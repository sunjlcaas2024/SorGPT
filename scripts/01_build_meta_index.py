import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

MODEL_PATH = "./models/bge-m3"
BATCH_ADD_SIZE = 100
EMBED_BATCH_SIZE = 8

CONFIG = {
    "english": {
        "csv": "/vol/sunjilin/website/data/publication/english_content.csv",
        "save": "./faiss_index_meta_english"
    },
    "chinese": {
        "csv": "/vol/sunjilin/website/data/publication/chinese_content.csv",
        "save": "./faiss_index_meta_chinese"
    }
}

_EN_AUTHORS_COLS = ["Author Full Names","Authors","AF","AU","authors","作者","作　者","第一作者"]
_EN_TITLE_COLS = ["filename", "Article Title","Title","TI","article title","title","题名","题　名","标题","文章标题"]
_EN_ABSTRACT_COLS = ["Abstract","AB","abstract","摘要","文摘","文　摘"]
_EN_KEYWORDS_COLS = ["Author Keywords","Keywords","DE","ID","keywords","关键字","关键词"]
_EN_DOI_COLS = ["DOI","doi","DI"]
_EN_YEAR_COLS = ["Publication Year","Year","PY","publication year","发表年","年","出版年"]
_EN_JOURNAL_COLS = ["Source Title","Journal","SO","source title","journal","刊名","刊　名","期刊名称","来源刊名"]
_FILENAME_CANDS = ["匹配的PDF文件名","filename","file_name","pdf_name","pdf","File Name","Filename","PDF","file","source","names"]

class BgeEmbeddingsWrapper(Embeddings):
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
    def embed_query(self, text):
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()[0]

def norm_text(s):
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return "" if s.lower() == "nan" else s

def normalize_colname(x):
    x = str(x).strip()
    x = x.replace(" ", "").replace("\u3000", "")
    return x

def clean_keywords(s):
    s = norm_text(s)
    if not s:
        return ""
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ;")

def pick(row, candidates, fallback=""):
    for c in candidates:
        if c in row:
            v = norm_text(row.get(c, ""))
            if v:
                return v
    compact_row = {normalize_colname(k): v for k, v in row.items()}
    for c in candidates:
        ck = normalize_colname(c)
        if ck in compact_row:
            v = norm_text(compact_row.get(ck, ""))
            if v:
                return v
    return fallback

def detect_filename_col(cols):
    for c in cols:
        if c in _FILENAME_CANDS:
            return c
    low_map = {normalize_colname(c): c for c in cols}
    for cand in _FILENAME_CANDS:
        k = normalize_colname(cand)
        if k in low_map:
            return low_map[k]
    return cols[-1]

def sentence_case(s):
    s = norm_text(s)
    if not s:
        return s
    alpha_chars = [c for c in s if c.isalpha()]
    if not alpha_chars:
        return s
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if upper_ratio <= 0.8:
        return s
    ACRONYMS = {
        "QTL","GWAS","SNP","DNA","RNA","mRNA","NAC","MYB","WRKY","YABBY","PCR","SSR","RFLP",
        "AFLP","RIL","NIL","DH","BC","F2","ABA","GA","IAA","JA","SA","ROS","SOD","POD","CAT",
        "APX","GFP","YFP","FISH","SDS","PAGE","HPLC","GC","MS","NMR","ATP","NADPH","FADH","CoA",
        "CDS","UTR","ORF","INDEL","CNV","SV","LD","PCA","BLUP","GBS","WGS","RAD","NGS","SMRT",
        "ONT","Hi-C","ChIP","ATAC"
    }
    words = s.split()
    out = []
    for i, w in enumerate(words):
        core = w.rstrip(".,;:?!")
        punct = w[len(core):]
        if core.upper() in ACRONYMS:
            out.append(core.upper() + punct)
        elif re.match(r"^[A-Za-z][A-Za-z0-9.]*[0-9][A-Za-z0-9.]*$", core):
            out.append(w)
        elif i == 0:
            out.append(core[:1].upper() + core[1:].lower() + punct)
        else:
            out.append(core.lower() + punct)
    return " ".join(out)

def title_case(s):
    s = norm_text(s)
    if not s:
        return s
    alpha_chars = [c for c in s if c.isalpha()]
    if not alpha_chars:
        return s
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if upper_ratio <= 0.8:
        return s
    small_words = {"a","an","the","and","or","of","in","on","at","to","for","with","by","from"}
    ws = s.lower().split()
    return " ".join([w.capitalize() if i == 0 or w not in small_words else w for i, w in enumerate(ws)])

def format_authors(authors_raw):
    authors_raw = norm_text(authors_raw)
    if not authors_raw:
        return ""
    parts = [a.strip() for a in re.split(r"[;|]", authors_raw) if a.strip()]
    out = []
    for p in parts[:6]:
        if "," in p:
            seg = [x.strip() for x in p.split(",", 1)]
            last = seg[0]
            first = seg[1] if len(seg) > 1 else ""
            initials = "".join(w[0].upper() + "." for w in first.split() if w)
            out.append(f"{last}, {initials}" if initials else last)
        else:
            out.append(p)
    formatted = "; ".join(out)
    if len(parts) > 6:
        formatted += " et al."
    return formatted

def read_csv_robust(csv_path):
    for enc in ["utf-8-sig", "gb18030", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=enc, engine="python")
        except Exception:
            continue
    raise RuntimeError(f"无法读取 CSV: {csv_path}")

def build_meta_text(item):
    title = item.get("title", "")
    keywords = item.get("keywords", "")
    abstract = item.get("abstract", "")[:1200]
    authors_raw = item.get("authors_raw", "")
    journal = item.get("journal", "")
    year = item.get("year", "")
    doi = item.get("doi", "")
    filename = item.get("filename", "")
    parts = [
        f"Title: {title}",
        f"Title: {title}",
        f"Authors: {authors_raw}",
        f"Keywords: {keywords}",
        f"Keywords: {keywords}",
        f"Abstract: {abstract}",
        f"Journal: {journal}",
        f"Year: {year}",
        f"DOI: {doi}",
        f"Filename: {filename}"
    ]
    return "\n".join([p for p in parts if norm_text(p.split(':', 1)[-1])])

def load_metadata_rows(csv_path):
    df = read_csv_robust(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    print(f"\n[诊断] {os.path.basename(csv_path)} 列名:")
    print("  " + " | ".join(cols))
    filename_col = detect_filename_col(cols)
    df = df.drop_duplicates(subset=[filename_col])
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {os.path.basename(csv_path)}"):
        d = row.to_dict()
        filename = norm_text(d.get(filename_col, ""))
        if not filename:
            continue
        authors_raw = pick(d, _EN_AUTHORS_COLS)
        title = pick(d, _EN_TITLE_COLS)
        abstract = pick(d, _EN_ABSTRACT_COLS)
        keywords = clean_keywords(pick(d, _EN_KEYWORDS_COLS))
        doi = pick(d, _EN_DOI_COLS)
        year = pick(d, _EN_YEAR_COLS)
        journal = pick(d, _EN_JOURNAL_COLS)
        rows.append({
            "filename": filename,
            "authors": format_authors(authors_raw),
            "authors_raw": authors_raw,
            "title": sentence_case(title),
            "abstract": abstract,
            "keywords": keywords,
            "doi": doi,
            "year": year,
            "journal": title_case(journal)
        })
    print(f"[元数据] 共读取 {len(rows)} 条")
    return rows

def build_meta_index(lang, csv_path, save_path):
    if os.path.exists(save_path):
        print(f"\n>>> 索引 {save_path} 已存在，跳过。")
        return
    print(f"\n>>> 正在构建 {lang.upper()} 元数据索引...")
    rows = load_metadata_rows(csv_path)
    docs = [Document(page_content=build_meta_text(item), metadata=item) for item in rows]
    if not docs:
        print(f"[警告] {lang} 没有可用于建库的文档")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Embedding] device = {device}")
    base_model = SentenceTransformer(MODEL_PATH, device=device)
    if device == "cuda":
        try:
            base_model.half()
        except Exception:
            pass
    model = BgeEmbeddingsWrapper(base_model, batch_size=EMBED_BATCH_SIZE)

    vector_db = None
    for i in tqdm(range(0, len(docs), BATCH_ADD_SIZE), desc=f"Building {lang} meta index"):
        batch_docs = docs[i:i+BATCH_ADD_SIZE]
        if vector_db is None:
            vector_db = FAISS.from_documents(batch_docs, model)
        else:
            vector_db.add_documents(batch_docs)
        vector_db.save_local(save_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f">>> {lang.upper()} 元数据索引构建完成: {save_path}")
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    for lang, conf in CONFIG.items():
        build_meta_index(lang, conf["csv"], conf["save"])
