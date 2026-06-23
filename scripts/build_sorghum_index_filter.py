import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import pandas as pd
import fitz
import torch
import gc
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# -----------------------------
# 统一配置
# -----------------------------
MODEL_PATH = "/vol/sunjilin/website/data/agent/models/bge-m3/"
BASE_SAVE_DIR = "/vol/sunjilin/website/data/agent"
INDEX_PREFIX = "faiss_v2"   # 修改这里即可统一改所有库名

SCALES = [
    {"name": "fine",  "size": 500,  "overlap": 50,  "batch": 256},
    {"name": "std",   "size": 1000, "overlap": 100, "batch": 128},
    {"name": "large", "size": 1500, "overlap": 200, "batch": 64},
    {"name": "para",  "size": None, "overlap": 0,   "batch": 128},
]

# -----------------------------
# 过滤规则
# -----------------------------
_TRUNCATE_SECTION_PATTERNS = re.compile(
    r"^\s*(references|bibliography|参考文献|文献|works cited|literature cited"
    r"|supplementary|supplemental materials?|supporting information"
    r"|appendix|附录)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SKIP_SECTION_PATTERNS = re.compile(
    r"^\s*(acknowledgements?|acknowledgments?|funding|conflict of interest"
    r"|author contributions?|data availability|ethics statement"
    r"|abbreviations|致谢|资金|作者贡献|数据可用性)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_REF_LINE_PATTERN = re.compile(r"^\s*(\[\d+\]|\d+\.)\s+[A-Z]")
_HEADER_FOOTER_PATTERN = re.compile(
    r"(^\s*\d+\s*$"
    r"|©\s*\d{4}"
    r"|all rights reserved"
    r"|www\.\S+\.\S+"
    r"|doi:\s*10\.\d{4}"
    r"|received:\s*\d"
    r"|accepted:\s*\d"
    r"|published:\s*\d"
    r"|volume\s+\d+.*issue\s+\d+"
    r"|^\s*page\s+\d+\s+of\s+\d+\s*$"
    r")",
    re.IGNORECASE,
)

def clean_pdf_text(text: str) -> str:
    match = _TRUNCATE_SECTION_PATTERNS.search(text)
    if match:
        text = text[:match.start()]
    sections_to_remove = []
    for m in _SKIP_SECTION_PATTERNS.finditer(text):
        start = m.start()
        rest = text[m.end():]
        next_section = re.search(r"\n\s*\n\s*[A-Z\u4e00-\u9fff][^\n]{0,60}\n", rest)
        end = m.end() + (next_section.start() if next_section else len(rest))
        sections_to_remove.append((start, end))
    for start, end in reversed(sections_to_remove):
        text = text[:start] + text[end:]
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _HEADER_FOOTER_PATTERN.search(stripped):
            continue
        if re.match(r"^\d{1,4}$", stripped):
            continue
        if len(stripped) < 15 and not re.search(r"[.!?。]$", stripped):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"-\n(\s*)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_paragraphs(text: str) -> list:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= 80]

def merge_short_paragraphs(paragraphs: list, min_len: int = 200) -> list:
    merged = []
    buffer = ""
    for p in paragraphs:
        buffer = (buffer + "\n\n" + p) if buffer else p
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer)
    return merged

def split_para_mode(text: str, max_para_size: int = 2000) -> list:
    paragraphs = split_into_paragraphs(text)
    paragraphs = merge_short_paragraphs(paragraphs, min_len=150)
    final = []
    for p in paragraphs:
        if len(p) <= max_para_size:
            final.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buffer = ""
            for sent in sentences:
                if len(buffer) + len(sent) + 1 <= max_para_size:
                    buffer = (buffer + " " + sent).strip() if buffer else sent
                else:
                    if buffer:
                        final.append(buffer)
                    buffer = sent
            if buffer:
                final.append(buffer)
    return final

def smart_split(text: str, chunk_size: int, chunk_overlap: int) -> list:
    paragraphs = split_into_paragraphs(text)
    paragraphs = merge_short_paragraphs(paragraphs, min_len=chunk_size // 3)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    final_chunks = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            final_chunks.append(p)
        else:
            final_chunks.extend(splitter.split_text(p))
    return final_chunks

def is_noise_chunk(text: str) -> bool:
    text = text.strip()
    if len(text) < 80:
        return True
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return True
    ref_lines = sum(1 for l in non_empty if _REF_LINE_PATTERN.match(l))
    if ref_lines / len(non_empty) > 0.35:
        return True
    doi_lines = sum(1 for l in non_empty if "doi:" in l.lower())
    if doi_lines / len(non_empty) > 0.4:
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / max(len(text), 1) < 0.25:
        return True
    avg_line_len = sum(len(l.strip()) for l in non_empty) / len(non_empty)
    if avg_line_len < 20 and len(non_empty) > 5:
        return True
    return False

# -----------------------------
# 适配器类
# -----------------------------
class BgeEmbeddingsWrapper(Embeddings):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
    def embed_documents(self, texts):
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True
        ).tolist()
    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]

# -----------------------------
# 单文件解析（单进程，避免多进程静默失败）
# -----------------------------
def parse_single_pdf(fpath: str, meta: dict, scale_name: str, size, overlap) -> list:
    try:
        fname = os.path.basename(fpath)
        with fitz.open(fpath) as doc:
            pages_text = [page.get_text() for page in doc]
            text = "\n\n".join(pages_text)
        if len(text.strip()) < 100:
            return []
        text = clean_pdf_text(text)
        if len(text.strip()) < 100:
            return []
        if scale_name == "para":
            chunks = split_para_mode(text, max_para_size=2000)
        else:
            chunks = smart_split(text, chunk_size=size, chunk_overlap=overlap)
        docs = []
        for c in chunks:
            if is_noise_chunk(c):
                continue
            docs.append(Document(
                page_content=c,
                metadata={"source": fname, **meta, "scale": scale_name}
            ))
        return docs
    except Exception as e:
        print(f"    [警告] 解析失败: {os.path.basename(fpath)} — {e}")
        return []

# -----------------------------
# 主建库函数
# -----------------------------
def build_single_language(lang: str, csv_path: str, pdf_dir: str):
    print(f"\n{'='*60}")
    print(f"开始构建 {lang.upper()} 语言索引库")
    print(f"库名前缀: {INDEX_PREFIX}")
    print(f"CSV: {csv_path}")
    print(f"PDF目录: {pdf_dir}")
    print(f"保存目录: {BASE_SAVE_DIR}")
    print(f"{'='*60}")

    # 读取 CSV
    print("正在读取元数据 CSV...")
    df = None
    for enc in ["utf-8-sig", "gb18030", "utf-8"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc, engine="python")
            print(f"    CSV 读取成功（编码: {enc}），共 {len(df)} 条记录")
            break
        except Exception as e:
            print(f"    尝试编码 {enc} 失败: {e}")
    if df is None:
        print(f"[错误] 无法读取 CSV: {csv_path}")
        return

    filename_col = df.columns[-1]
    print(f"    文件名列识别为: '{filename_col}'")
    df = df.drop_duplicates(subset=[filename_col])
    meta_map = {
        k: {**v, "names": k}
        for k, v in df.set_index(filename_col).to_dict("index").items()
    }

    # 扫描 PDF
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
    print(f"    共发现 {len(pdf_files)} 个 PDF 文件")
    if not pdf_files:
        print("[错误] PDF 目录为空，退出。")
        return

    # 加载 embedding 模型
    print("\n正在加载 Embedding 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    使用设备: {device}")
    base_model = SentenceTransformer(MODEL_PATH, device=device)
    if device == "cuda":
        try:
            base_model.half()
            print("    已启用 float16 半精度")
        except Exception:
            pass
    print("    Embedding 模型加载完成")

    for scale in SCALES:
        scale_name = scale["name"]
        size = scale["size"]
        overlap = scale["overlap"]
        batch = scale["batch"]

        # 库名格式：faiss_v2_english_fine
        save_path = os.path.join(BASE_SAVE_DIR, f"{INDEX_PREFIX}_{lang}_{scale_name}")

        if os.path.exists(save_path):
            print(f"\n>>> 库已存在，跳过: {save_path}")
            continue

        mode_desc = "自然段模式" if scale_name == "para" else f"chunk_size={size}, overlap={overlap}"
        print(f"\n>>> 开始构建: {lang.upper()} - {scale_name.upper()} ({mode_desc})")
        print(f"    保存路径: {save_path}")

        model = BgeEmbeddingsWrapper(base_model, batch_size=batch)
        vector_db = None
        total_kept = 0
        failed = 0

        for i, fname in enumerate(tqdm(pdf_files, desc=f"{lang}-{scale_name}", ncols=80)):
            fpath = os.path.join(pdf_dir, fname)
            meta = meta_map.get(fname, {})
            docs = parse_single_pdf(fpath, meta, scale_name, size, overlap)
            if not docs:
                failed += 1
                continue
            total_kept += len(docs)
            try:
                if vector_db is None:
                    vector_db = FAISS.from_documents(docs, model)
                else:
                    vector_db.add_documents(docs)
            except Exception as e:
                print(f"\n    [警告] 向量化失败: {fname} — {e}")
                failed += 1
                continue
            # 每 50 个文件保存一次
            if (i + 1) % 50 == 0 and vector_db is not None:
                vector_db.save_local(save_path)
                tqdm.write(f"    [自动保存] {i+1}/{len(pdf_files)} 文件，chunk 数: {total_kept}")

        if vector_db is not None:
            vector_db.save_local(save_path)
            print(f"\n    ✓ {scale_name} 库构建完成")
            print(f"      入库 chunk 数: {total_kept}")
            print(f"      失败文件数:   {failed}")
            print(f"      保存路径:     {save_path}")
        else:
            print(f"\n    [警告] {scale_name} 库无有效数据，请检查 PDF 和 CSV。")

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"\n{'='*60}")
    print(f"{lang.upper()} 全部索引库构建完成")
    print(f"生成的库：")
    for scale in SCALES:
        p = os.path.join(BASE_SAVE_DIR, f"{INDEX_PREFIX}_{lang}_{scale['name']}")
        status = "✓ 已生成" if os.path.exists(p) else "✗ 未生成"
        print(f"  {status}  {p}")
    print(f"{'='*60}")
