import os
import re
import json
import uuid
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# =========================
# 配置区
# =========================
SOURCE_DIRS = [
    "./data/pdfs/chinese",
    "./data/pdfs/english",
]
INDEX_DIR = "./data/index/faiss_index_sorghum"
MANIFEST_PATH = "./data/index/manifest.json"
PAPERS_JSONL = "./data/index/papers.jsonl"
CHUNKS_JSONL = "./data/index/chunks.jsonl"

BGE_MODEL_PATH = "./models/bge-m3"
TOKENIZER_PATH = "./models/qwen"   # 仅用于 token 计数，可替换为本地 Qwen tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 280
CHUNK_OVERLAP = 60
MIN_PARAGRAPH_CHARS = 80
MIN_TOTAL_TEXT_CHARS = 300

# 如果你有本地 OpenAI 兼容服务，可用于元数据纠正（可选）
USE_LLM_META_FIX = False
LLM_BASE_URL = "http://127.0.0.1:30000/v1"
LLM_MODEL_NAME = "/data/models/Qwen/Qwen3.5-32B"
LLM_API_KEY = "EMPTY"


# =========================
# Embedding 封装
# =========================
class BgeM3Embeddings:
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path, device=DEVICE)
        if DEVICE == "cuda":
            self.model.half()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


# =========================
# 工具函数
# =========================
def ensure_dirs():
    Path("./data/index").mkdir(parents=True, exist_ok=True)


def sha256_file(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> Dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        return {"files": {}, "created_at": None, "updated_at": None}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def list_pdf_files() -> List[str]:
    files = []
    for sd in SOURCE_DIRS:
        if not os.path.exists(sd):
            continue
        for root, _, filenames in os.walk(sd):
            for fn in filenames:
                if fn.lower().endswith(".pdf"):
                    files.append(os.path.join(root, fn))
    return sorted(files)


def detect_language_by_path(file_path: str) -> str:
    p = file_path.lower()
    if "chinese" in p or "zh" in p:
        return "zh"
    if "english" in p or "en" in p:
        return "en"
    return "unknown"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[^a-z0-9\u4e00-\u9fff ]", "", title)
    return title


def extract_doi(text: str) -> str:
    m = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', text, re.I)
    return m.group(1).strip() if m else ""


def extract_year(text: str) -> str:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if not years:
        return ""
    # 优先取第一页前部最早出现的合理年份
    return years[0]


def heuristic_extract_title_and_authors(first_page_text: str, file_name: str) -> Tuple[str, str]:
    lines = [x.strip() for x in first_page_text.splitlines() if x.strip()]
    lines = [x for x in lines if len(x) > 2]

    # 简单过滤页眉页脚/网址/doi
    filtered = []
    for line in lines[:25]:
        if "http" in line.lower() or "doi" in line.lower():
            continue
        if len(line) > 300:
            continue
        filtered.append(line)

    if not filtered:
        return os.path.splitext(file_name)[0], ""

    title = filtered[0]
    authors = ""
    if len(filtered) > 1:
        # 第二行往往是作者行，但并不总是
        candidate = filtered[1]
        if len(candidate) < 200:
            authors = candidate

    return title, authors


def format_reference(meta: Dict[str, Any]) -> str:
    """
    标准化到一种常见科研参考文献样式：
    Authors. Title. Journal. Year;Volume(Issue):Pages. doi:...
    """
    authors = meta.get("authors", "").strip()
    title = meta.get("title", "").strip()
    journal = meta.get("journal", "").strip()
    year = meta.get("year", "").strip()
    volume = meta.get("volume", "").strip()
    issue = meta.get("issue", "").strip()
    pages = meta.get("pages", "").strip()
    doi = meta.get("doi", "").strip()

    parts = []
    if authors:
        parts.append(authors.rstrip(".") + ".")
    if title:
        parts.append(title.rstrip(".") + ".")
    if journal:
        parts.append(journal.rstrip(".") + ".")

    vol_issue_pages = ""
    if year:
        vol_issue_pages += year
    if volume:
        vol_issue_pages += f";{volume}"
        if issue:
            vol_issue_pages += f"({issue})"
    if pages:
        vol_issue_pages += f":{pages}"
    if vol_issue_pages:
        parts.append(vol_issue_pages + ".")
    if doi:
        parts.append(f"doi:{doi}")

    ref = " ".join(parts).strip()
    return ref if ref else meta.get("source_file", "Unknown source")


def maybe_fix_metadata_with_llm(first_page_text: str, current_meta: Dict[str, Any]) -> Dict[str, Any]:
    if not USE_LLM_META_FIX:
        return current_meta

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        prompt = f"""
请从下面论文首页文本中尽量抽取 bibliographic metadata，返回 JSON：
{{
  "title": "",
  "authors": "",
  "journal": "",
  "year": "",
  "volume": "",
  "issue": "",
  "pages": "",
  "doi": ""
}}

要求：
1. 只返回 JSON；
2. 不确定就留空；
3. 不要编造。

论文首页文本：
{first_page_text[:6000]}
"""
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        for k, v in parsed.items():
            if v:
                current_meta[k] = v
    except Exception:
        pass

    return current_meta


def extract_paragraph_blocks(pdf_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    返回：
    - paragraph_records: [{text, page_start, page_end, section}]
    - paper_meta
    """
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    all_page_text = []
    paragraph_records = []

    first_page_text = ""

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        if page_idx == 0:
            first_page_text = page.get_text()

        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda x: (x[1], x[0]))  # y, x 排序

        current_buf = []
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            text = normalize_text(text)
            if len(text) < 20:
                continue

            # 页码/页眉页脚过滤
            if re.fullmatch(r"\d+", text):
                continue
            low = text.lower()
            if "copyright" in low or "all rights reserved" in low:
                continue

            current_buf.append(text)

        page_text = "\n".join(current_buf).strip()
        all_page_text.append(page_text)

        # 以 block 为基础形成段落记录
        for blk in current_buf:
            if len(blk) < MIN_PARAGRAPH_CHARS:
                continue
            paragraph_records.append({
                "text": blk,
                "page_start": page_idx + 1,
                "page_end": page_idx + 1,
                "section": ""
            })

    doc.close()

    full_text = normalize_text("\n".join(all_page_text))
    if len(full_text) < MIN_TOTAL_TEXT_CHARS:
        raise ValueError("PDF 可提取文本过少，可能是扫描件或图片型 PDF。")

    title, authors = heuristic_extract_title_and_authors(first_page_text, file_name)
    meta = {
        "title": title,
        "authors": authors,
        "journal": "",
        "year": extract_year(first_page_text),
        "volume": "",
        "issue": "",
        "pages": "",
        "doi": extract_doi(first_page_text),
        "source_file": file_name,
        "file_path": pdf_path,
        "language": detect_language_by_path(pdf_path),
    }
    meta = maybe_fix_metadata_with_llm(first_page_text, meta)
    meta["reference_text"] = format_reference(meta)
    return paragraph_records, meta


def write_jsonl(path: str, rows: List[Dict[str, Any]], mode: str = "a"):
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# 主逻辑
# =========================
def main():
    ensure_dirs()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False)),
        separators=["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""]
    )

    embed_model = BgeM3Embeddings(BGE_MODEL_PATH)

    all_files = list_pdf_files()
    print(f"发现 PDF 数量: {len(all_files)}")

    current_state = {fp: sha256_file(fp) for fp in all_files}
    manifest = load_manifest()
    old_state = manifest.get("files", {})

    deleted_files = sorted(set(old_state.keys()) - set(current_state.keys()))
    modified_files = sorted([fp for fp in current_state if fp in old_state and current_state[fp] != old_state[fp]])
    new_files = sorted([fp for fp in current_state if fp not in old_state])

    print(f"新增文件: {len(new_files)}")
    print(f"修改文件: {len(modified_files)}")
    print(f"删除文件: {len(deleted_files)}")

    rebuild_required = bool(modified_files or deleted_files)

    if rebuild_required:
        print("检测到修改/删除文件。为避免旧向量残留，本次执行全量重建索引。")
        if os.path.exists(PAPERS_JSONL):
            os.remove(PAPERS_JSONL)
        if os.path.exists(CHUNKS_JSONL):
            os.remove(CHUNKS_JSONL)
        if os.path.exists(INDEX_DIR):
            # 清理旧索引目录
            for root, dirs, files in os.walk(INDEX_DIR, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(INDEX_DIR)
        files_to_process = all_files
        vector_db = None
    else:
        files_to_process = new_files
        if os.path.exists(INDEX_DIR) and new_files:
            print("加载已有索引，准备增量添加。")
            vector_db = FAISS.load_local(
                INDEX_DIR,
                embed_model,
                allow_dangerous_deserialization=True
            )
        else:
            vector_db = None

    if not files_to_process:
        print("没有需要处理的新文件。索引保持不变。")
        manifest["files"] = current_state
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_manifest(manifest)
        return

    paper_rows = []
    chunk_rows = []
    documents = []

    seen_doc_hashes = set()
    seen_titles = set()

    start = time.time()

    for file_path in tqdm(files_to_process, desc="处理 PDF"):
        file_hash = current_state[file_path]
        if file_hash in seen_doc_hashes:
            continue
        seen_doc_hashes.add(file_hash)

        try:
            paragraphs, paper_meta = extract_paragraph_blocks(file_path)
        except Exception as e:
            print(f"跳过 {os.path.basename(file_path)}: {e}")
            continue

        title_norm = normalize_title(paper_meta.get("title", ""))
        if title_norm and title_norm in seen_titles:
            # 单次建库中的标题级去重
            print(f"标题疑似重复，跳过: {paper_meta.get('title', os.path.basename(file_path))}")
            continue
        if title_norm:
            seen_titles.add(title_norm)

        paper_id = str(uuid.uuid4())
        paper_meta["paper_id"] = paper_id
        paper_meta["file_sha256"] = file_hash
        paper_meta["reference_text"] = format_reference(paper_meta)

        paper_rows.append(paper_meta)

        chunk_index = 0
        for para in paragraphs:
            text = para["text"]
            sub_chunks = splitter.split_text(text)

            for chunk_text in sub_chunks:
                token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
                if token_count < 20:
                    continue

                chunk_id = str(uuid.uuid4())
                meta = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title": paper_meta.get("title", ""),
                    "authors": paper_meta.get("authors", ""),
                    "journal": paper_meta.get("journal", ""),
                    "year": paper_meta.get("year", ""),
                    "volume": paper_meta.get("volume", ""),
                    "issue": paper_meta.get("issue", ""),
                    "pages": paper_meta.get("pages", ""),
                    "doi": paper_meta.get("doi", ""),
                    "reference_text": paper_meta.get("reference_text", ""),
                    "source_file": paper_meta.get("source_file", ""),
                    "file_path": paper_meta.get("file_path", ""),
                    "file_sha256": paper_meta.get("file_sha256", ""),
                    "language": paper_meta.get("language", ""),
                    "page_start": para["page_start"],
                    "page_end": para["page_end"],
                    "section": para.get("section", ""),
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                }

                documents.append(Document(page_content=chunk_text, metadata=meta))
                chunk_rows.append(meta)
                chunk_index += 1

    if not documents:
        print("本次无有效 chunk，未生成索引。")
        return

    print(f"有效 chunk 数量: {len(documents)}")

    if vector_db is None:
        vector_db = FAISS.from_documents(documents, embed_model)
    else:
        vector_db.add_documents(documents)

    vector_db.save_local(INDEX_DIR)
    write_jsonl(PAPERS_JSONL, paper_rows, mode="a")
    write_jsonl(CHUNKS_JSONL, chunk_rows, mode="a")

    manifest["files"] = current_state
    if not manifest.get("created_at"):
        manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_manifest(manifest)

    print(f"索引已保存到: {INDEX_DIR}")
    print(f"耗时: {(time.time() - start) / 60:.2f} 分钟")


if __name__ == "__main__":
    main()
