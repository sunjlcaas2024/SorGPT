# -*- coding: utf-8 -*-
"""
rebuild_meta_index.py
=====================
重建 faiss_v2_meta_english 元数据索引库。
支持同时读取多个 CSV，合并去重后建库。

使用方法：
    cd /vol/sunjilin/website/data/agent
    python rebuild_meta_index.py

注意：
    - 会先备份现有的 faiss_v2_meta_english 到 faiss_v2_meta_english_bak
    - 建库完成后自动替换
    - 建库时间约 5-15 分钟，取决于 CSV 行数和 GPU 速度
"""

import os
import re
import shutil
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ══════════════════════════════════════════════════════════════
# ★ 配置区（按实际路径修改）
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/vol/sunjilin/website/data/agent/models/bge-m3/"
SAVE_PATH  = "/vol/sunjilin/website/data/agent/faiss_v2_meta_english"

# 两个 CSV 都加进来
CSV_LIST = [
    "/vol/sunjilin/website/data/publication/english_content.csv",
    "/vol/sunjilin/website/data/publication/updated_records.csv",
]

EMBED_BATCH_SIZE = 8
BATCH_ADD_SIZE   = 200   # 每批写入 FAISS 的文档数

# ══════════════════════════════════════════════════════════════
# 列名候选
# ══════════════════════════════════════════════════════════════
_AUTHORS_COLS  = ["Author Full Names","Authors","AF","AU","authors","作者","第一作者"]
_TITLE_COLS    = ["Article Title","Title","TI","article title","title","题名","标题","文章标题"]
_ABSTRACT_COLS = ["Abstract","AB","abstract","摘要","文摘"]
_KEYWORDS_COLS = ["Author Keywords","Keywords","DE","ID","keywords","关键字","关键词","Keywords Plus"]
_DOI_COLS      = ["DOI","doi","DI","DOI Link"]
_YEAR_COLS     = ["Publication Year","Year","PY","publication year","年","出版年"]
_JOURNAL_COLS  = ["Source Title","Journal","SO","source title","journal","刊名","期刊名称","来源刊名"]
_FILENAME_CANDS= ["names","Names","filename","Filename","file_name","pdf_name","PDF","source"]


def norm_text(s) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000"," ").replace("\xa0"," ").strip()
    s = re.sub(r"\s+"," ",s)
    return "" if s.lower() == "nan" else s


def pick(row: dict, candidates: list) -> str:
    for c in candidates:
        v = norm_text(row.get(c,""))
        if v:
            return v
    # 忽略大小写再找一遍
    row_lower = {k.lower(): v for k, v in row.items()}
    for c in candidates:
        v = norm_text(row_lower.get(c.lower(),""))
        if v:
            return v
    return ""


def find_col(cols: list, candidates: list):
    for c in candidates:
        if c in cols:
            return c
    cols_lower = {col.lower(): col for col in cols}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig","utf-8","gb18030","gbk","latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python",
                             on_bad_lines="skip")
            if len(df) > 0:
                print(f"    读取成功（编码:{enc}，{len(df)} 行）: {os.path.basename(path)}")
                return df
        except Exception:
            continue
    raise RuntimeError(f"无法读取: {path}")


def build_meta_text(row: dict, pdf_fname: str, meta_fname: str) -> str:
    """
    构建元数据检索文本。
    title 和 keywords 各重复一次，提升检索权重。
    """
    title    = pick(row, _TITLE_COLS) or meta_fname
    authors  = pick(row, _AUTHORS_COLS)
    abstract = pick(row, _ABSTRACT_COLS)[:1200]
    keywords = pick(row, _KEYWORDS_COLS)
    journal  = pick(row, _JOURNAL_COLS)
    year     = pick(row, _YEAR_COLS)
    doi      = pick(row, _DOI_COLS)

    parts = [
        f"Title: {title}",
        f"Title: {title}",          # 重复提升权重
        f"Authors: {authors}",
        f"Keywords: {keywords}",
        f"Keywords: {keywords}",    # 重复提升权重
        f"Abstract: {abstract}",
        f"Journal: {journal}",
        f"Year: {year}",
        f"DOI: {doi}",
        f"Filename: {pdf_fname}",
    ]
    return "\n".join(p for p in parts if norm_text(p.split(":",1)[-1]))


class BgeWrapper(Embeddings):
    def __init__(self, model, batch_size=8):
        self.model      = model
        self.batch_size = batch_size
    def embed_documents(self, texts):
        return self.model.encode(
            texts, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False
        ).tolist()
    def embed_query(self, text):
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        ).tolist()[0]


def load_all_rows(csv_list: list) -> list:
    """
    读取所有 CSV，以 pdf_fname（带.pdf）为唯一键去重合并。
    返回 list of dict，每条包含 pdf_fname / meta_fname / row。
    """
    seen_keys = set()
    all_rows  = []

    for csv_path in csv_list:
        if not os.path.exists(csv_path):
            print(f"  [跳过] 文件不存在: {csv_path}")
            continue

        df   = read_csv_robust(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)

        # 识别两个文件名列
        pdf_col  = find_col(cols, ["names","Names","NAMES"])        # 带 .pdf
        meta_col = find_col(cols, ["filename","Filename","FILENAME"])# 不带 .pdf

        if not pdf_col and not meta_col:
            # fallback：用最后一列
            pdf_col = cols[-1]
            print(f"    [警告] 未找到文件名列，使用最后一列: {pdf_col}")

        for row in df.to_dict("records"):
            pdf_fname  = norm_text(row.get(pdf_col,  "")) if pdf_col  else ""
            meta_fname = norm_text(row.get(meta_col, "")) if meta_col else ""

            # 主唯一键：带 .pdf 的文件名
            key = pdf_fname or (meta_fname + ".pdf" if meta_fname else "")
            if not key:
                continue
            key_lower = key.lower()
            if key_lower in seen_keys:
                continue
            seen_keys.add(key_lower)

            all_rows.append({
                "pdf_fname":  pdf_fname,
                "meta_fname": meta_fname,
                "row":        row,
            })

    print(f"\n  去重后总计: {len(all_rows)} 条文献元数据")
    return all_rows


def build_index(csv_list: list, model_path: str, save_path: str):
    print("=" * 60)
    print("开始重建元数据索引库")
    print(f"  CSV 来源: {csv_list}")
    print(f"  保存路径: {save_path}")
    print("=" * 60)

    # 1. 备份旧库
    bak_path = save_path + "_bak"
    if os.path.exists(save_path):
        if os.path.exists(bak_path):
            shutil.rmtree(bak_path)
        shutil.copytree(save_path, bak_path)
        print(f"\n[备份] 旧库已备份到: {bak_path}")

    # 2. 读取所有 CSV
    print("\n[步骤1] 读取 CSV 元数据...")
    all_rows = load_all_rows(csv_list)
    if not all_rows:
        print("[错误] 没有读取到任何元数据，退出。")
        return

    # 3. 构建 Document 列表
    print("\n[步骤2] 构建 Document 列表...")
    docs = []
    for item in tqdm(all_rows, desc="构建 Document", ncols=80):
        pdf_fname  = item["pdf_fname"]
        meta_fname = item["meta_fname"]
        row        = item["row"]

        text = build_meta_text(row, pdf_fname, meta_fname)

        meta = {
            "filename": pdf_fname or meta_fname,
            "source":   pdf_fname or meta_fname,
            "title":    pick(row, _TITLE_COLS) or meta_fname,
            "authors":  pick(row, _AUTHORS_COLS),
            "journal":  pick(row, _JOURNAL_COLS),
            "year":     pick(row, _YEAR_COLS),
            "doi":      pick(row, _DOI_COLS),
            "keywords": pick(row, _KEYWORDS_COLS),
        }
        docs.append(Document(page_content=text, metadata=meta))

    print(f"  共构建 {len(docs)} 条 Document")

    # 4. 加载 Embedding 模型
    print("\n[步骤3] 加载 Embedding 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")
    base_model = SentenceTransformer(model_path, device=device)
    if device == "cuda":
        try:
            base_model.half()
            print("  已启用 float16 半精度")
        except Exception:
            pass
    embed_model = BgeWrapper(base_model, batch_size=EMBED_BATCH_SIZE)

    # 5. 分批写入 FAISS
    print(f"\n[步骤4] 向量化并写入 FAISS（每批 {BATCH_ADD_SIZE} 条）...")
    vector_db = None
    total = len(docs)

    for i in tqdm(range(0, total, BATCH_ADD_SIZE), desc="建库进度", ncols=80):
        batch = docs[i: i + BATCH_ADD_SIZE]
        if vector_db is None:
            vector_db = FAISS.from_documents(batch, embed_model)
        else:
            vector_db.add_documents(batch)

        # 每 1000 条自动保存一次
        if (i + BATCH_ADD_SIZE) % 1000 == 0 or (i + BATCH_ADD_SIZE) >= total:
            vector_db.save_local(save_path)
            tqdm.write(f"  [自动保存] 已处理 {min(i+BATCH_ADD_SIZE, total)}/{total}")

        if device == "cuda":
            torch.cuda.empty_cache()

    # 6. 最终保存
    if vector_db is not None:
        vector_db.save_local(save_path)
        print(f"\n✅ 元数据库重建完成！")
        print(f"   文献总数: {total}")
        print(f"   保存路径: {save_path}")

        # 验证库大小
        size_mb = sum(
            os.path.getsize(os.path.join(save_path, f))
            for f in os.listdir(save_path)
        ) / 1024 / 1024
        print(f"   库大小:   {size_mb:.1f} MB")

        if size_mb < 10:
            print("   ⚠️  库较小，请确认 CSV 数据是否完整")
        else:
            print("   ✓  库大小正常")
    else:
        print("[错误] 未生成任何索引，请检查 CSV 内容")

    # 7. 清理
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("重建完成。原库备份位于:", bak_path)
    print("如新库正常，可删除备份: rm -rf", bak_path)
    print("=" * 60)


if __name__ == "__main__":
    build_index(CSV_LIST, MODEL_PATH, SAVE_PATH)
