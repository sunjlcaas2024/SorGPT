#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_fix.py — 修复后运行，验证 citation_map 命中率
放到 rag_project/ 目录下：cd rag_project && python verify_fix.py
"""
import sys
sys.path.insert(0, ".")

CSV   = "/vol/sunjilin/website/data/publication/english_content.csv"
BASE  = "/vol/sunjilin/website/data/agent"
MODEL = f"{BASE}/models/bge-m3/"
INDEX = f"{BASE}/faiss_v2_english_fine"

print("=" * 60)
print("步骤1：验证 citation_map 加载")
print("=" * 60)
from metadata_loader import load_citation_map, safe_get_ref_info
cmap = load_citation_map([CSV])
print(f"  citation_map 条目数: {len(cmap)}")
pdf_keys = [k for k in cmap.keys() if k.endswith(".pdf")][:3]
print(f"  带.pdf 的 key 示例: {pdf_keys}")
if pdf_keys:
    info = cmap[pdf_keys[0]]
    print(f"\n  示例（{pdf_keys[0][:50]}）:")
    print(f"    title:   {info['title'][:60]}")
    print(f"    authors: {info['authors'][:60]}")
    print(f"    journal: {info['journal'][:50]}")
    print(f"    year:    {info['year']}")
    print(f"    doi:     {info['doi'][:40]}")

print("\n" + "=" * 60)
print("步骤2：验证 FAISS source 匹配率")
print("=" * 60)
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class E(Embeddings):
    def __init__(self, m): self.m = m
    def embed_documents(self, t): return self.m.encode(t, normalize_embeddings=True).tolist()
    def embed_query(self, t): return self.m.encode([t], normalize_embeddings=True).tolist()[0]

print("  加载模型和索引...")
m  = SentenceTransformer(MODEL)
db = FAISS.load_local(INDEX, E(m), allow_dangerous_deserialization=True)
sources = list({doc.metadata.get("source","") for doc in db.docstore._dict.values() if doc.metadata.get("source","")})
print(f"  FAISS 唯一 source 数: {len(sources)}")
print(f"  source 示例: {sources[:2]}")

hit, miss, miss_ex = 0, 0, []
for src in sources:
    info = safe_get_ref_info(src, cmap)
    if info["title"] or info["authors"]: hit += 1
    else:
        miss += 1
        if len(miss_ex) < 5: miss_ex.append(src)

print(f"\n  命中: {hit}  未命中: {miss}  命中率: {hit/(hit+miss)*100:.1f}%")
if miss_ex:
    print("  未命中示例（不在CSV中的文献）:")
    for s in miss_ex: print(f"    {repr(s)}")
else:
    print("  ✅ 全部命中！")

print("\n" + "=" * 60)
print("步骤3：预览参考文献格式")
print("=" * 60)
from utils import build_citation_string
results = db.similarity_search_with_score("sorghum drought ABA signaling", k=3)
for i, (doc, score) in enumerate(results, 1):
    src  = doc.metadata.get("source", "")
    info = safe_get_ref_info(src, cmap)
    print(build_citation_string(info, i, src))
    print(f'    > "{doc.page_content[:180].replace(chr(10)," ")}..."')
    print()
