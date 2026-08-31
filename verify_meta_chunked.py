# -*- coding: utf-8 -*-
"""verify_meta_chunked.py ── 新旧 meta 索引召回对比（swap 前预验证）"""
import os, sys, re, pickle

H = "/vol/sunjilin/website/data/agent/sorghum_rag"
sys.path.insert(0, H); os.chdir(H)

from langchain_community.vectorstores import FAISS
from embeddings import BgeEmbeddingsWrapper

EMB = BgeEmbeddingsWrapper()
OLD_EN = "/vol/sunjilin/website/data/agent/faiss_v3_meta_english"
NEW_EN = OLD_EN + ".chunked"

K = 600  # 对应 _k = k*2 = 600（非中文题 review）

def basename(fn): return os.path.basename(fn or "").lower()

def load(path):
    return FAISS.load_local(path, EMB, allow_dangerous_deserialization=True)

def whole_docs(path):
    """只读 docstore 的整篇条目（排除 _sent 句子）。"""
    with open(os.path.join(path, "index.pkl"), "rb") as f:
        obj = pickle.load(f)
    d = getattr(obj[0], "_dict", obj[0])
    return [doc for doc in d.values() if doc is not None
            and not (doc.metadata or {}).get("_sent")]

def known_hyz(docs):
    s = set()
    for doc in docs:
        if "hongyingzi" in (doc.page_content or "").lower():
            s.add(basename((doc.metadata or {}).get("filename") or ""))
    return s

def pool(db, q, k=K):
    res = db.similarity_search_with_score(q, k=k)
    seen, out = set(), []
    for doc, score in res:
        md = doc.metadata or {}
        fn = md.get("filename") or md.get("source") or ""
        uniq = basename(fn) or md.get("title", "")
        if uniq in seen:
            continue
        seen.add(uniq)
        out.append((uniq, float(score), md.get("title", "")[:70], md.get("_sent", False)))
    return out

QUERIES = [
    "Hongyingzi",
    "Moutai liquor",
    "glutinous sorghum baijiu",
    "grain composition liquor",
]

def main():
    docs_old = whole_docs(OLD_EN)
    hyz = known_hyz(docs_old)
    print(f"[ground truth] 整篇含 hongyingzi 论文: {len(hyz)} 篇\n", flush=True)
    print("载入旧/新英文索引...", flush=True)
    old_db = load(OLD_EN)
    new_db = load(NEW_EN)
    print("载入完成\n", flush=True)

    for q in QUERIES:
        po = pool(old_db, q)
        pn = pool(new_db, q)
        ro = len([u for u, *_ in po if u in hyz])
        rn = len([u for u, *_ in pn if u in hyz])
        print(f"Q: {q!r}")
        print(f"   旧: distinct={len(po)}  hongyingzi召回={ro}/{len(hyz)}")
        print(f"   新: distinct={len(pn)}  hongyingzi召回={rn}/{len(hyz)}")
        # 新索引 top5 样例（带 _sent 标记检查）
        for u, s, t, _sent in pn[:5]:
            print(f"     top[{s:.3f}] sent={_sent} {u} | {t}")
        print(flush=True)

    # E048: Chen 2025 论文（标题含 comprehensive omics resource）
    q = "E048 T2T genome assembly telomere"
    for name, db in (("旧", old_db), ("新", new_db)):
        found = [(u, s) for u, s, t, _ in pool(db, q) if "omics resource" in t.lower()]
        print(f"[E048] {name} 索引池中 Chen 论文: {found[:2] if found else '未命中'}")
    # CHBZ: Li 2024 论文
    q = "CHBZ sorghum genome assembly"
    for name, db in (("旧", old_db), ("新", new_db)):
        found = [(u, s) for u, s, t, _ in pool(db, q) if "chbz" in t.lower() or "btx623" in t.lower() and "assembly" in t.lower()]
        print(f"[CHBZ] {name} 索引池中相关论文: {len(found)} 篇, e.g. {found[0][0] if found else '未命中'}")

if __name__ == "__main__":
    main()
