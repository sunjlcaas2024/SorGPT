# -*- coding: utf-8 -*-
"""
rebuild_zh_filtered.py ── 中文文献去噪：剔除明显与高粱不相关的文献（2026-08-31）
================================================================================
背景
----
中文语料 16805 篇中混入 ~1173 篇与高粱无关的文献（水稻/玉米/食用菌/财经等，
标题/摘要/关键词全无高粱信号）。本次将其剔除并重建全部受影响索引。

剔除标准（保守口径，用户确认）
----------------------------
标题+摘要+关键词 任一含 高粱/甜高粱/粒用高粱/sorghum/Sorghum/S. bicolor/
红缨子/高梁/蜀黍/秫秫 → 保留；否则剔除。

产物
----
1. 剔除清单            zh_removed_sources.txt   (~1173 个 filename)
2. 过滤后的 CSV        /vol/.../publication/chinese_content.csv (先备份 .bak.zhfilter)
3. 重建的全文索引      faiss_index_chinese_{fine,std,large} 暂存为 .zhfilter 目录
4. 重建的中文 meta 索引 faiss_index_meta_chinese 暂存为 .zhfilter 目录

索引重建方法（关键）
------------------
当前 zh 索引是 IVFPQ 压缩索引（原始 float32 向量不可恢复），但 chunk 文本完整
保存在 index.pkl docstore 中。因此：从 docstore 取保留 chunk 文本 → BGE-M3
重嵌入（与运行时同模型同 normalize）→ IndexFlatIP → 转 IVFPQ（配方对齐
rebuild_indexes.py：nlist=clamp(√N,[256,4096]), m=64, nbits=8, 内积, nprobe=64）
→ 写 index.faiss + index.pkl。meta 索引保持 flat（对齐 KEEP_FLAT）。

用法
----
# 1) 生成清单 + 过滤 CSV + 只重建（不换线上）
python rebuild_zh_filtered.py build
# 2) 备份原数据（build 成功后、swap 前执行）
python rebuild_zh_filtered.py backup
# 3) 确认无误后原子替换
python rebuild_zh_filtered.py swap
# 4) 重建 BM25 IDF（CPU，独立步骤）
python build_zh_bm25_idf.py
# 5) 重启服务

回滚：zhfilter_backup_20260831/ 或 *.old 目录直接覆盖；CSV 用 .bak.zhfilter 覆盖。
"""

import os, sys, csv, re, pickle, time, shutil, argparse, gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"] = "sequential"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 防显存碎片化 OOM

import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

_HERE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH    = "/vol/sunjilin/website/data/publication/chinese_content.csv"
MODEL_PATH  = "/vol/sunjilin/website/data/agent/models/bge-m3/"
REMOVED_LIST = os.path.join(_HERE, "zh_removed_sources.txt")
BACKUP_ROOT = "/vol/sunjilin/website/data/agent/zhfilter_backup_20260831"
SUFFIX      = ".zhfilter"

FULLTEXT_INDEX_PATHS = {
    "zh_fine":  "/vol/sunjilin/website/data/agent/faiss_index_chinese_fine",
    "zh_std":   "/vol/sunjilin/website/data/agent/faiss_index_chinese_std",
    "zh_large": "/vol/sunjilin/website/data/agent/faiss_index_chinese_large",
}
META_CHINESE = "/vol/sunjilin/website/data/agent/faiss_index_meta_chinese"
FULLTEXT_KEYS = ["zh_fine", "zh_std", "zh_large"]
# 每库嵌入 batch：chunk 越长显存需求越大（1500 字 large 用 64，避免注意力矩阵 OOM）
EMBED_BATCH = {"zh_fine": 128, "zh_std": 128, "zh_large": 64, "zh_meta": 64}

SORGHUM_PAT = re.compile(
    r"高粱|甜高粱|粒用高粱|sorghum|Sorghum|S\. bicolor|红缨子|高梁|蜀黍|秫秫"
)


def has_signal(s):
    return bool(SORGHUM_PAT.search(s or ""))


# ---------------------------------------------------------------- 清单与 CSV
def build_removal_set():
    # 优先用已持久化的清单（CSV 过滤后再重算会得到空集）
    if os.path.exists(REMOVED_LIST):
        removed = {l.strip() for l in open(REMOVED_LIST, encoding="utf-8") if l.strip()}
        print(f"[removal] 从 {REMOVED_LIST} 加载剔除清单: {len(removed)} 篇", flush=True)
        return removed
    removed = set()
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            if fn and not has_signal(
                row.get("title", "") + row.get("abstract", "") + row.get("keywords", "")
            ):
                removed.add(fn)
    return removed


def filter_csv(removed):
    if not os.path.exists(CSV_PATH + ".bak.zhfilter"):
        shutil.copy(CSV_PATH, CSV_PATH + ".bak.zhfilter")
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    kept = [r for r in rows if (r.get("filename") or "").strip() not in removed]
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    print(f"[CSV] {len(rows)} -> {len(kept)} (剔除 {len(rows) - len(kept)})", flush=True)


# ---------------------------------------------------------------- 嵌入模型
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_PATH, device=dev)
        if dev == "cuda":
            try:
                _model.half()
            except Exception:
                pass
        print(f"[model] device={dev} BGE-M3 loaded", flush=True)
    return _model


def embed(texts, batch=128, tag=""):
    m = get_model()
    vecs = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                    show_progress_bar=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    if not np.isfinite(vecs).all():
        bad = int((~np.isfinite(vecs).all(1)).sum())
        raise RuntimeError(f"[{tag}] {bad} 条非有限向量，中止（避免索引错位）")
    return vecs


# ---------------------------------------------------------------- IVFPQ 转换
def to_ivfpq(vecs, tag=""):
    N, dim = vecs.shape
    nlist = min(4096, max(256, int(N ** 0.5)))
    m, nbits = 64, 8
    print(f"[{tag}] IVFPQ: N={N:,} dim={dim} nlist={nlist} m={m} nbits={nbits}", flush=True)
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    train_vecs = vecs[:min(500_000, N)]
    t0 = time.time()
    ivf.train(train_vecs)
    print(f"[{tag}] train ok ({time.time() - t0:.0f}s)", flush=True)
    CH = 100_000
    for s in range(0, N, CH):
        ivf.add(vecs[s:s + CH])
        print(f"[{tag}] add {min(s + CH, N):,}/{N:,}", end="\r", flush=True)
    print(f"\n[{tag}] add done, ntotal={ivf.ntotal}", flush=True)
    ivf.nprobe = 64
    return ivf


# ---------------------------------------------------------------- 索引构建
def load_docs(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    docstore = obj[0] if isinstance(obj, tuple) else obj
    d = getattr(docstore, "_dict", docstore)
    return list(d.values()), len(d)


def write_store(stage, kept, vecs, ivf):
    os.makedirs(stage, exist_ok=True)
    faiss.write_index(ivf, os.path.join(stage, "index.faiss"))
    ds = InMemoryDocstore({i: doc for i, doc in enumerate(kept)})
    with open(os.path.join(stage, "index.pkl"), "wb") as f:
        pickle.dump((ds, {i: i for i in range(len(kept))}), f)
    print(f"[write] {stage} ({len(kept):,} docs, index.ntotal={ivf.ntotal})", flush=True)


def build_fulltext(key, removed):
    src = FULLTEXT_INDEX_PATHS[key]
    stage = src + SUFFIX
    if os.path.exists(os.path.join(stage, "index.faiss")):
        print(f"[{key}] 暂存已存在，跳过（断点续跑）", flush=True)
        return
    docs, total = load_docs(os.path.join(src, "index.pkl"))
    kept = [doc for doc in docs
            if os.path.basename((doc.metadata or {}).get("source") or "") not in removed]
    print(f"[{key}] chunks {total:,} -> {len(kept):,}", flush=True)
    if not kept:
        raise RuntimeError(f"[{key}] 保留 0 chunk，中止")
    vecs = embed([d.page_content for d in kept], batch=EMBED_BATCH[key], tag=key)
    ivf = to_ivfpq(vecs, tag=key)
    write_store(stage, kept, vecs, ivf)
    del docs, kept, vecs, ivf
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def build_meta(removed):
    src = META_CHINESE
    stage = src + SUFFIX
    if os.path.exists(os.path.join(stage, "index.faiss")):
        print(f"[zh_meta] 暂存已存在，跳过（断点续跑）", flush=True)
        return
    docs, total = load_docs(os.path.join(src, "index.pkl"))
    kept = [doc for doc in docs
            if os.path.basename((doc.metadata or {}).get("filename") or "") not in removed]
    print(f"[zh_meta] docs {total:,} -> {len(kept):,}", flush=True)
    vecs = embed([d.page_content for d in kept], batch=EMBED_BATCH["zh_meta"], tag="zh_meta")
    N, dim = vecs.shape
    flat = faiss.IndexFlatIP(dim)
    flat.add(vecs)
    write_store(stage, kept, vecs, flat)
    del docs, kept, vecs, flat
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------- 备份与替换
def do_backup():
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    for key in FULLTEXT_KEYS + ["zh_meta"]:
        src = FULLTEXT_INDEX_PATHS[key] if key != "zh_meta" else META_CHINESE
        dst = os.path.join(BACKUP_ROOT, os.path.basename(src))
        if os.path.exists(dst):
            print(f"[backup] skip(已存在) {dst}", flush=True)
            continue
        shutil.copytree(src, dst)
        print(f"[backup] {src} -> {dst}", flush=True)
    bak_csv = os.path.join(BACKUP_ROOT, "chinese_content.csv")
    if not os.path.exists(bak_csv) and os.path.exists(CSV_PATH + ".bak.zhfilter"):
        shutil.copy(CSV_PATH + ".bak.zhfilter", bak_csv)
        print(f"[backup] csv -> {bak_csv}", flush=True)
    bak_idf = os.path.join(BACKUP_ROOT, "zh_bm25_idf.pkl.bak")
    src_idf = os.path.join(_HERE, "zh_bm25_idf.pkl")
    if not os.path.exists(bak_idf) and os.path.exists(src_idf):
        shutil.copy(src_idf, bak_idf)
        print(f"[backup] idf -> {bak_idf}", flush=True)
    print("[backup] done", flush=True)


def do_swap():
    for key in FULLTEXT_KEYS + ["zh_meta"]:
        orig = FULLTEXT_INDEX_PATHS[key] if key != "zh_meta" else META_CHINESE
        stage = orig + SUFFIX
        if not os.path.exists(os.path.join(stage, "index.faiss")):
            print(f"[swap] 缺暂存 {stage}，跳过", flush=True)
            continue
        old = orig + ".old"
        if os.path.exists(old):
            shutil.rmtree(old)
        if os.path.exists(orig):
            os.rename(orig, old)
        os.rename(stage, orig)
        idx = faiss.read_index(os.path.join(orig, "index.faiss"))
        with open(os.path.join(orig, "index.pkl"), "rb") as f:
            obj = pickle.load(f)
        ndoc = len(getattr(obj[0], "_dict", obj[0]))
        ok = "OK" if ndoc == idx.ntotal else "!!MISMATCH!!"
        print(f"[swap] {os.path.basename(orig)} ntotal={idx.ntotal} ndoc={ndoc} {ok}", flush=True)
    print("[swap] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "backup", "swap"])
    args = ap.parse_args()

    if args.mode == "backup":
        do_backup()
        return
    if args.mode == "swap":
        do_swap()
        return

    removed = build_removal_set()
    print(f"[removal] 剔除候选论文数: {len(removed)}", flush=True)
    with open(REMOVED_LIST, "w", encoding="utf-8") as f:
        for fn in sorted(removed):
            f.write(fn + "\n")
    filter_csv(removed)
    for key in FULLTEXT_KEYS:
        t0 = time.time()
        build_fulltext(key, removed)
        print(f"[{key}] elapsed {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    build_meta(removed)
    print(f"[zh_meta] elapsed {time.time() - t0:.0f}s", flush=True)
    print("[build] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
