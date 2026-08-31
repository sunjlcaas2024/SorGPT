# -*- coding: utf-8 -*-
"""
rebuild_meta_chunked.py ── 整篇+句子级混合 meta 索引重建（2026-08-31）
================================================================================
背景
----
meta 检索把整篇摘要（title+authors+keywords+abstract+journal+doi+filename，
~2-3KB blob）嵌成 1 个向量，实体只在摘要出现一次时被稀释 → 实体查询
（hongyingzi/E048/红缨子）召回差。实验（diag_chunk.py）：整篇 vs 句子级
4 查询召回 = 3→9 / 7→12 / 3→7 / 12→19。

做法（用户确认"重建（推荐）"）
----------------------------
保留现有整篇向量（IndexFlat reconstruct，零回归），对每篇摘要句子级切分后
重嵌入追加。混合索引里每篇论文 = 1 整篇 + N 句子条目。retrieve_metadata 按
文件名去重取最高分（已支持，无需改去重逻辑）。

产物
----
faiss_v3_meta_english.chunked   (24,955 整篇 + ~180K 句子)
faiss_index_meta_chinese.chunked (15,018 整篇 + ~110K 句子)

用法
----
python rebuild_meta_chunked.py check   # 只统计摘要提取/句子数，验证正确性（不写盘）
python rebuild_meta_chunked.py build   # 提取+重嵌入+写暂存（断点续跑，~50min GPU）
python rebuild_meta_chunked.py backup  # 备份原索引目录
python rebuild_meta_chunked.py swap    # 原子替换（.old 留底）

回滚：meta_chunked_backup_20260901/ 直接覆盖，或 *.old 目录。
"""

import os, sys, re, pickle, time, shutil, argparse, gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"] = "sequential"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 防显存碎片化 OOM

import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

MODEL_PATH  = "/vol/sunjilin/website/data/agent/models/bge-m3/"
BACKUP_ROOT = "/vol/sunjilin/website/data/agent/meta_chunked_backup_20260901"
SUFFIX      = ".chunked"

META_INDEX_PATHS = {
    "english": "/vol/sunjilin/website/data/agent/faiss_v3_meta_english",
    "chinese": "/vol/sunjilin/website/data/agent/faiss_index_meta_chinese",
}
EMBED_BATCH = {"english": 128, "chinese": 128}   # 句子短(~150字)，batch 128 无 OOM
_MIN_SENT   = 8                                   # 句子最少字符，过滤噪声碎片

# 英文 meta page_content 中 Abstract 段：直到下一个 section 头(Journal:/DOI:/Filename:)
_EN_ABS = re.compile(r"Abstract:\s*(.*?)(?=\n[A-Z][A-Za-z]+:)", re.S)
_ZH_ABS = re.compile(r"Abstract[:：]\s*(.*?)(?=\n[A-Za-z一-鿿]+[:：])", re.S)


def load_db(src):
    with open(os.path.join(src, "index.pkl"), "rb") as f:
        obj = pickle.load(f)
    docstore = obj[0] if isinstance(obj, tuple) else obj
    i2d = obj[1] if isinstance(obj, tuple) and len(obj) > 1 else None
    idx = faiss.read_index(os.path.join(src, "index.faiss"))
    d = getattr(docstore, "_dict", docstore)
    return idx, d, i2d


def extract_abstract(doc):
    """中文 meta 把摘要存 metadata.abstract（直接取）；英文从 page_content 的 Abstract: 段解析。"""
    md = doc.metadata or {}
    abs_meta = (md.get("abstract") or "").strip()
    if abs_meta:
        return abs_meta
    pc = doc.page_content or ""
    for pat in (_EN_ABS, _ZH_ABS):
        m = pat.search(pc)
        if m and len(m.group(1).strip()) > 20:
            return m.group(1).strip()
    return None


def split_sentences(text, is_cn):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if is_cn:
        # 中文按句读切分；夹在中文里的英文句也切开（"综述 A；Ref: B.C.D."）
        out = []
        for p in re.split(r"(?<=[。！？；])\s*", text):
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Za-z\"'(0-9])", p.strip()):
                s = s.strip()
                if len(s) >= _MIN_SENT:
                    out.append(s)
        return out
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(0-9])", text)
    return [s.strip() for s in parts if len(s.strip()) >= _MIN_SENT]


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


# ---------------------------------------------------------------- 三阶段
def extract_pass(lang, is_cn, persist=True):
    """遍历原索引：reconstruct 现有向量 + 摘要句子切分。persist 时写暂存。"""
    src = META_INDEX_PATHS[lang]
    stage = src + SUFFIX
    ex_file = os.path.join(stage, "_extracted.pkl")
    if os.path.exists(ex_file):
        with open(ex_file, "rb") as f:
            extracted = pickle.load(f)
        print(f"[{lang}] 复用提取: {len(extracted):,} 句子", flush=True)
        return len(extracted)

    idx, d, i2d = load_db(src)
    N = idx.ntotal
    t0 = time.time()
    whole_vecs = np.zeros((N, idx.d), dtype=np.float32)
    extracted, n_noabs = [], 0
    for i in range(N):
        doc = d[i2d[i]]
        whole_vecs[i] = idx.reconstruct(i)          # 现有向量原样保留（零回归）
        abs_text = extract_abstract(doc)
        if abs_text:
            for s in split_sentences(abs_text, is_cn):
                extracted.append((i, s))
        else:
            n_noabs += 1
        if (i + 1) % 20000 == 0:
            print(f"[{lang}] 遍历 {i+1}/{N}, 句子 {len(extracted):,}", flush=True)

    stats = {"N": N, "no_abstract": n_noabs, "sentences": len(extracted),
             "avg_sent": round(len(extracted) / max(1, N - n_noabs), 1)}
    print(f"[{lang}] 提取完成 {stats} ({(time.time()-t0)/60:.1f}min)", flush=True)
    if persist:
        os.makedirs(stage, exist_ok=True)
        np.save(os.path.join(stage, "_whole_vecs.npy"), whole_vecs)
        with open(ex_file, "wb") as f:
            pickle.dump(extracted, f)
        with open(ex_file + ".stats", "wb") as f:
            pickle.dump(stats, f)
    del idx, d, whole_vecs
    gc.collect()
    return len(extracted)


def embed_pass(lang):
    """句子重嵌入（批次 npy 断点续跑）。"""
    src = META_INDEX_PATHS[lang]
    stage = src + SUFFIX
    vec_dir = os.path.join(stage, "_sentvecs")
    os.makedirs(vec_dir, exist_ok=True)
    with open(os.path.join(stage, "_extracted.pkl"), "rb") as f:
        extracted = pickle.load(f)
    sent_texts = [s for _, s in extracted]
    B = EMBED_BATCH[lang]
    nbatch = (len(sent_texts) + B - 1) // B
    m = get_model()
    done = 0
    t0 = time.time()
    for bi in range(nbatch):
        bfile = os.path.join(vec_dir, f"b{bi:06d}.npy")
        if os.path.exists(bfile):
            done += 1
            continue
        chunk = sent_texts[bi * B:(bi + 1) * B]
        vecs = np.asarray(m.encode(chunk, batch_size=len(chunk),
                                   normalize_embeddings=True), dtype=np.float32)
        if not np.isfinite(vecs).all():
            raise RuntimeError(f"[{lang}] batch {bi} 非有限向量，中止")
        np.save(bfile, vecs)
        done += 1
        if done % 100 == 0:
            print(f"[{lang}] 嵌入 {done}/{nbatch} batches "
                  f"({min((bi+1)*B, len(sent_texts))}/{len(sent_texts)}, "
                  f"{(time.time()-t0)/60:.0f}min)", flush=True)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    if done < nbatch:
        raise RuntimeError(f"[{lang}] 批次未完成 {done}/{nbatch}")
    return nbatch


def combine_pass(lang):
    """合并整篇向量+句子向量，写 index.faiss + index.pkl。"""
    src = META_INDEX_PATHS[lang]
    stage = src + SUFFIX
    if os.path.exists(os.path.join(stage, "index.faiss")):
        print(f"[{lang}] index.faiss 已存在，跳过", flush=True)
        return
    idx, d, i2d = load_db(src)
    N = idx.ntotal
    with open(os.path.join(stage, "_extracted.pkl"), "rb") as f:
        extracted = pickle.load(f)
    vec_dir = os.path.join(stage, "_sentvecs")
    batches = sorted(f for f in os.listdir(vec_dir) if f.endswith(".npy"))
    nbatch = (len(extracted) + EMBED_BATCH[lang] - 1) // EMBED_BATCH[lang]
    if len(batches) != nbatch:
        raise RuntimeError(f"[{lang}] 批次不全 {len(batches)}/{nbatch}")
    sent_vecs = np.vstack([np.load(os.path.join(vec_dir, f)) for f in batches])
    whole_vecs = np.load(os.path.join(stage, "_whole_vecs.npy"))
    assert sent_vecs.shape[0] == len(extracted), (sent_vecs.shape, len(extracted))
    assert whole_vecs.shape[0] == N
    all_vecs = np.vstack([whole_vecs, sent_vecs])
    flat = faiss.IndexFlatIP(idx.d)
    flat.add(all_vecs)

    # docstore：整篇保持原文档（无 _sent 标记）在前，句子条目（_sent=True）在后
    docs = [d[i2d[i]] for i in range(N)]
    for (pidx, s) in extracted:
        pdoc = d[i2d[pidx]]
        md = dict(pdoc.metadata or {})
        md["_sent"] = True
        docs.append(Document(page_content=s, metadata=md))
    assert len(docs) == len(all_vecs), (len(docs), len(all_vecs))

    faiss.write_index(flat, os.path.join(stage, "index.faiss"))
    ds = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
    with open(os.path.join(stage, "index.pkl"), "wb") as f:
        pickle.dump((ds, {i: i for i in range(len(docs))}), f)
    print(f"[{lang}] 完成 {stage}: {len(docs):,} docs, ntotal={flat.ntotal}", flush=True)
    del idx, d, docs, all_vecs
    gc.collect()


# ---------------------------------------------------------------- 备份与替换
def do_backup():
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    for lang, src in META_INDEX_PATHS.items():
        dst = os.path.join(BACKUP_ROOT, os.path.basename(src))
        if os.path.exists(dst):
            print(f"[backup] skip(已存在) {dst}", flush=True)
            continue
        shutil.copytree(src, dst)
        print(f"[backup] {src} -> {dst}", flush=True)
    print("[backup] done", flush=True)


def do_swap():
    for lang, orig in META_INDEX_PATHS.items():
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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "build", "backup", "swap"])
    args = ap.parse_args()

    if args.mode == "backup":
        do_backup()
        return
    if args.mode == "swap":
        do_swap()
        return
    if args.mode == "check":
        for lang, is_cn in [("english", False), ("chinese", True)]:
            extract_pass(lang, is_cn, persist=False)
        print("[check] DONE")
        return

    # build
    for lang, is_cn in [("english", False), ("chinese", True)]:
        t0 = time.time()
        n = extract_pass(lang, is_cn)
        print(f"[{lang}] 提取 {n:,} 句子 ({(time.time()-t0)/60:.1f}min)", flush=True)
        t0 = time.time()
        embed_pass(lang)
        print(f"[{lang}] 嵌入完成 ({(time.time()-t0)/60:.1f}min)", flush=True)
        combine_pass(lang)
    print("[build] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
