#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""zh-v7: 构建中文 BM25 IDF（jieba 分词）。

遍历中文全文索引 zh_fine/zh_std/zh_large 的 docstore，提取 chunk 文本，
用 BM25Scorer(lang="zh") 分词统计 df/avgdl，输出 zh_bm25_idf.pkl。
多进程并行分词；产出后由 retriever._get_bm25_zh() 懒加载。

用法: python build_zh_bm25_idf.py
"""
import sys, os, pickle, math, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FULLTEXT_INDEX_PATHS
from bm25_scorer import BM25Scorer

ZH_KEYS = ["zh_fine", "zh_std", "zh_large"]
MIN_LEN = 50


def extract_texts():
    seen, texts = set(), []
    for key in ZH_KEYS:
        path = FULLTEXT_INDEX_PATHS.get(key)
        if not path:
            print("NO PATH", key, flush=True)
            continue
        pkl = os.path.join(path, "index.pkl")
        if not os.path.exists(pkl):
            print("NO PKL", pkl, flush=True)
            continue
        t0 = time.time()
        with open(pkl, "rb") as f:
            obj = pickle.load(f)
        docstore = obj[0] if isinstance(obj, tuple) else obj
        d = getattr(docstore, "_dict", docstore)
        n = 0
        for k, v in d.items():
            txt = v.page_content if hasattr(v, "page_content") else str(v)
            th = txt[:200]
            if len(txt) > MIN_LEN and th not in seen:
                seen.add(th)
                texts.append(txt)
                n += 1
        del d, obj, docstore
        print(f"{key}: +{n} unique (total {len(texts)}, {round(time.time()-t0,1)}s)", flush=True)
    print("TOTAL unique:", len(texts), flush=True)
    return texts


def _worker(txts):
    sc = BM25Scorer(k1=1.2, b=0.75, lang="zh")  # 首次调用触发 jieba 懒加载
    stop = sc._stopwords
    df, lens = Counter(), []
    for t in txts:
        toks = [x for x in sc._tokenize_for_lang(t) if x not in stop]
        for x in set(toks):
            df[x] += 1
        lens.append(len(toks))
    return df, lens


def main():
    t0 = time.time()
    texts = extract_texts()
    N = len(texts)
    if N == 0:
        print("no texts extracted, abort", flush=True)
        return 1
    import multiprocessing as mp
    nw = min(16, mp.cpu_count())
    shards = [texts[i::nw] for i in range(nw)]
    del texts
    print(f"starting {nw} workers on {N} texts", flush=True)
    with mp.Pool(nw) as pool:
        res = pool.map(_worker, shards)
    df, lens = Counter(), []
    for d, l in res:
        df.update(d)
        lens.extend(l)
    avgdl = sum(lens) / max(N, 1)
    sc = BM25Scorer(k1=1.2, b=0.75, lang="zh")
    sc.N, sc.avgdl = N, avgdl
    for t, dft in df.items():
        sc.idf[t] = math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)
    sc.save("zh_bm25_idf.pkl")
    sz = os.path.getsize("zh_bm25_idf.pkl") / 1024 / 1024
    print(f"DONE N={N} vocab={len(sc.idf)} avgdl={avgdl:.1f} size={sz:.1f}MB elapsed={round(time.time()-t0,1)}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
