# -*- coding: utf-8 -*-
"""
retriever.py
========================
【作用】
实现 SorGPT 的检索主逻辑，包括：
1. 元数据检索（metadata retrieval）
2. 全文检索（full-text retrieval）
3. 根据问题类型自动选择不同粒度索引库
4. 对检索结果附加 BM25 lexical 打分（v2: BM25 替代简单 token overlap）

【改进记录】
- v2: BM25 scoring 替代 _simple_lexical_overlap
  BM25 带有 IDF 加权和词频饱和，解决了原实现中所有词等权重的问题。
  参考文献: Robertson & Zaragoza (2009).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

import faiss
from langchain_community.vectorstores import FAISS

from config import (
    META_INDEX_PATHS, FULLTEXT_INDEX_PATHS, TOP_META_K, TOP_CHUNK_K,
    COUNT_QUERY_FETCH_K, QUERY_TYPE_TO_INDEXES,
    DEFAULT_NPROBE, USE_FAISS_GPU, GPU_DEVICE
)
from utils import basename_lower, norm_text
from embeddings import BgeEmbeddingsWrapper

# BM25 scorer - lazy loaded
_bm25_scorer: Optional["BM25Scorer"] = None
_BM25_IDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_idf.pkl")
_BM25_WEIGHT = 0.25  # λ: BM25 weight in final_score (grid search optimal on eval set)

def _get_bm25() -> Optional["BM25Scorer"]:
    """Lazy-load BM25 scorer with pre-computed IDF."""
    global _bm25_scorer
    if _bm25_scorer is None:
        if os.path.exists(_BM25_IDF_PATH):
            from bm25_scorer import BM25Scorer
            _bm25_scorer = BM25Scorer.load(_BM25_IDF_PATH)
        else:
            # IDF not built yet; will fall back to simple overlap
            return None
    return _bm25_scorer


@dataclass
class MetaPaper:
    filename: str = ""
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    doi: str = ""
    score: float = 0.0
    meta_text: str = ""
    lang: str = ""


@dataclass
class ChunkHit:
    source: str
    content: str
    raw_score: float
    final_score: float
    granularity: str
    lang: str
    section_type: str = ""


class Retriever:
    """
    SorGPT 检索器。
    包含元数据检索 + 全文检索。
    """

    def __init__(self, embed_model: BgeEmbeddingsWrapper,
                 citation_map: Dict[str, Dict[str, str]]):
        self.embed_model = embed_model
        self.citation_map = citation_map

        # 加载元数据索引（保持原来的 LangChain 检索方式，库小，CPU 足够）
        self.meta_dbs = {
            lang: FAISS.load_local(
                path,
                self.embed_model,
                allow_dangerous_deserialization=True
            )
            for lang, path in META_INDEX_PATHS.items()
        }

        # 加载全文索引（新增：CPU + 可选 GPU）
        self.fulltext_dbs = {}
        for key, path in FULLTEXT_INDEX_PATHS.items():
            store = FAISS.load_local(
                path,
                self.embed_model,
                allow_dangerous_deserialization=True
            )
            cpu_index = store.index

            # 设置 CPU nprobe
            try:
                if hasattr(cpu_index, "nprobe"):
                    cpu_index.nprobe = DEFAULT_NPROBE
                else:
                    ivf = faiss.extract_index_ivf(cpu_index)
                    if ivf is not None:
                        ivf.nprobe = DEFAULT_NPROBE
            except Exception:
                pass

            gpu_index = None
            using_gpu = False
            gpu_res = None

            if USE_FAISS_GPU:
                try:
                    gpu_res = faiss.StandardGpuResources()
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = False
                    gpu_index = faiss.index_cpu_to_gpu(gpu_res, GPU_DEVICE, cpu_index, co)

                    try:
                        if hasattr(gpu_index, "nprobe"):
                            gpu_index.nprobe = DEFAULT_NPROBE
                        else:
                            ivf = faiss.extract_index_ivf(gpu_index)
                            if ivf is not None:
                                ivf.nprobe = DEFAULT_NPROBE
                    except Exception:
                        pass

                    using_gpu = True
                    print(f"[OK] fulltext {key} 已加载到 GPU")
                except Exception as e:
                    print(f"[WARN] fulltext {key} GPU 加载失败，回退 CPU: {e}")

            self.fulltext_dbs[key] = {
                "store": store,
                "cpu_index": cpu_index,
                "gpu_index": gpu_index,
                "using_gpu": using_gpu,
                "gpu_res": gpu_res,
            }

    def retrieve_metadata(self, user_query: str, en_keywords: str,
                          query_type: str, journal_filter: str = None) -> List[MetaPaper]:
        """
        元数据检索：对 metadata 库检索，找候选文献池。
        """
        hybrid_query = (f"{user_query}\nEnglish keywords: {en_keywords}"
                        if en_keywords else user_query)
        k = COUNT_QUERY_FETCH_K if query_type in ("count", "review", "gene_list") else TOP_META_K

        papers: List[MetaPaper] = []
        seen = set()

        for lang, db in self.meta_dbs.items():
            results = db.similarity_search_with_score(hybrid_query, k=k)
            for doc, score in results:
                md = doc.metadata or {}
                fname = norm_text(md.get("filename", "")) or norm_text(md.get("source", ""))
                uniq = basename_lower(fname) or md.get("title", "")
                if uniq in seen:
                    continue
                seen.add(uniq)
                papers.append(MetaPaper(
                    filename=fname,
                    title=md.get("title", ""),
                    authors=md.get("authors", ""),
                    journal=md.get("journal", ""),
                    year=md.get("year", ""),
                    doi=md.get("doi", ""),
                    score=float(score),
                    meta_text=doc.page_content,
                    lang=lang,
                ))

        papers.sort(key=lambda x: x.score)
        return papers[:k]

    def choose_indexes(self, query_type: str) -> List[str]:
        """
        根据问题类型选择全文索引库。
        """
        return QUERY_TYPE_TO_INDEXES.get(query_type, ["en_std"])

    def _simple_lexical_overlap(self, query: str, content: str) -> float:
        """
        [DEPRECATED v1] 简化版 lexical overlap，所有词等权重。
        保留以兼容未安装 BM25 IDF 时的 fallback。

        新版使用 _bm25_score() 替代。
        """
        q_tokens = set(norm_text(query).lower().split())
        c_tokens = set(norm_text(content).lower().split())
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / max(1, len(q_tokens))

    def _bm25_score(self, query: str, content: str) -> float:
        """
        [v2] BM25 scoring with IDF weighting.

        Falls back to simple overlap if BM25 IDF not available.
        Returns score in [0, 1] range for compatibility with the
        subtractive scoring framework.
        """
        bm25 = _get_bm25()
        if bm25 is None:
            # Fallback to simple overlap
            return self._simple_lexical_overlap(query, content)
        # Raw BM25 (unbounded) → normalized by corpus-specific max
        raw = bm25.score(query, content)
        # Clip to reasonable range; BM25 scores > 15 are extremely rare
        # for scientific text queries
        return min(raw / 15.0, 1.0)

    def retrieve_fulltext(self, user_query: str, en_keywords: str,
                          allowed_papers: List[MetaPaper],
                          query_type: str) -> List[ChunkHit]:
        """
        全文检索：
        1. 根据 query_type 选择库
        2. 只在 allowed_papers 范围内保留结果
        3. 计算 lexical overlap
        4. 输出 ChunkHit 列表供 reranker 使用
        """
        if query_type in {"locate", "count", "boundary"}:
            return []

        allowed = {basename_lower(p.filename) for p in allowed_papers if p.filename}
        hybrid_query = (f"{user_query}\nEnglish keywords: {en_keywords}"
                        if en_keywords else user_query)

        merged_hits: List[ChunkHit] = []
        seen = set()

        query_vec = self.embed_model.embed_query_np(hybrid_query)

        # Dynamic TOP_K: count/review types need more candidates to compensate for paper recall
        _dynamic_mult = 8 if query_type in ("count", "review", "gene_list") else 4
        chosen_indexes = self.choose_indexes(query_type)

        # Track content hashes for std/large dedup
        _content_hashes = {}  # source -> set of content prefixes

        for index_name in chosen_indexes:
            loaded = self.fulltext_dbs[index_name]
            store = loaded["store"]
            index = loaded["gpu_index"] if loaded["using_gpu"] and loaded["gpu_index"] is not None else loaded["cpu_index"]

            scores, ids = index.search(query_vec, TOP_CHUNK_K * _dynamic_mult)

            granularity = index_name.split("_")[-1]   # fine / std / large / para
            lang = index_name.split("_")[0]           # en

            for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
                if idx < 0:
                    continue

                docstore_id = store.index_to_docstore_id.get(idx)
                if docstore_id is None:
                    continue

                doc = store.docstore.search(docstore_id)
                if doc is None:
                    continue

                md = doc.metadata or {}
                src = md.get("source", "")
                src_key = basename_lower(src)

                if allowed and src_key not in allowed:
                    continue

                snippet_key = (src_key, doc.page_content[:120])
                if snippet_key in seen:
                    continue

                # v3: std/large dedup — if same source + >70% content overlap with existing chunk,
                # keep the one with better granularity match for this query_type
                content_preview = doc.page_content[:200]
                if src_key in _content_hashes:
                    is_dup = False
                    for prev_preview, prev_gran in _content_hashes[src_key]:
                        # Simple overlap check: shared words ratio
                        words_new = set(content_preview.lower().split())
                        words_old = set(prev_preview.lower().split())
                        if words_new and words_old:
                            overlap_ratio = len(words_new & words_old) / min(len(words_new), len(words_old))
                            if overlap_ratio > 0.7:
                                # Keep the granularity that better matches query_type
                                preferred_gran = {"review": "para", "mechanism": "para", "gene_list": "para"}.get(query_type, "fine")
                                if granularity == preferred_gran and prev_gran != preferred_gran:
                                    # Replace previous with this one
                                    merged_hits = [h for h in merged_hits if not (h.source == src_key and h.granularity == prev_gran)]
                                    _content_hashes[src_key] = [(p, g) for p, g in _content_hashes[src_key] if g != prev_gran]
                                    _content_hashes[src_key].append((content_preview, granularity))
                                else:
                                    is_dup = True
                                    break
                    if is_dup:
                        continue
                else:
                    _content_hashes[src_key] = []
                _content_hashes[src_key].append((content_preview, granularity))

                seen.add(snippet_key)

                # [v2] BM25 lexical scoring (IDF-weighted)
                lexical = self._bm25_score(hybrid_query, doc.page_content)

                # FAISS index.search returns cosine similarity, higher = better.
                # Negate to fit “smaller = better” sort convention.
                raw_score = -float(score)
                final_score = raw_score - _BM25_WEIGHT * lexical

                merged_hits.append(ChunkHit(
                    source=src,
                    content=doc.page_content,
                    raw_score=raw_score,
                    final_score=final_score,
                    granularity=granularity,
                    lang=lang,
                    section_type=md.get("section_type", ""),
                ))

        merged_hits.sort(key=lambda x: x.final_score)
        return merged_hits

    def group_hits_by_source(self, hits: List[ChunkHit]) -> Dict[str, List[ChunkHit]]:
        """
        按 source 分组，方便分析证据来源分布。
        """
        grouped = defaultdict(list)
        for h in hits:
            grouped[basename_lower(h.source)].append(h)
        return grouped
