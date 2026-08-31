# -*- coding: utf-8 -*-
import json
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
import json
from utils import basename_lower, norm_text
from metadata_loader import ZH_SKIP_JOURNALS
from embeddings import BgeEmbeddingsWrapper

# BM25 scorer - lazy loaded
_bm25_scorer: Optional["BM25Scorer"] = None
_BM25_IDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_idf.pkl")
_BM25_WEIGHT = 0.25  # λ: BM25 weight in final_score (grid search optimal on eval set)

# zh-v7: 中文 BM25（jieba 分词，独立中文 IDF）。与英文同模式懒加载。
_bm25_zh_scorer: Optional["BM25Scorer"] = None
_ZH_BM25_IDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zh_bm25_idf.pkl")
_ZH_BM25_WEIGHT = 0.15  # λ_zh 保守起步，A/B 0.15/0.20/0.25
_ZH_BM25_CAP = 30.0     # 中文 BM25 归一化上限（校准: 30 题中文 × 2000 采样 chunk，raw p95=24.1/p100=33.2）
_ZH_BM25_TOP = 100      # 每 zh 索引最多算 BM25 的向量序前 N 个存活候选（jieba 延迟控制）

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

def _get_bm25_zh() -> Optional["BM25Scorer"]:
    """[zh-v7] Lazy-load Chinese BM25 scorer (jieba). None → 中文路径保持纯向量。"""
    global _bm25_zh_scorer
    if _bm25_zh_scorer is None:
        if os.path.exists(_ZH_BM25_IDF_PATH):
            from bm25_scorer import BM25Scorer
            _bm25_zh_scorer = BM25Scorer.load(_ZH_BM25_IDF_PATH)
        else:
            return None
    return _bm25_zh_scorer


# ============ zh-v5: 中文检索查询重写 ============
# 问题: "介绍红缨子" 这类带泛化动词的中文查询，BGE-M3 查询嵌入被"介绍"带偏，
#       zh 索引检索不到红缨子论文；且 2-3 字品种名("红缨子")单独嵌入也不可靠。
# 做法: 剥离句首泛化框架词；若核心不足 4 汉字且无高粱语境词，补"高粱"锚点。
#       仅用于 zh 索引的检索 query，不改变传给 LLM 的原始问题；en 检索 query 不变。
_ZH_FRAMING = [
    "介绍一下", "请介绍", "介绍下", "说一下", "聊一聊", "我想了解", "我想知道",
    "说说", "讲讲", "谈谈", "简述", "概述", "描述", "讲解",
    "什么是", "是什么", "怎么样", "怎么", "怎样", "如何",
    "请问", "麻烦", "帮我", "关于", "介绍",
]

def _strip_zh_framing(s: str) -> str:
    """最长优先剥离句首泛化框架词。"""
    words = sorted(_ZH_FRAMING, key=len, reverse=True)
    changed = True
    while changed and s:
        changed = False
        for w in words:
            if s.startswith(w):
                s = s[len(w):].lstrip(" ，,、")
                changed = True
                break
    return s.strip(" ，,、")

_SG_CONTEXT = ("高粱", "高梁", "sorghum", "Sorghum")

def _build_zh_retrieval_query(user_query: str) -> str:
    """生成中文索引专用检索 query（英文查询原样返回）。"""
    cn = sum(1 for c in user_query if "一" <= c <= "鿿")
    if cn / max(len(user_query), 1) <= 0.15:
        return user_query
    core = _strip_zh_framing(user_query)
    if not core:
        return user_query
    core_cn = sum(1 for c in core if "一" <= c <= "鿿")
    if core_cn and core_cn <= 3 and not any(k in core for k in _SG_CONTEXT):
        return core + "高粱"
    return core


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
    bm25_score: float = 0.0  # zh-v7/P0-1: 归一化 BM25 分，供 reranker density gate 读取


class Retriever:
    """
    SorGPT 检索器。
    包含元数据检索 + 全文检索。
    """

    def __init__(self, embed_model: BgeEmbeddingsWrapper,
                 citation_map: Dict[str, Dict[str, str]]):
        self.embed_model = embed_model
        self.citation_map = citation_map
        # Load gene→paper index for metadata enrichment
        _gi_path = os.path.join(os.path.dirname(__file__), "gene_index.json")
        self.gene_index = {}
        if os.path.exists(_gi_path):
            try:
                with open(_gi_path) as _f:
                    self.gene_index = json.load(_f)
            except Exception:
                pass

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
        # zh-v5: 中文索引用重写后的检索 query（剥离泛化动词+短实体补高粱锚点）
        zh_query = _build_zh_retrieval_query(user_query)
        k = COUNT_QUERY_FETCH_K if query_type in ("count", "review", "gene_list") else TOP_META_K

        papers: List[MetaPaper] = []
        seen = set()

        # zh-v2: 判断查询语言。中文问题时英文 meta 索引加配额，扩大 allowed 英文文献池，
        # 供全文检索补充英文证据（株高/基础科学类中文语料不足）。
        _cn_chars = sum(1 for c in user_query if "一" <= c <= "鿿")
        _is_cn = _cn_chars / max(len(user_query), 1) > 0.15

        for lang, db in self.meta_dbs.items():
            _k = k
            if _is_cn and lang == "english":
                _k = max(k, k * 2)  # 英文 meta 多取，扩大英文候选
            _q = zh_query if (lang == "chinese" and zh_query != user_query) else hybrid_query
            results = db.similarity_search_with_score(_q, k=_k)
            for doc, score in results:
                md = doc.metadata or {}
                # 无关中文期刊（电影/生活/农业推广类）不入检索池
                if md.get("journal") in ZH_SKIP_JOURNALS:
                    continue
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

        if _is_cn:
            # zh-v2: 中文问题按语言配额保底：英文 meta 直接保底进池（不参与全局排序淘汰），
            # 避免被中文分数挤掉，确保 allowed 池含足够英文文献供全文补充。
            _en = sorted((p for p in papers if p.lang == "english"),
                         key=lambda x: x.score)[: max(k // 3, 80)]
            _zh = sorted((p for p in papers if p.lang == "chinese"),
                         key=lambda x: x.score)[: k]
            papers = _en + _zh
        else:
            papers.sort(key=lambda x: x.score)
            papers = papers[:k]

        # Gene index lookup: if query contains gene symbols, inject
        # matching papers from the gene index directly into the pool
        _gk = []
        for _w in user_query.replace(",", " ").replace("?", " ").split():
            _w = _w.strip('?!.,()[]{}"' + "'")
            if not _w: continue
            if "Sobic." in _w or "SbiHYZ." in _w:
                _gk.append(_w)
            elif len(_w) >= 3 and _w[0].isupper() and any(x.isupper() for x in _w[1:]):
                _gk.append(_w)
        if _gk and self.gene_index:
            _seen = {basename_lower(p.filename) or p.title for p in papers}
            for _paper_key, _gene_map in self.gene_index.items():
                _match = False
                for _g in _gk:
                    if _g in _gene_map or _g.lower() in str(_gene_map).lower():
                        _match = True
                        break
                if not _match: continue
                if _paper_key in _seen: continue
                _seen.add(_paper_key)
                # Create a MetaPaper with gene info in the title for downstream
                _gene_str = "; ".join(f"{k}={v}" for k, v in list(_gene_map.items())[:5])
                papers.append(MetaPaper(
                    filename=_paper_key,
                    title=f"[GeneIndex] {_gene_str}",
                    authors="", journal="", year="", doi="",
                    score=0.1,  # good score to rank well
                    meta_text=f"Gene index match: {_gene_str}",
                    lang="en"))

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

    def _bm25_score_zh(self, query: str, content: str) -> float:
        """[zh-v7] 中文 BM25（jieba 分词，独立中文 IDF），返回 [0,1]。

        zh_bm25_idf.pkl 缺失时返回 0 → 与现状纯向量排序完全一致（零退化）。
        """
        bm25 = _get_bm25_zh()
        if bm25 is None:
            return 0.0
        raw = bm25.score(query, content)
        return min(raw / _ZH_BM25_CAP, 1.0)

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

        # zh-v5: 中文全文索引用重写后的 query_vec，en 索引保持原 query_vec
        zh_query = _build_zh_retrieval_query(user_query)
        zh_query_vec = (self.embed_model.embed_query_np(zh_query)
                        if zh_query != user_query else None)

        merged_hits: List[ChunkHit] = []
        seen = set()

        query_vec = self.embed_model.embed_query_np(hybrid_query)

        # Dynamic TOP_K: count/review types need more candidates to compensate for paper recall
        _dynamic_mult = 8 if query_type in ("count", "review", "gene_list") else 4
        chosen_indexes = self.choose_indexes(query_type)

        # ── 按查询语言过滤索引库：中文问题只搜中文库，英文问题只搜英文库 ──
        chinese_chars = sum(1 for c in user_query if "一" <= c <= "鿿")
        is_cn = chinese_chars / max(len(user_query), 1) > 0.15
        if is_cn:
            # zh-v1: 中文问题以中文索引为主，英文索引自动补充（以事实为主）。
            # 英文补充索引用较小 K，避免英文 chunk 淹没中文证据。
            zh_ = [i for i in chosen_indexes if i.startswith("zh_")]
            en_ = [i for i in chosen_indexes if i.startswith("en_")]
            chosen_indexes = zh_ + en_
        else:
            chosen_indexes = [i for i in chosen_indexes if i.startswith("en_")]
        # 确保至少有一个索引被选中（fallback）
        if not chosen_indexes:
            chosen_indexes = self.choose_indexes(query_type)

        # Track content hashes for std/large dedup
        _content_hashes = {}  # source -> set of content prefixes

        for index_name in chosen_indexes:
            loaded = self.fulltext_dbs[index_name]
            store = loaded["store"]
            index = loaded["gpu_index"] if loaded["using_gpu"] and loaded["gpu_index"] is not None else loaded["cpu_index"]

            # zh-v7: 每个 zh 索引的 BM25 预算（jieba 延迟控制）
            zh_budget = _ZH_BM25_TOP if (is_cn and index_name.startswith("zh_")) else None

            # zh-v1: 混合检索时，英文补充索引只搜 TOP_CHUNK_K 个候选
            _search_k = TOP_CHUNK_K * _dynamic_mult
            if is_cn and index_name.startswith("en_"):
                _search_k = TOP_CHUNK_K
            _qvec = zh_query_vec if (zh_query_vec is not None and index_name.startswith("zh_")) else query_vec
            scores, ids = index.search(_qvec, _search_k)

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
                # zh-v1: 中文混合池统一用向量相似度（lexical=0）：
                #   BM25 tokenize 会把 CJK 字符替换为空格而失效，中文 chunk 只残留英文
                #   token，与英文关键词匹配是噪声；且中英 BM25 尺度不一致会破坏混合池排序。
                # zh-v7: 中文路径引入 jieba 分词 + 独立中文 IDF，恢复 BM25 参与排序。
                #   - 中文 zh_ 索引：用 zh-v5 重写后的 zh_query 打中文 BM25，权重 _ZH_BM25_WEIGHT；
                #   - 英文路径(is_cn=False)：公式与参数完全不变，逐字节等价；
                #   - 中文问题搜到的英文补充索引：维持纯向量（lexical=0，防中英尺度干扰）。
                # zh-v7: jieba 分词慢，每 zh 索引只给向量序前 _ZH_BM25_TOP 个存活候选算 BM25。
                if is_cn and index_name.startswith("zh_") and zh_budget:
                    bm25_score = self._bm25_score_zh(zh_query, doc.page_content)
                    lex_weight = _ZH_BM25_WEIGHT
                    zh_budget -= 1
                elif not is_cn:
                    bm25_score = self._bm25_score(hybrid_query, doc.page_content)
                    lex_weight = _BM25_WEIGHT
                else:
                    bm25_score = 0.0
                    lex_weight = _BM25_WEIGHT

                # FAISS index.search returns cosine similarity, higher = better.
                # Negate to fit “smaller = better” sort convention.
                raw_score = -float(score)
                final_score = raw_score - lex_weight * bm25_score

                merged_hits.append(ChunkHit(
                    source=src,
                    content=doc.page_content,
                    raw_score=raw_score,
                    final_score=final_score,
                    granularity=granularity,
                    lang=lang,
                    section_type=md.get("section_type", ""),
                    bm25_score=bm25_score,
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
