# -*- coding: utf-8 -*-
"""
reranker.py  ── 引用次数 + 期刊权重混合版 (v2)
================================================
核心改动 (v2):
1. 新增 _citation_bonus_normalized(): 用 log(年均引用) 替代纯期刊名权重。
   学术依据: 引用次数是比期刊名更客观的论文质量信号。
   - 对数变换防止极端值垄断 (10000引 vs 50引 → log)
   - 年份归一化消除时间偏倚 (新论文引用少是正常的)
2. 渐进式过渡: quality_bonus = α × journal_bonus + (1-α) × citation_bonus
   默认 α=0.3，可在 config 中调整。
3. 未匹配到引用数据时自动回退期刊权重 (fallback)。
"""

import os
import sqlite3
import math
from typing import Dict, List, Optional

from config import FINAL_CONTEXT_K, EVIDENCE_LIMITS
from utils import get_journal_score, basename_lower
from metadata_loader import safe_get_ref_info
from retriever import ChunkHit

# ── 引用次数缓存路径 ─────────────────────────────────────────
_CITATION_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citation_cache.db")

# ── 渐进式过渡系数 (0=纯引用, 1=纯期刊) ──────────────────────
# 建议: 先用 0.5 跑一周观察，确认引用数据覆盖率和质量后降至 0.0
_JOURNAL_BLEND_ALPHA = 0.3  # 30% journal + 70% citation


# ════════════════════════════════════════════════════════════════
# 引用次数 bonus (新)
# ════════════════════════════════════════════════════════════════

def _normalize_doi(doi: str) -> str:
    """Normalize DOI for DB lookup."""
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    return doi


def _lookup_citation_count(doi: str) -> Optional[int]:
    """Look up citation count from local SQLite cache."""
    doi = _normalize_doi(doi)
    if not doi:
        return None
    try:
        conn = sqlite3.connect(_CITATION_DB_PATH)
        row = conn.execute(
            "SELECT citation_count FROM citation_cache WHERE doi=?",
            (doi,)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _citation_bonus_normalized(citation_count: Optional[int],
                                 publication_year: Optional[int] = None) -> float:
    """
    计算引用次数 bonus (年份归一化)。

    使用 log10(cites_per_year) 做非线性映射，确保:
    - 10000 引论文不会完全压倒 50 引论文 (对数压缩)
    - 2025 年发表的 20 引论文不会被 2010 年的 500 引论文不公平压制 (年份归一化)

    学术依据:
        Nogueira et al. (2019). Multi-Stage Document Ranking with BERT.
        Cites/year is a standard proxy for per-paper impact.
    """
    if citation_count is None or citation_count < 0:
        return 0.0

    # Year normalization: compute citations per year since publication
    if publication_year and publication_year > 2000:
        years_since_pub = max(1, 2026 - publication_year)
        cites_per_year = citation_count / years_since_pub
    else:
        # Year unknown: use raw count with conservative treatment
        cites_per_year = citation_count / 10.0  # assume ~10 years old

    if cites_per_year <= 0:
        return 0.0

    log_cpy = math.log10(cites_per_year)

    # Piecewise mapping: log10(cites/year) → bonus [0, 0.15]
    # Thresholds calibrated for plant biology (lower than CS/medicine):
    # A paper with 20+ cites/year is well-cited in plant science.
    if log_cpy >= 2.0:    return 0.15   # ≥100 cites/year: exceptional
    if log_cpy >= 1.7:    return 0.12   # ≥50 cites/year: very high impact
    if log_cpy >= 1.3:    return 0.09   # ≥20 cites/year: well-cited
    if log_cpy >= 1.0:    return 0.06   # ≥10 cites/year: solid impact
    if log_cpy >= 0.7:    return 0.03   # ≥5 cites/year: modest impact
    if log_cpy >= 0.3:    return 0.01   # ≥2 cites/year: niche
    return 0.0


# ════════════════════════════════════════════════════════════════
# 期刊权重 bonus (保留，作为 fallback)
# ════════════════════════════════════════════════════════════════

def _journal_bonus(journal_score: float) -> float:
    """
    [v1 保留] 期刊分级权重，作为引用数据不可用时的 fallback。
    注意: 这是保守的三档版本 (max 0.12)，非生产版放大版。
    """
    if journal_score >= 9.0:
        return 0.12
    if journal_score >= 7.5:
        return 0.07
    if journal_score >= 5.0:
        return 0.03
    return 0.0


def _quality_bonus(ref_info: Dict[str, str]) -> float:
    """
    [v2] 综合质量 bonus：新论文用期刊权重，成熟论文用引用次数。

    策略（按优先级）:
    1. 引用数据完全缺失 → 纯期刊权重 fallback
    2. 新论文（<3年 + 引用<5次）→ 纯期刊权重（引用数不可靠）
    3. 较新论文（<5年 + 引用<3次）→ 50%期刊 + 50%引用
    4. 成熟论文（≥5年 或 引用充足）→ 纯引用次数（更客观）

    学术依据:
        新论文尚未积累引用，期刊声誉是合理的预期质量信号。
        老论文引用少则是真实的质量信号，不应被期刊权重掩盖。
    """
    doi = ref_info.get("doi", "")
    journal = ref_info.get("journal", "")
    year_str = ref_info.get("year", "")

    # Parse publication year
    year = None
    if year_str:
        try:
            year = int(float(year_str))
        except (ValueError, TypeError):
            pass

    citation_count = _lookup_citation_count(doi)
    journal_score = get_journal_score(journal)
    j_bonus = _journal_bonus(journal_score)

    # Case 1: No citation data at all → pure journal fallback
    if citation_count is None:
        return j_bonus

    years_since_pub = max(1, 2026 - year) if year and year > 2000 else 10

    # Case 2: New paper (<3 years) with few citations → journal is more reliable
    if citation_count < 5 and years_since_pub <= 3:
        return j_bonus

    # Case 3: Relatively new (<5 years) with very few citations → blend
    if citation_count < 3 and years_since_pub <= 5:
        c_bonus = _citation_bonus_normalized(citation_count, year)
        return 0.5 * j_bonus + 0.5 * c_bonus

    # Case 4: Mature paper → pure citation-based bonus (objective)
    return _citation_bonus_normalized(citation_count, year)


# ════════════════════════════════════════════════════════════════
# Reranker 主类
# ════════════════════════════════════════════════════════════════

class Reranker:

    def __init__(self, citation_map: Dict[str, Dict[str, str]]):
        self.citation_map = citation_map

    def rerank(self, hits: List[ChunkHit], query_type: str) -> List[ChunkHit]:
        """重排：粒度 + section + 文献质量 (引用次数为主) 三维度打分。"""
        reranked = []
        for hit in hits:
            ref_info = safe_get_ref_info(hit.source, self.citation_map)

            # ── 粒度加成 ──
            gran_bonus = {
                "fine":  0.12,
                "std":   0.08,
                "large": 0.04,
                "para":  0.06,
            }.get(hit.granularity, 0.0)

            # 问题类型 × 粒度的交叉加成
            if query_type in {"review", "mechanism"} and hit.granularity == "para":
                gran_bonus = 0.20
            elif query_type in {"factoid", "gene_function"} and hit.granularity == "fine":
                gran_bonus = 0.18

            # ── 章节加成 ──
            section_bonus = {
                "abstract":     0.12,
                "results":      0.08,
                "discussion":   0.08,
                "introduction": 0.03,
                "methods":     -0.03,
                "references":  -0.50,
            }.get((hit.section_type or "").lower(), 0.0)

            # ── 文献质量加成 (v2: 引用次数 + 期刊 fallback) ──
            quality_bonus = _quality_bonus(ref_info)

            final_score = (
                hit.final_score
                - gran_bonus
                - section_bonus
                - quality_bonus
            )
            reranked.append(ChunkHit(
                source=hit.source,
                content=hit.content,
                raw_score=hit.raw_score,
                final_score=final_score,
                granularity=hit.granularity,
                lang=hit.lang,
                section_type=hit.section_type,
            ))

        reranked.sort(key=lambda x: x.final_score)
        return reranked

    def diversify_and_trim(
        self, hits: List[ChunkHit], query_type: str
    ) -> List[ChunkHit]:
        """多源去重 + 裁剪 + v3 BM25 density gate。"""
        limit = EVIDENCE_LIMITS.get(query_type, FINAL_CONTEXT_K)

        # v3: Paper-level BM25 density gate — 过滤掉只在1-2个chunk中擦边命中关键词、
        # 但整体不相关的论文（如bioenergy QTL on Chr07被误当作height gene on Chr07）
        if query_type in ("gene_list", "gene_function", "factoid", "qtl_gwas"):
            from collections import defaultdict as _dd
            import numpy as _np
            _paper_scores = _dd(list)
            for h in hits:
                _paper_scores[basename_lower(h.source)].append(max(0, getattr(h, 'bm25_score', 0)))
            
            _keep_papers = set()
            for _src, _scores in _paper_scores.items():
                _avg = _np.mean(_scores)
                _cnt = len(_scores)
                if query_type == "gene_list":
                    # 需要多chunk支撑 或 高BM25均值
                    if _avg >= 0.5 or _cnt >= 3:
                        _keep_papers.add(_src)
                else:
                    # gene_function / factoid / qtl_gwas: 1-2个chunk但BM25高即可
                    if _avg >= 0.3:
                        _keep_papers.add(_src)
            
            # 确保至少有 limit/2 篇论文，防止过度过滤
            if len(_keep_papers) < limit // 2:
                _sorted = sorted(_paper_scores.items(), key=lambda x: _np.mean(x[1]), reverse=True)
                for _src, _ in _sorted[:limit//2]:
                    _keep_papers.add(_src)
            
            hits = [h for h in hits if basename_lower(h.source) in _keep_papers]

        # 不同问题类型每篇最多多少 chunk
        if query_type in {"review", "gene_list"}:
            max_per_source = 1      # 综述类强制多样性
        elif query_type in {"mechanism", "qtl_gwas"}:
            max_per_source = 2
        else:
            max_per_source = 2      # factoid / gene_function

        by_source: Dict[str, int] = {}
        selected: List[ChunkHit] = []

        for hit in hits:
            key = basename_lower(hit.source)
            cnt = by_source.get(key, 0)
            if cnt >= max_per_source:
                continue
            selected.append(hit)
            by_source[key] = cnt + 1
            if len(selected) >= limit:
                break

        return selected
