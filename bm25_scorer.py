# -*- coding: utf-8 -*-
"""
bm25_scorer.py
==============
BM25-based lexical scorer to replace simple token overlap in retriever.py.

学术依据 / Academic Basis:
    Robertson & Zaragoza (2009). The Probabilistic Relevance Framework:
    BM25 and Beyond. Foundations and Trends in Information Retrieval.

核心改进 / Key Improvements over simple overlap:
    1. IDF weighting: rare terms (gene names like "DW3") get higher weight
       than frequent terms ("the", "gene").
    2. Term frequency saturation: f(k1+1)/(f+k1) prevents one term from
       dominating the score.
    3. Document length normalization: b=0.75 prevents longer chunks from
       having an unfair advantage.

评分公式 / Scoring Formula:
    BM25(q, d) = Σ IDF(t) × [f(t,d) × (k1+1)] / [f(t,d) + k1 × (1-b + b×|d|/avgdl)]
    where:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        f(t,d) = term frequency of t in document d
        |d| = document length in tokens
        avgdl = average document length across corpus
        k1 = 1.2  (term frequency saturation parameter)
        b  = 0.75 (length normalization parameter)

使用方式 / Usage:
    # Offline: Build IDF on server
    python bm25_scorer.py --build --output bm25_idf.pkl

    # Online: Load and score
    scorer = BM25Scorer.load("bm25_idf.pkl")
    score = scorer.score(query_text, chunk_text)
"""

import os
import re
import json
import pickle
import argparse
import logging
from typing import Dict, List, Set, Optional
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# BM25 Scorer
# ════════════════════════════════════════════════════════════════

# ── zh-v7: 中文停用词 + jieba 懒加载 ─────────────────────────────
# 中文 BM25 需要 jieba 分词；懒加载避免拖慢服务冷启动。
_ZH_STOPWORDS = {
    "的", "了", "在", "是", "和", "与", "及", "或", "之", "于", "被", "将",
    "为", "以", "中", "对", "等", "并", "而", "从", "到", "向", "由", "把",
    "让", "就", "都", "很", "也", "其", "这", "那", "个", "种", "一个",
    "一种", "进行", "通过", "主要", "这些", "那些", "什么", "怎么", "如何",
    "哪些", "多少", "是否", "以及", "但是", "因此", "本文", "研究", "分析",
    "结果", "表明", "可以", "需要", "认为", "相关", "影响", "作用", "不同",
    "其中", "之间", "或者", "还是",
}

_JIEBA = None


def _get_jieba():
    global _JIEBA
    if _JIEBA is None:
        import jieba  # 懒加载：不拖慢启动
        _JIEBA = jieba
    return _JIEBA


class BM25Scorer:
    """
    BM25 scorer with pre-computed IDF over the chunk corpus.

    Parameters
    ----------
    k1 : float
        Term frequency saturation. Higher = less saturation (raw TF matters more).
        Default 1.2 is standard in IR literature.
    b : float
        Length normalization. 0 = no normalization, 1 = full normalization.
        Default 0.75 is standard.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, lang: str = "en"):
        self.k1 = k1
        self.b = b
        self.lang = lang  # "en" 用英文 tokenizer；"zh" 用 jieba 中文分词
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0  # total number of documents in corpus
        self._stopwords: Set[str] = self._load_stopwords()

    # ── Tokenization ───────────────────────────────────────────

    @staticmethod
    def _load_stopwords() -> Set[str]:
        """Minimal stopword set for scientific text. """
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "about",
            "between", "under", "over", "above", "below", "up", "down",
            "and", "or", "not", "but", "if", "then", "else", "when",
            "this", "that", "these", "those", "it", "its", "we", "they",
            "he", "she", "which", "who", "whom", "whose", "also",
            "than", "more", "less", "very", "too", "just", "only",
            "such", "each", "all", "both", "few", "most", "other",
            "some", "any", "no", "nor", "so", "thus", "therefore",
            "however", "although", "because", "since", "while", "where",
            "how", "what", "here", "there", "et", "al",
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase word tokens, keeping gene names intact."""
        text = (text or "").lower().strip()
        # Normalize: keep alphanumeric, hyphens, underscores (gene names!)
        text = re.sub(r"[^a-z0-9\s\-_\.]", " ", text)
        # Split
        tokens = [t.strip(".-_") for t in text.split() if t.strip(".-_")]
        return [t for t in tokens if len(t) >= 2]  # filter single chars

    @staticmethod
    def tokenize_zh(text: str) -> List[str]:
        """中英混合分词（zh-v7）：中文段 jieba.cut，英文/数字串整体保留小写。

        中文 chunk 常含英文残留（基因名 DW3/SbTFL1、术语），故英文/数字段
        走与原 tokenize 相同的正则逻辑（基因名不断开），中文段 jieba 切词
        并滤除中文停用词。查询与语料双侧都必须用同一分词。
        """
        text = (text or "").strip()
        if not text:
            return []
        jieba = _get_jieba()
        out: List[str] = []
        for seg in re.split(r"([一-鿿]+)", text):
            if not seg:
                continue
            if seg[0] >= "一" and seg[0] <= "鿿":
                # 纯中文段 → jieba 切词
                for w in jieba.cut(seg):
                    w = w.strip()
                    if w and w not in _ZH_STOPWORDS:
                        out.append(w)
            else:
                # 英文/数字段 → 原 tokenize 逻辑
                seg_l = re.sub(r"[^a-z0-9\s\-_\.]", " ", seg.lower())
                for t in seg_l.split():
                    t = t.strip(".-_")
                    if len(t) >= 2:
                        out.append(t)
        return out

    def _tokenize_for_lang(self, text: str) -> List[str]:
        """按 scorer 语言分发分词。英文路径与原有 tokenize 逐字节一致。"""
        return self.tokenize_zh(text) if self.lang == "zh" else self.tokenize(text)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Basic text normalization (mirrors utils.norm_text)."""
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    # ── Corpus Building (Offline) ───────────────────────────────

    def build_from_texts(self, texts: List[str], stopword_filter: bool = True):
        """
        Build IDF dictionary from a list of document texts.

        Parameters
        ----------
        texts : List[str]
            All chunk texts in the corpus.
        stopword_filter : bool
            Whether to exclude stopwords from IDF computation.
        """
        self.N = len(texts)
        if self.N == 0:
            logger.warning("Empty corpus provided to BM25Scorer.build_from_texts()")
            return

        df: Counter = Counter()        # document frequency
        doc_lengths: List[int] = []

        for i, text in enumerate(texts):
            tokens = self._tokenize_for_lang(text)
            if stopword_filter:
                tokens = [t for t in tokens if t not in self._stopwords]
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] += 1
            doc_lengths.append(len(tokens))

            if (i + 1) % 50000 == 0:
                logger.info(f"  Processed {i+1}/{self.N} documents...")

        self.avgdl = np.mean(doc_lengths) if doc_lengths else 1.0

        # Compute IDF: Robertson-Sparck Jones formula with smoothing
        for t, dft in df.items():
            self.idf[t] = np.log((self.N - dft + 0.5) / (dft + 0.5) + 1.0)

        logger.info(f"BM25 IDF built: N={self.N}, vocab_size={len(self.idf)}, "
                     f"avgdl={self.avgdl:.1f}")

    def build_from_faiss_indexes(self, index_paths: List[str],
                                  model_path: str = None) -> int:
        """
        Build IDF by extracting all chunk texts from FAISS indexes.

        This method requires the FAISS indexes to be accessible and is
        designed to run on the production server.

        Parameters
        ----------
        index_paths : List[str]
            Paths to FAISS fulltext indexes (e.g., faiss_v3_english_fine).
        model_path : str, optional
            Path to embedding model (BGE-M3). If None, uses config.

        Returns
        -------
        int : total number of chunks processed.
        """
        all_texts: List[str] = []
        seen = set()

        # Import locally to avoid dependency when only loading IDF
        import sys
        import faiss
        from langchain_community.vectorstores import FAISS
        from config import MODEL_PATH as _default_model

        model_path = model_path or _default_model

        for idx_path in index_paths:
            logger.info(f"Loading FAISS index: {idx_path}")
            try:
                # Need embedding model to load FAISS
                # Use a lightweight approach: read docstore directly if possible
                store_path = Path(idx_path)
                if not store_path.exists():
                    logger.warning(f"Index not found: {idx_path}")
                    continue

                # FAISS.load_local requires embeddings, but we can read
                # the docstore pickle directly for text extraction
                import pickle as _pickle
                docstore_file = store_path / "index.pkl"
                if docstore_file.exists():
                    with open(docstore_file, "rb") as f:
                        docstore = _pickle.load(f)
                    # Iterate through docstore to extract texts
                    if hasattr(docstore, '_dict'):
                        for k, doc in docstore._dict.items():
                            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                            text_hash = text[:120]
                            if text_hash not in seen:
                                seen.add(text_hash)
                                all_texts.append(text)
                    elif hasattr(docstore, 'search'):
                        logger.info(f"  Docstore type: {type(docstore)}, trying alternative extraction")
                        # Fallback: iterate through all index_to_docstore_id mappings
                else:
                    logger.warning(f"  No index.pkl found in {idx_path}")
            except Exception as e:
                logger.error(f"Failed to extract texts from {idx_path}: {e}")
                continue

        logger.info(f"Extracted {len(all_texts)} unique chunks from {len(index_paths)} indexes")
        self.build_from_texts(all_texts)
        return len(all_texts)

    # ── Query-Time Scoring ─────────────────────────────────────

    def score(self, query: str, document: str, stopword_filter: bool = True) -> float:
        """
        Compute BM25 score for a query-document pair.

        Parameters
        ----------
        query : str
            Query text (user question + expanded keywords).
        document : str
            Chunk text.
        stopword_filter : bool
            Whether to filter stopwords from query and document tokens.

        Returns
        -------
        float
            Raw BM25 score. Higher = more relevant.
            NOT normalized — caller should normalize if needed.
        """
        if not self.idf:
            # IDF not built; fall back to simple overlap (backward compat)
            return self._simple_overlap_fallback(query, document)

        q_tokens = self._tokenize_for_lang(query)
        d_tokens = self._tokenize_for_lang(document)

        if stopword_filter:
            q_tokens = [t for t in q_tokens if t not in self._stopwords]
            d_tokens = [t for t in d_tokens if t not in self._stopwords]

        if not q_tokens or not d_tokens:
            return 0.0

        d_len = len(d_tokens)
        tf = Counter(d_tokens)

        score = 0.0
        for t in set(q_tokens):  # unique query terms only (standard BM25)
            idf_t = self.idf.get(t, 0.0)
            if idf_t <= 0.0:
                continue
            f_td = tf.get(t, 0)
            if f_td == 0:
                continue

            # BM25 term score
            numerator = f_td * (self.k1 + 1.0)
            denominator = f_td + self.k1 * (1.0 - self.b + self.b * d_len / max(1.0, self.avgdl))
            score += idf_t * numerator / denominator

        return score

    def score_normalized(self, query: str, document: str,
                          max_score: Optional[float] = None) -> float:
        """
        Compute BM25 score normalized to [0, 1] range.
        If max_score is provided, divides by it. Otherwise returns raw.
        """
        raw = self.score(query, document)
        if max_score and max_score > 0:
            return min(raw / max_score, 1.0)
        return raw

    def _simple_overlap_fallback(self, query: str, document: str) -> float:
        """Fallback when IDF is not available (same as old lexical overlap)."""
        q_tokens = set(self._tokenize_for_lang(query))
        c_tokens = set(self._tokenize_for_lang(document))
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / max(1, len(q_tokens))

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str):
        """Save IDF dictionary and parameters to disk."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "lang": self.lang,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "N": self.N,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 scorer saved to {path} (N={self.N}, lang={self.lang}, vocab={len(self.idf)})")

    @classmethod
    def load(cls, path: str) -> "BM25Scorer":
        """Load pre-computed BM25 scorer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        scorer = cls(k1=data["k1"], b=data["b"], lang=data.get("lang", "en"))
        scorer.idf = data["idf"]
        scorer.avgdl = data["avgdl"]
        scorer.N = data["N"]
        logger.info(f"BM25 scorer loaded from {path} (N={scorer.N}, lang={scorer.lang}, vocab={len(scorer.idf)})")
        return scorer


# ════════════════════════════════════════════════════════════════
# CLI: Build IDF from FAISS indexes on the server
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 IDF builder for SorGPT")
    parser.add_argument("--build", action="store_true", help="Build IDF dictionary")
    parser.add_argument("--output", type=str, default="bm25_idf.pkl", help="Output path")
    parser.add_argument("--index-dir", type=str,
                        default="/vol/sunjilin/website/data/agent",
                        help="Directory containing FAISS indexes")
    parser.add_argument("--indexes", nargs="*",
                        default=["faiss_v3_english_fine", "faiss_v3_english_std",
                                 "faiss_v3_english_large", "faiss_v3_english_para"],
                        help="FAISS index names to process")
    args = parser.parse_args()

    if args.build:
        index_paths = [os.path.join(args.index_dir, name) for name in args.indexes]
        existing = [p for p in index_paths if os.path.exists(p)]
        logger.info(f"Found {len(existing)}/{len(index_paths)} indexes: {existing}")

        scorer = BM25Scorer(k1=1.2, b=0.75)
        n = scorer.build_from_faiss_indexes(existing)
        scorer.save(args.output)
        logger.info(f"Done. {n} chunks processed, saved to {args.output}")
    else:
        logger.info("Use --build to build IDF, or import BM25Scorer for scoring.")
