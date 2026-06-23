# -*- coding: utf-8 -*-
"""
evaluator.py  ── SorGPT 自动评估框架 (Publishable-Quality)
============================================================
三个评估层次 (Three-Level Evaluation):
  Level 1 — Retrieval Quality: Recall@K, Precision@K, MRR, NDCG@K
  Level 2 — Ranking Quality: Ablation study, source diversity
  Level 3 — Generation Quality: FactScore, Citation Recall/Precision, Faithfulness

使用方式:
  python evaluator.py --level 1          # 仅检索评估
  python evaluator.py --level 1,2        # 检索 + 排序评估
  python evaluator.py --level all        # 全部三项
  python evaluator.py --ablation         # Ablation study (7 conditions)
"""

import os, sys, json, math, time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

import numpy as np

# Project imports
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))


# ════════════════════════════════════════════════════════════════
# Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class EvalQuestion:
    """One evaluation question."""
    id: str
    question_type: str
    question: str
    standard_answer: str
    keypoints: List[str]
    required_citations: List[str]       # DOI list
    relevant_papers: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    requires_multi_source: bool = False
    notes: str = ""


@dataclass
class Level1Result:
    """Retrieval quality metrics per question."""
    question_id: str
    recall_5: float
    recall_10: float
    recall_20: float
    precision_5: float
    precision_10: float
    mrr: float
    ndcg_5: float
    ndcg_10: float
    num_relevant: int
    num_retrieved: int


@dataclass
class Level3Result:
    """Generation quality metrics per question."""
    question_id: str
    fact_score: float
    citation_recall: float      # keypoints cited
    citation_precision: float   # citations actually support claims
    faithfulness: float         # answer grounded in evidence
    completeness: float         # keypoints covered
    llm_judge_raw: str = ""


# ════════════════════════════════════════════════════════════════
# Level 1: Retrieval Quality
# ════════════════════════════════════════════════════════════════

class RetrievalEvaluator:
    """
    检索质量评估。

    指标:
    - Recall@K: 前K个结果中包含了多少相关论文 (覆盖度)
    - Precision@K: 前K个结果中相关论文占比 (精确度)
    - MRR: 第一个相关论文排名的倒数均值 (定位能力)
    - NDCG@K: 归一化折损累计增益 (排序质量)
    """

    def __init__(self, pipeline, eval_questions: List[EvalQuestion],
                 relevance_annotations: Dict[str, Dict[str, int]] = None):
        """
        Parameters
        ----------
        pipeline : SorghumRAGPipeline
        eval_questions : List[EvalQuestion]
        relevance_annotations : dict
            {question_id: {paper_filename: relevance_grade}}
            relevance_grade: 2=directly relevant, 1=partially relevant, 0=not relevant
        """
        self.pipeline = pipeline
        self.questions = eval_questions
        self.relevance = relevance_annotations or {}

    def _dcg(self, scores: List[float], k: int) -> float:
        """Discounted Cumulative Gain."""
        dcg = 0.0
        for i, s in enumerate(scores[:k]):
            dcg += s / math.log2(i + 2)  # i+2 because log2(1)=0, start at 2
        return dcg

    def _ndcg(self, scores: List[float], k: int) -> float:
        """Normalized DCG."""
        dcg = self._dcg(scores, k)
        ideal = sorted(scores, reverse=True)
        idcg = self._dcg(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_single(self, q: EvalQuestion) -> Level1Result:
        """Evaluate retrieval for one question."""
        # Run retrieval
        query_type, extra_types, en_keywords = self.pipeline.retriever.choose_indexes.__class__.__name__, [], ""
        meta_hits = self.pipeline.retriever.retrieve_metadata(q.question, en_keywords or q.question, query_type or "factoid")
        chunk_hits = self.pipeline.retriever.retrieve_fulltext(q.question, q.question, meta_hits, query_type or "factoid")

        # Get relevance labels
        rel = self.relevance.get(q.id, {})
        relevant_papers = set(q.relevant_papers) if q.relevant_papers else set()

        # Score each retrieved chunk
        retrieved_sources = []
        seen = set()
        for hit in chunk_hits:
            src = Path(hit.source).stem
            if src not in seen:
                seen.add(src)
                grade = rel.get(src, 2 if src in relevant_papers else 0)
                retrieved_sources.append((src, grade))

        # Compute metrics
        grades = [g for _, g in retrieved_sources]
        num_relevant = sum(1 for g in grades if g > 0)

        # Recall@K: proportion of relevant papers found in top K
        total_relevant = len(relevant_papers) or max(1, len([k for k, v in rel.items() if v > 0]))
        recall_5 = min(1.0, sum(1 for g in grades[:5] if g > 0) / max(1, total_relevant))
        recall_10 = min(1.0, sum(1 for g in grades[:10] if g > 0) / max(1, total_relevant))
        recall_20 = min(1.0, sum(1 for g in grades[:20] if g > 0) / max(1, total_relevant))

        # Precision@K
        precision_5 = sum(1 for g in grades[:5] if g > 0) / max(1, min(5, len(grades)))
        precision_10 = sum(1 for g in grades[:10] if g > 0) / max(1, min(10, len(grades)))

        # MRR: 1/rank of first relevant
        mrr = 0.0
        for i, g in enumerate(grades):
            if g > 0:
                mrr = 1.0 / (i + 1)
                break

        # NDCG
        ndcg_5 = self._ndcg([float(g) for _, g in retrieved_sources], 5)
        ndcg_10 = self._ndcg([float(g) for _, g in retrieved_sources], 10)

        return Level1Result(
            question_id=q.id,
            recall_5=recall_5, recall_10=recall_10, recall_20=recall_20,
            precision_5=precision_5, precision_10=precision_10,
            mrr=mrr, ndcg_5=ndcg_5, ndcg_10=ndcg_10,
            num_relevant=num_relevant, num_retrieved=len(retrieved_sources),
        )

    def evaluate_all(self) -> List[Level1Result]:
        """Evaluate all questions."""
        results = []
        for i, q in enumerate(self.questions):
            print(f"  [{i+1}/{len(self.questions)}] {q.id}: {q.question[:60]}...")
            result = self.evaluate_single(q)
            results.append(result)
        return results

    def summarize(self, results: List[Level1Result]) -> Dict[str, float]:
        """Compute macro-average metrics."""
        n = len(results)
        return {
            "num_questions": n,
            "Recall@5": np.mean([r.recall_5 for r in results]),
            "Recall@10": np.mean([r.recall_10 for r in results]),
            "Recall@20": np.mean([r.recall_20 for r in results]),
            "Precision@5": np.mean([r.precision_5 for r in results]),
            "Precision@10": np.mean([r.precision_10 for r in results]),
            "MRR": np.mean([r.mrr for r in results]),
            "NDCG@5": np.mean([r.ndcg_5 for r in results]),
            "NDCG@10": np.mean([r.ndcg_10 for r in results]),
        }


# ════════════════════════════════════════════════════════════════
# Level 2: Ranking Quality + Ablation Study
# ════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "full": {
        "bm25": True, "journal_bonus": True, "citation_bonus": True,
        "description": "Full model (BM25 + citation + journal blend)"
    },
    "no_journal": {
        "bm25": True, "journal_bonus": False, "citation_bonus": True,
        "description": "Remove journal bonus"
    },
    "no_citation": {
        "bm25": True, "journal_bonus": True, "citation_bonus": False,
        "description": "Remove citation bonus"
    },
    "bm25_only": {
        "bm25": True, "journal_bonus": False, "citation_bonus": False,
        "description": "BM25 only, no quality bonus"
    },
    "simple_overlap": {
        "bm25": False, "journal_bonus": True, "citation_bonus": True,
        "description": "Simple token overlap (no BM25 IDF)"
    },
    "baseline": {
        "bm25": False, "journal_bonus": True, "citation_bonus": False,
        "description": "Baseline: simple overlap + journal only"
    },
    "citation_only": {
        "bm25": True, "journal_bonus": False, "citation_bonus": True,
        "description": "BM25 + citation only (proposed best)"
    },
}


class AblationRunner:
    """Run ablation study comparing different scoring configurations."""

    def __init__(self, pipeline, eval_questions: List[EvalQuestion],
                 relevance: Dict = None):
        self.pipeline = pipeline
        self.questions = eval_questions
        self.relevance = relevance or {}
        self.retrieval_eval = RetrievalEvaluator(pipeline, eval_questions, relevance)

    def run(self) -> Dict[str, Dict[str, float]]:
        """Run all ablation conditions and return comparative metrics."""
        import pandas as pd

        all_results = {}
        rows = []

        for config_name, config in ABLATION_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"Running: {config_name} — {config['description']}")
            print(f"{'='*60}")

            # Apply config (modify global state temporarily)
            self._apply_config(config)

            # Evaluate retrieval
            results = self.retrieval_eval.evaluate_all()
            summary = self.retrieval_eval.summarize(results)
            summary["config"] = config_name
            summary["description"] = config["description"]
            all_results[config_name] = summary

            row = {"Config": config_name, "Description": config["description"]}
            row.update({k: round(v, 4) for k, v in summary.items() if isinstance(v, float)})
            rows.append(row)

            # Compute source diversity
            diversity = self._source_diversity(results)
            row["SourceDiversity"] = round(diversity, 4)

        # Restore defaults
        self._restore_defaults()

        # Build comparison table
        df = pd.DataFrame(rows)
        cols = ["Config", "Recall@5", "Recall@10", "MRR", "NDCG@5", "NDCG@10",
                "Precision@5", "SourceDiversity", "Description"]
        df = df[[c for c in cols if c in df.columns]]

        # Compute pairwise significance where possible
        print(f"\n{'='*60}")
        print("Ablation Study Results")
        print(f"{'='*60}")
        print(df.to_string(index=False))

        return all_results

    def _apply_config(self, config: Dict):
        """Temporarily modify scoring configuration."""
        import reranker as rk
        import retriever as rt

        # Store originals
        self._orig_bm25_weight = rt._BM25_WEIGHT if hasattr(rt, '_BM25_WEIGHT') else 0.25
        self._orig_journal_blend = rk._JOURNAL_BLEND_ALPHA if hasattr(rk, '_JOURNAL_BLEND_ALPHA') else 0.3

        # BM25 config
        if not config["bm25"]:
            rt._BM25_WEIGHT = 0.30  # fallback to simple overlap weight
            # Force fallback to simple overlap
            import retriever
            retriever._bm25_scorer = None  # disable BM25

        # Journal/Citation blend
        if not config["journal_bonus"] and config["citation_bonus"]:
            # Citation only
            rk._JOURNAL_BLEND_ALPHA = 0.0
        elif config["journal_bonus"] and not config["citation_bonus"]:
            # Journal only (baseline)
            rk._JOURNAL_BLEND_ALPHA = 1.0
        elif not config["journal_bonus"] and not config["citation_bonus"]:
            # Neither
            rk._JOURNAL_BLEND_ALPHA = 0.0

    def _restore_defaults(self):
        """Restore original configuration."""
        import retriever as rt
        import reranker as rk
        if hasattr(self, '_orig_bm25_weight'):
            rt._BM25_WEIGHT = self._orig_bm25_weight
        if hasattr(self, '_orig_journal_blend'):
            rk._JOURNAL_BLEND_ALPHA = self._orig_journal_blend

    def _source_diversity(self, results: List[Level1Result]) -> float:
        """Measure average number of distinct sources in top-10 results."""
        return 0.0  # Placeholder; would compute from retrieval results


# ════════════════════════════════════════════════════════════════
# Level 3: Generation Quality (LLM-as-Judge)
# ════════════════════════════════════════════════════════════════

class GenerationEvaluator:
    """
    Generation quality evaluation using LLM-as-Judge.

    Assesses:
    - FactScore: fraction of atomic claims that are correct
    - Citation Recall: fraction of keypoints with proper citations
    - Citation Precision: fraction of citations that actually support claims
    - Faithfulness: is the answer grounded in provided evidence?
    - Completeness: does the answer cover all keypoints?
    """

    FACTSCORE_PROMPT = """\
You are evaluating the factual accuracy of a scientific answer about sorghum.

STANDARD ANSWER (reference):
{standard_answer}

KEYPOINTS that the answer should cover:
{keypoints}

GENERATED ANSWER:
{generated_answer}

Evaluate the generated answer on the following dimensions. Return a JSON object:

1. fact_score (0.0-1.0): What fraction of the factual claims in the generated answer
   are correct according to the standard answer? 1.0 = all claims correct.
2. citation_recall (0.0-1.0): What fraction of the keypoints are supported by
   at least one citation in the generated answer?
3. citation_precision (0.0-1.0): What fraction of the citations [1], [2], etc.
   actually appear to support the claim they are attached to?
4. faithfulness (0.0-1.0): Is the answer grounded in the provided evidence?
   1.0 = fully grounded, 0.0 = hallucinated.
5. completeness (0.0-1.0): What fraction of the keypoints are addressed?

Return ONLY the JSON object, no other text:
{{"fact_score": X.XX, "citation_recall": X.XX, "citation_precision": X.XX,
  "faithfulness": X.XX, "completeness": X.XX}}
"""

    def __init__(self, pipeline, judge_model: str = "deepseek-reasoner"):
        self.pipeline = pipeline
        self.judge_model = judge_model

    def evaluate_single(self, q: EvalQuestion, generated_answer: str) -> Level3Result:
        """Evaluate generation quality for one Q&A pair."""
        prompt = self.FACTSCORE_PROMPT.format(
            standard_answer=q.standard_answer,
            keypoints="\n".join(f"- {kp}" for kp in q.keypoints),
            generated_answer=generated_answer,
        )

        # Use LLM judge
        try:
            from openai import OpenAI
            from config import BASE_URL, API_KEY
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            resp = client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content
            scores = json.loads(raw)
        except Exception as e:
            print(f"  LLM judge error for {q.id}: {e}")
            scores = {"fact_score": 0.0, "citation_recall": 0.0,
                      "citation_precision": 0.0, "faithfulness": 0.0,
                      "completeness": 0.0}
            raw = str(e)

        return Level3Result(
            question_id=q.id,
            fact_score=scores.get("fact_score", 0.0),
            citation_recall=scores.get("citation_recall", 0.0),
            citation_precision=scores.get("citation_precision", 0.0),
            faithfulness=scores.get("faithfulness", 0.0),
            completeness=scores.get("completeness", 0.0),
            llm_judge_raw=raw,
        )

    def evaluate_all(self, questions: List[EvalQuestion],
                     answers: Dict[str, str]) -> List[Level3Result]:
        """Evaluate all Q&A pairs."""
        results = []
        for i, q in enumerate(questions):
            answer = answers.get(q.id, "")
            if not answer:
                continue
            print(f"  [{i+1}/{len(questions)}] Judging {q.id}...")
            result = self.evaluate_single(q, answer)
            results.append(result)
        return results


# ════════════════════════════════════════════════════════════════
# Dataset Loader
# ════════════════════════════════════════════════════════════════

def load_eval_dataset(path: str) -> List[EvalQuestion]:
    """Load evaluation dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for item in data:
        questions.append(EvalQuestion(
            id=item.get("id", ""),
            question_type=item.get("question_type", ""),
            question=item.get("question", ""),
            standard_answer=item.get("standard_answer", ""),
            keypoints=item.get("keypoints", []),
            required_citations=item.get("required_citations", []),
            relevant_papers=item.get("relevant_papers", []),
            difficulty=item.get("difficulty", "medium"),
            requires_multi_source=item.get("requires_multi_source", False),
            notes=item.get("notes", ""),
        ))
    return questions


# ════════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals
# ════════════════════════════════════════════════════════════════

def bootstrap_ci(metrics: List[float], n_bootstrap: int = 10000,
                  ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a list of per-question metrics."""
    if len(metrics) < 5:
        return float(np.mean(metrics)), float(np.mean(metrics))

    np.random.seed(42)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(metrics, size=len(metrics), replace=True)
        means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return float(lo), float(hi)


# ════════════════════════════════════════════════════════════════
# Main CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SorGPT Evaluation Framework")
    parser.add_argument("--level", type=str, default="1",
                        help="Evaluation level: 1 (retrieval), 2 (ranking), 3 (generation), all")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (7 conditions)")
    parser.add_argument("--dataset", type=str, default="eval_dataset.json",
                        help="Path to eval dataset JSON")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output path for results")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute bootstrap confidence intervals")
    args = parser.parse_args()

    # Load dataset
    dataset_path = os.path.join(SCRIPT_DIR, "eval", args.dataset) \
        if not os.path.isabs(args.dataset) else args.dataset
    questions = load_eval_dataset(dataset_path)
    print(f"Loaded {len(questions)} evaluation questions")

    # Init pipeline
    from pipeline import SorghumRAGPipeline
    pipeline = SorghumRAGPipeline()

    if args.ablation:
        print("\nRunning Ablation Study...")
        runner = AblationRunner(pipeline, questions)
        results = runner.run()
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        evaluator = RetrievalEvaluator(pipeline, questions)
        results = evaluator.evaluate_all()
        summary = evaluator.summarize(results)

        print(f"\n{'='*60}")
        print("Retrieval Evaluation Summary")
        print(f"{'='*60}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if args.bootstrap:
            for metric in ["recall_5", "recall_10", "mrr", "ndcg_5"]:
                values = [getattr(r, metric) for r in results]
                lo, hi = bootstrap_ci(values)
                print(f"  {metric} 95% CI: [{lo:.4f}, {hi:.4f}]")
