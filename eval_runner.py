#!/usr/bin/env python3
"""
SorGPT 运行时自动评估管线 (Runtime Auto-Evaluation Pipeline)
=============================================================

方法论来源 (Methodology):
  - AutoNuggetizer (Pradeep et al., SIGIR 2025):
      LLM 从检索文献中自动提取 atomic nuggets，检查答案覆盖度
  - RAGChecker (Ru et al., NeurIPS 2024):
      Claim 级分解 + NLI entailment 检测，分离 retriever/generator 诊断
  - RAGEval (Zhu et al., 2024):
      Schema → Config → QA → Keypoint 全自动生成，零人工

评估流程 (7 Phases):
  Phase 1 ─ Query SorGPT API      → 获取答案 + 检索上下文 + 参考文献
  Phase 2 ─ Nugget Extraction     → LLM 从检索 chunk 中提取原子事实
  Phase 3 ─ Claim Decomposition   → LLM 将答案分解为原子声明
  Phase 4 ─ Claim Verification    → entailment: claim vs retrieved chunks
  Phase 5 ─ Nugget Coverage       → 答案覆盖了多少检索到的 nuggets
  Phase 6 ─ Answer Relevance      → 答案是否切题 (1-5)
  Phase 7 ─ Aggregate Metrics     → 按 SorGPT 类型 × RAGEval 类型 × RAGChecker 维度

零人工标注 (Zero Human Annotation):
  - 输入: 纯问题集（无标准答案、无 keypoints、无 nuggets）
  - 所有评估信号均从检索文献中自动提取
  - 人类仅在最终报告中审阅统计结果

Usage:
  python eval_runner.py --phase 1          # 仅查询 SorGPT API
  python eval_runner.py --phase 2          # 仅提取 nuggets
  python eval_runner.py --phase 1-4        # 查询 + nuggets + claims + 验证
  python eval_runner.py --phase all        # 完整 7 阶段评估
  python eval_runner.py --resume           # 从断点续跑
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

# ============================================================
# Configuration
# ============================================================

# SorGPT API
SORGPT_BASE_URL = "http://localhost:8000"  # internal server IP
SORGPT_API_KEY = os.environ.get("SORGPT_API_KEY", "")
SORGPT_REQUEST_DELAY = 12  # seconds between queries (respect rate limit)

# Evaluation LLM (DeepSeek Reasoner — same as SorGPT generator, but used for entailment/verification)
EVAL_BASE_URL = "https://api.deepseek.com/v1"
EVAL_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
EVAL_MODEL = "deepseek-reasoner"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(SCRIPT_DIR, "eval_questions_clean_200.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eval_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_ANSWERS_PATH      = os.path.join(OUTPUT_DIR, "phase1_raw_answers.json")
NUGGETS_PATH          = os.path.join(OUTPUT_DIR, "phase2_nuggets.json")
CLAIMS_PATH           = os.path.join(OUTPUT_DIR, "phase3_claims.json")
VERDICTS_PATH         = os.path.join(OUTPUT_DIR, "phase4_verdicts.json")
COVERAGE_PATH         = os.path.join(OUTPUT_DIR, "phase5_coverage.json")
RELEVANCE_PATH        = os.path.join(OUTPUT_DIR, "phase6_relevance.json")
FINAL_REPORT_JSON     = os.path.join(OUTPUT_DIR, "phase7_report.json")
FINAL_REPORT_MD       = os.path.join(OUTPUT_DIR, "phase7_report.md")


# ============================================================
# Data Structures
# ============================================================

@dataclass
class RawResponse:
    """Phase 1 output: raw SorGPT API response."""
    question_id: str
    question_zh: str
    question_en: str
    subtype: str
    rageval_type: str
    difficulty: str
    domain: str
    query: str               # actual query sent
    query_type: str          # SorGPT classification result
    answer: str              # generated answer
    references: List[str]    # reference list
    raw_response: dict       # full API response
    timestamp: str = ""
    error: str = ""

@dataclass
class Nugget:
    """Phase 2 output: atomic fact extracted from retrieved chunks."""
    nugget_id: str
    text: str
    source_refs: List[str]   # [1], [2], etc.
    category: str = ""       # gene_id / chromosome / function / phenotype / statistic / other

@dataclass
class NuggetSet:
    """All nuggets for one question."""
    question_id: str
    nuggets: List[Nugget]
    total_chunks: int
    extraction_timestamp: str = ""
    error: str = ""

@dataclass
class Claim:
    """Phase 3 output: atomic claim from generated answer."""
    claim_id: str
    text: str
    claim_type: str = ""  # gene_id / chromosome / function / phenotype / statistic / citation / other

@dataclass
class ClaimSet:
    """All claims for one question."""
    question_id: str
    claims: List[Claim]
    decomposition_timestamp: str = ""
    error: str = ""

@dataclass
class Verdict:
    """Phase 4 output: entailment verdict for one claim."""
    claim_id: str
    claim_text: str
    verdict: str            # SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED
    supporting_sources: List[str]  # [1], [3] etc.
    explanation: str = ""

@dataclass
class VerdictSet:
    """All verdicts for one question."""
    question_id: str
    verdicts: List[Verdict]
    verification_timestamp: str = ""
    error: str = ""

@dataclass
class CoverageResult:
    """Phase 5 output: nugget coverage check."""
    nugget_id: str
    nugget_text: str
    covered: str            # COVERED / PARTIALLY / MISSED
    answer_excerpt: str = ""

@dataclass
class CoverageSet:
    """All coverage results for one question."""
    question_id: str
    coverage: List[CoverageResult]
    coverage_timestamp: str = ""
    error: str = ""

@dataclass
class RelevanceScore:
    """Phase 6 output: answer relevance assessment."""
    question_id: str
    relevance_score: int    # 1-5
    explanation: str = ""
    error: str = ""

@dataclass
class QuestionMetrics:
    """Per-question aggregated metrics."""
    question_id: str
    subtype: str
    rageval_type: str
    difficulty: str
    domain: str

    # RAGChecker Overall Metrics
    precision: float = 0.0      # supported claims / total claims
    recall: float = 0.0         # covered nuggets / total nuggets
    f1: float = 0.0

    # RAGChecker Retriever Metrics
    claim_recall: float = 0.0   # nuggets covered by answer / total nuggets (how well answer uses retrieval)
    context_precision: float = 0.0  # (approximated by relevance * nugget density)

    # RAGChecker Generator Metrics
    faithfulness: float = 0.0   # supported claims / total claims
    hallucination_rate: float = 0.0  # unsupported claims / total claims
    context_utilization: float = 0.0  # claims with source support / total claims

    # Answer quality
    answer_relevance: float = 0.0  # 1-5 normalized to 0-1
    total_claims: int = 0
    total_nuggets: int = 0

    error: str = ""


# ============================================================
# LLM Client
# ============================================================

class EvalLLM:
    """LLM client for evaluation tasks (nugget extraction, entailment, etc.)."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=EVAL_API_KEY, base_url=EVAL_BASE_URL)

    def call(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Make a single LLM call."""
        try:
            resp = self.client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[ERROR] {e}"

    def call_with_json(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.0, max_tokens: int = 4096) -> dict:
        """Call LLM and parse JSON response."""
        raw = self.call(system_prompt, user_prompt, temperature, max_tokens)
        # Try to extract JSON from response
        try:
            # Find JSON block
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            # Try parsing
            raw = raw.strip()
            if raw.startswith("{"):
                return json.loads(raw)
            elif raw.startswith("["):
                return {"items": json.loads(raw)}
            else:
                return {"raw": raw, "parse_error": "No JSON structure found"}
        except (json.JSONDecodeError, ValueError) as e:
            return {"raw": raw, "parse_error": str(e)}


# ============================================================
# SorGPT API Client
# ============================================================

class SorGPTClient:
    """Client for querying the SorGPT API."""

    def __init__(self, base_url: str = SORGPT_BASE_URL, api_key: str = SORGPT_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        import requests
        self.requests = requests

    def ask(self, question: str) -> dict:
        """Send a question to SorGPT and get answer + references."""
        try:
            resp = self.requests.post(
                f"{self.base_url}/ask",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key,
                },
                json={"question": question},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
        except Exception as e:
            return {"error": str(e)}

    def health_check(self) -> bool:
        """Check if SorGPT API is reachable."""
        try:
            resp = self.requests.get(f"{self.base_url}/", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False


# ============================================================
# Phase 1: Query SorGPT API
# ============================================================

def phase1_query_sorgpt(questions: List[dict], resume: bool = True) -> List[RawResponse]:
    """Query SorGPT API for all questions, collect raw answers."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Querying SorGPT API ({len(questions)} questions)")
    print(f"{'='*60}")

    # Load existing if resuming
    existing = {}
    if resume and os.path.exists(RAW_ANSWERS_PATH):
        with open(RAW_ANSWERS_PATH) as f:
            for item in json.load(f):
                existing[item["question_id"]] = item
        print(f"Resuming: {len(existing)} already queried")

    client = SorGPTClient()

    # Health check
    if not client.health_check():
        print("⚠️  WARNING: SorGPT API health check failed. Proceeding anyway...")
    else:
        print("✅ SorGPT API health check passed")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        if qid in existing and not existing[qid].get("error"):
            results.append(RawResponse(**existing[qid]))
            print(f"  [{i+1}/{len(questions)}] {qid} ⏭️  (cached)")
            continue

        # Use Chinese question (primary language of the system)
        query = q["question_zh"]
        print(f"  [{i+1}/{len(questions)}] {qid} [{q['subtype']}] {query[:60]}...", end=" ", flush=True)

        resp = client.ask(query)
        timestamp = datetime.now().isoformat()

        if "error" in resp:
            raw = RawResponse(
                question_id=qid, question_zh=q["question_zh"],
                question_en=q["question_en"], subtype=q["subtype"],
                rageval_type=q["rageval_type"], difficulty=q["difficulty"],
                domain=q["domain"], query=query, query_type="",
                answer="", references=[], raw_response=resp,
                timestamp=timestamp, error=resp["error"]
            )
            print(f"❌ {resp['error']}")
        else:
            raw = RawResponse(
                question_id=qid, question_zh=q["question_zh"],
                question_en=q["question_en"], subtype=q["subtype"],
                rageval_type=q["rageval_type"], difficulty=q["difficulty"],
                domain=q["domain"], query=resp.get("query", query),
                query_type=resp.get("query_type", ""),
                answer=resp.get("answer", ""),
                references=resp.get("references", []),
                raw_response=resp, timestamp=timestamp,
            )
            print(f"✅ [{raw.query_type}] {len(raw.answer)} chars, {len(raw.references)} refs")

        results.append(raw)

        # Save incrementally
        with open(RAW_ANSWERS_PATH, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

        # Rate limit
        if i < len(questions) - 1:
            time.sleep(SORGPT_REQUEST_DELAY)

    print(f"\nPhase 1 complete: {len([r for r in results if not r.error])}/{len(results)} successful")
    return results


# ============================================================
# Phase 2: Nugget Extraction (AutoNuggetizer)
# ============================================================

NUGGET_EXTRACTION_SYSTEM = """\
You are an expert scientific fact extractor. Your task is to extract ALL verifiable
atomic facts ("nuggets") from a set of research literature snippets provided as context.

Each nugget must be:
1. A SINGLE atomic fact (not a compound statement)
2. DIRECTLY stated in the provided snippets (not inferred)
3. Verifiable against the source text
4. Specific (include gene IDs, chromosome numbers, statistics, etc.)

Categories: gene_id, chromosome_location, molecular_function, phenotype,
            statistic_value, citation_claim, mechanism_step, other

Output format: JSON array of nugget objects.
"""

def phase2_extract_nuggets(raw_answers: List[RawResponse], resume: bool = True) -> List[NuggetSet]:
    """Extract atomic nuggets from retrieved references."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: AutoNugget Extraction ({len(raw_answers)} questions)")
    print(f"{'='*60}")

    existing = {}
    if resume and os.path.exists(NUGGETS_PATH):
        with open(NUGGETS_PATH) as f:
            for item in json.load(f):
                existing[item["question_id"]] = item
        print(f"Resuming: {len(existing)} already processed")

    llm = EvalLLM()
    results = []

    for i, raw in enumerate(raw_answers):
        qid = raw.question_id
        if qid in existing and not existing[qid].get("error"):
            ns = NuggetSet(
                question_id=qid,
                nuggets=[Nugget(**n) for n in existing[qid].get("nuggets", [])],
                total_chunks=existing[qid].get("total_chunks", 0),
                extraction_timestamp=existing[qid].get("extraction_timestamp", ""),
            )
            results.append(ns)
            print(f"  [{i+1}/{len(raw_answers)}] {qid} ⏭️  (cached)")
            continue

        if raw.error:
            results.append(NuggetSet(question_id=qid, nuggets=[], total_chunks=0, error=raw.error))
            print(f"  [{i+1}/{len(raw_answers)}] {qid} ⏭️  (upstream error)")
            continue

        # Build context from references
        refs_text = raw.references if raw.references else []
        if not refs_text:
            # No references = nothing to extract nuggets from
            results.append(NuggetSet(
                question_id=qid, nuggets=[], total_chunks=0,
                extraction_timestamp=datetime.now().isoformat()
            ))
            print(f"  [{i+1}/{len(raw_answers)}] {qid} ⏭️  (no references)")
            continue

        # Truncate context if too long
        context = "\n\n---\n\n".join(refs_text)
        max_context = 12000
        if len(context) > max_context:
            context = context[:max_context] + "\n\n[...truncated...]"

        user_prompt = f"""\
RESEARCH SNIPPETS (from sorghum scientific literature):
{context}

TASK: Extract ALL verifiable atomic facts (nuggets) from these snippets.
For each nugget, identify which source reference it comes from ([1], [2], etc.).

Output as JSON array:
```json
[
  {{"nugget_id": "N1", "text": "DW3 encodes a P-glycoprotein involved in auxin transport", "source_refs": ["[1]"], "category": "molecular_function"}},
  ...
]
```
"""
        print(f"  [{i+1}/{len(raw_answers)}] {qid} [{raw.subtype}] extracting...", end=" ", flush=True)

        resp = llm.call_with_json(NUGGET_EXTRACTION_SYSTEM, user_prompt)
        timestamp = datetime.now().isoformat()

        try:
            items = resp.get("items", resp.get("nuggets", []))
            if not items and isinstance(resp, list):
                items = resp
            nuggets = [
                Nugget(
                    nugget_id=n.get("nugget_id", f"N{j+1}"),
                    text=n.get("text", ""),
                    source_refs=n.get("source_refs", []),
                    category=n.get("category", "other"),
                )
                for j, n in enumerate(items)
            ]
            ns = NuggetSet(
                question_id=qid, nuggets=nuggets,
                total_chunks=len(refs_text),
                extraction_timestamp=timestamp,
            )
            print(f"✅ {len(nuggets)} nuggets")
        except Exception as e:
            ns = NuggetSet(question_id=qid, nuggets=[], total_chunks=len(refs_text),
                          extraction_timestamp=timestamp, error=str(e))
            print(f"❌ parse error: {e}")

        results.append(ns)

        # Save incrementally
        with open(NUGGETS_PATH, 'w', encoding='utf-8') as f:
            json.dump([asdict(ns) for ns in results], f, ensure_ascii=False, indent=2)

        time.sleep(1)  # gentle rate limit for DeepSeek API

    total_nuggets = sum(len(ns.nuggets) for ns in results)
    print(f"\nPhase 2 complete: {total_nuggets} total nuggets extracted")
    return results


# ============================================================
# Phase 3: Claim Decomposition (RAGChecker-style)
# ============================================================

CLAIM_DECOMPOSITION_SYSTEM = """\
You are an expert at decomposing scientific answers into atomic factual claims.
Each claim must be:
1. A SINGLE, independently verifiable factual statement
2. Directly from the answer text (not inferred)
3. Self-contained (include the subject, e.g. "DW3 gene" not "it")

Claim types: gene_id | chromosome_location | molecular_function | phenotype |
             statistic_value | citation_claim | mechanism_step | comparative_claim | other

Output format: JSON array of claim objects.
"""

def phase3_decompose_claims(raw_answers: List[RawResponse], resume: bool = True) -> List[ClaimSet]:
    """Decompose each answer into atomic claims."""
    print(f"\n{'='*60}")
    print(f"PHASE 3: Claim Decomposition ({len(raw_answers)} answers)")
    print(f"{'='*60}")

    existing = {}
    if resume and os.path.exists(CLAIMS_PATH):
        with open(CLAIMS_PATH) as f:
            for item in json.load(f):
                existing[item["question_id"]] = item
        print(f"Resuming: {len(existing)} already processed")

    llm = EvalLLM()
    results = []

    for i, raw in enumerate(raw_answers):
        qid = raw.question_id
        if qid in existing and not existing[qid].get("error"):
            cs = ClaimSet(
                question_id=qid,
                claims=[Claim(**c) for c in existing[qid].get("claims", [])],
                decomposition_timestamp=existing[qid].get("decomposition_timestamp", ""),
            )
            results.append(cs)
            print(f"  [{i+1}/{len(raw_answers)}] {qid} ⏭️  (cached)")
            continue

        if raw.error or not raw.answer:
            results.append(ClaimSet(question_id=qid, claims=[], error=raw.error or "no answer"))
            print(f"  [{i+1}/{len(raw_answers)}] {qid} ⏭️  (no answer)")
            continue

        user_prompt = f"""\
QUESTION: {raw.question_zh}

ANSWER:
{raw.answer[:8000]}

TASK: Decompose the answer into ALL atomic factual claims.
Output as JSON array:
```json
[
  {{"claim_id": "C1", "text": "DW3 gene is located on chromosome 7", "claim_type": "chromosome_location"}},
  ...
]
```
"""
        print(f"  [{i+1}/{len(raw_answers)}] {qid} [{raw.subtype}] decomposing {len(raw.answer)} chars...", end=" ", flush=True)

        resp = llm.call_with_json(CLAIM_DECOMPOSITION_SYSTEM, user_prompt)
        timestamp = datetime.now().isoformat()

        try:
            items = resp.get("items", resp.get("claims", []))
            if not items and isinstance(resp, list):
                items = resp
            claims = [
                Claim(
                    claim_id=c.get("claim_id", f"C{j+1}"),
                    text=c.get("text", ""),
                    claim_type=c.get("claim_type", "other"),
                )
                for j, c in enumerate(items)
            ]
            cs = ClaimSet(question_id=qid, claims=claims, decomposition_timestamp=timestamp)
            print(f"✅ {len(claims)} claims")
        except Exception as e:
            cs = ClaimSet(question_id=qid, claims=[], decomposition_timestamp=timestamp, error=str(e))
            print(f"❌ parse error: {e}")

        results.append(cs)

        with open(CLAIMS_PATH, 'w', encoding='utf-8') as f:
            json.dump([asdict(cs) for cs in results], f, ensure_ascii=False, indent=2)

        time.sleep(1)

    total_claims = sum(len(cs.claims) for cs in results)
    print(f"\nPhase 3 complete: {total_claims} total claims decomposed")
    return results


# ============================================================
# Phase 4: Claim Verification (Entailment)
# ============================================================

VERIFICATION_SYSTEM = """\
You are an expert scientific fact-checker. Your task is to verify whether each
atomic claim from a generated answer is supported by the provided research snippets.

For each claim, determine:
- SUPPORTED: The claim is directly stated or clearly implied by the snippets.
             Provide the source reference numbers.
- PARTIALLY_SUPPORTED: The snippets provide partial evidence but key details
                       are missing or differ.
- UNSUPPORTED: The claim cannot be found in the snippets. This includes facts
               that MAY be true but are not in the provided text.

Output format: JSON array of verdict objects.
"""

def phase4_verify_claims(raw_answers: List[RawResponse], claim_sets: List[ClaimSet],
                         resume: bool = True) -> List[VerdictSet]:
    """Verify each claim against retrieved context."""
    print(f"\n{'='*60}")
    print(f"PHASE 4: Claim Verification ({sum(len(cs.claims) for cs in claim_sets)} claims)")
    print(f"{'='*60}")

    existing = {}
    if resume and os.path.exists(VERDICTS_PATH):
        with open(VERDICTS_PATH) as f:
            for item in json.load(f):
                existing[item["question_id"]] = item
        print(f"Resuming: {len(existing)} already processed")

    # Index raw answers by ID
    raw_index = {r.question_id: r for r in raw_answers}
    llm = EvalLLM()
    results = []

    for i, cs in enumerate(claim_sets):
        qid = cs.question_id
        if qid in existing and not existing[qid].get("error"):
            vs = VerdictSet(
                question_id=qid,
                verdicts=[Verdict(**v) for v in existing[qid].get("verdicts", [])],
                verification_timestamp=existing[qid].get("verification_timestamp", ""),
            )
            results.append(vs)
            print(f"  [{i+1}/{len(claim_sets)}] {qid} ⏭️  (cached)")
            continue

        if not cs.claims:
            results.append(VerdictSet(question_id=qid, verdicts=[]))
            print(f"  [{i+1}/{len(claim_sets)}] {qid} ⏭️  (no claims)")
            continue

        raw = raw_index.get(qid)
        if not raw or raw.error:
            results.append(VerdictSet(question_id=qid, verdicts=[], error="upstream error"))
            continue

        refs_text = "\n\n---\n\n".join(raw.references) if raw.references else "[No references available]"
        max_context = 10000
        if len(refs_text) > max_context:
            refs_text = refs_text[:max_context] + "\n\n[...truncated...]"

        claims_text = "\n".join(f"[{c.claim_id}] {c.text}" for c in cs.claims)

        user_prompt = f"""\
RESEARCH SNIPPETS:
{refs_text}

CLAIMS TO VERIFY:
{claims_text}

TASK: For each claim, determine if it is SUPPORTED by the research snippets.
Cite specific source references when supported.

Output as JSON array:
```json
[
  {{"claim_id": "C1", "verdict": "SUPPORTED", "supporting_sources": ["[1]", "[3]"], "explanation": "Snippet [1] states..."}},
  ...
]
```
"""
        print(f"  [{i+1}/{len(claim_sets)}] {qid} verifying {len(cs.claims)} claims...", end=" ", flush=True)

        resp = llm.call_with_json(VERIFICATION_SYSTEM, user_prompt)
        timestamp = datetime.now().isoformat()

        try:
            items = resp.get("items", resp.get("verdicts", []))
            if not items and isinstance(resp, list):
                items = resp
            verdicts = [
                Verdict(
                    claim_id=v.get("claim_id", f"V{j+1}"),
                    claim_text=v.get("claim_text", ""),
                    verdict=v.get("verdict", "UNSUPPORTED"),
                    supporting_sources=v.get("supporting_sources", []),
                    explanation=v.get("explanation", ""),
                )
                for j, v in enumerate(items)
            ]
            vs = VerdictSet(question_id=qid, verdicts=verdicts, verification_timestamp=timestamp)

            supported = sum(1 for v in verdicts if v.verdict == "SUPPORTED")
            partial = sum(1 for v in verdicts if v.verdict == "PARTIALLY_SUPPORTED")
            unsupported = sum(1 for v in verdicts if v.verdict == "UNSUPPORTED")
            print(f"✅ {supported}S/{partial}P/{unsupported}U")
        except Exception as e:
            vs = VerdictSet(question_id=qid, verdicts=[], verification_timestamp=timestamp, error=str(e))
            print(f"❌ parse error: {e}")

        results.append(vs)

        with open(VERDICTS_PATH, 'w', encoding='utf-8') as f:
            json.dump([asdict(vs) for vs in results], f, ensure_ascii=False, indent=2)

        time.sleep(1)

    print(f"\nPhase 4 complete")
    return results


# ============================================================
# Phase 5: Nugget Coverage (AutoNuggetizer-style)
# ============================================================

COVERAGE_SYSTEM = """\
You are checking whether atomic facts from research literature are covered
in a generated answer. For each nugget (fact from literature), determine:
- COVERED: The answer includes this fact (even if rephrased)
- PARTIALLY: The answer mentions related information but misses key details
- MISSED: The answer does not include this fact at all

Output format: JSON array.
"""

def phase5_check_coverage(raw_answers: List[RawResponse], nugget_sets: List[NuggetSet],
                          resume: bool = True) -> List[CoverageSet]:
    """Check how many nuggets are covered by the answer."""
    print(f"\n{'='*60}")
    print(f"PHASE 5: Nugget Coverage Check")
    print(f"{'='*60}")

    existing = {}
    if resume and os.path.exists(COVERAGE_PATH):
        with open(COVERAGE_PATH) as f:
            for item in json.load(f):
                existing[item["question_id"]] = item
        print(f"Resuming: {len(existing)} already processed")

    raw_index = {r.question_id: r for r in raw_answers}
    llm = EvalLLM()
    results = []

    for i, ns in enumerate(nugget_sets):
        qid = ns.question_id
        if qid in existing and not existing[qid].get("error"):
            cov = CoverageSet(
                question_id=qid,
                coverage=[CoverageResult(**c) for c in existing[qid].get("coverage", [])],
                coverage_timestamp=existing[qid].get("coverage_timestamp", ""),
            )
            results.append(cov)
            continue

        if not ns.nuggets:
            results.append(CoverageSet(question_id=qid, coverage=[]))
            continue

        raw = raw_index.get(qid)
        if not raw or raw.error or not raw.answer:
            results.append(CoverageSet(question_id=qid, coverage=[], error="upstream error"))
            continue

        nuggets_text = "\n".join(f"[{n.nugget_id}] {n.text}" for n in ns.nuggets)

        user_prompt = f"""\
ANSWER:
{raw.answer[:6000]}

FACTS TO CHECK (from research literature):
{nuggets_text}

TASK: For each fact, determine if it is covered in the answer.
Output as JSON array:
```json
[
  {{"nugget_id": "N1", "covered": "COVERED", "answer_excerpt": "..."}},
  ...
]
```
"""
        print(f"  [{i+1}/{len(nugget_sets)}] {qid} checking {len(ns.nuggets)} nuggets...", end=" ", flush=True)

        resp = llm.call_with_json(COVERAGE_SYSTEM, user_prompt)
        timestamp = datetime.now().isoformat()

        try:
            items = resp.get("items", resp.get("coverage", []))
            if not items and isinstance(resp, list):
                items = resp
            coverage = [
                CoverageResult(
                    nugget_id=c.get("nugget_id", ""),
                    nugget_text=c.get("nugget_text", ""),
                    covered=c.get("covered", "MISSED"),
                    answer_excerpt=c.get("answer_excerpt", ""),
                )
                for c in items
            ]
            cov = CoverageSet(question_id=qid, coverage=coverage, coverage_timestamp=timestamp)
            covered = sum(1 for c in coverage if c.covered == "COVERED")
            partial = sum(1 for c in coverage if c.covered == "PARTIALLY")
            missed = sum(1 for c in coverage if c.covered == "MISSED")
            print(f"✅ {covered}C/{partial}P/{missed}M")
        except Exception as e:
            cov = CoverageSet(question_id=qid, coverage=[], coverage_timestamp=timestamp, error=str(e))
            print(f"❌ parse error: {e}")

        results.append(cov)

        with open(COVERAGE_PATH, 'w', encoding='utf-8') as f:
            json.dump([asdict(cov) for cov in results], f, ensure_ascii=False, indent=2)

        time.sleep(1)

    print(f"\nPhase 5 complete")
    return results


# ============================================================
# Phase 6: Answer Relevance
# ============================================================

RELEVANCE_SYSTEM = """\
You are evaluating whether a generated answer actually addresses the user's question.
Score on a 1-5 scale:
1 = Completely off-topic / does not address the question
2 = Tangentially related but mostly irrelevant
3 = Partially addresses the question but misses key aspects
4 = Addresses the question well with minor omissions
5 = Fully and precisely addresses the question

Output format: JSON object with score and explanation.
"""

def phase6_score_relevance(raw_answers: List[RawResponse], resume: bool = True) -> List[RelevanceScore]:
    """Score answer relevance for each response."""
    print(f"\n{'='*60}")
    print(f"PHASE 6: Answer Relevance Scoring")
    print(f"{'='*60}")

    existing = {}
    if resume and os.path.exists(RELEVANCE_PATH):
        with open(RELEVANCE_PATH) as f:
            for item in json.load(f):
                existing[item["question_id"]] = item
        print(f"Resuming: {len(existing)} already processed")

    llm = EvalLLM()
    results = []

    for i, raw in enumerate(raw_answers):
        qid = raw.question_id
        if qid in existing and not existing[qid].get("error"):
            rs = RelevanceScore(**existing[qid])
            results.append(rs)
            continue

        if raw.error or not raw.answer:
            results.append(RelevanceScore(question_id=qid, relevance_score=0,
                                          explanation="upstream error", error=raw.error or "no answer"))
            continue

        user_prompt = f"""\
QUESTION: {raw.question_zh}

ANSWER:
{raw.answer[:5000]}

TASK: Score how well the answer addresses the question (1-5).
Output as JSON:
```json
{{"score": 4, "explanation": "..."}}
```
"""
        print(f"  [{i+1}/{len(raw_answers)}] {qid} scoring...", end=" ", flush=True)

        resp = llm.call_with_json(RELEVANCE_SYSTEM, user_prompt)

        try:
            score = int(resp.get("score", 0))
            explanation = resp.get("explanation", "")
            rs = RelevanceScore(question_id=qid, relevance_score=score, explanation=explanation)
            print(f"✅ {score}/5")
        except Exception as e:
            rs = RelevanceScore(question_id=qid, relevance_score=0, explanation="", error=str(e))
            print(f"❌ parse error: {e}")

        results.append(rs)

        with open(RELEVANCE_PATH, 'w', encoding='utf-8') as f:
            json.dump([asdict(rs) for rs in results], f, ensure_ascii=False, indent=2)

        time.sleep(1)

    avg = sum(r.relevance_score for r in results if r.relevance_score > 0) / max(1, len([r for r in results if r.relevance_score > 0]))
    print(f"\nPhase 6 complete: average relevance = {avg:.2f}/5")
    return results


# ============================================================
# Phase 7: Aggregate Metrics
# ============================================================

def phase7_compute_metrics(
    questions: List[dict],
    raw_answers: List[RawResponse],
    nugget_sets: List[NuggetSet],
    claim_sets: List[ClaimSet],
    verdict_sets: List[VerdictSet],
    coverage_sets: List[CoverageSet],
    relevance_scores: List[RelevanceScore],
) -> List[QuestionMetrics]:
    """Compute all evaluation metrics per question and aggregate."""
    print(f"\n{'='*60}")
    print(f"PHASE 7: Computing Final Metrics")
    print(f"{'='*60}")

    # Index everything by question_id
    q_index = {q["id"]: q for q in questions}
    raw_idx = {r.question_id: r for r in raw_answers}
    nug_idx = {n.question_id: n for n in nugget_sets}
    clm_idx = {c.question_id: c for c in claim_sets}
    vrd_idx = {v.question_id: v for v in verdict_sets}
    cov_idx = {c.question_id: c for c in coverage_sets}
    rel_idx = {r.question_id: r for r in relevance_scores}

    metrics_list = []

    for qid in q_index:
        q = q_index[qid]
        raw = raw_idx.get(qid)
        nugs = nug_idx.get(qid)
        claims = clm_idx.get(qid)
        verds = vrd_idx.get(qid)
        covs = cov_idx.get(qid)
        rel = rel_idx.get(qid)

        m = QuestionMetrics(
            question_id=qid,
            subtype=q["subtype"],
            rageval_type=q["rageval_type"],
            difficulty=q["difficulty"],
            domain=q["domain"],
        )

        if raw and raw.error:
            m.error = raw.error
            metrics_list.append(m)
            continue

        # Count totals
        m.total_nuggets = len(nugs.nuggets) if nugs else 0
        m.total_claims = len(claims.claims) if claims else 0

        if m.total_claims == 0 and m.total_nuggets == 0:
            metrics_list.append(m)
            continue

        # RAGChecker Overall Metrics
        if verds and verds.verdicts:
            supported = sum(1 for v in verds.verdicts if v.verdict == "SUPPORTED")
            partial = sum(1 for v in verds.verdicts if v.verdict == "PARTIALLY_SUPPORTED")
            unsupported = sum(1 for v in verds.verdicts if v.verdict == "UNSUPPORTED")
            total_v = len(verds.verdicts)

            m.precision = (supported + 0.5 * partial) / max(1, total_v)
            m.faithfulness = m.precision  # same calculation
            m.hallucination_rate = unsupported / max(1, total_v)
            m.context_utilization = (supported + partial) / max(1, total_v)

        # Claim Recall (nugget coverage)
        if covs and covs.coverage:
            covered = sum(1 for c in covs.coverage if c.covered == "COVERED")
            partial_cov = sum(1 for c in covs.coverage if c.covered == "PARTIALLY")
            total_cov = len(covs.coverage)
            m.recall = (covered + 0.5 * partial_cov) / max(1, total_cov)
            m.claim_recall = m.recall

        # Context Precision (approximated)
        if nugs and nugs.total_chunks > 0:
            if nugs.nuggets:
                m.context_precision = min(1.0, len(nugs.nuggets) / (nugs.total_chunks * 3))
            else:
                m.context_precision = 0.0

        # F1
        if m.precision > 0 or m.recall > 0:
            m.f1 = 2 * m.precision * m.recall / max(0.001, m.precision + m.recall)

        # Answer Relevance
        if rel and rel.relevance_score > 0:
            m.answer_relevance = rel.relevance_score / 5.0

        metrics_list.append(m)

    # Save
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump([asdict(m) for m in metrics_list], f, ensure_ascii=False, indent=2)

    # Generate markdown report
    _generate_markdown_report(metrics_list, questions)

    print(f"\nPhase 7 complete: metrics for {len(metrics_list)} questions")
    return metrics_list


def _generate_markdown_report(metrics: List[QuestionMetrics], questions: List[dict]):
    """Generate comprehensive evaluation report in markdown."""

    valid = [m for m in metrics if not m.error and (m.total_claims > 0 or m.total_nuggets > 0)]
    errors = [m for m in metrics if m.error]

    # Aggregate by SorGPT subtype
    by_subtype = defaultdict(list)
    for m in valid:
        by_subtype[m.subtype].append(m)

    # Aggregate by RAGEval type
    by_rageval = defaultdict(list)
    for m in valid:
        by_rageval[m.rageval_type].append(m)

    # Aggregate by difficulty
    by_diff = defaultdict(list)
    for m in valid:
        by_diff[m.difficulty].append(m)

    # Aggregate by domain
    by_domain = defaultdict(list)
    for m in valid:
        by_domain[m.domain].append(m)

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    lines = []
    lines.append("# SorGPT Runtime Auto-Evaluation Report")
    lines.append(f"\n**Generated**: {datetime.now().isoformat()}")
    lines.append(f"**Total questions**: {len(metrics)}")
    lines.append(f"**Successful evaluations**: {len(valid)}")
    lines.append(f"**Errors**: {len(errors)}")
    lines.append(f"\n**Methodology**: AutoNuggetizer (SIGIR 2025) + RAGChecker (NeurIPS 2024) + RAGEval (2024)")
    lines.append(f"**Zero human annotations**: All metrics computed automatically from retrieved literature\n")

    # Overall
    lines.append("## 1. Overall Metrics\n")
    lines.append(f"| Metric | Mean | Median | Min | Max |")
    lines.append(f"|--------|------|--------|-----|-----|")
    for name, key in [("Faithfulness", "faithfulness"), ("Claim Recall", "claim_recall"),
                       ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
                       ("Hallucination Rate", "hallucination_rate"),
                       ("Context Utilization", "context_utilization"),
                       ("Answer Relevance", "answer_relevance")]:
        vals = [getattr(m, key) for m in valid if getattr(m, key) > 0]
        if vals:
            lines.append(f"| {name} | {avg(vals):.3f} | {sorted(vals)[len(vals)//2]:.3f} | {min(vals):.3f} | {max(vals):.3f} |")

    # By SorGPT Type
    lines.append("\n## 2. By SorGPT Question Type\n")
    sorgpt_order = ['factoid', 'functional', 'mechanism', 'qtl_gwas', 'review',
                    'comparative', 'comprehensive', 'gene_list', 'locate', 'count', 'boundary']
    lines.append(f"| Type | N | Faithfulness | Claim Recall | F1 | Hallucination | Relevance |")
    lines.append(f"|------|---|-------------|--------------|----|---------------|-----------|")
    for st in sorgpt_order:
        items = by_subtype.get(st, [])
        if items:
            lines.append(f"| {st} | {len(items)} | {avg([m.faithfulness for m in items]):.3f} | "
                        f"{avg([m.claim_recall for m in items]):.3f} | {avg([m.f1 for m in items]):.3f} | "
                        f"{avg([m.hallucination_rate for m in items]):.3f} | {avg([m.answer_relevance for m in items]):.3f} |")

    # By RAGEval Type
    lines.append("\n## 3. By RAGEval Question Type (Zhu et al., 2024)\n")
    lines.append(f"| Type | N | Faithfulness | Claim Recall | F1 | Hallucination |")
    lines.append(f"|------|---|-------------|--------------|----|---------------|")
    for rt in ['Factual', 'Numerical Comparison', 'Information Integration',
               'Multi-hop Reasoning', 'Summary', 'Time-series', 'Irrelevant / Unanswerable']:
        items = by_rageval.get(rt, [])
        if items:
            lines.append(f"| {rt} | {len(items)} | {avg([m.faithfulness for m in items]):.3f} | "
                        f"{avg([m.claim_recall for m in items]):.3f} | {avg([m.f1 for m in items]):.3f} | "
                        f"{avg([m.hallucination_rate for m in items]):.3f} |")

    # By Difficulty
    lines.append("\n## 4. By Difficulty\n")
    lines.append(f"| Difficulty | N | Faithfulness | F1 | Hallucination | Relevance |")
    lines.append(f"|------------|---|-------------|----|---------------|-----------|")
    for d in ['easy', 'medium', 'hard']:
        items = by_diff.get(d, [])
        if items:
            lines.append(f"| {d} | {len(items)} | {avg([m.faithfulness for m in items]):.3f} | "
                        f"{avg([m.f1 for m in items]):.3f} | {avg([m.hallucination_rate for m in items]):.3f} | "
                        f"{avg([m.answer_relevance for m in items]):.3f} |")

    # By Domain
    lines.append("\n## 5. By Scientific Domain\n")
    lines.append(f"| Domain | N | Faithfulness | F1 | Hallucination |")
    lines.append(f"|--------|---|-------------|----|---------------|")
    for dom in sorted(by_domain.keys()):
        items = by_domain[dom]
        lines.append(f"| {dom} | {len(items)} | {avg([m.faithfulness for m in items]):.3f} | "
                    f"{avg([m.f1 for m in items]):.3f} | {avg([m.hallucination_rate for m in items]):.3f} |")

    # Per-question detail
    lines.append("\n## 6. Per-Question Detail\n")
    lines.append(f"| ID | Type | Difficulty | Claims | Nuggets | Faith. | Recall | F1 | Halluc. | Relev. |")
    lines.append(f"|----|------|-----------|--------|---------|--------|--------|----|---------|--------|")
    for m in metrics:
        if m.error:
            lines.append(f"| {m.question_id} | {m.subtype} | {m.difficulty} | ❌ {m.error} | | | | | | |")
        else:
            lines.append(f"| {m.question_id} | {m.subtype} | {m.difficulty} | {m.total_claims} | "
                        f"{m.total_nuggets} | {m.faithfulness:.2f} | {m.claim_recall:.2f} | "
                        f"{m.f1:.2f} | {m.hallucination_rate:.2f} | {m.answer_relevance:.2f} |")

    # Error items
    if errors:
        lines.append("\n## 7. Errors\n")
        for m in errors:
            lines.append(f"- **{m.question_id}** [{m.subtype}]: {m.error}")

    lines.append(f"\n---\n*Report generated by SorGPT Auto-Evaluation Pipeline*")
    lines.append(f"*Frameworks: AutoNuggetizer (SIGIR 2025), RAGChecker (NeurIPS 2024), RAGEval (2024)*")

    with open(FINAL_REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {FINAL_REPORT_MD}")


# ============================================================
# Main
# ============================================================

def parse_phase(arg: str) -> set:
    """Parse phase argument like '1', '1-4', 'all'."""
    if arg == "all":
        return {1, 2, 3, 4, 5, 6, 7}
    if "-" in arg:
        parts = arg.split("-")
        return set(range(int(parts[0]), int(parts[1]) + 1))
    return {int(arg)}

def main():
    parser = argparse.ArgumentParser(description="SorGPT Runtime Auto-Evaluation Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                       help="Phases to run: 1, 1-4, all, etc.")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from saved intermediate results")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh, ignore cached results")
    args = parser.parse_args()

    phases = parse_phase(args.phase)
    resume = not args.no_resume

    print(f"Running phases: {sorted(phases)}")
    print(f"Resume mode: {resume}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load questions
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Phase 1: Query SorGPT
    if 1 in phases:
        raw_answers = phase1_query_sorgpt(questions, resume=resume)
    else:
        if os.path.exists(RAW_ANSWERS_PATH):
            with open(RAW_ANSWERS_PATH) as f:
                raw_answers = [RawResponse(**item) for item in json.load(f)]
            print(f"Loaded {len(raw_answers)} cached raw answers")
        else:
            print("ERROR: Phase 1 output not found. Run with --phase 1 first.")
            sys.exit(1)

    # Phase 2: Nugget Extraction
    if 2 in phases:
        nugget_sets = phase2_extract_nuggets(raw_answers, resume=resume)
    else:
        if os.path.exists(NUGGETS_PATH):
            with open(NUGGETS_PATH) as f:
                nugget_sets = [NuggetSet(
                    question_id=item["question_id"],
                    nuggets=[Nugget(**n) for n in item.get("nuggets", [])],
                    total_chunks=item.get("total_chunks", 0),
                    extraction_timestamp=item.get("extraction_timestamp", ""),
                    error=item.get("error", ""),
                ) for item in json.load(f)]
            print(f"Loaded {len(nugget_sets)} cached nugget sets")
        else:
            nugget_sets = []

    # Phase 3: Claim Decomposition
    if 3 in phases:
        claim_sets = phase3_decompose_claims(raw_answers, resume=resume)
    else:
        if os.path.exists(CLAIMS_PATH):
            with open(CLAIMS_PATH) as f:
                claim_sets = [ClaimSet(
                    question_id=item["question_id"],
                    claims=[Claim(**c) for c in item.get("claims", [])],
                    decomposition_timestamp=item.get("decomposition_timestamp", ""),
                    error=item.get("error", ""),
                ) for item in json.load(f)]
            print(f"Loaded {len(claim_sets)} cached claim sets")
        else:
            claim_sets = []

    # Phase 4: Claim Verification
    if 4 in phases:
        verdict_sets = phase4_verify_claims(raw_answers, claim_sets, resume=resume)
    else:
        if os.path.exists(VERDICTS_PATH):
            with open(VERDICTS_PATH) as f:
                verdict_sets = [VerdictSet(
                    question_id=item["question_id"],
                    verdicts=[Verdict(**v) for v in item.get("verdicts", [])],
                    verification_timestamp=item.get("verification_timestamp", ""),
                    error=item.get("error", ""),
                ) for item in json.load(f)]
            print(f"Loaded {len(verdict_sets)} cached verdict sets")
        else:
            verdict_sets = []

    # Phase 5: Nugget Coverage
    if 5 in phases:
        coverage_sets = phase5_check_coverage(raw_answers, nugget_sets, resume=resume)
    else:
        if os.path.exists(COVERAGE_PATH):
            with open(COVERAGE_PATH) as f:
                coverage_sets = [CoverageSet(
                    question_id=item["question_id"],
                    coverage=[CoverageResult(**c) for c in item.get("coverage", [])],
                    coverage_timestamp=item.get("coverage_timestamp", ""),
                    error=item.get("error", ""),
                ) for item in json.load(f)]
            print(f"Loaded {len(coverage_sets)} cached coverage sets")
        else:
            coverage_sets = []

    # Phase 6: Answer Relevance
    if 6 in phases:
        relevance_scores = phase6_score_relevance(raw_answers, resume=resume)
    else:
        if os.path.exists(RELEVANCE_PATH):
            with open(RELEVANCE_PATH) as f:
                relevance_scores = [RelevanceScore(**item) for item in json.load(f)]
            print(f"Loaded {len(relevance_scores)} cached relevance scores")
        else:
            relevance_scores = []

    # Phase 7: Compute Final Metrics
    if 7 in phases:
        metrics = phase7_compute_metrics(
            questions, raw_answers, nugget_sets, claim_sets,
            verdict_sets, coverage_sets, relevance_scores
        )
    else:
        if os.path.exists(FINAL_REPORT_JSON):
            with open(FINAL_REPORT_JSON) as f:
                metrics = [QuestionMetrics(**item) for item in json.load(f)]
            print(f"Loaded {len(metrics)} cached metrics")
        else:
            metrics = []

    print(f"\n{'='*60}")
    print(f"Evaluation pipeline complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files:")
    for f in [RAW_ANSWERS_PATH, NUGGETS_PATH, CLAIMS_PATH, VERDICTS_PATH,
               COVERAGE_PATH, RELEVANCE_PATH, FINAL_REPORT_JSON, FINAL_REPORT_MD]:
        if os.path.exists(f):
            print(f"  ✅ {os.path.basename(f)}")
        else:
            print(f"  ⬜ {os.path.basename(f)} (not yet generated)")


if __name__ == "__main__":
    main()
