# -*- coding: utf-8 -*-
"""
granularity_eval.py
========================
【作用】
对比 fine / std / large / para 四个粒度索引库的检索质量，
以及 v1（旧库）vs v2（新库）的整体差异。

【输出】
granularity_eval_results.xlsx，包含以下 Sheet：
  01_原始检索结果   — 每题每粒度的详细命中记录
  02_粒度汇总对比   — 四粒度在各指标上的均值对比
  03_题型×粒度矩阵  — 热力图数据（题型 × 粒度 × 指标）
  04_v1_vs_v2对比   — 新旧库整体对比（可选，旧库存在时自动启用）
  05_建议           — 自动生成的优化建议

【使用方法】
1. 把本脚本放到 rag_project/ 目录下（与 config.py 同级）
2. 按需修改下方 CONFIG 区
3. 运行：python granularity_eval.py
"""

import os
import sys
import re
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ── 把 rag_project 加入路径，使得可以直接 import 项目模块 ──
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# ══════════════════════════════════════════════════════════════
# ★ 用户配置区（根据实际情况修改）
# ══════════════════════════════════════════════════════════════
CONFIG = {
    # 基础路径
    "base_dir": "/vol/sunjilin/website/data/agent",

    # v2 索引（新库，必须存在）
    "v2_indexes": {
        "en_fine":  "faiss_v2_english_fine",
        "en_std":   "faiss_v2_english_std",
        "en_large": "faiss_v2_english_large",
        "en_para":  "faiss_v2_english_para",   # para 库，可选
    },
    "v2_meta": "faiss_v2_meta_english",        # 可能是 faiss_v2_meta_english/faiss_meta_english

    # v1 索引（旧库，若不存在会自动跳过 v1 vs v2 对比）
    "v1_indexes": {
        "en_fine":  "faiss_index_english_fine",
        "en_std":   "faiss_index_english_std",
        "en_large": "faiss_index_english_large",
    },
    "v1_meta": "faiss_index_meta_english",

    # embedding 模型
    "model_path": "/vol/sunjilin/website/data/agent/models/bge-m3/",

    # 元数据 CSV
    "csv_path": "/vol/sunjilin/website/data/publication/english_content.csv",

    # 评测数据集（JSON，格式见下方说明）
    "eval_dataset": "eval_dataset.json",

    # 输出目录
    "output_dir": "granularity_eval_output",

    # 检索参数
    "top_k": 20,          # 每个库检索多少条
    "final_k": 5,         # 取前 K 条做精度评估

    # 是否启用 LLM 相关性打分（需要本地大模型 API 可用）
    "enable_llm_score": False,
}

# ══════════════════════════════════════════════════════════════
# 评测数据集格式说明（eval_dataset.json）
# 如果你已有 files/eval_dataset.json 可直接复用
# 格式如下：
# [
#   {
#     "id": "Q001",
#     "question_type": "基因功能类",
#     "question": "请介绍高粱 SbMADS51 基因的功能",
#     "standard_answer": "...",
#     "keypoints": ["MADS-box 转录因子", "调控开花时间"],
#     "required_citations": ["10.1093/plphys/kiac123"],
#     "relevant_papers": ["PaperA.pdf", "PaperB.pdf"]   # 可选，已知相关文献
#   },
#   ...
# ]
# ══════════════════════════════════════════════════════════════

QTYPE_ORDER = [
    "事实速查类", "基因功能类", "分子机制类",
    "QTL / GWAS类", "文献定位类", "综述比较类", "边界拒答类"
]

GRANULARITY_CN = {
    "en_fine":  "Fine (500字符)",
    "en_std":   "Std (1000字符)",
    "en_large": "Large (1500字符)",
    "en_para":  "Para (自然段)",
}


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def norm_text(s) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return "" if s.lower() == "nan" else s


def basename_lower(x: str) -> str:
    x = os.path.basename(norm_text(x)).lower()
    return re.sub(r"\.pdf$", "", x, flags=re.I)


def resolve_index_path(base_dir: str, index_name: str) -> Optional[str]:
    """
    自动解析索引路径，兼容 faiss_v2_meta_english/faiss_meta_english 嵌套情况。
    """
    candidate = os.path.join(base_dir, index_name)
    if os.path.exists(os.path.join(candidate, "index.faiss")):
        return candidate
    # 尝试子目录
    if os.path.isdir(candidate):
        for sub in os.listdir(candidate):
            sub_path = os.path.join(candidate, sub)
            if os.path.exists(os.path.join(sub_path, "index.faiss")):
                print(f"  [路径] {index_name} 在子目录 {sub} 中找到索引")
                return sub_path
    return None


def load_faiss_db(path: str, embed_model):
    """加载单个 FAISS 库，失败返回 None。"""
    try:
        from langchain_community.vectorstores import FAISS
        db = FAISS.load_local(path, embed_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"  [警告] 加载索引失败: {path} — {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════
def compute_lexical_overlap(query: str, content: str) -> float:
    """词汇重叠率（作为相关性代理指标之一）"""
    q_tokens = set(norm_text(query).lower().split())
    c_tokens = set(norm_text(content).lower().split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


def compute_keypoint_coverage(content: str, keypoints: List[str]) -> float:
    """检查 chunk 覆盖了多少关键点（精确字符串匹配）"""
    if not keypoints:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for kp in keypoints if kp.lower() in content_lower)
    return hits / len(keypoints)


def compute_paper_recall(
    retrieved_sources: List[str],
    relevant_papers: List[str]
) -> float:
    """命中率：检索结果中有多少是已知相关文献"""
    if not relevant_papers:
        return None
    retrieved_basenames = {basename_lower(s) for s in retrieved_sources}
    relevant_basenames  = {basename_lower(p) for p in relevant_papers}
    hits = retrieved_basenames & relevant_basenames
    return len(hits) / len(relevant_basenames)


def compute_diversity(sources: List[str]) -> float:
    """来源多样性：不同文献数 / 总检索数"""
    if not sources:
        return 0.0
    unique = len({basename_lower(s) for s in sources})
    return unique / len(sources)


def compute_avg_chunk_length(contents: List[str]) -> float:
    if not contents:
        return 0.0
    return sum(len(c) for c in contents) / len(contents)


def compute_noise_ratio(contents: List[str]) -> float:
    """
    估算噪声块比例（参考 build_sorghum_index_filter.py 的 is_noise_chunk 逻辑）。
    """
    if not contents:
        return 0.0
    noise_count = 0
    for c in contents:
        c = c.strip()
        if len(c) < 80:
            noise_count += 1
            continue
        lines = c.splitlines()
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            noise_count += 1
            continue
        alpha_count = sum(1 for ch in c if ch.isalpha())
        if alpha_count / max(len(c), 1) < 0.25:
            noise_count += 1
            continue
    return noise_count / len(contents)


# ══════════════════════════════════════════════════════════════
# 单次检索评测
# ══════════════════════════════════════════════════════════════
@dataclass
class RetrievalResult:
    index_key: str           # en_fine / en_std / en_large / en_para
    index_version: str       # v1 / v2
    question_id: str
    question_type: str
    question: str

    # 检索耗时（秒）
    latency: float = 0.0

    # 检索到的文档
    sources: List[str] = field(default_factory=list)
    contents: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    # 指标
    top1_score: float = 0.0
    avg_score: float = 0.0
    lexical_overlap_avg: float = 0.0
    keypoint_coverage_max: float = 0.0   # top-K 中最好的覆盖率
    keypoint_coverage_avg: float = 0.0   # top-K 均值
    paper_recall: Optional[float] = None  # 若有 relevant_papers
    source_diversity: float = 0.0
    avg_chunk_length: float = 0.0
    noise_ratio: float = 0.0
    retrieved_count: int = 0


def eval_single_index(
    db,
    index_key: str,
    index_version: str,
    item: dict,
    top_k: int,
    final_k: int,
) -> RetrievalResult:
    """对单个问题在单个索引库上做检索并计算指标。"""
    result = RetrievalResult(
        index_key=index_key,
        index_version=index_version,
        question_id=item["id"],
        question_type=item["question_type"],
        question=item["question"],
    )
    keypoints = item.get("keypoints", [])
    relevant_papers = item.get("relevant_papers", [])

    try:
        t0 = time.time()
        raw = db.similarity_search_with_score(item["question"], k=top_k)
        result.latency = time.time() - t0

        docs   = [r[0] for r in raw]
        scs    = [float(r[1]) for r in raw]

        result.retrieved_count = len(docs)
        result.sources  = [d.metadata.get("source", "") for d in docs]
        result.contents = [d.page_content for d in docs]
        result.scores   = scs

        # 取 final_k 条做精度计算
        top_sources  = result.sources[:final_k]
        top_contents = result.contents[:final_k]
        top_scores   = result.scores[:final_k]

        if top_scores:
            result.top1_score = top_scores[0]
            result.avg_score  = sum(top_scores) / len(top_scores)

        # 词汇重叠
        overlaps = [compute_lexical_overlap(item["question"], c)
                    for c in top_contents]
        result.lexical_overlap_avg = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # 关键点覆盖
        coverages = [compute_keypoint_coverage(c, keypoints)
                     for c in top_contents]
        result.keypoint_coverage_max = max(coverages) if coverages else 0.0
        result.keypoint_coverage_avg = sum(coverages) / len(coverages) if coverages else 0.0

        # 文献召回率
        result.paper_recall = compute_paper_recall(top_sources, relevant_papers)

        # 多样性与质量
        result.source_diversity  = compute_diversity(top_sources)
        result.avg_chunk_length  = compute_avg_chunk_length(top_contents)
        result.noise_ratio       = compute_noise_ratio(top_contents)

    except Exception as e:
        print(f"    [错误] {index_key} / {item['id']}: {e}")

    return result


# ══════════════════════════════════════════════════════════════
# 综合评分（加权合并各指标为单一分数，便于排名）
# ══════════════════════════════════════════════════════════════
def composite_score(r: RetrievalResult) -> float:
    """
    综合评分公式（满分 1.0）。
    可根据实际需要调整权重。
    """
    # dense score：FAISS L2 距离，越小越好，归一化为 [0,1]
    # 实际范围通常在 0~2，用 1 - score/2 做近似
    dense_score = max(0.0, 1.0 - r.avg_score / 2.0) if r.avg_score else 0.0

    score = (
        dense_score                    * 0.30 +
        r.lexical_overlap_avg          * 0.15 +
        r.keypoint_coverage_avg        * 0.30 +
        r.source_diversity             * 0.10 +
        (1.0 - r.noise_ratio)          * 0.10 +
        (1.0 - min(r.latency, 5) / 5)  * 0.05   # 速度奖励，超5秒不得分
    )
    return round(score, 4)


# ══════════════════════════════════════════════════════════════
# 建议生成
# ══════════════════════════════════════════════════════════════
def generate_recommendations(df_summary: pd.DataFrame) -> List[str]:
    recs = []
    dims = ["keypoint_coverage_avg", "source_diversity", "noise_ratio", "latency"]

    best_kp = df_summary["keypoint_coverage_avg"].idxmax()
    recs.append(
        f"✅ 关键点覆盖率最高：{best_kp}（{df_summary.loc[best_kp,'keypoint_coverage_avg']:.3f}）"
        f"，适合需要精确事实的 factoid / gene_function 类问题。"
    )

    best_div = df_summary["source_diversity"].idxmax()
    recs.append(
        f"✅ 来源多样性最高：{best_div}（{df_summary.loc[best_div,'source_diversity']:.3f}）"
        f"，适合 review / mechanism 类综合问题。"
    )

    worst_noise = df_summary["noise_ratio"].idxmax()
    best_noise  = df_summary["noise_ratio"].idxmin()
    recs.append(
        f"⚠️  噪声率最高：{worst_noise}（{df_summary.loc[worst_noise,'noise_ratio']:.3f}）"
        f"，噪声率最低：{best_noise}（{df_summary.loc[best_noise,'noise_ratio']:.3f}）。"
        f"若 {worst_noise} 噪声率>0.15，建议重新检查过滤规则。"
    )

    fastest = df_summary["latency"].idxmin()
    slowest = df_summary["latency"].idxmax()
    recs.append(
        f"⏱  延迟最低：{fastest}（{df_summary.loc[fastest,'latency']:.3f}s）"
        f"，延迟最高：{slowest}（{df_summary.loc[slowest,'latency']:.3f}s）。"
    )

    best_comp = df_summary["composite_score"].idxmax()
    recs.append(
        f"🏆 综合评分最优粒度：{best_comp}（{df_summary.loc[best_comp,'composite_score']:.4f}）"
        f"，建议在 config.py 中将该粒度排在 QUERY_TYPE_TO_INDEXES 首位。"
    )

    # 检查各粒度是否在合理区间
    for idx, row in df_summary.iterrows():
        if row["noise_ratio"] > 0.20:
            recs.append(
                f"❌ {idx} 噪声率 {row['noise_ratio']:.2%} > 20%，"
                f"强烈建议检查 is_noise_chunk 过滤逻辑或重建该粒度索引。"
            )
        if row["source_diversity"] < 0.40:
            recs.append(
                f"⚠️  {idx} 来源多样性 {row['source_diversity']:.2%} < 40%，"
                f"说明该粒度库存在严重重复命中，考虑增大 chunk_overlap 或降低 TOP_CHUNK_K。"
            )
        if row["keypoint_coverage_avg"] < 0.20:
            recs.append(
                f"⚠️  {idx} 关键点覆盖率 {row['keypoint_coverage_avg']:.2%} < 20%，"
                f"说明该粒度 chunk 过于碎片化或过于宏观，与问题语义匹配不足。"
            )

    recs.append(
        "📌 通用建议：在 QUERY_TYPE_TO_INDEXES 中，"
        "factoid/gene_function 优先 fine，mechanism/review 优先 large/std，"
        "qtl_gwas 使用 std+fine 组合效果较好。"
    )
    return recs


# ══════════════════════════════════════════════════════════════
# 导出 Excel
# ══════════════════════════════════════════════════════════════
def export_excel(
    all_results: List[RetrievalResult],
    output_path: str,
    recommendations: List[str],
    v1_v2_compare: Optional[pd.DataFrame],
):
    rows = []
    for r in all_results:
        rows.append({
            "版本":            r.index_version,
            "粒度":            GRANULARITY_CN.get(r.index_key, r.index_key),
            "粒度Key":         r.index_key,
            "题目ID":          r.question_id,
            "问题类型":        r.question_type,
            "问题":            r.question[:80],
            "检索耗时(s)":     round(r.latency, 4),
            "检索文档数":       r.retrieved_count,
            "Top1向量距离":    round(r.top1_score, 4),
            "TopK平均距离":    round(r.avg_score, 4),
            "词汇重叠率(avg)": round(r.lexical_overlap_avg, 4),
            "关键点覆盖(max)": round(r.keypoint_coverage_max, 4),
            "关键点覆盖(avg)": round(r.keypoint_coverage_avg, 4),
            "文献召回率":       round(r.paper_recall, 4) if r.paper_recall is not None else "N/A（无ground-truth）",
            "来源多样性":       round(r.source_diversity, 4),
            "平均chunk长度":   round(r.avg_chunk_length, 1),
            "噪声率(估算)":    round(r.noise_ratio, 4),
            "综合评分":        composite_score(r),
            "命中文献(前5)":   " | ".join(r.sources[:5]),
        })

    df_raw = pd.DataFrame(rows)

    # Sheet02: 粒度汇总（v2 only）
    df_v2 = df_raw[df_raw["版本"] == "v2"].copy()
    numeric_cols = [
        "检索耗时(s)", "Top1向量距离", "TopK平均距离",
        "词汇重叠率(avg)", "关键点覆盖(max)", "关键点覆盖(avg)",
        "来源多样性", "平均chunk长度", "噪声率(估算)", "综合评分"
    ]
    df_summary = (
        df_v2.groupby("粒度")[numeric_cols]
        .mean()
        .round(4)
    )
    df_summary.index.name = "粒度"
    # 用原始 key 做索引，方便 generate_recommendations
    df_summary_keyed = (
        df_v2.groupby("粒度Key")[numeric_cols]
        .mean()
        .round(4)
    )
    df_summary_keyed.columns = [c.replace("(s)", "").replace("(avg)", "_avg")
                                 .replace("(max)", "_max").replace("(估算)", "")
                                 .strip("_").replace(" ", "_")
                                 for c in df_summary_keyed.columns]
    df_summary_keyed.rename(columns={
        "检索耗时":            "latency",
        "词汇重叠率_avg":      "lexical_overlap_avg",
        "关键点覆盖_avg":      "keypoint_coverage_avg",
        "关键点覆盖_max":      "keypoint_coverage_max",
        "来源多样性":          "source_diversity",
        "噪声率":              "noise_ratio",
        "综合评分":            "composite_score",
    }, inplace=True)

    # Sheet03: 题型 × 粒度矩阵
    df_matrix = df_v2.pivot_table(
        values="综合评分",
        index="问题类型",
        columns="粒度",
        aggfunc="mean",
    ).round(4)
    df_matrix = df_matrix.reindex(
        [qt for qt in QTYPE_ORDER if qt in df_matrix.index]
    )

    # Sheet04: 题型 × 粒度 × 关键点覆盖
    df_kp_matrix = df_v2.pivot_table(
        values="关键点覆盖(avg)",
        index="问题类型",
        columns="粒度",
        aggfunc="mean",
    ).round(4)
    df_kp_matrix = df_kp_matrix.reindex(
        [qt for qt in QTYPE_ORDER if qt in df_kp_matrix.index]
    )

    # Sheet05: 建议
    recs = generate_recommendations(df_summary_keyed)
    recommendations.extend(recs)
    df_recs = pd.DataFrame({
        "序号": range(1, len(recommendations) + 1),
        "建议内容": recommendations,
    })

    # 写出
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="01_原始检索结果", index=False)
        df_summary.to_excel(writer, sheet_name="02_粒度汇总对比")
        df_matrix.to_excel(writer, sheet_name="03_题型×粒度综合评分")
        df_kp_matrix.to_excel(writer, sheet_name="04_题型×粒度关键点覆盖")
        if v1_v2_compare is not None:
            v1_v2_compare.to_excel(writer, sheet_name="05_v1_vs_v2对比", index=False)
        df_recs.to_excel(writer, sheet_name="06_建议", index=False)

    print(f"\n✅ 评测结果已导出: {out}")
    print(f"\n=== 粒度汇总 ===")
    print(df_summary.to_string())
    print(f"\n=== 综合评分矩阵（题型×粒度）===")
    print(df_matrix.to_string())


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def main():
    base_dir    = CONFIG["base_dir"]
    output_dir  = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载评测数据集 ────────────────────────────────────
    dataset_path = Path(CONFIG["eval_dataset"])
    if not dataset_path.exists():
        # 自动生成一个最小样例数据集
        print(f"[INFO] 未找到 {dataset_path}，生成样例数据集...")
        sample = [
            {
                "id": "Q001",
                "question_type": "基因功能类",
                "question": "What is the function of SbMADS51 gene in sorghum?",
                "standard_answer": "SbMADS51 is a MADS-box transcription factor that regulates flowering time in sorghum.",
                "keypoints": ["MADS-box", "transcription factor", "flowering time"],
                "required_citations": [],
                "relevant_papers": [],
            },
            {
                "id": "Q002",
                "question_type": "分子机制类",
                "question": "How does ABA signaling regulate drought tolerance in sorghum?",
                "standard_answer": "ABA signaling involves SnRK2 kinases and PP2C phosphatases to regulate stomatal closure.",
                "keypoints": ["ABA", "SnRK2", "PP2C", "stomatal closure", "drought"],
                "required_citations": [],
                "relevant_papers": [],
            },
            {
                "id": "Q003",
                "question_type": "QTL / GWAS类",
                "question": "What QTLs are associated with grain protein content in sorghum?",
                "standard_answer": "QTLs on chromosomes 1, 2, 4, 9 were identified in BTx623×IS3620C population.",
                "keypoints": ["QTL", "grain protein", "BTx623", "chromosomes"],
                "required_citations": [],
                "relevant_papers": [],
            },
            {
                "id": "Q004",
                "question_type": "综述比较类",
                "question": "What are the main mechanisms of stay-green trait in sorghum?",
                "standard_answer": "Stay-green involves delayed senescence, maintained chlorophyll, and nitrogen remobilization.",
                "keypoints": ["stay-green", "senescence", "chlorophyll", "nitrogen"],
                "required_citations": [],
                "relevant_papers": [],
            },
            {
                "id": "Q005",
                "question_type": "事实速查类",
                "question": "What is the chromosome location of SbDW3 gene in sorghum?",
                "standard_answer": "SbDW3 is located on chromosome 7 of sorghum genome.",
                "keypoints": ["chromosome 7", "SbDW3"],
                "required_citations": [],
                "relevant_papers": [],
            },
        ]
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f"  已生成样例数据集: {dataset_path}")
        print(f"  建议用你已有的 files/eval_dataset.json 替换它以获得更准确的评测结果。")

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] 加载评测数据集: {len(dataset)} 条问题")

    # ── 2. 加载 Embedding 模型 ───────────────────────────────
    print("\n[INFO] 加载 Embedding 模型...")
    import torch
    from sentence_transformers import SentenceTransformer
    from langchain_core.embeddings import Embeddings

    class _EmbWrap(Embeddings):
        def __init__(self, m, bs=64):
            self.m, self.bs = m, bs
        def embed_documents(self, texts):
            return self.m.encode(texts, batch_size=self.bs,
                                 normalize_embeddings=True,
                                 show_progress_bar=False).tolist()
        def embed_query(self, text):
            return self.m.encode([text], normalize_embeddings=True,
                                 show_progress_bar=False).tolist()[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = SentenceTransformer(CONFIG["model_path"], device=device)
    if device == "cuda":
        try:
            base_model.half()
        except Exception:
            pass
    embed_model = _EmbWrap(base_model)
    print(f"  设备: {device}")

    # ── 3. 加载 v2 索引库 ────────────────────────────────────
    print("\n[INFO] 加载 v2 索引库...")
    v2_dbs = {}
    for key, name in CONFIG["v2_indexes"].items():
        path = resolve_index_path(base_dir, name)
        if path:
            db = load_faiss_db(path, embed_model)
            if db:
                v2_dbs[key] = db
                print(f"  ✓ v2/{key}: {path}")
        else:
            print(f"  ✗ v2/{key}: 未找到 {name}")

    if not v2_dbs:
        print("[ERROR] 未加载到任何 v2 索引库，请检查路径配置。")
        return

    # ── 4. 加载 v1 索引库（可选）────────────────────────────
    print("\n[INFO] 加载 v1 索引库（若存在）...")
    v1_dbs = {}
    for key, name in CONFIG["v1_indexes"].items():
        path = resolve_index_path(base_dir, name)
        if path:
            db = load_faiss_db(path, embed_model)
            if db:
                v1_dbs[key] = db
                print(f"  ✓ v1/{key}: {path}")
        else:
            print(f"  - v1/{key}: 不存在，跳过")

    # ── 5. 开始评测 ──────────────────────────────────────────
    print(f"\n[INFO] 开始评测 ({len(dataset)} 题 × {len(v2_dbs)} 个v2库"
          f"{f' + {len(v1_dbs)} 个v1库' if v1_dbs else ''})...")

    all_results: List[RetrievalResult] = []
    top_k, final_k = CONFIG["top_k"], CONFIG["final_k"]

    for i, item in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {item['id']} | {item['question_type']}")

        # v2 各粒度
        for key, db in v2_dbs.items():
            r = eval_single_index(db, key, "v2", item, top_k, final_k)
            all_results.append(r)

        # v1 各粒度（仅 fine/std/large，与 v2 对应粒度对比）
        for key, db in v1_dbs.items():
            if key in v2_dbs:   # 只对比双方都有的粒度
                r = eval_single_index(db, key, "v1", item, top_k, final_k)
                all_results.append(r)

    # ── 6. v1 vs v2 对比表 ───────────────────────────────────
    v1_v2_compare = None
    if v1_dbs:
        rows_cmp = []
        numeric_cols = [
            "keypoint_coverage_avg", "source_diversity",
            "noise_ratio", "latency", "composite_score"
        ]
        for key in v1_dbs:
            if key not in v2_dbs:
                continue
            v1_items = [r for r in all_results if r.index_version == "v1" and r.index_key == key]
            v2_items = [r for r in all_results if r.index_version == "v2" and r.index_key == key]

            def avg(lst, attr):
                vals = [getattr(x, attr) for x in lst
                        if getattr(x, attr) is not None]
                return round(sum(vals) / len(vals), 4) if vals else None

            def comp_avg(lst):
                vals = [composite_score(x) for x in lst]
                return round(sum(vals) / len(vals), 4) if vals else None

            for attr, label in [
                ("keypoint_coverage_avg", "关键点覆盖(avg)"),
                ("source_diversity",      "来源多样性"),
                ("noise_ratio",           "噪声率"),
                ("latency",               "检索耗时(s)"),
            ]:
                v1_val = avg(v1_items, attr)
                v2_val = avg(v2_items, attr)
                delta  = round(v2_val - v1_val, 4) if v1_val and v2_val else None
                rows_cmp.append({
                    "粒度":      GRANULARITY_CN.get(key, key),
                    "指标":      label,
                    "v1均值":    v1_val,
                    "v2均值":    v2_val,
                    "Delta(v2-v1)": delta,
                    "改善方向":  "↑ 越大越好" if attr not in ("noise_ratio", "latency") else "↓ 越小越好",
                    "是否提升":  (
                        "✅" if (delta and delta > 0 and attr not in ("noise_ratio", "latency"))
                        or (delta and delta < 0 and attr in ("noise_ratio", "latency"))
                        else ("❌" if delta else "N/A")
                    ),
                })
            # 综合评分
            v1_comp = comp_avg(v1_items)
            v2_comp = comp_avg(v2_items)
            delta_comp = round(v2_comp - v1_comp, 4) if v1_comp and v2_comp else None
            rows_cmp.append({
                "粒度": GRANULARITY_CN.get(key, key),
                "指标": "综合评分",
                "v1均值": v1_comp,
                "v2均值": v2_comp,
                "Delta(v2-v1)": delta_comp,
                "改善方向": "↑ 越大越好",
                "是否提升": "✅" if delta_comp and delta_comp > 0 else ("❌" if delta_comp else "N/A"),
            })
        v1_v2_compare = pd.DataFrame(rows_cmp)

    # ── 7. 导出 Excel ────────────────────────────────────────
    out_path = output_dir / "granularity_eval_results.xlsx"
    export_excel(
        all_results,
        str(out_path),
        recommendations=[],
        v1_v2_compare=v1_v2_compare,
    )


if __name__ == "__main__":
    main()
