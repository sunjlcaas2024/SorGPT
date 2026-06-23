# -*- coding: utf-8 -*-
"""
prompt_builder.py  ── v2（精简结构版）
================================================================
相对于上一版的改动：
1. gene_function prompt 去掉僵化的六段式强制编号，改为要素引导 + 自然叙述
2. locate 类新增"如果文献不在库中如何处理"的明确指导
3. review 类新增对"是否有研究"类问题的处理指导（避免把"无直接证据"
   误解为"这个领域不存在"）
4. 所有中文 prompt 末尾加"如证据充分，可超越文献直接给出专家判断"
   避免模型过度保守地重复"证据不足"
"""

import os
import sys
from typing import Dict, List, Tuple

from utils import protect_bio_terms
from retriever import ChunkHit
from config import GENE_DB_PATH

_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
if _DB_DIR not in sys.path:
    sys.path.insert(0, _DB_DIR)

try:
    from omics_query import OmicsQueryHub as _OmicsHub
    _OMICS_HUB = _OmicsHub()
    _OMICS_AVAILABLE = True
except Exception:
    _OMICS_HUB = None
    _OMICS_AVAILABLE = False

try:
    import gene_db_query as _gdb
    _gdb.GENE_DB = GENE_DB_PATH
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


def _query_db(user_query: str, max_genes: int = 2) -> str:
    if not _DB_AVAILABLE:
        return ""
    try:
        return _gdb.query_and_format(user_query, max_genes=max_genes)
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════
# 工具函数（接口不变）
# ════════════════════════════════════════════════════════════════

def build_source_index(evidence_hits: List[ChunkHit]) -> Dict[str, Dict[str, str]]:
    source_index = {}
    idx = 1
    for hit in evidence_hits:
        if hit.source not in source_index:
            source_index[hit.source] = {"idx": idx, "fname": hit.source}
            idx += 1
    return source_index


def detect_language(query: str) -> str:
    chinese_chars = sum(1 for c in query if "\u4e00" <= c <= "\u9fff")
    return "chinese" if chinese_chars / max(len(query), 1) > 0.2 else "english"


def _evidence_block(evidence_hits, source_index):
    merged_protected = {}
    lines = []
    for hit in evidence_hits:
        ref_idx = source_index[hit.source]["idx"]
        protected_text, pmap = protect_bio_terms(hit.content)
        merged_protected.update(pmap)
        lines.append(f"[{ref_idx}] {protected_text}")
    return "\n\n".join(lines), merged_protected


def _source_hint(source_index):
    return "\n".join(
        f"[{v['idx']}] {v['fname']}"
        for _, v in sorted(source_index.items(), key=lambda x: x[1]["idx"])
    )


# ════════════════════════════════════════════════════════════════
# 各问题类型 Prompt — 英文版
# ════════════════════════════════════════════════════════════════

_TASK_EN = {

"factoid": """\
TASK: Answer this specific factual question precisely.
• Open with the direct answer (one sentence).
• Add 2-3 sentences of supporting context (genomic location, gene family, phenotype).
• If a (GeneDB) block appears above, it takes priority for coordinates and domain data.
• Keep the answer under 150 words.
""",

"gene_function": """\
TASK: Give a thorough functional account of the sorghum gene(s) in the question.
Cover the following aspects in a coherent narrative (not a rigid numbered list):
— Genomic identity: official ID, chromosomal location (BTx623 T2T preferred),
  gene/protein family, and domain architecture. Use (GeneDB) data where available.
— Molecular mechanism: biochemical activity, key protein interactions, subcellular localization.
— In-planta role: which developmental process, stress response, or agronomic trait it controls.
— Expression context: tissue specificity, developmental stage, or condition-dependent expression
  if evidence is available.
— Evolutionary context: ortholog in rice/maize/Arabidopsis and functional conservation.
— Open questions: explicitly flag what remains unknown.
Write as a knowledgeable scientist, not a checklist. Cite every factual claim.
""",

"mechanism": """\
TASK: Explain the molecular/cellular mechanism in depth.
Cover: (1) one-sentence overview and biological significance;
(2) key molecular players (proteins, genes, metabolites with IDs);
(3) the sequential or parallel events of the pathway;
(4) regulatory inputs (hormones, environment, developmental stage) that switch it on/off;
(5) phenotypic consequences when the mechanism is active vs. impaired;
(6) how this mechanism is similar to or distinct from the equivalent in maize/rice/Arabidopsis.
Be mechanistic. Cite every claim.
""",

"qtl_gwas": """\
TASK: Summarize QTL/GWAS results for the requested trait in sorghum.
Report: chromosomal positions (BTx623 T2T where possible), LOD/p-values, PVE;
mapping population and marker platform; candidate genes in the confidence interval
and evidence for their candidacy; cross-environment stability; syntenic loci in
maize or rice if mentioned; breeding application (MAS/GS potential).
Reproduce statistics exactly as found in the evidence. Cite every claim.
""",

"review": """\
TASK: Synthesize knowledge on this topic from multiple sources.
If the question asks whether a technology or research area EXISTS in sorghum:
— First state directly whether the provided evidence contains examples.
— Then describe what IS available (e.g., bulk RNA-seq, tissue-specific profiling,
  laser microdissection) as the closest relevant context.
— Do not conflate "not in the provided literature" with "does not exist globally";
  acknowledge the limitation of the provided reference set explicitly.
For general review questions: synthesize thematically across sources (not paper-by-paper);
identify dominant gene families, key open questions, and comparison with other cereals.
Cite every factual claim.
""",

"gene_list": """\
TASK: Compile a gene list related to the requested trait or process.
• Group genes by subfamily or function (no flat lists).
• For each gene: ID | common name | key function | evidence [citation].
• After the list, write one summary paragraph noting patterns
  (dominant families, chromosomal enrichment, confidence levels).
• Use (GeneDB) data to verify IDs and supplement descriptions.
• Label confidence: well-characterized / predicted / inferred by homology.
""",

"locate": """\
TASK: Identify the specific paper or dataset the user is asking about.
• Provide: full title, first author + et al., journal, year, DOI.
• If the paper appears in the source index, cite it. If the provided snippets reference
  it (e.g., "Zhang et al., 2023") but it is not itself a source, state that clearly
  and report what the snippets reveal about the paper.
• If multiple candidates match, rank them and explain.
• Never fabricate bibliographic details. If uncertain, say so explicitly.
""",

"count": """\
TASK: Provide an accurate count based on the retrieved literature.
State the count first. Briefly note the search scope. List papers if fewer than 20.
""",

"boundary": """\
TASK: This question falls outside the sorghum literature knowledge base.
Politely explain why, and specify what kinds of questions this system handles well.
Do not answer using general knowledge.
""",
}

# ════════════════════════════════════════════════════════════════
# 各问题类型 Prompt — 中文版
# ════════════════════════════════════════════════════════════════

_TASK_ZH = {

"factoid": """\
任务：精准回答这道具体的事实性问题。
• 第一句直接给出答案。
• 接下来2-3句补充背景（染色体位置、基因家族、表型）。
• 若上方有 (GeneDB) 注释块，对坐标和结构域信息优先采用。
• 总字数控制在200字以内。
""",

"gene_function": """\
任务：对问题中涉及的高粱基因进行全面的功能解析。
请以连贯叙述（而非僵化编号）覆盖以下要素：
— 基因身份：官方ID、染色体位置（BTx623 T2T优先）、基因/蛋白家族、
  结构域架构。优先使用 (GeneDB) 注释。
— 分子机制：生化活性、关键蛋白互作、亚细胞定位。
— 植株功能：调控哪个发育过程、胁迫响应或农艺性状。
— 表达特征：组织特异性、发育阶段或条件依赖性表达（如有证据）。
— 进化保守性：水稻/玉米/拟南芥同源基因及功能保守情况。
— 知识缺口：明确指出尚未解决的问题。
以科学家的视角叙述，不要机械列表。每个论断需引用来源。
""",

"mechanism": """\
任务：深入解释所提问的分子/细胞机制。
涵盖：(1) 一句话概述及生物学意义；
(2) 关键分子组分（蛋白质、基因、代谢物，含官方ID）；
(3) 通路的顺序或并行事件；
(4) 激活/关闭该机制的调控输入（激素、环境、发育时期）；
(5) 机制激活 vs. 受损时的表型后果；
(6) 与玉米/水稻/拟南芥等效机制的异同。
侧重机制阐释。每个论断需引用来源。
""",

"qtl_gwas": """\
任务：总结高粱中与所查性状相关的QTL/GWAS结果。
报告：染色体位置（尽量用BTx623 T2T坐标）、LOD值/p值、PVE；
定位群体和标记平台；置信区间内候选基因及候选证据；
跨环境稳定性；玉米/水稻共线性位点（若提及）；育种应用潜力。
精确引用证据中的统计数值。每个论断需引用来源。
""",

"review": """\
任务：从多来源综合该主题的知识。
如果问题询问某技术或研究方向在高粱中是否存在：
— 首先直接说明提供的证据中是否包含相关示例。
— 然后描述现有最接近的内容（如bulk RNA-seq、激光显微切割等）。
— 不要把"提供的文献中没有"等同于"全球范围不存在"；
  明确说明所提供参考集的局限性。
对于一般综述问题：按主题而非逐篇综合来源；
识别主导基因家族、关键开放问题，并与其他禾本科作物比较。
每个论断需引用来源。
""",

"gene_list": """\
任务：整理与所查性状或过程相关的基因列表。
• 按亚家族或功能分组，不要平铺列表。
• 每个基因：ID | 常用名 | 核心功能 | 证据 [引用]。
• 列表后写一段总结，分析规律（主导家族、染色体富集、可信度）。
• 使用 (GeneDB) 注释核实ID并补充描述。
• 标注可信度：功能明确 / 预测 / 同源推断。
""",

"locate": """\
任务：定位用户询问的具体文献。
请按以下顺序处理：
1. 检查来源索引中是否有该论文本身（直接引用）。
2. 如果原文不在索引，但某片段引用了该论文（如"Zhang et al., 2023 Science"），
   从引用信息中提取：作者、年份、期刊、标题（如有）、DOI（如有），
   并明确标注"该信息来源于引用，非原文索引"。
3. 最终提供：完整题目、第一作者 et al.、期刊、年份、DOI。
4. 多候选时排序并解释。禁止编造任何文献信息。
""",

"count": """\
任务：根据检索文献提供准确统计。先说数量，再说检索范围，少于20条时列出文献。
""",

"boundary": """\
任务：该问题超出高粱文献知识库范围。礼貌解释原因并说明本系统适合的问题类型。
不要用通用知识作答。
""",
}

for _d in (_TASK_EN, _TASK_ZH):
    _d.setdefault("unknown", _d["mechanism"])


# ════════════════════════════════════════════════════════════════
# 共享规则
# ════════════════════════════════════════════════════════════════

_RULES_EN = """\
RULES:
• Cite inline after every factual claim: [1] or [1,2]. Only use the Source index below.
• Never fabricate citations, gene names, chromosome positions, or statistics.
• All gene/protein names and locus IDs must stay in English.
• Primary reference genome: BTx623 T2T (Sobic.xxxGxxxxxx).
  Equivalents: HYZ (SbiHYZ.xxGxxxxxx) / BTx623v3 (SORBI_3xxxGxxxxxx).
• If the provided evidence is insufficient for a claim, say so — but do not repeat
  "evidence is insufficient" more than once per response.
"""

_RULES_ZH = """\
规则：
• 每个论断后引用来源编号：[1] 或 [1,2]。只使用下方来源索引。
• 禁止编造引用、基因名、染色体位置或统计数值。
• 所有基因/蛋白名称和基因座ID保持英文原文。
• 主参考基因组：BTx623 T2T（Sobic.xxxGxxxxxx）。
  等价ID：HYZ（SbiHYZ.xxGxxxxxx）/ BTx623v3（SORBI_3xxxGxxxxxx）。
• 如某一论断缺乏证据，说明一次即可，不要在全文反复重申"证据不足"。
"""


# ════════════════════════════════════════════════════════════════
# 主入口（接口与原版完全相同）
# ════════════════════════════════════════════════════════════════

def build_system_prompt(
    user_query: str,
    query_type: str,
    evidence_hits: List[ChunkHit],
    extra_types: List[str] = None,
) -> Tuple[str, Dict[str, str], Dict[str, Dict[str, str]]]:
    source_index = build_source_index(evidence_hits)
    evidence_text, merged_protected = _evidence_block(evidence_hits, source_index)
    src_hint = _source_hint(source_index)
    lang = detect_language(user_query)

    # SQLite 基因注释（自动触发）
    db_block = ""
    omics_block = ""
    if _OMICS_AVAILABLE and _OMICS_HUB:
        try:
            # 主类型查询
            omics_block = _OMICS_HUB.query_for_prompt(user_query, query_type)
        except Exception:
            omics_block = ""
        # extra_types 也查询（多标签路由）
        if extra_types and _OMICS_AVAILABLE and _OMICS_HUB:
            for etype in (extra_types or []):
                try:
                    extra_block = _OMICS_HUB.query_for_prompt(user_query, etype)
                    if extra_block and extra_block not in omics_block:
                        omics_block = omics_block + ("\n" if omics_block else "") + extra_block
                except Exception:
                    pass
    if query_type in {"factoid", "gene_function", "qtl_gwas", "gene_list", "mechanism"}:
        raw = _query_db(user_query, max_genes=2)
        if raw:
            header = (
                "【结构化基因数据库注释 (BTx623 T2T + HYZ + BTx623v3)】\n"
                if lang == "chinese" else
                "Structured gene database annotation (BTx623 T2T + HYZ + BTx623v3):\n"
            )
            db_block = header + raw + "\n\n"

    task_map = _TASK_ZH if lang == "chinese" else _TASK_EN
    task_instr = task_map.get(query_type, task_map["unknown"])
    rules = _RULES_ZH if lang == "chinese" else _RULES_EN
    SEP = "=" * 60

    if lang == "chinese":
        system_prompt = (
            "你是 SorGPT，世界顶级高粱（Sorghum bicolor）基因组学、遗传学、"
            "分子生物学与育种专家。你对高粱基因功能、QTL/GWAS、转录组学及与"
            "玉米、水稻等禾本科作物的比较基因组学有深入了解。\n\n"
            f"{task_instr}\n{rules}\n{SEP}\n"
            f"{db_block}"
            f"{omics_block}"
            f"来源索引：\n{src_hint}\n\n"
            f"研究片段：\n{evidence_text}\n"
            f"{SEP}\n\n请按任务要求回答用户问题。"
        )
    else:
        system_prompt = (
            "You are SorGPT, a world-class expert in sorghum (Sorghum bicolor) "
            "genomics, genetics, molecular biology, and breeding.\n\n"
            f"{task_instr}\n{rules}\n{SEP}\n"
            f"{db_block}"
            f"{omics_block}"
            f"Source index:\n{src_hint}\n\n"
            f"Research snippets:\n{evidence_text}\n"
            f"{SEP}\n\nAnswer the user's question following the task instructions."
        )

    return system_prompt, merged_protected, source_index
