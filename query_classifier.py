# -*- coding: utf-8 -*-
"""
query_classifier.py  ── 多标签路由版
======================================
核心改变：
classify_query_type() 返回 (primary_type, extra_types, en_keywords)
- primary_type:  主类型，决定 prompt 模板和检索策略
- extra_types:   附加类型列表，用于追加检索和数据库查询
- en_keywords:   英文检索关键词

典型场景：
  "AT1基因在盐碱地的QTL定位"
    → primary="gene_function", extra=["qtl_gwas"], kw=...
  "高粱产量相关基因有哪些，主要定位在哪条染色体"
    → primary="gene_list",    extra=["qtl_gwas"], kw=...
  "BTx623基因组有多大"
    → primary="factoid",      extra=[], kw=...
  "PF02365的基因有多少个"
    → primary="factoid",      extra=["count"], kw=...
"""

import re, os, json
from typing import Tuple, List
from utils import norm_text

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(f):
    p = os.path.join(_BASE_DIR, f)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

# ── 基因名正则（支持中文前后紧跟）──────────────────────────────
_GENE_RE = re.compile(
    r"""
    Sobic\.\d{3}G\d{6}
    | SbiHYZ\.\d{2}G\d{6}
    | SORBI_3\d{3}G\d{6}
    | (?<![a-zA-Z0-9])(?:
        AT1|SbAT1|GS3|BR2|
        DW[1-4]|SbDW[1-4]|
        MA[1-7]|SbMA[1-7]|
        TB1|SbTB1|SH1|SbSH1|
        Y1|RCN1|B1|AltSB|ARG1|BY1|GC1|
        SnRK\d*|WRKY\d+|NAC\d+|MYB\d+|
        bHLH\d+|ERF\d+|MADS\d+|
        SbMADS\d+|SbBX\d*|SbCYP\d+|SbWRKY\d+|
        Stg[1-4]|SbFT\d*|SbVRN\d*
    )(?![a-zA-Z0-9])
    | (?<![a-zA-Z0-9])[A-Z]{2,5}\d{1,3}[a-z]?(?![a-zA-Z0-9])
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ── Pfam / IPR / GO 编号正则 ─────────────────────────────────────
_DB_ID_RE = re.compile(
    r'\bPF\d{5}\b|\bIPR\d{6}\b|\bGO:\d{7}\b|\bPANTHER:PT\w+\b',
    re.IGNORECASE,
)

def _has_gene(q: str) -> bool:
    return bool(_GENE_RE.search(q))

def _has_db_id(q: str) -> bool:
    return bool(_DB_ID_RE.search(q))

# ── 基因名 → 检索词注入 ──────────────────────────────────────────
_GENE_CTX = {
    r"AT1|SbAT1":          "alkaline tolerance G protein gamma subunit Gγ alkaline sensitivity",
    r"DW1|SbDW1":          "dwarfism height brassinosteroid sorghum",
    r"DW2|SbDW2":          "dwarfism height protein kinase sorghum",
    r"DW3|SbDW3":          "dwarfism height auxin transport P-glycoprotein sorghum",
    r"DW4|SbDW4":          "dwarfism height sorghum",
    r"MA[1-7]|SbMA[1-7]":  "maturity photoperiod flowering time circadian clock sorghum",
    r"TB1|SbTB1":          "tiller branching domestication maize sorghum",
    r"SH1|SbSH1":          "shattering seed dispersal domestication sorghum",
    r"Y1":                 "yellow seed carotenoid pigment sorghum",
    r"B1":                 "brown midrib lignin cell wall sorghum",
    r"SnRK1":              "energy sensing kinase ABA starch metabolism sorghum",
    r"GS3":                "grain size G protein gamma subunit rice sorghum",
    r"AltSB":              "aluminum tolerance MATE sorghum",
    r"ARG1":               "fungal resistance NLR immune receptor sorghum",
    r"Stg[1-4]":           "stay-green drought tolerance post-flowering sorghum",
}

# ── 话题 → 检索词注入 ─────────────────────────────────────────────
_TOPIC_CTX = {
    "单细胞":     "single cell RNA sequencing scRNA-seq cell atlas",
    "空间转录组": "spatial transcriptomics Visium gene expression",
    "scRNA":      "single cell RNA sequencing cell type atlas transcriptome",
    "驯化":       "domestication selection sweep crop wild ancestor sorghum",
    "干旱":       "drought stress tolerance transcriptome ABA signaling sorghum",
    "drought":    "drought stress tolerance ABA transcriptome sorghum",
    "盐碱":       "saline alkaline stress tolerance ion transport sorghum",
    "alkaline":   "alkaline stress tolerance G protein AT1 pH sorghum",
    "开花":       "flowering time photoperiod maturity circadian sorghum",
    "flowering":  "flowering time photoperiod maturity locus sorghum",
    "产量":       "yield grain size QTL GWAS agronomic trait sorghum",
    "耐铝":       "aluminum tolerance MATE transporter AltSB sorghum",
    "stay-green": "stay-green Stg1 Stg2 Stg3 Stg4 drought senescence sorghum",
    "持绿":       "stay-green Stg1 Stg2 Stg3 Stg4 drought senescence sorghum",
}

def _build_keywords(query: str) -> str:
    zh_to_en      = _load_json("keywords_zh2en.json")
    domain_inject = _load_json("domain_injection.json")
    kws = []
    q_lower = query.lower()

    for zh, en in zh_to_en.items():
        if zh in query:
            kws.append(en)

    kws.extend(re.findall(r"[A-Za-z][A-Za-z0-9_\-\.]*[A-Za-z0-9]|[A-Za-z]{2,}", query))

    for pat, ctx in _GENE_CTX.items():
        if re.search(r'(?<![a-zA-Z0-9])(?:' + pat + r')(?![a-zA-Z0-9])',
                     query, re.IGNORECASE):
            kws.append(ctx)

    for trigger, ctx in _TOPIC_CTX.items():
        if trigger.lower() in q_lower:
            kws.append(ctx)

    for key, inj in domain_inject.items():
        if key in q_lower or key in query:
            kws.append(inj)

    if not any('\u4e00' <= c <= '\u9fff' for c in query):
        kws.append(norm_text(query))

    if "sorghum" not in " ".join(kws).lower():
        kws.append("sorghum")

    seen, result = set(), []
    for kw in kws:
        k = kw.strip().lower()
        if k and k not in seen:
            seen.add(k)
            result.append(kw.strip())
    return ", ".join(result) if result else norm_text(query)


# ═══════════════════════════════════════════════════════════════
# 信号检测函数（每个返回 bool，组合使用）
# ═══════════════════════════════════════════════════════════════

def _is_boundary(q): return any(p in q for p in [
    "价格","收益","更好吃","明年产量","治疗糖尿病","市场走势",
    "2025年田间","2030年","预测","商业化前景","会上涨","会下跌",
    "tastes better","taste better","better taste","更好吃","哪个更好吃",
    "market price","stock price","cure diabetes","forecast yield",
    "predict","商业化","未来方向预测","年产量预测",
    # v2 fix: boundary patterns that were misclassified as mechanism
    "多赚多少钱","每亩多赚","经济效益比较","是否绝对安全",
    "临床证据","补贴政策","补贴金额","能减肥","每天吃多少克",
    "运动补剂","化妆品","抗衰老","碳达峰","碳中和","碳汇量",
    "审批进度","商业化时间表","哪个更好喝","哪个品种排名","最佳品种",
    "RCT临床试验","乳清蛋白","绝对安全",
    "profit","subsidy","cosmetic","anti-aging","carbon neutral",
    "clinical trial","weight loss","taste better than",
])

def _is_count(q): return any(p in q for p in [
    "多少篇","几篇","统计","列出所有","有哪些文章","有哪些文献","有多少个","有多少","有几",
    "how many papers","how many articles","list all","count of papers","how many cloned genes","how many genes have been cloned",
])

def _is_factoid(q, q_orig): return any(p in q for p in [
    # 数量/大小类
    "多少个","多少条","多少种","多少成员","有几个","有多少","多大","有多少bp","有多少mb",
    "how many genes","how many members","how many",
    # 位置/编号类
    "基因id","染色体位置","位于哪条","在第几号","基因组大小","染色体数目",
    "gene id","chromosome location","what chromosome","how big is",
    "genome size","how large",
    # 结构域类（PF/IPR编号）
]) or _has_db_id(q_orig)

def _is_locate(q): return any(p in q for p in [
    "哪篇","哪一篇","谁发表","发表在哪","首次报道","题目是什么",
    "具体文章","具体论文","原文是","发表在nature","发表在science",
    "发表在cell","发表在molecular plant","发表于","哪本期刊",
    "which paper","which article","first reported","find the paper",
    "published in science","published in nature","published in cell",
    "which journal","doi of","citation for",
    "find the paper","find the article","find the study",
    "representative paper","key paper","landmark paper","seminal paper",
    "请找","找到这篇","找出这篇",
    # v2 fix: locate patterns that were misclassified as gene_function
    "请提供作者","提供DOI","提供完整引用","请提供DOI",
    "提供作者列表","发表在哪个期刊","哪年发表","第一作者是谁",
    "克隆论文","首次鉴定论文","首次克隆论文","原始克隆",
    "查找.*论文","查证.*论文","经典综述.*DOI",
    "提供.*引用信息","提供.*引用","完整引用信息",
    "provide.*citation","provide.*DOI","provide.*full citation",
    "which journal","first author","original paper",
    "classic review","first identified","first reported",
    "查找那篇","找到那篇",
])

def _is_qtl(q): return any(p in q for p in [
    "qtl","gwas","位点","关联分析","连锁分析","遗传图谱","定位",
    "quantitative trait","locus","loci","genome-wide association",
    "linkage mapping","snp association","marker assisted","mapping",
])

def _is_gene_function(q, q_orig):
    has_gene = _has_gene(q_orig)
    if not has_gene:
        return False
    # 含机制词时让 mechanism 接管，不走 gene_function
    mech_words = [
        "how does","how do","mechanism","pathway","signaling",
        "promoter","inversion","molecular mechanism","regulate the",
        "分子机制","调控机制","信号通路",
    ]
    if any(p in q for p in mech_words):
        return False
    func_kws = [
        "功能","作用","介绍","分析","是什么","编码","表达",
        "讲","解释","什么基因","基因是","有什么","做什么",
        "突变","缺失","过表达","敲除","表型",
        "function","role","what does","what is",
        "describe","explain","encode","expression",
        "mutant","knockout","overexpression","phenotype","about",
    ]
    has_verb = any(p in q for p in func_kws)
    is_short = len(q.strip()) <= 40
    is_gene_suffix = q.rstrip().endswith(("基因", "gene", "蛋白", "protein"))
    return has_verb or is_short or is_gene_suffix

def _is_review(q): return any(p in q for p in [
    "进展","综述","比较","总结","盘点","梳理","是否有","有没有","目前是否",
    "单细胞","空间转录组","scRNA","spatial transcriptomics",
    "驯化","选择信号","选择清除","应用前景","未来方向",
    "review","overview","summary","progress","advances","comparison",
    "current understanding","recent advances","what is known",
    "domestication","selection sweep","prospects","perspective",
    "compare","comparison","similarities","differences","versus","vs.",
    "异同","相同点","不同点","对比","相比","与...相比",
])

def _is_gene_list(q): return any(p in q for p in [
    "有哪些基因","哪些基因","关键基因","已知基因","鉴定到的基因","候选基因","相关基因","功能基因","调控基因","有哪些","相关的基因",
    "有哪些转录因子","what genes","which genes","genes involved in",
    "genes associated with","genes controlling","key genes for",
])

def _is_mechanism(q): return any(p in q for p in [
    "机制","通路","调控","信号","互作","如何","怎么",
    "mechanism","pathway","regulation of","signaling","interaction",
    "how does","why does","what happens",
])


# ═══════════════════════════════════════════════════════════════
# 主函数：多标签路由
# ═══════════════════════════════════════════════════════════════

def classify_query_type(query: str) -> Tuple[str, List[str], str]:
    """
    返回 (primary_type, extra_types, en_keywords)

    primary_type:  主类型（决定 prompt 模板）
    extra_types:   附加类型列表（追加检索 + 数据库查询）
    en_keywords:   英文检索关键词
    """
    q      = norm_text(query).lower()
    q_orig = norm_text(query)
    kw     = _build_keywords(query)

    # ── boundary（最高优先级，直接返回）────────────────────────
    # 年份预测/田间试验类：即使含基因名也是 boundary
    if _is_boundary(q) or any(p in q for p in [
        "2025年田间","2026年","2027年","2028年","2029年","2030年",
        "田间试验中的产量","未来几年","预测产量",
    ]):
        return "boundary", [], kw

    # ── count（文献统计，直接返回）──────────────────────────────
    if _is_count(q):
        return "count", [], kw

    # ── locate（文献定位，直接返回）─────────────────────────────
    if _is_locate(q):
        return "locate", [], kw

    # ────────────────────────────────────────────────────────────
    # 以下：收集所有匹配的类型标签，再决定主次
    # ────────────────────────────────────────────────────────────
    tags = set()

    if _is_factoid(q, q_orig):  tags.add("factoid")
    if _is_qtl(q):              tags.add("qtl_gwas")
    if _is_gene_function(q, q_orig): tags.add("gene_function")
    if _is_review(q):           tags.add("review")
    if _is_gene_list(q):        tags.add("gene_list")
    if _is_mechanism(q):        tags.add("mechanism")
    if _has_gene(q_orig) and not tags:
        tags.add("gene_function")

    # ── locate 优先：即使用户问了基因名，如果核心意图是找论文 → locate 为主 ──
    locate_strong = any(p in q for p in [
        "请提供作者","提供DOI","提供完整引用","完整引用",
        "提供作者","第一作者是谁","哪篇论文","查找.*论文",
        "发表在哪个期刊","哪年发表","克隆论文","首次鉴定",
        "provide.*citation","first author","first reported",
    ])
    if locate_strong and "locate" in tags:
        tags = {"locate"}  # override all other signals
    elif locate_strong:
        tags.add("locate")

    # ── 无标签兜底 ──────────────────────────────────────────────
    if not tags:
        tags.add("review")  # v2: changed from mechanism to review (less aggressive default)

    # ── 主类型优先级（从高到低）─────────────────────────────────
    PRIMARY_ORDER = [
        "factoid",       # 有明确数量/位置答案
        "gene_function", # 基因功能解析
        "qtl_gwas",      # 遗传定位
        "gene_list",     # 基因列表
        "review",        # 综述比较
        "mechanism",     # 分子机制
    ]
    primary = next((t for t in PRIMARY_ORDER if t in tags), "mechanism")
    extra   = sorted(tags - {primary})  # 附加类型

    # ── 特殊组合规则 ─────────────────────────────────────────────
    # "基因有哪些 + 定位" → gene_list 为主，qtl_gwas 为辅
    if "gene_list" in tags and "qtl_gwas" in tags:
        primary = "gene_list"
        extra   = ["qtl_gwas"] + [t for t in sorted(tags - {"gene_list", "qtl_gwas"})]

    # "基因功能 + QTL" → gene_function 为主，qtl_gwas 为辅
    if "gene_function" in tags and "qtl_gwas" in tags:
        primary = "gene_function"
        extra   = ["qtl_gwas"] + [t for t in extra if t != "qtl_gwas"]

    # 含明确 GWAS/QTL 词且问题重心是定位结果 → qtl_gwas 升主
    gwas_dominant = any(p in q for p in [
        "gwas", "qtl", "haplotype", "单倍型", "连锁不平衡",
        "gwas和单倍型", "gwas 和", "gwas结果",
    ])
    intro_words = any(p in q for p in ["介绍","相关基因的","相关基因"])
    if gwas_dominant and intro_words and "qtl_gwas" in tags:
        primary = "qtl_gwas"
        extra   = [t for t in sorted(tags - {"qtl_gwas"})]

    # "factoid + count类" → factoid 为主，不追加
    if "factoid" in tags and len(tags) == 1:
        extra = []

    return primary, extra, kw
