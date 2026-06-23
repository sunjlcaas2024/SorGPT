# -*- coding: utf-8 -*-
"""
utils.py
========================
【作用】
存放项目中的通用工具函数，避免在多个脚本中重复写相同逻辑。
主要包括：
1. 文本清洗
2. 列名标准化
3. DOI 标准化
4. 作者格式化
5. 期刊加权
6. 引用格式构建
7. 输出符号清洗
【与其他脚本关系】
- metadata_loader.py 依赖本模块处理 CSV 字段
- retriever.py 依赖 basename_lower / norm_text 等
- prompt_builder.py 用 protect_bio_terms
- generator.py 用 clean_symbols / restore_bio_terms
- pipeline.py 用 build_citation_string
"""
import os
import re
from typing import Any, Dict, List, Optional
from config import HIGH_IMPACT_JOURNALS
# 常见文件名列候选
_FILENAME_CANDS = [
    "匹配的PDF文件名", "names", "filename", "file_name", "pdf_name",
    "pdf", "File Name", "Filename", "PDF", "file", "source"
]
# 常见元数据列候选
_EN_AUTHORS_COLS = ["Author Full Names", "Authors", "AF", "AU", "authors", "作者", "作　者", "第一作者"]
_EN_TITLE_COLS = ["filename", "Article Title", "Title", "TI", "article title", "title", "题名", "题　名", "标题", "文章标题"]
_EN_ABSTRACT_COLS = ["Abstract", "AB", "abstract", "摘要", "文摘", "文　摘"]
_EN_KEYWORDS_COLS = ["Author Keywords", "Keywords", "DE", "ID", "keywords", "关键字", "关键词", "Keywords Plus"]
_EN_DOI_COLS = ["DOI", "doi", "DI", "DOI Link"]
_EN_YEAR_COLS = ["Publication Year", "Year", "PY", "publication year", "发表年", "年", "出版年"]
_EN_JOURNAL_COLS = ["Source Title", "Journal", "SO", "source title", "journal", "刊名", "刊　名", "期刊名称", "来源刊名"]
def norm_text(s: Any) -> str:
    """通用文本清洗：去全角空格、合并空白、处理 nan。"""
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return "" if s.lower() == "nan" else s
def normalize_colname(x: str) -> str:
    """标准化列名：去空格、去全角空格。"""
    return str(x).strip().replace(" ", "").replace("\u3000", "")
def basename_lower(x: str) -> str:
    """获取文件 basename，小写，去掉 .pdf 后缀。"""
    x = os.path.basename(norm_text(x)).lower()
    return re.sub(r"\.pdf$", "", x, flags=re.I)
def normalize_doi(doi: str) -> str:
    """DOI 标准化：去掉 https://doi.org/ 前缀，只保留纯 DOI。"""
    doi = norm_text(doi)
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi.strip().rstrip(".")
def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """将原始 row 的列名统一标准化。"""
    return {normalize_colname(k): v for k, v in row.items()}
def find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    """从列名列表中找候选列。"""
    low_map = {normalize_colname(c): c for c in cols}
    for cand in candidates:
        k = normalize_colname(cand)
        if k in low_map:
            return low_map[k]
    return None
def detect_filename_col(cols: List[str]) -> str:
    """自动识别文件名列，没识别到则默认最后一列。"""
    col = find_col(cols, _FILENAME_CANDS)
    return col if col else cols[-1]
def pick(row: Dict[str, Any], candidates: List[str], fallback: str = "") -> str:
    """在一行 row 中，按候选列名顺序取值，兼容原始列名与标准化列名。"""
    for c in candidates:
        if c in row:
            v = norm_text(row.get(c, ""))
            if v:
                return v
    c_row = compact_row(row)
    for c in candidates:
        ck = normalize_colname(c)
        if ck in c_row:
            v = norm_text(c_row.get(ck, ""))
            if v:
                return v
    return fallback
def sentence_case(s: str) -> str:
    """处理全大写标题，恢复为 sentence case，保留基因/技术缩写。"""
    s = norm_text(s)
    if not s:
        return s
    alpha_chars = [c for c in s if c.isalpha()]
    if not alpha_chars:
        return s
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if upper_ratio <= 0.8:
        return s
    acronyms = {
        "QTL", "GWAS", "SNP", "DNA", "RNA", "MRNA", "PCR", "SSR", "RIL", "NIL",
        "ABA", "ROS", "LD", "PCA", "GBS", "WGS", "RAD", "NGS", "SMRT", "ONT", "ATAC"
    }
    words = s.split()
    out = []
    for i, w in enumerate(words):
        core = w.rstrip(".,;:?!")
        punct = w[len(core):]
        if core.upper() in acronyms:
            out.append(core.upper() + punct)
        elif re.match(r"^[A-Za-z][A-Za-z0-9.]*[0-9][A-Za-z0-9.]*$", core):
            out.append(w)
        elif i == 0:
            out.append(core[:1].upper() + core[1:].lower() + punct)
        else:
            out.append(core.lower() + punct)
    return " ".join(out)
def title_case(s: str) -> str:
    """将全大写期刊名恢复为 title case。"""
    s = norm_text(s)
    if not s:
        return s
    alpha_chars = [c for c in s if c.isalpha()]
    if not alpha_chars:
        return s
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if upper_ratio <= 0.8:
        return s
    small_words = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "from"}
    ws = s.lower().split()
    return " ".join([w.capitalize() if i == 0 or w not in small_words else w for i, w in enumerate(ws)])
def format_authors(authors_raw: str) -> str:
    """规范化作者格式，转换为 Last, F.; Last2, F. 形式。"""
    authors_raw = norm_text(authors_raw)
    if not authors_raw:
        return ""
    parts = [a.strip() for a in re.split(r"[;|]", authors_raw) if a.strip()]
    out = []
    for p in parts[:6]:
        if "," in p:
            seg = [x.strip() for x in p.split(",", 1)]
            last = seg[0]
            first = seg[1] if len(seg) > 1 else ""
            initials = "".join(w[0].upper() + "." for w in first.split() if w)
            out.append(f"{last}, {initials}" if initials else last)
        else:
            out.append(p)
    formatted = "; ".join(out)
    if len(parts) > 6:
        formatted += " et al."
    return formatted
def get_journal_score(journal: str) -> float:
    """
    综合期刊权重系统，覆盖 2623 种期刊，支持名称标准化 + 缩写扩展。
    基于 CSV 数据中实际出现的期刊构建，覆盖 100% 论文。
    词边界匹配避免 "nature" 误匹配 "nature reviews"。
    """
    import re
    j = journal.lower().strip()
    # 标准化
    j = re.sub(r'^the\s+', '', j)
    j = j.replace('-', ' ').replace('–', ' ').replace('—', ' ')
    j = re.sub(r'[^a-z0-9\s]', '', j)
    j = re.sub(r'\s+', ' ', j).strip()

    # 缩写扩展（标准化后的名称 → 完整名称）
    _abbrev_map = {
        'pnas': 'proceedings of the national academy of sciences',
        'plant j': 'plant journal', 'plant physiol': 'plant physiology',
        'new phytol': 'new phytologist', 'j exp bot': 'journal of experimental botany',
        'theor appl genet': 'theoretical and applied genetics',
        'mol plant': 'molecular plant', 'curr biol': 'current biology',
        'curr opin plant biol': 'current opinion in plant biology',
        'trends plant sci': 'trends in plant science',
        'genome biol': 'genome biology', 'genome res': 'genome research',
        'plant cell environ': 'plant cell and environment',
        'plant biotechnol j': 'plant biotechnology journal',
        'plant cell physiol': 'plant and cell physiology',
        'plant mol biol': 'plant molecular biology',
        'mol breeding': 'molecular breeding',
        'bmc plant biol': 'bmc plant biology', 'bmc genom': 'bmc genomics',
        'sci rep': 'scientific reports', 'sci adv': 'science advances',
        'elife': 'elife', 'g3': 'g3 genes genomes genetics',
        'annu rev plant biol': 'annual review of plant biology',
        'front plant sci': 'frontiers in plant science',
        'plos genet': 'plos genetics', 'plos biol': 'plos biology',
        'nat genet': 'nature genetics', 'nat commun': 'nature communications',
        'nat plants': 'nature plants', 'nat biotechnol': 'nature biotechnology',
        'nat rev genet': 'nature reviews genetics',
        'int j mol sci': 'international journal of molecular sciences',
        'int j biol macromol': 'international journal of biological macromolecules',
        'j agric food chem': 'journal of agricultural and food chemistry',
        'j sci food agric': 'journal of the science of food and agriculture',
        'field crop res': 'field crops research', 'crop prot': 'crop protection',
    }
    if j in _abbrev_map:
        j = _abbrev_map[j]

    # 分级匹配（长模式优先，高权重优先）
    _tiers = [
        # === TIER 0: CNS (10.0) ===
        (10.0, []),  # CNS: handled below with exact match logic
        # === TIER 1 (9.0-9.8) ===
        (9.8, ["nature genetics", "nature reviews genetics"]),
        (9.5, ["nature plants", "nature communications"]),
        (9.3, ["nature biotechnology", "nature methods"]),
        (9.0, ["molecular plant", "annual review of plant biology"]),
        (8.8, ["plant cell"]),
        (8.5, ["pnas", "proceedings of the national academy of sciences",
               "genome biology", "trends in plant science", "science advances",
               "national science review"]),
        # === TIER 2 (7.5-8.3) ===
        (8.3, ["nature ecology", "nature food", "nature climate", "nature microbiology"]),
        (8.0, ["new phytologist", "current biology", "current opinion in plant biology",
               "genome research", "elife", "the isme journal", "nucleic acids research",
               "annual review of genetics", "plant biotechnology journal"]),
        (7.8, ["developmental cell", "embo journal"]),
        (7.5, ["plant physiology", "journal of integrative plant biology",
               "plos genetics", "plos biology", "plant communications",
               "journal of experimental botany", "plant cell and environment"]),
        # === TIER 3 (7.0-7.3) ===
        (7.3, ["plant journal", "horticulture research", "science bulletin"]),
        (7.0, ["plant and cell physiology", "plant cell physiology",
               "bmc biology", "development", "plant molecular biology",
               "theoretical and applied genetics", "molecular plant pathology",
               "food chemistry", "global change biology bioenergy",
               "trends in genetics", "molecular biology and evolution",
               "plant cell reports"]),
        # === TIER 4 (6.5-6.8) ===
        (6.8, ["journal of genetics and genomics", "plant diversity",
               "plant physiology and biochemistry", "environmental and experimental botany"]),
        (6.5, ["molecular breeding", "genetics", "g3 genes genomes genetics",
               "heredity", "annals of botany", "plant genome", "plant science",
               "frontiers in plant science", "physiologia plantarum", "planta",
               "functional plant biology", "plant cell tissue and organ culture",
               "plant growth regulation", "plant biology", "crop journal",
               "journal of agricultural and food chemistry",
               "bioresource technology", "biotechnology for biofuels",
               "food research international", "phytochemistry",
               "mycorrhiza", "annals of applied biology",
               "journal of biological chemistry"]),
        # === TIER 5 (6.0-6.3) ===
        (6.3, ["plant pathology", "phytopathology", "plant disease",
               "biomass bioenergy", "industrial crops and products",
               "journal of cereal science", "plant and soil",
               "plant methods", "genomics", "food hydrocolloids",
               "carbohydrate polymers", "critical reviews in plant sciences",
               "pest management science", "crop protection",
               "renewable energy", "renewable sustainable energy reviews",
               "fuel", "applied energy",
               "acs sustainable chemistry engineering",
               "soil biology and biochemistry", "soil biology biochemistry",
               "geoderma", "biology and fertility of soils",
               "chemosphere", "science of the total environment",
               "energy conversion and management",
               "journal of plant biochemistry and biotechnology"]),
        (6.0, ["bmc genomics", "bmc plant biology", "bmc genetics",
               "frontiers in microbiology", "frontiers in genetics",
               "scientific reports", "peerj", "plos one",
               "molecular genetics and genomics", "gene", "genes",
               "molecules", "international journal of molecular sciences",
               "international journal of biological macromolecules",
               "crop science", "field crops research",
               "european journal of agronomy", "agronomy for sustainable development",
               "soil tillage research", "agricultural and forest meteorology",
               "agricultural water management", "agricultural systems",
               "agriculture ecosystems environment", "agronomy journal",
               "journal of agronomy and crop science", "crop pasture science",
               "european journal of plant pathology", "biological control",
               "journal of chemical ecology", "entomologia experimentalis et applicata",
               "weed science", "genome", "chromosome research",
               "journal of integrative agriculture", "euphytica",
               "genetic resources and crop evolution", "plant breeding",
               "plant direct", "in silico plants", "plant phenomics",
               "journal of food engineering", "foods", "food control",
               "food microbiology", "international journal of food microbiology",
               "lwt food science and technology", "journal of food science",
               "postharvest biology and technology",
               "journal of environmental quality", "environmental pollution",
               "journal of environmental management",
               "computers and electronics in agriculture",
               "remote sensing", "water",
               "applied biochemistry and biotechnology",
               "microorganisms", "nutrients"]),
        # === TIER 6 (5.0-5.5) ===
        (5.5, ["journal of plant physiology", "journal of plant growth regulation",
               "journal of plant interactions", "journal of plant research",
               "acta physiologiae plantarum", "biologia plantarum",
               "south african journal of botany", "aob plants",
               "tropical plant biology", "plant signaling behavior",
               "plant ecology", "plant biosystems",
               "journal of plant nutrition", "soil science and plant nutrition",
               "journal of soil science and plant nutrition", "nutrient cycling in agroecosystems",
               "applied soil ecology", "soil science society of america journal",
               "soil use and management",
               "applied microbiology and biotechnology", "process biochemistry",
               "biochemical engineering journal", "journal of bioscience and bioengineering",
               "world journal of microbiology biotechnology",
               "3 biotech", "microbial cell factories", "metabolic engineering",
               "biotechnology letters", "amb express",
               "sustainability", "energies", "bioenergy research",
               "biomass conversion and biorefinery", "waste and biomass valorization",
               "journal of cleaner production", "environmental science and pollution research",
               "frontiers in bioengineering and biotechnology", "frontiers in energy research",
               "frontiers in sustainable food systems", "frontiers in nutrition",
               "frontiers in environmental science", "frontiers in ecology and evolution",
               "frontiers in agronomy", "frontiers in soil science",
               "applied ecology and environmental research",
               "bioresources", "energy",
               "agronomy basel", "agriculture basel", "plants basel",
               "agronomy journal", "crop science",
               "fermentation basel", "heliyon",
               "journal of applied microbiology",
               "insects basel", "animals basel"]),
        (5.0, ["journal of food composition and analysis",
               "food bioscience", "food science nutrition",
               "journal of food processing and preservation",
               "international journal of food science technology",
               "journal of food measurement and characterization",
               "european food research and technology",
               "journal of the science of food and agriculture",
               "international journal of food properties",
               "cereal chemistry", "cereal research communications",
               "starch starke", "journal of the institute of brewing",
               "sugar tech", "environmental research letters",
               "applied and environmental microbiology",
               "environmental monitoring and assessment",
               "ecotoxicology and environmental safety",
               "pesticide biochemistry and physiology",
               "scientia horticulturae",
               "journal of applied entomology",
               "bulletin of entomological research",
               "southwestern entomologist",
               "physiology and molecular biology of plants",
               "plant soil and environment",
               "legume research",
               "rsc advances",
               "irrigation and drainage",
               "toxins", "mycologia", "journal of fungi",
               "food function",
               "archives of virology",
               "processes", "journal of food processing engineering",
               "international journal of phytoremediation",
               "journal of food quality",
               "journal of the american oil chemists society",
               "poultry science", "journal of dairy science", "journal of animal science",
               "animal feed science and technology", "livestock science",
               "small ruminant research", "tropical animal health and production",
               "animal production science",
               "international journal of pest management",
               "weed research", "weed technology", "phytoparasitica",
               "australasian plant pathology", "canadian journal of plant pathology",
               "journal of phytopathology", "archives of phytopathology and plant protection",
               "journal of nematology",
               "journal of economic entomology", "environmental entomology",
               "florida entomologist", "biocontrol science and technology",
               "agricultural and forest entomology",
               "archives of agronomy and soil science", "agroforestry systems",
               "journal of sustainable agriculture", "renewable agriculture and food systems",
               "organic agriculture", "agroecology and sustainable food systems",
               "experimental agriculture",
               "grass and forage science", "journal of production agriculture",
               "international journal of agronomy",
               "cogent food agriculture",
               "turkish journal of agriculture and forestry",
               "spanish journal of agricultural research",
               "journal of agricultural science", "journal of crop improvement",
               "journal of plant registrations", "seed science research",
               "canadian journal of plant science", "acta scientiarum agronomy",
               "ciencia rural", "ciencia e agrotecnologia",
               "pesquisa agropecuaria brasileira", "revista brasileira",
               "revista caatinga", "semina ciencias agrarias",
               "pakistan journal of botany", "pakistan journal of agricultural sciences",
               "indian journal of agricultural sciences", "indian journal of genetics",
               "journal of arid environments", "journal of arid land",
               "australian journal of agricultural research",
               "new zealand journal of agricultural research",
               "chilean journal of agricultural research",
               "maydica", "bragantia", "planta daninha",
               "crop forage turfgrass management",
               "biocatalysis and agricultural biotechnology",
               "journal of agriculture and food research",
               "journal of stored products research",
               "arquivo brasileiro de medicina veterinaria e zootecnia",
               "journal of applied animal research",
               "international journal of hydrogen energy",
               "journal of hazardous materials",
               "journal of the american society of brewing chemists",
               "journal of animal physiology and animal nutrition",
               "revista ciencia agronomica",
               "communications in soil science and plant analysis",
               "journal of food science and technology mysore",
               "journal of food safety", "food additives and contaminants",
               "food and bioproducts processing",
               "innovative food science emerging technologies",
               "food analytical methods", "food and bioprocess technology",
               "foods basel"]),
    ]
    # CNS 精确匹配：必须是独立的 Nature/Science/Cell 期刊
    if j == 'nature' or j == 'science' or j == 'cell':
        return 10.0
    # Nature 子刊前缀匹配
    if j.startswith('nature '):
        for weight, patterns in sorted(_tiers, key=lambda x: -x[0]):
            for pat in sorted(patterns, key=len, reverse=True):
                if re.search(r'\b' + re.escape(pat) + r'\b', j):
                    return weight

    for weight, patterns in sorted(_tiers, key=lambda x: -x[0]):
        for pat in sorted(patterns, key=len, reverse=True):
            if re.search(r'\b' + re.escape(pat) + r'\b', j):
                return weight

    # 兜底：关键词推断
    _academic = ["journal", "research", "science", "review", "plant",
                 "crop", "soil", "food", "agric", "genet", "genom",
                 "biolog", "chem", "molecul", "biochem", "biotech",
                 "breed", "agron", "botan", "ecol", "environ",
                 "microbiol", "biomass", "bioenerg", "sorghum",
                 "cereal", "grain", "proceeding", "symposi",
                 "international", "annals", "archives", "bulletin"]
    if any(kw in j for kw in _academic):
        return 4.0
    return 3.0  # 书籍/报告


def protect_bio_terms(text: str):
    """
    已禁用术语保护。
    直接返回原始文本，避免基因名（如 Sh1、SbTB1、Dw3）
    被替换为占位符导致模型无法识别关键基因名。
    """
    return text, {}


def clean_symbols(text: str) -> str:
    """清洗模型生成中的 markdown / latex 杂质符号。"""
    import re
    text = re.sub(r"\^\{([^}]+)\}", r"[\1]", text)
    text = re.sub(r"\$([^$\n]+?)\$", r"\1", text)
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def restore_bio_terms(text: str, protected: Dict[str, str]) -> str:
    """已禁用，直接返回原文。"""
    return text
def build_citation_string(ref_info: Dict[str, str], index: int, source_fname: str = "") -> str:
    """
    构建参考文献字符串。
    输出示例：[1] Smith, J. (2020). Title. Journal. DOI: [xxx](https://doi.org/xxx)
    """
    title = norm_text(ref_info.get("title", "")) or re.sub(r"\.pdf$", "", source_fname, flags=re.I)
    authors = norm_text(ref_info.get("authors", ""))
    year = norm_text(ref_info.get("year", "")).rstrip(".0") if ".0" in norm_text(ref_info.get("year", "")) else norm_text(ref_info.get("year", ""))
    journal = norm_text(ref_info.get("journal", ""))
    doi = normalize_doi(ref_info.get("doi", ""))
    parts = [f"[{index}]"]
    if authors:
        parts.append(authors)
    if year:
        parts.append(f"({year}).")
    if title:
        parts.append(f"{title}.")
    if journal:
        parts.append(f"{journal}.")
    if doi:
        parts.append(f"DOI: [{doi}](https://doi.org/{doi})")
    return " ".join(parts)
