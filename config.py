# -*- coding: utf-8 -*-
"""
config.py  ── 全局配置（新增 SQLite 数据库路径）
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["MKL_THREADING_LAYER"] = "sequential"  # was GNU, changed to sequential to fix MKL 2025 deadlock
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# -----------------------------
# Embedding 模型
# -----------------------------
MODEL_PATH = "/vol/sunjilin/website/data/agent/models/bge-m3/"

# -----------------------------
# 小模型（关键词扩展）
# -----------------------------
SMALL_MODEL_PATH = "/vol/sunjilin/website/data/agent/models/Qwen/Qwen2.5-7B-Instruct"

# -----------------------------
# 本地大模型 API（已注释，切换到 DeepSeek）
# -----------------------------
# BASE_URL        = "http://10.122.14.169:30000/v1"
# API_KEY         = "EMPTY"
# LOCAL_MODEL_NAME = "/data/models/Qwen/Qwen3.5-397B-A17B"

# -----------------------------
# Dashscope Qwen API (local cost-saving)
# -----------------------------
BASE_URL        = "https://api.deepseek.com/v1"
# 密钥加固(2026-08-30): DeepSeek key 仅从环境变量 DEEPSEEK_API_KEY 注入(start.sh source .env)，
# 源码不保留任何 key 字面量。未设置时启动即报错，防止静默使用过期 key。
API_KEY         = os.environ.get("DEEPSEEK_API_KEY") or ""
if not API_KEY:
    raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY，请在 .env 配置后通过 start.sh 启动")
LOCAL_MODEL_NAME = "deepseek-v4-flash"

# -----------------------------
# 元数据索引路径
# -----------------------------
META_INDEX_PATHS = {
    "english": "/vol/sunjilin/website/data/agent/faiss_v3_meta_english",
    "chinese": "/vol/sunjilin/website/data/agent/faiss_index_meta_chinese",
}

# -----------------------------
# 全文索引路径（英文四库）
# -----------------------------
FULLTEXT_INDEX_PATHS = {
    "en_fine":  "/vol/sunjilin/website/data/agent/faiss_v3_english_fine",
    "en_std":   "/vol/sunjilin/website/data/agent/faiss_v3_english_std",
    "en_large": "/vol/sunjilin/website/data/agent/faiss_v3_english_large",
    "en_para":  "/vol/sunjilin/website/data/agent/faiss_v3_english_para",
    "zh_fine":  "/vol/sunjilin/website/data/agent/faiss_index_chinese_fine",
    "zh_std":   "/vol/sunjilin/website/data/agent/faiss_index_chinese_std",
    "zh_large": "/vol/sunjilin/website/data/agent/faiss_index_chinese_large",
}

# -----------------------------
# 元数据 CSV 路径
# -----------------------------
CSV_PATHS = [
    "/vol/sunjilin/website/data/publication/english_content_merged.csv",
    "/vol/sunjilin/website/data/publication/chinese_content.csv",
]

# -----------------------------
# SQLite 基因注释数据库
# -----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GENE_DB_PATH = os.path.join(_THIS_DIR, "db", "sorghum_genes.db")

# -----------------------------
# 检索参数
# -----------------------------
TOP_META_K           = 200
TOP_CHUNK_K          = 30
FINAL_CONTEXT_K      = 12
COUNT_QUERY_FETCH_K  = 300
COUNT_QUERY_MAX_SHOW = 200

# -----------------------------
# 基因序列展示 (sequence 类型)
# -----------------------------
SEQ_PREVIEW_BP      = 60      # 核苷酸序列预览长度(bp)
SEQ_PREVIEW_AA      = 30      # 蛋白序列预览长度(aa)
GET_SEQ_URL         = "http://127.0.0.1:8001/glapi/mongo/get-seq/"
SEQ_SYMBOL_MAP_PATH = os.path.join(_THIS_DIR, "gene_symbol_map.json")

# -----------------------------
# FAISS 运行参数
# -----------------------------
DEFAULT_NPROBE = 256
USE_FAISS_GPU  = False
GPU_DEVICE     = 0

# -----------------------------
# zh-v8: 中文问题翻译成英文进英文检索（跨语言证据兼容）
# True = 中文题把英文翻译同时用于英文索引向量检索 + 英文证据 BM25 词法打分；
# False = 回退到 zh-v7 行为（英文索引用混合向量、英文证据纯向量）。
# -----------------------------
TRANSLATE_ZH_TO_EN = True

# -----------------------------
# 各问题类型的参考文献上限
# -----------------------------
REFERENCE_LIMITS = {
    "factoid":      6,
    "gene_function":10,
    "mechanism":    12,
    "qtl_gwas":     12,
    "locate":       3,
    "review":       15,
    "gene_list":    15,
    "count":        20,
    "boundary":     0,
    "sequence":     6,
}

# -----------------------------
# 各问题类型的最终证据数量上限
# -----------------------------
EVIDENCE_LIMITS = {
    "factoid":      12,
    "gene_function":20,
    "mechanism":    24,
    "qtl_gwas":     16,
    "locate":       0,
    "review":       24,
    "gene_list":    24,
    "count":        0,
    "boundary":     0,
    "sequence":     12,
}

# -----------------------------
# 各问题类型对应的全文库选择策略
# -----------------------------
QUERY_TYPE_TO_INDEXES = {
    "factoid":      ["en_fine", "en_std", "zh_fine", "zh_std"],
    "gene_function":["en_fine", "en_std", "en_large", "zh_fine", "zh_std", "zh_large"],
    "mechanism":    ["en_std", "en_large", "en_fine", "en_para", "zh_std", "zh_large", "zh_fine"],
    "qtl_gwas":     ["en_std", "en_fine", "en_large", "zh_std", "zh_fine", "zh_large"],
    "review":       ["en_para", "en_large", "en_std", "en_fine", "zh_large", "zh_std", "zh_fine"],
    "gene_list":    ["en_fine", "en_std", "en_large", "en_para", "zh_fine", "zh_std", "zh_large"],
    "locate":       [],
    "count":        [],
    "boundary":     [],
    "sequence":     ["en_fine", "en_std", "zh_fine", "zh_std"],
}

# -----------------------------
# section 类型加权
# -----------------------------
SECTION_BONUS = {
    "abstract":     0.10,
    "results":      0.08,
    "discussion":   0.08,
    "introduction": 0.03,
    "methods":     -0.03,
    "references":  -0.50,
}

# -----------------------------
# 高水平期刊加权
# -----------------------------
HIGH_IMPACT_JOURNALS = {
    # Tier 0: CNS 顶刊
    "nature":                       10.0,
    "science":                      10.0,
    "cell":                         10.0,
    # Tier 1: 大子刊 / 顶级植物
    "nature genetics":               9.8,
    "nature plants":                 9.5,
    "nature communications":         9.3,
    "nature biotechnology":          9.0,
    "molecular plant":               9.0,
    "plant cell":                    8.8,
    "pnas":                          8.5,
    "genome biology":                8.5,
    # Tier 2: 高水平植物/农业
    "new phytologist":               8.0,
    "plant biotechnology journal":   7.8,
    "plant physiology":              7.5,
    "journal of experimental botany":7.3,
    "the plant journal":             7.0,
    "trends in plant science":       8.5,
    "current opinion in plant biol": 8.0,
    "current biology":               7.5,
    "plos genetics":                 7.5,
    "genome research":               8.0,
    "nucleic acids research":        7.5,
    "the isme journal":              8.0,
    "elife":                         7.5,
    # Tier 3: 优秀期刊
    "plant cell and environment":    7.0,
    "journal of integrative plant":  7.0,
    "plant communic":                7.5,
    "horticulture research":         7.0,
    "science advances":              8.5,
    "science bulletin":              7.0,
    "bmc biology":                   6.8,
    "development":                   6.5,
    "plant journal for cell":        6.8,
    "theoretical and applied genet": 6.5,
    "plant and cell physiology":     6.8,
    "frontiers in plant science":    5.0,
    "bmc genomics":                  5.5,
    "bmc plant biology":             5.5,
    "scientific reports":            5.0,
    "plant molecular biology":       6.0,
    "journal of cereal science":     4.0,
    "field crops research":          4.5,
    # Chinese journals (CNKI composite IF mapped to JCR scale)
    "作物学报":               5.5,
    "中国农业科学":           6.0,
    "核农学报":               4.0,
    "植物遗传资源学报":       4.0,
    "麦类作物学报":           3.5,
    "中国油料作物学报":       3.0,
    "玉米科学":               2.5,
    "大豆科学":               2.0,
    "山东农业科学":           2.0,
    "植物生理学报":           2.5,
    "江苏农业学报":           2.0,
    "华北农学报":             1.5,
    "河南农业科学":           1.5,
    "江西农业学报":           1.5,
    "湖北农业科学":           1.0,
    "西北农业学报":           1.0,
    "安徽农业科学":           0.5,
}
