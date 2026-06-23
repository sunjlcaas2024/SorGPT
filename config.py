# -*- coding: utf-8 -*-
"""
config.py  ── 全局配置（新增 SQLite 数据库路径）
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["MKL_THREADING_LAYER"] = "GNU"
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
# DeepSeek V4 Pro API
# -----------------------------
BASE_URL        = "https://api.deepseek.com/v1"
API_KEY         = os.environ.get("DEEPSEEK_API_KEY", "")
LOCAL_MODEL_NAME = "deepseek-reasoner"

# -----------------------------
# 元数据索引路径
# -----------------------------
META_INDEX_PATHS = {
    "english": "/vol/sunjilin/website/data/agent/faiss_v3_meta_english",
}

# -----------------------------
# 全文索引路径（英文四库）
# -----------------------------
FULLTEXT_INDEX_PATHS = {
    "en_fine":  "/vol/sunjilin/website/data/agent/faiss_v3_english_fine",
    "en_std":   "/vol/sunjilin/website/data/agent/faiss_v3_english_std",
    "en_large": "/vol/sunjilin/website/data/agent/faiss_v3_english_large",
    "en_para":  "/vol/sunjilin/website/data/agent/faiss_v3_english_para",
}

# -----------------------------
# 元数据 CSV 路径
# -----------------------------
CSV_PATHS = [
    "/vol/sunjilin/website/data/publication/english_content_merged.csv",
]

# -----------------------------
# SQLite 基因注释数据库
# -----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GENE_DB_PATH = os.path.join(_THIS_DIR, "db", "sorghum_genes.db")

# -----------------------------
# 检索参数
# -----------------------------
TOP_META_K           = 120
TOP_CHUNK_K          = 30
FINAL_CONTEXT_K      = 12
COUNT_QUERY_FETCH_K  = 300
COUNT_QUERY_MAX_SHOW = 200

# -----------------------------
# FAISS 运行参数
# -----------------------------
DEFAULT_NPROBE = 64
USE_FAISS_GPU  = False
GPU_DEVICE     = 0

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
}

# -----------------------------
# 各问题类型对应的全文库选择策略
# -----------------------------
QUERY_TYPE_TO_INDEXES = {
    "factoid":      ["en_fine", "en_std"],
    "gene_function":["en_fine", "en_std", "en_large"],
    "mechanism":    ["en_std", "en_large", "en_fine", "en_para"],
    "qtl_gwas":     ["en_std", "en_fine", "en_large"],
    "review":       ["en_para", "en_large", "en_std", "en_fine"],
    "gene_list":    ["en_fine", "en_std", "en_large", "en_para"],
    "locate":       [],
    "count":        [],
    "boundary":     [],
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
}
