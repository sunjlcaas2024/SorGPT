# -*- coding: utf-8 -*-
"""
metadata_loader.py  ── 修复版
========================
修复说明：
1. read_csv_robust 增加 latin-1 兜底编码，解决中间行 UTF-8 解码崩溃问题
2. load_citation_map 优先使用 names 列（带.pdf，与 FAISS source 一致）作为主 key
3. _register_all_keys 注册带/不带 .pdf、basename、小写等所有变体
4. title 字段从 Source Title / Article Title 取，不再误用 filename 列（无扩展名标题）
"""
from typing import Dict, List, Optional
import os
import re
import pandas as pd
from utils import (
    norm_text, detect_filename_col, pick, sentence_case, title_case,
    format_authors, normalize_doi,
)

_EN_AUTHORS_COLS  = ["Author Full Names", "Authors", "AF", "AU", "authors", "作者", "作　者", "第一作者"]
# 注意：filename 列内容是无扩展名标题，不再作为 title 的候选来源
_EN_TITLE_COLS    = ["Article Title", "Title", "TI", "article title", "title",
                     "题名", "题　名", "标题", "文章标题"]
_EN_ABSTRACT_COLS = ["Abstract", "AB", "abstract", "摘要", "文摘", "文　摘"]
_EN_KEYWORDS_COLS = ["Author Keywords", "Keywords", "DE", "ID", "keywords",
                     "关键字", "关键词", "Keywords Plus"]
_EN_DOI_COLS      = ["DOI", "doi", "DI", "DOI Link"]
_EN_YEAR_COLS     = ["Publication Year", "Year", "PY", "publication year", "发表年", "年", "出版年"]
_EN_JOURNAL_COLS  = ["Source Title", "Journal", "SO", "source title", "journal",
                     "刊名", "刊　名", "期刊名称", "来源刊名"]


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    """
    鲁棒读取 CSV。
    按优先级尝试多种编码，latin-1 兜底（不会有解码错误）。
    on_bad_lines='skip' 跳过格式异常行，不崩溃。
    """
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"]:
        try:
            df = pd.read_csv(
                csv_path,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
            )
            if len(df) > 0:
                return df
        except (UnicodeDecodeError, Exception):
            continue
    raise RuntimeError(f"无法读取 CSV: {csv_path}")


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    """按优先级查找列名，支持大小写不敏感匹配。"""
    # 完全匹配优先
    for c in candidates:
        if c in cols:
            return c
    # 忽略大小写
    cols_lower = {col.lower(): col for col in cols}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _register_all_keys(
    citation_map: Dict[str, Dict],
    info: Dict[str, str],
    fname: str,
) -> None:
    """
    把一条文献的所有可能 key 格式注册到 citation_map。
    覆盖：原始值、basename、去.pdf、小写变体。
    """
    if not fname:
        return
    fname = norm_text(fname)
    base  = os.path.basename(fname)
    stem  = re.sub(r"\.pdf$", "", base, flags=re.I)

    for key in {fname, base, stem,
                fname.lower(), base.lower(), stem.lower(),
                fname.strip(), base.strip(), stem.strip()}:
        if key:
            citation_map[key] = info


def load_citation_map(csv_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """
    构建 citation_map。

    CSV 结构（english_content.csv）：
      names    列（最后列）：带 .pdf 的完整文件名，与 FAISS source 完全一致 → 主 key
      filename 列（第4列）：不带 .pdf 的标题字符串 → 备用 key
      Source Title 列：期刊/会议名称 → title 字段来源之一

    所有 key 变体全部注册，确保 safe_get_ref_info 一定能命中。
    """
    citation_map: Dict[str, Dict[str, str]] = {}

    # v3: filter irrelevant Chinese journals
    _zh_skip = {'电影评介','电影文学','大众电影','当代电影','农村百事通','农村新技术','农村科学实验','农村科技','现代农业科技','现代农村科技','新农业','农技服务','农业机械'}
    _skipped = 0

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue

        df = read_csv_robust(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)

        # names 列：带 .pdf，与 FAISS source 一致（优先）
        pdf_col  = _find_col(cols, ["names", "Names", "NAMES"])
        # filename 列：不带 .pdf 的标题（备用 key，同时可作 title 兜底）
        meta_col = _find_col(cols, ["filename", "Filename", "FILENAME"])

        for row in df.to_dict("records"):
            pdf_fname  = norm_text(row.get(pdf_col,  "")) if pdf_col  else ""
            meta_fname = norm_text(row.get(meta_col, "")) if meta_col else ""

            # 至少有一个文件名才处理
            primary = pdf_fname or meta_fname
            if not primary:
                continue

            authors_raw = pick(row, _EN_AUTHORS_COLS)
            # title 优先从 Article Title / Source Title 取，
            # 兜底才用 filename 列（无扩展名标题）
            title = sentence_case(pick(row, _EN_TITLE_COLS))
            if not title and meta_fname:
                title = sentence_case(meta_fname)

            abstract = pick(row, _EN_ABSTRACT_COLS)
            keywords = pick(row, _EN_KEYWORDS_COLS)
            doi      = normalize_doi(pick(row, _EN_DOI_COLS))
            year     = pick(row, _EN_YEAR_COLS)
            journal  = title_case(pick(row, _EN_JOURNAL_COLS))

            info = {
                "authors":     format_authors(authors_raw),
                "authors_raw": authors_raw,
                "title":       title,
                "abstract":    abstract,
                "keywords":    keywords,
                "doi":         doi,
                "year":        year,
                "journal":     journal,
            }

            # 两列的所有变体全部注册
            _register_all_keys(citation_map, info, pdf_fname)
            _register_all_keys(citation_map, info, meta_fname)

    return citation_map


def safe_get_ref_info(fname: str, citation_map: Dict[str, Dict]) -> Dict[str, str]:
    """
    安全读取文献信息，自动尝试多种 key 格式。
    """
    fname = norm_text(fname)
    base  = os.path.basename(fname)
    stem  = re.sub(r"\.pdf$", "", base, flags=re.I)

    for key in [fname, base, stem,
                fname.lower(), base.lower(), stem.lower()]:
        if key and key in citation_map:
            return citation_map[key]

    return {
        "authors": "", "authors_raw": "", "title": "",
        "abstract": "", "keywords": "", "doi": "", "year": "", "journal": "",
    }
