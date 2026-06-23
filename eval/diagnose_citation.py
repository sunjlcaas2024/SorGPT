# -*- coding: utf-8 -*-
"""
fix_patch.py
============
本文件包含两处修复的完整替换代码，直接复制到对应文件即可。

修复1：build_sorghum_index_filter.py — 加强参考文献过滤
修复2：pipeline.py                  — 诊断 citation_map 匹配问题并修复
"""

# ══════════════════════════════════════════════════════════════════
# 修复1：build_sorghum_index_filter.py
# 替换原来的 _TRUNCATE_SECTION_PATTERNS / is_noise_chunk / clean_pdf_text
# ══════════════════════════════════════════════════════════════════

"""
【替换 build_sorghum_index_filter.py 顶部的正则定义区】

问题：
  - _TRUNCATE_SECTION_PATTERNS 要求 references 独占一整行（^\s*...\s*$）
    但 PDF 提取后 References 标题可能紧跟页码、空格或下一行内容，导致匹配失败
  - is_noise_chunk 对 "36.  Lin, Z." 这类编号参考文献行的检测阈值过宽

修复：
  1. 新增宽松版截断正则，允许 references 前有数字/空格/tab
  2. is_noise_chunk 额外检测"作者年份期刊"模式的密集行
  3. clean_pdf_text 在截断后再做一次尾部清洗
"""

FIXED_FILTER_CODE = r'''
import re

# ── 截断正则（宽松版，兼容"2  References"、"References\n"等格式）──
_TRUNCATE_SECTION_PATTERNS = re.compile(
    r"(?:^|\n)[ \t\d]*"          # 允许行首有数字/空格（如页码）
    r"(references|bibliography|参考文献|文献|works cited|literature cited"
    r"|supplementary|supplemental materials?|supporting information"
    r"|appendix|附录)"
    r"[ \t]*(?:\n|$)",            # 结尾允许有空格
    re.IGNORECASE,
)

# ── 跳过章节（致谢等，不影响截断逻辑）────────────────────────────
_SKIP_SECTION_PATTERNS = re.compile(
    r"^\s*(acknowledgements?|acknowledgments?|funding|conflict of interest"
    r"|author contributions?|data availability|ethics statement"
    r"|abbreviations|致谢|资金|作者贡献|数据可用性)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ── 参考文献行模式（加强版）─────────────────────────────────────
_REF_LINE_PATTERN = re.compile(
    r"^\s*(?:"
    r"\[\d+\]"                   # [1] 格式
    r"|\d{1,3}\."                # 1. / 36. 格式
    r"|\d{1,3}\s"                # 36 Lin 格式（数字+空格）
    r")\s*[A-Z\u4e00-\u9fff]"    # 后跟大写字母或中文
)

# ── 作者-年份-期刊行模式（用于检测参考文献密集段）────────────────
_AUTHOR_YEAR_PATTERN = re.compile(
    r"[A-Z][a-z]+,?\s+[A-Z]\..*(?:19|20)\d{2}"  # "Lin, Z. et al. 2012"
    r"|(?:19|20)\d{2}[;,\.]"                      # 或年份结尾
    r"|Nat\.?\s+Genet|Plant\s+Cell|Science|Nature|PNAS"  # 期刊缩写
    r"|doi:\s*10\.\d{4}"
)

# ── 页眉页脚模式（不变）──────────────────────────────────────────
_HEADER_FOOTER_PATTERN = re.compile(
    r"(^\s*\d+\s*$"
    r"|©\s*\d{4}"
    r"|all rights reserved"
    r"|www\.\S+\.\S+"
    r"|doi:\s*10\.\d{4}"
    r"|received:\s*\d"
    r"|accepted:\s*\d"
    r"|published:\s*\d"
    r"|volume\s+\d+.*issue\s+\d+"
    r"|^\s*page\s+\d+\s+of\s+\d+\s*$"
    r")",
    re.IGNORECASE,
)


def clean_pdf_text(text: str) -> str:
    """
    清洗 PDF 提取文本。
    修复：截断时取更早的匹配位置，防止 references 标题行残留。
    """
    match = _TRUNCATE_SECTION_PATTERNS.search(text)
    if match:
        # 取匹配起始位置截断，确保 references 标题本身也被去掉
        text = text[:match.start()]

    # 跳过致谢等章节
    sections_to_remove = []
    for m in _SKIP_SECTION_PATTERNS.finditer(text):
        start = m.start()
        rest = text[m.end():]
        next_section = re.search(r"\n\s*\n\s*[A-Z\u4e00-\u9fff][^\n]{0,60}\n", rest)
        end = m.end() + (next_section.start() if next_section else len(rest))
        sections_to_remove.append((start, end))
    for start, end in reversed(sections_to_remove):
        text = text[:start] + text[end:]

    # 逐行清洗
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _HEADER_FOOTER_PATTERN.search(stripped):
            continue
        if re.match(r"^\d{1,4}$", stripped):
            continue
        if len(stripped) < 15 and not re.search(r"[.!?。]$", stripped):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"-\n(\s*)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── 修复新增：尾部再扫一遍，去掉末尾残留的参考文献行簇 ──
    tail_lines = text.splitlines()
    # 从末尾向前找，若连续 >=4 行都匹配参考文献行特征，截断
    cutoff = len(tail_lines)
    ref_run = 0
    for i in range(len(tail_lines) - 1, -1, -1):
        l = tail_lines[i].strip()
        if not l:
            continue
        if _REF_LINE_PATTERN.match(l) or _AUTHOR_YEAR_PATTERN.search(l):
            ref_run += 1
        else:
            if ref_run >= 4:
                cutoff = i + 1
            ref_run = 0
    text = "\n".join(tail_lines[:cutoff]).strip()

    return text.strip()


def is_noise_chunk(text: str) -> bool:
    """
    判断 chunk 是否为噪声。
    修复：加强对参考文献密集段的检测。
    """
    text = text.strip()
    if len(text) < 80:
        return True

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return True

    # ── 原有规则 ──
    ref_lines = sum(1 for l in non_empty if _REF_LINE_PATTERN.match(l))
    if ref_lines / len(non_empty) > 0.30:   # 从 0.35 收紧到 0.30
        return True

    doi_lines = sum(1 for l in non_empty if "doi:" in l.lower())
    if doi_lines / len(non_empty) > 0.35:   # 从 0.40 收紧到 0.35
        return True

    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / max(len(text), 1) < 0.25:
        return True

    avg_line_len = sum(len(l.strip()) for l in non_empty) / len(non_empty)
    if avg_line_len < 20 and len(non_empty) > 5:
        return True

    # ── 修复新增：作者-年份-期刊行密集检测 ──────────────────────
    author_year_lines = sum(
        1 for l in non_empty if _AUTHOR_YEAR_PATTERN.search(l)
    )
    if author_year_lines / len(non_empty) > 0.50:
        return True

    # ── 修复新增：行均长度<60 且大量以大写字母开头的短行（典型参考文献） ──
    if avg_line_len < 80 and len(non_empty) >= 6:
        short_ref_like = sum(
            1 for l in non_empty
            if len(l.strip()) < 120
            and (_REF_LINE_PATTERN.match(l) or _AUTHOR_YEAR_PATTERN.search(l))
        )
        if short_ref_like / len(non_empty) > 0.45:
            return True

    return False
'''

print("=" * 60)
print("修复1 代码已就绪，将上面 FIXED_FILTER_CODE 中的函数")
print("替换到 build_sorghum_index_filter.py 对应位置即可。")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════
# 修复2：pipeline.py / metadata_loader.py
# 诊断 citation_map key 与 chunk source 的匹配问题
# ══════════════════════════════════════════════════════════════════

DIAGNOSTIC_CODE = '''
# ── 诊断脚本（在服务器上单独运行）──────────────────────────────
# python3 diagnose_citation.py

import os, re
import pandas as pd
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class E(Embeddings):
    def __init__(self, m): self.m = m
    def embed_documents(self, t):
        return self.m.encode(t, normalize_embeddings=True).tolist()
    def embed_query(self, t):
        return self.m.encode([t], normalize_embeddings=True).tolist()[0]

BASE = "/vol/sunjilin/website/data/agent"
CSV  = "/vol/sunjilin/website/data/publication/english_content.csv"

# 1. 读 CSV，看文件名列的实际值
df = pd.read_csv(CSV, encoding="utf-8-sig", engine="python")
print("=== CSV 列名 ===")
print(df.columns.tolist())
filename_col = df.columns[-1]
print(f"\\n文件名列: {repr(filename_col)}")
print("\\n前5个文件名示例:")
for v in df[filename_col].dropna().head(5):
    print(f"  {repr(str(v))}")

# 2. 读 FAISS，看 source 字段
m   = SentenceTransformer(f"{BASE}/models/bge-m3/")
db  = FAISS.load_local(
    f"{BASE}/faiss_v2_english_fine", E(m),
    allow_dangerous_deserialization=True
)
print("\\n=== FAISS chunk source 前5个 ===")
for i, (k, doc) in enumerate(db.docstore._dict.items()):
    print(f"  {repr(doc.metadata.get('source', ''))}")
    if i >= 4: break

# 3. 检查是否能匹配
csv_fnames = set(str(v).strip() for v in df[filename_col].dropna())
print(f"\\nCSV 文件名总数: {len(csv_fnames)}")

faiss_sources = set(
    doc.metadata.get("source", "")
    for doc in db.docstore._dict.values()
)
print(f"FAISS source 总数: {len(faiss_sources)}")

matched   = faiss_sources & csv_fnames
unmatched = faiss_sources - csv_fnames
print(f"\\n直接匹配数: {len(matched)}")
print(f"未匹配数:    {len(unmatched)}")
if unmatched:
    print("\\n前5个未匹配的 FAISS source:")
    for s in list(unmatched)[:5]:
        print(f"  {repr(s)}")
    print("\\n对应 CSV 中最接近的文件名:")
    for s in list(unmatched)[:5]:
        base = os.path.basename(s).lower().replace(".pdf","")
        candidates = [f for f in csv_fnames
                      if base in f.lower() or f.lower().replace(".pdf","") in base]
        print(f"  FAISS: {repr(s)}")
        print(f"    CSV候选: {candidates[:3]}")
'''

print("\n诊断脚本如下，保存为 diagnose_citation.py 后运行：")
print(DIAGNOSTIC_CODE)


# ══════════════════════════════════════════════════════════════════
# 修复2b：metadata_loader.py — 加强 key 注册，兼容更多格式
# 将下面这个函数替换 metadata_loader.py 中的 load_citation_map
# ══════════════════════════════════════════════════════════════════

FIXED_CITATION_MAP_CODE = '''
def load_citation_map(csv_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """
    从多个 CSV 中加载文献元数据，构建 citation_map。
    修复：注册更多 key 格式，覆盖 FAISS source 字段的各种写法。
    """
    citation_map: Dict[str, Dict[str, str]] = {}

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue

        df = read_csv_robust(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)
        filename_col = detect_filename_col(cols)

        for row in df.to_dict("records"):
            fname = norm_text(row.get(filename_col, ""))
            if not fname:
                continue

            authors_raw = pick(row, _EN_AUTHORS_COLS)
            title   = sentence_case(pick(row, _EN_TITLE_COLS))
            abstract= pick(row, _EN_ABSTRACT_COLS)
            keywords= pick(row, _EN_KEYWORDS_COLS)
            doi     = normalize_doi(pick(row, _EN_DOI_COLS))
            year    = pick(row, _EN_YEAR_COLS)
            journal = title_case(pick(row, _EN_JOURNAL_COLS))

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

            # ── 注册所有可能的 key 格式 ──────────────────────────
            keys_to_register = set()

            base = os.path.basename(fname)          # "Lin2012_SH1.pdf"
            stem = re.sub(r"\.pdf$", "", base, flags=re.I)  # "Lin2012_SH1"

            # 原始值及其变体
            for candidate in [fname, base, stem,
                               fname.strip(), base.strip(), stem.strip(),
                               fname.lower(), base.lower(), stem.lower()]:
                keys_to_register.add(candidate)

            # 去掉路径分隔符后的版本（防止 source 存了绝对路径）
            for sep in ["/", "\\\\"]:
                parts = fname.split(sep)
                if len(parts) > 1:
                    keys_to_register.add(parts[-1])
                    keys_to_register.add(
                        re.sub(r"\.pdf$", "", parts[-1], flags=re.I)
                    )

            for key in keys_to_register:
                if key:
                    citation_map[key] = info

    return citation_map
'''

print("\n修复2b：将上面 FIXED_CITATION_MAP_CODE 中的函数")
print("替换到 metadata_loader.py 中的 load_citation_map 即可。")


# ══════════════════════════════════════════════════════════════════
# 修复3：pipeline.py — 证据片段展示截断 & 清洗
# 修复参考文献区展示的 chunk 内容过于原始（含特殊字符）
# ══════════════════════════════════════════════════════════════════

FIXED_REF_LIST_CODE = '''
    def _build_reference_list(
        self,
        source_index: Dict[str, Dict[str, str]],
        selected_hits: List[ChunkHit],
        query_type: str,
    ) -> List[str]:
        """
        构建参考文献列表。
        修复：
        1. 证据片段清洗（去除特殊字符、多余空白、编号行）
        2. 只显示语义连贯的前两句，避免截断到参考文献段
        """
        ref_limit = REFERENCE_LIMITS.get(query_type, 6)
        sorted_items = sorted(source_index.items(), key=lambda x: x[1]["idx"])

        hits_by_source: Dict[str, List[ChunkHit]] = {}
        for hit in selected_hits:
            key = hit.source
            if key not in hits_by_source:
                hits_by_source[key] = []
            hits_by_source[key].append(hit)

        ref_lines = []
        for _, info in sorted_items[:ref_limit]:
            fname = info["fname"]
            idx   = info["idx"]
            ref   = safe_get_ref_info(fname, self.citation_map)
            ref_lines.append(build_citation_string(ref, idx, fname))

            source_hits = hits_by_source.get(fname, [])
            for hit in source_hits[:2]:
                preview = _clean_chunk_preview(hit.content)
                if preview:
                    ref_lines.append(f\'    > "{preview}"\')
            if source_hits:
                ref_lines.append("")

        return ref_lines


# ── 新增辅助函数（放在 pipeline.py 顶部，import 区之后）──────────
import re as _re

def _clean_chunk_preview(content: str, max_chars: int = 280) -> str:
    """
    清洗 chunk 内容，只取前两个完整句子作为证据预览。
    去掉：数字编号行、特殊符号、多余空白、参考文献行。
    """
    # 1. 去掉明显的参考文献行
    ref_pat = _re.compile(
        r"^\\s*(?:\\[\\d+\\]|\\d{1,3}[\\. ]\\s*[A-Z]|doi:\\s*10\\.).*$",
        _re.MULTILINE | _re.IGNORECASE,
    )
    content = ref_pat.sub("", content)

    # 2. 去掉页眉页脚残留
    content = _re.sub(
        r"©\\s*\\d{4}|all rights reserved|www\\.\\S+|doi:\\s*10\\.\\d{4}[^\\s]*",
        "", content, flags=_re.IGNORECASE
    )

    # 3. 合并空白，去掉孤立数字行
    lines = [l.strip() for l in content.splitlines()]
    lines = [l for l in lines if l and not _re.match(r"^\\d{1,4}$", l)]
    content = " ".join(lines)
    content = _re.sub(r"\\s+", " ", content).strip()

    # 4. 取前两句（句号/问号/感叹号截断）
    sentences = _re.split(r"(?<=[.!?])\\s+", content)
    preview = ""
    for sent in sentences:
        if len(preview) + len(sent) + 1 <= max_chars:
            preview = (preview + " " + sent).strip() if preview else sent
        else:
            break

    # 5. 如果仍然太长，硬截断并加省略号
    if len(preview) > max_chars:
        preview = preview[:max_chars].rsplit(" ", 1)[0] + "..."

    return preview
'''

print("\n修复3：将上面 FIXED_REF_LIST_CODE 替换 pipeline.py 中的")
print("_build_reference_list 方法，并在文件顶部加入 _clean_chunk_preview 函数。")
print("\n完成以上三处修复后，重建索引库即可解决参考文献片段乱码问题。")
print("注意：修复1（过滤规则）需要重新建库才能生效；修复2/3 改完直接重启 app.py 即可。")
