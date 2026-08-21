# -*- coding: utf-8 -*-
"""
sequence_fetcher.py — 基因序列提取 + 符号→ID 解析。

给 pipeline.py（答案序列注入）和 api_server.py（/sequence 代理端点）共用。
序列来源：sorghumdb get-seq 接口（samtools faidx 从 BTx623 T2T 基因组提取）。
基因符号→ID 映射来自 gene_symbol_map.json（sorghumdb 权威克隆基因 + 文献补录）。
"""
import json
import os
import re
import urllib.request

from config import SEQ_PREVIEW_BP, SEQ_PREVIEW_AA, GET_SEQ_URL, SEQ_SYMBOL_MAP_PATH

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sobic / HYZ / BTx623v3 三种官方 ID
_ID_RE = re.compile(
    r"Sobic\.\d{3}G\d{6}|SbiHYZ\.\d{2}G\d{6}|SORBI_3\d{3}G\d{6}",
    re.IGNORECASE,
)


def _load_symbol_map():
    path = SEQ_SYMBOL_MAP_PATH or os.path.join(_BASE_DIR, "gene_symbol_map.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("_comment", None)
        return {k: v for k, v in data.items() if v}
    except Exception:
        return {}


_SYMBOL_MAP = _load_symbol_map()


def resolve_gene_id(token):
    """token 可能是 Sobic ID 或基因符号；返回标准 ID 或 None。"""
    if not token:
        return None
    t = str(token).strip()
    m = _ID_RE.search(t)
    if m:
        return m.group(0)
    tl = t.lower()
    for sym, gid in _SYMBOL_MAP.items():
        if sym.lower() == tl:
            return gid
    return None


def resolve_genes_from_query(query):
    """从用户问题里提取基因（ID 或符号），解析成去重的 Sobic ID 列表。"""
    if not query:
        return []
    ids = []
    # 1) 直接 ID
    for m in _ID_RE.finditer(query):
        g = m.group(0)
        if g not in ids:
            ids.append(g)
    # 2) 符号（长符号优先，避免子串误匹配）
    for sym in sorted(_SYMBOL_MAP.keys(), key=len, reverse=True):
        pat = r"(?<![a-zA-Z0-9])" + re.escape(sym) + r"(?![a-zA-Z0-9])"
        if re.search(pat, query, re.IGNORECASE):
            g = _SYMBOL_MAP[sym]
            if g not in ids:
                ids.append(g)
    return ids[:3]


def detect_seq_type(query):
    """根据问题判断序列类型：dna / cds / protein。默认 dna。"""
    q = (query or "").lower()
    if any(p in q for p in ["protein", "peptide", "amino acid", "氨基酸", "蛋白序列", "蛋白质序列"]):
        return "protein"
    if any(p in q for p in ["cds", "coding", "编码序列", "编码区"]):
        return "cds"
    return "dna"


def fetch_sequence(gene_id, timeout=30):
    """调 get-seq，返回原始 dict；失败返回 None。"""
    try:
        payload = json.dumps({"geneid": gene_id}).encode("utf-8")
        req = urllib.request.Request(
            GET_SEQ_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _extract(d, seq_type):
    """从 get-seq 原始返回里取出指定类型序列字符串 + 位置信息。"""
    if not d or d.get("code") != 1:
        return "", None
    info = (d.get("gene") or {}).get("info") or {}
    meta = {"chr": info.get("chr", ""), "start": info.get("start"), "end": info.get("end")}
    if seq_type == "protein":
        pep = d.get("pep") or {}
        if pep:
            k = next(iter(pep))
            return pep[k], meta
        return "", meta
    if seq_type == "cds":
        cds = d.get("cds") or {}
        if cds:
            k = next(iter(cds))
            return cds[k], meta
        return "", meta
    return info.get("sequence", ""), meta


def build_sequence_blocks(gene_id, seq_type="dna", lang="chinese"):
    """一次 fetch，返回 (prompt上下文行, 序列预览块)。失败返回 ("", "")。

    上下文行用于注入 system prompt，让 LLM 序言采用正确的 ID/类型/长度/位置，
    避免模型凭记忆编造错误的 Sobic ID。
    """
    seq, meta = _extract(fetch_sequence(gene_id), seq_type)
    if not seq:
        return "", ""
    is_protein = seq_type == "protein"
    n = SEQ_PREVIEW_AA if is_protein else SEQ_PREVIEW_BP
    head = seq[:n]
    unit = "aa" if is_protein else "bp"
    if seq_type == "protein":
        label_zh, label_en = "氨基酸序列", "amino acid"
    elif seq_type == "cds":
        label_zh, label_en = "CDS序列", "CDS"
    else:
        label_zh, label_en = "核苷酸序列", "nucleotide"
    loc = ""
    if meta.get("chr"):
        _chr = str(meta["chr"]).strip()
        if _chr.lower().startswith("chr"):
            _chr = _chr[3:]
        loc = " chr%s:%s-%s" % (_chr, meta["start"], meta["end"])
    detail_url = "http://www.sorghumdb.com/geneDetail?gene=%s&search=%s" % (gene_id, gene_id)
    if lang == "chinese":
        ctx = "基因 %s 的%s：全长 %d %s，位置%s。" % (gene_id, label_zh, len(seq), unit, loc)
        preview = (
            "【%s %s（前 %d %s，全长 %d %s）】\n%s...\n\n"
            "[在 sorghumdb 查看完整序列](%s)"
            % (gene_id, label_zh, len(head), unit, len(seq), unit, head, detail_url)
        )
    else:
        ctx = "%s %s: length %d %s, location%s." % (gene_id, label_en, len(seq), unit, loc)
        preview = (
            "[%s %s sequence (first %d %s of %d %s)]\n%s...\n\n"
            "[View full sequence on sorghumdb](%s)"
            % (gene_id, label_en, len(head), unit, len(seq), unit, head, detail_url)
        )
    return ctx, preview


def build_sequence_preview(gene_id, seq_type="dna", lang="chinese"):
    """生成注入答案的序列预览文本（前 N bp/aa）。保留供回归测试/兼容调用。"""
    _, preview = build_sequence_blocks(gene_id, seq_type=seq_type, lang=lang)
    return preview


def fetch_and_format(gene_id, seq_type="dna"):
    """供 /sequence 端点：返回干净 dict。"""
    gid = resolve_gene_id(gene_id)
    if not gid:
        return {"code": 0, "msg": "无法解析该基因标识"}
    d = fetch_sequence(gid)
    if not d or d.get("code") != 1:
        return {"code": 0, "msg": "序列提取失败"}
    seq, meta = _extract(d, seq_type)
    return {
        "code": 1,
        "gene_id": gid,
        "type": seq_type,
        "sequence": seq,
        "length": len(seq),
        "chr": meta.get("chr"),
        "start": meta.get("start"),
        "end": meta.get("end"),
    }
