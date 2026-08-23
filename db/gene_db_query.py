#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gene_db_query.py
================
SorGPT 基因注释查询接口层。
将 SQLite 查询结果格式化为可注入 RAG prompt 的文本块。

使用方式（在 pipeline.py 中）：
    from gene_db_query import query_gene_annotation, format_for_prompt
    ann = query_gene_annotation("Sobic.001G000400")
    block = format_for_prompt(ann)
    # 将 block 拼入 system_prompt
"""

import sqlite3
import re
import os
import json
from typing import Optional, Dict, List, Tuple

GENE_DB = os.path.join(os.path.dirname(__file__), '../db/sorghum_genes.db')

# ── 符号 → Sobic ID 权威映射（zh-v3：合并 gene_symbol_map.json，JSON 优先）─────
def _load_symbol_map():
    """基因符号→Sobic ID 映射。gene_symbol_map.json 优先，硬编码兜底。"""
    _m = {
        "AT1": "Sobic.001G341700", "SbAT1": "Sobic.001G341700",
        "SH1": "Sobic.001G152901",
        "AltSB": "Sobic.003G403000",
        "ARG1": "Sobic.007G085350",
        "DW3": "Sobic.007G163800", "SbDW3": "Sobic.007G163800",
        "DW1": "Sobic.009G229800", "SbDW1": "Sobic.009G229800",
        "DW2": "Sobic.006G067700", "SbDW2": "Sobic.006G067700",
        "MA1": "Sobic.006G057600", "SbMA1": "Sobic.006G057600",
        "MA2": "Sobic.006G095600",
        "MA3": "Sobic.010G230100",
        "TB1": "Sobic.001G121200", "SbTB1": "Sobic.001G121200",
        "BY1": "Sobic.002G379600",
        "GC1": "Sobic.010G022600",
        "Y1": "Sobic.006G030400",
        "Tan1": "Sobic.004G280200",
        "B1": "Sobic.004G071000",
        "RCN1": "Sobic.003G361100",
    }
    try:
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gene_symbol_map.json")
        if os.path.exists(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _d = json.load(_f)
            _d.pop("_comment", None)
            for _k, _v in _d.items():
                if _k and _v:
                    _m[_k] = _v  # JSON 覆盖硬编码（统一冲突 ID）
            # 大小写归一：JSON 已收录的符号（小写相同）删掉硬编码残留，
            # 避免 MA1(旧) 与 Ma1(新) 同时命中造成重复 ID。
            _jlow = {str(k).lower() for k in _d}
            for _k in list(_m.keys()):
                _kl = str(_k).lower()
                if _kl in _jlow and _k not in _d:
                    del _m[_k]
    except Exception:
        pass
    return _m


SYMBOL_MAP = _load_symbol_map()


# ──────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(GENE_DB)
    c.row_factory = sqlite3.Row
    return c

def _extract_gene_ids(query: str) -> List[str]:
    """从用户问题中提取高粱基因 ID（支持Sobic格式 + 常用基因名）"""
    patterns = [
        r'Sobic\.\d{3}G\d{6}',
        r'SORBI_3\d{3}G\d{6}',
        r'SbiHYZ\.\d{2}G\d{6}',
        r'mikado\.[A-Za-z0-9]+',
    ]
    ids = []
    for pat in patterns:
        ids.extend(re.findall(pat, query))

    # 常用基因名 → Sobic ID 映射（zh-v3: 合并 gene_symbol_map.json，JSON 优先）
    _NAME_MAP = SYMBOL_MAP
    for name, sobic_id in _NAME_MAP.items():
        if re.search(r'(?<![a-zA-Z0-9])' + re.escape(name) + r'(?![a-zA-Z0-9])',
                     query, re.IGNORECASE):
            if sobic_id not in ids:
                ids.append(sobic_id)

    return list(dict.fromkeys(ids))

def _fmt_pos(chr_, start, end, strand) -> str:
    if chr_ and start and end:
        return f"{chr_}:{start:,}-{end:,}({strand or '?'})"
    return "N/A"

# ──────────────────────────────────────────────────────────────────
# 查询函数
# ──────────────────────────────────────────────────────────────────
def query_gene_annotation(gene_id: str) -> Optional[Dict]:
    """
    查询单个基因的完整注释信息。
    支持 BTx623 ID / HYZ ID / v3 ID 三种输入格式。
    """
    conn = _conn()
    cur  = conn.cursor()

    # 自动识别输入格式，统一转为 BTx623 主键
    if gene_id.startswith('SbiHYZ'):
        row = cur.execute(
            "SELECT * FROM genes WHERE hyz_id=?", (gene_id,)
        ).fetchone()
    elif gene_id.startswith('SORBI_3'):
        row = cur.execute(
            "SELECT * FROM genes WHERE v3_id=?", (gene_id,)
        ).fetchone()
    else:
        row = cur.execute(
            "SELECT * FROM genes WHERE gene_id=?", (gene_id,)
        ).fetchone()

    if not row:
        conn.close()
        return None

    result = dict(row)
    gid = result['gene_id']

    # Pfam
    result['pfam'] = [dict(r) for r in cur.execute(
        "SELECT pfam_id, pfam_name, pfam_desc FROM pfam WHERE gene_id=?",
        (gid,)
    ).fetchall()]

    # GO
    result['go'] = [dict(r) for r in cur.execute(
        "SELECT go_id, go_name, namespace FROM go_annotation WHERE gene_id=?",
        (gid,)
    ).fetchall()]

    # 其他功能注释（按 db_name 分组）
    func_rows = cur.execute(
        """SELECT db_name, term_id, term_name FROM func_annotation
           WHERE gene_id=? ORDER BY db_name""",
        (gid,)
    ).fetchall()
    func_by_db: Dict[str, List] = {}
    for r in func_rows:
        db = r['db_name']
        func_by_db.setdefault(db, []).append({
            'term_id': r['term_id'], 'term_name': r['term_name']
        })
    result['func_by_db'] = func_by_db

    # 同线性对应
    result['orthologs'] = [dict(r) for r in cur.execute(
        "SELECT genome, ortho_id, position FROM orthologs WHERE btx623_id=?",
        (gid,)
    ).fetchall()]

    conn.close()
    return result


def query_by_pfam(pfam_id: str, limit: int = 5):
    """
    按 Pfam ID 查询含该结构域的基因。
    返回格式化文本字符串（适合注入 prompt）。
    """
    import sqlite3, os
    db   = GENE_DB if GENE_DB else os.path.join(os.path.dirname(__file__), "sorghum_genes.db")
    conn = sqlite3.connect(db)

    meta = conn.execute(
        "SELECT pfam_name, pfam_desc FROM pfam WHERE pfam_id=? LIMIT 1", (pfam_id,)
    ).fetchone()
    pfam_name = meta[0] if meta else pfam_id
    pfam_desc = meta[1][:200] if meta and meta[1] else ""

    n_genes = conn.execute(
        "SELECT COUNT(DISTINCT gene_id) FROM pfam WHERE pfam_id=?", (pfam_id,)
    ).fetchone()[0]

    rows = conn.execute("""
        SELECT p.gene_id, g.chr, g.start, g.end, g.strand,
               g.hyz_id, g.rice_id
        FROM pfam p
        JOIN genes g ON p.gene_id = g.gene_id
        WHERE p.pfam_id = ?
        ORDER BY g.chr, g.start
        LIMIT ?
    """, (pfam_id, limit)).fetchall()
    conn.close()

    lines = [
        f"(PfamDB) {pfam_id} | 家族名称: {pfam_name}",
        f"  功能: {pfam_desc[:150]}" if pfam_desc else "",
        f"  高粱BTx623 T2T中共 {n_genes} 个基因含此结构域",
    ]
    if rows:
        lines.append(f"  代表性基因（按染色体排序，前{len(rows)}个）:")
        for r in rows:
            gene_id, chr_, s, e, strand = r[0], r[1], r[2], r[3], r[4]
            hyz   = f" | HYZ:{r[5]}" if r[5] else ""
            rice  = f" | Rice:{r[6]}" if r[6] else ""
            lines.append(f"    {gene_id} | {chr_}:{s:,}-{e:,}({strand}){hyz}{rice}")
    return "\n".join(l for l in lines if l)

def query_by_go(go_id: str, limit: int = 20) -> List[Dict]:
    """按 GO term 查询基因列表"""
    conn = _conn()
    rows = conn.execute("""
        SELECT g.gene_id, g.chr, g.start, g.end,
               go.go_name, go.namespace
        FROM genes g
        JOIN go_annotation go ON g.gene_id = go.gene_id
        WHERE go.go_id = ?
        ORDER BY g.chr, g.start
        LIMIT ?
    """, (go_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_by_region(chr_: str, start: int, end: int) -> List[Dict]:
    """按染色体区间查询基因"""
    conn = _conn()
    rows = conn.execute("""
        SELECT gene_id, chr, start, end, strand, hyz_id, v3_id, rice_id
        FROM genes
        WHERE chr=? AND start>=? AND end<=?
        ORDER BY start
    """, (chr_, start, end)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_ortholog(gene_id: str) -> Dict[str, str]:
    """查询某个基因在三个基因组中的对应 ID"""
    conn = _conn()
    row = conn.execute(
        "SELECT gene_id, hyz_id, v3_id, rice_id FROM genes WHERE gene_id=?",
        (gene_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {}
    return {
        'BTx623_T2T': row['gene_id'],
        'HYZ':        row['hyz_id'] or '.',
        'BTx623v3':   row['v3_id']  or '.',
        'Rice':       row['rice_id'] or '.',
    }

# ──────────────────────────────────────────────────────────────────
# 格式化为 RAG prompt 文本块
# ──────────────────────────────────────────────────────────────────
def format_for_prompt(ann: Dict, max_terms: int = 5) -> str:
    """
    将注释字典格式化为注入 system_prompt 的文本块。
    控制长度，避免 prompt 爆炸。
    """
    gid = ann['gene_id']
    pos = _fmt_pos(ann.get('chr'), ann.get('start'),
                   ann.get('end'), ann.get('strand'))
    lines = [
        f"(GeneDB) {gid} | {pos}",
    ]

    # 同线性对应
    orthos = ann.get('orthologs', [])
    if orthos:
        o_parts = [f"{o['genome']}:{o['ortho_id']}" for o in orthos[:3]]
        lines.append(f"  Synteny: {' | '.join(o_parts)}")

    # 水稻同源
    if ann.get('rice_id'):
        lines.append(f"  Rice homolog: {ann['rice_id']}")

    # Pfam（最多 max_terms 个）
    if ann.get('pfam'):
        pfam_parts = []
        for p in ann['pfam'][:max_terms]:
            desc = p.get('pfam_desc') or p.get('pfam_name') or ''
            pfam_parts.append(f"{p['pfam_id']}({desc})" if desc else p['pfam_id'])
        lines.append(f"  Pfam domains: {'; '.join(pfam_parts)}")

    # GO（只显示 biological_process，限 max_terms 个）
    go_bp = [g for g in ann.get('go', []) if g.get('namespace') == 'biological_process']
    if not go_bp:
        go_bp = ann.get('go', [])[:max_terms]
    if go_bp:
        go_parts = [f"{g['go_id']}({g['go_name']})" if g.get('go_name') else g['go_id']
                    for g in go_bp[:max_terms]]
        lines.append(f"  GO (biological process): {'; '.join(go_parts)}")

    # 其他重要数据库（IPR / PANTHER）
    fdb = ann.get('func_by_db', {})
    for db in ['interpro', 'panther']:
        terms = fdb.get(db, [])[:3]
        if terms:
            t_parts = [f"{t['term_id']}({t['term_name']})" if t.get('term_name') else t['term_id']
                       for t in terms]
            lines.append(f"  {db.capitalize()}: {'; '.join(t_parts)}")

    return '\n'.join(lines)


def _is_generic_list_query(user_query: str) -> bool:
    """泛列举类问题（问题本身不含具体基因名，如'驯化基因有哪些'）。"""
    _marks = ["哪些", "有哪些", "有哪", "基因有", "调控基因", "控制基因", "相关基因",
              "参与", "list", "gene list", "genes", "count", "多少", "gene(s)"]
    _q = (user_query or "").lower()
    return any(m in _q for m in _marks)


def _format_symbol_ref_block() -> str:
    """输出 已知符号→Sobic ID 参考对照块（仅作 ID 映射，非证据来源）。"""
    if not SYMBOL_MAP:
        return ""
    lines = ["【已知基因符号 → Sobic ID 参考对照（BTx623 T2T）】"]
    lines.append("以下符号→ID 对照仅用于把来源片段中出现的符号映射为基因座 ID，不是证据来源；")
    lines.append("只有来源片段实际提到该符号时才可使用对应 ID，来源未提及的基因禁止引入。")
    for _name, _id in sorted(SYMBOL_MAP.items()):
        lines.append("- %s = %s" % (_name, _id))
    return "\n".join(lines)


def query_and_format(user_query: str, max_genes: int = 2) -> str:
    """
    主入口：从用户问题中提取基因ID / Pfam / IPR / GO 编号，
    查询并格式化为 prompt 文本块。
    """
    import re as _re
    blocks = []

    # ── 1. 基因 ID 查询（原有逻辑）──────────────────────────
    gene_ids = _extract_gene_ids(user_query)
    for gid in gene_ids[:max_genes]:
        ann = query_gene_annotation(gid)
        if ann:
            blocks.append(format_for_prompt(ann))

    # ── 1.5 泛列举兜底（zh-v3）：中文问题无具体基因名时注入符号映射参考 ──
    # 仅中文问题注入（英文路径文献片段已自带 ID，避免额外上下文影响英文质量）
    _cn = sum(1 for c in user_query if "一" <= c <= "鿿") / max(len(user_query), 1) > 0.15
    if _cn and not gene_ids and _is_generic_list_query(user_query):
        _ref = _format_symbol_ref_block()
        if _ref:
            blocks.append(_ref)

    # ── 2. Pfam ID 自动检测 ──────────────────────────────────
    for m in _re.finditer(r'(?<![a-zA-Z0-9])PF(\d{5})(?![a-zA-Z0-9])', user_query, _re.IGNORECASE):
        pfam_id = "PF" + m.group(1)
        r = query_by_pfam(pfam_id, limit=5)
        if r:
            blocks.append(r)

    # ── 3. IPR ID 自动检测 ───────────────────────────────────
    for m in _re.finditer(r'(?<![a-zA-Z0-9])IPR(\d{6})(?![a-zA-Z0-9])', user_query, _re.IGNORECASE):
        ipr_id = "IPR" + m.group(1)
        r = query_count_by_db("interpro", ipr_id)
        if r:
            blocks.append(r)

    # ── 4. GO ID 自动检测 ────────────────────────────────────
    for m in _re.finditer(r'(?<![a-zA-Z0-9])GO:(\d{7})(?![a-zA-Z0-9])', user_query, _re.IGNORECASE):
        go_id = "GO:" + m.group(1)
        r = query_count_by_db("go", go_id)
        if r:
            blocks.append(r)

    # ── 5. 数量类问题：补充统计摘要 ─────────────────────────
    if blocks and any(p in user_query for p in [
        "多少个","多少","how many","count","统计"
    ]):
        # 已在各查询函数中包含数量，无需额外处理
        pass

    return '\n\n'.join(blocks)


# ──────────────────────────────────────────────────────────────────
# 快速验证（直接运行时）
# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    gid = sys.argv[1] if len(sys.argv) > 1 else 'Sobic.001G000400'
    print(f"\n查询基因: {gid}")
    ann = query_gene_annotation(gid)
    if ann:
        print(format_for_prompt(ann))
        print(f"\n原始字段数: {len(ann)}")
        print(f"Pfam 条目: {len(ann.get('pfam',[]))}")
        print(f"GO 条目: {len(ann.get('go',[]))}")
        print(f"同线性对应: {len(ann.get('orthologs',[]))}")
        print(f"\n跨基因组ID：{query_ortholog(gid)}")
    else:
        print(f"未找到 {gid}，请先运行 parse_combined.py 建库")


def query_count_by_db(db_name: str, term_id: str) -> str:
    """按数据库名和term_id统计基因数量（CDD/PANTHER/IPR/GO等）。"""
    import sqlite3, os
    db   = GENE_DB if GENE_DB else os.path.join(os.path.dirname(__file__), "sorghum_genes.db")
    conn = sqlite3.connect(db)
    if db_name.upper() == "GO":
        n = conn.execute(
            "SELECT COUNT(DISTINCT gene_id) FROM go_annotation WHERE go_id=?", (term_id,)
        ).fetchone()[0]
        name_row = conn.execute(
            "SELECT go_name FROM go_annotation WHERE go_id=? AND go_name!=\'\' LIMIT 1", (term_id,)
        ).fetchone()
        name = name_row[0] if name_row else ""
    elif db_name.upper() in ("PFAM", "PF"):
        n = conn.execute(
            "SELECT COUNT(DISTINCT gene_id) FROM pfam WHERE pfam_id=?", (term_id,)
        ).fetchone()[0]
        nr = conn.execute("SELECT pfam_name FROM pfam WHERE pfam_id=? LIMIT 1",(term_id,)).fetchone()
        name = nr[0] if nr else ""
    else:
        n = conn.execute(
            "SELECT COUNT(DISTINCT gene_id) FROM func_annotation WHERE db_name=? AND term_id=?",
            (db_name.lower(), term_id)
        ).fetchone()[0]
        name = ""
    conn.close()
    label = f"{name} " if name and name != term_id else ""
    return f"(DBCount) {db_name}:{term_id} {label}→ 高粱BTx623 T2T中共 {n} 个基因"
