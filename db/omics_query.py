#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omics_query.py  ── 多组学统一查询接口
=======================================
放置在：rag_project/db/omics_query.py

被 prompt_builder.py 调用，根据问题类型查询不同的组学库，
将结果格式化为可直接注入 system prompt 的文本块。

外部只需调用：
    from omics_query import OmicsQueryHub
    hub = OmicsQueryHub()
    block = hub.query_for_prompt(user_query, query_type)
"""

import os
import re
import sqlite3
from typing import Optional, Dict, List

_BASE = os.path.dirname(os.path.abspath(__file__))

# 各库路径
_GENE_DB    = os.path.join(_BASE, "sorghum_genes.db")
_KNOWN_DB   = os.path.join(_BASE, "known_genes.db")
_QTL_DB     = os.path.join(_BASE, "qtl.db")
_OMICS_DB   = os.path.join(_BASE, "omics.db")

# 基因名识别正则（与 query_classifier 保持一致）
_GENE_RE = re.compile(
    r"""
    Sobic\.\d{3}G\d{6}
    | SbiHYZ\.\d{2}G\d{6}
    | SORBI_3\d{3}G\d{6}
    | (?<![a-zA-Z0-9])(?:AT1|DW[1-4]|MA[1-7]|TB1|SH1|AltSB|ARG1
           |SnRK\d*|WRKY\d+|NAC\d+|MYB\d+|bHLH\d+
           |ERF\d+|MADS\d+|SbMADS\d+|SbDW\d|SbTB\d
           |SbMA\d|SbSH\d|SbBX\d*|SbCYP\d+|SbWRKY\d+
           |GS3|BR2|SbAT1|BY1|GC1|qTGW1a|Awn1|D
    )(?![a-zA-Z0-9])
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _conn(db_path):
    if not os.path.exists(db_path):
        return None
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c

def _fmt(v, default="N/A"):
    if v is None or str(v).strip() in ("", "None", "nan"): return default
    return str(v).strip()


class OmicsQueryHub:
    """统一查询入口，根据 query_type 自动选择数据库。"""

    # ── 已知功能基因查询 ─────────────────────────────────────────
    def query_known_gene(self, name_or_id: str) -> Optional[Dict]:
        """按基因名或 Sobic ID 查询已知功能基因信息。"""
        conn = _conn(_KNOWN_DB)
        if not conn: return None
        cur = conn.cursor()
        row = cur.execute("""
            SELECT * FROM known_genes
            WHERE gene_name LIKE ? OR gene_id LIKE ?
            LIMIT 1
        """, (f"%{name_or_id}%", f"%{name_or_id}%")).fetchone()
        conn.close()
        return dict(row) if row else None

    def query_known_genes_by_trait(self, trait_kw: str, limit: int = 8) -> List[Dict]:
        """按性状关键词查询已知功能基因。"""
        conn = _conn(_KNOWN_DB)
        if not conn: return []
        rows = conn.execute("""
            SELECT gene_name, gene_id, trait, annotation, first_author, doi
            FROM known_genes WHERE trait LIKE ? OR annotation LIKE ?
            ORDER BY gene_name LIMIT ?
        """, (f"%{trait_kw}%", f"%{trait_kw}%", limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── QTL 查询 ─────────────────────────────────────────────────
    def query_qtl_by_trait(self, trait_kw: str, limit: int = 6) -> List[Dict]:
        """按性状关键词查询 QTL 位点。"""
        conn = _conn(_QTL_DB)
        if not conn: return []
        rows = conn.execute("""
            SELECT qtl_id, qtl_class, trait, trait_desc, publication, population,
                   t2t_chr, t2t_start, t2t_end, v3_chr, v3_start, v3_end,
                   n_genes_v3, url
            FROM qtl_loci WHERE trait LIKE ? OR trait_desc LIKE ?
            ORDER BY t2t_chr, t2t_start LIMIT ?
        """, (f"%{trait_kw}%", f"%{trait_kw}%", limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def query_qtl_by_gene(self, gene_id: str, limit: int = 5) -> List[Dict]:
        """查询某基因位于哪些 QTL 区间内。"""
        conn = _conn(_QTL_DB)
        if not conn: return []
        rows = conn.execute("""
            SELECT q.qtl_id, q.trait, q.trait_desc, q.publication,
                   q.t2t_chr, q.t2t_start, q.t2t_end, q.url
            FROM qtl_genes g JOIN qtl_loci q ON g.qtl_id = q.qtl_id
            WHERE g.gene_id = ? LIMIT ?
        """, (gene_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── 表型查询 ─────────────────────────────────────────────────
    def query_trait_stats(self, trait_kw: str) -> Optional[Dict]:
        """查询某量化性状的基本统计（均值/范围/样本数）。"""
        conn = _conn(_OMICS_DB)
        if not conn: return None
        row = conn.execute("""
            SELECT trait,
                   COUNT(DISTINCT sample_id) as n_samples,
                   ROUND(AVG(value),3) as mean_val,
                   ROUND(MIN(value),3) as min_val,
                   ROUND(MAX(value),3) as max_val,
                   unit,
                   GROUP_CONCAT(DISTINCT location) as locations,
                   GROUP_CONCAT(DISTINCT year)     as years
            FROM phenotype_quant WHERE trait LIKE ? AND value IS NOT NULL
            GROUP BY trait LIMIT 1
        """, (f"%{trait_kw}%",)).fetchone()
        conn.close()
        return dict(row) if row else None

    def query_trait_list(self) -> List[str]:
        """返回所有量化性状名称列表（用于检索召回辅助）。"""
        conn = _conn(_OMICS_DB)
        if not conn: return []
        rows = conn.execute(
            "SELECT DISTINCT trait FROM phenotype_quant ORDER BY trait"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # ── 代谢物查询 ───────────────────────────────────────────────
    def query_metabolite(self, name_kw: str, limit: int = 5) -> List[Dict]:
        """按代谢物名查询元信息（类别/通路）。"""
        conn = _conn(_OMICS_DB)
        if not conn: return []
        rows = conn.execute("""
            SELECT name, superclass, class, pathway
            FROM metabolite_meta WHERE name LIKE ? LIMIT ?
        """, (f"%{name_kw}%", limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def query_metabolite_by_pathway(self, pathway_kw: str, limit: int = 8) -> List[Dict]:
        """按通路关键词查询代谢物。"""
        conn = _conn(_OMICS_DB)
        if not conn: return []
        rows = conn.execute("""
            SELECT name, superclass, class, pathway
            FROM metabolite_meta WHERE pathway LIKE ? OR class LIKE ?
            LIMIT ?
        """, (f"%{pathway_kw}%", f"%{pathway_kw}%", limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ══════════════════════════════════════════════════════════════
    # 格式化输出（注入 prompt 的文本块）
    # ══════════════════════════════════════════════════════════════

    def format_known_gene(self, info: Dict) -> str:
        lines = [
            f"(KnownGene) {_fmt(info.get('gene_name'))} | ID: {_fmt(info.get('gene_id'))}",
            f"  Trait: {_fmt(info.get('trait'))}",
            f"  Annotation: {_fmt(info.get('annotation'))}",
            f"  Causative variant: {_fmt(info.get('causative_variant'))}",
            f"  Key reference: {_fmt(info.get('first_author'))}",
        ]
        if info.get("doi"):
            lines.append(f"  DOI: {info['doi']}")
        return "\n".join(lines)

    def format_qtl_list(self, qtl_list: List[Dict]) -> str:
        if not qtl_list: return ""
        # 取第一个 QTL 的 trait 作为标题（去下划线）
        trait_name = _fmt(qtl_list[0].get("trait") or qtl_list[0].get("trait_desc",""))
        trait_name = trait_name.replace("_", " ").strip()
        lines = [f"(QTLDB) {trait_name} — {len(qtl_list)} QTL loci:"]
        for q in qtl_list:
            pos = "position N/A"
            if q.get("t2t_chr") and q.get("t2t_start"):
                pos = f"BTx623 T2T {q['t2t_chr']}:{int(q['t2t_start']):,}-{int(q['t2t_end']):,}"
            elif q.get("v3_chr") and q.get("v3_start"):
                pos = f"BTx623v3 {q['v3_chr']}:{int(q['v3_start']):,}-{int(q['v3_end']):,}"
            n_genes = _fmt(q.get("n_genes_v3"), "?")
            pub = _fmt(q.get("publication")).replace("_", " ")
            qtl_id = _fmt(q.get("qtl_id"))
            lines.append(
                f"  {qtl_id} | {pos} | ~{n_genes} candidate genes | {pub}"
            )
        return "\n".join(lines)

    def format_trait_stats(self, stats: Dict) -> str:
        if not stats: return ""
        trait_display = _fmt(stats.get('trait','')). replace('_',' ')
        return (
            f"(PhenomeDB) Trait: {trait_display} | "
            f"N={_fmt(stats.get('n_samples'))} samples | "
            f"Mean={_fmt(stats.get('mean_val'))} {_fmt(stats.get('unit'))} | "
            f"Range=[{_fmt(stats.get('min_val'))}, {_fmt(stats.get('max_val'))}] | "
            f"Locations: {_fmt(stats.get('locations'))} | "
            f"Years: {_fmt(stats.get('years'))}"
        )

    def format_metabolite_list(self, mets: List[Dict]) -> str:
        if not mets: return ""
        lines = [f"(MetabolomeDB) Related metabolites:"]
        for m in mets:
            lines.append(
                f"  {_fmt(m.get('name'))} | {_fmt(m.get('class'))} "
                f"| pathway: {_fmt(m.get('pathway'))}"
            )
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # 主入口：根据 query_type 自动组装数据块
    # ══════════════════════════════════════════════════════════════

    def query_for_prompt(self, user_query: str, query_type: str,
                         max_chars: int = 800) -> str:
        blocks = []
        q_lower = user_query.lower()
        gene_names = list(dict.fromkeys(_GENE_RE.findall(user_query)))

        # ── 中英文 → 数据库性状名映射 ────────────────────────────
        TRAIT_MAP = {
            "抗旱": "Stay-green", "干旱": "Stay-green",
            "stay-green": "Stay-green", "持绿": "Stay-green",
            "水分利用": "Water use efficiency",
            "耐盐": "Relative salt injury rate",
            "盐碱": "Relative salt injury rate",
            "耐铝": "Aluminium tolerance", "铝毒": "Aluminium tolerance",
            "产量": "Grain yield", "grain yield": "Grain yield",
            "籽粒产量": "Grain yield",
            "籽粒重": "Grain weight", "粒重": "Grain weight",
            "粒数": "Grain number",
            "株高": "Height (plant height)", "plant height": "Height (plant height)",
            "高度": "Height (plant height)",
            "开花": "Days to flowering", "花期": "Days to flowering",
            "flowering": "Days to flowering",
            "成熟": "Maturity", "maturity": "Maturity",
            "生物量": "Biomass", "biomass": "Biomass",
            "茎径": "Stem diameter", "茎粗": "Stem diameter",
            "叶长": "Leaf length", "叶面积": "Leaf area",
            "叶绿素": "Leaf chlorophyll content",
            "根长": "Root length", "根重": "Root dry weight",
            "根角": "Root angle", "根": "Root length",
            "单宁": "Tannin content", "tannin": "Tannin content",
            "蛋白质": "Protein content", "protein": "Protein content",
            "淀粉": "Starch", "starch": "Starch",
            "花青素": "Anthocyanin level",
            "蜡质": "Epicuticular wax",
            "抗病": "Grain mould resistance",
            "霉变": "Grain mould resistance",
            "独脚金": "Resistance to Striga",
            "striga": "Resistance to Striga",
            "分蘖": "Tiller number", "tiller": "Tiller number",
            "芒": "Awn presence", "awn": "Awn presence",
            "出苗": "Emergence rate", "发芽": "Germination rate",
            "倒伏": "Lodging tolerance",
        }

        trait_kws = []
        for kw, db_trait in TRAIT_MAP.items():
            if kw.lower() in q_lower:
                if db_trait not in trait_kws:
                    trait_kws.append(db_trait)

        # 无匹配时对 qtl_gwas 用 Stay-green 兜底
        if query_type == "qtl_gwas" and not trait_kws:
            trait_kws = ["Stay-green"]

        # ── 已知功能基因 ──────────────────────────────────────────
        if query_type in {"gene_function", "factoid", "mechanism", "qtl_gwas"}:
            for gname in gene_names[:2]:
                info = self.query_known_gene(gname)
                if info:
                    blocks.append(self.format_known_gene(info))

        # ── QTL 位点 ──────────────────────────────────────────────
        if query_type in {"qtl_gwas", "gene_function", "review"}:
            for gene_id in gene_names[:1]:
                qtl_list = self.query_qtl_by_gene(gene_id, limit=3)
                if qtl_list:
                    blocks.append(self.format_qtl_list(qtl_list))
                    break
            for kw in trait_kws[:2]:
                qtl_list = self.query_qtl_by_trait(kw, limit=5)
                if qtl_list:
                    blocks.append(self.format_qtl_list(qtl_list))
                    break

        # ── 表型统计 ──────────────────────────────────────────────
        if query_type in {"review", "mechanism", "qtl_gwas"}:
            for kw in trait_kws[:1]:
                stats = self.query_trait_stats(kw)
                if stats:
                    blocks.append(self.format_trait_stats(stats))
                    break

        # ── 代谢物 ────────────────────────────────────────────────
        if query_type in {"mechanism", "review"}:
            met_kws = []
            for w in ["flavonoid","phenolic","anthocyanin","lignin","tannin",
                      "ABA","abscisic","jasmonic","ROS","carotenoid",
                      "类黄酮","酚类","花青素","木质素","脱落酸"]:
                if w.lower() in q_lower:
                    met_kws.append(w)
            for kw in met_kws[:1]:
                mets = self.query_metabolite_by_pathway(kw, limit=4)
                if mets:
                    blocks.append(self.format_metabolite_list(mets))
                    break

        if not blocks:
            return ""
        result = "\n\n".join(blocks)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n...(截断)"
        return result


# ── 快速验证 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    hub = OmicsQueryHub()
    print("=== 测试1：已知基因查询 AT1 ===")
    info = hub.query_known_gene("AT1")
    if info: print(hub.format_known_gene(info))
    else: print("  未找到（known_genes.db 未建库？）")

    print("\n=== 测试2：QTL 性状查询（drought）===")
    qtls = hub.query_qtl_by_trait("drought", limit=3)
    if qtls: print(hub.format_qtl_list(qtls))
    else: print("  未找到（qtl.db 未建库？）")

    print("\n=== 测试3：prompt 注入测试 ===")
    block = hub.query_for_prompt("介绍AT1基因的功能", "gene_function")
    print(block if block else "  （无数据）")
