# -*- coding: utf-8 -*-
"""
pipeline.py
========================
【作用】
这是整个 SorGPT RAG 系统的总控脚本。
负责把所有模块串起来，形成完整问答流程。
【完整流程】
1. 问题分类
2. 规则提取英文检索关键词（从外部 JSON 热加载，无需重启）
3. metadata 检索
4. （必要时）全文检索
5. rerank + 多源去重
6. prompt 构建
7. 调用大模型生成答案（只调用一次）
8. 组装参考文献（每条文献后紧跟证据片段）

【修复记录】
- _clean_chunk_preview：清洗证据片段，去掉参考文献编号行、页眉页脚残留
- _build_reference_list：调用 _clean_chunk_preview，输出干净的证据预览
"""
import re
import json
import os
from typing import Dict, Any, List, Iterator
from config import CSV_PATHS, REFERENCE_LIMITS, COUNT_QUERY_MAX_SHOW
from embeddings import BgeEmbeddingsWrapper
from metadata_loader import load_citation_map, safe_get_ref_info
from query_classifier import classify_query_type
from retriever import Retriever, MetaPaper, ChunkHit
from reranker import Reranker
from prompt_builder import build_system_prompt
from generator import AnswerGenerator
from utils import build_citation_string, norm_text

# -----------------------------
# 证据片段清洗（修复：去掉参考文献行、页眉页脚残留）
# -----------------------------
_REF_LINE_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d{1,3}[\. ]\s*[A-Z]|doi:\s*10\.).*$",
    re.MULTILINE | re.IGNORECASE,
)
_NOISE_RE = re.compile(
    r"©\s*\d{4}|all rights reserved|www\.\S+\.\S+",
    re.IGNORECASE,
)

def _clean_chunk_preview(content: str, max_chars: int = 280) -> str:
    """
    清洗 chunk 内容，取前两个完整句子作为证据预览。
    去掉：参考文献编号行、页眉页脚残留、孤立数字行、多余空白。
    """
    # 1. 去掉参考文献编号行（[1] / 36. Lin / doi:10. 开头）
    content = _REF_LINE_RE.sub("", content)
    # 2. 去掉页眉页脚
    content = _NOISE_RE.sub("", content)
    # 3. 逐行清洗：去空行、去孤立数字行
    lines = [l.strip() for l in content.splitlines()]
    lines = [l for l in lines if l and not re.match(r"^\d{1,4}$", l)]
    content = " ".join(lines)
    content = re.sub(r"\s+", " ", content).strip()
    # 4. 取前两个完整句子（句号/问号/感叹号断句）
    sentences = re.split(r"(?<=[.!?])\s+", content)
    preview = ""
    for sent in sentences:
        if len(preview) + len(sent) + 1 <= max_chars:
            preview = (preview + " " + sent).strip() if preview else sent
        else:
            break
    # 5. 超长则硬截断加省略号
    if len(preview) > max_chars:
        preview = preview[:max_chars].rsplit(" ", 1)[0] + "..."
    return preview

# -----------------------------
# 外部 JSON 词典加载（热更新：每次问答时重新读取，修改 json 无需重启）
# -----------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(filename: str) -> dict:
    path = os.path.join(_BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def extract_keywords_by_rule(query: str) -> str:
    """
    基于规则从问题中提取英文检索关键词。
    支持中英文双语输入。
    每次调用时热加载 JSON 词典，修改词典文件后无需重启程序。
    """
    zh_to_en = _load_json("keywords_zh2en.json")
    domain_injection = _load_json("domain_injection.json")
    keywords = []
    q_lower = query.lower()
    # 1. 中文词典匹配
    for zh, en in zh_to_en.items():
        if zh in query:
            keywords.append(en)
    # 2. 保留原问题中已有的英文词（基因名、术语等）
    en_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-\.]*[A-Za-z0-9]|[A-Za-z]{2,}", query)
    keywords.extend(en_tokens)
    # 3. 领域专项注入
    for key, injection in domain_injection.items():
        if key in q_lower or key in query:
            keywords.append(injection)
    # 4. 纯英文问题：直接把原始问题加入关键词基础
    chinese_char_count = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    if chinese_char_count == 0:
        keywords.append(norm_text(query))
    # 5. 确保 sorghum 始终在关键词中
    if "sorghum" not in " ".join(keywords).lower():
        keywords.append("sorghum")
    # 6. 去重拼接
    seen = set()
    result = []
    for kw in keywords:
        k = kw.strip().lower()
        if k and k not in seen:
            seen.add(k)
            result.append(kw.strip())
    return ", ".join(result) if result else norm_text(query)


class SorghumRAGPipeline:
    """
    SorGPT 总控 pipeline。
    """
    def __init__(self):
        self.embed_model = BgeEmbeddingsWrapper()
        self.citation_map = load_citation_map(CSV_PATHS)
        self.retriever = Retriever(self.embed_model, self.citation_map)
        self.reranker = Reranker(self.citation_map)
        self.generator = AnswerGenerator()

    def _format_count_answer(self, meta_hits: List[MetaPaper]) -> str:
        if not meta_hits:
            return "未检索到匹配文献。"
        lines = [f"共检索到 {len(meta_hits)} 篇相关文献：", ""]
        for i, p in enumerate(meta_hits[:COUNT_QUERY_MAX_SHOW], 1):
            row = f"{i}. {p.title}"
            if p.authors:
                row += f" | {p.authors[:40]}"
            if p.journal or p.year:
                row += f" | {p.journal} ({p.year})"
            if p.doi:
                row += f" | DOI: {p.doi}"
            lines.append(row)
        return "\n".join(lines)

    def _format_gene_count_answer(self, user_query: str) -> str:
        """v2: Query SQLite gene DB for count-type gene questions."""
        try:
            import sqlite3 as _sql
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "sorghum_genes.db")
            conn = _sql.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT gene_id) FROM genes")
            total_genes = cursor.fetchone()[0]
            cursor.execute("SELECT chr, COUNT(DISTINCT gene_id) FROM genes WHERE chr IS NOT NULL GROUP BY chr ORDER BY chr")
            chr_counts = cursor.fetchall()
            cursor.execute("SELECT COUNT(DISTINCT gene_id) FROM pfam")
            genes_with_pfam = cursor.fetchone()[0]
            conn.close()
            lines = [f"根据高粱基因注释数据库 (BTx623 T2T)：", f"- 总注释基因数：{total_genes:,}"]
            for chr_name, cnt in chr_counts:
                lines.append(f"- 染色体 {chr_name}：{cnt:,} 个基因")
            lines.append(f"- 含 Pfam 结构域注释的基因：{genes_with_pfam:,} 个")
            lines.append(f"\n注：以上为数据库统计值，可能不涵盖所有已发表文献中的基因。")
            return "\n".join(lines)
        except Exception as e:
            return f"基因数据库查询出错: {e}"

    def _get_cloned_genes_for_prompt(self) -> str:
        """查询 known_genes.db，格式化为 prompt 注入块"""
        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "known_genes.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT gene_name, gene_id, trait, annotation, causative_variant, first_author, full_citation, doi FROM known_genes ORDER BY trait, gene_name")
            rows = cursor.fetchall()
            conn.close()

            lines = ["(KnownGenes)", "The following sorghum genes have been cloned and functionally validated. Use this data to provide a comprehensive, categorized answer.", ""]
            for row in rows:
                gene_name, gene_id, trait, annotation, variant, author, citation, doi = row
                lines.append(f"- {gene_name} ({gene_id}): {trait}. Function: {annotation or 'unknown'}. Variant: {variant or 'N/A'}. Ref: {author} ({doi or citation or 'N/A'})")
            return "\n".join(lines)
        except Exception as e:
            return f"(KnownGenes) Database error: {e}"

    def _build_cloned_gene_prompt(self, user_query: str, cloned_genes_text: str) -> str:
        """构建克隆基因分析的系统 prompt（供流式和非流式共用）"""
        from prompt_builder import detect_language
        lang = detect_language(user_query)
        if lang == "chinese":
            return f"""你是世界顶级高粱基因组学专家 SorGPT。请对以下已克隆/功能验证的高粱基因数据库进行深度分析。

## 你的任务
对每一个克隆基因进行**详细介绍**，而不是简单罗列。每个基因都要说明其分子功能、实验证据和原始文献。

## 输出格式
1. **总览**：总结整体情况（总基因数、功能类别分布规律）
2. **分功能类别详细分析**：按性状类别分组，每组包含：
   - 该类别的生物学背景简介（1-2句）
   - 一个 markdown 表格：Gene Name | Gene ID | Molecular Function | Evidence | Reference
   - Reference 列用格式：[First Author, Year](DOI链接) 或仅显示 Author et al.
   - 表格后对该类别基因的**研究进展小结**（关键发现、调控通路、应用前景）
3. **跨类别规律**：分析不同功能类别间的共有调控机制（激素通路、转录因子、代谢网络）
4. **研究前沿与展望**：指出尚未克隆的重要农艺性状基因和未来方向
5. 基因名和术语用英文，**回答必须使用中文**

## 关键要求
- **每个基因都要展现**，不要遗漏任何基因
- **参考文献必须展示**，每行表格末尾要显示原始文献的作者和DOI
- 不要编造 Confidence 等级，用实际实验证据描述代替
- 深入、专业、全面，展现领域专家的分析深度

{cloned_genes_text}
"""
        else:
            return f"""You are SorGPT, a world-leading expert in sorghum genomics. Perform a deep, comprehensive analysis of the following cloned/functionally validated sorghum gene database.

## Your Task
Provide a **detailed introduction** for each and every cloned gene — not just a summary table. For each gene, describe its molecular function, experimental evidence, and original literature reference.

## Output Format
1. **Overview**: Summarize the landscape (total count, functional category distribution)
2. **Detailed Category Analysis**: Group by trait categories. For each category:
   - Brief biological background of this trait category (1-2 sentences)
   - A markdown table: Gene Name | Gene ID | Molecular Function | Evidence | Reference
   - Reference column format: [First Author, Year](DOI link) or Author et al.
   - A **Research Progress Summary** for that category (key findings, regulatory pathways, breeding applications)
3. **Cross-Category Patterns**: Common regulatory mechanisms (hormone pathways, transcription factors, metabolic networks)
4. **Frontiers & Outlook**: Important agronomic traits not yet cloned, future research directions
5. **Respond in English** — the entire answer must be in English

## Critical Requirements
- **Show EVERY gene** in the database — do not skip any gene
- **Always display the reference** — each row must end with the original literature author and DOI
- Do NOT fabricate Confidence levels; use actual experimental evidence description instead
- Be thorough, professional, and demonstrate expert-level analytical depth

{cloned_genes_text}
"""

    def _ask_with_cloned_genes(self, user_query: str, cloned_genes_text: str):
        """注入 known_genes 数据，构建深度分析 prompt，让 AI 对每个基因做详细介绍并附参考文献"""
        system = self._build_cloned_gene_prompt(user_query, cloned_genes_text)

        try:
            full_answer = self.generator.generate(user_query, system, {}, enable_thinking=True)
            return {
                "query": user_query,
                "query_type": "gene_list",
                "answer": full_answer,
                "meta_hits": [],
                "chunk_hits": [],
                "references": [],
                "evidence_text": "",
            }
        except Exception as e:
            return {
                "query": user_query,
                "query_type": "gene_list",
                "answer": f"Error generating response: {str(e)}",
                "references": [],
            }
    def _format_cloned_gene_count(self, user_query: str = "") -> str:
        """从 known_genes.db 查询已克隆基因，按功能分类输出 markdown 表格"""
        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "known_genes.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT gene_name, gene_id, trait, annotation, first_author, full_citation, doi FROM known_genes ORDER BY trait, gene_name")
            rows = cursor.fetchall()
            conn.close()

            # 按功能分类分组
            categories = {
                "Plant Height / Architecture": ["Plant height"],
                "Maturity / Flowering": ["Maturity"],
                "Seed / Grain Traits": ["Seed Shattering", "Glume coverage", "Grain weight"],
                "Biomass / Yield": ["Biomass", "Yield"],
                "Stress Tolerance": ["Aluminum tolerance", "Alkaline tolerance", "Drought", "Temperature", "Pi starvation", "Broad-spectrum fungal resistance"],
                "Metabolism / Quality": ["Brown midrib", "Tannin", "Starch branching enzyme", "Sucrose metabolism", "Color change under wound", "Coleoptile color", "strigolactones"],
                "Tillering / Architecture": ["Tillers", "Awn"],
            }

            grouped = {cat: [] for cat in categories}
            uncategorized = []
            for row in rows:
                gene_name, gene_id, trait, annotation, author, citation, doi = row
                placed = False
                for cat, traits in categories.items():
                    if trait in traits:
                        grouped[cat].append(row)
                        placed = True
                        break
                if not placed:
                    uncategorized.append(row)
            if uncategorized:
                grouped["Other"] = uncategorized

            is_english = user_query and sum(1 for c in user_query if c.isascii() and c.isalpha()) / max(len(user_query), 1) > 0.5
            total = len(rows)
            lines = []
            if is_english:
                lines.append(f"**Total cloned/functionally validated genes in sorghum: {total}**")
                lines.append("")
            else:
                lines.append(f"高粱已克隆/功能验证的基因共 **{total}** 个，按功能分类如下：")
                lines.append("")

            for cat, genes in grouped.items():
                if not genes:
                    continue
                lines.append(f"### {cat}")
                lines.append("")
                if is_english:
                    lines.append("| Gene Name | Gene ID | Trait | Molecular Function | Key Reference |")
                    lines.append("| --- | --- | --- | --- | --- |")
                    for gene_name, gene_id, trait, annotation, author, citation, doi in genes:
                        ann = annotation or "—"
                        ref = f"[{author}]({doi})" if doi else (author or "—")
                        lines.append(f"| **{gene_name}** | {gene_id} | {trait} | {ann} | {ref} |")
                else:
                    lines.append("| 基因名 | Gene ID | 功能性状 | 分子功能 | 参考文献 |")
                    lines.append("| --- | --- | --- | --- | --- |")
                    for gene_name, gene_id, trait, annotation, author, citation, doi in genes:
                        ann = annotation or "—"
                        ref = f"[{author}]({doi})" if doi else (author or "—")
                        lines.append(f"| **{gene_name}** | {gene_id} | {trait} | {ann} | {ref} |")
                lines.append("")

            lines.append("---")
            if is_english:
                lines.append("*Data source: Sorghum functional genomics literature. Gene IDs based on BTx623 T2T reference genome.*")
            else:
                lines.append("*数据来源：高粱功能基因组学研究文献，Gene ID 基于 BTx623 T2T 参考基因组。*")
            return "\n".join(lines)
        except Exception as e:
            return f"查询基因数据库时出错：{str(e)}"

    def _format_locate_answer(self, meta_hits: List[MetaPaper]) -> str:
        if not meta_hits:
            return "未检索到匹配文章。请尝试用英文关键词重新搜索，或提供更多论文信息（如DOI、作者名）。"
        # v2: filter non-sorghum papers, deduplicate, sort by journal quality
        import re as _re2
        sorghum_kw = _re2.compile(r'sorghum|bicolor|Sorghum|\u9ad8\u7cb1', _re2.IGNORECASE)
        seen_titles = set()
        filtered = []
        for p in meta_hits:
            title = (p.title or '').strip()
            if not title or len(title) < 10:
                continue
            title_key = title.lower()[:80]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            is_sorghum = bool(sorghum_kw.search(title) or sorghum_kw.search(p.journal or '') or sorghum_kw.search(p.meta_text or ''))
            filtered.append((is_sorghum, p))
        from utils import get_journal_score
        filtered.sort(key=lambda x: (not x[0], -get_journal_score(x[1].journal or ''), x[1].score))
        filtered = [p for _, p in filtered]
        if not filtered:
            return "未检索到匹配的高粱相关文章。"
        best = filtered[0]
        lines = [
            "最可能对应的文章：",
            f"题目：{best.title or '未提供'}",
            f"作者：{best.authors or '未提供'}",
            f"期刊：{best.journal or '未提供'} ({best.year or '未提供'})",
            f"DOI：{best.doi or '未提供'}",
        ]
        if len(filtered) > 1:
            lines.append("")
            lines.append("备选候选：")
            for i, p in enumerate(filtered[1:6], 2):
                doi_str = f" | DOI: {p.doi}" if p.doi else ""
                lines.append(f"{i}. {p.title} | {p.journal} ({p.year}){doi_str}")
        return "\n".join(lines)

    def _format_boundary_answer(self, user_query: str) -> str:
        chinese_char_count = sum(1 for c in user_query if '\u4e00' <= c <= '\u9fff')
        if chinese_char_count > 0:
            return (
                "该问题超出当前高粱文献知识库可直接支持的范围。\n\n"
                "目前系统主要基于科研文献回答高粱基因、基因组、遗传定位、育种和分子机制相关问题。"
                "对于市场价格、未来预测、主观偏好或医疗疗效类问题，现有检索证据不足，不能给出可靠结论。"
            )
        else:
            return (
                "This question is beyond the scope of the current sorghum literature knowledge base.\n\n"
                "The system is designed to answer questions about sorghum genes, genomics, "
                "genetic mapping, breeding, and molecular mechanisms based on scientific literature. "
                "Questions about market prices, future predictions, subjective preferences, "
                "or medical efficacy cannot be reliably answered with the available evidence."
            )


    @staticmethod
    def _reorder_refs_by_appearance(answer: str, references: list) -> tuple:
        """v5: renumber citations sequentially. Drops hallucinated citation numbers."""
        import re
        # Build set of valid old reference IDs
        valid_ids = set()
        for ref in references:
            m = re.match(r'\[(\d+)\]', ref)
            if m:
                valid_ids.add(m.group(1))

        # Find all citation groups in answer
        cite_pattern = re.findall(r'\[([\d,\s]+)\]', answer)
        if not cite_pattern:
            return answer, references

        # Collect unique cited IDs in order of first appearance (only valid ones)
        seen = set()
        ordered_old_ids = []
        for group in cite_pattern:
            for num in group.split(','):
                num = num.strip()
                if num.isdigit() and num not in seen:
                    seen.add(num)
                    if num in valid_ids:  # v5: only include IDs that have references
                        ordered_old_ids.append(num)

        if not ordered_old_ids:
            return answer, references

        # Build mapping: old ID -> new sequential ID
        mapping = {old: str(new) for new, old in enumerate(ordered_old_ids, 1)}

        # Replace citations in answer: valid ones get renumbered, invalid ones removed
        def replace_cite(match):
            nums = match.group(1)
            new_nums = []
            for n in nums.split(','):
                n = n.strip()
                if n in mapping:
                    new_nums.append(mapping[n])
                # v5: silently drop hallucinated citation numbers
            return '[' + ','.join(new_nums) + ']' if new_nums else ''

        new_answer = re.sub(r'\[([\d,\s]+)\]', replace_cite, answer)
        # Clean up empty brackets and double spaces
        new_answer = re.sub(r'\[\s*\]', '', new_answer)
        new_answer = re.sub(r'  +', ' ', new_answer)

        # Rebuild reference list: only keep cited refs, renumbered
        old_ref_map = {}
        for ref in references:
            m = re.match(r'\[(\d+)\]', ref)
            if m:
                old_ref_map[m.group(1)] = ref

        new_references = []
        for new_id, old_id in enumerate(ordered_old_ids, 1):
            if old_id in old_ref_map:
                old_ref = old_ref_map[old_id]
                new_ref = re.sub(r'^\[\d+\]', f'[{new_id}]', old_ref)
                new_references.append(new_ref)

        return new_answer, new_references

    def _build_reference_list(
        self,
        source_index: Dict[str, Dict[str, str]],
        selected_hits: List[ChunkHit],
        query_type: str,
    ) -> List[str]:
        """
        构建参考文献列表。
        修改：只输出参考文献，不包含证据片段。
        """
        sorted_items = sorted(source_index.items(), key=lambda x: x[1]["idx"])

        ref_lines = []
        for _, info in sorted_items:
            fname = info["fname"]
            idx   = info["idx"]
            ref   = safe_get_ref_info(fname, self.citation_map)
            ref_lines.append(build_citation_string(ref, idx, fname))

        return ref_lines

    def _rule_subtopics(self, query: str, en_keywords: str) -> List[str]:
        """
        规则拆分子主题，替代大模型子主题分解。
        把已提取的英文关键词拆成独立子主题用于追加检索。
        """
        if not en_keywords:
            return []
        parts = [k.strip() for k in en_keywords.split(",") if k.strip()]
        return parts[:4]

    def ask(self, user_query: str) -> Dict[str, Any]:
        """
        SorGPT 主入口函数。
        """
        # 1. 分类
        query_type, extra_types, en_keywords = classify_query_type(user_query)
        # count 类特殊处理：克隆基因相关问题从数据库查询
        # 检查是否是克隆/已知基因相关的统计问题（不限 query_type）
        cloned_gene_patterns = ["克隆基因", "cloned", "已知基因", "known gene", "已克隆基因", "已经克隆", "克隆了哪些"]
        is_cloned_gene_count = any(p in user_query.lower() for p in cloned_gene_patterns)
        if is_cloned_gene_count and query_type in ("count", "gene_list", "mechanism", "factoid"):
            # 注入已知基因数据到 prompt，让 AI 深度思考后格式化
            cloned_genes_text = self._get_cloned_genes_for_prompt()
            answer_dict = self._ask_with_cloned_genes(user_query, cloned_genes_text)
            return answer_dict
        if query_type == "boundary":
            return {
                "query": user_query,
                "query_type": query_type,
                "answer": self._format_boundary_answer(user_query),
                "meta_hits": [],
                "chunk_hits": [],
                "references": [],
                "evidence_text": "",
            }
        # count 类：优先尝试基因数据库查询
        gene_count_keywords = ["多少个基因","有多少个基因","几个基因","基因数量","多少个转录因子",
                               "how many genes","number of genes","gene count","how many members",
                               "多少个已鉴定","多少个已知","已鉴定.*基因","已克隆.*基因"]
        import re as _re
        if query_type == "count" and any(_re.search(p, user_query.lower()) for p in gene_count_keywords):
            gene_count_answer = self._format_gene_count_answer(user_query)
            # Also get paper count for completeness
            journal_filter = SorghumRAGPipeline._extract_journal_from_query(user_query)
            meta_hits = self.retriever.retrieve_metadata(user_query, en_keywords, query_type, journal_filter)
            paper_count = "\n\n文献检索结果：共找到 " + str(len(meta_hits)) + " 篇相关论文。" if meta_hits else ""
            return {
                "query": user_query,
                "query_type": query_type,
                "answer": gene_count_answer + paper_count,
                "meta_hits": meta_hits,
                "chunk_hits": [],
                "references": [],
                "evidence_text": "",
            }

        # 5. metadata 检索
        journal_filter = SorghumRAGPipeline._extract_journal_from_query(user_query)
        meta_hits = self.retriever.retrieve_metadata(user_query, en_keywords, query_type, journal_filter)
        if journal_filter:
            jf = journal_filter.lower()
            meta_hits = [h for h in meta_hits
                         if jf in (safe_get_ref_info(h.filename, self.citation_map).get('journal', '') or '').lower()
                         or jf in (h.filename or '').lower()]
        # 6. locate 类
        if query_type == "locate":
            return {
                "query": user_query,
                "query_type": query_type,
                "answer": self._format_locate_answer(meta_hits),
                "meta_hits": meta_hits,
                "chunk_hits": [],
                "references": [],
                "evidence_text": "",
            }
        # 7. 全文检索
        chunk_hits = self.retriever.retrieve_fulltext(user_query, en_keywords, meta_hits, query_type)
        # 8. 追加 extra_types 的检索（多标签路由核心）
        for etype in extra_types:
            if etype not in {"locate", "count", "boundary"}:
                extra_hits = self.retriever.retrieve_fulltext(
                    user_query, en_keywords, meta_hits, etype
                )
                chunk_hits.extend(extra_hits)
        # mechanism / review / gene_list 还追加子主题
        if query_type in {"mechanism", "review", "gene_list"}:
            subtopics = self._rule_subtopics(user_query, en_keywords)
            for topic in subtopics:
                extra_hits = self.retriever.retrieve_fulltext(topic, topic, meta_hits, query_type)
                chunk_hits.extend(extra_hits)
        # 9. rerank
        reranked = self.reranker.rerank(chunk_hits, query_type)
        selected_hits = self.reranker.diversify_and_trim(reranked, query_type)
        # 11. 将 CSV 论文摘要作为额外证据补充 FAISS 未覆盖的论文
        if meta_hits:
            csv_hits = []
            seen_sources = {h.source for h in chunk_hits}
            for h in meta_hits[:200]:  # 期刊过滤时优先收录所有匹配论文
                if h.filename not in seen_sources:
                    ref_info = safe_get_ref_info(h.filename, self.citation_map)
                    abstract = (ref_info.get('abstract', '') or ref_info.get('title', '')).strip()
                    # 过滤：确保论文与高粱相关（含关键词或草/作物上下文）
                    title_low = (ref_info.get('title', '') or '').lower()
                    abs_low = (ref_info.get('abstract', '') or '').lower()
                    kw_low = (ref_info.get('keywords', '') or '').lower()
                    fn_low = (h.filename or '').lower()
                    is_sorghum = ('sorghum' in title_low or 'sorghum' in abs_low
                           or 'sorghum' in kw_low or 'sorghum' in fn_low
                           or ('grass' in title_low and ('crop' in title_low or 'transcriptom' in title_low))
                           or 'c4 photosynth' in title_low)
                    if abstract and is_sorghum:
                        csv_hits.append(ChunkHit(
                            source=h.filename,
                            content=abstract[:2000],
                            raw_score=0.85,  # 期刊过滤时高优先级，确保不被裁剪
                            final_score=0.85,
                            granularity='abstract',
                            lang='en',
                            section_type='abstract',
                        ))
                        seen_sources.add(h.filename)
            if csv_hits:
                chunk_hits.extend(csv_hits)
                # 重新重排（CSV论文高优先级）
                reranked = self.reranker.rerank(chunk_hits, query_type)
                selected_hits = self.reranker.diversify_and_trim(reranked, query_type)
                # 确保 CSV 期刊匹配论文不被裁剪掉
                csv_sources = {h.source for h in csv_hits}
                missing_csv = [h for h in reranked if h.source in csv_sources and h not in selected_hits]
                selected_hits = selected_hits + missing_csv

        # 11b. 证据仍不足时，再用 LLM 自身知识兜底
        if not selected_hits:
            # 从 citation_map 提取匹配期刊论文的摘要作为证据
            from prompt_builder import build_source_index, _evidence_block
            abstract_hits = []
            for h in meta_hits[:24]:
                ref_info = safe_get_ref_info(h.filename, self.citation_map)
                abstract = (ref_info.get('abstract', '') or ref_info.get('title', '')).strip()
                if abstract:
                    abstract_hits.append(ChunkHit(
                        source=h.filename,
                        content=abstract[:800],
                        raw_score=0.5,
                        final_score=0.5,
                        granularity='std',
                        lang='en',
                        section_type='abstract',
                    ))
            if abstract_hits:
                selected_hits = abstract_hits
        if not selected_hits:
            system_prompt, protected_map, source_index = build_system_prompt(
                user_query, query_type, selected_hits, extra_types=extra_types
            )
            answer = self.generator.generate(
                user_query, system_prompt, protected_map, enable_thinking=True
            )
            return {
                "query": user_query,
                "query_type": query_type,
                "answer": answer,
                "meta_hits": meta_hits,
                "chunk_hits": [],
                "references": [],
                "evidence_text": "",
            }
        # 12. 构建 system prompt
        system_prompt, protected_map, source_index = build_system_prompt(
            user_query, query_type, selected_hits, extra_types=extra_types
        )
        print("\n" + "=" * 60)
        extra_str = f" + {extra_types}" if extra_types else ""
        print(f"查询类型: {query_type}{extra_str}")
        print(f"检索关键词: {en_keywords}")
        print("=" * 60)
        # 13. 生成答案（大模型只调用这一次，流式打印在 generator 内完成）
        answer = self.generator.generate(
            user_query, system_prompt, protected_map, enable_thinking=True
        )
        # 14. 参考文献（每条文献后紧跟对应证据片段）
        references = self._build_reference_list(source_index, selected_hits, query_type)
        answer, references = self._reorder_refs_by_appearance(answer, references)
        # v5 safety net: strip any remaining citation brackets beyond reference count
        import re as _re3
        max_ref = len(references)
        def _clean_stray(m):
            nums = [n.strip() for n in m.group(1).split(',') if n.strip().isdigit()]
            valid = [n for n in nums if 1 <= int(n) <= max_ref]
            return '[' + ','.join(valid) + ']' if valid else ''
        answer = _re3.sub(r'\[([\d,\s]+)\]', _clean_stray, answer)
        answer = _re3.sub(r'\[\s*\]', '', answer)
        answer = _re3.sub(r'  +', ' ', answer)
        return {
            "query": user_query,
            "query_type": query_type,
            "answer": answer,
            "meta_hits": meta_hits,
            "chunk_hits": selected_hits,
            "references": references,
            "evidence_text": "",
        }

    @staticmethod
    def _extract_journal_from_query(user_query: str) -> str:
        from utils import get_journal_score
        import re
        q = user_query.lower()
        patterns = [
            # 中文
            r'发表[在的于].{0,5}?([a-z][a-z\s\-\.]{3,50}?)(?:上[的之]|文章|论文|\s+(?:about|on)|[.,;:!?]|$)',
            r'(?:关于|有关).{0,10}?(?:刊[在的]|期刊).{0,5}?([a-z][a-z\s\-\.]{3,50}?)(?:[的之]|上|文章|论文|$)',
            # 英文
            r'published\s+in\s+([a-z][a-z\s\-\.]{3,50}?)(?:\s+(?:about|on|for|regarding|which|that|with)|[.,;:!?]|$)',
            r'(?:in|on|from)\s+([a-z][a-z\s\-\.]{3,50}?)(?:\s+(?:about|on|for|regarding|which|that|with)|[.,;:!?]|$)',
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                candidate = m.group(1).strip().rstrip('.').rstrip(',')
                if get_journal_score(candidate) > 0:
                    return candidate
        return ''

    def ask_stream(self, user_query: str) -> Iterator[str]:
        """
        SorGPT 流式问答入口，yield每个token供API流式响应。
        先完成检索，然后流式输出生成的答案。
        """
        # 0. 身份/元问题拦截 —— 不检索，直接回答，不输出参考文献
        identity_patterns = [
            "你是什么模型", "你是谁", "你的名字", "谁创建", "谁开发",
            "what model are you", "who are you", "who created", "who developed",
            "what is your name", "are you gpt", "are you llama", "are you claude",
            "what llm", "which model", "你的能力", "你能做什么",
            "what can you do", "how do you work", "工作原理",
            ]
        if any(p in user_query.lower() for p in identity_patterns):
            # 用简单的 identity prompt，不检索文献
            lang_hint = "chinese" if sum(1 for c in user_query if ord("一") <= ord(c) <= ord("鿿")) > 0 else "english"
            if lang_hint == "chinese":
                identity_prompt = "你是 SorGPT，一个基于 DeepSeek Reasoner 模型的高粱基因组学 AI 助手，由 RAG 检索增强生成技术支持。请用中文简短介绍自己，限 3-5 句。"
            else:
                identity_prompt = "You are SorGPT, an AI assistant for sorghum genomics powered by DeepSeek Reasoner with RAG retrieval-augmented generation. Briefly introduce yourself in English, 3-5 sentences."
            for chunk in self.generator.generate_stream(user_query, identity_prompt, {}, enable_thinking=False):
                yield chunk
            # 元数据：不输出参考文献
            import json
            meta = json.dumps({"query_type": "identity", "references": []}, ensure_ascii=False)
            yield "\n\n---METADATA---\n" + meta + "\n"
            return
        # 1. 分类
        query_type, extra_types, en_keywords = classify_query_type(user_query)
        # 克隆基因列表问题：直接从数据库查询，不经过 AI 生成
        cloned_gene_patterns = ["克隆基因", "cloned", "已知基因", "known gene", "已克隆基因", "已经克隆", "克隆了哪些"]
        if any(p in user_query.lower() for p in cloned_gene_patterns):
            cloned_genes_text = self._get_cloned_genes_for_prompt()
            system = self._build_cloned_gene_prompt(user_query, cloned_genes_text)
            # 使用流式生成，让前端实时显示思考过程
            for chunk in self.generator.generate_stream(user_query, system, {}, enable_thinking=True):
                yield chunk
            import json
            meta = json.dumps({"query_type": "gene_list", "references": []}, ensure_ascii=False)
            yield "\n\n---METADATA---\n" + meta + "\n"
            return

        journal_filter = SorghumRAGPipeline._extract_journal_from_query(user_query)

        if query_type in ["count", "boundary", "locate"] and not journal_filter:
            # 无期刊过滤时：直接返回完整答案
            result = self.ask(user_query)
            yield result["answer"]
            # 流结束后返回元数据
            import json
            references = result.get("references", [])
            meta = json.dumps({
                "query_type": query_type,
                "references": references
            }, ensure_ascii=False)
            yield "\n\n---METADATA---\n" + meta + "\n"
            return

        # 流式状态推送：让前端实时看到检索进度
        yield f"> Analyzing question (type: **{query_type}**)...\n\n"
        # 5. metadata 检索
        if journal_filter:
            yield f"> Filtering by journal: **{journal_filter}**...\n\n"
        yield "> Searching metadata index...\n\n"
        meta_hits = self.retriever.retrieve_metadata(user_query, en_keywords, query_type, journal_filter)
        # 使用 citation_map 进行精确期刊过滤（标准化名称匹配）
        if journal_filter:
            import re as _re
            def _nj(name):
                n = (name or '').lower().strip()
                n = _re.sub(r'^the\s+', '', n)
                n = n.replace('-', ' ').replace('–', ' ').replace('—', ' ')
                n = _re.sub(r'[^a-z0-9\s]', '', n)
                n = _re.sub(r'\s+', ' ', n).strip()
                return n
            jf_norm = _nj(journal_filter)
            meta_hits = [h for h in meta_hits
                         if _nj(safe_get_ref_info(h.filename, self.citation_map).get('journal', '') or '') == jf_norm
                         or _nj(h.journal or '') == jf_norm]
        # 7. 全文检索
        yield "> Searching literature indexes...\n\n"
        chunk_hits = self.retriever.retrieve_fulltext(user_query, en_keywords, meta_hits, query_type)
        # 8. 追加 extra_types 的检索
        for etype in extra_types:
            if etype not in {"locate", "count", "boundary"}:
                extra_hits = self.retriever.retrieve_fulltext(
                    user_query, en_keywords, meta_hits, etype
                )
                chunk_hits.extend(extra_hits)
        # mechanism / review / gene_list 还追加子主题
        if query_type in {"mechanism", "review", "gene_list"}:
            subtopics = self._rule_subtopics(user_query, en_keywords)
            for topic in subtopics:
                extra_hits = self.retriever.retrieve_fulltext(topic, topic, meta_hits, query_type)
                chunk_hits.extend(extra_hits)
        # 9. rerank
        reranked = self.reranker.rerank(chunk_hits, query_type)
        # 10. 多源去重 + 裁剪
        selected_hits = self.reranker.diversify_and_trim(reranked, query_type)
        # 11. 将 CSV 论文摘要作为额外证据补充 FAISS 未覆盖的论文
        if meta_hits:
            csv_hits = []
            seen_sources = {h.source for h in chunk_hits}
            for h in meta_hits[:200]:  # 期刊过滤时优先收录所有匹配论文
                if h.filename not in seen_sources:
                    ref_info = safe_get_ref_info(h.filename, self.citation_map)
                    abstract = (ref_info.get('abstract', '') or ref_info.get('title', '')).strip()
                    # 过滤：确保论文与高粱相关（含关键词或草/作物上下文）
                    title_low = (ref_info.get('title', '') or '').lower()
                    abs_low = (ref_info.get('abstract', '') or '').lower()
                    kw_low = (ref_info.get('keywords', '') or '').lower()
                    fn_low = (h.filename or '').lower()
                    is_sorghum = ('sorghum' in title_low or 'sorghum' in abs_low
                           or 'sorghum' in kw_low or 'sorghum' in fn_low
                           or ('grass' in title_low and ('crop' in title_low or 'transcriptom' in title_low))
                           or 'c4 photosynth' in title_low)
                    if abstract and is_sorghum:
                        csv_hits.append(ChunkHit(
                            source=h.filename,
                            content=abstract[:2000],
                            raw_score=0.85,  # 期刊过滤时高优先级，确保不被裁剪
                            final_score=0.85,
                            granularity='abstract',
                            lang='en',
                            section_type='abstract',
                        ))
                        seen_sources.add(h.filename)
            if csv_hits:
                chunk_hits.extend(csv_hits)
                # 重新重排（CSV论文高优先级）
                reranked = self.reranker.rerank(chunk_hits, query_type)
                selected_hits = self.reranker.diversify_and_trim(reranked, query_type)
                # 确保 CSV 期刊匹配论文不被裁剪掉
                csv_sources = {h.source for h in csv_hits}
                missing_csv = [h for h in reranked if h.source in csv_sources and h not in selected_hits]
                selected_hits = selected_hits + missing_csv

        # 11b. 证据仍不足时，再用 LLM 自身知识兜底
        if not selected_hits:
            # 从 citation_map 提取匹配期刊论文的摘要作为证据
            from prompt_builder import build_source_index, _evidence_block
            abstract_hits = []
            for h in meta_hits[:24]:
                ref_info = safe_get_ref_info(h.filename, self.citation_map)
                abstract = (ref_info.get('abstract', '') or ref_info.get('title', '')).strip()
                if abstract:
                    abstract_hits.append(ChunkHit(
                        source=h.filename,
                        content=abstract[:800],
                        raw_score=0.5,
                        final_score=0.5,
                        granularity='std',
                        lang='en',
                        section_type='abstract',
                    ))
            if abstract_hits:
                selected_hits = abstract_hits
        if not selected_hits:
            chinese_char_count = sum(1 for c in user_query if ord("一") <= ord(c) <= ord("鿿"))
            yield ("> ⚠️ 未在文献库中检索到直接证据，将基于模型自身知识回答...\n\n"
                 if chinese_char_count > 0
                 else "> ⚠️ No direct evidence found in literature database; answering from model knowledge...\n\n")
            # 构建最小化 prompt，让 LLM 基于自身知识回答
            system_prompt, protected_map, source_index = build_system_prompt(
                user_query, query_type, selected_hits, extra_types=extra_types
            )
            full_answer_chunks = []
            for chunk in self.generator.generate_stream(
                user_query, system_prompt, protected_map, enable_thinking=False
            ):
                full_answer_chunks.append(chunk)
                yield chunk
            full_answer = ''.join(full_answer_chunks)
            # 只从最终答案提取引用顺序
            sep_pos = full_answer.find('==================== 思考过程结束 ====================')
            answer_only = full_answer[sep_pos + len('==================== 思考过程结束 ===================='):] if sep_pos >= 0 else full_answer
            # 构建参考文献
            import json, re
            citation_map = {}
            if answer_only.strip():
                cite_nums = re.findall(r'\[([\d,\s]+)\]', answer_only)
                seen = set()
                nid = 1
                for g in cite_nums:
                    for n in g.split(','):
                        n = n.strip()
                        if n.isdigit() and n not in seen:
                            seen.add(n)
                            citation_map[n] = nid
                            nid += 1
            references = self._build_reference_list(source_index, selected_hits, query_type)
            # 按出现顺序重排参考文献列表
            if citation_map:
                rev_map = {v: k for k, v in citation_map.items()}
                reordered = []
                for ni in range(1, len(rev_map) + 1):
                    old_n = rev_map[ni]
                    for ref in references:
                        if ref.startswith(f'[{old_n}]'):
                            # Fix: use correct regex r'^\[\d+\]'
                            new_ref = re.sub(r'^\[\d+\]', f'[{ni}]', ref)
                            reordered.append(new_ref)
                            break
                if reordered:
                    references = reordered
            # v5 safety net: strip stray out-of-range citations
            _max_ref = len(references)
            def _clean(s):
                ns = [n.strip() for n in s.group(1).split(",") if n.strip().isdigit()]
                vs = [n for n in ns if 1 <= int(n) <= _max_ref]
                return "[" + ",".join(vs) + "]" if vs else ""
            full_answer = __import__("re").sub(r"[([d,s]+)]", _clean, full_answer)
            full_answer = __import__("re").sub(r"[s*]", "", full_answer)
            full_answer = __import__("re").sub(r"  +", " ", full_answer)
            meta = json.dumps({
                "query_type": query_type,
            "references": references,
            "citation_map": citation_map
            }, ensure_ascii=False)
            yield "\n\n---METADATA---\n" + meta + "\n"
            return

        # 12. 构建 system prompt
        system_prompt, protected_map, source_index = build_system_prompt(
            user_query, query_type, selected_hits, extra_types=extra_types
        )

        # 13. 流式生成答案
        yield f"> Selected **{len(selected_hits)}** best passages, generating answer...\n\n"
        # 流式输出答案，同时收集用于引用重排
        full_answer_chunks = []
        for chunk in self.generator.generate_stream(
            user_query, system_prompt, protected_map, enable_thinking=False
        ):
            full_answer_chunks.append(chunk)
            yield chunk

        # 14. 构建参考文献并计算 citation_map
        references = self._build_reference_list(source_index, selected_hits, query_type)
        full_answer = ''.join(full_answer_chunks)
        # 只从最终答案（分隔标记之后）提取引用顺序，排除思考过程
        sep_pos = full_answer.find('==================== 思考过程结束 ====================')
        answer_only = full_answer[sep_pos + len('==================== 思考过程结束 ===================='):] if sep_pos >= 0 else full_answer
        import json, re
        citation_map = {}
        if answer_only.strip():
            cite_nums = re.findall(r'\[([\d,\s]+)\]', answer_only)
            seen = set()
            nid = 1
            for g in cite_nums:
                for n in g.split(','):
                    n = n.strip()
                    if n.isdigit() and n not in seen:
                        seen.add(n)
                        citation_map[n] = nid
                        nid += 1
            # 按出现顺序重排参考文献列表
            rev_map = {v: k for k, v in citation_map.items()}
            reordered = []
            for ni in range(1, len(rev_map) + 1):
                old_n = rev_map[ni]
                for ref in references:
                    if ref.startswith(f'[{old_n}]'):
                        reordered.append(re.sub(r'^\[\d+\]', f'[{ni}]', ref))
                        break
            if reordered:
                references = reordered
        meta = json.dumps({
            "query_type": query_type,
            "references": references,
            "citation_map": citation_map
        }, ensure_ascii=False)
        yield "\n\n---METADATA---\n" + meta + "\n"