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
from typing import Dict, Any, List, Iterator, Tuple, Optional, Set
from config import CSV_PATHS, REFERENCE_LIMITS, COUNT_QUERY_MAX_SHOW
from embeddings import BgeEmbeddingsWrapper
from metadata_loader import load_citation_map, safe_get_ref_info
from query_classifier import classify_query_type
from retriever import Retriever, MetaPaper, ChunkHit
from reranker import Reranker
from prompt_builder import build_system_prompt
from generator import AnswerGenerator
from sequence_fetcher import resolve_genes_from_query, build_sequence_blocks, detect_seq_type
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



    def _format_cloned_gene_count(self) -> str:
        """从 known_genes.db 查询已克隆/已知功能基因的数量"""
        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "known_genes.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 查询总数
            cursor.execute("SELECT COUNT(*) FROM known_genes")
            total_count = cursor.fetchone()[0]
            
            # 查询所有基因详情
            cursor.execute("SELECT gene_name, gene_id, trait FROM known_genes ORDER BY gene_name")
            genes = cursor.fetchall()
            conn.close()
            
            # 构建回答
            lines = [
                f"高粱已克隆/功能验证的基因共 <strong>{total_count}</strong> 个：",
                "",
                "以下是目前已克隆的主要基因列表：",
                "",
            ]
            
            for i, (gene_name, gene_id, trait) in enumerate(genes, 1):
                trait_str = trait if trait else "功能未知"
                lines.append(f"{i}. {gene_name} ({gene_id}) - {trait_str}")
            
            lines.extend([
                "",
                "注：数据来源于高粱功能基因组学研究文献，随着研究进展可能有所更新。",
            ])
            return "\n".join(lines)
        except Exception as e:
            return f"查询基因数据库时出错：{str(e)}"

    def _get_cloned_genes_for_prompt(self) -> str:
        try:
            import sqlite3
            dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "known_genes.db")
            conn = sqlite3.connect(dp)
            rows = conn.execute("SELECT gene_name,gene_id,trait,annotation,causative_variant,first_author,full_citation,doi FROM known_genes ORDER BY trait,gene_name").fetchall()
            conn.close()
            lines = ["(KnownGenes)", "Cloned/validated sorghum genes:", ""]
            for r in rows:
                gn,gid,tr,an,va,au,ci,doi = r
                dl = f"https://doi.org/{doi}" if doi and doi.strip() else (ci or "N/A")
                lines.append(f"- {gn} ({gid}): {tr}. Function: {an or 'unknown'}. Variant: {va or 'N/A'}. Ref: {au} ({dl})")
            return "\n".join(lines)
        except Exception as e:
            return f"(KnownGenes) Error: {e}"

    def _build_cloned_gene_prompt(self, uq, ct):
        from prompt_builder import detect_language
        if detect_language(uq) == "chinese":
            return f"你是SorGPT，高粱AI智能问答助手。请对以下已克隆/功能验证的高粱基因数据库进行深度分析。\n\n## 你的任务\n对每一个克隆基因进行详细介绍，而不是简单罗列。每个基因都要说明其分子功能、实验证据和原始文献。\n\n## 输出格式\n1. 总览：总结整体情况\n2. 分功能类别详细分析：按性状类别分组，每组包含生物学背景、表格(Gene Name | Gene ID | Molecular Function | Evidence | Reference)和研究进展小结\n3. 跨类别规律\n4. 研究前沿与展望\n5. 基因名和术语用英文，回答必须使用中文\n\n## 关键要求\n每个基因都要展现，参考文献必须展示，不要编造Confidence等级\n\n{ct}"
        return f"You are SorGPT, a world-class expert in sorghum genomics. Perform a deep, comprehensive analysis of the following cloned/functionally validated sorghum gene database.\n\n## Your Task\nProvide a detailed introduction for each and every cloned gene.\n\n## Output Format\n1. Overview\n2. Detailed Category Analysis with tables (Gene Name | Gene ID | Molecular Function | Evidence | Reference)\n3. Cross-Category Patterns\n4. Frontiers & Outlook\n\n## Key Requirements\nCover every gene, show references, do not fabricate confidence levels\n\n{ct}"

    def _ask_with_cloned_genes(self, uq, ct):
        s = self._build_cloned_gene_prompt(uq, ct)
        a = self.generator.generate(uq, s, {}, enable_thinking=False)
        return {"query": uq, "query_type": "gene_list", "answer": a, "references": []}

    def _format_locate_answer(self, meta_hits: List[MetaPaper]) -> str:
        if not meta_hits:
            return "未检索到匹配文章。"

        # 用 citation_map 补全年份/期刊/DOI（元数据索引的 year 字段可能为空）
        def _enrich(hit: MetaPaper) -> Dict[str, str]:
            info = safe_get_ref_info(hit.filename, self.citation_map)
            year = norm_text(hit.year or info.get("year", ""))
            if year.endswith(".0"):
                year = year[:-2]
            return {
                "title": hit.title or info.get("title", "") or "未提供",
                "authors": hit.authors or info.get("authors", "") or "未提供",
                "journal": hit.journal or info.get("journal", "") or "未提供",
                "year": year or "未提供",
                "doi": hit.doi or info.get("doi", "") or "未提供",
            }

        best = _enrich(meta_hits[0])
        lines = [
            "最可能对应的文章：",
            f"题目：{best['title']}",
            f"作者：{best['authors']}",
            f"期刊：{best['journal']} ({best['year']})",
            f"DOI：{best['doi']}",
        ]
        if len(meta_hits) > 1:
            lines.append("")
            lines.append("备选候选：")
            for i, p in enumerate(meta_hits[1:4], 2):
                e = _enrich(p)
                lines.append(f"{i}. {e['title']} | {e['journal']} ({e['year']})")
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

    def _extract_cited_indices(self, answer: str) -> Optional[Set[int]]:
        """解析正文中的 [n] / [n, m] 引用编号；无引用返回 None（走全量兜底）。"""
        if not answer:
            return None
        found = set()
        for _m in re.finditer(r"\[(\d[\d,\s]*)\]", answer):
            for _part in _m.group(1).split(","):
                _part = _part.strip()
                if _part.isdigit():
                    found.add(int(_part))
        return found or None

    def _build_reference_list(
        self,
        source_index: Dict[str, Dict[str, str]],
        selected_hits: List[ChunkHit],
        query_type: str,
        cited_indices: Optional[Set[int]] = None,
    ) -> List[str]:
        """
        构建参考文献列表。只输出参考文献，不包含证据片段。
        若给定 cited_indices（正文实际 [n] 引用的编号），只保留被引用的文献；
        未给定时保留全部池子文献（兼容旧行为）。
        """
        # ref_limit removed to avoid ghost citations
        sorted_items = sorted(source_index.items(), key=lambda x: x[1]["idx"])

        # 只保留与池子真实编号相交的被引编号；交集为空（编号越界/解析异常）→ 回退全量
        idx_set = None
        if cited_indices:
            idx_set = {info["idx"] for info in source_index.values()} & cited_indices
            if not idx_set:
                idx_set = None

        ref_lines = []
        for _, info in sorted_items:
            fname = info["fname"]
            idx   = info["idx"]
            if idx_set is not None and idx not in idx_set:
                continue
            ref  = safe_get_ref_info(fname, self.citation_map)
            line = build_citation_string(ref, idx, fname)
            if line.strip():
                ref_lines.append(line)

        return ref_lines

    def _sequence_blocks(self, user_query: str) -> Tuple[str, str]:
        """sequence 类型：解析基因 → 返回 (prompt上下文, 追加序列块)。

        上下文注入 system prompt 让 LLM 序言采用正确 ID；序列块在生成后追加，
        保证输出序列真实、不被模型改写。
        """
        try:
            gene_ids = resolve_genes_from_query(user_query)
        except Exception:
            return "", ""
        if not gene_ids:
            return "", ""
        seq_type = detect_seq_type(user_query)
        lang = "chinese" if sum(1 for c in user_query if ord("一") <= ord(c) <= ord("鿿")) > 0 else "english"
        ctx_lines, blocks = [], []
        for gid in gene_ids:
            try:
                c, b = build_sequence_blocks(gid, seq_type=seq_type, lang=lang)
            except Exception:
                c, b = "", ""
            if c:
                ctx_lines.append("- " + c)
            if b:
                blocks.append(b)
        if not blocks:
            return "", ""
        ctx = ""
        if ctx_lines:
            header = "解析出的基因：\n" if lang == "chinese" else "Resolved gene(s):\n"
            ctx = header + "\n".join(ctx_lines)
        return ctx, "\n\n" + "\n\n".join(blocks)
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
        if query_type == "count":
            # 检查是否是克隆/已知基因相关的统计问题
            cloned_gene_patterns = ["克隆", "cloned", "已知基因", "known gene", "已克隆基因"]
            is_cloned_gene_count = any(p in user_query for p in cloned_gene_patterns)
            
            if is_cloned_gene_count:
                # 从 known_genes.db 查询
                answer = self._format_cloned_gene_count()
            else:
                # 原有文献统计逻辑
                meta_hits = self.retriever.retrieve_metadata(user_query, en_keywords, query_type)
                answer = self._format_count_answer(meta_hits)
            
            return {
                "query": user_query,
                "query_type": query_type,
                "answer": answer,
                "meta_hits": meta_hits if not is_cloned_gene_count else [],
                "chunk_hits": [],
                "references": [],
                "evidence_text": "",
            }
        # 3b. Cloned gene database lookup
        cloned_gene_patterns = ["克隆基因", "cloned gene", "known gene", "已克隆基因"]
        is_cloned = any(p in user_query.lower() for p in cloned_gene_patterns)
        if is_cloned and query_type in ("count", "gene_list", "mechanism", "factoid", "review", "gene_function"):
            cloned_genes_text = self._get_cloned_genes_for_prompt()
            return self._ask_with_cloned_genes(user_query, cloned_genes_text)

        # 4. boundary 类
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
        # 5. metadata 检索
        meta_hits = self.retriever.retrieve_metadata(user_query, en_keywords, query_type)
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
        # 10. 多源去重 + 裁剪
        selected_hits = self.reranker.diversify_and_trim(reranked, query_type)
        # 10b. 序列解析（sequence 类型提前解析，避免"证据不足"早退并修正 ID）
        seq_ctx, seq_block = "", ""
        if query_type == "sequence":
            seq_ctx, seq_block = self._sequence_blocks(user_query)
        # 11. 证据不足
        if not selected_hits and not seq_block:
            chinese_char_count = sum(1 for c in user_query if '\u4e00' <= c <= '\u9fff')
            no_evidence_msg = (
                "未检索到足够的全文证据，现有检索证据有限。"
                if chinese_char_count > 0
                else "Insufficient evidence retrieved from the literature database."
            )
            return {
                "query": user_query,
                "query_type": query_type,
                "answer": no_evidence_msg,
                "meta_hits": meta_hits,
                "chunk_hits": [],
                "references": [],
                "evidence_text": "",
            }
        # 12. 构建 system prompt
        system_prompt, protected_map, source_index = build_system_prompt(
            user_query, query_type, selected_hits, extra_types=extra_types, seq_context=seq_ctx
        )
        print("\n" + "=" * 60)
        extra_str = f" + {extra_types}" if extra_types else ""
        print(f"查询类型: {query_type}{extra_str}")
        print(f"检索关键词: {en_keywords}")
        print("=" * 60)
        # 13. 生成答案（大模型只调用这一次，流式打印在 generator 内完成）
        answer = self.generator.generate(
            user_query, system_prompt, protected_map, enable_thinking=False
        )
        # 13b. 序列注入（sequence 类型）
        if query_type == "sequence":
            if seq_block:
                answer = answer + seq_block
        # 14. 参考文献（只保留正文实际 [n] 引用的文献）
        cited = self._extract_cited_indices(answer)
        references = self._build_reference_list(source_index, selected_hits, query_type, cited)
        return {
            "query": user_query,
            "query_type": query_type,
            "answer": answer,
            "meta_hits": meta_hits,
            "chunk_hits": selected_hits,
            "references": references,
            "evidence_text": "",
        }

    def ask_stream(self, user_query: str) -> Iterator[str]:
        """
        SorGPT 流式问答入口，yield每个token供API流式响应。
        先完成检索，然后流式输出生成的答案。
        """
        # 1. 分类
        query_type, extra_types, en_keywords = classify_query_type(user_query)
        if query_type in ["count", "boundary", "locate"]:
            # 这些类型直接返回完整答案，不需要流式生成
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

        # 4b. 克隆基因数据库查询
        cloned_gene_patterns = ["克隆基因", "cloned gene", "known gene", "已克隆基因"]
        is_cloned = any(p in user_query.lower() for p in cloned_gene_patterns)
        if is_cloned and query_type in ("count", "gene_list", "mechanism", "factoid", "review", "gene_function"):
            cloned_genes_text = self._get_cloned_genes_for_prompt()
            system = self._build_cloned_gene_prompt(user_query, cloned_genes_text)
            for chunk in self.generator.generate_stream(user_query, system, {}, enable_thinking=False):
                yield chunk
            # Send empty references for cloned gene answers
            import json as _json
            yield "\n\n---METADATA---\n" + _json.dumps({"query_type": "gene_list", "references": []}, ensure_ascii=False) + "\n"
            return

        # 5. metadata 检索
        meta_hits = self.retriever.retrieve_metadata(user_query, en_keywords, query_type)
        # 7. 全文检索
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
        # 10b. 序列解析（sequence 类型提前解析，避免"证据不足"早退并修正 ID）
        seq_ctx, seq_block = "", ""
        if query_type == "sequence":
            seq_ctx, seq_block = self._sequence_blocks(user_query)
        # 11. 证据不足
        if not selected_hits and not seq_block:
            chinese_char_count = sum(1 for c in user_query if ord("一") <= ord(c) <= ord("鿿"))
            no_evidence_msg = (
                "未检索到足够的全文证据，现有检索证据有限。"
                if chinese_char_count > 0
                else "Insufficient evidence retrieved from the literature database."
            )
            yield no_evidence_msg
            return

        # 12. 构建 system prompt
        system_prompt, protected_map, source_index = build_system_prompt(
            user_query, query_type, selected_hits, extra_types=extra_types, seq_context=seq_ctx
        )

        # 13. 流式生成答案（同时累计正文，用于过滤参考文献）
        answer_parts = []
        for chunk in self.generator.generate_stream(
            user_query, system_prompt, protected_map, enable_thinking=False
        ):
            answer_parts.append(chunk)
            yield chunk

        # 13b. 序列注入（sequence 类型）
        if query_type == "sequence":
            if seq_block:
                answer_parts.append(seq_block)
                yield seq_block

        # 14. 流结束后返回元数据（参考文献只保留正文实际 [n] 引用的）
        cited = self._extract_cited_indices("".join(answer_parts))
        references = self._build_reference_list(source_index, selected_hits, query_type, cited)
        import json
        meta = json.dumps({
            "query_type": query_type,
            "references": references
        }, ensure_ascii=False)
        yield "\n\n---METADATA---\n" + meta + "\n"