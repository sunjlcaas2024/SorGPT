#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_patch_server.py — 给 SorGPT 后端打「基因序列展示」补丁。

对每个 (file, old, new) 做精确替换：先备份 .bak，再断言 old 恰好出现 1 次后替换。
任一步失败即报错退出（不改写），便于核对。
"""
import os
import sys
import shutil

BASE = "/vol/sunjilin/website/data/agent/sorghum_rag"

# 每个元素: (相对路径, 旧串, 新串)
EDITS = []

# ============ config.py ============
EDITS += [
    ("config.py",
     'COUNT_QUERY_FETCH_K  = 300\nCOUNT_QUERY_MAX_SHOW = 200',
     'COUNT_QUERY_FETCH_K  = 300\nCOUNT_QUERY_MAX_SHOW = 200\n\n'
     '# -----------------------------\n'
     '# 基因序列展示 (sequence 类型)\n'
     '# -----------------------------\n'
     'SEQ_PREVIEW_BP      = 60      # 核苷酸序列预览长度(bp)\n'
     'SEQ_PREVIEW_AA      = 30      # 蛋白序列预览长度(aa)\n'
     'GET_SEQ_URL         = "http://127.0.0.1:8001/glapi/mongo/get-seq/"\n'
     'SEQ_SYMBOL_MAP_PATH = os.path.join(_THIS_DIR, "gene_symbol_map.json")'),
    # REFERENCE_LIMITS: 加 sequence
    ("config.py",
     '    "count":        20,\n    "boundary":     0,\n}',
     '    "count":        20,\n    "boundary":     0,\n    "sequence":     6,\n}'),
    # EVIDENCE_LIMITS: 加 sequence
    ("config.py",
     '    "count":        0,\n    "boundary":     0,\n}',
     '    "count":        0,\n    "boundary":     0,\n    "sequence":     12,\n}'),
    # QUERY_TYPE_TO_INDEXES: 加 sequence
    ("config.py",
     '    "count":        [],\n    "boundary":     [],\n}',
     '    "count":        [],\n    "boundary":     [],\n'
     '    "sequence":     ["en_fine", "en_std", "zh_fine", "zh_std"],\n}'),
]

# ============ query_classifier.py ============
EDITS += [
    # 新增 _is_sequence 函数（插在 _is_mechanism 之后）
    ("query_classifier.py",
     'def _is_mechanism(q): return any(p in q for p in [\n'
     '    "机制","通路","调控","信号","互作","如何","怎么",\n'
     '    "mechanism","pathway","regulation of","signaling","interaction",\n'
     '    "how does","why does","what happens",\n'
     '])',
     'def _is_mechanism(q): return any(p in q for p in [\n'
     '    "机制","通路","调控","信号","互作","如何","怎么",\n'
     '    "mechanism","pathway","regulation of","signaling","interaction",\n'
     '    "how does","why does","what happens",\n'
     '])\n\n'
     'def _is_sequence(q, q_orig):\n'
     '    """基因序列类问题：需同时命中序列关键词 + 基因标识。"""\n'
     '    if not _has_gene(q_orig):\n'
     '        return False\n'
     '    return any(p in q for p in [\n'
     '        "核苷酸序列","核苷酸","核酸序列","基因序列","dna序列","cds序列","编码序列",\n'
     '        "蛋白序列","蛋白质序列","氨基酸序列",\n'
     '        "nucleotide sequence","dna sequence","gene sequence","cds sequence",\n'
     '        "coding sequence","protein sequence","amino acid sequence","peptide sequence",\n'
     '    ])'),
    # 收集 tag
    ("query_classifier.py",
     '    if _is_mechanism(q):        tags.add("mechanism")',
     '    if _is_mechanism(q):        tags.add("mechanism")\n'
     '    if _is_sequence(q, q_orig): tags.add("sequence")'),
    # sequence 独占（插在无标签兜底之前）
    ("query_classifier.py",
     '    # ── 无标签兜底 ──────────────────────────────────────────────\n'
     '    if not tags:\n'
     '        tags.add("review")  # v2: changed from mechanism to review (less aggressive default)',
     '    # ── sequence 独占：序列问题不追加其他检索 ────────────────────\n'
     '    if "sequence" in tags:\n'
     '        tags = {"sequence"}\n\n'
     '    # ── 无标签兜底 ──────────────────────────────────────────────\n'
     '    if not tags:\n'
     '        tags.add("review")  # v2: changed from mechanism to review (less aggressive default)'),
    # PRIMARY_ORDER 加 sequence 在首位
    ("query_classifier.py",
     '    PRIMARY_ORDER = [\n'
     '        "factoid",       # 有明确数量/位置答案',
     '    PRIMARY_ORDER = [\n'
     '        "sequence",      # 基因序列\n'
     '        "factoid",       # 有明确数量/位置答案'),
]

# ============ prompt_builder.py ============
EDITS += [
    # EN sequence 任务（插在 boundary 之后）
    ("prompt_builder.py",
     '"boundary": """\\\n'
     'TASK: This question falls outside the sorghum literature knowledge base.\n'
     'Politely explain why, and specify what kinds of questions this system handles well.\n'
     'Do not answer using general knowledge.\n'
     '""",\n'
     '}',
     '"boundary": """\\\n'
     'TASK: This question falls outside the sorghum literature knowledge base.\n'
     'Politely explain why, and specify what kinds of questions this system handles well.\n'
     'Do not answer using general knowledge.\n'
     '""",\n\n'
     '"sequence": """\\\n'
     'TASK: Answer a question about a gene\'s sequence (nucleotide / CDS / protein) in sorghum.\n'
     '• Identify the gene and give its official Sobic.xxxGxxxxxx ID in 1-2 sentences (name, locus, function).\n'
     '• State which sequence type the user asked for (DNA / CDS / protein).\n'
     '• Do NOT write out the nucleotide or amino-acid sequence yourself — the exact sequence is appended automatically below.\n'
     '• Keep under 120 words.\n'
     '""",\n'
     '}'),
    # ZH sequence 任务（插在 boundary 之后）
    ("prompt_builder.py",
     '"boundary": """\\\n'
     '任务：该问题超出高粱文献知识库范围。礼貌解释原因并说明本系统适合的问题类型。\n'
     '不要用通用知识作答。\n'
     '""",\n'
     '}',
     '"boundary": """\\\n'
     '任务：该问题超出高粱文献知识库范围。礼貌解释原因并说明本系统适合的问题类型。\n'
     '不要用通用知识作答。\n'
     '""",\n\n'
     '"sequence": """\\\n'
     '任务：回答关于高粱基因序列（核苷酸 / CDS / 蛋白）的问题。\n'
     '• 用1-2句说明基因的官方 Sobic.xxxGxxxxxx ID、名称/位置和功能。\n'
     '• 说明用户所问的序列类型（DNA / CDS / 蛋白）。\n'
     '• 不要自己写出核苷酸或氨基酸序列——正确序列会在下方自动附上。\n'
     '• 字数控制在150字以内。\n'
     '""",\n'
     '}'),
]

# ============ pipeline.py ============
EDITS += [
    # import sequence_fetcher
    ("pipeline.py",
     'from generator import AnswerGenerator\n'
     'from utils import build_citation_string, norm_text',
     'from generator import AnswerGenerator\n'
     'from sequence_fetcher import resolve_genes_from_query, build_sequence_preview, detect_seq_type\n'
     'from utils import build_citation_string, norm_text'),
    # 新增 _sequence_preview_block 方法（插在 _build_reference_list 之后）
    ("pipeline.py",
     '    def _rule_subtopics(self, query: str, en_keywords: str) -> List[str]:',
     '    def _sequence_preview_block(self, user_query: str) -> str:\n'
     '        """sequence 类型：从 query 解析基因 → 取真实序列 → 生成前 N bp/aa 预览。"""\n'
     '        try:\n'
     '            gene_ids = resolve_genes_from_query(user_query)\n'
     '        except Exception:\n'
     '            return ""\n'
     '        if not gene_ids:\n'
     '            return ""\n'
     '        seq_type = detect_seq_type(user_query)\n'
     '        lang = "chinese" if sum(1 for c in user_query if "\\u4e00" <= c <= "\\u9fff") > 0 else "english"\n'
     '        blocks = []\n'
     '        for gid in gene_ids:\n'
     '            try:\n'
     '                b = build_sequence_preview(gid, seq_type=seq_type, lang=lang)\n'
     '            except Exception:\n'
     '                b = ""\n'
     '            if b:\n'
     '                blocks.append(b)\n'
     '        if not blocks:\n'
     '            return ""\n'
     '        return "\\n\\n" + "\\n\\n".join(blocks)\n\n'
     '    def _rule_subtopics(self, query: str, en_keywords: str) -> List[str]:'),
    # ask() 注入（answer 生成后）
    ("pipeline.py",
     '        # 14. 参考文献（每条文献后紧跟对应证据片段）\n'
     '        references = self._build_reference_list(source_index, selected_hits, query_type)\n'
     '        return {\n'
     '            "query": user_query,\n'
     '            "query_type": query_type,\n'
     '            "answer": answer,',
     '        # 13b. 序列注入（sequence 类型）\n'
     '        if query_type == "sequence":\n'
     '            seq_block = self._sequence_preview_block(user_query)\n'
     '            if seq_block:\n'
     '                answer = answer + seq_block\n'
     '        # 14. 参考文献（每条文献后紧跟对应证据片段）\n'
     '        references = self._build_reference_list(source_index, selected_hits, query_type)\n'
     '        return {\n'
     '            "query": user_query,\n'
     '            "query_type": query_type,\n'
     '            "answer": answer,'),
    # ask_stream() 注入（流结束后，元数据之前）
    ("pipeline.py",
     '        # 14. 流结束后返回元数据（包含参考文献等）\n'
     '        references = self._build_reference_list(source_index, selected_hits, query_type)',
     '        # 13b. 序列注入（sequence 类型）\n'
     '        if query_type == "sequence":\n'
     '            seq_block = self._sequence_preview_block(user_query)\n'
     '            if seq_block:\n'
     '                yield seq_block\n\n'
     '        # 14. 流结束后返回元数据（包含参考文献等）\n'
     '        references = self._build_reference_list(source_index, selected_hits, query_type)'),
]

# ============ api_server.py ============
EDITS += [
    # import sequence_fetcher
    ("api_server.py",
     'from pipeline import SorghumRAGPipeline\n'
     'from config import *',
     'from pipeline import SorghumRAGPipeline\n'
     'from config import *\n'
     'import sequence_fetcher'),
    # 请求模型
    ("api_server.py",
     'class QuestionRequest(BaseModel):\n'
     '    question: str\n'
     '    stream: bool = False',
     'class QuestionRequest(BaseModel):\n'
     '    question: str\n'
     '    stream: bool = False\n\n'
     'class SequenceRequest(BaseModel):\n'
     '    geneid: str\n'
     '    type: str = "dna"'),
    # /sequence 端点（插在 /stats 之前）
    ("api_server.py",
     '@app.get("/stats")',
     '@app.post("/sequence")\n'
     'async def get_gene_sequence(request: SequenceRequest, session: UserSession = Depends(get_current_user)):\n'
     '    """基因序列代理端点：符号自动解析 → get-seq → 返回序列。"""\n'
     '    try:\n'
     '        return sequence_fetcher.fetch_and_format(request.geneid, request.type)\n'
     '    except Exception as e:\n'
     '        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")\n\n'
     '@app.get("/stats")'),
]


def main():
    ok = True
    for rel, old, new in EDITS:
        path = os.path.join(BASE, rel)
        if not os.path.exists(path):
            print(f"[SKIP] {rel} 不存在", file=sys.stderr)
            ok = False
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        n = text.count(old)
        if n != 1:
            print(f"[FAIL] {rel}: 旧串出现 {n} 次（应为1），跳过", file=sys.stderr)
            # 打印旧串前 80 字符便于排查
            print(f"       old 片段: {old[:80]!r}", file=sys.stderr)
            ok = False
            continue
        # 备份（仅首次）
        bak = path + ".bak.seq"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))
        print(f"[OK] {rel}: +{new.count(chr(10)) - old.count(chr(10))} 行")
    if ok:
        print("\n全部补丁应用成功")
    else:
        print("\n存在失败项，请检查后再重启", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
