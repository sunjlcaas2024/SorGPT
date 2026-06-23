#!/usr/bin/env python
"""
gene_scanner.py - 从全文 chunk 中自动提取基因信息，补充 sorghum_genes.db

三阶段流水线:
  Phase 1: regex 扫描所有 chunk → 候选基因名列表
  Phase 2: 与现有 DB 交叉比对 → 新基因候选
  Phase 3: LLM 批量提取新基因的结构化信息 → 写入 DB
"""

import os, sys, re, json, pickle, sqlite3, time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FULLTEXT_INDEX_PATHS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "db", "sorghum_genes.db")
KNOWN_DB = os.path.join(SCRIPT_DIR, "db", "known_genes.db")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gene_scan_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 高粱基因名正则
GENE_PATTERNS = [
    (re.compile(r'\b(Sobic\.\d{3}G\d{6})\b', re.I), "sobic_id"),
    (re.compile(r'\b(SbiHYZ\.\d{2}G\d{6})\b', re.I), "hyz_id"),
    (re.compile(r'\b(SORBI_3\d{1}G\d{6})\b', re.I), "sorbi3_id"),
    (re.compile(r'\b(Sb[A-Z][A-Za-z0-9]{1,15})\b'), "gene_symbol"),
    (re.compile(r'\b([A-Z][a-z]{0,2}\d{1,2}[A-Z]?\d{0,2})\b'), "classic_symbol"),
]

# 已知的非基因缩写 (过滤噪音)
NON_GENES = {
    "DNA", "RNA", "PCR", "QTL", "GWAS", "MAS", "SNP", "SSR", "RFLP",
    "ATP", "ADP", "NADPH", "NADH", "mRNA", "tRNA", "rRNA", "cDNA",
    "WT", "CK", "MOCK", "CTRL",
}


def _load_chunk_texts(index_keys=None) -> List[str]:
    """从 FAISS 索引加载所有 chunk 文本 (不加载模型，只读 pickle)。"""
    if index_keys is None:
        index_keys = ["en_fine", "en_std", "en_large", "en_para"]

    all_texts = []
    seen = set()
    for key in index_keys:
        path = FULLTEXT_INDEX_PATHS.get(key)
        if not path or not os.path.isdir(path):
            continue
        pkl_path = os.path.join(path, "index.pkl")
        print("Loading %s..." % key)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        docstore = data[0]
        count = 0
        for k, doc in docstore._dict.items():
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            th = text[:200]
            if th not in seen and len(text) > 50:
                seen.add(th)
                all_texts.append(text)
                count += 1
        print("  %s: %d chunks" % (key, count))
    print("Total: %d unique chunks" % len(all_texts))
    return all_texts


def phase1_scan(all_texts=None, output_file=None):
    """Phase 1: regex 扫描所有 chunk，统计基因名出现频率。"""
    if all_texts is None:
        all_texts = _load_chunk_texts()
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "gene_candidates.json")

    gene_counts = Counter()
    gene_contexts = defaultdict(list)

    print("Scanning %d chunks for gene patterns..." % len(all_texts))
    for i, text in enumerate(all_texts):
        for pattern, ptype in GENE_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                m_clean = str(m).strip()
                if len(m_clean) < 3:
                    continue
                if m_clean.upper() in NON_GENES:
                    continue
                gene_counts[m_clean] += 1
                if len(gene_contexts[m_clean]) < 3:
                    start = max(0, text.find(m_clean) - 100)
                    end = min(len(text), text.find(m_clean) + len(m_clean) + 200)
                    gene_contexts[m_clean].append(text[start:end])

        if (i+1) % 500000 == 0:
            print("  %d/%d... %d unique genes" % (i+1, len(all_texts), len(gene_counts)))

    print("Scan complete. %d unique gene symbols." % len(gene_counts))

    top = gene_counts.most_common(30)
    print("\nTop 30 gene mentions:")
    for name, cnt in top:
        print("  %s: %d" % (name, cnt))

    output = {
        "gene_counts": dict(gene_counts.most_common()),
        "gene_contexts": {k: v[:3] for k, v in list(gene_contexts.items())[:50000]},
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print("Saved to %s" % output_file)
    return gene_counts


def phase2_crossref(gene_counts=None):
    """Phase 2: 与现有 DB 交叉比对，找出新基因。"""
    if gene_counts is None:
        cand_file = os.path.join(OUTPUT_DIR, "gene_candidates.json")
        with open(cand_file, "r") as f:
            data = json.load(f)
        gene_counts = Counter(data["gene_counts"])

    conn = sqlite3.connect(DB_PATH)
    known_ids = set()
    for row in conn.execute("SELECT gene_id FROM genes").fetchall():
        known_ids.add(row[0])
    conn.close()

    if os.path.exists(KNOWN_DB):
        conn2 = sqlite3.connect(KNOWN_DB)
        try:
            for row in conn2.execute("SELECT gene_name FROM known_genes").fetchall():
                known_ids.add(row[0].strip())
        except:
            pass
        conn2.close()

    # build uppercase lookup
    known_upper = {k.upper() for k in known_ids}

    novel = set()
    matched = set()
    for gene_name in gene_counts:
        if gene_name in known_ids or gene_name.upper() in known_upper:
            matched.add(gene_name)
        else:
            novel.add(gene_name)

    novel_filtered = {g for g in novel if gene_counts[g] >= 3}

    print("Known (matched): %d" % len(matched))
    print("Novel candidates: %d (>=3 mentions: %d)" % (len(novel), len(novel_filtered)))

    novel_file = os.path.join(OUTPUT_DIR, "novel_genes_filtered.json")
    novel_list = sorted(novel_filtered, key=lambda g: -gene_counts[g])
    with open(novel_file, "w") as f:
        json.dump({
            "count": len(novel_list),
            "genes": novel_list,
            "frequencies": {g: gene_counts[g] for g in novel_list},
        }, f, indent=2)
    print("Saved to %s" % novel_file)

    print("\nTop 30 novel gene candidates:")
    for g in novel_list[:30]:
        print("  %s: %d mentions" % (g, gene_counts[g]))

    return novel_filtered, matched


def phase3_llm_extract(novel_genes=None, model="deepseek-reasoner", limit=500):
    """Phase 3: 用 LLM 从文献上下文中提取新基因信息，写入 DB。"""
    cand_file = os.path.join(OUTPUT_DIR, "gene_candidates.json")
    with open(cand_file, "r") as f:
        data = json.load(f)
    contexts = data.get("gene_contexts", {})

    if novel_genes is None:
        novel_file = os.path.join(OUTPUT_DIR, "novel_genes_filtered.json")
        with open(novel_file, "r") as f:
            nd = json.load(f)
        novel_genes = nd["genes"][:limit]

    novel_genes = novel_genes[:limit]

    from openai import OpenAI
    from config import BASE_URL, API_KEY

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    conn = sqlite3.connect(DB_PATH)

    PROMPT_TEMPLATE = """You are a sorghum genomics expert. Extract structured gene annotation.

GENE SYMBOL: {gene_name}

CONTEXT SNIPPETS:
{snippets}

Return ONLY this JSON (no other text):
{{"gene_symbol":"...","full_name":"...","sobic_id":"...","chr":"...","molecular_function":"...","biological_process":"...","protein_family":"...","reference":"..."}}

If insufficient data, return: {{"gene_symbol":"{gene_name}","insufficient_data":true}}"""

    added = 0
    for i, gene_name in enumerate(novel_genes):
        ctx_list = contexts.get(gene_name, ["No context available."])
        snippets = "\n---\n".join(ctx_list[:3])
        prompt = PROMPT_TEMPLATE.format(gene_name=gene_name, snippets=snippets[:3000])

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=500,
            )
            raw = resp.choices[0].message.content
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                info = json.loads(json_match.group(0))
                if info.get("insufficient_data"):
                    continue

                sobic_id = info.get("sobic_id", "") or gene_name
                conn.execute(
                    "INSERT OR IGNORE INTO genes(gene_id, chr, start, end, strand) VALUES(?,?,?,?,?)",
                    (sobic_id, info.get("chr", ""), 0, 0, "+")
                )
                if info.get("molecular_function"):
                    conn.execute(
                        "INSERT INTO func_annotation(gene_id, db_name, term_id, term_name) VALUES(?,?,?,?)",
                        (sobic_id, "gene_scanner", "molecular_function", info["molecular_function"][:500])
                    )
                if info.get("biological_process"):
                    conn.execute(
                        "INSERT INTO func_annotation(gene_id, db_name, term_id, term_name) VALUES(?,?,?,?)",
                        (sobic_id, "gene_scanner", "biological_process", info["biological_process"][:500])
                    )
                conn.commit()
                added += 1
                print("  + %s -> %s: %s" % (gene_name, sobic_id, info.get("full_name", "")[:80]))

            time.sleep(0.3)
        except Exception as e:
            print("  ! %s error: %s" % (gene_name, str(e)[:100]))

    conn.close()
    print("\nDone. Added %d new gene annotations." % added)
    return added


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SorGPT Gene Scanner")
    p.add_argument("--phase1", action="store_true")
    p.add_argument("--phase2", action="store_true")
    p.add_argument("--phase3", action="store_true")
    p.add_argument("--model", default="deepseek-reasoner")
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()

    if args.phase1:
        texts = _load_chunk_texts()
        phase1_scan(texts)
    elif args.phase2:
        phase2_crossref()
    elif args.phase3:
        phase3_llm_extract(model=args.model, limit=args.limit)
    else:
        print("Usage: python gene_scanner.py --phase1 | --phase2 | --phase3")
        print("  --phase1: regex scan all chunks for gene symbols")
        print("  --phase2: cross-reference with existing DB")
        print("  --phase3: LLM extraction + DB insertion")
