#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_fix_seq_trigger.py — 给 _is_sequence 加裸 "cds"/"cdna" 触发词。
修复 Q63 "Extract the CDS of SbTFL1" 与 "CDS 序列"(含空格) 未命中序列关键词的问题。
"""
import shutil

p = "/vol/sunjilin/website/data/agent/sorghum_rag/query_classifier.py"
s = open(p, encoding="utf-8").read()

old = '''        "nucleotide sequence","dna sequence","gene sequence","cds sequence",
        "coding sequence","protein sequence","amino acid sequence","peptide sequence",
    ])'''
new = '''        "nucleotide sequence","dna sequence","gene sequence","cds sequence",
        "coding sequence","protein sequence","amino acid sequence","peptide sequence",
        "cds","cdna",  # 裸缩写：Extract the CDS of SbTFL1 / "CDS 序列"(含空格)
    ])'''

n = s.count(old)
assert n == 1, f"old 出现 {n} 次，应为1"
shutil.copy2(p, p + ".bak.trigger")
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("OK: _is_sequence 已加 cds/cdna 触发词")
