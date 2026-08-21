#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_seq_regression.py — 序列功能回归测试（不依赖完整 RAG 模型加载）。

在 sorghum_rag 环境运行，只 import 轻量模块（config/query_classifier/sequence_fetcher），
验证：1) 分类器把 6 道评估序列题路由到 "sequence"；2) 符号→ID 解析正确；
3) get-seq 能取到序列并生成前 N bp/aa 预览。
"""
import sys

sys.path.insert(0, "/vol/sunjilin/website/data/agent/sorghum_rag")

from query_classifier import classify_query_type
from sequence_fetcher import (
    resolve_genes_from_query,
    detect_seq_type,
    build_sequence_preview,
)

# (题号, 题目, 期望序列类型, 期望解析出的 Sobic ID)
CASES = [
    ("Q55", "Extract the protein sequence of Sobic.001G394400.", "protein", "Sobic.001G394400"),
    ("Q58", "Extract the protein sequence of Sobic.001G010200.", "protein", "Sobic.001G010200"),
    ("Q60", "Extract the CDS sequence of SbDREB1A (Sobic.001G411200).", "cds", "Sobic.001G411200"),
    ("Q61", "Extract the protein sequence of SbCYP79A1.", "protein", "Sobic.001G012300"),
    ("Q63", "Extract the CDS of SbTFL1 (Sobic.006G160900).", "cds", "Sobic.006G160900"),
    ("Q64", "Extract the protein sequence of AltSB (SbMATE).", "protein", "Sobic.003G403000"),
]

# 中文序列题（额外验证中文路由 + 符号解析）
CASES_ZH = [
    ("SbCYP79A1的核苷酸序列是什么", "dna", "Sobic.001G012300"),
    ("SbMATE的蛋白序列是什么", "protein", "Sobic.003G403000"),
    ("请给出 Sobic.001G394400 的 CDS 序列", "cds", "Sobic.001G394400"),
]


def main():
    all_cases = CASES + [("ZH", q, s, g) for q, s, g in CASES_ZH]

    print("=" * 76)
    print("1) 分类器路由测试（期望 type=sequence）")
    n_fail = 0
    for qid, q, _st, _g in all_cases:
        qt, tags, _ = classify_query_type(q)
        ok = qt == "sequence"
        n_fail += 0 if ok else 1
        print(f"  [{qid}] type={qt!r:<10} {'OK' if ok else 'FAIL  <<<<'}  <- {q[:55]}")

    print("=" * 76)
    print("2) 符号解析 + 序列类型判定")
    for qid, q, stype, expect_gid in all_cases:
        gids = resolve_genes_from_query(q)
        dt = detect_seq_type(q)
        ok_g = expect_gid in gids
        ok_t = dt == stype
        mark = "OK" if (ok_g and ok_t) else "FAIL"
        n_fail += 0 if (ok_g and ok_t) else 1
        print(f"  [{qid}] resolve={gids} (expect {expect_gid}) {'' if ok_g else '<<<GENE'} "
              f"| type={dt} (expect {stype}) {'' if ok_t else '<<<TYPE'}")

    print("=" * 76)
    print("3) get-seq 序列注入预览（前 N bp/aa）")
    for qid, q, stype, expect_gid in all_cases:
        gids = resolve_genes_from_query(q)
        dt = detect_seq_type(q)
        if not gids:
            print(f"  [{qid}] 无解析结果，跳过")
            continue
        gid = gids[0]
        try:
            b = build_sequence_preview(gid, seq_type=dt, lang="chinese")
            one = b.replace("\n", " / ") if b else "(空)"
            print(f"  [{qid}] {gid}: {one[:130]}")
            if not b:
                n_fail += 1
        except Exception as e:
            n_fail += 1
            print(f"  [{qid}] {gid} FAIL: {e}")

    print("=" * 76)
    print(f"结果: {'✅ 全部通过' if n_fail == 0 else '❌ %d 项失败' % n_fail}")


if __name__ == "__main__":
    main()
