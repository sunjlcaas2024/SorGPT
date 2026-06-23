# -*- coding: utf-8 -*-
"""
query_expander.py
========================
【作用】
使用本地小模型对用户问题进行检索增强，包括：
1. 生成英文检索关键词
2. 对复杂机制/综述问题生成子主题

【为什么需要它】
因为用户经常用中文提问，但高水平文献大多是英文。
如果直接拿中文问题做向量检索，容易召回不足。
因此要先把问题转为英文关键词。

【输入】
- 用户问题
- query_type（复杂题才需要子主题分解）

【输出】
- 英文关键词字符串
- 或子主题列表

【与其他脚本关系】
- pipeline.py 调用本模块
- retriever.py 用扩展后的关键词进行 metadata / fulltext 检索
"""

import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from config import SMALL_MODEL_PATH
from utils import norm_text


class QueryExpander:
    """
    负责 query rewrite / keyword expansion / subtopic decomposition。
    使用 modelscope 加载本地 Qwen2.5-7B 小模型。
    """

    def __init__(self):
    	self.tokenizer = AutoTokenizer.from_pretrained(
        	SMALL_MODEL_PATH, trust_remote_code=True
    	)
    	self.model = AutoModelForCausalLM.from_pretrained(
        	SMALL_MODEL_PATH,
        	torch_dtype=torch.float16,
        	trust_remote_code=True,
    	).cuda()  # 直接放到 GPU，不用 device_map="auto"
    	self.model.eval()
    def _call(self, system: str, user: str, max_new_tokens: int = 64) -> str:
        """
        通用小模型调用接口。
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return norm_text(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

    def get_english_keywords(self, query: str) -> str:
        """
        将用户问题转换为英文检索关键词。

        输出格式：
        "sorghum drought tolerance, ABA signaling, ROS scavenging"
        """
        try:
            return self._call(
                system=(
                    "Convert the user question into concise English academic retrieval keywords "
                    "for sorghum literature search. Keep gene names, locus names, QTL/GWAS terms, "
                    "numbers, species names, and technical terms. Output only comma-separated keywords."
                ),
                user=query,
                max_new_tokens=64,
            )
        except Exception:
            return ""

    def get_subtopics_for_complex_query(self, query: str, query_type: str):
        """
        对 mechanism / review 类问题进行子主题拆分。

        例如：
        "高粱抗旱机制有哪些"
        可能拆成：
        - ABA signaling
        - ROS scavenging
        - root traits
        - transcription factors
        """
        if query_type not in {"mechanism", "review"}:
            return []

        try:
            result = self._call(
                system=(
                    "Break the query into 3-5 concise English subtopics for literature retrieval. "
                    "Return only one subtopic per line. No numbering. No explanation."
                ),
                user=query,
                max_new_tokens=64,
            )
            lines = [norm_text(x) for x in result.splitlines()]
            return [x for x in lines if x]
        except Exception:
            return []
