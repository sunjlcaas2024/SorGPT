# -*- coding: utf-8 -*-
"""
translator.py ── 中文问题 → 英文翻译（检索用，zh-v8）
======================================================
跨语言证据兼容第一步：中文问题翻译成英文，供英文索引的向量检索 + 英文证据的
BM25 词法打分使用。

设计要点：
1. 复用 DeepSeek（deepseek-v4-flash），enable_thinking=False 秒出，温度 0；
2. 进程内缓存：retrieve_fulltext 一次请求被调多次（主/extra/子主题），
   同一问题只翻译一次；
3. 无 CJK（已是英文）快速路径，不调 API；
4. 翻译失败返回 None，调用方回退到 zh-v7 行为（零退化）；
5. 线程安全（api_server 多线程请求）。

依赖：config 的 BASE_URL/API_KEY/LOCAL_MODEL_NAME（与 generator.py 同源）。
"""

import re
import threading

from config import BASE_URL, API_KEY, LOCAL_MODEL_NAME

_CACHE: dict = {}
_LOCK = threading.Lock()
_CLIENT = None
_MAX_CACHE = 1024

_TRANSLATE_SYSTEM_PROMPT = (
    "你是科学文献翻译。把下面的中文高粱研究问题翻译成简洁的英文检索语句。"
    "保留基因名、位点ID（Sobic./Sb...）、专业术语不翻译。只输出英文译文，"
    "不要任何解释、引号或多余内容。"
)


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        _CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _CLIENT


def _has_cjk(text: str) -> bool:
    """与 retriever 的 is_cn 判定同口径（CJK 占比 > 0.15）。"""
    cn = sum(1 for c in text if "一" <= c <= "鿿")
    return cn / max(len(text), 1) > 0.15


def translate_zh_to_en(text: str):
    """中文→英文翻译；英文文本原样返回；失败返回 None（调用方回退）。

    Returns
    -------
    str | None
        - 无 CJK 的英文文本 → 原样返回（快速路径，不调 API）
        - 中文翻译成功 → 英文译文（含进程内缓存）
        - 中文翻译失败 → None
    """
    text = (text or "").strip()
    if not text:
        return None
    if not _has_cjk(text):
        return text

    with _LOCK:
        if text in _CACHE:
            return _CACHE[text]

    try:
        stream = _get_client().chat.completions.create(
            model=LOCAL_MODEL_NAME,
            messages=[
                {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            stream=True,
            timeout=20,
            # 与 generator.py 一致：真正关闭思考，秒出
            extra_body={"thinking": {"type": "disabled"}},
        )
        out = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
        out = re.sub(r"\s+", " ", out).strip()
        if not out:
            return None
        with _LOCK:
            if len(_CACHE) >= _MAX_CACHE:
                _CACHE.clear()
            _CACHE[text] = out
        return out
    except Exception:
        return None
