# -*- coding: utf-8 -*-
"""
generator.py
========================
【作用】
调用本地 Qwen API，根据构建好的 prompt 生成最终答案。
【功能】
1. 流式输出，打印思考过程和最终回答
2. 支持 thinking 模式
3. 生成后清洗 markdown / latex 噪声
4. 恢复被保护的生物学术语
【与其他脚本关系】
- pipeline.py 调用本模块
- prompt_builder.py 提供 system_prompt 和 protected_map
"""
from openai import OpenAI
from config import BASE_URL, API_KEY, LOCAL_MODEL_NAME
from utils import clean_symbols, restore_bio_terms


class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def generate(self, user_query: str, system_prompt: str, protected_map: dict, enable_thinking: bool = True) -> str:
        full_content = ""
        try:
            stream = self.client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.0,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking
                    }
                } if enable_thinking else {},
            )
            is_thinking = True
            print("\n" + "=" * 20 + " 深度思考过程 " + "=" * 20)
            for chunk in stream:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    print(delta.reasoning_content, end="", flush=True)
                # 打印最终回答
                elif hasattr(delta, "content") and delta.content:
                    if is_thinking:
                        print("\n" + "=" * 20 + " 最终结论 " + "=" * 20)
                        is_thinking = False
                    print(delta.content, end="", flush=True)
                    full_content += delta.content
            text = clean_symbols(full_content)
            text = restore_bio_terms(text, protected_map)
            return text
        except Exception as e:
            return f"[ERROR] generation failed: {e}"

    def generate_stream(self, user_query: str, system_prompt: str, protected_map: dict, enable_thinking: bool = True):
        """流式生成，yield每个token块供API使用

        输出格式:
        - 思考内容: 直接yield，无前缀
        - 分隔标记: ===== 思考过程结束 =====
        - 最终内容: 直接yield，无前缀
        """
        try:
            stream = self.client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.0,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking
                    }
                } if enable_thinking else {},
            )
            full_content = ""
            reasoning_content = ""
            in_thinking = True

            for chunk in stream:
                delta = chunk.choices[0].delta
                # 思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    yield delta.reasoning_content
                # 最终内容
                elif hasattr(delta, "content") and delta.content:
                    if in_thinking:
                        # 思考阶段结束，输出分隔标记
                        in_thinking = False
                        yield "\n\n==================== 思考过程结束 ====================\n\n"
                    full_content += delta.content
                    yield delta.content

            # 清洗
            if full_content:
                full_content = clean_symbols(full_content)
                full_content = restore_bio_terms(full_content, protected_map)
        except Exception as e:
            yield f"[ERROR] generation failed: {e}"
