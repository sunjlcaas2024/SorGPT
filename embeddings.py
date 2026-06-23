# -*- coding: utf-8 -*-
"""
embeddings.py
========================
【作用】
封装本地 BGE-M3 embedding 模型，提供统一的向量接口。
使其兼容 LangChain / FAISS 的 Embeddings 接口。

【输入】
- 文本列表（embed_documents）
- 单条查询（embed_query）

【输出】
- 向量列表

【与其他脚本关系】
- retriever.py 加载 FAISS 索引时需要本模块
- pipeline.py 初始化 Retriever 时会实例化本类
"""

from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from config import MODEL_PATH


class BgeEmbeddingsWrapper(Embeddings):
    """
    对 SentenceTransformer 进行简单封装，
    使其具备 LangChain 标准的 Embeddings 接口。
    """

    def __init__(self, model_path: str = MODEL_PATH, batch_size: int = 64):
        """
        初始化 embedding 模型。

        参数：
        - model_path: 本地 BGE-M3 模型目录
        - batch_size: 批量编码大小
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)

        if device == "cuda":
            try:
                self.model.half()
            except Exception:
                pass

        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量对文档进行向量化。
        用于 FAISS 建库或批量检索。
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        对单条查询进行向量化。
        用于 similarity_search 等操作。
        """
        return self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()[0]

    def embed_query_np(self, text: str) -> np.ndarray:
        """
        对单条查询进行向量化，返回 faiss 可直接 search 的 numpy 格式。
        """
        vec = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)
