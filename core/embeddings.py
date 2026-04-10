"""
向量化模型 —— 将文本映射到高维向量空间

学习要点：
- Embedding 将文本转换为固定维度的向量，使语义相似的文本在向量空间中距离更近
- all-MiniLM-L6-v2 是英文优化模型（384维），中文可换用 text2vec-base-chinese
- 首次运行时模型会自动下载（约 80MB），需要网络连接
"""

import logging
import numpy as np
from functools import lru_cache
from pathlib import Path
from config import EMBED_MODEL_NAME, HF_ENDPOINT

# 模型选择说明：
# - all-MiniLM-L6-v2: 英文优化，384维，轻量快速（默认）
# - shibing624/text2vec-base-chinese: 中文优化
# - BAAI/bge-small-zh-v1.5: 中文优化，性能更好


@lru_cache(maxsize=1)
def get_embed_model():
    """
    获取向量化模型（单例 + 缓存）

    首次调用时加载模型，后续调用直接返回缓存的实例。
    """
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    logging.info(f"加载向量化模型: {EMBED_MODEL_NAME}")

    model_source = EMBED_MODEL_NAME
    if not Path(EMBED_MODEL_NAME).exists():
        repo_id = EMBED_MODEL_NAME if "/" in EMBED_MODEL_NAME else f"sentence-transformers/{EMBED_MODEL_NAME}"
        logging.info(f"通过镜像下载嵌入模型: {repo_id} (endpoint={HF_ENDPOINT})")
        model_source = snapshot_download(repo_id=repo_id, endpoint=HF_ENDPOINT)

    model = SentenceTransformer(model_source)
    logging.info(f"向量化模型加载完成，输出维度: {model.get_sentence_embedding_dimension()}")
    return model


def encode_texts(texts, show_progress=False):
    """
    将文本列表编码为向量

    Args:
        texts: 文本列表
        show_progress: 是否显示进度条

    Returns:
        numpy 数组，形状为 (n_texts, embedding_dim)
    """
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return np.array(embeddings).astype('float32')


def encode_query(query):
    """
    将单个查询文本编码为向量

    Returns:
        numpy 数组，形状为 (1, embedding_dim)
    """
    model = get_embed_model()
    embedding = model.encode([query])
    return np.array(embedding).astype('float32')
