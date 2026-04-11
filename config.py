"""
配置中心 —— 环境变量加载、模型参数、自动检测机制

学习要点：
- 了解如何通过 .env 文件管理敏感配置（API Key）
- 了解 RAG 系统中的关键超参数及其作用
- 理解 LLM 后端的自动检测与回退机制
"""

import os
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：加载环境变量
# 优先加载 .env（用户配置），不存在则回退到 example.env（示例配置）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning("⚠️ 未找到 .env 文件，已回退加载 example.env。建议：复制 example.env 为 .env 后填写真实配置")
load_dotenv(dotenv_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：API 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "serpapi").lower()
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY")
WEB_SEARCH_API_URL = os.getenv("WEB_SEARCH_API_URL", "")
WEB_SEARCH_ENGINE = os.getenv("WEB_SEARCH_ENGINE", "google")

LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER_NAME", "Custom API")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模型名称配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")


def has_remote_llm_config():
    return all([
        LLM_API_KEY and LLM_API_KEY.strip(),
        LLM_API_URL and LLM_API_URL.strip(),
        LLM_MODEL_NAME and LLM_MODEL_NAME.strip(),
    ])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：RAG 超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
HYBRID_ALPHA = 0.7
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5
MAX_RETRIEVAL_ITERATIONS = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：运行时环境配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
requests.adapters.DEFAULT_RETRIES = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：LLM 后端自动检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_default_model():
    """
    自动检测可用的 LLM 后端，返回默认模型选择

    检测优先级：
    1. 远程 API 已配置 -> 默认使用云端 API
    2. 本地 Ollama 服务可用 -> 默认使用本地模型
    3. 都不可用 -> 返回 api 并提示用户配置
    """
    if has_remote_llm_config() and not LLM_API_KEY.startswith("Your"):
        logging.info("✅ 检测到远程 LLM API 配置，默认使用云端模型")
        return "api"

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            logging.info("✅ 检测到本地 Ollama 服务，默认使用本地模型")
            return "ollama"
    except Exception:
        pass

    logging.warning("⚠️ 未检测到可用 LLM 后端，请配置远程 API 或启动 Ollama")
    return "api"

DEFAULT_MODEL_CHOICE = detect_default_model()
