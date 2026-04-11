"""
联网搜索集成。

RAG 主流程会把联网搜索结果作为额外上下文交给大模型。
不同搜索厂商的返回格式不同，本模块负责把结果统一成：
{"title": ..., "url": ..., "snippet": ..., "timestamp": ...}
"""

import logging
import requests
from config import (
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_API_KEY,
    WEB_SEARCH_API_URL,
    WEB_SEARCH_ENGINE,
)


def check_web_search_config():
    """检查是否配置了有效的联网搜索 API Key。"""
    return bool(
        WEB_SEARCH_API_KEY
        and WEB_SEARCH_API_KEY.strip()
        and not WEB_SEARCH_API_KEY.startswith("Your")
    )


def serpapi_search(query, num_results=5):
    """通过 SerpAPI 搜索，并把返回结果转换成项目内部统一格式。"""
    if not WEB_SEARCH_API_KEY:
        raise ValueError("未配置 WEB_SEARCH_API_KEY")

    try:
        api_url = WEB_SEARCH_API_URL or "https://serpapi.com/search"
        params = {
            "engine": WEB_SEARCH_ENGINE,
            "q": query,
            "api_key": WEB_SEARCH_API_KEY,
            "num": num_results,
            "hl": "zh-CN",
            "gl": "cn",
        }
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        return _parse_serpapi_results(response.json())
    except Exception as e:
        logging.error(f"联网搜索失败: {str(e)}")
        return []


def _parse_serpapi_results(data):
    """解析 SerpAPI 返回结果，转换成项目内部统一格式。"""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "timestamp": item.get("date"),
            })

    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"),
            "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"),
            "timestamp": None,
        })

    return results


def search_web(query, num_results=5):
    """执行联网搜索，搜索结果只作为 LLM 上下文，不写入 FAISS 索引。"""
    if WEB_SEARCH_PROVIDER == "serpapi":
        results = serpapi_search(query, num_results)
    else:
        logging.error(f"暂不支持的联网搜索提供商: {WEB_SEARCH_PROVIDER}")
        return []

    if not results:
        logging.info("联网搜索没有返回结果")
    else:
        logging.info(f"联网搜索返回 {len(results)} 条结果")
    return results
