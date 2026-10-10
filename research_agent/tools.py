"""Research Tools.

This module provides search and content processing utilities for the research agent,
using Tavily for URL discovery and fetching full webpage content.
"""

import httpx #python的网络请求库，直接访问网页
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

tavily_client = TavilyClient()

#Agent调用的是tavily_search，而tavily_search内部再调用普通Python函数fetch_webpage_content
def fetch_webpage_content(url: str, timeout: float = 10.0) -> str: #输入网址，输出网页内容，转成markdown格式
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout) #访问网页
        response.raise_for_status() #检查请求是否成功，如果不成功会抛出异常
        return markdownify(response.text) #将网页内容转换为markdown格式，方便在Agent中处理和展示
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    include_domains: list[str] | None = None,
) -> str:
    """Search the web and return several concise candidate results.

    Args:
        query: Search query to execute.
        include_domains: Optional domains that search results must come from.

    Returns:
        Titles, URLs, relevance scores, and short summaries.
    """
    search_results = tavily_client.search(
        query=query,
        search_depth="basic",
        topic="general",
        max_results=3,
        include_domains=include_domains,
        include_answer=False,
        include_raw_content=False,
    )

    results = search_results.get("results", [])
    if not results:
        return f"No search results found for: {query}"

    result_texts = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        score = result.get("score", "")

        result_texts.append(
            f"""## Result {index}
Title: {title}
URL: {url}
Relevance score: {score}
Summary: {content}
"""
        )

    return f"""Found {len(results)} candidate results for "{query}":

{chr(10).join(result_texts)}"""


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str: #并没有真正的思考功能，只是一个占位符，记录反思内容
    #think_tool像人为设计出来的一个“反思步骤/状态节点”，真正进行思考的还是LLM
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}" #确认反思已被记录以供决策使用
