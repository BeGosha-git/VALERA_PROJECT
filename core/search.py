"""Internet search module using DuckDuckGo (free, no API key needed)."""

from typing import Optional

from duckduckgo_search import DDGS
from loguru import logger

from config import settings


def search_web(
    query: str,
    max_results: int = 5,
    region: Optional[str] = None,
) -> list[dict]:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        region: Search region (e.g., "ru-ru", "us-en").

    Returns:
        List of results with 'title', 'url', 'body' keys.
    """
    if not settings.search_enabled:
        logger.warning("Internet search is disabled in config.")
        return []

    region = region or settings.search_region

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=query,
                region=region,
                max_results=max_results,
            ))
        logger.info(f"Search '{query}': {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def format_search_results(results: list[dict]) -> str:
    """Format search results as a text block for the model.

    Args:
        results: List of search result dicts.

    Returns:
        Formatted text string.
    """
    if not results:
        return "По запросу ничего не найдено."

    lines = ["Результаты поиска в интернете:"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Без названия")
        body = r.get("body", "")
        href = r.get("href", "")
        # Truncate body for prompt efficiency
        body_short = body[:300] + "..." if len(body) > 300 else body
        lines.append(f"{i}. {title}\n   {body_short}\n   {href}")

    return "\n".join(lines)


def search_and_format(query: str, max_results: int = 5) -> str:
    """Search and return formatted results. Convenience function."""
    results = search_web(query, max_results)
    return format_search_results(results)
