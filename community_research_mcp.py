#!/usr/bin/env python3
"""
Community Research MCP Server (Simplified)

Searches Stack Overflow, Reddit, GitHub, HackerNews, and more to find
real solutions from real developers.
"""

import asyncio
import logging
import os
import time
from collections.abc import Coroutine
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# Load environment variables
try:
    from dotenv import load_dotenv

    _ = load_dotenv()
except ImportError:
    pass

# Import API integrations
from api import (
    search_brave,
    search_discourse,
    search_firecrawl,
    search_github,
    search_hackernews,
    search_lobsters,
    search_serper,
    search_stackoverflow,
    search_tavily,
)

# Import core utilities
from core import (
    deduplicate_results,
)

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# API Keys
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize MCP server
mcp = FastMCP("community_research_mcp")

# Constants
API_TIMEOUT = 60.0


# =============================================================================
# Reddit Search (built-in, no external module needed)
# =============================================================================


async def search_reddit(query: str, language: str) -> list[dict[str, str | int]]:
    """Search Reddit programming subreddits."""
    subreddit_map = {
        "python": "python+learnpython+django+flask",
        "javascript": "javascript+node+reactjs+webdev",
        "typescript": "typescript+javascript+webdev",
        "rust": "rust+learnrust",
        "go": "golang",
        "java": "java+javahelp",
        "csharp": "csharp+dotnet",
        "cpp": "cpp+cplusplus",
    }

    subreddit = subreddit_map.get(language.lower(), "programming+webdev")
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {"q": query, "sort": "relevance", "limit": 15, "restrict_sr": "on"}
    headers = {"User-Agent": "CommunityResearchMCP/1.0"}

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params, headers=headers)
            _ = response.raise_for_status()
            data: dict[str, Any] = response.json()

            results: list[dict[str, str | int]] = []
            children: list[dict[str, Any]] = data.get("data", {}).get("children", [])
            for item in children:
                post: dict[str, Any] = item.get("data", {})
                title: str = post.get("title", "")
                permalink: str = post.get("permalink", "")
                score: int = post.get("score", 0)
                num_comments: int = post.get("num_comments", 0)
                selftext: str = post.get("selftext", "") or ""
                subreddit: str = post.get("subreddit", "")
                results.append(
                    {
                        "title": title,
                        "url": f"https://www.reddit.com{permalink}",
                        "score": score,
                        "comments": num_comments,
                        "snippet": selftext[:500],
                        "subreddit": subreddit,
                    }
                )
            return results
    except Exception as e:
        logger.warning(f"Reddit search failed: {e}")
        return []


# =============================================================================
# Aggregate Search
# =============================================================================


async def aggregate_search(
    query: str,
    language: str,
    max_results_per_source: int = 15,
) -> dict[str, Any]:
    """Search all available sources in parallel."""

    async def safe_search(
        name: str, coro: Coroutine[Any, Any, list[dict[str, Any]]]
    ) -> tuple[str, list[dict[str, Any]], float, str | None]:
        """Wrap search in error handling."""
        start = time.time()
        try:
            # Directly await the coroutine with error handling
            results = await coro
            return name, results or [], time.time() - start, None
        except Exception as e:
            logger.warning(f"{name} search failed: {e}")
            return name, [], time.time() - start, str(e)

    # Build search tasks for all sources
    tasks = [
        safe_search("stackoverflow", search_stackoverflow(query, language)),
        safe_search("github", search_github(query, language)),
        safe_search("reddit", search_reddit(query, language)),
        safe_search("hackernews", search_hackernews(query)),
        safe_search("lobsters", search_lobsters(query)),
        safe_search("discourse", search_discourse(query, language)),
    ]

    # Add premium sources if API keys are available
    if BRAVE_SEARCH_API_KEY:
        tasks.append(safe_search("brave", search_brave(query, language)))
    if SERPER_API_KEY:
        tasks.append(safe_search("serper", search_serper(query, language)))
    if TAVILY_API_KEY:
        tasks.append(safe_search("tavily", search_tavily(query, language)))
    if FIRECRAWL_API_KEY:
        tasks.append(safe_search("firecrawl", search_firecrawl(query, language)))

    # Run all searches in parallel
    results_list = await asyncio.gather(*tasks)

    # Aggregate results
    all_results: dict[str, list[dict[str, Any]]] = {}
    audit_log: list[dict[str, Any]] = []

    for name, results, duration, error in results_list:
        # Cap results per source
        all_results[name] = results[:max_results_per_source]
        audit_log.append(
            {
                "source": name,
                "count": len(results),
                "duration_ms": round(duration * 1000, 2),
                "error": error,
            }
        )

    # Deduplicate across sources
    deduped = deduplicate_results(all_results)

    return {
        "results": deduped,
        "audit": audit_log,
        "total": sum(len(v) for v in deduped.values()),
    }


# =============================================================================
# MCP Tool
# =============================================================================


@mcp.tool(
    name="community_search",
    annotations=ToolAnnotations(
        title="Search Developer Communities",
        readOnlyHint=True,
    ),
)
async def community_search(
    query: str,
    language: str = "python",
    max_results: int = 15,
) -> dict[str, Any]:
    """
    Search developer communities for real-world solutions.

    Searches: Stack Overflow, GitHub Issues, Reddit, HackerNews, Lobsters,
    Discourse, and web search APIs (Brave, Serper, Tavily, Firecrawl if configured).

    Args:
        query: What to search for (be specific!)
        language: Programming language context (default: python)
        max_results: Max results per source (default: 15)

    Returns:
        Results from all sources with titles, URLs, scores, and snippets
    """
    # Validate
    if len(query.strip()) < 5:
        return {"error": "Query too short. Please be more specific."}

    # Search all sources
    search_results = await aggregate_search(
        query=query.strip(),
        language=language,
        max_results_per_source=min(max_results, 25),
    )

    return {
        "query": query,
        "language": language,
        "total_results": search_results["total"],
        "sources_searched": len(search_results["audit"]),
        "results": search_results["results"],
        "audit": search_results["audit"],
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()
