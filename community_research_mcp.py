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
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
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
    get_circuit_breaker,
    resilient_api_call,
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


async def search_reddit(query: str, language: str) -> List[Dict[str, Any]]:
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
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("data", {}).get("children", []):
                post = item.get("data", {})
                results.append(
                    {
                        "title": post.get("title", ""),
                        "url": f"https://www.reddit.com{post.get('permalink', '')}",
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "snippet": (post.get("selftext", "") or "")[:500],
                        "subreddit": post.get("subreddit", ""),
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
) -> Dict[str, Any]:
    """Search all available sources in parallel."""

    async def safe_search(name: str, coro):
        """Wrap search in error handling."""
        start = time.time()
        try:
            circuit = get_circuit_breaker(name)
            results = await circuit.call_async(resilient_api_call, coro)
            return name, results or [], time.time() - start, None
        except Exception as e:
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
    all_results: Dict[str, List] = {}
    audit_log = []

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
    annotations={
        "title": "Search Developer Communities",
        "readOnlyHint": True,
    },
)
async def community_search(
    query: str,
    language: str = "python",
    max_results: int = 15,
) -> Dict[str, Any]:
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
