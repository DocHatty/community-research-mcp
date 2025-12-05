"""
Hacker News Search (via Algolia).

Search HN for high-quality tech discussions, filtering for
substantive posts with significant community engagement.

API: https://hn.algolia.com/api
Rate Limits: Generous (Algolia hosted)
"""

import logging
from typing import Any, Optional

import httpx

__all__ = ["search"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://hn.algolia.com/api/v1"
API_TIMEOUT = 30.0

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Search Function
# ══════════════════════════════════════════════════════════════════════════════


def _simplify_query(query: str) -> str:
    """
    Simplify query for HN's Algolia search which doesn't handle long complex queries well.

    Extracts the most important 3-4 terms, removing filler words and keeping
    technical terms, project names, and key concepts.
    """
    import re

    # Common filler words to remove
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "how",
        "what",
        "when",
        "where",
        "why",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "vs",
        "versus",
        "best",
        "good",
        "better",
        "using",
        "use",
        "used",
        "comparison",
        "compare",
        "comparing",
        "between",
        "about",
        "into",
    }

    # Extract words, preserving case for technical terms
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", query)

    # Filter and prioritize
    key_terms = []
    for word in words:
        lower = word.lower()
        # Skip stop words and very short words
        if lower in stop_words or len(word) < 3:
            continue
        # Prioritize: CamelCase, ALL_CAPS, or longer technical terms
        if (
            any(c.isupper() for c in word[1:])  # CamelCase like FastAPI
            or word.isupper()  # Acronyms like API, SQL
            or len(word) >= 4
        ):  # Meaningful terms
            key_terms.append(word)

    # Take top 4 unique terms (preserving order)
    seen = set()
    unique_terms = []
    for term in key_terms:
        lower = term.lower()
        if lower not in seen:
            seen.add(lower)
            unique_terms.append(term)
            if len(unique_terms) >= 4:
                break

    # If we got too few terms, fall back to original (truncated)
    if len(unique_terms) < 2:
        return " ".join(query.split()[:4])

    return " ".join(unique_terms)


async def search(
    query: str,
    *,
    min_points: int = 5,
    max_results: int = 30,
    search_type: str = "story",
    simplify: bool = True,
) -> list[dict[str, Any]]:
    """
    Search Hacker News via Algolia.

    Args:
        query: Search query string
        min_points: Minimum points/upvotes filter (default: 5, lowered for broader results)
        max_results: Maximum results to return
        search_type: Type of content - 'story', 'comment', or 'all'
        simplify: Simplify long queries for better HN search results (default: True)

    Returns:
        List of posts with title, url, points, comments, snippet

    Example:
        >>> results = await search("rust async", min_points=100)
    """
    # Simplify complex queries for better Algolia search results
    search_query = (
        _simplify_query(query) if simplify and len(query.split()) > 4 else query
    )

    params = {
        "query": search_query,
        "hitsPerPage": min(max_results, 50),
    }

    if search_type != "all":
        params["tags"] = search_type

    # Only apply points filter if above threshold (allows niche topics to surface)
    if min_points > 0:
        params["numericFilters"] = f"points>{min_points}"

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(f"{API_BASE}/search", params=params)
            response.raise_for_status()
            data = response.json()

            return [
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("url")
                    or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    "points": hit.get("points", 0),
                    "comments": hit.get("num_comments", 0),
                    "author": hit.get("author", ""),
                    "snippet": (hit.get("story_text") or "")[:500],
                    "source": "hackernews",
                }
                for hit in data.get("hits", [])
            ]

    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_hackernews = search
