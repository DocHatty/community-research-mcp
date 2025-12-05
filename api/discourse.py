"""
Discourse Forums Search.

Search language-specific Discourse communities for technical discussions.
Uses public JSON endpoints available on all Discourse instances.

Supported Forums: Rust, Elixir, Swift, Julia, Python, and more
Rate Limits: Varies by forum
"""

import logging
from typing import Any, Optional

import httpx

__all__ = ["search", "FORUMS"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_TIMEOUT = 30.0

logger = logging.getLogger(__name__)

# Language-specific Discourse forums
FORUMS: dict[str, str] = {
    "rust": "https://users.rust-lang.org",
    "elixir": "https://elixirforum.com",
    "swift": "https://forums.swift.org",
    "julia": "https://discourse.julialang.org",
    "python": "https://discuss.python.org",
    "ruby": "https://discuss.rubyonrails.org",
    "ember": "https://discuss.emberjs.com",
    "kubernetes": "https://discuss.kubernetes.io",
    "pytorch": "https://discuss.pytorch.org",
    "terraform": "https://discuss.hashicorp.com",
}

DEFAULT_FORUM = "https://meta.discourse.org"

# ══════════════════════════════════════════════════════════════════════════════
# Search Function
# ══════════════════════════════════════════════════════════════════════════════


async def _search_single_forum(
    query: str,
    base_url: str,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search a single Discourse forum."""
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(
                f"{base_url}/search.json",
                params={"q": query},
            )
            response.raise_for_status()
            data = response.json()

            return [
                {
                    "title": topic.get("title", ""),
                    "url": f"{base_url}/t/{topic.get('slug', '')}/{topic.get('id', '')}",
                    "views": topic.get("views", 0),
                    "replies": max(0, topic.get("posts_count", 1) - 1),
                    "likes": topic.get("like_count", 0),
                    "solved": topic.get("has_accepted_answer", False),
                    "snippet": (topic.get("blurb") or "")[:500],
                    "source": f"discourse:{base_url.split('//')[1].split('/')[0]}",
                }
                for topic in data.get("topics", [])[:max_results]
            ]

    except Exception as e:
        logger.warning(f"Search failed on {base_url}: {e}")
        return []


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    forum_url: Optional[str] = None,
    max_results: int = 20,
    search_multiple: bool = True,
) -> list[dict[str, Any]]:
    """
    Search Discourse forums for discussions.

    Args:
        query: Search query string
        language: Programming language to prioritize (but searches others too)
        forum_url: Direct forum URL (overrides language selection)
        max_results: Maximum results to return per forum
        search_multiple: If True, search multiple relevant forums concurrently

    Returns:
        List of topics with title, url, views, replies, likes

    Example:
        >>> results = await search("async await", language="rust")
        >>> results = await search("deployment", forum_url="https://discuss.kubernetes.io")
    """
    import asyncio

    # If specific forum URL provided, only search that one
    if forum_url:
        return await _search_single_forum(query, forum_url.rstrip("/"), max_results)

    # Determine which forums to search
    forums_to_search: list[str] = []

    if language:
        # Prioritize the language-specific forum if it exists
        lang_lower = language.lower()
        if lang_lower in FORUMS:
            forums_to_search.append(FORUMS[lang_lower])

        # Add related forums based on language ecosystem
        related_forums = {
            "python": ["pytorch"],  # Python users often use PyTorch
            "javascript": ["ember"],
            "typescript": ["ember"],
            "rust": [],
            "go": ["kubernetes", "terraform"],  # Go is common in DevOps
            "java": ["kubernetes"],
        }
        for related in related_forums.get(lang_lower, []):
            if related in FORUMS and FORUMS[related] not in forums_to_search:
                forums_to_search.append(FORUMS[related])

    # If no language match or search_multiple, add high-traffic general forums
    if search_multiple and len(forums_to_search) < 3:
        priority_forums = ["python", "pytorch", "kubernetes", "rust"]
        for forum_key in priority_forums:
            if forum_key in FORUMS and FORUMS[forum_key] not in forums_to_search:
                forums_to_search.append(FORUMS[forum_key])
                if len(forums_to_search) >= 4:
                    break

    # Fallback if still empty
    if not forums_to_search:
        forums_to_search = [DEFAULT_FORUM]

    # Search all selected forums concurrently
    tasks = [
        _search_single_forum(query, forum_url, max_results // len(forums_to_search) + 2)
        for forum_url in forums_to_search
    ]

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results from all forums
    all_results: list[dict[str, Any]] = []
    for result in results_list:
        if isinstance(result, list):
            all_results.extend(result)

    # Sort by engagement (views + replies + likes)
    all_results.sort(
        key=lambda x: x.get("views", 0)
        + x.get("replies", 0) * 10
        + x.get("likes", 0) * 5,
        reverse=True,
    )

    return all_results[:max_results]


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_discourse = search
