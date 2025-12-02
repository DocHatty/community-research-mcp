"""
Google Grounding API via Gemini.

Uses Gemini's built-in Google Search grounding to find real-time web content.
This is particularly effective for niche topics where community sources have sparse coverage.

API: https://ai.google.dev/gemini-api/docs/google-search
Free Tier: 500 RPD (requests per day) for Flash models
Paid: $35 per 1,000 grounded prompts after free tier

This source is designed as a FALLBACK when primary community sources
(Stack Overflow, GitHub, Reddit) return sparse or no results.
"""

import json
import logging
import os
from typing import Any, Optional

import httpx

__all__ = ["search", "search_grounded", "is_available"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-2.0-flash"  # Best balance of speed/quality for grounding
API_TIMEOUT = 30.0

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Availability Check
# ══════════════════════════════════════════════════════════════════════════════


def is_available() -> bool:
    """Check if Google Grounding is available (API key configured)."""
    return bool(API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# Search Functions
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    max_results: int = 10,
    focus: str = "community",  # "community", "docs", "all"
) -> list[dict[str, Any]]:
    """
    Search using Google Grounding via Gemini API.

    This leverages Gemini's google_search tool to find real-time web content.
    Particularly useful for:
    - Niche topics with sparse community coverage
    - Recent API changes not yet discussed widely
    - Bleeding-edge library versions

    Args:
        query: Search query string
        language: Programming language context
        max_results: Target number of results
        focus: Search focus - "community" biases toward SO/Reddit/GitHub,
               "docs" toward official docs, "all" for everything

    Returns:
        List of results with title, url, snippet, source

    Example:
        >>> results = await search("OpenRouter Gemini transcription", language="python")
    """
    if not API_KEY:
        logger.debug("Skipped: GOOGLE_API_KEY/GEMINI_API_KEY not set")
        return []

    # Build the search prompt based on focus
    full_query = f"{language} {query}".strip() if language else query

    if focus == "community":
        system_prompt = """You are a search assistant focused on finding REAL developer solutions.
Search for discussions, issues, and solutions from:
- Stack Overflow answers
- GitHub issues and discussions
- Reddit programming communities
- Developer blog posts with working examples

Return the most relevant URLs with summaries of what solution each provides."""
    elif focus == "docs":
        system_prompt = """You are a search assistant focused on finding official documentation.
Search for:
- Official API documentation
- Library/framework guides
- Release notes and changelogs
- Migration guides

Return the most relevant URLs with summaries."""
    else:
        system_prompt = """You are a search assistant finding developer resources.
Search for any relevant content including documentation, discussions, and examples.
Return the most relevant URLs with summaries."""

    user_prompt = f"""Find the best resources for this developer question:

{full_query}

For each result, provide:
1. The URL
2. The title/topic
3. A brief summary of what solution or information it provides

Focus on practical, actionable content that actually helps solve the problem."""

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.1,  # Low temp for factual search
            "maxOutputTokens": 2048,
        },
    }

    try:
        url = f"{API_BASE}/models/{MODEL}:generateContent?key={API_KEY}"

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        results = []

        # Extract grounding metadata (URLs from search)
        grounding_metadata = data.get("candidates", [{}])[0].get(
            "groundingMetadata", {}
        )

        grounding_chunks = grounding_metadata.get("groundingChunks", [])
        grounding_supports = grounding_metadata.get("groundingSupports", [])
        search_queries = grounding_metadata.get("webSearchQueries", [])

        # Extract URLs from grounding chunks
        for chunk in grounding_chunks:
            web_info = chunk.get("web", {})
            url = web_info.get("uri", "")
            title = web_info.get("title", "")

            if url:
                results.append(
                    {
                        "title": title or url,
                        "url": url,
                        "snippet": "",  # Will be enriched from supports
                        "source": "google_grounding",
                        "grounded": True,
                    }
                )

        # Enrich with support text
        for support in grounding_supports:
            segment = support.get("segment", {})
            text = segment.get("text", "")
            chunk_indices = support.get("groundingChunkIndices", [])

            for idx in chunk_indices:
                if idx < len(results):
                    if results[idx]["snippet"]:
                        results[idx]["snippet"] += " " + text
                    else:
                        results[idx]["snippet"] = text

        # Also parse the model's response text for any additional URLs/info
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                text = part.get("text", "")
                if text:
                    # Extract any URLs mentioned in the response
                    import re

                    url_pattern = r'https?://[^\s<>"\')\]]+'
                    found_urls = re.findall(url_pattern, text)
                    existing_urls = {r["url"] for r in results}

                    for found_url in found_urls:
                        if found_url not in existing_urls:
                            results.append(
                                {
                                    "title": found_url.split("/")[-1] or "Resource",
                                    "url": found_url,
                                    "snippet": "Found in grounded response",
                                    "source": "google_grounding",
                                    "grounded": True,
                                }
                            )

        # Store search queries used for transparency
        if results and search_queries:
            results[0]["_search_queries"] = search_queries

        return results[:max_results]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error("Invalid Google API key")
        elif e.response.status_code == 429:
            logger.warning("Google Grounding rate limit exceeded")
        elif e.response.status_code == 400:
            logger.warning(f"Bad request: {e.response.text[:200]}")
        else:
            logger.warning(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        return []
    except Exception as e:
        logger.error(f"Google Grounding search failed: {e}")
        return []


async def search_niche_topic(
    query: str,
    language: Optional[str] = None,
    *,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Specialized search for niche topics that lack community coverage.

    This uses a two-phase approach:
    1. Search for any existing discussions/resources
    2. If sparse, search for related/alternative approaches

    Args:
        query: The niche topic to search
        language: Programming language
        context: Additional context about what the user is trying to achieve

    Returns:
        Dict with 'primary_results', 'alternative_approaches', and 'suggestions'
    """
    if not API_KEY:
        return {
            "error": "GOOGLE_API_KEY/GEMINI_API_KEY not configured",
            "primary_results": [],
            "alternative_approaches": [],
            "suggestions": [],
        }

    full_query = f"{language} {query}".strip() if language else query

    # Phase 1: Direct search
    prompt = f"""I'm searching for developer resources on a potentially niche topic:

{full_query}

{f"Context: {context}" if context else ""}

Please:
1. Search for any existing discussions, documentation, or solutions
2. If this is a niche/new topic with limited coverage, identify:
   - Alternative approaches that achieve the same goal
   - Related technologies with better documentation
   - Official sources even if sparse
3. Provide actionable suggestions

For each resource found, provide the URL and a brief summary."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,
        },
    }

    try:
        url = f"{API_BASE}/models/{MODEL}:generateContent?key={API_KEY}"

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        # Extract grounding data
        grounding = data.get("candidates", [{}])[0].get("groundingMetadata", {})

        primary_results = []
        for chunk in grounding.get("groundingChunks", []):
            web = chunk.get("web", {})
            if web.get("uri"):
                primary_results.append(
                    {
                        "title": web.get("title", ""),
                        "url": web.get("uri"),
                        "source": "google_grounding",
                    }
                )

        # Get the model's analysis
        response_text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            response_text = " ".join(p.get("text", "") for p in parts)

        return {
            "primary_results": primary_results,
            "model_analysis": response_text,
            "search_queries_used": grounding.get("webSearchQueries", []),
            "is_niche": len(primary_results) < 3,
        }

    except Exception as e:
        logger.error(f"Niche topic search failed: {e}")
        return {
            "error": str(e),
            "primary_results": [],
            "alternative_approaches": [],
            "suggestions": [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_grounded = search
