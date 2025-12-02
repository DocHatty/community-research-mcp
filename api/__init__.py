"""
Community Research API Integrations.

This module provides async search functions for developer communities,
Q&A sites, and web search APIs. All functions follow a consistent pattern:

    async def search(query: str, language: str = None, **options) -> list[dict]

Each result dict contains at minimum: title, url, snippet, source

Available Sources:
    COMMUNITY (Free)     stackexchange, github, hackernews, lobsters, discourse
    WEB SEARCH (API key) serper, tavily, brave, firecrawl
    FALLBACK             google_grounding (Gemini + Google Search)

Configuration via environment variables or .env file:
    SERPER_API_KEY, TAVILY_API_KEY, BRAVE_SEARCH_API_KEY, FIRECRAWL_API_KEY
    GITHUB_TOKEN (optional), STACKEXCHANGE_API_KEY (optional)
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Community Sources (Free)
from api.stackexchange import (
    search as search_stackexchange,
    search_multi as search_stackexchange_multi,
    search_stackoverflow,
    enrich_with_answers as enrich_stackexchange_answers,
    fetch_accepted_answer,
    SITES as STACKEXCHANGE_SITES,
)

from api.github import search as search_github_issues, search_github

from api.hackernews import search as search_hackernews_posts, search_hackernews

from api.lobsters import search as search_lobsters_posts, search_lobsters

from api.discourse import (
    search as search_discourse_forums,
    search_discourse,
    FORUMS as DISCOURSE_FORUMS,
)

# Web Search APIs (Require Keys)
from api.serper import (
    search as search_serper_web,
    search_news as search_serper_news,
    search_serper,
    get_related as get_serper_related,
    get_serper_related_searches,
)

from api.tavily import (
    search as search_tavily_web,
    search_news as search_tavily_news,
    search_tavily,
    extract as extract_tavily_content,
    extract_tavily,
)

from api.brave import (
    search as search_brave_web,
    search_news as search_brave_news,
    search_brave,
)

from api.firecrawl import (
    search as search_firecrawl_web,
    scrape as scrape_firecrawl_page,
    map_site as map_firecrawl_site,
    search_firecrawl,
    scrape_firecrawl,
    map_firecrawl,
)

# Google Grounding (Gemini + Google Search) - Fallback for niche topics
try:
    from api.google_grounding import (
        is_available as google_grounding_available,
        search as search_google_grounded,
    )

    GOOGLE_GROUNDING_AVAILABLE = True
except ImportError:
    GOOGLE_GROUNDING_AVAILABLE = False
    search_google_grounded = None

    def google_grounding_available():
        return False

from api.stackexchange import (
    search_multi as search_stackexchange_multi,
)
from api.tavily import (
    extract as extract_tavily_content,
)
from api.tavily import (
    extract_tavily,
    search_tavily,
)
from api.tavily import (
    search as search_tavily_web,
)
from api.tavily import (
    search_news as search_tavily_news,
)

# Google Grounding (Gemini + Google Search) - Fallback for niche topics
try:
    from api.google_grounding import (
        is_available as google_grounding_available,
    )
    from api.google_grounding import (
        search as search_google_grounded,
    )
    GOOGLE_GROUNDING_AVAILABLE = True
except ImportError:
    GOOGLE_GROUNDING_AVAILABLE = False
    search_google_grounded = None

    def google_grounding_available():
        return False

__all__ = [
    # Stack Exchange
    "search_stackexchange",
    "search_stackexchange_multi",
    "search_stackoverflow",
    "enrich_stackexchange_answers",
    "fetch_accepted_answer",
    "STACKEXCHANGE_SITES",
    # GitHub
    "search_github_issues",
    "search_github",
    # HackerNews
    "search_hackernews_posts",
    "search_hackernews",
    # Lobsters
    "search_lobsters_posts",
    "search_lobsters",
    # Discourse
    "search_discourse_forums",
    "search_discourse",
    "DISCOURSE_FORUMS",
    # Serper (Google)
    "search_serper_web",
    "search_serper_news",
    "search_serper",
    "get_serper_related",
    "get_serper_related_searches",
    # Tavily
    "search_tavily_web",
    "search_tavily_news",
    "search_tavily",
    "extract_tavily_content",
    "extract_tavily",
    # Brave
    "search_brave_web",
    "search_brave_news",
    "search_brave",
    # Firecrawl
    "search_firecrawl_web",
    "scrape_firecrawl_page",
    "map_firecrawl_site",
    "search_firecrawl",
    "scrape_firecrawl",
    "map_firecrawl",
    # Google Grounding
    "search_google_grounded",
    "google_grounding_available",
    "GOOGLE_GROUNDING_AVAILABLE",
]
