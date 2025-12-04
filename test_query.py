#!/usr/bin/env python3
"""Test script to query the community research MCP server."""

import asyncio
import json
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from community_research_mcp import community_search


async def main():
    query = "best Whisper version speech recognition 2024 community recommendation"
    print(f"Searching for: {query}\n")

    result = await community_search(query=query, language="python", max_results=10)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
