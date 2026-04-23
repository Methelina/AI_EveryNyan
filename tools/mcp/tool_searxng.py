"""
MCP server providing web search (via SearXNG) and URL content extraction tools.
Exposes two tools: web_search for meta-search, fetch_url for page content retrieval.

\\tools\\mcp\\tool_searxng.py

Version:     0.1.0
Author:      pytraveler
Updated:     2026-04-23

Patch Notes v0.1.0:
  [+] fetch_url: trafilatura as primary extractor with markdownify fallback.
  [+] fetch_url: extracts metadata (title, author, date) via bare_extraction.
  [+] fetch_url: favor_precision=True for aggressive noise filtering.
"""

import os
import sys
import httpx
import trafilatura
from datetime import datetime
from fastmcp import FastMCP
from markdownify import markdownify as md

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:2597")

mcp = FastMCP("searxng")

# Файл для отладки (будет в корне проекта или в папке logs)
DEBUG_LOG = os.path.join(os.path.dirname(__file__), "../../logs/mcp_debug.log")
DEBUG_LOG = os.path.abspath(DEBUG_LOG)
os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)


def log_debug(msg: str):
    """Безопасно пишем в файл, не трогая stdout/stderr."""
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")


@mcp.tool()
async def web_search(
    query: str,
    categories: str = "general",
    language: str = "auto",
    max_results: int = 10,
) -> str:
    """Meta-search the web via SearXNG and return markdown-formatted results."""
    log_debug(f"=== web_search called ===")
    log_debug(f"Query: {query}")
    log_debug(
        f"Categories: {categories}, language: {language}, max_results: {max_results}"
    )

    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "language": language,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])[:max_results]
    log_debug(f"Found {len(results)} raw results")

    if not results:
        log_debug("No results, returning empty")
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"### {i}. {title}")
        lines.append(f"**URL:** {url}")
        if snippet:
            lines.append(f"\n{snippet}")
        lines.append("")

    formatted = "\n".join(lines)
    log_debug(f"Formatted result length: {len(formatted)} characters")
    log_debug(f"Preview: {formatted[:500]}...")
    log_debug("=== web_search finished ===\n")

    return formatted


@mcp.tool()
async def fetch_url(url: str, max_length: int = 5000) -> str:
    """Fetch a URL and return its main content as markdown."""
    log_debug("=== fetch_url called ===")
    log_debug(f"URL: {url}, max_length: {max_length}")

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    log_debug(f"Downloaded HTML: {len(html)} characters")

    downloaded = trafilatura.bare_extraction(
        html,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        favor_precision=True,
    )

    if not downloaded or not downloaded.text:
        log_debug("trafilatura failed, falling back to markdownify")
        text = md(html, strip=["img", "script", "style"])
        text = "\n".join(line for line in text.splitlines() if line.strip())
        if not text.strip():
            log_debug("markdownify also returned empty content")
            log_debug("=== fetch_url finished ===\n")
            return "Could not extract content from this page."
        log_debug(f"markdownify fallback: {len(text)} characters")
    else:
        log_debug(
            f"Extracted — title: {downloaded.title}, "
            f"author: {downloaded.author}, date: {downloaded.date}"
        )

        parts = []
        if downloaded.title:
            parts.append(f"# {downloaded.title}")
        if downloaded.author:
            parts.append(f"Author: {downloaded.author}")
        if downloaded.date:
            parts.append(f"Date: {downloaded.date}")
        if len(parts) > 1:
            parts.append("")
        parts.append(downloaded.text)

        text = "\n".join(parts)

    if len(text) > max_length:
        text = text[:max_length] + "\n\n... (truncated)"
        log_debug(f"Truncated to {max_length} characters")

    log_debug(f"Result length: {len(text)} characters")
    log_debug(f"Preview: {text[:500]}...")
    log_debug("=== fetch_url finished ===\n")

    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")
