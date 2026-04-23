"""
\tools\mcp\tool_searxng.py
"""
import os
import sys
import httpx
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
    log_debug(f"Categories: {categories}, language: {language}, max_results: {max_results}")

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
    """Fetch a URL and return its content as markdown."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    text = md(html, strip=["img", "script", "style"])
    text = "\n".join(line for line in text.splitlines() if line.strip())

    if len(text) > max_length:
        text = text[:max_length] + "\n\n... (truncated)"

    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")