import os

import httpx
from fastmcp import FastMCP
from markdownify import markdownify as md

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:2597")

mcp = FastMCP("searxng")


@mcp.tool()
async def web_search(
    query: str,
    categories: str = "general",
    language: str = "auto",
    max_results: int = 10,
) -> str:
    """Meta-search the web via SearXNG and return markdown-formatted results.

    Args:
        query: Search query string.
        categories: SearXNG category (general, news, images, videos, science, etc.).
        language: Language code or "auto".
        max_results: Maximum number of results to return.
    """
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
    if not results:
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

    return "\n".join(lines)


@mcp.tool()
async def fetch_url(url: str, max_length: int = 5000) -> str:
    """Fetch a URL and return its content as markdown.

    Args:
        url: The URL to fetch.
        max_length: Maximum characters to return.
    """
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
