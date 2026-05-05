"""
MCP server providing browser interaction tools.

Tools:
  - open_url: open a URL in the user's default web browser

/tools/mcp/tool_browser.py

Version:     0.1.0
Author:      pytraveler
Created:     2026-05-05
"""

import sys
import webbrowser
import re
from urllib.parse import urlparse

from fastmcp import FastMCP

mcp = FastMCP("browser")

_URL_PATTERN = re.compile(
    r"^https?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def _validate_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: Only http:// and https:// URLs are allowed (got '{parsed.scheme}://')."
    if not parsed.hostname:
        return "Error: URL must have a valid hostname."
    return None


@mcp.tool()
async def open_url(
    url: str,
    new_window: bool = False,
) -> str:
    """
    Open a URL in the user's default web browser.

    Use this when the user asks to open a website, view a link, or browse
    to a specific URL. The URL is validated before opening.

    Parameters:
    - url: The full URL to open (must start with http:// or https://).
    - new_window: Open in a new browser window instead of a tab. Default: false (new tab).
    """
    try:
        url = url.strip()
        if not url:
            return "Error: No URL provided."

        err = _validate_url(url)
        if err:
            return err

        if new_window:
            opened = webbrowser.open_new(url)
        else:
            opened = webbrowser.open_new_tab(url)

        if opened:
            return f"Opened in browser: {url}"
        else:
            return f"Warning: Could not open browser for URL: {url}. The system may have no default browser configured."

    except Exception as e:
        return f"Error opening URL: {type(e).__name__}: {e}"


if __name__ == "__main__":
    print(f"[MCP] browser: Starting MCP server (tool_browser.py)", file=sys.stderr)
    print(f"[MCP] browser: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
