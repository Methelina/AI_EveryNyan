"""
MCP server providing web search (via SearXNG) and URL content extraction tools.
Exposes two tools: web_search for meta-search, fetch_url for page content retrieval.

\\tools\\mcp\\tool_searxng.py

Version:     0.4.0
Author:      pytraveler
Updated:     2026-04-23

Patch Notes v0.4.0 (by Soror L.'.L.'):
  [+] Integrated support for Playwright and Nodriver scrapers.
  [+] Added FETCH_MODE variable to select content engine (legacy/playwright/nodriver).
  [+] Playwright and Nodriver use a shared isolated browser path (.cache/ms-playwright).
  [*] fetch_url split into three methods: fetch_legacy (bs4+httpx), fetch_playwright, fetch_nodriver.
  [*] Both new methods use shared HTML cleaning and Markdown conversion logic.

Patch Notes v0.3.0 (by Soror L.'.L.'):
  [CHANGE] fetch_url fully rewritten to "bs4 + markdownify" method (no trafilatura).
  [+] fetch_url: preliminary HTML cleaning via BeautifulSoup (remove script/style/nav/footer/...).
  [+] fetch_url: converting cleaned HTML to Markdown via markdownify.
  [+] fetch_url: added brief stderr output (URL and extracted text size).
  [REMOVED] trafilatura is no longer used (proven less stable on complex sites).
  [+] Configuration variables moved to top of file (limits, timeouts, paths).
"""

import os
import sys
import glob
import httpx
from datetime import datetime
from fastmcp import FastMCP

# ============================================================================
# CONFIGURATION
# ============================================================================

# SCRAPER MODE SELECTION: 'legacy', 'playwright', 'nodriver'
# 'legacy' - fast httpx + bs4 (no JS).
# 'playwright' - full browser (accurate, but heavier).
# 'nodriver' - CDP browser (fast, lightweight, renders JS).
FETCH_MODE = os.environ.get("FETCH_MODE", "playwright")

# Browser Paths (Isolated Installation)
# Assuming script is in <repo>/tools/mcp/
# Repo root is 3 levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISOLATED_BROWSER_PATH = os.path.join(REPO_ROOT, ".cache", "ms-playwright")

# Set environment variable for Playwright to see local browser
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = ISOLATED_BROWSER_PATH

# Limits for fetch_url (extracted text size)
DEFAULT_MAX_FETCH_LENGTH = 500000          # Default max characters
MAX_FETCH_LENGTH_GLOBAL = 500000          # Absolute max

# Limits for web_search
DEFAULT_MAX_SEARCH_RESULTS = 10
MAX_SEARCH_RESULTS_GLOBAL = 50

# Timeouts
HTTP_TIMEOUT = 30

# SearXNG URL
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:2597")

# Logging
DEBUG_LOG = os.path.join(os.path.dirname(__file__), "../../logs/mcp_debug.log")
DEBUG_LOG = os.path.abspath(DEBUG_LOG)

# HTML Cleaning Settings
TAGS_TO_REMOVE = ['script', 'style', 'meta', 'link', 'noscript',
                  'header', 'footer', 'nav', 'aside', 'form', 'button',
                  'iframe', 'svg', 'canvas', 'command', 'embed', 'object']
MD_STRIP_TAGS = ['img', 'script', 'style', 'nav', 'footer', 'header', 'aside']

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[MCP] WARNING: BeautifulSoup4 not installed.", file=sys.stderr)

try:
    from markdownify import markdownify as md
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    print("[MCP] WARNING: markdownify not installed.", file=sys.stderr)

# Scraper availability checks
PLAYWRIGHT_AVAILABLE = False
NODRIVER_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

try:
    import nodriver
    NODRIVER_AVAILABLE = True
except ImportError:
    pass

mcp = FastMCP("searxng")

# Create log folder if missing
os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_debug(msg: str):
    """Write safely to file without touching stdout/stderr."""
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")

def get_chrome_executable_path() -> str | None:
    """
    Attempts to find Chrome executable in the isolated Playwright installation folder.
    Required for nodriver as it might not pick up Playwright env vars automatically.
    """
    # Playwright folder structure: .cache/ms-playwright/chromium-<version>/chrome-<os>/chrome.exe
    # Search for last chromium version
    base_path = os.path.join(ISOLATED_BROWSER_PATH, "chromium-*")
    chromium_dirs = glob.glob(base_path)
    if not chromium_dirs:
        log_debug(f"Chromium not found in {base_path}")
        return None
    
    # Take last alphabetical (usually latest version)
    chromium_dir = sorted(chromium_dirs)[-1]
    
    # Search for binary folder (chrome-win or mac, linux)
    possible_paths = [
        os.path.join(chromium_dir, "chrome-win", "chrome.exe"),
        os.path.join(chromium_dir, "chrome-linux", "chrome"),
        os.path.join(chromium_dir, "chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    # Try glob if standard paths failed
    found = glob.glob(os.path.join(chromium_dir, "**", "chrome.exe"), recursive=True)
    if found:
        return found[0]
    
    return None

def clean_html_with_bs4(html: str) -> str:
    """Removes noisy tags, returns cleaned HTML."""
    if not BS4_AVAILABLE:
        return html
    try:
        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')

        for tag in TAGS_TO_REMOVE:
            for t in soup.find_all(tag):
                t.decompose()

        # Remove empty elements (except br, p, hr)
        for tag in soup.find_all():
            if len(tag.get_text(strip=True)) == 0 and tag.name not in ['br', 'p', 'hr']:
                tag.decompose()

        return str(soup)
    except Exception as e:
        log_debug(f"Error in clean_html_with_bs4: {e}")
        return html

# ============================================================================
# CONTENT FETCHING METHODS (LEGACY, PLAYWRIGHT, NODRIVER)
# ============================================================================

async def fetch_legacy(url: str, max_length: int) -> str:
    """Original method: httpx + bs4."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log_debug(f"Legacy fetch error: {e}")
            raise

async def fetch_playwright(url: str, max_length: int) -> str:
    """Playwright method: render JS, download HTML."""
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("Playwright library not found.")
    
    log_debug(f"Using Playwright for {url}")
    try:
        async with async_playwright() as p:
            # Launch browser. 
            # headless=True works better in newer Playwright (new headless mode default)
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            )
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            )
            
            # Set load timeout
            await page.goto(url, wait_until="domcontentloaded", timeout=HTTP_TIMEOUT * 1000)
            
            # Small delay for simple JS scripts
            # await page.wait_for_timeout(1000) # Optional
            
            html = await page.content()
            await browser.close()
            return html
    except Exception as e:
        log_debug(f"Playwright fetch error: {e}")
        raise

async def fetch_nodriver(url: str, max_length: int) -> str:
    """Nodriver method: CDP, fast and lightweight."""
    if not NODRIVER_AVAILABLE:
        raise ImportError("Nodriver library not found.")
    
    log_debug(f"Using Nodriver for {url}")
    try:
        # Determine browser path if isolated
        browser_path = get_chrome_executable_path()
        
        if browser_path:
            log_debug(f"Nodriver using browser path: {browser_path}")
            browser = await nodriver.start(browser_executable_path=browser_path)
        else:
            log_debug("Nodriver using system/default browser path")
            browser = await nodriver.start()

        tab = browser.main_tab
        await tab.get(url)
        
        # Wait for load completion (nodriver waits automatically)
        html = await tab.content
        await browser.stop()
        return html
    except Exception as e:
        log_debug(f"Nodriver fetch error: {e}")
        raise

# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
async def web_search(
    query: str,
    categories: str = "general",
    language: str = "auto",
    max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
) -> str:
    """Meta-search the web via SearXNG and return markdown-formatted results."""
    log_debug(f"=== web_search called ===")
    if max_results > MAX_SEARCH_RESULTS_GLOBAL:
        max_results = MAX_SEARCH_RESULTS_GLOBAL

    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "language": language,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
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
async def fetch_url(url: str, max_length: int = DEFAULT_MAX_FETCH_LENGTH) -> str:
    """
    Fetch a URL, clean it with BeautifulSoup, and return markdown.
    Uses FETCH_MODE (legacy/playwright/nodriver) to fetch HTML.
    """
    log_debug(f"=== fetch_url called (Mode: {FETCH_MODE}) ===")
    log_debug(f"URL: {url}, max_length: {max_length}")

    # Limit max_length
    if max_length > MAX_FETCH_LENGTH_GLOBAL:
        max_length = MAX_FETCH_LENGTH_GLOBAL

    html = ""
    fetch_method_used = FETCH_MODE

    try:
        # Select fetch method
        if FETCH_MODE == "playwright":
            html = await fetch_playwright(url, max_length)
        elif FETCH_MODE == "nodriver":
            html = await fetch_nodriver(url, max_length)
        else:
            # Fallback to legacy
            fetch_method_used = "legacy"
            html = await fetch_legacy(url, max_length)
            
    except Exception as e:
        log_debug(f"Error with {fetch_method_used}: {e}. Trying legacy fallback.")
        # If selected method fails, try legacy
        try:
            html = await fetch_legacy(url, max_length)
            fetch_method_used = "legacy (fallback)"
        except Exception as e_legacy:
            log_debug(f"Legacy fetch also failed: {e_legacy}")
            return f"Error: Failed to fetch URL using {fetch_method_used} and legacy fallback. Details: {e_legacy}"

    log_debug(f"Downloaded HTML ({fetch_method_used}): {len(html)} characters")

    # Cleaning and conversion (shared block for all methods)
    cleaned_html = clean_html_with_bs4(html)
    log_debug(f"Cleaned HTML (bs4): {len(cleaned_html)} characters")

    if not MD_AVAILABLE:
        # Fallback if no markdownify
        try:
            soup = BeautifulSoup(cleaned_html, 'lxml' if BS4_AVAILABLE else 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        except Exception:
            text = "Error: markdownify missing and BeautifulSoup extraction failed."
    else:
        # Convert to Markdown
        text = md(cleaned_html, strip=MD_STRIP_TAGS)
        text = "\n".join(line for line in text.splitlines() if line.strip())

        # If empty - try raw text
        if not text.strip() and BS4_AVAILABLE:
            log_debug("markdownify gave empty result, falling back to bs4.get_text()")
            try:
                soup = BeautifulSoup(cleaned_html, 'lxml')
                raw_text = soup.get_text(separator='\n', strip=True)
                text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
            except Exception as e:
                log_debug(f"bs4.get_text() fallback failed: {e}")

    # Truncation
    if len(text) > max_length:
        text = text[:max_length] + "\n\n... (truncated)"

    log_debug(f"Final length: {len(text)} characters")
    print(f"[MCP] fetch_url ({fetch_method_used}): {url} -> extracted {len(text)} chars", file=sys.stderr, flush=True)

    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")