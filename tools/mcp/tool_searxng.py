"""
MCP server providing web search (via SearXNG) and URL content extraction tools.
Exposes two tools: web_search for meta-search, fetch_url for page content retrieval.

\\tools\\mcp\\tool_searxng.py

Version:     0.6.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v0.6.0 (by Soror L'.L'.):
  [+] fetch_url chunking: long documents are no longer silently truncated.
      Full text is split on paragraph boundaries into ~30k-char chunks, written
      to temp\\fetch_parts\\<urlhash>\\part_NN.md (project temp per policy),
      part 1 returned inline with a manifest so the agent can read the rest
      via the read_file tool. Fixes the "half-read instruction" failure mode
      (500k-char silent truncation / LLM context overflow).
  [+] Stale fetch_parts directories (>24h) are pruned on each fetch.
  [*] max_length now caps the TOTAL document; truncation (rare) is loudly
      marked in the manifest instead of a bare "... (truncated)".

Patch Notes v0.5.0 (by Soror L.'.L.'):
  [FIX] fetch_nodriver: tab.content -> tab.get_content() (nodriver API drift -
      nodriver fetch mode was 100% broken, AttributeError on every call).
  [FIX] fetch_nodriver: wait 2.5s after page load - JS-rendered pages (Pikabu,
      SPA) had empty DOM at serialize time; profile redirected to project temp\.
  [FIX] clean_html_with_bs4: lxml silently drops content past libxml2 nesting
      depth (~256, Vue/React sites exceed it) - heuristic fallback to
      html.parser when a large document yields a tiny body.
  [+] web_search routing: local SearXNG stays plain HTTP; public instances
      (SEARXNG_URL non-local or SEARXNG_FALLBACK_URLS set) are queried ONLY
      via nodriver (undetected Chromium) - plain HTTP gets 429/403 on public
      instances (see temp\\recon\\searx_instances_probe.recon.md).
  [+] Per-call fallback rotation across SEARXNG_FALLBACK_URLS with polite
      delay; httpx search wrapped in try/except with a readable error message.

Patch Notes v0.4.2 (by Soror L.'.L.'):
  [OPT] ISOLATED_BROWSER_PATH now respects PLAYWRIGHT_BROWSERS_PATH from env (set by .bat).
  [FIX] Falls back to relative path calculation if env var is missing.
  [FIX] Updated get_chrome_executable_path to detect 'chromium_headless_shell' directories.

Patch Notes v0.4.1 (by Soror L.'.L.'):
  [FIX] Corrected ISOLATED_BROWSER_PATH to point to 'playwright_browsers'.
  [FIX] Updated get_chrome_executable_path to detect 'chromium_headless_shell' directories.
"""

import os
import sys
import glob
import json
import asyncio
import httpx
from datetime import datetime
from fastmcp import FastMCP

# ============================================================================
# CONFIGURATION
# ============================================================================

# SCRAPER MODE SELECTION: 'legacy', 'playwright', 'nodriver'
FETCH_MODE = os.environ.get("FETCH_MODE", "playwright")

# ============================================================================
# BROWSER PATH RESOLUTION (Priority: Env Var -> Relative Calculation)
# ============================================================================

# 1. Try to get path from Environment Variable (set by run_ai_everynyan.bat)
ISOLATED_BROWSER_PATH = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")

# 2. Fallback: Calculate path relative to this script if Env Var is missing
if not ISOLATED_BROWSER_PATH:
    # Assuming script is in <repo>/tools/mcp/
    # Repo root is 3 levels up.
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ISOLATED_BROWSER_PATH = os.path.join(REPO_ROOT, "playwright_browsers")
    print(f"[MCP] No PLAYWRIGHT_BROWSERS_PATH env var found, using calculated path: {ISOLATED_BROWSER_PATH}", file=sys.stderr)

# Ensure Playwright library sees this path
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = ISOLATED_BROWSER_PATH

# ============================================================================
# GENERAL SETTINGS
# ============================================================================

# Limits for fetch_url (extracted text size)
DEFAULT_MAX_FETCH_LENGTH = 500000          # Default max characters
MAX_FETCH_LENGTH_GLOBAL = 500000          # Absolute max

# Long-document chunking: instead of truncating, the full text is written to
# disk in LLM-sized parts; part 1 goes inline, the rest are read by the agent
# via the read_file tool. ~30k chars ~= 7-8k tokens, safe for local models.
FETCH_CHUNK_SIZE = 30000
FETCH_PARTS_MAX_AGE_SEC = 24 * 3600       # prune stale part dirs on each fetch

# Limits for web_search
DEFAULT_MAX_SEARCH_RESULTS = 10
MAX_SEARCH_RESULTS_GLOBAL = 50

# Timeouts
HTTP_TIMEOUT = 30

# SearXNG URL
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:2597")

# Public fallback chain (JSON list) + routing, set by src\runtime.py
SEARXNG_FALLBACK_URLS_RAW = os.environ.get("SEARXNG_FALLBACK_URLS", "[]")
try:
    SEARXNG_FALLBACK_URLS = json.loads(SEARXNG_FALLBACK_URLS_RAW)
    if not isinstance(SEARXNG_FALLBACK_URLS, list):
        SEARXNG_FALLBACK_URLS = []
except Exception:
    SEARXNG_FALLBACK_URLS = []

import time as _time

def _is_local_searxng(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

# Route through the browser when the endpoint is not a trusted local instance.
SEARXNG_VIA_BROWSER = (
    os.environ.get("SEARXNG_VIA_BROWSER", "0") == "1"
    or not _is_local_searxng(SEARXNG_URL)
)

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


def _fetch_parts_root() -> str:
    """Project temp\\fetch_parts (repo root = 3 levels up from tools\\mcp\\)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "temp", "fetch_parts")


def _prune_stale_part_dirs(max_age_sec: int = FETCH_PARTS_MAX_AGE_SEC) -> None:
    """Remove fetch_parts subfolders older than max_age (temp hygiene)."""
    import time as _t
    root = _fetch_parts_root()
    try:
        now = _t.time()
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path) and now - os.path.getmtime(path) > max_age_sec:
                import shutil
                shutil.rmtree(path, ignore_errors=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        log_debug(f"fetch_parts prune failed: {e}")


def _split_into_chunks(text: str, size: int) -> list:
    """Split text into <=size chunks on paragraph boundaries (never mid-word);
    single paragraphs larger than the chunk are hard-split."""
    paragraphs = text.split("\n\n")
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        para_len = len(para) + 2
        if len(para) > size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            for i in range(0, len(para), size):
                chunks.append(para[i:i + size])
            continue
        if current and current_len + para_len > size:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += para_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _emit_chunked_document(text: str, url: str, max_total: int) -> str:
    """Write the document to temp as numbered .md parts; return part 1 inline
    plus a manifest pointing the agent at the rest (via read_file).
    Short documents (fitting one chunk) return bare, as fetch always did."""
    import hashlib

    if len(text) <= FETCH_CHUNK_SIZE:
        return text

    chunks = _split_into_chunks(text, FETCH_CHUNK_SIZE)

    # Rare: an absurdly huge document exceeding the global cap.
    if len(text) > max_total:
        kept, acc = [], 0
        for c in chunks:
            if acc + len(c) > max_total:
                break
            kept.append(c)
            acc += len(c)
        dropped = len(chunks) - len(kept)
        chunks = kept
        truncation_note = (
            f"\n[DOCUMENT TRUNCATED at max_total={max_total} chars: "
            f"{dropped} part(s) dropped - the text is INCOMPLETE]"
        )
    else:
        truncation_note = ""

    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    parts_dir = os.path.join(_fetch_parts_root(), url_hash)
    os.makedirs(parts_dir, exist_ok=True)

    width = max(2, len(str(len(chunks))))
    for i, chunk in enumerate(chunks[1:], start=2):  # part 1 is returned inline
        part_path = os.path.join(parts_dir, f"part_{i:0{width}d}.md")
        with open(part_path, "w", encoding="utf-8") as f:
            f.write(chunk)

    manifest_lines = [
        f"[DOCUMENT: {len(chunks)} parts, {len(text)} chars total. "
        f"Part 1/{len(chunks)} inline below.]",
        "[Read the remaining parts via the read_file tool:]",
    ]
    for i in range(2, len(chunks) + 1):
        manifest_lines.append(f"  {os.path.join(parts_dir, f'part_{i:0{width}d}.md')}")
    manifest = "\n".join(manifest_lines) + truncation_note

    header = f"--- PART 1/{len(chunks)} ---\n\n"
    return manifest + "\n\n" + header + chunks[0]

def get_chrome_executable_path() -> str | None:
    """
    Attempts to find Chrome or Chrome Headless Shell executable in the isolated installation.
    Searches for both full chromium and chromium_headless_shell folders.
    """
    # Search for full chromium first
    base_path = os.path.join(ISOLATED_BROWSER_PATH, "chromium-*")
    chromium_dirs = glob.glob(base_path)
    
    # Fallback: Look for headless shell if full chromium not found
    if not chromium_dirs:
        log_debug("Full chromium not found, searching for chromium_headless_shell...")
        base_path = os.path.join(ISOLATED_BROWSER_PATH, "chromium_headless_shell-*")
        chromium_dirs = glob.glob(base_path)

    if not chromium_dirs:
        log_debug(f"No browser found in {ISOLATED_BROWSER_PATH}")
        return None
    
    # Take last alphabetical (usually latest version)
    chromium_dir = sorted(chromium_dirs)[-1]
    
    # Prioritize full chrome.exe over chrome-headless-shell.exe
    possible_paths = [
        # Full Chromium
        os.path.join(chromium_dir, "chrome-win", "chrome.exe"),
        os.path.join(chromium_dir, "chrome-linux", "chrome"),
        os.path.join(chromium_dir, "chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium"),
        # Headless Shell (Newer Playwright versions)
        os.path.join(chromium_dir, "chrome-headless-shell-win64", "chrome-headless-shell.exe"),
        os.path.join(chromium_dir, "chrome-headless-shell-linux", "chrome-headless-shell"),
        os.path.join(chromium_dir, "chrome-headless-shell-mac", "chrome-headless-shell"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    # Try glob if standard paths failed
    found = glob.glob(os.path.join(chromium_dir, "**", "chrome.exe"), recursive=True)
    if found:
        return found[0]
        
    found_shell = glob.glob(os.path.join(chromium_dir, "**", "chrome-headless-shell.exe"), recursive=True)
    if found_shell:
        return found_shell[0]
    
    return None

def clean_html_with_bs4(html: str) -> str:
    """Removes noisy tags, returns cleaned HTML."""
    if not BS4_AVAILABLE:
        return html
    try:
        try:
            soup = BeautifulSoup(html, 'lxml')
            # libxml2 (lxml) silently DROPS content nested deeper than its
            # depth limit (~256) - Vue/React apps (Pikabu etc.) exceed it and
            # the story text vanishes without an exception. Detect the mangling
            # by an implausibly small body and fall back to html.parser.
            if len(html) > 50000:
                body = soup.find("body")
                text_len = len(body.get_text(strip=True)) if body else 0
                # A large document must yield a proportionally large text
                # body; a tiny residue means libxml2 dropped the deep tree.
                if text_len < max(200, len(html) // 200):
                    raise ValueError("lxml dropped deeply-nested content")
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

        # Keep the throwaway browser profile inside the project (temp-file policy)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        profile_dir = os.path.join(repo_root, "temp", "uc_profile")
        os.makedirs(profile_dir, exist_ok=True)

        start_kwargs = {"headless": True, "user_data_dir": profile_dir}
        if browser_path:
            log_debug(f"Nodriver using browser path: {browser_path}")
            start_kwargs["browser_executable_path"] = browser_path
        else:
            log_debug("Nodriver using system/default browser path")

        browser = await nodriver.start(**start_kwargs)

        tab = browser.main_tab
        await tab.get(url)
        # JS-heavy pages (Pikabu, SPA) render content client-side - give the
        # DOM a moment to populate before serializing.
        await asyncio.sleep(2.5)

        html = await tab.get_content()
        try:
            await browser.stop()
        except Exception:  # teardown noise on Windows
            pass
        await asyncio.sleep(0.5)  # let subprocess transports close quietly
        return html
    except Exception as e:
        log_debug(f"Nodriver fetch error: {e}")
        raise

# ============================================================================
# MCP TOOLS
# ============================================================================

async def _search_via_nodriver(base_url: str, params: dict) -> dict:
    """Fetch SearXNG JSON results through undetected headless Chromium.

    Public instances filter plain HTTP clients, so the request must carry a
    real browser fingerprint. Returns the parsed results dict.
    """
    if not NODRIVER_AVAILABLE:
        raise RuntimeError("nodriver library not available for public SearXNG query")

    from urllib.parse import urlencode

    browser_path = get_chrome_executable_path()
    # Project-temp nodriver profile (repo root = 3 levels up from tools\mcp\).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    profile_dir = os.path.join(repo_root, "temp", "uc_profile")
    os.makedirs(profile_dir, exist_ok=True)

    start_kwargs = {"headless": True, "user_data_dir": profile_dir}
    if browser_path:
        start_kwargs["browser_executable_path"] = browser_path

    url = base_url.rstrip("/") + "/search?" + urlencode({**params, "format": "json"})
    browser = None
    try:
        browser = await nodriver.start(**start_kwargs)
        page = await browser.get(url)
        html = await page.get_content()
    finally:
        if browser is not None:
            try:
                await browser.stop()
            except Exception:  # teardown noise on Windows
                pass
            await asyncio.sleep(0.5)  # let subprocess transports close quietly

    # SearXNG renders JSON inside <pre>; extract the JSON substring robustly.
    start = html.find("{")
    end = html.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"no JSON in response from {base_url} (blocked or error page)")
    return json.loads(html[start:end + 1])


async def _search_routed(params: dict) -> dict:
    """Route the search: local -> plain HTTP; public -> nodriver with fallback
    rotation across SEARXNG_FALLBACK_URLS."""
    if not SEARXNG_VIA_BROWSER:
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                resp = await client.get(f"{SEARXNG_URL}/search", params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"local SearXNG at {SEARXNG_URL} failed: {exc}. "
                "Remedy: run .\\run_searxng.bat"
            )

    candidates = [SEARXNG_URL] + [u for u in SEARXNG_FALLBACK_URLS if u != SEARXNG_URL]
    last_err = None
    for i, candidate in enumerate(candidates):
        if i > 0:
            await asyncio.sleep(2)  # be polite to rate-limited public instances
        try:
            log_debug(f"web_search via nodriver -> {candidate}")
            data = await _search_via_nodriver(candidate, params)
            if candidate != SEARXNG_URL:
                log_debug(f"web_search fallback instance engaged: {candidate}")
            return data
        except Exception as exc:
            log_debug(f"web_search via nodriver failed for {candidate}: {exc}")
            last_err = exc
    raise RuntimeError(
        f"all SearXNG instances failed (tried {len(candidates)}). "
        f"Last error: {last_err}. Remedy: run .\\run_searxng.bat for a local backend."
    )


@mcp.tool()
async def web_search(
    query: str,
    categories: str = "general",
    language: str = "auto",
    max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
) -> str:
    """Meta-search the web via SearXNG and return markdown-formatted results."""
    log_debug(f"=== web_search called ===")
    log_debug(f"Routing: SEARXNG_VIA_BROWSER={SEARXNG_VIA_BROWSER}, url={SEARXNG_URL}, fallbacks={SEARXNG_FALLBACK_URLS}")
    if max_results > MAX_SEARCH_RESULTS_GLOBAL:
        max_results = MAX_SEARCH_RESULTS_GLOBAL

    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "language": language,
    }

    try:
        data = await _search_routed(params)
    except Exception as exc:
        return (
            f"ERROR: web search unavailable: {exc}"
        )

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
    Long documents are NOT truncated: part 1 is returned inline and the rest
    is written as numbered .md chunks under temp\\fetch_parts\\ - read them
    with the read_file tool.
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

    # Chunked emission: short docs return inline as before; long docs are
    # written to temp as numbered .md parts (part 1 inline + manifest) so the
    # agent can read the whole document via read_file - no silent truncation.
    _prune_stale_part_dirs()
    result = _emit_chunked_document(text, url, max_length)

    log_debug(f"Final length: {len(text)} characters")
    print(f"[MCP] fetch_url ({fetch_method_used}): {url} -> extracted {len(text)} chars", file=sys.stderr, flush=True)

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")