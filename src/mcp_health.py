"""
Availability probing and fallback resolution for external MCP backends (SearXNG).

The FastMCP stdio subprocess always starts and always advertises its tools, so
tool discovery alone says nothing about whether SearXNG is actually serving.

Routing policy (empirically validated against 75 public instances, see
temp\\recon\\searx_instances_probe.recon.md):
- LOCAL instances (localhost / 127.0.0.1) are probed and queried over plain
  HTTP (urllib) - a trusted endpoint, no reason to pay for a browser.
- PUBLIC fallback instances are probed and queried ONLY through nodriver
  (undetected headless Chromium): simple clients get 429/403 on 67 of 75
  instances, while the browser-grade fingerprint passes where plain HTTP does
  not. playwright is deliberately not used as a probe tier (weaker pass rate,
  innerText blindness for JSON-in-<pre> responses).

Health cache: the resolved URL is reused across the session; a failed instance
goes into cooldown and the next candidate is picked lazily on re-resolve.

src/mcp_health.py
Version:     1.1.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.1.0 (Soror L'.L'.):
  [+] aprobe_searxng()/aresolve_searxng_url(): local-direct vs public-via-browser
      routing with a session health cache and per-instance cooldown.
  [+] probe_searxng_via_nodriver(): undetected-Chromium probe (JSON round-trip),
      browser profile redirected into project temp\ (temp-file policy).
  [+] DEFAULT_FALLBACK_URLS: instances verified to pass ALL probe methods.

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] New module: probe_searxng() via GET /search?q=ping&format=json.
"""

import asyncio
import json
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from logger import logger

SEARXNG_PROBE_TIMEOUT_SEC = 3.0
NODRIVER_PROBE_TIMEOUT_SEC = 30.0
INSTANCE_COOLDOWN_SEC = 300.0        # 5 min per-instance cooldown after failure
CACHE_TTL_SEC = 6 * 3600.0           # re-resolve the whole chain every 6 h

# Public instances verified (2026-09-29) to answer /search?format=json from
# plain HTTP AND both headless browsers. Kept deliberately short: public
# SearXNG is rate-limit-sensitive; users can extend via searxng_fallback_urls.
DEFAULT_FALLBACK_URLS: List[str] = [
    "https://search.mectov.my.id",
    "https://sx.xo.st",
]

# nodriver writes its throwaway profile here (project temp, per temp-file policy).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
NODRIVER_PROFILE_DIR = _PROJECT_ROOT / "temp" / "uc_profile"


def _find_chrome_executable() -> Optional[str]:
    """Locate the project's isolated Chromium for nodriver (same convention as
    tools\\mcp\\tool_searxng.py::get_chrome_executable_path, simplified)."""
    import glob

    base = _PROJECT_ROOT / "playwright_browsers"
    for pattern in ("chromium-*/chrome-win*/chrome.exe", "chromium-*/chrome-win/chrome.exe"):
        hits = sorted(glob.glob(str(base / pattern)))
        if hits:
            return hits[-1]
    hits = sorted(glob.glob(str(base / "chromium_headless_shell-*/**/chrome-headless-shell.exe"), recursive=True))
    return hits[-1] if hits else None


def _search_probe_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/search?q=ping&format=json"


def is_local_url(url: str) -> bool:
    """True for localhost / loopback endpoints."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def probe_searxng(base_url: str, timeout_sec: float = SEARXNG_PROBE_TIMEOUT_SEC) -> bool:
    """Return True if SearXNG answers a minimal JSON search request.

    A real /search round-trip is used (rather than /healthz, which SearXNG
    does not expose by default) - a 200 with parseable JSON body means the
    meta-search engine is actually functional. Used ONLY for local instances.
    """
    try:
        with urllib.request.urlopen(_search_probe_url(base_url), timeout=timeout_sec) as resp:
            return resp.status == 200
    except Exception:
        return False


async def aprobe_searxng_via_nodriver(
    base_url: str, timeout_sec: float = NODRIVER_PROBE_TIMEOUT_SEC
) -> bool:
    """Probe a PUBLIC instance through undetected headless Chromium.

    Plain HTTP clients are rate-limited/blocked on the vast majority of public
    SearXNG instances, so browser-grade fingerprinting is the only reliable
    probe there. Returns True if the JSON search round-trip succeeds.
    """
    try:
        import nodriver as uc
    except ImportError:
        logger.warning("[MCP] fallback: nodriver not installed, cannot probe public SearXNG")
        return False

    NODRIVER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    browser = None
    try:
        start_kwargs = {"headless": True, "user_data_dir": str(NODRIVER_PROFILE_DIR)}
        chrome = _find_chrome_executable()
        if chrome:
            start_kwargs["browser_executable_path"] = chrome
        browser = await uc.start(**start_kwargs)
        page = await browser.get(_search_probe_url(base_url))
        body = await page.evaluate("document.body ? document.body.innerText : ''")
        return bool(body) and body.lstrip().startswith("{")
    except Exception as exc:
        logger.debug("[MCP] nodriver probe failed for %s: %s", base_url, exc)
        return False
    finally:
        if browser is not None:
            try:
                await browser.stop()
            except Exception:  # noqa: BLE001 - teardown noise on Windows
                pass
            # Let nodriver's subprocess transports close before the caller's
            # loop moves on - avoids 'Event loop is closed' GC noise.
            await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# Session health cache: resolved URL + per-instance cooldowns.
# ---------------------------------------------------------------------------
_health_cache = {
    "url": None,           # currently chosen base URL
    "via_browser": False,  # query path for that URL
    "resolved_at": 0.0,
    "cooldowns": {},       # url -> monotonic timestamp until which it is skipped
}


def _now() -> float:
    return time.monotonic()


def _in_cooldown(url: str) -> bool:
    until = _health_cache["cooldowns"].get(url, 0.0)
    return _now() < until


def mark_searxng_failure(url: str, cooldown_sec: float = INSTANCE_COOLDOWN_SEC) -> None:
    """Put an instance into cooldown and drop it from the cache if it was chosen."""
    _health_cache["cooldowns"][url] = _now() + cooldown_sec
    if _health_cache["url"] == url:
        _health_cache["url"] = None
        _health_cache["resolved_at"] = 0.0
    logger.warning(
        "[MCP] fallback: SearXNG instance %s failed, cooldown %ss",
        url, cooldown_sec,
    )


def get_resolved_searxng() -> Tuple[Optional[str], bool]:
    """Currently chosen (url, via_browser); (None, False) if nothing is cached."""
    return _health_cache["url"], _health_cache["via_browser"]


def reset_searxng_health() -> None:
    """Test hook: clear the session cache."""
    _health_cache.update({"url": None, "via_browser": False, "resolved_at": 0.0, "cooldowns": {}})


async def aresolve_searxng_url(
    primary_url: str,
    fallback_urls: Optional[List[str]] = None,
) -> Tuple[Optional[str], bool]:
    """Pick a working SearXNG base URL and its query path.

    Order: session cache (if fresh) -> primary (local: HTTP probe; public:
    nodriver probe) -> fallbacks (nodriver probe each, skipping cooldowned).
    Returns (url, via_browser). (None, False) means nothing is reachable and
    the caller should disable web_search.
    """
    cached = _health_cache["url"]
    if cached and not _in_cooldown(cached) and (_now() - _health_cache["resolved_at"]) < CACHE_TTL_SEC:
        return cached, _health_cache["via_browser"]

    fallbacks = list(fallback_urls or DEFAULT_FALLBACK_URLS)

    # 1) Primary.
    if not _in_cooldown(primary_url):
        if is_local_url(primary_url):
            if probe_searxng(primary_url):
                _cache_set(primary_url, via_browser=False)
                return primary_url, False
        elif await aprobe_searxng_via_nodriver(primary_url):
            _cache_set(primary_url, via_browser=True)
            return primary_url, True

    # 2) Public fallbacks - browser-only probing (plain HTTP gets blocked there).
    for candidate in fallbacks:
        if candidate == primary_url or _in_cooldown(candidate):
            continue
        if await aprobe_searxng_via_nodriver(candidate):
            _cache_set(candidate, via_browser=True)
            return candidate, True

    return None, False


def _cache_set(url: str, via_browser: bool) -> None:
    _health_cache["url"] = url
    _health_cache["via_browser"] = via_browser
    _health_cache["resolved_at"] = _now()


def encode_fallback_env(fallback_urls: List[str]) -> str:
    """Serialize fallback list for the SEARXNG_FALLBACK_URLS env var."""
    return json.dumps(list(fallback_urls))
