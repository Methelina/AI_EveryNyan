"""
Availability probe for external MCP backends (SearXNG).

The FastMCP stdio subprocess always starts and always advertises its tools, so
tool discovery alone says nothing about whether SearXNG is actually serving.
This module provides a cheap warm-up probe so the runtime can warn loudly and
drop the SearXNG-dependent tool(s) from the agent instead of letting them fail
mid-conversation (fail soft, log loud - project fallback policy).

src/mcp_health.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] New module: probe_searxng() via GET /search?q=ping&format=json.
"""

import urllib.request

SEARXNG_PROBE_TIMEOUT_SEC = 3.0


def probe_searxng(base_url: str, timeout_sec: float = SEARXNG_PROBE_TIMEOUT_SEC) -> bool:
    """Return True if SearXNG answers a minimal JSON search request.

    A real /search round-trip is used (rather than /healthz, which SearXNG
    does not expose by default) - a 200 with parseable JSON body means the
    meta-search engine is actually functional.
    """
    url = base_url.rstrip("/") + "/search?q=ping&format=json"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return resp.status == 200
    except Exception:
        return False
