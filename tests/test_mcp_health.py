"""
Unit tests for src\\mcp_health.py SearXNG availability probe and resolver.

tests/test_mcp_health.py
Version:     1.1.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.1.0 (Soror L'.L'.):
  [+] Tests for is_local_url() and aresolve_searxng_url(): local-direct,
      fallback-via-browser, cooldown rotation, cache reuse, all-dead paths.

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests for probe_searxng(): reachable / refused / HTTP error / non-200 paths.
"""

import asyncio
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mcp_health  # noqa: E402

URL = "http://localhost:2597"


@pytest.fixture(autouse=True)
def _reset_health():
    mcp_health.reset_searxng_health()
    yield
    mcp_health.reset_searxng_health()


def _resp(status):
    r = MagicMock()
    r.status = status
    r.__enter__ = lambda s: r
    r.__exit__ = MagicMock(return_value=False)
    return r


def test_probe_true_on_200():
    with patch("urllib.request.urlopen", return_value=_resp(200)):
        assert mcp_health.probe_searxng(URL) is True


def test_probe_normalizes_trailing_slash():
    with patch("urllib.request.urlopen", return_value=_resp(200)) as op:
        assert mcp_health.probe_searxng(URL + "/") is True
        called = str(op.call_args[0][0])
        assert re.fullmatch(r"http://localhost:2597/search\?q=ping&format=json", called)


def test_probe_false_on_connection_refused():
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        assert mcp_health.probe_searxng(URL, timeout_sec=0.1) is False


def test_probe_false_on_http_error():
    import urllib.error
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.HTTPError(URL, 500, "ISE", {}, None)):
        assert mcp_health.probe_searxng(URL, timeout_sec=0.1) is False


def test_probe_false_on_non_200_status():
    with patch("urllib.request.urlopen", return_value=_resp(503)):
        assert mcp_health.probe_searxng(URL, timeout_sec=0.1) is False


# ---------------------------------------------------------------------------
# is_local_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("http://localhost:2597", True),
    ("http://127.0.0.1:8080", True),
    ("http://[::1]:2597", True),
    ("https://searx.be", False),
    ("https://search.mectov.my.id", False),
    ("not a url", False),
])
def test_is_local_url(url, expected):
    assert mcp_health.is_local_url(url) is expected


# ---------------------------------------------------------------------------
# aresolve_searxng_url
# ---------------------------------------------------------------------------

LOCAL = "http://localhost:2597"
PUB1 = "https://search.mectov.my.id"
PUB2 = "https://sx.xo.st"


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_resolve_local_direct():
    with patch.object(mcp_health, "probe_searxng", return_value=True) as p, \
         patch.object(mcp_health, "aprobe_searxng_via_nodriver", new=AsyncMock()) as np:
        url, via_browser = _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1]))
        assert (url, via_browser) == (LOCAL, False)
        p.assert_called_once_with(LOCAL)
        np.assert_not_awaited()


def test_resolve_falls_back_to_public_via_browser():
    with patch.object(mcp_health, "probe_searxng", return_value=False), \
         patch.object(mcp_health, "aprobe_searxng_via_nodriver", new=AsyncMock(side_effect=[True])) as np:
        url, via_browser = _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1, PUB2]))
        assert (url, via_browser) == (PUB1, True)
        assert np.await_count == 1  # stops at first live candidate


def test_resolve_skips_cooldowned_candidates():
    mcp_health.mark_searxng_failure(PUB1, cooldown_sec=60)
    with patch.object(mcp_health, "probe_searxng", return_value=False), \
         patch.object(mcp_health, "aprobe_searxng_via_nodriver", new=AsyncMock(side_effect=[True])) as np:
        url, _ = _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1, PUB2]))
        assert url == PUB2
        # PUB1 skipped: only PUB2 probed
        np.assert_awaited_once_with(PUB2)


def test_resolve_all_dead_returns_none():
    with patch.object(mcp_health, "probe_searxng", return_value=False), \
         patch.object(mcp_health, "aprobe_searxng_via_nodriver", new=AsyncMock(return_value=False)):
        assert _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1])) == (None, False)


def test_resolve_uses_session_cache_without_reprobing():
    with patch.object(mcp_health, "probe_searxng", return_value=True) as p:
        _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1]))
        _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1]))
        assert p.call_count == 1  # second call served from cache


def test_resolve_after_failure_rotates_to_next():
    with patch.object(mcp_health, "probe_searxng", return_value=False), \
         patch.object(mcp_health, "aprobe_searxng_via_nodriver",
                      new=AsyncMock(side_effect=[True, True])) as np:
        url1, _ = _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1, PUB2]))
        assert url1 == PUB1
        mcp_health.mark_searxng_failure(PUB1)
        url2, _ = _run(mcp_health.aresolve_searxng_url(LOCAL, [PUB1, PUB2]))
        assert url2 == PUB2
        assert np.await_count == 2  # PUB1 re-probed once after cache drop, then PUB2


def test_encode_fallback_env_roundtrip():
    import json
    assert json.loads(mcp_health.encode_fallback_env([PUB1, PUB2])) == [PUB1, PUB2]
