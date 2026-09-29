"""
Unit tests for src\\mcp_health.py SearXNG availability probe.

tests/test_mcp_health.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests for probe_searxng(): reachable / refused / HTTP error / non-200 paths.
"""

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mcp_health  # noqa: E402

URL = "http://localhost:2597"


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
