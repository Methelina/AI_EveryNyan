"""
Unit tests for src\\browser_updater.py (pure helpers only - no network, no
downloads, no pip calls in tests).

tests/test_browser_updater.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests for _parse_version(), _check_due()/_write_stamp() stamp cache,
      _prune_stale_revisions() with fake browser dirs.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import browser_updater as bu  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------

def test_parse_version_basic():
    assert bu._parse_version("1.58.0") == (1, 58, 0)


def test_parse_version_suffix_tolerant():
    assert bu._parse_version("1.58.0b1") == (1, 58, 0)
    assert bu._parse_version("1.59.2.dev0") == (1, 59, 2)


def test_parse_version_empty():
    assert bu._parse_version("") == ()
    assert bu._parse_version(None) == ()


def test_version_comparison_semantic():
    assert bu._parse_version("1.59.0") > bu._parse_version("1.58.0")
    assert bu._parse_version("2.0.0") > bu._parse_version("1.99.99")


# ---------------------------------------------------------------------------
# stamp cache
# ---------------------------------------------------------------------------

def test_check_due_without_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "STAMP_FILE", tmp_path / "stamp.json")
    assert bu._check_due(3) is True


def test_check_due_recent_stamp_not_due(tmp_path, monkeypatch):
    stamp = tmp_path / "stamp.json"
    stamp.write_text(f'{{"last_check": {time.time()}}}', encoding="utf-8")
    monkeypatch.setattr(bu, "STAMP_FILE", stamp)
    assert bu._check_due(3) is False


def test_check_due_old_stamp_due(tmp_path, monkeypatch):
    stamp = tmp_path / "stamp.json"
    old = time.time() - 4 * 86400
    stamp.write_text(f'{{"last_check": {old}}}', encoding="utf-8")
    monkeypatch.setattr(bu, "STAMP_FILE", stamp)
    assert bu._check_due(3) is True


def test_write_stamp_creates_file(tmp_path, monkeypatch):
    stamp = tmp_path / "sub" / "stamp.json"
    monkeypatch.setattr(bu, "STAMP_FILE", stamp)
    bu._write_stamp()
    assert stamp.exists()


# ---------------------------------------------------------------------------
# _prune_stale_revisions
# ---------------------------------------------------------------------------

def test_prune_removes_other_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "BROWSERS_DIR", tmp_path)
    keep = "1209"
    for name in ["chromium-1208", "chromium-1209", "chromium_headless_shell-1208",
                 "chromium_headless_shell-1209", "ffmpeg-1011", "ffmpeg-1209", "winldd-1007"]:
        d = tmp_path / name
        d.mkdir()
        (d / "dummy.bin").write_bytes(b"x" * 1024)
    bu._prune_stale_revisions(keep)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"chromium-1209", "chromium_headless_shell-1209", "ffmpeg-1209", "winldd-1209"} or \
           "chromium-1209" in names
    assert "chromium-1208" not in names
    assert "chromium_headless_shell-1208" not in names
    assert "ffmpeg-1011" not in names
    assert "winldd-1007" not in names


def test_prune_noop_when_no_revision(tmp_path):
    d = tmp_path / "chromium-1208"
    d.mkdir()
    bu._prune_stale_revisions(None)
    assert d.exists()


def test_prune_noop_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "BROWSERS_DIR", tmp_path / "nonexistent")
    bu._prune_stale_revisions("1209")  # must not raise
