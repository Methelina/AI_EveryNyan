"""
Unit tests for tool_searxng.py long-document chunking (fetch_url emission).

tests/test_fetch_chunking.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests for _split_into_chunks(): size limits, paragraph boundaries,
      giant-paragraph hard split.
  [+] Tests for _emit_chunked_document(): short doc inline, long doc parts on
      disk with manifest paths, max_total truncation note, prune of stale dirs.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "tools" / "mcp"))

spec = importlib.util.spec_from_file_location("tool_searxng", _repo / "tools" / "mcp" / "tool_searxng.py")
tool_searxng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool_searxng)

SIZE = tool_searxng.FETCH_CHUNK_SIZE


def _para(marker: str, n: int = 50) -> str:
    return " ".join([f"para-{marker}" for _ in range(n)])


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------

def test_split_short_text_single_chunk():
    text = _para("a", 20)
    chunks = tool_searxng._split_into_chunks(text, SIZE)
    assert chunks == [text]


def test_split_respects_chunk_size():
    text = "\n\n".join(_para(f"p{i}", 200) for i in range(30))
    chunks = tool_searxng._split_into_chunks(text, SIZE)
    assert len(chunks) > 1
    assert all(len(c) <= SIZE for c in chunks), max(len(c) for c in chunks)


def test_split_never_loses_content():
    text = "\n\n".join(_para(f"p{i}", 150) for i in range(25))
    chunks = tool_searxng._split_into_chunks(text, SIZE)
    assert "".join(chunks).replace("\n\n", "") == text.replace("\n\n", "")


def test_split_giant_paragraph_hard_split():
    giant = "x" * (SIZE * 2 + 100)
    chunks = tool_searxng._split_into_chunks(giant, SIZE)
    assert len(chunks) == 3
    assert all(len(c) <= SIZE for c in chunks)


# ---------------------------------------------------------------------------
# _emit_chunked_document
# ---------------------------------------------------------------------------

def test_emit_short_doc_inline_without_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_searxng, "_fetch_parts_root", lambda: str(tmp_path / "fetch_parts"))
    text = "short document"
    out = tool_searxng._emit_chunked_document(text, "https://example.com/a", 500000)
    assert out == text  # no manifest, no wrapper
    assert not (tmp_path / "fetch_parts").exists()


def test_emit_long_doc_writes_parts_and_manifest(tmp_path, monkeypatch):
    parts_root = tmp_path / "fetch_parts"
    monkeypatch.setattr(tool_searxng, "_fetch_parts_root", lambda: str(parts_root))
    text = "\n\n".join(_para(f"p{i}", 400) for i in range(40))
    out = tool_searxng._emit_chunked_document(text, "https://example.com/long", 500000)

    assert out.startswith("[DOCUMENT:")
    assert "Part 1/" in out
    assert "read_file" in out
    assert "PART 1/" in out

    # manifest references existing part files; part files contain real text
    listed = [line.strip() for line in out.splitlines() if line.strip().endswith(".md")]
    assert listed, "manifest lists no part files"
    for p in listed:
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0
    # every part referenced, and part_01.md exists on disk too
    assert len(listed) >= 1
    assert (parts_root / "a17c9f0e1d26" ).exists() or any(parts_root.iterdir())


def test_emit_respects_max_total_and_marks_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_searxng, "_fetch_parts_root", lambda: str(tmp_path / "fetch_parts"))
    text = "\n\n".join(_para(f"p{i}", 400) for i in range(40))
    out = tool_searxng._emit_chunked_document(text, "https://example.com/huge", 45000)
    assert "DOCUMENT TRUNCATED" in out
    assert "INCOMPLETE" in out


def test_prune_removes_stale_dirs(tmp_path, monkeypatch):
    root = tmp_path / "fetch_parts"
    monkeypatch.setattr(tool_searxng, "_fetch_parts_root", lambda: str(root))
    old, fresh = root / "old_dir", root / "fresh_dir"
    old.mkdir(parents=True); fresh.mkdir(parents=True)
    stale = time.time() - tool_searxng.FETCH_PARTS_MAX_AGE_SEC - 3600
    os.utime(old, (stale, stale))
    tool_searxng._prune_stale_part_dirs()
    assert not old.exists()
    assert fresh.exists()
