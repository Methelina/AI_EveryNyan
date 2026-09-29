"""
Unit tests for memory_manager.format_timestamp().

tests/test_format_timestamp.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests: datetime object, ISO string, None, garbage -> empty prefix.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_manager import format_timestamp  # noqa: E402


def test_datetime_object():
    assert format_timestamp(datetime(2026, 9, 29, 19, 5)) == "[2026-09-29 19:05]"


def test_iso_string():
    assert format_timestamp("2026-09-29T19:05:33.123456") == "[2026-09-29 19:05]"


def test_none_and_garbage_return_empty():
    assert format_timestamp(None) == ""
    assert format_timestamp("not a date") == ""
    assert format_timestamp(12345) == ""
