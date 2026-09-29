"""
Unit tests for src\\qdrant_backend.py warm-up probe and portable auto-spawn.

tests/test_qdrant_backend.py
Version:     1.0.1
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.1 (Soror L'.L'.):
  [*] Re-created after repo rollback (2026-09-29). Logic identical to v1.0.0.

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests for probe_ready() and ensure_qdrant(): ready / spawn-success /
      spawn-early-exit / no-exe paths (all external effects mocked).
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import qdrant_backend  # noqa: E402

URL = "http://localhost:6333"


@pytest.fixture(autouse=True)
def _reset_module_state():
    qdrant_backend._own_process = None
    yield
    qdrant_backend._own_process = None


def test_probe_ready_true_when_200():
    resp = MagicMock()
    resp.status = 200
    resp.__enter__ = lambda s: resp
    resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=resp):
        assert qdrant_backend.probe_ready(URL) is True


def test_probe_ready_false_on_connection_error():
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        assert qdrant_backend.probe_ready(URL, timeout_sec=0.1) is False


def test_ensure_ready_backend_no_spawn():
    with patch.object(qdrant_backend, "probe_ready", return_value=True) as probe, \
         patch.object(qdrant_backend, "_spawn_portable") as spawn:
        assert qdrant_backend.ensure_qdrant(URL) is True
        spawn.assert_not_called()
        assert probe.call_count == 1


def test_ensure_spawns_portable_when_url_silent():
    fake_proc = MagicMock(spec=subprocess.Popen)
    fake_proc.poll.return_value = None
    fake_proc.pid = 1234
    with patch.object(qdrant_backend, "probe_ready", side_effect=[False, True]), \
         patch.object(qdrant_backend, "PORTABLE_EXE") as exe, \
         patch.object(qdrant_backend, "_spawn_portable", return_value=fake_proc) as spawn, \
         patch.object(qdrant_backend, "atexit") as atexit_mock, \
         patch.object(qdrant_backend.time, "sleep"):
        exe.exists.return_value = True
        assert qdrant_backend.ensure_qdrant(URL) is True
        spawn.assert_called_once()
        atexit_mock.register.assert_called_once()


def test_ensure_fails_when_portable_exits_early():
    fake_proc = MagicMock(spec=subprocess.Popen)
    fake_proc.poll.return_value = 1
    fake_proc.returncode = 1
    with patch.object(qdrant_backend, "probe_ready", return_value=False), \
         patch.object(qdrant_backend, "PORTABLE_EXE") as exe, \
         patch.object(qdrant_backend, "_spawn_portable", return_value=fake_proc), \
         patch.object(qdrant_backend, "atexit"), \
         patch.object(qdrant_backend.time, "sleep"):
        exe.exists.return_value = True
        assert qdrant_backend.ensure_qdrant(URL) is False


def test_ensure_fails_without_exe_and_unreachable_url():
    with patch.object(qdrant_backend, "probe_ready", return_value=False), \
         patch.object(qdrant_backend, "PORTABLE_EXE") as exe, \
         patch.object(qdrant_backend, "_spawn_portable") as spawn:
        exe.exists.return_value = False
        assert qdrant_backend.ensure_qdrant(URL) is False
        spawn.assert_not_called()


def test_cleanup_terminates_only_own_process():
    own = MagicMock(spec=subprocess.Popen)
    own.poll.return_value = None
    qdrant_backend._own_process = own
    qdrant_backend._cleanup_own_backend()
    own.terminate.assert_called_once()
    assert qdrant_backend._own_process is None
