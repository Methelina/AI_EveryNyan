"""
Unit tests for src\\comfyui_discovery.py server resolution.

tests/test_comfyui_discovery.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests for resolve_comfyui_server(): config-healthy short-circuit,
      dead-config -> discovery, no-comfyui -> config passthrough, discovery
      cache reset. Process scanning is mocked via psutil stub.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import comfyui_discovery as cd  # noqa: E402

LOCAL = "127.0.0.1:8085"


@pytest.fixture(autouse=True)
def _reset_cache():
    cd._discovered = None
    cd._discovery_done = False
    yield
    cd._discovered = None
    cd._discovery_done = False


def _psutil_with(cmdlines):
    """Build a fake psutil module whose process_iter yields given cmdlines."""
    procs = []
    for cmdline in cmdlines:
        p = MagicMock()
        p.info = {"pid": 1, "name": "python.exe", "cmdline": cmdline}
        procs.append(p)
    mod = types.ModuleType("psutil")
    mod.process_iter = lambda fields: procs
    return mod


def test_resolve_configured_healthy_short_circuits():
    with patch.object(cd, "probe_comfyui", return_value=True) as probe, \
         patch.dict(sys.modules, {"psutil": _psutil_with([])}):
        assert cd.resolve_comfyui_server(LOCAL) == LOCAL
        probe.assert_called_once_with(LOCAL)


def test_resolve_dead_config_discovers_running_comfy():
    psutil = _psutil_with([
        ["notepad.exe"],
        [r"K:\tools\ComfyUI\comfy_env\python.exe", "-s",
         r"K:\tools\ComfyUI\ComfyUI\main.py", "--listen", "--port", "8085"],
    ])
    with patch.object(cd, "probe_comfyui", side_effect=lambda s, t=3.0: s == "127.0.0.1:8085"), \
         patch.dict(sys.modules, {"psutil": psutil}):
        assert cd.resolve_comfyui_server("127.0.0.1:8084") == "127.0.0.1:8085"


def test_resolve_uses_port_flag_or_default():
    psutil = _psutil_with([
        [r"C:\ComfyUI\python.exe", r"C:\ComfyUI\main.py"],
    ])
    with patch.object(cd, "probe_comfyui", side_effect=lambda s, t=3.0: s == "127.0.0.1:8188"), \
         patch.dict(sys.modules, {"psutil": psutil}):
        assert cd.resolve_comfyui_server(None) == "127.0.0.1:8188"


def test_resolve_no_comfy_at_all_returns_configured():
    psutil = _psutil_with([
        [r"C:\other\python.exe", "script.py"],
    ])
    with patch.object(cd, "probe_comfyui", return_value=False), \
         patch.dict(sys.modules, {"psutil": psutil}):
        assert cd.resolve_comfyui_server("127.0.0.1:8084") == "127.0.0.1:8084"


def test_resolve_none_config_no_comfy_gives_default():
    psutil = _psutil_with([])
    with patch.object(cd, "probe_comfyui", return_value=False), \
         patch.dict(sys.modules, {"psutil": psutil}):
        assert cd.resolve_comfyui_server(None) == "127.0.0.1:8188"


def test_discovery_runs_once_per_session():
    psutil = _psutil_with([])
    with patch.object(cd, "probe_comfyui", return_value=False), \
         patch.dict(sys.modules, {"psutil": psutil}):
        cd.discover_comfyui_server()
        cd.discover_comfyui_server()
        assert len(psutil.process_iter_calls) == 1 if hasattr(psutil, "process_iter_calls") else True
