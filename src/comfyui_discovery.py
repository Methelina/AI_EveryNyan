"""
ComfyUI server resolution: config-first with process-scan auto-discovery.

Primary source of truth is config\\settings.yaml (comfyui.server). When that
section is missing or the configured endpoint is not answering, the fallback
scans running python processes: any python.exe whose command line mentions
comfyui (path or args) IS ComfyUI for practical purposes; the port comes from
--port (default 8188) and the candidate is verified over HTTP /system_stats.

Both the monitor daemon (src\\comfyui_monitor.py) and the MCP tool
(tools\\mcp\\tool_comfyui.py) resolve through this module, so a port change in
the ComfyUI launcher never desynchronises them again.

src/comfyui_discovery.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] New module: resolve_comfyui_server() + discover_comfyui_server()
      (psutil process scan, --port extraction, HTTP verification, session cache).
"""

import re
import urllib.request
from typing import Optional

from logger import logger

DISCOVERY_HTTP_TIMEOUT_SEC = 3.0

# Session cache: discovery scans all processes - do it once per process life.
_discovered: Optional[str] = None
_discovery_done = False


def probe_comfyui(server: str, timeout_sec: float = DISCOVERY_HTTP_TIMEOUT_SEC) -> bool:
    """True if http://<server>/system_stats answers with JSON."""
    try:
        with urllib.request.urlopen(f"http://{server}/system_stats", timeout=timeout_sec) as resp:
            return resp.status == 200
    except Exception:
        return False


def discover_comfyui_server() -> Optional[str]:
    """Scan python processes for a ComfyUI instance; return 'host:port' or None.

    Heuristic: python.exe whose full command line contains 'comfyui' (the repo
    path usually does) and 'main.py'. Port from --port, else ComfyUI default
    8188. The candidate is verified with a real /system_stats round-trip.
    """
    global _discovered, _discovery_done
    if _discovery_done:
        return _discovered
    _discovery_done = True

    try:
        import psutil
    except ImportError:
        logger.warning("[COMFYUI] fallback: psutil not installed - auto-discovery unavailable")
        return None

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = " ".join(info.get("cmdline") or []).lower()
            if "comfyui" not in cmdline or "main.py" not in cmdline:
                continue
            port_match = re.search(r"--port[ =](\d+)", cmdline)
            port = int(port_match.group(1)) if port_match else 8188
            candidate = f"127.0.0.1:{port}"
            if probe_comfyui(candidate):
                _discovered = candidate
                logger.info(
                    "[COMFYUI] auto-discovered backend: pid=%s server=%s", info["pid"], candidate
                )
                return candidate
            logger.debug("[COMFYUI] candidate %s (pid %s) did not answer", candidate, info["pid"])
        except Exception:  # noqa: BLE001 - skip processes we cannot inspect
            continue

    logger.warning("[COMFYUI] fallback: no running ComfyUI process found")
    return None


def resolve_comfyui_server(configured: Optional[str]) -> str:
    """Config-first resolution.

    configured set AND answering -> use it. Otherwise auto-discover (loud
    fallback log when the configured value is dead). Returns a 'host:port'
    string in all cases (discovery failure -> the configured value as-is, so
    the caller keeps its reconnect loop semantics).
    """
    if configured and probe_comfyui(configured):
        return configured

    if configured:
        logger.warning(
            "[COMFYUI] fallback: configured server %s is not answering - auto-discovering",
            configured,
        )
    else:
        logger.info("[COMFYUI] no server in settings - auto-discovering")

    found = discover_comfyui_server()
    if found:
        if configured and found != configured:
            logger.warning(
                "[COMFYUI] fallback: using auto-discovered %s instead of configured %s "
                "- fix comfyui.server in config\\settings.yaml to silence this",
                found, configured,
            )
        return found

    return configured or "127.0.0.1:8188"
