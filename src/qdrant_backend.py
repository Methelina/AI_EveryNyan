"""
Qdrant backend warm-up probe with portable auto-spawn fallback.

Owns the "is Qdrant reachable?" question inside the Python runtime. If the
configured vector_db.url answers /readyz - report OK and move on. If not, and a
portable qdrant.exe exists in bin\\qd\\ (installed by install_ai_everynyan.ps1),
spawn it against the shared storage data\\qdrant_storage and wait. Docker
management stays in run_qdrant.ps1; this module only ever owns a process it
started itself and only tears that one down.

src/qdrant_backend.py
Version:     1.0.1
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.1 (Soror L'.L'.):
  [*] Re-created after repo rollback (2026-09-29). Logic identical to v1.0.0.

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] New module: probe_ready() + ensure_qdrant() with portable exe auto-spawn.
  [+] Tagged loud fallback logging per project fallback policy ([MEMORY] fallback:).
  [+] atexit cleanup limited to a backend this process started (own_backend flag).
"""

import atexit
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from logger import logger

# ---------------------------------------------------------------------------
# Paths resolved against the project root (two levels up from this file:
# src\\qdrant_backend.py -> project root).
# bin\\qd is the mandatory portable install target of the installer;
# data\\qdrant_storage is the shared storage both backends use, so data
# written by the Docker container is picked up by the portable exe untouched.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTABLE_EXE = _PROJECT_ROOT / "bin" / "qd" / "qdrant.exe"
STORAGE_DIR = _PROJECT_ROOT / "data" / "qdrant_storage"
LOG_DIR = _PROJECT_ROOT / "logs"

READY_TIMEOUT_SEC = 60
READY_POLL_SEC = 2

_own_process: Optional[subprocess.Popen] = None


def _ready_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/readyz"


def probe_ready(base_url: str, timeout_sec: float = 3.0) -> bool:
    """Return True if Qdrant answers GET /readyz with HTTP 200."""
    try:
        with urllib.request.urlopen(_ready_url(base_url), timeout=timeout_sec) as resp:
            return resp.status == 200
    except Exception:
        return False


def _spawn_portable() -> subprocess.Popen:
    """Start bin\\qd\\qdrant.exe against the shared storage. Caller must verify
    the exe exists and the port is still free-ish; readiness is awaited by the
    caller via probe_ready."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(STORAGE_DIR)
    out_log = open(LOG_DIR / "qdrant_portable.log", "ab", buffering=0)
    err_log = open(LOG_DIR / "qdrant_portable.err.log", "ab", buffering=0)
    process = subprocess.Popen(
        [str(PORTABLE_EXE)],
        cwd=str(PORTABLE_EXE.parent),
        env=env,
        stdout=out_log,
        stderr=err_log,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    logger.info(
        "[MEMORY] spawned portable Qdrant backend: pid=%s exe=%s storage=%s",
        process.pid, PORTABLE_EXE, STORAGE_DIR,
    )
    return process


def _cleanup_own_backend() -> None:
    """atexit hook: terminate only the qdrant process WE started. A backend
    started by run_qdrant.bat / Docker is never touched."""
    global _own_process
    proc = _own_process
    _own_process = None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
        logger.info("[MEMORY] stopped own portable Qdrant backend (pid was %s)", proc.pid)
    except Exception as exc:  # noqa: BLE001 - cleanup must never raise on exit
        logger.warning("[MEMORY] fallback: failed to stop own Qdrant backend cleanly: %s", exc)
        try:
            proc.kill()
        except Exception as exc2:  # noqa: BLE001
            logger.warning("[MEMORY] fallback: kill() also failed: %s", exc2)


def ensure_qdrant(base_url: str) -> bool:
    """Warm-up guarantee: make Qdrant at base_url reachable before components init.

    Returns True if a backend is answering /readyz. If the URL is silent and
    the portable exe is available, spawns it and waits (logged as a fallback).
    If nothing can be done, logs a loud error with the remedy and returns False
    so the caller can abort with a clear message instead of a raw traceback.
    """
    global _own_process
    if probe_ready(base_url):
        logger.info("[MEMORY] Qdrant backend OK: url=%s", base_url)
        return True

    logger.warning(
        "[MEMORY] fallback: Qdrant at %s is not answering /readyz", base_url
    )

    if PORTABLE_EXE.exists():
        logger.warning(
            "[MEMORY] fallback: spawning portable backend %s (storage=%s)",
            PORTABLE_EXE, STORAGE_DIR,
        )
        try:
            _own_process = _spawn_portable()
        except Exception as exc:
            logger.error(
                "[MEMORY] fallback: failed to spawn portable Qdrant: %s", exc
            )
            return False
        atexit.register(_cleanup_own_backend)

        deadline = time.monotonic() + READY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if _own_process.poll() is not None:
                logger.error(
                    "[MEMORY] fallback: portable Qdrant exited early, code=%s "
                    "(see logs\\qdrant_portable.err.log)",
                    _own_process.returncode,
                )
                return False
            if probe_ready(base_url):
                logger.info(
                    "[MEMORY] portable Qdrant backend ready: url=%s pid=%s",
                    base_url, _own_process.pid,
                )
                return True
            time.sleep(READY_POLL_SEC)

        logger.error(
            "[MEMORY] fallback: portable Qdrant did not become ready in %ss",
            READY_TIMEOUT_SEC,
        )
        return False

    logger.error(
        "[MEMORY] fallback: no Qdrant at %s and no portable exe at %s. "
        "Remedy: run .\\run_qdrant.bat (or reinstall via install_ai_everynyan.ps1).",
        base_url, PORTABLE_EXE,
    )
    return False


def abort_start() -> None:
    """Exit path used by the caller when ensure_qdrant() returned False."""
    sys.exit(
        "Qdrant vector DB is unreachable and no fallback backend is available. "
        "Start it with .\\run_qdrant.bat, then relaunch AI_EveryNyan."
    )
