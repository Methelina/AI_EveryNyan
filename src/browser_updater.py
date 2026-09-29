"""
Chromium auto-update for the project-isolated playwright_browsers install.

Browser revisions are pinned to the installed playwright package version, so
"update the browser" means: upgrade playwright via pip, re-run
`python -m playwright install chromium`, smoke-test the new browser, then prune
stale revision directories. pip and the playwright installer print their own
visible progress bars - output is streamed to the console, never captured.

Triggered from two places, both idempotent:
- the installer (install_ai_everynyan.ps1) on fresh installs;
- the runtime at startup (main.py), gated by config
  (browser_update.enabled, browser_update.check_interval_days) and a
  last-check stamp in temp\\ so the PyPI check does not run every launch.

Failure policy: every step is guarded - on any error the update is skipped
with a loud [INSTALL]/[APP] tagged warning and the app continues with the
currently installed browsers (old revision dirs are only pruned AFTER a
successful smoke test).

src/browser_updater.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] New module: update_browsers() - pip upgrade + playwright install +
      smoke test + stale revision pruning, streamed visible progress.
  [+] get_pypi_latest_version(), _current_revision() helpers, check-interval
      stamp cache in temp\\.
"""

import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from logger import logger

PYPI_JSON_URL = "https://pypi.org/pypi/playwright/json"
STAMP_FILE = Path(__file__).resolve().parent.parent / "temp" / "browser_update_stamp.json"
BROWSERS_DIR = Path(__file__).resolve().parent.parent / "playwright_browsers"
SMOKE_TEST_URL = "https://example.com"


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _parse_version(v: str) -> tuple:
    """'1.58.0' -> (1, 58, 0); tolerant of suffixes like '1.58.0b1'."""
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) if v else ()


def get_installed_playwright_version(python_exe: str) -> Optional[str]:
    try:
        out = subprocess.run(
            [python_exe, "-m", "pip", "show", "playwright"],
            capture_output=True, text=True, timeout=60,
        )
        match = re.search(r"^Version:\s*(.+)$", out.stdout, re.MULTILINE)
        return match.group(1).strip() if match else None
    except Exception as exc:
        logger.warning("[INSTALL] fallback: pip show playwright failed: %s", exc)
        return None


def get_pypi_latest_version(timeout_sec: float = 10.0) -> Optional[str]:
    try:
        req = urllib.request.Request(
            PYPI_JSON_URL, headers={"User-Agent": "AI_EveryNyan-browser-updater"}
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("info", {}).get("version")
    except Exception as exc:
        logger.warning("[INSTALL] fallback: PyPI version check failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Revision helpers
# ---------------------------------------------------------------------------

def _current_revision() -> Optional[str]:
    """Browser revision pinned by the installed playwright (e.g. '1208')."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium", "--dry-run"],
            capture_output=True, text=True, timeout=120,
        )
        match = re.search(r"playwright chromium v(\d+)", out.stdout)
        return match.group(1) if match else None
    except Exception as exc:
        logger.warning("[INSTALL] fallback: dry-run revision query failed: %s", exc)
        return None


def _prune_stale_revisions(keep_revision: Optional[str]) -> None:
    """Remove chromium*/ffmpeg*/winldd* dirs whose revision differs from the
    current one. Called only after a successful smoke test."""
    if not keep_revision or not BROWSERS_DIR.exists():
        return
    for pattern in ("chromium-*", "chromium_headless_shell-*", "ffmpeg-*", "winldd-*"):
        for path in BROWSERS_DIR.glob(pattern):
            rev = path.name.rsplit("-", 1)[-1]
            if rev != keep_revision:
                size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) // (1024 * 1024)
                shutil.rmtree(path, ignore_errors=True)
                logger.info("[INSTALL] pruned stale browser dir %s (freed ~%s MB)", path.name, size_mb)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test(python_exe: str, browsers_path: Path) -> bool:
    """Real headless launch + navigation against the NEW install."""
    script = (
        "from playwright.sync_api import sync_playwright\n"
        "with sync_playwright() as p:\n"
        "    b = p.chromium.launch(headless=True)\n"
        f"    pg = b.new_page(); pg.goto('{SMOKE_TEST_URL}', timeout=20000)\n"
        "    assert pg.title() is not None\n"
        "    b.close()\n"
        "print('SMOKE_OK')\n"
    )
    env_browsers = str(browsers_path)
    try:
        out = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True, text=True, timeout=90,
            env={**__import__("os").environ, "PLAYWRIGHT_BROWSERS_PATH": env_browsers},
        )
        ok = "SMOKE_OK" in out.stdout
        if not ok:
            logger.warning(
                "[INSTALL] fallback: browser smoke test failed (rc=%s): %s",
                out.returncode, (out.stderr or "")[-300:],
            )
        return ok
    except Exception as exc:
        logger.warning("[INSTALL] fallback: browser smoke test crashed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Check-interval stamp
# ---------------------------------------------------------------------------

def _check_due(interval_days: int) -> bool:
    try:
        if STAMP_FILE.exists():
            data = json.loads(STAMP_FILE.read_text(encoding="utf-8"))
            if time.time() - data.get("last_check", 0) < interval_days * 86400:
                return False
    except Exception:
        pass
    return True


def _write_stamp() -> None:
    try:
        STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        STAMP_FILE.write_text(json.dumps({"last_check": time.time()}), encoding="utf-8")
    except Exception as exc:
        logger.debug("[INSTALL] could not write update stamp: %s", exc)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _stream_run(cmd: list) -> int:
    """Run a subprocess streaming output straight to the console so pip /
    playwright download progress bars stay visible (never captured)."""
    proc = subprocess.Popen(cmd)
    return proc.wait()


def update_browsers(python_exe: str, browsers_path: Path) -> bool:
    """Full update cycle: pip upgrade -> browsers install -> smoke -> prune.
    Returns True if a new version was installed and verified."""
    installed = get_installed_playwright_version(python_exe)
    latest = get_pypi_latest_version()
    if not installed or not latest:
        logger.info("[INSTALL] browser update skipped: version unknown (installed=%s, latest=%s)", installed, latest)
        return False
    if _parse_version(latest) <= _parse_version(installed):
        logger.info("[INSTALL] playwright %s is up to date (PyPI: %s) - browsers untouched", installed, latest)
        return False

    print(f"\n[UPDATE] playwright {installed} -> {latest}: browsers will be updated (~150 MB, progress below)\n", flush=True)
    if _stream_run([python_exe, "-m", "pip", "install", "--upgrade", "playwright", "--no-warn-script-location"]) != 0:
        logger.warning("[INSTALL] fallback: pip upgrade failed - keeping playwright %s", installed)
        return False

    env_browsers = str(browsers_path)
    import os
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = env_browsers
    if _stream_run([python_exe, "-m", "playwright", "install", "chromium"]) != 0:
        logger.warning(
            "[INSTALL] fallback: browser download failed. Rollback: pip install playwright==%s", installed
        )
        return False

    new_rev = _current_revision()
    if not _smoke_test(python_exe, browsers_path):
        logger.warning(
            "[INSTALL] fallback: new browser FAILED smoke test. Old revision dirs kept. "
            "Rollback: pip install playwright==%s", installed,
        )
        return False

    _prune_stale_revisions(new_rev)
    logger.info(
        "[INSTALL] browsers updated: playwright %s -> %s (chromium rev %s, smoke test OK)",
        installed, latest, new_rev,
    )
    return True


def maybe_update_browsers(
    python_exe: str,
    browsers_path: Path,
    enabled: bool = True,
    check_interval_days: int = 3,
) -> None:
    """Runtime startup hook: gated, never raises, never blocks longer than the
    update itself takes (visible progress streams to the console)."""
    if not enabled:
        return
    if not _check_due(check_interval_days):
        return
    try:
        update_browsers(python_exe, browsers_path)
    except Exception as exc:
        logger.warning("[INSTALL] fallback: browser update crashed (app continues): %s", exc)
    finally:
        _write_stamp()
