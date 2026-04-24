"""
Exception logging utility for debugging unhandled errors.
Enabled by calling install_excepthook() (typically when AppSettings.debug is True).
Logs full tracebacks with frame locals to logs/debug_exceptions.log.

src\logging_exceptions.py
Version:     0.2.0
Author:      pytraveler
Updated:     2026-04-24

Patch Notes v0.2.0 [pytraveler]:
  [+] Added frame locals dump: _format_frame_locals walks tb_next chain and
      logs all local variables (including function arguments) per frame.
  [+] Last frame (crash site) is highlighted with '>>> ... <-- CRASH' marker.
  [+] _safe_repr with 500-char limit prevents log bloat on large objects.
  [+] Console output preserved: original sys.excepthook is called after logging.
  [+] Now controlled by AppSettings.debug via install_excepthook() in main.py.
  [-] Removed DEBUG_EXCEPTIONS env var dependency.

Patch Notes v0.1.0:
  [+] Initial release: excepthook that logs to logs/debug_exceptions.log.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

_exception_log_path: str = "logs/debug_exceptions.log"
_MAX_REPR_LEN = 500


def _safe_repr(value) -> str:
    try:
        r = repr(value)
    except Exception:
        return "<repr() failed>"
    if len(r) > _MAX_REPR_LEN:
        r = r[:_MAX_REPR_LEN] + f"... ({len(r)} chars total)"
    return r


def _format_frame_locals(tb) -> str:
    if tb is None:
        return ""

    frames = []
    current = tb
    while current is not None:
        frames.append(current)
        current = current.tb_next

    lines: list[str] = []
    for depth, frame_tb in enumerate(frames):
        frame = frame_tb.tb_frame
        code = frame.f_code
        locals_map = {k: v for k, v in frame.f_locals.items() if not k.startswith("__")}
        if not locals_map:
            continue

        is_last = frame_tb is frames[-1]
        prefix = ">>>" if is_last else "   "
        crash_tag = "  <-- CRASH" if is_last else ""

        lines.append(
            f"{prefix} [{depth}] {code.co_filename}:{frame_tb.tb_lineno}"
            f" in {code.co_name}{crash_tag}"
        )
        for name, val in locals_map.items():
            lines.append(f"        {name} = {_safe_repr(val)}")

    return "\n".join(lines) if lines else ""


def _write_exception(exc_type, exc_value, exc_tb, source: str = ""):
    try:
        Path(_exception_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(_exception_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Timestamp : {datetime.now().isoformat()}\n")
            if source:
                f.write(f"Source    : {source}\n")
            f.write(f"Type      : {exc_type.__name__}\n")
            f.write(f"Value     : {exc_value}\n")
            f.write("Traceback :\n")
            if exc_tb:
                f.write("".join(traceback.format_tb(exc_tb)))
            else:
                f.write(traceback.format_exc())
            locals_section = _format_frame_locals(exc_tb)
            if locals_section:
                f.write("\nLocals per frame :\n")
                f.write(locals_section)
                f.write("\n")
            f.write(f"{'=' * 70}\n")
    except Exception:
        pass


def _excepthook_handler(exc_type, exc_value, exc_tb):
    _write_exception(exc_type, exc_value, exc_tb, source="sys.excepthook (unhandled)")
    _original_excepthook(exc_type, exc_value, exc_tb)


_original_excepthook = sys.excepthook


def install_excepthook():
    sys.excepthook = _excepthook_handler


def uninstall_excepthook():
    sys.excepthook = _original_excepthook
