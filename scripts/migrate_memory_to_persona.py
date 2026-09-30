# -*- coding: utf-8 -*-
"""
One-time migration of the shared legacy memory store to the persona layout.

Moves data/history.db (+ .wal/.dmp sidecars) into data/memory/EveryNyan/,
backing up the originals into temp/backup/<timestamp>/ first. Idempotent:
skips when the target already exists. The Qdrant collection needs no
migration - the default persona (EveryNyan) keeps the legacy collection name
by design (see runtime.persona_collection()).

scripts/migrate_memory_to_persona.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-30

Patch Notes v1.0.0 (Soror L.'.L.'.):
  [+] Initial migration script for the per-persona memory layout.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from logger import logger  # noqa: E402

PERSONA = "EveryNyan"
SIDECARS = (".wal", ".dmp", ".tmp")


def main() -> int:
    src = PROJECT / "data" / "history.db"
    dst_dir = PROJECT / "data" / "memory" / PERSONA
    dst = dst_dir / "history.db"

    if dst.exists():
        logger.info("[MIGRATE] Target %s already exists - nothing to do.", dst)
        return 0
    if not src.exists():
        logger.warning(
            "[MIGRATE] fallback: source %s not found (fresh install?) - nothing to migrate.", src
        )
        return 0

    backup_dir = PROJECT / "temp" / "backup" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, backup_dir / src.name)
    for ext in SIDECARS:
        sidecar = src.with_suffix(src.suffix + ext)
        if sidecar.exists():
            shutil.copy2(sidecar, backup_dir / sidecar.name)
    logger.info("[MIGRATE] Backup copied to %s", backup_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    for ext in SIDECARS:
        sidecar = src.with_suffix(src.suffix + ext)
        if sidecar.exists():
            shutil.move(str(sidecar), dst.with_suffix(dst.suffix + ext))
    logger.info("[MIGRATE] Moved %s -> %s", src, dst)
    logger.info("[MIGRATE] Done. Qdrant collection stays as-is (default persona keeps the legacy name).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
