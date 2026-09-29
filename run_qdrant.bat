:: Qdrant launcher wrapper for AI_EveryNyan - delegates to run_qdrant.ps1.
:: run_qdrant.bat
:: Version:     2.0.0
:: Author:      Soror L.'.L.'.
:: Updated:     2026-09-29
::
:: Patch Notes v2.0.0 (Soror L'.L'.):
::   [+] Replaced inline Docker-only logic with a thin wrapper over run_qdrant.ps1
::       (Docker-first, automatic portable fallback to bin\qd).
@echo off
chcp 65001 >nul
title AI_EveryNyan - Qdrant Launcher

:: Thin wrapper: all logic lives in run_qdrant.ps1
:: (Docker-first, automatic portable fallback to bin\qd).

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_qdrant.ps1"
