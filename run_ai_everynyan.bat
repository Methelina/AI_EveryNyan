:: AI_EveryNyan runtime launcher wrapper - delegates to run_ai_everynyan.ps1.
:: run_ai_everynyan.bat
:: Version:     1.0.0
:: Author:      Soror L.'.L.'.
:: Updated:     2026-09-29
::
:: Patch Notes v1.0.0 (Soror L'.L'.):
::   [+] Created: README references run_ai_everynyan.bat but only the .ps1
::       existed - thin wrapper for consistency with run_qdrant.bat.
@echo off
chcp 65001 >nul
title AI_EveryNyan

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_ai_everynyan.ps1" %*
