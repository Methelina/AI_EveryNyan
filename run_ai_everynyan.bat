@echo off
chcp 65001 >nul
title AI_EveryNyan Chat

:: === Paths ===
set "ROOT=%~dp0"
set "ENV=%ROOT%env"
set "CONFIG=%ROOT%config"
set "DATA=%ROOT%data"

:: === Environment ===
set "HF_HOME=%ROOT%hf_cache"
set "QDRANT_URL=http://localhost:6333"
set "PYTHONUNBUFFERED=1"
set "QT_AUTO_SCREEN_SCALE_FACTOR=0"
set "QT_SCALE_FACTOR=1"

if not exist "%ENV%\python.exe" (
    echo [ERROR] Environment not found. Run install_ai_everynyan.ps1 first.
    pause
    exit /b 1
)

cd /d "%ROOT%"
"%ENV%\python.exe" src/main.py --config "%CONFIG%\settings.yaml" --data-dir "%DATA%"
pause