@echo off
chcp 65001 >nul
title AI_EveryNyan Chat

set "ROOT=%~dp0"
set "PROJECT=%ROOT%AI_EveryNyan"
set "ENV=%PROJECT%\env"
set "CONFIG=%PROJECT%\config"
set "DATA=%PROJECT%\data"

:: Env vars
set "HF_HOME=%ROOT%hf_cache"
set "QDRANT_URL=http://localhost:6333"
set "PYTHONUNBUFFERED=1"

:: DearPyGui HiDPI fix
set "QT_AUTO_SCREEN_SCALE_FACTOR=0"
set "QT_SCALE_FACTOR=1"

if not exist "%ENV%\python.exe" (
    echo [ERROR] Environment not found. Run install_ai_everynyan.ps1 first.
    pause
    exit /b 1
)

cd /d "%PROJECT%"
"%ENV%\python.exe" src/main.py --config "%CONFIG%\settings.yaml" --data-dir "%DATA%"
pause