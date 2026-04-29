:: ---------------------------------------------------------
:: /run_ai_everynyan.bat
:: Version: 1.1.0
:: Author:  Soror L.'.L.'.
:: Updated: 2026-04-23
::
:: Patchnote v1.1.0:
::   [+] Added PLAYWRIGHT_BROWSERS_PATH to use isolated Chromium from project.
::
:: Patchnote v1.0.0:
::   [+] Initial launch script.
:: ---------------------------------------------------------

@echo off
chcp 65001 >nul
title AI_EveryNyan Chat Launcher by L.'.L.'.
:: ===========================================
echo.
echo   ██▓        ██▓    ██▓        ██▓
echo  ▓██▒              ▓██▒
echo  ▒██░              ▒██░
echo  ▒██░              ▒██░
echo  ░██████▒ ██▓  ██▓ ░██████▒ ██▓  ██▓
echo  ░ ▒░▓  ░ ▒▓▒  ▒▓▒ ░ ▒░▓  ░ ▒▓▒  ▒▓▒
echo  ░ ░ ▒  ░ ░▒   ░▒  ░ ░ ▒  ░ ░▒   ░▒
echo    ░ ░    ░    ░     ░ ░    ░    ░
echo      ░  ░  ░    ░      ░  ░  ░    ░
echo.
echo  ===========================================
echo    AI_EveryNyan Chat Launcher by L.'.L.'. 
echo    Launcher Version: 1.1.0
echo  ===========================================
echo.

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

:: Playwright browsers path (isolated Chromium)
set "PLAYWRIGHT_BROWSERS_PATH=%ROOT%playwright_browsers"

if not exist "%ENV%\python.exe" (
    echo [ERROR] Environment not found. Run install_ai_everynyan.ps1 first.
    pause
    exit /b 1
)

cd /d "%ROOT%"
"%ENV%\python.exe" src/main.py --config "%CONFIG%\settings.yaml" --data-dir "%DATA%"
pause