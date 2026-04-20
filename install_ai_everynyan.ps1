<#
.SYNOPSIS
    Минимальный установщик AI_EveryNyan.
    Стек: DearPyGui + LangChain + Qdrant(Docker) + Pydantic
    Без Telegram, SD, vLLM.
#>

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# === Конфигурация ===
$ProjectName = "AI_EveryNyan"
$PythonVersion = "3.11"
$ScriptDir = $PSScriptRoot
$InstallPath = Join-Path $ScriptDir $ProjectName
$EnvPath = Join-Path $InstallPath "env"
$DataPath = Join-Path $InstallPath "data"
$ConfigPath = Join-Path $InstallPath "config"

# === Функции ===

function Write-Header {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  AI_EveryNyan Minimal Installer        ║" -ForegroundColor Cyan
    Write-Host "║  DearPyGui • LangChain • Qdrant        ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Test-Command {
    param([string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function New-LauncherBat {
    param([string]$FilePath)
    $BatContent = @"
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
"@
    $Utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($FilePath, $BatContent, $Utf8NoBom)
    Write-Host "  [+] Created: $FilePath" -ForegroundColor Green
}

function New-SettingsYaml {
    param([string]$OutputPath)
    $Content = @"
# AI_EveryNyan Minimal Config
llm:
  backend: "ollama"
  base_url: "http://localhost:11434/v1"
  chat_model: "qwen2.5:7b"
  embedding_model: "nomic-embed-text"
  timeout: 60

vector_db:
  mode: "docker"
  url: "http://localhost:6333"
  collection_name: "everynyan_diary"
  embedding_dim: 768
  min_relevance: 0.5

diary:
  storage_dir: "data/diary"
  token_dump_threshold: 8000
  plagiarism_threshold: 0.85

gui:
  title: "AI_EveryNyan"
  width: 900
  height: 700
  theme: "dark"

logging:
  level: "INFO"
  file: "logs/app.log"

debug: false
"@
    $Utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($OutputPath, $Content, $Utf8NoBom)
    Write-Host "  [+] Created: $OutputPath" -ForegroundColor Green
}

# === Основной процесс ===

Write-Header

# [1/5] Проверка зависимостей
Write-Host "[1/5] Checking system dependencies..." -ForegroundColor Yellow
$Missing = @()
if (!(Test-Command "git")) { $Missing += "Git" }
if (!(Test-Command "conda")) { $Missing += "Conda" }
if (!(Test-Command "docker")) { $Missing += "Docker" }

if ($Missing.Count -gt 0) {
    Write-Host "ERROR: Missing: $($Missing -join ', ')" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  [+] Git, Conda, Docker found" -ForegroundColor Green

# [2/5] Создание структуры папок
Write-Host "`n[2/5] Creating project structure..." -ForegroundColor Yellow
@"
data/diary
data/qdrant_storage
config
logs
hf_cache
temp
src
"@ -split "`n" | ForEach-Object {
    $dir = Join-Path $InstallPath $_.Trim()
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "  [+] Directories created" -ForegroundColor Green

# [3/5] Подготовка исходников
Write-Host "`n[3/5] Preparing source code..." -ForegroundColor Yellow
$SrcPath = Join-Path $InstallPath "src"
if (!(Test-Path (Join-Path $SrcPath "main.py"))) {
    $Stub = @"
# AI_EveryNyan Minimal Entry Point
import os, sys, time

def main():
    print("AI_EveryNyan starting...")
    print("Qdrant URL:", os.getenv("QDRANT_URL", "http://localhost:6333"))
    print("Press Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
"@
    New-Item -ItemType File -Path (Join-Path $SrcPath "main.py") -Value $Stub -Force | Out-Null
    Write-Host "  [+] Created stub src/main.py" -ForegroundColor Green
} else {
    Write-Host "  [+] src/main.py already exists" -ForegroundColor Gray
}

# [4/5] Создание окружения + установка зависимостей
Write-Host "`n[4/5] Setting up Python environment..." -ForegroundColor Yellow

$PythonExe = Join-Path $EnvPath "python.exe"

if (!(Test-Path $EnvPath)) {
    Write-Host "  Creating Conda env: $EnvPath" -ForegroundColor Gray
    conda create --prefix $EnvPath python=$PythonVersion pip -y -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create Conda env" -ForegroundColor Red
        exit 1
    }
}

Write-Host "  Installing requirements..." -ForegroundColor Gray
$ReqFile = Join-Path $ScriptDir "requirements.txt"
if (Test-Path $ReqFile) {
    & $PythonExe -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN: pip install completed with warnings" -ForegroundColor Yellow
    }
}

# Пост-проверка импортов (ИСПРАВЛЕНО: по одному модулю, через полный путь к python.exe)
Write-Host "  Verifying critical imports..." -ForegroundColor Gray
$Imports = @(
    "dearpygui.dearpygui",
    "langchain_core", 
    "qdrant_client", 
    "pydantic", 
    "asyncio"
)
$AllOK = $true
foreach ($imp in $Imports) {
    & $PythonExe -c "import $imp" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    [+] $imp" -ForegroundColor Green
    } else {
        Write-Host "    [!] $imp FAILED" -ForegroundColor Yellow
        $AllOK = $false
    }
}
if ($AllOK) {
    Write-Host "  [+] All imports successful" -ForegroundColor Green
} else {
    Write-Host "  [!] Some imports failed - check errors above" -ForegroundColor Yellow
}

# [5/5] Генерация конфига + запускатора
Write-Host "`n[5/5] Generating config and launcher..." -ForegroundColor Yellow

if (!(Test-Path (Join-Path $ConfigPath "settings.yaml"))) {
    New-SettingsYaml -OutputPath (Join-Path $ConfigPath "settings.yaml")
}
New-LauncherBat -FilePath (Join-Path $ScriptDir "run_ai_everynyan.bat")

# === Финал ===
Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Installation complete!               ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start Qdrant:  .\run_qdrant.bat" -ForegroundColor White
Write-Host "  2. Ensure Ollama is running (if using ollama backend)" -ForegroundColor Gray
Write-Host "  3. Launch app:    .\run_ai_everynyan.bat" -ForegroundColor White
Write-Host ""
Write-Host "Config: $ConfigPath\settings.yaml" -ForegroundColor Gray
Write-Host "Python: $PythonExe" -ForegroundColor Gray
Read-Host "Press Enter to finish"