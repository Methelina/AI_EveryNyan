<#
.SYNOPSIS
    Установщик AI_EveryNyan (плоская структура).
    Создает Conda-окружение, устанавливает зависимости, генерирует конфиги и запускаторы.
    Идмпотентен: безопасен для повторного запуска.
#>

# === Настройка кодировки ===
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# === Конфигурация путей ===
$ProjectRoot = $PSScriptRoot
$PythonVersion = "3.11"
$EnvPath = Join-Path $ProjectRoot "env"
$PythonExe = Join-Path $EnvPath "python.exe"
$ConfigPath = Join-Path $ProjectRoot "config"
$SrcPath = Join-Path $ProjectRoot "src"
$DataPath = Join-Path $ProjectRoot "data"
$LogsPath = Join-Path $ProjectRoot "logs"
$CachePath = Join-Path $ProjectRoot "hf_cache"
$TempPath = Join-Path $ProjectRoot "temp"
$ReqFile = Join-Path $ProjectRoot "requirements.txt"

# === Вспомогательные функции ===

function Write-Status {
    param([string]$Message, [string]$Level = "INFO")
    $color = switch ($Level) {
        "ERROR"   { "Red" }
        "WARN"    { "Yellow" }
        "SUCCESS" { "Green" }
        default   { "White" }
    }
    Write-Host "[$Level] $Message" -ForegroundColor $color
    $logLine = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    if (!(Test-Path $LogsPath)) { New-Item -ItemType Directory -Path $LogsPath -Force | Out-Null }
    Add-Content -Path (Join-Path $LogsPath "install.log") -Value $logLine -Encoding UTF8
}

function Test-Command {
    param([string]$Cmd)
    return $null -ne (Get-Command $Cmd -ErrorAction SilentlyContinue)
}

function Invoke-WithRetry {
    param([scriptblock]$Script, [int]$MaxAttempts = 3, [int]$DelaySec = 5)
    $attempt = 0
    while ($attempt -lt $MaxAttempts) {
        try {
            & $Script
            if ($LASTEXITCODE -eq 0) { return $true }
        } catch {}
        $attempt++
        if ($attempt -ge $MaxAttempts) { return $false }
        Start-Sleep -Seconds $DelaySec
    }
    return $false
}

function New-LauncherBat {
    param([string]$OutPath)
    $BatContent = @"
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
"@
    $Utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($OutPath, $BatContent, $Utf8NoBom)
    Write-Status "Создан запускатор: $OutPath" "SUCCESS"
}

function New-SettingsExample {
    param([string]$OutPath)
    $Content = @"
# AI_EveryNyan Configuration Template
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
    [System.IO.File]::WriteAllText($OutPath, $Content, $Utf8NoBom)
    Write-Status "Создан шаблон: $OutPath" "SUCCESS"
}

# === Основной процесс ===

Write-Status "╔════════════════════════════════════════╗" "INFO"
Write-Status "║  AI_EveryNyan Installer                ║" "INFO"
Write-Status "╚════════════════════════════════════════╝" "INFO"

# 1. Проверка зависимостей
Write-Status "[1/6] Проверка системных зависимостей..." "INFO"
$Missing = @()
if (!(Test-Command "git")) { $Missing += "Git" }
if (!(Test-Command "conda")) { $Missing += "Conda (Miniforge/Anaconda)" }
if (!(Test-Command "docker")) { $Missing += "Docker" }

if ($Missing.Count -gt 0) {
    Write-Status "ОШИБКА: Отсутствуют: $($Missing -join ', ')" "ERROR"
    Read-Host "Нажмите Enter для выхода"
    exit 1
}
Write-Status "  [+] Git, Conda, Docker найдены" "SUCCESS"

# 2. Создание структуры папок
Write-Status "`n[2/6] Создание структуры проекта..." "INFO"
$Dirs = @($SrcPath, $ConfigPath, $DataPath, $LogsPath, $CachePath, $TempPath, "$DataPath\qdrant_storage")
foreach ($dir in $Dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Status "  [+] Директории готовы" "SUCCESS"

# 3. Шаблоны конфигов
Write-Status "`n[3/6] Подготовка конфигурации..." "INFO"
$SettingsExample = Join-Path $ConfigPath "settings.yaml.example"
$SettingsActual = Join-Path $ConfigPath "settings.yaml"

if (!(Test-Path $SettingsExample)) {
    New-SettingsExample -OutPath $SettingsExample
}
if (!(Test-Path $SettingsActual)) {
    Copy-Item $SettingsExample $SettingsActual
    Write-Status "  [+] Создан config/settings.yaml (отредактируйте под себя)" "INFO"
} else {
    Write-Status "  [+] config/settings.yaml уже существует" "INFO"
}

# 4. Conda-окружение + pip install
Write-Status "`n[4/6] Настройка Python-окружения..." "INFO"

if (!(Test-Path $EnvPath)) {
    Write-Status "  Создание Conda env: $EnvPath" "INFO"
    $createOk = Invoke-WithRetry {
        conda create --prefix $EnvPath python=$PythonVersion pip -y -q
        if ($LASTEXITCODE -ne 0) { throw "conda create failed" }
    }
    if (!$createOk) {
        Write-Status "ОШИБКА: Не удалось создать окружение" "ERROR"
        exit 1
    }
}

# Установка зависимостей
if (Test-Path $ReqFile) {
    Write-Status "  Установка requirements..." "INFO"
    $pipOk = Invoke-WithRetry {
        & $PythonExe -m pip install -r $ReqFile
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    }
    if ($pipOk) { Write-Status "  [+] Зависимости установлены" "SUCCESS" }
    else { Write-Status "  [!] Предупреждение при pip install" "WARN" }
} else {
    Write-Status "  [!] requirements.txt не найден в корне" "WARN"
}

# 5. Проверка импортов
Write-Status "`n[5/6] Проверка критических импортов..." "INFO"
$Modules = @("dearpygui.dearpygui", "langchain_core", "qdrant_client", "pydantic", "asyncio")
$AllOK = $true
foreach ($mod in $Modules) {
    & $PythonExe -c "import $mod" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Status "    [+] $mod" "SUCCESS"
    } else {
        Write-Status "    [!] $mod НЕ НАЙДЕН" "ERROR"
        $AllOK = $false
    }
}
if ($AllOK) { Write-Status "  [+] Все импорты успешны" "SUCCESS" }
else { Write-Status "  [!] Ошибка импорта. Проверьте логи или переустановите окружение." "ERROR" }

# 6. Генерация запускатора
Write-Status "`n[6/6] Создание запускатора..." "INFO"
New-LauncherBat -OutPath (Join-Path $ProjectRoot "run_ai_everynyan.bat")

# === Финал ===
Write-Status "" "INFO"
Write-Status "╔════════════════════════════════════════╗" "SUCCESS"
Write-Status "║   Установка завершена!                 ║" "SUCCESS"
Write-Status "╚════════════════════════════════════════╝" "SUCCESS"
Write-Status "" "INFO"
Write-Status "Следующие шаги:" "INFO"
Write-Status "  1. Запустите Qdrant: .\run_qdrant.bat" "INFO"
Write-Status "  2. Убедитесь, что Ollama запущен (если используется)" "INFO"
Write-Status "  3. Запустите приложение: .\run_ai_everynyan.bat" "INFO"
Write-Status "" "INFO"
Write-Status "Конфиг: $ConfigPath\settings.yaml" "INFO"
Write-Status "Логи: $LogsPath\install.log" "INFO"

Read-Host "`nНажмите Enter для завершения"
exit 0