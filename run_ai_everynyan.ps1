<#
.SYNOPSIS
    AI_EveryNyan Chat Launcher by L.'.L.'.
    Version: 1.3.0
    Updated: 2026-04-30

.DESCRIPTION
    Patchnote v1.3.0:
      [*] Единый стиль логирования (INFO/ERROR/FATAL) без таймстемпов.
      [*] Вывод проверки серверов в формате "[INFO] Launcher: [Server] Status: OK".
    ...
#>

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "AI_EveryNyan Chat Launcher by L.'.L.'."

Write-Host @"

   ██▓        ██▓    ██▓        ██▓
  ▓██▒              ▓██▒
  ▒██░              ▒██░
  ▒██░              ▒██░
  ░██████▒ ██▓  ██▓ ░██████▒ ██▓  ██▓
  ░ ▒░▓  ░ ▒▓▒  ▒▓▒ ░ ▒░▓  ░ ▒▓▒  ▒▓▒
  ░ ░ ▒  ░ ░▒   ░▒  ░ ░ ▒  ░ ░▒   ░▒
    ░ ░    ░    ░     ░ ░    ░    ░
      ░  ░  ░    ░      ░  ░  ░    ░
  ===========================================
    AI_EveryNyan Chat Launcher by L.'.L.'.
    Version: 1.3.0
  ===========================================

"@

$ROOT = $PSScriptRoot
$ENV = Join-Path $ROOT "env"
$CONFIG = Join-Path $ROOT "config"
$DATA = Join-Path $ROOT "data"
$CONFIG_FILE = Join-Path $CONFIG "settings.yaml"

$env:HF_HOME = Join-Path $ROOT "hf_cache"
$env:QDRANT_URL = "http://localhost:6333"
$env:PYTHONUNBUFFERED = "1"
$env:QT_AUTO_SCREEN_SCALE_FACTOR = "0"
$env:QT_SCALE_FACTOR = "1"
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $ROOT "playwright_browsers"

# Проверка venv и конфига
if (-not (Test-Path "$ENV\python.exe")) {
    Write-Host "[ERROR] Launcher: Environment not found. Run install_ai_everynyan.ps1 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
if (-not (Test-Path $CONFIG_FILE)) {
    Write-Host "[ERROR] Launcher: Config file not found: $CONFIG_FILE" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# ===== Парсинг settings.yaml =====
$yaml = Get-Content $CONFIG_FILE -Encoding UTF8
$section = $null
$config = @{
    chat_mode       = $null
    embedding_mode  = $null
    ollama_base_url = $null
    llama_base_url  = $null
    qdrant_url      = $null
}

function CleanValue($raw) {
    $val = $raw -replace '\s*#.*$', ''
    $val = $val.Trim()
    $val = $val -replace '^"(.*)"$', '$1'
    return $val
}

foreach ($line in $yaml) {
    if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
    if ($line -match '^(\S.*?):\s*(.*)$') {
        $key = $Matches[1]
        $rest = $Matches[2]
        if ($rest -eq '') {
            $section = $key
        } else {
            $section = $null
            switch ($key) {
                'chat_mode'      { $config.chat_mode      = CleanValue $rest }
                'embedding_mode' { $config.embedding_mode = CleanValue $rest }
            }
        }
        continue
    }
    if ($section) {
        switch ($section) {
            'ollama'    { if ($line -match '^\s+base_url:\s+(.+)$') { $config.ollama_base_url = CleanValue $Matches[1] } }
            'llama'     { if ($line -match '^\s+base_url:\s+(.+)$') { $config.llama_base_url  = CleanValue $Matches[1] } }
            'vector_db' { if ($line -match '^\s+url:\s+(.+)$')      { $config.qdrant_url      = CleanValue $Matches[1] } }
        }
    }
}

# ===== Функция извлечения origin =====
function Get-Origin([string]$url) {
    if (-not $url) { return $null }
    try {
        $uri = [System.Uri]$url
        return "$($uri.Scheme)://$($uri.Host):$($uri.Port)"
    } catch {
        return $url
    }
}

# ===== Функция проверки сервера (возвращает хеш с Success и Error) =====
function Test-ServerAvailability {
    param(
        [string]$Url,
        [string]$ExpectedInBody = $null
    )
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        if ($ExpectedInBody) {
            if ($response.Content -notmatch [regex]::Escape($ExpectedInBody)) {
                $bodySample = $response.Content.Substring(0, [Math]::Min(200, $response.Content.Length))
                return @{ Success = $false; Error = "Unexpected response content. Got: $bodySample" }
            }
        }
        return @{ Success = $true; Error = $null }
    } catch {
        return @{ Success = $false; Error = $_.Exception.Message }
    }
}

# ===== Определение проверяемых серверов =====
$serversToCheck = @()

# Qdrant
if ($config.qdrant_url) {
    $serversToCheck += @{
        Name     = 'Qdrant'
        Url      = $config.qdrant_url.TrimEnd('/') + '/'
        Expected = '"qdrant'
    }
} else {
    Write-Host "[WARNING] Launcher: Qdrant URL not configured in settings.yaml." -ForegroundColor Yellow
}

# Chat server
if ($config.chat_mode -eq 'ollama') {
    if (-not $config.ollama_base_url) {
        Write-Host "[ERROR] Launcher: chat_mode=ollama but no ollama.base_url in config" -ForegroundColor Red
        exit 1
    }
    $origin = Get-Origin $config.ollama_base_url
    $serversToCheck += @{
        Name     = 'Ollama (chat)'
        Url      = "$origin/"
        Expected = 'Ollama is running'
    }
} elseif ($config.chat_mode -eq 'llama') {
    if (-not $config.llama_base_url) {
        Write-Host "[ERROR] Launcher: chat_mode=llama but no llama.base_url in config" -ForegroundColor Red
        exit 1
    }
    $origin = Get-Origin $config.llama_base_url
    $serversToCheck += @{
        Name     = 'Llama.cpp (chat)'
        Url      = "$origin/health"
        Expected = 'ok'
    }
} else {
    Write-Host "[WARNING] Launcher: Unknown chat_mode '$($config.chat_mode)', skipping chat server check." -ForegroundColor Yellow
}

# Embedding server (Ollama, отдельно от чата)
if ($config.embedding_mode -eq 'ollama' -and $config.chat_mode -ne 'ollama') {
    if (-not $config.ollama_base_url) {
        Write-Host "[ERROR] Launcher: embedding_mode=ollama but no ollama.base_url" -ForegroundColor Red
        exit 1
    }
    $origin = Get-Origin $config.ollama_base_url
    $serversToCheck += @{
        Name     = 'Ollama (embedding)'
        Url      = "$origin/"
        Expected = 'Ollama is running'
    }
}

# ===== Проверка всех серверов =====
$allOk = $true
foreach ($srv in $serversToCheck) {
    $checkResult = Test-ServerAvailability -Url $srv.Url -ExpectedInBody $srv.Expected
    if ($checkResult.Success) {
        Write-Host "[INFO] Launcher: [$($srv.Name)] Status: OK"
    } else {
        Write-Host "[ERROR] Launcher: [$($srv.Name)] Status: FAILED - $($checkResult.Error)" -ForegroundColor Red
        $allOk = $false
    }
}

if (-not $allOk) {
    Write-Host "[FATAL] Launcher: One or more required services are not running. Exiting." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Launcher: All services OK.`n"

# ===== Запуск приложения =====
Set-Location $ROOT
& "$ENV\python.exe" src/main.py --config "$CONFIG\settings.yaml" --data-dir "$DATA"
Read-Host "Press Enter to exit"