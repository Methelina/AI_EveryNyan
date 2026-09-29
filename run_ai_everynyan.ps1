<#
.SYNOPSIS
    AI_EveryNyan Chat Launcher by L.'.L.'.
    Version: 1.4.0
    Updated: 2026-09-29

.DESCRIPTION
    Patchnote v1.4.0:
      [+] Убийство зомби-процессов AI_EveryNyan (python.exe из project-local
          env) при старте лаунчера, до проверки сервисов.
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
    Version: 1.4.0
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

# ===== Убийство зомби-процессов AI_EveryNyan =====
# Осиротевшие инстансы бота держат data/history.db, Qdrant-сессии и порт GUI.
# Два канала поиска: exe из project-local env ИЛИ заголовок консоли
# "AI_EveryNyan ..." (main.py ставит его через SetConsoleTitleW).
$projectPython = (Join-Path $ENV "python.exe").ToLowerInvariant()
$zombiePids = @{}

foreach ($p in @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue)) {
    if ($p.ExecutablePath -and $p.ExecutablePath.ToLowerInvariant() -eq $projectPython) {
        $zombiePids[[uint32]$p.ProcessId] = "exe=$($p.ExecutablePath)"
    }
}
foreach ($gp in @(Get-Process -Name python -ErrorAction SilentlyContinue)) {
    if ($gp.MainWindowTitle -and $gp.MainWindowTitle -match 'AI_EveryNyan') {
        $zombiePids[[uint32]$gp.Id] = "title='$($gp.MainWindowTitle)'"
    }
}

if ($zombiePids.Count -gt 0) {
    foreach ($entry in $zombiePids.GetEnumerator()) {
        Write-Host "[RUNNER] [INFO] Killing zombie AI_EveryNyan process: PID=$($entry.Key) $($entry.Value)"
        Stop-Process -Id $entry.Key -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
} else {
    Write-Host "[RUNNER] [INFO] No zombie AI_EveryNyan processes found."
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
            'ollama' {
                if ($line -match '^\s+base_url:\s+(.+)$')   { $config.ollama_base_url  = CleanValue $Matches[1] }
                if ($line -match '^\s+chat_model:\s+(.+)$') { $config.ollama_chat_model = CleanValue $Matches[1] }
            }
            'llama' {
                if ($line -match '^\s+base_url:\s+(.+)$')   { $config.llama_base_url    = CleanValue $Matches[1] }
                if ($line -match '^\s+chat_model:\s+(.+)$') { $config.llama_chat_model  = CleanValue $Matches[1] }
            }
            'openai_compat' {
                if ($line -match '^\s+base_url:\s+(.+)$')   { $config.openai_base_url = CleanValue $Matches[1] }
                if ($line -match '^\s+api_key:\s+(.+)$')    { $config.openai_api_key  = CleanValue $Matches[1] }
                if ($line -match '^\s+chat_model:\s+(.+)$') { $config.openai_chat_model = CleanValue $Matches[1] }
            }
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
        [string]$ExpectedInBody = $null,
        [hashtable]$Headers = @{}
    )
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -UseBasicParsing -Headers $Headers -ErrorAction Stop
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
} elseif ($config.chat_mode -eq 'openai') {
    if (-not $config.openai_base_url) {
        Write-Host "[ERROR] Launcher: chat_mode=openai but no openai_compat.base_url in config" -ForegroundColor Red
        exit 1
    }
    # Keep the full base_url INCLUDING its path (/v1 etc.) - Get-Origin would
    # strip the path and turn /v1/models into /models (404 on cloud APIs).
    $modelsUrl = $config.openai_base_url.TrimEnd('/') + '/models'
    $headers = @{}
    if ($config.openai_api_key) {
        $headers['Authorization'] = "Bearer $($config.openai_api_key)"
    }
    $serversToCheck += @{
        Name     = 'OpenAI-compat (chat)'
        Url      = $modelsUrl
        Headers  = $headers
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
    $checkResult = Test-ServerAvailability -Url $srv.Url -ExpectedInBody $srv.Expected -Headers $srv.Headers
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

# ===== Проверка выбранной чат-модели (advisory: предупреждаем, но не блокируем) =====
function Test-ChatModelAvailability {
    param([string]$BaseUrl, [string]$ApiKey, [string]$Model)
    $headers = @{ 'Content-Type' = 'application/json' }
    if ($ApiKey) { $headers['Authorization'] = "Bearer $ApiKey" }
    $body = @{ model = $Model; messages = @(@{ role = 'user'; content = 'ping' }); max_tokens = 1 } | ConvertTo-Json -Depth 4
    try {
        $resp = Invoke-WebRequest -Uri "$BaseUrl/chat/completions" -Method POST -Headers $headers -Body $body -TimeoutSec 25 -UseBasicParsing -ErrorAction Stop
        return @{ Status = 'OK'; Error = $null }
    } catch {
        $code = $null; $msg = $_.Exception.Message
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            # pwsh 7: response body lands in ErrorDetails.Message
            $errBody = $_.ErrorDetails.Message
            if ($errBody -match '"message"\s*:\s*"([^"]{0,160})') { $msg = $Matches[1] }
        }
        if ($code -eq 200) { return @{ Status = 'OK'; Error = $null } }
        if ($code -eq 429 -or $msg -match 'rate limit') { return @{ Status = 'RATE_LIMITED'; Error = $msg } }
        if ($code -eq 404 -or $code -eq 410 -or $msg -match 'retired|not found|does not exist|invalid model') {
            return @{ Status = 'DEAD'; Error = "$code $msg" }
        }
        return @{ Status = 'UNKNOWN'; Error = "$code $msg" }
    }
}

$chatModelCheck = $null
switch ($config.chat_mode) {
    'ollama' {
        if ($config.ollama_chat_model -and $config.ollama_base_url) {
            $chatModelCheck = @{ Model = $config.ollama_chat_model
                                 Result = (Test-ChatModelAvailability -BaseUrl ($config.ollama_base_url -replace '/v1/?$','/v1') -ApiKey $config.ollama_api_key -Model $config.ollama_chat_model) }
        }
    }
    'llama' {
        if ($config.llama_chat_model -and $config.llama_base_url) {
            $chatModelCheck = @{ Model = $config.llama_chat_model
                                 Result = (Test-ChatModelAvailability -BaseUrl $config.llama_base_url -ApiKey $config.llama_api_key -Model $config.llama_chat_model) }
        }
    }
    'openai' {
        if ($config.openai_chat_model -and $config.openai_base_url) {
            $chatModelCheck = @{ Model = $config.openai_chat_model
                                 Result = (Test-ChatModelAvailability -BaseUrl $config.openai_base_url -ApiKey $config.openai_api_key -Model $config.openai_chat_model) }
        }
    }
}

if ($chatModelCheck) {
    $m = $chatModelCheck.Model; $r = $chatModelCheck.Result
    switch ($r.Status) {
        'OK'           { Write-Host "[INFO] Launcher: [Chat model] '$m' is alive and answering." -ForegroundColor Green }
        'RATE_LIMITED' { Write-Host "[WARNING] Launcher: [Chat model] '$m' is reachable but rate-limited right now - first messages may retry silently." -ForegroundColor Yellow }
        'DEAD'         { Write-Host "[WARNING] Launcher: [Chat model] '$m' looks DEAD ($($r.Error)). Update chat_model in config\settings.yaml - the app will start, but replies will fail." -ForegroundColor Yellow }
        default        { Write-Host "[WARNING] Launcher: [Chat model] '$m' check inconclusive ($($r.Error)) - continuing anyway." -ForegroundColor Yellow }
    }
} else {
    Write-Host "[WARNING] Launcher: [Chat model] chat_model not found in settings for mode '$($config.chat_mode)' - skipping model check." -ForegroundColor Yellow
}

# ===== Запуск приложения =====
Set-Location $ROOT
& "$ENV\python.exe" src/main.py --config "$CONFIG\settings.yaml" --data-dir "$DATA"
Read-Host "Press Enter to exit"