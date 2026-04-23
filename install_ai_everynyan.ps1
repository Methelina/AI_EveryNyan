<#
.SYNOPSIS
    Установщик AI_EveryNyan (плоская структура).
    Создает Conda-окружение, устанавливает зависимости, скачивает bge-m3 в Ollama,
    устанавливает spaCy и языковые модели, генерирует полный конфиг settings.yaml,
    устанавливает Playwright и Chromium в папку проекта (изолированно).
    Идемпотентен: безопасен для повторного запуска.

    \install_ai_everynyan.ps1
    Version: 0.7.1
    Author: Soror L.'.L.'.
    Updated: 2026-04-23
#>

# Patchnote v0.7.1:
#   - Удалено создание скриптов запуска (run_ai_everynyan.bat и .example) – пользователь сам настраивает.
#   + Добавлена проверка наличия Chromium в playwright_browsers после установки.
#   * Улучшена справка по переменной PLAYWRIGHT_BROWSERS_PATH для существующего bat-файла.

# Patchnote v0.7.0:
#   + Добавлена изолированная установка Playwright Chromium в папку проекта
#       (через PLAYWRIGHT_BROWSERS_PATH).
#   * Обновлен список проверки импортов: добавлены playwright и nodriver.

# Patchnote v0.6.0:
#   + Добавлены проверки импортов для всех новых зависимостей из requirements.txt
#       (aiohttp, httpx, duckdb, langchain*, openai, fastmcp, markdownify, langgraph и др.)


# ==========================================
Write-Host " ===========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  ██▓        ██▓    ██▓        ██▓" -ForegroundColor Yellow
Write-Host " ▓██▒              ▓██▒" -ForegroundColor Yellow
Write-Host " ▒██░              ▒██░" -ForegroundColor Yellow
Write-Host " ▒██░              ▒██░" -ForegroundColor Yellow
Write-Host " ░██████▒ ██▓  ██▓ ░██████▒ ██▓  ██▓" -ForegroundColor Yellow
Write-Host " ░ ▒░▓  ░ ▒▓▒  ▒▓▒ ░ ▒░▓  ░ ▒▓▒  ▒▓▒" -ForegroundColor Yellow
Write-Host " ░ ░ ▒  ░ ░▒   ░▒  ░ ░ ▒  ░ ░▒   ░▒" -ForegroundColor Yellow
Write-Host "   ░ ░    ░    ░     ░ ░    ░    ░" -ForegroundColor Yellow
Write-Host "     ░  ░  ░    ░      ░  ░  ░    ░" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ===========================================" -ForegroundColor Green
Write-Host "    EveryNyan AI by L.'.L.'." -ForegroundColor Yellow
Write-Host "    AI_EveryNyan Installer v0.7.1" -ForegroundColor Green
Write-Host ""
# ==========================================

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
$PlaywrightBrowsersPath = Join-Path $ProjectRoot "playwright_browsers"

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

function New-SettingsExample {
    param([string]$OutPath)
    $Content = @"
# AI_EveryNyan Configuration
# Version: 0.13.0
# \config\settings.yaml

# Выбор активного LLM-бэкенда для чата: "ollama" или "llama"
chat_mode: "ollama"   # режим для генерации ответов

# Отдельная настройка для эмбеддингов (можно указать ollama или другой сервер)
embedding_mode: "ollama"   # можно "ollama", "llama", или "custom"

# Настройки для Ollama (используется как для чата, так и для эмбеддингов, если выбраны)
ollama:
  base_url: "http://127.0.0.1:11434/v1"
  api_key: "ollama"
  chat_model: "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest" # или "deepseek-v3.1:671b-cloud"
  embedding_model: "bge-m3:latest"         # установлен инсталлятором
  timeout: 120
  temperature: 0.7
  max_tokens: 8192
  token_dump_threshold: 20000

# Настройки для внешнего LLaMA-сервера (только для чата, эмбеддинги не поддерживаются)
llama:
  base_url: "http://127.0.0.1:8088/v1"
  api_key: ""
  chat_model: "Falcon-H1R-7B-Q8_0.gguf"
  timeout: 1800
  temperature: 0.7
  max_tokens: 16304
  token_dump_threshold: 20000

vector_db:
  url: "http://localhost:6333"
  collection: "everynyan_diary"
  embedding_dim: 1024   # ВАЖНО: bge-m3 использует 1024-мерные вектора

rag:
  top_k: 10
  similarity_threshold: 0.65
  enable_metadata_filtering: false

context:
  max_history_messages: 20
  warn_if_context_exceeds: 20

diary:
  storage_dir: "data/diary"
  plagiarism_threshold: 0.97
  injection_max_length: 5000
  summary_prompt: |
    You are updating your personal diary. Process the provided context and write a reflective entry.
    Time window: last 24–48h. Focus on facts, emotions, relationships, and actionable insights.
    
    <rules>
    1. ALWAYS separate distinct thoughts/events with markdown lines `---`.
    2. Each section must be self-sufficient (50–300 words).
    3. Follow the exact structure below for EVERY section.
    4. Output ONLY the formatted text. No greetings, no explanations, no markdown outside the structure.
    5. DO NOT invent facts. If uncertain, state it explicitly.
    6. Use canonical names for entities (e.g., "Linda" instead of "that girl").
    7. Keep retrieval cues short, keyword-rich, and searchable.
    8. At the end of each section, include a JSON block with metadata as shown below.
    </rules>

    <output_format>
    **Timestamp:** [Absolute time or time window]
    **Source Event:** [What triggered this memory]
    **Outcomes:** [Concrete results, decisions, or emotional shifts]

    **Entities:** [Canonical names of people, places, systems]
    **Key Messages:** 
    - "[Verbatim quote 1]"
    - "[Verbatim quote 2]" (include context if needed)

    **Topics/Tags:** #[tag1] #[tag2] #[tag3]
    **Importance Score:** [0.0–1.0] – [1-sentence rationale]
    **Affect:** Valence [-1..1, where -1=depressed/misery, +1=ecstasy/elation], Arousal [-1..1, where -1=aversion/withdrawal, +1=intense approach/desire], plus a short emotion label.
    **Relationships:** [How entities relate to each other or to you]

    **Retrieval Cues:** ["search phrase 1", "search phrase 2", "search phrase 3"]
    **Photo Descriptions:** [If applicable: detailed visual description + filename]
    **Contradictions/Uncertainties:** [Conflicts, missing info, or things needing verification]

    Then output a JSON object with the following fields (no extra text after JSON):
    ```json
    {
      "entities": ["list", "of", "canonical", "names"],
      "topics": ["#tag1", "#tag2"],
      "retrieval_cues": ["phrase1", "phrase2"],
      "key_facts": ["important quote or fact"],
      "affect_valence": 0.0,
      "affect_arousal": 0.0,
      "emotion_label": "short emotion description",
      "importance_score": 0.0,
      "relationships": {"A-B": "friend", "B-C": "colleague"},
      "contradictions": ["any uncertainties"]
    }
    </output_format>

anti_repeat:
  trigger_avg: 0.73
  trigger_max: 0.69
  max_history: 32

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
    Write-Status "Создан шаблон конфигурации: $OutPath" "SUCCESS"
}

# === Основной процесс ===

Write-Status "╔════════════════════════════════════════╗" "INFO"
Write-Status "║  AI_EveryNyan Installer v0.7.1         ║" "INFO"
Write-Status "╚════════════════════════════════════════╝" "INFO"

# 1. Проверка зависимостей
Write-Status "[1/8] Проверка системных зависимостей..." "INFO"
$Missing = @()
if (!(Test-Command "git")) { $Missing += "Git" }
if (!(Test-Command "conda")) { $Missing += "Conda (Miniforge/Anaconda)" }
if (!(Test-Command "docker")) { $Missing += "Docker" }
if (!(Test-Command "ollama")) { $Missing += "Ollama" }

if ($Missing.Count -gt 0) {
    Write-Status "ОШИБКА: Отсутствуют: $($Missing -join ', ')" "ERROR"
    Read-Host "Нажмите Enter для выхода"
    exit 1
}
Write-Status "  [+] Git, Conda, Docker, Ollama найдены" "SUCCESS"

# 2. Создание структуры папок
Write-Status "`n[2/8] Создание структуры проекта..." "INFO"
$Dirs = @($SrcPath, $ConfigPath, $DataPath, $LogsPath, $CachePath, $TempPath, "$DataPath\qdrant_storage", $PlaywrightBrowsersPath)
foreach ($dir in $Dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Status "  [+] Директории готовы" "SUCCESS"

# 3. Генерация конфигурации (новый полный шаблон)
Write-Status "`n[3/8] Подготовка конфигурации..." "INFO"
$SettingsExample = Join-Path $ConfigPath "settings.yaml.example"
$SettingsActual = Join-Path $ConfigPath "settings.yaml"

if (!(Test-Path $SettingsExample)) {
    New-SettingsExample -OutPath $SettingsExample
}
if (!(Test-Path $SettingsActual)) {
    Copy-Item $SettingsExample $SettingsActual
    Write-Status "  [+] Создан config/settings.yaml (актуальная структура)" "INFO"
} else {
    Write-Status "  [+] config/settings.yaml уже существует (проверьте вручную при необходимости)" "INFO"
}

# 4. Установка Ollama-модели bge-m3:latest
Write-Status "`n[4/8] Проверка embedding-модели bge-m3:latest..." "INFO"

$modelInstalled = $false
try {
    $ollamaList = & ollama list 2>$null
    if ($ollamaList -match "bge-m3") {
        $modelInstalled = $true
    }
} catch {}

if ($modelInstalled) {
    Write-Status "  [+] bge-m3:latest уже установлена в Ollama" "SUCCESS"
} else {
    Write-Status "  Скачивание bge-m3:latest (~1.2 GB)..." "INFO"
    $pullOk = Invoke-WithRetry -Script {
        & ollama pull bge-m3:latest
        if ($LASTEXITCODE -ne 0) { throw "ollama pull failed" }
    } -MaxAttempts 3 -DelaySec 10

    if ($pullOk) {
        Write-Status "  [+] bge-m3:latest успешно установлена" "SUCCESS"
    } else {
        Write-Status "  [!] Не удалось скачать bge-m3:latest. Запустите вручную: ollama pull bge-m3:latest" "WARN"
    }
}

# 5. Conda-окружение + pip install
Write-Status "`n[5/8] Настройка Python-окружения..." "INFO"

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

# Установка зависимостей из requirements.txt
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

# 5b. Установка Playwright и изолированного Chromium
Write-Status "`n[5b/8] Установка Playwright и Chromium в проект..." "INFO"

# Убедимся, что playwright установлен
& $PythonExe -m pip install playwright>=1.40.0
if ($LASTEXITCODE -ne 0) {
    Write-Status "  [!] Не удалось установить playwright" "WARN"
} else {
    # Сохраняем старую переменную окружения, если была
    $oldPlaywrightPath = $env:PLAYWRIGHT_BROWSERS_PATH
    try {
        $env:PLAYWRIGHT_BROWSERS_PATH = $PlaywrightBrowsersPath
        Write-Status "  Установка Chromium в $PlaywrightBrowsersPath ..." "INFO"
        & $PythonExe -m playwright install chromium
        if ($LASTEXITCODE -eq 0) {
            Write-Status "  [+] Chromium успешно установлен в папку проекта (изолированно)" "SUCCESS"
        } else {
            Write-Status "  [!] Ошибка при установке Chromium (код $LASTEXITCODE)" "WARN"
        }
    } finally {
        # Восстанавливаем прежнее значение
        if ($oldPlaywrightPath) { $env:PLAYWRIGHT_BROWSERS_PATH = $oldPlaywrightPath }
        else { Remove-Item Env:PLAYWRIGHT_BROWSERS_PATH -ErrorAction SilentlyContinue }
    }
}

# 6. Установка spaCy и языковых моделей
Write-Status "`n[6/8] Установка spaCy и языковых моделей..." "INFO"

# Проверка, установлен ли spaCy
$spacyInstalled = & $PythonExe -c "import spacy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Status "  Установка spaCy через pip..." "INFO"
    & $PythonExe -m pip install spacy>=3.7.0
    if ($LASTEXITCODE -ne 0) {
        Write-Status "  [!] Не удалось установить spaCy" "WARN"
    } else {
        Write-Status "  [+] spaCy установлен" "SUCCESS"
    }
} else {
    Write-Status "  [+] spaCy уже установлен" "SUCCESS"
}

# Функция проверки наличия модели spaCy
function Test-SpacyModel {
    param([string]$ModelName)
    $checkScript = @"
import spacy
try:
    spacy.load('$ModelName')
    print('OK')
except OSError:
    print('MISSING')
"@
    $result = & $PythonExe -c $checkScript
    return $result.Trim() -eq "OK"
}

# Установка русской модели
$ruModel = "ru_core_news_sm"
if (Test-SpacyModel -ModelName $ruModel) {
    Write-Status "  [+] Модель $ruModel уже установлена" "SUCCESS"
} else {
    Write-Status "  Установка модели $ruModel (~40 MB)..." "INFO"
    & $PythonExe -m spacy download $ruModel
    if ($LASTEXITCODE -eq 0) {
        Write-Status "  [+] Модель $ruModel установлена" "SUCCESS"
    } else {
        Write-Status "  [!] Не удалось установить $ruModel" "WARN"
    }
}

# Установка английской модели
$enModel = "en_core_web_sm"
if (Test-SpacyModel -ModelName $enModel) {
    Write-Status "  [+] Модель $enModel уже установлена" "SUCCESS"
} else {
    Write-Status "  Установка модели $enModel (~12 MB)..." "INFO"
    & $PythonExe -m spacy download $enModel
    if ($LASTEXITCODE -eq 0) {
        Write-Status "  [+] Модель $enModel установлена" "SUCCESS"
    } else {
        Write-Status "  [!] Не удалось установить $enModel" "WARN"
    }
}

# 7. Проверка критических импортов (включая playwright и nodriver)
Write-Status "`n[7/8] Проверка критических импортов..." "INFO"

$Modules = @(
    # Основные библиотеки
    "dearpygui.dearpygui",
    "langchain_core",
    "qdrant_client",
    "pydantic",
    "asyncio",
    "spacy",
    # Сетевые и базы данных
    "aiohttp",
    "httpx",
    "duckdb",
    # LangChain и интеграции
    "langchain",
    "langchain_community",
    "langchain_qdrant",
    "langchain_openai",
    "openai",
    # Конфиги и утилиты
    "pydantic_settings",
    "dotenv",
    "structlog",
    "yaml",
    "tqdm",
    # MCP инструменты
    "fastmcp",
    "markdownify",
    "langchain_mcp_adapters",
    "langgraph",
    # Browser automation
    "playwright.async_api",
    "nodriver"      # опционально, но проверяем
)

$AllOK = $true
foreach ($mod in $Modules) {
    & $PythonExe -c "import $mod" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Status "    [+] $mod" "SUCCESS"
    } else {
        # nodriver может отсутствовать – не считаем ошибкой
        if ($mod -eq "nodriver") {
            Write-Status "    [?] $mod не установлен (опционально)" "WARN"
        } else {
            Write-Status "    [!] $mod НЕ НАЙДЕН" "ERROR"
            $AllOK = $false
        }
    }
}

# Дополнительная проверка моделей spaCy
$ruCheck = & $PythonExe -c "import spacy; spacy.load('ru_core_news_sm')" 2>$null
if ($LASTEXITCODE -eq 0) { Write-Status "    [+] ru_core_news_sm (spacy)" "SUCCESS" }
else { Write-Status "    [!] ru_core_news_sm не найдена" "WARN"; $AllOK = $false }

$enCheck = & $PythonExe -c "import spacy; spacy.load('en_core_web_sm')" 2>$null
if ($LASTEXITCODE -eq 0) { Write-Status "    [+] en_core_web_sm (spacy)" "SUCCESS" }
else { Write-Status "    [!] en_core_web_sm не найдена" "WARN"; $AllOK = $false }

if ($AllOK) { Write-Status "  [+] Все импорты и модели успешны" "SUCCESS" }
else { Write-Status "  [!] Ошибка импорта. Проверьте логи или переустановите окружение." "ERROR" }

# 8. Проверка Chromium и рекомендации по запуску
Write-Status "`n[8/8] Проверка изолированного Chromium..." "INFO"
if (Test-Path $PlaywrightBrowsersPath) {
    $chromiumDirs = Get-ChildItem -Path $PlaywrightBrowsersPath -Directory -Filter "chromium-*" -ErrorAction SilentlyContinue
    if ($chromiumDirs) {
        Write-Status "  [+] Chromium установлен в: $PlaywrightBrowsersPath" "SUCCESS"
    } else {
        Write-Status "  [!] Папка playwright_browsers существует, но Chromium не найден. Возможно, установка не завершилась." "WARN"
    }
} else {
    Write-Status "  [!] Папка playwright_browsers не найдена. Chromium не установлен." "WARN"
}

Write-Status "`n=== Важно для запуска приложения ===" "INFO"
Write-Status "Убедитесь, что в вашем run_ai_everynyan.bat (или любом другом скрипте запуска) установлена переменная окружения:" "INFO"
Write-Status "  set PLAYWRIGHT_BROWSERS_PATH=%~dp0playwright_browsers" "WARN"
Write-Status "Это обеспечит использование изолированного Chromium из проекта." "INFO"

# === Финал ===
Write-Status "" "INFO"
Write-Status "╔════════════════════════════════════════╗" "SUCCESS"
Write-Status "║   Установка завершена!                 ║" "SUCCESS"
Write-Status "╚════════════════════════════════════════╝" "SUCCESS"
Write-Status "" "INFO"
Write-Status "Следующие шаги:" "INFO"
Write-Status "  1. Убедитесь, что внешние запускаторы (run_qdrant.bat, run_searxng.bat) присутствуют в корне проекта" "INFO"
Write-Status "  2. Запустите Qdrant (run_qdrant.bat)" "INFO"
Write-Status "  3. Запустите Ollama (ollama serve)" "INFO"
Write-Status "  4. Запустите SearXNG (run_searxng.bat)" "INFO"
Write-Status "  5. Добавьте переменную PLAYWRIGHT_BROWSERS_PATH в ваш run_ai_everynyan.bat (см. рекомендацию выше)" "INFO"
Write-Status "  6. Запустите приложение через run_ai_everynyan.bat" "INFO"
Write-Status "" "INFO"
Write-Status "Конфиг: $ConfigPath\settings.yaml" "INFO"
Write-Status "Логи:   $LogsPath\install.log" "INFO"
Write-Status "Playwright браузер: $PlaywrightBrowsersPath (изолирован)" "INFO"

Read-Host "`nНажмите Enter для завершения"
exit 0