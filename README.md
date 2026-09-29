# 🐱 AI_EveryNyan v0.16.2 — Smart Desktop Assistant with Hybrid Memory & MCP Tools

> *"Haru, everynyan! How are you!? Fine, thank you! I wish I were a bird!"*

**AI_EveryNyan** is a desktop AI application with an advanced memory architecture and support for external tools via MCP (Model Context Protocol). Unlike ordinary chatbots, EveryNyan keeps a "digital diary", summarizes conversations, can search the web (through SearXNG) and extract content from web pages.

The application is built on Python, uses local models (Ollama, a LLaMA server, or **any OpenAI-compatible API**) and a hybrid storage system: **DuckDB** for exact history and **Qdrant** for semantic associations.
**Current version:** v0.16.2 — with fully unified multi-backend support (Ollama / LLaMA / OpenAI-compatible), streaming responses, an MCP agent, colored logging of tool calls, and self-healing backends (Qdrant, SearXNG, ComfyUI).

---

## ✨ Key Features

### 🧠 Hybrid Memory Architecture (DuckDB + Qdrant)
Storage is split into two levels:
*   **DuckDB (SQL):** Structured chat chronology, diary entries, and exact metadata.
*   **Qdrant (Vector DB):** Embeddings of dialogues and thoughts. Provides RAG (Retrieval-Augmented Generation) — search by meaning.

### ⏳ Sliding Window & Smart Dump
A unique context management algorithm:
1.  When the buffer overflows, the AI writes a diary entry, capturing the gist, emotions, and entities.
2.  The text is split into semantic blocks, checked for plagiarism (via Qdrant) and stored.
3.  Working memory is cleared, the dialogue continues from a clean slate, but the knowledge is preserved forever.

### 📋 Structured Diary Metadata (JSON + Circumplex Model)
Each diary entry contains LLM-extracted metadata:
- Entities (`entities`), tags (`topics`), retrieval cues (`retrieval_cues`).
- **Affect:** valence (−1…+1) and arousal (−1…+1) per the Russell model, plus a textual emotion label.
- Importance, relationships, contradictions, and key facts.

### 🛡️ Graceful Shutdown
The application intercepts shutdown signals. Before exiting, it force-calls `dump_context_to_memory` so that no thought is lost.

### 🚫 Semantic Anti-Repeat
Cosine similarity of new messages against history. If the AI starts repeating itself (average similarity > 0.73 or max > 0.69), a loop is detected and the topic is changed.

### 👁️ AI Internal Thoughts UI
The "Internal Thoughts" panel in DearPyGui shows processes in real time: RAG queries, tool calls, errors, model reasoning_content. In v0.16.2 **colored logging of MCP tools** was added: calls (yellow) and results (green/red).

### 🔌 Three Backends: Ollama, LLaMA, OpenAI-compatible
- **Ollama** — primary mode, full streaming and metadata support.
- **LLaMA** — mode for local servers (e.g. `llama-server`). A critical bug in answer saving was fully fixed in v0.16.1, and a fallback to direct LLM on agent errors was added in v0.16.2.
- **OpenAI-compatible** (`openai_compat`) — any server exposing `/chat/completions` + `/models`: OpenAI, OpenRouter, Mistral, LM Studio, vLLM. Configured by three fields: `base_url`, `api_key`, `chat_model`. Switchable live from the GUI.

### 🧹 Lemmatization for RAG (via spaCy)
Message and diary text before indexing is processed through `ru_core_news_sm` and `en_core_web_sm`, which improves search quality.

### 🔧 MCP Tools (Model Context Protocol)
All tools are auto-discovered from `tools/mcp/tool_*.py` and connected via stdio transport. Each tool is an independent FastMCP server.

#### 🌐 Web Search and Navigation (SearXNG)
- **`web_search`** — anonymous meta-search through SearXNG (no API keys). Results in Markdown.
- **`fetch_url`** — extraction of web page content with three modes: `legacy` (httpx+bs4), `playwright` (full browser), `nodriver` (light CDP). Automatic HTML → Markdown cleanup.
- **`open_url`** — opening a link in the user's system browser.

#### 🧮 Math and Conversion
- **`calculate`** — safe evaluator for mathematical expressions (arithmetic, trigonometry, logarithms, `pi`, `e`).
- **`convert_units`** — unit conversion: temperature (C/F/K), length, weight, data volume, time.

#### 💱 Exchange Rates
- **`convert_currency`** — convert an amount between currencies at live rates (no API keys, ~150+ currencies).
- **`exchange_rate`** — current rate for a pair or group of currencies.
- **`list_currencies`** — list of all supported currencies.

#### 🕐 Date and Time
- **`get_datetime`** — current date/time with timezone and arbitrary formatting support.
- **`get_timestamp`** — Unix timestamp and ISO 8601.
- **`format_datetime`** — format a timestamp into human-readable form.
- **`time_diff`** — human-readable difference between two dates.

#### 🌤️ Weather
- **`get_weather`** — current weather for a city/coordinates (via wttr.in, no API keys).
- **`get_forecast`** — 3-day forecast.

#### 🎲 Random Value Generation
- **`generate_uuid`** — UUID v4 (random) or v7 (time-ordered).
- **`generate_random_string`** — passwords, tokens, arbitrary strings with charset choice.
- **`generate_random_number`** — random numbers (int/float) in a range.
- **`pick_random`** — random choice from a list.

#### 📁 File System Sandbox (Workspace)
Read-only only; all paths are validated against `workspace_dir` (path traversal and symlink escapes are blocked).
- **`read_file`** — read a file (text/binary, with offset/limit).
- **`check_path`** — check existence and type of a path.
- **`list_directory`** — directory contents with sorting and filtering.
- **`search_files`** — search files by glob pattern.
- **`directory_tree`** — directory tree.
- **`file_info`** — file metadata (size, dates, line count, SHA-256).
- **`grep_content`** — search within file contents (regex).

#### 🖥️ System Information
- **`get_system_info`** — OS, hostname, uptime, CPU, RAM, disk (with section filtering).

#### 👁️ Vision (Image Analysis)
- **`describe_image`** — image analysis via a VL-model. Auto-detects the vision capability of the active chat model (Ollama / llama.cpp). Two modes: `structured_json` (detailed JSON: person/scene) and `free_text`. Supports local files and URLs.

#### 🎨 Image Generation (ComfyUI)
- **`generate_image`** — generation through ComfyUI: submit a workflow, wait via HTTP polling, save results to disk.
- **`list_workflows`** — list available workflow JSON files from `workflows/`.

#### 🧑‍🎨 Character Management
- **`update_character_appearance`** — change the active character's appearance via LLM: hair, clothing, makeup, accessories, etc. Projection system (`reprojection/`) with Pydantic validation and locked fields.

#### 🔧 Infrastructure
- **Colored logging** of all tool calls and results in the SYSTEM LOG panel.
- **Agent error fallback** — if the MCP agent crashes, the system switches to a direct LLM call.
- **Auto-discovery** — every `tool_*.py` is connected automatically through `__init__.py`.
- **SearXNG fallback chain** (`mcp_health`) — local instance over plain HTTP; if it is down, public instances are probed ONLY via nodriver (plain HTTP gets 429/403 there), with a session health cache and per-instance cooldowns; if everything is dead, `web_search` is dropped from the agent (fail soft, log loud).
- **Dead-model handling** — 410/retired/429 errors get friendly messages; rate limits are retried by a Kilo-style silent ladder (10/30/60s) that never spams the chat; after exhaustion a fail-fast cooldown prevents retry loops, and changing chat settings resets it. Synthetic error replies are NEVER persisted to memory.
- **Stale-agent guard** — after a model switch the react agent is rebuilt inline before generation, so the old backend can never answer.
- **Time awareness** — the system prompt carries the current local time and history messages carry `[YYYY-MM-DD HH:MM]` prefixes, so the character can reason about when things happened. Echoed prefixes are stripped from replies.
- **Copyable chat** — double-click a message to copy its full text to the clipboard.

### 🗄️ Qdrant: Docker-first + Portable Fallback
The vector DB can operate in two modes with fully compatible data:
- **Docker container** (`qdrant/qdrant`) — primary mode, managed by `run_qdrant.ps1`.
- **Portable binary** (`bin\qd\qdrant.exe`, v1.19.1) — automatic fallback when Docker is unavailable, also self-started by the runtime (`src/qdrant_backend.py`) if Qdrant is not running.

Both backends serve the same `http://localhost:6333` and the same storage `data\qdrant_storage` (segment formats and WAL are byte-for-byte compatible) — collection survives the Docker ↔ exe switch in both directions, and the Python code doesn't distinguish the backend at all. For a forced backend choice, set `$use_portable_qd = 1` in `run_qdrant.ps1`.

### 🖥️ ComfyUI: Auto-discovery
`comfyui.server` from `settings.yaml` is the single source of truth. If the configured endpoint is not answering, the runtime auto-discovers a running ComfyUI by scanning python processes (cmdline heuristic + `--port` + HTTP verification) — a port change in the ComfyUI launcher never desynchronizes the monitor or `generate_image`.

### 🔄 Chromium Auto-update (`browser_updater`)
Browser revisions are pinned to the playwright package version. At runtime startup (and in the installer) the updater checks PyPI, upgrades playwright, re-downloads Chromium, runs a real smoke test, and only then prunes old revisions. Gated by `browser_update.enabled` + `check_interval_days`; download progress is visible in the console.

---

## 🚀 Quick Start (Automatic Install)

The application includes an automatic PowerShell installer.

### Requirements
- **PowerShell**
- **Conda** (Miniforge or Anaconda)
- **Docker Desktop** (optional — for Qdrant in a container; without it Qdrant runs via the portable binary `bin\qd`)
- **Ollama** or any LLaMA-server with an OpenAI-compatible API

### 1. Install
```powershell
.\install_ai_everynyan.ps1
```

**What the script does:**
- Checks Git, Conda, Ollama (Docker — optional, warns if absent).
- Creates the folder structure and a Conda environment `env` with Python 3.12.
- Installs dependencies from `requirements.txt`.
- Downloads the embedding model **`bge-m3`** via Ollama.
- Downloads and unpacks the **portable Qdrant** into `bin\qd\` (mandatory step [2b/8] — guaranteed fallback backend).
- Generates `config/settings.yaml` (dim=1024).

### 2. Configuration (Optional)
Edit `config/settings.yaml`. Example:
```yaml
chat_mode: "ollama"   # or "llama", or "openai"
ollama:
  chat_model: "qwen2.5:7b"
llama:
  base_url: "http://127.0.0.1:8088/v1"
  chat_model: "Falcon-H1R-7B-Q8_0.gguf"
openai_compat:         # any /chat/completions + /models server
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  chat_model: "gpt-4o-mini"
# Chromium auto-update at runtime startup (visible download progress)
browser_update:
  enabled: true
  check_interval_days: 3
# For MCP tools (SearXNG)
searxng_url: "http://localhost:2597"
searxng_fallback_urls: []   # empty -> built-in verified public instances
```

> **Note:** `embedding_model` must be `bge-m3:latest`, `embedding_dim` — `1024`.

### 3. Launch
1. **Qdrant:** `.\run_qdrant.bat` — Docker-first; if Docker is unavailable (or `$use_portable_qd = 1` in `run_qdrant.ps1`), the portable binary is auto-downloaded and started from `bin\qd\`. The runtime also self-starts the portable backend if Qdrant was not launched. The Python side does not notice the difference (the same `http://localhost:6333`, the same storage `data\qdrant_storage`).
2. **SearXNG (for web search):** `.\run_searxng.bat` — if the container is not up, the runtime falls back to verified public instances (probed via headless browser); if those are dead too, `web_search` is excluded from the MCP agent with a warning (fetch_url works without it).
3. **Ollama / LLaMA-server / OpenAI-compatible endpoint** — the launcher also pings the selected `chat_model` and warns if it is dead or rate-limited before the app starts.
4. **Application:** `.\run_ai_everynyan.bat`

---

## 🗂️ Project Structure

```
AI_EveryNyan/
├── install_ai_everynyan.ps1   # installer
├── run_ai_everynyan.bat
├── run_qdrant.bat               # wrapper (logic lives in run_qdrant.ps1)
├── run_qdrant.ps1               # Qdrant launcher: Docker-first, portable fallback in bin\qd
├── src/
│   ├── main.py                # v0.17.10: MCP agent, colored logging, error handling, retry ladder
│   ├── runtime.py             # v0.17.11: runtime state, Qdrant/SearXNG warmup, chat modes, cooldown
│   ├── qdrant_backend.py      # v1.0.1: auto-start portable Qdrant (bin\qd)
│   ├── mcp_health.py          # v1.1.0: SearXNG fallback resolver (nodriver probes, health cache)
│   ├── comfyui_discovery.py   # v1.0.0: ComfyUI server auto-discovery (process scan)
│   ├── browser_updater.py     # v1.0.0: Chromium auto-update (pip + smoke test + prune)
│   ├── memory_manager.py      # v0.7.1: JSON metadata, circumplex model, timestamp helper
│   └── query_preprocessor.py  # v0.2.0: lemmatization
├── tools/mcp/
│   ├── __init__.py             # auto-discovery of tool_*.py, MultiServerMCPClient
│   ├── tool_searxng.py         # v0.6.0: SearXNG + fetch_url (chunked output, nodriver routing)
│   ├── tool_comfyui.py         # v0.1.0: image generation (ComfyUI API, server auto-discovery)
│   └── ...                     # 10 more tool servers
├── config/
│   ├── settings.yaml
│   ├── searxng_settings.reference.yml  # tracked reference of tuned SearXNG engines
│   └── character/
│       ├── base.yaml
│       └── appearance.yaml
├── tests/                      # 58 unit tests (backends, probes, chunking, config)
├── data/ (history.db, qdrant_storage)
└── logs/
```

---

## 🧩 How It Works

### 1. Asynchronous Architecture
GUI (DearPyGui) on the main thread, LLM requests and RAG on a background `asyncio` loop.

### 2. Message Pipeline (v0.16.2)
1.  **Anti-Repeat** — check for semantic loops.
2.  **RAG** — search in Qdrant (with similarity threshold).
3.  **Keyword search fallback** — if no vectors, search in DuckDB.
4.  **MCP Agent** — LangGraph agent with tools. Calls and results are color-logged in the thoughts panel.
5.  **Fallback** — on agent error → direct LLM (Ollama or LLaMA).
6.  **Streaming** (LLaMA) — the answer appears token by token.
7.  **Saving** the dialogue to DuckDB and Qdrant with metadata extraction.
8.  **Smart Dump** — on context overflow a diary entry with JSON metadata is written.

### 3. MCP Tools (12 servers, 30+ tools)
| Server | Tools | Description |
| :--- | :--- | :--- |
| `tool_searxng` | `web_search`, `fetch_url` | Web search (SearXNG) and page extraction |
| `tool_browser` | `open_url` | Open links in the browser |
| `tool_calculator` | `calculate`, `convert_units` | Math and unit conversion |
| `tool_currency` | `convert_currency`, `exchange_rate`, `list_currencies` | Exchange rates |
| `tool_datetime` | `get_datetime`, `get_timestamp`, `format_datetime`, `time_diff` | Date and time |
| `tool_weather` | `get_weather`, `get_forecast` | Weather (wttr.in) |
| `tool_random` | `generate_uuid`, `generate_random_string`, `generate_random_number`, `pick_random` | Random value generation |
| `tool_workspace` | `read_file`, `check_path`, `list_directory`, `search_files`, `directory_tree`, `file_info`, `grep_content` | File system sandbox (read-only) |
| `tool_system` | `get_system_info` | System information |
| `tool_vision` | `describe_image` | Image analysis (VL models) |
| `tool_comfyui` | `generate_image`, `list_workflows` | Image generation (ComfyUI) |
| `tool_character` | `update_character_appearance` | Character appearance management |

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **GUI** | `DearPyGui` |
| **LLM** | `LangChain` + `Ollama` / `LlamaChatModel` (custom adapter) |
| **MCP** | `fastmcp`, `langgraph` (create_react_agent) |
| **Vector DB** | `Qdrant` |
| **SQL DB** | `DuckDB` |
| **Config** | `Pydantic Settings` |
| **Async** | `asyncio` |
| **NLP** | `spaCy` |
| **Web scraping** | `httpx`, `BeautifulSoup4`, `markdownify`, `playwright` / `nodriver` |

---

## 🗺️ Roadmap Milestones (up to v0.16.2)

### Already Implemented
- ✅ Full dual-backend support (Ollama + LLaMA) via a unified LangChain interface.
- ✅ Streaming responses for the LLaMA mode.
- ✅ Structured diary metadata (JSON + circumplex affect model).
- ✅ Text lemmatization for improved RAG (spaCy).
- ✅ **MCP tools (12 servers, 30+ tools)** — auto-discovery, stdio transport, colored logging, fallback.
- ✅ **Web search (SearXNG)** and **URL extraction** (legacy/playwright/nodriver).
- ✅ **Open links in browser** (`open_url`).
- ✅ **Calculator and unit conversion** (temperature, length, weight, data, time).
- ✅ **Currency conversion** (~150+ currencies, no API keys).
- ✅ **Date and time** with timezones, formatting and difference calculation.
- ✅ **Weather** — current weather and 3-day forecast (wttr.in).
- ✅ **Random value generation** — UUID, strings, numbers, list pick.
- ✅ **Read-only file system sandbox** — read, search, tree, grep, metadata.
- ✅ **System information** — OS, CPU, RAM, disks.
- ✅ **Vision — image analysis** via VL models (auto-detects vision capability of the chat model).
- ✅ **Image generation (ComfyUI)** — submit workflow, save results.
- ✅ **Character appearance management** — projections, Pydantic validation, locked fields.
- ✅ Fixed critical bug with `None` answers in LLaMA mode.
- ✅ Graceful shutdown and colored MCP logging.
- ✅ **Portable Qdrant** (`bin\qd`) — Docker-first launcher, mandatory installer step, auto-start by runtime when launch was forgotten.
- ✅ **SearXNG fallback chain** — local instance → public instances (nodriver-probed, health cache + cooldowns) → `web_search` excluded; error replies are never persisted to memory.
- ✅ **OpenAI-compatible chat mode** — any `/chat/completions` server via three config fields, live-switchable in the GUI.
- ✅ **Time awareness** — current time in the system prompt, timestamped history; echoed prefixes stripped from replies.
- ✅ **Silent rate-limit ladder + fail-fast cooldown** — per-minute quotas waited out without chat spam; dead backends error instantly instead of looping.
- ✅ **ComfyUI auto-discovery** — config-first server resolution with process-scan fallback; quiet connection logging (transitions only).
- ✅ **Chromium auto-update** — PyPI check at startup (config-gated), visible download progress, smoke test, stale-revision pruning.
- ✅ **Copyable chat** — double-click a message to copy it.
- ✅ **Launcher model ping** — the preflight warns if the selected `chat_model` is dead or rate-limited before the app starts.

### In Active Development (WIP)
- 🚧 **Internal thoughts (proactivity)** — periodic generation of random thoughts saved to the diary.
- 🚧 **Confidence & Usage Count** — adding confidence and usage counts to metadata for RAG weighting.
- 🚧 **Background memory cleanup** — removing false and stale entries.

### Future Plans
- **Dynamic RAG adaptation** (auto-tuning `top_k` and threshold).
- **Semantic Chunker** — smart splitting of long texts.
- **NER Enrichment** — automatically extracting names, dates.
- **Reminders and scheduler**.

---

> 💡 **Note:** The project uses automatic model and environment setup. On errors, check `logs/app.log` and `logs/mcp_debug.log`.

*Haru everynyan, and happy coding!* 🐾✨
