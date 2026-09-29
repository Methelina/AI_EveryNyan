"""
Shared runtime state and dynamic reconfiguration for AI_EveryNyan.
Holds all mutable global state (settings, LLM, vector store, etc.).
Provides component initialization, embedding/LLM reinit, and MCP agent setup.

/src/runtime.py
Version:     0.17.11
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v0.17.11 (Soror L'.L'.):
  [+] chat_mode "openai": generic OpenAI-compatible backend (any /chat/completions
      + /models server) via settings.openai_compat. Routed through the same
      ChatOpenAI path as ollama; model list via Bearer /models.
  [+] LLM backend cooldown (mark_llm_cooldown / llm_cooldown_remaining /
      reset_llm_cooldown): after retry-ladder exhaustion new messages fail
      fast instead of re-entering the ladder; any chat-settings reinit resets.

Patch Notes v0.17.10 (Soror L'.L'.):
  [+] init_mcp_agent(): SearXNG URL resolution via mcp_health.aresolve_searxng_url
      - local instance -> plain HTTP as before;
      - local dead -> public fallbacks probed ONLY via nodriver (plain HTTP is
        blocked on public instances) with session health cache + cooldowns;
      - SEARXNG_FALLBACK_URLS passed to the MCP subprocess for per-call rotation;
      - nothing reachable -> web_search still dropped with the loud warning.

Patch Notes v0.17.9 (Soror L'.L'.):
  [+] init_components(): Qdrant warm-up via qdrant_backend.ensure_qdrant()
      (probe /readyz, auto-spawn portable bin\qd fallback, loud failure message).

Patch Notes v0.17.8 (Soror L'.L'.):
  [+] init_mcp_agent(): SearXNG warm-up probe (mcp_health.probe_searxng).
      If unreachable: loud [MCP] fallback warning + remedy hint, and the
      SearXNG-dependent tool (web_search) is dropped from the react agent so a
      dead backend never pollutes the LLM prompt. fetch_url stays (it does not
      need the SearXNG container).

Patch Notes v0.17.7 (by pytraveler):
  [+] comfyui_daemon: global reference to ComfyUIDaemon instance.
  [+] comfyui_generation_active: flag tracking active ComfyUI generation state.

Patch Notes v0.17.6 (by pytraveler):
  [+] Extracted from main.py: all global state variables.
  [+] init_components(), init_memory_manager(), init_query_preprocessor(), init_mcp_agent().
  [+] reinit_llm(), reinit_embeddings(), fetch_models_from_backend().
  [+] apply_chat_settings(), apply_embedding_settings(), reset_to_yaml_defaults().
  [+] Async helpers: run_async_loop(), submit_to_async().
  [*] No functional changes from original main.py code.
"""

import asyncio
from logger import logger
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from config import AppSettings
from llm_adapter import LlamaChatModel
from memory_manager import MemoryManager
from qdrant_backend import abort_start, ensure_qdrant
from query_preprocessor import QueryPreprocessor



# ============================================================================
# Core settings
# ============================================================================
settings: Optional[AppSettings] = None

# ============================================================================
# Infrastructure clients
# ============================================================================
qdrant_client: Optional[QdrantClient] = None
vector_store: Optional[QdrantVectorStore] = None
llm = None
embeddings: Optional[OpenAIEmbeddings] = None
memory_manager: Optional[MemoryManager] = None
mcp_client = None
react_agent = None
comfyui_daemon = None
comfyui_generation_active: bool = False 
query_preprocessor: Optional[QueryPreprocessor] = None

# ============================================================================
# Session state
# ============================================================================
session_context: List[Dict[str, str]] = []
anti_repeat_cache: List[Dict[str, Any]] = []

# ============================================================================
# Async infrastructure
# ============================================================================
async_loop: Optional[asyncio.AbstractEventLoop] = None
async_thread = None

# ============================================================================
# Lifecycle flags
# ============================================================================
_shutting_down: bool = False
_current_ai_message_tag = None

# ============================================================================
# Runtime overrides (for dynamic GUI changes)
# ============================================================================
runtime_chat_mode: str = "ollama"
runtime_embed_mode: str = "ollama"
runtime_chat_params: Dict[str, Any] = {}
runtime_embed_params: Dict[str, Any] = {}

# ============================================================================
# GUI splitter state
# ============================================================================
split_bottom_height: int = 120


def run_async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def submit_to_async(coro) -> asyncio.Future:
    if async_loop is None or not async_loop.is_running():
        logger.warning("Async loop not ready, running synchronously")
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, async_loop)


# ============================================================================
# Component initialization
# ============================================================================

def init_components():
    global qdrant_client, vector_store, llm, embeddings
    logger.info("Initializing components...")

    chat_cfg = settings.get_chat_config()
    embed_cfg = settings.get_embedding_config()

    logger.info(f"Chat mode: {settings.chat_mode}, endpoint: {chat_cfg.base_url}, model: {chat_cfg.chat_model}")
    logger.info(f"Embedding mode: {settings.embedding_mode}, endpoint: {embed_cfg.base_url}, model: {embed_cfg.embedding_model}")

    # Warm-up: guarantee a reachable Qdrant before touching the client.
    # If the URL is silent, qdrant_backend spawns the portable bin\qd\qdrant.exe
    # against the shared storage; failure exits with a clear remedy message.
    if not ensure_qdrant(settings.vector_db.url):
        abort_start()

    qdrant_client = QdrantClient(url=settings.vector_db.url)
    if not qdrant_client.collection_exists(settings.vector_db.collection):
        qdrant_client.create_collection(
            collection_name=settings.vector_db.collection,
            vectors_config=models.VectorParams(
                size=settings.vector_db.embedding_dim, distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Created collection: {settings.vector_db.collection}")

    embeddings = OpenAIEmbeddings(
        model=embed_cfg.embedding_model,
        openai_api_key=embed_cfg.api_key,
        openai_api_base=embed_cfg.base_url,
        check_embedding_ctx_length=False,
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.vector_db.collection,
        embedding=embeddings
    )

    if settings.chat_mode in ("ollama", "openai"):
        llm = ChatOpenAI(
            model=chat_cfg.chat_model,
            openai_api_key=chat_cfg.api_key,
            openai_api_base=chat_cfg.base_url,
            temperature=chat_cfg.temperature,
            timeout=chat_cfg.timeout,
            max_tokens=chat_cfg.max_tokens,
            max_retries=getattr(chat_cfg, "max_retries", 4),
            streaming=False
        )
        logger.info("LLM initialized as ChatOpenAI (mode=%s)", settings.chat_mode)
    else:
        api_key_to_use = chat_cfg.api_key if chat_cfg.api_key else "not-needed"
        llm = LlamaChatModel(
            base_url=chat_cfg.base_url,
            model=chat_cfg.chat_model,
            api_key=api_key_to_use,
            timeout=chat_cfg.timeout,
            temperature=chat_cfg.temperature,
            max_tokens=chat_cfg.max_tokens
        )
        logger.info("LLM initialized as LlamaChatModel (Native LangChain)")

    logger.info("Components initialized")


def init_memory_manager():
    global memory_manager
    memory_manager = MemoryManager()
    stats = memory_manager.get_stats()
    logger.info(
        f"MemoryManager initialized. Messages: {stats.get('total_messages', 0)}"
    )


def init_query_preprocessor():
    global query_preprocessor
    from gui import add_ai_thought
    query_preprocessor = QueryPreprocessor(add_thought_callback=add_ai_thought)
    logger.info("QueryPreprocessor initialized (spaCy lemmatization).")


async def init_mcp_agent():
    global mcp_client, react_agent
    try:
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from gui import add_ai_thought

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from langgraph.prebuilt import create_react_agent
        from tools.mcp import return_mcp_client
        from langchain_core.tools import StructuredTool

        # Warm-up resolution: the FastMCP stdio subprocess always starts and
        # always advertises its tools, so discovery alone proves nothing about
        # SearXNG. Routing policy (validated on 75 public instances):
        #   - local instance  -> plain HTTP probe + query;
        #   - local dead      -> public fallback chain probed ONLY via nodriver
        #     (plain HTTP gets 429/403 there), per-call rotation inside the tool;
        #   - nothing live    -> web_search is dropped from the agent so a dead
        #     backend never pollutes the LLM prompt (fail soft, log loud).
        from mcp_health import (
            DEFAULT_FALLBACK_URLS,
            aresolve_searxng_url,
            encode_fallback_env,
        )

        primary_url = getattr(settings, "searxng_url", "http://localhost:2597")
        configured = list(getattr(settings, "searxng_fallback_urls", None) or [])
        fallback_urls = configured if configured else list(DEFAULT_FALLBACK_URLS)

        searxng_url, searxng_via_browser = await aresolve_searxng_url(
            primary_url, fallback_urls
        )
        searxng_ok = searxng_url is not None

        mcp_env = {
            "SEARXNG_URL": searxng_url or primary_url,  # tool still needs a base
            "SEARXNG_FALLBACK_URLS": encode_fallback_env(fallback_urls),
        }
        if searxng_ok and searxng_via_browser:
            mcp_env["SEARXNG_VIA_BROWSER"] = "1"
            logger.warning(
                "[MCP] fallback: local SearXNG unreachable - using public instance "
                "%s via headless browser (queries leave your machine; privacy note). "
                "For a local backend run .\\run_searxng.bat",
                searxng_url,
            )

        mcp_client = return_mcp_client(**mcp_env)

        if not searxng_ok:
            logger.warning(
                "[MCP] fallback: SearXNG (local and all public fallbacks) is "
                "unreachable - web_search disabled for this session. "
                "Remedy: run .\\run_searxng.bat",
            )
            add_ai_thought(
                "[MCP] SearXNG offline - web search disabled (run .\\run_searxng.bat)",
                (255, 180, 80),
            )

        raw_tools = await mcp_client.get_tools()
        if raw_tools and not searxng_ok:
            dropped = [t.name for t in raw_tools if t.name == "web_search"]
            raw_tools = [t for t in raw_tools if t.name != "web_search"]
            if dropped:
                logger.info("[MCP] Disabled tool(s) (backend unreachable): %s", ", ".join(dropped))
        if raw_tools:
            def unwrap_tool(original_tool):
                async def _wrapper(**kwargs):
                    result = await original_tool.ainvoke(kwargs)
                    if isinstance(result, list) and result and isinstance(result[0], dict):
                        text_parts = [item.get('text', '') for item in result if item.get('type') == 'text']
                        if text_parts:
                            return "\n".join(text_parts)
                    return str(result)
                return StructuredTool.from_function(
                    coroutine=_wrapper,
                    name=original_tool.name,
                    description=original_tool.description,
                    args_schema=original_tool.args_schema,
                )
            tools = [unwrap_tool(t) for t in raw_tools]

            agent_model = llm
            if settings.chat_mode not in ("ollama", "openai"):
                chat_cfg = settings.get_chat_config()
                api_key = chat_cfg.api_key if chat_cfg.api_key else "not-needed"
                agent_model = ChatOpenAI(
                    model=chat_cfg.chat_model,
                    openai_api_key=api_key,
                    openai_api_base=chat_cfg.base_url,
                    temperature=chat_cfg.temperature,
                    timeout=chat_cfg.timeout,
                    max_tokens=chat_cfg.max_tokens,
                    streaming=False,
                )
                logger.info(
                    "[MCP] Using ChatOpenAI wrapper for react agent (llama mode)"
                )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                react_agent = create_react_agent(model=agent_model, tools=tools)
            tool_names = [t.name for t in tools]
            logger.info(f"[MCP] React agent initialized with tools: {tool_names}")
            add_ai_thought(
                f"[MCP] Agent ready: {len(tools)} tool(s) loaded ({', '.join(tool_names)})",
                (100, 200, 255),
            )
        else:
            logger.info("[MCP] No MCP tools discovered, running without tool support")
            add_ai_thought("[MCP] No tools found (standalone mode)", (200, 200, 150))
    except Exception as e:
        import traceback

        from gui import add_ai_thought
        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        logger.warning(
            f"[MCP] Failed to initialize MCP agent: {type(e).__name__}: {e}\n{''.join(tb_str)}"
        )
        add_ai_thought(f"[MCP] Init skipped: {type(e).__name__}: {e}", (255, 200, 100))
        mcp_client = None
        react_agent = None


# ============================================================================
# Dynamic runtime reconfiguration
# ============================================================================

# ---------------------------------------------------------------------------
# LLM backend cooldown: after repeated rate-limit failures, new messages fail
# fast (no retry ladder) until the cooldown expires or the user changes chat
# settings (any reinit resets it).
# ---------------------------------------------------------------------------
_llm_cooldown_until: float = 0.0

# Set by reinit_llm: the react agent still holds the PREVIOUS model until the
# async rebuild finishes. process_message checks it and rebuilds INLINE before
# generating, so a model switch can never be answered by the stale backend.
agent_needs_rebuild: bool = False


def mark_llm_cooldown(seconds: float = 300.0) -> None:
    global _llm_cooldown_until
    import time as _t
    _llm_cooldown_until = _t.monotonic() + seconds
    logger.warning(
        "[LLM] backend marked unhealthy - fast-fail cooldown for %ss "
        "(send a message again after that, or change chat settings to reset now)",
        seconds,
    )


def llm_cooldown_remaining() -> int:
    import time as _t
    return max(0, int(_llm_cooldown_until - _t.monotonic()))


def reset_llm_cooldown() -> None:
    global _llm_cooldown_until
    _llm_cooldown_until = 0.0


def chat_settings_for_mode(mode: str):
    """Settings section for a chat mode ('ollama' | 'llama' | 'openai')."""
    return {
        "ollama": settings.ollama,
        "openai": settings.openai_compat,
    }.get(mode, settings.llama)


def reinit_llm():
    global llm, react_agent, agent_needs_rebuild
    reset_llm_cooldown()  # user changed chat settings - trust the new backend
    agent_needs_rebuild = True  # stale agent must not answer with the old model
    mode = runtime_chat_mode
    params = runtime_chat_params.copy()
    defaults = chat_settings_for_mode(mode)
    logger.info(f"[DYNAMIC] Reinitializing LLM: mode={mode}, params={params}")

    if mode in ("ollama", "openai"):
        llm = ChatOpenAI(
            model=params.get("model", defaults.chat_model),
            openai_api_key=params.get("api_key", defaults.api_key),
            openai_api_base=params.get("base_url", defaults.base_url),
            temperature=params.get("temperature", defaults.temperature),
            timeout=params.get("timeout", defaults.timeout),
            max_tokens=params.get("max_tokens", defaults.max_tokens),
            max_retries=params.get("max_retries", getattr(defaults, "max_retries", 4)),
            streaming=False
        )
        logger.info("LLM reinitialized as ChatOpenAI (mode=%s)", mode)
    else:
        api_key = params.get("api_key", settings.llama.api_key) or "not-needed"
        llm = LlamaChatModel(
            base_url=params.get("base_url", settings.llama.base_url),
            model=params.get("model", settings.llama.chat_model),
            api_key=api_key,
            timeout=params.get("timeout", settings.llama.timeout),
            temperature=params.get("temperature", settings.llama.temperature),
            max_tokens=params.get("max_tokens", settings.llama.max_tokens)
        )
        logger.info("LLM reinitialized as LlamaChatModel")

    if react_agent:
        try:
            asyncio.run_coroutine_threadsafe(_recreate_mcp_agent(), async_loop)
        except Exception as e:
            logger.warning(f"Could not reinit MCP agent: {e}")


async def _recreate_mcp_agent():
    global mcp_client, react_agent, agent_needs_rebuild
    if mcp_client:
        raw_tools = await mcp_client.get_tools()
        if raw_tools:
            from langchain_core.tools import StructuredTool
            from langgraph.prebuilt import create_react_agent

            def unwrap_tool(original_tool):
                async def _wrapper(**kwargs):
                    result = await original_tool.ainvoke(kwargs)
                    if isinstance(result, list) and result and isinstance(result[0], dict):
                        text_parts = [item.get('text', '') for item in result if item.get('type') == 'text']
                        if text_parts:
                            return "\n".join(text_parts)
                    return str(result)
                return StructuredTool.from_function(
                    coroutine=_wrapper,
                    name=original_tool.name,
                    description=original_tool.description,
                    args_schema=original_tool.args_schema,
                )
            tools = [unwrap_tool(t) for t in raw_tools]
            agent_model = llm
            if runtime_chat_mode not in ("ollama", "openai"):
                agent_model = ChatOpenAI(
                    model=runtime_chat_params.get("model", settings.llama.chat_model),
                    openai_api_key=runtime_chat_params.get("api_key", "not-needed"),
                    openai_api_base=runtime_chat_params.get("base_url", settings.llama.base_url),
                    temperature=runtime_chat_params.get("temperature", settings.llama.temperature),
                    timeout=runtime_chat_params.get("timeout", settings.llama.timeout),
                    max_tokens=runtime_chat_params.get("max_tokens", settings.llama.max_tokens),
                    streaming=False,
                )
            react_agent = create_react_agent(model=agent_model, tools=tools)
            agent_needs_rebuild = False  # inline rebuild done - agent is current
            logger.info("[MCP] React agent reinitialized after chat change")


def reinit_embeddings():
    global embeddings, vector_store
    mode = runtime_embed_mode
    params = runtime_embed_params.copy()
    logger.info(f"[DYNAMIC] Reinitializing embeddings: mode={mode}, params={params}")

    if mode == "ollama":
        embeddings = OpenAIEmbeddings(
            model=params.get("model", settings.ollama.embedding_model),
            openai_api_key=params.get("api_key", settings.ollama.api_key),
            openai_api_base=params.get("base_url", settings.ollama.base_url),
            check_embedding_ctx_length=False,
        )
    else:
        logger.warning("Embedding mode 'llama' not supported, falling back to Ollama")
        embeddings = OpenAIEmbeddings(
            model=settings.ollama.embedding_model,
            openai_api_key=settings.ollama.api_key,
            openai_api_base=settings.ollama.base_url,
            check_embedding_ctx_length=False,
        )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.vector_db.collection,
        embedding=embeddings
    )
    logger.info("Embeddings and vector store reinitialized")


def fetch_models_from_backend(backend_type: str, base_url: str, api_key: str = "") -> List[str]:
    import requests
    try:
        if backend_type == "ollama":
            base = base_url.replace("/v1", "")
            response = requests.get(f"{base}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                return models
        else:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return models
            else:
                return [runtime_chat_params.get("model", "unknown")]
    except Exception as e:
        logger.warning(f"Failed to fetch models from {backend_type}: {e}")
    return []


def apply_chat_settings(ui_values: dict):
    global runtime_chat_mode, runtime_chat_params
    chat_mode_from_ui = ui_values.get("chat_mode")
    if chat_mode_from_ui:
        runtime_chat_mode = chat_mode_from_ui
        defaults = chat_settings_for_mode(runtime_chat_mode)
        runtime_chat_params["base_url"] = defaults.base_url
        runtime_chat_params["api_key"] = defaults.api_key

    runtime_chat_params.update({
        "model": ui_values.get("model", runtime_chat_params.get("model")),
        "temperature": ui_values.get("temperature", runtime_chat_params.get("temperature")),
        "max_tokens": ui_values.get("max_tokens", runtime_chat_params.get("max_tokens")),
        "timeout": ui_values.get("timeout", runtime_chat_params.get("timeout")),
    })
    runtime_chat_params = {k: v for k, v in runtime_chat_params.items() if v is not None}
    reinit_llm()
    from gui import add_ai_thought
    add_ai_thought(f"[GUI] Chat settings applied: mode={runtime_chat_mode}, model={runtime_chat_params.get('model')}")


def apply_embedding_settings(ui_values: dict):
    global runtime_embed_mode, runtime_embed_params
    runtime_embed_mode = ui_values.get("embed_mode", runtime_embed_mode)
    if runtime_embed_mode == "ollama":
        runtime_embed_params.update({
            "model": ui_values.get("model", settings.ollama.embedding_model),
            "base_url": settings.ollama.base_url,
            "api_key": settings.ollama.api_key,
        })
    else:
        runtime_embed_params.update({
            "model": settings.ollama.embedding_model,
            "base_url": settings.ollama.base_url,
            "api_key": settings.ollama.api_key,
        })
        from gui import add_ai_thought
        add_ai_thought("[GUI] LLaMA backend for embeddings not supported, using Ollama", (255,200,100))
    runtime_embed_params = {k: v for k, v in runtime_embed_params.items() if v is not None}
    reinit_embeddings()
    from gui import add_ai_thought
    add_ai_thought(f"[GUI] Embedding settings applied: mode={runtime_embed_mode}, model={runtime_embed_params.get('model')}")


def reset_to_yaml_defaults():
    global runtime_chat_mode, runtime_embed_mode, runtime_chat_params, runtime_embed_params, settings
    config_path = Path("config/settings.yaml")
    settings = AppSettings.from_yaml(str(config_path))

    runtime_chat_mode = settings.chat_mode
    runtime_embed_mode = settings.embedding_mode

    defaults = chat_settings_for_mode(runtime_chat_mode)
    runtime_chat_params = {
        "model": defaults.chat_model,
        "base_url": defaults.base_url,
        "api_key": defaults.api_key,
        "temperature": defaults.temperature,
        "max_tokens": defaults.max_tokens,
        "timeout": defaults.timeout,
    }
    if settings.embedding_mode == "ollama":
        runtime_embed_params = {
            "model": settings.ollama.embedding_model,
            "base_url": settings.ollama.base_url,
            "api_key": settings.ollama.api_key,
        }
    else:
        runtime_embed_params = {
            "model": settings.ollama.embedding_model,
            "base_url": settings.ollama.base_url,
            "api_key": settings.ollama.api_key,
        }
    reinit_llm()
    reinit_embeddings()
    from gui import add_ai_thought
    add_ai_thought("[GUI] Reset to settings.yaml defaults", (100,255,100))
