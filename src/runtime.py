"""
Shared runtime state and dynamic reconfiguration for AI_EveryNyan.
Holds all mutable global state (settings, LLM, vector store, etc.).
Provides component initialization, embedding/LLM reinit, and MCP agent setup.

/src/runtime.py
Version:     0.17.6
Author:      Soror L.'.L.'.
Updated:     2026-05-01

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

    if settings.chat_mode == "ollama":
        llm = ChatOpenAI(
            model=chat_cfg.chat_model,
            openai_api_key=chat_cfg.api_key,
            openai_api_base=chat_cfg.base_url,
            temperature=chat_cfg.temperature,
            timeout=chat_cfg.timeout,
            max_tokens=chat_cfg.max_tokens,
            streaming=False
        )
        logger.info("LLM initialized as ChatOpenAI (Ollama mode)")
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

        searxng_url = getattr(settings, "searxng_url", "http://localhost:2597")
        mcp_client = return_mcp_client(SEARXNG_URL=searxng_url)

        raw_tools = await mcp_client.get_tools()
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
            if settings.chat_mode != "ollama":
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

def reinit_llm():
    global llm, react_agent
    mode = runtime_chat_mode
    params = runtime_chat_params.copy()
    logger.info(f"[DYNAMIC] Reinitializing LLM: mode={mode}, params={params}")

    if mode == "ollama":
        llm = ChatOpenAI(
            model=params.get("model", settings.ollama.chat_model),
            openai_api_key=params.get("api_key", settings.ollama.api_key),
            openai_api_base=params.get("base_url", settings.ollama.base_url),
            temperature=params.get("temperature", settings.ollama.temperature),
            timeout=params.get("timeout", settings.ollama.timeout),
            max_tokens=params.get("max_tokens", settings.ollama.max_tokens),
            streaming=False
        )
        logger.info("LLM reinitialized as ChatOpenAI (Ollama)")
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
    global mcp_client, react_agent
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
            if runtime_chat_mode != "ollama":
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
        if runtime_chat_mode == "ollama":
            runtime_chat_params["base_url"] = settings.ollama.base_url
            runtime_chat_params["api_key"] = settings.ollama.api_key
        else:
            runtime_chat_params["base_url"] = settings.llama.base_url
            runtime_chat_params["api_key"] = settings.llama.api_key

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

    if runtime_chat_mode == "ollama":
        runtime_chat_params = {
            "model": settings.ollama.chat_model,
            "base_url": settings.ollama.base_url,
            "api_key": settings.ollama.api_key,
            "temperature": settings.ollama.temperature,
            "max_tokens": settings.ollama.max_tokens,
            "timeout": settings.ollama.timeout,
        }
    else:
        runtime_chat_params = {
            "model": settings.llama.chat_model,
            "base_url": settings.llama.base_url,
            "api_key": settings.llama.api_key,
            "temperature": settings.llama.temperature,
            "max_tokens": settings.llama.max_tokens,
            "timeout": settings.llama.timeout,
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
