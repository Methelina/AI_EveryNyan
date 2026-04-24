#!/usr/bin/env python3
"""
AI_EveryNyan - DearPyGui Chat with LangChain + Qdrant RAG + DuckDB History
Modular Character System + Smart Context Management + Structured Diary Metadata

\src\main.py
Version:     0.16.3 (MCP fix for llama mode)
Author:      Soror L.'.L.'.
Updated:     2026-04-24

Patch Notes v0.16.3 [pytraveler]:
  [FIX] MCP agent now initializes correctly in llama mode (chat_mode="llama").
        LlamaChatModel lacks bind_tools() (NotImplementedError in BaseChatModel),
        so a ChatOpenAI wrapper is created for the react agent in llama mode,
        leveraging the OpenAI-compatible API of llama-server.
  [+] Improved MCP init error logging: full exception type name and traceback
      instead of empty "Failed to initialize MCP agent: ".

Patch Notes v0.16.2:
  [+] MCP agent logging: tool calls and results are displayed in SYSTEM LOG with colors.
  [+] Added error handling for agent execution.
"""

import os
import sys
import asyncio
import logging
import threading
import signal
import json
import yaml
import re
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Literal, Callable
from datetime import datetime

import dearpygui.dearpygui as dpg
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from openai import BadRequestError, APITimeoutError, AsyncOpenAI

# LangChain Core Imports for the Custom Model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun

# Local imports
from memory_manager import MemoryManager, DiaryEntryMetadata
from query_preprocessor import QueryPreprocessor
import logging_exceptions

# ============================================================================
# LlamaChatModel (Native LangChain Adapter - FIXED v2)
# ============================================================================

class LlamaChatModel(BaseChatModel):
    """
    LangChain-compatible adapter for local LLaMA/OpenAI-compatible servers.
    Supports streaming and reasoning_content extraction.
    """
    
    base_url: str = Field(default="http://127.0.0.1:8088/v1")
    model: str = Field(default="Falcon-H1R-7B-Q8_0.gguf")
    api_key: str = Field(default="not-needed")
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 180
    
    _client: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )
    
    @property
    def _llm_type(self) -> str:
        return "llama-chat"
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        """LangChain messages -> OpenAI format."""
        result = []
        for m in messages:
            if isinstance(m, SystemMessage):
                result.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
            else:
                result.append({"role": "user", "content": m.content})
        return result
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Синхронная обертка (обязательна)."""
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Генератор для стриминга."""
        openai_messages = self._convert_messages(messages)
        
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            reasoning = getattr(delta, 'reasoning_content', "") or ""
            
            lc_chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content, 
                    additional_kwargs={"reasoning_content": reasoning}
                )
            )
            yield lc_chunk

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Обычный вызов (собирает стрим в кучу)."""
        result_content = ""
        result_reasoning = ""
        async for chunk in self._astream(messages, stop, run_manager, **kwargs):
            result_content += chunk.message.content
            if "reasoning_content" in chunk.message.additional_kwargs:
                result_reasoning += chunk.message.additional_kwargs["reasoning_content"]
        
        ai_message = AIMessage(
            content=result_content,
            additional_kwargs={"reasoning_content": result_reasoning}
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("AI_EveryNyan")


# ============================================================================
# Application Configuration (Pydantic + YAML)
# ============================================================================

class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "bge-m3:latest"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048
    token_dump_threshold: int = 20000


class LlamaSettings(BaseModel):
    base_url: str = "http://localhost:8088/v1"
    api_key: str = ""
    chat_model: str = "Falcon-H1R-7B-Q8_0.gguf"
    timeout: int = 180
    temperature: float = 0.7
    max_tokens: int = 4096
    token_dump_threshold: int = 20000


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "everynyan_diary"
    embedding_dim: int = 1024


class DiarySettings(BaseModel):
    storage_dir: str = "data/diary"
    plagiarism_threshold: float = 0.97
    injection_max_length: int = 5000
    summary_prompt: str = ""


class GUISettings(BaseModel):
    title: str = "AI_EveryNyan"
    width: int = 900
    height: int = 700
    theme: str = "dark"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "logs/app.log"


class AntiRepeatSettings(BaseModel):
    trigger_avg: float = 0.73
    trigger_max: float = 0.69
    max_history: int = 32


class RAGSettings(BaseModel):
    top_k: int = 10
    similarity_threshold: float = 0.65
    enable_metadata_filtering: bool = False


class ContextSettings(BaseModel):
    max_history_messages: int = 40
    warn_if_context_exceeds: int = 20


class AppSettings(BaseSettings):
    chat_mode: Literal["ollama", "llama"] = "ollama"
    embedding_mode: Literal["ollama", "custom"] = "ollama"
    
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    llama: LlamaSettings = Field(default_factory=LlamaSettings)
    
    vector_db: QdrantSettings = Field(default_factory=QdrantSettings)
    diary: DiarySettings = Field(default_factory=DiarySettings)
    gui: GUISettings = Field(default_factory=GUISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    anti_repeat: AntiRepeatSettings = Field(default_factory=AntiRepeatSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    debug: bool = False

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @classmethod
    def from_yaml(cls, path: str) -> "AppSettings":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load settings from {path}: {e}")
            raise
    
    def get_chat_config(self):
        if self.chat_mode == "ollama":
            return self.ollama
        else:
            return self.llama
    
    def get_embedding_config(self):
        if self.embedding_mode == "ollama":
            return self.ollama
        else:
            logger.warning("Custom embedding mode not implemented, falling back to ollama")
            return self.ollama

    def get_chat_config(self):
        if self.chat_mode == "ollama":
            return self.ollama
        else:
            return self.llama

    def get_embedding_config(self):
        if self.embedding_mode == "ollama":
            return self.ollama
        else:
            logger.warning(
                "Custom embedding mode not implemented, falling back to ollama"
            )
            return self.ollama


# ============================================================================
# Character Configuration
# ============================================================================


class CharacterBaseConfig(BaseModel):
    meta: dict = Field(default_factory=dict)
    prompt: str


class CharacterAppearanceConfig(BaseModel):
    meta: dict = Field(default_factory=dict)
    freeform: str


class CharacterConfig:
    BASE_PATH = Path("config/character/base.yaml")
    APPEARANCE_PATH = Path("config/character/appearance.yaml")

    @staticmethod
    def _load_yaml_file(filepath: Path, model: type[BaseModel]) -> BaseModel:
        if not filepath.exists():
            raise FileNotFoundError(f"Character config not found: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return model.model_validate(data)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    @classmethod
    def load_base(cls) -> CharacterBaseConfig:
        return cls._load_yaml_file(cls.BASE_PATH, CharacterBaseConfig)

    @classmethod
    def load_appearance(cls) -> CharacterAppearanceConfig:
        return cls._load_yaml_file(cls.APPEARANCE_PATH, CharacterAppearanceConfig)


# ============================================================================
# Global Objects & State
# ============================================================================

settings: Optional[AppSettings] = None
character_base: Optional[CharacterBaseConfig] = None
character_appearance: Optional[CharacterAppearanceConfig] = None

qdrant_client: Optional[QdrantClient] = None
vector_store: Optional[QdrantVectorStore] = None
llm = None
embeddings: Optional[OpenAIEmbeddings] = None
memory_manager: Optional[MemoryManager] = None
mcp_client = None
react_agent = None

query_preprocessor: Optional[QueryPreprocessor] = None

session_context: List[Dict[str, str]] = []
anti_repeat_cache: List[Dict[str, Any]] = []

async_loop: Optional[asyncio.AbstractEventLoop] = None
async_thread: Optional[threading.Thread] = None

_shutting_down: bool = False
_current_ai_message_tag = None


# ============================================================================
# AI Thoughts UI System
# ============================================================================


def add_ai_thought(text: str, color: Tuple[int, int, int] = (200, 200, 150)):
    logger.info(f"[AI_THOUGHT] {text}")
    try:
        if dpg.does_item_exist("thoughts_placeholder"):
            dpg.delete_item("thoughts_placeholder")
        timestamp = datetime.now().strftime("%H:%M:%S")
        with dpg.group(parent="ai_thoughts_area", horizontal=True):
            dpg.add_text(f"[{timestamp}] ", color=(100, 100, 100))
            dpg.add_text(text, color=color)
        dpg.set_y_scroll("ai_thoughts_area", 1e9)
    except Exception as e:
        logger.debug(f"Thought UI update skipped: {e}")


def update_ai_message_streaming(text: str):
    global _current_ai_message_tag
    if _current_ai_message_tag and dpg.does_item_exist(_current_ai_message_tag):
        dpg.set_value(_current_ai_message_tag, text)
    else:
        with dpg.group(parent="chat_area", horizontal=False):
            with dpg.group(horizontal=True):
                dpg.add_text("AI_EveryNyan:", color=(255,200,100))
            with dpg.group(indent=20):
                _current_ai_message_tag = dpg.add_text(text, wrap=max(200, dpg.get_viewport_width()-150))
        dpg.set_y_scroll("chat_area", 1e9)


def finalize_ai_message_streaming():
    global _current_ai_message_tag
    _current_ai_message_tag = None


# ============================================================================
# Async Helpers
# ============================================================================


def run_async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def submit_to_async(coro) -> asyncio.Future:
    if async_loop is None or not async_loop.is_running():
        logger.warning("Async loop not ready, running synchronously")
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, async_loop)


# ============================================================================
# Component Initialization
# ============================================================================


def init_components():
    global qdrant_client, vector_store, llm, embeddings
    logger.info("Initializing components...")
    
    chat_cfg = settings.get_chat_config()
    embed_cfg = settings.get_embedding_config()
    
    logger.info(f"Chat mode: {settings.chat_mode}, endpoint: {chat_cfg.base_url}, model: {chat_cfg.chat_model}")
    logger.info(f"Embedding mode: {settings.embedding_mode}, endpoint: {embed_cfg.base_url}, model: {embed_cfg.embedding_model}")
    
    # Qdrant
    qdrant_client = QdrantClient(url=settings.vector_db.url)
    if not qdrant_client.collection_exists(settings.vector_db.collection):
        qdrant_client.create_collection(
            collection_name=settings.vector_db.collection,
            vectors_config=models.VectorParams(
                size=settings.vector_db.embedding_dim, distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Created collection: {settings.vector_db.collection}")
    
    # Embeddings
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
    
    # LLM Initialization (Updated for v0.16.0)
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


def init_character():
    global character_base, character_appearance
    logger.info("Loading character configuration...")
    character_base = CharacterConfig.load_base()
    character_appearance = CharacterConfig.load_appearance()
    logger.info("Character brain loaded correctly. All systems nominal")


def init_memory_manager():
    global memory_manager
    memory_manager = MemoryManager()
    stats = memory_manager.get_stats()
    logger.info(
        f"MemoryManager initialized. Messages: {stats.get('total_messages', 0)}"
    )


def init_query_preprocessor():
    global query_preprocessor
    query_preprocessor = QueryPreprocessor(add_thought_callback=add_ai_thought)
    logger.info("QueryPreprocessor initialized (spaCy lemmatization).")


def _to_lc_messages(raw_msgs: list[dict]) -> list:
    result = []
    for m in raw_msgs:
        role, content = m["role"], m["content"]
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        else:
            result.append(SystemMessage(content=content))
    return result

# import traceback
async def init_mcp_agent():
    global mcp_client, react_agent
    try:
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from langgraph.prebuilt import create_react_agent
        from tools.mcp import return_mcp_client
        from langchain_core.tools import StructuredTool

        searxng_url = getattr(settings, "searxng_url", "http://localhost:2597")
        mcp_client = return_mcp_client(SEARXNG_URL=searxng_url)

        raw_tools = await mcp_client.get_tools()
        if raw_tools:
            # Обёртка: извлекаем текст из списка, который возвращает MCP
            def unwrap_tool(original_tool):
                async def _wrapper(**kwargs):
                    result = await original_tool.ainvoke(kwargs)
                    # Типичный результат от MCP: [{'type': 'text', 'text': '...'}]
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

            # LlamaChatModel не реализует bind_tools() (NotImplementedError в BaseChatModel).
            # Для react-agent создаём ChatOpenAI, т.к. llama-server предоставляет
            # OpenAI-совместимый API с поддержкой function/tool calling.
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

        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        logger.warning(
            f"[MCP] Failed to initialize MCP agent: {type(e).__name__}: {e}\n{''.join(tb_str)}"
        )
        add_ai_thought(f"[MCP] Init skipped: {type(e).__name__}: {e}", (255, 200, 100))
        mcp_client = None
        react_agent = None

def init_query_preprocessor():
    global query_preprocessor
    query_preprocessor = QueryPreprocessor(add_thought_callback=add_ai_thought)
    logger.info("QueryPreprocessor initialized (spaCy lemmatization).")


# ============================================================================
# RAG & Memory Management
# ============================================================================

async def query_memory(query: str, top_k: Optional[int] = None,
                      filter_meta: Optional[Dict] = None) -> str:
    if top_k is None:
        top_k = settings.rag.top_k
    if not vector_store:
        add_ai_thought("[RAG] SKIP: Vector store not initialized", (200, 150, 150))
        return ""

    add_ai_thought(f"[RAG] QUERY: \"{query[:60]}{'...' if len(query)>60 else ''}\" (k={top_k})", (180,220,255))
    if filter_meta:
        add_ai_thought(f"[RAG] FILTER: {filter_meta}", (180,180,200))

    try:
        qdrant_filter = None
        if filter_meta and settings.rag.enable_metadata_filtering:
            must_conditions = []
            for key, value in filter_meta.items():
                if isinstance(value, list):
                    must_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchAny(any=value)))
                else:
                    must_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        docs_with_scores = await vector_store.asimilarity_search_with_score(query, k=top_k, filter=qdrant_filter)
        if not docs_with_scores:
            add_ai_thought(
                f"[RAG] RESULT: No documents found (k={top_k})", (200, 150, 150)
            )
            return ""

        if settings.rag.similarity_threshold > 0:
            docs_with_scores = [(doc, score) for doc, score in docs_with_scores
                                if score >= settings.rag.similarity_threshold]
            if not docs_with_scores:
                add_ai_thought(
                    f"[RAG] RESULT: No documents above threshold {settings.rag.similarity_threshold}",
                    (200, 150, 150),
                )
                return ""

        docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        add_ai_thought(f"[RAG] FOUND: {len(docs)} document(s) (requested {top_k})", (150,255,150))

        formatted_memories = []
        for i, (doc, score) in enumerate(zip(docs, scores)):
            snippet = doc.page_content[:80].replace("\n", " ")
            add_ai_thought(f"  [{i+1}] score={score:.3f} | {snippet}...", (200,200,200))
            content = doc.page_content[:800]
            formatted_memories.append(f"--- Воспоминание {i+1} (релевантность {score:.2f}) ---\n{content}")

        return "\n\n".join(formatted_memories)
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        add_ai_thought(f"[RAG] ERROR: {e}", (255, 100, 100))
        return ""


async def keyword_search_in_history(query: str, limit: int = 3) -> str:
    if not memory_manager:
        return ""
    keywords = [w for w in query.lower().split() if len(w) > 3]
    if not keywords:
        return ""
    try:
        conn = memory_manager.conn
        conditions = " OR ".join([f"LOWER(content) LIKE '%{kw}%'" for kw in keywords])
        rows = conn.execute(
            f"""
            SELECT role, content, timestamp FROM chat_history
            WHERE {conditions} ORDER BY timestamp DESC LIMIT ?
        """,
            [limit],
        ).fetchall()
        if not rows:
            return ""
        result_parts = []
        for role, content, ts in rows:
            sender = "User" if role == "user" else "AI"
            result_parts.append(f"[{ts}] {sender}: {content[:300]}")
        return "\n\n".join(result_parts)
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return ""


async def keyword_search_in_history(query: str, limit: int = 3) -> str:
    if not memory_manager:
        return ""
    keywords = [w for w in query.lower().split() if len(w) > 3]
    if not keywords:
        return ""
    try:
        conn = memory_manager.conn
        conditions = " OR ".join([f"LOWER(content) LIKE '%{kw}%'" for kw in keywords])
        rows = conn.execute(f"""
            SELECT role, content, timestamp FROM chat_history
            WHERE {conditions} ORDER BY timestamp DESC LIMIT ?
        """, [limit]).fetchall()
        if not rows:
            return ""
        result_parts = []
        for role, content, ts in rows:
            sender = "User" if role == 'user' else "AI"
            result_parts.append(f"[{ts}] {sender}: {content[:300]}")
        return "\n\n".join(result_parts)
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return ""


async def check_plagiarism(text: str, threshold: float) -> bool:
    if not vector_store or not qdrant_client:
        return False
    try:
        query_vector = await embeddings.aembed_query(text)
        results = qdrant_client.query_points(
            collection_name=settings.vector_db.collection,
            query=query_vector,
            limit=1,
            with_payload=False,
            with_vectors=False,
        ).points
        if results and results[0].score > threshold:
            add_ai_thought(f"[MEM] BLOCK: Duplicate (sim={results[0].score:.2f})", (255,150,150))
            return True
        return False
    except Exception as e:
        logger.warning(f"Plagiarism check failed: {e}")
        return False


async def _extract_dialogue_metadata(user_text: str, ai_response: str) -> Dict[str, Any]:
    # Works with both ChatOpenAI and LlamaChatModel as both support ainvoke
    if settings.chat_mode != "ollama":
        # Для локальных моделей можно пропустить сложное извлечение, если нет инструкций
        return {"entities": [], "topics": [], "key_facts": []}
    prompt = f"""
Extract metadata from this conversation:
User: {user_text[:500]}
Assistant: {ai_response[:500]}

Return a JSON object with:
- "entities": list of canonical names (people, places, systems)
- "topics": list of topic tags (e.g., "#coding", "#project_x")
- "key_facts": list of important quotes or facts (max 3)

Output ONLY valid JSON, no extra text.
"""
    try:
        response = await llm.ainvoke([("system", "You are a metadata extractor. Output only JSON."), ("human", prompt)])
        content = response.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "entities": data.get("entities", []),
                "topics": data.get("topics", []),
                "key_facts": data.get("key_facts", [])
            }
    except Exception as e:
        logger.warning(f"Dialogue metadata extraction failed: {e}")
    return {"entities": [], "topics": [], "key_facts": []}


async def dump_context_to_memory():
    global session_context
    if not session_context:
        add_ai_thought("[SYS] DUMP: idle (no context)", (150,150,150))
        return
    msg_count = len(session_context)
    add_ai_thought(f"[SUM] START: processing {msg_count} messages", (200,200,100))
    for i, msg in enumerate(session_context[:5]):
        add_ai_thought(f"  [{i}] {msg['role']}: {msg['content'][:40]}...", (150,150,180))
    try:
        dialogue_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session_context])
        full_prompt = [("system", settings.diary.summary_prompt), ("human", f"Here is the conversation to summarize:\n\n{dialogue_text}")]
        
        if settings.chat_mode == "ollama":
            response = await llm.ainvoke(full_prompt)
            diary_entry = response.content
        else:
            # Unified approach using ainvoke for LlamaChatModel too
            response = await llm.ainvoke(full_prompt)
            diary_entry = response.content
            
        diary_entry = diary_entry.replace("-- -", "---").replace("- --", "---").replace("----", "---")
        sections = [s.strip() for s in diary_entry.split("---") if s.strip()]
        total_sections = len(sections)
        saved_count = 0
        for idx, section in enumerate(sections):
            if len(section) < 20:
                continue
            json_str = "{}"
            json_block = re.search(r'```json\s*(\{.*?\})\s*```', section, re.DOTALL)
            if json_block:
                json_str = json_block.group(1)
                section = re.sub(r'```json.*?```', '', section, flags=re.DOTALL).strip()
            else:
                json_match = re.search(r'(\{.*\})', section, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    section = section.replace(json_str, "").strip()
            clean_section = section[:500] if section else "No content"
            base_meta = {"timestamp": datetime.now().isoformat(), "section": f"{idx+1}/{total_sections}", "source": "context_dump"}
            try:
                parsed_meta = DiaryEntryMetadata.from_json(json_str, base_meta)
            except Exception as e:
                logger.warning(f"JSON parse fallback: {e}")
                parsed_meta = DiaryEntryMetadata(**base_meta)
            if await check_plagiarism(clean_section, settings.diary.plagiarism_threshold):
                continue
            if query_preprocessor:
                lemmatized = query_preprocessor.lemmatize_text(clean_section, remove_stopwords=False)
                parsed_meta.type_specific["lemmatized"] = lemmatized
            vector_store.add_texts(texts=[clean_section], metadatas=[parsed_meta.to_qdrant_payload()])
            if memory_manager:
                memory_manager.save_diary_summary(text=clean_section, index=idx, total=total_sections, meta=parsed_meta)
            saved_count += 1
        add_ai_thought(f"[DB] FINISH: {saved_count}/{total_sections} stored", (100,255,100))
        session_context.clear()
    except Exception as e:
        logger.error(f"dump_context_to_memory failed: {e}")
        add_ai_thought(f"[ERR] Dump failed: {e}", (255,100,100))


def check_anti_repetition_semantic(new_content: str) -> bool:
    global anti_repeat_cache
    if not anti_repeat_cache or not embeddings:
        return False
    try:
        new_embedding = embeddings.embed_query(new_content)
        max_sim, avg_sim = 0.0, 0.0
        for cached in anti_repeat_cache:
            cached_emb = cached.get("embedding")
            if cached_emb is None:
                continue
            sim = (sum(a*b for a,b in zip(new_embedding, cached_emb)) + 1) / 2
            max_sim = max(max_sim, sim)
            avg_sim += sim
        if anti_repeat_cache:
            avg_sim /= len(anti_repeat_cache)
        if max_sim > settings.anti_repeat.trigger_max or avg_sim > settings.anti_repeat.trigger_avg:
            add_ai_thought(f"[ANTIREPEAT] BLOCKED: max={max_sim:.2f} avg={avg_sim:.2f}", (255,200,100))
            return True
        anti_repeat_cache.append({"content": new_content[:200], "embedding": new_embedding, "timestamp": datetime.now()})
        if len(anti_repeat_cache) > settings.anti_repeat.max_history:
            anti_repeat_cache.pop(0)
        return False
    except Exception as e:
        logger.warning(f"Anti-repetition failed: {e}")
        return False


def build_system_prompt() -> str:
    if not character_base or not character_appearance:
        return "You are a helpful assistant."
    return f"""{character_base.prompt}

<visual_reference>
{character_appearance.freeform}
</visual_reference>

<instructions>
- В начале диалога ты можешь увидеть блок "Я вспоминаю:" — это твои собственные воспоминания, извлечённые из долговременной памяти.
- ОБЯЗАТЕЛЬНО используй эту информацию, чтобы ответить пользователю. Если там есть конкретные факты, упомяни их.
- Не придумывай то, чего нет в воспоминаниях. Если нужной информации нет, честно скажи об этом.
- Будь милой, дружелюбной, оставайся в образе EveryNyan.
- Отвечай от первого лица, используй she/her.
- Не используй markdown, если не просят.
</instructions>"""


# ============================================================================
# Message Processing (UNIFIED LOGIC v0.16.1 with MCP LOGGING)
# ============================================================================


async def process_message(user_text: str) -> str:
    global session_context

    if check_anti_repetition_semantic(user_text):
        return "I feel like we're going in circles. Let's talk about something new!"

    if not session_context and memory_manager:
        add_ai_thought("[CTX] Context empty, loading recent history from DuckDB", (200,150,100))
        fresh = memory_manager.get_recent_history(limit=settings.context.max_history_messages)
        session_context.extend(fresh)
        add_ai_thought(f"[CTX] Loaded {len(fresh)} messages", (150,255,150))

    async def attempt_generation():
        add_ai_thought(f"[CTX] Current context size: {len(session_context)} messages", (150,180,200))
        if len(session_context) > settings.context.warn_if_context_exceeds:
            add_ai_thought(f"[CTX] WARN: Large context ({len(session_context)}), may need dump soon", (255,200,100))

        rag_context = await query_memory(user_text)
        if not rag_context and memory_manager:
            add_ai_thought("[RAG] No vectors, trying DuckDB keyword search", (200,150,100))
            keyword_results = await keyword_search_in_history(user_text, limit=3)
            if keyword_results:
                rag_context = f"--- Найдено в истории чата ---\n{keyword_results}"
                add_ai_thought(f"[DB] Keyword search found results", (150,255,150))

        system_prompt = build_system_prompt()
        
        # --- UNIFIED MESSAGE CONSTRUCTION ---
        messages = [SystemMessage(content=system_prompt)]
        
        if rag_context:
            messages.append(AIMessage(content=f"Я вспоминаю:\n{rag_context}"))

        max_hist = settings.context.max_history_messages
        recent = (
            session_context[-max_hist:]
            if len(session_context) > max_hist
            else session_context
        )

        for msg in recent:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_text))

        content = ""

        if react_agent:
            add_ai_thought("[MCP] Using react agent with tools", (100, 200, 255))
            try:
                result = await react_agent.ainvoke({"messages": messages})
            except Exception as agent_err:
                logger.error(f"MCP agent execution failed: {agent_err}", exc_info=True)
                add_ai_thought(f"[MCP] Agent error: {agent_err}. Falling back to direct LLM.", (255, 100, 100))
                # fallback to direct LLM (ollama or llama)
                if settings.chat_mode == "ollama":
                    response = await llm.ainvoke(messages)
                    content = response.content
                else:
                    full_content = ""
                    async for chunk in llm.astream(messages):
                        msg_chunk = chunk.message if hasattr(chunk, 'message') else chunk
                        if msg_chunk.content:
                            full_content += msg_chunk.content
                            update_ai_message_streaming(full_content)
                    content = full_content
                finalize_ai_message_streaming()
                return content

            # ========== MCP TOOL LOGGING ==========
            for msg in result["messages"]:
                # Tool call request (AIMessage with tool_calls)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        # Жёлтый цвет для вызова инструмента
                        add_ai_thought(f"[TOOL] Call: {tool_name} args={tool_args}", (255, 220, 100))
                        logger.info(f"MCP TOOL CALL: {tool_name} {tool_args}")
                
                # Tool result (ToolMessage)
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', 'unknown')
                    result_preview = msg.content[:200] + ('...' if len(msg.content) > 200 else '')
                    # Зелёный цвет для успешного результата
                    add_ai_thought(f"[TOOL] Result from {tool_name}: {result_preview}", (100, 255, 100))
                    logger.info(f"MCP TOOL RESULT ({tool_name}): {msg.content}")
                    
                    # Если в результате есть явная ошибка - покажем красным
                    if "error" in msg.content.lower() or "exception" in msg.content.lower():
                        add_ai_thought(f"[TOOL] Error in {tool_name}: {msg.content[:300]}", (255, 100, 100))
            # ======================================
            
            content = result["messages"][-1].content
            final_msg = result["messages"][-1]
            reasoning = ""
            if hasattr(final_msg, "additional_kwargs"):
                reasoning = (
                    final_msg.additional_kwargs.get("reasoning_content", "") or ""
                )
            if not reasoning and hasattr(final_msg, "response_metadata"):
                reasoning = (
                    final_msg.response_metadata.get("reasoning_content", "") or ""
                )
            if reasoning:
                add_ai_thought(f"[REASONING]\n{reasoning}", (180, 180, 150))
        elif settings.chat_mode == "ollama":
            response = await llm.ainvoke(messages)
            content = response.content
            reasoning = response.response_metadata.get("reasoning_content", "") or ""
            if reasoning:
                add_ai_thought(f"[REASONING]\n{reasoning}", (180,180,150))
        else: 
            # LLaMA Mode - streaming
            full_content = ""
            full_reasoning = ""
            async for chunk in llm.astream(messages):
                msg = chunk.message if hasattr(chunk, 'message') else chunk
                if msg.content:
                    full_content += msg.content
                    update_ai_message_streaming(full_content)
                if "reasoning_content" in msg.additional_kwargs:
                    full_reasoning += msg.additional_kwargs["reasoning_content"]
            if full_reasoning:
                add_ai_thought(f"[REASONING]\n{full_reasoning}", (180,180,150))
            content = full_content

        return content

    try:
        return await attempt_generation()
    except (BadRequestError, APITimeoutError, TimeoutError) as e:
        error_str = str(e).lower()
        if "context length" in error_str or "exceeds" in error_str:
            add_ai_thought(f"[WARN] CONTEXT OVERFLOW: {len(session_context)} messages", (255,150,100))
            await dump_context_to_memory()
            if memory_manager:
                fresh = memory_manager.get_recent_history(limit=settings.context.max_history_messages)
                session_context.extend(fresh)
                add_ai_thought(f"[CTX] Rehydrated: loaded {len(fresh)} messages", (150,255,150))
            return await attempt_generation()
        elif "timeout" in error_str:
            add_ai_thought("[WARN] LLM request timed out. Please try again with a shorter message.", (255,200,100))
            return "I'm sorry, I took too long to think. Could you please repeat your question or make it shorter?"
        else:
            logger.error(f"LLM request failed: {e}")
            return f"Sorry, I encountered an error: {e}"
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return f"Sorry, I encountered an error: {e}"


async def save_to_memory(user_text: str, ai_response: str):
    global session_context
    if not ai_response or len(ai_response.strip()) == 0:
        add_ai_thought("[RAG] Skipped saving empty response", (255,150,150))
        return

    if memory_manager:
        memory_manager.save_message("user", user_text)
        memory_manager.save_message("assistant", ai_response)
        add_ai_thought("[DB] Saved dialogue to chat_history", (150,200,150))

    session_context.append({"role": "user", "content": user_text})
    session_context.append({"role": "assistant", "content": ai_response})
    add_ai_thought(f"[CTX] Context size: {len(session_context)} messages", (150,180,200))

    if vector_store:
        content = f"User: {user_text}\nAI: {ai_response}"
        if len(content) > 2000:
            content = content[:2000].rsplit('.', 1)[0] + '.'

        meta = await _extract_dialogue_metadata(user_text, ai_response)
        meta.update({"type": "dialogue", "timestamp": datetime.now().isoformat()})

        if query_preprocessor:
            lemmatized = query_preprocessor.lemmatize_text(content, remove_stopwords=False)
            meta["lemmatized"] = lemmatized
            add_ai_thought(f"[RAG] Lemmatized copy stored (length {len(lemmatized)})", (150,200,150))

        try:
            vector_store.add_texts(texts=[content], metadatas=[meta])
            add_ai_thought(f"[RAG] Saved dialogue with metadata", (150,255,150))
        except Exception as e:
            logger.warning(f"Failed to save to Qdrant: {e}")


# ============================================================================
# GUI
# ============================================================================


async def report_qdrant_status():
    if not qdrant_client:
        add_ai_thought("[RAG] Status: Qdrant client not available", (200,150,150))
        return
    try:
        info = qdrant_client.get_collection(settings.vector_db.collection)
        add_ai_thought(f"[RAG] Qdrant collection '{settings.vector_db.collection}': {info.points_count} vectors", (150,255,150))
        scroll = qdrant_client.scroll(collection_name=settings.vector_db.collection, limit=3, with_payload=True, with_vectors=False)
        for p in scroll[0]:
            ts = p.payload.get("metadata", {}).get("timestamp", "no timestamp")
            preview = str(p.payload.get("page_content", ""))[:60]
            add_ai_thought(f"  - {ts}: {preview}...", (180,180,180))
    except Exception as e:
        add_ai_thought(f"[RAG] Status error: {e}", (255,100,100))


def on_memory_report():
    add_ai_thought("[SYS] Generating memory report...", (200,200,100))
    submit_to_async(report_qdrant_status())
    if memory_manager:
        stats = memory_manager.get_stats()
        add_ai_thought(f"[DB] DuckDB: {stats.get('total_messages',0)} msgs, {stats.get('total_summaries',0)} summaries", (150,255,150))
        summaries = memory_manager.get_diary_summaries(limit=3)
        if summaries:
            for s in summaries:
                add_ai_thought(f"  - {s['timestamp']}: {s['text'][:80]}...", (180,180,180))


def find_available_font() -> Optional[str]:
    local = Path("data/fonts")
    for f in ["JetBrainsMonoNerdFont-Regular.ttf", "JetBrainsMonoNerdFont-Medium.ttf", "JetBrainsMonoNerdFont-Bold.ttf"]:
        p = local / f
        if p.exists():
            return str(p)
    for f in [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]:
        if Path(f).exists():
            return f
    return None


def setup_gui():
    dpg.create_context()
    font_path = find_available_font()
    if font_path:
        with dpg.font_registry():
            with dpg.font(font_path, 16) as main_font:
                pass
        dpg.bind_font(main_font)
    dpg.create_viewport(title=settings.gui.title, width=settings.gui.width, height=settings.gui.height, resizable=True)
    if settings.gui.theme == "dark":
        with dpg.theme() as dark_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25,25,35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40,40,60))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50,50,80))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220,220,220))
        dpg.bind_theme(dark_theme)
    with dpg.window(label="Chat", tag="main_window", no_title_bar=True, no_move=True, no_resize=False):
        with dpg.child_window(tag="chat_area", height=-200, border=False):
            if memory_manager:
                history = memory_manager.get_recent_history(limit=50)
                for msg in history:
                    color = (100,200,255) if msg['role'] == 'user' else (255,200,100)
                    sender = "You" if msg['role'] == 'user' else "AI_EveryNyan"
                    with dpg.group(horizontal=False):
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{sender}:", color=color)
                        with dpg.group(indent=20):
                            dpg.add_text(msg['content'], wrap=max(200, dpg.get_viewport_width()-150))
                        dpg.add_spacer(height=5)
            else:
                dpg.add_text("Welcome to AI_EveryNyan!", color=(150,150,200))
        with dpg.child_window(tag="ai_thoughts_area", height=120, label="[SYSTEM] LOG", border=True):
            dpg.add_text("[SYSTEM] STATUS: Idle", tag="thoughts_placeholder", color=(100,100,100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="user_input", width=-180, hint="Type your message...", on_enter=True, callback=on_send_message)
            dpg.add_button(label="Send", callback=on_send_message, width=80)
            dpg.add_button(label="Memory Report", callback=on_memory_report, width=100)
        dpg.add_text("", tag="status_text", color=(100,100,100))
    dpg.setup_dearpygui()
    dpg.show_viewport()


def add_chat_message(sender: str, text: str, color: tuple):
    dpg.set_y_scroll("chat_area", 1e9)
    with dpg.group(parent="chat_area", horizontal=False):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{sender}:", color=color)
        with dpg.group(indent=20):
            dpg.add_text(text, wrap=max(200, dpg.get_viewport_width()-150))
        dpg.add_spacer(height=5)
    dpg.set_y_scroll("chat_area", 1e9)


def on_send_message(sender, app_data):
    global _shutting_down
    if _shutting_down:
        return
    user_text = dpg.get_value("user_input").strip()
    if not user_text:
        return
    add_chat_message("You", user_text, (100,200,255))
    dpg.set_value("user_input", "")
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Thinking...")
    add_ai_thought(f"[IN] User: \"{user_text[:30]}{'...' if len(user_text)>30 else ''}\"", (150,200,255))
    future = submit_to_async(handle_async_response(user_text))
    def on_done(fut):
        try:
            fut.result()
        except Exception as e:
            logger.exception(f"Task failed: {e}")
            dpg.configure_item("user_input", enabled=True)
            dpg.set_value("status_text", f"Error: {e}")
    future.add_done_callback(on_done)


async def handle_async_response(user_text: str):
    try:
        response = await process_message(user_text)
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("AI_EveryNyan", response, (255,200,100))
        add_ai_thought("[SYS] Response generated.", (150,255,150))
        await save_to_memory(user_text, response)
    except Exception as e:
        logger.exception("Unhandled error")
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("Error", str(e), (255,100,100))
        add_ai_thought(f"[ERR] Handler Error: {e}", (255,100,100))


# ============================================================================
# Graceful Shutdown
# ============================================================================


def initiate_graceful_shutdown():
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    add_ai_thought("[SYS] SHUTDOWN: Saving data...", (255,150,150))
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Saving memories...")
    if session_context:
        try:
            future = asyncio.run_coroutine_threadsafe(
                dump_context_to_memory(), async_loop
            )
            future.result(timeout=30)
            add_ai_thought("[SYS] STATUS: Save complete.", (150,255,150))
        except Exception as e:
            logger.error(f"Shutdown dump failed: {e}")
    dpg.stop_dearpygui()


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    initiate_graceful_shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    global settings, async_loop, async_thread
    config_path = Path("config/settings.yaml")
    settings = AppSettings.from_yaml(str(config_path))
    Path(settings.diary.storage_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("hf_cache").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    if settings.debug:
        logging_exceptions.install_excepthook()

    logger.info(f"Starting AI_EveryNyan v0.16.2 (debug={settings.debug})")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    init_memory_manager()
    init_components()
    init_character()

    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(
        target=run_async_loop, args=(async_loop,), daemon=True
    )
    async_thread.start()

    setup_gui()
    init_query_preprocessor()

    future = asyncio.run_coroutine_threadsafe(init_mcp_agent(), async_loop)
    try:
        future.result(timeout=30)
    except Exception as e:
        logger.warning(f"[MCP] Agent initialization failed: {e}")
    add_ai_thought("[SYS] STATUS: Online. Ready.", (100, 255, 100))

    try:
        dpg.start_dearpygui()
    finally:
        add_ai_thought("[SYS] SHUTDOWN: Saving data...", (255, 150, 150))
        if session_context:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    dump_context_to_memory(), async_loop
                )
                future.result(timeout=300)
            except Exception as e:
                logger.error(f"Final dump failed: {e}")
        if mcp_client:
            logger.info("[MCP] Client released (no explicit close needed)")
        async_loop.call_soon_threadsafe(async_loop.stop)
        async_thread.join(timeout=2.0)
        if memory_manager:
            memory_manager.close()
        dpg.destroy_context()
        logger.info("[SYS] Shutdown complete.")


if __name__ == "__main__":
    main()