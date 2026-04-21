#!/usr/bin/env python3
"""
AI_EveryNyan - DearPyGui Chat with LangChain + Qdrant RAG + DuckDB History
Modular Character System + Smart Context Management + Structured Diary Metadata

\src\main.py
Version:     0.9.0
Author:      Soror L.'.L.'.
Updated:     2026-04-21

Patch Notes v0.9.0:
  [+] RAG: added metadata extraction for dialogue entries (entities, topics, key_facts).
  [+] Fixed dead adaptive threshold code (removed current_relevance_threshold).
  [+] Improved embedding normalization (normalize=True) for better vector similarity.
  [+] Enhanced JSON metadata parsing in dump_context_to_memory (robust regex fallback).
  [+] All AI thoughts now also logged to console/file.
  [*] Roadmap: re-ranker (3.3) and query preprocessing (3.4) planned.

Previous versions:
  v0.8.0: plain text RAG, DuckDB keyword fallback, auto-load history.
  v0.7.0: moved parameters to settings.yaml, JSON metadata, circumplex affect.
  v0.6.0: detailed memory reporting, Memory Report button.
  v0.5.0: universal diary metadata schema, Qdrant payload filtering.
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
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime

import dearpygui.dearpygui as dpg
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint, Filter, FieldCondition, MatchValue, MatchAny
from openai import BadRequestError

# Local imports
from memory_manager import MemoryManager, DiaryEntryMetadata

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger("AI_EveryNyan")


# ============================================================================
# Application Configuration (Pydantic + YAML)
# ============================================================================

class LLMSettings(BaseModel):
    backend: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "bge-m3:latest"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048
    token_dump_threshold: int = 20000


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "everynyan_diary"
    embedding_dim: int = 1024


class DiarySettings(BaseModel):
    storage_dir: str = "data/diary"
    plagiarism_threshold: float = 0.97
    injection_max_length: int = 5000
    summary_prompt: str = ""  # загружается из YAML


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
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: QdrantSettings = Field(default_factory=QdrantSettings)
    diary: DiarySettings = Field(default_factory=DiarySettings)
    gui: GUISettings = Field(default_factory=GUISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    anti_repeat: AntiRepeatSettings = Field(default_factory=AntiRepeatSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    debug: bool = False
    
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    @classmethod
    def from_yaml(cls, path: str) -> "AppSettings":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load settings from {path}: {e}")
            raise


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
llm: Optional[ChatOpenAI] = None
embeddings: Optional[OpenAIEmbeddings] = None
memory_manager: Optional[MemoryManager] = None

session_context: List[Dict[str, str]] = []
anti_repeat_cache: List[Dict[str, Any]] = []

async_loop: Optional[asyncio.AbstractEventLoop] = None
async_thread: Optional[threading.Thread] = None

_shutting_down: bool = False


# ============================================================================
# AI Thoughts UI System (with console logging)
# ============================================================================

def add_ai_thought(text: str, color: Tuple[int, int, int] = (200, 200, 150)):
    """Add a thought to the AI thoughts panel and also log to console."""
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
        logger.debug(f"Thought UI update skipped (early init): {e}")


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
    qdrant_client = QdrantClient(url=settings.vector_db.url)
    if not qdrant_client.collection_exists(settings.vector_db.collection):
        qdrant_client.create_collection(
            collection_name=settings.vector_db.collection,
            vectors_config=models.VectorParams(
                size=settings.vector_db.embedding_dim,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Created collection: {settings.vector_db.collection}")
    
    # 3.5: добавлена нормализация эмбеддингов
    embeddings = OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key="ollama",
        openai_api_base=settings.llm.base_url,
        # model_kwargs={"normalize": True},  # нормализация векторов
        check_embedding_ctx_length=False
        
    )
    
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.vector_db.collection,
        embedding=embeddings
    )
    llm = ChatOpenAI(
        model=settings.llm.chat_model,
        openai_api_key="ollama",
        openai_api_base=settings.llm.base_url,
        temperature=settings.llm.temperature,
        timeout=settings.llm.timeout,
        max_tokens=settings.llm.max_tokens,
        streaming=False
    )
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
    logger.info(f"MemoryManager initialized. Messages: {stats.get('total_messages', 0)}")


# ============================================================================
# RAG & Memory Management
# ============================================================================

async def query_memory(query: str, top_k: Optional[int] = None, 
                      filter_meta: Optional[Dict] = None) -> str:
    """Semantic search in Qdrant with plain text output."""
    if top_k is None:
        top_k = settings.rag.top_k
    if not vector_store:
        add_ai_thought("[RAG] SKIP: Vector store not initialized", (200,150,150))
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
            add_ai_thought(f"[RAG] RESULT: No documents found (k={top_k})", (200,150,150))
            return ""
        
        if settings.rag.similarity_threshold > 0:
            docs_with_scores = [(doc, score) for doc, score in docs_with_scores 
                                if score >= settings.rag.similarity_threshold]
            if not docs_with_scores:
                add_ai_thought(f"[RAG] RESULT: No documents above threshold {settings.rag.similarity_threshold}", (200,150,150))
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
        add_ai_thought(f"[RAG] ERROR: {e}", (255,100,100))
        return ""


# ---------- DuckDB keyword fallback ----------
async def keyword_search_in_history(query: str, limit: int = 3) -> str:
    """Fallback: search chat_history by keywords."""
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
            sender = "User" if role == "user" else "AI"
            result_parts.append(f"[{ts}] {sender}: {content[:300]}")
        return "\n\n".join(result_parts)
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return ""


# ---------- Plagiarism check ----------
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
            with_vectors=False
        ).points
        if results and results[0].score > threshold:
            add_ai_thought(f"[MEM] BLOCK: Duplicate (sim={results[0].score:.2f})", (255,150,150))
            return True
        return False
    except Exception as e:
        logger.warning(f"Plagiarism check failed: {e}")
        return False


# ---------- Metadata extraction for dialogues (2.8) ----------
async def _extract_dialogue_metadata(user_text: str, ai_response: str) -> Dict[str, Any]:
    """Extract entities, topics, key facts from a dialogue using LLM."""
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
        content = response.content.strip()
        # Извлекаем JSON из ответа
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


# ---------- Smart context dumping (improved JSON parsing) ----------
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
        response = await llm.ainvoke(full_prompt)
        diary_entry = response.content
        diary_entry = diary_entry.replace("-- -", "---").replace("- --", "---").replace("----", "---")
        sections = [s.strip() for s in diary_entry.split("---") if s.strip()]
        total_sections = len(sections)
        saved_count = 0
        for idx, section in enumerate(sections):
            if len(section) < 20:
                continue
            
            # 3.7: улучшенный парсинг JSON
            json_str = "{}"
            # Сначала ищем блок ```json ... ```
            json_block = re.search(r'```json\s*(\{.*?\})\s*```', section, re.DOTALL)
            if json_block:
                json_str = json_block.group(1)
                # Удаляем JSON-блок из текста
                section = re.sub(r'```json.*?```', '', section, flags=re.DOTALL).strip()
            else:
                # Ищем любой JSON-объект в тексте
                json_match = re.search(r'(\{.*\})', section, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # Пытаемся удалить найденный JSON из текста
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


# ============================================================================
# Prompt Building
# ============================================================================

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
# Message Processing
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
        
        messages = [{"role": "system", "content": build_system_prompt()}]
        
        if rag_context:
            messages.append({
                "role": "assistant",
                "content": f"Я вспоминаю:\n{rag_context}"
            })
        
        max_hist = settings.context.max_history_messages
        recent = session_context[-max_hist:] if len(session_context) > max_hist else session_context
        messages.extend(recent)
        messages.append({"role": "user", "content": user_text})
        
        response = await llm.ainvoke(messages)
        return response.content
    
    try:
        return await attempt_generation()
    except BadRequestError as e:
        if "context length" in str(e).lower() or "exceeds" in str(e).lower():
            add_ai_thought(f"[WARN] CONTEXT OVERFLOW: {len(session_context)} messages", (255,150,100))
            await dump_context_to_memory()
            if memory_manager:
                fresh = memory_manager.get_recent_history(limit=settings.context.max_history_messages)
                session_context.extend(fresh)
                add_ai_thought(f"[CTX] Rehydrated: loaded {len(fresh)} messages", (150,255,150))
            return await attempt_generation()
        else:
            raise
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return f"Sorry, I encountered an error: {e}"


async def save_to_memory(user_text: str, ai_response: str):
    global session_context
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
        
        # 2.8: извлекаем метаданные диалога
        meta = await _extract_dialogue_metadata(user_text, ai_response)
        meta.update({"type": "dialogue", "timestamp": datetime.now().isoformat()})
        
        try:
            vector_store.add_texts(texts=[content], metadatas=[meta])
            add_ai_thought(f"[RAG] Saved dialogue with metadata: entities={meta.get('entities', [])[:2]}...", (150,255,150))
        except Exception as e:
            logger.warning(f"Failed to save to Qdrant: {e}")


# ============================================================================
# GUI и вспомогательные функции
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
            future = asyncio.run_coroutine_threadsafe(dump_context_to_memory(), async_loop)
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
    
    logger.info(f"Starting AI_EveryNyan v0.9.0 (debug={settings.debug})")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    init_memory_manager()
    init_components()
    init_character()
    
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(async_loop,), daemon=True)
    async_thread.start()
    
    setup_gui()
    add_ai_thought("[SYS] STATUS: Online. Ready.", (100,255,100))
    try:
        dpg.start_dearpygui()
    finally:
        add_ai_thought("[SYS] SHUTDOWN: Saving data...", (255,150,150))
        if session_context:
            try:
                future = asyncio.run_coroutine_threadsafe(dump_context_to_memory(), async_loop)
                future.result(timeout=300)
            except Exception as e:
                logger.error(f"Final dump failed: {e}")
        async_loop.call_soon_threadsafe(async_loop.stop)
        async_thread.join(timeout=2.0)
        if memory_manager:
            memory_manager.close()
        dpg.destroy_context()
        logger.info("[SYS] Shutdown complete.")


if __name__ == "__main__":
    main()