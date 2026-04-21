#!/usr/bin/env python3
"""
AI_EveryNyan - DearPyGui Chat with LangChain + Qdrant RAG + DuckDB History
Modular Character System + Smart Context Management + Structured Diary Metadata

\src\main.py
Version:     0.7.0
Author:      Soror L.'.L.'.
Updated:     2026-04-21

Patch Notes v0.7.0:
  [+] All RAG, LLM and context parameters moved to settings.yaml
  [+] Added RAGSettings (top_k, similarity_threshold)
  [+] Added ContextSettings (max_history_messages, warn_if_context_exceeds)
  [+] LLMSettings now includes temperature and max_tokens
  [+] query_memory uses settings.rag.top_k as default
  [+] process_message uses settings.context.max_history_messages and warn_if_context_exceeds
  [+] Retry/rehydration uses settings.context.max_history_messages for limit

Patch Notes v0.6.0:
  [+] Enhanced memory reporting: detailed RAG query logs, context dump steps,
      sliding window events, Qdrant status, DuckDB statistics.
  [+] Added "Memory Report" button to GUI.
  [+] Improved anti-plagiarism and anti-repetition feedback.

Patch Notes v0.5.0:
  [+] Integrated universal diary metadata schema (entities, topics, emotion, etc.)
  [+] Added metadata parsing from LLM output via DiaryEntryMetadata model
  [+] Updated Qdrant operations to use structured payload filtering
  [+] Enhanced RAG queries with metadata-aware filtering support
  [-] Removed emojis; strict technical log format preserved throughout
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
    summary_prompt: str = """
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
**Emotion/Affect:** [Valence: pos/neg/neu, Arousal: high/med/low, Specific emotion]
**Relationships:** [How entities relate to each other or to you]

**Retrieval Cues:** ["search phrase 1", "search phrase 2", "search phrase 3"]
**Photo Descriptions:** [If applicable: detailed visual description + filename]
**Contradictions/Uncertainties:** [Conflicts, missing info, or things needing verification]
</output_format>
"""


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
    top_k: int = 3
    similarity_threshold: float = 0.0
    enable_metadata_filtering: bool = False


class ContextSettings(BaseModel):
    max_history_messages: int = 10
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
current_relevance_threshold: float = 0.5

async_loop: Optional[asyncio.AbstractEventLoop] = None
async_thread: Optional[threading.Thread] = None

# Graceful shutdown flag
_shutting_down: bool = False


# ============================================================================
# AI Thoughts UI System (Direct Updates)
# ============================================================================

def add_ai_thought(text: str, color: Tuple[int, int, int] = (200, 200, 150)):
    """Add a thought to the AI thoughts panel using technical format."""
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
    
    embeddings = OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key="ollama",
        openai_api_base=settings.llm.base_url,
        check_embedding_ctx_length=False,
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
    """
    Semantic search in Qdrant with optional metadata filtering.
    Reports details to UI and log.
    
    Args:
        query: Search query
        top_k: Number of results (default from settings.rag.top_k)
        filter_meta: Optional dict for metadata filtering
    """
    global current_relevance_threshold
    
    if top_k is None:
        top_k = settings.rag.top_k
    
    if not vector_store:
        add_ai_thought("[RAG] SKIP: Vector store not initialized", (200,150,150))
        return ""
    
    # Отчет о запросе
    add_ai_thought(f"[RAG] QUERY: \"{query[:60]}{'...' if len(query)>60 else ''}\" (k={top_k})", (180,220,255))
    if filter_meta:
        add_ai_thought(f"[RAG] FILTER: {filter_meta}", (180,180,200))
    
    try:
        # Build Qdrant filter if metadata filter provided
        qdrant_filter = None
        if filter_meta:
            must_conditions = []
            for key, value in filter_meta.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchAny(any=value)
                        )
                    )
                else:
                    must_conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)
        
        # Используем similarity_search_with_score для получения реальных score
        docs_with_scores = await vector_store.asimilarity_search_with_score(
            query, k=top_k, filter=qdrant_filter
        )
        
        if not docs_with_scores:
            add_ai_thought(f"[RAG] RESULT: No documents found (k={top_k})", (200,150,150))
            return ""
        
        # Фильтрация по similarity_threshold, если задан
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
            
            memory_xml = (
                f'<memory_piece id="{i}" relatedness="{score:.3f}">\n'
                f'{doc.page_content}\n'
                f'</memory_piece>'
            )
            formatted_memories.append(memory_xml)
        
        result = "\n\n".join(formatted_memories)
        
        # Adaptive threshold logic
        if formatted_memories:
            if len(formatted_memories) < top_k:
                current_relevance_threshold = max(0.3, current_relevance_threshold * 0.95)
                add_ai_thought(f"[RAG] THRESHOLD: lowered to {current_relevance_threshold:.2f}", (150,150,200))
            elif len(formatted_memories) == top_k:
                current_relevance_threshold = min(0.9, current_relevance_threshold * 1.05)
                add_ai_thought(f"[RAG] THRESHOLD: raised to {current_relevance_threshold:.2f}", (150,150,200))
        
        return result
        
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        add_ai_thought(f"[RAG] ERROR: {e}", (255,100,100))
        return ""


async def check_plagiarism(text: str, threshold: float) -> bool:
    """Check if text is too similar to existing entries in Qdrant."""
    if not vector_store or not qdrant_client:
        return False
    
    try:
        query_vector = await embeddings.aembed_query(text)
        
        results: List[ScoredPoint] = qdrant_client.query_points(
            collection_name=settings.vector_db.collection,
            query=query_vector,
            limit=1,
            with_payload=False,
            with_vectors=False
        ).points
        
        if results:
            similarity = results[0].score
            if similarity > threshold:
                logger.info(
                    f"[MEM] Skipped duplicate. "
                    f"Similarity: {similarity:.3f} > {threshold}"
                )
                add_ai_thought(f"[MEM] BLOCK: Duplicate content (similarity: {similarity:.2f})", (255, 150, 150))
                return True
        
        return False
        
    except Exception as e:
        logger.warning(f"Plagiarism check failed: {e}")
        return False


async def dump_context_to_memory():
    """
    Smart context dumping with JSON metadata extraction.
    LLM returns a diary section + a JSON block with structured metadata.
    """
    global session_context
    
    if not session_context:
        logger.info("[SYS] Context empty, nothing to dump")
        add_ai_thought("[SYS] DUMP: idle (no context)", (150,150,150))
        return
    
    msg_count = len(session_context)
    logger.info(f"[SYS] Dumping {msg_count} messages to memory...")
    add_ai_thought(f"[SUM] START: processing {msg_count} messages", (200,200,100))
    
    for i, msg in enumerate(session_context[:5]):
        add_ai_thought(f"  [{i}] {msg['role']}: {msg['content'][:40]}...", (150,150,180))
    
    try:
        dialogue_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session_context])
        
        full_prompt = [
            ("system", settings.diary.summary_prompt),
            ("human", f"Here is the conversation to summarize:\n\n{dialogue_text}")
        ]
        
        add_ai_thought("[LLM] Generating structured diary reflections with JSON metadata...", (150,200,255))
        response = await llm.ainvoke(full_prompt)
        diary_entry = response.content
        
        # Split by ---
        diary_entry = diary_entry.replace("-- -", "---").replace("- --", "---").replace("----", "---")
        sections = [s.strip() for s in diary_entry.split("---") if s.strip()]
        
        total_sections = len(sections)
        add_ai_thought(f"[PROC] LLM produced {total_sections} section(s)", (200,180,255))
        
        saved_count = 0
        for idx, section in enumerate(sections):
            if len(section) < 20:
                add_ai_thought(f"  Section {idx+1}: skipped (too short)", (200,150,150))
                continue
            
            # Extract JSON metadata from the section
            # Expected format: ... text ... ```json { ... } ``` or just {...}
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', section, re.DOTALL)
            if not json_match:
                # Fallback: find first { and last }
                start = section.find('{')
                end = section.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = section[start:end+1]
                else:
                    json_str = "{}"
            else:
                json_str = json_match.group(1)
            
            # Clean the main text: remove the JSON block and any trailing markers
            clean_section = re.sub(r'```json.*?```', '', section, flags=re.DOTALL).strip()
            if not clean_section:
                clean_section = section[:500]  # fallback
            
            # Parse metadata using DiaryEntryMetadata.from_json
            base_meta = {
                "timestamp": datetime.now().isoformat(),
                "section": f"{idx+1}/{total_sections}",
                "source": "context_dump"
            }
            try:
                parsed_meta = DiaryEntryMetadata.from_json(json_str, base_meta)
                add_ai_thought(f"  Section {idx+1}: valence={parsed_meta.affect_valence}, arousal={parsed_meta.affect_arousal}, label={parsed_meta.emotion_label}", (180,220,180))
                if parsed_meta.entities:
                    add_ai_thought(f"    Entities: {parsed_meta.entities[:3]}", (180,200,180))
                if parsed_meta.topics:
                    add_ai_thought(f"    Topics: {parsed_meta.topics[:3]}", (180,200,180))
            except Exception as e:
                logger.warning(f"JSON metadata parse error: {e}")
                add_ai_thought(f"  Section {idx+1}: metadata parse failed, using minimal", (200,150,150))
                parsed_meta = DiaryEntryMetadata(**base_meta)
            
            # Anti-plagiarism check
            is_duplicate = await check_plagiarism(clean_section, settings.diary.plagiarism_threshold)
            if is_duplicate:
                add_ai_thought(f"  Section {idx+1}: DUPLICATE, skipped", (255,150,150))
                continue
            
            # Save to Qdrant
            try:
                vector_store.add_texts(
                    texts=[clean_section],
                    metadatas=[parsed_meta.to_qdrant_payload()]
                )
                add_ai_thought(f"  Section {idx+1}: written to Qdrant", (150,255,150))
                
                if memory_manager:
                    memory_manager.save_diary_summary(
                        text=clean_section,
                        index=idx,
                        total=total_sections,
                        meta=parsed_meta
                    )
                    add_ai_thought(f"  Section {idx+1}: written to DuckDB", (150,255,150))
                
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save section {idx}: {e}")
                add_ai_thought(f"  Section {idx+1}: ERROR - {e}", (255,100,100))
        
        logger.info(f"[SYS] Saved {saved_count}/{total_sections} unique reflections")
        add_ai_thought(f"[DB] FINISH: {saved_count}/{total_sections} stored", (100,255,100))
        session_context.clear()
        add_ai_thought("[SYS] In-memory context cleared", (150,150,150))
        
    except Exception as e:
        logger.error(f"[SYS] Failed to dump context: {e}")
        add_ai_thought(f"[ERR] Dump failed: {e}", (255,100,100))


def check_anti_repetition_semantic(new_content: str) -> bool:
    """Semantic anti-repetition check."""
    global anti_repeat_cache
    
    if not anti_repeat_cache or not embeddings:
        return False
    
    try:
        new_embedding = embeddings.embed_query(new_content)
        
        max_similarity = 0.0
        avg_similarity = 0.0
        
        for cached in anti_repeat_cache:
            cached_embedding = cached.get("embedding")
            if cached_embedding is None:
                continue
            
            similarity = sum(a * b for a, b in zip(new_embedding, cached_embedding))
            similarity = (similarity + 1) / 2
            
            max_similarity = max(max_similarity, similarity)
            avg_similarity += similarity
        
        if anti_repeat_cache:
            avg_similarity /= len(anti_repeat_cache)
        
        if max_similarity > settings.anti_repeat.trigger_max:
            logger.warning(
                f"[ANTIREPEAT] Max similarity {max_similarity:.3f} > "
                f"{settings.anti_repeat.trigger_max}"
            )
            add_ai_thought(f"[ANTIREPEAT] BLOCKED: max_sim={max_similarity:.2f}", (255,200,100))
            return True
        
        if avg_similarity > settings.anti_repeat.trigger_avg:
            logger.warning(
                f"[ANTIREPEAT] Avg similarity {avg_similarity:.3f} > "
                f"{settings.anti_repeat.trigger_avg}"
            )
            add_ai_thought(f"[ANTIREPEAT] BLOCKED: avg_sim={avg_similarity:.2f}", (255,200,100))
            return True
        
        anti_repeat_cache.append({
            "content": new_content[:200],
            "embedding": new_embedding,
            "timestamp": datetime.now()
        })
        
        if len(anti_repeat_cache) > settings.anti_repeat.max_history:
            anti_repeat_cache.pop(0)
        
        return False
        
    except Exception as e:
        logger.warning(f"Anti-repetition check failed: {e}")
        return False


# ============================================================================
# Prompt Building
# ============================================================================

def build_system_prompt() -> str:
    """Assemble final system prompt from character config + instructions."""
    if not character_base or not character_appearance:
        return "You are a helpful assistant."
    
    return f"""{character_base.prompt}

<visual_reference>
{character_appearance.freeform}
</visual_reference>

<instructions>
- Use retrieved memories if they are relevant to the user's query.
- Be concise, friendly, and stay in character.
- Do not invent facts. If unsure, ask for clarification.
- When referring to yourself, use the name and pronouns defined in your base prompt.
- Format your responses naturally; avoid markdown unless requested.
</instructions>"""


# ============================================================================
# Message Processing (Main Orchestrator)
# ============================================================================

async def process_message(user_text: str) -> str:
    """Main message processing loop with Sliding Window support."""
    global session_context
    
    # 1. Anti-repetition check
    if check_anti_repetition_semantic(user_text):
        add_ai_thought("[AI] LOOP: Repetition detected. Switching topic.", (255, 200, 100))
        return "I feel like we're going in circles. Let's talk about something new!"
    
    # Inner function for retry logic
    async def attempt_generation():
        # Report current context size
        add_ai_thought(f"[CTX] Current context size: {len(session_context)} messages", (150,180,200))
        if len(session_context) > settings.context.warn_if_context_exceeds:
            add_ai_thought(f"[CTX] WARN: Large context ({len(session_context)}), may need dump soon", (255,200,100))
        
        # 2. RAG: retrieve relevant memories
        rag_context = await query_memory(user_text)
        
        # 3. Build system prompt
        system_prompt = build_system_prompt()
        
        # 4. Assemble messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"<retrieved_memories>\n{rag_context}\n</retrieved_memories>"
            })
        
        max_history = settings.context.max_history_messages
        recent = session_context[-max_history:] if len(session_context) > max_history else session_context
        messages.extend(recent)
        
        messages.append({"role": "user", "content": user_text})
        
        # 5. Call LLM
        response = await llm.ainvoke(messages)
        return response.content
    
    try:
        return await attempt_generation()
        
    except BadRequestError as e:
        if "context length" in str(e).lower() or "exceeds" in str(e).lower():
            logger.warning(f"[SYS] Context length exceeded (size={len(session_context)}). Dumping...")
            add_ai_thought(f"[WARN] CONTEXT OVERFLOW: {len(session_context)} messages, threshold ~{settings.llm.token_dump_threshold} tokens", (255,150,100))
            
            # Step 1: Summarize and save old context
            await dump_context_to_memory()
            
            # Step 2: Rehydrate context from DuckDB
            if memory_manager:
                logger.info("[SYS] Rehydrating context from DuckDB...")
                add_ai_thought("[DB] Loading recent history...", (150,200,255))
                fresh_history = memory_manager.get_recent_history(limit=settings.context.max_history_messages)
                
                if fresh_history:
                    session_context.extend(fresh_history)
                    logger.info(f"[SYS] Rehydrated with {len(fresh_history)} messages")
                    add_ai_thought(f"[CTX] Rehydrated: loaded {len(fresh_history)} messages from DuckDB", (150,255,150))
                    for msg in fresh_history:
                        add_ai_thought(f"  -> {msg['role']}: {msg['content'][:40]}...", (150,180,200))
            
            # Step 3: Retry generation
            logger.info("[SYS] Retrying generation...")
            add_ai_thought("[SYS] RETRY: Generating response...", (200,200,100))
            try:
                return await attempt_generation()
            except Exception as retry_e:
                logger.error(f"[SYS] Retry failed: {retry_e}")
                return "I tried to organize my memories, but my head is still spinning. Could we try a shorter sentence?"
        else:
            raise
            
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        add_ai_thought(f"[ERR] LLM Error: {e}", (255,100,100))
        return f"Sorry, I encountered an error: {e}"


async def save_to_memory(user_text: str, ai_response: str):
    """Save dialogue to both DuckDB (history) and Qdrant (RAG)."""
    global session_context
    
    # 1. Save to DuckDB
    if memory_manager:
        memory_manager.save_message("user", user_text)
        memory_manager.save_message("assistant", ai_response)
        add_ai_thought(f"[DB] Saved dialogue (user+assistant) to chat_history", (150,200,150))
    
    # 2. Add to in-memory session context
    session_context.append({"role": "user", "content": user_text})
    session_context.append({"role": "assistant", "content": ai_response})
    add_ai_thought(f"[CTX] Context size: {len(session_context)} messages", (150,180,200))
    
    # 3. Save to Qdrant (raw dialogue, minimal metadata)
    if vector_store:
        content = f"User: {user_text}\nAI: {ai_response}"
        max_chars = 2000
        
        if len(content) > max_chars:
            truncated = content[:max_chars].rsplit('.', 1)[0] + '.'
            logger.debug(f"Truncated dialogue from {len(content)} to {len(truncated)} chars")
            add_ai_thought(f"[RAG] Dialogue truncated to {len(truncated)} chars", (200,150,150))
            content = truncated
        
        try:
            vector_store.add_texts(
                texts=[content],
                metadatas=[{
                    "type": "dialogue",
                    "timestamp": datetime.now().isoformat()
                }]
            )
            add_ai_thought(f"[RAG] Saved raw dialogue to Qdrant", (150,255,150))
        except Exception as e:
            logger.warning(f"Failed to save to Qdrant: {e}")
            add_ai_thought(f"[RAG] ERROR saving dialogue: {e}", (255,100,100))


# ============================================================================
# Qdrant Status Reporting
# ============================================================================

async def report_qdrant_status():
    """Report number of points and sample entries from Qdrant."""
    if not qdrant_client or not settings:
        add_ai_thought("[RAG] Status: Qdrant client not available", (200,150,150))
        return
    try:
        collection_info = qdrant_client.get_collection(settings.vector_db.collection)
        points_count = collection_info.points_count
        add_ai_thought(f"[RAG] Qdrant collection '{settings.vector_db.collection}': {points_count} vectors", (150,255,150))
        
        # Get 3 most recent points (by timestamp if available, else just scroll)
        scroll_result = qdrant_client.scroll(
            collection_name=settings.vector_db.collection,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        points = scroll_result[0]
        if points:
            add_ai_thought(f"[RAG] Latest {len(points)} entries:", (150,200,200))
            for p in points:
                payload = p.payload
                # Try to get timestamp from metadata
                ts = payload.get("metadata", {}).get("timestamp", "no timestamp")
                text_preview = str(payload.get("page_content", ""))[:60]
                add_ai_thought(f"  - {ts}: {text_preview}...", (180,180,180))
        else:
            add_ai_thought("[RAG] No points found in collection", (200,200,150))
    except Exception as e:
        logger.error(f"Failed to get Qdrant status: {e}")
        add_ai_thought(f"[RAG] Status error: {e}", (255,100,100))


# ============================================================================
# GUI Setup (DearPyGui)
# ============================================================================

def find_available_font() -> Optional[str]:
    """Find first available font from preference list."""
    local_font_dir = Path("data/fonts")
    local_fonts_to_try = [
        "JetBrainsMonoNerdFont-Regular.ttf",
        "JetBrainsMonoNerdFont-Medium.ttf",
        "JetBrainsMonoNerdFont-Bold.ttf",
    ]

    for font_file in local_fonts_to_try:
        font_path = local_font_dir / font_file
        if font_path.exists():
            logger.info(f"[SYS] Using local font: {font_path}")
            return str(font_path)

    system_fonts_to_try = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]

    for font_path in system_fonts_to_try:
        if Path(font_path).exists():
            logger.info(f"[SYS] Using system fallback font: {font_path}")
            return font_path

    logger.warning("[SYS] No preferred fonts found, using DearPyGui default")
    return None


def on_memory_report():
    """Callback for Memory Report button."""
    add_ai_thought("[SYS] Generating memory report...", (200,200,100))
    submit_to_async(report_qdrant_status())
    if memory_manager:
        stats = memory_manager.get_stats()
        add_ai_thought(f"[DB] DuckDB stats: {stats.get('total_messages',0)} messages, {stats.get('total_summaries',0)} summaries", (150,255,150))
        # Show last 3 diary summaries
        summaries = memory_manager.get_diary_summaries(limit=3)
        if summaries:
            add_ai_thought(f"[DB] Last {len(summaries)} diary summaries:", (150,200,200))
            for s in summaries:
                add_ai_thought(f"  - {s['timestamp']}: {s['text'][:80]}...", (180,180,180))


def setup_gui():
    """Initialize DearPyGui context, viewport, and widgets."""
    dpg.create_context()
    
    # --- Font Loading ---
    font_path = find_available_font()
    if font_path:
        with dpg.font_registry():
            with dpg.font(font_path, 16) as main_font:
                pass
        dpg.bind_font(main_font)
        logger.info("[SYS] Font loaded successfully")

    # Viewport setup
    dpg.create_viewport(
        title=settings.gui.title,
        width=settings.gui.width,
        height=settings.gui.height,
        resizable=True
    )
    
    # Dark theme
    if settings.gui.theme == "dark":
        with dpg.theme() as dark_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 60))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 50, 80))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220))
        dpg.bind_theme(dark_theme)
    
    # Main window
    with dpg.window(
        label="Chat", 
        tag="main_window", 
        no_title_bar=True, 
        no_move=True, 
        no_resize=False
    ):
        # Scrollable chat area
        with dpg.child_window(tag="chat_area", height=-200, border=False):
            if memory_manager:
                history = memory_manager.get_recent_history(limit=50)
                for msg in history:
                    color = (100, 200, 255) if msg['role'] == 'user' else (255, 200, 100)
                    sender = "You" if msg['role'] == 'user' else "AI_EveryNyan"
                    
                    with dpg.group(parent="chat_area", horizontal=False):
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{sender}:", color=color)
                        with dpg.group(indent=20):
                            wrap_width = max(200, dpg.get_viewport_width() - 150)
                            dpg.add_text(msg['content'], wrap=wrap_width)
                        dpg.add_spacer(height=5)
            else:
                dpg.add_text(
                    "Welcome to AI_EveryNyan!", 
                    tag="welcome_text", 
                    color=(150, 150, 200)
                )
        
        # AI Thoughts Panel
        with dpg.child_window(tag="ai_thoughts_area", height=120, label="[SYSTEM] LOG", border=True):
            dpg.add_text("[SYSTEM] STATUS: Idle", tag="thoughts_placeholder", color=(100, 100, 100))
        
        # Input area with Memory Report button
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="user_input",
                width=-180,
                hint="Type your message...",
                on_enter=True,
                callback=on_send_message
            )
            dpg.add_button(label="Send", callback=on_send_message, width=80)
            dpg.add_button(label="Memory Report", callback=on_memory_report, width=100)
        
        # Status indicator
        dpg.add_text("", tag="status_text", color=(100, 100, 100))
    
    dpg.setup_dearpygui()
    dpg.show_viewport()


def add_chat_message(sender: str, text: str, color: tuple):
    """Add a message to the chat UI."""
    dpg.set_y_scroll("chat_area", 1e9)
    
    with dpg.group(parent="chat_area", horizontal=False):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{sender}:", color=color)
        with dpg.group(indent=20):
            wrap_width = max(200, dpg.get_viewport_width() - 150)
            dpg.add_text(text, wrap=wrap_width)
        dpg.add_spacer(height=5)
    
    dpg.set_y_scroll("chat_area", 1e9)


def on_send_message(sender, app_data):
    """GUI callback: handle user sending a message."""
    global _shutting_down
    if _shutting_down:
        return
        
    user_text = dpg.get_value("user_input").strip()
    if not user_text:
        return
    
    add_chat_message("You", user_text, color=(100, 200, 255))
    dpg.set_value("user_input", "")
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Thinking...")
    add_ai_thought(f"[IN] User: \"{user_text[:30]}{'...' if len(user_text)>30 else ''}\"", (150, 200, 255))
    
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
    """Async handler: process message and update UI."""
    try:
        response = await process_message(user_text)
        
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("AI_EveryNyan", response, color=(255, 200, 100))
        add_ai_thought("[SYS] Response generated.", (150, 255, 150))
        
        await save_to_memory(user_text, response)
        
    except Exception as e:
        logger.exception("Unhandled error in message handling")
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("Error", str(e), color=(255, 100, 100))
        add_ai_thought(f"[ERR] Handler Error: {e}", (255, 100, 100))


# ============================================================================
# Graceful Shutdown & Signal Handling
# ============================================================================

def initiate_graceful_shutdown():
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    
    logger.info("[SYS] Initiating graceful shutdown sequence...")
    add_ai_thought("[SYS] SHUTDOWN: Saving data...", (255, 150, 150))
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Saving memories...")
    
    if session_context:
        try:
            future = asyncio.run_coroutine_threadsafe(dump_context_to_memory(), async_loop)
            future.result(timeout=30)
            add_ai_thought("[SYS] STATUS: Save complete.", (150, 255, 150))
        except asyncio.TimeoutError:
            logger.warning("[SYS] Context dump timed out")
            add_ai_thought("[WARN] Save timed out.", (255, 200, 100))
        except Exception as e:
            logger.error(f"[SYS] Failed to dump context: {e}")
            add_ai_thought(f"[ERR] Dump failed: {e}", (255, 100, 100))
    else:
        add_ai_thought("[SYS] STATUS: Nothing to save.", (150, 150, 150))
    
    dpg.stop_dearpygui()


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    initiate_graceful_shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    global settings, async_loop, async_thread
    
    logger.info(f"Starting AI_EveryNyan v0.7.0 (debug={settings.debug})")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    init_memory_manager()
    init_components()
    init_character()
    
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(async_loop,), daemon=True)
    async_thread.start()
    
    setup_gui()
    
    logger.info("[GUI] Starting loop...")
    add_ai_thought("[SYS] STATUS: Online. Ready.", (100, 255, 100))
    
    try:
        dpg.start_dearpygui()
    finally:
        logger.info("[SYS] Initiating graceful shutdown sequence...")
        add_ai_thought("[SYS] SHUTDOWN: Saving data...", (255, 150, 150))
        dpg.configure_item("user_input", enabled=False)
        dpg.set_value("status_text", "Saving memories...")
        
        logger.info(f"DEBUG: session_context size at shutdown: {len(session_context)}")
        if session_context:
            for i, msg in enumerate(session_context[-5:], max(0, len(session_context)-5)):
                logger.debug(f"  [{i}] {msg['role']}: {msg['content'][:50]}...")
        
        if session_context:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    dump_context_to_memory(), 
                    async_loop
                )
                future.result(timeout=300)
                add_ai_thought("[SYS] STATUS: Save complete.", (150, 255, 150))
            except asyncio.TimeoutError:
                logger.warning("[SYS] Context dump timed out")
                add_ai_thought("[WARN] Save timed out.", (255, 200, 100))
            except Exception as e:
                logger.error(f"[SYS] Failed to dump context: {e}")
                add_ai_thought(f"[ERR] Dump failed: {e}", (255, 100, 100))
        else:
            logger.info("[SYS] No active context to save at shutdown")
            add_ai_thought("[SYS] STATUS: Nothing to save.", (150, 150, 150))
        
        logger.info("Shutting down background services...")
        add_ai_thought("[SYS] Stopping services...", (200, 200, 200))
        
        async_loop.call_soon_threadsafe(async_loop.stop)
        async_thread.join(timeout=2.0)
        
        if memory_manager:
            memory_manager.close()
        
        dpg.destroy_context()
        logger.info("[SYS] Shutdown complete. Goodbye!")


if __name__ == "__main__":
    config_path = Path("config/settings.yaml")
    settings = AppSettings.from_yaml(str(config_path))
    
    Path(settings.diary.storage_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("hf_cache").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    
    main()