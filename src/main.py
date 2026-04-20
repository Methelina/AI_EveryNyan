#!/usr/bin/env python3
"""
AI_EveryNyan - DearPyGui Chat with LangChain + Qdrant RAG + DuckDB History
Modular Character System + Smart Context Management + Graceful Shutdown

\src\main.py
Version:     0.4.3
Author:      Soror L.'.L.'.
Updated:     2026-04-21

Patch Notes v0.4.3:
  [!] Fixed: Qdrant plagiarism check using client.query() instead of non-existent search()
  [!] Fixed: DuckDB JSON serialization using json.dumps() instead of str()
  [+] Improved: Diary section splitting with better --- normalization
  [+] Improved: Text preservation in Qdrant (preserve newlines in page_content)
  
Dependencies:
  - dearpygui>=1.11.0
  - langchain>=0.2.0, langchain-openai>=0.1.0, langchain-qdrant>=0.1.0
  - qdrant-client[http]>=1.11.0
  - pydantic>=2.8.0, pydantic-settings>=2.3.0
  - duckdb>=0.10.0
  - ollama>=0.2.0
"""

import os
import sys
import asyncio
import logging
import threading
import signal
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import dearpygui.dearpygui as dpg
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint
from openai import BadRequestError

# Local imports
from memory_manager import MemoryManager

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
It's time to open diary and share your thoughts, emotions and feelings! 
Write shortly, but avoid missing details! Avoid plagiarism and copying prior pages.
Time window: last 24–48h.

<outputFormatting>
ALWAYS divide your diary pages with small (50-300 words) self-sufficient 
semantically coherent pieces of knowledge with markdown lines `---`.

For each section include:
- timestamps, source event, outcomes
- entities with canonical names
- key messages (verbatim, with context)
- topics/tags, importance score (0–1)
- emotion/affect, relationships
- retrieval cues (3–5 short phrases for future search)
- fine-grained photo descriptions if present
</outputFormatting>

DO NOT MAKE UP FACTS! IF UNSURE, DO NOT MAKE WEAK CONCLUSIONS!
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


class AppSettings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: QdrantSettings = Field(default_factory=QdrantSettings)
    diary: DiarySettings = Field(default_factory=DiarySettings)
    gui: GUISettings = Field(default_factory=GUISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    anti_repeat: AntiRepeatSettings = Field(default_factory=AntiRepeatSettings)
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
# AI Thoughts UI System (Direct Updates - DearPyGui is thread-safe)
# ============================================================================

def add_ai_thought(text: str, color: Tuple[int, int, int] = (200, 200, 150)):
    """
    Add a thought to the AI thoughts panel.
    
    DearPyGui is thread-safe for item configuration, so this can be called
    from async background threads without additional locking.
    """
    try:
        # Remove placeholder if it exists
        if dpg.does_item_exist("thoughts_placeholder"):
            dpg.delete_item("thoughts_placeholder")
        
        # Add new thought with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        with dpg.group(parent="ai_thoughts_area", horizontal=True):
            dpg.add_text(f"[{timestamp}] ", color=(100, 100, 100))
            dpg.add_text(text, color=color)
        
        # Auto-scroll to bottom
        dpg.set_y_scroll("ai_thoughts_area", 1e9)
        
    except Exception as e:
        # UI updates may fail if DearPyGui isn't fully initialized yet
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
        temperature=0.7,
        timeout=settings.llm.timeout,
        streaming=False
    )
    
    logger.info("Components initialized")


def init_character():
    global character_base, character_appearance
    logger.info("Loading character configuration...")
    character_base = CharacterConfig.load_base()
    character_appearance = CharacterConfig.load_appearance()
    logger.info("✓ Character brain loaded correctly. All systems nominal")


def init_memory_manager():
    global memory_manager
    memory_manager = MemoryManager()
    stats = memory_manager.get_stats()
    logger.info(f"✓ MemoryManager initialized. Messages: {stats.get('total_messages', 0)}")


# ============================================================================
# RAG & Memory Management
# ============================================================================

async def query_memory(query: str, top_k: int = 3) -> str:
    global current_relevance_threshold
    
    if not vector_store:
        return ""
    
    try:
        docs = await vector_store.asimilarity_search(query, k=top_k)
        if not docs:
            return ""
        
        formatted_memories = []
        for i, doc in enumerate(docs):
            relevance = doc.metadata.get("score", "high")
            # Preserve newlines in memory content for better LLM parsing
            memory_xml = (
                f'<memory_piece id="{i}" relatedness="{relevance}">\n'
                f'{doc.page_content}\n'
                f'</memory_piece>'
            )
            formatted_memories.append(memory_xml)
        
        result = "\n\n".join(formatted_memories)
        
        if formatted_memories:
            if len(formatted_memories) < top_k:
                current_relevance_threshold = max(0.3, current_relevance_threshold * 0.95)
            elif len(formatted_memories) == top_k:
                current_relevance_threshold = min(0.9, current_relevance_threshold * 1.05)
        
        return result
        
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        return ""


async def check_plagiarism(text: str, threshold: float) -> bool:
    """
    Check if text is too similar to existing entries in Qdrant.
    
    Uses QdrantClient.query() API (v1.11+ compatible).
    Returns True if duplicate found (should skip saving).
    """
    if not vector_store or not qdrant_client:
        return False
    
    try:
        # Get embedding for the new text
        query_vector = await embeddings.aembed_query(text)
        
        # Use QdrantClient.query() for compatibility with v1.11+
        results: List[ScoredPoint] = qdrant_client.query_points(
            collection_name=settings.vector_db.collection,
            query=query_vector,
            limit=1,
            with_payload=False,
            with_vectors=False
        ).points
        
        if results:
            similarity = results[0].score  # Cosine similarity [0, 1]
            if similarity > threshold:
                logger.info(
                    f"[PLAGIARISM] Skipped duplicate. "
                    f"Similarity: {similarity:.3f} > {threshold}"
                )
                add_ai_thought(f"🚫 Skipped duplicate memory (similarity: {similarity:.2f})", (255, 150, 150))
                return True
        
        return False
        
    except Exception as e:
        logger.warning(f"Plagiarism check failed: {e}")
        # Fail-safe: if check fails, assume not duplicate to avoid data loss
        return False


async def dump_context_to_memory():
    """
    Smart context dumping (like C++ diaryDumpMessages).
    
    1. LLM writes diary entry with sections separated by ---
    2. Each section is checked for plagiarism
    3. Unique sections saved to Qdrant (RAG) and DuckDB (history)
    4. In-memory context cleared
    """
    global session_context
    
    if not session_context:
        logger.info("[SLIDING WINDOW] Context empty, nothing to dump")
        add_ai_thought("ℹ️ No active context to save.", (150, 150, 150))
        return
    
    logger.info(f"[SLIDING WINDOW] Dumping {len(session_context)} messages to memory...")
    add_ai_thought(f"📝 Summarizing {len(session_context)} messages into diary...", (200, 200, 100))
    
    try:
        # 1. Format dialogue for LLM summarization
        dialogue_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in session_context
        ])
        
        # 2. Prompt LLM to write diary entry (using config prompt)
        full_prompt = [
            ("system", settings.diary.summary_prompt),
            ("human", f"Here is the conversation to summarize:\n\n{dialogue_text}")
        ]
        
        add_ai_thought("🧠 LLM is writing diary reflections...", (150, 200, 255))
        response = await llm.ainvoke(full_prompt)
        diary_entry = response.content
        
        # 3. Normalize separators: handle various --- formats
        # Replace common variations with standard markdown line
        diary_entry = diary_entry.replace("-- -", "---")
        diary_entry = diary_entry.replace("- --", "---")
        diary_entry = diary_entry.replace("----", "---")  # 4 dashes -> 3
        diary_entry = diary_entry.replace("\n---\n", "\n---\n")  # ensure consistent newlines
        
        # Split into sections, preserving content
        sections = [s.strip() for s in diary_entry.split("---") if s.strip()]
        
        saved_count = 0
        total_sections = len(sections)
        add_ai_thought(f"📖 Split into {total_sections} sections. Checking for plagiarism...", (200, 180, 255))
        
        for idx, section in enumerate(sections):
            # Skip very short sections (noise)
            if len(section) < 20:
                continue
            
            # 4. Anti-plagiarism check
            is_duplicate = await check_plagiarism(
                section, 
                settings.diary.plagiarism_threshold
            )
            if is_duplicate:
                continue
            
            # 5. Save to Qdrant (for RAG semantic search)
            try:
                # Preserve newlines in page_content for better readability
                vector_store.add_texts(
                    texts=[section],
                    metadatas=[{
                        "type": "diary_reflection",
                        "timestamp": datetime.now().isoformat(),
                        "section": f"{idx+1}/{total_sections}",
                        "source": "context_dump"
                    }]
                )
                
                # 6. Save to DuckDB (for structured history)
                if memory_manager:
                    memory_manager.save_diary_summary(
                        text=section,
                        index=idx,
                        total=total_sections,
                        meta={"type": "reflection"}
                    )
                
                saved_count += 1
                add_ai_thought(f"✅ Saved reflection {idx+1}/{total_sections}", (150, 255, 150))
                
            except Exception as e:
                logger.error(f"Failed to save section {idx}: {e}")
                continue
        
        logger.info(
            f"[SLIDING WINDOW] Saved {saved_count}/{total_sections} "
            f"unique reflections to memory"
        )
        add_ai_thought(f"💾 Diary updated: {saved_count}/{total_sections} memories stored.", (100, 255, 100))
        
        # 7. Clear in-memory context (Sliding Window reset)
        session_context.clear()
        logger.info("[SLIDING WINDOW] In-memory context cleared")
        add_ai_thought("🧹 Working memory cleared.", (150, 150, 150))
        
    except Exception as e:
        logger.error(f"[SLIDING WINDOW] Failed to dump context: {e}")
        add_ai_thought(f"❌ Failed to save memories: {e}", (255, 100, 100))


def check_anti_repetition_semantic(new_content: str) -> bool:
    """
    Semantic anti-repetition check (like C++ REPEAT_YOURSELF_TRIGGER_*).
    
    Compares new assistant response with recent history using embeddings.
    Returns True if content is too similar (should trigger topic change).
    """
    global anti_repeat_cache
    
    if not anti_repeat_cache or not embeddings:
        return False
    
    try:
        # Get embedding for new content
        new_embedding = embeddings.embed_query(new_content)
        
        max_similarity = 0.0
        avg_similarity = 0.0
        
        for cached in anti_repeat_cache:
            cached_embedding = cached.get("embedding")
            if cached_embedding is None:
                continue
            
            # Compute cosine similarity
            similarity = sum(a * b for a, b in zip(new_embedding, cached_embedding))
            # Normalize to [0, 1] range (cosine similarity is [-1, 1])
            similarity = (similarity + 1) / 2
            
            max_similarity = max(max_similarity, similarity)
            avg_similarity += similarity
        
        if anti_repeat_cache:
            avg_similarity /= len(anti_repeat_cache)
        
        # Check against thresholds (from config)
        if max_similarity > settings.anti_repeat.trigger_max:
            logger.warning(
                f"[ANTI-REPEAT] Max similarity {max_similarity:.3f} > "
                f"{settings.anti_repeat.trigger_max}"
            )
            return True
        
        if avg_similarity > settings.anti_repeat.trigger_avg:
            logger.warning(
                f"[ANTI-REPEAT] Avg similarity {avg_similarity:.3f} > "
                f"{settings.anti_repeat.trigger_avg}"
            )
            return True
        
        # Add new embedding to cache (with LRU eviction)
        anti_repeat_cache.append({
            "content": new_content[:200],  # Store snippet for debugging
            "embedding": new_embedding,
            "timestamp": datetime.now()
        })
        
        # Evict old entries if cache too large
        if len(anti_repeat_cache) > settings.anti_repeat.max_history:
            anti_repeat_cache.pop(0)
        
        return False
        
    except Exception as e:
        logger.warning(f"Anti-repetition check failed: {e}")
        return False  # Fail-safe: don't block on error


# ============================================================================
# Prompt Building
# ============================================================================

def build_system_prompt() -> str:
    """
    Assemble final system prompt from character config + instructions.
    """
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
    """
    Main message processing loop with Sliding Window support.
    
    Flow:
    1. Check for semantic repetition
    2. Build prompt with system + context + RAG + user message
    3. Call LLM
    4. Handle context overflow with smart dumping
    """
    global session_context
    
    # 1. Anti-repetition check (semantic)
    if check_anti_repetition_semantic(user_text):
        add_ai_thought("🔄 Detected repetition. Suggesting topic change.", (255, 200, 100))
        return "I feel like we're going in circles. Let's talk about something new! 🐾"
    
    # Inner function to avoid code duplication in retry logic
    async def attempt_generation():
        # 2. RAG: retrieve relevant memories
        rag_context = await query_memory(user_text)
        
        # 3. Build system prompt
        system_prompt = build_system_prompt()
        
        # 4. Assemble messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add RAG context if found (formatted as XML)
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"<retrieved_memories>\n{rag_context}\n</retrieved_memories>"
            })
        
        # Add recent session history (Sliding Window)
        max_history = 10
        recent = session_context[-max_history:] if len(session_context) > max_history else session_context
        messages.extend(recent)
        
        # Add current user message
        messages.append({"role": "user", "content": user_text})
        
        # 5. Call LLM
        response = await llm.ainvoke(messages)
        return response.content
    
    try:
        # First attempt
        return await attempt_generation()
        
    except BadRequestError as e:
        # Handle context length overflow (Sliding Window trigger)
        if "context length" in str(e).lower() or "exceeds" in str(e).lower():
            logger.warning(
                "[SLIDING WINDOW] Context length exceeded. "
                "Initiating smart dump sequence..."
            )
            add_ai_thought("⚠️ Context overflow! Compressing memory...", (255, 150, 100))
            
            # Step 1: Summarize and save old context to memory
            await dump_context_to_memory()
            
            # Step 2: Rehydrate context from DuckDB (recent history)
            if memory_manager:
                logger.info("[SLIDING WINDOW] Rehydrating context from DuckDB...")
                add_ai_thought("📥 Rehydrating recent chat history...", (150, 200, 255))
                fresh_history = memory_manager.get_recent_history(limit=10)
                
                if fresh_history:
                    session_context.extend(fresh_history)
                    logger.info(
                        f"[SLIDING WINDOW] Rehydrated with "
                        f"{len(fresh_history)} messages"
                    )
            
            # Step 3: Retry generation with fresh context
            logger.info("[SLIDING WINDOW] Retrying generation...")
            add_ai_thought("🔄 Retrying with fresh context...", (200, 200, 100))
            try:
                return await attempt_generation()
            except Exception as retry_e:
                logger.error(f"[SLIDING WINDOW] Retry failed: {retry_e}")
                return "I tried to organize my memories, but my head is still spinning. Could we try a shorter sentence?"
        else:
            # Other BadRequest errors: re-raise
            raise
            
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        add_ai_thought(f"❌ LLM Error: {e}", (255, 100, 100))
        return f"Sorry, I encountered an error: {e}"


async def save_to_memory(user_text: str, ai_response: str):
    """
    Save dialogue to both DuckDB (history) and Qdrant (RAG).
    
    Includes length truncation for embedding safety.
    """
    global session_context
    
    # 1. Save to DuckDB (structured history)
    if memory_manager:
        memory_manager.save_message("user", user_text)
        memory_manager.save_message("assistant", ai_response)
    
    # 2. Add to in-memory session context (Sliding Window)
    session_context.append({"role": "user", "content": user_text})
    session_context.append({"role": "assistant", "content": ai_response})
    
    # 3. Save to Qdrant (semantic RAG)
    if vector_store:
        # Truncate if too long for embedding model
        content = f"User: {user_text}\nAI: {ai_response}"
        max_chars = 2000  # Safe limit for most embedding models
        
        if len(content) > max_chars:
            # Smart truncation: cut at last sentence boundary
            truncated = content[:max_chars].rsplit('.', 1)[0] + '.'
            logger.debug(
                f"Truncated dialogue from {len(content)} to {len(truncated)} chars"
            )
            content = truncated
        
        try:
            vector_store.add_texts(
                texts=[content],
                metadatas=[{
                    "type": "dialogue",
                    "timestamp": datetime.now().isoformat()
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to save to Qdrant: {e}")


# ============================================================================
# GUI Setup (DearPyGui)
# ============================================================================

def find_available_font() -> Optional[str]:
    """
    Find first available font from preference list.
    Prioritizes fonts with Cyrillic support.
    """
    fonts_to_try = [
        r"C:\Windows\Fonts\JetBrainsMono Nerd Font.ttf",
        r"C:\Windows\Fonts\JetBrainsMonoNL-Regular.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\segoeui.ttf", 
        r"C:\Windows\Fonts\arial.ttf",
    ]
    
    for font_path in fonts_to_try:
        if Path(font_path).exists():
            logger.info(f"Using font: {font_path}")
            return font_path
    
    logger.warning("No preferred fonts found, using DearPyGui default")
    return None


def setup_gui():
    """Initialize DearPyGui context, viewport, and widgets."""
    dpg.create_context()
    
    # Font setup with Cyrillic support
    font_path = find_available_font()
    if font_path:
        with dpg.font_registry():
            with dpg.font(font_path, 16) as main_font:
                # Modern DearPyGui auto-detects character ranges
                pass
        dpg.bind_font(main_font)
        logger.info("✓ Font with Cyrillic support loaded")
    
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
            # Load and display recent history from DuckDB
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
        with dpg.child_window(tag="ai_thoughts_area", height=120, label="🧠 AI Internal Thoughts", border=True):
            dpg.add_text("🧠 Панель мыслей пуста", tag="thoughts_placeholder", color=(100, 100, 100))
        
        # Input area
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="user_input",
                width=-100,
                hint="Type your message...",
                on_enter=True,
                callback=on_send_message
            )
            dpg.add_button(label="Send", callback=on_send_message, width=80)
        
        # Status indicator
        dpg.add_text("", tag="status_text", color=(100, 100, 100))
    
    dpg.setup_dearpygui()
    dpg.show_viewport()


def add_chat_message(sender: str, text: str, color: tuple):
    """
    Add a message to the chat UI (must be called from main thread).
    
    Auto-scrolls to bottom after adding.
    """
    # Scroll to bottom before adding
    dpg.set_y_scroll("chat_area", 1e9)
    
    with dpg.group(parent="chat_area", horizontal=False):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{sender}:", color=color)
        with dpg.group(indent=20):
            wrap_width = max(200, dpg.get_viewport_width() - 150)
            dpg.add_text(text, wrap=wrap_width)
        dpg.add_spacer(height=5)
    
    # Scroll to bottom after adding
    dpg.set_y_scroll("chat_area", 1e9)


def on_send_message(sender, app_data):
    """GUI callback: handle user sending a message."""
    global _shutting_down
    if _shutting_down:
        return
        
    user_text = dpg.get_value("user_input").strip()
    if not user_text:
        return
    
    # Display user message
    add_chat_message("You", user_text, color=(100, 200, 255))
    dpg.set_value("user_input", "")
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Thinking...")
    add_ai_thought(f"💬 Received: \"{user_text[:30]}{'...' if len(user_text)>30 else ''}\"", (150, 200, 255))
    
    # Submit async processing to background loop
    future = submit_to_async(handle_async_response(user_text))
    
    # Callback when async task completes
    def on_done(fut):
        try:
            fut.result()  # Propagate any exceptions
        except Exception as e:
            logger.exception(f"Task failed: {e}")
            dpg.configure_item("user_input", enabled=True)
            dpg.set_value("status_text", f"Error: {e}")
    
    future.add_done_callback(on_done)


async def handle_async_response(user_text: str):
    """
    Async handler: process message and update UI.
    
    Runs in background asyncio thread; UI updates scheduled to main thread.
    """
    try:
        # Process message through LLM + RAG pipeline
        response = await process_message(user_text)
        
        # Update UI (must be on main thread - DearPyGui is not thread-safe)
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("AI_EveryNyan", response, color=(255, 200, 100))
        add_ai_thought("✅ Response generated & displayed.", (150, 255, 150))
        
        # Save to memory (async)
        await save_to_memory(user_text, response)
        
    except Exception as e:
        logger.exception("Unhandled error in message handling")
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("Error", str(e), color=(255, 100, 100))
        add_ai_thought(f"❌ Handler Error: {e}", (255, 100, 100))


# ============================================================================
# Graceful Shutdown & Signal Handling
# ============================================================================

def initiate_graceful_shutdown():
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    
    logger.info("🛑 Initiating graceful shutdown sequence...")
    add_ai_thought("🛑 Shutting down. Saving memories to diary...", (255, 150, 150))
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Saving memories... 🐾")
    
    # Run async dump in background loop
    if session_context:
        try:
            future = asyncio.run_coroutine_threadsafe(dump_context_to_memory(), async_loop)
            future.result(timeout=30)  # Wait up to 30s
            add_ai_thought("✅ Memory dump complete.", (150, 255, 150))
        except asyncio.TimeoutError:
            logger.warning("⚠ Context dump timed out")
            add_ai_thought("⚠️ Dump timed out. Exiting anyway.", (255, 200, 100))
        except Exception as e:
            logger.error(f"⚠ Failed to dump context: {e}")
            add_ai_thought(f"❌ Dump failed: {e}", (255, 100, 100))
    else:
        add_ai_thought("ℹ️ No active context to save.", (150, 150, 150))
    
    # Stop DearPyGui
    dpg.stop_dearpygui()


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    initiate_graceful_shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    global settings, async_loop, async_thread
    
    logger.info(f"Starting AI_EveryNyan v0.4.3 (debug={settings.debug})")
    
    # Register signal handlers for Ctrl+C and termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    init_memory_manager()
    init_components()
    init_character()
    
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(async_loop,), daemon=True)
    async_thread.start()
    
    setup_gui()
    
    logger.info("✓ Entering DearPyGui main loop...")
    add_ai_thought("🚀 System online. Ready for interaction.", (100, 255, 100))
    
    try:
        dpg.start_dearpygui()
    finally:
        # === Graceful shutdown: ALWAYS dump context on exit (window close OR signal) ===
        logger.info("🛑 Initiating graceful shutdown sequence...")
        add_ai_thought("🛑 Shutting down. Saving memories to diary...", (255, 150, 150))
        dpg.configure_item("user_input", enabled=False)
        dpg.set_value("status_text", "Saving memories... 🐾")
        
        # Debug: log context size at shutdown
        logger.info(f"DEBUG: session_context size at shutdown: {len(session_context)}")
        if session_context:
            for i, msg in enumerate(session_context[-5:], max(0, len(session_context)-5)):
                logger.debug(f"  [{i}] {msg['role']}: {msg['content'][:50]}...")
        
        # Сохраняем контекст, если есть что сохранять
        if session_context:
            try:
                # Запускаем дамп в фоновом async loop
                future = asyncio.run_coroutine_threadsafe(
                    dump_context_to_memory(), 
                    async_loop
                )
                # Ждём с таймаутом, чтобы не зависнуть
                future.result(timeout=300)
                add_ai_thought("✅ Memory dump complete.", (150, 255, 150))
            except asyncio.TimeoutError:
                logger.warning("⚠ Context dump timed out")
                add_ai_thought("⚠️ Dump timed out. Exiting anyway.", (255, 200, 100))
            except Exception as e:
                logger.error(f"⚠ Failed to dump context: {e}")
                add_ai_thought(f"❌ Dump failed: {e}", (255, 100, 100))
        else:
            logger.info("ℹ️ No active context to save at shutdown")
            add_ai_thought("ℹ️ No active context to save.", (150, 150, 150))
        
        # Останавливаем фоновые сервисы
        logger.info("Shutting down background services...")
        add_ai_thought("🔌 Disconnecting services...", (200, 200, 200))
        
        async_loop.call_soon_threadsafe(async_loop.stop)
        async_thread.join(timeout=2.0)
        
        if memory_manager:
            memory_manager.close()
        
        dpg.destroy_context()
        logger.info("✓ Shutdown complete. Goodbye! 🐾")


if __name__ == "__main__":
    config_path = Path("config/settings.yaml")
    settings = AppSettings.from_yaml(str(config_path))
    
    Path(settings.diary.storage_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("hf_cache").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    
    main()