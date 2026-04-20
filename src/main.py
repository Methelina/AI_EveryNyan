#!/usr/bin/env python3
"""
AI_EveryNyan - DearPyGui Chat with LangChain + Qdrant RAG + DuckDB History + Modular Character System
"""
import os
import sys
import asyncio
import logging
import threading
import yaml
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import dearpygui.dearpygui as dpg
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from openai import BadRequestError

# Локальный импорт менеджера памяти
from memory_manager import MemoryManager

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger("AI_EveryNyan")


# === Конфигурация приложения (settings.yaml) ===

class LLMSettings(BaseModel):
    backend: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    timeout: int = 60
    token_dump_threshold: int = 4000  # Порог сброса контекста в память


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "everynyan_diary"
    embedding_dim: int = 768


class DiarySettings(BaseModel):
    storage_dir: str = "data/diary"
    plagiarism_threshold: float = 0.85


class GUISettings(BaseModel):
    title: str = "AI_EveryNyan"
    width: int = 900
    height: int = 700
    theme: str = "dark"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "logs/app.log"


class AppSettings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: QdrantSettings = Field(default_factory=QdrantSettings)
    diary: DiarySettings = Field(default_factory=DiarySettings)
    gui: GUISettings = Field(default_factory=GUISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
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


# === Конфигурация персонажа ===

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
            raise FileNotFoundError(f"Character config file not found: {filepath}")
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


# === Глобальные объекты ===
settings: Optional[AppSettings] = None
character_base: Optional[CharacterBaseConfig] = None
character_appearance: Optional[CharacterAppearanceConfig] = None

qdrant_client: Optional[QdrantClient] = None
vector_store: Optional[QdrantVectorStore] = None
llm: Optional[ChatOpenAI] = None
embeddings: Optional[OpenAIEmbeddings] = None
memory_manager: Optional[MemoryManager] = None

# In-memory контекст сессии (список сообщений)
session_context: List[Dict[str, str]] = []

# Async loop
async_loop: Optional[asyncio.AbstractEventLoop] = None
async_thread: Optional[threading.Thread] = None


def run_async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def submit_to_async(coro) -> asyncio.Future:
    if async_loop is None or not async_loop.is_running():
        logger.warning("Async loop not ready, running synchronously")
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, async_loop)


# === Инициализация компонентов ===

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
        model_kwargs=(
            {"dimensions": settings.vector_db.embedding_dim} 
            if "text-embedding-3" in settings.llm.embedding_model 
            else {}
        )
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
    logger.info("Character brain loaded correctly. All systems nominal")


def init_memory_manager():
    global memory_manager
    memory_manager = MemoryManager()
    stats = memory_manager.get_stats()
    logger.info(f"MemoryManager initialized. Total messages: {stats['total_messages']}")


# === RAG и Управление Контекстом ===

async def query_memory(query: str, top_k: int = 3) -> str:
    """Семантический поиск в Qdrant с форматированием в XML"""
    if not vector_store:
        return ""
    
    try:
        docs = await vector_store.asimilarity_search(query, k=top_k)
        if not docs:
            return ""
        
        formatted_memories = []
        for i, doc in enumerate(docs):
            # Форматируем как XML для лучшего понимания LLM
            memory_xml = (
                f'<memory_piece id="{i}" relatedness="high">\n'
                f'{doc.page_content}\n'
                f'</memory_piece>'
            )
            formatted_memories.append(memory_xml)
            
        return "\n\n".join(formatted_memories)
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        return ""


async def dump_context_to_memory():
    """Сжимает текущий контекст сессии и сохраняет в Qdrant/DuckDB как одну запись"""
    global session_context
    
    if not session_context:
        return
        
    logger.info("Context overflow detected. Dumping to memory...")
    
    try:
        # Формируем текст для суммаризации
        context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session_context])
        
        # Простая суммаризация через LLM (можно улучшить)
        summary_prompt = f"Summarize the following conversation into key facts and emotional states in 2-3 sentences:\n\n{context_text}"
        summary_response = await llm.ainvoke([("human", summary_prompt)])
        summary = summary_response.content
        
        # Сохраняем резюме в Qdrant (RAG)
        if vector_store:
            vector_store.add_texts(
                texts=[f"[Session Summary] {summary}"],
                metadatas=[{"type": "summary", "timestamp": datetime.now().isoformat()}]
            )
        
        # Сохраняем полный лог в DuckDB (если нужно, но обычно там уже есть отдельные сообщения)
        # Здесь мы просто очищаем in-memory контекст, так как отдельные сообщения уже сохранены в handle_async_response
        
        session_context.clear()
        logger.info("Context dumped and cleared.")
        
    except Exception as e:
        logger.error(f"Failed to dump context: {e}")


def check_anti_repetition(new_message: str, history: List[Dict], threshold: float = 0.9) -> bool:
    """Простая проверка на точное совпадение последних ответов"""
    if not history:
        return False
    last_assistant_msg = next((msg['content'] for msg in reversed(history) if msg['role'] == 'assistant'), None)
    if last_assistant_msg and new_message.strip() == last_assistant_msg.strip():
        return True
    return False


# === Обработка сообщений ===

def build_system_prompt() -> str:
    if not character_base or not character_appearance:
        return "You are a helpful assistant."
    
    return f"""{character_base.prompt}

<visual_reference>
{character_appearance.freeform}
</visual_reference>

<instructions>
- Use retrieved memories if they are relevant.
- Be concise, friendly, and stay in character.
- Do not invent facts.
</instructions>"""


async def process_message(user_text: str) -> str:
    """Основной цикл обработки"""
    global session_context
    
    # 1. Проверка на повтор
    if check_anti_repetition(user_text, session_context):
        return "I feel like we're going in circles. Let's talk about something else! 🐾"

    # 2. Поиск в памяти (RAG)
    rag_context = await query_memory(user_text)
    
    # 3. Формирование промпта
    system_prompt = build_system_prompt()
    
    # Собираем сообщения для LLM: System + Session Context + RAG + Current User Message
    messages = [{"role": "system", "content": system_prompt}]
    
    # Добавляем RAG контекст, если есть
    if rag_context:
        messages.append({
            "role": "system", 
            "content": f"<retrieved_memories>\n{rag_context}\n</retrieved_memories>"
        })
        
    # Добавляем историю сессии (последние N сообщений)
    # Ограничиваем, чтобы не превысить контекстное окно слишком быстро
    max_history_len = 10
    recent_history = session_context[-max_history_len:] if len(session_context) > max_history_len else session_context
    messages.extend(recent_history)
    
    # Добавляем текущее сообщение пользователя
    messages.append({"role": "user", "content": user_text})
    
    # 4. Запрос к LLM
    try:
        response = await llm.ainvoke(messages)
        ai_response = response.content
        
        # Проверяем использование токенов (если доступно)
        # В LangChain/OpenAI это зависит от провайдера. Для Ollama может потребоваться парсинг заголовков или отдельный запрос.
        # Пока просто сохраняем.
        
        return ai_response
        
    except BadRequestError as e:
        if "context length" in str(e).lower():
            logger.warning("Context length exceeded. Triggering dump.")
            await dump_context_to_memory()
            return "My memory was getting full, so I organized my thoughts. Please ask again!"
        raise
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return f"Sorry, I encountered an error: {e}"


async def save_to_memory(user_text: str, ai_response: str):
    """Сохранение в DuckDB и Qdrant"""
    global session_context
    
    # Сохраняем в DuckDB (история)
    if memory_manager:
        memory_manager.save_message("user", user_text)
        memory_manager.save_message("assistant", ai_response)
        
    # Добавляем в in-memory контекст
    session_context.append({"role": "user", "content": user_text})
    session_context.append({"role": "assistant", "content": ai_response})
    
    # Сохраняем в Qdrant (для RAG)
    # Обрезаем если слишком длинно для эмбеддинга
    content = f"User: {user_text}\nAI: {ai_response}"
    max_chars = 2000
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
        
    if vector_store:
        try:
            vector_store.add_texts(
                texts=[content],
                metadatas=[{"type": "dialogue", "timestamp": datetime.now().isoformat()}]
            )
        except Exception as e:
            logger.warning(f"Failed to save to Qdrant: {e}")


# === GUI Логика ===

def find_available_font():
    fonts_to_try = [
        r"C:\Windows\Fonts\JetBrainsMono Nerd Font.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for font_path in fonts_to_try:
        if Path(font_path).exists():
            return font_path
    return None

def setup_gui():
    dpg.create_context()
    
    # Шрифт
    font_path = find_available_font()
    if font_path:
        with dpg.font_registry():
            with dpg.font(font_path, 16) as main_font:
                pass # Автодиапазоны в новых версиях DPG
        dpg.bind_font(main_font)
        logger.info(f"Font loaded: {font_path}")
    
    dpg.create_viewport(
        title=settings.gui.title,
        width=settings.gui.width,
        height=settings.gui.height,
        resizable=True
    )
    
    if settings.gui.theme == "dark":
        with dpg.theme() as dark_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 60))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 50, 80))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220))
        dpg.bind_theme(dark_theme)
    
    with dpg.window(label="Chat", tag="main_window", no_title_bar=True, no_move=True, no_resize=False):
        with dpg.child_window(tag="chat_area", height=-50, border=False):
            # Загрузка истории из DuckDB
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
                dpg.add_text("Welcome to AI_EveryNyan!", tag="welcome_text", color=(150, 150, 200))
        
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="user_input",
                width=-100,
                hint="Type your message...",
                on_enter=True,
                callback=on_send_message
            )
            dpg.add_button(label="Send", callback=on_send_message, width=80)
        
        dpg.add_text("", tag="status_text", color=(100, 100, 100))
    
    dpg.setup_dearpygui()
    dpg.show_viewport()


def add_chat_message(sender: str, text: str, color: tuple):
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
    user_text = dpg.get_value("user_input").strip()
    if not user_text:
        return
    
    add_chat_message("You", user_text, color=(100, 200, 255))
    dpg.set_value("user_input", "")
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Thinking...")
    
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
        add_chat_message("AI_EveryNyan", response, color=(255, 200, 100))
        
        await save_to_memory(user_text, response)
        
    except Exception as e:
        logger.exception("Unhandled error")
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("Error", str(e), color=(255, 100, 100))


# === Точка входа ===

def main():
    global settings, async_loop, async_thread
    
    logger.info(f"Starting AI_EveryNyan (debug={settings.debug})")
    
    # Инициализация
    init_memory_manager()
    init_components()
    init_character()
    
    # Async loop
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(async_loop,), daemon=True)
    async_thread.start()
    
    # GUI
    setup_gui()
    
    logger.info("Entering DearPyGui main loop...")
    dpg.start_dearpygui()
    
    # Shutdown
    async_loop.call_soon_threadsafe(async_loop.stop)
    async_thread.join(timeout=2.0)
    if memory_manager:
        memory_manager.close()
    dpg.destroy_context()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    config_path = Path("config/settings.yaml")
    settings = AppSettings.from_yaml(str(config_path))
    
    Path(settings.diary.storage_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("hf_cache").mkdir(parents=True, exist_ok=True)
    
    main()