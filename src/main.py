#!/usr/bin/env python3
"""
AI_EveryNyan - Minimal DearPyGui Chat with LangChain + Qdrant RAG
"""
import os
import sys
import asyncio
import logging
import threading
import yaml
from pathlib import Path
from typing import Optional, List, Callable, Any

import dearpygui.dearpygui as dpg
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

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


# === Конфигурация через Pydantic + YAML ===

class LLMSettings(BaseModel):
    backend: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    timeout: int = 60


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "everynyan_diary"
    embedding_dim: int = 768


class DiarySettings(BaseModel):
    storage_dir: str = "data/diary"
    token_dump_threshold: int = 8000
    plagiarism_threshold: float = 0.85


class GUISettings(BaseModel):
    title: str = "AI_EveryNyan"
    width: int = 900
    height: int = 700
    theme: str = "dark"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "logs/app.log"


class Settings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: QdrantSettings = Field(default_factory=QdrantSettings)
    diary: DiarySettings = Field(default_factory=DiarySettings)
    gui: GUISettings = Field(default_factory=GUISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    debug: bool = False
    
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """Загрузка настроек из YAML-файла"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load settings from {path}: {e}. Using defaults.")
            return cls()


# === Глобальные объекты ===
settings: Optional[Settings] = None
qdrant_client: Optional[QdrantClient] = None
vector_store: Optional[QdrantVectorStore] = None
llm: Optional[ChatOpenAI] = None
embeddings: Optional[OpenAIEmbeddings] = None

# === Async loop в отдельном потоке ===
async_loop: Optional[asyncio.AbstractEventLoop] = None
async_thread: Optional[threading.Thread] = None


def run_async_loop(loop: asyncio.AbstractEventLoop):
    """Запускает event loop в фоновом потоке"""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def submit_to_async(coro) -> asyncio.Future:
    """Отправляет корутину в фоновый asyncio-поток"""
    if async_loop is None or not async_loop.is_running():
        # Fallback: выполнить синхронно (может блокировать GUI)
        logger.warning("Async loop not ready, running synchronously")
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, async_loop)


# === Инициализация компонентов ===

def init_components():
    global qdrant_client, vector_store, llm, embeddings
    
    logger.info("Initializing components...")
    
    # Qdrant client
    qdrant_client = QdrantClient(url=settings.vector_db.url)
    
    # Создаём коллекцию если нет
    if not qdrant_client.collection_exists(settings.vector_db.collection):
        qdrant_client.create_collection(
            collection_name=settings.vector_db.collection,
            vectors_config=models.VectorParams(
                size=settings.vector_db.embedding_dim,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Created collection: {settings.vector_db.collection}")
    
    # Embeddings (OpenAI-compatible API)
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
    
    # Vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.vector_db.collection,
        embedding=embeddings
    )
    
    # LLM (OpenAI-compatible API)
    llm = ChatOpenAI(
        model=settings.llm.chat_model,
        openai_api_key="ollama",
        openai_api_base=settings.llm.base_url,
        temperature=0.7,
        timeout=settings.llm.timeout,
        streaming=False  # отключаем streaming для простоты
    )
    
    logger.info("Components initialized")


# === RAG: поиск в памяти ===

async def query_memory(query: str, top_k: int = 3) -> str:
    """Семантический поиск в Qdrant + форматирование результата"""
    if not vector_store:
        return ""
    
    try:
        docs = await vector_store.asimilarity_search(query, k=top_k)
        if not docs:
            return "No relevant memories found."
        
        result = "\n\n".join([
            f"[{i+1}] {doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        return f"<retrieved_memories>\n{result}\n</retrieved_memories>"
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        return f"Error retrieving memories: {e}"


# === Обработка сообщения пользователя ===

async def process_message(user_text: str) -> str:
    """Основной цикл: запрос к памяти → промпт → LLM → ответ"""
    
    # 1. Поиск релевантных записей
    memory_context = await query_memory(user_text)
    
    # 2. Формирование промпта
    system_prompt = """You are AI_EveryNyan, a helpful chat assistant.
Use retrieved memories if relevant. Be concise and friendly."""
    
    user_prompt = f"{memory_context}\n\nUser: {user_text}\nAI:"
    
    # 3. Запрос к LLM
    try:
        response = await llm.ainvoke([
            ("system", system_prompt),
            ("human", user_prompt)
        ])
        return response.content
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return f"Sorry, I encountered an error: {e}"


# === Сохранение диалога в память ===

async def save_to_memory(user_text: str, ai_response: str):
    """Сохраняет диалог в Qdrant для будущего RAG"""
    if not vector_store or not embeddings:
        return
    
    try:
        content = f"User: {user_text}\nAI: {ai_response}"
        vector_store.add_texts(
            texts=[content],
            metadatas=[{"type": "dialogue", "timestamp": str(Path.cwd())}]
        )
        logger.info("Saved dialogue to memory")
    except Exception as e:
        logger.warning(f"Failed to save to memory: {e}")


# === DearPyGui: GUI логика ===

def setup_gui():
    dpg.create_context()
    dpg.create_viewport(
        title=settings.gui.title,
        width=settings.gui.width,
        height=settings.gui.height,
        resizable=True
    )
    
    # Тема
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
            dpg.add_text("Welcome to AI_EveryNyan! Type a message below.", tag="welcome_text", color=(150, 150, 200))
        
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
    """Добавление сообщения в интерфейс чата"""
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
    """Обработчик отправки сообщения"""
    user_text = dpg.get_value("user_input").strip()
    if not user_text:
        return
    
    add_chat_message("You", user_text, color=(100, 200, 255))
    dpg.set_value("user_input", "")
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Thinking...")
    
    # Отправляем задачу в фоновый asyncio-поток
    future = submit_to_async(handle_async_response(user_text))
    
    # Опционально: обработать результат через callback (не блокируя GUI)
    def on_done(fut):
        try:
            result = fut.result()
            # Результат уже обработан внутри handle_async_response
        except Exception as e:
            logger.exception(f"Task failed: {e}")
            dpg.configure_item("user_input", enabled=True)
            dpg.set_value("status_text", f"Error: {e}")
    
    future.add_done_callback(on_done)


async def handle_async_response(user_text: str):
    """Асинхронный обработчик ответа (выполняется в фоне)"""
    try:
        response = await process_message(user_text)
        # Добавление в GUI должно быть из главного потока!
        # DearPyGui не потокобезопасен, поэтому используем dpg.configure_item через очередь
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("AI_EveryNyan", response, color=(255, 200, 100))
        
        # Сохраняем диалог в память
        await save_to_memory(user_text, response)
        
    except Exception as e:
        logger.exception("Unhandled error in message handling")
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("Error", str(e), color=(255, 100, 100))


# === Точка входа ===

def main():
    global settings, async_loop, async_thread
    
    logger.info(f"Starting AI_EveryNyan (debug={settings.debug})")
    
    # Запускаем asyncio в отдельном потоке
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(async_loop,), daemon=True)
    async_thread.start()
    
    # Инициализация компонентов (выполняется в главном потоке)
    init_components()
    
    # Настройка GUI
    setup_gui()
    
    # Запуск event loop DearPyGui (блокирует главный поток)
    logger.info("Entering DearPyGui main loop...")
    dpg.start_dearpygui()
    
    # Остановка (никогда не достигнется при обычном закрытии окна)
    async_loop.call_soon_threadsafe(async_loop.stop)
    async_thread.join(timeout=2.0)
    dpg.destroy_context()
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    # Загружаем настройки из YAML
    config_path = Path("config/settings.yaml")
    settings = Settings.from_yaml(str(config_path))
    
    # Создаём необходимые директории
    Path(settings.diary.storage_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("hf_cache").mkdir(parents=True, exist_ok=True)
    
    # Запускаем приложение
    main()