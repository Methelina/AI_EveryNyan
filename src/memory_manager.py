"""
Memory Manager for AI_EveryNyan.
Uses DuckDB to store structured chat history and metadata.
Complements Qdrant (semantic search) with precise SQL queries.
"""
import duckdb
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger("AI_EveryNyan.MemoryManager")

class MemoryManager:
    DB_PATH = "data/history.db"

    def __init__(self):
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Инициализирует соединение и создаёт таблицы, если их нет."""
        Path(self.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = duckdb.connect(self.DB_PATH)
        
        # Таблица сообщений чата
        # Убираем PRIMARY KEY и AUTOINCREMENT, так как DuckDB может ругаться на constraints в некоторых случаях.
        # Просто используем BIGINT для ID. Если нужно будет уникальное ID, можно генерировать его в Python.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id BIGINT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role VARCHAR NOT NULL, -- 'user' or 'assistant'
                content TEXT NOT NULL,
                session_id VARCHAR DEFAULT 'default', -- для разделения сессий в будущем
                metadata JSON -- для дополнительных данных (токены, инструменты и т.д.)
            )
        """)
        
        # Индекс для быстрого поиска по времени
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_history(timestamp)")
        
        logger.info(f"MemoryManager initialized. DB: {self.DB_PATH}")

    def save_message(self, role: str, content: str, meta: Optional[Dict] = None, session_id: str = "default"):
        """Сохраняет сообщение в историю."""
        try:
            # Генерируем простой ID на основе текущего времени в наносекундах, чтобы он был уникальным
            import time
            unique_id = int(time.time_ns())
            
            self.conn.execute("""
                INSERT INTO chat_history (id, role, content, metadata, session_id)
                VALUES (?, ?, ?, ?, ?)
            """, [unique_id, role, content, str(meta) if meta else None, session_id])
            self.conn.commit()
            logger.debug(f"Saved {role} message to history.")
        except Exception as e:
            logger.error(f"Failed to save message to history: {e}")

    def get_recent_history(self, limit: int = 20, session_id: str = "default") -> List[Dict[str, Any]]:
        """Получает последние N сообщений из истории."""
        try:
            result = self.conn.execute("""
                SELECT role, content, timestamp 
                FROM chat_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, [session_id, limit]).fetchall()
            
            # Возвращаем в прямом порядке (старые -> новые)
            return [{"role": r, "content": c, "timestamp": t} for r, c, t in reversed(result)]
        except Exception as e:
            logger.error(f"Failed to fetch recent history: {e}")
            return []

    def search_exact_match(self, query: str, session_id: str = "default") -> List[Dict[str, Any]]:
        """Поиск точного совпадения или подстроки в истории (SQL LIKE)."""
        try:
            # Используем ILIKE для регистронезависимого поиска
            result = self.conn.execute("""
                SELECT role, content, timestamp 
                FROM chat_history 
                WHERE session_id = ? AND content ILIKE ?
                ORDER BY timestamp DESC
            """, [session_id, f"%{query}%"]).fetchall()
            
            return [{"role": r, "content": c, "timestamp": t} for r, c, t in result]
        except Exception as e:
            logger.error(f"Failed to search history: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по базе."""
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
            last_msg = self.conn.execute("SELECT MAX(timestamp) FROM chat_history").fetchone()[0]
            return {"total_messages": count, "last_message_at": last_msg}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def close(self):
        """Закрывает соединение с БД."""
        if self.conn:
            self.conn.close()
            logger.info("MemoryManager connection closed.")