"""
Memory Manager for AI_EveryNyan.
Uses DuckDB for structured chat history and Qdrant for semantic RAG memory.
Provides Sliding Window mechanism and smart context dumping.

\src\memory_manager.py
Version:     0.3.2
Author:      Soror L.'.L.'.
Updated:     2026-04-21
Changes:
  - Fixed: JSON serialization for DuckDB metadata (json.dumps instead of str)
  - Fixed: Proper handling of None metadata values
  - Added: Input validation for diary summaries
"""

import duckdb
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger("AI_EveryNyan.MemoryManager")


class MemoryManager:
    """
    Unified memory manager for AI_EveryNyan.
    
    Responsibilities:
    - DuckDB: structured chat history (for Sliding Window rehydration)
    - Qdrant: semantic RAG memory (for knowledge retrieval)
    - Diary summaries: LLM-generated reflections stored in both
    
    Thread-safety: DuckDB connections are not thread-safe by default.
    This class should be used from a single thread, or with external locking.
    """
    
    DB_PATH = "data/history.db"
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize MemoryManager.
        
        Args:
            db_path: Optional path to DuckDB file. Defaults to DATA_PATH/history.db
        """
        self.conn = None
        self._db_path = db_path or self.DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize DuckDB connection and create tables if needed."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = duckdb.connect(self._db_path)
        
        # Chat history table (for Sliding Window)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id BIGINT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role VARCHAR NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                session_id VARCHAR DEFAULT 'default',
                metadata JSON
            )
        """)
        
        # Create indexes separately (DuckDB requirement)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_history(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_history(session_id)")
        
        # Diary summaries table (for LLM-generated reflections)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS diary_summaries (
                id BIGINT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary_text TEXT NOT NULL,
                section_index INT,
                total_sections INT,
                metadata JSON
            )
        """)
        
        # Index for diary summaries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_diary_timestamp ON diary_summaries(timestamp)")
        
        # Commit schema changes
        self.conn.commit()
        logger.info(f"MemoryManager initialized. DB: {self._db_path}")
    
    def save_message(self, role: str, content: str, 
                     meta: Optional[Dict] = None, 
                     session_id: str = "default") -> bool:
        """
        Save a chat message to DuckDB history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message text
            meta: Optional metadata dict (will be JSON-encoded)
            session_id: Session identifier for multi-session support
            
        Returns:
            True if saved successfully, False on error
        """
        try:
            unique_id = int(datetime.now().timestamp() * 1e9)
            # Proper JSON serialization for DuckDB
            meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
            self.conn.execute("""
                INSERT INTO chat_history (id, role, content, metadata, session_id)
                VALUES (?, ?, ?, ?, ?)
            """, [unique_id, role, content, meta_json, session_id])
            self.conn.commit()
            logger.debug(f"Saved {role} message ({len(content)} chars) to history")
            return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    def save_diary_summary(self, text: str, index: int, total: int, 
                          meta: Optional[Dict] = None) -> bool:
        """
        Save a diary summary section (from LLM summarization).
        
        Args:
            text: The summary text (one section, separated by ---)
            index: Section index (0-based)
            total: Total sections in this dump
            meta: Optional metadata (emotion, importance, retrieval_cues, etc.)
            
        Returns:
            True if saved successfully, False on error
        """
        try:
            # Validate input
            if not text or len(text.strip()) < 10:
                logger.warning("Skipping empty or too short diary summary")
                return False
            
            unique_id = int(datetime.now().timestamp() * 1e9)
            # Proper JSON serialization for DuckDB (double quotes, not single)
            meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
            
            self.conn.execute("""
                INSERT INTO diary_summaries (id, summary_text, section_index, total_sections, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [unique_id, text, index, total, meta_json])
            self.conn.commit()
            logger.debug(f"Saved diary summary {index+1}/{total} ({len(text)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to save diary summary: {e}")
            return False
    
    def get_recent_history(self, limit: int = 20, 
                          session_id: str = "default") -> List[Dict[str, Any]]:
        """
        Retrieve recent messages for Sliding Window rehydration.
        
        Args:
            limit: Maximum number of messages to return
            session_id: Session filter
            
        Returns:
            List of message dicts in chronological order (oldest first)
        """
        try:
            result = self.conn.execute("""
                SELECT role, content, timestamp 
                FROM chat_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, [session_id, limit]).fetchall()
            
            # Reverse to get chronological order (oldest -> newest)
            return [
                {"role": r, "content": c, "timestamp": t} 
                for r, c, t in reversed(result)
            ]
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}")
            return []
    
    def search_history(self, query: str, session_id: str = "default", 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Simple text search in chat history (SQL LIKE, case-insensitive).
        
        Note: For semantic search, use Qdrant via vector_store.
        
        Args:
            query: Search substring
            session_id: Session filter
            limit: Max results
            
        Returns:
            List of matching messages (newest first)
        """
        try:
            result = self.conn.execute("""
                SELECT role, content, timestamp 
                FROM chat_history 
                WHERE session_id = ? AND content ILIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, [session_id, f"%{query}%", limit]).fetchall()
            
            return [
                {"role": r, "content": c, "timestamp": t} 
                for r, c, t in result
            ]
        except Exception as e:
            logger.error(f"Failed to search history: {e}")
            return []
    
    def get_diary_summaries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent diary summaries (LLM-generated reflections).
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of summary dicts with metadata
        """
        try:
            result = self.conn.execute("""
                SELECT summary_text, section_index, total_sections, metadata, timestamp
                FROM diary_summaries
                ORDER BY timestamp DESC
                LIMIT ?
            """, [limit]).fetchall()
            
            return [
                {
                    "text": text,
                    "section": f"{idx+1}/{total}",
                    "metadata": json.loads(meta) if meta else {},
                    "timestamp": ts
                }
                for text, idx, total, meta, ts in result
            ]
        except Exception as e:
            logger.error(f"Failed to fetch diary summaries: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about stored data."""
        try:
            chat_count = self.conn.execute(
                "SELECT COUNT(*) FROM chat_history"
            ).fetchone()[0]
            
            diary_count = self.conn.execute(
                "SELECT COUNT(*) FROM diary_summaries"
            ).fetchone()[0]
            
            last_msg = self.conn.execute(
                "SELECT MAX(timestamp) FROM chat_history"
            ).fetchone()[0]
            
            return {
                "total_messages": chat_count,
                "total_summaries": diary_count,
                "last_message_at": last_msg
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self):
        """Close DuckDB connection gracefully."""
        if self.conn:
            try:
                self.conn.commit()
                self.conn.close()
                logger.info("MemoryManager connection closed")
            except Exception as e:
                logger.error(f"Error closing DB connection: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False