"""
Memory Manager for AI_EveryNyan.
Uses DuckDB for structured chat history and Qdrant for semantic RAG memory.
Provides Sliding Window mechanism and smart context dumping.

\src\memory_manager.py
Version:     0.7.0
Author:      Soror L.'.L.'.
Updated:     2026-04-21
Changes:
  [+] Switched to JSON-mode for metadata parsing (no regex).
  [+] Added circumplex model: affect_valence [-1..1], affect_arousal [-1..1].
  [+] Kept emotion_label as human-readable string.
  [+] Removed legacy regex parsing.
  [*] Improved from_json method with better error handling.
"""

import duckdb
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from pydantic import BaseModel, Field, validator, ValidationError

logger = logging.getLogger("AI_EveryNyan.MemoryManager")


# ============================================================================
# Pydantic Models for Structured Metadata (JSON-mode)
# ============================================================================

class DiaryEntryMetadata(BaseModel):
    """
    Universal metadata schema for diary entries.
    Now uses JSON output from LLM for reliable parsing.
    Implements circumplex model of affect (Russell, 1980):
        - valence: -1 (very negative/depressed) to +1 (very positive/ecstatic)
        - arousal: -1 (complete aversion/withdrawal) to +1 (intense approach/desire)
    """
    # Core fields (always present)
    type: str = Field(default="diary_reflection", description="Entry type")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    source: str = Field(default="context_dump", description="Source of entry")
    section: Optional[str] = Field(default=None, description="Section index, e.g., '1/5'")
    
    # Semantic fields
    entities: List[str] = Field(default_factory=list, description="Canonical entity names")
    topics: List[str] = Field(default_factory=list, description="Topic tags")
    retrieval_cues: List[str] = Field(default_factory=list, description="Search-friendly phrases")
    key_facts: List[str] = Field(default_factory=list, description="Extracted atomic facts")
    
    # Circumplex affect model (numeric + label)
    affect_valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0, 
        description="Valence: -1 (depressed/misery) to +1 (ecstasy/elation)")
    affect_arousal: Optional[float] = Field(default=None, ge=-1.0, le=1.0,
        description="Arousal: -1 (aversion/withdrawal) to +1 (intense approach/desire)")
    emotion_label: Optional[str] = Field(default=None, 
        description="Human-readable emotion, e.g., 'excited anticipation', 'mild irritation'")
    
    # Other fields
    importance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relationships: Dict[str, str] = Field(default_factory=dict)
    contradictions: List[str] = Field(default_factory=list)
    
    # For any extra fields (like source_event, outcomes) we keep them in type_specific
    type_specific: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant-compatible payload."""
        payload = self.dict(exclude_none=True, exclude={"type_specific"})
        # Flatten type_specific
        for key, value in self.type_specific.items():
            payload[f"ts_{key}"] = value
        return payload
    
    @classmethod
    def from_json(cls, json_str: str, base_meta: Optional[Dict] = None) -> "DiaryEntryMetadata":
        """
        Parse metadata from JSON string (embedded in LLM output).
        Expects a JSON object with fields like:
        {
            "entities": ["User", "Memory System"],
            "topics": ["#system_update", "#trust_building"],
            "affect_valence": 0.8,
            "affect_arousal": 0.6,
            "emotion_label": "excited anticipation",
            "importance_score": 0.9,
            ...
        }
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Using fallback minimal metadata.")
            data = {}
        
        # Merge with base_meta (timestamp, section, source)
        if base_meta:
            for k, v in base_meta.items():
                if k not in data or data[k] is None:
                    data[k] = v
        
        # Validate with Pydantic
        try:
            return cls(**data)
        except ValidationError as e:
            logger.warning(f"Validation error: {e}. Creating minimal entry.")
            # Fallback: create with only required fields
            return cls(
                timestamp=data.get("timestamp", datetime.now().isoformat()),
                source=data.get("source", "context_dump"),
                section=data.get("section")
            )


# ============================================================================
# MemoryManager (unchanged except imports)
# ============================================================================

class MemoryManager:
    DB_PATH = "data/history.db"
    
    def __init__(self, db_path: Optional[str] = None):
        self.conn = None
        self._db_path = db_path or self.DB_PATH
        self._init_db()
    
    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = duckdb.connect(self._db_path)
            logger.info(f"[DB] INIT: Connection established at {self._db_path}")
            
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
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_history(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_history(session_id)")
            
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
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_diary_timestamp ON diary_summaries(timestamp)")
            
            self.conn.commit()
            logger.info("[DB] SCHEMA: Tables checked/created successfully.")
        except Exception as e:
            logger.error(f"[DB] CRIT: Initialization failed: {e}")
            raise
    
    def save_message(self, role: str, content: str, meta: Optional[Dict] = None, session_id: str = "default") -> bool:
        try:
            unique_id = int(datetime.now().timestamp() * 1e9)
            meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
            self.conn.execute("""
                INSERT INTO chat_history (id, role, content, metadata, session_id)
                VALUES (?, ?, ?, ?, ?)
            """, [unique_id, role, content, meta_json, session_id])
            self.conn.commit()
            logger.info(f"[DB] WRITE: {role.upper()} msg stored ({len(content)} chars)")
            return True
        except Exception as e:
            logger.error(f"[DB] ERR: Message save failed: {e}")
            return False
    
    def save_diary_summary(self, text: str, index: int, total: int, 
                          meta: Optional[Union[Dict, DiaryEntryMetadata]] = None) -> bool:
        try:
            if not text or len(text.strip()) < 10:
                logger.warning("[DB] WARN: Skipping empty/too short diary summary")
                return False
            if isinstance(meta, DiaryEntryMetadata):
                meta_dict = meta.dict(exclude_none=True)
            else:
                meta_dict = meta or {}
            unique_id = int(datetime.now().timestamp() * 1e9)
            meta_json = json.dumps(meta_dict, ensure_ascii=False) if meta_dict else None
            self.conn.execute("""
                INSERT INTO diary_summaries (id, summary_text, section_index, total_sections, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [unique_id, text, index, total, meta_json])
            self.conn.commit()
            logger.info(f"[DB] SUMM: Diary entry {index+1}/{total} saved")
            return True
        except Exception as e:
            logger.error(f"[DB] ERR: Diary save failed: {e}")
            return False
    
    def get_recent_history(self, limit: int = 20, session_id: str = "default") -> List[Dict[str, Any]]:
        try:
            total_count = self.conn.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", [session_id]).fetchone()[0]
            result = self.conn.execute("""
                SELECT role, content, timestamp 
                FROM chat_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, [session_id, limit]).fetchall()
            history = [{"role": r, "content": c, "timestamp": t} for r, c, t in reversed(result)]
            logger.info(f"[DB] READ: Retrieved {len(history)} of {total_count} total messages (limit={limit})")
            return history
        except Exception as e:
            logger.error(f"[DB] ERR: History fetch failed: {e}")
            return []
    
    def get_diary_summaries(self, limit: int = 10, filter_meta: Optional[Dict] = None) -> List[Dict[str, Any]]:
        try:
            where_clause = ""
            params = [limit]
            if filter_meta:
                conditions = []
                for key, value in filter_meta.items():
                    if isinstance(value, dict):
                        if "$contains" in value:
                            conditions.append(f"metadata->>'{key}' LIKE ?")
                            params.append(f"%{value['$contains']}%")
                    else:
                        conditions.append(f"metadata->>'{key}' = ?")
                        params.append(str(value))
                if conditions:
                    where_clause = " AND " + " AND ".join(conditions)
            query = f"""
                SELECT summary_text, section_index, total_sections, metadata, timestamp
                FROM diary_summaries
                WHERE 1=1{where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            result = self.conn.execute(query, params).fetchall()
            summaries = [
                {
                    "text": text,
                    "section": f"{idx+1}/{total}",
                    "metadata": json.loads(meta) if meta else {},
                    "timestamp": ts
                }
                for text, idx, total, meta, ts in result
            ]
            logger.info(f"[DB] READ: Fetched {len(summaries)} diary summaries")
            return summaries
        except Exception as e:
            logger.error(f"[DB] ERR: Summaries fetch failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            chat_count = self.conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
            diary_count = self.conn.execute("SELECT COUNT(*) FROM diary_summaries").fetchone()[0]
            last_msg = self.conn.execute("SELECT MAX(timestamp) FROM chat_history").fetchone()[0]
            stats = {
                "total_messages": chat_count,
                "total_summaries": diary_count,
                "last_message_at": last_msg
            }
            logger.info(f"[DB] STAT: {chat_count} msgs, {diary_count} summaries")
            return stats
        except Exception as e:
            logger.error(f"[DB] ERR: Stats fetch failed: {e}")
            return {}
    
    def close(self):
        if self.conn:
            try:
                self.conn.commit()
                self.conn.close()
                logger.info("[DB] INFO: Connection closed successfully.")
            except Exception as e:
                logger.error(f"[DB] ERR: Close connection failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False