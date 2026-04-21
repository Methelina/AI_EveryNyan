"""
Memory Manager for AI_EveryNyan.
Uses DuckDB for structured chat history and Qdrant for semantic RAG memory.
Provides Sliding Window mechanism and smart context dumping.

\src\memory_manager.py
Version:     0.4.0
Author:      Soror L.'.L.'.
Updated:     2026-04-21
Changes:
  [+] Added Pydantic model for structured diary metadata validation.
  [+] Added helper methods for metadata parsing and filtering.
  [+] Updated save_diary_summary to support universal metadata schema.
  [-] Removed emojis; strict technical log format preserved.
"""

import duckdb
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from pydantic import BaseModel, Field, validator

logger = logging.getLogger("AI_EveryNyan.MemoryManager")


# ============================================================================
# Pydantic Models for Structured Metadata
# ============================================================================

class DiaryEntryMetadata(BaseModel):
    """
    Universal metadata schema for diary entries.
    Compatible with C++ original config::DIARY_PROMPT structure.
    """
    # Core fields (always present)
    type: str = Field(default="diary_reflection", description="Entry type: diary_reflection, dialogue, thought, etc.")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    source: str = Field(default="context_dump", description="Source: context_dump, proactive, manual, etc.")
    section: Optional[str] = Field(default=None, description="Section index, e.g., '1/5'")
    
    # Semantic fields (optional but recommended)
    entities: List[str] = Field(default_factory=list, description="Canonical entity names")
    topics: List[str] = Field(default_factory=list, description="Topic tags")
    retrieval_cues: List[str] = Field(default_factory=list, description="Search-friendly phrases")
    
    # Affective fields
    emotion_label: Optional[str] = Field(default=None, description="Emotion category")
    emotion_valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0, description="Valence: -1 (negative) to +1 (positive)")
    emotion_arousal: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Arousal: 0 (calm) to 1 (excited)")
    
    # Quality/priority fields
    importance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Importance: 0 (trivial) to 1 (critical)")
    confidence: Optional[float] = Field(default=None, ge=-1.0, le=1.0, description="Confidence: -1 (false) to +1 (ground truth)")
    
    # Relationship graph
    relationships: Dict[str, str] = Field(default_factory=dict, description="Entity relationships: 'A-B': 'type'")
    
    # Content flags
    contradictions: List[str] = Field(default_factory=list, description="Uncertainties or conflicting info")
    key_facts: List[str] = Field(default_factory=list, description="Extracted atomic facts")
    
    # Type-specific extensions (sandbox for custom fields)
    type_specific: Dict[str, Any] = Field(default_factory=dict, description="Custom fields per entry type")
    
    class Config:
        extra = "allow"  # Allow additional fields for future extensibility
    
    @validator('retrieval_cues', pre=True)
    def parse_cues(cls, v):
        """Handle both list and comma-separated string formats."""
        if isinstance(v, str):
            # Parse "cue1", "cue2" or cue1, cue2 formats
            cues = re.findall(r'"([^"]+)"|\'([^\']+)\'|([\w\s\-]+)', v)
            return [c[0] or c[1] or c[2].strip() for c in cues if any(c)]
        return v
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant-compatible payload (flat structure with nested support)."""
        payload = self.dict(exclude_none=True, exclude={"type_specific"})
        # Flatten type_specific fields with prefix to avoid collisions
        for key, value in self.type_specific.items():
            payload[f"ts_{key}"] = value
        return payload
    
    @classmethod
    def from_llm_output(cls, text: str, base_meta: Optional[Dict] = None) -> "DiaryEntryMetadata":
        """
        Parse structured metadata from LLM output using regex/keyword extraction.
        Fallback: return minimal metadata if parsing fails.
        """
        meta = base_meta or {}
        
        # Extract common patterns from markdown-style output
        def extract_list(pattern: str, text: str) -> List[str]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1)
                # Handle bullet lists: - item or ["item", "item"]
                items = re.findall(r'-\s*([^\n]+)|"([^"]+)"|\'([^\']+)\'', content)
                return [i[0] or i[1] or i[2] for i in items if any(i)]
            return []
        
        def extract_float(pattern: str, text: str) -> Optional[float]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
            return None
        
        def extract_str(pattern: str, text: str) -> Optional[str]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return None
        
        # Parse fields
        parsed = {
            "entities": extract_list(r'\*\*entities:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', text),
            "topics": extract_list(r'\*\*topics/tags:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', text),
            "retrieval_cues": extract_list(r'\*\*retrieval cues:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', text),
            "emotion_label": extract_str(r'\*\*emotion/affect:\*\*\s*.*?(\w+)', text),
            "importance_score": extract_float(r'\*\*importance score:\*\*\s*([\d.]+)', text),
            "key_facts": extract_list(r'\*\*key messages:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', text),
        }
        
        # Merge with base metadata and validate
        merged = {**meta, **{k: v for k, v in parsed.items() if v}}
        return cls(**merged)


# ============================================================================
# Core Class Definition
# ============================================================================

class MemoryManager:
    """
    Unified memory manager for AI_EveryNyan.
    
    Responsibilities:
    - DuckDB: structured chat history (for Sliding Window rehydration)
    - Qdrant: semantic RAG memory with structured metadata filtering
    - Diary summaries: LLM-generated reflections with universal schema
    
    Thread-safety: DuckDB connections are not thread-safe by default.
    This class should be used from a single thread, or with external locking.
    """
    
    DB_PATH = "data/history.db"


# ============================================================================
# Initialization & Database Schema
# ============================================================================

    def __init__(self, db_path: Optional[str] = None):
        """Initialize MemoryManager."""
        self.conn = None
        self._db_path = db_path or self.DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize DuckDB connection and create tables."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.conn = duckdb.connect(self._db_path)
            logger.info(f"[DB] INIT: Connection established at {self._db_path}")
            
            # Chat history table
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
            
            # Diary summaries table with expanded metadata support
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


# ============================================================================
# Data Persistence (Write Operations)
# ============================================================================

    def save_message(self, role: str, content: str, 
                     meta: Optional[Dict] = None, 
                     session_id: str = "default") -> bool:
        """Save a chat message to DuckDB history."""
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
        """
        Save a diary summary section with structured metadata.
        
        Args:
            text: Summary content
            index: Section index (0-based)
            total: Total sections in dump
            meta: Dict or DiaryEntryMetadata instance
        """
        try:
            if not text or len(text.strip()) < 10:
                logger.warning("[DB] WARN: Skipping empty/too short diary summary")
                return False
            
            # Convert metadata to dict if Pydantic model provided
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


# ============================================================================
# Data Retrieval (Read Operations)
# ============================================================================

    def get_recent_history(self, limit: int = 20, 
                          session_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieve recent messages for Sliding Window rehydration."""
        try:
            result = self.conn.execute("""
                SELECT role, content, timestamp 
                FROM chat_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, [session_id, limit]).fetchall()
            
            history = [
                {"role": r, "content": c, "timestamp": t} 
                for r, c, t in reversed(result)
            ]
            logger.info(f"[DB] READ: Fetched {len(history)} history messages")
            return history
        except Exception as e:
            logger.error(f"[DB] ERR: History fetch failed: {e}")
            return []
    
    def get_diary_summaries(self, limit: int = 10, 
                           filter_meta: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve diary summaries with optional metadata filtering.
        
        Args:
            limit: Max entries to return
            filter_meta: Dict with field filters, e.g., {"topics": ["food"], "importance_score": {"$gte": 0.7}}
        """
        try:
            # Build WHERE clause for metadata filtering (basic implementation)
            where_clause = ""
            params = [limit]
            
            if filter_meta:
                # Simple JSON containment checks (DuckDB supports -> operator)
                conditions = []
                for key, value in filter_meta.items():
                    if isinstance(value, dict):
                        # Handle operators: $gte, $lte, $contains, etc.
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
        """Return basic statistics."""
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


# ============================================================================
# Lifecycle Management
# ============================================================================

    def close(self):
        """Close DuckDB connection gracefully."""
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