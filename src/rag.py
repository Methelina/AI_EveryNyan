"""
RAG queries, anti-repeat checks, context dumping, and memory persistence for AI_EveryNyan.
Semantic search via Qdrant, keyword fallback via DuckDB, dialogue metadata extraction.

/src/rag.py
Version:     0.17.6
Author:      Soror L.'.L.'.
Updated:     2026-05-01

Patch Notes v0.17.6 (by pytraveler):
  [+] Extracted from main.py: query_memory(), keyword_search_in_history(), check_plagiarism().
  [+] dump_context_to_memory(), check_anti_repetition_semantic(), save_to_memory().
  [+] _extract_dialogue_metadata() for Qdrant metadata enrichment.
  [*] No functional changes from original main.py code.
"""

import re
import json
from logger import logger
from typing import Optional, Dict, Any, List

from datetime import datetime
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

import runtime
from memory_manager import DiaryEntryMetadata




def _add_ai_thought(text: str, color=(200, 200, 150)):
    from gui import add_ai_thought
    add_ai_thought(text, color)


async def query_memory(query: str, top_k: Optional[int] = None,
                       filter_meta: Optional[Dict] = None) -> str:
    if top_k is None:
        top_k = runtime.settings.rag.top_k
    if not runtime.vector_store:
        _add_ai_thought("[RAG] SKIP: Vector store not initialized", (200, 150, 150))
        return ""

    _add_ai_thought(f"[RAG] QUERY: \"{query[:60]}{'...' if len(query)>60 else ''}\" (k={top_k})", (180,220,255))
    if filter_meta:
        _add_ai_thought(f"[RAG] FILTER: {filter_meta}", (180,180,200))

    try:
        qdrant_filter = None
        if filter_meta and runtime.settings.rag.enable_metadata_filtering:
            must_conditions = []
            for key, value in filter_meta.items():
                if isinstance(value, list):
                    must_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchAny(any=value)))
                else:
                    must_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        docs_with_scores = await runtime.vector_store.asimilarity_search_with_score(query, k=top_k, filter=qdrant_filter)
        if not docs_with_scores:
            _add_ai_thought(
                f"[RAG] RESULT: No documents found (k={top_k})", (200, 150, 150)
            )
            return ""

        if runtime.settings.rag.similarity_threshold > 0:
            docs_with_scores = [(doc, score) for doc, score in docs_with_scores
                                if score >= runtime.settings.rag.similarity_threshold]
            if not docs_with_scores:
                _add_ai_thought(
                    f"[RAG] RESULT: No documents above threshold {runtime.settings.rag.similarity_threshold}",
                    (200, 150, 150),
                )
                return ""

        docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        _add_ai_thought(f"[RAG] FOUND: {len(docs)} document(s) (requested {top_k})", (150,255,150))

        formatted_memories = []
        for i, (doc, score) in enumerate(zip(docs, scores)):
            snippet = doc.page_content[:80].replace("\n", " ")
            _add_ai_thought(f"  [{i+1}] score={score:.3f} | {snippet}...", (200,200,200))
            content = doc.page_content[:800]
            formatted_memories.append(f"--- Воспоминание {i+1} (релевантность {score:.2f}) ---\n{content}")

        return "\n\n".join(formatted_memories)
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        _add_ai_thought(f"[RAG] ERROR: {e}", (255, 100, 100))
        return ""


async def keyword_search_in_history(query: str, limit: int = 3) -> str:
    if not runtime.memory_manager:
        return ""
    keywords = [w for w in query.lower().split() if len(w) > 3]
    if not keywords:
        return ""
    try:
        conn = runtime.memory_manager.conn
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


async def check_plagiarism(text: str, threshold: float) -> bool:
    if not runtime.vector_store or not runtime.qdrant_client:
        return False
    try:
        query_vector = await runtime.embeddings.aembed_query(text)
        results = runtime.qdrant_client.query_points(
            collection_name=runtime.settings.vector_db.collection,
            query=query_vector,
            limit=1,
            with_payload=False,
            with_vectors=False,
        ).points
        if results and results[0].score > threshold:
            _add_ai_thought(f"[MEM] BLOCK: Duplicate (sim={results[0].score:.2f})", (255,150,150))
            return True
        return False
    except Exception as e:
        logger.warning(f"Plagiarism check failed: {e}")
        return False


async def _extract_dialogue_metadata(user_text: str, ai_response: str) -> Dict[str, Any]:
    if runtime.settings.chat_mode != "ollama":
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
        response = await runtime.llm.ainvoke([("system", "You are a metadata extractor. Output only JSON."), ("human", prompt)])
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
    if not runtime.session_context:
        _add_ai_thought("[SYS] DUMP: idle (no context)", (150,150,150))
        return
    msg_count = len(runtime.session_context)
    _add_ai_thought(f"[SUM] START: processing {msg_count} messages", (200,200,100))
    for i, msg in enumerate(runtime.session_context[:5]):
        _add_ai_thought(f"  [{i}] {msg['role']}: {msg['content'][:40]}...", (150,150,180))
    try:
        dialogue_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in runtime.session_context])
        full_prompt = [("system", runtime.settings.diary.summary_prompt), ("human", f"Here is the conversation to summarize:\n\n{dialogue_text}")]

        if runtime.settings.chat_mode == "ollama":
            response = await runtime.llm.ainvoke(full_prompt)
            diary_entry = response.content
        else:
            response = await runtime.llm.ainvoke(full_prompt)
            diary_entry = response.content

        diary_entry = diary_entry.replace("-- -", "---").replace("- --", "---").replace("----", "---")
        sections = [s.strip() for s in diary_entry.split("---") if s.strip()]
        total_sections = len(sections)
        saved_count = 0
        skipped_count = 0
        for idx, section in enumerate(sections):
            if len(section) < 20:
                _add_ai_thought(f"  [SKIP] Section {idx+1} too short (<20 chars)", (255,200,100))
                skipped_count += 1
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

            if await check_plagiarism(clean_section, runtime.settings.diary.plagiarism_threshold):
                _add_ai_thought(f"  [SKIP] Section {idx+1} duplicate (plagiarism threshold)", (255,150,150))
                skipped_count += 1
                continue

            if runtime.query_preprocessor:
                lemmatized = runtime.query_preprocessor.lemmatize_text(clean_section, remove_stopwords=False)
                parsed_meta.type_specific["lemmatized"] = lemmatized

            runtime.vector_store.add_texts(texts=[clean_section], metadatas=[parsed_meta.to_qdrant_payload()])
            if runtime.memory_manager:
                runtime.memory_manager.save_diary_summary(text=clean_section, index=idx, total=total_sections, meta=parsed_meta)
            saved_count += 1
            _add_ai_thought(f"  [SAVE] Section {idx+1}/{total_sections} stored (len={len(clean_section)})", (100,255,100))

        _add_ai_thought(f"[DB] FINISH: {saved_count} saved, {skipped_count} skipped (from {total_sections} sections)", (100,255,100))
        runtime.session_context.clear()
    except Exception as e:
        logger.error(f"dump_context_to_memory failed: {e}")
        _add_ai_thought(f"[ERR] Dump failed: {e}", (255,100,100))


def check_anti_repetition_semantic(new_content: str) -> bool:
    if not runtime.anti_repeat_cache or not runtime.embeddings:
        return False
    try:
        new_embedding = runtime.embeddings.embed_query(new_content)
        max_sim, avg_sim = 0.0, 0.0
        for cached in runtime.anti_repeat_cache:
            cached_emb = cached.get("embedding")
            if cached_emb is None:
                continue
            sim = (sum(a*b for a,b in zip(new_embedding, cached_emb)) + 1) / 2
            max_sim = max(max_sim, sim)
            avg_sim += sim
        if runtime.anti_repeat_cache:
            avg_sim /= len(runtime.anti_repeat_cache)
        if max_sim > runtime.settings.anti_repeat.trigger_max or avg_sim > runtime.settings.anti_repeat.trigger_avg:
            _add_ai_thought(f"[ANTIREPEAT] BLOCKED: max={max_sim:.2f} avg={avg_sim:.2f}", (255,200,100))
            return True
        runtime.anti_repeat_cache.append({"content": new_content[:200], "embedding": new_embedding, "timestamp": datetime.now()})
        if len(runtime.anti_repeat_cache) > runtime.settings.anti_repeat.max_history:
            runtime.anti_repeat_cache.pop(0)
        return False
    except Exception as e:
        logger.warning(f"Anti-repetition failed: {e}")
        return False


async def save_to_memory(user_text: str, ai_response: str):
    if not ai_response or len(ai_response.strip()) == 0:
        _add_ai_thought("[RAG] Skipped saving empty response", (255,150,150))
        return

    if runtime.memory_manager:
        runtime.memory_manager.save_message("user", user_text)
        runtime.memory_manager.save_message("assistant", ai_response)
        _add_ai_thought("[DB] Saved dialogue to chat_history", (150,200,150))

    runtime.session_context.append({"role": "user", "content": user_text})
    runtime.session_context.append({"role": "assistant", "content": ai_response})
    _add_ai_thought(f"[CTX] Context size: {len(runtime.session_context)} messages", (150,180,200))

    if runtime.vector_store:
        content = f"User: {user_text}\nAI: {ai_response}"
        if len(content) > 2000:
            content = content[:2000].rsplit('.', 1)[0] + '.'

        meta = await _extract_dialogue_metadata(user_text, ai_response)
        meta.update({"type": "dialogue", "timestamp": datetime.now().isoformat()})

        if runtime.query_preprocessor:
            lemmatized = runtime.query_preprocessor.lemmatize_text(content, remove_stopwords=False)
            meta["lemmatized"] = lemmatized
            _add_ai_thought(f"[RAG] Lemmatized copy stored (length {len(lemmatized)})", (150,200,150))

        try:
            runtime.vector_store.add_texts(texts=[content], metadatas=[meta])
            _add_ai_thought(f"[RAG] Saved dialogue with metadata", (150,255,150))
        except Exception as e:
            logger.warning(f"Failed to save to Qdrant: {e}")
