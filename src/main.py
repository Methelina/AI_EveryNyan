#!/usr/bin/env python3
"""
AI_EveryNyan - DearPyGui Chat with LangChain + Qdrant RAG + DuckDB History
Modular Character System + Smart Context Management + Structured Diary Metadata

src/main.py
Version:     0.17.6 (Modular refactoring)
Author:      Soror L.'.L.'.
Updated:     2026-05-01

Patch Notes v0.17.6 (by pytraveler):
  [REFACTOR] Split monolithic main.py (2081 lines) into 6 focused modules:
    - config.py: Pydantic settings models, CharacterConfig, AppearanceProjection
    - llm_adapter.py: LlamaChatModel (LangChain adapter)
    - runtime.py: Shared global state, component init, dynamic reconfiguration
    - character.py: Character loading, projections, system prompt building
    - rag.py: RAG queries, anti-repeat, context dumping, memory persistence
    - gui.py: DearPyGui setup, callbacks, AI thoughts display
  [*] main.py reduced to ~180 lines: process_message orchestration + entry point.
  [*] No functional changes. All behavior preserved.
"""

import sys
import asyncio
import logging
import signal
import threading
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from openai import BadRequestError, APITimeoutError

import logging_exceptions
import runtime
from config import AppSettings
from runtime import (
    submit_to_async,
    init_components,
    init_memory_manager,
    init_query_preprocessor,
    init_mcp_agent,
    reinit_llm,
    reinit_embeddings,
    run_async_loop,
)
from character import init_character
from rag import (
    check_anti_repetition_semantic,
    dump_context_to_memory,
    query_memory,
    keyword_search_in_history,
    save_to_memory,
)
from gui import (
    add_ai_thought,
    update_ai_message_streaming,
    finalize_ai_message_streaming,
    setup_gui,
    refresh_models_list,
    save_window_geometry,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8", mode="a"),
    ],
)
from logger import logger


# ============================================================================
# Message Processing (UNIFIED LOGIC — central orchestrator)
# ============================================================================

async def process_message(user_text: str) -> str:
    if check_anti_repetition_semantic(user_text):
        return "I feel like we're going in circles. Let's talk about something new!"

    if not runtime.session_context and runtime.memory_manager:
        add_ai_thought("[CTX] Context empty, loading recent history from DuckDB", (200,150,100))
        fresh = runtime.memory_manager.get_recent_history(limit=runtime.settings.context.max_history_messages)
        runtime.session_context.extend(fresh)
        add_ai_thought(f"[CTX] Loaded {len(fresh)} messages", (150,255,150))

    async def attempt_generation():
        from character import build_system_prompt

        add_ai_thought(f"[CTX] Current context size: {len(runtime.session_context)} messages", (150,180,200))
        if len(runtime.session_context) > runtime.settings.context.warn_if_context_exceeds:
            add_ai_thought(f"[CTX] WARN: Large context ({len(runtime.session_context)}), may need dump soon", (255,200,100))

        rag_context = await query_memory(user_text)
        if not rag_context and runtime.memory_manager:
            add_ai_thought("[RAG] No vectors, trying DuckDB keyword search", (200,150,100))
            keyword_results = await keyword_search_in_history(user_text, limit=3)
            if keyword_results:
                rag_context = f"--- Найдено в истории чата ---\n{keyword_results}"
                add_ai_thought(f"[DB] Keyword search found results", (150,255,150))

        system_prompt = build_system_prompt()

        messages = [SystemMessage(content=system_prompt)]

        if rag_context:
            messages.append(AIMessage(content=f"Я вспоминаю:\n{rag_context}"))

        max_hist = runtime.settings.context.max_history_messages
        recent = (
            runtime.session_context[-max_hist:]
            if len(runtime.session_context) > max_hist
            else runtime.session_context
        )

        for msg in recent:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_text))

        content = ""

        if runtime.react_agent:
            add_ai_thought("[MCP] Using react agent with tools", (100, 200, 255))
            try:
                result = await runtime.react_agent.ainvoke({"messages": messages})
            except Exception as agent_err:
                logger.error(f"MCP agent execution failed: {agent_err}", exc_info=True)
                add_ai_thought(f"[MCP] Agent error: {agent_err}. Falling back to direct LLM.", (255, 100, 100))
                if runtime.runtime_chat_mode == "ollama":
                    response = await runtime.llm.ainvoke(messages)
                    content = response.content
                else:
                    full_content = ""
                    async for chunk in runtime.llm.astream(messages):
                        msg_chunk = chunk.message if hasattr(chunk, 'message') else chunk
                        if msg_chunk.content:
                            full_content += msg_chunk.content
                            update_ai_message_streaming(full_content)
                    content = full_content
                finalize_ai_message_streaming()
                return content

            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        add_ai_thought(f"[TOOL] Call: {tool_name} args={tool_args}", (255, 220, 100))
                        logger.info(f"MCP TOOL CALL: {tool_name} {tool_args}")

                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', 'unknown')
                    result_preview = msg.content[:200] + ('...' if len(msg.content) > 200 else '')
                    add_ai_thought(f"[TOOL] Result from {tool_name}: {result_preview}", (100, 255, 100))
                    logger.info(f"MCP TOOL RESULT ({tool_name}): {msg.content}")
                    if "error" in msg.content.lower() or "exception" in msg.content.lower():
                        add_ai_thought(f"[TOOL] Error in {tool_name}: {msg.content[:300]}", (255, 100, 100))

            content = result["messages"][-1].content
            final_msg = result["messages"][-1]
            reasoning = ""
            if hasattr(final_msg, "additional_kwargs"):
                reasoning = (
                    final_msg.additional_kwargs.get("reasoning_content", "") or ""
                )
            if not reasoning and hasattr(final_msg, "response_metadata"):
                reasoning = (
                    final_msg.response_metadata.get("reasoning_content", "") or ""
                )
            if reasoning:
                add_ai_thought(f"[REASONING]\n{reasoning}", (180, 180, 150))
        elif runtime.runtime_chat_mode == "ollama":
            response = await runtime.llm.ainvoke(messages)
            content = response.content
            reasoning = response.response_metadata.get("reasoning_content", "") or ""
            if reasoning:
                add_ai_thought(f"[REASONING]\n{reasoning}", (180,180,150))
        else:
            full_content = ""
            full_reasoning = ""
            async for chunk in runtime.llm.astream(messages):
                msg = chunk.message if hasattr(chunk, 'message') else chunk
                if msg.content:
                    full_content += msg.content
                    update_ai_message_streaming(full_content)
                if "reasoning_content" in msg.additional_kwargs:
                    full_reasoning += msg.additional_kwargs["reasoning_content"]
            if full_reasoning:
                add_ai_thought(f"[REASONING]\n{full_reasoning}", (180,180,150))
            content = full_content

        return content

    try:
        return await attempt_generation()
    except (BadRequestError, APITimeoutError, TimeoutError) as e:
        error_str = str(e).lower()
        if "context length" in error_str or "exceeds" in error_str:
            add_ai_thought(f"[WARN] CONTEXT OVERFLOW: {len(runtime.session_context)} messages", (255,150,100))
            await dump_context_to_memory()
            if runtime.memory_manager:
                fresh = runtime.memory_manager.get_recent_history(limit=runtime.settings.context.max_history_messages)
                runtime.session_context.extend(fresh)
                add_ai_thought(f"[CTX] Rehydrated: loaded {len(fresh)} messages", (150,255,150))
            return await attempt_generation()
        elif "timeout" in error_str:
            add_ai_thought("[WARN] LLM request timed out. Please try again with a shorter message.", (255,200,100))
            return "I'm sorry, I took too long to think. Could you please repeat your question or make it shorter?"
        else:
            logger.error(f"LLM request failed: {e}")
            return f"Sorry, I encountered an error: {e}"
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return f"Sorry, I encountered an error: {e}"


# ============================================================================
# Graceful Shutdown
# ============================================================================

def initiate_graceful_shutdown():
    if runtime._shutting_down:
        return
    runtime._shutting_down = True

    import dearpygui.dearpygui as dpg
    unsaved_count = len(runtime.session_context)
    add_ai_thought(f"[SYS] SHUTDOWN: {unsaved_count} unsaved messages in context", (255,150,150))

    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Saving memories...")

    if runtime.session_context:
        add_ai_thought(f"[SYS] SHUTDOWN: Dumping context to long-term memory...", (255,200,100))
        try:
            future = asyncio.run_coroutine_threadsafe(
                dump_context_to_memory(), runtime.async_loop
            )
            future.result(timeout=30)
            add_ai_thought("[SYS] STATUS: Context dump complete.", (150,255,150))
        except Exception as e:
            logger.error(f"Shutdown dump failed: {e}")
            add_ai_thought(f"[ERR] Shutdown dump failed: {e}", (255,100,100))
    else:
        add_ai_thought("[SYS] SHUTDOWN: No context to save.", (200,200,100))

    save_window_geometry()
    dpg.stop_dearpygui()


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    initiate_graceful_shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    config_path = Path("config/settings.yaml")
    runtime.settings = AppSettings.from_yaml(str(config_path))
    Path(runtime.settings.diary.storage_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("hf_cache").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    runtime.runtime_chat_mode = runtime.settings.chat_mode
    runtime.runtime_embed_mode = runtime.settings.embedding_mode
    if runtime.runtime_chat_mode == "ollama":
        runtime.runtime_chat_params = {
            "model": runtime.settings.ollama.chat_model,
            "base_url": runtime.settings.ollama.base_url,
            "api_key": runtime.settings.ollama.api_key,
            "temperature": runtime.settings.ollama.temperature,
            "max_tokens": runtime.settings.ollama.max_tokens,
            "timeout": runtime.settings.ollama.timeout,
        }
    else:
        runtime.runtime_chat_params = {
            "model": runtime.settings.llama.chat_model,
            "base_url": runtime.settings.llama.base_url,
            "api_key": runtime.settings.llama.api_key,
            "temperature": runtime.settings.llama.temperature,
            "max_tokens": runtime.settings.llama.max_tokens,
            "timeout": runtime.settings.llama.timeout,
        }
    if runtime.settings.embedding_mode == "ollama":
        runtime.runtime_embed_params = {
            "model": runtime.settings.ollama.embedding_model,
            "base_url": runtime.settings.ollama.base_url,
            "api_key": runtime.settings.ollama.api_key,
        }
    else:
        runtime.runtime_embed_params = {
            "model": runtime.settings.ollama.embedding_model,
            "base_url": runtime.settings.ollama.base_url,
            "api_key": runtime.settings.ollama.api_key,
        }

    if runtime.settings.debug:
        logging_exceptions.install_excepthook()

    logger.info(f"Starting AI_EveryNyan v0.18.0 (debug={runtime.settings.debug})")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    init_memory_manager()
    init_components()
    reinit_llm()
    reinit_embeddings()
    init_character()

    runtime.async_loop = asyncio.new_event_loop()
    runtime.async_thread = threading.Thread(
        target=run_async_loop, args=(runtime.async_loop,), daemon=True
    )
    runtime.async_thread.start()

    setup_gui()
    refresh_models_list()
    init_query_preprocessor()

    future = asyncio.run_coroutine_threadsafe(init_mcp_agent(), runtime.async_loop)
    try:
        future.result(timeout=30)
    except Exception as e:
        logger.warning(f"[MCP] Agent initialization failed: {e}")
    add_ai_thought("[SYS] STATUS: Online. Ready.", (100, 255, 100))

    try:
        import dearpygui.dearpygui as dpg
        dpg.start_dearpygui()
    finally:
        add_ai_thought(f"[SYS] SHUTDOWN: {len(runtime.session_context)} pending messages", (255,150,150))
        if runtime.session_context:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    dump_context_to_memory(), runtime.async_loop
                )
                future.result(timeout=300)
                add_ai_thought("[SYS] Final context dump successful.", (150,255,150))
            except Exception as e:
                logger.error(f"Final dump failed: {e}")
                add_ai_thought(f"[ERR] Final dump failed: {e}", (255,100,100))
        else:
            add_ai_thought("[SYS] No context to save on exit.", (200,200,100))
        if runtime.mcp_client:
            logger.info("[MCP] Client released (no explicit close needed)")
        runtime.async_loop.call_soon_threadsafe(runtime.async_loop.stop)
        runtime.async_thread.join(timeout=2.0)
        if runtime.memory_manager:
            runtime.memory_manager.close()
        save_window_geometry()
        dpg.destroy_context()
        logger.info("[SYS] Shutdown complete.")
        add_ai_thought("[SYS] Goodbye!", (100,255,100))


if __name__ == "__main__":
    main()
