#!/usr/bin/env python3
"""
AI_EveryNyan - DearPyGui Chat with LangChain + Qdrant RAG + DuckDB History
Modular Character System + Smart Context Management + Structured Diary Metadata

src/main.py
Version:     0.17.10
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v0.17.10 (Soror L'.L'.):
  [FIX] Dead-model handling: synthetic error replies ("Sorry, I encountered
      an error...", timeouts) are no longer RETURNED as assistant messages -
      they were being persisted to DuckDB/Qdrant/session_context as fake
      memories. process_message now raises GenerationError; the GUI shows the
      text but skips save_to_memory.
  [+] 410/retired/missing-model errors get a specific remedy message
      (update chat_model in settings.yaml) + [LLM] fallback log.
  [+] 429/rate-limit errors: Kilo-style silent outer retry ladder
      (RATE_LIMIT_DELAYS = 10/30/60s, ~100s patience) paced for
      per-minute provider quotas; SDK max_retries kept short so quota windows
      are not burned. Chat is never shown intermediate failures.
  [+] Fail-fast cooldown: after ladder exhaustion the backend is marked
      unhealthy (runtime.mark_llm_cooldown, 300s) - subsequent messages error
      INSTANTLY instead of re-entering the ladder, so the GUI is never trapped;
      changing chat settings resets the cooldown immediately.
  [+] chat_mode "openai": startup params resolve via
      runtime.chat_settings_for_mode() (ollama / llama / openai sections);
      ChatOpenAI paths accept the openai mode alongside ollama.

Patch Notes v0.17.9 (Soror L'.L'.):
  [+] Dialogue history messages with a timestamp get a "[YYYY-MM-DD HH:MM]"
      prefix at prompt assembly (format_timestamp) - paired with the <time>
      block in the system prompt, the model can reason about when each
      message was sent and how long ago past conversations happened.

Patch Notes v0.17.8 (by pytraveler):
  [+] ComfyUI daemon integration: initialize and start ComfyUIDaemon at startup
      with configurable check_interval from settings.
  [+] Graceful shutdown: stop ComfyUI monitor watchdog and daemon on exit.

Patch Notes v0.17.7 (by pytraveler):
  [+] ComfyUI image path injection: _extract_image_paths() extracts file paths from
      generate_image tool results and appends them to the LLM response text so that
      gui.py add_chat_message() can parse them and render image thumbnails inline.
  [+] Deduplication: only appends paths that are NOT already present in the LLM response,
      so paths are never duplicated even if the model mentions them verbatim.
  [~] Image paths become part of the response stored in DuckDB, so thumbnails appear
      on history reload as long as the files still exist on disk.

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
import os
import re
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
from memory_manager import format_timestamp
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

_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


class GenerationError(Exception):
    """Synthetic failure reply (dead model, timeout, API error).

    Raised instead of returning a fake assistant message, so the caller can
    show the text to the user WITHOUT persisting it to chat history / RAG -
    error texts must never become 'memories'.
    """
    pass


def _model_unavailable_hint(error_str: str) -> str:
    """User-facing remedy when the configured model is dead (404/410/retired)."""
    model = getattr(runtime.settings, "chat_mode", None)
    try:
        model = runtime.settings.get_chat_config().chat_model
    except Exception:
        pass
    logger.warning(
        "[LLM] fallback: chat model '%s' is unavailable (retired/missing). "
        "Remedy: update chat_model in config\\settings.yaml and restart",
        model,
    )
    return (
        f"My language model ('{model}') is currently unavailable "
        f"(the server reported it as retired or missing). "
        f"Please update chat_model in config/settings.yaml to an installed model and restart."
    )


_TIMESTAMP_ECHO_RE = re.compile(r"^\s*(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*)+")


def _strip_timestamp_echo(text: str) -> str:
    """Models sometimes COPY the "[YYYY-MM-DD HH:MM]" prefixes we add to
    history entries back into their reply (worst case doubled). Strip any
    leading timestamp prefixes from the reply before it is shown or saved."""
    return _TIMESTAMP_ECHO_RE.sub("", text, count=1)
    return (
        f"My language model ('{model}') is currently unavailable "
        f"(the server reported it as retired or missing). "
        f"Please update chat_model in config/settings.yaml to an installed model and restart me."
    )


def _extract_image_paths(content: str) -> list[str]:
    """Extract existing image file paths from tool result text."""
    paths = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and any(stripped.lower().endswith(ext) for ext in _IMAGE_EXTENSIONS):
            if Path(stripped).is_file():
                paths.append(stripped)
    return paths

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
            # Timestamped messages carry a "[YYYY-MM-DD HH:MM]" prefix so the
            # model can reason about when each message was sent (paired with
            # the current time in the <time> system block).
            content = msg["content"]
            ts_prefix = format_timestamp(msg.get("timestamp"))
            if ts_prefix:
                content = f"{ts_prefix} {content}"
            if msg["role"] == "user":
                messages.append(HumanMessage(content=content))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_text))

        content = ""

        if runtime.react_agent:
            if runtime.agent_needs_rebuild:
                # Chat settings changed moments ago: the scheduled async
                # rebuild may not have run yet - without this, the OLD model
                # answers the first messages after a model switch.
                logger.info("[MCP] Agent rebuild pending - rebuilding inline (stale model guard)")
                from runtime import _recreate_mcp_agent
                await _recreate_mcp_agent()
            add_ai_thought("[MCP] Using react agent with tools", (100, 200, 255))
            try:
                result = await runtime.react_agent.ainvoke({"messages": messages})
            except Exception as agent_err:
                logger.error(f"MCP agent execution failed: {agent_err}", exc_info=True)
                add_ai_thought(f"[MCP] Agent error: {agent_err}. Falling back to direct LLM.", (255, 100, 100))
                if runtime.runtime_chat_mode in ("ollama", "openai"):
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
                return _strip_timestamp_echo(content)

            collected_image_paths: list[str] = []
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
                    if tool_name == "generate_image":
                        collected_image_paths.extend(_extract_image_paths(msg.content))

            content = result["messages"][-1].content
            if collected_image_paths:
                missing = [p for p in collected_image_paths if p not in content]
                if missing:
                    content += "\n" + "\n".join(missing)
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
        elif runtime.runtime_chat_mode in ("ollama", "openai"):
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

        # Models sometimes copy the "[...]" timestamp prefixes from history
        # entries back into their reply - never let those leak into the
        # response or the stored history.
        content = _strip_timestamp_echo(content)

        return content

    # Rate-limit retry ladder: Kilo-style silent backoff - the chat is never
    # shown intermediate failures, the console gets one timer line per retry.
    # Steps sized for per-MINUTE provider quotas (Mistral free tier): the SDK
    # gives up quickly (max_retries=2), pacing is done by this loop.
    RATE_LIMIT_DELAYS = (10, 30, 60)  # seconds; ~100s total patience

    # Fast-fail cooldown: once the ladder is exhausted, subsequent messages
    # error INSTANTLY (no new ladders) until the cooldown expires or the user
    # changes chat settings - so a dead backend never traps the GUI.
    LLM_COOLDOWN_SEC = 300

    remaining = runtime.llm_cooldown_remaining()
    if remaining:
        raise GenerationError(
            f"My model backend failed repeatedly and is in cooldown for another "
            f"~{remaining}s. Change chat_model / backend in settings (applies "
            f"immediately) - or wait and send again."
        )

    def _classify_llm_error(e: Exception) -> str:
        s = str(e).lower()
        if isinstance(e, (APITimeoutError, TimeoutError)) or "timed out" in s:
            return "timeout"
        if "context length" in s or "exceeds" in s and "context" in s:
            return "context_overflow"
        if "429" in s or "rate limit" in s:
            return "rate_limit"
        if "410" in s or "retired" in s or "does not exist" in s or "not found" in s:
            return "model_unavailable"
        return "generic"

    rate_attempt = 0
    while True:
        try:
            return await attempt_generation()
        except GenerationError:
            raise
        except Exception as e:
            kind = _classify_llm_error(e)

            if kind == "context_overflow":
                add_ai_thought(f"[WARN] CONTEXT OVERFLOW: {len(runtime.session_context)} messages", (255,150,100))
                await dump_context_to_memory()
                if runtime.memory_manager:
                    fresh = runtime.memory_manager.get_recent_history(limit=runtime.settings.context.max_history_messages)
                    runtime.session_context.extend(fresh)
                    add_ai_thought(f"[CTX] Rehydrated: loaded {len(fresh)} messages", (150,255,150))
                continue

            if kind == "rate_limit" and rate_attempt < len(RATE_LIMIT_DELAYS):
                wait = RATE_LIMIT_DELAYS[rate_attempt]
                rate_attempt += 1
                logger.info(
                    "[LLM] Rate limited - silent retry %d/%d in %ds (model='%s')",
                    rate_attempt, len(RATE_LIMIT_DELAYS), wait,
                    runtime.settings.get_chat_config().chat_model,
                )
                await asyncio.sleep(wait)
                continue

            if kind == "rate_limit":
                logger.warning("[LLM] fallback: still rate limited after %d silent retries", len(RATE_LIMIT_DELAYS))
                runtime.mark_llm_cooldown(LLM_COOLDOWN_SEC)
                raise GenerationError(
                    "I'm being rate-limited by my model provider and my retries "
                    "ran out. I'll answer normally once the limit resets - "
                    "or switch chat_model / backend in settings to reach me right away."
                )

            if kind == "timeout":
                add_ai_thought("[WARN] LLM request timed out. Please try again with a shorter message.", (255,200,100))
                raise GenerationError(
                    "I'm sorry, I took too long to think. Could you please repeat your question or make it shorter?"
                )

            if kind == "model_unavailable":
                raise GenerationError(_model_unavailable_hint(str(e)))

            logger.error(f"LLM request failed: {e}")
            raise GenerationError(f"Sorry, I encountered an error: {e}")


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
    # Persona comes from the launcher (menu choice) via env; default EveryNyan.
    runtime.set_active_persona(os.environ.get("AI_EVERYNYAN_PERSONA"))
    logger.info(f"[APP] Active persona: {runtime.active_persona}")

    config_path = Path("config/settings.yaml")
    runtime.settings = AppSettings.from_yaml(str(config_path))
    Path(runtime.persona_diary_dir()).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("hf_cache").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    runtime.runtime_chat_mode = runtime.settings.chat_mode
    runtime.runtime_embed_mode = runtime.settings.embedding_mode
    _chat_defaults = runtime.chat_settings_for_mode(runtime.runtime_chat_mode)
    runtime.runtime_chat_params = {
        "model": _chat_defaults.chat_model,
        "base_url": _chat_defaults.base_url,
        "api_key": _chat_defaults.api_key,
        "temperature": _chat_defaults.temperature,
        "max_tokens": _chat_defaults.max_tokens,
        "timeout": _chat_defaults.timeout,
        "max_retries": getattr(_chat_defaults, "max_retries", 4),
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

    # Chromium auto-update (config-gated, interval-stamped, visible progress).
    # Runs before heavy init so the new browser is in place for this session.
    try:
        from browser_updater import maybe_update_browsers
        maybe_update_browsers(
            python_exe=sys.executable,
            browsers_path=Path("playwright_browsers").resolve(),
            enabled=runtime.settings.browser_update.enabled,
            check_interval_days=runtime.settings.browser_update.check_interval_days,
        )
    except Exception as exc:
        logger.warning("[INSTALL] fallback: browser update hook failed (app continues): %s", exc)

    # Name the console window so the launcher's zombie cleanup can find
    # orphaned instances by title even when ExecutablePath is unreadable.
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(f"AI_EveryNyan v0.18.0")
    except Exception as exc:
        logger.debug("[APP] fallback: could not set console title: %s", exc)

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

    from comfyui_monitor import ComfyUIDaemon
    from comfyui_discovery import resolve_comfyui_server

    resolved_server = resolve_comfyui_server(runtime.settings.comfyui.server)
    runtime.comfyui_daemon = ComfyUIDaemon(
        server=resolved_server,
        check_interval=runtime.settings.comfyui.daemon_check_interval,
    )
    runtime.comfyui_daemon.start()

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
        # Stop ComfyUI monitor watchdog before daemon
        from gui import _comfyui_watchdog_stop
        _comfyui_watchdog_stop.set()
        if runtime.comfyui_daemon:
            runtime.comfyui_daemon.stop()
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