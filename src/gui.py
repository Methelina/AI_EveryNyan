"""
DearPyGui GUI setup, control panel callbacks, AI thoughts display, and splitter management.
Handles viewport, theming, chat rendering, model selection, and runtime parameter controls.

/src/gui.py
Version:     0.17.7
Author:      Soror L.'.L.'.
Updated:     2026-05-03

Patch Notes v0.17.7 (by pytraveler):
  [+] Image thumbnails in chat: add_chat_message() now parses message text for image
      file paths (lines that are existing files ending in .png/.jpg/.webp/.bmp/.gif),
      strips them from the displayed text, and renders them as clickable thumbnails
      (max 280 px, aspect-ratio preserved) below the message body.
  [+] Modal image viewer: clicking a thumbnail opens a popup with the full-size image
      scaled to 80 % of the viewport, plus a Close button. Image and Close button
      resize dynamically when the modal window is resized (item_resize_handler).
  [+] Shared texture registry (chat_texture_registry) and image_button_theme created
      in setup_gui() — frameless transparent buttons with subtle hover border.
  [+] _parse_image_paths(), _add_image_thumbnail(), _open_image_modal() helpers.
  [~] History loading in setup_gui() now calls add_chat_message() instead of building
      widgets inline — so image thumbnails appear on reload too.

Patch Notes v0.17.6 (by pytraveler):
  [+] Extracted from main.py: setup_gui(), all DPG widget construction.
  [+] AI thoughts: add_ai_thought(), update_ai_message_streaming(), finalize_ai_message_streaming().
  [+] Splitter management: update_split_heights(), drag/resize handlers.
  [+] Control panel: on_chat_mode_changed(), on_embed_mode_changed(), refresh_models_list().
  [+] Message flow: on_send_message(), handle_async_response(), on_memory_report().
  [*] No functional changes from original main.py code.
  [FIX] Chat text wrapping: replaced all wrap=-1 with explicit pixel widths calculated from
        chat_area / ai_thoughts_area rect size. DearPyGui wrap=-1 fails inside nested
        dpg.group() containers with indent, causing single-line text overflow.
  [+] Added get_chat_wrap_width(), get_thoughts_wrap_width() helpers.
  [+] Added _chat_text_tags tracking list and _refresh_text_wrap_widths() to re-apply
        wrap widths on window resize (called from update_split_heights()).
  [~] Affected: add_chat_message(), update_ai_message_streaming(), add_ai_thought(),
        history loading in setup_gui().
  [+] Window geometry persistence: save/load viewport position (x, y) and size
        (width, height) to data/window_geometry.json on shutdown/startup.
  [+] Added load_window_geometry(), save_window_geometry(), apply_window_geometry().
  [~] save_window_geometry() called in initiate_graceful_shutdown() and main() finally block.
  [~] apply_window_geometry() called in setup_gui() after show_viewport().
"""

from logger import logger
from datetime import datetime
from json import loads, dumps
from pathlib import Path
from typing import Optional, Tuple
import uuid

import dearpygui.dearpygui as dpg

import runtime
from runtime import (
    fetch_models_from_backend,
    apply_chat_settings,
    apply_embedding_settings,
    reset_to_yaml_defaults,
)
from character import (
    refresh_character_list,
    on_character_selected,
)


_WINDOW_GEOMETRY_PATH = Path("data/window_geometry.json")


def load_window_geometry() -> Optional[dict]:
    try:
        if _WINDOW_GEOMETRY_PATH.exists():
            return loads(_WINDOW_GEOMETRY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"Could not load window geometry: {e}")
    return None


def save_window_geometry():
    try:
        if not dpg.is_viewport_ok():
            return
        pos = dpg.get_viewport_pos()
        w = dpg.get_viewport_width()
        h = dpg.get_viewport_height()
        _WINDOW_GEOMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _WINDOW_GEOMETRY_PATH.write_text(
            dumps({"x": pos[0], "y": pos[1], "width": w, "height": h}),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug(f"Could not save window geometry: {e}")


def apply_window_geometry():
    geom = load_window_geometry()
    if geom:
        try:
            dpg.set_viewport_pos([geom.get("x", 100), geom.get("y", 100)])
            w = geom.get("width")
            h = geom.get("height")
            if w and h:
                dpg.set_viewport_width(w)
                dpg.set_viewport_height(h)
        except Exception as e:
            logger.debug(f"Could not apply window geometry: {e}")


# Tags of add_text widgets in chat_area whose wrap width must track resize.
_chat_text_tags: list = []


def get_chat_wrap_width() -> int:
    """Return pixel width for text wrapping inside chat_area.

    Accounts for the 20 px indent and window margins.  Falls back to 600
    when the viewport has not been realised yet.
    """
    try:
        if dpg.does_item_exist("chat_area"):
            w = dpg.get_item_rect_size("chat_area")[0]
            if w > 80:
                return int(w) - 60          # indent(20) + side margins(40)
    except Exception:
        pass
    return 600


def get_thoughts_wrap_width() -> int:
    """Return pixel width for text wrapping inside ai_thoughts_area."""
    try:
        if dpg.does_item_exist("ai_thoughts_area"):
            w = dpg.get_item_rect_size("ai_thoughts_area")[0]
            if w > 60:
                return int(w) - 40
    except Exception:
        pass
    return 500


def _refresh_text_wrap_widths():
    """Re-apply wrap width to every tracked chat text widget."""
    w = get_chat_wrap_width()
    for tag in _chat_text_tags:
        try:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, wrap=w)
        except Exception:
            pass


# ============================================================================
# AI Thoughts UI System
# ============================================================================

def add_ai_thought(text: str, color: Tuple[int, int, int] = (200, 200, 150)):
    logger.info(f"[AI_THOUGHT] {text}")
    try:
        if dpg.does_item_exist("thoughts_placeholder"):
            dpg.delete_item("thoughts_placeholder")
        timestamp = datetime.now().strftime("%H:%M:%S")
        tw = get_thoughts_wrap_width()
        with dpg.group(parent="ai_thoughts_area", horizontal=True):
            dpg.add_text(f"[{timestamp}] ", color=(100, 100, 100))
            dpg.add_text(text, color=color, wrap=tw)
        dpg.set_y_scroll("ai_thoughts_area", 1e9)
    except Exception as e:
        logger.debug(f"Thought UI update skipped: {e}")


def update_ai_message_streaming(text: str):
    tag = runtime._current_ai_message_tag
    if tag and dpg.does_item_exist(tag):
        dpg.set_value(tag, text)
    else:
        wrap_w = get_chat_wrap_width()
        with dpg.group(parent="chat_area", horizontal=False):
            with dpg.group(horizontal=True):
                dpg.add_text("AI_EveryNyan:", color=(255,200,100))
            with dpg.group(indent=20):
                runtime._current_ai_message_tag = dpg.add_text(text, wrap=wrap_w)
                _chat_text_tags.append(runtime._current_ai_message_tag)
        dpg.set_y_scroll("chat_area", 1e9)


def finalize_ai_message_streaming():
    runtime._current_ai_message_tag = None


# ============================================================================
# GUI Splitter Management
# ============================================================================

def update_split_heights():
    if not dpg.does_item_exist("left_panel"):
        return
    try:
        left_height = dpg.get_item_rect_size("left_panel")[1]
    except:
        return
    if left_height <= 0:
        return

    # Reserve vertical space for the input row and status text below the splitter
    input_reserve = 0
    for tag in ("input_row", "status_text"):
        try:
            if dpg.does_item_exist(tag):
                h = dpg.get_item_rect_size(tag)[1]
                if h > 0:
                    input_reserve += h
        except Exception:
            pass
    if input_reserve < 20:
        input_reserve = 70  # fallback for first frame before layout settles
    input_reserve += 10     # spacing margin

    min_bottom = 50
    max_bottom = max(min_bottom, left_height - input_reserve - 50)
    bottom = min(max(runtime.split_bottom_height, min_bottom), max_bottom)
    chat_height = left_height - bottom - input_reserve
    if chat_height < 50:
        chat_height = 50
        bottom = left_height - chat_height - input_reserve
        if bottom < min_bottom:
            bottom = min_bottom
    dpg.configure_item("chat_area", height=chat_height)
    dpg.configure_item("ai_thoughts_area", height=bottom)
    runtime.split_bottom_height = bottom

    global _chat_text_tags
    _chat_text_tags = [t for t in _chat_text_tags if dpg.does_item_exist(t)]
    _refresh_text_wrap_widths()


def on_separator_drag(sender, app_data):
    dy = app_data[1]
    runtime.split_bottom_height -= dy
    update_split_heights()


def on_left_panel_resize():
    update_split_heights()


def on_chat_area_resize():
    update_split_heights()


# ============================================================================
# Chat message display
# ============================================================================

_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
_THUMB_MAX_SIZE = 280  # max thumbnail dimension in pixels


def _parse_image_paths(text: str) -> Tuple[str, list]:
    """Extract valid image file paths from message text.

    Lines that consist solely of an existing file path ending in an image
    extension are stripped from the text and collected separately.

    Returns (cleaned_text, list_of_existing_image_paths).
    """
    lines = text.split('\n')
    clean_lines: list[str] = []
    image_paths: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and any(stripped.lower().endswith(ext) for ext in _IMAGE_EXTENSIONS):
            if Path(stripped).is_file():
                image_paths.append(stripped)
                continue
        clean_lines.append(line)
    return '\n'.join(clean_lines), image_paths


def _open_image_modal(sender, app_data, user_data):
    """Open a modal popup showing the full-size image that resizes with the window."""
    tex_tag = user_data["tex_tag"]
    orig_w = user_data["orig_w"]
    orig_h = user_data["orig_h"]
    image_path = user_data["path"]

    modal_tag = f"img_modal_{uuid.uuid4().hex[:8]}"
    img_tag = f"{modal_tag}_img"
    btn_tag = f"{modal_tag}_btn"
    hr_tag = f"{modal_tag}_hr"

    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()
    max_w = int(vp_w * 0.8)
    max_h = int(vp_h * 0.8)

    ratio_w = max_w / orig_w if orig_w > max_w else 1.0
    ratio_h = max_h / orig_h if orig_h > max_h else 1.0
    ratio = min(ratio_w, ratio_h)
    display_w = int(orig_w * ratio)
    display_h = int(orig_h * ratio)

    def _on_modal_resize():
        if not dpg.does_item_exist(modal_tag):
            return
        try:
            win_size = dpg.get_item_rect_size(modal_tag)
            avail_w = win_size[0] - 24
            avail_h = win_size[1] - 84
        except Exception:
            return
        if avail_w < 50 or avail_h < 50:
            return
        r = min(avail_w / orig_w, avail_h / orig_h)
        new_w = max(int(orig_w * r), 50)
        new_h = max(int(orig_h * r), 50)
        try:
            if dpg.does_item_exist(img_tag):
                dpg.configure_item(img_tag, width=new_w, height=new_h)
            if dpg.does_item_exist(btn_tag):
                dpg.configure_item(btn_tag, width=new_w)
        except Exception:
            pass

    def _close_modal():
        if dpg.does_item_exist(hr_tag):
            dpg.delete_item(hr_tag)
        if dpg.does_item_exist(modal_tag):
            dpg.delete_item(modal_tag)

    with dpg.window(
        tag=modal_tag,
        modal=True,
        popup=True,
        label=Path(image_path).name,
        no_resize=False,
        width=display_w + 24,
        height=display_h + 84,
    ):
        dpg.add_image(tex_tag, tag=img_tag, width=display_w, height=display_h)
        dpg.add_spacer(height=5)
        dpg.add_button(
            label="Close",
            tag=btn_tag,
            width=display_w,
            callback=lambda s, a: _close_modal(),
        )

    with dpg.item_handler_registry(tag=hr_tag):
        dpg.add_item_resize_handler(callback=lambda s, a: _on_modal_resize())
    dpg.bind_item_handler_registry(modal_tag, hr_tag)


def _add_image_thumbnail(parent: str, image_path: str):
    """Add a clickable image thumbnail to a parent widget."""
    path = Path(image_path)
    if not path.exists():
        return

    try:
        width, height, channels, data = dpg.load_image(str(path))
    except Exception as e:
        logger.warning(f"[GUI] Failed to load thumbnail {image_path}: {e}")
        return

    tex_tag = f"img_tex_{uuid.uuid4().hex[:8]}"
    try:
        dpg.add_static_texture(
            width, height, data,
            tag=tex_tag,
            parent="chat_texture_registry",
        )
    except Exception:
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(width, height, data, tag=tex_tag)

    # Scale to thumbnail size preserving aspect ratio
    if max(width, height) > _THUMB_MAX_SIZE:
        ratio = _THUMB_MAX_SIZE / max(width, height)
        thumb_w = int(width * ratio)
        thumb_h = int(height * ratio)
    else:
        thumb_w = width
        thumb_h = height

    btn_tag = f"img_btn_{uuid.uuid4().hex[:8]}"
    dpg.add_image_button(
        tex_tag,
        tag=btn_tag,
        width=thumb_w,
        height=thumb_h,
        callback=_open_image_modal,
        user_data={
            "tex_tag": tex_tag,
            "orig_w": width,
            "orig_h": height,
            "path": image_path,
        },
        parent=parent,
    )

    if dpg.does_item_exist("image_button_theme"):
        dpg.bind_item_theme(btn_tag, "image_button_theme")


def add_chat_message(sender: str, text: str, color: tuple):
    """Add a chat message to the chat area.

    Automatically detects image file paths in the text and displays
    them as clickable thumbnails below the text content.
    """
    wrap_w = get_chat_wrap_width()
    dpg.set_y_scroll("chat_area", 1e9)

    clean_text, image_paths = _parse_image_paths(text)

    with dpg.group(parent="chat_area", horizontal=False):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{sender}:", color=color)
        with dpg.group(indent=20) as msg_body:
            if clean_text.strip():
                tag = dpg.add_text(clean_text, wrap=wrap_w)
                _chat_text_tags.append(tag)
            for img_path in image_paths:
                _add_image_thumbnail(parent=msg_body, image_path=img_path)
        dpg.add_spacer(height=5)
    dpg.set_y_scroll("chat_area", 1e9)


# ============================================================================
# Font discovery
# ============================================================================

def find_available_font() -> Optional[str]:
    local = Path("data/fonts")
    for f in ["JetBrainsMonoNerdFont-Regular.ttf", "JetBrainsMonoNerdFont-Medium.ttf", "JetBrainsMonoNerdFont-Bold.ttf"]:
        p = local / f
        if p.exists():
            return str(p)
    for f in [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]:
        if Path(f).exists():
            return f
    return None


# ============================================================================
# Control panel callbacks
# ============================================================================

def on_chat_mode_changed(sender, app_data):
    runtime.runtime_chat_mode = app_data
    if app_data == "ollama":
        runtime.runtime_chat_params.update({
            "base_url": runtime.settings.ollama.base_url,
            "api_key": runtime.settings.ollama.api_key,
            "model": runtime.settings.ollama.chat_model,
            "temperature": runtime.settings.ollama.temperature,
            "max_tokens": runtime.settings.ollama.max_tokens,
            "timeout": runtime.settings.ollama.timeout,
        })
    else:
        runtime.runtime_chat_params.update({
            "base_url": runtime.settings.llama.base_url,
            "api_key": runtime.settings.llama.api_key,
            "model": runtime.settings.llama.chat_model,
            "temperature": runtime.settings.llama.temperature,
            "max_tokens": runtime.settings.llama.max_tokens,
            "timeout": runtime.settings.llama.timeout,
        })
    dpg.set_value("chat_temp", runtime.runtime_chat_params["temperature"])
    dpg.set_value("chat_max_tokens", runtime.runtime_chat_params["max_tokens"])
    dpg.set_value("chat_timeout", runtime.runtime_chat_params["timeout"])
    dpg.set_value("chat_model_hidden", runtime.runtime_chat_params["model"])
    refresh_models_list()
    apply_chat_settings(runtime.runtime_chat_params)


def on_embed_mode_changed(sender, app_data):
    runtime.runtime_embed_mode = app_data
    if runtime.runtime_embed_mode == "ollama":
        runtime.runtime_embed_params.update({
            "model": runtime.settings.ollama.embedding_model,
            "base_url": runtime.settings.ollama.base_url,
            "api_key": runtime.settings.ollama.api_key,
        })
    else:
        runtime.runtime_embed_params.update({
            "model": runtime.settings.ollama.embedding_model,
            "base_url": runtime.settings.ollama.base_url,
            "api_key": runtime.settings.ollama.api_key,
        })
        add_ai_thought("[GUI] LLaMA backend for embeddings not supported, using Ollama", (255,200,100))
    runtime.reinit_embeddings()
    add_ai_thought(f"[GUI] Embedding backend set to {runtime.runtime_embed_mode}", (100,255,100))


def refresh_models_list():
    backend = dpg.get_value("chat_mode_radio")
    if backend == "ollama":
        url = runtime.settings.ollama.base_url
        api_key = runtime.settings.ollama.api_key
    else:
        url = runtime.settings.llama.base_url
        api_key = runtime.settings.llama.api_key

    models = fetch_models_from_backend(backend, url, api_key)
    if models:
        dpg.configure_item("chat_model_combo", items=models)
        current_model = runtime.runtime_chat_params.get("model")
        if current_model in models:
            dpg.set_value("chat_model_combo", current_model)
        else:
            dpg.set_value("chat_model_combo", models[0])
        dpg.set_value("chat_model_hidden", dpg.get_value("chat_model_combo"))
        add_ai_thought(f"[GUI] Loaded {len(models)} models from {backend}", (100,255,100))
    else:
        add_ai_thought(f"[GUI] Failed to fetch models from {backend}", (255,100,100))


def apply_chat_from_ui():
    ui_vals = {
        "chat_mode": dpg.get_value("chat_mode_radio"),
        "model": dpg.get_value("chat_model_hidden"),
        "temperature": dpg.get_value("chat_temp"),
        "max_tokens": dpg.get_value("chat_max_tokens"),
        "timeout": dpg.get_value("chat_timeout"),
    }
    if ui_vals["chat_mode"] == "ollama":
        ui_vals["base_url"] = runtime.runtime_chat_params.get("base_url", runtime.settings.ollama.base_url)
        ui_vals["api_key"] = runtime.runtime_chat_params.get("api_key", runtime.settings.ollama.api_key)
    else:
        ui_vals["base_url"] = runtime.runtime_chat_params.get("base_url", runtime.settings.llama.base_url)
        ui_vals["api_key"] = runtime.runtime_chat_params.get("api_key", runtime.settings.llama.api_key)
    apply_chat_settings(ui_vals)


def apply_embed_from_ui():
    ui_vals = {
        "embed_mode": dpg.get_value("embed_mode_radio"),
        "model": dpg.get_value("embed_model"),
        "base_url": dpg.get_value("embed_base_url"),
        "api_key": dpg.get_value("embed_api_key"),
    }
    apply_embedding_settings(ui_vals)


def reset_to_yaml_defaults_and_update_ui():
    reset_to_yaml_defaults()
    dpg.set_value("chat_mode_radio", runtime.runtime_chat_mode)
    dpg.set_value("chat_model_hidden", runtime.runtime_chat_params.get("model", ""))
    dpg.set_value("chat_temp", runtime.runtime_chat_params.get("temperature", 0.7))
    dpg.set_value("chat_max_tokens", runtime.runtime_chat_params.get("max_tokens", 2048))
    dpg.set_value("chat_timeout", runtime.runtime_chat_params.get("timeout", 120))
    dpg.set_value("embed_mode_radio", runtime.runtime_embed_mode)
    refresh_models_list()
    refresh_character_list()
    add_ai_thought("[GUI] Reset to settings.yaml defaults", (100,255,100))


# ============================================================================
# Send message callback
# ============================================================================

def on_send_message(sender, app_data):
    if runtime._shutting_down:
        return
    user_text = dpg.get_value("user_input").strip()
    if not user_text:
        return
    add_chat_message("You", user_text, (100,200,255))
    dpg.set_value("user_input", "")
    dpg.configure_item("user_input", enabled=False)
    dpg.set_value("status_text", "Thinking...")
    add_ai_thought(f"[IN] User: \"{user_text[:30]}{'...' if len(user_text)>30 else ''}\"", (150,200,255))
    future = runtime.submit_to_async(handle_async_response(user_text))
    def on_done(fut):
        try:
            fut.result()
        except Exception as e:
            logger.exception(f"Task failed: {e}")
            dpg.configure_item("user_input", enabled=True)
            dpg.set_value("status_text", f"Error: {e}")
    future.add_done_callback(on_done)


async def handle_async_response(user_text: str):
    from main import process_message
    from rag import save_to_memory
    try:
        response = await process_message(user_text)
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("AI_EveryNyan", response, (255,200,100))
        add_ai_thought("[SYS] Response generated.", (150,255,150))
        await save_to_memory(user_text, response)
    except Exception as e:
        logger.exception("Unhandled error")
        dpg.configure_item("user_input", enabled=True)
        dpg.set_value("status_text", "")
        add_chat_message("Error", str(e), (255,100,100))
        add_ai_thought(f"[ERR] Handler Error: {e}", (255,100,100))


# ============================================================================
# Memory report
# ============================================================================

async def report_qdrant_status():
    if not runtime.qdrant_client:
        add_ai_thought("[RAG] Status: Qdrant client not available", (200,150,150))
        return
    try:
        info = runtime.qdrant_client.get_collection(runtime.settings.vector_db.collection)
        add_ai_thought(f"[RAG] Qdrant collection '{runtime.settings.vector_db.collection}': {info.points_count} vectors", (150,255,150))
        scroll = runtime.qdrant_client.scroll(collection_name=runtime.settings.vector_db.collection, limit=3, with_payload=True, with_vectors=False)
        for p in scroll[0]:
            ts = p.payload.get("metadata", {}).get("timestamp", "no timestamp")
            preview = str(p.payload.get("page_content", ""))[:60]
            add_ai_thought(f"  - {ts}: {preview}...", (180,180,180))
    except Exception as e:
        add_ai_thought(f"[RAG] Status error: {e}", (255,100,100))


def on_memory_report():
    add_ai_thought("[SYS] Generating memory report...", (200,200,100))
    runtime.submit_to_async(report_qdrant_status())
    if runtime.memory_manager:
        stats = runtime.memory_manager.get_stats()
        add_ai_thought(f"[DB] DuckDB: {stats.get('total_messages',0)} msgs, {stats.get('total_summaries',0)} summaries", (150,255,150))
        summaries = runtime.memory_manager.get_diary_summaries(limit=3)
        if summaries:
            for s in summaries:
                add_ai_thought(f"  - {s['timestamp']}: {s['text'][:80]}...", (180,180,180))


# ============================================================================
# GUI Setup
# ============================================================================

def setup_gui():
    dpg.create_context()
    dpg.add_texture_registry(tag="chat_texture_registry", show=False)
    font_path = find_available_font()
    if font_path:
        with dpg.font_registry():
            with dpg.font(font_path, 16) as main_font:
                pass
        dpg.bind_font(main_font)
    dpg.create_viewport(title=runtime.settings.gui.title, width=runtime.settings.gui.width, height=runtime.settings.gui.height, resizable=True)
    if runtime.settings.gui.theme == "dark":
        with dpg.theme() as dark_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25,25,35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40,40,60))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50,50,80))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220,220,220))
        dpg.bind_theme(dark_theme)

    # Image button theme — transparent background, subtle hover border
    with dpg.theme(tag="image_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 60, 90, 100))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (80, 80, 120, 140))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (100, 100, 150, 80))
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)

    with dpg.window(label="Chat", tag="main_window", no_title_bar=True, no_move=True, no_resize=False, no_scrollbar=True):
        dpg.set_primary_window("main_window", True)

        with dpg.group(horizontal=True):
            with dpg.child_window(tag="left_panel", width=-300, border=False, no_scrollbar=True):
                with dpg.child_window(tag="chat_area", height=-200, border=False):
                    if runtime.memory_manager:
                        history = runtime.memory_manager.get_recent_history(limit=50)
                        for msg in history:
                            color = (100,200,255) if msg['role'] == 'user' else (255,200,100)
                            sender = "You" if msg['role'] == 'user' else "AI_EveryNyan"
                            add_chat_message(sender, msg['content'], color)
                    else:
                        dpg.add_text("Welcome to AI_EveryNyan!", color=(150,150,200))

                with dpg.child_window(tag="ai_thoughts_area", height=-1, label="[SYSTEM] LOG", border=True):
                    dpg.add_text("[SYSTEM] STATUS: Idle", tag="thoughts_placeholder", color=(100,100,100))

                with dpg.group(horizontal=True, tag="input_row"):
                    dpg.add_input_text(tag="user_input", width=-220, hint="Type your message...", on_enter=True, callback=on_send_message)
                    dpg.add_button(label="Send", callback=on_send_message, width=70)
                    dpg.add_button(label="Memory Report", callback=on_memory_report, width=130)
                dpg.add_text("", tag="status_text", color=(100,100,100))

                with dpg.item_handler_registry(tag="chat_resize_handler"):
                    dpg.add_item_resize_handler(callback=on_chat_area_resize)
                dpg.bind_item_handler_registry("chat_area", "chat_resize_handler")

                with dpg.item_handler_registry(tag="left_panel_resize_handler"):
                    dpg.add_item_resize_handler(callback=on_left_panel_resize)
                dpg.bind_item_handler_registry("left_panel", "left_panel_resize_handler")

            with dpg.child_window(width=330, border=True, label="Control Panel", horizontal_scrollbar=True):
                dpg.add_text("Chat Backend", color=(200,200,255))
                dpg.add_radio_button(tag="chat_mode_radio", items=["ollama", "llama"], default_value=runtime.runtime_chat_mode, horizontal=True, callback=on_chat_mode_changed)

                dpg.add_text(f"Ollama URL: {runtime.settings.ollama.base_url}", color=(150,150,200), wrap=270)
                dpg.add_text(f"LLaMA URL: {runtime.settings.llama.base_url}", color=(150,150,200), wrap=270)

                with dpg.group(horizontal=True):
                    dpg.add_combo(tag="chat_model_combo", label="Model", width=-50, default_value=runtime.runtime_chat_params.get("model", ""), callback=lambda s,a: dpg.set_value("chat_model_hidden", a))
                    dpg.add_button(label="↻", tag="refresh_models_btn", callback=lambda: refresh_models_list(), width=40)
                dpg.add_input_text(tag="chat_model_hidden", default_value=runtime.runtime_chat_params.get("model", ""), show=False)

                dpg.add_input_float(tag="chat_temp", label="Temperature", default_value=runtime.runtime_chat_params.get("temperature", 0.7), step=0.05, min_value=0.0, max_value=2.0)
                dpg.add_input_int(tag="chat_max_tokens", label="Max tokens", default_value=runtime.runtime_chat_params.get("max_tokens", 2048), step=256, min_value=1)
                dpg.add_input_int(tag="chat_timeout", label="Timeout (s)", default_value=runtime.runtime_chat_params.get("timeout", 120), step=10, min_value=10)
                dpg.add_button(label="Apply Chat Settings", callback=apply_chat_from_ui)
                dpg.add_spacer(height=10)

                dpg.add_text("Embedding Backend", color=(200,255,200))
                dpg.add_radio_button(tag="embed_mode_radio", items=["ollama", "llama"], default_value=runtime.runtime_embed_mode, horizontal=True, callback=on_embed_mode_changed)
                dpg.add_text(f"Ollama URL: {runtime.settings.ollama.base_url}", color=(150,150,200), wrap=270)
                dpg.add_text(f"Embedding model: {runtime.settings.ollama.embedding_model}", color=(150,150,200), wrap=270)
                dpg.add_button(label="Apply Embedding Settings", callback=apply_embed_from_ui)
                dpg.add_spacer(height=10)

                dpg.add_text("Character Appearance", color=(255,200,100))
                with dpg.group(horizontal=True):
                    dpg.add_combo(tag="character_combo", label="Appearance", width=-50, callback=on_character_selected)
                    dpg.add_button(label="Update", tag="update_char_list_btn", callback=lambda: refresh_character_list(), width=40)
                dpg.add_spacer(height=5)

                dpg.add_button(label="Reset to settings.yaml", callback=lambda: reset_to_yaml_defaults_and_update_ui())
                dpg.add_spacer(height=5)
                dpg.add_text("Note: Changing embedding model requires same vector dimension.", color=(200,150,100), wrap=270)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    apply_window_geometry()
    dpg.set_frame_callback(1, update_split_heights)

    refresh_character_list()
