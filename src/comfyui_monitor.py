#!/usr/bin/env python3
"""
ComfyUI Monitor Daemon for AI_EveryNyan — persistent connection, progress tracking, preview.

Provides a background daemon that maintains a WebSocket connection to ComfyUI,
tracks generation progress, receives preview frames, and exposes state for
the DearPyGui interface.

Integration with gui.py provides:
  - Progress bar near the "Reasoning..." line during ComfyUI generation
  - Live preview panel in the right control panel
  - Cancel button that interrupts generation and frees GPU memory

src/comfyui_monitor.py

Version:     0.4.1
Author:      pytraveler
Updated:     2026-05-05

Patch Notes v0.4.1 (by pytraveler):
  [+] Background daemon: persistent WebSocket to ComfyUI with auto-reconnect.
  [+] Configurable reconnect interval via settings.yaml (comfyui.daemon_check_interval).
  [+] Progress tracking: monitors generation progress from ComfyUI events.
  [+] Preview frames: receives binary preview data for DearPyGui texture updates.
  [+] Cancel with memory cleanup: POST /interrupt + POST /free to unload models.
  [+] GUI integration: state polling for gui.py progress bar and preview panel.
  [~] Refactored from standalone monitor (v0.4.0) into daemon class usable by main app.

Patch Notes v0.4.0 (by Soror L.'.L.):
  [FIX] Preview GUI: converted RGBA data to float list for dpg.set_value.
        Dynamic texture now correctly updates after strip of binary header.
  [+] Debug dump of preview frames continues.

Patch Notes v0.3.x … v0.1.0 (see previous)
"""

import os
import sys
import json
import struct
import threading
import time
import io
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, List

import websocket as _ws

from logger import logger


# ComfyUI binary event types (from comfy/protocol.py)
class BinaryEventTypes:
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    TEXT = 3
    PREVIEW_IMAGE_WITH_METADATA = 4


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


def _load_server_from_settings() -> str:
    try:
        if CONFIG_PATH.exists():
            import yaml
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = data.get("comfyui", {})
            if cfg and "server" in cfg:
                return cfg["server"]
    except Exception:
        pass
    return "127.0.0.1:8188"


class ComfyUIDaemon:
    """Background daemon maintaining persistent WebSocket connection to ComfyUI.

    - Connects on start, reconnects on failure after check_interval.
    - Receives progress events and preview binary frames.
    - Provides thread-safe state access for GUI polling.
    - Cancel & free memory via ComfyUI HTTP API.
    """

    FIXED_CLIENT_ID = "ai_everynyan"

    def __init__(self, server: str, check_interval: float = 5.0):
        self.server = server
        self.check_interval = check_interval
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._connected = False

        self._generating = False
        self._progress_value = 0
        self._progress_max = 1
        self._preview_bytes: Optional[bytes] = None
        self._current_prompt_id: Optional[str] = None
        self._preview_warned: bool = False  # one-shot warning when no previews arrive
        self._received_preview_this_gen: bool = False  # did we get any binary frame this generation?
        self._generation_done: bool = False  # set on executing(node=None), cleared on execution_start

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ComfyUIDaemon"
        )
        self._thread.start()
        logger.info(
            f"[ComfyUI Daemon] Started (server={self.server}, interval={self.check_interval}s)"
        )

    def stop(self):
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[ComfyUI Daemon] Stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._connect_and_listen()
            except Exception as e:
                logger.debug(f"[ComfyUI Daemon] Connection error: {e}")
            if self._stop_event.is_set():
                break
            logger.debug(
                f"[ComfyUI Daemon] Reconnecting in {self.check_interval}s..."
            )
            self._stop_event.wait(self.check_interval)

    def _connect_and_listen(self):
        ws = None
        last_msg_time = time.monotonic()
        try:
            ws = _ws.WebSocket()
            ws.connect(
                f"ws://{self.server}/ws?clientId={self.FIXED_CLIENT_ID}",
                timeout=30,
            )
            self._ws = ws
            last_msg_time = time.monotonic()
            with self._lock:
                self._connected = True
            logger.info(f"[ComfyUI Daemon] Connected to ws://{self.server}")

            # Check if ComfyUI is configured for preview output
            preview_msg = self.check_preview_config()
            if preview_msg:
                logger.warning(f"[ComfyUI Daemon] PREVIEW WARNING: {preview_msg}")

            while not self._stop_event.is_set():
                try:
                    ws.settimeout(10.0)
                    out = ws.recv()
                    last_msg_time = time.monotonic()
                    self._process_message(out)
                except _ws.WebSocketTimeoutException:
                    # Liveness check: if no message for 90s, reconnect
                    if time.monotonic() - last_msg_time > 90:
                        logger.info("[ComfyUI Daemon] No messages for 90s, reconnecting")
                        break
                    continue
                except _ws.WebSocketConnectionClosedException:
                    logger.info("[ComfyUI Daemon] WS closed by server, reconnecting")
                    break
                except Exception as e:
                    if self._stop_event.is_set():
                        break
                    logger.info(f"[ComfyUI Daemon] WS recv error: {e}")
                    break
        except Exception as e:
            logger.debug(f"[ComfyUI Daemon] Connect failed: {e}")
        finally:
            with self._lock:
                self._connected = False
            self._ws = None
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass

    def _process_message(self, msg):
        if isinstance(msg, str):
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                return
            msg_type = data.get("type")
            msg_data = data.get("data", {})
            if msg_type == "progress":
                value = msg_data.get("value", 0)
                max_val = msg_data.get("max", 1)
                logger.debug(f"[ComfyUI Daemon] WS progress: {value}/{max_val}")
                with self._lock:
                    # Guard: ComfyUI threads can race, sending progress AFTER
                    # executing(node=None).  Don't re-activate generating once
                    # the current prompt has been marked done.
                    if not self._generation_done:
                        self._generating = True
                    self._progress_value = value
                    self._progress_max = max_val
            elif msg_type == "executing":
                node = msg_data.get("node")
                pid = msg_data.get("prompt_id", "?")
                logger.debug(f"[ComfyUI Daemon] WS executing: node={node} prompt_id={pid}")
                if node is None:
                    with self._lock:
                        # Ignore stale completion events from previous/cancelled generations
                        if self._current_prompt_id is not None and pid != self._current_prompt_id:
                            logger.debug(
                                f"[ComfyUI Daemon] Ignoring stale executing_done for "
                                f"prompt_id={pid} (current={self._current_prompt_id})"
                            )
                            return
                        had_preview = self._received_preview_this_gen
                        self._generating = False
                        self._generation_done = True
                        self._current_prompt_id = None
                    logger.info(f"[ComfyUI Daemon] Generation ended (prompt_id={pid}), had_preview={had_preview}")
                    if not had_preview and not self._preview_warned:
                        self._preview_warned = True
                        logger.warning(
                            "[ComfyUI Daemon] Generation completed but NO preview frames were received. "
                            "ComfyUI needs --preview-method auto to send live previews. "
                            "Restart ComfyUI with: python main.py --preview-method auto"
                        )
            elif msg_type == "execution_start":
                pid = msg_data.get("prompt_id", "?")
                logger.info(f"[ComfyUI Daemon] WS execution_start: prompt_id={pid}")
                with self._lock:
                    self._generating = True
                    self._generation_done = False
                    self._current_prompt_id = pid
                    self._preview_warned = False
                    self._received_preview_this_gen = False
            elif msg_type == "execution_error":
                logger.warning(f"[ComfyUI Daemon] WS execution_error: {msg_data}")
                with self._lock:
                    self._generating = False
                    self._generation_done = True
            elif msg_type == "status":
                queue_remaining = msg_data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                logger.debug(f"[ComfyUI Daemon] WS status: queue_remaining={queue_remaining}")
            else:
                logger.debug(f"[ComfyUI Daemon] WS unknown text event: type={msg_type}")
        elif isinstance(msg, bytes):
            self._handle_binary_frame(msg)

    def _handle_binary_frame(self, data: bytes):
        """Parse ComfyUI binary WebSocket frame.

        Format (from ComfyUI protocol.py / server.py):
          Bytes 0-3: event type as big-endian uint32
          Event 1 (PREVIEW_IMAGE):        [4B type=1][4B format_code][image bytes]
          Event 4 (PREVIEW_IMAGE_WITH_METADATA): [4B type=4][4B json_len][json][image bytes]
        """
        if len(data) < 8:
            logger.debug(f"[ComfyUI Daemon] Binary frame too short ({len(data)} bytes), ignoring")
            return

        event_type = struct.unpack(">I", data[:4])[0]
        logger.debug(
            f"[ComfyUI Daemon] WS binary frame: {len(data)} bytes, "
            f"event_type={event_type}, header_hex={data[:16].hex()}"
        )

        if event_type == BinaryEventTypes.PREVIEW_IMAGE:
            # [4B event=1][4B format: 1=JPEG, 2=PNG][image bytes]
            with self._lock:
                self._preview_bytes = data
                self._received_preview_this_gen = True

        elif event_type == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
            # [4B event=4][4B json_length][json bytes][image bytes]
            with self._lock:
                self._preview_bytes = data
                self._received_preview_this_gen = True

        elif event_type == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            with self._lock:
                self._preview_bytes = data
                self._received_preview_this_gen = True

        else:
            logger.debug(
                f"[ComfyUI Daemon] Unknown binary event type {event_type}, "
                f"storing raw ({len(data)} bytes)"
            )
            with self._lock:
                self._preview_bytes = data

    def _health_check(self) -> bool:
        try:
            req = urllib.request.Request(f"http://{self.server}/")
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False

    def check_preview_config(self) -> Optional[str]:
        """Check ComfyUI system_stats for preview configuration.

        Returns a diagnostic message or None if previews appear enabled.
        ComfyUI needs --preview-method auto or --preview-method taesd to
        send binary preview frames via WebSocket.
        """
        try:
            req = urllib.request.Request(f"http://{self.server}/system_stats")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            argv = data.get("system", {}).get("argv", [])
            version = data.get("system", {}).get("comfyui_version", "unknown")

            has_preview_flag = any(
                "--preview-method" in arg for arg in argv
            )
            # Some ComfyUI configs use --enable-cors-header or config files
            # Check if preview-method is in any arg form
            preview_method = None
            for i, arg in enumerate(argv):
                if arg == "--preview-method" and i + 1 < len(argv):
                    preview_method = argv[i + 1]
                elif arg.startswith("--preview-method="):
                    preview_method = arg.split("=", 1)[1]

            if preview_method and preview_method.lower() not in ("none", "no"):
                return None  # Previews should be enabled

            msg = (
                f"ComfyUI v{version} is running WITHOUT --preview-method flag. "
                f"Live previews will NOT be sent via WebSocket. "
                f"Restart ComfyUI with: python main.py --preview-method auto"
            )
            if not has_preview_flag:
                logger.warning(f"[ComfyUI Daemon] {msg}")
            return msg

        except Exception as e:
            logger.debug(f"[ComfyUI Daemon] Could not check preview config: {e}")
            return None

    def check_queue_http(self) -> dict:
        """Poll ComfyUI /queue via HTTP."""
        try:
            req = urllib.request.Request(f"http://{self.server}/queue")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}

    def get_state(self) -> dict:
        with self._lock:
            ws_state = {
                "connected": self._connected,
                "generating": self._generating,
                "progress_value": self._progress_value,
                "progress_max": self._progress_max,
                "preview_bytes": self._preview_bytes,
                "prompt_id": self._current_prompt_id,
            }
        return ws_state

    def consume_preview(self):
        """Clear preview bytes after the GUI has successfully processed them."""
        with self._lock:
            self._preview_bytes = None

    def cancel_and_free(self):
        """Interrupt current ComfyUI execution and free GPU memory."""
        logger.info("[ComfyUI Daemon] Cancel & free requested")
        try:
            req = urllib.request.Request(
                f"http://{self.server}/interrupt",
                data=b"{}",
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            urllib.request.urlopen(req, timeout=10)
            logger.info("[ComfyUI Daemon] Interrupt sent")
        except Exception as e:
            logger.debug(f"[ComfyUI Daemon] Interrupt failed: {e}")

        time.sleep(0.5)

        try:
            body = json.dumps(
                {"unload_models": True, "free_memory": True}
            ).encode()
            req = urllib.request.Request(
                f"http://{self.server}/free",
                data=body,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            urllib.request.urlopen(req, timeout=10)
            logger.info("[ComfyUI Daemon] Models unloaded, memory freed")
        except Exception as e:
            logger.debug(f"[ComfyUI Daemon] Free failed: {e}")

        with self._lock:
            self._generating = False
            self._generation_done = True
            self._current_prompt_id = None
            self._progress_value = 0
            self._progress_max = 1
            self._preview_bytes = None


def strip_preview_header(data: bytes) -> bytes:
    """Strip ComfyUI binary frame header to get raw image bytes.

    ComfyUI binary frame formats:
      Event 1 (PREVIEW_IMAGE):        [4B type=1][4B format_code][image bytes]
      Event 4 (PREVIEW_IMAGE_WITH_METADATA): [4B type=4][4B json_len][json][image bytes]

    Fallback: search for JPEG/PNG magic bytes if header parsing fails.
    """
    if len(data) < 8:
        return data

    event_type = struct.unpack(">I", data[:4])[0]

    if event_type == BinaryEventTypes.PREVIEW_IMAGE:
        # Skip 8-byte header (4B event type + 4B format code)
        return data[8:]

    elif event_type == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
        # Skip 4B event type, read JSON metadata length, skip JSON, rest is image
        json_len = struct.unpack(">I", data[4:8])[0]
        image_start = 8 + json_len
        if image_start < len(data):
            return data[image_start:]
        return data[8:]

    elif event_type == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
        return data[8:]

    # Fallback: search for JPEG/PNG magic bytes (in case format is unexpected)
    for magic in (b"\xff\xd8", b"\x89PNG"):
        idx = data.find(magic)
        if idx != -1 and idx < 64:  # magic should be near the start
            return data[idx:]

    # Last resort: skip 8 bytes (minimum ComfyUI header)
    return data[8:]


def rgba_to_float_list(data: bytes) -> List[float]:
    return [b / 255.0 for b in data]


def make_square_preview(img_data: bytes, size: int = 224) -> Optional[bytes]:
    """Convert raw preview bytes into RGBA square bytes for DearPyGui texture.

    Handles ComfyUI binary frame formats:
      Event 1: [4B type=1][4B format_code][image bytes]
      Event 4: [4B type=4][4B json_len][json][image bytes]
    """
    try:
        from PIL import Image as _PILImage

        # Decode event type from header
        event_type_str = "unknown"
        if len(img_data) >= 4:
            evt = struct.unpack(">I", img_data[:4])[0]
            event_type_str = {1: "PREVIEW_IMAGE", 2: "UNENCODED", 3: "TEXT", 4: "PREVIEW_WITH_META"}.get(evt, f"raw({evt})")

        logger.debug(
            f"[ComfyUI Monitor] make_square_preview: {len(img_data)} bytes, "
            f"event={event_type_str}"
        )
        clean = strip_preview_header(img_data)
        img = _PILImage.open(io.BytesIO(clean))
        img.thumbnail((size, size), _PILImage.LANCZOS)
        bg = _PILImage.new("RGBA", (size, size), (60, 60, 80, 255))
        offset = ((size - img.width) // 2, (size - img.height) // 2)
        bg.paste(img, offset)
        return bg.tobytes("raw", "RGBA")
    except Exception as e:
        logger.warning(f"[ComfyUI Monitor] make_square_preview failed ({len(img_data)} bytes): {e}")
        return None


# ---------------------------------------------------------------------------
# Standalone mode (for testing without the main app)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import dearpygui.dearpygui as dpg
    from PIL import Image

    server = os.environ.get("COMFYUI_SERVER", _load_server_from_settings())
    print(f"[ComfyUI Monitor] Standalone mode — server: {server}", file=sys.stderr)
    daemon = ComfyUIDaemon(server=server, check_interval=10.0)
    daemon.start()

    dpg.create_context()
    dpg.add_texture_registry(tag="tex_registry", show=False)

    placeholder = Image.new("RGBA", (224, 224), (60, 60, 80, 255))
    floats = rgba_to_float_list(placeholder.tobytes("raw", "RGBA"))
    dpg.add_dynamic_texture(
        224, 224, floats, tag="preview_tex", parent="tex_registry"
    )

    with dpg.theme() as dark_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 35))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220))

    dpg.bind_theme(dark_theme)
    dpg.create_viewport(
        title="ComfyUI Monitor (standalone)", width=500, height=400
    )
    dpg.setup_dearpygui()

    with dpg.window(tag="main_window", no_title_bar=True):
        dpg.set_primary_window("main_window", True)
        dpg.add_text("ComfyUI Monitor (standalone)", color=(255, 200, 100))
        dpg.add_progress_bar(
            tag="progress_bar", width=-1, height=20, default_value=0.0
        )
        dpg.add_image(
            "preview_tex", tag="preview_image", width=224, height=224
        )
        dpg.add_button(
            label="Cancel & Free",
            callback=lambda: daemon.cancel_and_free(),
        )
        dpg.add_text("", tag="status_text")

    _sa_last = 0.0

    def _sa_tick():
        global _sa_last
        now = time.perf_counter()
        if now - _sa_last >= 0.1:
            _sa_last = now
            state = daemon.get_state()
            if state["generating"]:
                frac = state["progress_value"] / max(state["progress_max"], 1)
                dpg.configure_item(
                    "progress_bar",
                    default_value=frac,
                    overlay=f"{state['progress_value']}/{state['progress_max']}",
                )
            else:
                dpg.configure_item(
                    "progress_bar", default_value=0.0, overlay="idle"
                )
            conn = "Connected" if state["connected"] else "Disconnected"
            gen = "Generating" if state["generating"] else "Idle"
            dpg.set_value("status_text", f"{conn} | {gen}")
            if state["preview_bytes"]:
                rgba = make_square_preview(state["preview_bytes"], 224)
                if rgba:
                    dpg.set_value("preview_tex", rgba_to_float_list(rgba))
        dpg.set_frame_callback(dpg.get_frame_count() + 1, _sa_tick)

    dpg.show_viewport()
    dpg.set_frame_callback(1, _sa_tick)
    dpg.start_dearpygui()
    daemon.stop()
    dpg.destroy_context()
