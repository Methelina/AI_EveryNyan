"""
MCP server providing image generation tools via ComfyUI API.
Exposes two tools: generate_image and list_workflows.

Queues a ComfyUI workflow, waits for completion via websocket,
collects output images, saves them to disk and returns file paths.

Reads settings from config/settings.yaml (ComfyUISettings),
with environment variable overrides.

/tools/mcp/tool_comfyui.py

Version:     0.1.0
Author:      Soror L.'.L.'.
Updated:     2026-05-03

Patch Notes v0.1.0 (by pytraveler):
  [+] Initial implementation: generate_image and list_workflows MCP tools.
  [+] ComfyUI API integration: queue workflow, websocket progress, image collection.
  [+] Config via settings.yaml (ComfyUISettings) with env var overrides.

Patch Notes v0.1.0 (by Soror L.'.L.):
  [*] Refactored config loading: reads settings.yaml via ComfyUISettings model,
      falls back to env vars, then to hardcoded defaults.
  [*] Added ComfyUISettings to src/config.py AppSettings.
"""

import os
import sys
import json
import uuid
import urllib.request
import urllib.parse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_LOG = LOG_DIR / "mcp_comfyui.log"

mcp = FastMCP("comfyui")


def _log(msg: str):
    ts = datetime.now().isoformat()
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")


def _load_settings() -> dict:
    defaults = {
        "server": "127.0.0.1:8084",
        "workflow_dir": str(REPO_ROOT / "workflows"),
        "output_dir": str(REPO_ROOT / "data" / "comfyui_output"),
        "http_timeout": 120,
    }
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            comfyui_cfg = data.get("comfyui", {})
            if comfyui_cfg:
                wd = comfyui_cfg.get("workflow_dir", defaults["workflow_dir"])
                od = comfyui_cfg.get("output_dir", defaults["output_dir"])
                if not Path(wd).is_absolute():
                    wd = str(REPO_ROOT / wd)
                if not Path(od).is_absolute():
                    od = str(REPO_ROOT / od)
                return {
                    "server": comfyui_cfg.get("server", defaults["server"]),
                    "workflow_dir": wd,
                    "output_dir": od,
                    "http_timeout": comfyui_cfg.get("http_timeout", defaults["http_timeout"]),
                }
    except Exception as e:
        print(f"[MCP] comfyui: Failed to load settings.yaml: {e}", file=sys.stderr)
    return defaults


_yaml_settings = _load_settings()

COMFYUI_SERVER = os.environ.get("COMFYUI_SERVER", _yaml_settings["server"])
WORKFLOW_DIR = Path(os.environ.get("COMFYUI_WORKFLOW_DIR", _yaml_settings["workflow_dir"]))
OUTPUT_DIR = Path(os.environ.get("COMFYUI_OUTPUT_DIR", _yaml_settings["output_dir"]))
HTTP_TIMEOUT = int(os.environ.get("COMFYUI_HTTP_TIMEOUT", str(_yaml_settings["http_timeout"])))


def _queue_prompt(prompt: dict, client_id: str) -> dict:
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        f"http://{COMFYUI_SERVER}/prompt", data=data
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def _get_history(prompt_id: str) -> dict:
    url = f"http://{COMFYUI_SERVER}/history/{prompt_id}"
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def _get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    data = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    url = f"http://{COMFYUI_SERVER}/view?{data}"
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


def _wait_for_completion(ws, prompt_id: str) -> None:
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
        elif isinstance(out, bytes):
            continue


def _collect_output_images(prompt_id: str) -> Dict[str, List[bytes]]:
    history = _get_history(prompt_id)
    if prompt_id not in history:
        return {}
    outputs = history[prompt_id].get("outputs", {})
    result: Dict[str, List[bytes]] = {}
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            images_data = []
            for img in node_output["images"]:
                img_bytes = _get_image(img["filename"], img["subfolder"], img["type"])
                images_data.append(img_bytes)
            result[node_id] = images_data
    return result


def _find_prompt_node(workflow: dict, prompt_type: str) -> Optional[str]:
    """Find the node ID for a positive or negative prompt node."""
    title_keywords = {
        "positive": ("pos", "positive"),
        "negative": ("neg", "negative"),
    }
    keywords = title_keywords[prompt_type]

    # --- Strategy 1: _meta.title match ---
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title", "").lower()
        if any(kw in title for kw in keywords):
            if "text" in node.get("inputs", {}):
                _log(f"_find_prompt_node({prompt_type}): matched by _meta.title on node {node_id} ('{title}')")
                return node_id

    # --- Strategy 2: class_type + wiring ---
    clip_nodes: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "CLIPTextEncode":
            if "text" in node.get("inputs", {}):
                clip_nodes.append(node_id)

    # 2a: trace through guider / sampler "positive"/"negative" inputs
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        ref = inputs.get(prompt_type)  # "positive" or "negative"
        if isinstance(ref, list) and len(ref) == 2:
            referenced_node = str(ref[0])
            if referenced_node in clip_nodes:
                _log(f"_find_prompt_node({prompt_type}): matched by wiring from node {node_id} -> {referenced_node}")
                return referenced_node

    # 2b: exactly 2 CLIPTextEncode nodes — assume first=pos, second=neg
    if len(clip_nodes) == 2:
        sorted_nodes = sorted(clip_nodes)
        chosen = sorted_nodes[0] if prompt_type == "positive" else sorted_nodes[1]
        _log(f"_find_prompt_node({prompt_type}): matched by positional order -> {chosen}")
        return chosen

    _log(f"_find_prompt_node({prompt_type}): no matching node found")
    return None


def _apply_workflow_overrides(
    workflow: dict,
    positive_prompt: Optional[str],
    negative_prompt: Optional[str],
    node_overrides: Optional[Dict[str, Dict[str, Any]]],
) -> dict:
    if positive_prompt is not None:
        nid = _find_prompt_node(workflow, "positive")
        if nid:
            workflow[nid]["inputs"]["text"] = positive_prompt
        else:
            _log("WARNING: could not find positive prompt node in workflow")
    if negative_prompt is not None:
        nid = _find_prompt_node(workflow, "negative")
        if nid:
            workflow[nid]["inputs"]["text"] = negative_prompt
        else:
            _log("WARNING: could not find negative prompt node in workflow")
    if node_overrides:
        for node_id, fields in node_overrides.items():
            if node_id in workflow:
                for key, value in fields.items():
                    if "inputs" in workflow[node_id]:
                        workflow[node_id]["inputs"][key] = value
    return workflow


def _save_images(images: Dict[str, List[bytes]], prompt_id: str) -> List[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for node_id, img_list in images.items():
        for idx, img_bytes in enumerate(img_list):
            filename = f"{node_id}_{prompt_id}_{idx}.png"
            filepath = OUTPUT_DIR / filename
            filepath.write_bytes(img_bytes)
            saved.append(str(filepath))
    return saved


# ============================================================================
# MCP TOOLS
# ============================================================================


@mcp.tool()
async def generate_image(
    positive_prompt: str,
    negative_prompt: str = "",
    workflow_file: str = "",
    node_overrides: str = "",
) -> str:
    """
    This tool requested when user ask for photo, image generation, selfie or make an art. 
    Generate an image using ComfyUI by submitting a workflow with the given prompts.
    Returns the local file paths of the generated images.

    Parameters:
    - positive_prompt: The main image description (e.g. 'masterpiece, best quality, a girl in a garden').
    - negative_prompt: What to avoid (e.g. 'lowres, bad anatomy, watermark'). Optional.
    - workflow_file: Filename of a workflow JSON in the workflows directory. If empty, uses 'default.json'.
    - node_overrides: JSON string with per-node input overrides, e.g. '{"1":{"ckpt_name":"model.safetensors"}}'. Optional.
    """
    import websocket as _ws

    _log(f"generate_image called: positive={positive_prompt!r}, negative={negative_prompt!r}")

    wf_name = workflow_file.strip() or "default.json"
    wf_path = WORKFLOW_DIR / wf_name
    if not wf_path.exists():
        return f"Error: Workflow file not found: {wf_path}"

    try:
        with open(wf_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        _log(f"Failed to load workflow: {e}")
        return f"Error: Failed to load workflow: {e}"

    overrides = {}
    if node_overrides.strip():
        try:
            overrides = json.loads(node_overrides)
        except json.JSONDecodeError as e:
            return f"Error: Invalid node_overrides JSON: {e}"

    workflow = _apply_workflow_overrides(
        workflow,
        positive_prompt if positive_prompt else None,
        negative_prompt if negative_prompt else None,
        overrides if overrides else None,
    )

    client_id = str(uuid.uuid4())

    try:
        result = _queue_prompt(workflow, client_id)
    except Exception as e:
        _log(f"Failed to queue prompt: {e}")
        return f"Error: Failed to queue prompt to ComfyUI at {COMFYUI_SERVER}: {e}"

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        _log(f"No prompt_id in response: {result}")
        return f"Error: ComfyUI did not return a prompt_id. Response: {result}"

    _log(f"Queued prompt_id={prompt_id}, waiting for completion...")

    try:
        ws = _ws.WebSocket()
        ws.connect(f"ws://{COMFYUI_SERVER}/ws?clientId={client_id}", timeout=HTTP_TIMEOUT)
        _wait_for_completion(ws, prompt_id)
        ws.close()
    except Exception as e:
        _log(f"WebSocket error: {e}")
        return f"Error: WebSocket communication with ComfyUI failed: {e}"

    images = _collect_output_images(prompt_id)
    if not images:
        return f"ComfyUI completed (prompt_id={prompt_id}) but produced no output images."

    saved = _save_images(images, prompt_id)
    _log(f"Generated {len(saved)} image(s), prompt_id={prompt_id}")

    lines = [f"Generated {len(saved)} image(s) (prompt_id={prompt_id}):"]
    for path in saved:
        lines.append(f"  {path}")
    return "\n".join(lines)


@mcp.tool()
async def list_workflows() -> str:
    """
    List available ComfyUI workflow JSON files in the workflows directory.
    Returns filenames that can be passed to generate_image as the workflow_file parameter.
    """
    if not WORKFLOW_DIR.exists():
        return f"Workflow directory does not exist: {WORKFLOW_DIR}"

    files = sorted(WORKFLOW_DIR.glob("*.json"))
    if not files:
        return f"No .json workflow files found in {WORKFLOW_DIR}"

    lines = [f"Available workflows in {WORKFLOW_DIR}:"]
    for f in files:
        size_kb = f.stat().st_size / 1024
        lines.append(f"  {f.name} ({size_kb:.1f} KB)")
    lines.append(f"\nTotal: {len(files)} workflow(s)")
    return "\n".join(lines)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print(f"[MCP] comfyui: Starting MCP server (tool_comfyui.py)", file=sys.stderr)
    print(f"[MCP] comfyui: ComfyUI server: {COMFYUI_SERVER}", file=sys.stderr)
    print(f"[MCP] comfyui: Workflow dir: {WORKFLOW_DIR}", file=sys.stderr)
    print(f"[MCP] comfyui: Output dir: {OUTPUT_DIR}", file=sys.stderr)
    print(f"[MCP] comfyui: HTTP timeout: {HTTP_TIMEOUT}s", file=sys.stderr)
    mcp.run(transport="stdio")
