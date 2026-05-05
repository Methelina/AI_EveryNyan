#!/usr/bin/env python3
"""
MCP server providing vision-language tasks using VL models.
Exposes one tool: describe_image – accepts local path or URL, returns structured JSON description.
Uses PIL for validation, security (PNG conversion, resize), reads config from settings.yaml.

Auto-detects whether the active chat LLM (Ollama or llama.cpp) supports vision
and preferentially uses it when vision.prefer_chat_model is enabled.

/tools/mcp/tool_vision.py

Version:     0.3.5
Author:      Soror L.'.L.'.
Updated:     2026-05-05

Patch Notes v0.3.5 (by pytraveler):
  [FIX] Added docstring to describe_image tool so the LLM can understand
        what the tool does, when to call it, and what each parameter means.

Patch Notes v0.3.4 (by pytraveler):
  [+] Auto-detect vision capability of the active chat LLM (Ollama / llama.cpp).
  [+] If chat model supports vision and prefer_chat_model is enabled, use it instead
      of the dedicated vision model.
  [+] New config key: vision.prefer_chat_model (default: true).
  [+] OpenAI-compatible multimodal endpoint support for llama.cpp backends.
  [+] Fallback: if chat model vision call fails, retry with dedicated vision model.

Patch Notes v0.3.3:
  [FIX] Added console debug output when run standalone to verify config loading.
  [+] Now prints loaded configuration to stderr on startup.

Patch Notes v0.3.2:
  [FIX] Corrected REPO_ROOT path: now uses 3 parents instead of 4.
  [FIX] Ensures logs directory exists before writing debug log.
"""

import os
import sys
import base64
import io
import json
import yaml
import httpx
from pathlib import Path
from datetime import datetime

from fastmcp import FastMCP
from PIL import Image

# ============================================================================
# PATH RESOLUTION – read config from project root
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # <repo>/
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"

# ============================================================================
# LOGGING
# ============================================================================
DEBUG_LOG = REPO_ROOT / "logs" / "mcp_vision.log"
DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)

def log_debug(msg: str):
    timestamp = datetime.now().isoformat()
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {msg}\n")

def report_to_console(msg: str):
    print(f"[MCP] vision: {msg}", file=sys.stderr, flush=True)
    log_debug(msg)

# ============================================================================
# KNOWN VISION MODEL FAMILIES (Ollama details.families)
# ============================================================================
VISION_FAMILIES = {
    "clip", "llava", "llava-llama3", "mllama", "minicpm-v",
    "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "pixtral",
    "phi3-vision", "fuyu", "cogvlm", "internvl2",
}

# ============================================================================
# LOAD CONFIGURATION FROM YAML
# ============================================================================
def load_full_config() -> dict:
    default_vision = {
        "enabled": True,
        "model": "qwen3-vl:235b-cloud",
        "prefer_chat_model": True,
        "default_prompt": "What do you see in this image? Describe in detail.",
        "prompt_mode": "structured_json",
        "max_image_size_mb": 20,
        "resize_size": 1024,
    }

    try:
        report_to_console(f"Loading config from: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        vision = {**default_vision, **data.get("vision", {})}

        chat_mode = data.get("chat_mode", "ollama")
        ollama_cfg = data.get("ollama", {})
        llama_cfg = data.get("llama", {})

        config = {
            "vision": vision,
            "chat_mode": chat_mode,
            "ollama_base_url": ollama_cfg.get("base_url", "http://localhost:11434/v1"),
            "ollama_chat_model": ollama_cfg.get("chat_model", "qwen2.5:7b"),
            "ollama_api_key": ollama_cfg.get("api_key", "ollama"),
            "llama_base_url": llama_cfg.get("base_url", "http://localhost:8088/v1"),
            "llama_chat_model": llama_cfg.get("chat_model", ""),
            "llama_api_key": llama_cfg.get("api_key", ""),
        }

        report_to_console(
            f"Config: vision_model={vision['model']}, "
            f"prefer_chat={vision['prefer_chat_model']}, "
            f"chat_mode={chat_mode}"
        )
        return config

    except Exception as e:
        (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
        log_debug(f"Failed to load config: {e}, using defaults")
        report_to_console(f"WARN: Failed to load config: {e}, using defaults")
        return {
            "vision": default_vision,
            "chat_mode": "ollama",
            "ollama_base_url": "http://localhost:11434/v1",
            "ollama_chat_model": "qwen2.5:7b",
            "ollama_api_key": "ollama",
            "llama_base_url": "http://localhost:8088/v1",
            "llama_chat_model": "",
            "llama_api_key": "",
        }


full_config = load_full_config()

VISION_CFG = full_config["vision"]
CHAT_MODE = full_config["chat_mode"]
OLLAMA_BASE_URL = full_config["ollama_base_url"].replace("/v1", "")
LLAMA_BASE_URL = full_config["llama_base_url"]

VISION_MODEL = VISION_CFG["model"]
PREFER_CHAT_MODEL = VISION_CFG["prefer_chat_model"]
PROMPT_MODE = VISION_CFG["prompt_mode"]
DEFAULT_PROMPT = VISION_CFG["default_prompt"]
MAX_IMAGE_SIZE_MB = VISION_CFG["max_image_size_mb"]
RESIZE_SIZE = VISION_CFG["resize_size"]

# ============================================================================
# VISION CAPABILITY DETECTION
# ============================================================================

def _strip_v1(url: str) -> str:
    if url.endswith("/v1"):
        return url[:-3]
    return url


def detect_ollama_vision(base_url: str, model_name: str) -> bool:
    try:
        url = _strip_v1(base_url) + "/api/show"
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json={"name": model_name})
            resp.raise_for_status()
            data = resp.json()

        details = data.get("details", {})
        families = details.get("families", [])
        family = details.get("family", "")

        all_families = set(f.lower() for f in families)
        if family:
            all_families.add(family.lower())

        for vf in VISION_FAMILIES:
            if vf in all_families:
                report_to_console(
                    f"Ollama model '{model_name}' IS vision-capable "
                    f"(matched family '{vf}', all families: {all_families})"
                )
                return True

        model_info = data.get("model_info", {})
        for key in model_info:
            if "clip" in key.lower() or "vision" in key.lower() or "mmproj" in key.lower():
                report_to_console(
                    f"Ollama model '{model_name}' IS vision-capable "
                    f"(matched model_info key '{key}')"
                )
                return True

        report_to_console(
            f"Ollama model '{model_name}' is NOT vision-capable "
            f"(families: {all_families})"
        )
        return False

    except Exception as e:
        report_to_console(f"Ollama vision detection failed for '{model_name}': {e}")
        return False


def _find_modalities(obj) -> dict | None:
    """Recursively search for the 'modalities' dict anywhere in the JSON tree."""
    if isinstance(obj, dict):
        if "modalities" in obj and isinstance(obj["modalities"], dict):
            return obj["modalities"]
        for v in obj.values():
            result = _find_modalities(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_modalities(item)
            if result is not None:
                return result
    return None


def _find_keys_recursive(obj, target_keys: set) -> str | None:
    """Recursively search for any of target_keys in the JSON tree. Returns the found key or None."""
    if isinstance(obj, dict):
        for k in obj:
            if k in target_keys:
                return k
        for v in obj.values():
            result = _find_keys_recursive(v, target_keys)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_keys_recursive(item, target_keys)
            if result is not None:
                return result
    return None


def detect_llama_vision(base_url: str) -> bool:
    """
    Detect vision capability of a llama.cpp server.

    llama-server /props returns structured JSON with 'modalities' dict
    somewhere in the tree, e.g.:
        "modalities": {"vision": true, "audio": false}

    The 'modalities' key may appear at any nesting level depending on
    the llama.cpp build version, so we search recursively.
    We check the exact boolean value of modalities.vision.
    """
    try:
        props_url = _strip_v1(base_url) + "/props"
        with httpx.Client(timeout=10) as client:
            resp = client.get(props_url)
            if resp.status_code == 404:
                # Older llama.cpp may not have /props — try /health
                health_url = _strip_v1(base_url) + "/health"
                resp = client.get(health_url)
            resp.raise_for_status()

            data = resp.json()

        # Primary check: find 'modalities' dict anywhere in the tree
        modalities = _find_modalities(data)
        if isinstance(modalities, dict):
            vision_flag = modalities.get("vision")
            if isinstance(vision_flag, bool):
                if vision_flag:
                    report_to_console(
                        "llama.cpp server IS vision-capable "
                        f"(modalities.vision={vision_flag})"
                    )
                    return True
                else:
                    report_to_console(
                        "llama.cpp server is NOT vision-capable "
                        f"(modalities.vision={vision_flag})"
                    )
                    return False

        # Fallback: check for mmproj / clip / projector metadata keys
        # anywhere in the tree (some older builds expose these)
        found_key = _find_keys_recursive(data, {"mmproj", "clip", "projector"})
        if found_key:
            report_to_console(
                f"llama.cpp server IS vision-capable "
                f"(found key '{found_key}' in /props)"
            )
            return True

        report_to_console(
            "llama.cpp server is NOT vision-capable "
            "(no modalities.vision or projector metadata found)"
        )
        return False

    except Exception as e:
        report_to_console(f"llama.cpp vision detection failed: {e}")
        return False


def detect_vision_capability() -> dict | None:
    if not PREFER_CHAT_MODEL:
        report_to_console("prefer_chat_model is disabled, using dedicated vision model")
        return None

    if CHAT_MODE == "ollama":
        model_name = full_config["ollama_chat_model"]
        base_url = full_config["ollama_base_url"]
        if model_name and detect_ollama_vision(base_url, model_name):
            return {
                "backend": "ollama",
                "model": model_name,
                "base_url": base_url,
            }
    elif CHAT_MODE == "llama":
        model_name = full_config["llama_chat_model"]
        base_url = full_config["llama_base_url"]
        if model_name and detect_llama_vision(base_url):
            return {
                "backend": "llama",
                "model": model_name,
                "base_url": base_url,
            }

    report_to_console("Chat model does NOT support vision, falling back to dedicated vision model")
    return None


CHAT_VISION = detect_vision_capability()

if CHAT_VISION:
    ACTIVE_MODEL = CHAT_VISION["model"]
    ACTIVE_BACKEND = CHAT_VISION["backend"]
    ACTIVE_BASE_URL = CHAT_VISION["base_url"]
    report_to_console(
        f"ACTIVE: Using chat model '{ACTIVE_MODEL}' ({ACTIVE_BACKEND}) for vision"
    )
else:
    ACTIVE_MODEL = VISION_MODEL
    ACTIVE_BACKEND = "ollama"
    ACTIVE_BASE_URL = OLLAMA_BASE_URL
    report_to_console(
        f"ACTIVE: Using dedicated vision model '{ACTIVE_MODEL}' (ollama fallback)"
    )

mcp = FastMCP("vision")

# ============================================================================
# FULL STRUCTURED PROMPT (with value options, etc., + short description)
# ============================================================================

STRUCTURED_PROMPT = """You are a skilled AI visual analyst. Analyze the image thoroughly step by step.

Based on the content, output ONLY a single JSON object using one of the two schemas below (human or scene). NO extra text, NO markdown. Use "N/A" if information is not visible or not applicable. Use "etc." to indicate that other values not listed are also possible. For free text fields, provide a concise description in your own words. For measurements and sizes, you may use approximate values (e.g., "about 165 cm", "~70B", "medium build") when exact numbers are not clearly visible.

=== SCHEMA A: HUMAN / CHARACTER (use if any human or humanoid is the main subject) ===
{
  "subject_type": "human",
  "short_visual_description": "string (one or two sentences summarizing the whole image: who, what, where, mood – e.g., 'A young woman taking a selfie in a cozy room, smiling, wearing a black blouse, upper body visible.')",
  "age_estimate": "10yo|20yo|30yo|40yo|50yo|60yo+|N/A|etc.",
  "height_cm": "string (approximate, e.g., 'about 165 cm') or N/A",
  "race": "asian|caucasian|african|mongoloid|mixed|N/A|etc.",
  "gender": "female|male|non-binary|N/A|etc.",
  "face_shape": "oval|diamond|triangle|round|square|heart|oblong|N/A|etc.",
  "measurements_cm": {
    "bust": "string (approx or exact integer) or N/A",
    "waist": "string (approx or exact integer) or N/A",
    "hips": "string (approx or exact integer) or N/A",
    "high_hips": "string (approx or exact integer) or N/A",
    "back_waist": "string (approx or exact integer) or N/A",
    "front_waist": "string (approx or exact integer) or N/A",
    "inseam": "string (approx or exact integer) or N/A",
    "sleeve_length": "string (approx or exact integer) or N/A",
    "legs_length_3_4": "string (approx or exact integer) or N/A",
    "full_legs_length": "string (approx or exact integer) or N/A"
  },
  "breast_type": "almost flat|flat|tiny|small|childish|medium|large|N/A|etc.",
  "cap_size": "XS|S|M|L|XL|N/A|etc.",
  "constitution": "Slender|Childish|Athletic|Curvy|Petite|Plus-size|Stocky|Lean|Muscular|Soft|Average|Pear-shaped|Apple-shaped|Rectangular|Hourglass|Tall and Lean|Short and Stocky|Heavy-set|Ectomorphic|Mesomorphic|Endomorphic|N/A|etc.",
  "features": "string (free text: hair, skin, expression, scars, tattoos, etc.)",
  "attire_clothing": "string (free text: e.g., naked, underwear, swimsuit, casual, business, evening gown, costume, uniform, etc. Include upper/lower/footwear details or 'N/A')",
  "footwear": "barefoot|sneakers|heels|boots|sandals|loafers|none|N/A|etc.",
  "accessories": "string (free text: glasses, hat, jewelry, bag, watch, or 'N/A')",
  "pose": "standing|sitting|lying|walking|running|jumping|bending|kneeling|etc.|N/A",
  "body_parts_details": "string (free text: visible limbs, torso, back, specific features or 'N/A')",
  "erotical_assets": "none|suggestive|cleavage|cameltoe|nipple outline|erected nipples|nudity|N/A|etc.",
  "secondary_objects_props": "string (free text: items held, props, interactions, or 'N/A')",
  "environment": "string (free text: indoor/outdoor, room type, nature, urban, etc.)",
  "lighting": "string (free text: natural, artificial, studio, low key, high key, golden hour, etc.)",
  "composition": "string (free text: close-up, full body, mid shot, rule of thirds, leading lines, etc.)",
  "camera_specs": {
    "type": "DSLR|mirrorless|smartphone|webcam|action cam|film|unknown|N/A|etc.",
    "model": "string or N/A",
    "lens": "string (e.g., 50mm, 85mm macro) or N/A",
    "focal_length": "string (mm) or N/A",
    "aperture": "string (f number) or N/A",
    "bokeh": "present|absent|strong|weak|N/A",
    "iso": "string or N/A",
    "noise": "none|low|medium|high|N/A",
    "medium": "digital|film|N/A",
    "white_balance": "auto|custom|warm|cool|N/A",
    "gamma": "string or N/A"
  },
  "atmosphere": "string (free text: mood, emotions evoked, color grade, or 'N/A')",
  "style": "photorealistic|vintage|surreal|painting|3D|watercolor|anime|comic|editorial|street|candid|N/A|etc.",
  "custom_elements": "string (free text: brands, text, logos, cultural items, or 'N/A')",
  "extra_controlnet": "dwpose|depth|openpose|none|N/A|etc."
}

=== SCHEMA B: SCENE / LANDSCAPE / OBJECT (use if NO human is the main subject) ===
{
  "subject_type": "scene",
  "short_visual_description": "string (one or two sentences summarizing the whole scene: what, where, mood, key elements – e.g., 'A calm beach at sunset with gentle waves and orange sky.')",
  "scene_type": "landscape|cityscape|interior|still life|animal|object|abstract|N/A|etc.",
  "location": "string (free text: e.g., forest, beach, office, kitchen, street, or 'N/A')",
  "time_of_day": "dawn|day|dusk|night|indoor no natural light|N/A|etc.",
  "season": "spring|summer|autumn|winter|N/A|etc.",
  "weather": "sunny|cloudy|rainy|snowy|foggy|stormy|clear|N/A|etc.",
  "main_focal_point": "string (free text: central object or area of interest)",
  "environment_background": "string (free text: textures, colors, depth, seamless elements, or 'N/A')",
  "lighting": "string (free text: natural, window, artificial, neon, flash, ambient, etc.)",
  "composition": "string (free text: wide angle, telephoto, macro, panoramic, etc.)",
  "rule_of_thirds": "yes|no|partial|N/A",
  "camera_specs": {
    "camera_type": "DSLR|mirrorless|smartphone|webcam|action cam|film|unknown|N/A|etc.",
    "camera_model": "string or N/A",
    "lens": "string or N/A",
    "focal_length": "string or N/A",
    "aperture": "string or N/A",
    "bokeh": "present|absent|N/A",
    "iso": "string or N/A",
    "noise": "none|low|medium|high|N/A",
    "medium": "digital|film|N/A",
    "white_balance": "auto|custom|N/A",
    "gamma": "string or N/A"
  },
  "atmosphere": "string (free text: mood, color grade, evoked feeling, or 'N/A')",
  "style": "photorealistic|vintage|surreal|painting|3D|watercolor|anime|editorial|street|N/A|etc.",
  "secondary_objects": "string (free text: props, animals, vehicles, furniture, or 'N/A')",
  "custom_elements": "string (free text: brands, text, signs, cultural references, or 'N/A')",
  "extra_controlnet": "dwpose|depth|openpose|none|N/A|etc.",
  "event_action": "string (free text: what is happening, if anything, or 'N/A')"
}

IMPORTANT: 
- Use the exact string values from the lists above where applicable, but you are free to add other values using "etc." as a hint. For free-text fields (marked "string"), write a concise natural language description.
- For measurements and sizes, you may use approximate values (e.g., "about 165 cm", "~70B", "medium build") – you are not forced to give exact numbers.
- If a field is not visible or not relevant, use "N/A".
- Output ONLY the JSON object — no markdown, no extra text.
"""

# ============================================================================
# HELPERS
# ============================================================================

def normalize_image_to_png_base64(image_data: bytes) -> str:
    img = Image.open(io.BytesIO(image_data))
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
    else:
        img = img.convert('RGB')
    
    width, height = img.size
    max_dim = max(width, height)
    if max_dim > RESIZE_SIZE:
        ratio = RESIZE_SIZE / max_dim
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        report_to_console(f"Resized from {width}x{height} to {new_width}x{new_height}")
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    return base64.b64encode(png_bytes).decode('utf-8')

async def download_image_data(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"URL does not point to an image (Content-Type: {content_type})")
        if len(resp.content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit")
        return resp.content

def load_local_image_data(path: str) -> bytes:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    size = os.path.getsize(path)
    if size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit")
    with open(path, "rb") as f:
        return f.read()

# ============================================================================
# VL CALL DISPATCHERS
# ============================================================================

async def call_ollama_vl(model: str, base_url: str, prompt: str, image_base64: str) -> str:
    api_url = _strip_v1(base_url) + "/api/generate"
    async with httpx.AsyncClient(timeout=180) as client:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            }
        }
        response = await client.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()


async def call_openai_vl(model: str, base_url: str, prompt: str, image_base64: str, api_key: str = "") -> str:
    api_url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""


async def call_vision_model(model: str, backend: str, base_url: str, prompt: str, image_base64: str) -> str:
    if backend == "ollama":
        return await call_ollama_vl(model, base_url, prompt, image_base64)
    else:
        api_key = ""
        if backend == "llama":
            api_key = full_config.get("llama_api_key", "")
        return await call_openai_vl(model, base_url, prompt, image_base64, api_key)

# ============================================================================
# MCP TOOL
# ============================================================================
@mcp.tool()
async def describe_image(
    image_source: str,
    prompt: str = DEFAULT_PROMPT,
    is_url: bool = False
) -> str:
    """
    Analyze an image using a vision-language model and return a detailed description.

    Use this tool whenever the user sends an image, references an image file or URL,
    or asks to describe / analyze what is visible in a picture.
    For example: "What's in this image?", "Describe this photo", "Analyze this picture".

    The tool downloads the image (if URL) or reads it from disk (if local path),
    converts it to PNG, resizes if needed, and sends it to the configured VL model.

    In structured_json mode (default), returns a comprehensive JSON object describing
    the image content using one of two schemas:
      - Schema A (human/character): age, gender, body type, attire, pose, environment, etc.
      - Schema B (scene/object): scene type, location, lighting, composition, atmosphere, etc.
    In free_text mode, returns a natural language description based on the given prompt.

    Parameters:
    - image_source: Path to a local image file OR a URL pointing to an image.
    - prompt: Custom text prompt for the VL model (only used in free_text mode).
              Ignored in structured_json mode. Default: "What do you see in this image?"
    - is_url: Set to True if image_source is a URL, False (default) if it is a local file path.
    """
    report_to_console(
        f"describe_image called: source={image_source[:100]}, is_url={is_url}, "
        f"mode={PROMPT_MODE}, active_model={ACTIVE_MODEL} ({ACTIVE_BACKEND})"
    )
    try:
        if is_url:
            report_to_console(f"Downloading from URL: {image_source[:80]}")
            raw_bytes = await download_image_data(image_source)
        else:
            report_to_console(f"Reading local file: {image_source}")
            raw_bytes = load_local_image_data(image_source)

        report_to_console("Validating and converting image to PNG...")
        img_b64 = normalize_image_to_png_base64(raw_bytes)

        if PROMPT_MODE == "structured_json":
            final_prompt = STRUCTURED_PROMPT
            report_to_console("Using structured_json prompt (person/scene schema)")
        else:
            final_prompt = prompt
            report_to_console(f"Using free_text prompt: {prompt[:80]}...")

        report_to_console(f"Sending to model: {ACTIVE_MODEL} (backend: {ACTIVE_BACKEND})")

        try:
            description = await call_vision_model(
                ACTIVE_MODEL, ACTIVE_BACKEND, ACTIVE_BASE_URL,
                final_prompt, img_b64
            )
        except Exception as primary_err:
            if CHAT_VISION is not None:
                report_to_console(
                    f"Chat model '{ACTIVE_MODEL}' failed ({primary_err}), "
                    f"falling back to dedicated vision model '{VISION_MODEL}'"
                )
                description = await call_ollama_vl(
                    VISION_MODEL, OLLAMA_BASE_URL, final_prompt, img_b64
                )
            else:
                raise

        report_to_console(f"Success: received {len(description)} chars from model")
        return description

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        report_to_console(f"ERROR: {error_msg}")
        return f"Error analyzing image: {error_msg}"

# ============================================================================
# MAIN ENTRY POINT (for standalone testing)
# ============================================================================
if __name__ == "__main__":
    print(f"[MCP] vision: Starting MCP server (tool_vision.py)", file=sys.stderr)
    print(f"[MCP] vision: REPO_ROOT = {REPO_ROOT}", file=sys.stderr)
    print(f"[MCP] vision: Config path = {CONFIG_PATH}", file=sys.stderr)
    print(f"[MCP] vision: Chat mode = {CHAT_MODE}", file=sys.stderr)
    print(f"[MCP] vision: Dedicated vision model = {VISION_MODEL}", file=sys.stderr)
    print(f"[MCP] vision: Prefer chat model = {PREFER_CHAT_MODEL}", file=sys.stderr)
    print(f"[MCP] vision: Active model = {ACTIVE_MODEL} (backend: {ACTIVE_BACKEND})", file=sys.stderr)
    print(f"[MCP] vision: Prompt mode = {PROMPT_MODE}", file=sys.stderr)
    print(f"[MCP] vision: Resize size = {RESIZE_SIZE}px", file=sys.stderr)
    print(f"[MCP] vision: Max image size = {MAX_IMAGE_SIZE_MB}MB", file=sys.stderr)
    print(f"[MCP] vision: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")
