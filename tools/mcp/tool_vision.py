#!/usr/bin/env python3
"""
MCP server providing vision-language tasks using Ollama VL models.
Exposes one tool: describe_image – accepts local path or URL, returns structured JSON description.
Uses PIL for validation, security (PNG conversion, resize), reads config from settings.yaml.

\\tools\\mcp\\tool_vision.py

Version:     0.3.3
Author:      Soror L.'.L.'.
Updated:     2026-04-29

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
# LOAD CONFIGURATION FROM YAML
# ============================================================================
def load_vision_config():
    default_config = {
        "enabled": True,
        "model": "qwen3-vl:235b-cloud",
        "default_prompt": "What do you see in this image? Describe in detail.",
        "prompt_mode": "structured_json",
        "max_image_size_mb": 20,
        "resize_size": 1024,
    }
    try:
        # Print debug info to stderr (visible in console when run standalone)
        print(f"[DEBUG] Loading config from: {CONFIG_PATH}", file=sys.stderr)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        vision = data.get("vision", {})
        config = {**default_config, **vision}
        print(f"[DEBUG] Config loaded: model={config['model']}, mode={config['prompt_mode']}, resize={config['resize_size']}",
              file=sys.stderr)
        return config
    except Exception as e:
        # Ensure logs directory exists before writing
        (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
        with open(REPO_ROOT / "logs" / "mcp_vision.log", "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.now().isoformat()} Failed to load config: {e}, using defaults\n")
        print(f"[WARN] Failed to load config: {e}, using defaults", file=sys.stderr)
        return default_config

config = load_vision_config()

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
VISION_MODEL = config["model"]
PROMPT_MODE = config["prompt_mode"]
DEFAULT_PROMPT = config["default_prompt"]
MAX_IMAGE_SIZE_MB = config["max_image_size_mb"]
RESIZE_SIZE = config["resize_size"]

# Ensure logs directory exists
DEBUG_LOG = REPO_ROOT / "logs" / "mcp_vision.log"
DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)

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
def log_debug(msg: str):
    timestamp = datetime.now().isoformat()
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {msg}\n")

def report_to_console(msg: str):
    print(f"[MCP] vision: {msg}", file=sys.stderr, flush=True)
    log_debug(msg)

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

async def call_ollama_vl(prompt: str, image_base64: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            }
        }
        response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

# ============================================================================
# MCP TOOL
# ============================================================================
@mcp.tool()
async def describe_image(
    image_source: str,
    prompt: str = DEFAULT_PROMPT,
    is_url: bool = False
) -> str:
    report_to_console(f"describe_image called: source={image_source[:100]}, is_url={is_url}, mode={PROMPT_MODE}")
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

        report_to_console(f"Sending to VL model: {VISION_MODEL}")
        description = await call_ollama_vl(final_prompt, img_b64)

        report_to_console(f"Success: received {len(description)} chars from VL model")
        return description

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        report_to_console(f"ERROR: {error_msg}")
        return f"Error analyzing image: {error_msg}"

# ============================================================================
# MAIN ENTRY POINT (for standalone testing)
# ============================================================================
if __name__ == "__main__":
    # Print startup information to stderr
    print(f"[MCP] vision: Starting MCP server (tool_vision.py)", file=sys.stderr)
    print(f"[MCP] vision: REPO_ROOT = {REPO_ROOT}", file=sys.stderr)
    print(f"[MCP] vision: Config path = {CONFIG_PATH}", file=sys.stderr)
    print(f"[MCP] vision: Vision model = {VISION_MODEL}", file=sys.stderr)
    print(f"[MCP] vision: Prompt mode = {PROMPT_MODE}", file=sys.stderr)
    print(f"[MCP] vision: Resize size = {RESIZE_SIZE}px", file=sys.stderr)
    print(f"[MCP] vision: Max image size = {MAX_IMAGE_SIZE_MB}MB", file=sys.stderr)
    print(f"[MCP] vision: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")