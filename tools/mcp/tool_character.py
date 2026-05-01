#!/usr/bin/env python3
"""
MCP server providing a tool to update character appearance projections.
Exposes one tool: update_character_appearance – accepts appearance_set name and text description,
calls the configured LLM (from settings.yaml) to generate a modified JSON, validates and saves the projection.

\\tools\\mcp\\tool_character.py

Version:     0.4.0
Author:      Soror L.'.L.'.
Created:     2026-04-30

Patch Notes v0.4.1:
  [ENHANCEMENT] Expanded default locked_fields to cover all core physical traits
                (anatomy, measurements, constitution, etc.) to prevent accidental changes.
  [DOCS] Improved tool description with more diverse examples (nails, makeup, etc.)
          to help LLM recognise a wider range of appearance change requests.

Patch Notes v0.4.0:
  [REFACTOR] Argument renamed from character_name to appearance_set
             to match the new persona/appearance architecture.
  [FIX] Projection path now based on appearance_set instead of character_name.

Patch Notes v0.3.0:
  [SIMPLIFY] Always uses the chat_model from settings.yaml – no dynamic fallback logic.
  [LOG] Removed verbose debug messages: only errors and final result are reported.

Patch Notes v0.2.0:
  [CHANGE] Dynamically selected a suitable Ollama chat model at startup
           (excluded embedding model, preferred configured chat_model).
  [*] This logic was later removed in v0.3.0.

Patch Notes v0.1.0:
  Initial release with update_character_appearance tool,
  Pydantic validation, projection management, and locked fields support.
"""

import os
import sys
import json
import yaml
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastmcp import FastMCP
from pydantic import BaseModel, ValidationError

# ============================================================================
# PATH RESOLUTION – read config from project root
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # <repo>/
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"
CHARACTER_DIR = REPO_ROOT / "config" / "character"
REPROJECTION_DIR = CHARACTER_DIR / "reprojection"

# ============================================================================
# Pydantic Model for Projection Validation (minimal, compatible with main.py)
# ============================================================================
class AppearanceProjection(BaseModel):
    """Minimal validation model for projection JSON files."""
    character_name: str
    freeform: Optional[str] = None
    short_visual_description: Optional[str] = None
    # Allow any additional fields
    class Config:
        extra = "allow"

# ============================================================================
# LOAD CONFIGURATION FROM YAML (simplified)
# ============================================================================
def load_character_config():
    default_config = {
        "ollama": {
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "chat_model": "deepseek-v3.1:671b-cloud",
            "timeout": 1800,
            "temperature": 0.7,
            "max_tokens": 8192,
        }
    }
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        ollama_cfg = data.get("ollama", {})
        config = {
            "base_url": ollama_cfg.get("base_url", default_config["ollama"]["base_url"]),
            "api_key": ollama_cfg.get("api_key", default_config["ollama"]["api_key"]),
            "chat_model": ollama_cfg.get("chat_model", default_config["ollama"]["chat_model"]),
            "timeout": ollama_cfg.get("timeout", default_config["ollama"]["timeout"]),
            "temperature": 0.2,          # low for structured JSON edits
            "max_tokens": ollama_cfg.get("max_tokens", default_config["ollama"]["max_tokens"]),
        }
        return config
    except Exception as e:
        print(f"[MCP] character: Failed to load settings.yaml: {e}. Using defaults.", file=sys.stderr)
        return default_config["ollama"]

config = load_character_config()
LLM_MODEL = config["chat_model"]
LLM_BASE_URL = config["base_url"]
LLM_API_KEY = config["api_key"]
LLM_TIMEOUT = config["timeout"]
LLM_TEMPERATURE = config["temperature"]
LLM_MAX_TOKENS = config["max_tokens"]

# ============================================================================
# LOGGING (errors only)
# ============================================================================
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_LOG = LOG_DIR / "mcp_character.log"

def log_error(msg: str):
    timestamp = datetime.now().isoformat()
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} ERROR {msg}\n")

# ============================================================================
# FastMCP instance
# ============================================================================
mcp = FastMCP("character")

# ============================================================================
# INTERNAL HELPERS (updated for appearance_set)
# ============================================================================
def get_original_appearance(appearance_set: str) -> Optional[dict]:
    """Load original appearance JSON for the given appearance set."""
    original_path = CHARACTER_DIR / f"appearance_{appearance_set}.json"
    if not original_path.exists():
        return None
    try:
        with open(original_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Error reading original appearance: {e}")
        return None

def get_projection_path(appearance_set: str) -> Path:
    """Return the path for a projection file based on appearance_set."""
    return REPROJECTION_DIR / f"appearance_{appearance_set}.projection.json"

def load_projection(appearance_set: str) -> Optional[dict]:
    """Load current projection or create from original if missing/invalid."""
    projection_path = get_projection_path(appearance_set)
    if projection_path.exists():
        try:
            with open(projection_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            AppearanceProjection.model_validate(data)
            # Ensure consistency
            if data.get("character_name") != appearance_set:
                raise ValueError("Mismatched appearance_set in projection")
            return data
        except Exception as e:
            log_error(f"Invalid projection for {appearance_set}, recreating from original: {e}")
    # Create from original
    original = get_original_appearance(appearance_set)
    if not original:
        return None
    REPROJECTION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(projection_path, "w", encoding="utf-8") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)
        return original
    except Exception as e:
        log_error(f"Failed to create projection from original: {e}")
        return None

def save_projection(appearance_set: str, data: dict) -> bool:
    """Save validated projection to disk."""
    projection_path = get_projection_path(appearance_set)
    REPROJECTION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(projection_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_error(f"Failed to save projection: {e}")
        return False

async def call_llm_for_update(current_json: dict, change_request: str) -> dict:
    system_prompt = (
        "You are a precise JSON editor for character appearance data.\n"
        "You receive the current appearance JSON and a change request.\n"
        "Your task is to output the COMPLETE updated JSON object.\n"
        "- Modify ONLY the fields relevant to the request.\n"
        "- Update the short_visual_description depend on the changes.\n"
        "- Keep all other fields unchanged.\n"
        "- Preserve the exact structure and data types.\n"
        "- Do NOT alter locked_fields: character_name, race, gender etc.\n"
        "- Return ONLY the JSON object, without markdown, code fences, or explanations.\n"
    )
    user_prompt = f"Current appearance JSON:\n{json.dumps(current_json, ensure_ascii=False)}\n\nChange request: {change_request}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT, headers=headers) as client:
        resp = await client.post(f"{LLM_BASE_URL}/chat/completions", json=payload)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
    
    # Extract JSON from possible markdown wrapping
    if "```json" in content:
        parts = content.split("```json")
        if len(parts) > 1:
            end_block = parts[1].split("```")
            if len(end_block) > 0:
                content = end_block[0].strip()
    elif "```" in content:
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1].strip()
    # Find first { and matching }
    start = content.find('{')
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    stack = 0
    end = -1
    for i, ch in enumerate(content[start:], start):
        if ch == '{':
            stack += 1
        elif ch == '}':
            stack -= 1
            if stack == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unbalanced JSON braces in LLM response")
    json_str = content[start:end+1]
    return json.loads(json_str)

def apply_locked_fields(original_data: dict, new_data: dict, locked_fields: List[str]) -> dict:
    for field in locked_fields:
        if field in original_data:
            new_data[field] = original_data[field]
    return new_data

# ============================================================================
# MCP TOOL
# ============================================================================
@mcp.tool()
async def update_character_appearance(
    appearance_set: str,
    change_request: str,
    locked_fields: str = "character_name,race,gender,subject_type,origin,age_estimate,height_cm,face_shape,measurements_cm,breast_type,cap_size,constitution,anatomy"
) -> str:
    """
    Modify the current character's visual look, clothing, hair, or body details.
    
    Use this tool whenever the user asks to change how the current character looks,
    including but not limited to: painting nails, applying makeup, dyeing hair,
    changing hairstyle, putting on clothing or accessories, removing items,
    altering footwear, adding or removing scars, tattoos, jewelry, etc.
    
    Any request that implies a change to the character's appearance should trigger this tool.
    For example:
    - "paint your nails black"
    - "put on red lipstick"
    - "tie your hair into a ponytail"
    - "change into a white dress"
    - "remove your earrings"
    
    The tool will take the CURRENT character appearance (from the active projection), 
    apply your described changes, and save the updated version. 
    The character's visual description in the next system prompt will then reflect the new state.
    
    Provide ONLY the changes to LLM; do not describe what remains unchanged.
    
    Parameters:
    - appearance_set: The exact name of the currently active appearance set (e.g., 'Frieren').
    - change_request: A detailed, natural language description of the desired appearance changes.
    - locked_fields: Comma-separated list of fields that MUST remain unchanged (default: 'character_name,race,gender, etc').
    """
    
    # 1. Load current projection (or create from original)
    current_data = load_projection(appearance_set)
    if current_data is None:
        return f"Error: Could not load or create projection for appearance set '{appearance_set}'. Ensure an original appearance_{appearance_set}.json exists."

    # 2. Locked fields
    locked_list = [f.strip() for f in locked_fields.split(",") if f.strip()]
    if "character_name" not in locked_list:
        locked_list.append("character_name")

    # 3. LLM call
    try:
        updated_data = await call_llm_for_update(current_data, change_request)
    except Exception as e:
        log_error(f"LLM call failed: {e}")
        return f"Error: Failed to obtain updated appearance from LLM: {e}"

    # 4. Apply locked fields
    updated_data = apply_locked_fields(current_data, updated_data, locked_list)

    # 5. Validate
    try:
        AppearanceProjection.model_validate(updated_data)
    except ValidationError as ve:
        log_error(f"Validation failed: {ve}")
        return f"Error: Updated appearance data is invalid: {ve}"

    # 6. Save
    if not save_projection(appearance_set, updated_data):
        return "Error: Failed to save updated projection to disk."

    # 7. Build diff
    changes = [key for key in updated_data if updated_data.get(key) != current_data.get(key) and key not in locked_list]
    if changes:
        return f"Successfully updated {appearance_set}'s appearance. Changed fields: {', '.join(changes)}."
    else:
        return f"No visible changes made to {appearance_set}'s appearance (or only locked fields were altered)."

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print(f"[MCP] character: Starting MCP server (tool_character.py)", file=sys.stderr)
    print(f"[MCP] character: Chat model: {LLM_MODEL} (from settings.yaml)", file=sys.stderr)
    print(f"[MCP] character: Ready to accept stdio MCP connections", file=sys.stderr)
    mcp.run(transport="stdio")