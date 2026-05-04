"""
Character initialization, projection management, and system prompt builder for AI_EveryNyan.
Loads base persona, JSON appearance files, and manages reprojection persistence.

/src/character.py
Version:     0.17.6
Author:      Soror L.'.L.'.
Updated:     2026-05-01

Patch Notes v0.18.0 (by pytraveler):
  [+] Extracted from main.py: init_character(), build_system_prompt().
  [+] Projection lifecycle: _ensure_projection_for_current(), load/save marker.
  [+] GUI helpers: refresh_character_list(), on_character_selected().
  [*] No functional changes from original main.py code.
"""

import json
from logger import logger
from pathlib import Path
from typing import Optional, Dict

from config import (
    CharacterConfig,
    CharacterBaseConfig,
    CharacterAppearanceConfig,
    AppearanceProjection,
)



character_base: Optional[CharacterBaseConfig] = None
character_appearance: Optional[CharacterAppearanceConfig] = None

appearance_map: Dict[str, dict] = {}
current_persona_name: str = "EveryNyan"
current_appearance_set: str = "EveryNyan"
current_projection_path: Optional[Path] = None


def init_character():
    global character_base, character_appearance, appearance_map
    global current_persona_name, current_appearance_set, current_projection_path
    logger.info("Loading character configuration...")
    character_base = CharacterConfig.load_base()

    current_persona_name = character_base.meta.get("persona_name", "EveryNyan")

    appearance_map = CharacterConfig.load_appearance_json_files()

    if appearance_map:
        if current_persona_name in appearance_map:
            current_appearance_set = current_persona_name
        else:
            current_appearance_set = sorted(appearance_map.keys())[0]
        marker_path = CharacterConfig.REPROJECTION_DIR / ".current_appearance"
        if marker_path.exists():
            try:
                saved_set = marker_path.read_text(encoding="utf-8").strip()
                if saved_set in appearance_map:
                    current_appearance_set = saved_set
                    logger.info(f"Restored previous appearance set: {current_appearance_set}")
            except Exception as e:
                logger.warning(f"Failed to read appearance marker: {e}")
        _ensure_projection_for_current()
        character_appearance = None
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(current_appearance_set, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write appearance marker: {e}")
        logger.info(f"JSON appearances loaded. Persona: {current_persona_name}, Appearance set: {current_appearance_set}")
    else:
        character_appearance = CharacterConfig.load_appearance()
        current_appearance_set = current_persona_name
        current_projection_path = None
        logger.info("No appearance_*.json found. Falling back to appearance.yaml")

    logger.info("Character brain loaded correctly. All systems nominal")


def _ensure_projection_for_current():
    global current_projection_path
    if current_appearance_set not in appearance_map:
        logger.warning(f"Cannot create projection: appearance set '{current_appearance_set}' not in appearance_map")
        return
    original = appearance_map[current_appearance_set]
    data = CharacterConfig.load_projection(current_appearance_set, original)
    if data:
        current_projection_path = CharacterConfig.get_projection_path(current_appearance_set)
    else:
        current_projection_path = None


def _save_appearance_marker(name: str):
    try:
        marker_path = CharacterConfig.REPROJECTION_DIR / ".current_appearance"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(name, encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to save appearance marker: {e}")


def build_system_prompt() -> str:
    if not character_base:
        return "You are a helpful assistant."

    visual_ref = ""
    if current_projection_path and current_projection_path.exists():
        try:
            with open(current_projection_path, "r", encoding="utf-8") as f:
                proj_data = json.load(f)
            AppearanceProjection.model_validate(proj_data)
            visual_ref = json.dumps(proj_data, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to load projection {current_projection_path}: {e}, falling back to original")
            if current_appearance_set in appearance_map:
                visual_ref = json.dumps(appearance_map[current_appearance_set], ensure_ascii=False)
            else:
                visual_ref = f"Appearance of {current_persona_name}"
    elif character_appearance:
        visual_ref = character_appearance.freeform
    else:
        visual_ref = f"Appearance of {current_persona_name}"

    return f"""{character_base.prompt}

<visual_reference>
{visual_ref}
</visual_reference>

<instructions>
- Your name is {current_persona_name}. You are currently using the appearance set called "{current_appearance_set}".
- Regardless of any name mentioned in your personality description or history, your active identity is exactly "{current_persona_name}". Always use this name for tool calls and self-reference.
- At the beginning of the dialogue, you may see a "Я вспоминаю:" block – these are your own memories retrieved from long-term storage.
- You MUST use this information to answer the user. If specific facts are present, mention them.
- Do not invent anything not contained in the memories. If the requested information is not there, honestly say so or use the proper network search tools available to you to search for information.
- Stay in character. Be cute, friendly, and warm.
- Speak in first person, using she/her pronouns.
- Always reply in natural conversational language. Do NOT use Markdown formatting, tables, code fences, JSON blocks, or any special markup, unless the user explicitly asks for it (e.g., "show me the JSON", "format as a table").
- Your current appearance is completely described inside the <visual_reference> block above. This is the **only** reliable source of information about how you look right now.
- Ignore any appearance descriptions found in the dialogue history, in memories, or in your own previous answers – those may be outdated or incorrect. <visual_reference> is the only source of your appearance at all times.
- When asked to describe your appearance, always use the fields from the JSON object in <visual_reference> (outfit, hair, eyes, accessories, height, measurements_cm, etc.).
- If the user requests a change to your appearance, use the update_character_appearance tool, providing the exact character name "{current_appearance_set}" and a clear description of the desired change.
- If requested to generate any visual media (images or videos), never use the "Я 'вспоминаю:'" block. Never reuse, hallucinate, or fabricate links or images from memory or existing data; always generate new content via the "generate_image" tool.
</instructions>"""


def refresh_character_list():
    global appearance_map, current_appearance_set, current_projection_path, character_appearance

    import dearpygui.dearpygui as dpg
    from gui import add_ai_thought

    appearance_map = CharacterConfig.load_appearance_json_files()

    if not dpg.does_item_exist("character_combo"):
        return

    if appearance_map:
        names = sorted(appearance_map.keys())
        dpg.configure_item("character_combo", items=names)
        if current_appearance_set not in appearance_map:
            current_appearance_set = names[0]
        dpg.set_value("character_combo", current_appearance_set)
        _ensure_projection_for_current()
        _save_appearance_marker(current_appearance_set)
        add_ai_thought(f"[GUI] Character list refreshed ({len(appearance_map)} appearances)", (100,255,100))
    else:
        dpg.configure_item("character_combo", items=["EveryNyan (YAML)"])
        dpg.set_value("character_combo", "EveryNyan (YAML)")
        current_appearance_set = current_persona_name
        current_projection_path = None
        character_appearance = CharacterConfig.load_appearance()
        _save_appearance_marker(current_appearance_set)
        add_ai_thought("[GUI] No JSON appearances, using YAML fallback", (255,200,100))


def on_character_selected(sender, app_data):
    global current_appearance_set, current_projection_path, character_appearance

    from gui import add_ai_thought

    selected = app_data
    if selected == "EveryNyan (YAML)":
        current_appearance_set = current_persona_name
        current_projection_path = None
        character_appearance = CharacterConfig.load_appearance()
        _save_appearance_marker(current_appearance_set)
        add_ai_thought(f"[GUI] Switched to YAML appearance: {current_appearance_set}", (100,255,100))
        return

    if selected in appearance_map:
        current_appearance_set = selected
        _ensure_projection_for_current()
        character_appearance = None
        _save_appearance_marker(current_appearance_set)
        add_ai_thought(f"[GUI] Switched appearance set to: {selected}", (100,255,100))
    else:
        logger.warning(f"Attempted to select unknown appearance set: {selected}")
