"""
Application configuration models and character config loader for AI_EveryNyan.
Pydantic-based settings with YAML override. Character YAML/JSON file loader.

/src/config.py
Version:     0.17.7
Author:      Soror L.'.L.'.
Updated:     2026-05-05

Patch Notes v0.17.7 (by pytraveler):
  [+] WorkspaceSettings model: sandboxed file access config for LLM tools
      (workspace_dir, max_file_size, max_read_lines, max_read_bytes, etc.).
  [+] ComfyUISettings.daemon_check_interval: configurable reconnect interval.
  [+] AppSettings.workspace: new field with WorkspaceSettings defaults.

Patch Notes v0.17.6 (by pytraveler):
  [+] Extracted from main.py: AppSettings, all sub-settings models.
  [+] CharacterConfig, AppearanceProjection, CharacterBaseConfig, CharacterAppearanceConfig.
  [*] No functional changes from original main.py code.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List

import yaml
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic_settings import BaseSettings
from typing import Literal

from logger import logger


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "bge-m3:latest"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048
    token_dump_threshold: int = 20000


class LlamaSettings(BaseModel):
    base_url: str = "http://localhost:8088/v1"
    api_key: str = ""
    chat_model: str = "Falcon-H1R-7B-Q8_0.gguf"
    timeout: int = 180
    temperature: float = 0.7
    max_tokens: int = 4096
    token_dump_threshold: int = 20000


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "everynyan_diary"
    embedding_dim: int = 1024


class DiarySettings(BaseModel):
    storage_dir: str = "data/diary"
    plagiarism_threshold: float = 0.97
    injection_max_length: int = 5000
    summary_prompt: str = ""


class GUISettings(BaseModel):
    title: str = "AI_EveryNyan"
    width: int = 900
    height: int = 700
    theme: str = "dark"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "logs/app.log"


class AntiRepeatSettings(BaseModel):
    trigger_avg: float = 0.73
    trigger_max: float = 0.69
    max_history: int = 32


class RAGSettings(BaseModel):
    top_k: int = 10
    similarity_threshold: float = 0.65
    enable_metadata_filtering: bool = False


class ContextSettings(BaseModel):
    max_history_messages: int = 40
    warn_if_context_exceeds: int = 20


class WorkspaceSettings(BaseModel):
    workspace_dir: str = "."
    max_file_size: int = 10 * 1024 * 1024
    max_read_lines: int = 2000
    max_read_bytes: int = 1024 * 1024
    max_search_results: int = 200
    allow_hidden_files: bool = False
    max_line_count_size: int = 1024 * 1024


class ComfyUISettings(BaseModel):
    server: str = "127.0.0.1:8084"
    workflow_dir: str = "workflows"
    output_dir: str = "data/comfyui_output"
    http_timeout: int = 120
    daemon_check_interval: float = 5.0


class AppSettings(BaseSettings):
    chat_mode: Literal["ollama", "llama"] = "ollama"
    embedding_mode: Literal["ollama", "custom"] = "ollama"

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    llama: LlamaSettings = Field(default_factory=LlamaSettings)

    vector_db: QdrantSettings = Field(default_factory=QdrantSettings)
    diary: DiarySettings = Field(default_factory=DiarySettings)
    gui: GUISettings = Field(default_factory=GUISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    anti_repeat: AntiRepeatSettings = Field(default_factory=AntiRepeatSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    comfyui: ComfyUISettings = Field(default_factory=ComfyUISettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    debug: bool = False

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @classmethod
    def from_yaml(cls, path: str) -> "AppSettings":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load settings from {path}: {e}")
            raise

    def get_chat_config(self):
        if self.chat_mode == "ollama":
            return self.ollama
        else:
            return self.llama

    def get_embedding_config(self):
        if self.embedding_mode == "ollama":
            return self.ollama
        else:
            logger.warning("Custom embedding mode not implemented, falling back to ollama")
            return self.ollama


class AppearanceProjection(BaseModel):
    character_name: str
    freeform: Optional[str] = None
    short_visual_description: Optional[str] = None

    class Config:
        extra = "allow"


class CharacterBaseConfig(BaseModel):
    meta: dict = Field(default_factory=dict)
    prompt: str


class CharacterAppearanceConfig(BaseModel):
    meta: dict = Field(default_factory=dict)
    freeform: str


class CharacterConfig:
    BASE_PATH = Path("config/character/base.yaml")
    APPEARANCE_PATH = Path("config/character/appearance.yaml")
    APPERANCE_JSON_DIR = Path("config/character")
    REPROJECTION_DIR = Path("config/character/reprojection")

    @staticmethod
    def _load_yaml_file(filepath: Path, model: type[BaseModel]) -> BaseModel:
        if not filepath.exists():
            raise FileNotFoundError(f"Character config not found: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return model.model_validate(data)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    @classmethod
    def load_base(cls) -> CharacterBaseConfig:
        return cls._load_yaml_file(cls.BASE_PATH, CharacterBaseConfig)

    @classmethod
    def load_appearance(cls) -> CharacterAppearanceConfig:
        return cls._load_yaml_file(cls.APPEARANCE_PATH, CharacterAppearanceConfig)

    @classmethod
    def load_appearance_json_files(cls) -> Dict[str, dict]:
        appearance_map = {}
        if not cls.APPERANCE_JSON_DIR.exists():
            return appearance_map
        for fpath in sorted(cls.APPERANCE_JSON_DIR.glob("appearance_*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("character_name", fpath.stem)
                appearance_map[name] = data
            except Exception as e:
                logger.warning(f"Skipping invalid appearance JSON: {fpath} ({e})")
        return appearance_map

    @classmethod
    def get_projection_path(cls, character_name: str) -> Path:
        return cls.REPROJECTION_DIR / f"appearance_{character_name}.projection.json"

    @classmethod
    def create_projection_from_original(cls, character_name: str, original_data: dict) -> Optional[dict]:
        cls.REPROJECTION_DIR.mkdir(parents=True, exist_ok=True)
        projection_path = cls.get_projection_path(character_name)
        try:
            with open(projection_path, "w", encoding="utf-8") as f:
                json.dump(original_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created projection for {character_name}")
            return original_data
        except Exception as e:
            logger.error(f"Failed to create projection for {character_name}: {e}")
            return None

    @classmethod
    def load_projection(cls, character_name: str, original_data: dict) -> Optional[dict]:
        projection_path = cls.get_projection_path(character_name)
        if not projection_path.exists():
            return cls.create_projection_from_original(character_name, original_data)

        try:
            with open(projection_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            AppearanceProjection.model_validate(data)
            if data.get("character_name") != character_name:
                logger.warning(f"Projection mismatch for {character_name}, recreating from original")
                return cls.create_projection_from_original(character_name, original_data)
            return data
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.warning(f"Projection for {character_name} is invalid ({e}), recreating from original")
            return cls.create_projection_from_original(character_name, original_data)
