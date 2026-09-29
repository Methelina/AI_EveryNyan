"""
Unit tests for the "openai" chat mode configuration (generic OpenAI-compatible
backend) in src\\config.py.

tests/test_chat_modes.py
Version:     1.0.0
Author:      Soror L.'.L.'.
Updated:     2026-09-29

Patch Notes v1.0.0 (Soror L'.L'.):
  [+] Tests: get_chat_config for all three modes, literal accepts "openai",
      yaml round-trip with an openai_compat section.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import AppSettings  # noqa: E402


def _settings_with_mode(mode: str) -> AppSettings:
    return AppSettings(chat_mode=mode)


def test_get_chat_config_ollama():
    s = _settings_with_mode("ollama")
    assert s.get_chat_config() is s.ollama


def test_get_chat_config_llama():
    s = _settings_with_mode("llama")
    assert s.get_chat_config() is s.llama


def test_get_chat_config_openai():
    s = _settings_with_mode("openai")
    assert s.get_chat_config() is s.openai_compat


def test_openai_compat_defaults():
    s = _settings_with_mode("openai")
    cfg = s.get_chat_config()
    assert cfg.base_url.startswith("https://")
    assert cfg.chat_model
    assert hasattr(cfg, "api_key")


def test_yaml_roundtrip_with_openai_section(tmp_path):
    p = tmp_path / "settings.yaml"
    p.write_text(
        "chat_mode: openai\n"
        "openai_compat:\n"
        "  base_url: \"https://example.invalid/v1\"\n"
        "  api_key: \"sk-test\"\n"
        "  chat_model: \"test-model\"\n",
        encoding="utf-8",
    )
    s = AppSettings.from_yaml(str(p))
    assert s.chat_mode == "openai"
    assert s.openai_compat.base_url == "https://example.invalid/v1"
    assert s.openai_compat.api_key == "sk-test"
    assert s.get_chat_config() is s.openai_compat
