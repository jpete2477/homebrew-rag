from __future__ import annotations

import pytest

from homebrew_rag.config import Settings, get_settings


def test_defaults_are_sane():
    settings = Settings()
    assert settings.collection == "documents"
    assert settings.claude_model == "claude-opus-5"
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.api_key is None


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("RAG_COLLECTION", "tienta")
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_CLAUDE_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("RAG_ENABLE_REFUSAL_FALLBACK", "false")
    monkeypatch.setenv("RAG_API_KEY", "secret")

    settings = Settings.from_env()
    assert settings.collection == "tienta"
    assert settings.top_k == 9
    assert settings.claude_model == "claude-sonnet-5"
    assert settings.enable_refusal_fallback is False
    assert settings.api_key == "secret"


def test_blank_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RAG_COLLECTION", "")
    assert Settings.from_env().collection == "documents"


def test_non_integer_setting_fails_loudly(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "five")
    with pytest.raises(ValueError, match="RAG_TOP_K"):
        Settings.from_env()


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
