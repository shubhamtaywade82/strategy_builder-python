import os
from strategy_api.config import Settings

def test_settings_reads_ollama_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:11434")
    monkeypatch.setenv("OLLAMA_AGENT_MODEL", "test-model")
    s = Settings()
    assert s.ollama_base_url == "http://example:11434"
    assert s.ollama_agent_model == "test-model"

def test_settings_defaults():
    s = Settings()
    assert s.sqlite_path.endswith("sqlite.db")
    assert s.binance_fapi == "https://fapi.binance.com"
