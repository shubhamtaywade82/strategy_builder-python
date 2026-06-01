from __future__ import annotations
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_REPO_ROOT / ".env"), extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_agent_model: str = "qwen3.5:4b"
    binance_fapi: str = "https://fapi.binance.com"
    sqlite_path: str = str(_REPO_ROOT / "sqlite.db")
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    frontend_dist: str = str(_REPO_ROOT / "frontend" / "dist")

settings = Settings()
