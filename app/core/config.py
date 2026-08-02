"""Argo configuration. Single source of truth for env vars + defaults."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _env(key: str, default: str = "") -> str:
    """Get env var, strip whitespace, return default if empty."""
    val = os.environ.get(key, default)
    return val.strip() if val else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key, "").lower()
    if raw in ("1", "true", "yes", "on", "y", "t"):
        return True
    if raw in ("0", "false", "no", "off", "n", "f"):
        return False
    return default


class Settings:
    """All runtime configuration. Re-read on each instantiation to support tests."""

    # --- LLM endpoint ---
    LLM_API_KEY: str = _env("LLM_API_KEY", "")
    LLM_BASE_URL: str = _env("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_MODEL: str = _env("LLM_MODEL", "qwen/qwen3-coder:free")
    LLM_TIMEOUT: int = _env_int("LLM_TIMEOUT", 300)

    # --- Agent behavior ---
    AGENT_MAX_STEPS: int = _env_int("AGENT_MAX_STEPS", 40)
    AGENT_TEMPERATURE: float = _env_float("AGENT_TEMPERATURE", 0.1)
    AGENT_MAX_TOKENS: int = _env_int("AGENT_MAX_TOKENS", 8192)
    AGENT_CONTEXT_WINDOW: int = _env_int("AGENT_CONTEXT_WINDOW", 24000)
    AGENT_AUTO_APPROVE: bool = _env_bool("AGENT_AUTO_APPROVE", False)
    AGENT_BASH_TIMEOUT: int = _env_int("AGENT_BASH_TIMEOUT", 120)
    AGENT_LOOP_THRESHOLD: int = _env_int("AGENT_LOOP_THRESHOLD", 3)

    # --- Server ---
    ARGO_HOST: str = _env("ARGO_HOST", "0.0.0.0")
    ARGO_PORT: int = _env_int("ARGO_PORT", 8000)
    ARGO_WORKSPACE: Path = Path(_env("ARGO_WORKSPACE", "./workspaces/default")).resolve()
    ARGO_DB: Path = Path(_env("ARGO_DB", "./data/argo.db")).resolve()

    # --- Limits ---
    MAX_TOOL_OUTPUT: int = 12000
    MAX_FILE_READ: int = 16000
    MAX_LIST_ROWS: int = 500

    def ensure_dirs(self) -> None:
        """Create needed directories."""
        self.ARGO_WORKSPACE.mkdir(parents=True, exist_ok=True)
        self.ARGO_DB.parent.mkdir(parents=True, exist_ok=True)


# Singleton — instantiating the class is cheap (no I/O).
settings = Settings()
settings.ensure_dirs()
