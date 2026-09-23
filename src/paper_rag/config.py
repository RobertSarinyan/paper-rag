"""Load and validate application settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when a required application setting is missing."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Load application settings without overwriting environment variables."""
    load_dotenv(dotenv_path=Path(env_file), override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    missing = [
        name
        for name, value in (
            ("GEMINI_API_KEY", api_key),
            ("GEMINI_MODEL", model),
        )
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(
            f"Missing required setting(s): {joined}. Add them to .env or the shell environment."
        )

    return Settings(gemini_api_key=api_key, gemini_model=model)
