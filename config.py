import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

DEFAULT_MAX_RETRIES = 8
DEFAULT_MIN_SLEEP = 6.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_TEMPERATURE = 0.7


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_env: str
    embedding: str = "bge-m3"
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    min_sleep: float = DEFAULT_MIN_SLEEP
    max_delay: float = DEFAULT_MAX_DELAY


def _load_providers() -> dict[str, dict]:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("providers", {}) or {}


def get_provider(name: str) -> ProviderConfig:
    providers = _load_providers()
    if name not in providers:
        available = ", ".join(providers) or "(none configured)"
        raise KeyError(f"Provider '{name}' not found in {CONFIG_PATH}. Available: {available}")
    raw = providers[name]
    return ProviderConfig(
        name=name,
        model=raw.get("model", "gpt-4o"),
        base_url=raw.get("base_url"),
        api_env=raw.get("api_env", "OPENAI_API_KEY"),
        embedding=raw.get("embedding", "bge-m3"),
        temperature=raw.get("temperature", DEFAULT_TEMPERATURE),
        max_retries=raw.get("max_retries", DEFAULT_MAX_RETRIES),
        min_sleep=raw.get("min_sleep", DEFAULT_MIN_SLEEP),
        max_delay=raw.get("max_delay", DEFAULT_MAX_DELAY),
    )


def list_providers() -> list[str]:
    return list(_load_providers().keys())


config = _load_providers()