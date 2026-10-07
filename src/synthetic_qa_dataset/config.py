from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

DEFAULT_MAX_RETRIES = 8
DEFAULT_MIN_SLEEP = 6.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_TEMPERATURE = 0.7
DEFAULT_PROVIDER = "albert"


@dataclass
class ProviderConfig:
    """Configuration of a single LLM or embedding provider.

    Attributes
    ----------
    name:
        Unique provider name used as the key in ``config.yaml``.
    model:
        Model identifier passed to the provider API.
    base_url:
        Base URL of the provider API.
    api_env:
        Name of the environment variable holding the API key.
    embedding:
        Embedding model name used by this provider.
    temperature:
        Sampling temperature for LLM calls.
    max_retries:
        Maximum number of retry attempts on retryable errors.
    min_sleep:
        Base delay (seconds) for exponential backoff.
    max_delay:
        Upper bound (seconds) for the backoff delay.
    """

    name: str
    model: str
    base_url: str
    api_env: str
    embedding: str = "bge-m3"
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    min_sleep: float = DEFAULT_MIN_SLEEP
    max_delay: float = DEFAULT_MAX_DELAY


def _load_config() -> dict:
    """Load the full YAML configuration from ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration contents, or an empty dict if the file is
        missing or empty.
    """
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_providers() -> dict[str, dict]:
    """Return the ``providers`` mapping from the configuration.

    Returns
    -------
    dict[str, dict]
        Mapping of provider name to its raw configuration, empty if none.
    """
    return _load_config().get("providers", {}) or {}


def _provider_name(key: str) -> str:
    """Return the provider configured for ``key``.

    Falls back to a single provider (``DEFAULT_PROVIDER`` if configured, else
    the first configured one) so legacy configs without the top-level keys keep
    working.

    Parameters
    ----------
    key:
        Top-level configuration key, e.g. ``"llm_provider"``.

    Returns
    -------
    str
        Name of the provider to use for ``key``.
    """
    name = _load_config().get(key)
    if name:
        return name
    providers = _load_providers()
    if DEFAULT_PROVIDER in providers:
        return DEFAULT_PROVIDER
    if providers:
        return next(iter(providers))
    return DEFAULT_PROVIDER


def get_llm_provider_name() -> str:
    """Return the name of the configured LLM provider.

    Returns
    -------
    str
        Name of the LLM provider to use.
    """
    return _provider_name("llm_provider")


def get_embedding_provider_name() -> str:
    """Return the name of the configured embedding provider.

    Returns
    -------
    str
        Name of the embedding provider to use.
    """
    return _provider_name("embedding_provider")


def get_provider(name: str) -> ProviderConfig:
    """Build a ``ProviderConfig`` for the given provider name.

    Parameters
    ----------
    name:
        Provider name as configured under ``providers`` in ``config.yaml``.

    Returns
    -------
    ProviderConfig
        Populated configuration for the requested provider.

    Raises
    ------
    KeyError
        If ``name`` is not a configured provider.
    """
    providers = _load_providers()
    if name not in providers:
        available = ", ".join(providers) or "(none configured)"
        raise KeyError(
            f"Provider '{name}' not found in {CONFIG_PATH}. Available: {available}"
        )
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
    """Return the names of all configured providers.

    Returns
    -------
    list[str]
        Provider names as configured under ``providers``.
    """
    return list(_load_providers().keys())


config = _load_providers()
