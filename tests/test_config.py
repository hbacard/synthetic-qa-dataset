"""Unit tests for :mod:`synthetic_qa_dataset.config`."""

import pytest
import yaml

from synthetic_qa_dataset import config


def _write_config(tmp_path, contents: dict) -> None:
    """Write ``contents`` as YAML to ``tmp_path/config.yaml`` and point
    ``config.CONFIG_PATH`` at it.

    Parameters
    ----------
    tmp_path:
        Temporary directory used by the test.
    contents:
        Configuration mapping to serialize to YAML.
    """
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    config.CONFIG_PATH = path


def test_list_providers(monkeypatch, tmp_path):
    """``list_providers`` must return the configured provider names."""
    _write_config(
        tmp_path,
        {"providers": {"albert": {"model": "x"}, "ollama": {"model": "y"}}},
    )
    assert config.list_providers() == ["albert", "ollama"]


def test_list_providers_empty(monkeypatch, tmp_path):
    """``list_providers`` must be empty when no providers are configured."""
    _write_config(tmp_path, {"providers": {}})
    assert config.list_providers() == []


def test_get_provider_defaults(monkeypatch, tmp_path):
    """``get_provider`` must fill missing optional fields with defaults."""
    _write_config(
        tmp_path,
        {
            "providers": {
                "albert": {
                    "model": "llama-3.1-8b",
                    "base_url": "http://localhost:8000/v1",
                }
            }
        },
    )
    cfg = config.get_provider("albert")
    assert cfg.name == "albert"
    assert cfg.model == "llama-3.1-8b"
    assert cfg.base_url == "http://localhost:8000/v1"
    assert cfg.api_env == "OPENAI_API_KEY"
    assert cfg.embedding == "bge-m3"
    assert cfg.temperature == config.DEFAULT_TEMPERATURE
    assert cfg.max_retries == config.DEFAULT_MAX_RETRIES
    assert cfg.min_sleep == config.DEFAULT_MIN_SLEEP
    assert cfg.max_delay == config.DEFAULT_MAX_DELAY


def test_get_provider_unknown(monkeypatch, tmp_path):
    """``get_provider`` must raise ``KeyError`` for an unknown provider."""
    _write_config(tmp_path, {"providers": {"albert": {"model": "x"}}})
    with pytest.raises(KeyError):
        config.get_provider("does-not-exist")


def test_provider_name_uses_top_level_key(monkeypatch, tmp_path):
    """``_provider_name`` must prefer the top-level configuration key."""
    _write_config(
        tmp_path,
        {"llm_provider": "ollama", "providers": {"albert": {}, "ollama": {}}},
    )
    assert config._provider_name("llm_provider") == "ollama"


def test_provider_name_fallback_to_default(monkeypatch, tmp_path):
    """``_provider_name`` must fall back to DEFAULT_PROVIDER when the
    top-level key is absent but that provider is configured."""
    _write_config(tmp_path, {"providers": {"albert": {}, "ollama": {}}})
    assert config._provider_name("llm_provider") == config.DEFAULT_PROVIDER


def test_provider_name_fallback_to_first(monkeypatch, tmp_path):
    """``_provider_name`` must fall back to the first configured provider when
    DEFAULT_PROVIDER is not among them."""
    _write_config(tmp_path, {"providers": {"first": {}, "second": {}}})
    assert config._provider_name("llm_provider") == "first"


def test_provider_name_no_providers(monkeypatch, tmp_path):
    """``_provider_name`` must return DEFAULT_PROVIDER when none are set."""
    _write_config(tmp_path, {})
    assert config._provider_name("llm_provider") == config.DEFAULT_PROVIDER
