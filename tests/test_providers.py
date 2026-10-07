"""Unit tests for :mod:`synthetic_qa_dataset.providers`."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from synthetic_qa_dataset import providers
from synthetic_qa_dataset.providers import (
    RequestLimitError,
    _acall_with_retry,
    _build_messages,
    _call_with_retry,
    _is_retryable,
    _retry_delay,
    _schema_hint,
)


class _RateLimitError(Exception):
    """Exception carrying an HTTP ``429`` status code."""

    status_code = 429


class _ServerError(Exception):
    """Exception carrying a non-retryable HTTP status code."""

    status_code = 500


class _Answer(BaseModel):
    """Small pydantic schema used to exercise ``_schema_hint``."""

    text: str
    score: float
    valid: bool


class _BareFailure(Exception):
    """Exception without a ``status_code`` attribute."""


# ---------------------------------------------------------------------------
# _retry_delay
# ---------------------------------------------------------------------------


def test_retry_delay_doubles():
    """Delay must double with each attempt starting from ``min_sleep``."""
    assert _retry_delay(0, 6.0, 60.0) == 6.0
    assert _retry_delay(1, 6.0, 60.0) == 12.0
    assert _retry_delay(2, 6.0, 60.0) == 24.0


def test_retry_delay_is_capped():
    """Delay must never exceed ``max_delay``."""
    assert _retry_delay(10, 6.0, 60.0) == 60.0


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [_RateLimitError(), ConnectionError("down"), TimeoutError("slow")],
)
def test_is_retryable_true(exc):
    """Rate-limit and connection/timeout errors must be retryable."""
    assert _is_retryable(exc)


@pytest.mark.parametrize(
    "exc", [_ServerError(), _BareFailure("boom"), ValueError("bad")]
)
def test_is_retryable_false(exc):
    """Other errors must not be retryable."""
    assert not _is_retryable(exc)


def test_is_retryable_nested_response():
    """A ``429`` hidden on ``exc.response.status_code`` must be retryable."""

    class _Response:
        status_code = 429

    class _WithResponse(Exception):
        response = _Response()

    assert _is_retryable(_WithResponse())


# ---------------------------------------------------------------------------
# _call_with_retry
# ---------------------------------------------------------------------------


def test_call_with_retry_success(monkeypatch):
    """A successful call must return its value without retrying."""
    monkeypatch.setattr(providers, "_sleep_with_backoff", lambda *a: None)
    assert _call_with_retry(lambda: "ok") == "ok"


def test_call_with_retry_retries_then_succeeds(monkeypatch):
    """A retryable error followed by success must return the value."""
    monkeypatch.setattr(providers, "_sleep_with_backoff", lambda *a: None)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise _RateLimitError()
        return "recovered"

    assert _call_with_retry(flaky, max_retries=3) == "recovered"
    assert calls["count"] == 2


def test_call_with_retry_gives_up(monkeypatch):
    """Persistent retryable errors must raise ``RequestLimitError``."""
    monkeypatch.setattr(providers, "_sleep_with_backoff", lambda *a: None)

    def always_fails():
        raise _RateLimitError()

    with pytest.raises(RequestLimitError):
        _call_with_retry(always_fails, max_retries=2)


def test_call_with_retry_non_retryable_propagates(monkeypatch):
    """A non-retryable error must propagate immediately."""
    monkeypatch.setattr(providers, "_sleep_with_backoff", lambda *a: None)

    def fails():
        raise _ServerError()

    with pytest.raises(_ServerError):
        _call_with_retry(fails, max_retries=3)


# ---------------------------------------------------------------------------
# _acall_with_retry
# ---------------------------------------------------------------------------


def test_acall_with_retry_success(monkeypatch):
    """An async successful call must return its value without retrying."""

    async def _noop(*_args):
        pass

    monkeypatch.setattr(providers, "_async_sleep_with_backoff", _noop)

    async def ok():
        return "ok"

    assert asyncio.run(_acall_with_retry(ok)) == "ok"


def test_acall_with_retry_gives_up(monkeypatch):
    """Persistent retryable errors must raise ``RequestLimitError``."""

    async def _noop(*_args):
        pass

    monkeypatch.setattr(providers, "_async_sleep_with_backoff", _noop)

    async def always_fails():
        raise _RateLimitError()

    with pytest.raises(RequestLimitError):
        asyncio.run(_acall_with_retry(always_fails, max_retries=2))


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


def test_build_messages_without_schema():
    """Without a schema, a single user message and no kwargs are returned."""
    messages, kwargs = _build_messages("Hello", None)
    assert messages == [{"role": "user", "content": "Hello"}]
    assert kwargs == {}


def test_build_messages_with_schema():
    """With a schema, a structured prompt and JSON response format are set."""
    messages, kwargs = _build_messages("Hello", _Answer)
    assert kwargs == {"response_format": {"type": "json_object"}}
    assert messages[0]["role"] == "user"
    assert "Return ONLY a single JSON object" in messages[0]["content"]


# ---------------------------------------------------------------------------
# _schema_hint
# ---------------------------------------------------------------------------


def test_schema_hint_is_valid_json_matching_schema():
    """The hint must be a JSON object whose keys match the schema fields."""
    hint = json.loads(_schema_hint(_Answer))
    assert set(hint) == {"text", "score", "valid"}
    assert isinstance(hint["text"], str)
    assert isinstance(hint["score"], (int, float))
    assert hint["valid"] is True
