"""Provider-agnostic OpenAI-compatible clients with exponential-backoff retries.

Wraps ``OpenAI``/``AsyncOpenAI`` into deepeval-compatible LLM and embedding
models, adding retry with exponential backoff on rate-limit (HTTP 429) and
connection/timeout errors for any OpenAI-compatible provider.
"""

import asyncio
import json
import os
import time

from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .config import (
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_SLEEP,
    DEFAULT_TEMPERATURE,
    ProviderConfig,
)

RATE_LIMIT_STATUS = 429


class RequestLimitError(RuntimeError):
    """Raised when a call exhausts all retry attempts on retryable errors."""


def _is_retryable(exc: Exception) -> bool:
    """Return whether ``exc`` should trigger a retry.

    Parameters
    ----------
    exc:
        The exception raised by the API call.

    Returns
    -------
    bool
        True for rate-limit (HTTP 429) or connection/timeout errors.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == RATE_LIMIT_STATUS or isinstance(
        exc, (ConnectionError, TimeoutError)
    )


def _retry_delay(attempt: int, min_sleep: float, max_delay: float) -> float:
    """Compute the exponential backoff delay for an attempt.

    Parameters
    ----------
    attempt:
        Zero-based retry attempt number.
    min_sleep:
        Base delay (seconds) for the first retry.
    max_delay:
        Upper bound (seconds) on the delay.

    Returns
    -------
    float
        Delay (seconds) to sleep before retrying.
    """
    return min(min_sleep * (2**attempt), max_delay)


def _sleep_with_backoff(attempt: int, min_sleep: float, max_delay: float):
    """Block for the exponential backoff delay of an attempt.

    Parameters
    ----------
    attempt:
        Zero-based retry attempt number.
    min_sleep:
        Base delay (seconds) for the first retry.
    max_delay:
        Upper bound (seconds) on the delay.
    """
    time.sleep(_retry_delay(attempt, min_sleep, max_delay))


async def _async_sleep_with_backoff(attempt: int, min_sleep: float, max_delay: float):
    """Wait asynchronously for the exponential backoff delay of an attempt.

    Parameters
    ----------
    attempt:
        Zero-based retry attempt number.
    min_sleep:
        Base delay (seconds) for the first retry.
    max_delay:
        Upper bound (seconds) on the delay.
    """
    await asyncio.sleep(_retry_delay(attempt, min_sleep, max_delay))


def _call_with_retry(
    call,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_sleep: float = DEFAULT_MIN_SLEEP,
    max_delay: float = DEFAULT_MAX_DELAY,
    **kwargs,
):
    """Invoke ``call`` synchronously, retrying on retryable errors.

    Parameters
    ----------
    call:
        Callable to invoke with ``*args`` and ``**kwargs``.
    max_retries:
        Maximum number of retry attempts after the first call.
    min_sleep:
        Base delay (seconds) for exponential backoff.
    max_delay:
        Upper bound (seconds) on the backoff delay.

    Returns
    -------
    The return value of ``call``.

    Raises
    ------
    RequestLimitError
        If all attempts fail on retryable errors.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            if attempt >= max_retries:
                raise RequestLimitError(
                    f"Gave up after {max_retries + 1} attempts on rate-limit/connection error: {exc}"
                ) from exc
            _sleep_with_backoff(attempt, min_sleep, max_delay)
    raise last_exc


async def _acall_with_retry(
    call,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_sleep: float = DEFAULT_MIN_SLEEP,
    max_delay: float = DEFAULT_MAX_DELAY,
    **kwargs,
):
    """Invoke ``call`` asynchronously, retrying on retryable errors.

    Parameters
    ----------
    call:
        Awaitable callable to invoke with ``*args`` and ``**kwargs``.
    max_retries:
        Maximum number of retry attempts after the first call.
    min_sleep:
        Base delay (seconds) for exponential backoff.
    max_delay:
        Upper bound (seconds) on the backoff delay.

    Returns
    -------
    The awaited return value of ``call``.

    Raises
    ------
    RequestLimitError
        If all attempts fail on retryable errors.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await call(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            if attempt >= max_retries:
                raise RequestLimitError(
                    f"Gave up after {max_retries + 1} attempts on rate-limit/connection error: {exc}"
                ) from exc
            await _async_sleep_with_backoff(attempt, min_sleep, max_delay)
    raise last_exc


def init_client(base_url, api_key_env: str):
    """Create a synchronous OpenAI-compatible client.

    Parameters
    ----------
    base_url:
        Base URL of the provider API.
    api_key_env:
        Name of the environment variable holding the API key; falls back to
        the literal string ``"ollama"`` when unset.

    Returns
    -------
    OpenAI
        The synchronous client.
    """
    return OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "ollama"))


def init_async_client(base_url, api_key_env: str):
    """Create an asynchronous OpenAI-compatible client.

    Parameters
    ----------
    base_url:
        Base URL of the provider API.
    api_key_env:
        Name of the environment variable holding the API key; falls back to
        the literal string ``"ollama"`` when unset.

    Returns
    -------
    AsyncOpenAI
        The asynchronous client.
    """
    return AsyncOpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "ollama"))


def _deref(schema_obj: dict, json_schema: dict) -> dict:
    """Resolve a JSON schema ``$ref`` against the top-level definitions.

    Parameters
    ----------
    schema_obj:
        JSON schema object, possibly containing a ``$ref``.
    json_schema:
        Root JSON schema holding the ``$defs``/``definitions`` maps.

    Returns
    -------
    dict
        The referenced schema, or ``schema_obj`` unchanged if no ``$ref``.
    """
    ref = schema_obj.get("$ref")
    if ref:
        name = ref.split("/")[-1]
        return json_schema.get("$defs", {}).get(
            name, json_schema.get("definitions", {}).get(name, {})
        )
    return schema_obj


def _pick_type(schema_obj: dict) -> dict:
    """Pick the first non-null option of an ``anyOf``/``oneOf`` schema.

    Parameters
    ----------
    schema_obj:
        JSON schema object possibly using ``anyOf`` or ``oneOf``.

    Returns
    -------
    dict
        The first non-null sub-schema, or ``schema_obj`` unchanged.
    """
    for key in ("anyOf", "oneOf"):
        options = schema_obj.get(key) or []
        for opt in options:
            if opt.get("type") != "null":
                return opt
    return schema_obj


def _type_example(prop: dict, json_schema: dict):
    """Build an example value matching the type of a schema property.

    Parameters
    ----------
    prop:
        JSON schema describing the property.
    json_schema:
        Root JSON schema used to resolve references.

    Returns
    -------
    Example value for the property type (object, list, bool, int/float or str).
    """
    prop = _deref(prop, json_schema)
    prop = _pick_type(prop)
    ptype = prop.get("type")
    if "properties" in prop or "allOf" in prop or prop.get("$ref"):
        return _object_example(prop, json_schema)
    if ptype == "array":
        items = _deref(prop.get("items", {}), json_schema)
        items = _pick_type(items)
        return [_object_example(items, json_schema)] if "properties" in items else [""]
    if ptype == "boolean":
        return True
    if ptype in ("integer", "number"):
        return 0
    return ""


def _object_example(field_schema: dict, json_schema: dict):
    """Build an example object from a JSON schema's ``properties``.

    Parameters
    ----------
    field_schema:
        JSON schema describing an object.
    json_schema:
        Root JSON schema used to resolve references.

    Returns
    -------
    dict
        Example mapping of property name to example value.
    """
    merged = {}
    for key in ("allOf",):
        for part in field_schema.get(key) or []:
            merged.update(_object_example(_deref(part, json_schema), json_schema))
    field_schema = {**field_schema, **merged}
    props = field_schema.get("properties", {})
    return {name: _type_example(sub, json_schema) for name, sub in props.items()}


def _schema_hint(schema: BaseModel) -> str:
    """Serialize a Pydantic model as an example JSON object.

    Parameters
    ----------
    schema:
        Pydantic model describing the expected response structure.

    Returns
    -------
    str
        Compact JSON example used as a format hint.
    """
    json_schema = schema.model_json_schema()
    if json_schema.get("type") == "array":
        example = [_object_example(json_schema.get("items", {}), json_schema)]
    else:
        example = _object_example(json_schema, json_schema)
    return json.dumps(example, ensure_ascii=False)


def _build_messages(prompt: str, schema: BaseModel | None):
    """Build the chat messages and extra kwargs for an API call.

    When ``schema`` is provided, appends a structural hint and requests a
    JSON object response.

    Parameters
    ----------
    prompt:
        User prompt to send.
    schema:
        Optional Pydantic model used to build the JSON format hint.

    Returns
    -------
    tuple[list[dict], dict]
        The messages list and the extra request kwargs.
    """
    if schema is None:
        return [{"role": "user", "content": prompt}], {}
    hint = _schema_hint(schema)
    structured_prompt = (
        f"{prompt}\n\n"
        "Return ONLY a single JSON object (no markdown, no extra text) that matches "
        f"exactly this structure, filling in the values:\n{hint}"
    )
    return (
        [{"role": "user", "content": structured_prompt}],
        {"response_format": {"type": "json_object"}},
    )


class CustomOpenAI(DeepEvalBaseLLM):
    """OpenAI-compatible LLM wrapper with exponential-backoff retries.

    Attributes
    ----------
    model:
        Model identifier passed to the provider API.
    base_url:
        Base URL of the provider API.
    api_env:
        Name of the environment variable holding the API key.
    temperature:
        Sampling temperature for LLM calls.
    max_retries:
        Maximum number of retry attempts on retryable errors.
    min_sleep:
        Base delay (seconds) for exponential backoff.
    max_delay:
        Upper bound (seconds) on the backoff delay.
    """

    def __init__(
        self,
        model,
        base_url,
        api_env,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_sleep: float = DEFAULT_MIN_SLEEP,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        """Initialize the LLM wrapper and its sync/async clients.

        Parameters
        ----------
        model:
            Model identifier passed to the provider API.
        base_url:
            Base URL of the provider API.
        api_env:
            Name of the environment variable holding the API key.
        temperature:
            Sampling temperature for LLM calls.
        max_retries:
            Maximum number of retry attempts on retryable errors.
        min_sleep:
            Base delay (seconds) for exponential backoff.
        max_delay:
            Upper bound (seconds) on the backoff delay.
        """
        self.model = model
        super().__init__(model)
        self.base_url = base_url
        self.api_env = api_env
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_sleep = min_sleep
        self.max_delay = max_delay
        self._client = init_client(self.base_url, self.api_env)
        self._a_client = init_async_client(self.base_url, self.api_env)

    @classmethod
    def from_provider(cls, cfg: ProviderConfig):
        """Build a wrapper from a :class:`ProviderConfig`.

        Parameters
        ----------
        cfg:
            Provider configuration to instantiate from.

        Returns
        -------
        CustomOpenAI
            Configured LLM wrapper.
        """
        return cls(
            model=cfg.model,
            base_url=cfg.base_url,
            api_env=cfg.api_env,
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
            min_sleep=cfg.min_sleep,
            max_delay=cfg.max_delay,
        )

    def load_model(self):
        """Return the model identifier expected by deepeval."""
        return self.model

    def generate(self, prompt: str, schema: BaseModel | None = None):
        """Generate a response for ``prompt``, optionally as a Pydantic model.

        Parameters
        ----------
        prompt:
            User prompt to send.
        schema:
            Optional Pydantic model; when set, the response is parsed into it.

        Returns
        -------
        str or ``schema``
            The raw text answer, or a parsed instance of ``schema`` if provided.
        """
        messages, kwargs = _build_messages(prompt, schema)
        answer = _call_with_retry(
            self._client.chat.completions.create,
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
            **kwargs,
        )

        content = answer.choices[0].message.content
        if schema:
            return schema(**json.loads(content))
        return content

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        """Generate asynchronously, optionally as a Pydantic model.

        Parameters
        ----------
        prompt:
            User prompt to send.
        schema:
            Optional Pydantic model; when set, the response is parsed into it.

        Returns
        -------
        str or ``schema``
            The raw text answer, or a parsed instance of ``schema`` if provided.
        """
        messages, kwargs = _build_messages(prompt, schema)
        answer = await _acall_with_retry(
            self._a_client.chat.completions.create,
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
            **kwargs,
        )

        content = answer.choices[0].message.content
        if schema:
            return schema(**json.loads(content))
        return content

    def get_model_name(self):
        """Return the display name of the model."""
        return f"{self.model} (custom)"


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """OpenAI-compatible embedding wrapper with exponential-backoff retries.

    Attributes
    ----------
    model:
        Embedding model identifier passed to the provider API.
    base_url:
        Base URL of the provider API.
    api_env:
        Name of the environment variable holding the API key.
    max_retries:
        Maximum number of retry attempts on retryable errors.
    min_sleep:
        Base delay (seconds) for exponential backoff.
    max_delay:
        Upper bound (seconds) on the backoff delay.
    """

    def __init__(
        self,
        model: str,
        base_url,
        api_env,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_sleep: float = DEFAULT_MIN_SLEEP,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        """Initialize the embedding wrapper and its sync/async clients.

        Parameters
        ----------
        model:
            Embedding model identifier passed to the provider API.
        base_url:
            Base URL of the provider API.
        api_env:
            Name of the environment variable holding the API key.
        max_retries:
            Maximum number of retry attempts on retryable errors.
        min_sleep:
            Base delay (seconds) for exponential backoff.
        max_delay:
            Upper bound (seconds) on the backoff delay.
        """
        self.model = model
        super().__init__(model)
        self.base_url = base_url
        self.api_env = api_env
        self.max_retries = max_retries
        self.min_sleep = min_sleep
        self.max_delay = max_delay
        self._client = init_client(self.base_url, self.api_env)
        self._a_client = init_async_client(self.base_url, self.api_env)

    @classmethod
    def from_provider(cls, cfg: ProviderConfig):
        """Build an embedding wrapper from a :class:`ProviderConfig`.

        Parameters
        ----------
        cfg:
            Provider configuration to instantiate from.

        Returns
        -------
        CustomEmbeddingModel
            Configured embedding wrapper.
        """
        return cls(
            model=cfg.embedding,
            base_url=cfg.base_url,
            api_env=cfg.api_env,
            max_retries=cfg.max_retries,
            min_sleep=cfg.min_sleep,
            max_delay=cfg.max_delay,
        )

    def load_model(self):
        """Return the embedding model identifier expected by deepeval."""
        return self.model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text.

        Parameters
        ----------
        text:
            Text to embed.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        response = _call_with_retry(
            self._client.embeddings.create,
            input=text,
            model=self.model,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Parameters
        ----------
        texts:
            Texts to embed.

        Returns
        -------
        list[list[float]]
            The embedding vectors, one per input text.
        """
        response = _call_with_retry(
            self._client.embeddings.create,
            input=texts,
            model=self.model,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
        )
        return [item.embedding for item in response.data]

    async def a_embed_text(self, text: str) -> list[float]:
        """Embed a single text asynchronously.

        Parameters
        ----------
        text:
            Text to embed.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        response = await _acall_with_retry(
            self._a_client.embeddings.create,
            input=text,
            model=self.model,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
        )
        return response.data[0].embedding

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts asynchronously.

        Parameters
        ----------
        texts:
            Texts to embed.

        Returns
        -------
        list[list[float]]
            The embedding vectors, one per input text.
        """
        response = await _acall_with_retry(
            self._a_client.embeddings.create,
            input=texts,
            model=self.model,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
        )
        return [item.embedding for item in response.data]

    def get_model_name(self):
        """Return the display name of the model."""
        return f"{self.model} (custom)"


if __name__ == "__main__":
    import asyncio

    from .config import get_provider

    cfg = get_provider("albert")
    llm = CustomOpenAI.from_provider(cfg)
    res = asyncio.run(llm.a_generate(prompt="What's the capital of Canada"))
    print(res)
