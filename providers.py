import asyncio
import json
import os
import time

from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from config import (
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_SLEEP,
    DEFAULT_TEMPERATURE,
    ProviderConfig,
)

RATE_LIMIT_STATUS = 429


class RequestLimitError(RuntimeError):
    pass


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == RATE_LIMIT_STATUS:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


def _retry_delay(attempt: int, min_sleep: float, max_delay: float) -> float:
    return min(min_sleep * (2 ** attempt), max_delay)


def _sleep_with_backoff(attempt: int, min_sleep: float, max_delay: float):
    time.sleep(_retry_delay(attempt, min_sleep, max_delay))


async def _async_sleep_with_backoff(attempt: int, min_sleep: float, max_delay: float):
    await asyncio.sleep(_retry_delay(attempt, min_sleep, max_delay))


def _call_with_retry(call, *args, max_retries: int = DEFAULT_MAX_RETRIES, min_sleep: float = DEFAULT_MIN_SLEEP, max_delay: float = DEFAULT_MAX_DELAY, **kwargs):
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


async def _acall_with_retry(call, *args, max_retries: int = DEFAULT_MAX_RETRIES, min_sleep: float = DEFAULT_MIN_SLEEP, max_delay: float = DEFAULT_MAX_DELAY, **kwargs):
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
    return OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, 'ollama'))


def init_async_client(base_url, api_key_env: str):
    return AsyncOpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, 'ollama'))


def _deref(schema_obj: dict, json_schema: dict) -> dict:
    ref = schema_obj.get("$ref")
    if ref:
        name = ref.split("/")[-1]
        return json_schema.get("$defs", {}).get(
            name, json_schema.get("definitions", {}).get(name, {})
        )
    return schema_obj


def _pick_type(schema_obj: dict) -> dict:
    for key in ("anyOf", "oneOf"):
        options = schema_obj.get(key) or []
        for opt in options:
            if opt.get("type") != "null":
                return opt
    return schema_obj


def _type_example(prop: dict, json_schema: dict):
    prop = _deref(prop, json_schema)
    prop = _pick_type(prop)
    ptype = prop.get("type")
    if "properties" in prop or "allOf" in prop or prop.get("$ref"):
        return _object_example(prop, json_schema)
    if ptype == "array":
        items = _deref(prop.get("items", {}), json_schema)
        items = _pick_type(items)
        return (
            [_object_example(items, json_schema)] if "properties" in items else [""]
        )
    if ptype == "boolean":
        return True
    if ptype in ("integer", "number"):
        return 0
    return ""


def _object_example(field_schema: dict, json_schema: dict):
    merged = {}
    for key in ("allOf",):
        for part in field_schema.get(key) or []:
            merged.update(_object_example(_deref(part, json_schema), json_schema))
    field_schema = {**field_schema, **merged}
    props = field_schema.get("properties", {})
    return {name: _type_example(sub, json_schema) for name, sub in props.items()}


def _schema_hint(schema: BaseModel) -> str:
    json_schema = schema.model_json_schema()
    if json_schema.get("type") == "array":
        example = [_object_example(json_schema.get("items", {}), json_schema)]
    else:
        example = _object_example(json_schema, json_schema)
    return json.dumps(example, ensure_ascii=False)


def _build_messages(prompt: str, schema: BaseModel | None):
    if schema is None:
        return [{'role': 'user', 'content': prompt}], {}
    hint = _schema_hint(schema)
    structured_prompt = (
        f"{prompt}\n\n"
        "Return ONLY a single JSON object (no markdown, no extra text) that matches "
        f"exactly this structure, filling in the values:\n{hint}"
    )
    return (
        [{'role': 'user', 'content': structured_prompt}],
        {"response_format": {"type": "json_object"}},
    )


class CustomOpenAI(DeepEvalBaseLLM):
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
        return self.model

    def generate(self, prompt: str, schema: BaseModel | None = None):
        messages, kwargs = _build_messages(prompt, schema)
        answer = _call_with_retry(
            self._client.chat.completions.create,
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
            **kwargs)

        content = answer.choices[0].message.content
        if schema:
            return schema(**json.loads(content))
        return content

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        messages, kwargs = _build_messages(prompt, schema)
        answer = await _acall_with_retry(
            self._a_client.chat.completions.create,
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_retries=self.max_retries,
            min_sleep=self.min_sleep,
            max_delay=self.max_delay,
            **kwargs)

        content = answer.choices[0].message.content
        if schema:
            return schema(**json.loads(content))
        return content

    def get_model_name(self):
        return f"{self.model} (custom)"


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model: str,
        base_url,
        api_env,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_sleep: float = DEFAULT_MIN_SLEEP,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
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
        return cls(
            model=cfg.embedding,
            base_url=cfg.base_url,
            api_env=cfg.api_env,
            max_retries=cfg.max_retries,
            min_sleep=cfg.min_sleep,
            max_delay=cfg.max_delay,
        )

    def load_model(self):
        return self.model

    def embed_text(self, text: str) -> list[float]:
        response = _call_with_retry(
            self._client.embeddings.create, input=text, model=self.model,
            max_retries=self.max_retries, min_sleep=self.min_sleep, max_delay=self.max_delay)
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = _call_with_retry(
            self._client.embeddings.create, input=texts, model=self.model,
            max_retries=self.max_retries, min_sleep=self.min_sleep, max_delay=self.max_delay)
        return [item.embedding for item in response.data]

    async def a_embed_text(self, text: str) -> list[float]:
        response = await _acall_with_retry(
            self._a_client.embeddings.create, input=text, model=self.model,
            max_retries=self.max_retries, min_sleep=self.min_sleep, max_delay=self.max_delay)
        return response.data[0].embedding

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await _acall_with_retry(
            self._a_client.embeddings.create, input=texts, model=self.model,
            max_retries=self.max_retries, min_sleep=self.min_sleep, max_delay=self.max_delay)
        return [item.embedding for item in response.data]

    def get_model_name(self):
        return f"{self.model} (custom)"


if __name__ == "__main__":
    import asyncio

    from config import get_provider
    cfg = get_provider("albert")
    llm = CustomOpenAI.from_provider(cfg)
    res = asyncio.run(llm.a_generate(prompt="What's the capital of Canada"))
    print(res)
