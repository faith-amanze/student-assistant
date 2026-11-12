"""
llm.py - the ONLY file that knows about specific AI providers.

The rest of the app (main.py) calls two functions:
    stream_chat(messages)  -> yields text pieces as they arrive
    complete(messages, json_mode=False) -> returns the full reply as text

Both Mistral and OpenAI offer an "OpenAI-style" chat HTTP API, so the
differences between them are just a URL, an API key and a model name.
We keep those in the PROVIDERS table below. Adding a new provider
(e.g. Groq) is a matter of adding one more entry.
"""
import asyncio
import json
import os
import random
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv

load_dotenv()  # reads the .env file so os.getenv() can see your keys

# ---------------------------------------------------------------- providers
PROVIDERS = {
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key_env": "MISTRAL_API_KEY",
        "model_env": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-20b",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
}


# ------------------------------------------------------------------- errors
class LLMError(Exception):
    """Something went wrong talking to the AI provider."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(LLMError):
    """Provider kept answering 429 (too many requests) even after retries."""

    def __init__(self, message: str):
        super().__init__(message, status_code=429)


# ------------------------------------------------------------------- config
def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "mistral").lower().strip()


def _config():
    """Look up the current provider's settings (read fresh on every call)."""
    name = provider_name()
    if name not in PROVIDERS:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{name}'. Choose one of: {', '.join(PROVIDERS)}",
            status_code=500,
        )
    cfg = PROVIDERS[name]
    api_key = os.getenv(cfg["key_env"])
    if not api_key:
        raise LLMError(f"{cfg['key_env']} is not set. Add it to your .env file.", 500)
    model = os.getenv(cfg["model_env"], cfg["default_model"])
    return name, cfg["url"], api_key, model


# ------------------------------------------------------------- http client
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """One shared async HTTP client (re-using connections is faster)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ------------------------------------------------ retry / backoff on 429
def _wait_time(response: httpx.Response, attempt: int) -> float:
    """How long to sleep before retrying.

    1. If the provider sent a 'Retry-After' header, trust it.
    2. Otherwise use exponential backoff: 1s, 2s, 4s, 8s... plus a little
       random "jitter" so many clients don't all retry at the same instant.
    """
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass
    base = float(os.getenv("LLM_BACKOFF_BASE", "1.0"))
    return min(base * (2 ** attempt), 30.0) + random.uniform(0, base * 0.25)


async def _send(payload: dict, stream: bool) -> httpx.Response:
    """Send the request. If we get a 429, wait and try again."""
    name, url, api_key, model = _config()
    payload = {"model": model, **payload}
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
    client = get_client()

    for attempt in range(max_retries + 1):
        request = client.build_request(
            "POST",
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            response = await client.send(request, stream=stream)
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach {name}: {exc}", status_code=503)

        if response.status_code == 429:
            await response.aclose()
            if attempt == max_retries:
                raise RateLimitError(
                    f"{name} is rate-limiting us. Gave up after {max_retries} retries."
                )
            delay = _wait_time(response, attempt)
            print(f"[llm] 429 from {name}; retry {attempt + 1}/{max_retries} in {delay:.1f}s")
            await asyncio.sleep(delay)  # async sleep: other requests keep running
            continue

        if response.status_code >= 400:
            body = (await response.aread()).decode(errors="replace")[:300]
            await response.aclose()
            raise LLMError(f"{name} returned {response.status_code}: {body}", 502)

        return response

    raise LLMError("Unreachable")  # pragma: no cover


# ------------------------------------------------------------ public API
async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    """Yield the model's answer a few words at a time."""
    response = await _send({"messages": messages, "stream": True}, stream=True)
    try:
        async for line in response.aiter_lines():
            # Streaming uses "Server-Sent Events": lines like  data: {...json...}
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = (choices[0].get("delta") or {}).get("content")
            if text:
                yield text
    finally:
        await response.aclose()


async def complete(messages: list[dict], json_mode: bool = False) -> str:
    """Return the model's full answer. With json_mode=True the model must reply in JSON."""
    payload: dict = {"messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = await _send(payload, stream=False)
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected response format from provider: {exc}", 502)
