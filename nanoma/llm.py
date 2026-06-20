"""LLM abstraction: OpenAI-compatible and Anthropic-compatible clients."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from nanoma.cost import UsageRecord

logger = logging.getLogger("nanoma")

Message = dict[str, Any]
ToolDef = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: UsageRecord = field(default_factory=UsageRecord)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    http_timeout: float = field(default_factory=lambda: _default_http_timeout())


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def count_message_tokens(messages: list[Message]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        if isinstance(content, str):
            total += estimate_tokens(content)
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                total += estimate_tokens(json.dumps(tc.get("function", {}).get("arguments", "")))
    return total


# --- LLM logging ---

_log_dir: Path | None = None
_log_counter: int = 0


def set_log_dir(path: Path | str):
    global _log_dir
    _log_dir = Path(path)
    _log_dir.mkdir(parents=True, exist_ok=True)


# --- Main call ---

_RETRYABLE = {429, 500, 502, 503, 504}
_shared_client: httpx.AsyncClient | None = None


def _default_max_tokens() -> int:
    raw = os.environ.get("NANOMA_MAX_TOKENS")
    if not raw:
        return 16384
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid NANOMA_MAX_TOKENS=%r; using 16384", raw)
        return 16384
    return max(1, value)


def _default_http_timeout() -> float:
    raw = os.environ.get("NANOMA_HTTP_TIMEOUT")
    if not raw:
        return 180.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid NANOMA_HTTP_TIMEOUT=%r; using 180", raw)
        return 180.0
    return max(1.0, value)


def _omit_sampling_params() -> bool:
    return os.environ.get("NANOMA_OMIT_SAMPLING_PARAMS", "").strip().lower() in {"1", "true", "yes", "on"}


def _get_client(timeout: float = 180.0) -> httpx.AsyncClient:
    """Reuse a single httpx client across all LLM calls (connection pooling)."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=100))
    return _shared_client


def _discard_client():
    """Stop reusing the pooled client after transport-level failures.

    Do not close it here: other concurrent agent requests may still be using the
    same client object.
    """
    global _shared_client
    _shared_client = None


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _openai_tool_to_anthropic(tool: ToolDef) -> dict[str, Any]:
    fn = tool.get("function", tool)
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _openai_messages_to_anthropic(messages: list[Message]) -> tuple[str, list[Message]]:
    """Convert NanoMA's OpenAI-style history to Anthropic Messages format."""
    system_parts: list[str] = []
    out: list[Message] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            text = _message_content_to_text(msg.get("content"))
            if text:
                system_parts.append(text)
            continue

        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            text = _message_content_to_text(msg.get("content"))
            if text and text != "(empty)":
                content_blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args if isinstance(args, dict) else {"_raw": args},
                })
            if content_blocks:
                out.append({"role": "assistant", "content": content_blocks})
            continue

        if role == "tool":
            content = _message_content_to_text(msg.get("content"))
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                }],
            })
            continue

        # Anthropic only accepts user/assistant messages. Runtime messages from
        # inbox delivery are already user messages; unknown roles are safest as user.
        text = _message_content_to_text(msg.get("content"))
        if text:
            out.append({"role": "assistant" if role == "assistant" else "user", "content": text})

    # Anthropic rejects messages that contain only system content. NanoMA stores
    # the task in the system prompt, so add a small user turn to start the loop.
    if not any(m.get("role") == "user" for m in out):
        out.append({"role": "user", "content": "Begin the task now."})

    return "\n\n".join(system_parts), out


def _anthropic_stop_reason_to_openai(reason: str | None) -> str:
    if reason == "tool_use":
        return "tool_calls"
    if reason == "max_tokens":
        return "length"
    return "stop"


def _is_empty_bad_response(data: dict[str, Any]) -> bool:
    try:
        choice = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return False
    usage = data.get("usage", {})
    return (
        not choice.get("content")
        and not choice.get("tool_calls")
        and usage.get("total_tokens", 0) == 0
    )


async def openai_compatible_call(
    messages: list[Message],
    model: str,
    tools: list[ToolDef] | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int | None = None,
    retry_config: RetryConfig | None = None,
) -> LLMResponse:
    base_url = base_url or os.environ.get("NANOMA_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("NANOMA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No API key. Set NANOMA_API_KEY or OPENAI_API_KEY.")
    rc = retry_config or RetryConfig()

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body: dict[str, Any] = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens or _default_max_tokens(),
    }
    if not _omit_sampling_params():
        body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    t0 = time.time()
    client = _get_client(rc.http_timeout)
    last_err: Exception | None = None
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.3))
                continue
            resp.raise_for_status()
            data = resp.json()
            if _is_empty_bad_response(data) and attempt < rc.max_retries:
                delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.3))
                continue
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            if attempt == rc.max_retries:
                raise
            if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
                _discard_client()
                client = _get_client(rc.http_timeout)
            await asyncio.sleep(rc.base_delay * (2 ** attempt))
    else:
        raise last_err or RuntimeError("Retry exhausted")

    elapsed_ms = (time.time() - t0) * 1000

    # Log
    if _log_dir:
        global _log_counter
        _log_counter += 1
        safe_model = model.replace("/", "_").replace(":", "_")
        log_path = _log_dir / f"{_log_counter:05d}_{safe_model}.jsonl"
        try:
            log_path.write_text(json.dumps({"request": body, "response": data, "ms": elapsed_ms}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM log {log_path}: {e}")

    choice = data["choices"][0]["message"]
    usage_data = data.get("usage", {})
    cached = 0
    if pd := usage_data.get("prompt_tokens_details"):
        cached = pd.get("cached_tokens", 0)

    usage = UsageRecord(
        input_tokens=usage_data.get("prompt_tokens", 0),
        cached_input_tokens=cached,
        output_tokens=usage_data.get("completion_tokens", 0),
        model=model,
    )

    tool_calls = []
    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": fn["arguments"]}
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=args))

    return LLMResponse(content=choice.get("content"), tool_calls=tool_calls, usage=usage, raw=data)


async def anthropic_compatible_call(
    messages: list[Message],
    model: str,
    tools: list[ToolDef] | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int | None = None,
    retry_config: RetryConfig | None = None,
) -> LLMResponse:
    base_url = base_url or os.environ.get("NANOMA_LLM_BASE_URL", "https://api.anthropic.com/v1")
    api_key = api_key or os.environ.get("NANOMA_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No API key. Set NANOMA_API_KEY or ANTHROPIC_API_KEY.")
    rc = retry_config or RetryConfig()

    system, anthropic_messages = _openai_messages_to_anthropic(messages)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": os.environ.get("NANOMA_ANTHROPIC_VERSION", "2023-06-01"),
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens or _default_max_tokens(),
    }
    if not _omit_sampling_params():
        body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [_openai_tool_to_anthropic(t) for t in tools]
        body["tool_choice"] = {"type": "auto"}

    t0 = time.time()
    client = _get_client(rc.http_timeout)
    last_err: Exception | None = None
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/messages", headers=headers, json=body)
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.3))
                continue
            resp.raise_for_status()
            data = resp.json()
            if (
                not data.get("content")
                and data.get("usage", {}).get("input_tokens", 0) == 0
                and data.get("usage", {}).get("output_tokens", 0) == 0
                and attempt < rc.max_retries
            ):
                delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.3))
                continue
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            if attempt == rc.max_retries:
                raise
            if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
                _discard_client()
                client = _get_client(rc.http_timeout)
            await asyncio.sleep(rc.base_delay * (2 ** attempt))
    else:
        raise last_err or RuntimeError("Retry exhausted")

    elapsed_ms = (time.time() - t0) * 1000

    if _log_dir:
        global _log_counter
        _log_counter += 1
        safe_model = model.replace("/", "_").replace(":", "_")
        log_path = _log_dir / f"{_log_counter:05d}_{safe_model}.jsonl"
        try:
            log_path.write_text(json.dumps({"request": body, "response": data, "ms": elapsed_ms}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM log {log_path}: {e}")

    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []) or []:
        if block.get("type") == "text":
            content_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            args = block.get("input", {})
            tool_calls.append(ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=args if isinstance(args, dict) else {"_raw": args},
            ))

    usage_data = data.get("usage", {})
    usage = UsageRecord(
        input_tokens=usage_data.get("input_tokens", 0),
        cached_input_tokens=usage_data.get("cache_read_input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        model=model,
    )

    # Keep raw shape recognizable for tooling that expects OpenAI-ish metadata.
    raw = {
        **data,
        "_openai_compatible": {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "\n".join(p for p in content_parts if p),
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in tool_calls
                    ],
                },
                "finish_reason": _anthropic_stop_reason_to_openai(data.get("stop_reason")),
            }]
        },
    }

    return LLMResponse(
        content="\n".join(p for p in content_parts if p) or None,
        tool_calls=tool_calls,
        usage=usage,
        raw=raw,
    )


def default_llm_call(*args: Any, **kwargs: Any):
    protocol = os.environ.get("NANOMA_LLM_PROTOCOL", "openai").strip().lower()
    if protocol in {"anthropic", "anthropic-compatible", "messages"}:
        return anthropic_compatible_call(*args, **kwargs)
    return openai_compatible_call(*args, **kwargs)


def default_router(task: str, budget: float, allowed_models: list[str] | None = None) -> str:
    from nanoma.models import get_registry
    return get_registry().route(budget, allowed=allowed_models)
