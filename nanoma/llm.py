"""LLM abstraction: OpenAI-compatible client with retry."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from nanoma.cost import UsageRecord
from nanoma.env import load_dotenv

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
    max_retries: int = 8
    base_delay: float = 2.0
    max_delay: float = 90.0
    http_timeout: float = 300.0  # seconds; LLM calls can be slow with large contexts


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


def _get_client(timeout: float = 180.0) -> httpx.AsyncClient:
    """Reuse a single httpx client across all LLM calls (connection pooling)."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        max_connections = int(os.environ.get("NANOMA_HTTP_MAX_CONNECTIONS", "100"))
        _shared_client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=max_connections))
    return _shared_client


async def openai_compatible_call(
    messages: list[Message],
    model: str,
    tools: list[ToolDef] | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    retry_config: RetryConfig | None = None,
) -> LLMResponse:
    load_dotenv()
    base_url = base_url or os.environ.get("NANOMA_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("NANOMA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No API key. Set NANOMA_API_KEY or OPENAI_API_KEY.")
    rc = retry_config or RetryConfig()

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = int(os.environ.get("NANOMA_MAX_TOKENS", "8192"))

    body: dict[str, Any] = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": resolved_max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"

    t0 = time.time()
    client = _get_client(rc.http_timeout)
    last_err: Exception | None = None
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                delay = _retry_delay(resp, rc, attempt)
                logger.warning("Retrying LLM call after HTTP %s in %.1fs (attempt %s/%s)", resp.status_code, delay, attempt + 1, rc.max_retries)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last_err = e
            if attempt == rc.max_retries:
                raise
            delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
            delay = delay + random.uniform(0, delay * 0.5)
            logger.warning("Retrying LLM call after %s in %.1fs (attempt %s/%s)", type(e).__name__, delay, attempt + 1, rc.max_retries)
            await asyncio.sleep(delay)
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
    allowed_tool_names = _tool_names_from_schemas(tools or [])
    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": fn["arguments"]}
            name = fn["name"]
            if not allowed_tool_names or name in allowed_tool_names:
                tool_calls.append(ToolCall(id=tc["id"], name=name, arguments=args))
    elif choice.get("content"):
        tool_calls = _parse_text_tool_calls(choice.get("content") or "", allowed_tool_names)

    return LLMResponse(content=choice.get("content"), tool_calls=tool_calls, usage=usage, raw=data)


def _tool_names_from_schemas(tools: list[ToolDef]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if name:
            names.add(str(name))
    return names


_DSML_INVOKE_RE = re.compile(r"<｜DSML｜invoke\s+([^>]*)>(.*?)</｜DSML｜invoke>", re.DOTALL)
_DSML_PARAM_RE = re.compile(r"<｜DSML｜parameter\s+([^>]*)>(.*?)</｜DSML｜parameter>", re.DOTALL)
_ATTR_RE = re.compile(r'([A-Za-z_][\w:-]*)="([^"]*)"')


def _parse_text_tool_calls(content: str, allowed_tool_names: set[str] | None = None) -> list[ToolCall]:
    """Parse provider-specific DSML text tool calls when standard tool_calls is empty."""
    if "<｜DSML｜invoke" not in content:
        return []
    allowed = allowed_tool_names or set()
    calls: list[ToolCall] = []
    for idx, match in enumerate(_DSML_INVOKE_RE.finditer(content), start=1):
        attrs = _parse_dsml_attrs(match.group(1))
        name = attrs.get("name", "")
        if not name:
            continue
        if allowed and name not in allowed:
            continue
        args: dict[str, Any] = {}
        for param_match in _DSML_PARAM_RE.finditer(match.group(2)):
            param_attrs = _parse_dsml_attrs(param_match.group(1))
            param_name = param_attrs.get("name")
            if not param_name:
                continue
            args[param_name] = _parse_dsml_value(param_match.group(2), param_attrs)
        calls.append(ToolCall(id=f"text_tool_{idx}", name=name, arguments=args))
    return calls


def _parse_dsml_attrs(text: str) -> dict[str, str]:
    return {key: html.unescape(value) for key, value in _ATTR_RE.findall(text)}


def _parse_dsml_value(raw: str, attrs: dict[str, str]) -> Any:
    value = html.unescape(raw)
    if attrs.get("json") == "true":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if attrs.get("number") == "true":
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    if attrs.get("boolean") == "true":
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return value


def _retry_delay(resp: httpx.Response, rc: RetryConfig, attempt: int) -> float:
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), rc.max_delay)
        except ValueError:
            pass
    delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
    return delay + random.uniform(0, delay * 0.5)


def default_router(task: str, budget: float, allowed_models: list[str] | None = None) -> str:
    from nanoma.models import get_registry
    return get_registry().route(budget, allowed=allowed_models)
