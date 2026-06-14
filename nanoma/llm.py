"""LLM abstraction: OpenAI-compatible client with retry."""

from __future__ import annotations

import asyncio
import copy
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


class TransientEmptyLLMResponse(RuntimeError):
    """Provider returned HTTP 200 with no content, no tools, and zero usage."""


class TransientToolCallTransportMiss(RuntimeError):
    """Provider returned text instead of tool calls with zero usage on a tool turn."""


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


def _write_llm_log(model: str, payload: dict[str, Any]) -> None:
    if not _log_dir:
        return
    global _log_counter
    _log_counter += 1
    safe_model = model.replace("/", "_").replace(":", "_")
    log_path = _log_dir / f"{_log_counter:05d}_{safe_model}.jsonl"
    try:
        log_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write LLM log {log_path}: {e}")


def _response_text_preview(resp: httpx.Response, limit: int = 2000) -> str:
    try:
        text = resp.text
    except Exception:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _response_text_len(resp: httpx.Response) -> int | None:
    try:
        return len(resp.text)
    except Exception:
        return None


def _response_headers_preview(resp: httpx.Response) -> dict[str, str]:
    allowed = {
        "content-type",
        "content-length",
        "date",
        "retry-after",
        "server",
        "x-request-id",
        "x-correlation-id",
        "x-trace-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    }
    prefixes = ("cf-", "x-ratelimit-", "x-request-", "x-gateway-", "x-upstream-")
    headers: dict[str, str] = {}
    for key, value in resp.headers.items():
        lower = key.lower()
        if lower in allowed or lower.startswith(prefixes):
            headers[lower] = value
    return headers


def _http_response_log_payload(
    body: dict[str, Any],
    resp: httpx.Response,
    *,
    attempt: int,
    retry: bool,
    error: str,
    elapsed_ms: float,
    exception: str | None = None,
) -> dict[str, Any]:
    request = resp.request
    payload: dict[str, Any] = {
        "request": body,
        "request_method": request.method if request else "",
        "request_url": str(request.url) if request else "",
        "response_status": resp.status_code,
        "response_reason": resp.reason_phrase,
        "response_headers": _response_headers_preview(resp),
        "response_text": _response_text_preview(resp),
        "response_text_len": _response_text_len(resp),
        "attempt": attempt,
        "retry": retry,
        "error": error,
        "ms": elapsed_ms,
    }
    if exception:
        payload["exception"] = exception
    try:
        payload["response_json"] = resp.json()
    except Exception:
        pass
    return payload


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

    request_tools = _adapt_tool_schemas_for_model(tools or [], model)
    request_tool_choice = _adapt_tool_choice_for_model(tool_choice or "auto", model) if request_tools else None
    request_messages = _adapt_messages_for_schema_only_tool_turn(
        messages,
        model=model,
        tools=request_tools,
        requested_tool_choice=tool_choice,
    )

    body: dict[str, Any] = {
        "model": model, "messages": request_messages,
        "temperature": temperature, "max_tokens": resolved_max_tokens,
    }
    if request_tools:
        body["tools"] = request_tools
        body["tool_choice"] = request_tool_choice

    t0 = time.time()
    client = _get_client(rc.http_timeout)
    last_err: Exception | None = None
    repair_applied = False
    schema_tool_repair_count = 0
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                _write_llm_log(model, _http_response_log_payload(
                    body,
                    resp,
                    attempt=attempt + 1,
                    retry=True,
                    error=f"HTTP {resp.status_code}",
                    elapsed_ms=(time.time() - t0) * 1000,
                ))
                delay = _retry_delay(resp, rc, attempt)
                logger.warning("Retrying LLM call after HTTP %s in %.1fs (attempt %s/%s)", resp.status_code, delay, attempt + 1, rc.max_retries)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            try:
                _raise_for_transient_empty_response(data)
                _raise_for_transient_tool_call_transport_miss(data, model=model, tools=request_tools)
            except (TransientEmptyLLMResponse, TransientToolCallTransportMiss) as e:
                _write_llm_log(model, {
                    "request": body,
                    "response": data,
                    "attempt": attempt + 1,
                    "retry": attempt < rc.max_retries,
                    "error": type(e).__name__,
                    "ms": (time.time() - t0) * 1000,
                })
                raise
            schema_tool_miss = _schema_only_tool_choice_miss(
                data,
                model=model,
                tools=request_tools,
                requested_tool_choice=tool_choice,
                repair_count=schema_tool_repair_count,
            )
            if schema_tool_miss and attempt < rc.max_retries:
                target_tool = schema_tool_miss
                _write_llm_log(model, {
                    "request": body,
                    "response": data,
                    "attempt": attempt + 1,
                    "retry": True,
                    "error": "SchemaOnlyToolChoiceMiss",
                    "target_tool": target_tool,
                    "repair_count": schema_tool_repair_count + 1,
                    "ms": (time.time() - t0) * 1000,
                })
                schema_tool_repair_count += 1
                body = _body_with_schema_only_tool_retry_repair(body, target_tool, schema_tool_repair_count)
                delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
                delay = delay + random.uniform(0, delay * 0.5)
                logger.warning(
                    "Retrying LLM call after SchemaOnlyToolChoiceMiss for %s in %.1fs (attempt %s/%s)",
                    target_tool,
                    delay,
                    attempt + 1,
                    rc.max_retries,
                )
                await asyncio.sleep(delay)
                continue
            break
        except (
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.HTTPStatusError,
            TransientEmptyLLMResponse,
            TransientToolCallTransportMiss,
        ) as e:
            last_err = e
            retry = attempt < rc.max_retries
            if isinstance(e, httpx.HTTPStatusError):
                retry = retry and e.response.status_code in _RETRYABLE
            if isinstance(e, httpx.HTTPStatusError):
                _write_llm_log(model, _http_response_log_payload(
                    body,
                    e.response,
                    attempt=attempt + 1,
                    retry=retry,
                    error="HTTPStatusError",
                    elapsed_ms=(time.time() - t0) * 1000,
                    exception=str(e),
                ))
            if not retry:
                raise
            if _should_repair_tool_retry(e, model=model, tools=request_tools) and not repair_applied:
                body = _body_with_tool_retry_repair(body)
                repair_applied = True
            delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
            delay = delay + random.uniform(0, delay * 0.5)
            if isinstance(e, httpx.HTTPStatusError):
                logger.warning(
                    "Retrying LLM call after HTTPStatusError %s in %.1fs (attempt %s/%s)",
                    e.response.status_code,
                    delay,
                    attempt + 1,
                    rc.max_retries,
                )
            else:
                logger.warning("Retrying LLM call after %s in %.1fs (attempt %s/%s)", type(e).__name__, delay, attempt + 1, rc.max_retries)
            await asyncio.sleep(delay)
    else:
        raise last_err or RuntimeError("Retry exhausted")

    elapsed_ms = (time.time() - t0) * 1000

    _write_llm_log(model, {"request": body, "response": data, "ms": elapsed_ms})

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
                try:
                    args = json.loads(fn["arguments"], strict=False)
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": fn["arguments"]}
            name = fn["name"]
            if not allowed_tool_names or name in allowed_tool_names:
                tool_calls.append(ToolCall(id=tc["id"], name=name, arguments=args))
    elif choice.get("content"):
        tool_calls = _parse_text_tool_calls(choice.get("content") or "", allowed_tool_names)

    return LLMResponse(content=choice.get("content"), tool_calls=tool_calls, usage=usage, raw=data)


_REQUIRED_ARG_TOOL_COMPAT: dict[str, list[str]] = {
    "file_list": ["path"],
    "spawn_many": ["agents"],
    "query": ["filter"],
    "wait": ["agent_ids"],
    "set_status": ["action"],
    "bt_aggregate": ["directory"],
}

_SET_STATUS_ACTIONS = ["compact", "create", "done", "message", "read", "self_stop", "stop", "work"]


def _model_needs_required_arg_tool_schema(model: str) -> bool:
    mode = os.environ.get("NANOMA_LLM_TOOL_SCHEMA_COMPAT", "auto").strip().lower()
    if mode in {"0", "false", "off", "none"}:
        return False
    if mode in {"1", "true", "required_args"}:
        return True
    normalized = (model or "").strip().lower().replace("_", "-")
    return (
        normalized.endswith("deepseek-v4-pro")
        or normalized.endswith("/deepseek-v4-pro")
        or "deepseek-v4-pro-" in normalized
    )


def _adapt_tool_schemas_for_model(tools: list[ToolDef], model: str) -> list[ToolDef]:
    """Apply provider-specific schema compatibility without mutating global tool defs."""
    if not tools or not _model_needs_required_arg_tool_schema(model):
        return tools
    adapted = copy.deepcopy(tools)
    for tool in adapted:
        function = tool.get("function") or {}
        name = function.get("name")
        required = _REQUIRED_ARG_TOOL_COMPAT.get(str(name))
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if required:
            parameters["required"] = required
        elif "required" not in parameters and isinstance(properties, dict):
            parameters["required"] = []
        if name == "set_status":
            if isinstance(properties, dict) and isinstance(properties.get("action"), dict):
                properties["action"]["enum"] = list(_SET_STATUS_ACTIONS)
    return adapted


def _adapt_tool_choice_for_model(
    tool_choice: str | dict[str, Any] | None,
    model: str,
) -> str | dict[str, Any] | None:
    """DeepSeek pro thinking mode accepts tool schemas but rejects named tool_choice."""
    if isinstance(tool_choice, dict) and _model_needs_required_arg_tool_schema(model):
        return "auto"
    return tool_choice


def _named_tool_choice_name(tool_choice: str | dict[str, Any] | None) -> str:
    if not isinstance(tool_choice, dict):
        return ""
    function = tool_choice.get("function") or {}
    name = function.get("name")
    return str(name or "")


def _adapt_messages_for_schema_only_tool_turn(
    messages: list[Message],
    *,
    model: str,
    tools: list[ToolDef],
    requested_tool_choice: str | dict[str, Any] | None,
) -> list[Message]:
    if os.environ.get("NANOMA_SCHEMA_ONLY_SANITIZE_HISTORY", "1").strip().lower() in {"0", "false", "off"}:
        return messages
    if not tools or not _model_needs_required_arg_tool_schema(model):
        return messages
    target_tool = _named_tool_choice_name(requested_tool_choice)
    if not target_tool:
        return messages
    allowed = _tool_names_from_schemas(tools)
    if target_tool not in allowed:
        return messages

    sanitized: list[Message] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            summary = f"[Prior tool calls omitted for current schema-only `{target_tool}` turn.]"
            content = str(msg.get("content") or "").strip()
            if content:
                summary = f"{summary}\n{_sanitize_schema_only_text(content, target_tool)}"
            sanitized.append({"role": "assistant", "content": summary})
            continue
        if role == "tool":
            sanitized.append({
                "role": "user",
                "content": f"[Prior tool result omitted for current schema-only `{target_tool}` turn.]",
            })
            continue
        copied = dict(msg)
        if isinstance(copied.get("content"), str):
            copied["content"] = _sanitize_schema_only_text(str(copied.get("content") or ""), target_tool)
        sanitized.append(copied)
    return sanitized


def _sanitize_schema_only_text(text: str, target_tool: str) -> str:
    allowed = {target_tool, "tool", "tools", "tool_call", "tool_calls", "schema", "function"}
    known_tools = {
        "bash", "bt_aggregate", "compact", "create_agent", "file_list", "file_read",
        "file_replace", "file_write", "get_cost", "grep", "kill", "query", "rebirth",
        "send", "set_bio", "set_status", "shell", "spawn", "spawn_many", "submit",
        "transfer", "wait", "web_search",
    }
    sanitized = text
    for name in sorted(known_tools - allowed, key=len, reverse=True):
        sanitized = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "[omitted-tool]", sanitized)
    return sanitized


def _response_has_usable_tool_call(data: dict[str, Any], allowed_tool_names: set[str]) -> bool:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    message = (choices[0] or {}).get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if not allowed_tool_names:
            return True
        for tool_call in tool_calls:
            function = (tool_call or {}).get("function") or {}
            if str(function.get("name") or "") in allowed_tool_names:
                return True
        return False
    content = str(message.get("content") or "")
    return bool(_parse_text_tool_calls(content, allowed_tool_names))


def _schema_only_tool_choice_miss(
    data: dict[str, Any],
    *,
    model: str,
    tools: list[ToolDef],
    requested_tool_choice: str | dict[str, Any] | None,
    repair_count: int,
) -> str:
    max_repairs = int(os.environ.get("NANOMA_SCHEMA_ONLY_TOOL_REPAIR_RETRIES", "3"))
    if repair_count >= max(0, max_repairs) or not tools or not _model_needs_required_arg_tool_schema(model):
        return ""
    target_tool = _named_tool_choice_name(requested_tool_choice)
    if not target_tool:
        return ""
    available = _tool_names_from_schemas(tools)
    if target_tool not in available:
        return ""
    if _response_has_usable_tool_call(data, {target_tool}):
        return ""
    return target_tool


def _body_with_schema_only_tool_retry_repair(body: dict[str, Any], tool_name: str, repair_count: int = 1) -> dict[str, Any]:
    repaired = copy.deepcopy(body)
    messages = list(repaired.get("messages") or [])
    messages.append({
        "role": "user",
        "content": (
            "[Schema-only tool retry]\n"
            "This model does not support named tool_choice in thinking mode, so runtime provided the target "
            f"tool through the available tool schema instead. The previous response did not include an "
            f"executable `{tool_name}` tool_call. Retry #{repair_count}: ignore earlier intentions to call "
            f"tools outside the current schema. The only acceptable result for this turn "
            f"is an actual `{tool_name}` tool_call from the current tool schema. Do not answer with prose only."
        ),
    })
    repaired["messages"] = messages
    repaired["tool_choice"] = "auto"
    return repaired


def _should_repair_tool_retry(error: Exception, *, model: str, tools: list[ToolDef]) -> bool:
    if not tools or not _model_needs_required_arg_tool_schema(model):
        return False
    return isinstance(error, (TransientEmptyLLMResponse, TransientToolCallTransportMiss))


def _body_with_tool_retry_repair(body: dict[str, Any]) -> dict[str, Any]:
    repaired = copy.deepcopy(body)
    messages = list(repaired.get("messages") or [])
    messages.append({
        "role": "user",
        "content": (
            "[Provider retry repair]\n"
            "The previous provider response for this same turn did not return a valid tool_call. "
            "Continue the same selected action and return actual tool_calls from the available tools. "
            "Do not answer with prose only."
        ),
    })
    repaired["messages"] = messages
    return repaired


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


def _raise_for_transient_empty_response(data: dict[str, Any]) -> None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TransientEmptyLLMResponse("LLM response has no choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    usage = data.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    try:
        usage_total = int(total_tokens if total_tokens is not None else int(prompt_tokens or 0) + int(completion_tokens or 0))
    except (TypeError, ValueError):
        usage_total = 0
    if not str(content or "").strip() and not tool_calls and usage_total <= 0:
        raise TransientEmptyLLMResponse("LLM response has empty assistant message and zero usage")


def _raise_for_transient_tool_call_transport_miss(
    data: dict[str, Any],
    *,
    model: str,
    tools: list[ToolDef],
) -> None:
    if not tools or not _model_needs_required_arg_tool_schema(model):
        return
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    message = (choices[0] or {}).get("message") or {}
    content = str(message.get("content") or "")
    tool_calls = message.get("tool_calls") or []
    if tool_calls or _parse_text_tool_calls(content, _tool_names_from_schemas(tools)):
        return
    usage = data.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    try:
        usage_total = int(total_tokens if total_tokens is not None else int(prompt_tokens or 0) + int(completion_tokens or 0))
    except (TypeError, ValueError):
        usage_total = 0
    if content.strip() and usage_total <= 0:
        raise TransientToolCallTransportMiss("LLM response has text but no tool calls and zero usage on a tool turn")


def default_router(task: str, budget: float, allowed_models: list[str] | None = None) -> str:
    from nanoma.models import get_registry
    return get_registry().route(budget, allowed=allowed_models)
