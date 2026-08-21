"""LLM abstraction: OpenAI-compatible and Anthropic-compatible clients."""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import random
import re
import shlex
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
    max_retries: int = field(default_factory=lambda: _default_retry_max_retries())
    base_delay: float = field(default_factory=lambda: _default_retry_base_delay())
    max_delay: float = field(default_factory=lambda: _default_retry_max_delay())
    http_timeout: float = field(default_factory=lambda: _default_http_timeout())


def _openai_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return {"type": "function", "function": {"name": tool_choice}}
    return tool_choice


def _anthropic_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return {"type": "auto"}
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice in {"any", "required"}:
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "tool", "name": tool_choice}
    return tool_choice


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


def _redacted_request_summary(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    return {
        "model": body.get("model"),
        "message_count": len(messages) if isinstance(messages, list) else None,
        "tool_count": len(tools) if isinstance(tools, list) else None,
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "tool_choice": body.get("tool_choice"),
        "message_roles": [
            msg.get("role") for msg in messages[-12:]
            if isinstance(msg, dict)
        ],
        "tool_names": [
            ((tool.get("function") or {}).get("name"))
            for tool in tools
            if isinstance(tool, dict)
        ],
        "approx_body_chars": len(json.dumps(body, ensure_ascii=False, default=str)),
    }


def _openai_message_payload(msg: Message) -> Message:
    payload = dict(msg)
    if "content" in payload and payload.get("content") is None:
        payload["content"] = ""
    return payload


def _log_http_error(model: str, body: dict[str, Any], resp: httpx.Response, elapsed_ms: float) -> None:
    if not _log_dir:
        return
    global _log_counter
    _log_counter += 1
    safe_model = model.replace("/", "_").replace(":", "_")
    log_path = _log_dir / f"{_log_counter:05d}_{safe_model}_error.jsonl"
    try:
        log_path.write_text(json.dumps({
            "request_summary": _redacted_request_summary(body),
            "status_code": resp.status_code,
            "response_text": resp.text[:8000],
            "ms": elapsed_ms,
        }, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write LLM error log {log_path}: {e}")


_NON_RETRYABLE_PROVIDER_ERROR_CODES = {
    "authentication_error",
    "incorrect_api_key",
    "invalid_api_key",
    "invalid_authentication",
    "invalid_model",
    "model_not_found",
    "model_not_supported",
    "model_not_available",
    "unsupported_model",
}


def _provider_error_code(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        if code is not None:
            return str(code).strip().lower()
    return ""


def _is_non_retryable_provider_error(resp: httpx.Response) -> bool:
    code = _provider_error_code(resp)
    if code in _NON_RETRYABLE_PROVIDER_ERROR_CODES:
        return True
    text = resp.text[:1000].lower()
    permanent_markers = (
        "model_not_found",
        "no available channel for model",
        "invalid api key",
        "incorrect api key",
        "authentication failed",
    )
    return any(marker in text for marker in permanent_markers)


def _retry_sleep_seconds(attempt: int, rc: RetryConfig, resp: httpx.Response | None = None) -> float:
    delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
    if resp is not None:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), rc.max_delay))
            except ValueError:
                pass
    return delay + random.uniform(0, delay * 0.3)


# --- Main call ---

# The EdgeBench provider gateway has returned short-lived 401/403 responses and
# then accepted the same credential again without any configuration change.  A
# bare auth status is therefore retryable; an explicit invalid-key error is
# still rejected immediately by `_is_non_retryable_provider_error` above.
_RETRYABLE = {401, 403, 429, 500, 502, 503, 504}
_shared_client: httpx.AsyncClient | None = None
_shared_clients: dict[float, httpx.AsyncClient] = {}


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


def _omit_max_tokens() -> bool:
    return os.environ.get("NANOMA_OMIT_MAX_TOKENS", "").strip().lower() in {"1", "true", "yes", "on"}


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


def _default_retry_max_retries() -> int:
    raw = os.environ.get("NANOMA_LLM_MAX_RETRIES")
    if not raw:
        return 8
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid NANOMA_LLM_MAX_RETRIES=%r; using 8", raw)
        return 8
    return max(0, value)


def _default_retry_base_delay() -> float:
    raw = os.environ.get("NANOMA_LLM_RETRY_BASE_DELAY")
    if not raw:
        return 2.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid NANOMA_LLM_RETRY_BASE_DELAY=%r; using 2.0", raw)
        return 2.0
    return max(0.1, value)


def _default_retry_max_delay() -> float:
    raw = os.environ.get("NANOMA_LLM_RETRY_MAX_DELAY")
    if not raw:
        return 120.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid NANOMA_LLM_RETRY_MAX_DELAY=%r; using 120.0", raw)
        return 120.0
    return max(0.1, value)


def _omit_sampling_params() -> bool:
    return os.environ.get("NANOMA_OMIT_SAMPLING_PARAMS", "").strip().lower() in {"1", "true", "yes", "on"}


def _openai_extra_body() -> dict[str, Any]:
    raw = os.environ.get("NANOMA_OPENAI_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Invalid NANOMA_OPENAI_EXTRA_BODY JSON: %s", e)
        return {}
    if not isinstance(data, dict):
        logger.warning("NANOMA_OPENAI_EXTRA_BODY must be a JSON object")
        return {}
    return data


def _parse_text_tool_calls_enabled() -> bool:
    return os.environ.get("NANOMA_PARSE_TEXT_TOOL_CALLS", "").strip().lower() in {"1", "true", "yes", "on"}


def _text_only_tools_enabled() -> bool:
    return os.environ.get("NANOMA_TEXT_ONLY_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}


def _text_tool_prompt(tools: list[ToolDef]) -> str:
    lines = [
        "[Runtime text tool protocol]",
        "The current model endpoint is using text tool calls. Do not emit prose when you need a tool.",
        'Emit exactly one JSON object like {"tool":"shell","args":{"command":"..."}}.',
        "For shell only, a fenced bash block is also accepted.",
        "Available tools:",
    ]
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        arg_names = list(props)[:8]
        details = []
        if required:
            details.append("required=" + ",".join(str(item) for item in required[:8]))
        elif arg_names:
            details.append("args=" + ",".join(str(item) for item in arg_names))
        lines.append(f"- {name}" + (f" ({'; '.join(details)})" if details else ""))
    return "\n".join(lines)


def _inject_text_tool_prompt(messages: list[Message], tools: list[ToolDef]) -> list[Message]:
    prompt = _text_tool_prompt(tools)
    for idx, msg in enumerate(messages):
        if msg.get("role") == "system":
            out = list(messages)
            merged = dict(msg)
            existing = _message_content_to_text(merged.get("content"))
            merged["content"] = f"{prompt}\n\n{existing}" if existing else prompt
            out[idx] = merged
            return out

    inserted = False
    out: list[Message] = []
    for msg in messages:
        if not inserted and msg.get("role") != "system":
            out.append({"role": "system", "content": prompt})
            inserted = True
        out.append(msg)
    if not inserted:
        out.append({"role": "system", "content": prompt})
    return out


def _normalize_text_tool_name(name: str) -> str:
    name = name.strip().lstrip("@")
    if ":" in name:
        name = name.rsplit(":", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    lowered = name.lower()
    aliases = {
        "bash": "shell",
        "cmd": "shell",
        "command": "shell",
        "command_shell": "shell",
        "run_shell": "shell",
        "shell_run": "shell",
        "shellrun": "shell",
        "terminal": "shell",
        "web_search": "shell",
        "search": "shell",
        "google_search": "shell",
        "visit": "shell",
        "fetch": "shell",
        "browse": "shell",
        "open_url": "shell",
        "read_url": "shell",
        "read_page": "shell",
        "readpage": "shell",
        "web_fetch": "shell",
        "ws_file_read": "ws_read_file",
        "file_read": "ws_read_file",
        "read_file": "ws_read_file",
        "write_file": "ws_create_file",
    }
    return aliases.get(lowered, lowered)


def _text_arg_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text_arg_to_string(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _string_list_from_text_tool_args(args: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    raw: Any = None
    for key in keys:
        if key in args:
            raw = args[key]
            break
    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        values = [raw]
    out: list[str] = []
    for item in values:
        text = _text_arg_to_string(item).strip()
        if text:
            out.append(text)
    return out


def _serper_search_shell_command(args: dict[str, Any]) -> str:
    queries = _string_list_from_text_tool_args(
        args,
        ("q", "query", "queries", "search_terms", "search_query", "keywords"),
    )
    if not queries:
        return ""
    limit = args.get("num") or args.get("n") or args.get("limit") or 10
    try:
        limit = max(1, min(20, int(limit)))
    except Exception:
        limit = 10
    queries_json = json.dumps(queries, ensure_ascii=False)
    return f"""python3 - <<'PY'
import json
import os
import urllib.error
import urllib.request

queries = json.loads({queries_json!r})
api_key = os.environ.get("SERPER_API_KEY")
if not api_key:
    raise SystemExit("SERPER_API_KEY is not set")

for query in queries:
    payload = json.dumps({{"q": query, "num": {limit}}}).encode("utf-8")
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={{"X-API-KEY": api_key, "Content-Type": "application/json"}},
        method="POST",
    )
    print(f"QUERY: {{query}}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"SERPER HTTP {{exc.code}}: {{exc.read().decode('utf-8', 'replace')[:1000]}}")
        continue
    except Exception as exc:
        print(f"SERPER ERROR: {{type(exc).__name__}}: {{exc}}")
        continue
    for item in (data.get("organic") or [])[:{limit}]:
        print("- " + (item.get("title") or ""))
        print("  " + (item.get("link") or ""))
        print("  " + (item.get("snippet") or ""))
    answer_box = data.get("answerBox") or {{}}
    if answer_box:
        print("ANSWER_BOX:", json.dumps(answer_box, ensure_ascii=False)[:1200])
    knowledge_graph = data.get("knowledgeGraph") or {{}}
    if knowledge_graph:
        print("KNOWLEDGE_GRAPH:", json.dumps(knowledge_graph, ensure_ascii=False)[:1200])
    print()
PY"""


def _visit_url_shell_command(args: dict[str, Any]) -> str:
    urls = _string_list_from_text_tool_args(args, ("url", "urls", "link", "href"))
    if not urls:
        return ""
    urls_json = json.dumps(urls[:5], ensure_ascii=False)
    return f"""python3 - <<'PY'
import html
import json
import re
import urllib.error
import urllib.request

urls = json.loads({urls_json!r})
for url in urls:
    print("URL:", url)
    try:
        req = urllib.request.Request(url, headers={{"User-Agent": "Mozilla/5.0"}})
        with urllib.request.urlopen(req, timeout=90) as resp:
            content_type = resp.headers.get("content-type", "")
            data = resp.read(1_500_000)
            status = getattr(resp, "status", "unknown")
    except urllib.error.HTTPError as exc:
        print("HTTP ERROR:", exc.code)
        content_type = exc.headers.get("content-type", "")
        data = exc.read(200_000)
        status = exc.code
    except Exception as exc:
        print("FETCH ERROR:", type(exc).__name__, exc)
        continue
    print("STATUS:", status, "CONTENT_TYPE:", content_type, "BYTES:", len(data))
    text = data.decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style).*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\\s+", " ", text).strip()
    print(text[:40000])
PY"""


def _normalize_text_tool_args(name: str, args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        args = {"command": args} if name == "shell" else {"task": args}
    if not isinstance(args, dict):
        args = {}
    args = dict(args)
    if name in {"spawn", "spawn_many"}:
        if "task" not in args:
            for key in (
                "assignment", "instruction", "prompt", "message",
                "description", "unit", "query",
            ):
                if key in args:
                    args["task"] = args.pop(key)
                    break
        if "task" in args and not isinstance(args["task"], str):
            args["task"] = _text_arg_to_string(args["task"])
        for key in ("agent_name", "name", "label", "role", "command"):
            args.pop(key, None)
        if name == "spawn_many" and "agents" in args and isinstance(args["agents"], list):
            agents = []
            for item in args["agents"]:
                if isinstance(item, dict) and "task" not in item:
                    item = dict(item)
                    for key in (
                        "assignment", "instruction", "prompt", "message",
                        "description", "unit", "query",
                    ):
                        if key in item:
                            item["task"] = item.pop(key)
                            break
                    for key in ("agent_name", "name", "label", "role", "command"):
                        item.pop(key, None)
                    if "task" in item and not isinstance(item["task"], str):
                        item["task"] = _text_arg_to_string(item["task"])
                agents.append(item)
            args["agents"] = agents
    if name == "shell" and "command" not in args:
        search_command = _serper_search_shell_command(args)
        if search_command:
            args["command"] = search_command
        else:
            visit_command = _visit_url_shell_command(args)
            if visit_command:
                args["command"] = visit_command
    if name == "shell" and "command" not in args:
        for key in ("cmd", "code", "script"):
            if key in args:
                args["command"] = args[key]
                break
    if name == "shell" and isinstance(args.get("command"), str):
        command = args["command"].strip()
        command = re.sub(r"^```(?:bash|sh|shell)?\s*", "", command, flags=re.IGNORECASE)
        command = re.sub(r"\s*```\s*$", "", command).strip()
        args["command"] = command
        args.setdefault("timeout", int(os.environ.get("NANOMA_SHELL_TEXT_TOOL_TIMEOUT", "90")))
    if name == "set_status" and "status" not in args:
        positional = args.get("_positional")
        if isinstance(positional, list) and positional:
            args["status"] = positional[0]
    if name == "wait":
        if "agent_ids" not in args:
            for key in ("agents", "agent", "children"):
                if key in args:
                    args["agent_ids"] = args.pop(key)
                    break
        if isinstance(args.get("agent_ids"), str):
            raw_agent_ids = args["agent_ids"].strip()
            try:
                parsed_agent_ids = json.loads(raw_agent_ids.replace("'", '"'))
            except Exception:
                parsed_agent_ids = [
                    item.strip()
                    for item in re.split(r",|\s+", raw_agent_ids.strip("[]"))
                    if item.strip()
                ]
            args["agent_ids"] = parsed_agent_ids
    if name == "query" and "agent_id" not in args:
        for key in ("agent", "target"):
            if key in args:
                args["agent_id"] = args.pop(key)
                break
    if name in {"ws_create_file", "ws_append_file"}:
        if "path" not in args and "file" in args:
            args["path"] = args.pop("file")
        if "content" not in args and "text" in args:
            args["content"] = args.pop("text")
    if name == "ws_read_file":
        if "path" not in args and "file" in args:
            args["path"] = args.pop("file")
        args.pop("binary", None)
    return args


def _valid_shell_text_command(command: str) -> bool:
    stripped = str(command or "").strip()
    if not stripped:
        return False
    normalized = stripped.strip("`").strip().lower()
    if normalized in {
        "shell",
        "bash",
        "sh",
        "zsh",
        "command",
        "cmd",
        "terminal",
        "run shell",
        "shell run",
        "spawn",
        "spawn_many",
        "send",
        "wait",
        "query",
        "kill",
        "transfer",
        "set_bio",
        "set_status",
        "submit",
        "get_cost",
        "rebirth",
        "batch",
    }:
        return False
    if re.match(
        r"^(?:spawn|spawn_many|send|wait|query|kill|transfer|set_bio|set_status|submit|get_cost|rebirth|batch)\s*\(",
        normalized,
    ):
        return False
    return True


def _json_object_from_text(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("empty JSON text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    last_error: Exception | None = None
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError as e:
            last_error = e
    raise last_error or ValueError("no JSON object found")


def _text_tool_call_id(index: int) -> str:
    return f"text_tool_call_{int(time.time() * 1000)}_{index}"


def _literal_from_ast(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        return None


def _python_call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _parse_python_text_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _normalize_text_tool_name(_python_call_name(node))
        if name not in allowed:
            continue

        args: dict[str, Any] = {}
        for kw in node.keywords:
            if not kw.arg:
                continue
            value = _literal_from_ast(kw.value)
            if value is not None:
                args[kw.arg] = value

        if node.args:
            first = _literal_from_ast(node.args[0])
            if isinstance(first, dict):
                args = {**first, **args}
            elif first is not None:
                if name == "shell":
                    args.setdefault("command", str(first))
                elif name == "spawn_many" and isinstance(first, list):
                    args.setdefault("agents", first)
                else:
                    args.setdefault("task", str(first))

        add_call(name, args)


def _parse_bare_function_line_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    """Recover plain tool calls embedded as standalone lines in model prose."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*\((.*)\)\s*\.?$",
            stripped,
            flags=re.DOTALL,
        )
        if not match:
            continue
        name = _normalize_text_tool_name(match.group(1))
        if name not in allowed:
            continue
        add_call(name, _parse_function_call_args(match.group(2)))


def _parse_pythonish_spawn_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "spawn" not in allowed and "spawn_many" not in allowed:
        return
    calls: list[dict[str, str]] = []
    pattern = re.compile(r"\bspawn\s*\((.*?)\)", flags=re.DOTALL)
    for match in pattern.finditer(text):
        args = _parse_function_call_args(match.group(1))
        positional = args.get("_positional") or []
        task: Any = args.get("task") or args.get("message") or args.get("prompt") or args.get("description")
        if not task and len(positional) >= 2:
            task = positional[1]
        elif not task and len(positional) == 1:
            task = positional[0]
        if task and str(task).strip():
            calls.append({"task": str(task).strip()})
        if len(calls) >= 8:
            break
    if not calls:
        return
    if len(calls) > 1 and "spawn_many" in allowed:
        add_call("spawn_many", {"agents": calls})
        return
    if "spawn" in allowed:
        for item in calls:
            add_call("spawn", item)
    elif "spawn_many" in allowed:
        add_call("spawn_many", {"agents": calls})


def _parse_qwen_angle_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    normalized = text.replace("<@call>function<@>", "<@>")
    normalized = re.sub(r"</@>\s*([A-Za-z0-9_.:-]+)<\|", r"<@>\1<|", normalized)
    pattern = re.compile(
        r"<@>([A-Za-z0-9_.:-]+)(.*?)(?=<@>[A-Za-z0-9_.:-]+<\||</@>|$)",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(normalized):
        name = _normalize_text_tool_name(match.group(1))
        if name not in allowed:
            continue
        body = match.group(2)
        parts = re.split(r"<\|([^|<>]+)\|", body)
        args: dict[str, Any] = {}
        for idx in range(1, len(parts), 2):
            key = parts[idx].strip()
            value = parts[idx + 1] if idx + 1 < len(parts) else ""
            value = re.split(r"</@>|<@call>|<@>", value, maxsplit=1)[0]
            value = value.replace("<|mask_start|>", "").strip()
            if key and value:
                args[key] = value
        if args:
            add_call(name, args)


def _split_call_arguments(raw: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    depth = 0
    for ch in raw:
        if quote:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_call_argument_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip("\"'")


def _parse_function_call_args(raw: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    positional: list[Any] = []
    for part in _split_call_arguments(raw):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            args[key.strip()] = _parse_call_argument_value(value)
        else:
            positional.append(_parse_call_argument_value(part))
    if positional:
        args.setdefault("_positional", positional)
    return args


def _parse_qwen_tool_call_blocks(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    block_pattern = re.compile(
        r"<(?P<tag>tool_call|tool_code)>\s*(?P<body>.*?)(?:</(?P=tag)>|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in block_pattern.finditer(text):
        body = re.sub(r"</?(?:tool_call|tool_code)>", "", match.group("body"), flags=re.IGNORECASE).strip()
        if not body:
            continue
        if body.lower() in {"user", "assistant", "system", ">"} or re.fullmatch(r"(?:user|assistant|system)?\s*>?", body, flags=re.IGNORECASE):
            continue
        try:
            value = _json_object_from_text(body)
        except Exception:
            value = None
        if value is not None and _add_tool_calls_from_json_value(value, allowed, add_call):
            continue
        if _parse_qwen_jsonish_tool_call_body(body, allowed, add_call):
            continue
        fn_match = re.match(r"([A-Za-z_][A-Za-z0-9_.-]*)\s*\((.*)\)\s*$", body, flags=re.DOTALL)
        if fn_match:
            raw_name = fn_match.group(1)
            name = _normalize_text_tool_name(raw_name)
            args = _parse_function_call_args(fn_match.group(2))
            if name == "shell":
                command = args.get("command") or args.get("cmd") or (args.get("_positional") or [""])[0]
                add_call("shell", {"command": command})
            elif raw_name.lower() == "ws_file_read" and args.get("binary") and "shell" in allowed:
                path = str(args.get("path") or (args.get("_positional") or [""])[0])
                add_call("shell", {"command": f"ls -l {shlex.quote(path)} && file {shlex.quote(path)}"})
            else:
                add_call(name, args)
            continue
        bare = _normalize_text_tool_name(body)
        if bare == "send" and "wait" in allowed:
            add_call("wait", {"timeout": 120, "mode": "any"})
        elif bare in allowed:
            add_call(bare, {})


def _jsonish_unescape_text(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return (
            value.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
        )


def _extract_jsonish_string_field(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text, flags=re.IGNORECASE)
    if not match:
        return None
    tail = text[match.end():]
    end = re.search(
        r'"\s*(?=,\s*"[A-Za-z0-9_.-]+"\s*:|\}\s*(?:\}\s*)?$)',
        tail,
        flags=re.DOTALL,
    )
    if not end:
        return None
    return _jsonish_unescape_text(tail[:end.start()])


def _parse_qwen_jsonish_tool_call_body(
    body: str,
    allowed: set[str],
    add_call: Any,
) -> bool:
    """Recover Qwen text tool calls whose JSON strings contain raw newlines."""
    name_match = re.search(r'"(?:tool|call|name)"\s*:\s*"([A-Za-z0-9_.@-]+)"', body, flags=re.IGNORECASE)
    if not name_match:
        return False

    name = _normalize_text_tool_name(name_match.group(1))
    if name not in allowed:
        return False

    if name == "shell":
        command = (
            _extract_jsonish_string_field(body, "command")
            or _extract_jsonish_string_field(body, "cmd")
            or _extract_jsonish_string_field(body, "code")
            or _extract_jsonish_string_field(body, "script")
        )
        if not command:
            return False
        args: dict[str, Any] = {"command": command}
        timeout_match = re.search(r'"timeout"\s*:\s*(\d+)', body)
        if timeout_match:
            args["timeout"] = int(timeout_match.group(1))
        add_call(name, args)
        return True

    if name in {"ws_create_file", "ws_append_file"}:
        path = _extract_jsonish_string_field(body, "path") or _extract_jsonish_string_field(body, "file")
        content = _extract_jsonish_string_field(body, "content") or _extract_jsonish_string_field(body, "text")
        if path is None or content is None:
            return False
        add_call(name, {"path": path, "content": content})
        return True

    if name == "set_status":
        status = _extract_jsonish_string_field(body, "status")
        result = _extract_jsonish_string_field(body, "result")
        if not status:
            return False
        args = {"status": status}
        if result is not None:
            args["result"] = result
        add_call(name, args)
        return True

    return False


def _extract_bare_spawn_partitions(text: str) -> list[str]:
    partition_terms = (
        "brands",
        "companies",
        "broad subject areas",
        "subject areas",
        "broad subjects",
        "subjects",
        "states",
        "countries",
        "regions",
        "sources",
        "partitions",
        "broad categories",
        "categories",
        "universities",
        "disciplines",
    )
    term_pattern = "|".join(re.escape(term) for term in partition_terms)

    comment_match = re.search(r"#\s*spawn agents?\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if comment_match:
        labels: list[str] = []
        for piece in re.split(r",|;", comment_match.group(1)):
            piece = piece.strip()
            if not piece:
                continue
            paren = re.search(r"\(([^()]+)\)", piece)
            if paren:
                labels.append(paren.group(1).replace("_", " ").strip().title())
                continue
            piece = re.sub(r"^(?:subject|agent|researcher)[_\-\s]*\d*[_\-\s]*", "", piece, flags=re.IGNORECASE)
            piece = piece.replace("_", " ").replace("-", " ").strip()
            if piece:
                labels.append(piece.title())
        if len(labels) >= 2:
            return labels[:8]

    listed = re.search(
        rf"(?:{term_pattern}|spawn agents? for)(?:[^\n:]{{0,160}})?:\s*\n(?P<block>(?:\s*(?:[-*]|\d+[.)])\s+.+\n?)+)",
        text,
        flags=re.IGNORECASE,
    )
    if listed:
        candidates = []
        for line in listed.group("block").splitlines():
            item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip(" .;:")
            if 2 <= len(item) <= 120:
                candidates.append(item)
        if len(candidates) >= 2:
            return candidates[:8]

    blocks = re.finditer(
        rf"(?:the\s+)?(?:{term_pattern})\s*(?:are)?\s*:?\s*(?P<block>.*?)(?=\n\s*\n|I'll|I will|Let me|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    candidates: list[str] = []
    for match in blocks:
        block = match.group("block").strip()
        if not block:
            continue
        for line in block.splitlines():
            item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip(" .;:")
            if 2 <= len(item) <= 120 and not item.lower().startswith(("scope", "columns", "as of")):
                candidates.append(item)
        if len(candidates) < 2 and "," in block:
            pieces = re.split(r",|\band\b", block)
            candidates = [
                piece.strip(" .;:\n")
                for piece in pieces
                if 2 <= len(piece.strip(" .;:\n")) <= 80
            ]
        if len(candidates) >= 2:
            break

    if len(candidates) < 2:
        numbered = []
        for line in text.splitlines():
            match = re.match(r"\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", line)
            if not match:
                continue
            item = match.group(1).strip(" .;:")
            if 2 <= len(item) <= 100 and not item.lower().startswith(("i need", "scope", "columns")):
                numbered.append(item)
        if 2 <= len(numbered) <= 12:
            candidates = numbered

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = re.sub(r"\s+", " ", item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:8]


def _spawn_research_task(context: str, partition: str) -> str:
    return (
        "Parent planning/context:\n"
        f"{context}\n\n"
        f"Your independent WideSearch research partition: {partition}. "
        "Use public web evidence through the available shell tools, write concise findings "
        "to the shared workspace, then report back to the parent."
    )


def _extract_yamlish_section(text: str, header: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(header)}\s*:\s*(?:\|\s*)?\n",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    tail = text[match.end():]
    next_header = re.search(r"(?m)^[A-Za-z_][A-Za-z0-9_-]*\s*:\s*(?:\||$)", tail)
    block = tail[:next_header.start()] if next_header else tail
    return re.split(r"</(?:tool_call|command)>|<tool_call>", block, maxsplit=1, flags=re.IGNORECASE)[0]


def _parse_yamlish_list(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        match = re.match(r"\s*-\s*(.+?)\s*$", line)
        if match:
            items.append(match.group(1).strip().strip("'\""))
    return items


def _parse_yamlish_mapping(block: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_value
        if not current_key:
            return
        raw = "\n".join(part for part in current_value if part is not None).strip()
        if raw:
            try:
                value = ast.literal_eval(raw)
            except Exception:
                value = raw.strip("'\"")
            mapping[current_key] = str(value).strip()
        current_key = None
        current_value = []

    for line in block.splitlines():
        if re.match(r"\s*</?(?:tool_call|command)\b", line, flags=re.IGNORECASE):
            break
        match = re.match(r"\s{0,4}([A-Za-z0-9_.-]+)\s*:\s*(.*)$", line)
        if match:
            flush()
            current_key = match.group(1).strip()
            current_value = [match.group(2).strip()]
            continue
        if current_key and (line.startswith(" ") or line.startswith("\t")):
            current_value.append(line.strip())
    flush()
    return mapping


def _spawn_context_from_qwen_plan(text: str) -> str:
    context = re.sub(r"</?(?:think|tool_call|command)[^>]*>", "", text, flags=re.IGNORECASE)
    context = re.sub(
        r"(?ms)^agent_instructions\s*:\s*\|?.*?(?=^agent_names\s*:|^task_instructions\s*:|\Z)",
        "",
        context,
    )
    context = re.sub(r"(?ms)^agent_names\s*:.*?(?=^task_instructions\s*:|\Z)", "", context)
    context = re.sub(r"(?ms)^task_instructions\s*:.*", "", context)
    context = re.sub(r"(?m)^---\s*[A-Za-z0-9_.-]*spawn[A-Za-z0-9_.-]*\s*---\s*$", "", context)
    context = re.sub(r"\n{3,}", "\n\n", context).strip()
    if len(context) > 2500:
        context = context[:2500].rsplit("\n", 1)[0].strip()
    return context


def _extract_qwen_yamlish_spawn_agents(text: str) -> list[dict[str, str]]:
    if not re.search(r"(?im)^\s*(?:num_agents|agent_names|task_instructions|agent_instructions)\s*:", text):
        return []

    names = _parse_yamlish_list(_extract_yamlish_section(text, "agent_names"))
    tasks_by_name = _parse_yamlish_mapping(_extract_yamlish_section(text, "task_instructions"))
    context = _spawn_context_from_qwen_plan(text)

    ordered_names: list[str] = []
    for name in names:
        if name not in ordered_names:
            ordered_names.append(name)
    for name in tasks_by_name:
        if name not in ordered_names:
            ordered_names.append(name)

    agents: list[dict[str, str]] = []
    for name in ordered_names[:8]:
        instruction = tasks_by_name.get(name) or name.replace("_", " ")
        if not instruction:
            continue
        label = name.replace("_", " ")
        agents.append({
            "task": (
                "Parent planning/context:\n"
                f"{context}\n\n"
                f"Your independent WideSearch research partition: {label}.\n"
                f"{instruction}\n\n"
                "Use public web evidence through the available shell tools, write concise findings "
                "to the shared workspace, then report back to the parent."
            ).strip()
        })

    if agents:
        return agents

    partitions = _extract_bare_spawn_partitions(text)
    return [{"task": _spawn_research_task(context, partition)} for partition in partitions[:8]]


def _parse_qwen_yamlish_spawn_plan_tool_call(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "spawn_many" not in allowed and "spawn" not in allowed:
        return
    lowered = text.lower()
    if "spawn" not in lowered and "num_agents" not in lowered and "task_instructions:" not in lowered:
        return
    agents = _extract_qwen_yamlish_spawn_agents(text)
    if len(agents) < 2:
        return
    if "spawn_many" in allowed:
        add_call("spawn_many", {"agents": agents})
        return
    for agent in agents:
        add_call("spawn", {"task": agent.get("task", "")})


def _parse_qwen_nested_tag_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "shell" in allowed:
        pattern = re.compile(
            r"<tool_call>\s*shell\s*<tool_call>\s*command\s*(.*?)(?:</command>|(?=<tool_call>)|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(text):
            raw = re.sub(r"</?command[^>]*>", "", match.group(1), flags=re.IGNORECASE).strip()
            if not raw:
                continue
            value = _parse_call_argument_value(raw)
            add_call("shell", {"command": str(value)})


def _parse_qwen_at_prefixed_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "<tool_call>@" not in text.lower():
        return

    parts = re.split(r"<tool_call>\s*@", text, flags=re.IGNORECASE)
    awaiting_shell_command = False
    for part in parts[1:]:
        segment = re.split(r"</tool_call>|<tool_call>", part, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if not segment:
            continue
        first_line = segment.splitlines()[0].strip()
        tag = first_line.split(None, 1)[0].strip().lower()

        if tag in {"user", "assistant", "system"}:
            continue
        if tag in {"shell", "bash", "sh", "command_shell"}:
            awaiting_shell_command = "shell" in allowed
            remainder = "\n".join(segment.splitlines()[1:]).strip()
            if awaiting_shell_command and remainder:
                try:
                    args = _json_object_from_text(remainder)
                except Exception:
                    args = {"command": remainder}
                add_call("shell", args)
                awaiting_shell_command = False
            continue
        if awaiting_shell_command and tag in {"command", "cmd", "args", "argument"}:
            remainder = "\n".join(segment.splitlines()[1:]).strip()
            if remainder:
                try:
                    args = _json_object_from_text(remainder)
                except Exception:
                    args = {"command": remainder}
                add_call("shell", args)
                awaiting_shell_command = False
            continue
        if awaiting_shell_command:
            add_call("shell", {"command": segment})
            awaiting_shell_command = False
            continue

        name = _normalize_text_tool_name(tag)
        if name in allowed:
            remainder = first_line[len(tag):].strip()
            if not remainder:
                remainder = "\n".join(segment.splitlines()[1:]).strip()
            if remainder:
                try:
                    args = _json_object_from_text(remainder)
                except Exception:
                    args = {"command": remainder} if name == "shell" else {"task": remainder}
            else:
                args = {}
            add_call(name, args)


def _parse_pipe_delimited_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    pattern = re.compile(
        r"\|\|([A-Za-z0-9_.-]+):begin\|\|\|(.*?)\|\|\|\1:end\|\|\|",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        name = _normalize_text_tool_name(match.group(1))
        if name not in allowed:
            continue
        body = match.group(2).strip()
        if name == "shell":
            add_call(name, {"command": body})
        elif name in {"spawn", "spawn_many"}:
            if name == "spawn_many":
                add_call(name, {"agents": [{"task": body}]})
            else:
                add_call(name, {"task": body})
        else:
            add_call(name, {"_raw": body})


def _tool_call_args_from_item(item: dict[str, Any]) -> tuple[str | None, Any]:
    fn = item.get("function")
    name: Any = None
    args: Any = None

    if isinstance(fn, dict):
        name = fn.get("name") or fn.get("function")
        args = fn.get("arguments") or fn.get("args") or fn.get("parameters")
    elif isinstance(fn, str):
        name = fn

    # Qwen-style text JSON often uses {"tool": "spawn", "name": "agent_name", ...}.
    # In that shape, "name" is an argument, not the tool function name.
    name = name or item.get("tool") or item.get("call") or item.get("name")
    args = args or item.get("arguments") or item.get("args") or item.get("parameters")

    if args is None and name:
        args = {
            k: v for k, v in item.items()
            if k not in {"id", "type", "name", "tool", "call", "function"}
        }

    if isinstance(args, str):
        try:
            args = _json_object_from_text(args)
        except Exception:
            args = {"command": args} if _normalize_text_tool_name(str(name or "")) == "shell" else {"task": args}

    return str(name) if name else None, args


def _add_tool_calls_from_json_value(value: Any, allowed: set[str], add_call: Any) -> int:
    count = 0
    if isinstance(value, list):
        for item in value:
            count += _add_tool_calls_from_json_value(item, allowed, add_call)
        return count

    if not isinstance(value, dict):
        return 0

    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list):
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            name, args = _tool_call_args_from_item(item)
            if name:
                add_call(name, args)
                count += 1

    name, args = _tool_call_args_from_item(value)
    if name and _normalize_text_tool_name(name) in allowed:
        add_call(name, args)
        count += 1
    return count


def _parse_relaxed_embedded_json_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if '"tool"' not in text and "'tool'" not in text and '"function"' not in text and "'function'" not in text:
        return
    decoder = json.JSONDecoder()
    idx = 0
    count = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        count += _add_tool_calls_from_json_value(value, allowed, add_call)
        idx = start + max(end, 1)
        if count >= 12:
            break


def _parse_qwen_loose_json_tool_objects(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if '"tool"' not in text or '"args"' not in text:
        return
    candidates = re.finditer(
        r'"args"\s*:\s*(\{.*?\})\s*,\s*"tool"\s*:\s*"([A-Za-z0-9_.@-]+)"',
        text,
        flags=re.DOTALL,
    )
    spawned: list[dict[str, Any]] = []
    for match in candidates:
        try:
            args = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = _normalize_text_tool_name(match.group(2))
        if name in {"spawn", "spawn_many"}:
            task = args.get("task") or args.get("message") or args.get("prompt") or args.get("description")
            if task:
                spawned.append({"task": str(task)})
            continue
        add_call(name, args)

    if not spawned:
        return
    if "spawn_many" in allowed:
        add_call("spawn_many", {"agents": spawned[:8]})
    elif "spawn" in allowed:
        for item in spawned[:8]:
            add_call("spawn", item)


def _parse_embedded_json_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if _add_tool_calls_from_json_value(value, allowed, add_call):
            return


def _parse_qwen_double_pipe_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    pattern = re.compile(r"tool_call\|\|([A-Za-z0-9_.-]+)\|\|", flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in pattern.finditer(text):
        name = _normalize_text_tool_name(match.group(1))
        if name not in allowed:
            continue
        tail = text[match.end():].lstrip()
        try:
            args, _ = decoder.raw_decode(tail)
        except json.JSONDecodeError:
            raw = re.split(r"\s*\|{1,2}\s*tool_call\|\||</?tool_call>", tail, maxsplit=1, flags=re.IGNORECASE)[0]
            try:
                args = _json_object_from_text(raw)
            except Exception:
                args = {"command": raw.strip()} if name == "shell" else {"_raw": raw.strip()}
        add_call(name, args)


def _parse_command_shell_blocks(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "shell" not in allowed:
        return
    block_pattern = re.compile(r"<command_shell\b[^>]*>(.*?)(?:</command_shell>|$)", flags=re.IGNORECASE | re.DOTALL)
    for match in block_pattern.finditer(text):
        body = match.group(1).strip()
        if not body:
            continue
        timeout: int | None = None
        timeout_match = re.search(r"<timeout>\s*(\d+)\s*</timeout>", body, flags=re.IGNORECASE)
        if timeout_match:
            timeout = int(timeout_match.group(1))
        args_match = re.search(
            r"<args>\s*(.*?)(?=</args>|<timeout>|</command_shell>|$)",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        raw_args = args_match.group(1).strip() if args_match else body
        raw_args = re.sub(r"</?args>", "", raw_args, flags=re.IGNORECASE).strip()
        try:
            args = _json_object_from_text(raw_args)
        except Exception:
            command = re.sub(r"</?timeout>|\d+\s*</timeout>", "", raw_args, flags=re.IGNORECASE).strip()
            args = {"command": command}
        if isinstance(args, dict) and timeout is not None:
            args.setdefault("timeout", timeout)
        add_call("shell", args)


def _parse_parameter_named_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    pattern = re.compile(
        r"<parameter\s+name=[\"']?([A-Za-z0-9_.-]+)[\"']?\s*>(.*?)(?=<parameter\s+name=|</parameter>|</tool_call>|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        name = _normalize_text_tool_name(match.group(1))
        if name not in allowed:
            continue
        raw = re.sub(r"</?parameter[^>]*>", "", match.group(2), flags=re.IGNORECASE).strip()
        if not raw:
            continue
        try:
            args = _json_object_from_text(raw)
        except Exception:
            if name == "shell":
                command = re.sub(r"^\s*(?:command|cmd)\s*=\s*", "", raw, flags=re.IGNORECASE).strip()
                command = re.split(r"</(?:think|tool_call|parameter)>", command, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                args = {"command": command}
            else:
                args = {"_raw": raw}
        add_call(name, args)


def _parse_python_answer_writer_tool_call(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "shell" not in allowed:
        return
    fence_pattern = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in fence_pattern.finditer(text):
        body = match.group(1).strip()
        if "widesearch_answer.md" not in body or "open(" not in body:
            continue
        delimiter = "NANOMA_PY"
        while re.search(rf"^\s*{delimiter}\s*$", body, flags=re.MULTILINE):
            delimiter += "_END"
        command = f"python3 - <<'{delimiter}'\n{body}\n{delimiter}"
        add_call("shell", {"command": command})


def _parse_xmlish_tool_calls(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    qwen_tool_use_pattern = re.compile(
        r"<tool_use>\s*"
        r"<tool_use_name>\s*([A-Za-z0-9_.:-]+)\s*</tool_use_name>\s*"
        r"<tool_use_args>\s*(.*?)\s*(?:</tool_use_args>|</tool_use>|$)\s*"
        r"(?:</tool_use>)?",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in qwen_tool_use_pattern.finditer(text):
        name = _normalize_text_tool_name(match.group(1))
        if name not in allowed:
            continue
        body = match.group(2).strip()
        try:
            args = _json_object_from_text(body)
        except Exception:
            args = {"command": body} if name == "shell" else {"task": body}
        add_call(name, args)

    invoke_pattern = re.compile(
        r"<invoke\s+name=[\"']?([A-Za-z0-9_.:-]+)[\"']?\s*>(.*?)(?:</invoke>|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in invoke_pattern.finditer(text):
        name = _normalize_text_tool_name(match.group(1))
        if name not in allowed:
            continue
        body = match.group(2)
        args: dict[str, Any] = {}
        for p_match in re.finditer(
            r"<parameter\s+name=[\"']?([^\"'>\s]+)[\"']?\s*>(.*?)(?=<parameter\s+name=|</invoke>|$)",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            key = p_match.group(1).strip()
            value = re.sub(r"</?parameter[^>]*>", "", p_match.group(2), flags=re.IGNORECASE).strip()
            if not key:
                continue
            try:
                parsed_value = json.loads(value)
            except Exception:
                parsed_value = value.strip("\"'")
            args[key] = parsed_value
        if name == "wait" and "agents" in args and "agent_ids" not in args:
            args["agent_ids"] = args.pop("agents")
        add_call(name, args)

    subprocess_pattern = re.compile(
        r"<function>\s*\{?(?:__import__\(['\"]subprocess['\"]\)\.run|subprocess\.run|shell)\}?\s*</function>\s*"
        r"<parameter>\s*(?:cmd|command|shell)\s*</parameter>\s*"
        r"<parameter>\s*(.*?)\s*</parameter>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in subprocess_pattern.finditer(text):
        if "shell" in allowed:
            add_call("shell", {"command": match.group(1).strip()})


def _parse_call_function_fences(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    fence_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        lang = (match.group(1) or "").strip()
        body = match.group(2).strip()
        call_match = re.search(r"(?:call:)?function\{([A-Za-z0-9_.-]+)\}", lang, flags=re.IGNORECASE)
        if not call_match:
            continue
        name = _normalize_text_tool_name(call_match.group(1))
        if name not in allowed:
            continue
        if body:
            try:
                args = _json_object_from_text(body)
            except Exception:
                args = {"command": body} if name == "shell" else {"_raw": body}
        elif name == "wait":
            args = {"timeout": 120}
        elif name == "send":
            # Qwen sometimes emits a bare send call after spawning fully-tasked agents.
            # With no recipient/message, waiting for those agents is the only useful next action.
            if "wait" in allowed:
                add_call("wait", {"timeout": 120, "mode": "any"})
            continue
        else:
            args = {}
        add_call(name, args)


def _parse_qwen_masked_shell_blocks(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "shell" not in allowed:
        return

    block_patterns = (
        re.compile(r"<\|mask_start\|>\s*(.*?)(?:<\|mask_end\|>|$)", flags=re.DOTALL),
        re.compile(r"<tool_code>\s*(.*?)(?:</tool_code>|$)", flags=re.IGNORECASE | re.DOTALL),
    )
    for pattern in block_patterns:
        for match in pattern.finditer(text):
            command = match.group(1).strip()
            command = re.sub(r"</?(?:tool_code|tool_call|command)[^>]*>", "", command, flags=re.IGNORECASE).strip()
            if _valid_shell_text_command(command):
                add_call("shell", {"command": command})


def _parse_plain_shell_command(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "shell" not in allowed:
        return
    cleaned = re.sub(r"</?think>", "", text).strip()
    cleaned = cleaned.replace("<|mask_start|>", "").replace("<|mask_end|>", "")
    cleaned = re.sub(r"</?tool_code>", "", cleaned, flags=re.IGNORECASE).strip()
    if re.search(r"(?im)^\s*(?:num_agents|agent_names|task_instructions|agent_instructions)\s*:", cleaned):
        return
    lines = cleaned.splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        if re.match(r"\s*(?:curl|python3?|wget|cat|grep|rg|sed|awk|find|ls)\b", line):
            start = idx
            break
    if start is None:
        return
    command = "\n".join(lines[start:]).strip()
    if not command or len(command) > 8000:
        return
    add_call("shell", {"command": command})


def _parse_spawn_agent_plan_tool_call(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    if "spawn" not in allowed and "spawn_many" not in allowed:
        return
    stripped = text.strip()
    lowered = stripped.lower()
    if "spawn" not in lowered or "agent" not in lowered:
        return
    marker = re.search(
        r"#\s*spawn agents?|(?:^|\n)\s*agent\s*$|spawn(?:ing)?\s+\d+\s+(?:child\s+)?agents?|spawn agents? for|let me spawn (?:the )?agents?",
        stripped,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not marker:
        return

    context = re.sub(r"</?think>", "", stripped)
    context = re.sub(r"```[A-Za-z0-9_.-]*|```", "", context).strip()
    if len(context) > 2500:
        context = context[:2500].rsplit("\n", 1)[0]
    partitions = _extract_bare_spawn_partitions(context)
    if not partitions:
        return

    if "spawn_many" in allowed:
        add_call("spawn_many", {"agents": [{"task": _spawn_research_task(context, p)} for p in partitions]})
        return

    for partition in partitions:
        add_call("spawn", {"task": _spawn_research_task(context, partition)})


def _parse_bare_spawn_tool_call(
    text: str,
    allowed: set[str],
    add_call: Any,
) -> None:
    stripped = text.strip()
    if not re.search(r"(?:^|[>\n])\s*(?:<tool_call>\s*)*spawn\s*$", stripped, flags=re.IGNORECASE):
        return
    if "spawn_many" not in allowed and "spawn" not in allowed:
        return

    context = re.sub(r"</?think>", "", stripped)
    context = re.sub(r"(?:<tool_call>\s*)*spawn\s*$", "", context, flags=re.IGNORECASE).strip()
    if len(context) > 2500:
        context = context[:2500].rsplit("\n", 1)[0]
    partitions = _extract_bare_spawn_partitions(context)

    if partitions and "spawn_many" in allowed:
        agents = [
            {"task": _spawn_research_task(context, partition)}
            for partition in partitions
        ]
        add_call("spawn_many", {"agents": agents})
        return

    if partitions and "spawn" in allowed:
        for partition in partitions:
            add_call("spawn", {"task": _spawn_research_task(context, partition)})
        return

    add_call("spawn", {
        "task": (
            "Perform one independent research branch for the current WideSearch task using this "
            "parent planning/context. Use public web evidence through the available shell tools, "
            "write concise findings to the shared workspace, then report back to the parent.\n\n"
            f"{context}"
        )
    })


def _parse_text_tool_calls(content: str | None, tools: list[ToolDef] | None) -> list[ToolCall]:
    if not content or not tools:
        return []
    allowed = {
        str((tool.get("function") or {}).get("name") or "")
        for tool in tools
        if isinstance(tool, dict)
    }
    allowed.discard("")
    if not allowed:
        return []

    parsed: list[ToolCall] = []
    seen_calls: set[str] = set()

    def add_call(name: str, args: Any) -> None:
        name = _normalize_text_tool_name(name)
        if name not in allowed:
            return
        if name == "shell":
            normalized_args = _normalize_text_tool_args(name, args)
            if not _valid_shell_text_command(str(normalized_args.get("command", ""))):
                return
            args = normalized_args
        else:
            args = _normalize_text_tool_args(name, args)
        key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)}"
        if key in seen_calls:
            return
        seen_calls.add(key)
        parsed.append(ToolCall(
            id=_text_tool_call_id(len(parsed)),
            name=name,
            arguments=args,
        ))

    _parse_qwen_yamlish_spawn_plan_tool_call(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_qwen_nested_tag_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_qwen_at_prefixed_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_qwen_loose_json_tool_objects(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_pythonish_spawn_calls(content, allowed, add_call)
    if parsed:
        return parsed

    fence_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    for match in fence_pattern.finditer(content):
        lang = (match.group(1) or "").strip()
        lang_lower = lang.lower()
        body = match.group(2).strip()
        if not body:
            continue
        if (
            lang_lower in {"bash", "sh", "shell", "zsh"}
            or re.match(r"^(?:python3?|uv\s+run\s+python|python3?\s+-)\b", lang_lower)
        ):
            if re.match(r"^(?:python3?|uv\s+run\s+python|python3?\s+-)\b", lang_lower):
                body = f"{lang}\n{body}"
            if _valid_shell_text_command(body):
                add_call("shell", {"command": body})
            continue

        before_python = len(parsed)
        _parse_python_text_tool_calls(body, allowed, add_call)
        if len(parsed) > before_python:
            continue

        if lang_lower not in {"tools", "tool", "agent", "json", "function_call", "markdown", "md", "text", "python", "py"}:
            continue

        call_match = re.search(
            r"(?:^|\n)\s*call\s+(?:[A-Za-z0-9_.-]+\s+)?([A-Za-z0-9_.-]+)\s*\n(?P<args>.*)",
            body,
            flags=re.DOTALL,
        )
        if call_match:
            name = call_match.group(1)
            try:
                args = _json_object_from_text(call_match.group("args"))
            except Exception:
                args = {"_raw": call_match.group("args").strip()}
            add_call(name, args)
            continue

        try:
            value = _json_object_from_text(body)
        except Exception:
            continue
        _add_tool_calls_from_json_value(value, allowed, add_call)

    if parsed:
        return parsed

    _parse_qwen_tool_call_blocks(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_qwen_angle_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_pipe_delimited_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_qwen_double_pipe_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_command_shell_blocks(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_relaxed_embedded_json_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_embedded_json_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_parameter_named_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_python_answer_writer_tool_call(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_xmlish_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_call_function_fences(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_qwen_masked_shell_blocks(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_python_text_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_bare_function_line_tool_calls(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_bare_spawn_tool_call(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_spawn_agent_plan_tool_call(content, allowed, add_call)
    if parsed:
        return parsed

    _parse_plain_shell_command(content, allowed, add_call)
    if parsed:
        return parsed

    # Some Qwen chat templates emit a plain line followed by a JSON payload.
    call_match = re.search(
        r"(?:^|\n)\s*call\s+(?:[A-Za-z0-9_.-]+\s+)?([A-Za-z0-9_.-]+)\s*\n(?P<args>[{\[].*)",
        content,
        flags=re.DOTALL,
    )
    if call_match:
        try:
            args = _json_object_from_text(call_match.group("args"))
        except Exception:
            args = {"_raw": call_match.group("args").strip()}
        add_call(call_match.group(1), args)

    return parsed


def _get_client(timeout: float = 180.0) -> httpx.AsyncClient:
    """Reuse a single httpx client across all LLM calls (connection pooling)."""
    client = _shared_clients.get(timeout)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=100))
        _shared_clients[timeout] = client
    return client


def _discard_client():
    """Stop reusing the pooled client after transport-level failures.

    Do not close it here: other concurrent agent requests may still be using the
    same client object.
    """
    global _shared_client, _shared_clients
    _shared_client = None
    _shared_clients = {}


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
    tool_choice: Any = None,
    retry_config: RetryConfig | None = None,
) -> LLMResponse:
    base_url = base_url or os.environ.get("NANOMA_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("NANOMA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No API key. Set NANOMA_API_KEY or OPENAI_API_KEY.")
    rc = retry_config or RetryConfig()

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request_messages = list(messages)
    text_only_tools = bool(tools and _text_only_tools_enabled())
    if text_only_tools:
        request_messages = _inject_text_tool_prompt(request_messages, tools or [])
    if not any(msg.get("role") == "user" for msg in request_messages):
        # Some OpenAI-compatible servers reject tool-enabled, system-only chats.
        # NanoMA stores the initial task in the system prompt, so add a tiny
        # user turn to start the first agent loop without changing the task.
        request_messages.append({"role": "user", "content": "Begin the task now."})
    request_messages = [_openai_message_payload(msg) for msg in request_messages]
    body: dict[str, Any] = {
        "model": model, "messages": request_messages,
    }
    if not _omit_max_tokens():
        body["max_tokens"] = max_tokens or _default_max_tokens()
    if not _omit_sampling_params():
        body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
    if tools and not text_only_tools:
        body["tools"] = tools
        body["tool_choice"] = _openai_tool_choice(tool_choice)
    body.update(_openai_extra_body())

    t0 = time.time()
    client = _get_client(rc.http_timeout)
    last_err: Exception | None = None
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            if resp.status_code >= 400 and _is_non_retryable_provider_error(resp):
                _log_http_error(model, body, resp, (time.time() - t0) * 1000)
                resp.raise_for_status()
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                await asyncio.sleep(_retry_sleep_seconds(attempt, rc, resp))
                continue
            if resp.status_code >= 400:
                _log_http_error(model, body, resp, (time.time() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()
            if _is_empty_bad_response(data) and attempt < rc.max_retries:
                await asyncio.sleep(_retry_sleep_seconds(attempt, rc))
                continue
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            if attempt == rc.max_retries:
                raise
            response = e.response if isinstance(e, httpx.HTTPStatusError) else None
            if response is not None and (
                response.status_code not in _RETRYABLE
                or _is_non_retryable_provider_error(response)
            ):
                raise
            if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
                _discard_client()
                client = _get_client(rc.http_timeout)
            await asyncio.sleep(_retry_sleep_seconds(attempt, rc, response))
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
    elif _parse_text_tool_calls_enabled():
        tool_calls = _parse_text_tool_calls(choice.get("content"), tools)

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
    tool_choice: Any = None,
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
        body["tool_choice"] = _anthropic_tool_choice(tool_choice)

    t0 = time.time()
    client = _get_client(rc.http_timeout)
    last_err: Exception | None = None
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/messages", headers=headers, json=body)
            if resp.status_code >= 400 and _is_non_retryable_provider_error(resp):
                _log_http_error(model, body, resp, (time.time() - t0) * 1000)
                resp.raise_for_status()
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                await asyncio.sleep(_retry_sleep_seconds(attempt, rc, resp))
                continue
            if resp.status_code >= 400:
                _log_http_error(model, body, resp, (time.time() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()
            if (
                not data.get("content")
                and data.get("usage", {}).get("input_tokens", 0) == 0
                and data.get("usage", {}).get("output_tokens", 0) == 0
                and attempt < rc.max_retries
            ):
                await asyncio.sleep(_retry_sleep_seconds(attempt, rc))
                continue
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            if attempt == rc.max_retries:
                raise
            response = e.response if isinstance(e, httpx.HTTPStatusError) else None
            if response is not None and (
                response.status_code not in _RETRYABLE
                or _is_non_retryable_provider_error(response)
            ):
                raise
            if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
                _discard_client()
                client = _get_client(rc.http_timeout)
            await asyncio.sleep(_retry_sleep_seconds(attempt, rc, response))
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
