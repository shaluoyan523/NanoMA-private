"""Core runtime: Agent, Runtime, ReAct loop."""

from __future__ import annotations

import asyncio
import copy
import html
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable, Literal
from urllib.parse import parse_qsl, quote_plus, unquote, urlparse

from nanoma.cost import CostLedger, UsageRecord
from nanoma.delivery import DeliveryContract, publish_delivery_contract
from nanoma.llm import (
    LLMResponse, Message, RetryConfig, ToolCall, ToolDef,
    anthropic_compatible_call,
    count_message_tokens, default_llm_call, estimate_tokens,
    openai_compatible_call, set_log_dir,
)
from nanoma.scheduler import Scheduler
from nanoma.fixed_orchestration import (
    FixedAgentSpec,
    FixedCheckpoint,
    FixedOrchestrationPlan,
    FixedVerificationGate,
    load_fixed_orchestration_plan,
)
from nanoma.strategy import (
    CandidateEvidenceGateConfig,
    CandidateEvidenceGateStrategy,
    FinalEvidenceGateConfig,
    FinalEvidenceGateStrategy,
    CandidateFinalizeConfig,
    CandidateFinalizeStrategy,
    LowValueAgentKillConfig,
    LowValueAgentKillStrategy,
    RootRecoverySpawnConfig,
    RootRecoverySpawnStrategy,
    SpawnPortfolioConfig,
    SpawnPortfolioStrategy,
    StrategyAction,
    StrategyDecision,
    StrategyState,
    decision_from_intervention_item,
)
from nanoma.tools import WORK_TOOLS
from nanoma.plugins.workspace_tools import WORKSPACE_TOOLS

logger = logging.getLogger("nanoma")


class _CandidateDeliveryInterrupt(Exception):
    """Internal control flow used to restart a root turn after child delivery."""


class _RuntimeToolOverrideInterrupt(Exception):
    """Internal control flow used to restart a turn after an async tool override."""


# ─── ID Generation (NATO phonetic) ──────────────────────────────────────────

_NATO = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
]

ToolPolicyMode = Literal["off", "adaptive", "enforce"]
ShellCapability = Literal["web", "python", "fs", "process", "package", "system", "unknown"]

_CREATE_TOOLS = {"spawn", "spawn_many"}
_COORDINATION_TOOLS = {"send", "deliver_to_parent", "wait", "query", "kill", "transfer", "set_bio"}
# Above this size a merge diff compares size+mtime instead of hashing contents.
_MERGE_HASH_MAX_BYTES = 8 * 1024 * 1024
# At or above this size a file is hardlinked into a child's working copy instead
# of being duplicated: bulk payloads (datasets, archives, model weights) are read,
# never edited in place, so linking them keeps the copy complete but nearly free.
_MERGE_HARDLINK_MIN_BYTES = 1024 * 1024
_LIFECYCLE_TOOLS = {"get_cost", "set_status", "rebirth", "submit"}
_SHELL_TOOLS = {"shell", "tb_shell"}
_DELIVERY_READ_TOOLS = {
    "ws_read_file", "ws_grep", "ws_code_outline", "ws_read_symbol",
    "tb_read_file", "get_task_context",
}
_DELIVERY_WRITE_TOOLS = {
    "ws_create_file", "ws_append_file", "ws_replace_string",
    "ws_multi_replace", "ws_apply_patch", "tb_write_file",
}
_READ_TOOLS = _DELIVERY_READ_TOOLS | {"query", "get_cost"} | _SHELL_TOOLS
_WORK_TOOLS = _DELIVERY_WRITE_TOOLS | _SHELL_TOOLS | {"batch", "submit"}
_FINISH_TOOLS = {"set_status", "submit", "tb_write_file", "ws_create_file", "ws_append_file"}
_SUPERVISOR_TOPOLOGY_TOOL_ORDER = (
    "spawn_many", "spawn", "query", "wait", "kill", "send", "set_status",
)
_SUPERVISOR_FINISH_TOOL_ORDER = (
    "tb_write_file", "tb_read_file",
    "ws_create_file", "ws_append_file", "ws_replace_string",
    "ws_multi_replace", "ws_apply_patch",
    "ws_read_file", "ws_grep",
    "tb_shell", "shell", "submit", "set_status", "get_cost",
)
_SUPERVISOR_FINISH_TOOLS = set(_SUPERVISOR_FINISH_TOOL_ORDER)
_SUPERVISOR_DELIVERY_TOOLS = (
    _DELIVERY_WRITE_TOOLS
    | _SHELL_TOOLS
    | {"submit"}
)
_MALFORMED_WRITE_PENDING = "malformed_write_pending"
_CHILD_EVIDENCE_DELIVERY_PREFIX = "child_evidence_delivery_after:"
_OUTPUT_SUFFIXES = {
    ".answer", ".csv", ".html", ".json", ".jsonl", ".md", ".out",
    ".pdf", ".txt", ".tsv", ".xml", ".yaml", ".yml",
}
_SHELL_CAPABILITIES = {"web", "python", "fs", "process", "package", "system", "unknown"}
_BACKTICK_PATH_RE = re.compile(r"`([^`]+)`")
_NAMED_PATH_RE = re.compile(r"\bnamed\s+([A-Za-z0-9_./$-]+\.[A-Za-z0-9]+)", re.IGNORECASE)
_OUTPUT_PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./$-])([$A-Za-z0-9_./-]+\.(?:answer|csv|html|jsonl?|md|out|pdf|txt|tsv|xml|ya?ml))",
    re.IGNORECASE,
)
_MARKDOWN_TABLE_LINE_RE = re.compile(r"^\s*\|[^|\n]+\|", re.MULTILINE)
_CSVISH_LINE_RE = re.compile(r"^\s*[^,\t\n]+[,\t][^,\t\n]+(?:[,\t][^,\t\n]+)+\s*$")
_WEB_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_WEB_LOW_SIGNAL_PATTERNS = (
    "access denied",
    "attention required",
    "blocked",
    "captcha",
    "cloudflare",
    "forbidden",
    "no results",
    "not found",
    "permission denied",
    "rate limit",
    "too many requests",
    "robot check",
    "temporarily unavailable",
    "traceback",
    "unrecognized parameters",
    "unrecognized value for parameter",
    "validation-failure",
    "wikimedia error",
)
_WEB_HARD_BLOCK_PATTERNS = (
    "checking your connection",
    "verify you are human",
    "robot check",
    "captcha",
    "cf-chl-",
)
_WEB_SEARCH_DOMAINS = (
    "bing.com",
    "duckduckgo.com",
    "google.com",
    "mojeek.com",
    "search.brave.com",
    "searx.",
    "yahoo.com",
    "yandex.",
)
_TASK_KEYWORD_STOPWORDS = {
    "about", "after", "again", "answer", "article", "before", "being", "could",
    "final", "first", "given", "have", "into", "need", "only", "otherwise",
    "paper", "question", "return", "should", "their", "there", "these", "thing",
    "this", "using", "what", "when", "where", "which", "with", "without", "would",
}


def _first_effective_shell_command(command: str) -> str:
    """Return the first non-comment shell line for capability classification."""
    for line in command.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = re.sub(
            r"^(?:(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*="
            r"(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)\s*(?:(?:&&|;)\s*)?)+",
            "",
            stripped,
        ).lstrip()
        stripped = re.sub(
            r"^(?:cd\s+(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)\s*&&\s*)+",
            "",
            stripped,
        ).lstrip()
        return stripped
    return command.strip()


def _repair_bare_python_block(command: str) -> str:
    """Wrap a parsed fenced-Python body that arrived as a bare shell command."""
    lines = command.splitlines()
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None or lines[first].strip().lower() not in {"python", "python3"}:
        return command
    body = "\n".join(lines[first + 1:]).strip()
    if not body:
        return command
    try:
        compile(body, "<nanoma-fenced-python>", "exec")
    except SyntaxError:
        return command
    delimiter = "NANOMA_FENCED_PY"
    if any(line.strip() == delimiter for line in body.splitlines()):
        delimiter = "NANOMA_FENCED_PY_END"
    return f"python3 - <<'{delimiter}'\n{body}\n{delimiter}"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _looks_like_output_path(text: str) -> bool:
    text = text.strip().strip("'\".,);:")
    if not text or "://" in text:
        return False
    return Path(text).suffix.lower() in _OUTPUT_SUFFIXES


def _extract_expected_output_paths(task: str) -> list[str]:
    """Best-effort extraction for explicit deliverable paths in task prompts."""
    if not isinstance(task, str):
        try:
            task = json.dumps(task, ensure_ascii=False)
        except Exception:
            task = str(task)
    paths: list[str] = []
    for raw in re.findall(r"`+([^`\n]+?)`+", task):
        raw = raw.strip()
        if _looks_like_output_path(raw):
            paths.append(raw.strip("'\".,);:"))
    for match in _OUTPUT_PATH_TOKEN_RE.finditer(task):
        raw = match.group(1).strip()
        if _looks_like_output_path(raw):
            paths.append(raw.strip("'\".,);:"))
    for match in _NAMED_PATH_RE.finditer(task):
        raw = match.group(1).strip()
        if _looks_like_output_path(raw):
            paths.append(raw.strip("'\".,);:"))

    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        key = path.lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _task_expects_structured_collection(task: str) -> bool:
    """Whether the task asks for a multi-row/table-like deliverable.

    This is a generic task-shape signal, not a benchmark switch. It lets the
    low-concurrency bypass stay faithful for short-answer tasks while still
    allowing constraint to step in for long collection tasks stuck in web loops.
    """
    text = task.lower()
    if re.search(r"\|\s*[^|\n]+\s*\|", task):
        return True
    table_markers = (
        "markdown table",
        "one markdown table",
        "fenced ```markdown",
        "```markdown",
        "table columns",
        "columns in order",
        "column names",
        "year-by-year",
        "breakdown",
        "rows must be unique",
        "output the results in one markdown table",
        "表格",
        "列名",
        "景区名称",
        "不要拆分成多个markdown表格",
    )
    if any(marker in text for marker in table_markers):
        return True
    expected = _extract_expected_output_paths(task)
    return any(Path(path).suffix.lower() in {".csv", ".tsv", ".md"} for path in expected)


def classify_shell_capability(command: str) -> ShellCapability:
    """Classify a shell command for internal policy pruning.

    The public tool remains a single `shell` function. This classification is
    used only by the runtime to narrow what that tool may execute under
    constraint pressure.
    """
    cmd = _first_effective_shell_command(command)
    if not cmd:
        return "unknown"

    first = re.split(r"\s+", cmd, maxsplit=1)[0].split("/")[-1]
    lowered = command.lower()

    python_network_markers = (
        "http.client",
        "requests.",
        "urllib.",
        "socket.",
        "ssl.",
        "aiohttp",
        "httpx",
        "urlopen",
        "wrap_socket",
        "create_connection",
    )

    imports_network_module = bool(
        re.search(r"\b(?:import|from)\s+(?:aiohttp|http\.client|httpx|requests|socket|ssl|urllib)\b", lowered)
        or re.search(r"\bimport\s+[A-Za-z0-9_.,\s]*(?:socket|ssl|urllib|requests|httpx|aiohttp)\b", lowered)
    )

    if (
        first in {"curl", "wget"}
        or re.search(r"https?://", lowered)
        or any(marker in lowered for marker in python_network_markers)
        or imports_network_module
    ):
        return "web"
    if first in {"python", "python3", "python2"} or re.match(r"python\d?\s*<<", lowered):
        return "python"
    if first in {
        "cat", "cd", "cp", "du", "echo", "file", "find", "head", "ls", "mkdir",
        "grep", "mv", "pwd", "realpath", "rm", "rmdir", "sed", "sort", "stat",
        "tail", "tee", "touch", "tree", "uniq", "wc",
    }:
        return "fs"
    if first in {
        "c++", "cc", "clang", "clang++", "g++", "gcc", "go", "javac", "ps",
        "pkill", "kill", "killall", "jobs", "pgrep", "rustc", "sleep", "timeout",
    }:
        return "process"
    if first in {"apt", "apt-get", "brew", "conda", "npm", "npx", "pip", "pip3", "pnpm", "yarn"}:
        return "package"
    if first in {
        "bash", "chmod", "chown", "docker", "git", "make", "node", "perl", "ruby",
        "sh", "sudo", "tar", "unzip", "xz", "zip",
    }:
        return "system"
    return "unknown"


def _extract_web_urls(command: str) -> list[str]:
    urls: list[str] = []
    for match in _WEB_URL_RE.finditer(command):
        url = match.group(0).rstrip("),.;]")
        if url:
            urls.append(url)
    return urls


def _web_command_domain(command: str) -> str:
    urls = _extract_web_urls(command)
    if not urls:
        return ""
    try:
        return (urlparse(urls[0]).netloc or "").lower()
    except Exception:
        return ""


def _web_command_query(command: str) -> str:
    for url in _extract_web_urls(command):
        try:
            parsed = urlparse(url)
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        except Exception:
            continue
        value = (
            query.get("q")
            or query.get("query")
            or query.get("search")
            or query.get("srsearch")
            or query.get("title")
            or ""
        )
        if value:
            decoded = unquote(str(value)).strip()
            if re.search(r"[{}]|\$\(|\b(?:quote|quote_plus|urlencode)\s*\(", decoded):
                return ""
            return decoded
    return ""


def _web_command_writes_download(command: str) -> bool:
    """Return true when a web command explicitly persists the response to a file."""
    value = str(command or "")
    return bool(
        re.search(r"(?:^|\s)(?:-o|--output)(?:\s+|=)[^\s;&|]+", value)
        or re.search(r"(?:^|\s)(?:-O|--output-document)(?:\s+|=)[^\s;&|]+", value)
    )


def _web_command_writes_html_download(command: str) -> bool:
    """Return true when the persisted response is an HTML page, not a research asset."""
    value = str(command or "")
    matches = re.findall(
        r"(?:^|\s)(?:-o|--output|-O|--output-document)(?:\s+|=)([^\s;&|]+)",
        value,
    )
    return any(
        str(path).strip("'\"").lower().split("?", 1)[0].endswith((".html", ".htm"))
        for path in matches
    )


def _bing_rss_search_sync(query: str, limit: int = 8) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"provider": "bing_rss", "query": "", "results": []}
    url = f"https://www.bing.com/search?q={quote_plus(query)}&format=rss"
    request = urllib.request.Request(url, headers={"User-Agent": "NanoMA/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    results: list[dict[str, str]] = []
    for item in root.findall(".//item")[:max(1, int(limit))]:
        title = html.unescape(str(item.findtext("title") or "")).strip()
        link = str(item.findtext("link") or "").strip()
        description = html.unescape(str(item.findtext("description") or ""))
        description = re.sub(r"<[^>]+>", " ", description)
        description = re.sub(r"\s+", " ", description).strip()
        if title or link or description:
            results.append({
                "title": title[:500],
                "link": link[:1000],
                "snippet": description[:1200],
            })
    return {"provider": "bing_rss", "query": query, "results": results}


def _plain_html_text(value: str) -> str:
    value = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<style\b[^>]*>.*?</style>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _yahoo_result_url(value: str) -> str:
    value = html.unescape(str(value or "")).strip()
    match = re.search(r"/RU=([^/]+)/RK=", value)
    if match:
        return unquote(match.group(1))
    return value


def _yahoo_search_sync(query: str, limit: int = 8) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"provider": "yahoo_search", "query": "", "results": []}
    url = f"https://search.yahoo.com/search?p={quote_plus(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read().decode(errors="replace")
    pattern = re.compile(
        r'<div class="compTitle[^"]*">.*?<a[^>]*href="([^"]+)"[^>]*>.*?'
        r'<h3[^>]*>(.*?)</h3>.*?</div>\s*'
        r'(?:<div class="compText[^"]*">.*?<p[^>]*>(.*?)</p>)?',
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for href, title_html, snippet_html in pattern.findall(payload):
        title = _plain_html_text(title_html)
        link = _yahoo_result_url(href)
        snippet = _plain_html_text(snippet_html)
        if not title or not link or "search.yahoo.com/search" in link:
            continue
        results.append({
            "title": title[:500],
            "link": link[:1000],
            "snippet": snippet[:1200],
        })
        if len(results) >= max(1, int(limit)):
            break
    return {"provider": "yahoo_search", "query": query, "results": results}


def _structured_web_search_sync(query: str, limit: int = 8) -> dict[str, Any]:
    try:
        yahoo = _yahoo_search_sync(query, limit=limit)
        if yahoo.get("results"):
            return yahoo
    except Exception:
        pass
    return _bing_rss_search_sync(query, limit=limit)


def _web_command_signature(command: str) -> str:
    urls = _extract_web_urls(command)
    if not urls:
        return re.sub(r"\s+", " ", command.strip().lower())[:240]

    try:
        parsed = urlparse(urls[0])
    except Exception:
        return urls[0].lower()[:240]

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_text = (
        query.get("q")
        or query.get("query")
        or query.get("search")
        or query.get("srsearch")
        or query.get("title")
        or ""
    )
    if query_text:
        if re.search(r"[{}]|\$\(|\b(?:quote|quote_plus|urlencode)\s*\(", query_text):
            return ""
        query_part = unquote(query_text).lower()
    else:
        stable_params = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in {"api_key", "apikey", "key", "token", "access_token"}
        ][:6]
        query_part = "&".join(f"{k}={v}" for k, v in stable_params).lower()

    path = parsed.path.rstrip("/") or "/"
    return re.sub(
        r"\s+",
        " ",
        f"{(parsed.netloc or '').lower()}{path.lower()}?{query_part}",
    )[:240]


def _web_result_low_signal(result: Any) -> bool:
    if not isinstance(result, dict):
        return True

    try:
        exit_code = int(result.get("exit_code", 0))
    except Exception:
        exit_code = 0
    if exit_code != 0:
        return True

    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip()
    if len(text) < 80:
        return True

    lowered = text.lower()
    if any(pattern in lowered for pattern in _WEB_LOW_SIGNAL_PATTERNS):
        return True

    if re.search(r'"(?:totalhits|count|total|num_found)"\s*:\s*0\b', lowered):
        return True
    if re.search(r'"(?:items|results|search)"\s*:\s*\[\s*\]', lowered):
        return True

    return False


def _web_result_hard_block(result: Any) -> bool:
    """Detect explicit remote denial/challenge pages, not merely weak content."""
    if not isinstance(result, dict):
        return False
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip().lower()
    if not text:
        return False
    head = text[:12000]
    headings = " ".join(re.findall(
        r"<(?:title|h1)[^>]*>(.*?)</(?:title|h1)>",
        head,
        flags=re.DOTALL,
    ))
    headings = re.sub(r"<[^>]+>", " ", headings)
    headings = re.sub(r"\s+", " ", headings).strip()
    if any(pattern in headings for pattern in _WEB_HARD_BLOCK_PATTERNS):
        return True
    if (
        ("cf-chl-" in head or "challenge-platform" in head)
        and ("just a moment" in headings or "checking your connection" in head[:2000])
    ):
        return True
    status_pattern = r"(?:\b403\b.{0,80}\bforbidden\b|\b429\b.{0,80}\btoo many requests\b)"
    if re.search(status_pattern, headings, re.DOTALL):
        return True
    return len(text) <= 6000 and bool(re.search(status_pattern, head, re.DOTALL))


def _web_result_text(result: Any) -> str:
    if isinstance(result, dict):
        return f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip()
    return str(result)


def _stable_text_digest(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(normalized[:20000].encode("utf-8", errors="ignore")).hexdigest()


def _is_search_domain(domain: str) -> bool:
    domain = domain.lower()
    return any(marker in domain for marker in _WEB_SEARCH_DOMAINS)


def _task_keywords(task: str) -> set[str]:
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{4,}", task)
        if word.lower() not in _TASK_KEYWORD_STOPWORDS
    }
    return set(list(words)[:80])


def _task_keyword_overlap(task: str, text: str) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for word in _task_keywords(task) if word in text_lower)


def _structured_output_score(text: str) -> float:
    if not text:
        return 0.0
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    table_lines = len(_MARKDOWN_TABLE_LINE_RE.findall(text))
    csvish_lines = sum(1 for line in lines if _CSVISH_LINE_RE.match(line))
    fenced = "```" in text
    jsonish = text.lstrip().startswith(("{", "[")) and text.rstrip().endswith(("}", "]"))
    numeric_items = len(re.findall(r"(?:FY)?\d{4}|\$?\d+(?:\.\d+)?", text))
    score = 0.0
    if table_lines >= 3:
        score = max(score, min(1.0, 0.35 + table_lines / 18.0))
    if csvish_lines >= 4:
        score = max(score, min(1.0, 0.30 + csvish_lines / 20.0))
    if fenced and len(lines) >= 4:
        score = max(score, 0.50)
    if jsonish and len(text) >= 250:
        score = max(score, 0.55)
    if numeric_items >= 8 and len(lines) >= 4:
        score = max(score, min(0.85, 0.30 + numeric_items / 40.0))
    return _clamp01(score)


def _task_requires_source_content_evidence(task: str) -> bool:
    lowered = task.lower()
    return bool(
        re.search(
            r"\b(?:quote|quoted|quotation|wording|passage|exact\s+text|"
            r"stanza|verse|indent(?:ed|ation)?)\b",
            lowered,
        )
        or re.search(
            r"\b(?:line|word|citation)\b.{0,120}\b(?:match|differ|correct|actual|source)\b",
            lowered,
            flags=re.DOTALL,
        )
    )


def _is_bibliographic_metadata_record(text: str) -> bool:
    lowered = text.lower()
    # Crossref work records are useful routing evidence, but never expose the
    # source wording or layout needed for quotation and stanza questions.
    return all(
        marker in lowered
        for marker in ('"message-type"', '"reference-count"', '"doi"', '"publisher"')
    )


def _looks_like_candidate_output(task: str, text: str) -> bool:
    """Runtime-only signal that a tool result contains a deliverable-shaped answer."""
    if len(text) < 350:
        return False
    if (
        _task_requires_source_content_evidence(task)
        and _is_bibliographic_metadata_record(text)
    ):
        return False
    score = _structured_output_score(text)
    if score < 0.45:
        return False
    keyword_overlap = _task_keyword_overlap(task, text)
    numeric_items = len(re.findall(r"(?:FY)?\d{4}|\$?\d+(?:\.\d+)?", text))
    return keyword_overlap >= 2 or numeric_items >= 8 or (score >= 0.80 and len(text) >= 800)


class IdGenerator:
    def __init__(self):
        self._counter = 0

    def next(self) -> str:
        name = _NATO[self._counter % len(_NATO)]
        suffix = self._counter // len(_NATO)
        self._counter += 1
        return f"{name}-{suffix}" if suffix else name


# ─── Envelope ────────────────────────────────────────────────────────────────

@dataclass
class Envelope:
    from_id: str
    to_id: str
    content: str
    tokens: int
    timestamp: float
    priority: int = 0
    mode: Literal["immediate", "steer", "queue"] = "queue"


# ─── ResourceQuota ───────────────────────────────────────────────────────────

@dataclass
class ResourceQuota:
    budget: float = 10.0
    time_limit: float = 0.0      # 0 = unlimited
    max_turns: int = 200


# ─── Artifact ────────────────────────────────────────────────────────────────

@dataclass
class Artifact:
    path: str
    absolute_path: Path
    description: str = ""
    agent_id: str = ""


# ─── ToolContext ─────────────────────────────────────────────────────────────

@dataclass
class ToolContext:
    shared_dir: Path
    workspace_root: Path
    workspace_extra_roots: tuple[Path, ...] = ()
    shell_max_output: int = 10000
    shell_max_timeout: int = 30
    file_read_max_chars: int = 50000
    file_list_max_entries: int = 500
    grep_max_results: int = 100
    blocked_shell_patterns: list[str] = field(default_factory=list)
    allowed_shell_capabilities: set[str] = field(default_factory=lambda: set(_SHELL_CAPABILITIES))


# ─── Tool Policy ─────────────────────────────────────────────────────────────

@dataclass
class ToolPolicyState:
    create: float = 0.0
    read: float = 0.0
    message: float = 0.0
    work: float = 0.0
    finish: float = 0.0
    resource_pressure: float = 0.0
    dependency_pressure: float = 0.0
    artifact_gap: float = 0.0
    stagnation_pressure: float = 0.0
    delivery_pressure: float = 0.0
    delivery_phase: str = "explore"
    web_loop_pressure: float = 0.0
    topology_pressure: float = 0.0
    reconcile_phase: bool = False
    expected_outputs: int = 0
    missing_outputs: int = 0
    active_children: int = 0
    reason: str = "off"
    scoped_tools: list[str] = field(default_factory=list)
    removed_tools: list[str] = field(default_factory=list)
    bypassed: bool = False

    def as_event(self) -> dict[str, Any]:
        return {
            "weights": {
                "create": round(self.create, 3),
                "read": round(self.read, 3),
                "message": round(self.message, 3),
                "work": round(self.work, 3),
                "finish": round(self.finish, 3),
            },
            "pressures": {
                "resource": round(self.resource_pressure, 3),
                "dependency": round(self.dependency_pressure, 3),
                "artifact_gap": round(self.artifact_gap, 3),
                "stagnation": round(self.stagnation_pressure, 3),
                "delivery": round(self.delivery_pressure, 3),
                "web_loop": round(self.web_loop_pressure, 3),
                "topology": round(self.topology_pressure, 3),
            },
            "delivery_phase": self.delivery_phase,
            "reconcile_phase": self.reconcile_phase,
            "expected_outputs": self.expected_outputs,
            "missing_outputs": self.missing_outputs,
            "active_children": self.active_children,
            "reason": self.reason,
            "bypassed": self.bypassed,
            "scoped_tools": self.scoped_tools,
            "removed_tools": self.removed_tools,
        }


@dataclass
class ShellActivityState:
    web_calls: int = 0
    web_low_signal_calls: int = 0
    blocked_web_calls: int = 0
    large_output_calls: int = 0
    truncated_output_calls: int = 0
    consecutive_large_outputs: int = 0
    output_bytes: int = 0
    consecutive_web_low_signal: int = 0
    repeated_web_queries: int = 0
    repeated_web_domains: int = 0
    web_no_gain_calls: int = 0
    consecutive_web_no_gain: int = 0
    repeated_web_results: int = 0
    search_result_calls: int = 0
    task_keyword_hits: int = 0
    material_gain_calls: int = 0
    candidate_like_outputs: int = 0
    candidate_like_output_bytes: int = 0
    candidate_prepare_open_turn: int = 0
    candidate_prepare_open_non_delivery_tool_calls: int = 0
    candidate_prepare_finalize_turn: int = 0
    candidate_prepare_finalize_non_delivery_tool_calls: int = 0
    unavailable_tool_calls: int = 0
    web_recovery_calls_remaining: int = 0
    web_recovery_grants: int = 0
    web_recovery_last_write_calls: int = 0
    unique_web_signatures: set[str] = field(default_factory=set)
    unique_web_domains: set[str] = field(default_factory=set)
    unique_web_result_digests: set[str] = field(default_factory=set)
    hard_blocked_web_signatures: set[str] = field(default_factory=set)
    last_web_signature: str = ""
    last_web_domain: str = ""
    last_web_result_digest: str = ""
    web_domain_sprawl: float = 0.0
    web_saturation: float = 0.0
    web_loop_pressure: float = 0.0
    reconcile_after_web_loop: bool = False
    finalize_after_web_saturation: bool = False

    def as_event(self) -> dict[str, Any]:
        return {
            "web_calls": self.web_calls,
            "web_low_signal_calls": self.web_low_signal_calls,
            "blocked_web_calls": self.blocked_web_calls,
            "large_output_calls": self.large_output_calls,
            "truncated_output_calls": self.truncated_output_calls,
            "consecutive_large_outputs": self.consecutive_large_outputs,
            "output_bytes": self.output_bytes,
            "consecutive_web_low_signal": self.consecutive_web_low_signal,
            "repeated_web_queries": self.repeated_web_queries,
            "repeated_web_domains": self.repeated_web_domains,
            "web_no_gain_calls": self.web_no_gain_calls,
            "consecutive_web_no_gain": self.consecutive_web_no_gain,
            "repeated_web_results": self.repeated_web_results,
            "search_result_calls": self.search_result_calls,
            "task_keyword_hits": self.task_keyword_hits,
            "material_gain_calls": self.material_gain_calls,
            "candidate_like_outputs": self.candidate_like_outputs,
            "candidate_like_output_bytes": self.candidate_like_output_bytes,
            "candidate_prepare_open_turn": self.candidate_prepare_open_turn,
            "candidate_prepare_open_non_delivery_tool_calls": self.candidate_prepare_open_non_delivery_tool_calls,
            "candidate_prepare_finalize_turn": self.candidate_prepare_finalize_turn,
            "candidate_prepare_finalize_non_delivery_tool_calls": self.candidate_prepare_finalize_non_delivery_tool_calls,
            "unavailable_tool_calls": self.unavailable_tool_calls,
            "web_recovery_calls_remaining": self.web_recovery_calls_remaining,
            "web_recovery_grants": self.web_recovery_grants,
            "unique_web_signatures": len(self.unique_web_signatures),
            "unique_web_domains": len(self.unique_web_domains),
            "unique_web_result_digests": len(self.unique_web_result_digests),
            "hard_blocked_web_signatures": len(self.hard_blocked_web_signatures),
            "last_web_domain": self.last_web_domain,
            "web_domain_sprawl": round(self.web_domain_sprawl, 3),
            "web_saturation": round(self.web_saturation, 3),
            "web_loop_pressure": round(self.web_loop_pressure, 3),
            "reconcile_after_web_loop": self.reconcile_after_web_loop,
            "finalize_after_web_saturation": self.finalize_after_web_saturation,
        }


@dataclass
class DeliveryActivityState:
    write_calls: int = 0
    submit_calls: int = 0
    non_delivery_tool_calls: int = 0
    deliver_enter_turn: int = 0
    deliver_enter_non_delivery_tool_calls: int = 0
    candidate_ready_turn: int = 0
    candidate_ready_non_delivery_tool_calls: int = 0
    candidate_review_calls: int = 0
    candidate_source_review_calls: int = 0
    candidate_output_files: set[str] = field(default_factory=set)
    expected_output_files: set[str] = field(default_factory=set)
    container_candidate_output_files: set[str] = field(default_factory=set)
    last_write_turn: int = 0
    last_submit_turn: int = 0

    def as_event(self) -> dict[str, Any]:
        return {
            "write_calls": self.write_calls,
            "submit_calls": self.submit_calls,
            "non_delivery_tool_calls": self.non_delivery_tool_calls,
            "deliver_enter_turn": self.deliver_enter_turn,
            "deliver_enter_non_delivery_tool_calls": self.deliver_enter_non_delivery_tool_calls,
            "candidate_ready_turn": self.candidate_ready_turn,
            "candidate_ready_non_delivery_tool_calls": self.candidate_ready_non_delivery_tool_calls,
            "candidate_review_calls": self.candidate_review_calls,
            "candidate_source_review_calls": self.candidate_source_review_calls,
            "candidate_output_files": len(self.candidate_output_files),
            "expected_output_files": len(self.expected_output_files),
            "container_candidate_output_files": len(self.container_candidate_output_files),
            "last_write_turn": self.last_write_turn,
            "last_submit_turn": self.last_submit_turn,
        }


# ─── Configuration ───────────────────────────────────────────────────────────

def _runtime_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _runtime_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _runtime_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


@dataclass
class RuntimeConfig:
    max_agents: int = 1000
    max_depth: int = 100
    max_concurrent_llm: int = 50
    llm_admission_control: bool = field(
        default_factory=lambda: _runtime_env_bool("NANOMA_LLM_ADMISSION_CONTROL", False)
    )
    llm_min_start_spacing: float = field(
        default_factory=lambda: max(0.0, _runtime_env_float("NANOMA_LLM_MIN_START_SPACING", 0.0))
    )
    llm_large_context_tokens: int = field(
        default_factory=lambda: max(0, _runtime_env_int("NANOMA_LLM_LARGE_CONTEXT_TOKENS", 32000))
    )
    llm_large_context_spacing: float = field(
        default_factory=lambda: max(0.0, _runtime_env_float("NANOMA_LLM_LARGE_CONTEXT_SPACING", 0.0))
    )
    llm_overload_cooldown_seconds: float = field(
        default_factory=lambda: max(0.0, _runtime_env_float("NANOMA_LLM_OVERLOAD_COOLDOWN_SECONDS", 0.0))
    )
    llm_admission_max_delay: float = field(
        default_factory=lambda: max(0.0, _runtime_env_float("NANOMA_LLM_ADMISSION_MAX_DELAY", 120.0))
    )
    budget: float = 10.0
    max_total_tokens: int = 0
    tool_policy_soft_total_tokens: int = 0
    time_limit: float = 0.0
    max_turns: int = 200
    malformed_tool_repair_turns: int = field(
        default_factory=lambda: max(0, _runtime_env_int("NANOMA_MALFORMED_TOOL_REPAIR_TURNS", 3))
    )
    malformed_tool_fail_after: int = field(
        default_factory=lambda: max(1, _runtime_env_int("NANOMA_MALFORMED_TOOL_FAIL_AFTER", 8))
    )
    llm_400_compact_retry_enabled: bool = field(
        default_factory=lambda: _runtime_env_bool("NANOMA_LLM_400_COMPACT_RETRY", True)
    )
    llm_400_compact_keep_recent: int = field(
        default_factory=lambda: max(2, _runtime_env_int("NANOMA_LLM_400_COMPACT_KEEP_RECENT", 8))
    )
    llm_400_compact_max_retries_per_agent: int = field(
        default_factory=lambda: max(0, _runtime_env_int("NANOMA_LLM_400_COMPACT_MAX_RETRIES_PER_AGENT", 8))
    )
    allowed_models: list[str] | None = None
    disabled_tools: set[str] = field(default_factory=set)
    extra_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    context_compress_ratio: float = 0.8
    default_model: str = "deepseek-v4-flash"
    log_dir: Path | None = field(default_factory=lambda: Path("./logs"))
    workspace_root: Path = field(default_factory=lambda: Path("./workspace"))
    workspace_extra_roots: list[Path] = field(default_factory=list)
    shared_dir: str = "shared"
    delivery_contract: DeliveryContract | None = None
    system_extra_instructions: str = field(
        default_factory=lambda: os.environ.get("NANOMA_SYSTEM_EXTRA_INSTRUCTIONS", "")
    )
    retry: RetryConfig = field(default_factory=RetryConfig)
    # Runtime tool policy: state variables narrow tool availability without
    # injecting policy text into the agent loop.
    tool_policy_mode: ToolPolicyMode = "adaptive"
    tool_policy_spawn_min_weight: float = 0.30
    tool_policy_resource_threshold: float = 0.85
    tool_policy_dependency_threshold: float = 0.65
    tool_policy_dependency_window: int = 8
    tool_policy_finish_threshold: float = 0.75
    tool_policy_prune_tools: bool = False
    tool_policy_prune_min_tools: int = 6
    tool_policy_prune_pressure_start: float = 0.65
    tool_policy_prune_pressure_end: float = 0.95
    tool_policy_prune_preserve_tools: set[str] = field(default_factory=lambda: {
        "shell", "tb_shell", "get_cost", "set_status", "submit", "deliver_to_parent",
        "get_task_context", "ws_read_file", "tb_read_file", "ws_create_file", "ws_append_file", "tb_write_file",
    })
    tool_policy_prune_shell_capabilities: bool = False
    tool_policy_shell_capability_pressure_start: float = 0.55
    tool_policy_shell_capability_pressure_end: float = 0.95
    tool_policy_low_concurrency_bypass_enabled: bool = True
    tool_policy_low_concurrency_max_active_agents: int = 1
    tool_policy_low_concurrency_resource_threshold: float = 0.45
    tool_policy_low_concurrency_web_loop_threshold: float = 0.75
    tool_policy_web_saturation_enabled: bool = True
    tool_policy_web_saturation_min_calls: int = 10
    tool_policy_web_saturation_threshold: float = 0.85
    tool_policy_web_saturation_finalize_after_blocks: int = 2
    tool_policy_web_loop_reconcile_enabled: bool = True
    tool_policy_web_loop_min_calls: int = 16
    tool_policy_web_loop_no_gain_threshold: float = 0.70
    tool_policy_web_loop_consecutive_no_gain: int = 8
    tool_policy_web_loop_pressure_threshold: float = 0.72
    tool_policy_web_loop_recovery_calls: int = 3
    tool_policy_web_loop_shell_close_after_blocks: int = 3
    tool_policy_hard_constraint_topology_threshold: float = 0.10
    tool_policy_single_agent_web_reconcile_threshold: float = 0.92
    tool_policy_delivery_enabled: bool = True
    tool_policy_delivery_web_calls_start: int = 24
    tool_policy_delivery_web_calls_end: int = 64
    tool_policy_delivery_turns_start: int = 18
    tool_policy_delivery_turns_end: int = 52
    tool_policy_delivery_tool_calls_start: int = 36
    tool_policy_delivery_tool_calls_end: int = 96
    tool_policy_delivery_pressure_threshold: float = 0.55
    tool_policy_delivery_prepare_enabled: bool = True
    tool_policy_delivery_prepare_web_calls: int = 10
    tool_policy_delivery_prepare_min_candidate_outputs: int = 3
    tool_policy_delivery_prepare_local_candidate_outputs: int = 3
    tool_policy_delivery_prepare_resource_threshold: float = 0.72
    tool_policy_delivery_prepare_activity_threshold: float = 0.55
    tool_policy_delivery_prepare_hold_tool_calls: int = 12
    tool_policy_delivery_prepare_shell_window: int = 6
    tool_policy_delivery_consolidate_shell_close_after: int = 0
    tool_policy_delivery_prepare_blocked_web_finalize_after: int = 2
    tool_policy_delivery_prepare_finalize_shell_window: int = 2
    tool_policy_delivery_prepare_unavailable_stop_after: int = 3
    tool_policy_delivery_verify_enabled: bool = True
    tool_policy_delivery_verify_web_calls: int = 16
    tool_policy_delivery_verify_min_candidate_reviews: int = 1
    tool_policy_delivery_verify_min_source_reviews: int = 1
    tool_policy_delivery_verify_resource_cutoff: float = 0.92
    tool_policy_log_events: bool = False
    # Resource notification thresholds (fraction consumed, e.g. 0.5 = 50%)
    notify_thresholds: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.70, 0.80, 0.90, 0.95])
    # Compression / truncation settings
    compress_keep_recent: int = 6           # messages to keep verbatim during compression
    compress_max_messages: int = 40         # max old messages to include in summary
    compress_max_chars: int = 300           # max chars per message in summary (0 = unlimited)
    shell_max_output: int = 10000           # max chars for shell output (0 = unlimited)
    shell_max_timeout: int = 30             # max seconds per shell call (0 = unlimited)
    file_read_max_chars: int = 50000        # max chars for file_read (0 = unlimited)
    file_list_max_entries: int = 500        # max entries for file_list (0 = unlimited)
    grep_max_results: int = 100             # max grep results (0 = unlimited)
    blocked_shell_patterns: list[str] = field(default_factory=list)
    probe_dir: Path | None = None
    probe_every_llm: bool = False
    probe_stop_after: int = 0
    auto_checkpoint_enabled: bool = False
    auto_checkpoint_dir: Path | None = None
    auto_rollback_enabled: bool = False
    auto_rollback_max_attempts: int = 2
    auto_rollback_on_infra_failure: bool = True
    auto_rollback_on_candidate_failure: bool = True
    auto_rollback_instruction: str | None = None
    force_spawn_turns: int = 0
    force_spawn_many: bool = False
    force_spawn_auto_recover_no_tool: bool = True
    force_spawn_auto_recover_output_token_threshold: int = 2000
    # Children receive a bounded, runtime-owned task capsule instead of a copy
    # of the complete root prompt.  The full source remains available in small
    # on-demand chunks through get_task_context.
    task_capsule_enabled: bool = True
    task_capsule_max_tokens: int = 900
    task_capsule_inline_root_max_tokens: int = 650
    task_context_chunk_max_tokens: int = 1200
    candidate_delivery_ledger_enabled: bool = True
    candidate_convergence_enabled: bool = False
    candidate_convergence_time_fraction: float = 0.75
    candidate_convergence_max_root_turns: int = 2
    candidate_convergence_min_confidence: float = 0.50
    candidate_convergence_high_confidence_threshold: float = 0.90
    candidate_llm_interrupt_enabled: bool = True
    candidate_root_park_enabled: bool = False
    candidate_root_park_poll_seconds: float = 1.0
    single_tool_max_tokens: int = field(
        default_factory=lambda: max(
            0, _runtime_env_int("NANOMA_SINGLE_TOOL_MAX_TOKENS", 900)
        )
    )
    control_tool_max_tokens: int = field(
        default_factory=lambda: max(
            0, _runtime_env_int("NANOMA_CONTROL_TOOL_MAX_TOKENS", 600)
        )
    )
    delivery_tool_max_tokens: int = field(
        default_factory=lambda: max(
            0, _runtime_env_int("NANOMA_DELIVERY_TOOL_MAX_TOKENS", 384)
        )
    )
    child_first_turn_max_tokens: int = field(
        default_factory=lambda: max(
            0, _runtime_env_int("NANOMA_CHILD_FIRST_TURN_MAX_TOKENS", 600)
        )
    )
    candidate_deadline_fallback_enabled: bool = True
    child_delivery_tool_required: bool = True
    child_delivery_auto_complete: bool = True
    child_explicit_candidate_recovery_enabled: bool = True
    child_action_repair_enabled: bool = True
    child_action_repair_output_token_threshold: int = 2000
    child_action_delivery_override_after_output_limits: int = 2
    child_evidence_checkpoint_enabled: bool = True
    child_kill_delivery_grace_enabled: bool = True
    # A root's first completion attempt becomes the concrete candidate that an
    # ordinary child must inspect.  Revised candidates may be reviewed again,
    # up to this bounded number of rounds; the root can explicitly override a
    # disputed review with a reason instead of being trapped in a loop.
    final_candidate_review_enabled: bool = True
    final_candidate_review_max_rounds: int = 2
    final_candidate_review_fail_open: bool = True
    web_search_failover_after_low_signal: bool = False
    fixed_orchestration_profile: str = field(
        default_factory=lambda: os.environ.get("NANOMA_FIXED_ORCHESTRATION_PROFILE", "")
    )
    fixed_orchestration_config_path: Path | None = field(
        default_factory=lambda: (
            Path(value).expanduser()
            if (value := os.environ.get("NANOMA_FIXED_ORCHESTRATION_CONFIG", "").strip())
            else None
        )
    )
    fixed_orchestration_poll_seconds: float = field(
        default_factory=lambda: max(
            0.25, _runtime_env_float("NANOMA_FIXED_ORCHESTRATION_POLL_SECONDS", 2.0)
        )
    )
    probe_resume_instruction: str | None = None
    intervention_file: Path | None = field(
        default_factory=lambda: (
            Path(value).expanduser()
            if (value := os.environ.get("NANOMA_INTERVENTION_FILE", "").strip())
            else None
        )
    )
    strategy_log_events: bool = False
    auto_kill_low_value_agents: bool = False
    auto_kill_min_llm: int = 10
    auto_kill_max_llm_no_candidate: int = 14
    auto_kill_max_tokens_no_candidate: int = 160000
    auto_kill_max_llm_duplicate: int = 22
    auto_kill_max_agents: int = 3
    auto_kill_only_tool: bool = False
    auto_kill_query_before_kill: bool = False
    strategy_spawn_portfolio: bool = False
    strategy_spawn_portfolio_target_agents: int = 5
    strategy_spawn_portfolio_max_root_turn: int = 3
    strategy_root_recovery_spawn: bool = False
    strategy_root_recovery_min_turns: int = 6
    strategy_root_recovery_target_agents: int = 4
    strategy_root_recovery_min_shared_candidates: int = 1
    strategy_root_recovery_max_attempts: int = 2
    strategy_root_recovery_require_no_active_children: bool = True
    strategy_candidate_evidence_gate: bool = False
    strategy_candidate_evidence_min_root_turns: int = 4
    strategy_candidate_evidence_min_shared_candidates: int = 1
    strategy_candidate_evidence_required_files: int = 1
    strategy_candidate_evidence_max_attempts: int = 2
    strategy_final_evidence_gate: bool = False
    strategy_final_evidence_min_root_turns: int = 4
    strategy_final_evidence_min_shared_candidates: int = 1
    strategy_final_evidence_min_candidate_agents: int = 1
    strategy_final_evidence_query_coverage_ratio: float = 1.0
    strategy_final_evidence_require_verification: bool = True
    strategy_final_evidence_require_verification_review: bool = True
    strategy_final_evidence_allow_spawn_verifier: bool = True
    strategy_final_evidence_max_verifier_spawns: int = 2
    strategy_finalize_candidates: bool = False
    strategy_finalize_min_shared_candidates: int = 2
    strategy_finalize_after_turns: int = 8
    strategy_improve_after_turns: int = 6
    strategy_readable_bytes_threshold: int = 900
    strategy_max_improve_attempts: int = 2
    strategy_stop_improve_when_stalled: bool = True
    strategy_finalize_min_stable_observations: int = 0
    strategy_finalize_once_per_best: bool = False
    strategy_max_finalize_attempts: int = 0
    supervisor_enabled: bool = False
    supervisor_model: str = "claude-opus-4-8"
    supervisor_protocol: str = "anthropic"
    supervisor_base_url: str | None = None
    supervisor_api_key: str | None = None
    supervisor_max_calls: int = 0
    supervisor_max_tokens: int = 1200
    supervisor_temperature: float = 0.0
    supervisor_http_timeout: float = 45.0
    supervisor_max_retries: int = 1
    supervisor_error_backoff_turns: int = 4
    supervisor_error_fallback_spawn: bool = True
    supervisor_root_only: bool = False
    supervisor_min_turn_interval: int = 1
    supervisor_recent_events: int = 12
    supervisor_history_messages: int = 4
    supervisor_fail_open: bool = True
    supervisor_one_step_topology: bool = True
    supervisor_min_child_turns: int = 2
    supervisor_trigger_mode: str = "decision_points"
    supervisor_initial_root_turns: int = 1
    supervisor_stall_turns: int = 8
    supervisor_stall_interval: int = 6
    supervisor_leaf_stall_tokens: int = 120000
    supervisor_decomposition_min_turns: int = 8
    supervisor_decomposition_min_tokens: int = 300000
    supervisor_noop_backoff_threshold: int = 1
    supervisor_quantitative_combo_required: bool = True
    supervisor_payload_task_chars: int = 900
    supervisor_payload_agent_task_chars: int = 240
    supervisor_payload_message_chars: int = 320
    supervisor_retry_override_on_wrong_tool: bool = True
    supervisor_override_wrong_tool_limit: int = 1


# ─── Agent ───────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: str
    task: str
    model: str

    # Identity
    bio: str = ""  # mutable self-description, visible to all via query()

    # State
    status: Literal["running", "idle", "done", "failed"] = "running"
    history: list[Message] = field(default_factory=list)
    children: set[str] = field(default_factory=set)
    parent: str | None = None
    depth: int = 0
    result: str | None = None

    # Resources
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    context_tokens: int = 0
    context_limit: int = 128_000  # overridden from model registry at creation
    tokens_consumed: int = 0
    _created_at: float = field(default_factory=time.time)

    # Workspace
    workspace: Path = field(default_factory=lambda: Path("."))
    artifacts: list[Artifact] = field(default_factory=list)

    # Message inboxes (three priority levels)
    _queue_inbox: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)
    _steer_inbox: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)
    _immediate_inbox: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)

    # Internal
    _task: asyncio.Task | None = field(default=None, repr=False)
    _turns: int = 0
    _last_active: float = field(default_factory=time.time)
    _rebirth_pending: dict | None = field(default=None, repr=False)
    _notified_thresholds: set = field(default_factory=set)  # resource thresholds already fired
    _tool_calls: int = 0
    _no_tool_turns: int = 0
    _output_limit_no_tool_turns: int = 0
    _malformed_tool_turns: int = 0
    _http400_recoveries: int = 0
    _last_tool_policy: dict[str, Any] | None = field(default=None, repr=False)
    _shell_activity: ShellActivityState = field(default_factory=ShellActivityState, repr=False)
    _delivery_activity: DeliveryActivityState = field(default_factory=DeliveryActivityState, repr=False)
    _delivery_final_notice_sent: bool = field(default=False, repr=False)


# ─── Runtime ─────────────────────────────────────────────────────────────────

class Runtime:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        llm_call: Callable | None = None,
        router: Callable | None = None,
        on_event: Callable | None = None,
        checkpoint_hook: Callable[[Path], Any] | None = None,
        restore_hook: Callable[[Path], Any] | None = None,
    ):
        self.config = config or RuntimeConfig()
        self._fixed_plan = load_fixed_orchestration_plan(
            self.config.fixed_orchestration_profile,
            self.config.fixed_orchestration_config_path,
        )
        if (
            self.config.fixed_orchestration_profile
            and not self.config.fixed_orchestration_config_path
            and self._fixed_plan is None
        ):
            logger.warning(
                "Unknown fixed orchestration profile %r; orchestration disabled",
                self.config.fixed_orchestration_profile,
            )
        if self._fixed_plan and self.config.supervisor_enabled:
            logger.warning(
                "Disabling supervisor because fixed orchestration profile %s is active",
                self._fixed_plan.name,
            )
            self.config.supervisor_enabled = False
        self.ledger = CostLedger(total_budget=self.config.budget)
        self.agents: dict[str, Agent] = {}
        self._id_gen = IdGenerator()
        self._tool_context = ToolContext(
            shared_dir=self.config.workspace_root / self.config.shared_dir,
            workspace_root=self.config.workspace_root,
            workspace_extra_roots=tuple(self.config.workspace_extra_roots),
            shell_max_output=self.config.shell_max_output,
            shell_max_timeout=self.config.shell_max_timeout,
            file_read_max_chars=self.config.file_read_max_chars,
            file_list_max_entries=self.config.file_list_max_entries,
            grep_max_results=self.config.grep_max_results,
            blocked_shell_patterns=list(self.config.blocked_shell_patterns),
        )
        self.llm_call = llm_call or default_llm_call
        self.router = router
        scheduler_limit = self.config.max_concurrent_llm
        if self._fixed_plan:
            scheduler_limit = min(scheduler_limit, self._fixed_plan.max_concurrent_llm)
        self.scheduler = Scheduler(max_concurrent=scheduler_limit)
        self.on_event = on_event or (lambda e: None)
        self._start_time = time.time()
        self._excluded_time_seconds = 0.0
        self._excluded_time_by_reason: dict[str, float] = {}
        self._events: list[dict] = []  # all events for post-hoc analysis
        self._messages_sent: list[tuple[str, str, int]] = []  # (from, to, tokens) for comm graph
        self._emit_lock = threading.Lock()  # protects events.jsonl writes
        self._probe_counter = 0
        self._checkpoint_hook = checkpoint_hook
        self._restore_hook = restore_hook
        self._last_checkpoint_path: Path | None = None
        self._last_checkpoint_agent: str | None = None
        self._global_checkpoint_seq = 0
        self._global_checkpoints: list[dict[str, Any]] = []
        self._rollback_requested: dict[str, Any] | None = None
        self._rollback_attempts = 0
        self._rollback_history: list[dict[str, Any]] = []
        self._infra_failure_streak_by_agent: dict[str, int] = {}
        self._infra_failure_state_by_agent: dict[str, dict[str, Any]] = {}
        self._llm_admission_lock = asyncio.Lock()
        self._last_llm_start_time = 0.0
        self._llm_overload_cooldown_until = 0.0
        self._intervention_offsets: dict[str, int] = {}
        self._pending_tool_overrides: dict[str, dict[str, Any]] = {}
        self._tool_override_events: dict[str, asyncio.Event] = {}
        self._override_wrong_tool_count_by_agent: dict[str, int] = {}
        self._strategy_spawn_authorized: set[str] = set()
        self._candidate_improve_attempts: dict[str, int] = {}
        self._candidate_improve_baseline_bytes: dict[str, int | None] = {}
        self._candidate_finalize_attempts: dict[str, int] = {}
        self._candidate_evidence_gate_attempts: dict[str, int] = {}
        self._root_recovery_spawn_attempts: dict[str, int] = {}
        self._candidate_finalize_last_bytes: dict[str, int | None] = {}
        self._best_shared_candidate_observed_bytes: int | None = None
        self._best_shared_candidate_stable_observations: int = 0
        self._candidate_deliveries: list[dict[str, Any]] = []
        self._candidate_delivery_seq = 0
        self._candidate_delivery_events: dict[str, asyncio.Event] = {}
        self._delivery_contract_history: list[dict[str, Any]] = []
        self._final_candidate_reviews: dict[str, dict[str, Any]] = {}
        self._candidate_convergence_turn_by_root: dict[str, int] = {}
        self._candidate_convergence_notice_by_root: dict[str, tuple[int, bool]] = {}
        self._candidate_root_parked: set[str] = set()
        self._auto_kill_requested: set[str] = set()
        self._supervisor_calls = 0
        self._supervisor_tokens = 0
        self._supervisor_cost = 0.0
        self._supervisor_decisions: dict[str, int] = {}
        self._last_supervisor_turn_by_agent: dict[str, int] = {}
        self._last_supervisor_marker_by_agent: dict[str, tuple[Any, ...]] = {}
        self._last_supervisor_reason_by_agent: dict[str, str] = {}
        self._last_supervisor_action_by_agent: dict[str, tuple[Any, ...]] = {}
        self._last_supervisor_action_turn_by_agent: dict[str, int] = {}
        self._supervisor_action_repeat_by_agent: dict[str, int] = {}
        self._supervisor_skips: dict[str, int] = {}
        self._supervisor_noop_streak_by_agent: dict[str, int] = {}
        self._supervisor_errors = 0
        self._supervisor_error_backoff_until_by_agent: dict[str, int] = {}
        self._fixed_agent_ids: dict[str, str] = {}
        self._fixed_agent_keys: dict[str, str] = {}
        self._fixed_checkpoint_agents: dict[str, set[str]] = {}
        self._fixed_agent_specs: dict[str, FixedAgentSpec] = {}
        self._fixed_fired_checkpoints: set[str] = set()
        self._fixed_requested_checkpoints: set[str] = set()
        self._fixed_pending_checkpoint_by_parent: dict[str, str] = {}
        self._fixed_observed_spawns: list[dict[str, Any]] = []
        self._fixed_monitor_task: asyncio.Task | None = None
        self._fixed_orchestration_lock = asyncio.Lock()
        self._fixed_build_lock = asyncio.Lock()
        self._fixed_completion_waiting: set[str] = set()
        self._fixed_validated_agents: set[str] = set()
        self._fixed_candidate_validations: dict[str, dict[str, Any]] = {}
        self._fixed_best_gate_metric: float | None = None
        self._fixed_green_seq = 0
        self._fixed_last_green_path: Path | None = None
        self._fixed_has_verified_green = bool(
            self._fixed_plan and not self._fixed_plan.direct_build_gate
        )
        self._fixed_green_restores = 0
        self._fixed_source_write_seq = 0
        self._fixed_last_green_write_seq = 0
        self._fixed_source_writes_in_flight = 0
        self._fixed_source_writes_frozen = False
        # ── merge-submit-path orchestration (opt-in via NANOMA_MERGE_SUBMIT_PATH) ──
        # When enabled, every spawned child works on a private copy of the shared
        # submit path (workspace_extra_roots[0]); on completion the runtime folds
        # its diff back under a lock and keeps it only if a re-run of the agents'
        # own verification says the merged result still works and is no worse.
        # Role-agnostic: applies to any spawned child whatever it was assigned.
        self._merge_lock: "asyncio.Lock | None" = None
        self._merge_baseline_path: Path | None = None
        self._merge_best_metric: float | None = None
        self._merge_disabled_reason: str | None = None
        self._merge_last_child_metric: dict[str, float | None] = {}
        self._verify_metric_seen: dict[str, list[float]] = {}
        self._verify_probe_cache: dict[str, dict] = {}
        self._verify_calibration: dict[str, dict] = {}
        self._merge_best_command: str | None = None
        self._merge_best_signature = None
        # Seconds each parent has already spent blocked on its children, so the
        # aggregate wait is a budget for the run rather than for one attempt.
        self._aggregate_wait_spent: dict[str, float] = {}
        # The state each agent's in-flight submission is sending, read when it is
        # sent rather than when the verdict returns.
        self._submitted_signature: dict[str, frozenset] = {}
        self._official_last: dict | None = None
        self._delivery_tasks: set[asyncio.Task] = set()
        self._delivery_judging_closed = False
        self._merge_current_metric: float | None = None
        self._merge_best_rank: tuple | None = None
        self._merge_current_rank: tuple | None = None
        self._merge_scored_signature: frozenset | None = None
        self._strategies = [
            SpawnPortfolioStrategy(
                SpawnPortfolioConfig(
                    enabled=self.config.strategy_spawn_portfolio,
                    target_agents=self.config.strategy_spawn_portfolio_target_agents,
                    max_root_turn=self.config.strategy_spawn_portfolio_max_root_turn,
                    require_no_children=False,
                )
            ),
            RootRecoverySpawnStrategy(
                RootRecoverySpawnConfig(
                    enabled=self.config.strategy_root_recovery_spawn,
                    min_root_turns=self.config.strategy_root_recovery_min_turns,
                    target_agents=self.config.strategy_root_recovery_target_agents,
                    min_shared_candidates=self.config.strategy_root_recovery_min_shared_candidates,
                    max_attempts=self.config.strategy_root_recovery_max_attempts,
                    require_no_active_children=self.config.strategy_root_recovery_require_no_active_children,
                )
            ),
            CandidateEvidenceGateStrategy(
                CandidateEvidenceGateConfig(
                    enabled=self.config.strategy_candidate_evidence_gate,
                    min_root_turns=self.config.strategy_candidate_evidence_min_root_turns,
                    min_shared_candidates=self.config.strategy_candidate_evidence_min_shared_candidates,
                    required_verification_files=self.config.strategy_candidate_evidence_required_files,
                    max_attempts=self.config.strategy_candidate_evidence_max_attempts,
                )
            ),
            FinalEvidenceGateStrategy(
                FinalEvidenceGateConfig(
                    enabled=self.config.strategy_final_evidence_gate,
                    min_root_turns=self.config.strategy_final_evidence_min_root_turns,
                    min_shared_candidates=self.config.strategy_final_evidence_min_shared_candidates,
                    min_candidate_agents=self.config.strategy_final_evidence_min_candidate_agents,
                    query_coverage_ratio=self.config.strategy_final_evidence_query_coverage_ratio,
                    require_verification=self.config.strategy_final_evidence_require_verification,
                    require_verification_review=self.config.strategy_final_evidence_require_verification_review,
                    allow_spawn_verifier=self.config.strategy_final_evidence_allow_spawn_verifier,
                    max_verifier_spawns=self.config.strategy_final_evidence_max_verifier_spawns,
                )
            ),
            LowValueAgentKillStrategy(
                LowValueAgentKillConfig(
                    enabled=self.config.auto_kill_low_value_agents,
                    min_llm=self.config.auto_kill_min_llm,
                    max_llm_no_candidate=self.config.auto_kill_max_llm_no_candidate,
                    max_tokens_no_candidate=self.config.auto_kill_max_tokens_no_candidate,
                    max_llm_duplicate=self.config.auto_kill_max_llm_duplicate,
                    max_agents=self.config.auto_kill_max_agents,
                    kill_only_tool=self.config.auto_kill_only_tool,
                    query_before_kill=self.config.auto_kill_query_before_kill,
                )
            ),
            CandidateFinalizeStrategy(
                CandidateFinalizeConfig(
                    enabled=self.config.strategy_finalize_candidates,
                    min_shared_candidates=self.config.strategy_finalize_min_shared_candidates,
                    finalize_after_turns=self.config.strategy_finalize_after_turns,
                    improve_after_turns=self.config.strategy_improve_after_turns,
                    readable_bytes_threshold=self.config.strategy_readable_bytes_threshold,
                    max_improve_attempts=self.config.strategy_max_improve_attempts,
                    stop_improve_when_stalled=self.config.strategy_stop_improve_when_stalled,
                    min_stable_observations=self.config.strategy_finalize_min_stable_observations,
                    finalize_once_per_best=self.config.strategy_finalize_once_per_best,
                    max_finalize_attempts=self.config.strategy_max_finalize_attempts,
                )
            ),
        ]

        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        self._tool_context.shared_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_dir:
            set_log_dir(self.config.log_dir)

    # ─── Agent lifecycle ─────────────────────────────────────────────────

    def _fixed_repo_path(self) -> Path | None:
        if not self._fixed_plan:
            return None
        configured = os.path.expandvars(self._fixed_plan.task_repo).strip()
        if configured:
            return Path(configured).expanduser()
        if self.config.workspace_extra_roots:
            return Path(self.config.workspace_extra_roots[0]).expanduser()
        return None

    @staticmethod
    def _fixed_snapshot_ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".lake",
            ".nanoma-runtime-logs",
            ".nanoma-task-work",
            "__pycache__",
        }
        return {name for name in names if name in ignored}

    def _fixed_snapshot_roots(self) -> tuple[Path, ...]:
        if not self._fixed_plan:
            return ()
        roots: list[Path] = []
        for raw_path in self._fixed_plan.snapshot_paths:
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts:
                continue
            if path.as_posix() in {"", "."}:
                return ()
            roots.append(path)
        return tuple(roots)

    @staticmethod
    def _fixed_remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()

    def _fixed_copy_snapshot_item(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir() and not source.is_symlink():
            shutil.copytree(
                source,
                destination,
                ignore=self._fixed_snapshot_ignore,
                symlinks=True,
            )
        elif source.exists() or source.is_symlink():
            shutil.copy2(source, destination, follow_symlinks=False)

    def _fixed_snapshot_source(self, destination: Path) -> bool:
        source = self._fixed_repo_path()
        if (
            not self._fixed_plan
            or not self._fixed_plan.snapshot_enabled
            or source is None
            or not source.is_dir()
        ):
            return False
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        roots = self._fixed_snapshot_roots()
        if roots:
            destination.mkdir(parents=True, exist_ok=True)
            copied = False
            for relative in roots:
                item = source / relative
                if not item.exists() and not item.is_symlink():
                    continue
                self._fixed_copy_snapshot_item(item, destination / relative)
                copied = True
            if not copied:
                return False
        else:
            shutil.copytree(
                source,
                destination,
                ignore=self._fixed_snapshot_ignore,
                symlinks=True,
            )
        return True

    def _fixed_initialize_green_checkpoint(self, root_id: str) -> None:
        if not self._fixed_plan:
            return
        if not self._fixed_plan.snapshot_enabled:
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "baseline_without_snapshot",
                "path": None,
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })
            return
        path = self.config.workspace_root / "fixed-green" / "baseline"
        if self._fixed_snapshot_source(path):
            self._fixed_last_green_path = path
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "baseline",
                "path": str(path),
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })

    def _fixed_record_green_checkpoint(self, root_id: str) -> bool:
        self._fixed_green_seq += 1
        path = self.config.workspace_root / "fixed-green" / f"green-{self._fixed_green_seq:03d}"
        if self._fixed_plan and not self._fixed_plan.snapshot_enabled:
            self._fixed_last_green_path = None
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._fixed_has_verified_green = True
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "verification_gate_without_snapshot",
                "seq": self._fixed_green_seq,
                "path": None,
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })
            return True
        if self._fixed_snapshot_source(path):
            self._fixed_last_green_path = path
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._fixed_has_verified_green = True
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "verification_gate",
                "seq": self._fixed_green_seq,
                "path": str(path),
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })
            return True
        self._emit(root_id, "fixed_green_checkpoint_skipped", {
            "reason": "snapshot_failed",
            "path": str(path),
            "source_write_seq": self._fixed_source_write_seq,
        })
        return False

    def _fixed_restore_green_checkpoint(self, root_id: str, reason: str) -> bool:
        source = self._fixed_last_green_path
        destination = self._fixed_repo_path()
        if (
            not self._fixed_plan
            or not self._fixed_plan.snapshot_enabled
            or source is None
            or destination is None
            or not source.is_dir()
        ):
            return False
        roots = self._fixed_snapshot_roots()
        if roots:
            for relative in roots:
                target = destination / relative
                self._fixed_remove_path(target)
                item = source / relative
                if item.exists() or item.is_symlink():
                    self._fixed_copy_snapshot_item(item, target)
        else:
            snapshot_files = {
                path.relative_to(source)
                for path in source.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            for path in destination.rglob("*"):
                if not (path.is_file() or path.is_symlink()):
                    continue
                relative = path.relative_to(destination)
                if any(part in self._fixed_snapshot_ignore("", [part]) for part in relative.parts):
                    continue
                if relative not in snapshot_files:
                    self._fixed_remove_path(path)
            for relative in snapshot_files:
                target = destination / relative
                self._fixed_remove_path(target)
                self._fixed_copy_snapshot_item(source / relative, target)
        self._fixed_source_write_seq = self._fixed_last_green_write_seq
        self._fixed_green_restores += 1
        self._emit(root_id, "fixed_green_restore", {
            "source": str(source),
            "reason": reason,
            "restore_count": self._fixed_green_restores,
            "source_write_seq": self._fixed_source_write_seq,
            "elapsed_seconds": round(self.effective_elapsed(), 1),
        })
        return True

    def _fixed_normalize_owned_path(self, raw_path: str) -> str | None:
        repo = self._fixed_repo_path()
        if repo is None or not raw_path.strip():
            return None
        path = Path(raw_path.strip())
        if path.is_absolute():
            try:
                return path.resolve().relative_to(repo.resolve()).as_posix()
            except ValueError:
                return None
        normalized = path.as_posix().lstrip("./")
        marker = "se-bmk-intern/combinatorial-games/"
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
        return normalized

    def _fixed_tool_write_paths(self, tc: ToolCall) -> list[str]:
        if tc.name == "ws_apply_patch":
            patch = str(tc.arguments.get("input", ""))
            return re.findall(
                r"^\*\*\* (?:Add|Update|Delete) File:\s*(.+?)\s*$",
                patch,
                flags=re.MULTILINE,
            )
        path = tc.arguments.get("path")
        return [str(path)] if path else []

    def _fixed_source_paths_for_tool(self, tc: ToolCall) -> list[str]:
        if not self._fixed_plan or tc.name not in _DELIVERY_WRITE_TOOLS:
            return []
        normalized = (
            self._fixed_normalize_owned_path(raw_path)
            for raw_path in self._fixed_tool_write_paths(tc)
        )
        return list(dict.fromkeys(path for path in normalized if path is not None))

    def _fixed_write_block_reason(self, agent: Agent, tc: ToolCall) -> str | None:
        spec = self._fixed_agent_specs.get(agent.id)
        if not self._fixed_plan:
            return None
        source_paths = self._fixed_source_paths_for_tool(tc)
        if source_paths and self._fixed_source_writes_frozen:
            return "source writes are frozen while the root validates or submits a green checkpoint"
        if (
            source_paths
            and agent.parent is None
            and not self._fixed_plan.root_source_writes
        ):
            return "the fixed topology reserves source writes for assigned worker nodes"
        if spec is None:
            return None
        if spec.read_only:
            return f"fixed role {spec.key} is read-only"
        paths = self._fixed_tool_write_paths(tc)
        if not paths:
            return "fixed orchestration requires an attributable file path for every write"
        allowed = set(spec.allowed_paths)
        for raw_path in paths:
            normalized = self._fixed_normalize_owned_path(raw_path)
            if normalized is None:
                # Private NanoMA workspace handoffs are allowed.
                continue
            if "*" not in allowed and normalized not in allowed:
                return (
                    f"fixed role {spec.key} does not own {normalized}; "
                    f"allowed_paths={sorted(allowed)}"
                )
        return None

    @staticmethod
    def _fixed_shell_redirect_scan(command: str) -> str:
        masked = list(command)

        # Heredoc bodies are interpreter input, not shell syntax. Mask them so
        # comparisons inside a Python/Ruby/Perl program are not mistaken for
        # output redirects.
        heredoc_pattern = re.compile(
            r"<<-?\s*(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)"
            r"(?P=quote)"
        )
        search_from = 0
        while match := heredoc_pattern.search(command, search_from):
            body_start = command.find("\n", match.end())
            if body_start < 0:
                break
            terminator = re.search(
                rf"(?m)^[\t ]*{re.escape(match.group('tag'))}[\t ]*(?:\n|$)",
                command[body_start + 1:],
            )
            if terminator is None:
                search_from = match.end()
                continue
            body_end = body_start + 1 + terminator.start()
            for index in range(body_start + 1, body_end):
                masked[index] = " "
            search_from = body_start + 1 + terminator.end()

        # Shell operators inside quoted arguments belong to that argument. In
        # particular, Python comparisons in `python -c "..."` are not redirects.
        quote: str | None = None
        escaped = False
        for index, char in enumerate(command):
            if masked[index] == " " and char != " ":
                continue
            if escaped:
                masked[index] = " "
                escaped = False
                continue
            if quote is not None:
                masked[index] = " "
                if char == "\\" and quote == '"':
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                masked[index] = " "
            elif char == "\\":
                escaped = True
                masked[index] = " "

        return "".join(masked)

    def _fixed_shell_block_reason(self, agent: Agent, command: str) -> str | None:
        spec = self._fixed_agent_specs.get(agent.id)
        if not self._fixed_plan:
            return None
        # Only checkpoint-bound roles inherit fixed-topology shell contracts.
        # NanoMA-created adaptive children retain the runtime's normal behavior.
        if spec is None and agent.parent is not None:
            return None
        lowered = command.lower()
        destructive_git = re.search(
            r"\bgit\s+(?:reset|restore|checkout|clean|revert|switch|merge|cherry-pick)\b",
            lowered,
        )
        if destructive_git and agent.parent is not None:
            return "destructive or integrating git commands are reserved for the root integrator"
        # Discard harmless fd merges and /dev/null sinks before looking for
        # shell redirects that could mutate benchmark source files.
        redirect_scan = self._fixed_shell_redirect_scan(command)
        write_scan = re.sub(
            r"\d*>{1,2}\s*(?:&\d+|/dev/null)(?=\s|[;|&]|$)",
            "",
            redirect_scan.lower(),
        )
        write_markers = (
            r"(?:^|\s)(?:sed\s+-i|perl\s+-pi|tee\s|cp\s|mv\s|rm\s)",
            # Block redirects to files, but permit fd merges used to inspect
            # build output, such as `lake build 2>&1 | tail -80`.
            r"(?:^|[\s;|&])\d*>{1,2}(?!&)",
        )
        python_write_markers = (
            r"\b(?:write_text|write_bytes|writelines)\s*\(",
            r"\bopen\s*\([^)]*,\s*['\"][wax+]",
            r"\.(?:save|to_csv|to_excel|to_json|to_parquet|to_pickle|dump)\s*\(",
            r"\b(?:shutil\.)?(?:copy|copy2|copyfile|move|rmtree)\s*\(",
            r"\bos\.(?:remove|unlink|rename|replace|makedirs|mkdir)\s*\(",
            r"\bpath\s*\([^)]*\)\.(?:unlink|mkdir|rename|replace|touch)\s*\(",
            r"\bapply_patch\b",
        )
        python_source_write = bool(
            re.search(r"\bpython(?:3(?:\.\d+)?)?\b", lowered)
            and any(
                re.search(pattern, lowered, flags=re.DOTALL)
                for pattern in python_write_markers
            )
        )
        if (
            any(re.search(pattern, write_scan) for pattern in write_markers)
            or python_source_write
        ):
            return (
                "shell-based source writes are disabled for fixed orchestration roles; "
                "use ws_replace_string/ws_multi_replace/ws_apply_patch on owned files"
            )
        return None

    def _fixed_finish_block_reason(self, agent: Agent, tool_name: str) -> str | None:
        if not self._fixed_plan:
            return None
        if tool_name == "submit" and agent.parent is not None:
            return "only the fixed orchestration root may submit"
        gate = self._fixed_verification_gate()
        if tool_name == "submit" and gate and gate.argv == ("sforge-submit",):
            return "official submission is owned by the Runtime verification gate"
        if agent.parent is not None:
            return None
        if self._fixed_plan.direct_build_gate and not self._fixed_has_verified_green:
            return "no Runtime-verified green checkpoint is available"
        if self._fixed_source_writes_in_flight:
            return (
                f"{self._fixed_source_writes_in_flight} source write(s) are still in flight; "
                "wait for writers, then run the Runtime verification gate"
            )
        if self._fixed_source_write_seq != self._fixed_last_green_write_seq:
            return (
                "the source tree changed after the last green checkpoint; wait for writers and "
                "obtain a successful root lake build or Runtime verification gate before "
                "finishing or submitting"
            )
        return None

    @staticmethod
    def _fixed_is_full_lake_build(command: str) -> bool:
        # Only a bare build (optionally preceded by one `cd ... &&`) preserves
        # the process exit code. Pipelines and trailing shell commands must
        # never be accepted as green evidence.
        cleaned = re.sub(r"\d*>\s*&\d+", "", command).strip()
        if (
            not cleaned
            or re.search(r"[|;\n\r]", cleaned)
            or re.search(r"(?<!&)&(?!&)", cleaned)
        ):
            return False
        parts = re.split(r"\s*&&\s*", cleaned)
        if len(parts) == 2:
            try:
                cd_tokens = shlex.split(parts[0])
            except ValueError:
                return False
            if len(cd_tokens) not in {2, 3} or cd_tokens[0] != "cd":
                return False
            if len(cd_tokens) == 3 and cd_tokens[1] != "--":
                return False
            build = parts[1]
        elif len(parts) == 1:
            build = parts[0]
        else:
            return False
        try:
            return shlex.split(build) == ["lake", "build"]
        except ValueError:
            return False

    def _fixed_observe_shell_result(
        self,
        agent: Agent,
        command: str,
        result: Any,
        write_seq_before: int | None = None,
    ) -> None:
        if not self._fixed_plan or agent.parent is not None:
            return
        if not self._fixed_is_full_lake_build(command):
            return
        if self._fixed_plan.direct_build_gate:
            self._emit(agent.id, "fixed_green_checkpoint_skipped", {
                "reason": "agent_shell_build_is_not_authoritative",
                "command": command[:500],
            })
            return
        exit_code = result.get("exit_code") if isinstance(result, dict) else None
        if exit_code == 0:
            concurrent_write = (
                write_seq_before is not None
                and write_seq_before != self._fixed_source_write_seq
            )
            if concurrent_write or self._fixed_source_writes_in_flight:
                self._emit(agent.id, "fixed_green_checkpoint_skipped", {
                    "reason": "source_changed_during_build",
                    "write_seq_before": write_seq_before,
                    "write_seq_after": self._fixed_source_write_seq,
                    "writes_in_flight": self._fixed_source_writes_in_flight,
                })
            else:
                self._fixed_record_green_checkpoint(agent.id)
        elif self._fixed_restore_green_checkpoint(agent.id, "root_lake_build_failed"):
            notice = (
                "[Runtime green restore] The root lake build failed. Source files were restored "
                "to the last successful green checkpoint. Re-read the tree before continuing."
            )
            agent.history.append({"role": "user", "content": notice})

    def _fixed_verification_gate(self) -> FixedVerificationGate | None:
        if not self._fixed_plan or not self._fixed_plan.direct_build_gate:
            return None
        return self._fixed_plan.verification_gate or FixedVerificationGate(
            argv=("lake", "build"),
            timeout_seconds=max(
                30.0,
                float(os.environ.get("NANOMA_FIXED_BUILD_TIMEOUT", "900") or 900),
            ),
        )

    @staticmethod
    def _fixed_parse_gate_output(
        gate: FixedVerificationGate,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        output = f"{stdout}\n{stderr}"
        valid = exit_code == 0
        if re.search(r"^\s*Valid:\s*no\s*$", output, flags=re.IGNORECASE | re.MULTILINE):
            valid = False
        if re.search(r"^\s*[^:\n]+:\s*ERROR\s*$", output, flags=re.MULTILINE):
            valid = False

        metric = None
        metric_name = None
        score_match = re.search(
            r"^\s*Score:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
            output,
            flags=re.MULTILINE,
        )
        if score_match:
            metric = float(score_match.group(1))
            metric_name = "score"
        else:
            pass_rate_match = re.search(
                r"^\s*Pass rate:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))%\s*$",
                output,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if pass_rate_match:
                metric = float(pass_rate_match.group(1)) / 100.0
                metric_name = "pass_rate"

        if gate.acceptance == "exit-code":
            valid = exit_code == 0
        return {
            "judge_valid": valid,
            "metric": metric,
            "metric_name": metric_name,
        }

    def _fixed_gate_metric_accepted(
        self,
        gate: FixedVerificationGate,
        metric: float | None,
    ) -> bool:
        if gate.acceptance != "judge-valid-nondecreasing":
            return True
        if metric is None or self._fixed_best_gate_metric is None:
            return True
        tolerance = max(1e-12, abs(self._fixed_best_gate_metric) * 1e-9)
        if gate.score_direction == "minimize":
            return metric <= self._fixed_best_gate_metric + tolerance
        return metric + tolerance >= self._fixed_best_gate_metric

    async def _fixed_run_direct_lake_build(
        self,
        root_id: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Run the configured Runtime verification gate (legacy name retained)."""
        repo = self._fixed_repo_path()
        gate = self._fixed_verification_gate()
        if repo is None or not repo.is_dir() or gate is None:
            return {
                "accepted": False,
                "exit_code": -1,
                "reason": "verification_gate_unavailable",
            }
        gate_cwd = Path(os.path.expandvars(gate.cwd)).expanduser() if gate.cwd else repo
        if not gate_cwd.is_absolute():
            gate_cwd = repo / gate_cwd
        if not gate_cwd.is_dir():
            return {
                "accepted": False,
                "exit_code": -1,
                "reason": "verification_gate_cwd_unavailable",
                "cwd": str(gate_cwd),
            }
        async with self._fixed_build_lock:
            if self._fixed_source_writes_in_flight:
                return {
                    "accepted": False,
                    "exit_code": -1,
                    "reason": "source_write_in_flight",
                }
            write_seq_before = self._fixed_source_write_seq
            previously_frozen = self._fixed_source_writes_frozen
            self._fixed_source_writes_frozen = True
            started = time.time()
            proc: asyncio.subprocess.Process | None = None
            stdout = b""
            stderr = b""
            timed_out = False
            try:
                proc = await asyncio.create_subprocess_exec(
                    *gate.argv,
                    cwd=str(gate_cwd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=gate.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    proc.kill()
                    stdout, stderr = await proc.communicate()
            except Exception as exc:
                stderr = f"{type(exc).__name__}: {exc}".encode()
            finally:
                self._fixed_source_writes_frozen = previously_frozen

            exit_code = -1 if proc is None or timed_out else int(proc.returncode or 0)
            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")
            parsed = self._fixed_parse_gate_output(
                gate,
                exit_code,
                stdout_text,
                stderr_text,
            )
            source_changed = (
                write_seq_before != self._fixed_source_write_seq
                or bool(self._fixed_source_writes_in_flight)
            )
            valid = bool(parsed["judge_valid"])
            score_accepted = self._fixed_gate_metric_accepted(gate, parsed["metric"])
            gate_passed = valid and score_accepted and not source_changed
            accepted = gate_passed and self._fixed_record_green_checkpoint(root_id)
            if accepted:
                outcome = "accepted"
                if parsed["metric"] is not None:
                    self._fixed_best_gate_metric = float(parsed["metric"])
            else:
                restore_reason = (
                    "runtime_verification_source_changed"
                    if source_changed
                    else (
                        "runtime_verification_score_regression"
                        if valid and not score_accepted
                        else (
                            "runtime_verification_snapshot_failed"
                            if gate_passed
                            else "runtime_verification_failed"
                        )
                    )
                )
                self._fixed_restore_green_checkpoint(root_id, restore_reason)
                outcome = "restored"
            result = {
                "accepted": accepted,
                "exit_code": exit_code,
                "reason": reason,
                "outcome": outcome,
                "argv": list(gate.argv),
                "cwd": str(gate_cwd),
                "acceptance": gate.acceptance,
                "score_direction": gate.score_direction,
                "judge_valid": valid,
                "metric": parsed["metric"],
                "metric_name": parsed["metric_name"],
                "best_metric": self._fixed_best_gate_metric,
                "timed_out": timed_out,
                "source_changed": source_changed,
                "write_seq_before": write_seq_before,
                "write_seq_after": self._fixed_source_write_seq,
                "elapsed_seconds": round(time.time() - started, 3),
                "stdout_tail": stdout_text[-12000:],
                "stderr_tail": stderr_text[-12000:],
            }
            self._emit(root_id, "fixed_runtime_build_gate", result)
            self._emit(root_id, "fixed_runtime_verification_gate", result)
            return result

    def _fixed_elapsed_fraction(self) -> float:
        if self.config.time_limit <= 0:
            return 0.0
        return min(1.0, self.effective_elapsed() / self.config.time_limit)

    def _fixed_checkpoint(self, key: str) -> FixedCheckpoint | None:
        if not self._fixed_plan:
            return None
        return next((item for item in self._fixed_plan.checkpoints if item.key == key), None)

    def _fixed_missing_specs(self, checkpoint: FixedCheckpoint) -> list[FixedAgentSpec]:
        return [
            spec
            for spec in checkpoint.agents
            if spec.key not in self._fixed_agent_ids
        ]

    def _fixed_checkpoint_complete(self, key: str) -> bool:
        checkpoint = self._fixed_checkpoint(key)
        if checkpoint is None or key not in self._fixed_fired_checkpoints:
            return False
        spawned = self._fixed_checkpoint_agents.get(key, set())
        return bool(spawned) and all(
            self.agents[agent_id].status == "done"
            for agent_id in spawned
            if agent_id in self.agents
        )

    def _fixed_expire_agents(self) -> None:
        elapsed = self.effective_elapsed()
        current_task = asyncio.current_task()
        for agent_id, spec in list(self._fixed_agent_specs.items()):
            agent = self.agents.get(agent_id)
            if (
                agent is None
                or spec.end_elapsed_seconds <= 0
                or elapsed < spec.end_elapsed_seconds
                or agent.status not in {"running", "idle"}
            ):
                continue
            agent.status = "done"
            agent.result = agent.result or (
                f"[Runtime role deadline at {spec.end_elapsed_seconds:.0f}s; "
                "current candidate will be validated]"
            )
            if (
                agent._task is not None
                and agent._task is not current_task
                and not agent._task.done()
            ):
                agent._task.cancel()
            self._emit(agent_id, "fixed_agent_deadline", {
                "role": spec.key,
                "deadline_seconds": spec.end_elapsed_seconds,
                "elapsed_seconds": round(elapsed, 1),
            })

    async def _fixed_validate_completed_agents(self) -> None:
        if not self._fixed_plan or not self._fixed_plan.direct_build_gate:
            return
        gate = self._fixed_verification_gate()
        if gate is None:
            return
        root_id = self._fixed_agent_ids.get("root")
        if not root_id or root_id not in self.agents:
            return
        for agent_id, spec in list(self._fixed_agent_specs.items()):
            if agent_id in self._fixed_validated_agents:
                continue
            agent = self.agents.get(agent_id)
            if agent is None or agent.status not in {"done", "failed"}:
                continue
            if agent._task is not None and not agent._task.done():
                continue
            if spec.read_only:
                self._fixed_validated_agents.add(agent_id)
                self._fixed_candidate_validations[agent_id] = {
                    "role": spec.key,
                    "accepted": agent.status == "done",
                    "outcome": "read_only_complete" if agent.status == "done" else "read_only_failed",
                }
                continue

            other_writer_active = any(
                other_id != agent_id
                and not other_spec.read_only
                and (other := self.agents.get(other_id)) is not None
                and other.status in {"running", "idle"}
                for other_id, other_spec in self._fixed_agent_specs.items()
            )
            if other_writer_active or self._fixed_source_writes_in_flight:
                continue

            if agent.status == "failed":
                if self._fixed_source_write_seq != self._fixed_last_green_write_seq:
                    self._fixed_restore_green_checkpoint(
                        root_id,
                        f"fixed_writer_failed:{spec.key}",
                    )
                validation = {
                    "role": spec.key,
                    "accepted": False,
                    "outcome": "agent_failed_restored",
                }
            elif (
                self._fixed_source_write_seq == self._fixed_last_green_write_seq
                and not gate.always_run
            ):
                validation = {
                    "role": spec.key,
                    "accepted": self._fixed_has_verified_green,
                    "outcome": "no_source_change",
                }
            else:
                build = await self._fixed_run_direct_lake_build(
                    root_id,
                    reason=f"writer_complete:{spec.key}",
                )
                if build.get("reason") == "source_write_in_flight":
                    continue
                validation = {"role": spec.key, **build}

            self._fixed_validated_agents.add(agent_id)
            self._fixed_candidate_validations[agent_id] = validation
            accepted = bool(validation.get("accepted"))
            root = self.agents[root_id]
            root.history.append({
                "role": "user",
                "content": (
                    f"[Runtime candidate gate] {spec.key}: "
                    f"{'accepted as the new green checkpoint' if accepted else 'rejected; previous green restored'}. "
                    f"outcome={validation.get('outcome')}; "
                    f"metric={validation.get('metric')}; best={validation.get('best_metric')}."
                ),
            })
            self._emit(root_id, "fixed_candidate_validation", {
                "agent": agent_id,
                **validation,
            })

    def fixed_orchestration_completion_block_reason(self, agent: Agent) -> str | None:
        plan = self._fixed_plan
        if not plan or agent.parent is not None or not plan.completion_checkpoint:
            return None
        if self._fixed_checkpoint_complete(plan.completion_checkpoint):
            return None
        if self._fixed_elapsed_fraction() >= 0.95 or self.effective_elapsed() >= 6900:
            return None
        return (
            f"fixed orchestration is still active; wait for checkpoint "
            f"{plan.completion_checkpoint!r} and query its verifier before finishing"
        )

    def _fixed_dependency_terminal(self, agent_id: str) -> bool:
        agent = self.agents.get(agent_id)
        if agent is None or agent.status not in {"done", "failed"}:
            return False
        spec = self._fixed_agent_specs.get(agent_id)
        if (
            self._fixed_plan
            and self._fixed_plan.direct_build_gate
            and spec is not None
            and not spec.read_only
        ):
            return agent_id in self._fixed_validated_agents
        return True

    def _fixed_checkpoint_ready(self, checkpoint: FixedCheckpoint) -> bool:
        if any(dep not in self._fixed_fired_checkpoints for dep in checkpoint.depends_on):
            return False
        parent_id = self._fixed_agent_ids.get(checkpoint.parent_key)
        if not parent_id or parent_id not in self.agents:
            return False
        parent = self.agents[parent_id]
        age = max(0.0, time.time() - parent._created_at)
        elapsed = self.effective_elapsed()
        elapsed_fraction = self._fixed_elapsed_fraction()
        if (
            checkpoint.not_before_elapsed_seconds > 0
            and elapsed < checkpoint.not_before_elapsed_seconds
        ):
            return False
        if (
            checkpoint.require_parent_without_candidate
            and self._has_delivery_candidate(parent)
        ):
            return False
        if (
            checkpoint.min_parent_web_no_gain_calls > 0
            and parent._shell_activity.web_no_gain_calls
            < checkpoint.min_parent_web_no_gain_calls
        ):
            return False
        if (
            checkpoint.min_parent_unavailable_tool_calls > 0
            and parent._shell_activity.unavailable_tool_calls
            < checkpoint.min_parent_unavailable_tool_calls
        ):
            return False

        if checkpoint.mode == "settled":
            dependency_agents = set().union(*(
                self._fixed_checkpoint_agents.get(dep, set())
                for dep in checkpoint.depends_on
            )) if checkpoint.depends_on else set()
            if dependency_agents:
                return all(
                    self._fixed_dependency_terminal(agent_id)
                    for agent_id in dependency_agents
                    if agent_id in self.agents
                )
            return bool(
                (
                    checkpoint.fallback_elapsed_fraction > 0
                    and elapsed_fraction >= checkpoint.fallback_elapsed_fraction
                )
                or (
                    checkpoint.fallback_elapsed_seconds > 0
                    and elapsed >= checkpoint.fallback_elapsed_seconds
                )
            )

        if checkpoint.mode == "stalled":
            if parent.status in {"done", "failed"}:
                return False
            pending_override = self._pending_tool_overrides.get(parent_id) or {}
            if "deliver_to_parent" in set(pending_override.get("tools") or []):
                return False
            last_write_turn = parent._delivery_activity.last_write_turn
            stalled_turns = parent._turns - last_write_turn if last_write_turn > 0 else parent._turns
            stalled = stalled_turns >= max(1, checkpoint.min_stall_turns)
            progressed = parent._turns >= checkpoint.min_parent_turns
            age_fallback = (
                checkpoint.min_parent_age_seconds > 0
                and age >= checkpoint.min_parent_age_seconds
            )
            return stalled and (progressed or age_fallback)

        if parent.status in {"done", "failed"}:
            return True

        progress_signals = []
        if checkpoint.min_parent_turns > 0:
            progress_signals.append(parent._turns >= checkpoint.min_parent_turns)
        if checkpoint.min_parent_tool_calls > 0:
            progress_signals.append(parent._tool_calls >= checkpoint.min_parent_tool_calls)
        if checkpoint.min_parent_age_seconds > 0:
            progress_signals.append(age >= checkpoint.min_parent_age_seconds)
        fallback_signals = []
        if checkpoint.fallback_elapsed_seconds > 0:
            fallback_signals.append(elapsed >= checkpoint.fallback_elapsed_seconds)
        if checkpoint.fallback_elapsed_fraction > 0:
            fallback_signals.append(elapsed_fraction >= checkpoint.fallback_elapsed_fraction)
        if progress_signals:
            progress_ready = (
                all(progress_signals)
                if checkpoint.progress_logic == "all"
                else any(progress_signals)
            )
        else:
            progress_ready = not fallback_signals
        return progress_ready or any(fallback_signals)

    async def _notify_fixed_agent(self, to_id: str, message: str, mode: str = "steer") -> None:
        await self.deliver(Envelope(
            from_id="runtime-checkpoint",
            to_id=to_id,
            content=message,
            tokens=estimate_tokens(message),
            timestamp=time.time(),
            mode=mode,
        ))
        self._emit(to_id, "fixed_orchestration_message", {
            "from": "runtime-checkpoint",
            "message": message[:500],
        })

    def _fixed_dependency_handoff(self, checkpoint: FixedCheckpoint) -> str:
        handoffs: list[str] = []
        for dependency in checkpoint.depends_on:
            for agent_id in sorted(self._fixed_checkpoint_agents.get(dependency, set())):
                agent = self.agents.get(agent_id)
                if agent is None or not agent.result:
                    continue
                role = self._fixed_agent_keys.get(agent_id, agent.bio or agent_id)
                result = str(agent.result).strip()
                if not result:
                    continue
                handoffs.append(
                    f"[{dependency}/{role} from {agent_id}]\n{result[:2400]}"
                )
        if not handoffs:
            return ""
        return "\n\n".join(handoffs)[:6000]

    def _fixed_release_spawn_request(self, parent_id: str, checkpoint_key: str) -> None:
        if self._fixed_pending_checkpoint_by_parent.get(parent_id) == checkpoint_key:
            self._fixed_pending_checkpoint_by_parent.pop(parent_id, None)
        self._fixed_requested_checkpoints.discard(checkpoint_key)
        override = self._pending_tool_overrides.get(parent_id)
        if (
            override
            and (override.get("metadata") or {}).get("fixed_checkpoint") == checkpoint_key
        ):
            self._pending_tool_overrides.pop(parent_id, None)

    def _fixed_spawn_context(
        self,
        agent: Agent,
    ) -> tuple[FixedCheckpoint, list[FixedAgentSpec], dict[str, Any]] | None:
        checkpoint_key = self._fixed_pending_checkpoint_by_parent.get(agent.id)
        if not checkpoint_key:
            return None
        checkpoint = self._fixed_checkpoint(checkpoint_key)
        override = self._pending_tool_overrides.get(agent.id)
        metadata = dict((override or {}).get("metadata") or {})
        if checkpoint is None or metadata.get("fixed_checkpoint") != checkpoint_key:
            return None
        selected_keys = [str(item) for item in metadata.get("fixed_agent_keys") or []]
        specs_by_key = {spec.key: spec for spec in checkpoint.agents}
        selected = [specs_by_key[key] for key in selected_keys if key in specs_by_key]
        if not selected:
            return None
        return checkpoint, selected, metadata

    @staticmethod
    def _fixed_route_parent_task(spec: FixedAgentSpec, requested: Any) -> str | None:
        if isinstance(requested, dict):
            requested = (
                requested.get("task")
                or requested.get("description")
                or requested.get("unit")
            )
        parent_task = str(requested or "").strip()
        if not parent_task:
            return None
        runtime_contract = spec.task.strip()
        if parent_task == runtime_contract:
            return runtime_contract
        return (
            f"{runtime_contract}\n\n"
            "[Parent-authored execution plan]\n"
            "The runtime role and access contract above takes precedence over this plan.\n"
            f"{parent_task}"
        )

    def _fixed_prepare_spawn_tool_call(self, tc: ToolCall, agent: Agent) -> str | None:
        context = self._fixed_spawn_context(agent)
        if context is None:
            # A fixed topology adds checkpoint requests; it does not replace
            # NanoMA's ordinary, adaptive spawn behavior.
            return None
        checkpoint, specs, metadata = context
        required_tool = str(
            metadata.get("spawn_tool") or ("spawn_many" if len(specs) > 1 else "spawn")
        )
        if tc.name != required_tool:
            return (
                f"checkpoint {checkpoint.key} requires {required_tool}, not {tc.name}; "
                "use the single spawn tool exposed for this turn"
            )

        original_arguments = copy.deepcopy(tc.arguments)
        if tc.name == "spawn":
            if len(specs) != 1:
                return f"checkpoint {checkpoint.key} requires {len(specs)} children via spawn_many"
            routed_task = self._fixed_route_parent_task(specs[0], original_arguments)
            if routed_task is None:
                return (
                    f"checkpoint {checkpoint.key} requires the parent agent to author a non-empty "
                    "child task in NanoMA's spawn call"
                )
            tc.arguments = {"task": routed_task}
            requested_model = original_arguments.get("model")
            if requested_model:
                tc.arguments["model"] = requested_model
        else:
            requested = original_arguments.get("agents") or original_arguments.get("tasks") or []
            if not isinstance(requested, list) or len(requested) != len(specs):
                return (
                    f"checkpoint {checkpoint.key} requires exactly {len(specs)} child assignments "
                    "in one spawn_many call"
                )
            routed_agents: list[dict[str, Any]] = []
            for index, (spec, item) in enumerate(zip(specs, requested)):
                routed_task = self._fixed_route_parent_task(spec, item)
                if routed_task is None:
                    return (
                        f"checkpoint {checkpoint.key} requires a non-empty parent-authored task "
                        f"for child assignment {index + 1}"
                    )
                routed: dict[str, Any] = {"task": routed_task}
                if isinstance(item, dict) and item.get("model"):
                    routed["model"] = item["model"]
                routed_agents.append(routed)
            tc.arguments = {"agents": routed_agents}

        self._emit(agent.id, "fixed_spawn_tool_routed", {
            "checkpoint": checkpoint.key,
            "tool": tc.name,
            "agent_keys": [spec.key for spec in specs],
            "original_args_preview": str(original_arguments)[:500],
            "execution": "nanoma_tool",
            "parent_tasks_preserved": True,
            "runtime_contracts_applied": True,
        })
        return None

    @staticmethod
    def _spawn_tool_created_ids(tc: ToolCall, result: Any) -> list[str]:
        if not isinstance(result, dict) or "error" in result:
            return []
        if tc.name == "spawn":
            child_id = str(result.get("agent_id") or "").strip()
            return [child_id] if child_id else []
        created_ids: list[str] = []
        for item in result.get("created") or []:
            if not isinstance(item, dict):
                continue
            child_id = str(item.get("agent_id") or "").strip()
            if child_id:
                created_ids.append(child_id)
        return created_ids

    def _fixed_record_observed_spawn(
        self,
        tc: ToolCall,
        parent: Agent,
        child: Agent,
        *,
        spawn_origin: str,
        checkpoint: str | None = None,
        role_key: str | None = None,
    ) -> None:
        if any(item.get("child_id") == child.id for item in self._fixed_observed_spawns):
            return
        record = {
            "parent_id": parent.id,
            "parent_role": self._fixed_agent_keys.get(parent.id),
            "child_id": child.id,
            "child_role": role_key,
            "spawn_origin": spawn_origin,
            "checkpoint": checkpoint,
            "tool": tc.name,
            "tool_call_id": tc.id,
            "elapsed_seconds": round(self.effective_elapsed(), 3),
            "depth": child.depth,
            "model": child.model,
            "bio": child.bio,
            "task_preview": child.task[:500],
        }
        self._fixed_observed_spawns.append(record)
        if spawn_origin == "autonomous_spawn":
            self._emit(parent.id, "autonomous_spawn_observed", {
                "child": child.id,
                "parent_role": record["parent_role"],
                "depth": child.depth,
                "model": child.model,
                "spawn_tool": tc.name,
                "tool_call_id": tc.id,
                "task_preview": child.task[:500],
                "topology_update": True,
            })

    def _fixed_observed_topology(self) -> dict[str, Any]:
        spawn_by_child = {
            str(record.get("child_id")): record
            for record in self._fixed_observed_spawns
            if record.get("child_id")
        }
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for agent in sorted(self.agents.values(), key=lambda item: item._created_at):
            spawn = spawn_by_child.get(agent.id)
            if agent.parent is None:
                spawn_origin = "root"
            elif spawn:
                spawn_origin = str(spawn.get("spawn_origin") or "autonomous_spawn")
            elif agent.id in self._fixed_agent_specs:
                spawn_origin = "checkpoint_spawn"
            else:
                spawn_origin = "autonomous_spawn"
            fixed_role = self._fixed_agent_keys.get(agent.id)
            nodes.append({
                "id": agent.id,
                "parent_id": agent.parent,
                "depth": agent.depth,
                "status": agent.status,
                "model": agent.model,
                "bio": agent.bio,
                "fixed_role": fixed_role,
                "spawn_origin": spawn_origin,
                "checkpoint": spawn.get("checkpoint") if spawn else None,
                "start_elapsed_seconds": round(
                    max(0.0, agent._created_at - self._start_time), 3
                ),
                "task_preview": agent.task[:500],
            })
            if agent.parent is not None:
                edges.append({
                    "from": agent.parent,
                    "to": agent.id,
                    "spawn_origin": spawn_origin,
                    "checkpoint": spawn.get("checkpoint") if spawn else None,
                    "tool": spawn.get("tool") if spawn else None,
                    "elapsed_seconds": spawn.get("elapsed_seconds") if spawn else None,
                })
        autonomous = [
            copy.deepcopy(record)
            for record in self._fixed_observed_spawns
            if record.get("spawn_origin") == "autonomous_spawn"
        ]
        checkpoint = [
            copy.deepcopy(record)
            for record in self._fixed_observed_spawns
            if record.get("spawn_origin") == "checkpoint_spawn"
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "autonomous_spawns": autonomous,
            "checkpoint_spawns": checkpoint,
            "counts": {
                "nodes": len(nodes),
                "edges": len(edges),
                "autonomous_spawns": len(autonomous),
                "checkpoint_spawns": len(checkpoint),
            },
        }

    async def _fixed_record_spawn_tool_result(
        self,
        tc: ToolCall,
        parent: Agent,
        result: Any,
    ) -> Any:
        context = self._fixed_spawn_context(parent)
        created_ids = self._spawn_tool_created_ids(tc, result)
        if context is None:
            for child_id in created_ids:
                child = self.agents.get(child_id)
                if child is not None and child.parent == parent.id:
                    self._fixed_record_observed_spawn(
                        tc,
                        parent,
                        child,
                        spawn_origin="autonomous_spawn",
                    )
            return result
        checkpoint, specs, _metadata = context

        dependency_handoff = self._fixed_dependency_handoff(checkpoint)
        registered: list[str] = []
        for spec, child_id in zip(specs, created_ids):
            child = self.agents.get(child_id)
            if child is None or child.parent != parent.id:
                continue
            child.bio = spec.bio
            self._fixed_agent_ids[spec.key] = child.id
            self._fixed_agent_keys[child.id] = spec.key
            self._fixed_agent_specs[child.id] = spec
            self._fixed_checkpoint_agents.setdefault(checkpoint.key, set()).add(child.id)
            registered.append(spec.key)
            self._fixed_record_observed_spawn(
                tc,
                parent,
                child,
                spawn_origin="checkpoint_spawn",
                checkpoint=checkpoint.key,
                role_key=spec.key,
            )
            access = (
                "read-only"
                if spec.read_only
                else f"write only: {', '.join(spec.allowed_paths) or '(no paths configured)'}"
            )
            role_message = (
                f"[Runtime role binding for {spec.key}] Your parent created you through NanoMA's "
                f"{tc.name} tool at checkpoint {checkpoint.key}. Access contract: {access}. "
                f"Assignment: {(spec.assignment or spec.bio)[:800]}. Do not work outside this contract."
            )
            if dependency_handoff:
                role_message += (
                    "\n\n[Runtime dependency handoff]\nUse these completed upstream findings as "
                    f"evidence; verify them before editing.\n\n{dependency_handoff}"
                )
            await self._notify_fixed_agent(child.id, role_message)
            self._emit(parent.id, "fixed_agent_spawned", {
                "checkpoint": checkpoint.key,
                "agent_key": spec.key,
                "child": child.id,
                "bio": spec.bio,
                "model": child.model,
                "executor": "nanoma_tool",
                "spawn_tool": tc.name,
                "tool_call_id": tc.id,
            })

        self._fixed_release_spawn_request(parent.id, checkpoint.key)
        expected_keys = {spec.key for spec in checkpoint.agents}
        spawned_keys = {
            self._fixed_agent_keys[agent_id]
            for agent_id in self._fixed_checkpoint_agents.get(checkpoint.key, set())
            if agent_id in self._fixed_agent_keys
        }
        if expected_keys.issubset(spawned_keys):
            self._fixed_fired_checkpoints.add(checkpoint.key)
            self._emit(parent.id, "fixed_checkpoint_fired", {
                "checkpoint": checkpoint.key,
                "mode": checkpoint.mode,
                "agents": sorted(spawned_keys),
                "elapsed_seconds": round(self.effective_elapsed(), 1),
                "spawn_execution": "nanoma_tool",
            })
        else:
            self._emit(parent.id, "fixed_checkpoint_spawn_incomplete", {
                "checkpoint": checkpoint.key,
                "requested": [spec.key for spec in specs],
                "registered": registered,
                "result": str(result)[:1000],
            })

        if isinstance(result, dict):
            result = {
                **result,
                "fixed_checkpoint": checkpoint.key,
                "fixed_roles_registered": registered,
                "spawn_execution": "nanoma_tool",
            }
        return result

    async def _advance_fixed_checkpoint(self, checkpoint: FixedCheckpoint) -> None:
        if not self._fixed_plan:
            return
        if (
            checkpoint.key in self._fixed_fired_checkpoints
            or checkpoint.key in self._fixed_requested_checkpoints
        ):
            return
        parent_id = self._fixed_agent_ids.get(checkpoint.parent_key)
        if not parent_id or parent_id not in self.agents:
            return
        if parent_id in self._fixed_pending_checkpoint_by_parent:
            return
        parent = self.agents[parent_id]
        if parent.status == "failed":
            return
        missing_specs = self._fixed_missing_specs(checkpoint)
        if not missing_specs:
            return

        live_agents = sum(
            other.status in {"running", "idle"}
            for other in self.agents.values()
        )
        available_slots = min(
            max(0, self._fixed_plan.max_live_agents - live_agents),
            max(0, self.config.max_agents - len(self.agents)),
        )
        if available_slots <= 0:
            return

        selected: list[FixedAgentSpec] = []
        for spec in missing_specs:
            if self._fixed_plan.direct_build_gate and not spec.read_only:
                writer_active = any(
                    not other_spec.read_only
                    and (other := self.agents.get(other_id)) is not None
                    and (
                        other.status in {"running", "idle"}
                        or (
                            other.status in {"done", "failed"}
                            and other_id not in self._fixed_validated_agents
                        )
                    )
                    for other_id, other_spec in self._fixed_agent_specs.items()
                )
                if writer_active:
                    continue
            selected.append(spec)
            if len(selected) >= available_slots:
                break
            if self._fixed_plan.direct_build_gate and not spec.read_only:
                break

        if not selected:
            return

        spawn_tool = "spawn_many" if len(selected) > 1 else "spawn"
        role_lines: list[str] = []
        for index, spec in enumerate(selected, start=1):
            access = (
                "read-only"
                if spec.read_only
                else f"write only: {', '.join(spec.allowed_paths) or '(no paths configured)'}"
            )
            role_lines.append(
                f"{index}. role_key={spec.key}; role={spec.bio}; access={access}; "
                f"assignment={(spec.assignment or spec.bio)[:1000]}"
            )
        request_message = (
            f"[Runtime checkpoint {checkpoint.key}] This checkpoint changes your next action only. "
            f"Runtime will not create these agents. You must call NanoMA's {spawn_tool} tool now "
            f"to create exactly {len(selected)} child{'ren' if len(selected) != 1 else ''}. "
            "Do not use shell, file, query, wait, send, or set_status before the spawn call. "
            "The runtime will bind the canonical assignments and access contracts to the children "
            "created by that tool call. This one call contains only the listed checkpoint roles; "
            "after it completes, your normal NanoMA spawn autonomy resumes.\n\n"
            + "\n".join(role_lines)
        )
        self._fixed_requested_checkpoints.add(checkpoint.key)
        self._fixed_pending_checkpoint_by_parent[parent_id] = checkpoint.key
        self._pending_tool_overrides[parent_id] = {
            "source": "fixed_orchestration_checkpoint",
            "action": "tool_override",
            "strategy_action": "FIXED_CHECKPOINT_SPAWN",
            "tools": [spawn_tool],
            "reason": f"fixed checkpoint {checkpoint.key}",
            "message": "",
            "once": False,
            "created_at": time.time(),
            "wrong_tool_attempts": 0,
            "metadata": {
                "fixed_checkpoint": checkpoint.key,
                "fixed_agent_keys": [spec.key for spec in selected],
                "spawn_tool": spawn_tool,
            },
        }
        self._tool_override_events.setdefault(parent_id, asyncio.Event()).set()
        self._fixed_completion_waiting.discard(parent_id)
        await self._notify_fixed_agent(parent_id, request_message)
        if parent.status == "done":
            parent.status = "running"
            if parent._task is None or parent._task.done():
                self.start_agent(parent)
            self._emit(parent_id, "fixed_checkpoint_parent_reactivated", {
                "checkpoint": checkpoint.key,
            })
        self._emit(parent_id, "fixed_checkpoint_requested", {
            "checkpoint": checkpoint.key,
            "mode": checkpoint.mode,
            "agents": [spec.key for spec in selected],
            "spawn_tool": spawn_tool,
            "elapsed_seconds": round(self.effective_elapsed(), 1),
            "executor": "parent_agent_via_nanoma_tool",
        })

    async def _poll_fixed_orchestration_once(self) -> None:
        if not self._fixed_plan:
            return
        async with self._fixed_orchestration_lock:
            self._fixed_expire_agents()
            await self._fixed_validate_completed_agents()
            for checkpoint in self._fixed_plan.checkpoints:
                if checkpoint.key in self._fixed_fired_checkpoints:
                    continue
                if self._fixed_checkpoint_ready(checkpoint):
                    await self._advance_fixed_checkpoint(checkpoint)

    async def _fixed_orchestration_monitor(self) -> None:
        if not self._fixed_plan:
            return
        try:
            while True:
                for agent_id in self._load_runtime_interventions():
                    self._tool_override_events.setdefault(
                        agent_id, asyncio.Event()
                    ).set()
                await self._poll_fixed_orchestration_once()
                await asyncio.sleep(self.config.fixed_orchestration_poll_seconds)
        except asyncio.CancelledError:
            return

    def _root_agent_for(self, agent: Agent) -> Agent:
        """Return the root without copying ancestor histories into a child."""
        current = agent
        seen: set[str] = set()
        while current.parent and current.parent in self.agents and current.id not in seen:
            seen.add(current.id)
            current = self.agents[current.parent]
        return current

    @staticmethod
    def _slice_text_by_tokens(text: str, offset: int, max_tokens: int) -> str:
        """Take an exact character window whose estimated size fits the budget."""
        source = str(text or "")
        start = max(0, min(len(source), int(offset or 0)))
        remaining = source[start:]
        if not remaining or estimate_tokens(remaining) <= max_tokens:
            return remaining
        low, high = 0, len(remaining)
        while low < high:
            mid = (low + high + 1) // 2
            if estimate_tokens(remaining[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        return remaining[:max(1, low)]

    def _task_contract_excerpt(self, task: str) -> str:
        """Extract interface/constraint lines while preserving their exact wording.

        This is deliberately deterministic rather than an LLM summary: the capsule
        cannot silently rewrite a signature, unit, index convention, or return shape.
        Omitted prose remains retrievable through ``get_task_context``.
        """
        source = str(task or "").strip()
        if not source:
            return "(root task is empty)"
        inline_limit = max(64, int(self.config.task_capsule_inline_root_max_tokens))
        if estimate_tokens(source) <= inline_limit:
            return source

        lines = source.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        signal_selected: set[int] = set()
        signal = re.compile(
            r"(?:^\s*(?:async\s+def|def|class)\s+|->\s*[^:]+:?\s*$|"
            r"\b(?:args?|arguments?|parameters?|returns?|output|signature|interface|"
            r"shape|dtype|index(?:ing)?|units?|constant|constraints?|requirements?|"
            r"must|required|exactly|do not|allowed|forbidden|previous|prior)\b)",
            flags=re.IGNORECASE,
        )
        for index, line in enumerate(lines):
            if signal.search(line):
                signal_selected.update(
                    range(max(0, index - 1), min(len(lines), index + 2))
                )

        boundary_selected = set(nonempty[:4]) | set(nonempty[-10:])
        excerpt_lines: list[str] = ["[Interface and constraint lines]"]
        excerpt_lines.extend(lines[index] for index in sorted(signal_selected))
        remaining_boundaries = sorted(boundary_selected - signal_selected)
        if remaining_boundaries:
            excerpt_lines.append("[Task opening/closing lines]")
            excerpt_lines.extend(lines[index] for index in remaining_boundaries)
        excerpt_lines.append("[… fetch omitted source with get_task_context …]")
        excerpt = "\n".join(excerpt_lines).strip()
        return self._slice_text_by_tokens(excerpt, 0, inline_limit)

    def _build_child_task_capsule(self, parent: Agent) -> str:
        if not self.config.task_capsule_enabled:
            return ""
        root = self._root_agent_for(parent)
        root_task = str(root.task or "")
        digest = hashlib.sha256(root_task.encode("utf-8")).hexdigest()[:16]
        paths = _extract_expected_output_paths(root_task)
        retrieval_limit = max(64, int(self.config.task_context_chunk_max_tokens))
        header = (
            "[Runtime compact task capsule]\n"
            f"Root agent: {root.id}; source SHA-256: {digest}; source chars: {len(root_task)}.\n"
            "Your Task line above is the child-specific assignment. The exact root task is not "
            "duplicated into every child context. If omitted scientific/background details matter, "
            f"call get_task_context(section=\"root\", offset=0); each call is capped at "
            f"{retrieval_limit} tokens and returns next_offset/has_more.\n"
            + (
                "Explicit output paths: " + ", ".join(paths) + ".\n"
                if paths else ""
            )
            + "Exact contract excerpt (verbatim):\n"
        )
        capsule = header + self._task_contract_excerpt(root_task)
        budget = max(128, int(self.config.task_capsule_max_tokens))
        return self._slice_text_by_tokens(capsule, 0, budget)

    def task_context(
        self,
        agent: Agent,
        *,
        section: str = "root",
        offset: int = 0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return bounded, auditable task context without copying whole histories."""
        section = str(section or "root").strip().lower()
        root = self._root_agent_for(agent)
        if section == "root":
            source = str(root.task or "")
        elif section == "parent":
            parent = self.agents.get(agent.parent or "")
            source = str(parent.task or "") if parent is not None else str(root.task or "")
        elif section == "assignment":
            source = str(agent.task or "")
        elif section == "contract":
            source = self._task_contract_excerpt(str(root.task or ""))
        elif section == "final_candidate":
            review = self._final_candidate_reviews.get(root.id) or {}
            if agent.id not in {root.id, review.get("reviewer_id")}:
                return {"error": "final_candidate is visible only to the root and its active final reviewer"}
            source = str(review.get("candidate") or "")
            if not source:
                return {"error": "no final candidate is pending review"}
        else:
            return {
                "error": "unknown section",
                "available_sections": ["root", "parent", "assignment", "contract", "final_candidate"],
            }

        try:
            start = max(0, min(len(source), int(offset or 0)))
        except (TypeError, ValueError):
            return {"error": "offset must be a non-negative integer"}
        configured_limit = max(64, int(self.config.task_context_chunk_max_tokens))
        try:
            requested_limit = int(max_tokens) if max_tokens is not None else configured_limit
        except (TypeError, ValueError):
            return {"error": "max_tokens must be an integer"}
        effective_limit = max(64, min(configured_limit, requested_limit))
        content = self._slice_text_by_tokens(source, start, effective_limit)
        next_offset = min(len(source), start + len(content))
        return {
            "section": section,
            "content": content,
            "offset": start,
            "next_offset": next_offset,
            "has_more": next_offset < len(source),
            "total_chars": len(source),
            "chunk_tokens": estimate_tokens(content),
            "max_tokens": effective_limit,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }

    def _final_review_delivery(self, root_id: str, reviewer_id: str) -> dict[str, Any] | None:
        for record in reversed(self._candidate_deliveries):
            if (
                record.get("parent_id") == root_id
                and record.get("agent_id") == reviewer_id
                and str(record.get("source") or "") == "deliver_to_parent"
            ):
                return record
        return None

    async def ensure_final_candidate_review(
        self,
        agent: Agent,
        candidate: str,
        *,
        override_reason: str = "",
    ) -> dict[str, Any]:
        """Gate root completion on a review of the concrete submitted candidate."""
        if not self.config.final_candidate_review_enabled or agent.parent is not None:
            return {"ready": True, "reason": "disabled_or_not_root"}
        if not agent.children:
            return {"ready": True, "reason": "single_agent_path"}

        candidate = str(candidate or agent.result or "").strip()
        if not candidate:
            return {"ready": True, "reason": "no_text_candidate"}
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        override_reason = str(override_reason or "").strip()
        if override_reason:
            self._emit(agent.id, "final_candidate_review_overridden", {
                "candidate_sha256": digest,
                "reason": override_reason[:500],
            })
            return {"ready": True, "reason": "explicit_override", "candidate_sha256": digest}

        state = self._final_candidate_reviews.get(agent.id)
        if state and state.get("candidate_sha256") == digest:
            reviewer_id = str(state.get("reviewer_id") or "")
            reviewer = self.agents.get(reviewer_id)
            delivery = self._final_review_delivery(agent.id, reviewer_id)
            if delivery is not None:
                answer = str(delivery.get("answer") or "").strip()
                verdict = answer.split(None, 1)[0].upper().rstrip(":") if answer else ""
                state.update({"verdict": verdict, "review_seq": delivery.get("seq")})
                if verdict == "ACCEPT":
                    self._emit(agent.id, "final_candidate_review_accepted", {
                        "reviewer": reviewer_id,
                        "round": state.get("round"),
                        "candidate_sha256": digest,
                    })
                    return {
                        "ready": True,
                        "reason": "accepted",
                        "reviewer_id": reviewer_id,
                        "round": state.get("round"),
                        "candidate_sha256": digest,
                    }
                return {
                    "ready": False,
                    "reason": "revision_requested" if verdict == "REVISE" else "unclear_verdict",
                    "reviewer_id": reviewer_id,
                    "round": state.get("round"),
                    "verdict": answer[:200],
                    "evidence": str(delivery.get("evidence") or "")[:2000],
                    "instruction": (
                        "Revise the concrete result and call set_status(done, result=<revised candidate>) "
                        "again. If the reviewer is wrong, supply review_override_reason with a concise rationale."
                    ),
                }
            if reviewer is not None and reviewer.status not in {"done", "failed", "killed"}:
                return {
                    "ready": False,
                    "reason": "review_in_progress",
                    "reviewer_id": reviewer_id,
                    "round": state.get("round"),
                    "instruction": "Wait for the final reviewer to deliver its verdict, then retry set_status(done).",
                }
            if self.config.final_candidate_review_fail_open:
                self._emit(agent.id, "final_candidate_review_degraded", {
                    "reviewer": reviewer_id,
                    "reason": "reviewer_finished_without_formal_delivery",
                    "candidate_sha256": digest,
                })
                return {"ready": True, "reason": "reviewer_failed_open", "reviewer_id": reviewer_id}
            return {
                "ready": False,
                "reason": "reviewer_finished_without_formal_delivery",
                "reviewer_id": reviewer_id,
                "instruction": "Retry after collecting a formal reviewer delivery or use review_override_reason.",
            }

        previous_round = int((state or {}).get("round") or 0)
        max_rounds = max(1, int(self.config.final_candidate_review_max_rounds))
        if previous_round >= max_rounds:
            return {
                "ready": False,
                "reason": "review_round_limit",
                "rounds": previous_round,
                "instruction": (
                    "The candidate changed after the maximum review rounds. Either restore the last reviewed "
                    "candidate or call set_status with review_override_reason explaining why the change is safe."
                ),
            }

        round_number = previous_round + 1
        self._final_candidate_reviews[agent.id] = {
            "candidate": candidate,
            "candidate_sha256": digest,
            "round": round_number,
            "created_at": time.time(),
        }
        review_task = (
            "[Runtime final-candidate review]\n"
            "Review the root's concrete pending submission, not an independently invented solution. "
            "First call get_task_context(section=\"final_candidate\", offset=0) and continue with "
            "next_offset while has_more=true. Fetch section=\"contract\" and only the root-task chunks "
            "needed to resolve omitted details. Audit the exact signature, output shape/type, index base, "
            "units/constants, boundary and rounding conventions, and library/API spellings. Do not run hidden "
            "tests or use benchmark-private answers. Then call deliver_to_parent exactly once: answer must be "
            "either ACCEPT or REVISE; evidence must name concrete defects and the smallest correction. "
            "Do not place a replacement solution in answer."
        )
        if "spawn" in self.config.disabled_tools:
            spawn_result: dict[str, Any] = {"error": "spawn is disabled by runtime configuration"}
        else:
            # This is a bounded completion gate, not a new planning decision.
            # Authorize exactly this spawn so end-of-run policy pressure cannot
            # silently skip the review, then immediately restore normal policy.
            self._strategy_spawn_authorized.add(agent.id)
            try:
                spawn_result = await self._invoke_meta_spawn({"task": review_task}, agent)
            finally:
                self._strategy_spawn_authorized.discard(agent.id)
        reviewer_id = str((spawn_result or {}).get("agent_id") or "") if isinstance(spawn_result, dict) else ""
        if not reviewer_id:
            self._final_candidate_reviews.pop(agent.id, None)
            detail = str(spawn_result)[:500]
            self._emit(agent.id, "final_candidate_review_spawn_failed", {
                "round": round_number,
                "candidate_sha256": digest,
                "detail": detail,
            })
            if self.config.final_candidate_review_fail_open:
                return {"ready": True, "reason": "review_spawn_failed_open", "detail": detail}
            return {"ready": False, "reason": "review_spawn_failed", "detail": detail}

        self._final_candidate_reviews[agent.id]["reviewer_id"] = reviewer_id
        reviewer = self.agents.get(reviewer_id)
        if reviewer is not None:
            setattr(reviewer, "_final_candidate_reviewer_for", agent.id)
        self._emit(agent.id, "final_candidate_review_started", {
            "reviewer": reviewer_id,
            "round": round_number,
            "candidate_sha256": digest,
            "candidate_chars": len(candidate),
        })
        return {
            "ready": False,
            "reason": "review_started",
            "reviewer_id": reviewer_id,
            "round": round_number,
            "candidate_sha256": digest,
            "instruction": "Wait for the reviewer delivery, incorporate any concrete correction, then retry set_status(done).",
        }

    def create_agent(
        self,
        task: str,
        model: str | None = None,
        quota: ResourceQuota | None = None,
        parent: str | None = None,
        depth: int = 0,
    ) -> Agent:
        agent_id = self._id_gen.next()
        model = model or self.config.default_model
        quota = quota or ResourceQuota(
            budget=self.config.budget,
            time_limit=self.config.time_limit,
            max_turns=self.config.max_turns,
        )

        workspace = self.config.workspace_root / agent_id
        workspace.mkdir(parents=True, exist_ok=True)

        # Build context for sub-agents
        parent_context = None
        if parent and parent in self.agents:
            p = self.agents[parent]
            siblings = [sid for sid in p.children if sid in self.agents]
            sibling_info = ", ".join(f"{sid}({self.agents[sid].task[:30]})" for sid in siblings[:5])
            parent_context = {
                "parent_id": parent,
                "parent_task": p.task[:100],
                "siblings": sibling_info,
                "depth": depth,
                "task_capsule": self._build_child_task_capsule(p),
            }

        # Merge-submit-path: a spawned child works on a private copy of the task
        # directory so parallel children never collide. Anchor the copy path in
        # the child's task text (runtime-owned isolation, not a static prompt
        # guideline) before the system prompt is built.
        merge_copy_dir: Path | None = None
        if parent is not None and self._merge_active():
            merge_copy_dir = workspace / "_task_copy"
            scope = self._merge_scope_roots()
            contribution = ", ".join(p.as_posix() for p in scope) if scope else "the whole directory"
            task = (
                task
                + f"\n\n[Workspace] Work in {merge_copy_dir} — a complete private copy of "
                "the task directory (harness, datasets, dependencies included), not a stub. "
                "cd there and build, run and evaluate exactly as you would in the original. "
                f"Your contribution is whatever you change under: {contribution} — the runtime "
                "folds that back into the real submission when you finish. Do not edit the "
                "shared task directory directly. Register your check with the verify tool: the "
                "runtime re-runs it against the merged result, and work it cannot verify can "
                "never be treated as the best state."
            )

        system_prompt = self._build_system_prompt(agent_id, task, workspace, parent_context)

        agent = Agent(
            id=agent_id, task=task, model=model, quota=quota,
            parent=parent, depth=depth, workspace=workspace,
            history=[{"role": "system", "content": system_prompt}],
        )

        # Context limit from model registry
        try:
            from nanoma.models import get_registry
            agent.context_limit = get_registry().context_limit(model)
        except Exception:
            pass

        self.agents[agent_id] = agent
        agent._created_at = time.time()
        if parent and parent in self.agents:
            self.agents[parent].children.add(agent_id)

        if merge_copy_dir is not None:
            # Seed the private copy now that the agent is registered (it is
            # excluded from the "active children" check until _merge_copy is set,
            # so a fresh round re-snapshots the baseline correctly).
            self._merge_seed_child_copy(agent)

        self._emit(agent_id, "agent_new", {
            "task": task, "model": model, "budget": quota.budget if math.isfinite(quota.budget) else None,
            "parent": parent, "depth": depth,
        })
        return agent

    def _delivery_candidate_bases(self, agent: Agent) -> list[Path]:
        """Ordered roots that may contain a complete final artifact tree."""

        contract = self.config.delivery_contract
        if contract is None:
            return []
        candidates = [
            agent.workspace,
            contract.target_root,
            self._tool_context.shared_dir,
            self.config.workspace_root,
            *self.config.workspace_extra_roots,
        ]
        candidates.extend(
            other.workspace
            for other in sorted(self.agents.values(), key=lambda item: item.id)
            if other.id != agent.id
        )
        result: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            path = Path(candidate).expanduser()
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path.absolute())
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
        return result

    def finalize_delivery(
        self,
        agent: Agent,
        *,
        trigger: str,
        explicit_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Publish and validate the root agent's benchmark delivery contract.

        Child artifacts remain handoffs to their parent.  Only the root can
        publish the official destination, which prevents a late child from
        replacing a reconciled result behind the root's back.
        """

        contract = self.config.delivery_contract
        if contract is None or agent.parent is not None:
            return {"ready": True, "trigger": trigger, "contract": False}

        # Only the current submit call gets explicit priority. Later done/exit
        # checks prefer an already-published official tree, so an older artifact
        # record cannot overwrite newer reconciled work.
        explicit = list(explicit_paths or [])
        report = publish_delivery_contract(
            contract,
            candidate_bases=self._delivery_candidate_bases(agent),
            explicit_paths=explicit,
            trigger=trigger,
        ).as_dict()
        report["contract"] = True
        self._delivery_contract_history.append(copy.deepcopy(report))
        self._emit(agent.id, "delivery_contract_ready" if report["ready"] else "delivery_contract_blocked", {
            "trigger": trigger,
            "published": report.get("published", []),
            "satisfied": report.get("satisfied", []),
            "missing": report.get("missing", []),
            "checked_candidates": report.get("checked_candidates", [])[-20:],
        })
        return report

    def start_agent(self, agent: Agent):
        agent._task = asyncio.ensure_future(self._agent_loop(agent))

    def _llm_exception_status_code(self, exc: Exception) -> int | None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
        return None

    def _note_llm_failure_for_admission(self, exc: Exception) -> None:
        if not self.config.llm_admission_control:
            return
        cooldown = float(self.config.llm_overload_cooldown_seconds or 0.0)
        if cooldown <= 0:
            return
        status = self._llm_exception_status_code(exc)
        if status in {429, 500, 502, 503, 504}:
            self._llm_overload_cooldown_until = max(
                self._llm_overload_cooldown_until,
                time.time() + cooldown,
            )

    async def _admit_llm_call(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        scheduler_before = dict(self.scheduler.stats)
        context_tokens = agent.context_tokens or count_message_tokens(agent.history)
        admission = {
            "enabled": bool(self.config.llm_admission_control),
            "scheduler_before": scheduler_before,
            "context_tokens": context_tokens,
            "message_count": len(agent.history),
            "tool_count": len(turn_tools),
            "wait_ms": 0,
            "reason": "disabled",
        }
        if not self.config.llm_admission_control:
            return admission

        active_or_waiting = int(scheduler_before.get("active") or 0) + int(scheduler_before.get("waiting") or 0)
        spacing = 0.0
        reason = "none"
        if active_or_waiting > 0 and self.config.llm_min_start_spacing > 0:
            spacing = max(spacing, float(self.config.llm_min_start_spacing))
            reason = "concurrent_llm_start"
        if (
            active_or_waiting > 0
            and self.config.llm_large_context_tokens > 0
            and context_tokens >= self.config.llm_large_context_tokens
            and self.config.llm_large_context_spacing > 0
        ):
            spacing = max(spacing, float(self.config.llm_large_context_spacing))
            reason = "large_context_concurrent_start"

        async with self._llm_admission_lock:
            now = time.time()
            wait_until = max(self._last_llm_start_time + spacing, self._llm_overload_cooldown_until)
            wait_seconds = max(0.0, wait_until - now)
            if self.config.llm_admission_max_delay > 0:
                wait_seconds = min(wait_seconds, float(self.config.llm_admission_max_delay))
            if wait_seconds > 0:
                admission["reason"] = (
                    "provider_overload_cooldown"
                    if self._llm_overload_cooldown_until > now and self._llm_overload_cooldown_until >= wait_until
                    else reason
                )
                admission["wait_ms"] = int(wait_seconds * 1000)
                self._emit(agent.id, "llm_admission_wait", admission)
                await asyncio.sleep(wait_seconds)
            else:
                admission["reason"] = reason
            self._last_llm_start_time = time.time()
            admission["scheduler_after_wait"] = dict(self.scheduler.stats)
            return admission

    async def _maybe_await(self, value: Any) -> Any:
        if hasattr(value, "__await__"):
            return await value
        return value

    def effective_elapsed(self) -> float:
        return max(0.0, time.time() - self._start_time - self._excluded_time_seconds)

    async def _measure_excluded_time(self, reason: str, func: Callable[[], Any]) -> Any:
        started = time.time()
        try:
            value = func()
            return await self._maybe_await(value)
        finally:
            elapsed = time.time() - started
            self._excluded_time_seconds += elapsed
            self._excluded_time_by_reason[reason] = (
                self._excluded_time_by_reason.get(reason, 0.0) + elapsed
            )

    async def _write_auto_checkpoint(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
        *,
        reason: str,
    ) -> Path | None:
        if not (self.config.probe_every_llm or self.config.auto_checkpoint_enabled):
            return None
        if not (self.config.probe_dir or self.config.auto_checkpoint_dir):
            return None
        self._global_checkpoint_seq += 1
        checkpoint_record = {
            "seq": self._global_checkpoint_seq,
            "path": "",
            "agent": agent.id,
            "turn": agent._turns,
            "created_at": time.time(),
            "reason": reason,
            "active_agents": [
                other.id for other in self.agents.values()
                if other.status in {"running", "idle"}
            ],
            "agent_turns": {
                other.id: other._turns for other in self.agents.values()
            },
        }
        probe_path = await self._measure_excluded_time(
            "probe_write",
            lambda: self._write_probe(
                agent,
                turn_tools,
                tool_policy,
                reason=reason,
                global_checkpoint=checkpoint_record,
            ),
        )
        checkpoint_record["path"] = str(probe_path)
        self._last_checkpoint_path = probe_path
        self._last_checkpoint_agent = agent.id
        self._global_checkpoints.append(checkpoint_record)
        if len(self._global_checkpoints) > 500:
            self._global_checkpoints = self._global_checkpoints[-500:]
        hook_data = None
        if self._checkpoint_hook is not None:
            try:
                hook_data = await self._measure_excluded_time(
                    "checkpoint_hook",
                    lambda: self._checkpoint_hook(probe_path),
                )
            except Exception as exc:
                self._emit(agent.id, "checkpoint_hook_error", {
                    "path": str(probe_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
        if hook_data is not None:
            meta_path = probe_path.parent / "meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["external_snapshot"] = hook_data
                meta["global_checkpoint"] = checkpoint_record
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
        self._emit(agent.id, "probe" if reason == "manual_probe" else "checkpoint", {
            "path": str(probe_path),
            "turn": agent._turns,
            "reason": reason,
            "global_checkpoint": checkpoint_record,
            "tools": list(turn_tools),
            "tool_policy": tool_policy.as_event(),
            "external_snapshot": hook_data,
        })
        return probe_path

    def _is_infra_failure_exception(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "400 bad request",
                "401 unauthorized",
                "403 forbidden",
                "408 request timeout",
                "503 service unavailable",
                "503",
                "429",
                "bad request",
                "httpstatuserror",
                "rate limit",
                "server error",
                "temporarily unavailable",
                "connection reset",
                "read timeout",
                "connect timeout",
                "timeout",
            )
        )

    def _select_rollback_checkpoint(
        self,
        agent: Agent,
        *,
        preferred_path: Path | None = None,
        reason: str,
    ) -> tuple[Path | None, dict[str, Any] | None]:
        records = [
            record for record in self._global_checkpoints
            if record.get("path")
        ]
        if preferred_path is not None:
            preferred = str(preferred_path)
            for record in reversed(records):
                if record.get("path") == preferred:
                    return preferred_path, record
            return preferred_path, None

        if not records and self._last_checkpoint_path is not None:
            return self._last_checkpoint_path, {
                "path": str(self._last_checkpoint_path),
                "agent": self._last_checkpoint_agent,
            }
        if not records:
            return None, None

        if reason == "infra_failure":
            state = self._infra_failure_state_by_agent.get(agent.id) or {}
            max_seq = state.get("rollback_before_seq")
            if max_seq is None:
                max_seq = records[-1].get("seq")
            eligible = [
                record for record in records
                if int(record.get("seq") or 0) < int(max_seq or 0)
            ]
            if not eligible:
                eligible = records[:1]
            record = eligible[-1]
            self._infra_failure_state_by_agent[agent.id] = {
                **state,
                "rollback_before_seq": int(record.get("seq") or 0),
                "last_selected_path": record.get("path"),
            }
            return Path(record["path"]), record

        record = records[-1]
        return Path(record["path"]), record

    def _request_rollback(
        self,
        agent: Agent,
        *,
        reason: str,
        detail: str,
        checkpoint_path: Path | None = None,
    ) -> bool:
        if not self.config.auto_rollback_enabled:
            return False
        if self._rollback_requested is not None:
            return True
        if self._rollback_attempts >= max(0, self.config.auto_rollback_max_attempts):
            self._emit(agent.id, "rollback_skipped", {
                "reason": reason,
                "detail": detail[:500],
                "attempts": self._rollback_attempts,
                "max_attempts": self.config.auto_rollback_max_attempts,
            })
            return False
        selected_path, checkpoint_record = self._select_rollback_checkpoint(
            agent,
            preferred_path=checkpoint_path,
            reason=reason,
        )
        if selected_path is None:
            self._emit(agent.id, "rollback_skipped", {
                "reason": reason,
                "detail": detail[:500],
                "error": "no_checkpoint_available",
            })
            return False
        self._rollback_requested = {
            "agent": agent.id,
            "path": str(selected_path),
            "checkpoint_agent": (
                checkpoint_record.get("agent") if checkpoint_record else self._last_checkpoint_agent
            ),
            "global_checkpoint": checkpoint_record,
            "reason": reason,
            "detail": detail[:1200],
            "requested_at": time.time(),
            "attempt": self._rollback_attempts + 1,
        }
        self._emit(agent.id, "rollback_requested", self._rollback_requested)
        current = asyncio.current_task()
        for other in self.agents.values():
            task = other._task
            if task and not task.done() and task is not current:
                task.cancel()
        return True

    async def _perform_requested_rollback(self) -> None:
        request = self._rollback_requested
        if not request:
            return
        self._rollback_requested = None
        self._rollback_attempts += 1
        self._rollback_history.append(copy.deepcopy(request))
        probe_path = Path(request["path"])
        for agent in list(self.agents.values()):
            task = agent._task
            if task and not task.done():
                task.cancel()
        tasks = [a._task for a in self.agents.values() if a._task and not a._task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        active_agent = await self._measure_excluded_time(
            "probe_restore",
            lambda: self.restore_probe(probe_path),
        )
        if self._restore_hook is not None:
            try:
                await self._measure_excluded_time(
                    "restore_hook",
                    lambda: self._restore_hook(probe_path),
                )
            except Exception as exc:
                self._emit(active_agent, "restore_hook_error", {
                    "path": str(probe_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
        instruction = self.config.auto_rollback_instruction or (
            "The runtime rolled back to the previous saved node because the last branch hit an "
            "infrastructure or candidate failure. Re-run from this clean state. Treat failed branches "
            "as not covering the work. Reconsider topology before continuing: if the remaining work can "
            "be split into independent units, spawn short agents; if there are multiple plausible paths, "
            "spawn parallel candidate branches."
        )
        note = (
            "[Runtime rollback]\n"
            f"Reason: {request.get('reason')}\n"
            f"Detail: {request.get('detail')}\n"
            f"Restored checkpoint: {probe_path.parent.name}\n\n"
            f"{instruction}"
        )
        for agent in self.agents.values():
            if agent.status in {"running", "idle"}:
                agent.history.append({"role": "user", "content": note})
                agent.status = "running"
                agent._task = None
        self._last_checkpoint_path = probe_path
        self._last_checkpoint_agent = active_agent
        self._emit(active_agent, "rollback_restored", {
            "path": str(probe_path),
            "reason": request.get("reason"),
            "attempt": self._rollback_attempts,
            "active_agents": [
                agent.id for agent in self.agents.values()
                if agent.status in {"running", "idle"}
            ],
        })
        for agent in self.agents.values():
            if agent.status == "running":
                self.start_agent(agent)

    def _ensure_tool_call_responses(self, agent: Agent) -> bool:
        """Repair OpenAI tool-call adjacency before sending history to an LLM.

        OpenAI-compatible APIs reject histories where an assistant message with
        tool_calls is not immediately followed by one tool message per
        tool_call_id. Runtime interrupts, status changes, and rollbacks must not
        leave the conversation in that shape.
        """
        repaired = False
        new_history: list[Message] = []
        i = 0
        while i < len(agent.history):
            msg = agent.history[i]
            if msg.get("role") == "tool":
                repaired = True
                i += 1
                continue
            new_history.append(msg)
            tool_calls = msg.get("tool_calls") or []
            if msg.get("role") == "assistant" and tool_calls:
                expected = [tc.get("id") for tc in tool_calls if tc.get("id")]
                seen: set[str] = set()
                j = i + 1
                while j < len(agent.history) and agent.history[j].get("role") == "tool":
                    tool_msg = agent.history[j]
                    tool_call_id = tool_msg.get("tool_call_id")
                    if tool_call_id in expected and tool_call_id not in seen:
                        new_history.append(tool_msg)
                        seen.add(tool_call_id)
                    else:
                        repaired = True
                    j += 1
                for tool_call_id in expected:
                    if tool_call_id not in seen:
                        new_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({
                                "skipped": True,
                                "reason": "runtime_repaired_missing_tool_response",
                            }),
                        })
                        repaired = True
                i = j
                continue
            i += 1
        if repaired:
            agent.history = new_history
            self._emit(agent.id, "history_repaired", {
                "reason": "tool_call_adjacency",
                "messages": len(agent.history),
            })
        return repaired

    def _safe_recent_start(self, history: list[Message], keep_recent: int) -> int:
        """Return a recent-history boundary that preserves tool-call blocks."""
        if not history:
            return 0
        start = max(1, len(history) - max(1, keep_recent))
        while start > 1 and history[start].get("role") == "tool":
            start -= 1
        if start > 1:
            prev = history[start - 1]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                start -= 1
        return start

    async def run(self, task: str, model: str | None = None) -> str:
        """Run a single root agent to completion, then shut down all remaining agents."""
        root = self.create_agent(task, model=model)
        root_id = root.id
        if self._fixed_plan:
            self._fixed_agent_ids["root"] = root.id
            self._fixed_agent_keys[root.id] = "root"
            root.history.append({
                "role": "user",
                "content": self._fixed_plan.root_instructions,
            })
            self._emit(root.id, "fixed_orchestration_started", {
                "profile": self._fixed_plan.name,
                "checkpoints": [item.key for item in self._fixed_plan.checkpoints],
                "max_live_agents": self._fixed_plan.max_live_agents,
                "max_concurrent_llm": self._fixed_plan.max_concurrent_llm,
                "source_isolation": (
                    "shared-tree-serial"
                    if self._fixed_plan.direct_build_gate
                    else "shared-tree"
                ),
                "spawn_execution": "parent_agent_via_nanoma_tool",
                "supervisor_enabled": False,
            })
            self._fixed_initialize_green_checkpoint(root.id)
            if self._fixed_plan.direct_build_gate:
                baseline_build = await self._fixed_run_direct_lake_build(
                    root.id,
                    reason="runtime_start_baseline",
                )
                root.history.append({
                    "role": "user",
                    "content": (
                        "[Runtime verification gate] Initial task state "
                        f"{'passed' if baseline_build.get('accepted') else 'failed'} "
                        f"with exit_code={baseline_build.get('exit_code')}, "
                        f"metric={baseline_build.get('metric')}."
                    ),
                })
        self.start_agent(root)
        if self._fixed_plan:
            self._fixed_monitor_task = asyncio.create_task(
                self._fixed_orchestration_monitor()
            )
        while True:
            while not self._rollback_requested:
                active_tasks = [
                    agent._task for agent in self.agents.values()
                    if agent.status == "running" and agent._task is not None and not agent._task.done()
                ]
                if not active_tasks:
                    break
                done, _pending = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await asyncio.gather(*done, return_exceptions=True)
                root = self.agents.get(root_id) or root
                if root._task is not None and root._task.done() and not self._rollback_requested:
                    break
            if not self._rollback_requested:
                break
            await self._perform_requested_rollback()
            root = self.agents.get(root_id) or next(
                (a for a in self.agents.values() if a.parent is None),
                None,
            )
            if root is None:
                break
            if root._task is None or root._task.done():
                self.start_agent(root)
        # Freeze the shared source before cancelling children so the final scorer
        # never observes a half-written tree.
        if self._fixed_plan:
            self._fixed_source_writes_frozen = True
        # Clean up any still-running children
        if self._fixed_monitor_task and not self._fixed_monitor_task.done():
            self._fixed_monitor_task.cancel()
            await asyncio.gather(self._fixed_monitor_task, return_exceptions=True)
        running = [a for a in self.agents.values() if a.status in ("running", "idle") and a.id != root.id]
        if running:
            await self.shutdown()
        if (
            self._fixed_plan
            and self._fixed_source_write_seq != self._fixed_last_green_write_seq
        ):
            self._fixed_restore_green_checkpoint(root.id, "runtime_finished_with_dirty_source")
        # The harness scores whatever the submission path holds when the run
        # ends: collect the work of children that shutdown just cancelled, then
        # cash in the best measured state before handing it over.
        try:
            await self._await_delivery_tasks()
        except Exception as exc:
            self._emit("root", "delivery_drain_error", {"detail": str(exc)[:300]})
        try:
            await self._merge_promote_pending()
        except Exception as exc:
            self._emit("root", "merge_promote_error", {"detail": str(exc)[:300]})
        self._merge_restore_best()
        # The finalizer is runtime-owned and therefore also runs on deadline,
        # max-turn, cancellation, and other exits that bypass set_status.
        self.finalize_delivery(root, trigger="runtime_exit")
        return root.result or ""

    # ─── Message delivery ────────────────────────────────────────────────

    async def deliver(self, envelope: Envelope):
        agent = self.agents.get(envelope.to_id)
        if not agent:
            return
        # Track for stats
        self._messages_sent.append((envelope.from_id, envelope.to_id, envelope.tokens))
        if envelope.mode == "immediate":
            agent._immediate_inbox.put_nowait(envelope)
        elif envelope.mode == "steer":
            agent._steer_inbox.put_nowait(envelope)
        else:
            agent._queue_inbox.put_nowait(envelope)
        # Wake idle agents (guard against double-start)
        if agent.status == "idle":
            agent.status = "running"
            if agent._task is None or agent._task.done():
                self.start_agent(agent)

    def _record_candidate_delivery(
        self,
        agent: Agent,
        *,
        parent_id: str,
        answer: str,
        evidence: str = "",
        confidence: float = 0.5,
        method: str = "",
        source: str = "done",
    ) -> dict[str, Any] | None:
        """Persist a child candidate independently from inbox delivery."""
        if not self.config.candidate_delivery_ledger_enabled:
            return None
        answer = str(answer or "").strip()
        evidence = str(evidence or "").strip()
        if not answer and not evidence:
            return None
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        for existing in reversed(self._candidate_deliveries):
            if (
                existing.get("agent_id") == agent.id
                and existing.get("parent_id") == parent_id
                and existing.get("answer") == answer
            ):
                old_confidence = float(existing.get("confidence") or 0)
                source_rank = {
                    "done": 1,
                    "child_history": 2,
                    "child_response": 3,
                    "deliver_to_parent": 4,
                }
                existing_source = str(existing.get("source") or "")
                stronger_source = source_rank.get(source, 0) > source_rank.get(
                    existing_source, 0
                )
                confidence_upgrade_allowed = (
                    stronger_source or source == existing_source
                )
                stronger_delivery = (
                    confidence_upgrade_allowed and confidence > old_confidence
                ) or stronger_source
                if evidence and (not existing.get("evidence") or stronger_delivery):
                    existing["evidence"] = evidence
                if method and (not existing.get("method") or stronger_delivery):
                    existing["method"] = str(method).strip()
                if stronger_source:
                    existing["source"] = source
                if confidence_upgrade_allowed:
                    existing["confidence"] = max(old_confidence, confidence)
                existing["updated_at"] = time.time()
                event = self._candidate_delivery_events.get(parent_id)
                if event is not None and self._candidate_record_usable_for_convergence(existing):
                    event.set()
                if stronger_delivery:
                    self._emit(agent.id, "candidate_delivery_upgraded", {
                        "seq": existing.get("seq"),
                        "parent_id": parent_id,
                        "answer": answer[:300],
                        "confidence": existing["confidence"],
                        "method": str(existing.get("method") or "")[:120],
                        "source": existing.get("source"),
                    })
                return existing
        self._candidate_delivery_seq += 1
        record = {
            "seq": self._candidate_delivery_seq,
            "agent_id": agent.id,
            "parent_id": parent_id,
            "answer": answer,
            "evidence": evidence,
            "confidence": confidence,
            "method": str(method or "").strip(),
            "source": source,
            "created_at": time.time(),
        }
        self._candidate_deliveries.append(record)
        self._emit(agent.id, "candidate_delivery_recorded", {
            "seq": record["seq"],
            "parent_id": parent_id,
            "answer": answer[:300],
            "confidence": confidence,
            "method": record["method"][:120],
            "source": source,
        })
        event = self._candidate_delivery_events.get(parent_id)
        if event is not None and self._candidate_record_usable_for_convergence(record):
            event.set()
        return record

    async def _await_llm_response(
        self,
        agent: Agent,
        llm_awaitable: Awaitable[LLMResponse],
        timeout: float | None,
    ) -> LLMResponse:
        """Wait for an LLM, interrupting stale work after runtime state changes."""
        has_active_direct_child = any(
            child_id in self.agents
            and self.agents[child_id].status not in {"done", "failed"}
            for child_id in agent.children
        )
        should_interrupt_for_candidate = bool(
            self.config.candidate_llm_interrupt_enabled
            and self.config.candidate_convergence_enabled
            and (agent.parent is None or has_active_direct_child)
            and not self._candidate_delivery_records(agent)
        )
        candidate_event = (
            self._candidate_delivery_events.setdefault(agent.id, asyncio.Event())
            if should_interrupt_for_candidate
            else None
        )
        override_event = self._tool_override_events.setdefault(
            agent.id, asyncio.Event()
        )
        llm_task = asyncio.ensure_future(llm_awaitable)
        candidate_task = (
            asyncio.create_task(candidate_event.wait())
            if candidate_event is not None
            else None
        )
        override_task = asyncio.create_task(override_event.wait())
        waiters = {llm_task, override_task}
        if candidate_task is not None:
            waiters.add(candidate_task)
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if llm_task in done:
                return await llm_task
            if override_task in done and override_event.is_set():
                override_event.clear()
                llm_task.cancel()
                await asyncio.gather(llm_task, return_exceptions=True)
                raise _RuntimeToolOverrideInterrupt
            if (
                candidate_task is not None
                and candidate_task in done
                and candidate_event is not None
                and candidate_event.is_set()
            ):
                llm_task.cancel()
                await asyncio.gather(llm_task, return_exceptions=True)
                raise _CandidateDeliveryInterrupt
            llm_task.cancel()
            await asyncio.gather(llm_task, return_exceptions=True)
            raise asyncio.TimeoutError
        finally:
            pending_waiters = [
                task
                for task in (candidate_task, override_task)
                if task is not None
            ]
            for task in pending_waiters:
                task.cancel()
            await asyncio.gather(*pending_waiters, return_exceptions=True)

    def _llm_max_tokens_for_turn(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
    ) -> int | None:
        caps: list[int] = []
        if agent.parent is not None and agent._turns == 1:
            child_first_cap = max(0, int(self.config.child_first_turn_max_tokens))
            if child_first_cap > 0:
                caps.append(child_first_cap)
        if len(turn_tools) != 1:
            return min(caps) if caps else None
        single_cap = max(0, int(self.config.single_tool_max_tokens))
        if single_cap > 0:
            caps.append(single_cap)
        tool_name = next(iter(turn_tools))
        if tool_name in {
            "deliver_to_parent",
            "set_status",
            "wait",
        }:
            delivery_cap = max(0, int(self.config.delivery_tool_max_tokens))
            if delivery_cap > 0:
                caps.append(delivery_cap)
        if tool_name in {
            "spawn",
            "spawn_many",
            "query",
        }:
            control_cap = max(0, int(self.config.control_tool_max_tokens))
            if control_cap > 0:
                caps.append(control_cap)
        return min(caps) if caps else None

    def _root_should_park_for_candidate(self, agent: Agent) -> bool:
        if (
            not self.config.candidate_root_park_enabled
            or not self.config.candidate_convergence_enabled
            or (
                self._fixed_plan is not None
                and not self._fixed_plan.root_parks_after_spawn
            )
            or agent.parent is not None
            or not agent.children
        ):
            return False
        records = self._candidate_delivery_records(agent)
        children = [self.agents[child_id] for child_id in agent.children if child_id in self.agents]
        active_children = [
            child for child in children if child.status not in {"done", "failed"}
        ]
        formal_records = [
            record
            for record in records
            if (
                str(record.get("source") or "") == "deliver_to_parent"
                and bool(str(record.get("evidence") or "").strip())
            )
        ]
        if self.config.child_delivery_tool_required:
            required_formal_deliveries = min(2, len(children))
            if (
                required_formal_deliveries > 0
                and len(formal_records) >= required_formal_deliveries
            ):
                return False
        confidence_threshold = _clamp01(
            self.config.candidate_convergence_high_confidence_threshold
        )
        strong_candidate = any(
            (
                str(record.get("source") or "") == "deliver_to_parent"
                and bool(str(record.get("evidence") or "").strip())
            )
            or (
                confidence_threshold > 0
                and float(record.get("confidence") or 0) >= confidence_threshold
                and len(str(record.get("evidence") or "").strip()) >= 40
            )
            for record in records
        )
        if strong_candidate and not self.config.child_delivery_tool_required:
            return False
        if records and agent.quota.time_limit > 0:
            time_fraction = self.effective_elapsed() / agent.quota.time_limit
            if time_fraction >= self.config.candidate_convergence_time_fraction:
                return False
        return bool(active_children)

    async def _maybe_park_root_for_candidate(self, agent: Agent) -> bool:
        """Yield the root's model slot while any spawned child is still working."""
        if not self._root_should_park_for_candidate(agent):
            if agent.id in self._candidate_root_parked:
                self._candidate_root_parked.discard(agent.id)
                records = self._candidate_delivery_records(agent)
                self._emit(agent.id, "candidate_root_park_end", {
                    "candidate_count": len(records),
                    "formal_delivery_count": sum(
                        1 for record in records
                        if (
                            str(record.get("source") or "") == "deliver_to_parent"
                            and bool(str(record.get("evidence") or "").strip())
                        )
                    ),
                    "required_formal_deliveries": (
                        min(2, len(agent.children))
                        if self.config.child_delivery_tool_required else 0
                    ),
                    "active_children": sum(
                        1 for child_id in agent.children
                        if child_id in self.agents
                        and self.agents[child_id].status not in {"done", "failed"}
                    ),
                })
            return False

        if agent.id not in self._candidate_root_parked:
            self._candidate_root_parked.add(agent.id)
            self._emit(agent.id, "candidate_root_park_start", {
                "active_children": len(agent.children),
                "turn": agent._turns,
            })

        timeout = max(0.05, float(self.config.candidate_root_park_poll_seconds))
        if agent.quota.time_limit > 0:
            remaining = agent.quota.time_limit - self.effective_elapsed()
            if remaining <= 0:
                return False
            timeout = min(timeout, remaining)

        candidate_event = self._candidate_delivery_events.setdefault(agent.id, asyncio.Event())
        try:
            await asyncio.wait_for(candidate_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        self._harvest_child_candidates(agent)
        should_park = self._root_should_park_for_candidate(agent)
        if should_park and candidate_event.is_set():
            candidate_event.clear()
        if not should_park and agent.id in self._candidate_root_parked:
            self._candidate_root_parked.discard(agent.id)
            records = self._candidate_delivery_records(agent)
            self._emit(agent.id, "candidate_root_park_end", {
                "candidate_count": len(records),
                "formal_delivery_count": sum(
                    1 for record in records
                    if (
                        str(record.get("source") or "") == "deliver_to_parent"
                        and bool(str(record.get("evidence") or "").strip())
                    )
                ),
                "required_formal_deliveries": (
                    min(2, len(agent.children))
                    if self.config.child_delivery_tool_required else 0
                ),
                "active_children": sum(
                    1 for child_id in agent.children
                    if child_id in self.agents
                    and self.agents[child_id].status not in {"done", "failed"}
                ),
            })
        return should_park

    @staticmethod
    def _candidate_answer_key(answer: str) -> str:
        value = str(answer or "").strip().lower()
        value = re.sub(r"[\s`\"'.,;:!?()\[\]{}]+", " ", value)
        return value.strip()

    @staticmethod
    def _candidate_answer_usable(answer: str) -> bool:
        value = str(answer or "").strip()
        if not value or value.startswith("["):
            return False
        lowered = value.lower()
        if re.fullmatch(r"(?:\.{2,}|\u2026+|[-_?\s]+)", value):
            return False
        if re.search(r"<[^>]{1,120}>", value):
            return False
        if lowered in {"answer", "answer-only", "final answer", "candidate", "result"}:
            return False
        if lowered.startswith((
            "crash:",
            "unable to determine",
            "unable to complete",
            "wrote final answer",
            "wrote answer",
            "research complete",
            "verification complete",
        )):
            return False
        if re.match(r"^only\s+\d+\s+(?:of|/)\s*\d+\b", lowered):
            return False
        if (
            not re.search(r"\d", value)
            and re.search(
                r"^the\s+(?:absolute\s+)?difference\s+(?:in|between)\b.*\bbetween\b",
                lowered,
            )
        ):
            return False
        if lowered == "done":
            return False
        return len(value) <= 1200

    def _candidate_record_usable_for_convergence(self, record: dict[str, Any]) -> bool:
        """Keep weak placeholders in the ledger without steering root finalization."""
        if not self._candidate_answer_usable(str(record.get("answer") or "")):
            return False
        if (
            str(record.get("source") or "") == "done"
            and (
                not str(record.get("evidence") or "").strip()
                or not str(record.get("method") or "").strip()
            )
        ):
            return False
        try:
            confidence = float(record.get("confidence") or 0)
        except (TypeError, ValueError):
            return False
        return confidence >= _clamp01(self.config.candidate_convergence_min_confidence)

    def _candidate_delivery_records(self, parent: Agent) -> list[dict[str, Any]]:
        usable = [
            record
            for record in self._candidate_deliveries
            if record.get("parent_id") == parent.id
            and self._candidate_record_usable_for_convergence(record)
        ]
        latest_by_agent: dict[str, dict[str, Any]] = {}
        unowned: list[dict[str, Any]] = []
        for record in sorted(usable, key=lambda item: int(item.get("seq") or 0)):
            agent_id = str(record.get("agent_id") or "")
            if not agent_id:
                unowned.append(record)
                continue
            previous = latest_by_agent.get(agent_id)
            source = str(record.get("source") or "")
            if source == "deliver_to_parent":
                latest_by_agent[agent_id] = record
            elif previous is None or str(previous.get("source") or "") != "deliver_to_parent":
                latest_by_agent[agent_id] = record
        return sorted(
            [*unowned, *latest_by_agent.values()],
            key=lambda item: int(item.get("seq") or 0),
        )

    def _formal_candidate_delivery_records(
        self,
        parent: Agent,
    ) -> list[dict[str, Any]]:
        """Return one structurally complete formal delivery per direct child."""
        direct_children = set(parent.children)
        latest_by_agent: dict[str, dict[str, Any]] = {}
        for record in sorted(
            self._candidate_deliveries,
            key=lambda item: int(item.get("seq") or 0),
        ):
            agent_id = str(record.get("agent_id") or "")
            if (
                record.get("parent_id") != parent.id
                or agent_id not in direct_children
                or str(record.get("source") or "") != "deliver_to_parent"
                or not self._candidate_answer_usable(
                    str(record.get("answer") or "")
                )
                or not str(record.get("evidence") or "").strip()
            ):
                continue
            latest_by_agent[agent_id] = record
        return list(latest_by_agent.values())

    @staticmethod
    def _candidate_text_fragments(text: str) -> list[str]:
        """Expose explicit answer text embedded in serialized tool results."""
        value = str(text or "").strip()
        if not value:
            return []
        fragments: list[str] = []
        try:
            payload = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "result", "stdout", "content", "message"):
                item = payload.get(key)
                if isinstance(item, (str, int, float)) and str(item).strip():
                    fragments.append(str(item).strip())
        fragments.append(value)
        return fragments

    @staticmethod
    def _compact_candidate_evidence(text: str, *, limit: int = 6000) -> str:
        """Bound recovered evidence while preserving source context and conclusions."""
        value = str(text or "").strip()
        if limit <= 0 or len(value) <= limit:
            return value
        marker = "\n...[middle omitted by runtime]...\n"
        available = max(2, limit - len(marker))
        head_size = max(1, available // 3)
        tail_size = max(1, available - head_size)
        return value[:head_size].rstrip() + marker + value[-tail_size:].lstrip()

    def _explicit_candidate_from_text(self, text: str) -> tuple[str, str]:
        """Recover only candidates marked as final or encoded as an answer field."""
        if not self.config.child_explicit_candidate_recovery_enabled:
            return "", ""
        from nanoma.meta import _extract_candidate_answer

        for fragment in self._candidate_text_fragments(text):
            if not re.search(
                r"(?is)\bfinal\s+answer\b|[\"']answer[\"']\s*:|(?:^|\n)\s*answer\s*:|"
                r"\bdeliver_to_parent\b[^\n]{0,500}?[\"']?answer[\"']?\s*[:=]|"
                r"\b(?:ball\s+with\s+highest[^\n:]{0,100}|most\s+likely\s+ball|best\s+ball|winner|winning\s+ball)\s*:",
                fragment,
            ):
                continue
            answer = _extract_candidate_answer(fragment)
            if self._candidate_answer_usable(answer):
                return answer, fragment
        return "", ""

    def _harvest_child_candidates(self, parent: Agent) -> int:
        """Recover explicit child answers that were computed but never delivered."""
        if parent.parent is not None or not self.config.child_explicit_candidate_recovery_enabled:
            return 0
        recovered = 0
        existing_agents = {
            str(record.get("agent_id") or "")
            for record in self._candidate_delivery_records(parent)
        }
        for child_id in sorted(parent.children):
            child = self.agents.get(child_id)
            if child is None or child.id in existing_agents:
                continue
            for message in reversed(child.history):
                if message.get("role") not in {"assistant", "tool"}:
                    continue
                answer, evidence = self._explicit_candidate_from_text(str(message.get("content") or ""))
                if not answer:
                    continue
                record = self._record_candidate_delivery(
                    child,
                    parent_id=parent.id,
                    answer=answer,
                    evidence=self._compact_candidate_evidence(evidence),
                    confidence=0.7,
                    method="runtime explicit child-history recovery",
                    source="child_history",
                )
                if record is not None:
                    recovered += 1
                    existing_agents.add(child.id)
                    self._emit(child.id, "candidate_text_recovered", {
                        "parent_id": parent.id,
                        "answer": answer[:300],
                        "source": "child_history",
                    })
                break
        return recovered

    def _ensure_child_delivery_tool(
        self,
        agent: Agent,
        tools: dict[str, dict[str, Any]],
        all_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> tuple[dict[str, dict[str, Any]], ToolPolicyState]:
        """Keep the parent-delivery protocol available after every policy scope."""
        if (
            not self.config.child_delivery_tool_required
            or agent.parent is None
            or "deliver_to_parent" not in all_tools
            or "deliver_to_parent" in tools
            or "runtime_intervention" in tool_policy.reason
        ):
            return tools, tool_policy
        scoped = dict(tools)
        scoped["deliver_to_parent"] = all_tools["deliver_to_parent"]
        tool_policy.removed_tools = [
            name for name in tool_policy.removed_tools if name != "deliver_to_parent"
        ]
        tool_policy.scoped_tools = list(scoped)
        tool_policy.reason = ",".join(filter(None, [tool_policy.reason, "child_delivery_required"]))
        return scoped, tool_policy

    def _candidate_convergence_state(self, agent: Agent) -> dict[str, Any] | None:
        if (
            not self.config.candidate_convergence_enabled
            or agent.parent is not None
            or agent.status != "running"
        ):
            return None
        records = self._candidate_delivery_records(agent)
        if not records:
            return None
        first_turn = self._candidate_convergence_turn_by_root.setdefault(agent.id, agent._turns)
        active_children = [
            child_id for child_id in agent.children
            if child_id in self.agents and self.agents[child_id].status not in {"done", "failed"}
        ]
        required_formal_deliveries = (
            min(2, len(agent.children))
            if self.config.child_delivery_tool_required else 0
        )
        formal_delivery_count = len(
            self._formal_candidate_delivery_records(agent)
        )
        formal_delivery_ready = (
            required_formal_deliveries == 0
            or formal_delivery_count >= required_formal_deliveries
        )
        unique_answers = {
            self._candidate_answer_key(str(record.get("answer") or ""))
            for record in records
        }
        unique_answers.discard("")
        conflict = len(unique_answers) > 1
        confidence_threshold = _clamp01(
            self.config.candidate_convergence_high_confidence_threshold
        )
        high_confidence_ready = bool(
            not conflict
            and formal_delivery_ready
            and confidence_threshold > 0
            and any(
                str(record.get("source") or "") == "deliver_to_parent"
                and float(record.get("confidence") or 0) >= confidence_threshold
                and len(str(record.get("evidence") or "").strip()) >= 40
                and bool(str(record.get("method") or "").strip())
                for record in records
            )
        )
        time_fraction = 0.0
        if agent.quota.time_limit > 0:
            time_fraction = self.effective_elapsed() / agent.quota.time_limit
        turns_since = max(0, agent._turns - first_turn)
        root_reconcile_needed = bool(
            not active_children
            and not high_confidence_ready
            and time_fraction < self.config.candidate_convergence_time_fraction
            and turns_since < max(1, self.config.candidate_convergence_max_root_turns)
        )
        force_final = bool(
            high_confidence_ready
            or time_fraction >= self.config.candidate_convergence_time_fraction
            or turns_since >= max(1, self.config.candidate_convergence_max_root_turns)
            or (not active_children and not root_reconcile_needed)
        )
        if (
            active_children
            and not formal_delivery_ready
            and time_fraction < 0.95
        ):
            force_final = False
        return {
            "records": records,
            "active_children": active_children,
            "conflict": conflict,
            "high_confidence_ready": high_confidence_ready,
            "time_fraction": time_fraction,
            "turns_since": turns_since,
            "root_reconcile_needed": root_reconcile_needed,
            "formal_delivery_count": formal_delivery_count,
            "required_formal_deliveries": required_formal_deliveries,
            "formal_delivery_ready": formal_delivery_ready,
            "force_final": force_final,
        }

    def _maybe_inject_candidate_convergence_notice(self, agent: Agent) -> None:
        state = self._candidate_convergence_state(agent)
        if not state:
            return
        records = state["records"]
        signature = (max(int(record.get("seq") or 0) for record in records), bool(state["force_final"]))
        if self._candidate_convergence_notice_by_root.get(agent.id) == signature:
            return
        self._candidate_convergence_notice_by_root[agent.id] = signature
        rows = []
        for record in records[-8:]:
            evidence_excerpt = self._compact_candidate_evidence(
                str(record.get("evidence") or "not supplied"),
                limit=1200,
            )
            rows.append(
                f"- {record.get('agent_id')}: answer={str(record.get('answer') or '')[:300]!r}; "
                f"confidence={float(record.get('confidence') or 0):.2f}; "
                f"method={str(record.get('method') or 'unspecified')[:100]}; "
                f"evidence={evidence_excerpt}"
            )
        if state["force_final"]:
            action = (
                "This is the finalization window. Choose the best supported answer now, write the required "
                "final file, and call set_status(done, result=<answer-only>). Do not query, wait, spawn, "
                "call shell, or resume web research."
            )
        elif state["active_children"]:
            action = (
                "Stop open-ended research. You may spend at most one coordination turn querying or waiting "
                "for already-active children; otherwise compare the evidence, write the required final file, "
                "and call set_status(done, result=<answer-only>)."
            )
        else:
            action = (
                "All children are finished, but the candidate ledger is conflicting or lacks a formal "
                "high-confidence result. Use one focused root verification turn on the exact disputed facts. "
                "If more than one fact is disputed, inspect all of them in one bounded batch command or API "
                "call instead of spending the turn on one page. You may inspect candidate files or use "
                "shell/direct public sources, but do not restart broad research or spawn more agents. Then "
                "adjudicate the evidence and finalize."
            )
        conflict = (
            "Candidate answers conflict. Adjudicate by evidence and method, not arrival order or role name. "
            if state["conflict"] else
            "A supported child candidate is available. Use it unless the listed evidence contains a concrete contradiction. "
        )
        notice = (
            "[Runtime candidate convergence]\n"
            + conflict
            + action
            + "\nCandidate ledger:\n"
            + "\n".join(rows)
        )
        agent.history.append({"role": "user", "content": notice})
        self._emit(agent.id, "candidate_convergence_notice", {
            "candidate_count": len(records),
            "conflict": state["conflict"],
            "active_children": list(state["active_children"]),
            "high_confidence_ready": state["high_confidence_ready"],
            "time_fraction": round(float(state["time_fraction"]), 3),
            "turns_since": state["turns_since"],
            "root_reconcile_needed": state["root_reconcile_needed"],
            "formal_delivery_count": state["formal_delivery_count"],
            "required_formal_deliveries": state["required_formal_deliveries"],
            "formal_delivery_ready": state["formal_delivery_ready"],
            "force_final": state["force_final"],
        })

    def _apply_candidate_convergence_scope(
        self,
        agent: Agent,
        tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> tuple[dict[str, dict[str, Any]], ToolPolicyState]:
        if self._fixed_spawn_context(agent) is not None:
            return tools, tool_policy
        if "runtime_intervention" in tool_policy.reason:
            return tools, tool_policy
        state = self._candidate_convergence_state(agent)
        if not state:
            return tools, tool_policy
        allowed = {
            "ws_create_file", "ws_append_file", "ws_replace_string", "ws_multi_replace",
            "ws_apply_patch", "ws_read_file", "ws_grep", "submit", "set_status", "get_cost",
        }
        if not state["force_final"] and state["active_children"]:
            allowed.update({"query", "wait"})
        elif not state["force_final"] and state["root_reconcile_needed"]:
            allowed.add("shell")
        scoped = {name: tool for name, tool in tools.items() if name in allowed}
        if not scoped:
            return tools, tool_policy
        removed = set(tools) - set(scoped)
        if state["root_reconcile_needed"] and not state["force_final"]:
            tool_policy.finish = max(tool_policy.finish, 0.75)
            tool_policy.work = max(tool_policy.work, 0.65)
        else:
            tool_policy.finish = max(tool_policy.finish, 1.0)
            tool_policy.work = min(tool_policy.work, 0.25)
        tool_policy.create = 0.0
        tool_policy.removed_tools = sorted(set(tool_policy.removed_tools) | removed)
        tool_policy.scoped_tools = list(scoped)
        suffix = "candidate_convergence_final" if state["force_final"] else "candidate_convergence"
        tool_policy.reason = ",".join(filter(None, [tool_policy.reason, suffix]))
        return scoped, tool_policy

    def _active_child_completion_block_reason(self, agent: Agent) -> str | None:
        if not self._fixed_plan:
            return None
        active_children = [
            child_id
            for child_id in agent.children
            if child_id in self.agents
            and self.agents[child_id].status not in {"done", "failed"}
        ]
        if not active_children:
            agent._notified_thresholds.discard("active_child_finish_guard")
            return None
        confidence_threshold = _clamp01(
            self.config.candidate_convergence_high_confidence_threshold
        )
        required_formal_deliveries = (
            min(2, len(agent.children))
            if self.config.child_delivery_tool_required else 0
        )
        formal_delivery_count = len(
            self._formal_candidate_delivery_records(agent)
        )
        has_formal_high_confidence_candidate = any(
            str(record.get("source") or "") == "deliver_to_parent"
            and float(record.get("confidence") or 0) >= confidence_threshold
            and len(str(record.get("evidence") or "").strip()) >= 40
            for record in self._candidate_delivery_records(agent)
        )
        if (
            has_formal_high_confidence_candidate
            and formal_delivery_count >= required_formal_deliveries
        ):
            return None
        if (
            agent.quota.time_limit > 0
            and self.effective_elapsed() >= agent.quota.time_limit * 0.95
        ):
            return None
        return (
            "active child work is still in flight: "
            + ", ".join(active_children[:8])
            + "; wait for, query, or explicitly kill those children before completing"
        )

    def _apply_active_child_finish_guard(
        self,
        agent: Agent,
        tools: dict[str, dict[str, Any]],
        all_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> tuple[dict[str, dict[str, Any]], ToolPolicyState]:
        """Keep finalization from racing a child that is still producing evidence."""
        if self._fixed_spawn_context(agent) is not None:
            return tools, tool_policy
        block_reason = self._active_child_completion_block_reason(agent)
        if not block_reason:
            return tools, tool_policy
        finish_tools = (
            _DELIVERY_WRITE_TOOLS
            | {"submit", "set_status", "deliver_to_parent"}
        )
        finalizing = bool(
            set(tools).intersection(finish_tools)
            and (
                tool_policy.delivery_phase == "deliver"
                or "candidate_convergence_final" in tool_policy.reason
                or (
                    tool_policy.finish >= self.config.tool_policy_finish_threshold
                    and set(tools).issubset(
                        finish_tools | _DELIVERY_READ_TOOLS | {"get_cost"}
                    )
                )
            )
        )
        coordination_names = {"query", "wait", "send", "kill", "get_cost"}
        coordination_tools = {
            name: all_tools[name]
            for name in coordination_names
            if name in all_tools
        }
        if finalizing:
            scoped = coordination_tools
        else:
            scoped = dict(tools)
            scoped.update(coordination_tools)
        if not scoped:
            return tools, tool_policy
        removed = set(tools) - set(scoped)
        restored = set(scoped) - set(tools)
        if not finalizing and not restored:
            return tools, tool_policy
        if finalizing:
            tool_policy.finish = min(tool_policy.finish, 0.25)
            tool_policy.work = min(tool_policy.work, 0.25)
            tool_policy.message = max(tool_policy.message, 0.9)
        tool_policy.removed_tools = sorted(
            (set(tool_policy.removed_tools) | removed) - restored
        )
        tool_policy.scoped_tools = list(scoped)
        guard_reason = (
            "active_child_finish_guard"
            if finalizing
            else "active_child_coordination_restore"
        )
        tool_policy.reason = ",".join(
            filter(None, [tool_policy.reason, guard_reason])
        )
        if "active_child_finish_guard" not in agent._notified_thresholds:
            agent._notified_thresholds.add("active_child_finish_guard")
            instruction = (
                "Do not write or submit a guessed final answer. Use NanoMA "
                "wait/query for the active child, or kill it explicitly if its "
                "work is no longer needed."
                if finalizing
                else (
                    "NanoMA coordination tools have been restored alongside your "
                    "current research tools. Reconcile the active child's formal "
                    "delivery before completing."
                )
            )
            agent.history.append({
                "role": "user",
                "content": (
                    "[Runtime active-child finish guard]\n"
                    f"{block_reason}. {instruction}"
                ),
            })
            self._emit(agent.id, "active_child_finish_guard", {
                "active_children": [
                    child_id
                    for child_id in agent.children
                    if child_id in self.agents
                    and self.agents[child_id].status not in {"done", "failed"}
                ],
                "removed_tools": sorted(removed),
                "restored_tools": sorted(restored),
                "mode": "finalizing" if finalizing else "preserve_research",
            })
        return scoped, tool_policy

    def _deadline_candidate_answer(self, agent: Agent) -> str:
        if (
            not self.config.candidate_deadline_fallback_enabled
            or agent.parent is not None
        ):
            return ""
        records = self._candidate_delivery_records(agent)
        if not records:
            return ""
        best = max(
            records,
            key=lambda record: (
                float(record.get("confidence") or 0),
                bool(record.get("evidence")),
                int(record.get("seq") or 0),
            ),
        )
        return str(best.get("answer") or "").strip()

    # ─── ReAct loop ──────────────────────────────────────────────────────

    async def _agent_loop(self, agent: Agent):
        from nanoma.meta import META_TOOLS
        # Tool set: shell (universal primitive) + workspace (structured I/O) + meta (coordination)
        all_tools = {
            name: tool
            for name, tool in {
                **WORK_TOOLS,
                **WORKSPACE_TOOLS,
                **META_TOOLS,
                **self.config.extra_tools,
            }.items()
            if name not in self.config.disabled_tools
        }
        all_tools = self._merge_wrap_submit(agent, all_tools)

        try:
            while agent.status == "running":
                if agent.id in self._fixed_completion_waiting:
                    await self._poll_fixed_orchestration_once()
                    if (
                        agent.id in self._fixed_completion_waiting
                        and self.fixed_orchestration_completion_block_reason(agent)
                    ):
                        await asyncio.sleep(self.config.fixed_orchestration_poll_seconds)
                        continue
                    self._fixed_completion_waiting.discard(agent.id)
                if await self._maybe_park_root_for_candidate(agent):
                    continue
                agent._turns += 1
                agent._last_active = time.time()

                if agent.parent is None:
                    self._harvest_child_candidates(agent)

                # Turn limits
                if agent.quota.max_turns > 0 and agent._turns > agent.quota.max_turns:
                    agent.status = "done"
                    agent.result = agent.result or "[Max turns reached]"
                    break

                # Time limit
                if agent.quota.time_limit > 0:
                    elapsed = self.effective_elapsed()
                    if elapsed >= agent.quota.time_limit:
                        agent.status = "done"
                        deadline_answer = self._deadline_candidate_answer(agent)
                        if deadline_answer:
                            agent.result = deadline_answer
                            self._emit(agent.id, "candidate_deadline_fallback", {
                                "answer": deadline_answer[:300],
                                "elapsed_seconds": round(elapsed, 1),
                            })
                        else:
                            agent.result = agent.result or f"[Time limit at {elapsed:.0f}s]"
                        break

                # Budget enforcement — global budget check
                if self.ledger.remaining() <= 0:
                    agent.status = "failed"
                    agent.result = agent.result or "[GLOBAL BUDGET EXHAUSTED]"
                    self._emit(agent.id, "failed", {
                        "reason": "budget_exhausted",
                        "status": "failed",
                        "turns": agent._turns,
                        "result": agent.result,
                    })
                    break

                # Time-based budget drain — DISABLED
                # (Previously drained budget over wall-clock time, but this punishes
                # agents for legitimately waiting on dependencies)

                # Rebirth
                if agent._rebirth_pending:
                    self._execute_rebirth(agent)

                # Resource threshold notifications
                self._check_thresholds(agent)

                # Checkpoints steer the designated parent into a NanoMA spawn
                # tool call; the Runtime never creates child agents directly.
                await self._poll_fixed_orchestration_once()

                # Inject queued messages
                self._inject_messages(agent, agent._queue_inbox)
                self._inject_messages(agent, agent._steer_inbox)
                self._maybe_inject_candidate_convergence_notice(agent)

                # Context compression
                agent.context_tokens = count_message_tokens(agent.history)
                if agent.context_tokens > int(agent.context_limit * self.config.context_compress_ratio):
                    agent.history = await self._compress(agent.history)
                    agent.context_tokens = count_message_tokens(agent.history)

                # Spawn decisions in JUDGE mode are made at the planning moment
                # (inside meta_task_create, before the todolist is materialized),
                # not here — so the todolist is only ever created on no-spawn.
                turn_tools, tool_policy = self._apply_state_tool_policy(agent, all_tools)
                self._maybe_schedule_runtime_strategy(agent, turn_tools, tool_policy)
                await self._maybe_schedule_supervisor_strategy(agent, turn_tools, tool_policy)
                turn_tools, tool_policy = self._apply_intervention_tool_override(
                    agent, turn_tools, tool_policy
                )
                turn_tools, tool_policy = self._apply_candidate_convergence_scope(
                    agent, turn_tools, tool_policy
                )
                turn_tools, tool_policy = self._ensure_child_delivery_tool(
                    agent, turn_tools, all_tools, tool_policy
                )
                turn_tools, tool_policy = self._apply_active_child_finish_guard(
                    agent, turn_tools, all_tools, tool_policy
                )
                turn_tools, tool_policy = self._apply_spawn_todolist_gate(
                    agent, turn_tools, tool_policy
                )
                self._maybe_inject_delivery_finalization_notice(agent, tool_policy, turn_tools)
                self._ensure_tool_call_responses(agent)
                if (
                    self.config.force_spawn_turns > 0
                    and agent.parent is None
                    and not agent.children
                    and agent._turns <= (
                        self.config.force_spawn_turns
                        + max(0, self.config.malformed_tool_repair_turns)
                    )
                    and bool(_CREATE_TOOLS.intersection(all_tools))
                ):
                    create_tools = _CREATE_TOOLS
                    if self.config.force_spawn_many and "spawn_many" in all_tools:
                        create_tools = {"spawn_many"}
                    preferred = create_tools | {"get_cost", "set_status"}
                    forced = {
                        name: tool for name, tool in all_tools.items()
                        if name in preferred
                    }
                    if forced:
                        turn_tools = forced
                        tool_policy.reason = f"{tool_policy.reason},force_spawn_turns"
                        tool_policy.scoped_tools = list(turn_tools)
                        agent.history.append({
                            "role": "user",
                            "content": (
                                "[Probe branch tool override]\n"
                                "This is a root-only runtime instruction for the current turn. Do not copy it "
                                "into any child assignment. "
                                "For this resumed branch, shell and workspace write tools are intentionally unavailable "
                                "until you create at least one child agent. "
                                + (
                                    "Only spawn_many is available, so call spawn_many once with complementary agents, "
                                    "each with complete task context. "
                                    if "spawn_many" in forced and "spawn" not in forced else
                                    "Call spawn now. Prefer spawning separate solver, verifier, and golfer agents, "
                                    "each with complete task context. "
                                )
                                + "Omit the spawn "
                                f"model argument or use {self.config.default_model}; do not request other models. "
                                "Do not call shell."
                            ),
                        })
                tool_schemas = [t["schema"] for t in turn_tools.values()]
                agent._last_tool_policy = tool_policy.as_event()

                call_checkpoint_path: Path | None = None
                if self.config.probe_every_llm or self.config.auto_checkpoint_enabled:
                    call_checkpoint_path = await self._write_auto_checkpoint(
                        agent,
                        turn_tools,
                        tool_policy,
                        reason="manual_probe" if self.config.probe_every_llm else "auto_checkpoint",
                    )
                    if self.config.probe_stop_after > 0 and self._probe_counter >= self.config.probe_stop_after:
                        agent.status = "done"
                        agent.result = f"[Probe stop after {self._probe_counter} probes]"
                        break

                # LLM call
                admission = await self._admit_llm_call(agent, turn_tools)
                candidate_count_before_queue = (
                    len(self._candidate_delivery_records(agent))
                    if agent.parent is None else 0
                )
                queue_started = time.time()
                await self.scheduler.acquire()
                queue_wait_ms = (time.time() - queue_started) * 1000
                if agent.parent is None:
                    self._harvest_child_candidates(agent)
                    candidate_count_after_queue = len(self._candidate_delivery_records(agent))
                    if (
                        self.config.candidate_convergence_enabled
                        and candidate_count_after_queue > candidate_count_before_queue
                    ):
                        self._emit(agent.id, "candidate_arrived_while_llm_queued", {
                            "candidate_count_before": candidate_count_before_queue,
                            "candidate_count_after": candidate_count_after_queue,
                            "queue_wait_ms": round(queue_wait_ms, 3),
                            "stale_turn": agent._turns,
                        })
                        agent._turns = max(0, agent._turns - 1)
                        self.scheduler.release()
                        continue
                elapsed_before_llm = self.effective_elapsed()
                if (
                    agent.quota.time_limit > 0
                    and elapsed_before_llm >= agent.quota.time_limit
                ):
                    agent.status = "done"
                    deadline_answer = self._deadline_candidate_answer(agent)
                    agent.result = deadline_answer or agent.result or (
                        f"[Time limit at {elapsed_before_llm:.0f}s]"
                    )
                    self._emit(agent.id, "llm_deadline_skipped", {
                        "answer": deadline_answer[:300],
                        "elapsed_seconds": round(elapsed_before_llm, 1),
                        "queue_wait_ms": round(queue_wait_ms, 3),
                    })
                    self.scheduler.release()
                    break
                llm_started = time.time()
                turn_max_tokens = self._llm_max_tokens_for_turn(agent, turn_tools)
                self._emit(agent.id, "llm_start", {
                    "model": agent.model,
                    "turn": agent._turns,
                    "context_tokens": agent.context_tokens,
                    "message_count": len(agent.history),
                    "tool_count": len(turn_tools),
                    "max_tokens": turn_max_tokens,
                    "queue_wait_ms": round(queue_wait_ms, 3),
                    "scheduler": dict(self.scheduler.stats),
                    "admission": admission,
                })
                try:
                    try:
                        llm_kwargs: dict[str, Any] = {
                            "retry_config": self.config.retry,
                        }
                        if turn_max_tokens is not None:
                            llm_kwargs["max_tokens"] = turn_max_tokens
                        llm_awaitable = self.llm_call(
                            self._history_with_todo_reminder(agent), agent.model, tool_schemas,
                            **llm_kwargs,
                        )
                        if agent.quota.time_limit > 0:
                            remaining = max(
                                0.001,
                                agent.quota.time_limit - self.effective_elapsed(),
                            )
                            response = await self._await_llm_response(
                                agent, llm_awaitable, remaining
                            )
                        else:
                            response = await self._await_llm_response(
                                agent, llm_awaitable, None
                            )
                        self._infra_failure_streak_by_agent.pop(agent.id, None)
                        self._infra_failure_state_by_agent.pop(agent.id, None)
                    except _CandidateDeliveryInterrupt:
                        self._harvest_child_candidates(agent)
                        self._emit(agent.id, "llm_candidate_interrupted", {
                            "candidate_count": len(self._candidate_delivery_records(agent)),
                            "elapsed_seconds": round(self.effective_elapsed(), 1),
                            "llm_elapsed_ms": round((time.time() - llm_started) * 1000, 3),
                        })
                        continue
                    except _RuntimeToolOverrideInterrupt:
                        override = self._pending_tool_overrides.get(agent.id) or {}
                        self._emit(agent.id, "llm_tool_override_interrupted", {
                            "source": override.get("source"),
                            "strategy_action": override.get("strategy_action"),
                            "tools": list(override.get("tools") or []),
                            "elapsed_seconds": round(self.effective_elapsed(), 1),
                            "llm_elapsed_ms": round(
                                (time.time() - llm_started) * 1000, 3
                            ),
                        })
                        continue
                    except asyncio.TimeoutError:
                        elapsed = self.effective_elapsed()
                        if agent.parent is None:
                            self._harvest_child_candidates(agent)
                        deadline_answer = self._deadline_candidate_answer(agent)
                        agent.status = "done"
                        agent.result = deadline_answer or agent.result or f"[Time limit at {elapsed:.0f}s]"
                        self._emit(agent.id, "llm_deadline_cancelled", {
                            "answer": deadline_answer[:300],
                            "elapsed_seconds": round(elapsed, 1),
                            "llm_elapsed_ms": round((time.time() - llm_started) * 1000, 3),
                        })
                        break
                    except Exception as exc:
                        llm_elapsed_ms = (time.time() - llm_started) * 1000
                        self._note_llm_failure_for_admission(exc)
                        self._emit(agent.id, "llm_error", {
                            "model": agent.model,
                            "turn": agent._turns,
                            "context_tokens": agent.context_tokens,
                            "message_count": len(agent.history),
                            "tool_count": len(turn_tools),
                            "elapsed_ms": round(llm_elapsed_ms, 3),
                            "status_code": self._llm_exception_status_code(exc),
                            "exception": type(exc).__name__,
                            "detail": str(exc)[:1000],
                            "scheduler": dict(self.scheduler.stats),
                            "admission": admission,
                        })
                        if (
                            self.config.llm_400_compact_retry_enabled
                            and self._llm_exception_status_code(exc) == 400
                            and agent._http400_recoveries
                            < self.config.llm_400_compact_max_retries_per_agent
                        ):
                            before_messages = len(agent.history)
                            before_tokens = agent.context_tokens or count_message_tokens(agent.history)
                            agent._http400_recoveries += 1
                            repaired_before = self._ensure_tool_call_responses(agent)
                            agent.history = await self._compress(
                                agent.history,
                                keep_recent=self.config.llm_400_compact_keep_recent,
                            )
                            repaired_after = self._ensure_tool_call_responses(agent)
                            agent.history.append({
                                "role": "user",
                                "content": (
                                    "[Runtime note]\n"
                                    "The previous LLM request was rejected by the provider with HTTP 400. "
                                    "Older conversation history was compacted while preserving recent tool "
                                    "results. Continue from the summary and recent state; avoid replaying "
                                    "already completed work unless it is necessary."
                                ),
                            })
                            agent.context_tokens = count_message_tokens(agent.history)
                            self._emit(agent.id, "llm_400_context_compacted", {
                                "model": agent.model,
                                "turn": agent._turns,
                                "attempt": agent._http400_recoveries,
                                "before_messages": before_messages,
                                "after_messages": len(agent.history),
                                "before_context_tokens": before_tokens,
                                "after_context_tokens": agent.context_tokens,
                                "keep_recent": self.config.llm_400_compact_keep_recent,
                                "repaired_before": repaired_before,
                                "repaired_after": repaired_after,
                            })
                            continue
                        if (
                            self.config.auto_rollback_on_infra_failure
                            and self._is_infra_failure_exception(exc)
                        ):
                            self._infra_failure_streak_by_agent[agent.id] = (
                                self._infra_failure_streak_by_agent.get(agent.id, 0) + 1
                            )
                            if call_checkpoint_path is not None:
                                call_record = next(
                                    (
                                        record for record in reversed(self._global_checkpoints)
                                        if record.get("path") == str(call_checkpoint_path)
                                    ),
                                    None,
                                )
                                if call_record is not None:
                                    state = self._infra_failure_state_by_agent.get(agent.id) or {}
                                    self._infra_failure_state_by_agent[agent.id] = {
                                        **state,
                                        "failure_seq": int(call_record.get("seq") or 0),
                                        "rollback_before_seq": int(call_record.get("seq") or 0) + 1,
                                    }
                            preferred = call_checkpoint_path
                            if self._infra_failure_streak_by_agent[agent.id] > 1:
                                preferred = None
                            if self._request_rollback(
                                agent,
                                reason="infra_failure",
                                detail=f"{type(exc).__name__}: {exc}",
                                checkpoint_path=preferred,
                            ):
                                return
                        raise
                finally:
                    self.scheduler.release()
                llm_elapsed_ms = (time.time() - llm_started) * 1000

                # Record usage
                cost = self.ledger.record(agent.id, response.usage)
                agent.tokens_consumed += response.usage.total_tokens
                agent.quota.budget -= cost
                llm_event = {
                    "tokens": response.usage.total_tokens,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cached": response.usage.cached_input_tokens,
                    "cost": round(cost, 6),
                    "model": agent.model,
                    "tool_calls": [tc.name for tc in response.tool_calls] if response.tool_calls else [],
                    "has_content": bool(response.content),
                    "content_preview": (response.content or "")[:200],
                    "elapsed_ms": round(llm_elapsed_ms, 3),
                    "queue_wait_ms": round(queue_wait_ms, 3),
                    "scheduler": dict(self.scheduler.stats),
                    "admission": admission,
                }
                if self.config.tool_policy_log_events:
                    llm_event["tool_policy"] = tool_policy.as_event()
                self._emit(agent.id, "llm_done", llm_event)

                if (
                    self.config.force_spawn_auto_recover_no_tool
                    and self.config.force_spawn_many
                    and self.config.force_spawn_turns > 0
                    and agent.parent is None
                    and not agent.children
                    and agent._turns <= (
                        self.config.force_spawn_turns
                        + max(0, self.config.malformed_tool_repair_turns)
                    )
                    and "spawn_many" in turn_tools
                    and not response.tool_calls
                    and bool((response.content or "").strip())
                    and (
                        int(getattr(response.usage, "output_tokens", 0) or 0)
                        >= self.config.force_spawn_auto_recover_output_token_threshold
                        or self._looks_like_malformed_tool_intent(response.content or "")
                    )
                ):
                    parent_task = agent.task.strip()
                    direct_task = (
                        "Direct solver assignment. Solve the complete parent task below independently and "
                        "compactly using the available tools. For an attachment, inspect the attachment "
                        "directly and compute the requested answer. Call deliver_to_parent with an "
                        "answer-only candidate, decisive evidence, confidence, and method as soon as the "
                        "answer is supported.\n\nComplete parent task:\n"
                        + parent_task
                    )
                    verifier_task = (
                        "Independent verifier assignment. Solve the same complete parent task below using a "
                        "materially different method. Check the decisive observations and arithmetic rather "
                        "than repeating the direct solver's workflow. Call deliver_to_parent with an "
                        "answer-only candidate, decisive evidence, confidence, and method as soon as the "
                        "answer is supported.\n\nComplete parent task:\n"
                        + parent_task
                    )
                    response.tool_calls = [ToolCall(
                        id=f"runtime-auto-spawn-{agent.id}-{agent._turns}",
                        name="spawn_many",
                        arguments={"agents": [
                            {"task": direct_task},
                            {"task": verifier_task},
                        ]},
                    )]
                    self._emit(agent.id, "root_spawn_auto_recovered", {
                        "turn": agent._turns,
                        "output_tokens": int(getattr(response.usage, "output_tokens", 0) or 0),
                        "requested_children": 2,
                        "preview": (response.content or "")[:300],
                    })

                response_tool_names = [tc.name for tc in response.tool_calls] if response.tool_calls else []
                pending_override = self._pending_tool_overrides.get(agent.id) or {}
                pending_override_tools = set(
                    pending_override.get("last_scoped_tools")
                    or pending_override.get("tools")
                    or []
                )
                delivery_only_turn = set(turn_tools) == {"deliver_to_parent"}
                delivery_override_required = bool(
                    agent.parent
                    and not response.tool_calls
                    and response.content
                    and (
                        "deliver_to_parent" in pending_override_tools
                        or delivery_only_turn
                    )
                )
                malformed_tool_intent = bool(
                    not response.tool_calls
                    and response.content
                    and (
                        delivery_override_required
                        or self._looks_like_malformed_tool_intent(response.content)
                    )
                )
                if (
                    agent.parent
                    and not response.tool_calls
                    and response.content
                    and not delivery_override_required
                ):
                    recovered_answer, recovered_evidence = self._explicit_candidate_from_text(
                        response.content
                    )
                    if recovered_answer:
                        self._record_candidate_delivery(
                            agent,
                            parent_id=agent.parent,
                            answer=recovered_answer,
                            evidence=self._compact_candidate_evidence(recovered_evidence),
                            confidence=0.7,
                            method="runtime explicit child-response recovery",
                            source="child_response",
                        )
                        agent.history.append({"role": "assistant", "content": response.content})
                        agent.status = "done"
                        agent.result = recovered_answer
                        self._emit(agent.id, "candidate_text_recovered", {
                            "parent_id": agent.parent,
                            "answer": recovered_answer[:300],
                            "source": "child_response",
                            "malformed_tool_intent": malformed_tool_intent,
                        })
                        self._finalize_consumed_tool_override(agent, response_tool_names)
                        break
                if (
                    malformed_tool_intent
                ):
                    if self._looks_like_malformed_write_intent(response.content):
                        agent._notified_thresholds.add(_MALFORMED_WRITE_PENDING)
                        self._emit(agent.id, "malformed_write_pending", {
                            "turn": agent._turns,
                            "preview": response.content[:300],
                        })
                    agent._malformed_tool_turns += 1
                    self._emit(agent.id, "malformed_tool_output", {
                        "count": agent._malformed_tool_turns,
                        "preview": response.content[:300],
                    })
                    repair_turns = max(0, int(self.config.malformed_tool_repair_turns))
                    fail_after = max(repair_turns + 1, int(self.config.malformed_tool_fail_after))
                    if agent._malformed_tool_turns >= fail_after:
                        agent.history.append({"role": "assistant", "content": response.content})
                        agent.status = "failed"
                        agent.result = (
                            "repeated malformed tool output: model kept emitting tool-call-looking "
                            "text without executable tool_calls"
                        )
                        self._emit(agent.id, "runtime_tool_repair_failed", {
                            "count": agent._malformed_tool_turns,
                            "fail_after": fail_after,
                            "preview": response.content[:300],
                        })
                        self._finalize_consumed_tool_override(agent, response_tool_names)
                        break
                    else:
                        repair_message = self._malformed_tool_repair_message(
                            agent,
                            turn_tools,
                            response.content,
                        )
                        if agent._malformed_tool_turns > repair_turns:
                            repair_message += (
                                "\n\nThis is a repeated malformed tool-call loop. If you cannot emit "
                                "a real tool call immediately, call set_status(status=\"done\", "
                                "result=\"unable to emit required tool call\") instead of repeating "
                                "bare tags such as <shell>."
                            )
                        agent.history.append({"role": "assistant", "content": response.content})
                        agent.history.append({"role": "user", "content": repair_message})
                        agent._no_tool_turns += 1
                        self._emit(agent.id, "runtime_tool_repair", {
                            "count": agent._malformed_tool_turns,
                            "tools": list(turn_tools),
                            "message": repair_message[:300],
                        })
                        if not self._preserve_consumed_tool_override_for_repair(agent):
                            self._finalize_consumed_tool_override(agent, response_tool_names)
                        continue

                stop_after_response = False
                total_tokens_after_response = 0
                if self.config.max_total_tokens > 0:
                    total_tokens_after_response = sum(a.tokens_consumed for a in self.agents.values())
                    stop_after_response = total_tokens_after_response >= self.config.max_total_tokens

                # Process response
                if response.tool_calls:
                    agent._no_tool_turns = 0
                    agent._malformed_tool_turns = 0
                    responded_tool_call_ids: set[str] = set()
                    # Append assistant message with tool calls
                    agent.history.append({
                        "role": "assistant", "content": response.content,
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                            for tc in response.tool_calls
                        ],
                    })
                    # Execute tools
                    for i, tc in enumerate(response.tool_calls):
                        # Immediate interrupt check
                        if not agent._immediate_inbox.empty():
                            for skipped in response.tool_calls[i:]:
                                agent.history.append({
                                    "role": "tool", "tool_call_id": skipped.id,
                                    "content": json.dumps({"interrupted": True}),
                                })
                                responded_tool_call_ids.add(skipped.id)
                            self._inject_messages(agent, agent._immediate_inbox)
                            break

                        result = await self._execute_tool(tc, agent, turn_tools)
                        agent._tool_calls += 1
                        result_json = json.dumps(result, ensure_ascii=False, default=str)
                        agent.history.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": result_json[:8000],
                        })
                        responded_tool_call_ids.add(tc.id)
                        # Full tool event for viewer (no truncation on args, reasonable on result)
                        event_data = {
                            "tool": tc.name,
                            "args": tc.arguments,
                            "result": result_json[:5000],
                            "result_full_len": len(result_json),
                        }
                        event_data.update(self._tool_call_metadata(tc))
                        self._emit(agent.id, "tool_call", event_data)
                        candidate_failure = self._candidate_failure_reason(agent, tc, result)
                        if candidate_failure and self._request_rollback(
                            agent,
                            reason="candidate_failure",
                            detail=candidate_failure,
                        ):
                            break
                        if agent.status != "running":
                            break

                    for skipped in response.tool_calls:
                        if skipped.id in responded_tool_call_ids:
                            continue
                        agent.history.append({
                            "role": "tool",
                            "tool_call_id": skipped.id,
                            "content": json.dumps({
                                "skipped": True,
                                "reason": (
                                    "rollback_requested"
                                    if self._rollback_requested
                                    else f"agent_status_{agent.status}"
                                ),
                            }),
                        })

                    if self._rollback_requested:
                        return
                    # Steer messages before next LLM call
                    self._inject_messages(agent, agent._steer_inbox)

                elif response.content:
                    agent._no_tool_turns += 1
                    agent.history.append({"role": "assistant", "content": response.content})
                    if agent.parent and self.config.child_action_repair_enabled:
                        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
                        repair_threshold = max(
                            1,
                            int(self.config.child_action_repair_output_token_threshold),
                        )
                        if turn_max_tokens is not None:
                            repair_threshold = min(
                                repair_threshold,
                                max(1, int(turn_max_tokens)),
                            )
                        output_limit_repair = (
                            output_tokens
                            >= repair_threshold
                        )
                        action_message = (
                            "[Runtime child action required]\n"
                            "Your last response made no executable tool call and delivered no explicit final "
                            "candidate. On the next response, emit exactly one available tool call. Keep any "
                            "reasoning before it under 40 words. Use shell or a workspace tool to perform the "
                            "assigned work; if the answer is already supported, call deliver_to_parent now."
                        )
                        if output_limit_repair:
                            agent._output_limit_no_tool_turns += 1
                            action_message += (
                                " Your previous response likely hit the model output limit. Keep the entire next "
                                "response under 800 tokens, and keep any shell command compact: omit comments, "
                                "plots, repeated diagnostics, and explanatory code."
                            )
                        override_after = max(
                            0,
                            int(self.config.child_action_delivery_override_after_output_limits),
                        )
                        delivery_override_scheduled = False
                        if (
                            output_limit_repair
                            and override_after > 0
                            and agent._output_limit_no_tool_turns >= override_after
                            and "deliver_to_parent" in self._all_tools()
                            and agent.id not in self._pending_tool_overrides
                        ):
                            delivery_override_scheduled = True
                            action_message += (
                                " This branch has now exhausted the output limit repeatedly. Stop research and "
                                "explanation. The next turn is restricted to deliver_to_parent; call it as the "
                                "first and only action with the best supported answer and concise evidence "
                                "already present in history."
                            )
                            self._pending_tool_overrides[agent.id] = {
                                "source": "runtime_child_action_repair",
                                "action": "tool_override",
                                "strategy_action": "CHILD_DELIVERY_RECOVERY",
                                "tools": ["deliver_to_parent"],
                                "reason": "repeated output-limit prose without child delivery",
                                "message": (
                                    "Repeated output-limit prose prevented delivery. Do not explain or research "
                                    "further. Call deliver_to_parent now as the first and only action, using the "
                                    "best supported answer and concise evidence already in history."
                                ),
                                "once": False,
                                "created_at": time.time(),
                                "wrong_tool_attempts": 0,
                            }
                            self._emit(agent.id, "child_delivery_override_scheduled", {
                                "output_limit_no_tool_turns": agent._output_limit_no_tool_turns,
                                "tools": ["deliver_to_parent"],
                            })
                        agent.history.append({"role": "user", "content": action_message})
                        self._emit(agent.id, "child_action_required", {
                            "no_tool_turns": agent._no_tool_turns,
                            "available_tools": list(turn_tools),
                            "output_tokens": output_tokens,
                            "output_limit_threshold": repair_threshold,
                            "output_limit_repair": output_limit_repair,
                            "output_limit_no_tool_turns": agent._output_limit_no_tool_turns,
                            "delivery_override_scheduled": delivery_override_scheduled,
                        })
                    elif (
                        agent.parent is None
                        and self.config.force_spawn_turns > 0
                        and not agent.children
                        and agent._turns < (
                            self.config.force_spawn_turns
                            + max(0, self.config.malformed_tool_repair_turns)
                        )
                    ):
                        action_message = (
                            "[Runtime spawn action required]\n"
                            "Your last response did not execute a tool call. On the next response, do not "
                            "explain or research. Emit exactly one real spawn_many tool call containing the "
                            "requested complementary child assignments."
                        )
                        agent.history.append({"role": "user", "content": action_message})
                        self._emit(agent.id, "root_spawn_action_required", {
                            "no_tool_turns": agent._no_tool_turns,
                            "available_tools": list(turn_tools),
                        })
                else:
                    agent._no_tool_turns += 1
                    agent.history.append({"role": "assistant", "content": "(empty)"})

                self._finalize_consumed_tool_override(agent, response_tool_names)
                if agent.status != "running":
                    break
                if stop_after_response:
                    agent.status = "done"
                    agent.result = agent.result or f"[Max total tokens reached: {total_tokens_after_response}]"
                    for other in self.agents.values():
                        if other.status in {"running", "idle"}:
                            other.status = "done"
                            other.result = other.result or agent.result
                    self._emit(agent.id, "done", {
                        "status": "done",
                        "reason": "max_total_tokens",
                        "tokens": total_tokens_after_response,
                        "max_total_tokens": self.config.max_total_tokens,
                        "result": agent.result,
                    })
                    break
                if not self.ledger.can_afford(0):
                    agent.status = "failed"
                    agent.result = "Budget exhausted"
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Agent {agent.id} crashed: {e}")
            if (
                self.config.auto_rollback_on_infra_failure
                and self._is_infra_failure_exception(e)
                and self._request_rollback(
                    agent,
                    reason="infra_failure",
                    detail=f"{type(e).__name__}: {e}",
                )
            ):
                return
            agent.status = "failed"
            agent.result = f"Crash: {e}"
        finally:
            # Fold a finishing child's work back in. "killed" counts: the parent
            # routinely kills children right after they deliver, and their work
            # would otherwise be stranded in the copy. The eval gate still
            # rejects anything that regresses the score.
            if (
                getattr(agent, "_merge_copy", None)
                and agent.status in ("done", "killed")
            ):
                self._schedule_child_delivery(agent)
            # Only notify parent and emit done event if actually finished (not just idle)
            if agent.status in ("done", "failed"):
                if agent.parent and agent.parent in self.agents:
                    if self._candidate_answer_usable(agent.result or ""):
                        self._record_candidate_delivery(
                            agent,
                            parent_id=agent.parent,
                            answer=agent.result or "",
                            confidence=0.5,
                            method="child done result",
                            source="done",
                        )
                    result_preview = (agent.result or "")[:200]
                    death_msg = f"[Agent {agent.id} finished: {agent.status}] {result_preview}"
                    # What else is coming, so the first arrival is not mistaken for
                    # the whole set. The parent would otherwise have to spend a turn
                    # querying its siblings to find out — and usually does not.
                    parent_agent = self.agents.get(agent.parent)
                    if parent_agent is not None:
                        death_msg += self._outstanding_note(parent_agent)
                    msg_tokens = estimate_tokens(death_msg)
                    await self.deliver(Envelope(
                        from_id="system", to_id=agent.parent,
                        content=death_msg,
                        tokens=msg_tokens, timestamp=time.time(), mode="steer",
                    ))
                    self._emit(agent.parent, "send", {
                        "from": "system", "to": agent.parent,
                        "mode": "steer", "tokens": msg_tokens,
                        "message": death_msg,
                    })
                self._emit(agent.id, agent.status, {
                    "status": agent.status, "turns": agent._turns,
                    "tokens": agent.tokens_consumed, "result": (agent.result or "")[:500],
                    "artifacts": [a.path for a in agent.artifacts],
                })

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _check_thresholds(self, agent: Agent):
        """Inject resource notifications when consumption crosses configured thresholds.
        Delivered as system→agent messages via the normal send path."""
        thresholds = self.config.notify_thresholds
        if not thresholds:
            return

        elapsed = self.effective_elapsed()
        alerts = []

        # Time consumed
        if agent.quota.time_limit > 0:
            time_frac = elapsed / agent.quota.time_limit
            for t in thresholds:
                key = f"time_{t}"
                if time_frac >= t and key not in agent._notified_thresholds:
                    agent._notified_thresholds.add(key)
                    remaining = max(0, agent.quota.time_limit - elapsed)
                    alerts.append(f"⏱️ TIME {int(t*100)}% used — {remaining:.0f}s remaining")

        # Turns consumed
        if agent.quota.max_turns > 0:
            turn_frac = agent._turns / agent.quota.max_turns
            for t in thresholds:
                key = f"turns_{t}"
                if turn_frac >= t and key not in agent._notified_thresholds:
                    agent._notified_thresholds.add(key)
                    remaining = agent.quota.max_turns - agent._turns
                    alerts.append(f"🔄 TURNS {int(t*100)}% used — {remaining} turns remaining")

        # Budget consumed (global ledger)
        total_budget = self.ledger.total_budget
        if total_budget > 0:
            spent_frac = self.ledger.total_spent / total_budget
            spent_frac = max(0.0, min(1.0, spent_frac))
            for t in thresholds:
                key = f"budget_{t}"
                if spent_frac >= t and key not in agent._notified_thresholds:
                    agent._notified_thresholds.add(key)
                    remaining = self.ledger.remaining()
                    alerts.append(f"BUDGET {int(t*100)}% used — ${remaining:.4f} remaining of ${total_budget:.4f}")

        # Context window consumed
        if agent.context_limit > 0 and agent.context_tokens > 0:
            ctx_frac = agent.context_tokens / agent.context_limit
            for t in thresholds:
                key = f"context_{t}"
                if ctx_frac >= t and key not in agent._notified_thresholds:
                    agent._notified_thresholds.add(key)
                    remaining_pct = int((1 - ctx_frac) * 100)
                    alerts.append(f"🧠 CONTEXT {int(t*100)}% full — {remaining_pct}% capacity left. Consider rebirth().")

        # Deliver as system message via normal envelope path
        if alerts:
            notice = "[Resource Alert]\n" + "\n".join(alerts)
            envelope = Envelope(
                from_id="system", to_id=agent.id, content=notice,
                tokens=estimate_tokens(notice), timestamp=time.time(),
                mode="steer",
            )
            agent._steer_inbox.put_nowait(envelope)
            self._emit(agent.id, "send", {
                "from": "system", "to": agent.id,
                "mode": "steer", "tokens": envelope.tokens,
                "message": notice,
            })

    def _inject_messages(self, agent: Agent, queue: asyncio.Queue[Envelope]) -> list[Envelope]:
        msgs = []
        while not queue.empty():
            try:
                msgs.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        msgs.sort(key=lambda m: m.priority, reverse=True)
        for msg in msgs:
            agent.history.append({"role": "user", "content": f"[Message from {msg.from_id}]: {msg.content}"})
        return msgs

    def _looks_like_malformed_tool_intent(self, content: str) -> bool:
        """Detect model text that attempted a tool call but yielded no parsed call."""
        text = content or ""
        lowered = text.lower()
        if (
            "<tool_call" in lowered
            or "</tool_call>" in lowered
            or "<tool_use" in lowered
            or "</tool_use>" in lowered
            or "<invoke" in lowered
            or "</invoke>" in lowered
            or "<tool name=" in lowered
            or "</tool>" in lowered
            or "<function_calls" in lowered
            or "</function_calls>" in lowered
            or re.search(r"</?(?:shell|tool_code)\s*>", lowered)
        ):
            return True
        if re.search(r"['\"](?:tool|tool_calls|function_call)['\"]\s*:", text):
            return True
        if re.search(
            r"\b(?:spawn|spawn_many|shell|bash|command_shell|ws_file_read|ws_read_file|set_status|submit)\s*\(",
            text,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:task\s+(?:is\s+)?complete|task\s+complete|completed|answer\s+(?:has\s+been\s+)?written|files?\s+(?:have\s+been\s+)?written|done)\b",
            lowered,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"(?im)^\s*(?:shell|bash|sh|python|tool_code)\s*$", text):
            return True
        if re.search(r"```(?:json|tools?|function_call|bash|sh|shell|zsh)\s*\n", text, flags=re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _looks_like_malformed_write_intent(content: str) -> bool:
        """Recognize a failed artifact write that must be retried from the beginning."""
        value = str(content or "")
        return bool(re.search(
            r"\b(?:ws_create_file|ws_append_file|tb_write_file)\b",
            value,
            flags=re.IGNORECASE,
        ))

    def _malformed_tool_repair_message(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        content: str,
    ) -> str:
        available = sorted(turn_tools)
        examples: list[str] = []
        if "shell" in turn_tools:
            examples.append(
                'shell as exactly one fenced bash block with a real command, for example:\n'
                '```bash\n'
                'find "$WORKSPACE" "$SHARED" -maxdepth 4 -type f\n'
                '```'
            )
        if "spawn_many" in turn_tools:
            examples.append(
                'spawn_many as JSON, for example:\n'
                '{"tool":"spawn_many","args":{"agents":[{"task":"research partition A"},'
                '{"task":"research partition B"}]}}'
            )
        elif "spawn" in turn_tools:
            examples.append(
                'spawn as JSON, for example:\n'
                '{"tool":"spawn","args":{"task":"research this exact subproblem"}}'
            )
        if "ws_read_file" in turn_tools:
            examples.append(
                'ws_read_file as JSON, for example:\n'
                '{"tool":"ws_read_file","args":{"path":"relative/path.txt"}}'
            )
        if "set_status" in turn_tools:
            examples.append(
                'set_status as JSON when finished, for example:\n'
                '{"tool":"set_status","args":{"status":"done","result":"final answer"}}'
            )

        image_hint = ""
        if re.search(r"\b(?:image|attachment|picture|file)\b", content, flags=re.IGNORECASE) and "shell" in turn_tools:
            image_hint = (
                "\nIf you need to inspect an image or attachment, use the path already shown in the task. "
                "If unsure, first list local files with the shell example above."
            )

        examples_text = "\n\n".join(examples) if examples else "Use one available tool from the current schema."
        return (
            "[Runtime malformed tool-call repair]\n"
            "Your previous response looked like a tool call, but the runtime received zero executable tool_calls. "
            "No tool ran and no file content was changed. If that response attempted a create or write, assume "
            "the entire intended content is absent. Do not append a continuation or trust a stale target file; "
            "create or rewrite the complete artifact from its beginning. "
            "Do not repeat malformed <tool_call> tags, placeholder words such as `shell`, or prose like "
            "`I will call the tool`.\n"
            f"Available tools now: {available}\n"
            "On the next response, emit exactly one executable tool request and no explanatory prose.\n\n"
            f"{examples_text}"
            "\n\nIf this endpoint still cannot emit an executable tool call, use this text fallback exactly:\n"
            "FINAL ANSWER: <answer-only>\n"
            "EVIDENCE: <brief decisive evidence>\n"
            "The runtime can recover that explicit final-answer form and deliver it safely."
            f"{image_hint}"
        )

    def _path_candidates_for_agent(self, agent: Agent, raw_path: str) -> list[Path]:
        path_text = raw_path.strip().strip("'\".,);:")
        path_text = path_text.replace("$SHARED", self.config.shared_dir)
        path = Path(path_text)
        if path.is_absolute():
            return [path]

        candidates = [agent.workspace / path, self.config.workspace_root / path]
        if path.parts and path.parts[0] == self.config.shared_dir:
            candidates.append(self.config.workspace_root / path)
        else:
            candidates.append(self._tool_context.shared_dir / path)
        return candidates

    @staticmethod
    def _malformed_write_recovery_block_reason(agent: Agent, tc: ToolCall) -> str:
        if _MALFORMED_WRITE_PENDING not in agent._notified_thresholds:
            return ""
        if tc.name == "submit" or (
            tc.name == "set_status"
            and str(tc.arguments.get("status", "done")) == "done"
        ):
            return (
                "A previous complete artifact write was malformed, so no prefix was written. "
                "Rewrite the full artifact from the beginning before finishing."
            )
        if tc.name in {
            "ws_append_file", "ws_replace_string", "ws_multi_replace", "ws_apply_patch",
        }:
            return (
                "Incremental editing is blocked because the preceding complete write never ran. "
                "Use ws_create_file or tb_write_file to write the full artifact from the beginning."
            )
        if tc.name in {"shell", "tb_shell"} and re.search(
            r"(?<!>)>>(?!>)\s*[^\s;&|]+",
            str(tc.arguments.get("command", "")),
        ):
            return (
                "Shell append is blocked because the preceding complete write never ran. "
                "Use a full overwrite and include the artifact from its first line."
            )
        return ""

    @staticmethod
    def _malformed_write_recovery_completed(agent: Agent, tc: ToolCall, result: Any) -> bool:
        if _MALFORMED_WRITE_PENDING not in agent._notified_thresholds:
            return False
        if isinstance(result, dict) and result.get("error"):
            return False
        complete_write = tc.name in {"ws_create_file", "tb_write_file"}
        if tc.name in {"shell", "tb_shell"}:
            command = str(tc.arguments.get("command", ""))
            complete_write = bool(
                re.search(
                    r"(?<![>\d])>(?![>=])\s*(['\"]?)[A-Za-z0-9_./$-]+\.[A-Za-z0-9]+\1",
                    command,
                )
                or re.search(r"\bopen\s*\([^,\n]+,\s*['\"]w(?:t)?['\"]", command)
                or ".write_text(" in command
            )
        if complete_write:
            agent._notified_thresholds.discard(_MALFORMED_WRITE_PENDING)
        return complete_write

    def _expand_shell_runtime_paths(self, agent: Agent, command: str) -> str:
        replacements = {
            "SHARED": str(self._tool_context.shared_dir),
            "WORKSPACE": str(agent.workspace),
        }
        expanded = command
        for name, value in replacements.items():
            expanded = re.sub(
                rf"\$(?:\{{{name}\}}|{name})(?![A-Za-z0-9_])",
                lambda _match, replacement=value: replacement,
                expanded,
            )
        return expanded

    def _missing_expected_outputs(self, agent: Agent) -> list[str]:
        expected = _extract_expected_output_paths(agent.task)
        missing: list[str] = []
        artifact_paths = {a.path for a in agent.artifacts}
        artifact_names = {Path(a.path).name for a in agent.artifacts}
        for raw_path in expected:
            if raw_path in artifact_paths or Path(raw_path).name in artifact_names:
                continue
            if not any(path.exists() for path in self._path_candidates_for_agent(agent, raw_path)):
                missing.append(raw_path)
        return missing

    def _resource_pressure(self, agent: Agent) -> float:
        pressures: list[float] = []
        if self.ledger.total_budget > 0:
            pressures.append(_clamp01(self.ledger.total_spent / self.ledger.total_budget))
        if agent.quota.time_limit > 0:
            pressures.append(_clamp01(self.effective_elapsed() / agent.quota.time_limit))
        if agent.quota.max_turns > 0:
            pressures.append(_clamp01(agent._turns / agent.quota.max_turns))
        if self.config.max_total_tokens > 0:
            total_tokens = sum(a.tokens_consumed for a in self.agents.values())
            pressures.append(_clamp01(total_tokens / self.config.max_total_tokens))
        if self.config.tool_policy_soft_total_tokens > 0:
            total_tokens = sum(a.tokens_consumed for a in self.agents.values())
            pressures.append(_clamp01(total_tokens / self.config.tool_policy_soft_total_tokens))
        if agent.context_limit > 0 and agent.context_tokens > 0:
            pressures.append(_clamp01(agent.context_tokens / agent.context_limit))
        return max(pressures, default=0.0)

    def _hard_resource_pressure(self, agent: Agent) -> float:
        """Resource pressure from explicit run limits, excluding context size.

        Context pressure is useful once adaptive policy is active, but using it
        to leave low-concurrency bypass makes long single-agent reads diverge
        from the original runtime. The bypass should only close when the run is
        approaching a real budget/time/turn/token limit.
        """
        pressures: list[float] = []
        if self.ledger.total_budget > 0:
            pressures.append(_clamp01(self.ledger.total_spent / self.ledger.total_budget))
        if agent.quota.time_limit > 0:
            pressures.append(_clamp01(self.effective_elapsed() / agent.quota.time_limit))
        if agent.quota.max_turns > 0:
            pressures.append(_clamp01(agent._turns / agent.quota.max_turns))
        if self.config.max_total_tokens > 0:
            total_tokens = sum(a.tokens_consumed for a in self.agents.values())
            pressures.append(_clamp01(total_tokens / self.config.max_total_tokens))
        if self.config.tool_policy_soft_total_tokens > 0:
            total_tokens = sum(a.tokens_consumed for a in self.agents.values())
            pressures.append(_clamp01(total_tokens / self.config.tool_policy_soft_total_tokens))
        return max(pressures, default=0.0)

    def _active_agent_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.status not in {"done", "failed"})

    def _low_concurrency_bypass_active(self, agent: Agent) -> bool:
        """Whether adaptive constraints should stay completely out of the turn."""
        if (
            self.config.tool_policy_mode != "adaptive"
            or not self.config.tool_policy_low_concurrency_bypass_enabled
        ):
            return False
        if self.config.tool_policy_low_concurrency_max_active_agents <= 0:
            return False
        if self._active_agent_count() > self.config.tool_policy_low_concurrency_max_active_agents:
            return False
        resource_pressure = self._hard_resource_pressure(agent)
        if resource_pressure >= self.config.tool_policy_low_concurrency_resource_threshold:
            return False
        if _task_expects_structured_collection(agent.task):
            if (
                agent._shell_activity.reconcile_after_web_loop
                or (
                    self.config.tool_policy_low_concurrency_web_loop_threshold > 0
                    and agent._shell_activity.web_loop_pressure
                    >= self.config.tool_policy_low_concurrency_web_loop_threshold
                )
            ):
                return False
        return True

    def _apply_hard_tool_limits(
        self,
        agent: Agent,
        tools: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], set[str], list[str]]:
        scoped = dict(tools)
        removed: set[str] = set()
        reasons: list[str] = []
        spawn_unavailable = self._spawn_unavailable_reason(agent)
        if spawn_unavailable is not None:
            for name in _CREATE_TOOLS:
                if name in scoped:
                    removed.add(name)
                    scoped.pop(name, None)
            reasons.append(spawn_unavailable)
        return scoped, removed, reasons

    def _compute_tool_policy_state(self, agent: Agent) -> ToolPolicyState:
        expected = _extract_expected_output_paths(agent.task)
        missing = self._missing_expected_outputs(agent)
        artifact_gap = len(missing) / len(expected) if expected else 0.0

        active_children = [
            cid for cid in agent.children
            if cid in self.agents and self.agents[cid].status not in {"done", "failed"}
        ]
        dependency_window = max(1, self.config.tool_policy_dependency_window)
        dependency_pressure = len(active_children) / max(1, dependency_window, len(agent.children))
        total_agents = len(self.agents)
        child_pressure = _clamp01(len(agent.children) / max(1, dependency_window))
        active_child_pressure = _clamp01(len(active_children) / max(1, dependency_window))
        pool_pressure = (
            _clamp01((total_agents - 1) / max(1, self.config.max_agents - 1))
            if self.config.max_agents > 1
            else (1.0 if total_agents > 1 else 0.0)
        )
        topology_pressure = max(
            child_pressure,
            active_child_pressure,
            pool_pressure,
            0.35 if agent.parent else 0.0,
        )
        depth_pressure = _clamp01(agent.depth / self.config.max_depth) if self.config.max_depth > 0 else 1.0
        resource_pressure = self._resource_pressure(agent)
        stagnation_pressure = _clamp01(agent._no_tool_turns / 3.0)
        delivery_pressure, delivery_phase = self._compute_delivery_pressure(
            agent,
            expected_outputs=len(expected),
            missing_outputs=len(missing),
            resource_pressure=resource_pressure,
        )
        web_loop_pressure = agent._shell_activity.web_loop_pressure
        hard_constraint_pressure = max(
            topology_pressure,
            (
                web_loop_pressure
                if agent._shell_activity.web_calls >= self.config.tool_policy_web_loop_min_calls
                else 0.0
            ),
        )
        reconcile_phase = (
            self.config.tool_policy_web_loop_reconcile_enabled
            and agent._shell_activity.reconcile_after_web_loop
            and not self._has_delivery_candidate(agent)
            and (
                topology_pressure >= self.config.tool_policy_hard_constraint_topology_threshold
                or hard_constraint_pressure >= self.config.tool_policy_single_agent_web_reconcile_threshold
            )
        )

        can_create = self._spawn_unavailable_reason(agent) is None
        active_artifact_gap = artifact_gap * (1.0 - delivery_pressure)

        create = (
            0.40
            + 0.35 * active_artifact_gap
            + 0.25 * stagnation_pressure
            - 0.75 * resource_pressure
            - 0.35 * delivery_pressure
            - 0.25 * web_loop_pressure
            - 0.55 * dependency_pressure
            - 0.45 * depth_pressure
        )
        if agent.parent:
            create -= 0.20
        if not can_create:
            create = 0.0

        read = (
            0.25
            + 0.25 * stagnation_pressure
            + 0.20 * (0.0 if expected else 1.0)
            + 0.15 * dependency_pressure
            + 0.15 * delivery_pressure
            + 0.25 * web_loop_pressure
        )
        message = 0.20 + 0.65 * dependency_pressure + (0.10 if agent.children else 0.0)
        work = (
            0.45
            + 0.45 * active_artifact_gap
            + 0.15 * (1.0 - dependency_pressure)
            - 0.25 * resource_pressure
            - 0.15 * delivery_pressure
            - 0.25 * web_loop_pressure
        )
        finish = (
            0.10
            + (0.65 if expected and not missing else 0.0)
            + (0.20 if not expected and agent._turns > 1 else 0.0)
            + 0.25 * resource_pressure
            + 0.50 * delivery_pressure
            - 0.55 * active_artifact_gap
            - 0.45 * dependency_pressure
        )

        return ToolPolicyState(
            create=_clamp01(create),
            read=_clamp01(read),
            message=_clamp01(message),
            work=_clamp01(work),
            finish=_clamp01(finish),
            resource_pressure=resource_pressure,
            dependency_pressure=_clamp01(dependency_pressure),
            artifact_gap=_clamp01(artifact_gap),
            stagnation_pressure=stagnation_pressure,
            delivery_pressure=delivery_pressure,
            delivery_phase=delivery_phase,
            web_loop_pressure=web_loop_pressure,
            topology_pressure=topology_pressure,
            reconcile_phase=reconcile_phase,
            expected_outputs=len(expected),
            missing_outputs=len(missing),
            active_children=len(active_children),
        )

    def _compute_delivery_pressure(
        self,
        agent: Agent,
        *,
        expected_outputs: int,
        missing_outputs: int,
        resource_pressure: float,
    ) -> tuple[float, str]:
        if not self.config.tool_policy_delivery_enabled:
            return 0.0, "off"

        shell = agent._shell_activity
        delivery = agent._delivery_activity
        web_calls = shell.web_calls

        turn_start = max(1, self.config.tool_policy_delivery_turns_start)
        turn_end = max(turn_start + 1, self.config.tool_policy_delivery_turns_end)
        tool_start = max(1, self.config.tool_policy_delivery_tool_calls_start)
        tool_end = max(tool_start + 1, self.config.tool_policy_delivery_tool_calls_end)
        turn_pressure = _clamp01((agent._turns - turn_start) / (turn_end - turn_start))
        tool_call_pressure = _clamp01((agent._tool_calls - tool_start) / (tool_end - tool_start))
        non_delivery_pressure = _clamp01((delivery.non_delivery_tool_calls - tool_start) / (tool_end - tool_start))
        activity_pressure = max(turn_pressure, tool_call_pressure, non_delivery_pressure)

        low_signal_ratio = shell.web_low_signal_calls / max(1, web_calls)
        repeat_ratio = (shell.repeated_web_queries + shell.repeated_web_domains) / max(1, 2 * web_calls)
        large_output_pressure = max(
            _clamp01(shell.large_output_calls / 8.0),
            _clamp01(shell.truncated_output_calls / 4.0),
            _clamp01(shell.consecutive_large_outputs / 3.0),
            _clamp01(shell.output_bytes / 900000.0),
        )
        loop_pressure = max(
            shell.web_saturation,
            low_signal_ratio,
            repeat_ratio,
            shell.web_domain_sprawl,
            large_output_pressure,
        )

        expected_ready = (
            expected_outputs > 0
            and missing_outputs == 0
            and bool(delivery.expected_output_files)
        )
        candidate_ready = bool(delivery.candidate_output_files or agent.artifacts)
        written_candidate_ready = (
            delivery.write_calls > 0
            and (expected_outputs == 0 or missing_outputs == 0)
        )
        completion_signal = expected_ready or candidate_ready or written_candidate_ready or delivery.submit_calls > 0
        if completion_signal and delivery.candidate_ready_turn <= 0:
            delivery.candidate_ready_turn = agent._turns
            delivery.candidate_ready_non_delivery_tool_calls = delivery.non_delivery_tool_calls

        min_candidate_outputs = max(1, self.config.tool_policy_delivery_prepare_min_candidate_outputs)
        prepare_loop_pressure = max(shell.web_loop_pressure, loop_pressure)
        prepare_activity_ready = (
            activity_pressure >= self.config.tool_policy_delivery_prepare_activity_threshold
            and (
                shell.reconcile_after_web_loop
                or shell.web_saturation >= self.config.tool_policy_web_saturation_threshold
                or prepare_loop_pressure >= self.config.tool_policy_delivery_pressure_threshold
            )
        )
        prepare_web_calls_ready = (
            shell.web_calls >= self.config.tool_policy_delivery_prepare_web_calls
            or shell.blocked_web_calls > 0
            or (
                self.config.tool_policy_web_saturation_enabled
                and shell.web_calls >= self.config.tool_policy_web_saturation_min_calls
                and shell.web_saturation >= self.config.tool_policy_web_saturation_threshold
            )
        )
        local_candidate_ready = (
            shell.web_calls <= 1
            and shell.candidate_like_outputs >= max(
                min_candidate_outputs,
                self.config.tool_policy_delivery_prepare_local_candidate_outputs,
            )
        )
        prepare_web_calls_ready = prepare_web_calls_ready or local_candidate_ready
        candidate_like_signal = (
            self.config.tool_policy_delivery_prepare_enabled
            and not completion_signal
            and shell.candidate_like_outputs >= min_candidate_outputs
            and prepare_web_calls_ready
            and (
                local_candidate_ready
                or shell.blocked_web_calls > 0
                or
                resource_pressure >= self.config.tool_policy_delivery_prepare_resource_threshold
                or prepare_activity_ready
                or prepare_loop_pressure >= self.config.tool_policy_delivery_pressure_threshold
            )
        )
        if candidate_like_signal and shell.candidate_prepare_open_turn <= 0:
            shell.candidate_prepare_open_turn = agent._turns
            shell.candidate_prepare_open_non_delivery_tool_calls = delivery.non_delivery_tool_calls
        prepare_blocked_web_finalize = (
            self.config.tool_policy_delivery_prepare_enabled
            and not completion_signal
            and self._multi_item_evidence_scope_ready(agent)
            and shell.candidate_like_outputs >= min_candidate_outputs
            and shell.blocked_web_calls
            >= max(1, self.config.tool_policy_delivery_prepare_blocked_web_finalize_after)
        )
        prepare_window_open = (
            self.config.tool_policy_delivery_prepare_enabled
            and not completion_signal
            and shell.candidate_prepare_open_turn > 0
            and (
                delivery.non_delivery_tool_calls
                - shell.candidate_prepare_open_non_delivery_tool_calls
                < max(1, self.config.tool_policy_delivery_prepare_hold_tool_calls)
                or prepare_blocked_web_finalize
            )
        )

        verification_required = self._delivery_verification_required(agent, completion_signal)
        verification_satisfied = self._delivery_verification_satisfied(agent, resource_pressure)
        verification_open = verification_required and not verification_satisfied

        completion_pressure = 0.0
        if expected_ready:
            completion_pressure = 1.0
        elif candidate_ready:
            completion_pressure = 0.68
        elif written_candidate_ready:
            completion_pressure = 0.55
        elif delivery.submit_calls > 0:
            completion_pressure = 0.50

        # Generic completion pressure: expensive activity can make the runtime
        # consolidate, but it is not treated as evidence that the task is ready.
        # Only explicit candidate/completion signals can move the agent into the
        # delivery-only schema.
        pressure = _clamp01(
            0.62 * completion_pressure
            + 0.18 * resource_pressure * (1.0 if completion_signal else 0.25)
            + 0.12 * activity_pressure * (1.0 if completion_signal else 0.25)
            + 0.08 * loop_pressure * (1.0 if completion_signal else 0.0)
            + 0.18 * min(1.0, shell.candidate_like_outputs / 2.0)
            * (1.0 if candidate_like_signal or prepare_window_open else 0.0)
            + 0.18 * (1.0 if prepare_blocked_web_finalize else 0.0)
        )

        threshold = self.config.tool_policy_delivery_pressure_threshold
        hard_delivery = (
            (expected_ready and not verification_open)
            or (completion_signal and resource_pressure >= self.config.tool_policy_delivery_verify_resource_cutoff)
            or (completion_signal and activity_pressure >= 0.98)
            or (completion_signal and non_delivery_pressure >= 0.98)
        )
        prepare_delivery_overrun = candidate_like_signal and self._prepare_delivery_overrun(agent)
        web_loop_forced_delivery = self._web_loop_forced_delivery(agent)
        if verification_open and completion_signal:
            phase = "verify"
        elif hard_delivery or (completion_signal and pressure >= max(0.82, threshold + 0.25)):
            phase = "deliver"
        elif completion_signal and pressure >= threshold:
            phase = "consolidate"
        elif prepare_delivery_overrun or web_loop_forced_delivery:
            phase = "deliver"
        elif candidate_like_signal or prepare_window_open:
            phase = "prepare"
        elif max(resource_pressure, activity_pressure, loop_pressure) >= threshold:
            phase = "consolidate"
        else:
            phase = "explore"
        if phase == "deliver" and delivery.deliver_enter_turn <= 0:
            delivery.deliver_enter_turn = agent._turns
            delivery.deliver_enter_non_delivery_tool_calls = delivery.non_delivery_tool_calls
        return pressure, phase

    def _delivery_verification_required(self, agent: Agent, completion_signal: bool) -> bool:
        if not self.config.tool_policy_delivery_verify_enabled or not completion_signal:
            return False
        delivery = agent._delivery_activity
        if delivery.submit_calls > 0:
            return False
        if delivery.candidate_ready_turn <= 0:
            return False
        if agent._shell_activity.web_calls < self.config.tool_policy_delivery_verify_web_calls:
            return False
        return True

    def _delivery_verification_satisfied(self, agent: Agent, resource_pressure: float) -> bool:
        if resource_pressure >= self.config.tool_policy_delivery_verify_resource_cutoff:
            return True
        delivery = agent._delivery_activity
        min_candidate_reviews = max(0, self.config.tool_policy_delivery_verify_min_candidate_reviews)
        min_source_reviews = max(0, self.config.tool_policy_delivery_verify_min_source_reviews)
        if delivery.candidate_review_calls < min_candidate_reviews:
            return False
        if delivery.candidate_source_review_calls < min_source_reviews:
            return False
        return True

    def _prepare_shell_window_open(self, agent: Agent) -> bool:
        shell = agent._shell_activity
        if shell.candidate_prepare_open_turn <= 0:
            return True
        window = max(0, self.config.tool_policy_delivery_prepare_shell_window)
        if window <= 0:
            return False
        used = (
            agent._delivery_activity.non_delivery_tool_calls
            - shell.candidate_prepare_open_non_delivery_tool_calls
        )
        return used < window

    def _prepare_delivery_overrun(self, agent: Agent) -> bool:
        shell = agent._shell_activity
        delivery = agent._delivery_activity
        min_candidate_outputs = max(1, self.config.tool_policy_delivery_prepare_min_candidate_outputs)
        if shell.candidate_like_outputs < min_candidate_outputs:
            return False
        if (
            shell.blocked_web_calls
            >= max(1, self.config.tool_policy_delivery_prepare_blocked_web_finalize_after)
        ):
            return self._multi_item_evidence_scope_ready(agent)
        if shell.candidate_prepare_open_turn <= 0:
            return False
        used = delivery.non_delivery_tool_calls - shell.candidate_prepare_open_non_delivery_tool_calls
        return used >= max(1, self.config.tool_policy_delivery_prepare_hold_tool_calls)

    def _web_loop_forced_delivery(self, agent: Agent) -> bool:
        shell = agent._shell_activity
        if not self.config.tool_policy_web_loop_reconcile_enabled:
            return False
        if self._has_delivery_candidate(agent):
            return False
        if not shell.reconcile_after_web_loop:
            return False
        if shell.web_calls < max(1, self.config.tool_policy_web_loop_min_calls):
            return False
        no_gain_threshold = max(
            self.config.tool_policy_web_loop_min_calls,
            self.config.tool_policy_web_loop_consecutive_no_gain * 2,
        )
        if shell.consecutive_web_no_gain >= no_gain_threshold:
            return True
        calls = max(1, shell.web_calls)
        repeat_query_ratio = shell.repeated_web_queries / calls
        repeat_result_ratio = shell.repeated_web_results / calls
        repeated_loop = (
            shell.web_calls >= max(8, self.config.tool_policy_web_loop_min_calls // 2)
            and (
                repeat_query_ratio >= 0.45
                or repeat_result_ratio >= 0.45
                or (
                    (shell.repeated_web_queries + shell.repeated_web_results)
                    / max(1, 2 * calls)
                ) >= 0.40
            )
            and (
                shell.web_no_gain_calls >= max(4, int(calls * 0.35))
                or shell.consecutive_web_no_gain >= 4
            )
        )
        if repeated_loop:
            return True
        no_gain_ratio = shell.web_no_gain_calls / calls
        low_signal_ratio = shell.web_low_signal_calls / calls
        return (
            shell.web_calls >= no_gain_threshold
            and no_gain_ratio >= self.config.tool_policy_web_loop_no_gain_threshold
            and low_signal_ratio >= 0.65
        )

    def _has_delivery_candidate(self, agent: Agent) -> bool:
        expected = _extract_expected_output_paths(agent.task)
        missing = self._missing_expected_outputs(agent)
        delivery = agent._delivery_activity
        child_shell_candidate = (
            agent.parent is not None
            and bool(
                agent._shell_activity.candidate_like_outputs
                or agent._shell_activity.material_gain_calls
            )
            and any(
                str(key).startswith(_CHILD_EVIDENCE_DELIVERY_PREFIX)
                for key in agent._notified_thresholds
            )
        )
        return (
            bool(delivery.expected_output_files)
            or bool(agent.artifacts)
            or child_shell_candidate
            or (
                bool(delivery.candidate_output_files)
                and (not expected or bool(delivery.expected_output_files) or not missing)
            )
            or (delivery.write_calls > 0 and (not expected or not missing))
        )

    def _submitted_result_text(
        self,
        agent: Agent,
        arguments: dict[str, Any] | None,
        result: Any,
    ) -> tuple[str, str]:
        submitted = ""
        shared_copy = ""
        if isinstance(result, dict):
            submitted = str(result.get("submitted") or "")
            shared_copy = str(result.get("shared_copy") or "")
        if not submitted and isinstance(arguments, dict):
            submitted = str(arguments.get("path") or "")

        candidates: list[Path] = []
        for raw in (shared_copy, submitted):
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = agent.workspace / path
            candidates.append(path)
        if agent.artifacts:
            candidates.append(agent.artifacts[-1].absolute_path)
        delivery = agent._delivery_activity
        for raw in sorted(delivery.expected_output_files | delivery.candidate_output_files):
            path = Path(raw)
            if not path.is_absolute():
                path = agent.workspace / path
            candidates.append(path)

        for path in candidates:
            try:
                if not path.exists() or not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                for key in ("answer", "final_answer", "FINAL ANSWER"):
                    value = parsed.get(key)
                    if value is not None:
                        return str(value).strip(), submitted
            if path.name.lower() == "answer.json" or len(text) <= 2000:
                return text, submitted
        return "", submitted

    def _has_material_note_since_reconcile(self, agent: Agent) -> bool:
        shell = agent._shell_activity
        delivery = agent._delivery_activity
        return (
            shell.reconcile_after_web_loop
            and delivery.write_calls > shell.web_recovery_last_write_calls
            and not self._has_delivery_candidate(agent)
        )

    def _allowed_shell_capabilities(self, state: ToolPolicyState, agent: Agent | None = None) -> set[str]:
        """Return the current shell sub-capabilities allowed by constraints."""
        all_caps = set(_SHELL_CAPABILITIES)
        if (
            self.config.tool_policy_mode == "off"
            or not self.config.tool_policy_prune_shell_capabilities
            or state.bypassed
        ):
            return all_caps

        web_saturated = (
            agent is not None
            and self.config.tool_policy_web_saturation_enabled
            and agent._shell_activity.web_calls >= self.config.tool_policy_web_saturation_min_calls
            and agent._shell_activity.web_saturation >= self.config.tool_policy_web_saturation_threshold
        )
        web_loop_reconcile = (
            agent is not None
            and self.config.tool_policy_web_loop_reconcile_enabled
            and state.reconcile_phase
        )
        topology_web_loop = (
            agent is not None
            and self.config.tool_policy_web_loop_reconcile_enabled
            and agent._shell_activity.reconcile_after_web_loop
            and state.topology_pressure >= self.config.tool_policy_hard_constraint_topology_threshold
        )
        web_recovery_open = (
            (web_loop_reconcile or topology_web_loop)
            and agent is not None
            and agent._shell_activity.web_recovery_calls_remaining > 0
        )
        web_observation_open = (
            agent is not None
            and self.config.tool_policy_web_saturation_enabled
            and agent._shell_activity.web_calls < self.config.tool_policy_web_saturation_min_calls
        )
        web_pressure = (
            agent._shell_activity.web_saturation
            if agent is not None and agent._shell_activity.web_calls >= self.config.tool_policy_web_saturation_min_calls
            else 0.0
        )

        start = self.config.tool_policy_shell_capability_pressure_start
        end = self.config.tool_policy_shell_capability_pressure_end
        delivery_pressure = state.delivery_pressure if state.delivery_phase in {"prepare", "consolidate", "deliver"} else 0.0
        pressure = max(state.resource_pressure, state.finish, web_pressure, delivery_pressure, state.web_loop_pressure)
        if state.delivery_phase == "prepare":
            if agent is not None and not self._prepare_shell_window_open(agent):
                return {"python", "fs", "process", "unknown"}
            if web_saturated and not web_recovery_open:
                return {"python", "fs", "process", "unknown"}
            return {"web", "python", "fs", "process", "unknown"}

        if pressure < start and not web_saturated and not web_loop_reconcile:
            return all_caps

        if state.delivery_phase == "verify":
            return {"web", "python", "fs", "process", "unknown"}
        if state.delivery_phase == "consolidate":
            allowed = {"web", "python", "fs", "process", "unknown"}
            if web_saturated or (web_loop_reconcile and not web_recovery_open):
                allowed.discard("web")
            return allowed
        if state.delivery_phase == "deliver":
            if (
                agent is not None
                and self._prepare_delivery_overrun(agent)
                and not self._has_delivery_candidate(agent)
            ):
                return {"python", "fs", "process", "unknown"}
            if topology_web_loop and not web_recovery_open:
                return {"python", "fs", "process", "unknown"}
            if web_observation_open:
                return {"web", "python", "fs", "process", "unknown"}
            return {"python", "fs", "process", "unknown"}
        if web_loop_reconcile:
            if web_recovery_open:
                return {"web", "python", "fs", "process", "unknown"}
            return {"python", "fs", "process", "unknown"}

        if end <= start:
            ratio = 1.0
        else:
            ratio = _clamp01((pressure - start) / (end - start))

        if ratio < 0.35:
            allowed = {"web", "python", "fs", "process", "unknown"}
        elif ratio < 0.70:
            allowed = {"web", "python", "fs"}
        elif state.finish >= self.config.tool_policy_finish_threshold:
            allowed = {"python", "fs"}
        else:
            allowed = {"web", "python", "fs"}

        if web_observation_open:
            allowed.add("web")
        if web_saturated:
            allowed.discard("web")
        if web_loop_reconcile and not web_recovery_open:
            allowed.discard("web")
        return allowed

    def _refresh_tool_context_policy(self, agent: Agent) -> ToolPolicyState:
        if self.config.tool_policy_mode == "off" or self._low_concurrency_bypass_active(agent):
            self._tool_context.allowed_shell_capabilities = set(_SHELL_CAPABILITIES)
            return ToolPolicyState(reason="off", bypassed=True)
        state = self._compute_tool_policy_state(agent)
        override = self._pending_tool_overrides.get(agent.id) or {}
        override_tools = set(override.get("last_scoped_tools") or [])
        if override.get("active_turn") == agent._turns and override_tools.intersection(_SHELL_TOOLS):
            self._tool_context.allowed_shell_capabilities = set(_SHELL_CAPABILITIES)
            state.reason = ",".join(filter(None, [state.reason, "runtime_intervention_shell_capabilities"]))
            return state
        self._tool_context.allowed_shell_capabilities = self._allowed_shell_capabilities(state, agent)
        return state

    def _track_delivery_activity(
        self,
        agent: Agent,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        delivery = agent._delivery_activity
        expected = set(_extract_expected_output_paths(agent.task))
        candidate_names = {Path(p).name for p in delivery.candidate_output_files | delivery.expected_output_files}
        candidate_paths = {str(p) for p in delivery.candidate_output_files | delivery.expected_output_files}

        path_text = ""
        if isinstance(arguments, dict):
            raw_path = arguments.get("path") or arguments.get("src") or ""
            if isinstance(raw_path, str):
                path_text = raw_path

        wrote = False
        if tool_name in {
            "ws_create_file", "ws_append_file", "ws_replace_string",
            "ws_multi_replace", "ws_apply_patch", "submit", "transfer",
        }:
            wrote = True
        elif tool_name in {"shell", "tb_shell"}:
            command = str(arguments.get("command", "")) if isinstance(arguments, dict) else ""
            path_candidates = [
                m.group(1)
                for m in re.finditer(r"(?<!\d)(?:>|>>)\s*([A-Za-z0-9_./$-]+\.[A-Za-z0-9]+)", command)
            ]
            path_candidates.extend(
                m.group(3)
                for m in re.finditer(
                    r"\b(?:open|write_text)\s*\(\s*([rubfRUBF]*)(['\"])([^'\"]+\.[A-Za-z0-9]+)\2",
                    command,
                )
            )
            expected_names = {Path(p).name for p in expected}
            candidate_output_paths = [
                p.strip().strip("'\"")
                for p in path_candidates
                if (
                    _looks_like_output_path(p)
                    and Path(p.strip().strip("'\"")).name not in {"null"}
                    and (
                        not expected
                        or p.strip().strip("'\"") in expected
                        or Path(p.strip().strip("'\"")).name in expected_names
                    )
                    and (
                        expected
                        or any(
                            marker in Path(p.strip().strip("'\"")).name.lower()
                            for marker in {"answer", "final", "output", "response", "result", "solution", "submission"}
                        )
                    )
                )
            ]
            wrote = bool(
                re.search(r"(^|[;&|]\s*)(?:cat|printf|echo|python\d?|python3)\b", command)
                and bool(candidate_output_paths)
            )
            if not path_text and candidate_output_paths:
                path_text = candidate_output_paths[-1]
        elif tool_name == "tb_write_file":
            path_text = str(arguments.get("path") or arguments.get("file") or path_text)
            wrote = bool(path_text)

        activity_changed = False
        if wrote:
            delivery.write_calls += 1
            delivery.last_write_turn = agent._turns
            activity_changed = True
        elif tool_name not in {"submit", "set_status", "deliver_to_parent", "get_cost"}:
            delivery.non_delivery_tool_calls += 1
            activity_changed = True

        if tool_name == "submit":
            delivery.submit_calls += 1
            delivery.last_submit_turn = agent._turns
            activity_changed = True

        if path_text:
            clean_path = path_text.strip().strip("'\"")
            if _looks_like_output_path(clean_path):
                delivery.candidate_output_files.add(clean_path)
                if clean_path.startswith("/app/") or clean_path.startswith("/workspace/"):
                    delivery.container_candidate_output_files.add(clean_path)
                if clean_path in expected or Path(clean_path).name in {Path(p).name for p in expected}:
                    delivery.expected_output_files.add(clean_path)
                candidate_paths.add(clean_path)
                candidate_names.add(Path(clean_path).name)

        for raw_path in expected:
            if any(path.exists() for path in self._path_candidates_for_agent(agent, raw_path)):
                delivery.expected_output_files.add(raw_path)
                candidate_paths.add(raw_path)
                candidate_names.add(Path(raw_path).name)

        if wrote and delivery.candidate_output_files and delivery.candidate_ready_turn <= 0:
            delivery.candidate_ready_turn = agent._turns
            delivery.candidate_ready_non_delivery_tool_calls = delivery.non_delivery_tool_calls

        if delivery.candidate_ready_turn > 0 and tool_name in {"ws_read_file", "ws_grep", "shell", "tb_shell", "tb_read_file"}:
            reviewed_candidate = False
            if path_text:
                clean_path = path_text.strip().strip("'\"")
                reviewed_candidate = (
                    clean_path in candidate_paths
                    or Path(clean_path).name in candidate_names
                )
            if tool_name in {"shell", "tb_shell"}:
                command = str(arguments.get("command", "")) if isinstance(arguments, dict) else ""
                reviewed_candidate = reviewed_candidate or any(name and name in command for name in candidate_names)
            if reviewed_candidate:
                delivery.candidate_review_calls += 1

        if (
            delivery.candidate_ready_turn > 0
            and tool_name not in {"submit", "set_status", "get_cost"}
            and agent._turns > delivery.candidate_ready_turn
        ):
            source_review = False
            if tool_name in {"shell", "tb_shell"}:
                command = str(arguments.get("command", "")) if isinstance(arguments, dict) else ""
                capability = classify_shell_capability(command)
                source_review = capability in {"web", "python"}
            elif tool_name in {"ws_read_file", "ws_grep", "ws_code_outline", "ws_read_symbol", "tb_read_file"}:
                source_review = not path_text or Path(path_text).name not in candidate_names
            if source_review:
                delivery.candidate_source_review_calls += 1

        if self.config.tool_policy_log_events and activity_changed:
            self._emit(agent.id, "delivery_activity", delivery.as_event())

        if (
            wrote
            and self.config.tool_policy_web_loop_reconcile_enabled
            and self._has_material_note_since_reconcile(agent)
            and not self._low_concurrency_bypass_active(agent)
            and self._compute_tool_policy_state(agent).reconcile_phase
            and agent._shell_activity.web_recovery_calls_remaining <= 0
        ):
            agent._shell_activity.web_recovery_calls_remaining = max(0, self.config.tool_policy_web_loop_recovery_calls)
            agent._shell_activity.web_recovery_grants += 1
            agent._shell_activity.web_recovery_last_write_calls = delivery.write_calls
            if self.config.tool_policy_log_events:
                self._emit(agent.id, "shell_activity", agent._shell_activity.as_event())

    def _update_shell_activity(self, agent: Agent, command: str, result: Any) -> None:
        activity = agent._shell_activity
        result_text = ""
        if isinstance(result, dict):
            result_text = "\n".join(
                str(result.get(key) or "")
                for key in ("stdout", "stderr")
            )
            output_size = sum(
                len(str(result.get(key) or ""))
                for key in ("stdout", "stderr")
            )
            if result.get("stdout_file") or result.get("stderr_file") or "truncated " in result_text:
                activity.truncated_output_calls += 1
        else:
            result_text = str(result)
            output_size = len(str(result))

        activity.output_bytes += output_size
        lowered_result = result_text.lower()
        unavailable_markers = (
            "command not found",
            "not found",
            "no module named",
            "modulenotfounderror",
            "cannot stat",
            "no such file or directory",
            "permission denied",
        )
        if any(marker in lowered_result for marker in unavailable_markers):
            activity.unavailable_tool_calls += 1
        if output_size >= max(12000, int(self.config.shell_max_output * 0.60)):
            activity.large_output_calls += 1
            activity.consecutive_large_outputs += 1
        else:
            activity.consecutive_large_outputs = 0

        capability = classify_shell_capability(command)
        candidate_like_output = _looks_like_candidate_output(
            agent.task,
            result_text,
        )
        if (
            capability == "web"
            and (
                _web_result_low_signal(result)
                or _web_result_hard_block(result)
            )
        ):
            candidate_like_output = False
        if candidate_like_output:
            activity.candidate_like_outputs += 1
            activity.candidate_like_output_bytes += output_size

        if isinstance(result, dict) and result.get("blocked") and result.get("blocked_capability") == "web":
            activity.blocked_web_calls += 1
            if (
                self.config.tool_policy_web_saturation_enabled
                and self._multi_item_evidence_scope_ready(agent)
                and (
                    activity.web_saturation >= self.config.tool_policy_web_saturation_threshold
                    or activity.blocked_web_calls >= self.config.tool_policy_web_saturation_finalize_after_blocks
                )
                and self._has_delivery_candidate(agent)
            ):
                activity.finalize_after_web_saturation = True
                self._emit(agent.id, "shell_activity", activity.as_event())
            elif self.config.tool_policy_log_events:
                self._emit(agent.id, "shell_activity", activity.as_event())
            return

        if not self.config.tool_policy_prune_shell_capabilities and capability != "web":
            if self.config.tool_policy_log_events and (
                activity.large_output_calls
                or activity.truncated_output_calls
                or activity.unavailable_tool_calls
                or activity.candidate_like_outputs
            ):
                self._emit(agent.id, "shell_activity", activity.as_event())
            return

        if capability != "web":
            if self.config.tool_policy_log_events and (
                activity.large_output_calls
                or activity.truncated_output_calls
                or activity.unavailable_tool_calls
                or activity.candidate_like_outputs
            ):
                self._emit(agent.id, "shell_activity", activity.as_event())
            return

        activity.web_calls += 1
        pipeline_failed = bool(
            isinstance(result, dict)
            and result.get("exit_code") is not None
            and result.get("exit_code") not in (0, "0")
        )
        if pipeline_failed:
            # A failed local parser or pipeline does not establish that the remote URL is irrelevant.
            activity.consecutive_web_low_signal = 0
            activity.consecutive_web_no_gain = 0
            if self.config.tool_policy_log_events:
                self._emit(agent.id, "shell_activity", activity.as_event())
            return
        if activity.reconcile_after_web_loop and activity.web_recovery_calls_remaining > 0:
            activity.web_recovery_calls_remaining -= 1

        signature = _web_command_signature(command)
        domain = _web_command_domain(command)
        result_text = _web_result_text(result)
        result_digest = _stable_text_digest(result_text) if result_text else ""
        hard_blocked = _web_result_hard_block(result)

        repeated_query = bool(signature and signature in activity.unique_web_signatures)
        repeated_domain = bool(domain and domain == activity.last_web_domain)
        low_signal = _web_result_low_signal(result)
        repeated_result = bool(result_digest and result_digest in activity.unique_web_result_digests)
        search_result = _is_search_domain(domain)
        keyword_overlap = _task_keyword_overlap(agent.task, result_text)
        structured_score = _structured_output_score(result_text)
        structured_gain = (
            structured_score >= 0.55
            and not repeated_result
            and output_size >= 120
        )
        has_gain = (
            (not low_signal or structured_gain)
            and not repeated_result
            and (keyword_overlap >= 2 or structured_gain)
            and (not search_result or output_size >= 1200)
        )
        if has_gain:
            activity.material_gain_calls += 1

        if repeated_query:
            activity.repeated_web_queries += 1
        if repeated_domain:
            activity.repeated_web_domains += 1
        if repeated_result:
            activity.repeated_web_results += 1
        if search_result:
            activity.search_result_calls += 1
        if keyword_overlap > 0:
            activity.task_keyword_hits += 1
        if low_signal:
            activity.web_low_signal_calls += 1
            activity.consecutive_web_low_signal += 1
        else:
            activity.consecutive_web_low_signal = 0
        if has_gain:
            activity.consecutive_web_no_gain = 0
        else:
            activity.web_no_gain_calls += 1
            activity.consecutive_web_no_gain += 1

        if signature:
            activity.unique_web_signatures.add(signature)
            activity.last_web_signature = signature
        if domain:
            activity.unique_web_domains.add(domain)
            activity.last_web_domain = domain
        if result_digest:
            activity.unique_web_result_digests.add(result_digest)
            activity.last_web_result_digest = result_digest
        if hard_blocked and signature:
            activity.hard_blocked_web_signatures.add(signature)

        calls = max(1, activity.web_calls)
        low_signal_ratio = activity.web_low_signal_calls / calls
        repeat_query_ratio = activity.repeated_web_queries / calls
        repeat_domain_ratio = activity.repeated_web_domains / calls
        no_gain_ratio = activity.web_no_gain_calls / calls
        repeat_result_ratio = activity.repeated_web_results / calls
        search_result_ratio = activity.search_result_calls / calls
        # Two loop shapes are common:
        # 1. drilling into the same source/query, captured by repeat ratios;
        # 2. broad source hopping after evidence stalls, captured by sprawl.
        # Both are runtime-only signals and never become prompt text.
        activity.web_domain_sprawl = min(
            1.0,
            max(0, len(activity.unique_web_domains) - 6) / 10.0,
        )
        consecutive_ratio = min(1.0, activity.consecutive_web_low_signal / 4.0)
        volume_ratio = min(1.0, max(0, calls - self.config.tool_policy_web_saturation_min_calls + 1) / 16.0)

        activity.web_saturation = _clamp01(
            0.20 * low_signal_ratio
            + 0.15 * repeat_query_ratio
            + 0.10 * repeat_domain_ratio
            + 0.25 * activity.web_domain_sprawl
            + 0.05 * consecutive_ratio
            + 0.25 * volume_ratio
        )
        no_gain_pressure = _clamp01(
            0.34 * no_gain_ratio
            + 0.20 * min(1.0, activity.consecutive_web_no_gain / max(1, self.config.tool_policy_web_loop_consecutive_no_gain))
            + 0.14 * repeat_result_ratio
            + 0.12 * search_result_ratio
            + 0.12 * activity.web_domain_sprawl
            + 0.08 * volume_ratio
        )
        activity.web_loop_pressure = max(activity.web_saturation, no_gain_pressure)
        if (
            self.config.tool_policy_web_loop_reconcile_enabled
            and activity.web_calls >= self.config.tool_policy_web_loop_min_calls
            and not self._has_delivery_candidate(agent)
            and (
                no_gain_ratio >= self.config.tool_policy_web_loop_no_gain_threshold
                or activity.consecutive_web_no_gain >= self.config.tool_policy_web_loop_consecutive_no_gain
                or activity.web_loop_pressure >= self.config.tool_policy_web_loop_pressure_threshold
            )
        ):
            activity.reconcile_after_web_loop = True

        if (
            self.config.tool_policy_log_events
            or activity.web_saturation >= self.config.tool_policy_web_saturation_threshold
            or activity.reconcile_after_web_loop
        ):
            self._emit(agent.id, "shell_activity", activity.as_event())

    def _web_search_failover_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or not _is_search_domain(_web_command_domain(command))
            or not _web_result_low_signal(result)
        ):
            return ""
        signature = _web_command_signature(command)
        key = f"web_search_failover_guidance:{signature or _web_command_domain(command)}"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The public HTML search returned no usable evidence. Do not repeat this exact query. "
            "Runtime will attach a structured fallback search when available. Follow its direct links, or use "
            "a direct site/document URL or ordinary public API. Use a materially different query and do not "
            "search benchmark datasets or answer dumps."
        )

    def _web_failure_fuse_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or not _web_result_hard_block(result)
        ):
            return ""
        signature = _web_command_signature(command)
        key = f"web_failure_fuse_guidance:{signature}"
        if not signature or key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This exact URL returned an explicit 403/429 denial or human-verification challenge and is now "
            "fused. Do not fetch it again or keep parsing its challenge page. Switch to another independent "
            "source; if the task asks you to execute or verify a named deterministic procedure, perform a local "
            "deterministic reproduction and report the remote-source limitation with the local evidence."
        )

    def _direct_web_no_gain_guidance(self, agent: Agent, command: str) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or (
                _web_command_writes_download(command)
                and not _web_command_writes_html_download(command)
            )
            or agent._shell_activity.consecutive_web_no_gain <= 0
        ):
            return ""
        domain = _web_command_domain(command)
        signature = _web_command_signature(command)
        if not domain or _is_search_domain(domain) or not signature:
            return ""
        if signature != agent._shell_activity.last_web_signature:
            return ""
        if signature in agent._shell_activity.hard_blocked_web_signatures:
            return ""
        key = f"direct_web_no_gain_guidance:{signature}"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This direct fetch produced no task-aligned evidence. Treat the mismatch as negative evidence and "
            "do not fetch or grep the same URL again. Revisit the strongest named candidate already retrieved, "
            "or change the source or assumption. For a versioned repository, inspect the record's version "
            "history instead of inferring the relevant date from its identifier."
        )

    @staticmethod
    def _is_broad_arxiv_historical_search(command: str) -> bool:
        for raw_url in _extract_web_urls(command):
            try:
                parsed = urlparse(raw_url)
                domain = (parsed.netloc or "").lower()
            except Exception:
                continue
            if not domain.endswith("arxiv.org") or not parsed.path.rstrip("/").endswith("/search"):
                continue
            return True
        return False

    @staticmethod
    def _requires_constrained_arxiv_search(agent: Agent) -> bool:
        task = str(agent.task or "").lower()
        return (
            "runtime inherited research protocol" in task
            and "arxiv" in task
            and ("advanced html" in task or "title field" in task)
        )

    def _arxiv_broad_search_guidance(self, agent: Agent, command: str) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or not self._requires_constrained_arxiv_search(agent)
            or not self._is_broad_arxiv_historical_search(command)
        ):
            return ""
        key = "arxiv_broad_search_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This basic arXiv search page is sorted newest-first, so changing its query text or grep/head filters "
            "cannot reliably find a historical paper or verify its title and version date. Do not repeat another "
            "basic search-page variant. Use one arXiv advanced HTML query with separate author and title fields; "
            "the required parameter shape is advanced=&terms-0-term=<URLENCODED_AUTHOR>&terms-0-field=author&"
            "terms-1-term=<URLENCODED_TITLE>&terms-1-field=title. Parse the returned IDs structurally, then "
            "inspect candidate version metadata. If empty, switch to named bibliography candidates instead of "
            "changing the local grep expression."
        )

    def _arxiv_broad_search_block_reason(self, agent: Agent, command: str) -> str:
        if (
            "arxiv_broad_search_guidance" not in agent._notified_thresholds
            or not self._requires_constrained_arxiv_search(agent)
            or not self._is_broad_arxiv_historical_search(command)
        ):
            return ""
        return (
            "Repeated basic arXiv newest-first search blocked. Use advanced search with separate author and title "
            "fields, then verify candidate version dates; do not retry a basic search page with another query or grep."
        )

    @staticmethod
    def _is_malformed_arxiv_advanced_search(command: str) -> bool:
        command_field_pairs = {
            (index, field.lower())
            for index, field in re.findall(
                r"terms-(\d+)-field\s*=\s*(author|title)",
                command,
                flags=re.IGNORECASE,
            )
        }
        command_term_indexes = set(
            re.findall(r"terms-(\d+)-term\s*=", command, flags=re.IGNORECASE)
        )
        for raw_url in _extract_web_urls(command):
            try:
                parsed = urlparse(raw_url)
                domain = (parsed.netloc or "").lower()
                query_items = parse_qsl(parsed.query, keep_blank_values=True)
            except Exception:
                continue
            if (
                not domain.endswith("arxiv.org")
                or not parsed.path.rstrip("/").endswith("/search/advanced")
            ):
                continue
            fields = {
                str(value or "").lower()
                for key, value in query_items
                if str(key).lower().endswith("-field")
            }
            keys = {str(key or "").lower() for key, _ in query_items}
            # Python often builds this URL from adjacent f-string fragments, so
            # the URL extractor sees only the first literal ending at advanced=&.
            if keys == {"advanced"}:
                fields.update(
                    field
                    for index, field in command_field_pairs
                    if index in command_term_indexes
                )
            return (
                domain not in {"arxiv.org", "www.arxiv.org"}
                or "advanced" not in keys
                or not {"author", "title"}.issubset(fields)
            )
        return False

    def _arxiv_malformed_advanced_search_block_reason(self, agent: Agent, command: str) -> str:
        if (
            not self._requires_constrained_arxiv_search(agent)
            or not self._is_malformed_arxiv_advanced_search(command)
        ):
            return ""
        return (
            "Malformed arXiv advanced-search URL blocked. The advanced endpoint does not accept an opaque terms= "
            "expression, belongs on arxiv.org rather than export.arxiv.org, and only displays the form when the "
            "hidden advanced= parameter is absent. Use https://arxiv.org/search/advanced?advanced=&terms-0-term="
            "<URLENCODED_AUTHOR>&terms-0-field=author&terms-1-term="
            "<URLENCODED_TITLE>&terms-1-field=title, with separate operator parameters if needed; parse result "
            "IDs before checking version metadata."
        )

    def _arxiv_outdated_advanced_selector_block_reason(self, agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "runtime inherited research protocol" not in task
            or "/search/advanced" not in value
            or not re.search(
                r"select(?:_one)?\s*\(\s*[\"'][^\"']*\.list-result",
                value,
            )
        ):
            return ""
        return (
            "Outdated arXiv advanced-search selector blocked. Current result records use "
            "li.arxiv-result, with the identifier link under p.list-title a[href*='/abs/']; "
            ".list-result returns an empty list even when the downloaded HTML contains results."
        )

    def _arxiv_wrong_fulltext_html_source_block_reason(self, agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "runtime inherited research protocol" not in task
            or "ar5iv" not in task
            or not re.search(r"https?://(?:www\.)?arxiv\.org/html/\d{4}\.\d+", value)
        ):
            return ""
        return (
            "Wrong full-text HTML source blocked. This inherited protocol requires the converted full paper at "
            "https://ar5iv.labs.arxiv.org/html/<ARXIV_ID>; arxiv.org/html/<ARXIV_ID> can return a short fallback "
            "page without the bibliography or figures."
        )

    def _arxiv_saved_advanced_html_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        stdout = str(result.get("stdout") or "") if isinstance(result, dict) else ""
        if (
            "runtime inherited research protocol" not in task
            or "arxiv.org/search/advanced" not in value
            or not re.search(r"(?:^|\s)(?:-o|--output)\s+[^\s;&|]+", value)
            or not re.search(r"(?:^|\D)200(?:\D|$)", stdout)
        ):
            return ""
        key = "arxiv_saved_advanced_html_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The arXiv advanced HTML download succeeded with HTTP 200 and was saved locally. Do not classify "
            "this as rate limiting, retry the network request, or deliver an incomplete search report. Parse "
            "the saved file now with BeautifulSoup using li.arxiv-result and extract each identifier from "
            "p.list-title a[href*='/abs/']; then inspect candidate version histories."
        )

    def _arxiv_advanced_batch_timeout_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        stderr = str(result.get("stderr") or "") if isinstance(result, dict) else ""
        exit_code = result.get("exit_code") if isinstance(result, dict) else None
        if (
            "runtime inherited research protocol" not in task
            or "arxiv.org/search/advanced" not in value
            or not (exit_code == -1 or "timeout" in stderr.lower())
        ):
            return ""
        key = "arxiv_advanced_batch_timeout_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The batched arXiv request timed out, but earlier loop iterations may already have saved valid HTML. "
            "Do not discard them or rerun the full batch. First list the saved search HTML files and parse every "
            "nonempty file locally with li.arxiv-result; only then fetch missing authors, preferably concurrently "
            "with bounded per-request timeouts and flushed progress output."
        )

    def _arxiv_version_history_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        output = "\n".join(
            str(result.get(key) or "") for key in ("stdout", "stderr")
        ) if isinstance(result, dict) else str(result or "")
        if (
            "runtime inherited research protocol" not in task
            or "initial or revised arxiv version" not in task
            or "/search/advanced" not in value
            or not re.search(r"arxiv\.org/abs/\d{4}\.\d+", output, flags=re.IGNORECASE)
        ):
            return ""
        key = "arxiv_version_history_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The returned arXiv identifier prefix records only the initial submission month (v1), not every "
            "revision month. Do not prioritize or reject candidates by a YYMM-looking identifier. Inspect the "
            "submission/version history for every promising title-and-author candidate before applying the target "
            "month; candidates repeated under several independently searched authors or already named in the "
            "bibliography should be checked before a merely month-matching identifier."
        )

    @staticmethod
    def _is_linewise_bibliography_filter(agent: Agent, command: str) -> bool:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        return bool(
            "runtime inherited research protocol" in task
            and "bibliography items structurally" in task
            and ("split('\\n')" in value or 'split("\\n")' in value or ".splitlines(" in value)
            and re.search(r"\bfor\b[^\n]{0,100}\bline\b", value)
            and "2020" in value
            and any(marker in value for marker in ("frb", "fast radio", "180916", "burst"))
        )

    def _bibliography_line_filter_guidance(self, agent: Agent, command: str) -> str:
        if not self._is_linewise_bibliography_filter(agent, command):
            return ""
        key = "bibliography_line_filter_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This filter applies year/topic predicates to individual PDF text lines, but numbered citations often "
            "wrap across several lines. It can silently drop the relevant record. Do not repeat a line-oriented "
            "filter. Segment the normalized bibliography by numbered citation boundaries, join each record's "
            "continuation lines, then apply all predicates to the complete record and resolve any et al. author list "
            "through source metadata."
        )

    def _bibliography_line_filter_block_reason(self, agent: Agent, command: str) -> str:
        if (
            "bibliography_line_filter_guidance" not in agent._notified_thresholds
            or not self._is_linewise_bibliography_filter(agent, command)
        ):
            return ""
        return (
            "Repeated line-oriented bibliography filter blocked. Merge wrapped lines into complete numbered "
            "citation records before filtering by year, topic, or author."
        )

    @staticmethod
    def _partial_bibliography_scope_block_reason(agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "parse all bibliography items structurally" not in task
            or "fitz" not in value
            or not any(marker in value for marker in ("bibliography", "reference"))
        ):
            return ""
        assumes_suffix = bool(
            "last few pages" in value
            or "likely have references" in value
            or re.search(r"range\(\s*len\([^)]*\)\s*-\s*\d+", value)
        )
        if not assumes_suffix:
            return ""
        return (
            "Partial-bibliography shortcut blocked. The protocol requires all bibliography items, so do not "
            "assume an arbitrary last-N-page suffix contains the complete references. Scan page text once to "
            "locate the References/Bibliography heading, include that page and every following page, segment "
            "complete numbered records, and verify that the first parsed citation number is near the start."
        )

    @staticmethod
    def _pdf_full_page_dump_block_reason(agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "print only candidate captions and axis labels, not entire page text" not in task
            or "fitz" not in value
            or "get_text(" not in value
        ):
            return ""
        text_variables = set(re.findall(
            r"\b([a-z_]\w*)\s*=\s*(?:[a-z_]\w*\.)?get_text\(",
            value,
        ))
        dumps_page = any(
            re.search(rf"\bprint\(\s*{re.escape(variable)}\s*\)", value)
            for variable in text_variables
        ) or bool(re.search(r"\bprint\(\s*(?:[a-z_]\w*\.)?get_text\(\)\s*\)", value))
        if not dumps_page:
            return ""
        return (
            "Full-page PDF text dump blocked. The inherited protocol requires only bounded candidate captions "
            "and axis labels, and a whole page adds noise without proving plotted endpoints. Extract a short "
            "caption/label window from the already matched page, then render that page or figure crop and inspect "
            "the plot-frame borders plus tick spacing."
        )

    @staticmethod
    def _arxiv_identifier_month_assumption_block_reason(agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "runtime inherited research protocol" not in task
            or not (
                "arxiv" in task
                and "initial or revised" in task
                and "version" in task
            )
        ):
            return ""
        searches_identifier_prefix = bool(
            re.search(r"[\"']arxiv:20\d{2}[\"']\s+in\b", value)
            or re.search(r"\bgrep\b[^\n]{0,100}arxiv[:./]?20\d{2}", value)
            or re.search(r"\.\s*startswith\s*\(\s*[rubf]*[\"']20\d{2}", value)
            or re.search(
                r"\[\s*(?:0\s*)?:\s*4\s*\]\s*(?:==|!=)\s*[rubf]*[\"']20\d{2}",
                value,
            )
            or re.search(
                r"\bre\.(?:match|search|findall|finditer)\s*\(\s*[rubf]*[\"']"
                r"[^\"'\r\n]{0,40}20\d{2}",
                value,
            )
        )
        if not searches_identifier_prefix:
            return ""
        return (
            "arXiv identifier-month inference blocked. An identifier encodes the initial submission month, "
            "while the task explicitly allows a later revision date. Shortlist papers by title/authors/subject, "
            "then inspect every candidate's version history; do not search bibliography text for a YYMM prefix."
        )

    def _arxiv_exact_title_api_block_reason(self, agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = unquote(str(command or "")).lower()
        protocol_blocks_title_api = (
            "do not add exact 'ti:frb' api filters" in task
            or (
                "advanced html" in task
                and bool(re.search(
                    r"\bdo not\b[^.\r\n]{0,160}(?:exact-title|search_query|\bti\s*:)",
                    task,
                ))
            )
        )
        if (
            not self._requires_constrained_arxiv_search(agent)
            or not protocol_blocks_title_api
            or "export.arxiv.org/api/query" not in value
        ):
            return ""
        exact_title_query = False
        for match in re.finditer(
            r"export\.arxiv\.org/api/query[^\r\n]*",
            value,
        ):
            query_line = match.group(0)
            if "search_query=" not in query_line:
                continue
            if re.search(r"\bti\s*:", query_line):
                exact_title_query = True
                break
            variable_refs = re.findall(
                r"search_query=(?:\{([a-z_]\w*)\}|\$\{?([a-z_]\w*)\}?)",
                query_line,
            )
            for groups in variable_refs:
                variable = next((name for name in groups if name), "")
                if variable and re.search(
                    rf"\b{re.escape(variable)}\s*=\s*f?[\"'][^\r\n]*\bti\s*:",
                    value,
                ):
                    exact_title_query = True
                    break
            if exact_title_query:
                break
        if not exact_title_query:
            return ""
        return (
            "Exact-title arXiv Atom API query blocked by the inherited research protocol. Use arXiv advanced "
            "HTML with separate author and title fields, or inspect ordinary metadata for a named candidate; "
            "do not retry the API with a different ti: token."
        )

    def _arxiv_atom_feed_header_parse_block_reason(self, agent: Agent, command: str) -> str:
        value = str(command or "").lower()
        if (
            not self._requires_constrained_arxiv_search(agent)
            or "export.arxiv.org/api/query" not in value
            or "id_list" not in value
        ):
            return ""
        takes_first_title = "<title>" in value and bool(
            re.search(r"\b(?:titles?|title_matches)\s*\[\s*0\s*\]", value)
        )
        takes_first_updated = "<updated>" in value and bool(
            re.search(r"\b(?:updated|updates?|updated_matches)\s*\[\s*0\s*\]", value)
        )
        if not (takes_first_title or takes_first_updated):
            return ""
        return (
            "Atom feed-header parse blocked. Both the feed and its paper entry contain title/updated fields, so "
            "the first regex match is query metadata rather than paper metadata. Parse the XML and select the Atom "
            "entry element first, then read that entry's title, published, updated, and author/name children. Use "
            "the paper's submission history when every version date is required; never use the feed-level updated "
            "timestamp as a paper revision date."
        )

    @staticmethod
    def _pdf_cli_fallback_guidance(command: str, result: Any) -> str:
        if "pdftotext" not in str(command or "").lower():
            return ""
        if isinstance(result, dict):
            result_text = "\n".join(
                str(result.get(key) or "")
                for key in ("stdout", "stderr", "error")
            )
        else:
            result_text = str(result or "")
        if not re.search(
            r"(?:pdftotext[^\n]*(?:not found|no such file)|command not found)",
            result_text,
            flags=re.IGNORECASE,
        ):
            return ""
        return (
            "The pdftotext executable is unavailable. Do not install packages or retry that CLI. Use the "
            "already available Python PDF libraries immediately: run Python with `import fitz`, open the local "
            "PDF via `fitz.open(...)`, and iterate `page.get_text()`; use PyPDF2 only if importing fitz fails."
        )

    def _figure_panel_disambiguation_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        scope_text = f"{agent.task}\n{self.config.system_extra_instructions}"
        if (
            agent.parent is None
            or classify_shell_capability(command) != "python"
            or not re.search(r"\b(?:bottom|lower)\s+panels?\b", scope_text, flags=re.IGNORECASE)
            or not isinstance(result, dict)
        ):
            return ""
        output = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr"))
        page_mentions = set(re.findall(r"\bpage\s+(\d+)\b", output, flags=re.IGNORECASE))
        if (
            len(page_mentions) < 2
            or "figure" not in output.lower()
            or "panel" not in output.lower()
        ):
            return ""
        key = "figure_panel_disambiguation_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The PDF scan found several figure/page mentions for the target. Do not select the first figure that "
            "mentions the item. Extract every matching figure caption as a complete block, then choose only the "
            "caption that satisfies all wording qualifiers from the task, especially the named item and an explicit "
            "lower/bottom panel. Treat nearby prose and a caption describing several items as ambiguous until the "
            "exact panel label is verified; render that matched page only."
        )

    def _html_regex_parser_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            agent.parent is None
            or classify_shell_capability(command) != "python"
            or "bibliography" not in task
            or "structurally" not in task
            or ".html" not in value
            or not re.search(r"\bre\.(?:findall|search|finditer)\b", value)
            or not isinstance(result, dict)
        ):
            return ""
        output = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr"))
        if not re.search(
            r"(?:found\s+0\s+(?:bib|reference)|"
            r"(?:total\s+)?(?:bib(?:liography)?(?:\s+items?)?|refs?|references?)\s*:\s*0\b|"
            r"no\s+bibliography|no\s+references)",
            output,
            flags=re.IGNORECASE,
        ):
            return ""
        key = "html_regex_parser_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The tag-specific HTML regex returned zero bibliography records. Do not try another div/ol/li regex "
            "or dump every class name. Parse the saved HTML with BeautifulSoup and select the known CSS class "
            "independently of tag name (for ar5iv, `soup.select('.ltx_bibitem')`), then use each node's complete "
            "text and id/label. Inspect one selected node only if the selector itself is empty."
        )

    @staticmethod
    def _local_structured_candidate_count(result: Any) -> int:
        if not isinstance(result, dict):
            return 0
        output = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr"))
        arxiv_ids = set(re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", output))
        candidate_labels = set(re.findall(
            r"\b(?:candidate|bib(?:liography)?|ref(?:erence)?)\s*(?:#|no\.?\s*)?\[?(\d+)\]?",
            output,
            flags=re.IGNORECASE,
        ))
        return max(len(arxiv_ids), len(candidate_labels))

    @staticmethod
    def _local_delivery_audit_ready(command: str, result: Any) -> bool:
        """Recognize a successful local audit that already contains a final candidate."""
        if classify_shell_capability(command) not in {"python", "fs"}:
            return False
        if not isinstance(result, dict):
            return False
        if result.get("exit_code") not in (None, 0, "0"):
            return False
        output = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr"))
        if len(output) < 250:
            return False
        lowered = output.lower()
        if any(
            marker in lowered
            for marker in (
                "traceback (most recent call last)",
                "syntaxerror",
                "assertionerror",
                "verification failed",
            )
        ):
            return False
        audit_markers = (
            "sha256",
            "answer:",
            "count:",
            "hit pages",
            "qualifying",
            "total physical pages",
            "step 22",
            "step-22",
            "hex",
            "degree histogram",
            "path length",
            "min_year",
            "minimum year",
        )
        return sum(marker in lowered for marker in audit_markers) >= 2

    def _child_evidence_checkpoint_guidance(
        self,
        agent: Agent,
        *,
        candidate_like_before: int,
        material_gain_before: int,
        command: str = "",
        result: Any = None,
    ) -> str:
        scope_text = f"{agent.task}\n{self.config.system_extra_instructions}"
        expected_items = self._explicit_expected_evidence_count(scope_text)
        candidate_increased = (
            agent._shell_activity.candidate_like_outputs > candidate_like_before
        )
        material_increased = (
            agent._shell_activity.material_gain_calls > material_gain_before
        )
        capability = classify_shell_capability(command)
        local_candidate_count = self._local_structured_candidate_count(result)
        local_audit_gain = (
            candidate_increased
            and self._local_delivery_audit_ready(command, result)
        )
        local_candidate_gain = (
            candidate_increased
            and capability in {"python", "fs"}
            and (
                local_candidate_count >= max(1, expected_items)
                or local_audit_gain
            )
        )
        if (
            not self.config.child_evidence_checkpoint_enabled
            or agent.parent is None
            or (
                _web_command_writes_download(command)
                and not _web_command_writes_html_download(command)
            )
            or not candidate_increased
            or not (material_increased or local_candidate_gain)
        ):
            return ""
        overlap_author_guidance = ""
        if re.search(
            r"(?:\bone\s+of\s+(?:the\s+)?same\s+authors?\b|"
            r"\bany\s+(?:shared|overlapping)\s+authors?\b|"
            r"\bat\s+least\s+one\s+(?:shared|overlapping)\s+author\b|"
            r"\boverlapping\s+author\b)",
            scope_text,
            flags=re.IGNORECASE,
        ):
            overlap_author_guidance = (
                " Preserve the author-overlap quantifier: search shared authors separately (or with OR), "
                "not by requiring multiple authors with AND; a candidate needs at least one shared author "
                "before the remaining discriminators are checked. A bibliography entry abbreviated with "
                "'et al.' has an unresolved author list: fetch its complete metadata before rejecting it for "
                "lack of a shared author."
            )
        citation_guidance = (
            " If a decisive passage names a numbered citation, resolve that citation before another "
            "author- or date-wide search."
        )
        chart_span_guidance = ""
        if re.search(
            r"\b(?:plotted|plot|diagram|figure|chart)\b.{0,120}\b(?:axis|span|range|endpoint|time)\b|"
            r"\b(?:axis|span|range|endpoint)\b.{0,120}\b(?:plot|diagram|figure|chart)\b",
            scope_text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            chart_span_guidance = (
                " For a plotted-axis span, the first and last printed tick labels are not automatically the "
                "plot endpoints. Render and crop the actual figure, locate both frame borders, and verify "
                "whether they coincide with labeled ticks; otherwise use the tick spacing and border positions "
                "to derive the full displayed range."
            )
        evidence_count = agent._shell_activity.material_gain_calls
        if local_candidate_gain:
            evidence_count = max(
                evidence_count,
                local_candidate_count,
            )
        if expected_items > 1 and evidence_count < expected_items:
            return ""
        followup_threshold = max(3, expected_items + 1)
        followup = (
            agent._shell_activity.candidate_like_outputs >= followup_threshold
            and evidence_count >= followup_threshold
        )
        checkpoint_phase = "followup" if followup else "initial"
        key = f"child_evidence_checkpoint:{checkpoint_phase}"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        # A multi-item checkpoint often means the child has only produced a
        # candidate list. It still needs source resolution and one evidence row
        # per item, so keep the checkpoint advisory instead of removing every
        # research tool after an arbitrary number of calls.
        if expected_items <= 1:
            delivery_after = (
                agent._delivery_activity.non_delivery_tool_calls
                + (1 if followup or local_audit_gain else 2)
            )
            agent._notified_thresholds.add(
                f"{_CHILD_EVIDENCE_DELIVERY_PREFIX}{delivery_after}"
            )
        checkpoint_intro = (
            "Additional task-aligned candidates appeared after the initial checkpoint. Stop expanding the "
            "search and resolve the candidates already in hand. "
            if followup
            else ""
        )
        if expected_items > 1:
            return (
                checkpoint_intro
                + f"This multi-item task requests {expected_items} evidence rows/items and now has enough "
                "material results for a completeness checkpoint. Before launching another broad search, "
                "shortlist any named source candidate already present in the retrieved evidence and test it "
                "against every task discriminator. For versioned records, check both the initial and "
                "revision/updated dates."
                + citation_guidance
                + overlap_author_guidance
                + chart_span_guidance
                + " Verify that every required item has its own retrieved evidence before "
                "delivery. Continue only with narrowly targeted missing items; "
                "leave any unresolved item explicit rather than filling it from memory, then deliver."
            )
        return (
            checkpoint_intro
            + "This result contains material, task-aligned candidate evidence. Before any broad search, "
            "shortlist named candidates already present in this result and test them against all task "
            "discriminators. For versioned records, check both the initial and revision/updated dates."
            + citation_guidance
            + overlap_author_guidance
            + chart_span_guidance
            + " On the "
            "next turn, either call deliver_to_parent now, or make at most one narrowly targeted "
            "extraction/verification call and then deliver. Do not resume broad or open-ended searching."
        )

    def _child_has_formal_delivery(self, agent: Agent) -> bool:
        return any(
            record.get("agent_id") == agent.id
            and record.get("source") == "deliver_to_parent"
            for record in self._candidate_deliveries
        )

    @staticmethod
    def _child_has_material_delivery_evidence(agent: Agent) -> bool:
        return bool(
            agent._shell_activity.candidate_like_outputs > 0
            or agent._shell_activity.material_gain_calls > 0
        )

    def _child_evidence_checkpoint_delivery_due(self, agent: Agent) -> bool:
        if (
            not self.config.child_evidence_checkpoint_enabled
            or agent.parent is None
            or agent.status != "running"
            or self._child_has_formal_delivery(agent)
        ):
            return False
        thresholds: list[int] = []
        for key in agent._notified_thresholds:
            value = str(key)
            if not value.startswith(_CHILD_EVIDENCE_DELIVERY_PREFIX):
                continue
            try:
                thresholds.append(int(value.removeprefix(_CHILD_EVIDENCE_DELIVERY_PREFIX)))
            except ValueError:
                continue
        return bool(
            thresholds
            and agent._delivery_activity.non_delivery_tool_calls >= min(thresholds)
        )

    @staticmethod
    def _explicit_expected_evidence_count(text: str) -> int:
        """Extract a small explicit research scope such as 'exactly six rows'."""
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12,
        }
        value = str(text or "").lower()
        patterns = (
            r"\b(?:exactly|all)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
            r"\s+(?:[a-z][a-z0-9_-]*\s+){0,3}"
            r"(?:rows?|people|persons?|items?|records?|facts?|entries?|sources?)\b",
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
            r"[-\s]+(?:[a-z][a-z0-9_-]*[-\s]+){0,3}rows?\b",
        )
        for pattern in patterns:
            match = re.search(pattern, value)
            if not match:
                continue
            token = match.group(1)
            count = int(token) if token.isdigit() else number_words.get(token, 0)
            if count > 0:
                return min(count, 20)
        named_scope = re.search(
            r"\bexplicitly\s+(?:classify|verify|check)\s+(.{1,500}?)"
            r"(?:\s+against\b|;\s*|(?<!\b[a-z])\.(?:\s|$)|\n|$)",
            value,
            flags=re.IGNORECASE,
        )
        if named_scope and "," in named_scope.group(1):
            items = [
                re.sub(r"^(?:and|or)\s+", "", item.strip(" `\"'"))
                for item in re.split(r"\s*,\s*|\s*,?\s+and\s+", named_scope.group(1))
            ]
            named_items = [
                item for item in items
                if re.fullmatch(
                    r"[a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*){1,5}",
                    item,
                    flags=re.IGNORECASE,
                )
            ]
            if len(named_items) == len(items) and len(named_items) > 1:
                return min(len(named_items), 20)
        return 0

    def _multi_item_evidence_scope_ready(self, agent: Agent) -> bool:
        scope_text = f"{agent.task}\n{self.config.system_extra_instructions}"
        expected_items = self._explicit_expected_evidence_count(scope_text)
        return (
            expected_items <= 1
            or agent._shell_activity.material_gain_calls >= expected_items
        )

    def _large_html_extraction_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        if classify_shell_capability(command) != "web" or not isinstance(result, dict):
            return ""
        output = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr"))
        domain = _web_command_domain(command)
        if (
            len(output) < 8000
            or "wikipedia.org" not in domain
            or "<script" not in output.lower()
        ):
            return ""
        key = "large_html_extraction_guidance:wikipedia"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "Raw Wikipedia HTML produced mostly script/configuration boilerplate. Do not repeat grep or sed "
            "against the article HTML. Use the MediaWiki API with action=query, prop=extracts, explaintext=1, "
            "format=json, and a literal titles=PAGE_TITLE parameter, then parse query.pages[*].extract and "
            "select only the relevant paragraph."
        )

    def _child_shared_final_write_block_reason(self, agent: Agent, tc: ToolCall) -> str:
        if agent.parent is None or tc.name not in {
            "ws_create_file", "ws_append_file", "ws_replace_string",
            "ws_multi_replace", "ws_apply_patch", "tb_write_file",
        }:
            return ""
        raw_path = str(tc.arguments.get("path") or tc.arguments.get("file") or "").strip()
        if not raw_path:
            return ""
        expected_names = {
            Path(path).name for path in _extract_expected_output_paths(agent.task)
        }
        if not expected_names:
            return ""
        expanded = raw_path.replace("$SHARED", str(self._tool_context.shared_dir))
        path = Path(expanded)
        if not path.is_absolute():
            path = agent.workspace / path
        try:
            path.resolve().relative_to(Path(self._tool_context.shared_dir).resolve())
        except ValueError:
            return ""
        if path.name not in expected_names:
            return ""
        return (
            f"child agents may not create or overwrite the root-owned shared final artifact {path.name}; "
            "write evidence in the child workspace and call deliver_to_parent instead"
        )

    def _web_search_failover_block_reason(self, agent: Agent, command: str) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or agent._shell_activity.consecutive_web_low_signal <= 0
        ):
            return ""
        domain = _web_command_domain(command)
        signature = _web_command_signature(command)
        if not domain or not _is_search_domain(domain) or not signature:
            return ""
        if signature != agent._shell_activity.last_web_signature:
            return ""
        return (
            f"Repeated low-signal search query blocked on {domain}. Switch query or method now: use the "
            "structured Yahoo/Bing fallback, or open a direct source/public API."
        )

    def _web_failure_fuse_block_reason(self, agent: Agent, command: str) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
        ):
            return ""
        signature = _web_command_signature(command)
        if not signature or signature not in agent._shell_activity.hard_blocked_web_signatures:
            return ""
        return (
            "Repeated fetch of a URL with an explicit 403/429/challenge response was blocked. "
            "Use another independent source or switch to local deterministic reproduction."
        )

    def _direct_web_no_gain_block_reason(self, agent: Agent, command: str) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or (
                _web_command_writes_download(command)
                and not _web_command_writes_html_download(command)
            )
            or agent._shell_activity.consecutive_web_no_gain <= 0
        ):
            return ""
        domain = _web_command_domain(command)
        signature = _web_command_signature(command)
        if not domain or _is_search_domain(domain) or not signature:
            return ""
        if signature != agent._shell_activity.last_web_signature:
            return ""
        return (
            "Repeated no-gain fetch of the same direct URL was blocked. Treat the previous mismatch as "
            "negative evidence and change the candidate, source, or retrieval method."
        )

    def _apply_state_tool_policy(self, agent: Agent, tools: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], ToolPolicyState]:
        if self._child_evidence_checkpoint_delivery_due(agent) and "deliver_to_parent" in tools:
            scoped = {"deliver_to_parent": tools["deliver_to_parent"]}
            state = ToolPolicyState(reason="child_evidence_checkpoint_delivery")
            state.finish = 1.0
            state.work = 0.0
            state.message = 1.0
            state.removed_tools = sorted(set(tools) - set(scoped))
            state.scoped_tools = list(scoped)
            return scoped, state
        if self.config.tool_policy_mode == "off":
            state = ToolPolicyState(reason="off", bypassed=True)
            state.reason = "off"
            state.scoped_tools = list(tools)
            return tools, state

        original_order = {name: idx for idx, name in enumerate(tools)}
        if self._low_concurrency_bypass_active(agent):
            scoped, removed, hard_reasons = self._apply_hard_tool_limits(agent, tools)
            state = ToolPolicyState(reason="off", bypassed=True)
            state.reason = ",".join(["off", *hard_reasons])
            state.scoped_tools = list(scoped)
            state.removed_tools = sorted(removed)
            return scoped, state

        state = self._compute_tool_policy_state(agent)
        scoped, removed, reasons = self._apply_hard_tool_limits(agent, tools)

        if agent._shell_activity.finalize_after_web_saturation and self._has_delivery_candidate(agent):
            finalize_allowed = (
                {"submit", "set_status", "deliver_to_parent", "get_cost"}
                | _DELIVERY_READ_TOOLS
                | _DELIVERY_WRITE_TOOLS
            )
            finalized = {name: tool for name, tool in scoped.items() if name in finalize_allowed}
            if finalized:
                removed.update(set(scoped) - set(finalized))
                scoped = finalized
                state.finish = max(state.finish, 1.0)
                state.work = min(state.work, 0.25)
                state.read = min(state.read, 0.35)
                scoped = self._order_tools_by_policy(scoped, state, original_order)
                state.removed_tools = sorted(removed)
                state.scoped_tools = list(scoped)
                state.reason = "web_saturation_finalize"
                return scoped, state

        if state.reconcile_phase:
            reconcile_allowed = (
                {"query", "send", "deliver_to_parent", "wait", "get_cost", "rebirth", "set_status"}
                | _SHELL_TOOLS
                | _DELIVERY_READ_TOOLS
                | {"ws_create_file", "ws_append_file", "ws_replace_string", "tb_write_file"}
            )
            shell_closed_after_blocks = (
                self.config.tool_policy_web_loop_shell_close_after_blocks > 0
                and agent._shell_activity.blocked_web_calls
                >= self.config.tool_policy_web_loop_shell_close_after_blocks
                and agent._shell_activity.web_recovery_calls_remaining <= 0
                and not self._has_delivery_candidate(agent)
            )
            if shell_closed_after_blocks:
                reconcile_allowed.difference_update(_SHELL_TOOLS)
            reconciled = {name: tool for name, tool in scoped.items() if name in reconcile_allowed}
            if reconciled:
                removed.update(set(scoped) - set(reconciled))
                scoped = reconciled
                state.read = max(state.read, 0.70)
                state.message = max(state.message, 0.45)
                state.work = min(state.work, 0.25 if shell_closed_after_blocks else 0.45)
                if shell_closed_after_blocks:
                    state.finish = max(state.finish, 0.65)
                reasons.append("web_loop_reconcile")
                if shell_closed_after_blocks:
                    reasons.append("web_loop_shell_closed")

        if state.delivery_phase == "prepare":
            prepare_allowed = (
                {"submit", "set_status", "deliver_to_parent", "get_cost"}
                | _SHELL_TOOLS
                | _DELIVERY_READ_TOOLS
                | _DELIVERY_WRITE_TOOLS
            )
            prepare_shell_open = self._prepare_shell_window_open(agent)
            prepare_blocked_web_finalize = (
                self.config.tool_policy_delivery_prepare_enabled
                and self._multi_item_evidence_scope_ready(agent)
                and agent._shell_activity.candidate_like_outputs
                >= max(1, self.config.tool_policy_delivery_prepare_min_candidate_outputs)
                and agent._shell_activity.blocked_web_calls
                >= max(1, self.config.tool_policy_delivery_prepare_blocked_web_finalize_after)
            )
            if prepare_blocked_web_finalize and agent._shell_activity.candidate_prepare_finalize_turn <= 0:
                agent._shell_activity.candidate_prepare_finalize_turn = agent._turns
                agent._shell_activity.candidate_prepare_finalize_non_delivery_tool_calls = (
                    agent._delivery_activity.non_delivery_tool_calls
                )
            finalize_shell_window_open = (
                not prepare_blocked_web_finalize
                or (
                    agent._delivery_activity.non_delivery_tool_calls
                    - agent._shell_activity.candidate_prepare_finalize_non_delivery_tool_calls
                    < max(0, self.config.tool_policy_delivery_prepare_finalize_shell_window)
                )
            )
            if prepare_blocked_web_finalize:
                prepare_allowed = (
                    {"submit", "set_status", "deliver_to_parent"}
                    | _SHELL_TOOLS
                    | _DELIVERY_READ_TOOLS
                    | {"ws_create_file", "ws_append_file", "ws_replace_string", "tb_write_file"}
                )
            prepared = {name: tool for name, tool in scoped.items() if name in prepare_allowed}
            if prepared:
                removed.update(set(scoped) - set(prepared))
                scoped = prepared
                state.finish = max(state.finish, 0.82 if prepare_blocked_web_finalize else 0.62)
                state.work = min(state.work, 0.22 if prepare_blocked_web_finalize else (0.58 if prepare_shell_open else 0.30))
                state.read = max(state.read, 0.52)
                reasons.append("delivery_prepare")
                if not prepare_shell_open:
                    reasons.append("delivery_prepare_web_closed")
                if prepare_blocked_web_finalize:
                    reasons.append("delivery_prepare_blocked_web_finalize")
                    if not finalize_shell_window_open:
                        reasons.append("delivery_prepare_finalize_web_closed")

        if state.delivery_phase == "consolidate":
            close_after = int(self.config.tool_policy_delivery_consolidate_shell_close_after or 0)
            if close_after > 0 and agent._delivery_activity.non_delivery_tool_calls >= close_after:
                consolidate_allowed = (
                    {"submit", "set_status", "deliver_to_parent", "get_cost"}
                    | _DELIVERY_READ_TOOLS
                    | _DELIVERY_WRITE_TOOLS
                )
                consolidated = {name: tool for name, tool in scoped.items() if name in consolidate_allowed}
                if consolidated:
                    removed.update(set(scoped) - set(consolidated))
                    scoped = consolidated
                    state.finish = max(state.finish, 0.82)
                    state.work = min(state.work, 0.18)
                    state.read = max(state.read, 0.58)
                    reasons.append("delivery_consolidate_shell_closed")

        if state.delivery_phase == "verify":
            verify_allowed = (
                {"submit", "set_status", "deliver_to_parent", "get_cost"}
                | _SHELL_TOOLS
                | _DELIVERY_READ_TOOLS
                | _DELIVERY_WRITE_TOOLS
            )
            verified = {name: tool for name, tool in scoped.items() if name in verify_allowed}
            if verified:
                removed.update(set(scoped) - set(verified))
                scoped = verified
                state.finish = min(state.finish, 0.70)
                state.work = min(state.work, 0.55)
                state.read = max(state.read, 0.82)
                reasons.append("delivery_verify")

        if state.delivery_phase == "deliver":
            forced_prepare_delivery = self._prepare_delivery_overrun(agent) and not self._has_delivery_candidate(agent)
            forced_web_loop_delivery = self._web_loop_forced_delivery(agent)
            candidate_delivery = self._has_delivery_candidate(agent) or agent._delivery_activity.submit_calls > 0
            deliver_non_delivery_calls = (
                agent._delivery_activity.non_delivery_tool_calls
                - agent._delivery_activity.deliver_enter_non_delivery_tool_calls
            )
            local_finalize_window_open = (not candidate_delivery) and deliver_non_delivery_calls < 8
            deliver_allowed = (
                {"submit", "set_status", "deliver_to_parent", "get_cost"}
                | _DELIVERY_READ_TOOLS
                | _DELIVERY_WRITE_TOOLS
            )
            if candidate_delivery:
                deliver_allowed = _DELIVERY_READ_TOOLS | {"submit", "set_status", "deliver_to_parent", "get_cost"}
            elif forced_prepare_delivery or forced_web_loop_delivery:
                deliver_allowed = _DELIVERY_WRITE_TOOLS | {"submit", "set_status", "deliver_to_parent", "get_cost"}
                if local_finalize_window_open:
                    deliver_allowed.update(_SHELL_TOOLS)
            elif local_finalize_window_open:
                deliver_allowed.add("query")
                if not (
                    state.reconcile_phase
                    and agent._shell_activity.web_recovery_calls_remaining <= 0
                ):
                    deliver_allowed.update(_SHELL_TOOLS)
            delivered = {name: tool for name, tool in scoped.items() if name in deliver_allowed}
            if delivered:
                removed.update(set(scoped) - set(delivered))
                scoped = delivered
                state.finish = max(state.finish, 0.92)
                state.work = min(state.work, 0.35)
                state.read = max(state.read, 0.55)
                reasons.append("delivery_phase")
                if candidate_delivery:
                    reasons.append("delivery_candidate_final")
                if forced_prepare_delivery:
                    reasons.append("delivery_prepare_forced_final")
                if forced_web_loop_delivery:
                    reasons.append("delivery_web_loop_forced_final")
                if not local_finalize_window_open:
                    reasons.append("delivery_finalize_scope")

        spawn_reconcile_closed = state.reconcile_phase
        spawn_should_close = (
            spawn_reconcile_closed
            or state.resource_pressure >= self.config.tool_policy_resource_threshold
            or state.dependency_pressure >= self.config.tool_policy_dependency_threshold
            or (
                self.config.tool_policy_mode == "enforce"
                and state.create < self.config.tool_policy_spawn_min_weight
            )
        )
        if spawn_should_close:
            for name in _CREATE_TOOLS:
                if name in scoped:
                    removed.add(name)
                    scoped.pop(name, None)
            if spawn_reconcile_closed:
                reasons.append("web_loop_spawn_closed")
            else:
                reasons.append("spawn_closed")

        if state.finish >= self.config.tool_policy_finish_threshold:
            for name in _CREATE_TOOLS:
                if name in scoped:
                    removed.add(name)
                    scoped.pop(name, None)
            reasons.append("finish_weight_high")

        if self.config.tool_policy_mode == "enforce":
            if state.dependency_pressure >= self.config.tool_policy_dependency_threshold:
                allowed = _COORDINATION_TOOLS | _READ_TOOLS | _LIFECYCLE_TOOLS
                reasons.append("dependency_scope")
            elif state.finish >= self.config.tool_policy_finish_threshold:
                allowed = _READ_TOOLS | _LIFECYCLE_TOOLS | _DELIVERY_WRITE_TOOLS | _SHELL_TOOLS
                reasons.append("finish_scope")
            elif state.create >= self.config.tool_policy_spawn_min_weight and state.create >= state.work:
                allowed = _CREATE_TOOLS | _COORDINATION_TOOLS | _READ_TOOLS | _LIFECYCLE_TOOLS | _SHELL_TOOLS
                reasons.append("create_scope")
            else:
                allowed = _WORK_TOOLS | _READ_TOOLS | _LIFECYCLE_TOOLS | {"query"}
                reasons.append("work_scope")

            enforced = {name: tool for name, tool in scoped.items() if name in allowed}
            if enforced:
                removed.update(set(scoped) - set(enforced))
                scoped = enforced

        if not scoped:
            scoped = {name: tool for name, tool in tools.items() if name in _LIFECYCLE_TOOLS}
            if not scoped:
                scoped = tools
            reasons.append("fallback_scope")

        scoped = self._order_tools_by_policy(scoped, state, original_order)
        if self.config.tool_policy_prune_tools:
            scoped, pruned = self._prune_tools_by_policy(scoped, state)
            if pruned:
                removed.update(pruned)
                reasons.append("prune_tools")
        state.removed_tools = sorted(removed)
        state.scoped_tools = list(scoped)
        state.reason = ",".join(dict.fromkeys(reasons)) or "all_tools"
        return scoped, state

    def _maybe_inject_delivery_finalization_notice(
        self,
        agent: Agent,
        tool_policy: ToolPolicyState,
        turn_tools: dict[str, dict[str, Any]],
    ) -> None:
        if agent._delivery_final_notice_sent:
            return
        if tool_policy.delivery_phase != "deliver":
            return
        if not (self._prepare_delivery_overrun(agent) or self._web_loop_forced_delivery(agent)):
            return
        if self._has_delivery_candidate(agent):
            return
        if not ({"ws_create_file", "set_status"} <= set(turn_tools)):
            return
        expected_paths = _extract_expected_output_paths(agent.task)
        if expected_paths:
            path_text = ", ".join(f"`{path}`" for path in expected_paths)
            file_instruction = (
                f"Create the required final file(s): {path_text}. If both a workspace "
                "path and a shared path are named, write the same final answer to both."
            )
        else:
            file_instruction = (
                "Create the required final answer file named by the task. If no filename "
                "is named, create `answer.md`."
            )
        agent._delivery_final_notice_sent = True
        agent.history.append({
            "role": "user",
            "content": (
                "[Runtime delivery constraint]\n"
                "Web investigation has reached a no-gain loop or enough evidence has already "
                "been collected. Stop investigating now. Do not create scripts, do not call "
                "shell again, do not fetch more data, and do not call unavailable or "
                "network-related tools. "
                f"{file_instruction} Use ws_create_file or ws_append_file for file output, "
                "then call set_status(\"done\", result=\"wrote final answer\"). If evidence "
                "is imperfect, choose the best answer supported by the evidence already in the conversation."
            ),
        })

    def _load_runtime_interventions(self) -> set[str]:
        loaded_agents: set[str] = set()
        path = self.config.intervention_file
        if not path or not path.exists():
            return loaded_agents
        key = str(path)
        start = self._intervention_offsets.get(key, 0)
        try:
            with path.open("r", encoding="utf-8") as f:
                f.seek(start)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    decision = decision_from_intervention_item(item)
                    if decision is None:
                        continue
                    self._register_strategy_decision(decision)
                    self._emit(decision.target_agent_id, "strategy_decision", decision.as_event())
                    loaded_agents.add(decision.target_agent_id)
                self._intervention_offsets[key] = f.tell()
        except OSError:
            return loaded_agents
        return loaded_agents

    def _apply_intervention_tool_override(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> tuple[dict[str, dict[str, Any]], ToolPolicyState]:
        self._load_runtime_interventions()
        override = self._pending_tool_overrides.get(agent.id)
        if not override:
            return turn_tools, tool_policy

        all_enabled_tools = self._all_tools()
        stale_route_at_delivery_gate = (
            tool_policy.reason == "child_evidence_checkpoint_delivery"
            and bool(override.get("message_delivered"))
        )
        available_tools = turn_tools if stale_route_at_delivery_gate else all_enabled_tools
        requested = [name for name in override.get("tools", []) if name in available_tools]
        scoped = {name: available_tools[name] for name in requested}
        if not scoped:
            return turn_tools, tool_policy

        message = str(override.get("message") or "").strip()
        if override.get("action") in {"self_kill", "SELF_KILL"}:
            message = message or (
                "Supervisor intervention: your branch has been judged low-value "
                "after query-based inspection. Call kill(agent_id=<your own id>) now."
            )
        message_injected = False
        if message and not override.get("message_delivered"):
            if override.get("strategy_action") == "SUPERVISOR_SPAWN" or any(
                name in scoped for name in ("spawn", "spawn_many")
            ):
                spawn_tool = "spawn_many" if "spawn_many" in scoped else "spawn"
                message = (
                    f"Runtime topology override: call {spawn_tool} as your next tool call. "
                    "Do not call shell, file, read, write, wait, query, or get_cost first. "
                    "Create the child branches described below, then stop this turn.\n"
                    f"{message}"
                )
            agent.history.append({
                "role": "user",
                "content": f"[Runtime supervisor intervention]\n{message}",
            })
            override["message_delivered"] = True
            message_injected = True

        removed = sorted(set(turn_tools) - set(scoped))
        restored = sorted(set(scoped) - set(turn_tools))
        override["last_scoped_tools"] = list(scoped)
        override["active_turn"] = agent._turns
        override_event = self._tool_override_events.get(agent.id)
        if override_event is not None:
            override_event.clear()
        tool_policy.reason = f"{tool_policy.reason},runtime_intervention"
        tool_policy.scoped_tools = list(scoped)
        tool_policy.removed_tools = sorted(set(tool_policy.removed_tools) | set(removed))
        self._emit(agent.id, "runtime_intervention", {
            "action": override.get("action"),
            "source": override.get("source"),
            "reason": override.get("reason", ""),
            "tools": list(scoped),
            "removed_tools": removed,
            "restored_tools": restored,
            "message": message[:300] if message_injected else "",
            "message_injected": message_injected,
            "delivery_gate_preserved": stale_route_at_delivery_gate,
            "strategy_action": override.get("strategy_action"),
            "metadata": override.get("metadata", {}),
        })
        if override.get("once", True):
            override["consumed_turn"] = agent._turns
        return scoped, tool_policy

    def _finalize_consumed_tool_override(
        self,
        agent: Agent,
        response_tool_calls: list[str],
    ) -> None:
        override = self._pending_tool_overrides.get(agent.id)
        if not override or override.get("consumed_turn") != agent._turns:
            return
        if not override.get("once", True):
            return
        allowed = set(override.get("last_scoped_tools") or override.get("tools") or [])
        called = set(response_tool_calls or [])
        wrong_calls = sorted(called - allowed)
        if (
            self.config.supervisor_retry_override_on_wrong_tool
            and wrong_calls
            and not (called & allowed)
            and int(override.get("wrong_tool_attempts") or 0) < max(0, self.config.supervisor_override_wrong_tool_limit)
        ):
            attempts = int(override.get("wrong_tool_attempts") or 0) + 1
            override["wrong_tool_attempts"] = attempts
            override.pop("consumed_turn", None)
            reminder = (
                "Previous turn ignored the runtime topology override and called unavailable tools "
                f"{wrong_calls}. Retry the same override now using only {sorted(allowed)}."
            )
            override["message"] = f"{reminder}\n{override.get('message', '')}".strip()
            self._emit(agent.id, "runtime_override_retry", {
                "allowed_tools": sorted(allowed),
                "wrong_tool_calls": wrong_calls,
                "attempt": attempts,
                "strategy_action": override.get("strategy_action"),
            })
            return
        self._pending_tool_overrides.pop(agent.id, None)

    def _preserve_consumed_tool_override_for_repair(self, agent: Agent) -> bool:
        override = self._pending_tool_overrides.get(agent.id)
        if (
            not override
            or not override.get("once", True)
            or override.get("consumed_turn") != agent._turns
        ):
            return False
        allowed = list(override.get("last_scoped_tools") or override.get("tools") or [])
        if not allowed:
            return False
        override.pop("consumed_turn", None)
        self._emit(agent.id, "runtime_override_repair_preserved", {
            "allowed_tools": allowed,
            "malformed_count": agent._malformed_tool_turns,
            "strategy_action": override.get("strategy_action"),
        })
        return True

    def _agent_candidate_files(self, agent: Agent) -> list[Path]:
        if not agent.workspace.exists():
            return []
        files: list[Path] = []
        for path in agent.workspace.rglob("*"):
            if path.is_file():
                files.append(path)
        return files

    def _shared_candidate_files(self) -> list[Path]:
        shared = self._tool_context.shared_dir
        if not shared.exists():
            return []
        return [
            path
            for path in shared.rglob("*")
            if (
                path.is_file()
                and self._looks_like_candidate_file(path)
                and not self._looks_like_verification_file(path)
            )
        ]

    def _shared_verification_files(self) -> list[Path]:
        shared = self._tool_context.shared_dir
        if not shared.exists():
            return []
        return [
            path
            for path in shared.rglob("*")
            if path.is_file() and self._looks_like_verification_file(path)
        ]

    def _best_shared_candidate_bytes(self) -> int | None:
        sizes: list[int] = []
        for path in self._shared_candidate_files():
            try:
                sizes.append(path.stat().st_size)
            except OSError:
                continue
        return min(sizes) if sizes else None

    def _observe_best_shared_candidate_bytes(self, best_bytes: int | None) -> int:
        if best_bytes is None:
            self._best_shared_candidate_observed_bytes = None
            self._best_shared_candidate_stable_observations = 0
            return 0
        if best_bytes == self._best_shared_candidate_observed_bytes:
            self._best_shared_candidate_stable_observations += 1
        else:
            self._best_shared_candidate_observed_bytes = best_bytes
            self._best_shared_candidate_stable_observations = 1
        return self._best_shared_candidate_stable_observations

    def _looks_like_candidate_file(self, path: Path) -> bool:
        if path.suffix not in {".py", ".patch", ".diff", ".json"}:
            return False
        text = str(path).lower()
        name = path.name.lower()
        return (
            "candidate" in text
            or "solution" in name
            or "solver" in name
            or "golf" in name
            or "recursive" in name
            or "baseline" in name
            or "compact" in name
            or "model.patch" in name
        )

    def _looks_like_verification_file(self, path: Path) -> bool:
        if path.suffix not in {".json", ".jsonl", ".md", ".txt"}:
            return False
        text = str(path).lower()
        name = path.name.lower()
        return (
            "candidate_matrix" in name
            or "final_selection" in name
            or "verification" in text
            or "verifier" in text
            or "selection" in name
            or "ranking" in name
            or "rank" in name
        )

    def _agent_has_candidate_file(self, agent: Agent) -> bool:
        return any(
            self._looks_like_candidate_file(path)
            and not self._looks_like_verification_file(path)
            for path in self._agent_candidate_files(agent)
        )

    def _agent_tool_count(self, agent: Agent, tool: str) -> int:
        count = 0
        for event in self._events:
            event_type = event.get("event", event.get("type"))
            if event.get("agent") != agent.id or event_type != "tool_call":
                continue
            data = event.get("data") or {}
            if data.get("tool") == tool:
                count += 1
        return count

    def _candidate_agents_for_root(self, root: Agent) -> list[str]:
        ids: list[str] = []
        for cid in root.children:
            child = self.agents.get(cid)
            if child is None:
                continue
            delivery = child._delivery_activity
            if (
                self._agent_has_candidate_file(child)
                or bool(child.artifacts)
                or bool(delivery.candidate_output_files)
                or bool(delivery.expected_output_files)
            ):
                ids.append(cid)
        return sorted(ids)

    def _root_candidate_query_coverage(self, root: Agent, candidate_agent_ids: list[str]) -> int:
        if not candidate_agent_ids:
            return 0
        candidates = set(candidate_agent_ids)
        queried: set[str] = set()
        for event in self._events:
            event_type = event.get("event", event.get("type"))
            if event.get("agent") != root.id or event_type != "tool_call":
                continue
            data = event.get("data") or {}
            if data.get("tool") != "query":
                continue
            args = data.get("args") or {}
            target = str(args.get("agent_id") or "").strip()
            if target in candidates:
                queried.add(target)
            elif not target:
                queried.update(candidates)
        return len(queried)

    def _agent_verification_review_count(self, agent: Agent) -> int:
        verification_files = self._shared_verification_files()
        if not verification_files:
            return 0
        names = {path.name for path in verification_files}
        full_paths = {str(path) for path in verification_files}
        count = 0
        for event in self._events:
            event_type = event.get("event", event.get("type"))
            if event.get("agent") != agent.id or event_type != "tool_call":
                continue
            data = event.get("data") or {}
            tool = data.get("tool")
            if tool not in {"ws_read_file", "ws_grep", "shell", "query"}:
                continue
            args = data.get("args") or {}
            text = " ".join(str(v) for v in args.values() if isinstance(v, (str, int, float)))
            if any(path in text for path in full_paths) or any(name in text for name in names):
                count += 1
        return count

    def _agent_large_tool_results(self, agent: Agent, threshold: int = 3000) -> int:
        count = 0
        for event in self._events:
            event_type = event.get("event", event.get("type"))
            if event.get("agent") != agent.id or event_type != "tool_call":
                continue
            data = event.get("data") or {}
            try:
                size = int(data.get("result_full_len") or 0)
            except Exception:
                size = 0
            if size >= threshold:
                count += 1
        return count

    def _root_agent_id(self, agent: Agent) -> str:
        current = agent
        seen: set[str] = set()
        while current.parent and current.parent in self.agents and current.id not in seen:
            seen.add(current.id)
            current = self.agents[current.parent]
        return current.id

    def _candidate_improve_stalled(self, agent: Agent, best_bytes: int | None) -> bool:
        root_id = self._root_agent_id(agent)
        attempts = self._candidate_improve_attempts.get(root_id, 0)
        if attempts <= 0:
            return False
        baseline = self._candidate_improve_baseline_bytes.get(root_id)
        if baseline is None or best_bytes is None:
            return False
        return best_bytes >= baseline

    def _build_strategy_state(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> StrategyState:
        active_children = sum(
            1
            for cid in agent.children
            if cid in self.agents and self.agents[cid].status not in {"done", "failed"}
        )
        best_shared_bytes = self._best_shared_candidate_bytes()
        if agent.parent is None:
            stable_observations = self._observe_best_shared_candidate_bytes(best_shared_bytes)
        else:
            stable_observations = self._best_shared_candidate_stable_observations
        root_id = self._root_agent_id(agent)
        root_agent = self.agents.get(root_id, agent)
        candidate_agent_ids = self._candidate_agents_for_root(root_agent)
        return StrategyState(
            agent_id=agent.id,
            parent_id=agent.parent,
            is_root=agent.parent is None,
            status=agent.status,
            turns=agent._turns,
            tokens=agent.tokens_consumed,
            tool_calls=agent._tool_calls,
            no_tool_turns=agent._no_tool_turns,
            total_agents=len(self.agents),
            active_children=active_children,
            shared_candidate_count=len(self._shared_candidate_files()),
            candidate_agent_count=len(candidate_agent_ids),
            candidate_query_coverage=self._root_candidate_query_coverage(root_agent, candidate_agent_ids),
            shared_verification_count=len(self._shared_verification_files()),
            verification_review_calls=self._agent_verification_review_count(root_agent),
            has_local_candidate=self._agent_has_candidate_file(agent),
            query_calls=self._agent_tool_count(agent, "query"),
            kill_calls=self._agent_tool_count(agent, "kill"),
            set_status_calls=self._agent_tool_count(agent, "set_status"),
            large_tool_results=self._agent_large_tool_results(agent),
            requested_auto_kills=len(self._auto_kill_requested),
            candidate_improve_attempts=self._candidate_improve_attempts.get(root_id, 0),
            candidate_improve_stalled=self._candidate_improve_stalled(agent, best_shared_bytes),
            candidate_finalize_attempts=self._candidate_finalize_attempts.get(root_id, 0),
            candidate_evidence_gate_attempts=self._candidate_evidence_gate_attempts.get(root_id, 0),
            root_recovery_spawn_attempts=self._root_recovery_spawn_attempts.get(root_id, 0),
            last_finalized_candidate_bytes=self._candidate_finalize_last_bytes.get(root_id),
            already_auto_kill_requested=agent.id in self._auto_kill_requested,
            pending_override=agent.id in self._pending_tool_overrides,
            resource_pressure=tool_policy.resource_pressure,
            delivery_phase=tool_policy.delivery_phase,
            available_tools=tuple(turn_tools),
            best_shared_candidate_bytes=best_shared_bytes,
            best_shared_candidate_stable_observations=stable_observations,
        )

    def _register_strategy_decision(self, decision: StrategyDecision) -> None:
        raw_action = "self_kill" if "KILL" in decision.action or decision.action == "SELF_KILL" else "tool_override"
        if decision.action in {
            StrategyAction.SPAWN_PORTFOLIO,
            StrategyAction.IMPROVE_CANDIDATE,
            StrategyAction.SPAWN_RECOVERY,
            StrategyAction.SPAWN_VERIFIER,
            "SUPERVISOR_SPAWN",
        }:
            self._strategy_spawn_authorized.add(decision.target_agent_id)
        target_agent = self.agents.get(decision.target_agent_id)
        if decision.action == StrategyAction.IMPROVE_CANDIDATE and target_agent is not None:
            root_id = self._root_agent_id(target_agent)
            self._candidate_improve_attempts[root_id] = self._candidate_improve_attempts.get(root_id, 0) + 1
            metadata = dict(decision.metadata)
            try:
                baseline = metadata.get("best_shared_candidate_bytes")
            except AttributeError:
                baseline = None
            self._candidate_improve_baseline_bytes[root_id] = int(baseline) if baseline is not None else None
        if decision.action == StrategyAction.FINALIZE_CANDIDATE and target_agent is not None:
            root_id = self._root_agent_id(target_agent)
            self._candidate_finalize_attempts[root_id] = self._candidate_finalize_attempts.get(root_id, 0) + 1
            metadata = dict(decision.metadata)
            best_bytes = metadata.get("best_shared_candidate_bytes")
            self._candidate_finalize_last_bytes[root_id] = int(best_bytes) if best_bytes is not None else None
        if decision.action in {
            StrategyAction.QUERY_CANDIDATE_EVIDENCE,
            StrategyAction.SPAWN_VERIFIER,
        } and target_agent is not None:
            root_id = self._root_agent_id(target_agent)
            self._candidate_evidence_gate_attempts[root_id] = (
                self._candidate_evidence_gate_attempts.get(root_id, 0) + 1
            )
        if decision.action == StrategyAction.SPAWN_RECOVERY and target_agent is not None:
            root_id = self._root_agent_id(target_agent)
            self._root_recovery_spawn_attempts[root_id] = (
                self._root_recovery_spawn_attempts.get(root_id, 0) + 1
            )
        self._pending_tool_overrides[decision.target_agent_id] = {
            "source": decision.source,
            "action": raw_action,
            "strategy_action": decision.action,
            "tools": list(decision.tools),
            "reason": decision.reason,
            "message": decision.message,
            "once": decision.once,
            "created_at": time.time(),
            "wrong_tool_attempts": 0,
            "metadata": dict(decision.metadata),
        }

    def _agent_summary_for_supervisor(self, agent: Agent, root_agent: Agent) -> dict[str, Any]:
        delivery = agent._delivery_activity
        return {
            "id": agent.id,
            "parent": agent.parent,
            "children": sorted(agent.children),
            "depth": agent.depth,
            "status": agent.status,
            "bio": agent.bio[:300],
            "task_preview": agent.task[:max(80, self.config.supervisor_payload_agent_task_chars)],
            "turns": agent._turns,
            "tokens": agent.tokens_consumed,
            "tool_calls": agent._tool_calls,
            "has_candidate": self._agent_has_candidate_file(agent) or bool(agent.artifacts),
            "candidate_outputs": len(delivery.candidate_output_files),
            "expected_outputs": len(delivery.expected_output_files),
            "result_preview": (agent.result or "")[:400],
            "queried_by_root": self._root_has_queried_agent(root_agent, agent.id),
        }

    def _root_has_queried_agent(self, root_agent: Agent, target_id: str) -> bool:
        for event in self._events:
            event_type = event.get("event", event.get("type"))
            if event.get("agent") != root_agent.id or event_type != "tool_call":
                continue
            data = event.get("data") or {}
            if data.get("tool") != "query":
                continue
            args = data.get("args") or {}
            if str(args.get("agent_id") or "").strip() == target_id:
                return True
        return False

    def _supervisor_progress_signature(self, agent: Agent) -> tuple[Any, ...]:
        child_status = tuple(
            sorted(
                (cid, self.agents[cid].status)
                for cid in agent.children
                if cid in self.agents
            )
        )
        delivery = agent._delivery_activity
        return (
            child_status,
            len(delivery.candidate_output_files),
            len(delivery.container_candidate_output_files),
            len(delivery.expected_output_files),
            delivery.write_calls,
            delivery.submit_calls,
            delivery.candidate_ready_turn,
            agent._shell_activity.unavailable_tool_calls,
        )

    def _recent_events_for_supervisor(self, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        out: list[dict[str, Any]] = []
        for event in self._events[-max(limit * 4, limit):]:
            event_type = event.get("event", event.get("type"))
            if event_type not in {
                "agent_new", "spawn", "done", "failed", "tool_call",
                "strategy_decision", "runtime_intervention", "llm_done",
            }:
                continue
            data = event.get("data") or {}
            compact: dict[str, Any] = {
                "type": event_type,
                "agent": event.get("agent"),
            }
            if event_type == "tool_call":
                compact["tool"] = data.get("tool")
                args = data.get("args") or {}
                compact["args"] = {
                    k: (str(v)[:180] if isinstance(v, str) else v)
                    for k, v in args.items()
                    if k in {"agent_id", "mode", "status", "description", "task", "command", "path"}
                }
                compact["result_len"] = data.get("result_full_len")
            elif event_type == "llm_done":
                compact["tool_calls"] = data.get("tool_calls", [])
                compact["tokens"] = data.get("tokens")
                compact["content_preview"] = str(data.get("content_preview") or "")[:180]
            else:
                compact["data"] = {
                    k: (str(v)[:220] if isinstance(v, str) else v)
                    for k, v in data.items()
                    if k in {"child", "status", "result", "reason", "action", "source", "strategy_action", "tools", "parent"}
                }
            out.append(compact)
        return out[-limit:]

    def _recent_agent_messages_for_supervisor(self, agent: Agent, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        messages: list[dict[str, Any]] = []
        message_chars = max(120, int(self.config.supervisor_payload_message_chars or 320))
        for msg in agent.history[1:]:
            role = msg.get("role")
            if role not in {"user", "assistant", "tool"}:
                continue
            content = str(msg.get("content") or "")
            if not content:
                continue
            messages.append({"role": role, "content": content[:message_chars]})
        return messages[-limit:]

    def _supervisor_evidence_flow(self, agent: Agent, root_agent: Agent, limit: int) -> dict[str, Any]:
        """Compact generic evidence signals for topology supervision."""
        scan_limit = max(limit * 12, 80)
        recent_writes: list[dict[str, Any]] = []
        recent_finishes: list[dict[str, Any]] = []
        recent_failures: list[dict[str, Any]] = []
        recent_verification: list[dict[str, Any]] = []
        candidate_mentions: list[dict[str, Any]] = []
        query_edges: list[dict[str, Any]] = []

        failure_re = re.compile(
            r"\b(assertionerror|assert\b|failed\b|failure\b|timeout|timed out|"
            r"traceback|exception|error:|does not exist|not found|unexpected value|"
            r"mismatch|incorrect|wrong|out of bounds|oob)\b",
            re.IGNORECASE,
        )
        success_re = re.compile(
            r"\b(passed|success|successful|all tests pass|0 failed|reward['\"]?\s*[:=]\s*1|"
            r"score['\"]?\s*[:=]\s*1(?:\.0+)?)\b",
            re.IGNORECASE,
        )
        strong_success_re = re.compile(
            r"\b(all tests pass(?:ed)?|0 failed|reward['\"]?\s*[:=]\s*1(?:\.0+)?|"
            r"score['\"]?\s*[:=]\s*1(?:\.0+)?)\b",
            re.IGNORECASE,
        )
        verification_re = re.compile(
            r"\b(verify|verification|verifier|test|tests|pytest|judge|check|"
            r"expected|actual|passed|failed|reward|score)\b",
            re.IGNORECASE,
        )
        candidate_re = re.compile(
            r"\b(best|top|candidate|answer|result|expected|actual|rank|"
            r"mean|score|verdict|success|failure)\b",
            re.IGNORECASE,
        )

        for event in self._events[-scan_limit:]:
            event_type = event.get("event", event.get("type"))
            event_agent = event.get("agent")
            data = event.get("data") or {}
            if event_type == "tool_call":
                tool = str(data.get("tool") or "")
                args = data.get("args") or {}
                result_text = str(data.get("result") or "")
                command_or_path = str(args.get("command") or args.get("path") or "")[:240]
                if tool in _DELIVERY_WRITE_TOOLS or tool in {"submit"}:
                    recent_writes.append({
                        "agent": event_agent,
                        "tool": tool,
                        "target": command_or_path,
                        "result": result_text[:240],
                    })
                if tool == "set_status":
                    recent_finishes.append({
                        "agent": event_agent,
                        "status": str(args.get("status") or "")[:80],
                        "result": str(args.get("result") or "")[:320],
                    })
                if tool == "query":
                    query_edges.append({
                        "from": event_agent,
                        "to": str(args.get("agent_id") or "")[:80],
                    })
                combined = f"{command_or_path}\n{result_text}"
                if failure_re.search(combined):
                    recent_failures.append({
                        "agent": event_agent,
                        "tool": tool,
                        "snippet": combined[:500],
                    })
                if verification_re.search(combined):
                    recent_verification.append({
                        "agent": event_agent,
                        "tool": tool,
                        "snippet": combined[:500],
                    })
                if candidate_re.search(result_text):
                    candidate_mentions.append({
                        "agent": event_agent,
                        "tool": tool,
                        "snippet": result_text[:500],
                    })
            elif event_type in {"done", "failed"}:
                result = str(data.get("result") or data.get("reason") or "")
                recent_finishes.append({
                    "agent": event_agent,
                    "status": event_type,
                    "result": result[:320],
                })
                if failure_re.search(result) or event_type == "failed":
                    recent_failures.append({
                        "agent": event_agent,
                        "tool": event_type,
                        "snippet": result[:500],
                    })

        active_children = [
            cid for cid in agent.children
            if cid in self.agents and self.agents[cid].status not in {"done", "failed"}
        ]
        done_children = [
            cid for cid in agent.children
            if cid in self.agents and self.agents[cid].status in {"done", "failed"}
        ]
        return {
            "recent_writes": recent_writes[-6:],
            "recent_finishes": recent_finishes[-6:],
            "recent_failures": recent_failures[-8:],
            "recent_verification": recent_verification[-8:],
            "candidate_mentions": candidate_mentions[-8:],
            "query_edges": query_edges[-8:],
            "current_agent_children": {
                "active": active_children[:12],
                "done_or_failed": done_children[:12],
            },
            "flags": {
                "has_recent_write": bool(recent_writes),
                "has_recent_failure": bool(recent_failures),
                "has_recent_verification": bool(recent_verification),
                "has_recent_successful_verification": any(
                    (
                        strong_success_re.search(str(item.get("snippet") or ""))
                        or (
                            success_re.search(str(item.get("snippet") or ""))
                            and not failure_re.search(str(item.get("snippet") or ""))
                        )
                    )
                    for item in recent_verification[-4:]
                ),
                "has_candidate_mentions": bool(candidate_mentions),
                "has_unqueried_done_children": any(
                    cid in self.agents
                    and self.agents[cid].status in {"done", "failed"}
                    and not self._root_has_queried_agent(root_agent, cid)
                    for cid in root_agent.children
                ),
            },
        }

    def _agent_effective_evidence(self, agent: Agent) -> bool:
        delivery = agent._delivery_activity
        result = (agent.result or "").strip()
        return bool(
            agent.tokens_consumed > 0
            and (
                result
                or agent.artifacts
                or delivery.candidate_output_files
                or delivery.expected_output_files
                or delivery.container_candidate_output_files
                or delivery.write_calls > 0
            )
        )

    def _agent_quantitative_signals(
        self,
        agent: Agent,
        state: StrategyState,
        turn_tools: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        delivery = agent._delivery_activity
        expected = _extract_expected_output_paths(agent.task)
        missing = self._missing_expected_outputs(agent)
        root_id = self._root_agent_id(agent)
        root_agent = self.agents.get(root_id, agent)
        child_agents = [self.agents[cid] for cid in agent.children if cid in self.agents]
        done_children = [child for child in child_agents if child.status in {"done", "failed"}]
        ineffective_done_children = [
            child.id for child in done_children if not self._agent_effective_evidence(child)
        ]
        active_children = [child for child in child_agents if child.status not in {"done", "failed"}]
        root_total_tokens = sum(max(0, a.tokens_consumed) for a in self.agents.values()) or 1
        agent_token_share = round(agent.tokens_consumed / root_total_tokens, 3)
        root_token_share = round(root_agent.tokens_consumed / root_total_tokens, 3)
        recent_unavailable = agent._shell_activity.unavailable_tool_calls
        workspace_candidates = len(delivery.candidate_output_files)
        container_candidates = len(delivery.container_candidate_output_files)
        local_candidates = workspace_candidates + len(agent.artifacts)
        root_candidates = state.shared_candidate_count + state.candidate_agent_count
        has_any_candidate = bool(local_candidates or container_candidates or root_candidates)
        has_expected = bool(delivery.expected_output_files) or (bool(expected) and not missing)
        delivery_gap = (
            bool(expected)
            and not has_expected
            and (
                agent._turns >= max(2, self.config.supervisor_min_child_turns)
                or delivery.write_calls > 0
                or state.delivery_phase in {"prepare", "consolidate", "verify", "deliver"}
            )
        )
        tool_mismatch = recent_unavailable >= 2 or (
            state.delivery_phase in {"prepare", "deliver", "verify"}
            and recent_unavailable >= 1
        )
        stalled_or_ineffective_branch = bool(ineffective_done_children) or (
            agent._turns >= max(1, self.config.supervisor_decomposition_min_turns)
            and not has_any_candidate
            and (
                agent.tokens_consumed >= max(0, self.config.supervisor_leaf_stall_tokens)
                or delivery.non_delivery_tool_calls >= 8
                or agent._tool_calls >= 10
            )
        )
        unverified_candidate = has_any_candidate and state.shared_verification_count == 0 and (
            state.delivery_phase in {"verify", "deliver"}
            or agent._turns >= max(1, self.config.supervisor_decomposition_min_turns)
            or delivery.candidate_ready_turn > 0
        )
        topology_bottleneck = (
            agent.parent is None
            and agent._turns >= max(1, self.config.supervisor_stall_turns)
            and (
                root_token_share >= 0.60
                or (
                    len(active_children) <= 1
                    and not has_any_candidate
                    and agent.tokens_consumed >= max(0, self.config.supervisor_decomposition_min_tokens)
                )
            )
        )
        path_misalignment = (
            workspace_candidates > 0
            and container_candidates == 0
            and any(str(path).startswith("/app/") for path in expected)
        )
        active_signals = [
            name
            for name, enabled in {
                "delivery_gap": delivery_gap,
                "tool_mismatch": tool_mismatch,
                "stalled_or_ineffective_branch": stalled_or_ineffective_branch,
                "unverified_candidate": unverified_candidate,
                "topology_bottleneck": topology_bottleneck,
            }.items()
            if enabled
        ]
        return {
            "active": active_signals,
            "delivery_gap": {
                "active": delivery_gap,
                "expected_paths": expected[:8],
                "missing_expected_paths": missing[:8],
                "expected_outputs_seen": len(delivery.expected_output_files),
                "workspace_candidates": workspace_candidates,
                "container_candidates": container_candidates,
                "write_calls": delivery.write_calls,
                "submit_calls": delivery.submit_calls,
                "delivery_phase": state.delivery_phase,
            },
            "tool_mismatch": {
                "active": tool_mismatch,
                "unavailable_tool_calls": recent_unavailable,
                "available_tools": sorted(turn_tools),
                "delivery_phase": state.delivery_phase,
            },
            "stalled_or_ineffective_branch": {
                "active": stalled_or_ineffective_branch,
                "ineffective_done_children": ineffective_done_children[:12],
                "turns": agent._turns,
                "tokens": agent.tokens_consumed,
                "tool_calls": agent._tool_calls,
                "non_delivery_tool_calls": delivery.non_delivery_tool_calls,
                "has_any_candidate": has_any_candidate,
            },
            "unverified_candidate": {
                "active": unverified_candidate,
                "has_any_candidate": has_any_candidate,
                "shared_verification_count": state.shared_verification_count,
                "verification_review_calls": state.verification_review_calls,
                "candidate_ready_turn": delivery.candidate_ready_turn,
            },
            "topology_bottleneck": {
                "active": topology_bottleneck,
                "root_token_share": root_token_share,
                "agent_token_share": agent_token_share,
                "active_children": len(active_children),
                "done_children": len(done_children),
                "total_agents": len(self.agents),
                "root_turns": root_agent._turns,
                "root_tokens": root_agent.tokens_consumed,
            },
            "path_misalignment": {
                "active": path_misalignment,
                "workspace_candidates": sorted(delivery.candidate_output_files)[:8],
                "container_candidates": sorted(delivery.container_candidate_output_files)[:8],
            },
        }

    def _build_supervisor_payload(
        self,
        agent: Agent,
        state: StrategyState,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> dict[str, Any]:
        root_id = self._root_agent_id(agent)
        root_agent = self.agents.get(root_id, agent)
        active = [a for a in self.agents.values() if a.status not in {"done", "failed"}]
        done = [a for a in self.agents.values() if a.status in {"done", "failed"}]
        edges = [
            [a.parent, a.id]
            for a in self.agents.values()
            if a.parent is not None
        ]
        tool_counts: dict[str, int] = {}
        for event in self._events:
            event_type = event.get("event", event.get("type"))
            if event_type != "tool_call":
                continue
            tool = (event.get("data") or {}).get("tool")
            if tool:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        infra_failed = self._infra_failed_agents()
        effective_child_count = sum(
            1
            for cid in root_agent.children
            if cid in self.agents and cid not in infra_failed
        )
        child_status = {
            status: sum(1 for cid in agent.children if cid in self.agents and self.agents[cid].status == status)
            for status in ("running", "idle", "done", "failed")
        }
        agent_candidate_files = [str(path.relative_to(agent.workspace)) for path in self._agent_candidate_files(agent)[:12]]
        quantitative_signals = self._agent_quantitative_signals(agent, state, turn_tools)
        supervisor_allowed_tools = sorted(self._all_tools())
        payload = {
            "current_agent": {
                "id": agent.id,
                "task": agent.task[:max(200, self.config.supervisor_payload_task_chars)],
                "bio": agent.bio[:500],
                "parent": agent.parent,
                "children": sorted(agent.children),
                "depth": agent.depth,
                "status": agent.status,
                "turns": agent._turns,
                "tokens": agent.tokens_consumed,
                "available_tools": supervisor_allowed_tools,
                "current_turn_tools": sorted(turn_tools),
                "tool_calls": agent._tool_calls,
                "no_tool_turns": agent._no_tool_turns,
                "child_status": child_status,
                "has_local_candidate": self._agent_has_candidate_file(agent),
                "candidate_files": agent_candidate_files,
                "candidate_outputs": len(agent._delivery_activity.candidate_output_files),
                "expected_outputs": len(agent._delivery_activity.expected_output_files),
                "container_candidate_outputs": len(agent._delivery_activity.container_candidate_output_files),
                "write_calls": agent._delivery_activity.write_calls,
                "submit_calls": agent._delivery_activity.submit_calls,
                "unavailable_tool_calls": agent._shell_activity.unavailable_tool_calls,
                "recent_messages": self._recent_agent_messages_for_supervisor(
                    agent, self.config.supervisor_history_messages
                ),
            },
            "topology": {
                "root": root_id,
                "total_agents": len(self.agents),
                "active_agents": len(active),
                "done_agents": len(done),
                "max_depth": max((a.depth for a in self.agents.values()), default=0),
                "edges": edges,
                "agents": [
                    self._agent_summary_for_supervisor(a, root_agent)
                    for a in sorted(self.agents.values(), key=lambda x: (x.depth, x.id))
                ],
            },
            "coordination_state": {
                "spawn_count": tool_counts.get("spawn", 0),
                "query_count": tool_counts.get("query", 0),
                "kill_count": tool_counts.get("kill", 0),
                "wait_count": tool_counts.get("wait", 0),
                "candidate_agent_count": state.candidate_agent_count,
                "candidate_query_coverage": state.candidate_query_coverage,
                "shared_candidate_count": state.shared_candidate_count,
                "shared_verification_count": state.shared_verification_count,
                "verification_review_calls": state.verification_review_calls,
                "resource_pressure": round(tool_policy.resource_pressure, 3),
                "delivery_phase": tool_policy.delivery_phase,
                "infra_failed_agents": sorted(infra_failed),
                "root_effective_children": effective_child_count,
            },
            "recent_events": self._recent_events_for_supervisor(self.config.supervisor_recent_events),
            "evidence_flow": self._supervisor_evidence_flow(
                agent, root_agent, self.config.supervisor_recent_events
            ),
            "quantitative_signals": quantitative_signals,
            "decision_contract": {
                "allowed_decisions": ["noop", "tool_override", "self_kill"],
                "allowed_tools": supervisor_allowed_tools,
                "notes": [
                    "Maintain evidence resolution for the current best candidate: support, counter-evidence, disagreement, unvalidated assumptions, evidence independence, and cost to resolve uncertainty.",
                    "Forecast near-future network progress before choosing an action; prefer topology changes over more local computation when the forecast is drift.",
                    "Do not finish while unresolved material counter-evidence threatens the deliverable, unless it is stale, low-confidence, non-actionable, or outweighed by stronger independent evidence.",
                    "Do not treat a single self-consistent reasoning chain as resolved when a cheap independent check could catch a material error.",
                    "If uncertainty is missing evidence, choose query or wait; if it is branch disagreement, query branches or spawn a short adjudicator only when existing evidence cannot resolve it.",
                    "If a candidate failed, route failure evidence to the most capable existing branch; spawn repair only when no existing branch can act on it.",
                    "If the main risk is unvalidated assumptions, choose the cheapest independent validation path: query existing evidence, wait, direct check, or a short validation branch.",
                    "If a branch consumes tokens without adding independent evidence, query if uncertain, then kill, wait, or narrow the work with a shorter branch.",
                    "Choose finish when the required deliverable exists and remaining uncertainty is low, non-material, or more expensive to resolve than the expected correctness gain.",
                    "Choose exactly one next topology behavior.",
                    "For spawn, allowed_tools must contain only spawn_many or spawn.",
                    "For query/wait/send/kill, allowed_tools must contain only that action tool.",
                    "For finish, include at least one concrete delivery tool.",
                    "Return compact JSON; omit task-solving content.",
                ],
            },
        }
        return payload

    def _extract_supervisor_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        candidates = [text]
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fence:
            candidates.insert(0, fence.group(1))
        brace = re.search(r"(\{.*\})", text, re.S)
        if brace:
            candidates.append(brace.group(1))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _supervisor_decision_from_json(
        self,
        item: dict[str, Any],
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        payload: dict[str, Any],
    ) -> StrategyDecision | None:
        decision = str(item.get("decision") or item.get("action") or "noop").strip().lower()
        if decision in {"", "none", "noop", "no_op"}:
            return None
        if decision not in {"tool_override", "self_kill"}:
            return None
        target_id = str(item.get("target_agent_id") or agent.id).strip()
        if target_id not in self.agents:
            target_id = agent.id
        if self.config.supervisor_root_only and self.agents[target_id].parent is not None:
            return None
        raw_tools = item.get("allowed_tools") or item.get("tools") or []
        if not isinstance(raw_tools, list):
            raw_tools = []
        current_available = set(turn_tools)
        all_available = set(self._all_tools()) - set(self.config.disabled_tools)
        available = set(all_available)
        if not available:
            available = set(current_available)
        raw_tool_names = [str(t) for t in raw_tools]
        if self.config.supervisor_one_step_topology:
            topology_action = str(item.get("topology_action") or "").strip().lower()
            if topology_action in {"noop", "no_op", "none"}:
                return None
            if not topology_action:
                for name in raw_tool_names:
                    if name in {"spawn", "query", "wait", "kill", "send", "set_status"}:
                        topology_action = {
                            "set_status": "finish",
                        }.get(name, name)
                        break
            one_step_tools: set[str]
            spawn_count = item.get("spawn_count")
            try:
                spawn_count_value = int(spawn_count)
            except (TypeError, ValueError):
                spawn_count_value = 0
            if decision == "self_kill" or topology_action == "kill":
                one_step_tools = {"kill"}
            elif topology_action == "spawn" or (not topology_action and "spawn" in raw_tool_names):
                one_step_tools = {"spawn_many" if spawn_count_value > 1 and "spawn_many" in available else "spawn"}
            elif topology_action == "query" or (not topology_action and "query" in raw_tool_names):
                one_step_tools = {"query"}
            elif topology_action == "wait" or (not topology_action and "wait" in raw_tool_names):
                one_step_tools = {"wait"}
            elif topology_action == "send" or (not topology_action and "send" in raw_tool_names):
                one_step_tools = {"send"}
            elif topology_action == "finish" or "set_status" in raw_tool_names:
                requested_finish_tools = {
                    name for name in raw_tool_names
                    if name in _SUPERVISOR_FINISH_TOOLS
                }
                delivery_tools = requested_finish_tools & _SUPERVISOR_DELIVERY_TOOLS & available
                if not delivery_tools:
                    delivery_tools = set()
                    for name in _SUPERVISOR_FINISH_TOOL_ORDER:
                        if name in available and name in _SUPERVISOR_DELIVERY_TOOLS:
                            delivery_tools.add(name)
                            break
                one_step_tools = delivery_tools | {
                    name for name in ("tb_read_file", "ws_read_file", "ws_grep", "set_status")
                    if name in available
                }
            else:
                one_step_tools = set(raw_tool_names) & (
                    set(_SUPERVISOR_TOPOLOGY_TOOL_ORDER) | _SUPERVISOR_FINISH_TOOLS
                )
            if topology_action == "finish" or (
                "set_status" in raw_tool_names and one_step_tools & _SUPERVISOR_FINISH_TOOLS
            ):
                tool_order = _SUPERVISOR_FINISH_TOOL_ORDER
            else:
                tool_order = (*_SUPERVISOR_TOPOLOGY_TOOL_ORDER, *_SUPERVISOR_FINISH_TOOL_ORDER)
            seen_tools: set[str] = set()
            tools = []
            for name in tool_order:
                if name in seen_tools:
                    continue
                if name in one_step_tools and name in available:
                    tools.append(name)
                    seen_tools.add(name)
            if topology_action == "finish" and not (set(tools) & _SUPERVISOR_DELIVERY_TOOLS):
                self._emit(agent.id, "supervisor_unexecutable_finish", {
                    "target_agent_id": target_id,
                    "raw_allowed_tools": raw_tool_names,
                    "available_tools": sorted(available),
                    "turn_tools": sorted(current_available),
                    "reason": "finish action has no executable delivery tool",
                })
                return None
            if topology_action == "finish":
                evidence_flow = payload.get("evidence_flow") or {}
                evidence_flags = evidence_flow.get("flags") or {}
                if (
                    evidence_flags.get("has_recent_failure")
                    and not evidence_flags.get("has_recent_successful_verification")
                ):
                    self._emit(agent.id, "supervisor_finish_safety_blocked", {
                        "target_agent_id": target_id,
                        "reason": "recent failure evidence without later successful verification",
                        "raw_allowed_tools": raw_tool_names,
                    })
                    return None
        else:
            tools = [name for name in raw_tool_names if name in available]
        if decision == "self_kill" and not tools:
            tools = ["kill"] if "kill" in available else []
        if decision == "tool_override" and not tools:
            return None
        if decision == "tool_override" and any(t in tools for t in ("spawn", "spawn_many")):
            spawn_tool = "spawn_many" if "spawn_many" in tools else "spawn"
            tools = [spawn_tool]
        if (
            decision == "tool_override"
            and not self.config.supervisor_one_step_topology
            and "spawn" not in tools
            and "set_status" in available
            and "set_status" not in tools
        ):
            tools.append("set_status")
        if (
            decision == "tool_override"
            and "get_cost" in available
            and "get_cost" not in tools
            and not self.config.supervisor_one_step_topology
        ):
            tools.append("get_cost")

        message = str(item.get("message") or "").strip()
        if self.config.supervisor_one_step_topology:
            topology_action = str(item.get("topology_action") or "").strip().lower()
            spawn_count = item.get("spawn_count")
            try:
                spawn_count = int(spawn_count)
            except (TypeError, ValueError):
                spawn_count = None
            allowed_text = ", ".join(tools)
            if "spawn" in tools or "spawn_many" in tools:
                count_text = f" Supervisor estimated spawn_count={spawn_count}." if spawn_count else ""
                decomposition_bits: list[str] = []
                for key, label in (("spawn_units", "Spawn units"),):
                    value = item.get(key)
                    if isinstance(value, list) and value:
                        try:
                            text = json.dumps(value[:6], ensure_ascii=False)
                        except TypeError:
                            text = str(value[:6])
                        decomposition_bits.append(f"{label}: {text}")
                decomposition_text = ""
                if decomposition_bits:
                    decomposition_text = " " + " ".join(decomposition_bits)
                message = (
                    "Single-step topology intervention: choose spawn as the next action."
                    f"{count_text} Create child branches according to the supervisor's one-step topology decision. "
                    "Do not solve locally on this turn. Use the available spawn tool now."
                    f"{decomposition_text}"
                )
            elif decision == "self_kill" or "kill" in tools:
                message = (
                    "Single-step topology intervention: inspect/query evidence if needed, then kill only a "
                    "clearly low-value branch. Do not modify the task answer on this turn."
                )
            elif any(t in tools for t in ("query", "wait", "send")):
                action_text = topology_action or "query/wait"
                message = (
                    f"Single-step topology intervention: use {action_text} next to collect or route branch "
                    "evidence. If there is a failed verification, candidate conflict, or unqueried "
                    "finished branch, gather or forward that exact evidence before doing any local work."
                )
            elif topology_action == "finish" or any(t in tools for t in ("tb_write_file", "ws_create_file", "submit", "set_status")):
                message = (
                    "Single-step topology intervention: finish the current deliverable now. "
                    "Use the available read/write delivery tools to create or update the required final artifact, "
                    "then call set_status(done, result=...) or submit. Do not start new exploration or spawn new agents. "
                    "Only finish if branch evidence is consistent and no recent verification failure contradicts the artifact."
                )
            else:
                message = (
                    f"Single-step topology intervention: restrict the next turn to structural tools "
                    f"[{allowed_text}]. Do not include task-specific answer content."
                )
        else:
            spawn_plan = item.get("spawn_plan")
            if spawn_plan:
                try:
                    plan_text = json.dumps(spawn_plan, ensure_ascii=False, indent=2)
                except TypeError:
                    plan_text = str(spawn_plan)
                message = f"{message}\n\nSupervisor spawn/task plan:\n{plan_text}".strip()
        if self.config.supervisor_one_step_topology and len(message) > 700:
            message = message[:700]
        if len(message) > 3500:
            message = message[:3500]
        reason = str(item.get("reason") or "topology-aware supervisor decision").strip()
        if len(reason) > 800:
            reason = reason[:800]
        action = "SUPERVISOR_SELF_KILL" if decision == "self_kill" else str(item.get("strategy_action") or "SUPERVISOR_TOOL_OVERRIDE")
        if "spawn" in tools or "spawn_many" in tools:
            action = "SUPERVISOR_SPAWN"
        action_signature = (
            target_id,
            str(item.get("topology_action") or "").strip().lower(),
            tuple(tools),
            self._supervisor_progress_signature(self.agents.get(target_id, agent)),
        )
        previous_signature = self._last_supervisor_action_by_agent.get(target_id)
        previous_turn = self._last_supervisor_action_turn_by_agent.get(target_id, -10**9)
        target_turn = self.agents.get(target_id, agent)._turns
        duplicate_cooldown = max(2, int(getattr(self.config, "supervisor_stall_interval", 3) or 3))
        if (
            previous_signature == action_signature
            and not any(t in tools for t in ("spawn", "spawn_many", "kill"))
            and target_turn - previous_turn < duplicate_cooldown
        ):
            repeats = self._supervisor_action_repeat_by_agent.get(target_id, 0) + 1
            self._supervisor_action_repeat_by_agent[target_id] = repeats
            if repeats >= 1:
                self._supervisor_skip(agent, "duplicate_supervisor_action")
                self._emit(agent.id, "supervisor_duplicate_action_suppressed", {
                    "target_agent_id": target_id,
                    "topology_action": action_signature[1],
                    "tools": list(tools),
                    "repeat_count": repeats,
                    "cooldown_turns": duplicate_cooldown,
                })
                return None
        else:
            self._supervisor_action_repeat_by_agent[target_id] = 0
        self._last_supervisor_action_by_agent[target_id] = action_signature
        self._last_supervisor_action_turn_by_agent[target_id] = self.agents.get(target_id, agent)._turns
        return StrategyDecision(
            source="opus_topology_supervisor",
            action=action,
            target_agent_id=target_id,
            tools=tuple(tools),
            reason=reason,
            message=message,
            once=bool(item.get("once", True)),
            metadata={
                "tool_learning": {
                    "tool_family": "delete" if decision == "self_kill" else "control",
                    "tool_name": "kill" if decision == "self_kill" else ("spawn" if "spawn" in tools else "tool_override"),
                    "supervisor_model": self.config.supervisor_model,
                    "topology_aware": True,
                },
                "forecast": str(item.get("forecast") or "")[:500],
                "risk": str(item.get("risk") or "")[:80],
                "payload_state": {
                    "current_agent": payload.get("current_agent", {}).get("id"),
                    "total_agents": payload.get("topology", {}).get("total_agents"),
                    "active_agents": payload.get("topology", {}).get("active_agents"),
                    "shared_candidate_count": payload.get("coordination_state", {}).get("shared_candidate_count"),
                    "shared_verification_count": payload.get("coordination_state", {}).get("shared_verification_count"),
                },
                "raw_supervisor": item,
            },
        )

    def _supervisor_event_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self._events:
            event_type = event.get("event", event.get("type"))
            if event_type in {
                "agent_new", "spawn", "done", "failed", "send",
                "strategy_decision", "runtime_intervention",
            }:
                counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def _infra_failed_agents(self) -> set[str]:
        failed: set[str] = set()
        for aid, agent in self.agents.items():
            if agent.status != "failed":
                continue
            result = (agent.result or "").lower()
            if (
                "503 service unavailable" in result
                or "429" in result
                or "rate limit" in result
                or "server error" in result
            ) and agent.tokens_consumed == 0:
                failed.add(aid)
        return failed

    def _supervisor_skip(self, agent: Agent, reason: str) -> None:
        self._supervisor_skips[reason] = self._supervisor_skips.get(reason, 0) + 1
        self._last_supervisor_reason_by_agent[agent.id] = reason

    def _supervisor_decision_point(
        self,
        agent: Agent,
        state: StrategyState,
        tool_policy: ToolPolicyState,
    ) -> tuple[bool, str, tuple[Any, ...]]:
        mode = (self.config.supervisor_trigger_mode or "decision_points").strip().lower()
        if mode in {"always", "every_turn", "all"}:
            return True, "always", ("always", agent._turns)
        if mode in {"off", "manual", "none"}:
            return False, "trigger_mode_off", ()
        decomposition_mode = mode in {"task_decomposition", "decomposition", "work_inventory"}

        event_counts = self._supervisor_event_counts()
        infra_failed = self._infra_failed_agents()
        active_children = state.active_children
        child_count = len(agent.children)
        effective_child_count = sum(
            1
            for cid in agent.children
            if cid in self.agents and cid not in infra_failed
        )
        root_has_children = child_count > 0
        root_candidates = state.shared_candidate_count + state.candidate_agent_count
        has_verification = state.shared_verification_count > 0
        done_agents = sum(1 for a in self.agents.values() if a.status in {"done", "failed"})
        failed_agents = sum(1 for a in self.agents.values() if a.status == "failed")
        marker = (
            child_count,
            active_children,
            done_agents,
            failed_agents,
            root_candidates,
            state.candidate_query_coverage,
            state.shared_verification_count,
            state.delivery_phase,
            event_counts.get("send", 0),
            event_counts.get("spawn", 0),
            event_counts.get("failed", 0),
            agent._delivery_activity.write_calls,
            agent._delivery_activity.candidate_ready_turn,
            agent._shell_activity.unavailable_tool_calls,
            len(agent._delivery_activity.container_candidate_output_files),
        )

        last_marker = self._last_supervisor_marker_by_agent.get(agent.id)
        topology_changed = last_marker is not None and marker != last_marker
        first_root_window = (
            agent.parent is None
            and agent._turns <= max(0, self.config.supervisor_initial_root_turns)
            and "spawn" in state.available_tools
        )
        new_branch_evidence = (
            agent.parent is None
            and root_has_children
            and topology_changed
            and (
                done_agents > 0
                or failed_agents > 0
                or event_counts.get("send", 0) > 0
                or root_candidates > 0
                or has_verification
            )
        )
        needs_initial_branching = (
            agent.parent is None
            and effective_child_count == 0
            and agent._turns <= max(2, self.config.supervisor_stall_turns)
            and "spawn" in state.available_tools
            and root_candidates == 0
            and not has_verification
        )
        root_infra_branches_failed = (
            agent.parent is None
            and child_count > 0
            and effective_child_count == 0
            and bool(infra_failed.intersection(agent.children))
            and root_candidates == 0
            and not has_verification
            and "spawn" in state.available_tools
            and topology_changed
        )
        root_stalled_without_candidates = (
            agent.parent is None
            and agent._turns >= max(1, self.config.supervisor_stall_turns)
            and root_candidates == 0
            and not has_verification
            and (
                not root_has_children
                or active_children == 0
                or failed_agents > 0
            )
            and agent._turns % max(1, self.config.supervisor_stall_interval) == 0
        )
        quantitative_signals = self._agent_quantitative_signals(agent, state, state.available_tools)
        signal_names = list(quantitative_signals.get("active") or [])
        root_needs_evidence_routing = (
            agent.parent is None
            and root_candidates > 0
            and (
                state.candidate_query_coverage < state.candidate_agent_count
                or (
                    state.shared_candidate_count > 0
                    and state.shared_verification_count == 0
                )
            )
            and topology_changed
        )
        needs_candidate_forecast = (
            "unverified_candidate" in signal_names
            and agent._turns >= max(0, self.config.supervisor_min_child_turns)
            and (
                topology_changed
                or agent._delivery_activity.candidate_ready_turn > 0
                or state.candidate_query_coverage < state.candidate_agent_count
                or state.shared_verification_count == 0
            )
        )
        leaf_tool_stuck = (
            agent.parent is not None
            and not state.has_local_candidate
            and state.shared_candidate_count == 0
            and agent._shell_activity.unavailable_tool_calls >= 3
            and agent._turns >= max(0, self.config.supervisor_min_child_turns)
            and (
                topology_changed
                or agent._turns % max(1, self.config.supervisor_stall_interval) == 0
            )
        )
        leaf_long_without_candidate = (
            agent.parent is not None
            and not state.has_local_candidate
            and state.shared_candidate_count == 0
            and agent._turns >= max(0, self.config.supervisor_min_child_turns)
            and (
                agent.tokens_consumed >= max(0, self.config.supervisor_leaf_stall_tokens)
                or state.delivery_phase in {"prepare", "deliver"}
                or agent._delivery_activity.write_calls > 0
            )
            and agent._turns % max(1, self.config.supervisor_stall_interval) == 0
        )
        noisy_quantitative_signal = False
        if self.config.supervisor_quantitative_combo_required:
            signal_set = set(signal_names)
            noisy_quantitative_signal = (
                signal_set == {"delivery_gap"}
                or (
                    signal_set == {"delivery_gap", "unverified_candidate"}
                    and not topology_changed
                    and state.shared_verification_count == 0
                    and state.candidate_query_coverage >= state.candidate_agent_count
                )
            )
        quantitative_signal_point = (
            decomposition_mode
            and bool(signal_names)
            and not noisy_quantitative_signal
            and not state.pending_override
            and (
                topology_changed
                or agent._turns % max(1, self.config.supervisor_stall_interval) == 0
                or "tool_mismatch" in signal_names
            )
        )

        if first_root_window:
            reason = "initial_root_topology"
        elif needs_initial_branching:
            reason = "early_root_no_children"
        elif root_infra_branches_failed:
            reason = "infra_branches_failed"
        elif new_branch_evidence:
            reason = "branch_evidence_changed"
        elif root_stalled_without_candidates:
            reason = "root_stalled_without_candidates"
        elif root_needs_evidence_routing:
            reason = "candidate_evidence_routing"
        elif needs_candidate_forecast:
            reason = "candidate_forecast"
        elif quantitative_signal_point:
            reason = "quantitative_" + "_".join(signal_names[:3])
        elif leaf_tool_stuck:
            reason = "leaf_tool_stuck"
        elif leaf_long_without_candidate:
            reason = "leaf_long_without_candidate"
        else:
            return False, "not_decision_point", marker

        previous_action = self._last_supervisor_action_by_agent.get(agent.id)
        if previous_action is not None:
            previous_topology_action = previous_action[1]
            previous_progress = previous_action[3]
            previous_turn = self._last_supervisor_action_turn_by_agent.get(agent.id, -10**9)
            if (
                previous_topology_action in {"wait", "query", "finish"}
                and previous_progress == self._supervisor_progress_signature(agent)
                and agent._turns - previous_turn < max(2, self.config.supervisor_stall_interval)
                and reason not in {
                    "branch_evidence_changed",
                    "candidate_evidence_routing",
                    "candidate_forecast",
                    "infra_branches_failed",
                }
            ):
                return False, "duplicate_supervisor_progress", marker

        if last_marker == marker and reason == self._last_supervisor_reason_by_agent.get(agent.id):
            return False, f"duplicate_{reason}", marker
        noop_streak = self._supervisor_noop_streak_by_agent.get(agent.id, 0)
        if noop_streak >= max(1, self.config.supervisor_noop_backoff_threshold) and reason not in {
            "branch_evidence_changed",
            "candidate_evidence_routing",
            "candidate_forecast",
            "root_stalled_without_candidates",
            "infra_branches_failed",
        }:
            return False, "noop_backoff", marker
        return True, reason, marker

    def _supervisor_fallback_decision(
        self,
        agent: Agent,
        trigger_reason: str,
        payload: dict[str, Any],
        turn_tools: dict[str, dict[str, Any]],
    ) -> StrategyDecision | None:
        if not self.config.supervisor_error_fallback_spawn:
            return None
        if agent.parent is not None:
            return None
        if trigger_reason not in {"initial_root_topology", "early_root_no_children", "infra_branches_failed"}:
            return None
        if "spawn" not in turn_tools:
            return None
        return StrategyDecision(
            source="opus_topology_supervisor_fallback",
            action="SUPERVISOR_SPAWN",
            target_agent_id=agent.id,
            tools=tuple(t for t in ("spawn", "get_cost") if t in turn_tools),
            reason=f"Supervisor unavailable at {trigger_reason}; local topology fallback keeps multi-agent branching alive.",
            message=(
                "Single-step topology fallback: choose spawn as the next action. "
                "Create complementary child branches according to uncovered work units. "
                "Do not solve locally on this turn."
            ),
            once=True,
            metadata={
                "tool_learning": {
                    "tool_family": "control",
                    "tool_name": "spawn",
                    "supervisor_model": self.config.supervisor_model,
                    "topology_aware": True,
                    "fallback": True,
                },
                "payload_state": {
                    "current_agent": payload.get("current_agent", {}).get("id"),
                    "total_agents": payload.get("topology", {}).get("total_agents"),
                    "active_agents": payload.get("topology", {}).get("active_agents"),
                    "shared_candidate_count": payload.get("coordination_state", {}).get("shared_candidate_count"),
                    "shared_verification_count": payload.get("coordination_state", {}).get("shared_verification_count"),
                },
            },
        )

    async def _maybe_schedule_supervisor_strategy(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> None:
        if not self.config.supervisor_enabled:
            return
        if agent.status != "running" or agent.id in self._pending_tool_overrides:
            return
        if agent.parent is not None and agent._turns < max(0, self.config.supervisor_min_child_turns):
            return
        if self.config.supervisor_root_only and agent.parent is not None:
            return
        if self.config.supervisor_max_calls > 0 and self._supervisor_calls >= self.config.supervisor_max_calls:
            return
        backoff_until = self._supervisor_error_backoff_until_by_agent.get(agent.id, -10**9)
        if agent._turns < backoff_until:
            self._supervisor_skip(agent, "error_backoff")
            return
        last_turn = self._last_supervisor_turn_by_agent.get(agent.id, -10**9)
        if agent._turns - last_turn < max(1, self.config.supervisor_min_turn_interval):
            self._supervisor_skip(agent, "min_turn_interval")
            return

        state = self._build_strategy_state(agent, turn_tools, tool_policy)
        should_call, trigger_reason, trigger_marker = self._supervisor_decision_point(agent, state, tool_policy)
        if not should_call:
            self._supervisor_skip(agent, trigger_reason)
            if trigger_marker:
                self._last_supervisor_marker_by_agent.setdefault(agent.id, trigger_marker)
            return
        payload = self._build_supervisor_payload(agent, state, turn_tools, tool_policy)
        payload["supervisor_trigger"] = {
            "reason": trigger_reason,
            "marker": list(trigger_marker),
        }
        system = (
            "You are a topology-aware multi-agent runtime supervisor. "
            "You do not solve the task or provide domain content. Your job is to preserve "
            "evidence resolution in the agent network. For the current best candidate, reason "
            "about supporting evidence, material counter-evidence, unresolved branch disagreement, "
            "unvalidated assumptions, independence of evidence, and the token value of reducing "
            "uncertainty. Forecast the likely near-future progress of the current topology from "
            "agent states and evidence flow, then choose exactly one next structural tool behavior. "
            "Prefer topology changes only when they resolve valuable uncertainty, add independent "
            "evidence, cover independent work, adjudicate competing candidates, or avoid long-agent "
            "drift. "
            "Return one compact JSON object only."
        )
        user = (
            "Given this NanoMA runtime snapshot, decide whether the current agent should "
            "continue normally or be restricted to exactly one next topology action: spawn, "
            "query, send, wait, kill, finish, or noop. "
            "First forecast what will probably happen if no intervention is made: will the "
            "network converge, waste tokens in a long branch, need more branch evidence, need "
            "verification/adjudication, need a repair branch, or be ready to deliver? Base the "
            "action on that forecast, not only on the current quantitative signal. "
            "Before choosing the action, resolve these questions: "
            "1) What candidate outcome currently appears best? "
            "2) What evidence supports it? "
            "3) Which assumptions behind it remain unvalidated, and are they material? "
            "4) What counter-evidence or unresolved disagreement materially threatens it? "
            "5) Is an independent validation signal available at a token cost justified by the possible correctness gain? "
            "6) Which single topology action best resolves the highest-value uncertainty? "
            "Do not finish while unresolved material counter-evidence threatens the current deliverable, "
            "unless that counter-evidence is stale, low-confidence, non-actionable, or outweighed by "
            "stronger independent evidence. Do not treat a candidate as resolved merely because one "
            "agent's reasoning is internally consistent. When the current confidence rests on a single "
            "evidence chain, the possible error would materially change the outcome, and a low-cost "
            "independent validation signal is available, obtain that signal before finish. Use the "
            "cheapest suitable path: query or wait for existing evidence, route evidence with send, "
            "perform a direct check when finish tools allow it, or spawn one short validation/adjudication "
            "branch only when existing evidence cannot resolve the uncertainty. If uncertainty is caused "
            "by a failed candidate, route the failure to the most capable existing branch; spawn repair "
            "only when no existing branch can act on it. If a branch is consuming tokens without adding "
            "independent evidence, query its state, then kill or narrow it. Finish when the remaining "
            "uncertainty is low, non-material, independently covered enough, or more expensive to resolve "
            "than the expected correctness gain. "
            "For allowed_tools, list only tools needed for that topology action. For spawn, "
            "use only spawn_many or spawn. For query, only query. For send, only send. For wait, only wait. "
            "For finish, include a concrete delivery tool such as tb_write_file, ws_create_file, "
            "shell/tb_shell, or submit. Do not add diagnostic or implementation tools to a "
            "topology action. Use noop if topology is already adequate. "
            "Do not include hidden tests, verifier logs, solution paths, task-specific algorithms, "
            "or multi-step plans. Avoid /tmp unless the public task instruction explicitly names it.\n\n"
            "Output schema:\n"
            "{\n"
            '  "decision": "noop|tool_override|self_kill",\n'
            '  "target_agent_id": "agent id",\n'
            '  "allowed_tools": ["tool", "..."],\n'
            '  "topology_action": "spawn|query|send|wait|kill|finish|noop",\n'
            '  "spawn_count": 0,\n'
            '  "spawn_units": [{"unit": "short work unit", "deliverable": "artifact/patch/report/verdict"}],\n'
            '  "forecast": "one sentence: likely next progress without intervention",\n'
            '  "risk": "serial_drift|missing_branch|missing_evidence|unvalidated_assumption|single_chain_evidence|material_counter_evidence|candidate_conflict|uncertainty_not_worth_cost|ready_to_deliver|adequate",\n'
            '  "message": "one short imperative topology instruction",\n'
            '  "reason": "short rationale linking forecast to action",\n'
            '  "once": true\n'
            "}\n\n"
            f"Snapshot JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        supervisor_retry = RetryConfig(
            max_retries=max(0, self.config.supervisor_max_retries),
            base_delay=self.config.retry.base_delay,
            max_delay=self.config.retry.max_delay,
            http_timeout=max(1.0, self.config.supervisor_http_timeout),
        )
        try:
            protocol = self.config.supervisor_protocol.strip().lower()
            if protocol in {"anthropic", "anthropic-compatible", "messages"}:
                response = await anthropic_compatible_call(
                    messages,
                    self.config.supervisor_model,
                    tools=None,
                    base_url=self.config.supervisor_base_url,
                    api_key=self.config.supervisor_api_key,
                    temperature=self.config.supervisor_temperature,
                    max_tokens=self.config.supervisor_max_tokens,
                    retry_config=supervisor_retry,
                )
            else:
                response = await openai_compatible_call(
                    messages,
                    self.config.supervisor_model,
                    tools=None,
                    base_url=self.config.supervisor_base_url,
                    api_key=self.config.supervisor_api_key,
                    temperature=self.config.supervisor_temperature,
                    max_tokens=self.config.supervisor_max_tokens,
                    retry_config=supervisor_retry,
                )
        except Exception as exc:
            self._supervisor_errors += 1
            self._supervisor_error_backoff_until_by_agent[agent.id] = (
                agent._turns + max(1, self.config.supervisor_error_backoff_turns)
            )
            self._emit(agent.id, "opus_supervisor_error", {"error": f"{type(exc).__name__}: {exc}"})
            fallback = self._supervisor_fallback_decision(agent, trigger_reason, payload, turn_tools)
            if fallback is not None:
                self._supervisor_decisions[fallback.action] = self._supervisor_decisions.get(fallback.action, 0) + 1
                self._register_strategy_decision(fallback)
                self._emit(agent.id, "strategy_decision", fallback.as_event())
            if self.config.supervisor_fail_open:
                return
            raise

        cost = response.usage.cost_usd()
        self._supervisor_calls += 1
        self._supervisor_tokens += response.usage.total_tokens
        self._supervisor_cost += cost
        self._last_supervisor_turn_by_agent[agent.id] = agent._turns
        raw_text = response.content or ""
        parsed = self._extract_supervisor_json(raw_text)
        self._emit(agent.id, "opus_supervisor_done", {
            "model": self.config.supervisor_model,
            "tokens": response.usage.total_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost": round(cost, 6),
            "content_preview": raw_text[:500],
            "parsed": parsed if parsed is not None else None,
            "trigger_reason": trigger_reason,
        })
        if parsed is None:
            self._supervisor_noop_streak_by_agent[agent.id] = (
                self._supervisor_noop_streak_by_agent.get(agent.id, 0) + 1
            )
            self._last_supervisor_marker_by_agent[agent.id] = trigger_marker
            self._last_supervisor_reason_by_agent[agent.id] = trigger_reason
            return
        decision = self._supervisor_decision_from_json(parsed, agent, turn_tools, payload)
        if decision is None:
            self._supervisor_decisions["noop"] = self._supervisor_decisions.get("noop", 0) + 1
            self._supervisor_noop_streak_by_agent[agent.id] = (
                self._supervisor_noop_streak_by_agent.get(agent.id, 0) + 1
            )
            self._last_supervisor_marker_by_agent[agent.id] = trigger_marker
            self._last_supervisor_reason_by_agent[agent.id] = trigger_reason
            return
        self._supervisor_noop_streak_by_agent[agent.id] = 0
        self._last_supervisor_marker_by_agent[agent.id] = trigger_marker
        self._last_supervisor_reason_by_agent[agent.id] = trigger_reason
        self._supervisor_decisions[decision.action] = self._supervisor_decisions.get(decision.action, 0) + 1
        self._register_strategy_decision(decision)
        self._emit(agent.id, "strategy_decision", decision.as_event())

    def _maybe_schedule_runtime_strategy(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> None:
        fixed_context = self._fixed_spawn_context(agent)
        if fixed_context is not None:
            checkpoint, _specs, metadata = fixed_context
            self._emit(agent.id, "runtime_strategy_suppressed_by_fixed_checkpoint", {
                "checkpoint": checkpoint.key,
                "spawn_tool": metadata.get("spawn_tool"),
                "turn": agent._turns,
            })
            return

        state = self._build_strategy_state(agent, turn_tools, tool_policy)
        if self.config.strategy_log_events:
            self._emit(agent.id, "strategy_state", state.as_event())

        for strategy in self._strategies:
            decision = strategy.decide(state)
            if decision is None:
                continue
            if decision.action in {
                StrategyAction.SPAWN_PORTFOLIO,
                StrategyAction.IMPROVE_CANDIDATE,
                StrategyAction.SPAWN_RECOVERY,
                StrategyAction.SPAWN_VERIFIER,
            }:
                if "spawn" in self.config.disabled_tools:
                    continue
                if self._spawn_unavailable_reason(agent) is not None:
                    continue
            self._register_strategy_decision(decision)
            if decision.source == "low_value_agent_kill":
                self._auto_kill_requested.add(decision.target_agent_id)
                self._emit(agent.id, "auto_kill_scheduled", {
                    "reason": decision.reason,
                    "tools": list(decision.tools),
                    "shared_candidates": state.shared_candidate_count,
                    "turns": state.turns,
                    "tokens": state.tokens,
                    "strategy_action": decision.action,
                })
            elif decision.source == "spawn_portfolio":
                self._emit(agent.id, "spawn_portfolio_scheduled", {
                    "reason": decision.reason,
                    "tools": list(decision.tools),
                    "total_agents": state.total_agents,
                    "active_children": state.active_children,
                    "shared_candidates": state.shared_candidate_count,
                    "strategy_action": decision.action,
                })
            elif decision.source == "root_recovery_spawn":
                self._emit(agent.id, "root_recovery_spawn_scheduled", {
                    "reason": decision.reason,
                    "tools": list(decision.tools),
                    "total_agents": state.total_agents,
                    "active_children": state.active_children,
                    "shared_candidates": state.shared_candidate_count,
                    "attempts": state.root_recovery_spawn_attempts,
                    "strategy_action": decision.action,
                })
            elif decision.source == "candidate_evidence_gate":
                self._emit(agent.id, "candidate_evidence_gate_scheduled", {
                    "reason": decision.reason,
                    "tools": list(decision.tools),
                    "shared_candidates": state.shared_candidate_count,
                    "shared_verification": state.shared_verification_count,
                    "active_children": state.active_children,
                    "attempts": state.candidate_evidence_gate_attempts,
                    "strategy_action": decision.action,
                })
            elif decision.source == "final_evidence_gate":
                self._emit(agent.id, "final_evidence_gate_scheduled", {
                    "reason": decision.reason,
                    "tools": list(decision.tools),
                    "shared_candidates": state.shared_candidate_count,
                    "candidate_agents": state.candidate_agent_count,
                    "candidate_query_coverage": state.candidate_query_coverage,
                    "shared_verification": state.shared_verification_count,
                    "verification_review_calls": state.verification_review_calls,
                    "delivery_phase": state.delivery_phase,
                    "attempts": state.candidate_evidence_gate_attempts,
                    "strategy_action": decision.action,
                })
            elif decision.source == "candidate_finalize":
                self._emit(agent.id, "candidate_finalize_scheduled", {
                    "reason": decision.reason,
                    "tools": list(decision.tools),
                    "shared_candidates": state.shared_candidate_count,
                    "best_shared_candidate_bytes": state.best_shared_candidate_bytes,
                    "turns": state.turns,
                    "strategy_action": decision.action,
                })
            self._emit(agent.id, "strategy_decision", decision.as_event())
            break

    def _tool_policy_score(self, tool_name: str, state: ToolPolicyState) -> float:
        if state.reconcile_phase:
            if tool_name in {"ws_read_file", "ws_grep"}:
                return 1.1
            if tool_name in {"ws_create_file", "ws_append_file", "ws_replace_string"}:
                return 1.0
            if tool_name in {"query", "spawn", "send", "wait"}:
                return 0.9
            if tool_name == "set_status":
                return 0.75
            if tool_name == "shell":
                return 0.65
            if tool_name == "get_cost":
                return 0.5

        if state.delivery_phase == "prepare":
            if tool_name == "deliver_to_parent":
                return 1.3
            if tool_name in {"ws_create_file", "ws_append_file"}:
                return 1.25
            if tool_name in {"ws_replace_string", "ws_multi_replace", "ws_apply_patch"}:
                return 1.05
            if tool_name == "submit":
                return 0.96
            if tool_name == "set_status":
                return 0.92
            if tool_name in {"ws_read_file", "ws_grep"}:
                return 0.84
            if tool_name == "shell":
                return 0.72
            if tool_name == "get_cost":
                return 0.25

        if state.delivery_phase == "verify":
            if tool_name == "deliver_to_parent":
                return 1.15
            if tool_name in {"ws_read_file", "ws_grep"}:
                return 1.2
            if tool_name == "shell":
                return 1.05
            if tool_name in {"ws_replace_string", "ws_multi_replace", "ws_apply_patch", "ws_create_file", "ws_append_file"}:
                return 0.95
            if tool_name == "get_cost":
                return 0.45
            if tool_name == "submit":
                return 0.18
            if tool_name == "set_status":
                return 0.16

        if state.delivery_phase == "deliver":
            if tool_name == "deliver_to_parent":
                return 1.35
            if tool_name == "submit":
                return 1.25
            if tool_name == "set_status":
                return 1.2
            if tool_name in {"ws_create_file", "ws_append_file", "ws_replace_string", "ws_multi_replace", "ws_apply_patch"}:
                return 1.1
            if tool_name == "ws_read_file":
                return 0.82
            if tool_name == "shell":
                return 0.55
            if tool_name == "query":
                return 0.45
            if tool_name == "get_cost":
                return 0.25

        scores: list[float] = []
        if tool_name in _CREATE_TOOLS:
            scores.append(state.create)
        if tool_name in _COORDINATION_TOOLS:
            scores.append(state.message)
        if tool_name in _READ_TOOLS:
            scores.append(state.read)
        if tool_name in _WORK_TOOLS:
            scores.append(state.work)
        if tool_name in _FINISH_TOOLS:
            scores.append(state.finish)
        elif tool_name in _LIFECYCLE_TOOLS:
            scores.append(max(state.read, state.finish))
        return max(scores, default=0.0)

    def _order_tools_by_policy(
        self,
        tools: dict[str, dict[str, Any]],
        state: ToolPolicyState,
        original_order: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        # Structural steering only: preferred tools appear earlier in the schema
        # list. No policy text or weights are injected into the agent loop.
        names = sorted(
            tools,
            key=lambda name: (
                -self._tool_policy_score(name, state),
                original_order.get(name, len(original_order)),
                name,
            ),
        )
        return {name: tools[name] for name in names}

    def _prune_tools_by_policy(
        self,
        tools: dict[str, dict[str, Any]],
        state: ToolPolicyState,
    ) -> tuple[dict[str, dict[str, Any]], set[str]]:
        """Optionally shrink the tool schema based on runtime pressure.

        This is structural steering only: no policy text or weights are added to
        the agent history. Tools have already been ordered by policy score, so
        pruning keeps the strongest prefix plus a small set of finalization
        primitives needed to produce an answer.
        """
        if not tools:
            return tools, set()

        start = self.config.tool_policy_prune_pressure_start
        end = self.config.tool_policy_prune_pressure_end
        pressure = max(state.resource_pressure, state.finish, state.delivery_pressure, state.web_loop_pressure)
        if pressure < start:
            return tools, set()

        if end <= start:
            ratio = 1.0
        else:
            ratio = _clamp01((pressure - start) / (end - start))

        total = len(tools)
        min_tools = max(1, self.config.tool_policy_prune_min_tools)
        target = math.ceil(total - ratio * max(0, total - min_tools))
        target = max(1, min(total, target))

        preserve = set(self.config.tool_policy_prune_preserve_tools)
        if state.reconcile_phase:
            preserve.discard("submit")
            preserve.update({
                "shell", "tb_shell", "ws_read_file", "tb_read_file", "ws_grep",
                "ws_create_file", "ws_append_file", "tb_write_file", "query", "set_status",
            })
        if state.delivery_phase == "prepare":
            preserve.discard("get_cost")
            preserve.update({
                "shell", "tb_shell", "ws_read_file", "tb_read_file", "ws_grep",
                "ws_create_file", "ws_append_file", "tb_write_file", "submit", "set_status",
            })
        if state.delivery_phase == "deliver":
            preserve.discard("get_cost")
            preserve.update({
                "shell", "tb_shell", "set_status", "submit", "ws_read_file", "tb_read_file",
                "ws_create_file", "ws_append_file", "tb_write_file",
            })
        keep: set[str] = {name for name in tools if name in preserve}

        for name in tools:
            if len(keep) >= target:
                break
            keep.add(name)

        pruned = {name for name in tools if name not in keep}
        if not pruned:
            return tools, set()

        scoped = {name: tool for name, tool in tools.items() if name in keep}
        if not scoped:
            fallback = {
                name: tool
                for name, tool in tools.items()
                if name in _LIFECYCLE_TOOLS or name in _FINISH_TOOLS
            }
            scoped = fallback or tools
            pruned = set(tools) - set(scoped)
        return scoped, pruned

    def current_tool_policy(self, agent: Agent) -> dict[str, Any]:
        all_tools = self._all_tools()
        _, state = self._apply_state_tool_policy(agent, all_tools)
        return state.as_event()

    def spawn_policy_violation(self, agent: Agent) -> str | None:
        if self._fixed_spawn_context(agent) is not None:
            return None
        if agent.id in self._strategy_spawn_authorized:
            return None
        if self.config.tool_policy_mode == "off":
            return None
        unavailable = self._spawn_unavailable_reason(agent)
        if unavailable:
            return unavailable
        if self._low_concurrency_bypass_active(agent):
            return None
        state = self._compute_tool_policy_state(agent)
        if state.reconcile_phase:
            return "web loop reconcile active"
        if state.resource_pressure >= self.config.tool_policy_resource_threshold:
	            return f"resource pressure {state.resource_pressure:.2f} >= {self.config.tool_policy_resource_threshold:.2f}"
        if state.dependency_pressure >= self.config.tool_policy_dependency_threshold:
            return f"dependency pressure {state.dependency_pressure:.2f} >= {self.config.tool_policy_dependency_threshold:.2f}"
        if self.config.tool_policy_mode == "enforce" and state.create < self.config.tool_policy_spawn_min_weight:
            return f"create weight {state.create:.2f} < {self.config.tool_policy_spawn_min_weight:.2f}"
        return None

    def _spawn_unavailable_reason(self, agent: Agent) -> str | None:
        if "spawn" in self.config.disabled_tools:
            return "spawn disabled"
        if agent.depth + 1 > self.config.max_depth:
            return f"max depth {self.config.max_depth} reached"
        if len(self.agents) >= self.config.max_agents:
            return f"max agents {self.config.max_agents} reached"
        return None

    @staticmethod
    def _parse_spawn_judge(text: str | None) -> dict | None:
        """Extract the judge's decision JSON from a (possibly fenced) response."""
        import json
        import re
        if not text:
            return None
        raw = str(text).strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
        else:
            i, j = raw.find("{"), raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                raw = raw[i:j + 1]
        try:
            data = json.loads(raw)
        except Exception:
            return None
        if not isinstance(data, dict) or "spawn" not in data:
            return None
        subs = data.get("subagents") or []
        clean_subs = []
        if isinstance(subs, list):
            for s in subs:
                if isinstance(s, dict) and str(s.get("task") or "").strip():
                    clean_subs.append({
                        "subject": str(s.get("subject") or "").strip(),
                        "task": str(s.get("task")).strip(),
                        "role": str(s.get("role") or "").strip().lower(),
                    })
                elif isinstance(s, str) and s.strip():
                    clean_subs.append({"subject": "", "task": s.strip(), "role": ""})
        return {
            "spawn": bool(data.get("spawn")),
            "reasoning": str(data.get("reasoning") or "").strip(),
            "subagents": clean_subs,
        }

    def _spawn_judge_model(self, agent: "Agent") -> str:
        """Use the working agent's model for its topology decision."""
        return str(getattr(agent, "model", "") or self.config.default_model or "").strip()

    def _render_history_for_judge(self, agent: "Agent", max_chars: int = 48000) -> str:
        """Serialize the parent agent's working context (exactly what IT sees).

        The spawn judge is given the same context as the parent, not a thin
        checklist, so its decision is grounded in what has actually been explored.
        """
        parts: list[str] = []
        for m in getattr(agent, "history", []) or []:
            if not isinstance(m, dict):
                parts.append(str(m))
                continue
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                segs = []
                for c in content:
                    if isinstance(c, dict):
                        segs.append(str(c.get("text") or c.get("content") or ""))
                    else:
                        segs.append(str(c))
                content = "\n".join(s for s in segs if s)
            content = str(content or "")
            if m.get("tool_calls"):
                try:
                    names = ", ".join(
                        (tc.get("function", {}) or {}).get("name", "?")
                        for tc in m["tool_calls"]
                    )
                except Exception:
                    names = ""
                if names:
                    content = (content + f"\n[tool_calls: {names}]").strip()
            if not content:
                continue
            parts.append(f"[{role}] {content}")
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = "...(earlier context truncated)...\n\n" + text[-max_chars:]
        return text

    async def _spawn_judge_at_plan(self, agent: Agent, plan_hint: str = "") -> bool:
        """At a fresh planning moment, let the working model decide whether to spawn.

        Called from `meta_task_create` BEFORE any todolist is materialized, so the
        list is only ever created on the no-spawn path (decide first, plan second).
        The judge sees the SAME working context the parent sees (agent.history),
        not just a checklist. Children are NOT isolated: each can call
        query(agent_id='<parent>', messages=-1) to read the parent's context, so
        assignments need not restate everything. The judge is told NOT to reason
        about cost/budget and to lean toward spawning (speed matters). Returns True
        iff children were spawned. Best-effort: any failure degrades to "no spawn".
        """
        import os
        if os.environ.get("NANOMA_SPAWN_TODOLIST_JUDGE") != "1":
            return False
        remaining_slots = max(0, self.config.max_agents - len(self.agents))
        if remaining_slots <= 0:
            return False
        # Hard runtime constraint: benchmark work runs in this container's memory
        # cgroup, so bound the fan-out regardless of what the judge wants.
        cap = self._max_parallel_children()
        if cap is not None:
            remaining_slots = min(remaining_slots, max(0, cap - self._active_children_total()))
            if remaining_slots <= 0:
                self._emit(agent.id, "spawn_judge_skipped", {
                    "reason": "memory_capacity",
                    "active_children": self._active_children_total(),
                    "cap": cap,
                })
                return False
        # Don't re-fan-out while this agent already has children in flight; let it
        # integrate their results first.
        active_children = [
            a for a in self.agents.values()
            if getattr(a, "parent", None) == agent.id
            and getattr(a, "status", None) not in ("done", "failed", "killed")
        ]
        if active_children:
            return False
        judge_model = self._spawn_judge_model(agent)
        if not judge_model:
            return False

        context_text = self._render_history_for_judge(agent)
        sys_msg = (
            "You are the planning judge for a multi-agent research system. At a "
            "planning moment you decide whether the parent agent should fan its next "
            "phase out to parallel child agents, and if so how to split it.\n"
            "Children run in PARALLEL. A child starts in a fresh context but is NOT "
            "isolated: it can call query(agent_id='<parent>', messages=-1) to read the "
            "parent's full context and findings on demand. So an assignment does NOT "
            "need to restate everything — give a clear objective plus what to pull from "
            "the parent.\n"
            "Spawn is worth doing in ANY of these cases:\n"
            "1) DECOMPOSITION: the phase splits into parts that can progress at the same time.\n"
            "2) PARALLEL REASONING & VERIFICATION: the answer/artifact is uncertain or "
            "error-prone and benefits from being produced and independently cross-checked. "
            "Spawn a solver plus verifier(s) that use a materially different method/source.\n"
            "3) CONTEXT ISOLATION: a part will generate a large volume of intermediate "
            "material; offload it to a child that returns only a distilled result.\n"
            "4) METHOD PORTFOLIO / EXPLORATION: the best approach is uncertain; spawn one "
            "explorer per materially different approach and keep the best.\n"
            "Every child already receives its OWN private copy of the task directory and the "
            "runtime folds each child's changes back safely, so children never collide — you "
            "do not need to tell them to copy files or avoid each other.\n"
            "Bias toward spawning: parallel children finish sooner and can cross-check "
            "each other, so when parallelism is even plausibly useful, spawn. Do NOT "
            "reason about token cost, money, or budget — that is handled by the runtime, "
            "not you, and speed is what matters. Only decline when the next phase is a "
            "genuinely single, indivisible step where nothing could be gained by any "
            "parallel worker, verifier, or explorer. Reply with STRICT JSON only, no prose."
        )
        user_msg = (
            f"Parent task:\n{agent.task}\n\n"
            f"Parent agent id (children query this): {agent.id}\n\n"
            "Parent's current working context (this is exactly what the parent sees):\n"
            f"{context_text}\n\n"
            + (f"The parent is about to plan this next: {plan_hint}\n\n" if plan_hint else "")
            + f"You may spawn up to {remaining_slots} parallel children.\n\n"
            "Return JSON exactly like:\n"
            '{"spawn": true|false, "reasoning": "<one sentence>", '
            '"subagents": [{"subject": "<short label>", "role": "solver|verifier|explorer|worker", '
            '"task": "<objective + what to query from the parent>"}]}\n'
            "For a verification split give the verifier(s) a DIFFERENT method/source from "
            "the solver; for an exploration split give each explorer a DIFFERENT approach. "
            "If spawn is false, return an empty subagents list."
        )

        decision = None
        try:
            resp = await self.llm_call(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                judge_model,
                None,
            )
            decision = self._parse_spawn_judge(getattr(resp, "content", None))
        except Exception as exc:
            self._emit(agent.id, "spawn_judge_error", {"model": judge_model, "detail": str(exc)[:300]})

        if not decision:
            self._emit(agent.id, "spawn_judge_decision", {
                "model": judge_model, "spawn": False, "reason": "no/invalid judge output", "children": 0,
            })
            return False

        subs = decision["subagents"] if decision["spawn"] else []
        subs = subs[:remaining_slots]
        self._emit(agent.id, "spawn_judge_decision", {
            "model": judge_model,
            "spawn": bool(decision["spawn"]) and bool(subs),
            "reasoning": decision.get("reasoning", "")[:300],
            "requested_children": len(decision["subagents"]),
            "children": len(subs),
            "roles": [s.get("role") or "worker" for s in subs],
        })
        if not subs:
            return False  # no-spawn: caller proceeds to create the todolist

        spawned_any = False
        for sub in subs:
            child_task = (
                str(sub.get("task", "")).strip()
                + f"\n\n[Context] You are a child of agent '{agent.id}'. Its full working "
                f"context and findings are available on demand: call "
                f'query(agent_id="{agent.id}", messages=-1) to read them (or messages=N '
                f"for the last N). Pull what you need instead of restating."
            )
            try:
                result = await self._invoke_meta_spawn(
                    {"task": child_task, "model": judge_model},
                    agent,
                )
            except Exception as exc:
                self._emit(agent.id, "spawn_judge_error", {"stage": "spawn", "detail": str(exc)[:300]})
                break
            if not isinstance(result, dict) or result.get("error"):
                self._emit(agent.id, "spawn_judge_error", {"stage": "spawn", "detail": str(result)[:300]})
                break
            spawned_any = True
        return spawned_any

    def _delivery_orchestration_report(self, parent: Agent, delivered: Agent) -> str:
        """What else is in flight, and where contributions overlap.

        The judge cannot reason about whether a delivery needs cross-checking
        without knowing what the siblings are doing to the same files. Overlap is
        the strongest signal there is: on ann_vector_search_qps four children
        each rewrote the same two files, and the combination — which none of them
        had ever run — was what got submitted.
        """
        baseline = self._merge_baseline_path
        siblings = [
            a for a in self.agents.values()
            if getattr(a, "parent", None) == parent.id and a.id != delivered.id
        ]
        touched: dict[str, set[str]] = {}
        lines: list[str] = []
        for agent in [delivered, *sorted(siblings, key=lambda a: a.id)]:
            files, deleted = self._merge_own_changes(agent)
            changed = set(files) | set(deleted)
            for rel in changed:
                touched.setdefault(rel, set()).add(agent.id)
            spec = getattr(agent, "_verify_spec", None)
            promoted = getattr(agent, "_merge_promoted_sig", None) is not None
            lines.append(
                f"- {agent.id} [{agent.status}]"
                f" task: {str(agent.task or '')[:200].replace(chr(10), ' ')}\n"
                f"    changed files: {sorted(changed)[:8] or 'none'}"
                f" | delivered already: {promoted}"
                f" | own registered check: {(spec or {}).get('name') or 'NONE'}"
                f" | its last measurement: {self._merge_last_child_metric.get(agent.id)}"
            )
        overlaps = {rel: sorted(who) for rel, who in touched.items() if len(who) > 1}
        report = "Agents on this fan-out round:\n" + "\n".join(lines)
        report += (
            f"\n\nFiles more than one agent changed: "
            f"{json.dumps(overlaps) if overlaps else 'none'}"
        )
        report += (
            f"\n\nBest state measured so far: {self._merge_best_metric}"
            f"\nA check the runtime can re-run against the submission: "
            f"{(self._verification_spec_for(parent) or {}).get('name') or 'NONE REGISTERED'}"
        )
        return report

    def _delivery_spawn_in_flight(self, parent: Agent) -> bool:
        return any(
            getattr(a, "_spawned_at_delivery", False)
            and getattr(a, "status", None) not in ("done", "failed", "killed")
            for a in self.agents.values()
        )

    async def _spawn_judge_at_delivery(
        self, delivered: Agent, delivery: dict | None
    ) -> bool:
        """At a delivery, let the judge decide whether to spawn a cross-checker.

        The planning moment is not the only place worth a spawn decision. A
        delivery is when sibling contributions land on the same files and the
        merged artifact becomes something no single agent has run — and the
        agent best placed to sort that out is one assigned to do exactly that,
        not the runtime on a timer. Roles are assignments, not agent classes:
        this spawns an ordinary agent whose task happens to be verification.

        Its deliverable is a check registered with `verify`, so the conclusion
        becomes something the runtime can re-run rather than prose in a message.
        Best-effort: any failure degrades to no spawn.
        """
        if os.environ.get("NANOMA_SPAWN_TODOLIST_JUDGE") != "1":
            return False
        if not self._merge_active():
            return False
        parent_id = getattr(delivered, "parent", None)
        parent = self.agents.get(parent_id) if parent_id else None
        if parent is None or parent.status in ("done", "failed", "killed"):
            return False
        if not delivery or delivery.get("status") != "delivered":
            return False
        if self._delivery_spawn_in_flight(parent):
            self._emit(delivered.id, "spawn_judge_skipped", {
                "stage": "delivery", "reason": "verifier_already_running",
            })
            return False
        remaining_slots = max(0, self.config.max_agents - len(self.agents))
        cap = self._max_parallel_children()
        if cap is not None:
            remaining_slots = min(
                remaining_slots, max(0, cap - self._active_children_total())
            )
        if remaining_slots <= 0:
            self._emit(delivered.id, "spawn_judge_skipped", {
                "stage": "delivery", "reason": "capacity",
                "active_children": self._active_children_total(), "cap": cap,
            })
            return False

        judge_model = self._spawn_judge_model(parent)
        if not judge_model:
            return False
        report = self._delivery_orchestration_report(parent, delivered)
        sys_msg = (
            "You are the orchestration judge for a multi-agent engineering system. A "
            "child agent has just delivered work into the shared submission. You decide "
            "whether to spawn an agent to independently cross-check the current merged "
            "state, and what to assign it.\n"
            "Why this moment matters: each agent verifies its own private copy, but the "
            "submission is the MERGE of several contributions, which no one has run. "
            "When several agents change the same files, a merged result can fail while "
            "every individual contribution passed.\n"
            "A cross-checking assignment is worth it when ANY of these hold:\n"
            "1) OVERLAP: more than one agent changed the same files, so the combination "
            "is untested.\n"
            "2) NO RUNNABLE CHECK: no check is registered, or the registered one fails to "
            "run against the merged submission — nothing can measure what ships.\n"
            "3) UNCONFIRMED CLAIM: the delivery reports a result that has not been "
            "reproduced against the merged state by anyone else.\n"
            "4) SIBLINGS STILL IN FLIGHT: more deliveries are coming that will land on the "
            "same files, so a check that works is needed before they do.\n"
            "Decline when the merged state already verifies cleanly with a registered "
            "check and no sibling is touching the same files.\n"
            "The agent you spawn is an ordinary agent: it gets its own private copy of the "
            "submission and can call query(agent_id='<id>', messages=-1) on the parent or "
            "any sibling to read their context. Its DELIVERABLE is a check registered via "
            "the verify tool that runs against the submission and prints a machine-readable "
            "verdict — a reproducible check, not a written opinion. If the merged state is "
            "broken it should identify which contribution broke it. Do not reason about "
            "cost or budget. Reply with STRICT JSON only, no prose."
        )
        user_msg = (
            f"Overall task:\n{str(parent.task or '')[:2000]}\n\n"
            f"Parent agent id: {parent.id}\n"
            f"Just delivered: {delivered.id}\n"
            f"Its delivery outcome: verified={delivery.get('verified')} "
            f"metric={delivery.get('metric')} kept={delivery.get('kept')}\n"
            f"Files it changed: {delivery.get('changed_files')}\n"
            f"Verification output from the merged state:\n"
            f"{str(delivery.get('verification_output') or '(none)')[:1500]}\n\n"
            f"{report}\n\n"
            f"You may spawn up to {remaining_slots} agent(s); one is usually enough.\n\n"
            "Return JSON exactly like:\n"
            '{"spawn": true|false, "reasoning": "<one sentence>", '
            '"subagents": [{"subject": "<short label>", "role": "verifier", '
            '"task": "<what to check, against what, and what to query from whom>"}]}\n'
            "If spawn is false, return an empty subagents list."
        )

        decision = None
        try:
            resp = await self.llm_call(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                judge_model,
                None,
            )
            decision = self._parse_spawn_judge(getattr(resp, "content", None))
        except Exception as exc:
            self._emit(delivered.id, "spawn_judge_error", {
                "stage": "delivery", "model": judge_model, "detail": str(exc)[:300],
            })
        if not decision:
            return False
        subs = (decision["subagents"] if decision["spawn"] else [])[:remaining_slots]
        self._emit(delivered.id, "spawn_judge_decision", {
            "stage": "delivery",
            "model": judge_model,
            "spawn": bool(decision["spawn"]) and bool(subs),
            "reasoning": decision.get("reasoning", "")[:300],
            "children": len(subs),
        })
        if not subs:
            return False

        spawned_any = False
        for sub in subs:
            task = (
                str(sub.get("task", "")).strip()
                + f"\n\n[Context] You are a child of agent '{parent.id}'. Agent "
                f"'{delivered.id}' just delivered into the shared submission. Read any "
                f'agent\'s context on demand with query(agent_id="<id>", messages=-1).'
                "\n[Deliverable] A check registered with the verify tool, printing a final "
                'line like {"ok": true, "metric": <number>}. The runtime re-runs it '
                "against the merged submission on every later delivery, so it outlasts "
                "you — getting it to run reliably is worth more than a verdict in prose."
            )
            try:
                result = await self._invoke_meta_spawn(
                    {"task": task, "model": judge_model},
                    parent,
                )
            except Exception as exc:
                self._emit(delivered.id, "spawn_judge_error", {
                    "stage": "delivery_spawn", "detail": str(exc)[:300],
                })
                break
            if not isinstance(result, dict) or result.get("error"):
                self._emit(delivered.id, "spawn_judge_error", {
                    "stage": "delivery_spawn", "detail": str(result)[:300],
                })
                break
            for child_id in self._spawn_tool_created_ids_from_result(result):
                child = self.agents.get(child_id)
                if child is not None:
                    child._spawned_at_delivery = True
            spawned_any = True
        return spawned_any

    @staticmethod
    def _spawn_tool_created_ids_from_result(result: dict) -> list[str]:
        ids = []
        for key in ("agent_id", "child", "child_id", "id"):
            value = result.get(key)
            if isinstance(value, str) and value:
                ids.append(value)
        children = result.get("children")
        if isinstance(children, list):
            ids.extend(c for c in children if isinstance(c, str))
        return ids

    async def _invoke_meta_spawn(self, spawn_args: dict, agent: Agent) -> dict:
        from nanoma.meta import meta_spawn
        return await meta_spawn(spawn_args, agent, self)

    # ─── merge-submit-path orchestration ─────────────────────────────────
    # The shared submit path (workspace_extra_roots[0], e.g. EdgeBench's
    # task_cwd) is a single resource that `submit` is hard-wired to. To let
    # parallel children contribute without colliding, each child works on a
    # private copy and the runtime folds the child's diff back on completion,
    # serialized by a lock and gated by the submission score (keep only if it
    # did not regress). This is role-agnostic: the child's assigned role only
    # shaped its task text, never this plumbing.
    #
    # Two independent notions, deliberately kept apart:
    #   working copy  — what the child runs in. Must be the COMPLETE task
    #                   directory (harness, datasets, deps), or the child cannot
    #                   build, test or score anything. See _merge_clone_workdir.
    #   merge scope   — which files count as a contribution, i.e. the submission
    #                   subpaths. Only this is snapshotted, diffed and applied.
    #                   See _merge_scope_roots / _merge_rel_files.
    # Conflating them is what turned a 509MB ann-benchmarks tree into either an
    # OOM (copy everything) or unrunnable 12KB stubs (copy only the scope).

    def _merge_active(self) -> bool:
        import os
        if os.environ.get("NANOMA_MERGE_SUBMIT_PATH") != "1":
            return False
        if self._merge_disabled_reason:
            return False
        return self._merge_target() is not None

    def _merge_scope_roots(self) -> tuple[Path, ...]:
        """Relative subpaths that merge is restricted to (the submission paths).

        Reads NANOMA_MERGE_PATHS, else SFORGE_SUBMIT_PATHS (which EdgeBench
        already injects). Without this, a task directory that embeds datasets
        (e.g. ann-benchmarks: 509MB of `data/` for a 12KB submission) would be
        copied per child and hashed on every diff. An empty result means the
        whole tree, which is only safe for small task directories.
        """
        import os
        raw = os.environ.get("NANOMA_MERGE_PATHS") or os.environ.get("SFORGE_SUBMIT_PATHS") or ""
        roots: list[Path] = []
        for token in raw.replace(",", " ").split():
            token = token.strip().strip('"').strip("'")
            if not token:
                continue
            candidate = Path(token)
            if candidate.is_absolute() or ".." in candidate.parts:
                continue
            posix = candidate.as_posix().rstrip("/")
            if posix in {"", ".", "./"}:
                return ()  # "." = submit everything; fall back to the whole tree
            roots.append(Path(posix))
        return tuple(roots)

    def _merge_scope_stats(self, root: Path) -> tuple[int, int]:
        """(total_bytes, file_count) of the merge scope under `root`."""
        total = 0
        count = 0
        for path in self._merge_rel_files(root).values():
            try:
                if path.is_symlink():
                    continue
                total += path.stat().st_size
                count += 1
            except OSError:
                continue
        return total, count

    def _merge_clone_cost(self, source: Path) -> tuple[int, int]:
        """(bytes actually duplicated, file count) for one complete working copy.

        Only files below the hardlink threshold are duplicated; larger ones are
        linked, so they cost nothing. Explicit read-only dependency/cache roots
        are represented by one symlink and pruned from the walk. Stat-only walk,
        no reads.
        """
        duplicated = 0
        count = 0
        for current, dirnames, filenames in os.walk(source):
            relative_dir = Path(current).relative_to(source)
            dirnames[:] = [
                name for name in dirnames
                if not self._merge_clone_skip(name)
                and not self._merge_clone_path_is_shared(relative_dir / name)
            ]
            for name in filenames:
                if self._merge_clone_skip(name):
                    continue
                if self._merge_clone_path_is_shared(relative_dir / name):
                    continue
                path = Path(current) / name
                try:
                    if path.is_symlink():
                        continue
                    size = path.stat().st_size
                except OSError:
                    continue
                count += 1
                if size < _MERGE_HARDLINK_MIN_BYTES:
                    duplicated += size
        return duplicated, count

    def _merge_check_copy_budget(self, target: Path) -> bool:
        """Disable merge (once) if a per-child working copy is too expensive.

        Measures what a copy actually costs (duplicated bytes, not the tree size)
        so a task that is large only because of linkable datasets stays eligible.
        """
        import os
        max_mb = float(os.environ.get("NANOMA_MERGE_MAX_MB", "200") or 200)
        max_files = int(os.environ.get("NANOMA_MERGE_MAX_FILES", "40000") or 40000)
        duplicated, count = self._merge_clone_cost(target)
        too_big = max_mb > 0 and duplicated > max_mb * 1024 * 1024
        too_many = max_files > 0 and count > max_files
        if not (too_big or too_many):
            return True
        self._merge_disabled_reason = (
            f"per-child working copy too expensive: {duplicated / (1024 * 1024):.1f}MB "
            f"duplicated / {count} files (limits {max_mb}MB / {max_files} files)"
        )
        self._emit("root", "merge_disabled", {
            "reason": self._merge_disabled_reason,
            "scope": [p.as_posix() for p in self._merge_scope_roots()] or ["<whole tree>"],
            "shared_clone_roots": [
                p.as_posix() for p in self._merge_shared_clone_roots()
            ],
            "hint": "raise NANOMA_MERGE_MAX_MB / NANOMA_MERGE_MAX_FILES to re-enable",
        })
        return False

    def _merge_target(self) -> Path | None:
        if self.config.workspace_extra_roots:
            p = Path(self.config.workspace_extra_roots[0]).expanduser()
            if p.is_dir():
                return p
        return None

    def _merge_get_lock(self) -> "asyncio.Lock":
        if self._merge_lock is None:
            self._merge_lock = asyncio.Lock()
        return self._merge_lock

    def _merge_root_dir(self) -> Path:
        return self.config.workspace_root / "_merge"

    @staticmethod
    def _merge_ignored_part(name: str) -> bool:
        return name in {
            ".git", ".lake", ".nanoma-runtime-logs", ".nanoma-task-work",
            "__pycache__", "_task_copy", "_merge",
        }

    @staticmethod
    def _merge_clone_skip(name: str) -> bool:
        """Excluded from a child's working copy: runtime artefacts only.

        Narrower than _merge_ignored_part on purpose — build metadata such as
        .git and .lake is never submitted but is often needed to run the task.
        """
        return name in {
            ".nanoma-runtime-logs", ".nanoma-task-work", "__pycache__",
            "_task_copy", "_merge",
        }

    def _merge_shared_clone_roots(self) -> tuple[Path, ...]:
        """Large immutable roots that child worktrees may reference by symlink.

        ``NANOMA_MERGE_SHARED_PATHS`` is an explicit, whitespace/comma-separated
        allowlist relative to the complete task directory. It is intended for
        dependency caches such as Lean's ``.lake/packages`` and for baseline
        trees outside the submission scope. Those trees make a runnable clone
        several gigabytes and hundreds of thousands of files even though the
        agent must never submit or modify them.

        A path is accepted only when it is outside every submitted scope or is
        already excluded from merge diffs (for example, a path below ``.lake``).
        This prevents an accidental setting from sharing editable deliverables.
        """
        raw = os.environ.get("NANOMA_MERGE_SHARED_PATHS", "")
        if not raw.strip():
            return ()
        scopes = self._merge_scope_roots()
        roots: list[Path] = []
        for token in raw.replace(",", " ").split():
            candidate = Path(token.strip().strip('"').strip("'"))
            if (
                not candidate.parts
                or candidate.is_absolute()
                or ".." in candidate.parts
                or candidate.as_posix() in {"", ".", "./"}
            ):
                continue
            ignored_by_merge = any(
                self._merge_ignored_part(part) for part in candidate.parts
            )
            overlaps_scope = not scopes or any(
                candidate == scope
                or scope in candidate.parents
                or candidate in scope.parents
                for scope in scopes
            )
            if overlaps_scope and not ignored_by_merge:
                continue
            if candidate not in roots:
                roots.append(candidate)
        return tuple(roots)

    def _merge_clone_path_is_shared(self, relative: Path) -> bool:
        """Whether ``relative`` is the root of an explicitly shared subtree."""
        return relative in self._merge_shared_clone_roots()

    def _merge_restore_snapshot(self, snapshot: Path, target: Path) -> None:
        """Overlay a snapshot back onto the submission path, within the scope.

        Deliberately not a copy_tree: that clears the destination first, and the
        destination here holds the harness, the datasets and every child's
        working copy. Restoring by diff touches only the scoped files.
        """
        changed, deleted = self._merge_diff(snapshot, target)
        if changed or deleted:
            self._merge_apply(changed, deleted, target)

    def _merge_copy_tree(self, source: Path, dest: Path) -> bool:
        """Snapshot the merge scope of `source` into a fresh `dest`.

        For snapshots only — `dest` is cleared first, so it must never be the
        live submission path. Use _merge_restore_snapshot to go the other way.
        """
        target = self._merge_target()
        if target is not None and dest.resolve() == target.resolve():
            logger.error(f"refusing to clear the live submission path: {dest}")
            return False
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            roots = self._merge_scope_roots()
            if not roots:
                shutil.copytree(
                    source, dest,
                    ignore=self._fixed_snapshot_ignore, symlinks=True,
                )
                return True
            dest.mkdir(parents=True, exist_ok=True)
            for relative in roots:
                item = source / relative
                if not item.exists() and not item.is_symlink():
                    continue
                self._fixed_copy_snapshot_item(item, dest / relative)
            return True
        except Exception as exc:
            logger.warning(f"merge copy_tree failed ({source} -> {dest}): {exc}")
            return False

    def _merge_ensure_baseline(self) -> Path | None:
        """Snapshot the submit path as the common ancestor for a fan-out round.

        Reused while any merge-child is still active (so siblings diff against
        the same ancestor); re-snapshotted at the start of a fresh round.
        """
        target = self._merge_target()
        if target is None:
            return None
        active = [
            a for a in self.agents.values()
            if getattr(a, "_merge_copy", None)
            and getattr(a, "status", None) not in ("done", "failed", "killed")
        ]
        existing = self._merge_baseline_path
        if existing and existing.is_dir() and active:
            return existing
        if not self._merge_check_copy_budget(target):
            return None
        dest = self._merge_root_dir() / "baseline"
        if not self._merge_copy_tree(target, dest):
            return existing
        self._merge_baseline_path = dest
        self._emit("root", "merge_baseline_snapshot", {"path": str(dest), "source": str(target)})
        return dest

    def _merge_clone_workdir(self, source: Path, dest: Path) -> tuple[int, int] | None:
        """Clone the COMPLETE task directory as a child's working copy.

        Directories are always real, so files the child creates or removes stay
        local (build output, logs, benchmark results). Small files are duplicated
        so every edit a child could plausibly make is isolated; files at or above
        _MERGE_HARDLINK_MIN_BYTES are hardlinked, which keeps the tree complete
        and runnable at nearly zero cost. Explicit immutable cache/dependency
        roots are symlinked back to the common task tree. Returns
        (duplicated_bytes, private_files).
        """
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"merge clone init failed ({dest}): {exc}")
            return None
        duplicated = 0
        files = 0
        for current, dirnames, filenames in os.walk(source):
            relative = Path(current).relative_to(source)
            target_dir = dest / relative
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                dirnames[:] = []
                continue
            private_dirs: list[str] = []
            for name in dirnames:
                if self._merge_clone_skip(name):
                    continue
                child_relative = relative / name
                if self._merge_clone_path_is_shared(child_relative):
                    try:
                        os.symlink(
                            str((source / child_relative).resolve()),
                            target_dir / name,
                            target_is_directory=True,
                        )
                    except Exception as exc:
                        logger.warning(
                            "merge shared-root link failed (%s): %s",
                            child_relative,
                            exc,
                        )
                        return None
                    continue
                private_dirs.append(name)
            dirnames[:] = private_dirs
            for name in filenames:
                if self._merge_clone_skip(name):
                    continue
                src = Path(current) / name
                dst = target_dir / name
                try:
                    if self._merge_clone_path_is_shared(relative / name):
                        os.symlink(str(src.resolve()), dst)
                        continue
                    if src.is_symlink():
                        os.symlink(os.readlink(src), dst)
                        files += 1
                        continue
                    size = src.stat().st_size
                    if size >= _MERGE_HARDLINK_MIN_BYTES:
                        try:
                            os.link(src, dst)
                        except OSError:
                            # different filesystem or link limit: fall back to a copy
                            shutil.copy2(src, dst)
                            duplicated += size
                    else:
                        shutil.copy2(src, dst)
                        duplicated += size
                    files += 1
                except Exception:
                    continue
        return duplicated, files

    def _merge_seed_child_copy(self, agent: Agent) -> Path | None:
        """Give a freshly created child a complete, private, runnable task copy."""
        baseline = self._merge_ensure_baseline()
        if baseline is None:
            return None
        target = self._merge_target()
        if target is None:
            return None
        copy_dir = agent.workspace / "_task_copy"
        cost = self._merge_clone_workdir(target, copy_dir)
        if cost is None:
            return None
        # Rewind just the diffable surface to the round baseline, so the child's
        # diff carries its own work and not drift that landed after the snapshot.
        for relative in self._merge_scope_roots():
            source = baseline / relative
            if not source.exists() and not source.is_symlink():
                continue
            destination = copy_dir / relative
            self._fixed_remove_path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._fixed_copy_snapshot_item(source, destination)
        agent._merge_copy = copy_dir
        self._emit(agent.id, "merge_child_seed", {
            "copy": str(copy_dir), "duplicated_bytes": cost[0], "files": cost[1],
            "shared_clone_roots": [
                path.as_posix() for path in self._merge_shared_clone_roots()
            ],
        })
        return copy_dir

    def _merge_agent_baseline(self, agent: Agent) -> Path | None:
        """What this agent was handed, which is what its diff is measured against.

        Usually the round baseline. It differs once a copy has been refreshed to
        a later state of the submission: from then on the agent's own work is
        what it changed since that refresh, not since the run began.
        """
        own = getattr(agent, "_merge_base", None)
        if own and Path(own).is_dir():
            return Path(own)
        return self._merge_baseline_path

    def _merge_own_changes(self, agent: Agent) -> tuple[dict[str, Path], set[str]]:
        """What this agent has changed in its own copy, or nothing."""
        copy_dir = getattr(agent, "_merge_copy", None)
        baseline = self._merge_agent_baseline(agent)
        if not copy_dir or baseline is None or not Path(copy_dir).is_dir():
            return {}, set()
        try:
            return self._merge_diff(Path(copy_dir), baseline)
        except Exception:
            return {}, set()

    def _merge_rel_files(self, root: Path) -> dict[str, Path]:
        """Files under `root`, restricted to the merge scope."""
        out: dict[str, Path] = {}
        scope = self._merge_scope_roots()
        bases = [root / rel for rel in scope] if scope else [root]
        for base in bases:
            if not base.exists() and not base.is_symlink():
                continue
            if base.is_file() or base.is_symlink():
                rel = base.relative_to(root)
                if not any(self._merge_ignored_part(part) for part in rel.parts):
                    out[rel.as_posix()] = base
                continue
            for p in base.rglob("*"):
                if p.is_dir() and not p.is_symlink():
                    continue
                rel = p.relative_to(root)
                if any(self._merge_ignored_part(part) for part in rel.parts):
                    continue
                out[rel.as_posix()] = p
        return out

    @staticmethod
    def _merge_same_file(a: Path, b: Path) -> bool:
        # Diffs run repeatedly (per promote, per uniqueness check), so avoid
        # hashing large payloads: fall back to size+mtime past a threshold.
        try:
            if a.is_symlink() or b.is_symlink():
                return os.readlink(a) == os.readlink(b)
            sa, sb = a.stat(), b.stat()
            if sa.st_size != sb.st_size:
                return False
            if sa.st_size > _MERGE_HASH_MAX_BYTES:
                # nanosecond mtime: copy2/copytree preserve it exactly, so an
                # untouched file still compares equal while a same-size rewrite
                # is not mistaken for "unchanged".
                return sa.st_mtime_ns == sb.st_mtime_ns
            import hashlib
            return hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()
        except Exception:
            return False

    def _merge_diff(self, copy_dir: Path, baseline: Path) -> tuple[dict[str, Path], set[str]]:
        """(changed_or_added {rel: srcpath}, deleted {rel}) of copy vs baseline."""
        cur = self._merge_rel_files(copy_dir)
        base = self._merge_rel_files(baseline)
        changed = {
            rel: src for rel, src in cur.items()
            if rel not in base or not self._merge_same_file(src, base[rel])
        }
        deleted = set(base) - set(cur)
        return changed, deleted

    def _merge_apply(self, changed: dict[str, Path], deleted: set[str], target: Path) -> None:
        for rel, src in changed.items():
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                self._fixed_remove_path(dst)
            if src.is_symlink():
                os.symlink(os.readlink(src), dst)
            else:
                shutil.copy2(src, dst, follow_symlinks=False)
        for rel in deleted:
            self._fixed_remove_path(target / rel)

    # ─── agent-owned verification ────────────────────────────────────────
    # Agents already measure their own work constantly (39 of 40 shell calls in
    # one ann_vector_search_qps run were benchmark runs), but each measurement
    # was a throwaway `python -c` whose result survived only as prose. Nobody
    # could reproduce it and — the expensive part — nobody ever measured the
    # artifact that actually got submitted: a config merged from three children
    # replaced a working solution without a single run against it.
    #
    # So a verification is a runtime-owned object: the agent declares the
    # command, the runtime executes it, and the command is stored with the
    # working root replaced by a placeholder so the runtime can re-run the very
    # same check against a merged result. A claim nobody else can reproduce is
    # not evidence.

    VERIFY_WORKDIR_TOKEN = "{WORKDIR}"

    def _verify_working_root(self, agent: Agent) -> Path | None:
        """Where this agent's work lives: its private copy, else the task dir."""
        copy_dir = getattr(agent, "_merge_copy", None)
        if copy_dir and Path(copy_dir).is_dir():
            return Path(copy_dir)
        return self._merge_target()

    def _verify_portable_command(self, command: str, agent: Agent) -> str:
        """Replace concrete working roots with the placeholder, so the same
        command can later be run against the merged submission path.

        Paths continuing into the runtime workspace are left alone. It only
        exists at its original location — the working copy deliberately omits it
        — so rewriting it points the command at a directory that is not there.
        That is what killed the first check an agent tried to register: a `tee`
        into its workspace log became a path inside the copy, and the whole
        command died in a second.
        """
        import re
        out = command
        roots = []
        working = self._verify_working_root(agent)
        target = self._merge_target()
        for root in (working, target):
            if root is not None:
                roots.append(str(root).rstrip("/"))
        guard = self._merge_workspace_guard_segment(target) if target else None
        suffix = rf"(?!/{re.escape(guard)}\b)" if guard else ""
        # longest first: a child's copy path contains the task path as a prefix
        for root in sorted(set(roots), key=len, reverse=True):
            out = re.sub(re.escape(root) + suffix, self.VERIFY_WORKDIR_TOKEN, out)
        return out

    @staticmethod
    def _parse_verification_output(text: str) -> tuple[bool | None, float | None]:
        """(ok, metric) from the last machine-readable line of a check's output.

        Accepts a trailing JSON object or a `VERIFY: ok=... metric=...` line.
        Anything else counts as unparsed, which is treated as unverified rather
        than as success — a check whose verdict cannot be read is not a pass.
        """
        import json as _json
        import re
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        for line in reversed(lines[-25:]):
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = _json.loads(line)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if "ok" not in payload and "metric" not in payload:
                    continue
                ok = payload.get("ok")
                metric = payload.get("metric")
                return (
                    bool(ok) if ok is not None else None,
                    float(metric) if isinstance(metric, (int, float))
                    and not isinstance(metric, bool) else None,
                )
            match = re.match(r"^VERIFY:\s*(.*)$", line, re.IGNORECASE)
            if match:
                fields = dict(re.findall(r"(\w+)\s*=\s*(\S+)", match.group(1)))
                if not fields:
                    continue
                raw_ok = fields.get("ok")
                ok = None if raw_ok is None else raw_ok.lower() in ("1", "true", "yes")
                try:
                    metric = float(fields["metric"]) if "metric" in fields else None
                except ValueError:
                    metric = None
                return ok, metric
        return None, None

    async def _run_verification(
        self, command: str, workdir: Path, timeout: float
    ) -> dict[str, Any]:
        """Execute a verification command against `workdir` and read its verdict."""
        resolved = command.replace(self.VERIFY_WORKDIR_TOKEN, str(workdir))
        started = time.time()
        # Same launcher and environment as the shell tool, so a check behaves
        # identically to the command the agent ran by hand.
        env = {
            **os.environ,
            "WORKSPACE": str(workdir),
            "SHARED": str(self.config.workspace_root / "shared"),
        }
        try:
            proc = await asyncio.create_subprocess_shell(
                resolved,
                cwd=str(workdir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "ok": False, "metric": None, "exit_code": None,
                    "timed_out": True, "seconds": round(time.time() - started, 1),
                    "output": f"verification timed out after {timeout}s",
                    "readable": False,
                }
            text = stdout.decode(errors="replace")
        except Exception as exc:
            return {
                "ok": False, "metric": None, "exit_code": -1, "timed_out": False,
                "seconds": round(time.time() - started, 1),
                "output": f"{type(exc).__name__}: {exc}",
                "readable": False,
            }
        parsed_ok, metric = self._parse_verification_output(text)
        ok = parsed_ok
        if ok is None:
            ok = proc.returncode == 0 and metric is not None
        return {
            "ok": bool(ok) and proc.returncode == 0,
            "metric": metric,
            "exit_code": proc.returncode,
            "timed_out": False,
            "seconds": round(time.time() - started, 1),
            "output": text[-4000:],
            # Whether a verdict could be read at all. A check that reports a
            # failure has run; one that crashed before printing has not, and
            # only the first of those can gate anything.
            "readable": parsed_ok is not None or metric is not None,
        }

    @staticmethod
    def _verify_rank(
        ok: bool | None, metric: float | None, higher_is_better: bool = True
    ) -> tuple | None:
        """Sortable quality of a verification. A failing check always loses, so
        no metric, however good, can carry a broken artifact into the submission."""
        if ok is None:
            return None
        oriented = None
        if metric is not None:
            oriented = metric if higher_is_better else -metric
        return (1 if ok else 0, oriented if oriented is not None else 0.0)

    async def _submit_as_root(
        self, args, agent: Agent, runtime, handler, *, aggregate: bool = True
    ) -> Any:
        """Fold in what is still in flight, then gate, then spend the submission.

        An official submission is a statement about the whole workspace, so it
        waits for the work still running and folds it in as one merge. Otherwise
        it scores a state that no check has passed and that nobody has finished
        writing.

        `aggregate=False` skips the wait for a caller reached after the run has
        already drained its deliveries. There, a child still marked running is
        one that was killed and will never deliver, so waiting the full budget
        would spend the end of the wall clock to learn nothing — and the closing
        submission is the one the round is scored on.
        """
        wait = (
            await self._await_outstanding_deliveries(agent)
            if aggregate
            else {"waited": [], "timed_out": False, "holding": {}}
        )
        waited = wait["waited"]
        try:
            await self._merge_promote_pending()
        except Exception as exc:
            self._emit(agent.id, "merge_promote_error", {"detail": str(exc)[:300]})
        blocked = self._submit_incomplete_block(agent, wait)
        if blocked is None:
            blocked = await self._submit_preflight(agent)
        if blocked is not None:
            if waited:
                blocked["waited_for"] = waited
            return blocked
        # Read here, not at the call site: everything above changes the state this
        # submission is about, so the aggregate is what gets sent and judged.
        self._note_state_being_submitted(agent, "submit", args)
        result = await handler(args, agent, runtime)
        if waited:
            result = dict(result) if isinstance(result, dict) else {"result": result}
            result["waited_for"] = waited
        return result

    async def submit_official(
        self,
        reason: str = "final",
        agent: Agent | None = None,
        *,
        wait_for_deliveries: bool = False,
    ) -> Any:
        """Spend an official submission through the gates an agent's own goes through.

        The gates live in the per-turn tool table, so every submission that does
        not originate in a model tool call used to skip them. The EdgeBench
        adapter's own closing submission is one of those, and it is the decisive
        one: on the 2026-07-29 run it shipped `agent-2`, and because it went
        straight to `sforge-submit` its verdict was never parsed, never
        calibrated against the local check, and never written to the ledger — so
        the next iteration began without knowing what the last one had scored.

        `wait_for_deliveries` defaults to False for the caller this exists for:
        one reached after the run has drained its deliveries, where a child still
        marked running is one that was killed, and waiting its budget out would
        spend the end of the wall clock on work that is not coming. Queued merges
        are still folded in either way. A caller submitting mid-run, while
        children are genuinely still producing, wants True — otherwise it scores
        a state nobody has finished writing.
        """
        submit = (self.config.extra_tools or {}).get("submit")
        if not submit or "handler" not in submit:
            return None
        target = agent or self._submit_acting_agent()
        if target is None:
            # No agent ever ran: nothing to gate on, nobody to attribute the
            # verdict to, and a handler that expects an agent would fail on a
            # stand-in. Declining lets the caller submit directly, so a run that
            # produced no agents still gets scored.
            return None

        args = {"reason": reason}
        self._note_state_being_submitted(target, "submit", args)
        try:
            if self._merge_active() and not getattr(target, "_merge_copy", None):
                result = await self._submit_as_root(
                    args, target, self, submit["handler"], aggregate=wait_for_deliveries
                )
            elif self._submit_gate_enabled():
                blocked = await self._submit_preflight(target)
                result = blocked if blocked is not None else await submit["handler"](
                    args, target, self
                )
            else:
                result = await submit["handler"](args, target, self)
        except Exception as exc:
            self._emit(target.id, "merge_submit_error", {"detail": str(exc)[:300]})
            return None

        # The tool-call path calibrates from the dispatch loop; this path has no
        # dispatch loop, so the verdict would otherwise be read by nobody.
        try:
            await self._calibrate_from_tool_result(target, "submit", args, result)
        except Exception as exc:
            self._emit(target.id, "official_calibration_error", {"detail": str(exc)[:200]})
        return result

    def _submit_acting_agent(self) -> Agent | None:
        """Who a runtime-initiated submission is attributed to: the root agent."""
        for candidate in self.agents.values():
            if getattr(candidate, "parent", None) in (None, ""):
                return candidate
        return next(iter(self.agents.values()), None)

    def _merge_change_signature(self, changed: dict[str, Path], deleted: set[str]) -> frozenset:
        """Fingerprint of a diff, so an unchanged copy is not promoted twice.

        Content-hashed within the merge scope: same-size edits are common, and a
        cheaper size+mtime key would silently drop a real change.
        """
        import hashlib
        items: set[tuple] = {("-", rel) for rel in deleted}
        for rel, src in changed.items():
            try:
                stat = src.stat()
                if stat.st_size > _MERGE_HASH_MAX_BYTES:
                    items.add((rel, stat.st_size, stat.st_mtime_ns))
                else:
                    items.add((rel, hashlib.md5(src.read_bytes()).hexdigest()))
            except OSError:
                items.add((rel, "unreadable"))
        return frozenset(items)

    # ─── experiment ledger ───────────────────────────────────────────────
    # The keep-best ratchet below is driven entirely by the agent's own check,
    # so a run whose check never passes the submission path has no memory of
    # what it measured at all. One such run failed its check on all five merges
    # and then shrank the scored config to fit a local timeout, deleting the
    # parameters that had produced its best result — while the judge had already
    # scored that state.
    #
    # Nothing needs measuring to fix that: the judge's verdict arrives through
    # the submit tool and `_calibrate_with_official` already parses it and
    # already computes the signature of the state it applies to. It then keeps a
    # single boolean from the pair and discards the score. The ledger keeps the
    # pair, durably, so what the run has measured survives an agent that cannot
    # author a working check — and survives the agent's own forgetting.

    def _ledger_path(self) -> Path:
        return self._merge_root_dir() / "ledger.jsonl"

    def _ledger_higher_is_better(self) -> bool:
        """Direction of the judge's score, mirroring `verify`'s own parameter."""
        return os.environ.get("NANOMA_OFFICIAL_LOWER_IS_BETTER") != "1"

    @staticmethod
    def _ledger_digest(signature: frozenset | None) -> str | None:
        """A short stable name for a state, so entries can be compared by value."""
        if signature is None:
            return None
        parts = sorted(repr(item) for item in signature)
        return hashlib.md5("\n".join(parts).encode()).hexdigest()[:12]

    def _ledger_snapshot_dir(self, digest: str) -> Path:
        return self._merge_root_dir() / "ledger" / digest

    def _ledger_entries(self) -> list[dict]:
        path = self._ledger_path()
        if not path.is_file():
            return []
        entries = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a torn final line must not hide the rest
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def _ledger_best(self) -> dict | None:
        """The best measurement on record, read from disk rather than memory."""
        scored = [
            e for e in self._ledger_entries()
            if isinstance(e.get("metric"), (int, float)) and e.get("counts", True)
        ]
        if not scored:
            return None
        sign = 1 if self._ledger_higher_is_better() else -1
        return max(scored, key=lambda e: sign * float(e["metric"]))

    def _ledger_note(
        self,
        source: str,
        metric: float | None,
        *,
        signature: frozenset | None,
        agent_id: str = "root",
        counts: bool = True,
        extra: dict | None = None,
    ) -> dict | None:
        """Record one (state, measurement) pair and snapshot a new best state.

        `counts=False` records an observation that must not become a state to
        fall back to — a verdict the judge rejected as invalid still says
        something true about the state, but not that it is worth returning to.
        """
        if metric is None or not self._merge_active():
            return None
        digest = self._ledger_digest(signature)
        entry = {
            "at": time.time(),
            "source": source,
            "metric": float(metric),
            "state": digest,
            "counts": bool(counts),
            "agent": agent_id,
        }
        if extra:
            entry.update(extra)

        previous_best = self._ledger_best()
        try:
            self._ledger_path().parent.mkdir(parents=True, exist_ok=True)
            with self._ledger_path().open("a") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._emit(agent_id, "ledger_error", {"detail": str(exc)[:200]})
            return None

        if counts and digest is not None:
            sign = 1 if self._ledger_higher_is_better() else -1
            improved = previous_best is None or (
                sign * float(metric) > sign * float(previous_best.get("metric", 0.0))
            )
            if improved:
                self._ledger_keep_state(digest, metric, agent_id)
        self._emit(agent_id, "ledger_note", {
            "source": source, "metric": metric, "state": digest, "counts": counts,
        })
        return entry

    def _ledger_keep_state(self, digest: str, metric: float, agent_id: str) -> None:
        """Snapshot the state a measurement applies to, so it can be restored.

        Only the best is kept. This is reached once per improvement, so keeping
        each one would grow the workspace by a full copy of the submission scope
        every time the run got better — up to NANOMA_MERGE_MAX_MB apiece, inside a
        container with a fixed disk. Superseded states stay on the record as
        numbers; `experiments` reports them as measured but not restorable.
        """
        target = self._merge_target()
        if target is None or not self._merge_snapshot_affordable(target):
            return
        destination = self._ledger_snapshot_dir(digest)
        if destination.is_dir():
            return  # this state is already kept; measuring it twice changes nothing
        if not self._merge_copy_tree(target, destination):
            return
        self._emit(agent_id, "ledger_snapshot", {"state": digest, "metric": metric})
        for stale in destination.parent.iterdir():
            if stale.is_dir() and stale.name != digest:
                shutil.rmtree(stale, ignore_errors=True)

    def _ledger_restore(self, digest: str) -> bool:
        """Put a recorded state back into the submission path."""
        source = self._ledger_snapshot_dir(digest)
        target = self._merge_target()
        if target is None or not source.is_dir():
            return False
        changed, deleted = self._merge_diff(source, target)
        if changed or deleted:
            self._merge_apply(changed, deleted, target)
        self._emit("root", "ledger_restored", {
            "state": digest, "changed": len(changed), "deleted": len(deleted),
        })
        return True

    def _ledger_state_metrics(
        self, digest: str | None, source: str | None = None
    ) -> list[float]:
        """Every metric recorded for one state, optionally from one channel only.

        Channels are not interchangeable: an official score and a local check's
        number measure different things on different scales, so pooling them
        would compare quantities that were never comparable.
        """
        if digest is None:
            return []
        return [
            float(e["metric"])
            for e in self._ledger_entries()
            if e.get("state") == digest
            and isinstance(e.get("metric"), (int, float))
            and (source is None or e.get("source") == source)
        ]

    # ─── how much of a difference is a difference ─────────────────────────────
    # Re-measuring one unchanged state is not expected to return one number. On
    # ann_vector_search_qps, 17 rounds submitted a byte-identical archive and
    # scored 2564 to 4160 QPS — a 16% coefficient of variation, and through that
    # task's log_max rescale, 0.00 to 15.08 official points for the same bytes.
    # The spread is a property of the measurement, not of judge load: it is
    # 12-16% whether evaluations run three at a time or seventeen.
    #
    # Everything that decided "this state is worse, roll it back" or "this state
    # is worse, refuse to ship it" compared two single numbers with no margin,
    # so on a difference smaller than the noise those decisions were coin flips
    # that presented themselves as measurements. These estimate the noise from
    # the run's own repeated measurements and refuse to call a difference real
    # until it clears it.

    def _measurement_spread(self, source: str) -> tuple[float, float] | None:
        """Scatter of repeat measurements of a single unchanged state.

        Returns `(relative, absolute)` — a coefficient of variation and a
        standard deviation in the metric's own units. Both, because neither
        alone survives every metric a task might report: a relative figure is
        meaningless for a metric that sits near or crosses zero (a delta, a
        margin, a signed error), while an absolute one measured on states of one
        magnitude under-protects states of another.

        Estimated from this run's own record rather than assumed, because the
        scatter is a property of the task's measurement and cannot be known in
        advance. Returns None until some state has been measured twice. A
        deterministic check measured twice gives (0, 0), which restores exact
        comparison — that is the intended answer, not a degenerate one.
        """
        by_state: dict[str, list[float]] = {}
        for entry in self._ledger_entries():
            metric, state = entry.get("metric"), entry.get("state")
            if entry.get("source") != source or state is None:
                continue
            if isinstance(metric, (int, float)):
                by_state.setdefault(state, []).append(float(metric))

        relative, absolute = [], []
        for metrics in by_state.values():
            if len(metrics) < 2:
                continue
            mean = sum(metrics) / len(metrics)
            variance = sum((m - mean) ** 2 for m in metrics) / (len(metrics) - 1)
            sigma = variance**0.5
            absolute.append(sigma)
            if abs(mean) > sigma:
                # Only where a ratio means something. A state whose repeats
                # straddle zero has a mean smaller than its own scatter, and its
                # coefficient of variation is an artefact of dividing by nearly
                # nothing — 424% for repeats of -0.01 and 0.02. Letting that into
                # the estimate would set a margin no regression could clear.
                relative.append(sigma / abs(mean))
        if not absolute:
            return None
        return (
            sum(relative) / len(relative) if relative else 0.0,
            sum(absolute) / len(absolute),
        )

    def _measurement_noise(self, source: str) -> float | None:
        """The relative half of the scatter, for reporting."""
        spread = self._measurement_spread(source)
        return None if spread is None else spread[0]

    def _noise_margin(self, source: str, at: float = 1.0) -> float | None:
        """The gap a difference must clear at magnitude `at`, in metric units."""
        spread = self._measurement_spread(source)
        if spread is None:
            return None
        try:
            k = float(os.environ.get("NANOMA_NOISE_MARGIN_SIGMAS", "2") or 2)
        except ValueError:
            k = 2.0
        relative, absolute = spread
        # Whichever regime is worse at this magnitude. Multiplicative noise
        # dominates for a large value, additive for a small one, and taking the
        # larger keeps the margin honest without having to decide which the
        # task's measurement is.
        return k * max(relative * abs(at), absolute)

    def _metric_is_worse(
        self, candidate: list[float], reference: list[float], source: str
    ) -> bool | None:
        """Whether `candidate` measures worse than `reference` beyond the noise.

        Returns None when the run cannot yet tell — no measurement on one side,
        or no repeat anywhere from which to estimate the scatter. Callers read
        None according to which mistake costs more; see the two call sites.
        """
        if not candidate or not reference:
            return None

        def median(values: list[float]) -> float:
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[mid]
            return (ordered[mid - 1] + ordered[mid]) / 2

        here, there = median(candidate), median(reference)
        # Sized at the reference's magnitude, not the candidate's: the reference
        # is the established measurement, and scaling by the candidate would let
        # a wildly bad one inflate the margin it has to clear.
        margin = self._noise_margin(source, at=abs(there))
        if margin is None:
            return None
        sign = 1 if self._ledger_higher_is_better() else -1
        return sign * (there - here) > margin

    def _verify_state_path(self) -> Path:
        return self._merge_root_dir() / "verification.json"

    def _verify_load_state(self) -> None:
        """Adopt the check and ratchet floor proven by earlier iterations.

        A long EdgeBench run is a series of fresh NanoMA processes against one
        task directory. Without this, every iteration starts blind: the check
        somebody already got working is gone, and the ratchet floor resets, so a
        later round can quietly ship something worse than a measured earlier one.
        """
        if getattr(self, "_verify_state_loaded", False):
            return
        self._verify_state_loaded = True
        path = self._verify_state_path()
        if not path.is_file():
            return
        try:
            state = json.loads(path.read_text())
        except Exception:
            return
        spec = state.get("spec")
        if isinstance(spec, dict) and spec.get("command"):
            self._verify_persisted_spec = spec
        calibration = state.get("calibration")
        if isinstance(calibration, dict):
            for cmd, rec in calibration.items():
                if isinstance(rec, dict):
                    self._verify_calibration.setdefault(cmd, rec)
        seen = state.get("metric_seen")
        if isinstance(seen, dict):
            for cmd, values in seen.items():
                for v in values or []:
                    self._verify_note_metric(cmd, v)
        rank = state.get("best_rank")
        if isinstance(rank, list) and rank and self._merge_best_dir().is_dir():
            self._merge_best_rank = tuple(rank)
            self._merge_best_metric = state.get("best_metric")
            self._merge_best_command = state.get("best_command")
            self._emit("root", "verify_state_loaded", {
                "metric": self._merge_best_metric,
                "check": (spec or {}).get("name"),
            })

    def _verify_save_state(self, spec: dict | None = None) -> None:
        if not self._merge_active():
            return
        # Read before writing, or the first registration of a new iteration
        # overwrites the floor an earlier one measured — which is why nothing
        # was ever actually inherited.
        self._verify_load_state()
        spec = spec or getattr(self, "_verify_persisted_spec", None)
        path = self._verify_state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "spec": spec,
                "best_rank": list(self._merge_best_rank) if self._merge_best_rank else None,
                "best_metric": self._merge_best_metric,
                "best_command": self._merge_best_command,
                "metric_seen": self._verify_metric_seen,
                "calibration": self._verify_calibration,
            }, indent=2))
        except OSError:
            pass

    def _verify_register_spec(self, spec: dict) -> None:
        """Record a newly proven check for this run and the ones after it."""
        self._verify_load_state()
        self._verify_persisted_spec = spec
        # A different check means a different scale, so a floor measured by the
        # old one is not something the new one can be compared against. Keyed on
        # the check actually in force, not on every registration: one run
        # registered eight checks in a row, and resetting each time threw away
        # the only floor that meant anything.
        in_force = (self._authoritative_spec() or {}).get("command")
        if self._merge_best_command is not None and in_force != self._merge_best_command:
            self._merge_best_rank = None
            self._merge_best_metric = None
            self._merge_best_command = None
        self._verify_save_state(spec)

    @staticmethod
    def _official_parse(text: str) -> dict | None:
        """Read the judge's verdict out of what sforge-submit printed.

        This is the feedback channel the harness gives the agent — score, pass
        rate and the names of what failed are printed for it to read. The
        host-side auto-evals are a separate, admin-only view and are none of our
        business.
        """
        if not text or "Results" not in text:
            return None
        pass_rate = None
        m = re.search(r"Pass rate:\s+([\d.]+)%", text)
        if m:
            pass_rate = float(m.group(1)) / 100.0
        passed = re.search(r"Passed:\s+(\d+)/(\d+)", text)
        if pass_rate is None and passed and int(passed.group(2)):
            pass_rate = int(passed.group(1)) / int(passed.group(2))
        if pass_rate is None and "All tests passed!" in text:
            pass_rate = 1.0
        if pass_rate is None:
            return None
        score = None
        m = re.search(r"Score:\s+([-\d.]+)", text)
        if m:
            try:
                score = float(m.group(1))
            except ValueError:
                score = None
        if passed is None and score is None:
            # A pass rate with no tally and no score is not a verdict — reading a
            # fragment as one produced "invalid, 100% passed", which would have
            # blamed a check for a rejection nobody made.
            return None
        failed = re.findall(r"^\s+- (.+)$", text, flags=re.MULTILINE)
        round_id = None
        m = re.search(r"^\s+(\S+) Results\s*$", text, flags=re.MULTILINE)
        if m:
            round_id = m.group(1)
        return {
            "valid": "Valid:       no" not in text and "Valid: no" not in text,
            "pass_rate": pass_rate,
            "score": score,
            "failed": [f.strip() for f in failed][:20],
            "round": round_id,
        }

    @staticmethod
    def _is_submission_call(name: str, args: dict | None) -> bool:
        """Whether this tool call is an official submission being sent."""
        if name == "submit":
            return True
        if name != "shell":
            return False
        command = str((args or {}).get("command") or "")
        return "sforge-submit" in command and "--list" not in command

    def _note_state_being_submitted(
        self, agent: Agent, name: str, args: dict | None
    ) -> None:
        """Record which state a submission is sending, before it is judged.

        A verdict arrives minutes after the submission that earned it, and the
        submit call blocks for that whole time while other agents keep editing the
        shared submit path. Reading the workspace when the verdict lands therefore
        files the authoritative measurement against a state that did not produce
        it: one run stamped an official 0.0 and an official 4468.0 onto the same
        digest 31 seconds apart, and snapshotted whatever happened to be present
        as the state to fall back to. The ledger is only worth having if its pairs
        are true, so the state is read at the moment it is sent.
        """
        if not self._merge_active() or not self._is_submission_call(name, args):
            return
        target = self._merge_target()
        if target is None:
            return
        self._submitted_signature[agent.id] = self._merge_scope_signature(target)

    async def _calibrate_from_tool_result(
        self, agent: Agent, name: str, args: dict | None, result: Any
    ) -> None:
        """Catch a judge verdict however the agent went and get it."""
        if not self._merge_active() or not isinstance(result, dict):
            return
        if not self._is_submission_call(name, args):
            return
        text = "\n".join(
            str(result.get(key) or "")
            for key in ("stdout", "output", "submission", "stderr")
        )
        try:
            await self._calibrate_with_official(agent, text)
        except Exception as exc:
            self._emit(agent.id, "official_calibration_error", {"detail": str(exc)[:200]})

    async def _calibrate_with_official(self, agent: Agent, text: str) -> dict | None:
        """Check the local instrument against the judge's verdict on the same state.

        The local check is the gate; this is the occasional calibration of it. A
        check can pass a state the judge rejects — one measured throughput while
        the task also required a recall floor, so the ratchet happily climbed
        towards submissions that scored zero. The disagreement is a hard fact,
        and it goes to the agents as a fact.
        """
        if not self._merge_active():
            return None
        official = self._official_parse(text)
        if official is None:
            return None
        self._official_last = official
        # The state as it was when this submission left, not as it is now that the
        # verdict has come back. Falls back to the live scope for a verdict that
        # arrived without a matching send, which is better than recording nothing.
        signature = self._submitted_signature.pop(agent.id, None)
        if signature is None:
            target = self._merge_target()
            signature = self._merge_scope_signature(target) if target else None
        judged_ok = bool(official["valid"]) and (official["pass_rate"] or 0.0) >= 1.0
        # Only comparable when the local check actually passed this same state.
        local_ok = (
            signature is not None
            and signature == getattr(self, "_merge_scored_signature", None)
        )
        self._emit(agent.id, "official_verdict", {
            "pass_rate": official["pass_rate"], "score": official["score"],
            "valid": official["valid"], "round": official["round"],
            "local_passed_this_state": local_ok,
        })
        # The authoritative measurement of this exact state. Recorded before the
        # calibration bookkeeping below, which reduces the pair to one boolean
        # and is the only thing that used to survive.
        # `counts` means "a state worth returning to", so a verdict where nothing
        # passed is excluded for the same reason an invalid one is. It is still
        # recorded — that a state scores nothing is worth knowing — but it must
        # not become the best. One run's only official verdict was a broken state
        # scoring 0, which was snapshotted as the state to fall back to and then
        # used to refuse two later submissions of a state its own check had just
        # verified at 18000: nothing is "worse" than a state where nothing passed.
        passed_something = (
            official["pass_rate"] is None or float(official["pass_rate"]) > 0
        )
        self._ledger_note(
            "official",
            official["score"],
            signature=signature,
            agent_id=agent.id,
            counts=bool(official["valid"]) and passed_something,
            extra={
                "round": official["round"],
                "pass_rate": official["pass_rate"],
                "valid": bool(official["valid"]),
            },
        )
        spec = self._authoritative_spec()
        if not local_ok or judged_ok or spec is None:
            if local_ok and judged_ok and spec is not None:
                self._verify_mark_calibration(spec, agreed=True)
            return official
        self._verify_mark_calibration(spec, agreed=False)
        # The state the local check called best is one the judge rejects, so it
        # is not somewhere to rewind to at the end of the run.
        if signature == getattr(self, "_merge_best_signature", None):
            self._merge_best_rank = None
            self._merge_best_metric = None
            self._merge_best_command = None
            self._merge_best_signature = None
            self._emit("root", "merge_best_dropped", {"reason": "judge rejected it"})
        note = (
            f"[Calibration] The judge scored the state your check had passed: "
            f"valid={official['valid']}, pass rate {official['pass_rate']:.0%}"
            + (f", score {official['score']}" if official["score"] is not None else "")
            + f". Failing: {official['failed'] or ['(not named)']}. "
            f"The check now gating every merge is '{spec.get('name') or spec.get('command','')[:60]}', "
            f"registered by {spec.get('agent')}, and it passed this state — so what it "
            "measures and what the judge requires are not the same thing."
        )
        told = []
        for who in {agent.id, spec.get("agent")}:
            peer = self.agents.get(who or "")
            if peer is None or getattr(peer, "status", None) in ("failed", "killed"):
                continue
            await self.deliver(Envelope(
                from_id="system", to_id=peer.id, content=note,
                tokens=estimate_tokens(note), timestamp=time.time(), mode="steer",
            ))
            told.append(peer.id)
        self._emit(agent.id, "official_disagrees", {
            "pass_rate": official["pass_rate"], "failed": official["failed"][:10],
            "check": spec.get("name"), "told": told,
        })
        return official

    def _verify_contradicted(self, spec: dict) -> bool:
        """Whether the judge has disagreed with this check more often than not."""
        record = self._verify_calibration.get(spec.get("command", ""))
        if not record:
            return False
        return record.get("contradicted", 0) > record.get("agreed", 0)

    def _verify_mark_calibration(self, spec: dict, *, agreed: bool) -> None:
        """Remember whether the judge has ever contradicted this check."""
        command = spec.get("command", "")
        record = self._verify_calibration.setdefault(
            command, {"agreed": 0, "contradicted": 0}
        )
        record["agreed" if agreed else "contradicted"] += 1
        self._verify_save_state()

    def _verify_note_metric(
        self, command: str, metric: float | None, *, at_target: bool = True
    ) -> None:
        """Remember what a check has reported, so we can tell whether its number
        actually tracks the work.

        Only numbers measured against the submission path count. An agent's
        private copy is a different tree — it holds whatever the agent generated
        along the way, which the merge does not carry — so the same command
        legitimately reports different numbers in the two places. Counting both
        made a check look like it tracked state when all it tracked was which
        tree it ran in: one run registered a check that read 3622 in its own copy
        and 0 at the target, and those two values were enough to promote a
        permanent floor of zero.
        """
        if metric is None or not at_target:
            return
        seen = self._verify_metric_seen.setdefault(command, [])
        if metric not in seen:
            seen.append(metric)
            del seen[:-8]

    async def _verify_probe_at_target(
        self, command: str, timeout: float
    ) -> dict[str, Any] | None:
        """Can the gate run this check where it will have to run it?

        A check earns registration by passing where the agent ran it, which for a
        child is its own private copy. The gate re-runs it against the merged
        submission path instead, and those are different trees: the copy holds
        the benchmark output and datasets the agent generated, while the merge
        carries only files inside the submit scope. So a check reading its own
        `results/` passed on registration and then crashed at every single
        promotion — the gate answered `ok: false` for a whole run and nothing was
        ever kept, which is the failure this probe exists to catch.

        Returns None when there is no separate target to probe. Probed once per
        distinct command, because the probe costs a full run of the check.
        """
        target = self._merge_target()
        if target is None:
            return None
        cached = self._verify_probe_cache.get(command)
        if cached is not None:
            return cached
        async with self._merge_get_lock():
            result = await self._run_verification(command, target, timeout)
        self._verify_probe_cache[command] = result
        # Deliberately not recorded as a metric observation. The probe asks one
        # question — can a verdict be read here — and the merged tree does not
        # hold this agent's work yet, so its number describes a state nobody
        # submitted. Evidence that a check tracks the work comes from the
        # promotions, which measure states that were really assembled.
        return result

    def _verify_metric_is_proven(self, spec: dict) -> bool:
        """Whether this check's metric may be used as a ratchet floor.

        A check earns that two ways, and needs both.

        It must move: one run registered eight checks in a row that all said
        `ok, metric 0.0`, and as a floor that is vacuous — nothing can ever be
        "worse" than zero — so the gate silently degrades to accepting
        everything.

        It must also agree with the authority. A local number is a proxy for the
        score, and a proxy that ranks states differently from the judge is worse
        than no proxy: the ratchet then preserves and protects whichever state the
        proxy happens to prefer. One run's check reported 18138 for a state the
        judge would score on a scale where its best was 3454, and there was no
        mechanism that could notice. While the run cannot yet tell — fewer than
        two states measured by both — moving is enough.
        """
        if len(self._verify_metric_seen.get(spec.get("command", ""), [])) < 2:
            return False
        tracks = self._verify_metric_tracks_authority()
        if tracks is False:
            self._emit("root", "verify_metric_withdrawn", {
                "check": spec.get("name") or spec.get("command", "")[:80],
                "reason": "it ranks states differently from the judge",
            })
            return False
        return True

    def _verify_metric_tracks_authority(self) -> bool | None:
        """Whether the local check ranks states the way the judge does.

        Returns None while fewer than two states have been measured by both, so
        the caller reads it as "no evidence against". Compares orderings rather
        than values: the two need not be on one scale, only to agree on which
        state is better, which is all a ratchet floor uses them for.
        """
        by_state: dict[str, dict[str, float]] = {}
        for entry in self._ledger_entries():
            state, metric, source = (
                entry.get("state"), entry.get("metric"), entry.get("source"),
            )
            if state is None or not isinstance(metric, (int, float)):
                continue
            if source in ("official", "verify"):
                # Latest measurement of this state on this channel.
                by_state.setdefault(state, {})[source] = float(metric)

        paired = [v for v in by_state.values() if "official" in v and "verify" in v]
        if len(paired) < 2:
            return None
        sign = 1 if self._ledger_higher_is_better() else -1
        agree = disagree = 0
        for i, a in enumerate(paired):
            for b in paired[i + 1:]:
                official = sign * (a["official"] - b["official"])
                local = a["verify"] - b["verify"]
                if official == 0 or local == 0:
                    continue  # a tie on either side ranks nothing
                if (official > 0) == (local > 0):
                    agree += 1
                else:
                    disagree += 1
        if agree + disagree == 0:
            return None
        return disagree <= agree

    def _authoritative_spec(self) -> dict | None:
        """The one check the gate measures every merged result with.

        Deliberately not per-agent, even though an agent may have registered its
        own: promotions are ranked against a single shared floor, and two checks
        report on different scales — one agent's 1.0 and another's 0.33 are not
        comparable, so mixing them rolls back good work and keeps bad.

        A check that has been seen to move outranks a newer one that has not,
        so a stream of freshly registered `ok, metric 0.0` checks cannot displace
        one that actually measures the objective. A check inherited from an
        earlier iteration counts too, so a child that never declared one is not
        a dead end.
        """
        self._verify_load_state()
        candidates = [
            a._verify_spec for a in self.agents.values()
            if getattr(a, "_verify_spec", None)
        ]
        persisted = getattr(self, "_verify_persisted_spec", None)
        if persisted:
            candidates.append(persisted)
        if not candidates:
            return None
        return max(candidates, key=lambda s: (
            self._verify_metric_is_proven(s), s.get("at", 0.0),
        ))

    def _verification_spec_for(self, agent: Agent) -> dict | None:
        return self._authoritative_spec()

    async def _verify_submission_path(self, agent: Agent, spec: dict) -> dict:
        """Run a registered check against the shared submission path itself."""
        target = self._merge_target()
        result = await self._run_verification(
            spec["command"], target, float(spec.get("timeout", 900))
        )
        result["command"] = spec["command"]
        result["source_agent"] = spec.get("agent")
        self._verify_note_metric(spec["command"], result["metric"])
        self._emit(agent.id, "verify_merged", {
            "ok": result["ok"], "metric": result["metric"],
            "seconds": result["seconds"], "from": spec.get("agent"),
            "exit_code": result["exit_code"],
            # Without the output there is no way to tell a merge that measured
            # worse from a check that could not run against the shared tree.
            "output_tail": " ".join((result["output"] or "").split())[-400:],
        })
        return result

    def _record_verified_best(self, metric: float | None, rank: tuple) -> None:
        target = self._merge_target()
        if target is None:
            return
        if not rank or rank[0] != 1:
            return  # a state whose check failed is not a state to fall back to
        spec = self._authoritative_spec()
        if spec is not None and self._verify_contradicted(spec):
            # The judge has already shown this check passing a state it rejects,
            # so a state it approves is not something to rewind to. Without this
            # the next merge simply re-records what the last verdict threw out.
            self._emit("root", "merge_best_withheld", {"reason": "check contradicted"})
            return
        self._merge_current_rank = rank
        self._merge_current_metric = metric
        self._merge_scored_signature = self._merge_scope_signature(target)
        if self._merge_best_rank is not None and rank <= self._merge_best_rank:
            return
        if not self._merge_snapshot_affordable(target):
            return
        if not self._merge_copy_tree(target, self._merge_best_dir()):
            return
        self._merge_best_rank = rank
        self._merge_best_metric = metric
        self._merge_best_command = (self._authoritative_spec() or {}).get("command")
        self._merge_best_signature = self._merge_scored_signature
        self._verify_save_state()
        self._emit("root", "merge_best_snapshot", {"metric": metric, "verified": True})

    def _schedule_child_delivery(self, agent: Agent) -> asyncio.Task:
        """Hand the fold-back to a task of its own.

        Deliberately not awaited by the caller: this is reached from the agent
        loop's `finally`, and a parent usually kills a child right after it
        delivers, so that `finally` runs with a cancellation already pending —
        every await there raises CancelledError at once, which `except Exception`
        does not catch. One run folded back nothing at all that way. `run()`
        drains these before it ends.
        """
        task = asyncio.ensure_future(self._finalize_child_delivery(agent))
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_tasks.discard)
        return task

    async def _finalize_child_delivery(self, agent: Agent) -> None:
        """Fold a finishing child's work back in, then decide about cross-checking."""
        delivery = None
        try:
            delivery = await self._merge_promote_child(agent)
        except Exception as exc:
            self._emit(agent.id, "merge_promote_error", {"detail": str(exc)[:300]})
        if agent.status == "killed" or self._delivery_judging_closed:
            # The judge costs an extra model round-trip and can only pay off while there
            # is still time to act on it.
            return
        try:
            # A delivery is the other moment worth a spawn decision: it is when
            # sibling contributions start overlapping and the merged artifact
            # stops being anything a single agent has ever run.
            await asyncio.wait_for(
                self._spawn_judge_at_delivery(agent, delivery),
                timeout=self._DELIVERY_JUDGE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self._emit(agent.id, "spawn_judge_error", {
                "stage": "delivery", "detail": "judge timed out",
            })
        except Exception as exc:
            self._emit(agent.id, "spawn_judge_error", {
                "stage": "delivery", "detail": str(exc)[:300],
            })

    async def _await_delivery_tasks(self, timeout: float = 120.0) -> None:
        """Let outstanding fold-backs finish before the run is wrapped up."""
        self._delivery_judging_closed = True
        pending = [t for t in self._delivery_tasks if not t.done()]
        if not pending:
            return
        self._emit("system", "delivery_drain", {"pending": len(pending)})
        done, still = await asyncio.wait(pending, timeout=timeout)
        for t in still:
            t.cancel()
        if still:
            self._emit("system", "delivery_drain_timeout", {"abandoned": len(still)})

    async def _await_outstanding_deliveries(self, parent: Agent) -> dict:
        """Hold until the children that still owe work have handed it over.

        Bounded, and only for children that are actually producing: a parked one
        owes nothing and is waiting for the parent, so waiting on it would be a
        standoff.

        Reports how the wait ended, not just whom it waited on. It used to return
        the names alone, so the caller could not tell "everything is folded in"
        from "gave up with a child still holding files" and spent the submission
        identically either way. One run waited out the full 15 minutes on a child,
        submitted 31 seconds later, and the verdict it got back recorded that no
        local check had passed the state it sent.

        The budget is spent once per parent, not once per call. It used to be
        recomputed on entry, so every submission attempt bought another full wait
        on the same children: one run blocked its root for 2702 of its 6240
        seconds across three attempts, waiting on the same three children holding
        the same two files, and reached its deadline having submitted once.
        """
        spent = self._aggregate_wait_spent.get(parent.id, 0.0)
        budget = max(0.0, self._AGGREGATE_WAIT_SECONDS - spent)
        started = time.time()
        deadline = started + budget
        waited: list[str] = []
        timed_out = False
        holding: dict[str, list[str]] = {}
        try:
            while True:
                outstanding = self._outstanding_deliveries(parent)
                if not outstanding:
                    break
                if time.time() >= deadline:
                    timed_out = True
                    holding = {
                        cid: info["undelivered_files"]
                        for cid, info in outstanding.items() if info["undelivered_files"]
                    }
                    self._emit(parent.id, "aggregate_wait_timeout", {
                        "still_outstanding": sorted(outstanding),
                        "holding": holding,
                        "budget_seconds": budget,
                        "exhausted": budget <= 0,
                    })
                    break
                for cid in outstanding:
                    if cid not in waited:
                        waited.append(cid)
                        self._emit(parent.id, "aggregate_wait", {
                            "for": cid, "status": outstanding[cid]["status"],
                            "files": outstanding[cid]["undelivered_files"],
                            "budget_seconds": budget,
                        })
                await asyncio.sleep(self._AGGREGATE_POLL_SECONDS)
        finally:
            self._aggregate_wait_spent[parent.id] = spent + (time.time() - started)
        return {"waited": waited, "timed_out": timed_out, "holding": holding}

    def _wait_is_interrupted(self, agent: Agent, target_ids: list[str]) -> bool:
        """Whether something has arrived that is worth cutting a wait short for.

        The arrivals a wait is *for* are not interruptions. Treating them as such
        meant a parent asking for mode='all' across a fan-out came back on the
        first child's completion notice, holding one result out of four, and went
        on to aggregate a partial set — the thing it had asked not to do.
        """
        targets = set(target_ids or [])
        for inbox in (agent._immediate_inbox, agent._steer_inbox):
            for envelope in list(getattr(inbox, "_queue", ())):
                sender = getattr(envelope, "from_id", "")
                if sender in targets:
                    continue  # one of them reporting in
                content = getattr(envelope, "content", "") or ""
                if sender == "system" and any(t in content for t in targets):
                    continue  # the runtime saying one of them finished or delivered
                return True
        return False

    def _outstanding_deliveries(self, parent: Agent) -> dict[str, dict]:
        """Children that still owe this parent something.

        Either still working, or sitting on changes they have not delivered. A
        child parked in `idle` with nothing pending owes nothing — waiting on one
        of those would be waiting for a wake-up that only the parent can send.
        """
        out: dict[str, dict] = {}
        for child_id in getattr(parent, "children", []) or []:
            child = self.agents.get(child_id)
            if child is None or child.status in ("done", "failed", "killed"):
                continue
            changed, deleted = self._merge_own_changes(child)
            undelivered = bool(changed or deleted) and (
                self._merge_change_signature(changed, deleted)
                != getattr(child, "_merge_promoted_sig", None)
            )
            if child.status != "running" and not undelivered:
                continue
            out[child_id] = {
                "status": child.status,
                "undelivered_files": sorted(set(changed) | deleted)[:10],
            }
        return out

    def _outstanding_note(self, parent: Agent) -> str:
        """What is still coming, for a parent about to act on what has arrived."""
        outstanding = self._outstanding_deliveries(parent)
        if not outstanding:
            return ""
        holding = {
            cid: info["undelivered_files"]
            for cid, info in outstanding.items() if info["undelivered_files"]
        }
        note = (
            f"\n[Still to come] {sorted(outstanding)} have not reported yet."
        )
        if holding:
            note += f" Undelivered changes: {json.dumps(holding)}."
        note += (
            " wait(mode='all') blocks until they all report; their arrivals no longer "
            "cut it short."
        )
        return note

    def _merge_live_children(self, exclude: str = "") -> list[Agent]:
        return [
            a for a in self.agents.values()
            if a.id != exclude
            and getattr(a, "status", None) not in ("done", "failed", "killed")
            and getattr(a, "_merge_copy", None)
            and Path(a._merge_copy).is_dir()
        ]

    def _merge_refresh_idle_copies(self, exclude: str = "") -> list[str]:
        """Move copies that have nothing of their own at stake up to the submission.

        Eligibility is read off the copy rather than handed out as a role: a child
        that has not written anything loses nothing by being moved forward, and
        the moment it does write something it stops being refreshed and delivers
        its work like anyone else. That keeps an agent from reasoning about a
        state that no longer exists — measuring work that has since been replaced,
        or reporting on files nobody is going to ship.

        Called while holding the merge lock, so nobody is shown a half-applied tree.
        """
        target = self._merge_target()
        refreshed: list[str] = []
        if target is None:
            return refreshed
        for child in self._merge_live_children(exclude):
            changed, deleted = self._merge_own_changes(child)
            if changed or deleted:
                continue
            try:
                # The new baseline is taken first: a copy moved forward without
                # one reads as having changed everything the delivery brought,
                # and would re-deliver other agents' work as its own.
                base = self._merge_rebase_child(child, target)
                if base is None:
                    continue
                self._merge_restore_snapshot(target, Path(child._merge_copy))
                child._merge_base = base
                refreshed.append(child.id)
            except Exception as exc:
                self._emit(child.id, "merge_refresh_error", {"detail": str(exc)[:200]})
        return refreshed

    def _finish_time_situation(self, agent: Agent, tc) -> dict | None:
        """Hand an agent the state of play the first time it tries to finish.

        Whether to stay is a judgement that needs facts, and the one fact an
        agent cannot see is what its peers are doing to the same files right
        now. So the facts are delivered at the moment of the decision and the
        decision is left where it was — calling again finishes. Once per agent,
        so nobody can be held in.
        """
        if getattr(tc, "name", "") != "set_status":
            return None
        if str((getattr(tc, "arguments", None) or {}).get("status", "done")) != "done":
            return None
        if not self._merge_active() or agent.parent is None:
            return None
        if getattr(agent, "_finish_situation_shown", False):
            return None
        mine = self._merge_own_changes(agent)
        touched = set(mine[0]) | mine[1]
        # Files this agent has an interest in: what it changed, and — for one
        # that changed nothing — what its registered check measures, which is
        # the whole submission.
        peers: dict[str, list[str]] = {}
        for peer in self._merge_live_children(exclude=agent.id):
            changed, deleted = self._merge_own_changes(peer)
            files = set(changed) | deleted
            shared = sorted(files & touched) if touched else sorted(files)
            if shared:
                peers[peer.id] = shared[:10]
        if not peers:
            return None
        agent._finish_situation_shown = True
        self._emit(agent.id, "finish_situation", {"peers": peers})
        return {
            "status_not_set": "done",
            "still_in_flight": peers,
            "detail": (
                "These agents are still working on files you have a stake in, so what "
                "you looked at is not the last word on them. Anything you registered "
                "with the verify tool keeps being re-run on every later merge whether "
                "you stay or not. Call set_status again to finish; set_status(idle) "
                "costs nothing and wakes you when one of them delivers."
            ),
        }

    def _merge_rebase_child(self, agent: Agent, source: Path) -> Path | None:
        """Snapshot what an agent is being handed, to measure its own work against."""
        if not self._merge_snapshot_affordable(source):
            return None
        base = self._merge_root_dir() / "base" / agent.id
        return base if self._merge_copy_tree(source, base) else None

    async def _notify_peers_of_delivery(
        self, delivered: Agent, changed: list[str], kept: bool,
        verified: bool, metric: float | None, refreshed: list[str],
    ) -> None:
        """Tell whoever this delivery affects, straight into their inbox.

        Not via the parent: a relay makes every peer wait on the parent noticing
        and forwarding, and the parent is the busiest agent in the round. Who
        gets told follows from what happened to them — their copy moved, or the
        delivery touched a file they are editing — so nobody has to be nominated
        as the recipient in advance.
        """
        if not kept and not refreshed:
            return
        landed = set(changed)
        contributors = sorted({
            a.id for a in self.agents.values()
            if getattr(a, "_merge_promoted_sig", None) is not None
        })
        outcome = (
            f"kept (verified={verified}, metric={metric})" if kept
            else "rolled back after failing its check against the merged submission"
        )
        head = (
            f"[Submission changed] {delivered.id} delivered {sorted(landed)[:10]} and it "
            f"was {outcome}. The submission now holds work from {contributors or ['none']}."
        )
        for peer in self._merge_live_children(exclude=delivered.id):
            own_changed, own_deleted = self._merge_own_changes(peer)
            overlap = sorted((set(own_changed) | own_deleted) & landed)
            if peer.id in refreshed:
                tail = (
                    " Your copy had nothing of your own in it, so it now matches the "
                    "current submission — re-read anything you had looked at."
                )
            elif overlap:
                tail = (
                    f" You are editing the same files: {overlap}. Your copy is left as it "
                    "is, so your delivery will be diffed against what you were handed and "
                    "will overwrite theirs in those files."
                )
            else:
                continue
            note = head + tail
            await self.deliver(Envelope(
                from_id="system", to_id=peer.id, content=note,
                tokens=estimate_tokens(note), timestamp=time.time(), mode="steer",
            ))
            self._emit(peer.id, "delivery_notified", {
                "delivered_by": delivered.id, "kept": kept,
                "refreshed": peer.id in refreshed, "overlap": overlap[:10],
            })

    async def _merge_promote_child(self, agent: Agent, reason: str = "") -> dict | None:
        """Fold a child's diff into the submission path, gated by a re-run check.

        Serialized by a lock. The child's changes are computed against the
        immutable round baseline and applied onto the (accumulating) submission
        path — and then the runtime runs the registered verification against
        that merged result. Not against the child's own copy: every party
        verifying only its own artifact is exactly how a config merged from
        three children replaced a working solution unmeasured. A merged state
        that fails, or verifies worse than the best so far, is rolled back.
        """
        if not self._merge_active():
            return None
        copy_dir = getattr(agent, "_merge_copy", None)
        baseline = self._merge_agent_baseline(agent)
        target = self._merge_target()
        if not copy_dir or not Path(copy_dir).is_dir() or baseline is None or target is None:
            return None
        changed, deleted = self._merge_diff(Path(copy_dir), baseline)
        if not changed and not deleted:
            self._emit(agent.id, "merge_promote_skip", {"reason": "no_changes"})
            return {
                "status": "nothing_to_deliver",
                "detail": f"No changes under the submission paths in {copy_dir}.",
            }
        signature = self._merge_change_signature(changed, deleted)
        if signature == getattr(agent, "_merge_promoted_sig", None):
            self._emit(agent.id, "merge_promote_skip", {"reason": "unchanged_since_last"})
            return {
                "status": "unchanged_since_last_delivery",
                "metric": self._merge_last_child_metric.get(agent.id),
                "detail": "Nothing changed since your last delivery; edit files first.",
            }
        spec = self._verification_spec_for(agent)
        async with self._merge_get_lock():
            prev = self._merge_root_dir() / "prev"
            have_prev = self._merge_copy_tree(target, prev)
            self._merge_apply(changed, deleted, target)
            verification = None
            rank = None
            if spec is not None:
                verification = await self._verify_submission_path(agent, spec)
                rank = self._verify_rank(
                    verification["ok"], verification["metric"],
                    bool(spec.get("higher_is_better", True)),
                )
            best_rank = self._merge_best_rank
            proven = self._verify_metric_is_proven(spec) if spec else False
            if verification is not None and verification.get("metric") is not None:
                # Recorded so this channel can estimate its own scatter. Never
                # counts towards the official best: a local check's number is not
                # the judge's verdict and the two are not on one scale.
                self._ledger_note(
                    "verify",
                    verification["metric"],
                    signature=self._merge_scope_signature(target),
                    agent_id=agent.id,
                    counts=False,
                    extra={"ok": bool(verification["ok"])},
                )
            keep = True
            if rank is not None:
                # A merged state that fails its check is never kept, including
                # when nothing better has been measured yet. Defaulting to keep
                # in that case let four failing merges through in one round and
                # then snapshotted the last of them as the state to fall back to.
                if not verification["ok"]:
                    keep = False
                elif best_rank is not None and proven and rank < best_rank:
                    # Worse by this check — but a gap smaller than what
                    # re-measuring the same state would move it is not a gap.
                    #
                    # Unlike the submission gate, an unknown scatter here falls
                    # back to the strict comparison. Each promotion verifies a
                    # different merged state, so this check rarely measures one
                    # state twice and a margin may never become estimable; being
                    # inert until it did would mean keeping every broken merge,
                    # which is the failure the rollback exists for. The costs are
                    # not symmetric either: a wrong rollback loses one delivery,
                    # which `prev` still holds, while a wrong refusal at the
                    # submission gate loses the round's score.
                    best_state = self._ledger_digest(self._merge_best_signature)
                    worse = self._metric_is_worse(
                        [verification["metric"]] if verification["metric"] is not None else [],
                        self._ledger_state_metrics(best_state, "verify")
                        or ([self._merge_best_metric] if self._merge_best_metric is not None else []),
                        "verify",
                    )
                    keep = worse is False
                    if keep:
                        self._emit(agent.id, "merge_keep_within_noise", {
                            "metric": verification["metric"],
                            "best": self._merge_best_metric,
                            "noise": self._measurement_noise("verify"),
                        })
            if keep and rank is not None and verification["ok"]:
                self._record_verified_best(verification["metric"], rank)
            if not keep and have_prev:
                self._merge_restore_snapshot(prev, target)
            agent._merge_promoted_sig = signature
            metric = verification["metric"] if verification else None
            self._merge_last_child_metric[agent.id] = metric
            refreshed = self._merge_refresh_idle_copies(exclude=agent.id)
            self._emit(agent.id, "merge_promote", {
                "changed": len(changed), "deleted": len(deleted),
                "verified": bool(verification and verification["ok"]),
                "metric": metric, "best": self._merge_best_metric, "kept": keep,
                "metric_proven": proven, "copies_refreshed": refreshed,
            })
        await self._notify_peers_of_delivery(
            agent, sorted(changed), keep,
            bool(verification and verification["ok"]), metric, refreshed,
        )
        if verification is None:
            note = (
                "No verification is registered, so this was merged unchecked and can "
                "never be treated as the best state. Register one with the verify tool "
                "so the runtime can prove a merged result still works."
            )
        elif keep:
            note = "Verified against the merged submission and kept."
        else:
            note = (
                "The merged submission verified worse than the best state, so it was "
                "rolled back. Your working copy is untouched — improve it and retry."
            )
        return {
            "status": "delivered",
            "verified": bool(verification and verification["ok"]),
            "metric": metric,
            "best_metric": self._merge_best_metric,
            "kept": keep,
            "metric_proven": proven,
            "changed_files": sorted(changed)[:20],
            "deleted_files": sorted(deleted)[:20],
            "note": note,
            "verification_output": (verification or {}).get("output", "")[-1500:],
        }

    def _merge_workspace_guard_segment(self, target: Path) -> str | None:
        """First path segment of the runtime workspace when it lives inside the
        task directory (EdgeBench puts it at task_cwd/.nanoma-task-work)."""
        try:
            relative = self.config.workspace_root.resolve().relative_to(target.resolve())
        except (ValueError, OSError):
            return None
        return relative.parts[0] if relative.parts else None

    def _merge_redirect_tool_args(self, agent: Agent, tc: "ToolCall") -> None:
        """Rewrite shared-task-directory paths to a merge child's private copy.

        Isolation cannot be advisory. The harness prompt names the shared task
        directory repeatedly ("cd {task_cwd} && ..."), children follow it over a
        single line of task text, and every sibling lands back on the same files
        — observed on ann_vector_search_qps, where four children edited the
        shared directory and their private copies stayed untouched. Rewriting
        the paths that filesystem tools actually receive makes the copy the only
        thing a child can reach, without needing container privileges or model
        compliance.
        """
        copy_dir = getattr(agent, "_merge_copy", None)
        if not copy_dir or not self._merge_active():
            return
        if not (tc.name in _SHELL_TOOLS or tc.name.startswith("ws_")):
            return
        target = self._merge_target()
        if target is None:
            return
        source = str(target).rstrip("/")
        destination = str(copy_dir).rstrip("/")
        if not source or source == destination:
            return

        import re
        # The copy lives under the workspace, which sits inside the task
        # directory, so a naive replace would corrupt paths that already point
        # at the copy. Only rewrite references that leave the workspace alone.
        guard = self._merge_workspace_guard_segment(target)
        pattern = re.escape(source) + (rf"(?!/{re.escape(guard)}\b)" if guard else "")
        hits = 0

        def rewrite(value):
            nonlocal hits
            if isinstance(value, str):
                if source not in value:
                    return value
                new_value, count = re.subn(pattern, destination, value)
                hits += count
                return new_value
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            if isinstance(value, dict):
                return {k: rewrite(v) for k, v in value.items()}
            return value

        if not isinstance(tc.arguments, dict):
            return
        tc.arguments = rewrite(tc.arguments)
        if hits:
            seen = getattr(agent, "_merge_redirects", 0) + hits
            agent._merge_redirects = seen
            if seen <= 3:
                self._emit(agent.id, "merge_path_redirect", {
                    "tool": tc.name, "from": source, "to": destination,
                })

    # ─── keep-best ratchet over the whole run ────────────────────────────
    # The child ratchet only guards what children promote, but the parent edits
    # the submission path directly and nothing checks whether its edits make the
    # score worse. On ann_vector_search_qps that cost the entire run: the parent
    # hand-merged three children's self-reported configs over a measured 3760,
    # the combined config failed validation, and the run ended at 0 with a peak
    # of 4509 already on record. So every measured score snapshots the
    # submission paths, and the best one is restored before the run hands off.

    def _merge_best_dir(self) -> Path:
        return self._merge_root_dir() / "best"

    def _merge_scope_signature(self, root: Path) -> frozenset:
        files = self._merge_rel_files(root)
        return self._merge_change_signature(files, set())

    def _merge_snapshot_affordable(self, target: Path) -> bool:
        import os
        max_mb = float(os.environ.get("NANOMA_MERGE_MAX_MB", "200") or 200)
        max_files = int(os.environ.get("NANOMA_MERGE_MAX_FILES", "40000") or 40000)
        total, count = self._merge_scope_stats(target)
        return not (
            (max_mb > 0 and total > max_mb * 1024 * 1024)
            or (max_files > 0 and count > max_files)
        )

    def _merge_restore_best(self) -> None:
        """Put the best verified state back before the run hands off.

        Restores when the live state verified worse than the best, and also when
        it was edited after its last verification: an unchecked edit is worth
        less than a state that is known to work.
        """
        if not self._merge_active() or self._merge_best_rank is None:
            return
        best_dir = self._merge_best_dir()
        target = self._merge_target()
        if target is None or not best_dir.is_dir():
            return
        current = self._merge_current_rank
        if current is not None and self._merge_scored_signature is not None:
            if self._merge_scope_signature(target) != self._merge_scored_signature:
                current = None  # edited since it was last measured
        if current is not None and current >= self._merge_best_rank:
            return
        changed, deleted = self._merge_diff(best_dir, target)
        if not changed and not deleted:
            return
        self._merge_apply(changed, deleted, target)
        self._emit("root", "merge_best_restored", {
            "best": self._merge_best_metric,
            "replaced": self._merge_current_metric if current is not None else None,
            "changed": len(changed),
            "deleted": len(deleted),
        })

    async def _merge_promote_pending(self) -> None:
        """Fold in children that still hold unpromoted work, verified once.

        A child promotes from its own agent task, which shutdown cancels, so
        anything a child was still working on when the root finished is dropped
        — it only used to survive because children wrote the shared directory
        directly. Applied as a single union and verified once: the union is a
        combination nobody has ever run, which is precisely the artifact that
        needs checking, and if it fails the keep-best restore rewinds it.
        """
        if not self._merge_active():
            return
        baseline = self._merge_baseline_path
        target = self._merge_target()
        if baseline is None or target is None:
            return
        union_changed: dict[str, Path] = {}
        union_deleted: set[str] = set()
        contributors: list[str] = []
        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            copy_dir = getattr(agent, "_merge_copy", None)
            if not copy_dir or not Path(copy_dir).is_dir():
                continue
            changed, deleted = self._merge_diff(
                Path(copy_dir), self._merge_agent_baseline(agent) or baseline
            )
            if not changed and not deleted:
                continue
            if self._merge_change_signature(changed, deleted) == getattr(
                agent, "_merge_promoted_sig", None
            ):
                continue
            union_changed.update(changed)
            union_deleted |= deleted
            contributors.append(agent.id)
        if not contributors:
            return
        union_deleted -= set(union_changed)
        root = next((a for a in self.agents.values() if a.parent is None), None)
        actor = root or self.agents[contributors[0]]
        spec = self._verification_spec_for(actor)
        async with self._merge_get_lock():
            self._merge_apply(union_changed, union_deleted, target)
            verification = None
            if spec is not None:
                verification = await self._verify_submission_path(actor, spec)
                rank = self._verify_rank(
                    verification["ok"], verification["metric"],
                    bool(spec.get("higher_is_better", True)),
                )
                if rank is not None:
                    self._record_verified_best(verification["metric"], rank)
        self._emit("root", "merge_promote_pending", {
            "contributors": contributors,
            "changed": len(union_changed),
            "deleted": len(union_deleted),
            "verified": bool(verification and verification["ok"]),
            "metric": (verification or {}).get("metric"),
        })

    def preserve_final_branches(self, output_dir: Path | str | None = None) -> dict[str, Any]:
        """Freeze the selected submission and every private child worktree.

        This is intentionally a terminal, post-delivery operation. It performs
        no verification, exposes no score to an agent, and does not promote or
        merge anything. EdgeBench calls it only after its closing official
        submission, so ``selected-final`` is byte-for-byte the state that was
        handed to the judge while child candidates remain independent.
        """
        from nanoma.branch_archive import preserve_branch_candidates

        target = self._merge_target()
        if target is None:
            return {
                "preserved": False,
                "reason": "no submission workspace configured",
                "logical_candidate_count": 0,
                "unique_artifact_count": 0,
            }

        root = next(
            (agent for agent in self.agents.values() if agent.parent is None),
            None,
        )
        candidates: list[dict[str, Any]] = [{
            "candidate_id": "selected-final",
            "source_root": str(target),
            "selected": True,
            "kind": "selected",
            "agent_id": root.id if root is not None else None,
            "spawn_parent_agent_id": None,
            "status": getattr(root, "status", None),
            "depth": getattr(root, "depth", 0),
            "turns": getattr(root, "_turns", 0),
            "tokens": getattr(root, "tokens_consumed", 0),
        }]
        for agent in sorted(self.agents.values(), key=lambda item: item.id):
            copy_dir = getattr(agent, "_merge_copy", None)
            if copy_dir is None:
                continue
            candidates.append({
                "candidate_id": f"{agent.id}-final",
                "source_root": str(copy_dir),
                "selected": False,
                "kind": "agent_branch",
                "agent_id": agent.id,
                # This is orchestration lineage, not artifact ancestry: nested
                # agents are seeded from the shared submission state.
                "spawn_parent_agent_id": agent.parent,
                "status": agent.status,
                "depth": agent.depth,
                "turns": agent._turns,
                "tokens": agent.tokens_consumed,
                "promoted_during_run": (
                    getattr(agent, "_merge_promoted_sig", None) is not None
                ),
                "last_local_metric": self._merge_last_child_metric.get(agent.id),
            })

        if output_dir is None:
            base = Path(self.config.log_dir) if self.config.log_dir else self.config.workspace_root
            output_dir = base / "candidate_branches"
        manifest = preserve_branch_candidates(
            candidates=candidates,
            output_dir=output_dir,
            scope_roots=self._merge_scope_roots(),
            ignored_parts={
                ".git", ".lake", ".nanoma-runtime-logs", ".nanoma-task-work",
                "__pycache__", "_task_copy", "_merge",
            },
            run_metadata={
                "runtime_started_at": self._start_time,
                "runtime_finished_at": time.time(),
                "agent_count": len(self.agents),
            },
        )
        manifest["preserved"] = True
        self._emit("root", "candidate_branches_preserved", {
            "path": manifest.get("output_dir"),
            "logical_candidates": manifest.get("logical_candidate_count", 0),
            "available_candidates": manifest.get("available_candidate_count", 0),
            "unique_artifacts": manifest.get("unique_artifact_count", 0),
            "selected_candidate_id": manifest.get("selected_candidate_id"),
        })
        return manifest

    # ─── the gate in front of an official submission ─────────────────────
    # A submission is scarce and irreversible: the judge scores the workspace as
    # it stands and the number goes on the record. One run spent six of them on a
    # state whose algorithm had been renamed, so every one scored zero for
    # "Nothing to run", and it spent one of those 31 seconds after its own
    # aggregate wait timed out, while the verdict it got back recorded
    # local_passed_this_state: false. Both were knowable beforehand, and that
    # signal was only ever computed afterwards, for calibration.

    PREFLIGHT_TIMEOUT_DEFAULT = 120.0

    def _submit_preflight_command(self) -> str:
        return (os.environ.get("NANOMA_SUBMIT_PREFLIGHT") or "").strip()

    def _submit_requires_verified_state(self) -> bool:
        return os.environ.get("NANOMA_SUBMIT_REQUIRE_VERIFIED") == "1"

    def _submit_requires_measured_state(self) -> bool:
        """Whether a submission may ship a state worse than one on record.

        Separate from NANOMA_SUBMIT_REQUIRE_VERIFIED because it rests on
        different evidence: that gate needs the agent to have authored a check
        that passes, this one needs only a measurement the run has already seen.
        A run whose check never works is exactly the run that needs this.

        On by default, but only where the ledger can hold anything: without a
        merge target there is no state to fingerprint and no verdict to attach to
        one, so leaving it enabled there would put a gate in front of every
        submission that could never have evidence to act on.
        """
        if not self._merge_active():
            return False
        return os.environ.get("NANOMA_SUBMIT_REQUIRE_MEASURED", "1") == "1"

    def _submit_requires_complete_aggregate(self) -> bool:
        """Whether a timed-out aggregate wait may still spend a submission."""
        return os.environ.get("NANOMA_SUBMIT_REQUIRE_COMPLETE", "1") == "1"

    def _submit_gate_enabled(self) -> bool:
        return (
            bool(self._submit_preflight_command())
            or self._submit_requires_verified_state()
            or self._submit_requires_measured_state()
        )

    async def _run_preflight(self, command: str, workdir: Path) -> tuple[int | None, str]:
        """Run a submission preflight; the exit code is the whole verdict.

        Unlike a `verify` check this prints nothing the runtime parses — it is a
        sanity condition the task imposes, such as the submitted algorithm still
        being discoverable under the name the judge runs.
        """
        try:
            timeout = float(
                os.environ.get("NANOMA_SUBMIT_PREFLIGHT_TIMEOUT")
                or self.PREFLIGHT_TIMEOUT_DEFAULT
            )
        except ValueError:
            timeout = self.PREFLIGHT_TIMEOUT_DEFAULT
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workdir),
                env={**os.environ, "WORKSPACE": str(workdir)},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:
            return -1, f"{type(exc).__name__}: {exc}"
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None, f"preflight timed out after {timeout:.0f}s"
        return proc.returncode, stdout.decode(errors="replace")[-2000:]

    def _submit_incomplete_block(self, agent: Agent, wait: dict) -> dict | None:
        """Stop a timed-out aggregate wait from silently becoming a submission.

        The wait exists so a submission describes the whole workspace rather than
        "a state that no check has passed and that nobody has finished writing" —
        and when it timed out, that is exactly what got sent, because the timeout
        was invisible at the call site. Waiting out the full budget read to the
        agent as permission to go.

        Refused once, then allowed. A child can hang for the rest of the run, and
        a gate that refuses forever would turn that into a zero — the cost of
        being wrong here is the whole score. So the first attempt is refused with
        the names, the files and the ways out; submitting again proceeds. The
        point is to make the choice informed, not to make it for the agent.
        """
        if not self._submit_requires_complete_aggregate():
            return None
        if not wait.get("timed_out"):
            agent._aggregate_wait_refused = False
            return None
        holding = wait.get("holding") or {}
        if not holding:
            return None  # outstanding, but holding nothing: nothing is being lost
        if getattr(agent, "_aggregate_wait_refused", False):
            self._emit(agent.id, "submit_incomplete_allowed", {"holding": holding})
            return None

        agent._aggregate_wait_refused = True
        children = sorted(holding)
        reason = (
            f"the wait for {children} ran out after "
            f"{self._AGGREGATE_WAIT_SECONDS:.0f}s and they are still holding changes "
            f"that belong in this submission: {json.dumps(holding)}. Submitting now "
            "sends a state that is knowably unfinished. Either kill them and take "
            "what has already merged, message them for what they have, or submit "
            "again to go ahead without it — this refusal is not repeated."
        )
        self._emit(agent.id, "submit_blocked", {
            "reason": "aggregate_incomplete", "holding": holding,
            "waited_for": wait.get("waited"),
        })
        return {"submitted": False, "blocked": "aggregate_incomplete", "reason": reason}

    def _submit_regression_block(self, agent: Agent) -> dict | None:
        """Refuse to ship a state that is worse than one already measured.

        The keep-best ratchet restores a better state only at the end of the run
        and only when the agent's own check produced it. Nothing stopped a
        submission from being spent, mid-run, on a state the run had already
        measured as worse — or on one nobody had measured while a better one sat
        on record. Both happened: one run shrank the scored config to fit a local
        timeout, deleting the parameters behind its own best score, and spent its
        remaining submissions on the smaller one.

        Only refuses on evidence: without a recorded measurement there is
        nothing to be worse than, and a state that is not worse beyond the
        measurement noise passes. It used to compare this state's best single
        number against the record's best single number, which on a task whose
        repeat measurements vary by 16% meant refusing states that were fine and
        passing states that were worse, both at around chance.
        """
        if not self._submit_requires_measured_state():
            return None
        best = self._ledger_best()
        target = self._merge_target()
        if best is None or target is None or not best.get("state"):
            return None

        current = self._ledger_digest(self._merge_scope_signature(target))
        if current == best["state"]:
            return None

        best_metric = float(best["metric"])
        source = str(best.get("source") or "official")
        here = self._ledger_state_metrics(current, source)
        there = self._ledger_state_metrics(best["state"], source) or [best_metric]
        sign = 1 if self._ledger_higher_is_better() else -1
        if here and self._metric_is_worse(here, there, source) is not True:
            # Measured, and not worse by more than the noise. That includes the
            # case where the run cannot yet estimate its noise: a difference it
            # cannot distinguish from measurement scatter is not grounds to
            # spend nothing, and refusing here costs the whole score.
            return None

        measured = (
            f"this state measured {max(here, key=lambda m: sign * m):g}"
            if here
            else "this state has never been measured"
        )
        restorable = self._ledger_snapshot_dir(best["state"]).is_dir()
        reason = (
            f"a better state is already on record: {best_metric:g}"
            + (f" (round {best['round']})" if best.get("round") else "")
            + f", while {measured}. Spending a submission here would ship the worse "
            "of the two."
            + (
                f" The recorded state is kept at {self._ledger_snapshot_dir(best['state'])} "
                "— restore it, or measure this one and beat it."
                if restorable
                else " Measure this state before spending a submission on it."
            )
        )
        self._emit(agent.id, "submit_blocked", {
            "reason": "worse_than_measured",
            "best_metric": best_metric,
            "best_state": best["state"],
            "current_state": current,
            "current_metrics": here,
            "noise": self._measurement_noise(source),
            "margin": self._noise_margin(source, at=best_metric),
            "restorable": restorable,
        })
        return {"submitted": False, "blocked": "worse_than_measured", "reason": reason}

    async def _submit_preflight(self, agent: Agent) -> dict | None:
        """Why this submission must not be spent, or None to let it through.

        Deliberately inert when there is nothing to judge with: a run that
        registered no working check must still be able to submit, or a broken
        gate turns into a zero. It only refuses when the evidence to refuse on
        actually exists.
        """
        if not self._submit_gate_enabled():
            return None

        if self._submit_requires_verified_state():
            spec = self._authoritative_spec()
            target = self._merge_target()
            signature = self._merge_scope_signature(target) if target else None
            if spec is not None and signature is not None and signature != getattr(
                self, "_merge_scored_signature", None
            ):
                reason = (
                    "no check has passed the workspace as it stands. The registered "
                    f"check ({spec.get('name') or 'unnamed'}) has not been run against "
                    "this state, so the submission would score something nobody has "
                    "measured. Run `verify` on it first; if it fails, fix the state "
                    "instead of spending a submission on it."
                )
                self._emit(agent.id, "submit_blocked", {
                    "reason": "unverified_state", "check": spec.get("name"),
                })
                return {"submitted": False, "blocked": "unverified_state", "reason": reason}

        regression = self._submit_regression_block(agent)
        if regression is not None:
            return regression

        command = self._submit_preflight_command()
        if command:
            workdir = self._merge_target() or agent.workspace
            code, output = await self._run_preflight(command, Path(workdir))
            if code != 0:
                tail = " ".join((output or "").split())[-600:]
                self._emit(agent.id, "submit_blocked", {
                    "reason": "preflight_failed", "exit_code": code,
                    "output_tail": tail[-400:],
                })
                return {
                    "submitted": False,
                    "blocked": "preflight_failed",
                    "reason": (
                        "the submission preflight failed, so the judge would reject "
                        f"this state before scoring it. `{command}` exited {code}: "
                        f"{tail}"
                    ),
                }
        return None

    def _merge_wrap_submit(self, agent: Agent, tools: dict[str, dict]) -> dict[str, dict]:
        """Wrap `submit`: deliver a child's own work first, and gate what ships.

        A child works in a private copy, so a raw submit would send the parent's
        state to the judge — a number about somebody else's work. Its changes
        are merged and verified first; whether to then spend an official
        submission stays the agent's own call.

        The gate runs last, immediately before the submission is spent, so it
        judges the state that will actually be scored rather than the one that
        existed before the merge.
        """
        if "submit" not in tools:
            return tools
        original = tools["submit"]

        if not self._merge_active():
            if not self._submit_gate_enabled():
                return tools

            async def gated_submit(args, ag, runtime):
                blocked = await self._submit_preflight(ag)
                if blocked is not None:
                    return blocked
                return await original["handler"](args, ag, runtime)

            wrapped_plain = dict(original)
            wrapped_plain["handler"] = gated_submit
            patched_plain = dict(tools)
            patched_plain["submit"] = wrapped_plain
            return patched_plain

        is_child = bool(getattr(agent, "_merge_copy", None))
        if not is_child:
            async def aggregate_then_submit(args, ag, runtime):
                return await self._submit_as_root(args, ag, runtime, original["handler"])

            wrapped_parent = dict(original)
            wrapped_parent["handler"] = aggregate_then_submit
            patched_parent = dict(tools)
            patched_parent["submit"] = wrapped_parent
            return patched_parent

        async def merge_submit(args, ag, runtime):
            delivery = await self._merge_promote_child(
                ag, reason=str((args or {}).get("reason") or f"child {ag.id} submit")
            )
            if delivery is None:
                blocked = await self._submit_preflight(ag)
                if blocked is not None:
                    return blocked
                self._note_state_being_submitted(ag, "submit", args)
                return await original["handler"](args, ag, runtime)
            if not delivery.get("kept"):
                return delivery
            blocked = await self._submit_preflight(ag)
            if blocked is not None:
                return {"delivery": delivery, "submission": blocked}
            # After the child's own work went in, so the verdict is filed against
            # the state that was sent rather than the one that preceded it.
            self._note_state_being_submitted(ag, "submit", args)
            result = await original["handler"](args, ag, runtime)
            return {"delivery": delivery, "submission": result}

        wrapped = dict(original)
        wrapped["handler"] = merge_submit
        patched = dict(tools)
        patched["submit"] = wrapped
        return patched

    def _container_memory_limit_bytes(self) -> int | None:
        """Best-effort memory-cgroup limit for this process (v2 then v1)."""
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                raw = Path(path).read_text().strip()
            except OSError:
                continue
            if not raw or raw == "max":
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if 0 < value < (1 << 62):
                return value
        return None

    def _active_children_total(self) -> int:
        return sum(
            1 for a in self.agents.values()
            if a.parent and getattr(a, "status", None) not in ("done", "failed", "killed")
        )

    def _max_parallel_children(self) -> int | None:
        """Cap on concurrently running children, derived from the memory limit.

        Benchmark work (builds, evaluations, dataset loads) runs inside the same
        memory cgroup as the runtime, so unbounded fan-out can OOM the entire
        run — which is exactly what killed the ann-benchmarks task. This is a
        hard runtime constraint, deliberately NOT delegated to the model.
        Returns None when no limit can be determined (no cap).
        """
        import os
        explicit = os.environ.get("NANOMA_MAX_PARALLEL_CHILDREN", "").strip()
        if explicit:
            try:
                value = int(explicit)
                return value if value > 0 else None
            except ValueError:
                pass
        limit = self._container_memory_limit_bytes()
        if not limit:
            return None
        per_child_mb = float(os.environ.get("NANOMA_CHILD_MEM_MB", "1500") or 1500)
        reserve_mb = float(os.environ.get("NANOMA_RUNTIME_RESERVE_MB", "1500") or 1500)
        if per_child_mb <= 0:
            return None
        usable_mb = max(0.0, limit / (1024 * 1024) - reserve_mb)
        return max(1, int(usable_mb // per_child_mb))

    def _spawn_memory_block_reason(self) -> str | None:
        """Why a new child must not start right now (memory headroom)."""
        cap = self._max_parallel_children()
        if cap is None:
            return None
        active = self._active_children_total()
        if active < cap:
            return None
        limit = self._container_memory_limit_bytes()
        limit_mb = f"{limit / (1024 * 1024):.0f}MB" if limit else "unknown"
        return (
            f"memory headroom: {active}/{cap} child agents already running under a "
            f"{limit_mb} memory limit; wait for one to finish before spawning more"
        )

    def _active_contributor_ids(self, exclude_ids: "set[str] | frozenset[str]" = frozenset()) -> list[str]:
        """Ids of still-running agents that hold a private submit-path copy."""
        exclude = {x for x in exclude_ids if x}
        out: list[str] = []
        for a in self.agents.values():
            if a.id in exclude:
                continue
            if getattr(a, "status", None) in ("done", "failed", "killed"):
                continue
            if getattr(a, "_merge_copy", None):
                out.append(a.id)
        return out

    def _agents_touching_files(
        self,
        files: "set[str]",
        exclude_ids: "set[str] | frozenset[str]" = frozenset(),
    ) -> dict[str, list[str]]:
        """Map active-agent-id -> which of `files` that agent is currently editing.

        Computed from each active agent's private copy diffed against the round
        baseline, so a receiver can tell whether an incoming diff overlaps work
        that other agents have started but not yet delivered.
        """
        baseline = self._merge_baseline_path
        if not files or baseline is None or not Path(baseline).is_dir():
            return {}
        wanted = set(files)
        exclude = {x for x in exclude_ids if x}
        out: dict[str, list[str]] = {}
        for a in self.agents.values():
            if a.id in exclude:
                continue
            if getattr(a, "status", None) in ("done", "failed", "killed"):
                continue
            copy = getattr(a, "_merge_copy", None)
            if not copy or not Path(copy).is_dir():
                continue
            changed, deleted = self._merge_own_changes(a)
            touched = (set(changed) | set(deleted)) & wanted
            if touched:
                out[a.id] = sorted(touched)
        return out

    def _apply_spawn_todolist_gate(self, agent, turn_tools, tool_policy):
        """Keep same-model planning as the only model-facing spawn path.

        Direct `spawn`, `spawn_many`, and `task_spawn` calls are never offered
        by this branch. At a `task_create` planning node the worker's own model
        may approve the internal `meta_spawn` execution primitive through
        `_spawn_judge_at_plan`.

        Only ever removes tools (never adds), so it is safe to run after the
        state-based scoping passes.
        """
        removed: list[str] = []
        gated = dict(turn_tools)
        for raw in ("spawn", "spawn_many", "task_spawn"):
            if raw in gated:
                gated.pop(raw, None)
                removed.append(raw)

        if removed:
            try:
                existing = list(getattr(tool_policy, "removed_tools", []) or [])
                tool_policy.removed_tools = sorted(set(existing) | set(removed))
                base = getattr(tool_policy, "reason", "") or ""
                reason = "same_model_spawn_judge_only"
                tool_policy.reason = f"{base},{reason}" if base else reason
                tool_policy.scoped_tools = list(gated)
            except Exception:
                pass
        return gated, tool_policy

    _VERIFY_NUDGE_AFTER_TURNS = 6
    _VERIFY_NUDGE_LIMIT = 3
    _DELIVERY_JUDGE_TIMEOUT = 180.0
    _AGGREGATE_WAIT_SECONDS = float(os.environ.get("NANOMA_AGGREGATE_WAIT_SECONDS", "900") or 900)
    _AGGREGATE_POLL_SECONDS = 2.0

    def _verification_nudge(self, agent: Agent) -> str | None:
        """Prompt to register a check, and follow up when an attempt fails.

        Agents measure their work constantly but as throwaway one-liners, so
        without this the machinery stays inert. One prompt is not enough either:
        in a live round all four agents were nudged, one tried, its command died
        in a second, and nobody tried again for the rest of the iteration. A
        failed attempt is exactly when another word is worth saying.
        """
        if not self._merge_active() or "verify" not in self._all_tools():
            return None
        if getattr(agent, "_verify_spec", None):
            return None
        if agent._turns < self._VERIFY_NUDGE_AFTER_TURNS:
            return None
        sent = getattr(agent, "_verify_nudges", 0)
        failures = getattr(agent, "_verify_failures", 0)
        if sent and not (failures >= sent and sent < self._VERIFY_NUDGE_LIMIT):
            return None
        agent._verify_nudges = sent + 1
        if not sent:
            return (
                "<system-reminder>You have not registered a check with the verify tool. "
                "Your own measurements are not visible to the runtime, so nothing can be "
                "re-run against the merged result of several agents' work — and that "
                "merged state, which nobody measures, is what actually gets submitted. "
                "Register the check you already run (build, tests, the task's benchmark) "
                'so it ends with a line like {"ok": true, "metric": 3760.0}. Until then, '
                "no state can be identified as the best one to fall back to."
                "</system-reminder>"
            )
        reason = getattr(agent, "_verify_last_failure", "") or "it did not pass"
        return (
            f"<system-reminder>Your verify attempt did not register: {reason} "
            "Nothing is measuring your work yet, so it cannot be kept over anything "
            "else. Try again with a command you have already seen succeed in the "
            "shell, and append the verdict rather than rewriting the whole thing, e.g. "
            "`<your working command> && python3 -c \"import json;print(json.dumps("
            "{'ok': True, 'metric': <the number you care about>}))\"`. Keep it inside "
            "your own working directory and give it a timeout long enough to finish."
            "</system-reminder>"
        )

    def _history_with_todo_reminder(self, agent: Agent) -> list:
        """agent.history plus ephemeral trailing reminders, if any.

        The reminders (Claude-Code-style state re-injection + staleness nudge)
        are transient: appended only to the per-call message list, never to the
        persisted history, so they always reflect current state and do not
        accumulate in context. Best-effort — if the optimizations package is
        absent or the tools are disabled, returns agent.history unchanged.
        """
        extras: list[str] = []
        tools = self._all_tools()
        if "task_create" in tools:
            try:
                from optimizations.todo_tools import render_todo_reminder
                reminder = render_todo_reminder(agent, self)
            except Exception:
                reminder = None
            if reminder:
                extras.append(reminder)
                self._emit(agent.id, "todo_reminder_injected", {
                    "turn": agent._turns,
                    "chars": len(reminder),
                })
        try:
            nudge = self._verification_nudge(agent)
        except Exception:
            nudge = None
        if nudge:
            extras.append(nudge)
            self._emit(agent.id, "verify_nudge_injected", {"turn": agent._turns})
        if not extras:
            return agent.history
        return [*agent.history, *({"role": "user", "content": e} for e in extras)]

    def _all_tools(self) -> dict[str, dict[str, Any]]:
        from nanoma.meta import META_TOOLS
        return {
            name: tool
            for name, tool in {
                **WORK_TOOLS,
                **WORKSPACE_TOOLS,
                **META_TOOLS,
                **self.config.extra_tools,
            }.items()
            if name not in self.config.disabled_tools
        }

    def _execute_rebirth(self, agent: Agent):
        params = agent._rebirth_pending
        agent._rebirth_pending = None
        summary = params["summary"]
        files = params.get("files", [])
        new_task = params.get("new_task")
        new_bio = params.get("new_bio")

        # ─── Save full conversation to file before wiping ────────────────
        import time as _time
        timestamp = _time.strftime("%Y%m%d_%H%M%S")
        archive_filename = f"{agent.id}_{timestamp}.jsonl"
        archive_path = agent.workspace / ".rebirth" / archive_filename
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with open(archive_path, "w", encoding="utf-8") as f:
            for msg in agent.history:
                f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")

        archive_rel = f".rebirth/{archive_filename}"
        # ─────────────────────────────────────────────────────────────────

        system_msg = agent.history[0] if agent.history else None
        agent.history = []
        if system_msg:
            if new_task:
                system_msg["content"] = system_msg["content"].replace(
                    f"Your task: {agent.task}", f"Your task: {new_task}")
                agent.task = new_task
            agent.history.append(system_msg)
        if new_bio:
            agent.bio = new_bio

        file_section = ""
        if files:
            file_section = "\n\nKey files:\n" + "\n".join(f"- {f}" for f in files)

        msg_count = sum(1 for _ in open(archive_path))
        rebirth_notice = (
            f"[Rebirth — context reset]\n\n"
            f"⚠️ Your previous conversation ({msg_count} messages) "
            f"has been compressed and saved to: `{archive_rel}`\n"
            f"Use `ws_read_file` or `shell('cat {archive_rel}')` to review if needed.\n\n"
            f"## Progress\n{summary}{file_section}"
        )
        agent.history.append({"role": "user", "content": rebirth_notice})
        agent.context_tokens = count_message_tokens(agent.history)

    async def _execute_tool(self, tc: ToolCall, agent: Agent, tools: dict) -> Any:
        self._merge_redirect_tool_args(agent, tc)
        fixed_context = self._fixed_spawn_context(agent)
        if fixed_context is not None:
            checkpoint, specs, metadata = fixed_context
            required_tool = str(
                metadata.get("spawn_tool")
                or ("spawn_many" if len(specs) > 1 else "spawn")
            )
            if tc.name != required_tool:
                reason = (
                    f"checkpoint {checkpoint.key} requires {required_tool}, not {tc.name}; "
                    "the attempted tool came from a turn that started before the checkpoint"
                )
                self._emit(agent.id, "fixed_checkpoint_stale_tool_block", {
                    "checkpoint": checkpoint.key,
                    "attempted_tool": tc.name,
                    "required_tool": required_tool,
                    "reason": reason,
                })
                return {
                    "error": reason,
                    "checkpoint": checkpoint.key,
                    "required_tool": required_tool,
                    "instruction": (
                        f"Discard the stale action and call {required_tool} as the next NanoMA "
                        "tool. Do not deliver, finish, query, wait, or continue research first."
                    ),
                }

        tool_info = tools.get(tc.name)
        if not tool_info:
            agent._shell_activity.unavailable_tool_calls += 1
            state = self._compute_tool_policy_state(agent)
            if state.delivery_phase == "deliver" and self._has_delivery_candidate(agent):
                candidate_result, candidate_path = self._submitted_result_text(agent, None, {})
                if candidate_result:
                    agent.status = "done"
                    agent.result = candidate_result
                    self._emit(agent.id, "delivery_unavailable_done", {
                        "tool": tc.name,
                        "candidate": candidate_path,
                        "result_preview": candidate_result[:200],
                    })
            stop_after_unavailable = int(self.config.tool_policy_delivery_prepare_unavailable_stop_after or 0)
            if (
                stop_after_unavailable > 0
                and state.delivery_phase == "prepare"
                and agent._shell_activity.unavailable_tool_calls >= stop_after_unavailable
            ):
                agent.status = "done"
                agent.result = agent.result or "[Stopped after repeated unavailable tool calls during delivery preparation]"
            return {
                "error": f"Unknown or unavailable tool this turn: {tc.name}",
                "available_tools": sorted(tools),
                "instruction": (
                    "Use only one of the available_tools listed here. If you have enough evidence, "
                    "write the final artifact with ws_create_file and call set_status(done, result=...)."
                ),
            }
        malformed_write_block = self._malformed_write_recovery_block_reason(agent, tc)
        if malformed_write_block:
            self._emit(agent.id, "malformed_write_recovery_block", {
                "tool": tc.name,
                "reason": malformed_write_block,
            })
            return {
                "error": malformed_write_block,
                "required_behavior": (
                    "write the complete artifact from its first line with ws_create_file, "
                    "tb_write_file, or a shell overwrite"
                ),
            }
        handler = tool_info["handler"]
        try:
            if tool_info.get("is_meta"):
                requested_completion = (
                    tc.name in {"submit", "deliver_to_parent"}
                    or (
                        tc.name == "set_status"
                        and str(tc.arguments.get("status", "done")) == "done"
                    )
                )
                if requested_completion:
                    active_child_block = self._active_child_completion_block_reason(agent)
                    if active_child_block:
                        self._emit(agent.id, "active_child_completion_block", {
                            "tool": tc.name,
                            "reason": active_child_block,
                        })
                        return {
                            "error": active_child_block,
                            "required_tool_behavior": [
                                "query", "wait", "send", "kill", "get_cost"
                            ],
                        }
                if self._fixed_plan and tc.name in _CREATE_TOOLS:
                    fixed_spawn_error = self._fixed_prepare_spawn_tool_call(tc, agent)
                    if fixed_spawn_error:
                        self._emit(agent.id, "fixed_orchestration_spawn_block", {
                            "tool": tc.name,
                            "checkpoint_role": self._fixed_agent_keys.get(
                                agent.id, "unmanaged-child"
                            ),
                            "reason": fixed_spawn_error,
                        })
                        return {
                            "error": fixed_spawn_error,
                            "required_behavior": (
                                "continue assigned scope until a checkpoint asks this agent to "
                                "call NanoMA's spawn or spawn_many tool"
                            ),
                        }
                if tc.name in {"set_status", "submit"}:
                    requested_done = (
                        tc.name == "submit"
                        or str(tc.arguments.get("status", "done")) == "done"
                    )
                    if (
                        requested_done
                        and agent.parent is None
                        and self._fixed_plan
                        and self._fixed_plan.direct_build_gate
                        and self._fixed_source_write_seq != self._fixed_last_green_write_seq
                        and not self._fixed_source_writes_in_flight
                    ):
                        await self._fixed_run_direct_lake_build(
                            agent.id,
                            reason=f"root_finish:{tc.name}",
                        )
                    fixed_block_reason = (
                        self.fixed_orchestration_completion_block_reason(agent)
                        if requested_done
                        else None
                    )
                    if not fixed_block_reason and requested_done:
                        fixed_block_reason = self._fixed_finish_block_reason(agent, tc.name)
                    if fixed_block_reason:
                        initial_checkpoint_fired = bool(
                            self._fixed_plan
                            and self._fixed_plan.checkpoints
                            and self._fixed_plan.checkpoints[0].key in self._fixed_fired_checkpoints
                        )
                        if initial_checkpoint_fired:
                            self._fixed_completion_waiting.add(agent.id)
                        self._emit(agent.id, "fixed_orchestration_completion_block", {
                            "tool": tc.name,
                            "reason": fixed_block_reason,
                        })
                        return {
                            "error": fixed_block_reason,
                            "required_tool_behavior": ["query", "wait", "send", "get_cost"],
                        }
                    final_block_reason = self._final_evidence_block_reason(agent)
                    if final_block_reason:
                        self._emit(agent.id, "final_evidence_tool_block", {
                            "tool": tc.name,
                            "reason": final_block_reason,
                        })
                        return {
                            "error": final_block_reason,
                            "required_tool_behavior": ["query", "wait", "spawn", "ws_read_file", "get_cost"],
                        }
                state = self._compute_tool_policy_state(agent)
                finalize_forced = (
                    not self._low_concurrency_bypass_active(agent)
                    and (
                        agent._shell_activity.finalize_after_web_saturation
                        or state.delivery_phase == "deliver"
                    )
                )
                if (
                    tc.name == "set_status"
                    and finalize_forced
                ):
                    requested_status = str(tc.arguments.get("status", "done"))
                    requested_result = str(tc.arguments.get("result", "") or "").strip()
                    reason = (
                        "after web saturation"
                        if agent._shell_activity.finalize_after_web_saturation
                        else "during finalization"
                    )
                    if requested_status == "idle":
                        return {
                            "error": (
                                f"idle is unavailable {reason}; write the deliverable, then call "
                                "set_status(done, result=...) or submit instead"
                            ),
                            "required_status": "done",
                        }
                    if requested_status == "done" and not requested_result:
                        return {
                            "error": (
                                f"empty done result is unavailable {reason}; write the deliverable, "
                                "then call set_status(done, result=<answer>) or submit"
                            ),
                            "required_status": "done",
                            "required_result": "non-empty answer",
                        }
                freeze_for_finish = bool(
                    self._fixed_plan
                    and agent.parent is None
                    and (
                        tc.name == "submit"
                        or (
                            tc.name == "set_status"
                            and str(tc.arguments.get("status", "done")) == "done"
                        )
                    )
                )
                held = self._finish_time_situation(agent, tc)
                if held is not None:
                    return held
                if freeze_for_finish:
                    self._fixed_source_writes_frozen = True
                self._note_state_being_submitted(agent, tc.name, tc.arguments)
                try:
                    result = await handler(tc.arguments, agent, self)
                except Exception:
                    if freeze_for_finish:
                        self._fixed_source_writes_frozen = False
                    if self._fixed_plan and tc.name in _CREATE_TOOLS:
                        checkpoint_key = self._fixed_pending_checkpoint_by_parent.get(agent.id)
                        if checkpoint_key:
                            self._fixed_release_spawn_request(agent.id, checkpoint_key)
                    raise
                if self._fixed_plan and tc.name in _CREATE_TOOLS:
                    result = await self._fixed_record_spawn_tool_result(tc, agent, result)
                self._track_delivery_activity(agent, tc.name, tc.arguments, result)
                await self._calibrate_from_tool_result(agent, tc.name, tc.arguments, result)
                if (
                    tc.name == "submit"
                    and agent.status == "running"
                    and isinstance(result, dict)
                    and "error" not in result
                    and finalize_forced
                ):
                    submitted_result, submitted_path = self._submitted_result_text(agent, tc.arguments, result)
                    agent.status = "done"
                    agent.result = submitted_result or agent.result or f"[Submitted {submitted_path or 'artifact'}]"
                    self._emit(agent.id, "delivery_submit_done", {
                        "submitted": submitted_path or result.get("submitted"),
                        "result_preview": (agent.result or "")[:200],
                    })
                if freeze_for_finish and (
                    agent.status == "running"
                    or (isinstance(result, dict) and "error" in result)
                ):
                    self._fixed_source_writes_frozen = False
                return result
            else:
                child_final_block = self._child_shared_final_write_block_reason(agent, tc)
                if child_final_block:
                    self._emit(agent.id, "child_shared_final_write_block", {
                        "tool": tc.name,
                        "path": str(tc.arguments.get("path") or tc.arguments.get("file") or ""),
                        "reason": child_final_block,
                    })
                    return {
                        "error": child_final_block,
                        "required_behavior": (
                            "write a private evidence note if needed, then call deliver_to_parent with the "
                            "answer, evidence, confidence, and method"
                        ),
                    }
                if self._fixed_plan and tc.name in _DELIVERY_WRITE_TOOLS:
                    ownership_block = self._fixed_write_block_reason(agent, tc)
                    if ownership_block:
                        self._emit(agent.id, "fixed_ownership_block", {
                            "tool": tc.name,
                            "role": self._fixed_agent_keys.get(agent.id),
                            "reason": ownership_block,
                            "paths": self._fixed_tool_write_paths(tc),
                        })
                        return {
                            "error": ownership_block,
                            "required_behavior": "edit only assigned files or send a handoff to parent",
                        }
                if self._fixed_plan and tc.name in {"shell", "tb_shell"}:
                    shell_block = self._fixed_shell_block_reason(
                        agent,
                        str(tc.arguments.get("command", "")),
                    )
                    if shell_block:
                        self._emit(agent.id, "fixed_shell_write_block", {
                            "tool": tc.name,
                            "role": self._fixed_agent_keys.get(agent.id),
                            "reason": shell_block,
                        })
                        return {
                            "error": shell_block,
                            "required_behavior": "use read/build shell commands and structured owned-file writes",
                        }
                if (
                    self._fixed_plan
                    and agent.parent is not None
                    and tc.name in {"shell", "tb_shell"}
                    and re.search(
                        r"(?:^|[;&|\s])sforge-submit(?:\s|$)",
                        str(tc.arguments.get("command", "")),
                    )
                ):
                    self._emit(agent.id, "fixed_orchestration_submit_block", {
                        "tool": tc.name,
                        "checkpoint_role": self._fixed_agent_keys.get(agent.id, "unmanaged-child"),
                    })
                    return {
                        "error": "only the fixed orchestration root may run sforge-submit",
                        "required_behavior": "send build-backed handoff to parent",
                    }
                command = str(tc.arguments.get("command", ""))
                if tc.name in {"shell", "tb_shell"}:
                    repaired_python = _repair_bare_python_block(command)
                    if repaired_python != command:
                        tc.arguments["command"] = repaired_python
                        command = repaired_python
                        self._emit(agent.id, "shell_fenced_python_repaired", {
                            "tool": tc.name,
                            "body_lines": max(0, len(repaired_python.splitlines()) - 2),
                        })
                    shared_references = len(re.findall(r"\$(?:\{SHARED\}|SHARED)", command))
                    workspace_references = len(re.findall(r"\$(?:\{WORKSPACE\}|WORKSPACE)", command))
                    expanded_command = self._expand_shell_runtime_paths(agent, command)
                    if expanded_command != command:
                        tc.arguments["command"] = expanded_command
                        command = expanded_command
                        self._emit(agent.id, "shell_runtime_paths_expanded", {
                            "tool": tc.name,
                            "shared_references": shared_references,
                            "workspace_references": workspace_references,
                        })
                if tc.name == "shell":
                    wrong_fulltext_source_block = (
                        self._arxiv_wrong_fulltext_html_source_block_reason(agent, command)
                    )
                    if wrong_fulltext_source_block:
                        self._emit(agent.id, "arxiv_wrong_fulltext_html_source_block", {
                            "reason": wrong_fulltext_source_block,
                        })
                        return {
                            "error": wrong_fulltext_source_block,
                            "required_behavior": "use the ar5iv converted full-paper HTML source",
                        }
                    outdated_advanced_selector_block = (
                        self._arxiv_outdated_advanced_selector_block_reason(agent, command)
                    )
                    if outdated_advanced_selector_block:
                        self._emit(agent.id, "arxiv_outdated_advanced_selector_block", {
                            "reason": outdated_advanced_selector_block,
                        })
                        return {
                            "error": outdated_advanced_selector_block,
                            "required_behavior": (
                                "parse li.arxiv-result and its p.list-title identifier link"
                            ),
                        }
                    malformed_advanced_block = self._arxiv_malformed_advanced_search_block_reason(
                        agent,
                        command,
                    )
                    if malformed_advanced_block:
                        self._emit(agent.id, "arxiv_malformed_advanced_search_block", {
                            "domain": _web_command_domain(command),
                            "reason": malformed_advanced_block,
                        })
                        return {
                            "error": malformed_advanced_block,
                            "required_behavior": (
                                "use separate numbered author and title field parameters"
                            ),
                        }
                    identifier_month_block = self._arxiv_identifier_month_assumption_block_reason(
                        agent,
                        command,
                    )
                    if identifier_month_block:
                        self._emit(agent.id, "arxiv_identifier_month_assumption_block", {
                            "reason": identifier_month_block,
                        })
                        return {
                            "error": identifier_month_block,
                            "required_behavior": (
                                "inspect candidate version histories instead of inferring an identifier prefix"
                            ),
                        }
                    atom_header_block = self._arxiv_atom_feed_header_parse_block_reason(
                        agent,
                        command,
                    )
                    if atom_header_block:
                        self._emit(agent.id, "arxiv_atom_feed_header_parse_block", {
                            "reason": atom_header_block,
                        })
                        return {
                            "error": atom_header_block,
                            "required_behavior": (
                                "select the Atom entry before reading paper metadata fields"
                            ),
                        }
                    exact_title_api_block = self._arxiv_exact_title_api_block_reason(agent, command)
                    if exact_title_api_block:
                        self._emit(agent.id, "arxiv_exact_title_api_block", {
                            "domain": _web_command_domain(command),
                            "reason": exact_title_api_block,
                        })
                        return {
                            "error": exact_title_api_block,
                            "required_behavior": (
                                "use arXiv advanced HTML fields or named-candidate metadata"
                            ),
                        }
                    full_page_dump_block = self._pdf_full_page_dump_block_reason(agent, command)
                    if full_page_dump_block:
                        self._emit(agent.id, "pdf_full_page_dump_block", {
                            "reason": full_page_dump_block,
                        })
                        return {
                            "error": full_page_dump_block,
                            "required_behavior": (
                                "print bounded caption/axis evidence and render the matched figure page"
                            ),
                        }
                    partial_bibliography_block = self._partial_bibliography_scope_block_reason(
                        agent,
                        command,
                    )
                    if partial_bibliography_block:
                        self._emit(agent.id, "partial_bibliography_scope_block", {
                            "reason": partial_bibliography_block,
                        })
                        return {
                            "error": partial_bibliography_block,
                            "required_behavior": (
                                "locate the bibliography heading and parse every numbered record from there"
                            ),
                        }
                    bibliography_filter_block = self._bibliography_line_filter_block_reason(agent, command)
                    if bibliography_filter_block:
                        self._emit(agent.id, "bibliography_line_filter_block", {
                            "reason": bibliography_filter_block,
                        })
                        return {
                            "error": bibliography_filter_block,
                            "required_behavior": (
                                "segment complete numbered citations, join continuation lines, then filter"
                            ),
                        }
                    arxiv_scope_block = self._arxiv_broad_search_block_reason(agent, command)
                    if arxiv_scope_block:
                        self._emit(agent.id, "arxiv_broad_search_block", {
                            "domain": _web_command_domain(command),
                            "reason": arxiv_scope_block,
                        })
                        return {
                            "error": arxiv_scope_block,
                            "required_behavior": (
                                "use arXiv advanced author-plus-title fields or named candidate metadata"
                            ),
                        }
                    failure_fuse_block = self._web_failure_fuse_block_reason(agent, command)
                    if failure_fuse_block:
                        self._emit(agent.id, "web_failure_fuse_block", {
                            "signature": _web_command_signature(command),
                            "domain": _web_command_domain(command),
                            "reason": failure_fuse_block,
                        })
                        return {
                            "error": failure_fuse_block,
                            "required_behavior": (
                                "use another source or perform a local deterministic reproduction, then deliver"
                            ),
                        }
                    no_gain_block = self._direct_web_no_gain_block_reason(agent, command)
                    if no_gain_block:
                        self._emit(agent.id, "direct_web_no_gain_block", {
                            "signature": _web_command_signature(command),
                            "domain": _web_command_domain(command),
                            "reason": no_gain_block,
                        })
                        return {
                            "error": no_gain_block,
                            "required_behavior": (
                                "change the candidate, source, or retrieval method; do not retry this URL"
                            ),
                        }
                    failover_block = self._web_search_failover_block_reason(agent, command)
                    if failover_block:
                        self._emit(agent.id, "web_search_failover_block", {
                            "domain": _web_command_domain(command),
                            "reason": failover_block,
                        })
                        return {
                            "error": failover_block,
                            "required_behavior": (
                                "change query or method; use the structured fallback or a direct public source"
                            ),
                        }
                root_lake_build = bool(
                    self._fixed_plan
                    and agent.parent is None
                    and tc.name in {"shell", "tb_shell"}
                    and self._fixed_is_full_lake_build(command)
                )
                if root_lake_build and self._fixed_source_writes_in_flight:
                    return {
                        "error": (
                            f"{self._fixed_source_writes_in_flight} source write(s) are still in flight; "
                            "wait for writers before starting the root lake build"
                        ),
                        "required_behavior": "query/wait for writers, then retry lake build",
                    }
                source_write_paths = self._fixed_source_paths_for_tool(tc)
                write_seq_before = self._fixed_source_write_seq
                if root_lake_build:
                    self._fixed_source_writes_frozen = True
                if source_write_paths:
                    self._fixed_source_writes_in_flight += 1
                self._refresh_tool_context_policy(agent)
                self._note_state_being_submitted(agent, tc.name, tc.arguments)
                try:
                    result = await handler(tc.arguments, agent.workspace, self._tool_context)
                except Exception:
                    if root_lake_build:
                        self._fixed_source_writes_frozen = False
                    raise
                finally:
                    if source_write_paths:
                        self._fixed_source_writes_in_flight = max(
                            0, self._fixed_source_writes_in_flight - 1
                        )
                if source_write_paths and not (
                    isinstance(result, dict) and "error" in result
                ):
                    self._fixed_source_write_seq += 1
                    self._emit(agent.id, "fixed_source_dirty", {
                        "role": self._fixed_agent_keys.get(agent.id, "root"),
                        "tool": tc.name,
                        "paths": source_write_paths,
                        "source_write_seq": self._fixed_source_write_seq,
                        "last_green_write_seq": self._fixed_last_green_write_seq,
                    })
                if tc.name == "shell":
                    candidate_like_before = agent._shell_activity.candidate_like_outputs
                    material_gain_before = agent._shell_activity.material_gain_calls
                    self._update_shell_activity(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    pdf_cli_guidance = self._pdf_cli_fallback_guidance(command, result)
                    if pdf_cli_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_pdf_cli_fallback_guidance": pdf_cli_guidance,
                        }
                        self._emit(agent.id, "pdf_cli_fallback_guidance", {
                            "guidance": pdf_cli_guidance,
                        })
                    bibliography_filter_guidance = self._bibliography_line_filter_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                    )
                    if bibliography_filter_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_bibliography_filter_guidance": bibliography_filter_guidance,
                        }
                        self._emit(agent.id, "bibliography_line_filter_guidance", {
                            "guidance": bibliography_filter_guidance[:500],
                        })
                    arxiv_scope_guidance = self._arxiv_broad_search_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                    )
                    if arxiv_scope_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_arxiv_search_guidance": arxiv_scope_guidance,
                        }
                        self._emit(agent.id, "arxiv_broad_search_guidance", {
                            "domain": _web_command_domain(command),
                            "guidance": arxiv_scope_guidance[:500],
                        })
                    version_history_guidance = self._arxiv_version_history_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    if version_history_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_arxiv_version_history_guidance": version_history_guidance,
                        }
                        self._emit(agent.id, "arxiv_version_history_guidance", {
                            "guidance": version_history_guidance[:500],
                        })
                    saved_advanced_html_guidance = self._arxiv_saved_advanced_html_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    if saved_advanced_html_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_arxiv_saved_html_guidance": saved_advanced_html_guidance,
                        }
                        self._emit(agent.id, "arxiv_saved_advanced_html_guidance", {
                            "guidance": saved_advanced_html_guidance[:500],
                        })
                    advanced_batch_timeout_guidance = self._arxiv_advanced_batch_timeout_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    if advanced_batch_timeout_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_arxiv_batch_timeout_guidance": advanced_batch_timeout_guidance,
                        }
                        self._emit(agent.id, "arxiv_advanced_batch_timeout_guidance", {
                            "guidance": advanced_batch_timeout_guidance[:500],
                        })
                    failure_fuse_guidance = self._web_failure_fuse_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    if failure_fuse_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_web_failure_guidance": failure_fuse_guidance,
                        }
                        self._emit(agent.id, "web_failure_fuse_guidance", {
                            "signature": _web_command_signature(command),
                            "domain": _web_command_domain(command),
                            "guidance": failure_fuse_guidance[:500],
                        })
                    no_gain_guidance = (
                        ""
                        if saved_advanced_html_guidance
                        else self._direct_web_no_gain_guidance(
                            agent,
                            str(tc.arguments.get("command", "")),
                        )
                    )
                    if no_gain_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_direct_web_no_gain_guidance": no_gain_guidance,
                        }
                        self._emit(agent.id, "direct_web_no_gain_guidance", {
                            "signature": _web_command_signature(command),
                            "domain": _web_command_domain(command),
                            "guidance": no_gain_guidance[:500],
                        })
                    failover_guidance = self._web_search_failover_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    if failover_guidance and isinstance(result, dict):
                        fallback_query = _web_command_query(command)
                        fallback_search: dict[str, Any] = {}
                        fallback_error = ""
                        if fallback_query:
                            try:
                                fallback_search = await asyncio.to_thread(
                                    _structured_web_search_sync,
                                    fallback_query,
                                )
                            except Exception as exc:
                                fallback_error = f"{type(exc).__name__}: {exc}"
                        if fallback_search.get("results"):
                            failover_guidance = (
                                "Runtime automatically ran the same query through a structured public search. Review "
                                "runtime_fallback_search below, follow promising direct links, and do not "
                                "repeat the empty HTML search."
                            )
                        result = {
                            **result,
                            "runtime_guidance": failover_guidance,
                            **(
                                {"runtime_fallback_search": fallback_search}
                                if fallback_search.get("results") else {}
                            ),
                            **(
                                {"runtime_fallback_error": fallback_error}
                                if fallback_error else {}
                            ),
                        }
                        self._emit(agent.id, "web_search_failover_guidance", {
                            "domain": _web_command_domain(command),
                            "guidance": failover_guidance[:500],
                            "fallback_provider": fallback_search.get("provider"),
                            "fallback_results": len(fallback_search.get("results") or []),
                            "fallback_error": fallback_error[:500],
                        })
                    html_guidance = self._large_html_extraction_guidance(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    if html_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_html_extraction_guidance": html_guidance,
                        }
                        self._emit(agent.id, "large_html_extraction_guidance", {
                            "domain": _web_command_domain(command),
                            "guidance": html_guidance,
                        })
                    figure_guidance = self._figure_panel_disambiguation_guidance(
                        agent,
                        command,
                        result,
                    )
                    if figure_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_figure_panel_guidance": figure_guidance,
                        }
                        self._emit(agent.id, "figure_panel_disambiguation_guidance", {
                            "guidance": figure_guidance,
                        })
                    html_parser_guidance = self._html_regex_parser_guidance(
                        agent,
                        command,
                        result,
                    )
                    if html_parser_guidance and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_html_parser_guidance": html_parser_guidance,
                        }
                        self._emit(agent.id, "html_regex_parser_guidance", {
                            "guidance": html_parser_guidance,
                        })
                    evidence_checkpoint = self._child_evidence_checkpoint_guidance(
                        agent,
                        candidate_like_before=candidate_like_before,
                        material_gain_before=material_gain_before,
                        command=command,
                        result=result,
                    )
                    if evidence_checkpoint and isinstance(result, dict):
                        result = {
                            **result,
                            "runtime_evidence_checkpoint": evidence_checkpoint,
                        }
                        self._emit(agent.id, "child_evidence_checkpoint", {
                            "candidate_like_outputs": agent._shell_activity.candidate_like_outputs,
                            "material_gain_calls": agent._shell_activity.material_gain_calls,
                            "guidance": evidence_checkpoint,
                        })
                    self._refresh_tool_context_policy(agent)
                    self._fixed_observe_shell_result(
                        agent,
                        command,
                        result,
                        write_seq_before=write_seq_before,
                    )
                if root_lake_build:
                    self._fixed_source_writes_frozen = False
                if self._malformed_write_recovery_completed(agent, tc, result):
                    self._emit(agent.id, "malformed_write_recovered", {
                        "tool": tc.name,
                        "turn": agent._turns,
                    })
                self._track_delivery_activity(agent, tc.name, tc.arguments, result)
                await self._calibrate_from_tool_result(agent, tc.name, tc.arguments, result)
                return result
        except Exception as e:
            return {"error": str(e)}

    def _candidate_failure_reason(self, agent: Agent, tc: ToolCall, result: Any) -> str | None:
        if not (
            self.config.auto_rollback_enabled
            and self.config.auto_rollback_on_candidate_failure
        ):
            return None
        if tc.name not in {"shell", "tb_shell", "submit"}:
            return None
        text = ""
        exit_code: int | None = None
        if isinstance(result, dict):
            if result.get("blocked"):
                return None
            if "exit_code" in result:
                try:
                    exit_code = int(result.get("exit_code"))
                except (TypeError, ValueError):
                    exit_code = None
            text = "\n".join(
                str(result.get(key) or "")
                for key in ("stdout", "stderr", "error")
                if result.get(key)
            )
        else:
            text = str(result or "")
        command = str(tc.arguments.get("command", "")) if isinstance(tc.arguments, dict) else ""
        lower_text = text.lower()
        lower_command = command.lower()
        command_words = set(re.findall(r"[a-zA-Z0-9_+.-]+", lower_command))
        explicit_check_command = bool(
            command_words.intersection(
                {
                    "test",
                    "pytest",
                    "unittest",
                    "check",
                    "verify",
                    "validate",
                    "compile",
                    "make",
                    "gcc",
                    "g++",
                    "clang",
                }
            )
            or "cargo test" in lower_command
            or "go test" in lower_command
            or "npm test" in lower_command
        )

        candidate_context = bool(
            agent._delivery_activity.write_calls
            or agent._delivery_activity.candidate_ready_turn > 0
            or agent._delivery_activity.candidate_output_files
            or agent._delivery_activity.container_candidate_output_files
            or explicit_check_command
        )
        if not candidate_context:
            return None
        strong_failure = any(
            marker in lower_text
            for marker in (
                "assertionerror",
                "traceback (most recent call last)",
                "compilation failed",
                "compile error",
                "syntaxerror",
                "test failed",
                "tests failed",
                "failed tests",
                "pytest failed",
                "verification failed",
                "validation failed",
                "candidate failed",
                "not found in python-chess",
                "image similarity check failed",
                "mismatch",
            )
        )
        explicit_pass = any(
            marker in lower_text
            for marker in (
                "all tests passed",
                "success",
            )
        )
        if exit_code not in (None, 0) and (strong_failure or explicit_check_command):
            return (
                f"tool={tc.name} exit_code={exit_code}; command={command[:220]}; "
                f"failure={text[:600]}"
            )
        if strong_failure and not (exit_code == 0 and explicit_pass):
            return f"tool={tc.name}; command={command[:220]}; failure={text[:600]}"
        return None

    def _tool_call_metadata(self, tc: ToolCall) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if tc.name != "shell":
            return metadata
        try:
            requested_timeout = int(tc.arguments.get("timeout", 30))
        except (TypeError, ValueError):
            requested_timeout = 30
        requested_timeout = max(1, requested_timeout)
        max_timeout = int(getattr(self._tool_context, "shell_max_timeout", 30) or 0)
        if max_timeout > 0 and requested_timeout > max_timeout:
            metadata.update({
                "timeout_capped": True,
                "requested_timeout": requested_timeout,
                "effective_timeout": max_timeout,
            })
        return metadata

    def _final_evidence_block_reason(self, agent: Agent) -> str | None:
        if not self.config.strategy_final_evidence_gate:
            return None
        if agent.parent is not None or agent.status != "running":
            return None
        if agent._turns < self.config.strategy_final_evidence_min_root_turns:
            return None
        shared_candidates = len(self._shared_candidate_files())
        if shared_candidates < self.config.strategy_final_evidence_min_shared_candidates:
            return None

        candidate_agent_ids = self._candidate_agents_for_root(agent)
        candidate_agents = len(candidate_agent_ids)
        if candidate_agents >= self.config.strategy_final_evidence_min_candidate_agents:
            ratio = max(0.0, min(1.0, self.config.strategy_final_evidence_query_coverage_ratio))
            required = max(1, math.ceil(candidate_agents * ratio))
            covered = self._root_candidate_query_coverage(agent, candidate_agent_ids)
            if covered < required:
                return (
                    f"final evidence gate: query candidate-producing child agents first "
                    f"({covered}/{required} candidate queries covered)"
                )

        verification_count = len(self._shared_verification_files())
        if self.config.strategy_final_evidence_require_verification and verification_count <= 0:
            if self.config.strategy_final_evidence_allow_spawn_verifier:
                return "final evidence gate: spawn or query a verifier/selector branch before finalization"
            return "final evidence gate: collect coordination evidence before finalization"

        if (
            self.config.strategy_final_evidence_require_verification_review
            and verification_count > 0
            and self._agent_verification_review_count(agent) <= 0
        ):
            return "final evidence gate: read/query the shared verification or selection evidence before finalization"
        return None

    async def _compress(self, history: list[Message], keep_recent: int | None = None) -> list[Message]:
        """Compress old messages into a summary, keeping recent ones intact."""
        keep_recent = keep_recent or self.config.compress_keep_recent
        if len(history) <= keep_recent + 2:
            return history
        system = history[0]
        recent_start = self._safe_recent_start(history, keep_recent)
        old = history[1:recent_start]
        recent = history[recent_start:]
        max_msgs = self.config.compress_max_messages
        max_chars = self.config.compress_max_chars
        parts = []
        for msg in (old[-max_msgs:] if max_msgs > 0 else old):
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if max_chars > 0:
                content = content[:max_chars]
            if content:
                parts.append(f"[{role}]: {content}")
        summary = {"role": "assistant", "content": f"[Compressed {len(old)} messages]\n" + "\n".join(parts)}
        return [system, summary] + recent

    def _build_system_prompt(self, agent_id: str, task: str, workspace: Path, parent_context: dict | None = None) -> str:
        shared = self._tool_context.shared_dir
        time_info = ""
        if self.config.time_limit > 0:
            time_info = f"\nTime limit: {self.config.time_limit:.0f}s total. Check with get_cost()."

        web_search_line = ""
        if self.config.web_search_failover_after_low_signal:
            web_search_line = (
                "- For public web search, use ordinary public search or direct sources. Runtime automatically "
                "attaches a structured Yahoo/Bing fallback after an empty HTML search; follow those direct links "
                "instead of repeating the failed query. A direct URL that returns an explicit 403/429 or "
                "human-verification challenge is fused; switch source or use local deterministic reproduction."
            )

        # Context about spawner (for sub-agents)
        context_section = ""
        if parent_context:
            capsule = str(parent_context.get("task_capsule") or "").strip()
            capsule_section = f"\n## Task Capsule\n{capsule}\n" if capsule else ""
            context_section = f"""
## Your Context
- Spawned by: agent "{parent_context['parent_id']}" (task: {parent_context['parent_task']})
- Peers working in parallel: {parent_context['siblings'] or 'none yet'}
- Depth: {parent_context['depth']}
- On your first response, perform the assigned work with one concrete tool call. Do not spend a full turn on prose-only planning; keep any text before the tool call under 40 words.
- When your evidence supports an answer, call deliver_to_parent(answer=..., evidence=..., confidence=..., method=...) exactly once, then call set_status("done", result=<same answer>).
- Do not use query or send for final result delivery. If evidence is incomplete, deliver your best concise candidate with lower confidence instead of returning an empty result.
- The root owns final files in the shared workspace. Do not create or overwrite a shared final artifact named by the parent task; keep research notes private and use deliver_to_parent for your handoff.
{web_search_line}
{capsule_section}
"""

        coordination_tools = {"spawn", "send", "deliver_to_parent", "wait", "query", "kill", "transfer", "set_bio", "get_task_context"}
        has_coordination_tools = bool(coordination_tools - self.config.disabled_tools)
        if has_coordination_tools:
            meta_line = "- meta tools: coordination (spawn, send, deliver_to_parent, wait, query, kill, transfer, etc.)"
        else:
            meta_line = "- meta tools: single-agent lifecycle and deliverables (get_cost, set_status, rebirth, submit, batch)"

        extra_section = ""
        extra = self.config.system_extra_instructions.strip()
        if extra:
            extra_section = f"""
## Runtime Environment
{extra}
"""

        delivery_section = ""
        contract = self.config.delivery_contract
        if contract is not None and not parent_context:
            required = ", ".join(str(path) for path in contract.required_targets())
            delivery_section = f"""
## Runtime Delivery Contract
- Required final outputs: {required}
- `submit` accepts a file or a complete directory tree. The runtime preserves the
  surrounding tree and atomically publishes it to the required destination.
- `set_status(done)` is refused while a required output is missing or empty. A
  failed submit is not completion; fix the path or submit the complete tree.
"""

        return f"""You are agent "{agent_id}" in a multi-agent system.

Task: {task}
Workspace: {workspace} (private to you)
Shared: {shared} (visible to all agents){time_info}
{context_section}
{extra_section}
{delivery_section}
## Tool Philosophy
Available tools can change from turn to turn. Only call tools that are present in the
current tool schema. If a tool is absent, switch to one of the available tools instead
of requesting the absent tool again.
When you need a tool, emit an actual tool call. If your model cannot emit native
tool_calls, write exactly one executable tool request as a fenced bash block for shell,
or as JSON like {{"tool": "shell", "args": {{"command": "..."}}}}. Do not merely say
"I will search", "let me inspect", or invent tool names such as ws_file_read.
If a shell result says a capability such as web/network access is blocked, do not retry
that capability. Use local shell/file operations and the evidence already collected to
finish the required deliverable and call set_status(done, result=...).

When available, tools are organized in 3 layers:
- shell: universal primitive for system commands such as mkdir, mv, ls, find, git,
  pip, curl, and Python one-liners.
- ws_* tools: structured operations (file create/read/edit, grep, code outline)
{meta_line}
"""

    def _emit(self, agent_id: str, event_type: str, data: dict | None = None):
        now = time.time()
        event = {"agent": agent_id, "event": event_type, "data": data or {}, "t": now}
        self._events.append(event)
        self.on_event(event)
        # Write to events.jsonl for the viewer (thread-safe)
        if self.config.log_dir:
            trace_event = {
                "type": event_type, "agent": agent_id,
                "ts": now, "rel_ts": round(now - self._start_time, 3),
                "data": data or {},
            }
            try:
                events_file = self.config.log_dir / "events.jsonl"
                line = json.dumps(trace_event, ensure_ascii=False, default=str, allow_nan=False) + "\n"
                with self._emit_lock:
                    with open(events_file, "a") as f:
                        f.write(line)
            except Exception:
                pass

    # ─── Public API ──────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "agents": {
                aid: {"status": a.status, "task": a.task[:80], "bio": a.bio, "turns": a._turns}
                for aid, a in self.agents.items()
            },
            "cost": self.ledger.summary(),
            "scheduler": self.scheduler.stats,
        }

    def stats(self) -> dict[str, Any]:
        """Comprehensive post-run statistics."""
        elapsed = self.effective_elapsed()
        agents = list(self.agents.values())
        n = len(agents)
        if n == 0:
            return {"error": "no agents"}

        # ─── Tokens ──────────────────────────────────────────────────────
        total_tokens = sum(a.tokens_consumed for a in agents)
        token_per_agent = [a.tokens_consumed for a in agents]

        # ─── Agent lifecycle ─────────────────────────────────────────────
        statuses = {}
        for a in agents:
            statuses[a.status] = statuses.get(a.status, 0) + 1
        depths = [a.depth for a in agents]
        max_depth = max(depths) if depths else 0
        turns_list = [a._turns for a in agents]

        # Peak concurrency (estimate from events)
        running_at = {}  # agent_id -> (start_time, end_time)
        for e in self._events:
            aid = e["agent"]
            t = e["t"]
            if e["event"] == "agent_new":
                running_at[aid] = [t, None]
            elif e["event"] in {"done", "failed"}:
                if aid in running_at:
                    running_at[aid][1] = t
        # Fill in end times for agents still running
        for aid in running_at:
            if running_at[aid][1] is None:
                running_at[aid][1] = time.time()
        # Compute peak
        peak_concurrent = 0
        if running_at:
            all_times = []
            for aid, (s, e) in running_at.items():
                all_times.append((s, 1))
                all_times.append((e, -1))
            all_times.sort()
            current = 0
            for _, delta in all_times:
                current += delta
                peak_concurrent = max(peak_concurrent, current)

        # ─── Tool usage ──────────────────────────────────────────────────
        tool_counts: dict[str, int] = {}
        for e in self._events:
            if e["event"] == "tool_call":
                name = e["data"].get("tool", "?")
                tool_counts[name] = tool_counts.get(name, 0) + 1
        total_tool_calls = sum(tool_counts.values())
        tool_sorted = sorted(tool_counts.items(), key=lambda x: -x[1])

        # ─── Communication ───────────────────────────────────────────────
        total_messages = len(self._messages_sent)
        msg_tokens_total = sum(t for _, _, t in self._messages_sent)

        # Communication graph: edges between agents
        comm_edges: dict[tuple[str, str], int] = {}
        agents_who_sent: set[str] = set()
        agents_who_received: set[str] = set()
        for frm, to, tok in self._messages_sent:
            edge = (frm, to)
            comm_edges[edge] = comm_edges.get(edge, 0) + 1
            agents_who_sent.add(frm)
            agents_who_received.add(to)

        # Degree: out-degree = unique recipients per sender
        out_degree: dict[str, set] = {}
        in_degree: dict[str, set] = {}
        for frm, to, _ in self._messages_sent:
            out_degree.setdefault(frm, set()).add(to)
            in_degree.setdefault(to, set()).add(frm)
        max_out = max((len(v) for v in out_degree.values()), default=0)
        max_in = max((len(v) for v in in_degree.values()), default=0)
        busiest_sender = max(out_degree.items(), key=lambda x: len(x[1]), default=("none", set()))
        busiest_receiver = max(in_degree.items(), key=lambda x: len(x[1]), default=("none", set()))

        # ─── Budget ──────────────────────────────────────────────────────
        cost = self.ledger.total_spent
        tokens_per_dollar = total_tokens / cost if cost > 0 else 0
        fixed_observed_topology = (
            self._fixed_observed_topology()
            if self._fixed_plan
            else {"nodes": [], "edges": [], "autonomous_spawns": [], "checkpoint_spawns": []}
        )

        # ─── Compile result ──────────────────────────────────────────────
        return {
            "overview": {
                "elapsed_seconds": round(elapsed, 1),
                "wall_elapsed_seconds": round(time.time() - self._start_time, 1),
                "excluded_time_seconds": round(self._excluded_time_seconds, 1),
                "excluded_time_by_reason": {
                    reason: round(value, 1)
                    for reason, value in sorted(self._excluded_time_by_reason.items())
                },
                "total_cost_usd": round(cost, 4),
                "total_tokens": total_tokens,
                "total_tokens_with_supervisor": total_tokens + self._supervisor_tokens,
                "total_cost_usd_with_supervisor": round(cost + self._supervisor_cost, 4),
                "tokens_per_dollar": int(tokens_per_dollar),
            },
            "supervisor": {
                "enabled": self.config.supervisor_enabled,
                "model": self.config.supervisor_model if self.config.supervisor_enabled else None,
                "calls": self._supervisor_calls,
                "errors": self._supervisor_errors,
                "tokens": self._supervisor_tokens,
                "cost_usd": round(self._supervisor_cost, 6),
                "decisions": dict(self._supervisor_decisions),
                "trigger_mode": self.config.supervisor_trigger_mode,
                "skips": dict(self._supervisor_skips),
            },
            "fixed_orchestration": {
                "enabled": self._fixed_plan is not None,
                "profile": self._fixed_plan.name if self._fixed_plan else None,
                "spawn_execution": (
                    "parent_agent_via_nanoma_tool" if self._fixed_plan else None
                ),
                "spawn_policy": (
                    "checkpoint_supplements_autonomous" if self._fixed_plan else None
                ),
                "autonomous_spawn_enabled": self._fixed_plan is not None,
                "observed_topology": fixed_observed_topology,
                "fired_checkpoints": sorted(self._fixed_fired_checkpoints),
                "requested_checkpoints": sorted(self._fixed_requested_checkpoints),
                "pending_checkpoint_by_parent": dict(
                    sorted(self._fixed_pending_checkpoint_by_parent.items())
                ),
                "agent_roles": {
                    key: agent_id for key, agent_id in sorted(self._fixed_agent_ids.items())
                },
                "checkpoint_agents": {
                    key: sorted(agent_ids)
                    for key, agent_ids in sorted(self._fixed_checkpoint_agents.items())
                },
                "completion_checkpoint": (
                    self._fixed_plan.completion_checkpoint if self._fixed_plan else None
                ),
                "completion_ready": bool(
                    self._fixed_plan
                    and self._fixed_checkpoint_complete(
                        self._fixed_plan.completion_checkpoint
                    )
                ),
                "ownership": {
                    self._fixed_agent_keys.get(agent_id, agent_id): {
                        "read_only": spec.read_only,
                        "allowed_paths": list(spec.allowed_paths),
                    }
                    for agent_id, spec in self._fixed_agent_specs.items()
                },
                "root_source_writes": (
                    self._fixed_plan.root_source_writes if self._fixed_plan else None
                ),
                "direct_build_gate": (
                    self._fixed_plan.direct_build_gate if self._fixed_plan else False
                ),
                "source_isolation": (
                    "shared-tree-serial"
                    if self._fixed_plan and self._fixed_plan.direct_build_gate
                    else "shared-tree"
                ),
                "verification_gate": (
                    {
                        "argv": list(self._fixed_plan.verification_gate.argv),
                        "cwd": self._fixed_plan.verification_gate.cwd,
                        "timeout_seconds": self._fixed_plan.verification_gate.timeout_seconds,
                        "acceptance": self._fixed_plan.verification_gate.acceptance,
                        "score_direction": self._fixed_plan.verification_gate.score_direction,
                        "always_run": self._fixed_plan.verification_gate.always_run,
                    }
                    if self._fixed_plan and self._fixed_plan.verification_gate
                    else None
                ),
                "best_gate_metric": self._fixed_best_gate_metric,
                "snapshot_enabled": (
                    self._fixed_plan.snapshot_enabled if self._fixed_plan else False
                ),
                "snapshot_paths": (
                    list(self._fixed_plan.snapshot_paths) if self._fixed_plan else []
                ),
                "has_verified_green": self._fixed_has_verified_green,
                "candidate_validations": {
                    self._fixed_agent_keys.get(agent_id, agent_id): {
                        key: value
                        for key, value in validation.items()
                        if key not in {"stdout_tail", "stderr_tail"}
                    }
                    for agent_id, validation in self._fixed_candidate_validations.items()
                },
                "green_checkpoint_seq": self._fixed_green_seq,
                "last_green_path": (
                    str(self._fixed_last_green_path) if self._fixed_last_green_path else None
                ),
                "green_restores": self._fixed_green_restores,
                "source_write_seq": self._fixed_source_write_seq,
                "last_green_write_seq": self._fixed_last_green_write_seq,
                "source_dirty": (
                    self._fixed_source_write_seq != self._fixed_last_green_write_seq
                ),
                "source_writes_in_flight": self._fixed_source_writes_in_flight,
                "source_writes_frozen": self._fixed_source_writes_frozen,
            },
            "rollback": {
                "enabled": self.config.auto_rollback_enabled,
                "attempts": self._rollback_attempts,
                "history": copy.deepcopy(self._rollback_history),
                "last_checkpoint": str(self._last_checkpoint_path) if self._last_checkpoint_path else None,
            },
            "delivery_contract": {
                "enabled": self.config.delivery_contract is not None,
                "ready": bool(
                    self._delivery_contract_history
                    and self._delivery_contract_history[-1].get("ready")
                ),
                "attempts": len(self._delivery_contract_history),
                "published_trees": sum(
                    len(record.get("published") or [])
                    for record in self._delivery_contract_history
                ),
                "last": (
                    copy.deepcopy(self._delivery_contract_history[-1])
                    if self._delivery_contract_history else None
                ),
            },
            "agents": {
                "total_spawned": n,
                "peak_concurrent": peak_concurrent,
                "max_depth": max_depth,
                "status_breakdown": statuses,
                "avg_turns": round(sum(turns_list) / n, 1),
                "max_turns": max(turns_list),
                "min_turns": min(turns_list),
                "avg_tokens": int(total_tokens / n),
                "max_tokens_agent": max(agents, key=lambda a: a.tokens_consumed).id,
                "max_tokens_value": max(token_per_agent),
            },
            "tools": {
                "total_calls": total_tool_calls,
                "unique_tools_used": len(tool_counts),
                "top_tools": tool_sorted[:10],
                "spawn_count": tool_counts.get("spawn", 0),
                "send_count": tool_counts.get("send", 0),
                "deliver_to_parent_count": tool_counts.get("deliver_to_parent", 0),
                "query_count": tool_counts.get("query", 0),
                "wait_count": tool_counts.get("wait", 0),
            },
            "candidate_delivery": {
                "records": len(self._candidate_deliveries),
                "usable_records": sum(
                    1 for record in self._candidate_deliveries
                    if self._candidate_record_usable_for_convergence(record)
                ),
                "convergence_roots": len(self._candidate_convergence_turn_by_root),
                "sources": dict(sorted(Counter(
                    str(record.get("source") or "unknown")
                    for record in self._candidate_deliveries
                ).items())),
            },
            "communication": {
                "total_messages": total_messages,
                "total_message_tokens": msg_tokens_total,
                "unique_edges": len(comm_edges),
                "agents_who_sent": len(agents_who_sent),
                "agents_who_received": len(agents_who_received),
                "max_out_degree": max_out,
                "max_in_degree": max_in,
                "busiest_sender": f"{busiest_sender[0]} → {len(busiest_sender[1])} recipients",
                "busiest_receiver": f"{busiest_receiver[0]} ← {len(busiest_receiver[1])} senders",
                "top_edges": sorted(comm_edges.items(), key=lambda x: -x[1])[:5],
            },
            "per_agent": [
                {
                    "id": a.id, "status": a.status, "depth": a.depth,
                    "turns": a._turns, "tokens": a.tokens_consumed,
                    "children": len(a.children), "bio": a.bio[:50],
                    "last_tool_policy": copy.deepcopy(a._last_tool_policy),
                    "shell_activity": a._shell_activity.as_event(),
                    "delivery_activity": a._delivery_activity.as_event(),
                }
                for a in sorted(agents, key=lambda a: -a.tokens_consumed)
            ],
        }

    async def shutdown(self):
        """Force-terminate all running agents."""
        for a in self.agents.values():
            if a.status in ("running", "idle"):
                a.status = "done"
            if a._task and not a._task.done():
                a._task.cancel()
        tasks = [a._task for a in self.agents.values() if a._task and not a._task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _queue_snapshot(self, queue: asyncio.Queue[Envelope]) -> list[Envelope]:
        return list(getattr(queue, "_queue", []))

    def _make_queue(self, items: list[Envelope]) -> asyncio.Queue[Envelope]:
        queue: asyncio.Queue[Envelope] = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)
        return queue

    def _agent_snapshot(self, agent: Agent) -> dict[str, Any]:
        return {
            "id": agent.id,
            "task": agent.task,
            "model": agent.model,
            "bio": agent.bio,
            "status": agent.status,
            "history": copy.deepcopy(agent.history),
            "children": sorted(agent.children),
            "parent": agent.parent,
            "depth": agent.depth,
            "result": agent.result,
            "quota": copy.deepcopy(agent.quota),
            "context_tokens": agent.context_tokens,
            "context_limit": agent.context_limit,
            "tokens_consumed": agent.tokens_consumed,
            "created_at": agent._created_at,
            "workspace": str(agent.workspace),
            "artifacts": copy.deepcopy(agent.artifacts),
            "queue_inbox": self._queue_snapshot(agent._queue_inbox),
            "steer_inbox": self._queue_snapshot(agent._steer_inbox),
            "immediate_inbox": self._queue_snapshot(agent._immediate_inbox),
            "turns": agent._turns,
            "last_active": agent._last_active,
            "rebirth_pending": copy.deepcopy(agent._rebirth_pending),
            "notified_thresholds": sorted(agent._notified_thresholds),
            "tool_calls": agent._tool_calls,
            "no_tool_turns": agent._no_tool_turns,
            "malformed_tool_turns": agent._malformed_tool_turns,
            "http400_recoveries": agent._http400_recoveries,
            "last_tool_policy": copy.deepcopy(agent._last_tool_policy),
            "shell_activity": {
                "web_calls": agent._shell_activity.web_calls,
                "web_low_signal_calls": agent._shell_activity.web_low_signal_calls,
                "blocked_web_calls": agent._shell_activity.blocked_web_calls,
                "large_output_calls": agent._shell_activity.large_output_calls,
                "truncated_output_calls": agent._shell_activity.truncated_output_calls,
                "consecutive_large_outputs": agent._shell_activity.consecutive_large_outputs,
                "output_bytes": agent._shell_activity.output_bytes,
                "consecutive_web_low_signal": agent._shell_activity.consecutive_web_low_signal,
                "repeated_web_queries": agent._shell_activity.repeated_web_queries,
                "repeated_web_domains": agent._shell_activity.repeated_web_domains,
                "web_no_gain_calls": agent._shell_activity.web_no_gain_calls,
                "consecutive_web_no_gain": agent._shell_activity.consecutive_web_no_gain,
                "repeated_web_results": agent._shell_activity.repeated_web_results,
                "search_result_calls": agent._shell_activity.search_result_calls,
                "task_keyword_hits": agent._shell_activity.task_keyword_hits,
                "material_gain_calls": agent._shell_activity.material_gain_calls,
                "candidate_like_outputs": agent._shell_activity.candidate_like_outputs,
                "candidate_like_output_bytes": agent._shell_activity.candidate_like_output_bytes,
                "candidate_prepare_open_turn": agent._shell_activity.candidate_prepare_open_turn,
                "candidate_prepare_open_non_delivery_tool_calls": agent._shell_activity.candidate_prepare_open_non_delivery_tool_calls,
                "candidate_prepare_finalize_turn": agent._shell_activity.candidate_prepare_finalize_turn,
                "candidate_prepare_finalize_non_delivery_tool_calls": agent._shell_activity.candidate_prepare_finalize_non_delivery_tool_calls,
                "unavailable_tool_calls": agent._shell_activity.unavailable_tool_calls,
                "web_recovery_calls_remaining": agent._shell_activity.web_recovery_calls_remaining,
                "web_recovery_grants": agent._shell_activity.web_recovery_grants,
                "web_recovery_last_write_calls": agent._shell_activity.web_recovery_last_write_calls,
                "unique_web_signatures": sorted(agent._shell_activity.unique_web_signatures),
                "unique_web_domains": sorted(agent._shell_activity.unique_web_domains),
                "unique_web_result_digests": sorted(agent._shell_activity.unique_web_result_digests),
                "hard_blocked_web_signatures": sorted(agent._shell_activity.hard_blocked_web_signatures),
                "last_web_signature": agent._shell_activity.last_web_signature,
                "last_web_domain": agent._shell_activity.last_web_domain,
                "last_web_result_digest": agent._shell_activity.last_web_result_digest,
                "web_domain_sprawl": agent._shell_activity.web_domain_sprawl,
                "web_saturation": agent._shell_activity.web_saturation,
                "web_loop_pressure": agent._shell_activity.web_loop_pressure,
                "reconcile_after_web_loop": agent._shell_activity.reconcile_after_web_loop,
                "finalize_after_web_saturation": agent._shell_activity.finalize_after_web_saturation,
            },
            "delivery_activity": {
                "write_calls": agent._delivery_activity.write_calls,
                "submit_calls": agent._delivery_activity.submit_calls,
                "non_delivery_tool_calls": agent._delivery_activity.non_delivery_tool_calls,
                "deliver_enter_turn": agent._delivery_activity.deliver_enter_turn,
                "deliver_enter_non_delivery_tool_calls": agent._delivery_activity.deliver_enter_non_delivery_tool_calls,
                "candidate_ready_turn": agent._delivery_activity.candidate_ready_turn,
                "candidate_ready_non_delivery_tool_calls": agent._delivery_activity.candidate_ready_non_delivery_tool_calls,
                "candidate_review_calls": agent._delivery_activity.candidate_review_calls,
                "candidate_source_review_calls": agent._delivery_activity.candidate_source_review_calls,
                "candidate_output_files": sorted(agent._delivery_activity.candidate_output_files),
                "expected_output_files": sorted(agent._delivery_activity.expected_output_files),
                "container_candidate_output_files": sorted(agent._delivery_activity.container_candidate_output_files),
                "last_write_turn": agent._delivery_activity.last_write_turn,
                "last_submit_turn": agent._delivery_activity.last_submit_turn,
            },
        }

    def _restore_agent(
        self,
        data: dict[str, Any],
        old_workspace_root: Path | None = None,
    ) -> Agent:
        workspace = Path(data.get("workspace", self.config.workspace_root / data["id"]))
        if old_workspace_root is not None:
            try:
                workspace = self.config.workspace_root / workspace.relative_to(old_workspace_root)
            except ValueError:
                pass
        history = copy.deepcopy(data.get("history", []))
        if old_workspace_root is not None:
            old_shared = old_workspace_root / self.config.shared_dir
            new_shared = self.config.workspace_root / self.config.shared_dir
            for msg in history:
                content = msg.get("content")
                if isinstance(content, str):
                    content = content.replace(str(old_workspace_root), str(self.config.workspace_root))
                    content = content.replace(str(old_shared), str(new_shared))
                    msg["content"] = content
        agent = Agent(
            id=data["id"],
            task=data["task"],
            model=data["model"],
            bio=data.get("bio", ""),
            status=data.get("status", "running"),
            history=history,
            children=set(data.get("children", [])),
            parent=data.get("parent"),
            depth=int(data.get("depth", 0)),
            result=data.get("result"),
            quota=copy.deepcopy(data.get("quota", ResourceQuota())),
            context_tokens=int(data.get("context_tokens", 0)),
            context_limit=int(data.get("context_limit", 128000)),
            tokens_consumed=int(data.get("tokens_consumed", 0)),
            workspace=workspace,
            artifacts=copy.deepcopy(data.get("artifacts", [])),
        )
        agent._queue_inbox = self._make_queue(data.get("queue_inbox", []))
        agent._steer_inbox = self._make_queue(data.get("steer_inbox", []))
        agent._immediate_inbox = self._make_queue(data.get("immediate_inbox", []))
        agent._turns = int(data.get("turns", 0))
        agent._last_active = float(data.get("last_active", time.time()))
        agent._rebirth_pending = copy.deepcopy(data.get("rebirth_pending"))
        agent._notified_thresholds = set(data.get("notified_thresholds", []))
        agent._tool_calls = int(data.get("tool_calls", 0))
        agent._no_tool_turns = int(data.get("no_tool_turns", 0))
        agent._malformed_tool_turns = int(data.get("malformed_tool_turns", 0))
        agent._http400_recoveries = int(data.get("http400_recoveries", 0))
        agent._last_tool_policy = copy.deepcopy(data.get("last_tool_policy"))
        shell_activity = data.get("shell_activity") or {}
        agent._shell_activity = ShellActivityState(
            web_calls=int(shell_activity.get("web_calls", 0)),
            web_low_signal_calls=int(shell_activity.get("web_low_signal_calls", 0)),
            blocked_web_calls=int(shell_activity.get("blocked_web_calls", 0)),
            large_output_calls=int(shell_activity.get("large_output_calls", 0)),
            truncated_output_calls=int(shell_activity.get("truncated_output_calls", 0)),
            consecutive_large_outputs=int(shell_activity.get("consecutive_large_outputs", 0)),
            output_bytes=int(shell_activity.get("output_bytes", 0)),
            consecutive_web_low_signal=int(shell_activity.get("consecutive_web_low_signal", 0)),
            repeated_web_queries=int(shell_activity.get("repeated_web_queries", 0)),
            repeated_web_domains=int(shell_activity.get("repeated_web_domains", 0)),
            web_no_gain_calls=int(shell_activity.get("web_no_gain_calls", 0)),
            consecutive_web_no_gain=int(shell_activity.get("consecutive_web_no_gain", 0)),
            repeated_web_results=int(shell_activity.get("repeated_web_results", 0)),
            search_result_calls=int(shell_activity.get("search_result_calls", 0)),
            task_keyword_hits=int(shell_activity.get("task_keyword_hits", 0)),
            material_gain_calls=int(shell_activity.get("material_gain_calls", 0)),
            candidate_like_outputs=int(shell_activity.get("candidate_like_outputs", 0)),
            candidate_like_output_bytes=int(shell_activity.get("candidate_like_output_bytes", 0)),
            candidate_prepare_open_turn=int(shell_activity.get("candidate_prepare_open_turn", 0)),
            candidate_prepare_open_non_delivery_tool_calls=int(shell_activity.get("candidate_prepare_open_non_delivery_tool_calls", 0)),
            candidate_prepare_finalize_turn=int(shell_activity.get("candidate_prepare_finalize_turn", 0)),
            candidate_prepare_finalize_non_delivery_tool_calls=int(shell_activity.get("candidate_prepare_finalize_non_delivery_tool_calls", 0)),
            unavailable_tool_calls=int(shell_activity.get("unavailable_tool_calls", 0)),
            web_recovery_calls_remaining=int(shell_activity.get("web_recovery_calls_remaining", 0)),
            web_recovery_grants=int(shell_activity.get("web_recovery_grants", 0)),
            web_recovery_last_write_calls=int(shell_activity.get("web_recovery_last_write_calls", 0)),
            unique_web_signatures=set(shell_activity.get("unique_web_signatures", [])),
            unique_web_domains=set(shell_activity.get("unique_web_domains", [])),
            unique_web_result_digests=set(shell_activity.get("unique_web_result_digests", [])),
            hard_blocked_web_signatures=set(shell_activity.get("hard_blocked_web_signatures", [])),
            last_web_signature=str(shell_activity.get("last_web_signature", "")),
            last_web_domain=str(shell_activity.get("last_web_domain", "")),
            last_web_result_digest=str(shell_activity.get("last_web_result_digest", "")),
            web_domain_sprawl=float(shell_activity.get("web_domain_sprawl", 0.0)),
            web_saturation=float(shell_activity.get("web_saturation", 0.0)),
            web_loop_pressure=float(shell_activity.get("web_loop_pressure", 0.0)),
            reconcile_after_web_loop=bool(shell_activity.get("reconcile_after_web_loop", False)),
            finalize_after_web_saturation=bool(shell_activity.get("finalize_after_web_saturation", False)),
        )
        delivery_activity = data.get("delivery_activity") or {}
        agent._delivery_activity = DeliveryActivityState(
            write_calls=int(delivery_activity.get("write_calls", 0)),
            submit_calls=int(delivery_activity.get("submit_calls", 0)),
            non_delivery_tool_calls=int(delivery_activity.get("non_delivery_tool_calls", 0)),
            deliver_enter_turn=int(delivery_activity.get("deliver_enter_turn", 0)),
            deliver_enter_non_delivery_tool_calls=int(delivery_activity.get("deliver_enter_non_delivery_tool_calls", 0)),
            candidate_ready_turn=int(delivery_activity.get("candidate_ready_turn", 0)),
            candidate_ready_non_delivery_tool_calls=int(delivery_activity.get("candidate_ready_non_delivery_tool_calls", 0)),
            candidate_review_calls=int(delivery_activity.get("candidate_review_calls", 0)),
            candidate_source_review_calls=int(delivery_activity.get("candidate_source_review_calls", 0)),
            candidate_output_files=set(delivery_activity.get("candidate_output_files", [])),
            expected_output_files=set(delivery_activity.get("expected_output_files", [])),
            container_candidate_output_files=set(delivery_activity.get("container_candidate_output_files", [])),
            last_write_turn=int(delivery_activity.get("last_write_turn", 0)),
            last_submit_turn=int(delivery_activity.get("last_submit_turn", 0)),
        )
        return agent

    def _fixed_plan_fingerprint(self) -> str | None:
        if self._fixed_plan is None:
            return None
        payload = json.dumps(
            asdict(self._fixed_plan),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _runtime_probe_state(self) -> dict[str, Any]:
        return {
            "effective_elapsed_seconds": self.effective_elapsed(),
            "excluded_time_seconds": self._excluded_time_seconds,
            "excluded_time_by_reason": copy.deepcopy(self._excluded_time_by_reason),
            "pending_tool_overrides": copy.deepcopy(self._pending_tool_overrides),
            "override_wrong_tool_count_by_agent": copy.deepcopy(
                self._override_wrong_tool_count_by_agent
            ),
            "strategy_spawn_authorized": sorted(self._strategy_spawn_authorized),
            "candidate_deliveries": copy.deepcopy(self._candidate_deliveries),
            "candidate_delivery_seq": self._candidate_delivery_seq,
            "final_candidate_reviews": copy.deepcopy(self._final_candidate_reviews),
            "candidate_convergence_turn_by_root": copy.deepcopy(
                self._candidate_convergence_turn_by_root
            ),
            "candidate_convergence_notice_by_root": copy.deepcopy(
                self._candidate_convergence_notice_by_root
            ),
            "candidate_root_parked": sorted(self._candidate_root_parked),
            "rollback_attempts": self._rollback_attempts,
            "rollback_history": copy.deepcopy(self._rollback_history),
            "infra_failure_streak_by_agent": copy.deepcopy(
                self._infra_failure_streak_by_agent
            ),
            "infra_failure_state_by_agent": copy.deepcopy(
                self._infra_failure_state_by_agent
            ),
        }

    def _fixed_orchestration_probe_state(self) -> dict[str, Any]:
        last_green_relative = None
        last_green_absolute = None
        if self._fixed_last_green_path is not None:
            try:
                last_green_relative = str(
                    self._fixed_last_green_path.relative_to(self.config.workspace_root)
                )
            except ValueError:
                last_green_absolute = str(self._fixed_last_green_path)
        return {
            "profile": self._fixed_plan.name if self._fixed_plan else None,
            "plan_fingerprint": self._fixed_plan_fingerprint(),
            "agent_ids": copy.deepcopy(self._fixed_agent_ids),
            "agent_keys": copy.deepcopy(self._fixed_agent_keys),
            "checkpoint_agents": {
                key: sorted(agent_ids)
                for key, agent_ids in self._fixed_checkpoint_agents.items()
            },
            "agent_specs": copy.deepcopy(self._fixed_agent_specs),
            "fired_checkpoints": sorted(self._fixed_fired_checkpoints),
            "requested_checkpoints": sorted(self._fixed_requested_checkpoints),
            "pending_checkpoint_by_parent": copy.deepcopy(
                self._fixed_pending_checkpoint_by_parent
            ),
            "observed_spawns": copy.deepcopy(self._fixed_observed_spawns),
            "completion_waiting": sorted(self._fixed_completion_waiting),
            "validated_agents": sorted(self._fixed_validated_agents),
            "candidate_validations": copy.deepcopy(self._fixed_candidate_validations),
            "best_gate_metric": self._fixed_best_gate_metric,
            "green_seq": self._fixed_green_seq,
            "last_green_path_relative": last_green_relative,
            "last_green_path_absolute": last_green_absolute,
            "has_verified_green": self._fixed_has_verified_green,
            "green_restores": self._fixed_green_restores,
            "source_write_seq": self._fixed_source_write_seq,
            "last_green_write_seq": self._fixed_last_green_write_seq,
            "source_writes_in_flight": self._fixed_source_writes_in_flight,
            "source_writes_frozen": self._fixed_source_writes_frozen,
        }

    def _validate_probe_topology(self, state: dict[str, Any]) -> None:
        saved = state.get("fixed_orchestration")
        current_profile = self._fixed_plan.name if self._fixed_plan else None
        if saved is None:
            if current_profile is not None:
                raise ValueError(
                    "probe does not contain fixed orchestration state; refusing to "
                    "restart an active topology from its root"
                )
            return

        saved_profile = saved.get("profile")
        if saved_profile != current_profile:
            raise ValueError(
                "fixed orchestration profile mismatch: "
                f"probe={saved_profile!r}, runtime={current_profile!r}"
            )
        if saved_profile is not None:
            saved_fingerprint = saved.get("plan_fingerprint")
            current_fingerprint = self._fixed_plan_fingerprint()
            if not saved_fingerprint or saved_fingerprint != current_fingerprint:
                raise ValueError(
                    "fixed orchestration plan changed since the probe was written; "
                    "refusing to resume at a different topology node"
                )

    def _restore_runtime_probe_state(self, state: dict[str, Any]) -> None:
        saved = state.get("runtime_resume_state")
        if not saved:
            return
        elapsed = max(0.0, float(saved.get("effective_elapsed_seconds", 0.0)))
        excluded = max(0.0, float(saved.get("excluded_time_seconds", 0.0)))
        self._excluded_time_seconds = excluded
        self._excluded_time_by_reason = copy.deepcopy(
            saved.get("excluded_time_by_reason") or {}
        )
        self._start_time = time.time() - elapsed - excluded
        self._pending_tool_overrides = copy.deepcopy(
            saved.get("pending_tool_overrides") or {}
        )
        self._override_wrong_tool_count_by_agent = copy.deepcopy(
            saved.get("override_wrong_tool_count_by_agent") or {}
        )
        self._strategy_spawn_authorized = set(
            saved.get("strategy_spawn_authorized") or []
        )
        self._candidate_deliveries = copy.deepcopy(
            saved.get("candidate_deliveries") or []
        )
        self._candidate_delivery_seq = int(saved.get("candidate_delivery_seq") or 0)
        self._candidate_delivery_events = {}
        self._final_candidate_reviews = copy.deepcopy(
            saved.get("final_candidate_reviews") or {}
        )
        self._candidate_convergence_turn_by_root = copy.deepcopy(
            saved.get("candidate_convergence_turn_by_root") or {}
        )
        self._candidate_convergence_notice_by_root = copy.deepcopy(
            saved.get("candidate_convergence_notice_by_root") or {}
        )
        self._candidate_root_parked = set(saved.get("candidate_root_parked") or [])
        self._rollback_attempts = int(saved.get("rollback_attempts") or 0)
        self._rollback_history = copy.deepcopy(saved.get("rollback_history") or [])
        self._rollback_requested = None
        self._infra_failure_streak_by_agent = copy.deepcopy(
            saved.get("infra_failure_streak_by_agent") or {}
        )
        self._infra_failure_state_by_agent = copy.deepcopy(
            saved.get("infra_failure_state_by_agent") or {}
        )

    def _restore_fixed_orchestration_probe_state(
        self,
        state: dict[str, Any],
        old_workspace_root: Path | None,
    ) -> None:
        saved = state.get("fixed_orchestration")
        if not saved or saved.get("profile") is None:
            return
        self._fixed_agent_ids = copy.deepcopy(saved.get("agent_ids") or {})
        self._fixed_agent_keys = copy.deepcopy(saved.get("agent_keys") or {})
        self._fixed_checkpoint_agents = {
            key: set(agent_ids)
            for key, agent_ids in (saved.get("checkpoint_agents") or {}).items()
        }
        self._fixed_agent_specs = copy.deepcopy(saved.get("agent_specs") or {})
        self._fixed_fired_checkpoints = set(saved.get("fired_checkpoints") or [])
        self._fixed_requested_checkpoints = set(
            saved.get("requested_checkpoints") or []
        )
        self._fixed_pending_checkpoint_by_parent = copy.deepcopy(
            saved.get("pending_checkpoint_by_parent") or {}
        )
        self._fixed_observed_spawns = copy.deepcopy(saved.get("observed_spawns") or [])
        self._fixed_completion_waiting = set(saved.get("completion_waiting") or [])
        self._fixed_validated_agents = set(saved.get("validated_agents") or [])
        self._fixed_candidate_validations = copy.deepcopy(
            saved.get("candidate_validations") or {}
        )
        metric = saved.get("best_gate_metric")
        self._fixed_best_gate_metric = float(metric) if metric is not None else None
        self._fixed_green_seq = int(saved.get("green_seq") or 0)
        relative_green = saved.get("last_green_path_relative")
        absolute_green = saved.get("last_green_path_absolute")
        if relative_green:
            self._fixed_last_green_path = self.config.workspace_root / relative_green
        elif absolute_green:
            green_path = Path(absolute_green)
            if old_workspace_root is not None:
                try:
                    green_path = self.config.workspace_root / green_path.relative_to(
                        old_workspace_root
                    )
                except ValueError:
                    pass
            self._fixed_last_green_path = green_path
        else:
            self._fixed_last_green_path = None
        self._fixed_has_verified_green = bool(saved.get("has_verified_green", False))
        self._fixed_green_restores = int(saved.get("green_restores") or 0)
        self._fixed_source_write_seq = int(saved.get("source_write_seq") or 0)
        self._fixed_last_green_write_seq = int(
            saved.get("last_green_write_seq") or 0
        )
        # A restored process has no tool handler currently writing source files.
        self._fixed_source_writes_in_flight = 0
        self._fixed_source_writes_frozen = bool(
            saved.get("source_writes_frozen", False)
        )
        self._fixed_monitor_task = None

        known_agents = set(self.agents)
        referenced_agents = (
            set(self._fixed_agent_ids.values())
            | set(self._fixed_agent_keys)
            | set(self._fixed_agent_specs)
            | set(self._fixed_pending_checkpoint_by_parent)
            | set(self._fixed_completion_waiting)
            | set(self._fixed_validated_agents)
            | {
                agent_id
                for agent_ids in self._fixed_checkpoint_agents.values()
                for agent_id in agent_ids
            }
        )
        unknown_agents = sorted(referenced_agents - known_agents)
        if unknown_agents:
            raise ValueError(
                "fixed orchestration probe references missing agents: "
                + ", ".join(unknown_agents)
            )

    def _write_probe(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
        *,
        reason: str = "manual_probe",
        global_checkpoint: dict[str, Any] | None = None,
    ) -> Path:
        root_dir = self.config.probe_dir or self.config.auto_checkpoint_dir
        if not root_dir:
            raise RuntimeError("probe_dir or auto_checkpoint_dir is required when probes/checkpoints are enabled")
        self._probe_counter += 1
        probe_dir = root_dir / f"{self._probe_counter:05d}_{agent.id}_turn{agent._turns}"
        snapshot_dir = probe_dir / "workspace_snapshot"
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        probe_dir.mkdir(parents=True, exist_ok=True)
        if self.config.workspace_root.exists():
            ignore_names = {".probes", "probes"}
            try:
                checkpoint_relative = root_dir.resolve().relative_to(
                    self.config.workspace_root.resolve()
                )
            except ValueError:
                checkpoint_relative = None
            if checkpoint_relative is not None:
                if not checkpoint_relative.parts:
                    raise RuntimeError(
                        "probe/checkpoint directory cannot equal workspace_root"
                    )
                ignore_names.add(checkpoint_relative.parts[0])
            ignore = shutil.ignore_patterns(*sorted(ignore_names))
            shutil.copytree(self.config.workspace_root, snapshot_dir, ignore=ignore)
        probe_path = probe_dir / "state.pkl"
        if global_checkpoint is not None:
            global_checkpoint["path"] = str(probe_path)

        state = {
            "schema_version": 2,
            "created_at": time.time(),
            "active_agent": agent.id,
            "workspace_root": str(self.config.workspace_root),
            "probe_counter": self._probe_counter,
            "global_checkpoint_seq": (
                global_checkpoint.get("seq") if global_checkpoint else None
            ),
            "global_checkpoint": copy.deepcopy(global_checkpoint),
            "global_checkpoints": copy.deepcopy([
                *self._global_checkpoints,
                *([global_checkpoint] if global_checkpoint else []),
            ]),
            "id_counter": self._id_gen._counter,
            "ledger": copy.deepcopy(self.ledger),
            "messages_sent": copy.deepcopy(self._messages_sent),
            "events": copy.deepcopy(self._events),
            "agents": {
                agent_id: self._agent_snapshot(agent_obj)
                for agent_id, agent_obj in self.agents.items()
            },
            "turn_tools": list(turn_tools),
            "tool_policy": tool_policy.as_event(),
            "strategy_state": self._build_strategy_state(agent, turn_tools, tool_policy).as_event(),
            "runtime_resume_state": self._runtime_probe_state(),
            "fixed_orchestration": self._fixed_orchestration_probe_state(),
        }
        import pickle
        with probe_path.open("wb") as f:
            pickle.dump(state, f)

        (probe_dir / "meta.json").write_text(json.dumps({
            "schema_version": 2,
            "active_agent": agent.id,
            "probe_counter": self._probe_counter,
            "global_checkpoint_seq": (
                global_checkpoint.get("seq") if global_checkpoint else None
            ),
            "global_checkpoint": global_checkpoint,
            "reason": reason,
            "agent_turn": agent._turns,
            "turn_tools": list(turn_tools),
            "tool_policy": tool_policy.as_event(),
            "strategy_state": state["strategy_state"],
            "effective_elapsed_seconds": state["runtime_resume_state"][
                "effective_elapsed_seconds"
            ],
            "fixed_orchestration": {
                "profile": state["fixed_orchestration"]["profile"],
                "plan_fingerprint": state["fixed_orchestration"]["plan_fingerprint"],
                "fired_checkpoints": state["fixed_orchestration"]["fired_checkpoints"],
                "requested_checkpoints": state["fixed_orchestration"][
                    "requested_checkpoints"
                ],
                "pending_checkpoint_by_parent": state["fixed_orchestration"][
                    "pending_checkpoint_by_parent"
                ],
                "agent_ids": state["fixed_orchestration"]["agent_ids"],
            },
            "workspace_snapshot": str(snapshot_dir),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return probe_path

    def restore_probe(self, probe_path: Path) -> str:
        import pickle
        with probe_path.open("rb") as f:
            state = pickle.load(f)
        self._validate_probe_topology(state)
        snapshot_dir = probe_path.parent / "workspace_snapshot"
        if snapshot_dir.exists():
            if self.config.workspace_root.exists():
                shutil.rmtree(self.config.workspace_root)
            shutil.copytree(snapshot_dir, self.config.workspace_root)
            self._tool_context.workspace_root = self.config.workspace_root
            self._tool_context.workspace_extra_roots = tuple(self.config.workspace_extra_roots)
            self._tool_context.shared_dir = self.config.workspace_root / self.config.shared_dir

        self.ledger = copy.deepcopy(state["ledger"])
        self._messages_sent = copy.deepcopy(state.get("messages_sent", []))
        self._events = copy.deepcopy(state.get("events", []))
        self._id_gen._counter = int(state.get("id_counter", 0))
        self._probe_counter = int(state.get("probe_counter", 0))
        self._global_checkpoint_seq = int(state.get("global_checkpoint_seq") or 0)
        current_global_checkpoint = copy.deepcopy(state.get("global_checkpoint"))
        state_global_checkpoints = copy.deepcopy(state.get("global_checkpoints") or [])
        if state_global_checkpoints:
            self._global_checkpoints = [
                copy.deepcopy(record)
                for record in state_global_checkpoints
                if int(record.get("seq") or 0) <= self._global_checkpoint_seq
            ]
        else:
            self._global_checkpoints = [
                copy.deepcopy(record)
                for record in self._global_checkpoints
                if int(record.get("seq") or 0) <= self._global_checkpoint_seq
            ]
        if current_global_checkpoint:
            if not any(
                record.get("path") == current_global_checkpoint.get("path")
                for record in self._global_checkpoints
            ):
                self._global_checkpoints.append(current_global_checkpoint)
        probe_root = self.config.probe_dir or self.config.auto_checkpoint_dir
        if probe_root and probe_root.exists():
            max_counter = self._probe_counter
            for child in probe_root.iterdir():
                if not child.is_dir():
                    continue
                prefix = child.name.split("_", 1)[0]
                if prefix.isdigit():
                    max_counter = max(max_counter, int(prefix))
            self._probe_counter = max_counter
        old_workspace_root = state.get("workspace_root")
        if old_workspace_root is None and state.get("agents"):
            first_agent = next(iter(state["agents"].values()))
            old_workspace_root = str(Path(first_agent.get("workspace", "")).parent)
        old_workspace_root_path = Path(old_workspace_root) if old_workspace_root else None
        self.agents = {
            agent_id: self._restore_agent(agent_data, old_workspace_root_path)
            for agent_id, agent_data in state.get("agents", {}).items()
        }
        for agent in self.agents.values():
            agent._task = None
        self._restore_runtime_probe_state(state)
        self._restore_fixed_orchestration_probe_state(state, old_workspace_root_path)
        self._last_checkpoint_path = probe_path
        self._last_checkpoint_agent = str(state["active_agent"])
        return str(state["active_agent"])

    async def _continue_fixed_from_probe(self) -> str:
        root_id = self._fixed_agent_ids.get("root")
        root = self.agents.get(root_id or "")
        if root is None:
            root = next((agent for agent in self.agents.values() if agent.parent is None), None)
        if root is None:
            raise ValueError("fixed orchestration probe has no root agent")
        root_id = root.id

        for agent in self.agents.values():
            if agent.status == "running" and (agent._task is None or agent._task.done()):
                self.start_agent(agent)
        self._fixed_monitor_task = asyncio.create_task(
            self._fixed_orchestration_monitor()
        )
        try:
            while True:
                while not self._rollback_requested:
                    active_tasks = [
                        agent._task
                        for agent in self.agents.values()
                        if agent.status == "running"
                        and agent._task is not None
                        and not agent._task.done()
                    ]
                    if not active_tasks:
                        break
                    done, _pending = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    await asyncio.gather(*done, return_exceptions=True)
                    root = self.agents.get(root_id) or root
                    if (
                        root._task is not None
                        and root._task.done()
                        and not self._rollback_requested
                    ):
                        break
                if not self._rollback_requested:
                    break
                await self._perform_requested_rollback()
                root = self.agents.get(root_id) or next(
                    (agent for agent in self.agents.values() if agent.parent is None),
                    None,
                )
                if root is None:
                    break
                root_id = root.id
                if root._task is None or root._task.done():
                    self.start_agent(root)
        finally:
            self._fixed_source_writes_frozen = True
            if self._fixed_monitor_task and not self._fixed_monitor_task.done():
                self._fixed_monitor_task.cancel()
                await asyncio.gather(self._fixed_monitor_task, return_exceptions=True)

        running = [
            agent
            for agent in self.agents.values()
            if agent.status in ("running", "idle") and agent.id != root.id
        ]
        if running:
            await self.shutdown()
        if self._fixed_source_write_seq != self._fixed_last_green_write_seq:
            self._fixed_restore_green_checkpoint(
                root.id,
                "runtime_finished_with_dirty_source",
            )
        return root.result or ""

    async def continue_from_probe(self, probe_path: Path, agent_id: str | None = None) -> str:
        restored_active = self.restore_probe(probe_path)
        active = agent_id or restored_active
        if active not in self.agents:
            raise ValueError(f"probe does not contain agent {active!r}")
        agent = self.agents[active]
        resumed_agents = (
            self.agents.values() if self._fixed_plan else (agent,)
        )
        for resumed_agent in resumed_agents:
            if self.config.max_turns > 0:
                resumed_agent.quota.max_turns = self.config.max_turns
            if self.config.time_limit > 0:
                resumed_agent.quota.time_limit = self.config.time_limit
        if self.config.probe_resume_instruction:
            agent.history.append({
                "role": "user",
                "content": (
                    "[Probe branch instruction]\n"
                    f"{self.config.probe_resume_instruction}"
                ),
            })
        agent.status = "running"
        if self._fixed_plan:
            return await self._continue_fixed_from_probe()
        self.start_agent(agent)
        await agent._task
        running = [
            a for a in self.agents.values()
            if a.status in ("running", "idle") and a.id != agent.id
        ]
        if running:
            await self.shutdown()
        return agent.result or ""
