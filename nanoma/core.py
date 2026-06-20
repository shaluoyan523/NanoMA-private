"""Core runtime: Agent, Runtime, ReAct loop."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable, Literal
from urllib.parse import parse_qsl, unquote, urlparse

from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import (
    LLMResponse, Message, RetryConfig, ToolCall, ToolDef,
    count_message_tokens, default_llm_call, estimate_tokens, set_log_dir,
)
from nanoma.scheduler import Scheduler
from nanoma.tools import WORK_TOOLS
from nanoma.plugins.workspace_tools import WORKSPACE_TOOLS

logger = logging.getLogger("nanoma")

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
_COORDINATION_TOOLS = {"send", "wait", "query", "kill", "transfer", "set_bio"}
_LIFECYCLE_TOOLS = {"get_cost", "set_status", "rebirth", "submit"}
_READ_TOOLS = {"ws_read_file", "ws_grep", "ws_code_outline", "ws_read_symbol", "query", "get_cost", "shell"}
_WORK_TOOLS = {
    "shell", "ws_create_file", "ws_append_file", "ws_replace_string",
    "ws_multi_replace", "ws_apply_patch", "batch", "submit",
}
_FINISH_TOOLS = {"set_status", "submit"}
_OUTPUT_SUFFIXES = {
    ".answer", ".csv", ".html", ".json", ".jsonl", ".md", ".out",
    ".pdf", ".txt", ".tsv", ".xml", ".yaml", ".yml",
}
_SHELL_CAPABILITIES = {"web", "python", "fs", "process", "package", "system", "unknown"}
_BACKTICK_PATH_RE = re.compile(r"`([^`]+)`")
_NAMED_PATH_RE = re.compile(r"\bnamed\s+([A-Za-z0-9_./$-]+\.[A-Za-z0-9]+)", re.IGNORECASE)
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
    "robot check",
    "temporarily unavailable",
    "traceback",
    "validation-failure",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _looks_like_output_path(text: str) -> bool:
    text = text.strip().strip("'\".,);:")
    if not text or "://" in text:
        return False
    return Path(text).suffix.lower() in _OUTPUT_SUFFIXES


def _extract_expected_output_paths(task: str) -> list[str]:
    """Best-effort extraction for explicit deliverable paths in benchmark prompts."""
    paths: list[str] = []
    for match in _BACKTICK_PATH_RE.finditer(task):
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


def classify_shell_capability(command: str) -> ShellCapability:
    """Classify a shell command for internal policy pruning.

    The public tool remains a single `shell` function. This classification is
    used only by the runtime to narrow what that tool may execute under
    constraint pressure.
    """
    cmd = command.strip()
    if not cmd:
        return "unknown"

    first = re.split(r"\s+", cmd, maxsplit=1)[0].split("/")[-1]
    lowered = cmd.lower()

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
    if first in {"ps", "pkill", "kill", "killall", "jobs", "pgrep", "sleep", "timeout"}:
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
    shell_max_output: int = 10000
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
    expected_outputs: int = 0
    missing_outputs: int = 0
    active_children: int = 0
    reason: str = "off"
    scoped_tools: list[str] = field(default_factory=list)
    removed_tools: list[str] = field(default_factory=list)

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
            },
            "expected_outputs": self.expected_outputs,
            "missing_outputs": self.missing_outputs,
            "active_children": self.active_children,
            "reason": self.reason,
            "scoped_tools": self.scoped_tools,
            "removed_tools": self.removed_tools,
        }


@dataclass
class ShellActivityState:
    web_calls: int = 0
    web_low_signal_calls: int = 0
    blocked_web_calls: int = 0
    consecutive_web_low_signal: int = 0
    repeated_web_queries: int = 0
    repeated_web_domains: int = 0
    unique_web_signatures: set[str] = field(default_factory=set)
    unique_web_domains: set[str] = field(default_factory=set)
    last_web_signature: str = ""
    last_web_domain: str = ""
    web_domain_sprawl: float = 0.0
    web_saturation: float = 0.0
    finalize_after_web_saturation: bool = False

    def as_event(self) -> dict[str, Any]:
        return {
            "web_calls": self.web_calls,
            "web_low_signal_calls": self.web_low_signal_calls,
            "blocked_web_calls": self.blocked_web_calls,
            "consecutive_web_low_signal": self.consecutive_web_low_signal,
            "repeated_web_queries": self.repeated_web_queries,
            "repeated_web_domains": self.repeated_web_domains,
            "unique_web_signatures": len(self.unique_web_signatures),
            "unique_web_domains": len(self.unique_web_domains),
            "last_web_domain": self.last_web_domain,
            "web_domain_sprawl": round(self.web_domain_sprawl, 3),
            "web_saturation": round(self.web_saturation, 3),
            "finalize_after_web_saturation": self.finalize_after_web_saturation,
        }


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    max_agents: int = 1000
    max_depth: int = 100
    max_concurrent_llm: int = 50
    budget: float = 10.0
    max_total_tokens: int = 0
    tool_policy_soft_total_tokens: int = 0
    time_limit: float = 0.0
    max_turns: int = 200
    allowed_models: list[str] | None = None
    disabled_tools: set[str] = field(default_factory=set)
    context_compress_ratio: float = 0.8
    default_model: str = "deepseek-v4-flash"
    log_dir: Path | None = field(default_factory=lambda: Path("./logs"))
    workspace_root: Path = field(default_factory=lambda: Path("./workspace"))
    shared_dir: str = "shared"
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
        "shell", "get_cost", "set_status", "submit",
        "ws_read_file", "ws_create_file", "ws_append_file",
    })
    tool_policy_prune_shell_capabilities: bool = False
    tool_policy_shell_capability_pressure_start: float = 0.55
    tool_policy_shell_capability_pressure_end: float = 0.95
    tool_policy_web_saturation_enabled: bool = True
    tool_policy_web_saturation_min_calls: int = 10
    tool_policy_web_saturation_threshold: float = 0.85
    tool_policy_web_saturation_finalize_after_blocks: int = 2
    tool_policy_log_events: bool = False
    # Resource notification thresholds (fraction consumed, e.g. 0.5 = 50%)
    notify_thresholds: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.70, 0.80, 0.90, 0.95])
    # Compression / truncation settings
    compress_keep_recent: int = 6           # messages to keep verbatim during compression
    compress_max_messages: int = 40         # max old messages to include in summary
    compress_max_chars: int = 300           # max chars per message in summary (0 = unlimited)
    shell_max_output: int = 10000           # max chars for shell output (0 = unlimited)
    file_read_max_chars: int = 50000        # max chars for file_read (0 = unlimited)
    file_list_max_entries: int = 500        # max entries for file_list (0 = unlimited)
    grep_max_results: int = 100             # max grep results (0 = unlimited)
    blocked_shell_patterns: list[str] = field(default_factory=list)
    probe_dir: Path | None = None
    probe_every_llm: bool = False
    probe_stop_after: int = 0
    force_spawn_turns: int = 0
    probe_resume_instruction: str | None = None


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
    _last_tool_policy: dict[str, Any] | None = field(default=None, repr=False)
    _shell_activity: ShellActivityState = field(default_factory=ShellActivityState, repr=False)


# ─── Runtime ─────────────────────────────────────────────────────────────────

class Runtime:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        llm_call: Callable | None = None,
        router: Callable | None = None,
        on_event: Callable | None = None,
    ):
        self.config = config or RuntimeConfig()
        self.ledger = CostLedger(total_budget=self.config.budget)
        self.agents: dict[str, Agent] = {}
        self._id_gen = IdGenerator()
        self._tool_context = ToolContext(
            shared_dir=self.config.workspace_root / self.config.shared_dir,
            workspace_root=self.config.workspace_root,
            shell_max_output=self.config.shell_max_output,
            file_read_max_chars=self.config.file_read_max_chars,
            file_list_max_entries=self.config.file_list_max_entries,
            grep_max_results=self.config.grep_max_results,
            blocked_shell_patterns=list(self.config.blocked_shell_patterns),
        )
        self.llm_call = llm_call or default_llm_call
        self.router = router
        self.scheduler = Scheduler(max_concurrent=self.config.max_concurrent_llm)
        self.on_event = on_event or (lambda e: None)
        self._start_time = time.time()
        self._events: list[dict] = []  # all events for post-hoc analysis
        self._messages_sent: list[tuple[str, str, int]] = []  # (from, to, tokens) for comm graph
        self._emit_lock = threading.Lock()  # protects events.jsonl writes
        self._probe_counter = 0

        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        self._tool_context.shared_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_dir:
            set_log_dir(self.config.log_dir)

    # ─── Agent lifecycle ─────────────────────────────────────────────────

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
            }

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

        self._emit(agent_id, "agent_new", {
            "task": task, "model": model, "budget": quota.budget if math.isfinite(quota.budget) else None,
            "parent": parent, "depth": depth,
        })
        return agent

    def start_agent(self, agent: Agent):
        agent._task = asyncio.ensure_future(self._agent_loop(agent))

    async def run(self, task: str, model: str | None = None) -> str:
        """Run a single root agent to completion, then shut down all remaining agents."""
        root = self.create_agent(task, model=model)
        self.start_agent(root)
        await root._task
        # Clean up any still-running children
        running = [a for a in self.agents.values() if a.status in ("running", "idle") and a.id != root.id]
        if running:
            await self.shutdown()
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

    # ─── ReAct loop ──────────────────────────────────────────────────────

    async def _agent_loop(self, agent: Agent):
        from nanoma.meta import META_TOOLS
        # Tool set: shell (universal primitive) + workspace (structured I/O) + meta (coordination)
        all_tools = {
            name: tool
            for name, tool in {**WORK_TOOLS, **WORKSPACE_TOOLS, **META_TOOLS}.items()
            if name not in self.config.disabled_tools
        }

        try:
            while agent.status == "running":
                agent._turns += 1
                agent._last_active = time.time()

                # Turn limits
                if agent.quota.max_turns > 0 and agent._turns > agent.quota.max_turns:
                    agent.status = "done"
                    agent.result = agent.result or "[Max turns reached]"
                    break

                # Time limit
                if agent.quota.time_limit > 0:
                    elapsed = time.time() - self._start_time
                    if elapsed >= agent.quota.time_limit:
                        agent.status = "done"
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

                # Inject queued messages
                self._inject_messages(agent, agent._queue_inbox)
                self._inject_messages(agent, agent._steer_inbox)

                # Context compression
                agent.context_tokens = count_message_tokens(agent.history)
                if agent.context_tokens > int(agent.context_limit * self.config.context_compress_ratio):
                    agent.history = await self._compress(agent.history)
                    agent.context_tokens = count_message_tokens(agent.history)

                turn_tools, tool_policy = self._apply_state_tool_policy(agent, all_tools)
                if (
                    self.config.force_spawn_turns > 0
                    and agent.parent is None
                    and not agent.children
                    and agent._turns <= self.config.force_spawn_turns
                    and "spawn" in turn_tools
                ):
                    preferred = _CREATE_TOOLS | {"get_cost", "set_status"}
                    forced = {name: tool for name, tool in turn_tools.items() if name in preferred}
                    if forced:
                        turn_tools = forced
                        tool_policy.reason = f"{tool_policy.reason},force_spawn_turns"
                        tool_policy.scoped_tools = list(turn_tools)
                        agent.history.append({
                            "role": "user",
                            "content": (
                                "[Probe branch tool override]\n"
                                "For this resumed branch, shell and workspace write tools are intentionally unavailable "
                                "until you create at least one child agent. Call spawn now. Prefer spawning separate "
                                "solver, verifier, and golfer agents, each with complete task context. Omit the spawn "
                                f"model argument or use {self.config.default_model}; do not request other models. "
                                "Do not call shell."
                            ),
                        })
                tool_schemas = [t["schema"] for t in turn_tools.values()]
                agent._last_tool_policy = tool_policy.as_event()

                if self.config.probe_every_llm:
                    probe_path = self._write_probe(agent, turn_tools, tool_policy)
                    self._emit(agent.id, "probe", {
                        "path": str(probe_path),
                        "turn": agent._turns,
                        "tools": list(turn_tools),
                        "tool_policy": tool_policy.as_event(),
                    })
                    if self.config.probe_stop_after > 0 and self._probe_counter >= self.config.probe_stop_after:
                        agent.status = "done"
                        agent.result = f"[Probe stop after {self._probe_counter} probes]"
                        break

                # LLM call
                await self.scheduler.acquire()
                try:
                    response = await self.llm_call(
                        agent.history, agent.model, tool_schemas,
                        retry_config=self.config.retry,
                    )
                finally:
                    self.scheduler.release()

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
                }
                if self.config.tool_policy_log_events:
                    llm_event["tool_policy"] = tool_policy.as_event()
                self._emit(agent.id, "llm_done", llm_event)

                if self.config.max_total_tokens > 0:
                    total_tokens = sum(a.tokens_consumed for a in self.agents.values())
                    if total_tokens >= self.config.max_total_tokens:
                        agent.status = "done"
                        agent.result = agent.result or f"[Max total tokens reached: {total_tokens}]"
                        for other in self.agents.values():
                            if other.status in {"running", "idle"}:
                                other.status = "done"
                                other.result = other.result or agent.result
                        self._emit(agent.id, "done", {
                            "status": "done",
                            "reason": "max_total_tokens",
                            "tokens": total_tokens,
                            "max_total_tokens": self.config.max_total_tokens,
                            "result": agent.result,
                        })
                        break

                # Process response
                if response.tool_calls:
                    agent._no_tool_turns = 0
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
                            msgs = self._inject_messages(agent, agent._immediate_inbox)
                            for skipped in response.tool_calls[i:]:
                                agent.history.append({
                                    "role": "tool", "tool_call_id": skipped.id,
                                    "content": json.dumps({"interrupted": True}),
                                })
                            break

                        result = await self._execute_tool(tc, agent, turn_tools)
                        agent._tool_calls += 1
                        result_json = json.dumps(result, ensure_ascii=False, default=str)
                        agent.history.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": result_json[:8000],
                        })
                        # Full tool event for viewer (no truncation on args, reasonable on result)
                        self._emit(agent.id, "tool_call", {
                            "tool": tc.name,
                            "args": tc.arguments,
                            "result": result_json[:5000],
                            "result_full_len": len(result_json),
                        })

                    # Steer messages before next LLM call
                    self._inject_messages(agent, agent._steer_inbox)

                elif response.content:
                    agent._no_tool_turns += 1
                    agent.history.append({"role": "assistant", "content": response.content})
                else:
                    agent._no_tool_turns += 1
                    agent.history.append({"role": "assistant", "content": "(empty)"})

                if agent.status != "running":
                    break
                if not self.ledger.can_afford(0):
                    agent.status = "failed"
                    agent.result = "Budget exhausted"
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Agent {agent.id} crashed: {e}")
            agent.status = "failed"
            agent.result = f"Crash: {e}"
        finally:
            # Only notify parent and emit done event if actually finished (not just idle)
            if agent.status in ("done", "failed"):
                if agent.parent and agent.parent in self.agents:
                    result_preview = (agent.result or "")[:200]
                    death_msg = f"[Agent {agent.id} finished: {agent.status}] {result_preview}"
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

        elapsed = time.time() - self._start_time
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
            pressures.append(_clamp01((time.time() - self._start_time) / agent.quota.time_limit))
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
        if self.config.max_agents > 0:
            pressures.append(_clamp01(len(self.agents) / self.config.max_agents))
        return max(pressures, default=0.0)

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
        depth_pressure = _clamp01(agent.depth / self.config.max_depth) if self.config.max_depth > 0 else 1.0
        resource_pressure = self._resource_pressure(agent)
        stagnation_pressure = _clamp01(agent._no_tool_turns / 3.0)

        can_create = self._spawn_unavailable_reason(agent) is None

        create = (
            0.40
            + 0.55 * artifact_gap
            + 0.25 * stagnation_pressure
            - 0.75 * resource_pressure
            - 0.55 * dependency_pressure
            - 0.45 * depth_pressure
        )
        if agent.parent:
            create -= 0.20
        if not can_create:
            create = 0.0

        read = 0.25 + 0.25 * stagnation_pressure + 0.20 * (0.0 if expected else 1.0) + 0.15 * dependency_pressure
        message = 0.20 + 0.65 * dependency_pressure + (0.10 if agent.children else 0.0)
        work = 0.45 + 0.45 * artifact_gap + 0.15 * (1.0 - dependency_pressure) - 0.25 * resource_pressure
        finish = (
            0.10
            + (0.65 if expected and not missing else 0.0)
            + (0.20 if not expected and agent._turns > 1 else 0.0)
            + 0.25 * resource_pressure
            - 0.55 * artifact_gap
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
            expected_outputs=len(expected),
            missing_outputs=len(missing),
            active_children=len(active_children),
        )

    def _allowed_shell_capabilities(self, state: ToolPolicyState, agent: Agent | None = None) -> set[str]:
        """Return the current shell sub-capabilities allowed by constraints."""
        all_caps = set(_SHELL_CAPABILITIES)
        if (
            self.config.tool_policy_mode == "off"
            or not self.config.tool_policy_prune_shell_capabilities
        ):
            return all_caps

        web_saturated = (
            agent is not None
            and self.config.tool_policy_web_saturation_enabled
            and agent._shell_activity.web_calls >= self.config.tool_policy_web_saturation_min_calls
            and agent._shell_activity.web_saturation >= self.config.tool_policy_web_saturation_threshold
        )
        web_pressure = (
            agent._shell_activity.web_saturation
            if agent is not None and agent._shell_activity.web_calls >= self.config.tool_policy_web_saturation_min_calls
            else 0.0
        )

        start = self.config.tool_policy_shell_capability_pressure_start
        end = self.config.tool_policy_shell_capability_pressure_end
        pressure = max(state.resource_pressure, state.finish, web_pressure)
        if pressure < start and not web_saturated:
            return all_caps

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

        if web_saturated:
            allowed.discard("web")

        return allowed

    def _refresh_tool_context_policy(self, agent: Agent) -> ToolPolicyState:
        state = self._compute_tool_policy_state(agent)
        self._tool_context.allowed_shell_capabilities = self._allowed_shell_capabilities(state, agent)
        return state

    def _update_shell_activity(self, agent: Agent, command: str, result: Any) -> None:
        if not self.config.tool_policy_prune_shell_capabilities:
            return
        capability = classify_shell_capability(command)
        if capability != "web":
            return

        activity = agent._shell_activity
        if isinstance(result, dict) and result.get("blocked") and result.get("blocked_capability") == "web":
            activity.blocked_web_calls += 1
            if (
                self.config.tool_policy_web_saturation_enabled
                and activity.web_saturation >= self.config.tool_policy_web_saturation_threshold
                and activity.blocked_web_calls >= self.config.tool_policy_web_saturation_finalize_after_blocks
            ):
                activity.finalize_after_web_saturation = True
                self._emit(agent.id, "shell_activity", activity.as_event())
            return

        activity.web_calls += 1

        signature = _web_command_signature(command)
        domain = _web_command_domain(command)

        repeated_query = bool(signature and signature in activity.unique_web_signatures)
        repeated_domain = bool(domain and domain == activity.last_web_domain)
        low_signal = _web_result_low_signal(result)

        if repeated_query:
            activity.repeated_web_queries += 1
        if repeated_domain:
            activity.repeated_web_domains += 1
        if low_signal:
            activity.web_low_signal_calls += 1
            activity.consecutive_web_low_signal += 1
        else:
            activity.consecutive_web_low_signal = 0

        if signature:
            activity.unique_web_signatures.add(signature)
            activity.last_web_signature = signature
        if domain:
            activity.unique_web_domains.add(domain)
            activity.last_web_domain = domain

        calls = max(1, activity.web_calls)
        low_signal_ratio = activity.web_low_signal_calls / calls
        repeat_query_ratio = activity.repeated_web_queries / calls
        repeat_domain_ratio = activity.repeated_web_domains / calls
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

        if (
            self.config.tool_policy_log_events
            or activity.web_saturation >= self.config.tool_policy_web_saturation_threshold
        ):
            self._emit(agent.id, "shell_activity", activity.as_event())

    def _apply_state_tool_policy(self, agent: Agent, tools: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], ToolPolicyState]:
        state = self._compute_tool_policy_state(agent)
        if self.config.tool_policy_mode == "off":
            state.reason = "off"
            state.scoped_tools = list(tools)
            return tools, state

        scoped = dict(tools)
        original_order = {name: idx for idx, name in enumerate(tools)}
        removed: set[str] = set()
        reasons: list[str] = []

        if agent._shell_activity.finalize_after_web_saturation:
            finalize_allowed = {
                "ws_create_file", "ws_append_file", "ws_read_file",
                "submit", "set_status", "get_cost",
            }
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

        spawn_unavailable = self._spawn_unavailable_reason(agent)
        spawn_should_close = (
            spawn_unavailable is not None
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
            reasons.append(spawn_unavailable or "spawn_closed")

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
                allowed = _READ_TOOLS | _LIFECYCLE_TOOLS | {"shell"}
                reasons.append("finish_scope")
            elif state.create >= self.config.tool_policy_spawn_min_weight and state.create >= state.work:
                allowed = _CREATE_TOOLS | _COORDINATION_TOOLS | _READ_TOOLS | _LIFECYCLE_TOOLS | {"shell"}
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

    def _tool_policy_score(self, tool_name: str, state: ToolPolicyState) -> float:
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
        pressure = max(state.resource_pressure, state.finish)
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
        if self.config.tool_policy_mode == "off":
            return None
        unavailable = self._spawn_unavailable_reason(agent)
        if unavailable:
            return unavailable
        state = self._compute_tool_policy_state(agent)
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

    def _all_tools(self) -> dict[str, dict[str, Any]]:
        from nanoma.meta import META_TOOLS
        return {
            name: tool
            for name, tool in {**WORK_TOOLS, **WORKSPACE_TOOLS, **META_TOOLS}.items()
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
        tool_info = tools.get(tc.name)
        if not tool_info:
            return {
                "error": f"Unknown or unavailable tool this turn: {tc.name}",
                "available_tools": sorted(tools),
        }
        handler = tool_info["handler"]
        try:
            if tool_info.get("is_meta"):
                if (
                    tc.name == "set_status"
                    and agent._shell_activity.finalize_after_web_saturation
                ):
                    requested_status = str(tc.arguments.get("status", "done"))
                    requested_result = str(tc.arguments.get("result", "") or "").strip()
                    if requested_status == "idle":
                        return {
                            "error": (
                                "idle is unavailable after web saturation; "
                                "call set_status(done, result=...) or submit instead"
                            ),
                            "required_status": "done",
                        }
                    if requested_status == "done" and not requested_result:
                        return {
                            "error": (
                                "empty done result is unavailable after web saturation; "
                                "call set_status(done, result=<answer>) or submit instead"
                            ),
                            "required_status": "done",
                            "required_result": "non-empty answer",
                        }
                return await handler(tc.arguments, agent, self)
            else:
                self._refresh_tool_context_policy(agent)
                result = await handler(tc.arguments, agent.workspace, self._tool_context)
                if tc.name == "shell":
                    self._update_shell_activity(
                        agent,
                        str(tc.arguments.get("command", "")),
                        result,
                    )
                    self._refresh_tool_context_policy(agent)
                return result
        except Exception as e:
            return {"error": str(e)}

    async def _compress(self, history: list[Message], keep_recent: int | None = None) -> list[Message]:
        """Compress old messages into a summary, keeping recent ones intact."""
        keep_recent = keep_recent or self.config.compress_keep_recent
        if len(history) <= keep_recent + 2:
            return history
        system = history[0]
        old = history[1:-keep_recent]
        recent = history[-keep_recent:]
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

        # Context about spawner (for sub-agents)
        context_section = ""
        if parent_context:
            context_section = f"""
## Your Context
- Spawned by: agent "{parent_context['parent_id']}" (task: {parent_context['parent_task']})
- Peers working in parallel: {parent_context['siblings'] or 'none yet'}
- Depth: {parent_context['depth']}
- When you finish, send your spawner a message with results, then call set_status("done").
"""

        coordination_tools = {"spawn", "send", "wait", "query", "kill", "transfer", "set_bio"}
        has_coordination_tools = bool(coordination_tools - self.config.disabled_tools)
        if has_coordination_tools:
            meta_line = "- meta tools: coordination (spawn, send, wait, query, kill, transfer, etc.)"
        else:
            meta_line = "- meta tools: single-agent lifecycle and deliverables (get_cost, set_status, rebirth, submit, batch)"

        return f"""You are agent "{agent_id}" in a multi-agent system.

Task: {task}
Workspace: {workspace} (private to you)
Shared: {shared} (visible to all agents){time_info}
{context_section}
## Tool Philosophy
You have tools in 3 layers:
- shell: universal primitive. Use for mkdir, rm, mv, ls, find, tree, git, pip, curl, etc.
- ws_* tools: structured operations (file create/read/edit, grep, code outline)
{meta_line}

When in doubt, use shell. The ws_* tools exist only for operations shell can't do reliably.
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
        elapsed = time.time() - self._start_time
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

        # ─── Compile result ──────────────────────────────────────────────
        return {
            "overview": {
                "elapsed_seconds": round(elapsed, 1),
                "total_cost_usd": round(cost, 4),
                "total_tokens": total_tokens,
                "tokens_per_dollar": int(tokens_per_dollar),
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
                "query_count": tool_counts.get("query", 0),
                "wait_count": tool_counts.get("wait", 0),
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
            "last_tool_policy": copy.deepcopy(agent._last_tool_policy),
            "shell_activity": {
                "web_calls": agent._shell_activity.web_calls,
                "web_low_signal_calls": agent._shell_activity.web_low_signal_calls,
                "blocked_web_calls": agent._shell_activity.blocked_web_calls,
                "consecutive_web_low_signal": agent._shell_activity.consecutive_web_low_signal,
                "repeated_web_queries": agent._shell_activity.repeated_web_queries,
                "repeated_web_domains": agent._shell_activity.repeated_web_domains,
                "unique_web_signatures": sorted(agent._shell_activity.unique_web_signatures),
                "unique_web_domains": sorted(agent._shell_activity.unique_web_domains),
                "last_web_signature": agent._shell_activity.last_web_signature,
                "last_web_domain": agent._shell_activity.last_web_domain,
                "web_domain_sprawl": agent._shell_activity.web_domain_sprawl,
                "web_saturation": agent._shell_activity.web_saturation,
                "finalize_after_web_saturation": agent._shell_activity.finalize_after_web_saturation,
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
        agent._last_tool_policy = copy.deepcopy(data.get("last_tool_policy"))
        shell_activity = data.get("shell_activity") or {}
        agent._shell_activity = ShellActivityState(
            web_calls=int(shell_activity.get("web_calls", 0)),
            web_low_signal_calls=int(shell_activity.get("web_low_signal_calls", 0)),
            blocked_web_calls=int(shell_activity.get("blocked_web_calls", 0)),
            consecutive_web_low_signal=int(shell_activity.get("consecutive_web_low_signal", 0)),
            repeated_web_queries=int(shell_activity.get("repeated_web_queries", 0)),
            repeated_web_domains=int(shell_activity.get("repeated_web_domains", 0)),
            unique_web_signatures=set(shell_activity.get("unique_web_signatures", [])),
            unique_web_domains=set(shell_activity.get("unique_web_domains", [])),
            last_web_signature=str(shell_activity.get("last_web_signature", "")),
            last_web_domain=str(shell_activity.get("last_web_domain", "")),
            web_domain_sprawl=float(shell_activity.get("web_domain_sprawl", 0.0)),
            web_saturation=float(shell_activity.get("web_saturation", 0.0)),
            finalize_after_web_saturation=bool(shell_activity.get("finalize_after_web_saturation", False)),
        )
        return agent

    def _write_probe(
        self,
        agent: Agent,
        turn_tools: dict[str, dict[str, Any]],
        tool_policy: ToolPolicyState,
    ) -> Path:
        if not self.config.probe_dir:
            raise RuntimeError("probe_dir is required when probe_every_llm is enabled")
        self._probe_counter += 1
        probe_dir = self.config.probe_dir / f"{self._probe_counter:05d}_{agent.id}_turn{agent._turns}"
        snapshot_dir = probe_dir / "workspace_snapshot"
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        probe_dir.mkdir(parents=True, exist_ok=True)
        if self.config.workspace_root.exists():
            ignore = shutil.ignore_patterns(".probes", "probes")
            shutil.copytree(self.config.workspace_root, snapshot_dir, ignore=ignore)

        state = {
            "schema_version": 1,
            "created_at": time.time(),
            "active_agent": agent.id,
            "workspace_root": str(self.config.workspace_root),
            "probe_counter": self._probe_counter,
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
        }
        import pickle
        probe_path = probe_dir / "state.pkl"
        with probe_path.open("wb") as f:
            pickle.dump(state, f)

        (probe_dir / "meta.json").write_text(json.dumps({
            "schema_version": 1,
            "active_agent": agent.id,
            "probe_counter": self._probe_counter,
            "agent_turn": agent._turns,
            "turn_tools": list(turn_tools),
            "tool_policy": tool_policy.as_event(),
            "workspace_snapshot": str(snapshot_dir),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return probe_path

    def restore_probe(self, probe_path: Path) -> str:
        import pickle
        with probe_path.open("rb") as f:
            state = pickle.load(f)
        snapshot_dir = probe_path.parent / "workspace_snapshot"
        if snapshot_dir.exists():
            if self.config.workspace_root.exists():
                shutil.rmtree(self.config.workspace_root)
            shutil.copytree(snapshot_dir, self.config.workspace_root)
            self._tool_context.workspace_root = self.config.workspace_root
            self._tool_context.shared_dir = self.config.workspace_root / self.config.shared_dir

        self.ledger = copy.deepcopy(state["ledger"])
        self._messages_sent = copy.deepcopy(state.get("messages_sent", []))
        self._events = copy.deepcopy(state.get("events", []))
        self._id_gen._counter = int(state.get("id_counter", 0))
        self._probe_counter = int(state.get("probe_counter", 0))
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
        return str(state["active_agent"])

    async def continue_from_probe(self, probe_path: Path, agent_id: str | None = None) -> str:
        active = agent_id or self.restore_probe(probe_path)
        agent = self.agents[active]
        if self.config.max_turns > 0:
            agent.quota.max_turns = self.config.max_turns
        if self.config.time_limit > 0:
            agent.quota.time_limit = self.config.time_limit
        if self.config.probe_resume_instruction:
            agent.history.append({
                "role": "user",
                "content": (
                    "[Probe branch instruction]\n"
                    f"{self.config.probe_resume_instruction}"
                ),
            })
        agent.status = "running"
        self.start_agent(agent)
        await agent._task
        running = [
            a for a in self.agents.values()
            if a.status in ("running", "idle") and a.id != agent.id
        ]
        if running:
            await self.shutdown()
        return agent.result or ""
