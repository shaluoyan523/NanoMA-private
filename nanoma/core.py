"""Core runtime: Agent, Runtime, ReAct loop."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Awaitable, Literal

from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import (
    LLMResponse, Message, RetryConfig, ToolCall, ToolDef,
    count_message_tokens, estimate_tokens, openai_compatible_call, set_log_dir,
    _parse_text_tool_calls,
)
from nanoma.memory import ActiveTaskCard, ExperienceCard, MemoryBroker
from nanoma.sandbox import SandboxConfig, SandboxSession
from nanoma.scheduler import Scheduler
from nanoma.tools import WORK_TOOLS

logger = logging.getLogger("nanoma")

OrchestrationPreference = Literal["solo", "balanced", "parallel", "aggressive"]
ChildOrchestrationPreference = OrchestrationPreference | Literal["inherit_decayed", "task_adaptive"] | None
ShellMode = Literal["disabled", "controlled", "unrestricted"]
ORCHESTRATION_PREFERENCES = {"solo", "balanced", "parallel", "aggressive"}
ORCHESTRATION_ORDER: tuple[OrchestrationPreference, ...] = ("solo", "balanced", "parallel", "aggressive")
ORCHESTRATION_RANK: dict[OrchestrationPreference, int] = {pref: i for i, pref in enumerate(ORCHESTRATION_ORDER)}
ORCHESTRATION_DECAY: dict[OrchestrationPreference, OrchestrationPreference] = {
    "aggressive": "parallel",
    "parallel": "balanced",
    "balanced": "solo",
    "solo": "solo",
}
ORCHESTRATION_COMPLEXITY_TERMS: dict[str, int] = {
    "architecture": 2,
    "benchmark": 2,
    "compare": 1,
    "complex": 2,
    "concurrent": 2,
    "design": 1,
    "end-to-end": 2,
    "evaluate": 1,
    "full": 1,
    "full-scale": 2,
    "implementation": 1,
    "implement": 1,
    "integration": 2,
    "multi-agent": 2,
    "multiagent": 2,
    "parallel": 2,
    "review": 1,
    "synthesis": 1,
    "test matrix": 2,
    "workflow": 1,
    "全量": 2,
    "复杂": 2,
    "并发": 2,
    "并行": 2,
    "多agent": 2,
    "多智能体": 2,
    "基准": 2,
    "工作流": 1,
    "架构": 2,
    "测试矩阵": 2,
    "端到端": 2,
    "评估": 1,
    "集成": 2,
}
ORCHESTRATION_ATOMIC_TERMS: dict[str, int] = {
    "brief": 1,
    "grep": 2,
    "inspect": 1,
    "one file": 2,
    "quick": 1,
    "read one": 2,
    "single": 1,
    "smoke": 1,
    "small": 1,
    "一次": 2,
    "单个": 2,
    "单文件": 2,
    "只看": 1,
    "快速": 1,
    "简单": 1,
}


def _normalize_orchestration_preference(value: str | None, default: OrchestrationPreference = "balanced") -> OrchestrationPreference:
    if value in ORCHESTRATION_PREFERENCES:
        return value  # type: ignore[return-value]
    return default


def _decay_orchestration_preference(value: OrchestrationPreference) -> OrchestrationPreference:
    return ORCHESTRATION_DECAY[value]


def _preference_at_rank(rank: int) -> OrchestrationPreference:
    return ORCHESTRATION_ORDER[max(0, min(rank, len(ORCHESTRATION_ORDER) - 1))]


def _infer_task_orchestration_preference(
    *,
    parent_preference: OrchestrationPreference,
    task: str,
    role: str = "",
    create_type: str = "",
    relationship: str = "",
    group_id: str = "",
    workflow_prior: str = "",
    current_task_tags: list[str] | None = None,
) -> tuple[OrchestrationPreference, list[str]]:
    """Pick a child preference from assigned work instead of lineage depth."""
    tags = current_task_tags or []
    text = "\n".join(str(part) for part in [task, role, create_type, relationship, group_id, workflow_prior, " ".join(tags)] if part)
    normalized = text.lower()
    parent_rank = ORCHESTRATION_RANK[parent_preference]
    parallel_score = 0
    atomic_score = 0
    reasons: list[str] = []
    leaf_artifact_lane = _task_is_leaf_artifact_lane(
        task=task,
        role=role,
        group_id=group_id,
        current_task_tags=tags,
    )

    for term, weight in ORCHESTRATION_COMPLEXITY_TERMS.items():
        if term in normalized:
            parallel_score += weight
    for term, weight in ORCHESTRATION_ATOMIC_TERMS.items():
        if term in normalized:
            atomic_score += weight

    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", task))
    if bullet_count >= 3:
        parallel_score += 2
        reasons.append(f"{bullet_count} listed work items")
    elif bullet_count == 2:
        parallel_score += 1

    connector_count = len(re.findall(r"\b(?:and|plus|then)\b|以及|并且|同时|然后", normalized))
    if connector_count >= 3:
        parallel_score += 2
        reasons.append("multiple coordinated clauses")
    elif connector_count == 2:
        parallel_score += 1

    if workflow_prior:
        parallel_score += 1
        reasons.append(f"workflow_prior={workflow_prior}")
    if any(tag.startswith(("phase:", "feature:", "scope:full", "workflow:", "benchmark:")) for tag in tags):
        parallel_score += 1
        reasons.append("structured task tags")

    task_words = len(re.findall(r"\w+", task))
    if task and ((0 < task_words <= 8) or len(task.strip()) <= 24):
        atomic_score += 1
    if leaf_artifact_lane:
        atomic_score += 5
        reasons.append("leaf artifact lane")

    target_rank = parent_rank
    if parallel_score >= 7:
        target_rank = max(parent_rank, ORCHESTRATION_RANK["aggressive"])
    elif parallel_score >= 4:
        target_rank = max(parent_rank, ORCHESTRATION_RANK["parallel"])
    elif parallel_score >= 2:
        target_rank = max(parent_rank, ORCHESTRATION_RANK["balanced"])

    if atomic_score >= 3 and parallel_score < 4:
        target_rank = min(target_rank, ORCHESTRATION_RANK["solo"])
    elif atomic_score >= 2 and parallel_score < 3:
        target_rank = min(target_rank, ORCHESTRATION_RANK["balanced"])
    if leaf_artifact_lane:
        target_rank = min(target_rank, ORCHESTRATION_RANK["solo"])

    preference = _preference_at_rank(target_rank)
    if parallel_score:
        reasons.append(f"parallel_score={parallel_score}")
    if atomic_score:
        reasons.append(f"atomic_score={atomic_score}")
    if not reasons:
        reasons.append("no strong task signal; kept parent preference")
    if preference != parent_preference:
        reasons.append(f"resolved {parent_preference}->{preference}")
    else:
        reasons.append(f"resolved {preference}")
    return preference, reasons


def _expected_output_paths(task: str) -> list[str]:
    """Extract explicit shared output paths from task text."""
    paths: list[str] = []
    seen: set[str] = set()
    for path in _referenced_output_like_shared_paths(task):
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _looks_like_output_path(path: str) -> bool:
    if "<" in path or ">" in path or "*" in path:
        return False
    if path.startswith("shared/source/"):
        return False
    if re.fullmatch(r"shared/[^/]+/problems/[^/]+/(?:metadata\.json|statement\.md)", path):
        return False
    if re.fullmatch(r"shared/[^/]+/README\.md", path):
        return False
    if path.endswith("/"):
        return False
    suffix = Path(path).suffix.lower()
    return suffix in {".cpp", ".md", ".json", ".py", ".txt"}


def _referenced_output_like_shared_paths(task: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"`(shared/[^`]+?)`", task or ""):
        path = match.group(1).strip().rstrip(".,;:)]}")
        if _looks_like_output_path(path) and path not in seen:
            seen.add(path)
            paths.append(path)
    for match in re.finditer(r"(?<![\w/])shared/[^\s`'\"<>)\]}]+", task or ""):
        path = match.group(0).strip().rstrip(".,;:)]}")
        if _looks_like_output_path(path) and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _referenced_group_ids(task: str) -> list[str]:
    group_ids: list[str] = []
    seen: set[str] = set()
    patterns = (
        r"group_id\s*[=:]\s*`([^`]+)`",
        r"group_id\s*[=:]\s*['\"]([^'\"]+)['\"]",
        r"group_id\s*[=:]\s*([A-Za-z0-9_.:/@+-]+)",
        r"group_id\s+`([^`]+)`",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, task or "", flags=re.IGNORECASE):
            value = match.group(1).strip().rstrip(".,;)")
            if value and value not in seen:
                seen.add(value)
                group_ids.append(value)
    return group_ids


def _task_prefers_query_before_artifact(task: str) -> bool:
    text = (task or "").lower()
    if "query" in text or "group_id" in text:
        return True
    coordination_terms = (
        "coordinate", "coordinator", "peer", "peers", "summarize shared",
        "inspect shared", "discover", "artifact_commit", "remaining gaps",
        "协调", "汇总", "查询", "同伴", "其他agent",
    )
    return any(term in text for term in coordination_terms)


def _task_is_coordination_artifact_task(task: str) -> bool:
    if not _expected_output_paths(task) or not _task_prefers_query_before_artifact(task):
        return False
    text = (task or "").lower()
    coordination_terms = (
        "coordinate", "coordinator", "summarize", "summary", "report", "status",
        "inspect", "discover", "peer", "peers", "group_id", "remaining gaps",
        "artifact_commit", "file_write observations",
        "协调", "汇总", "总结", "报告", "状态", "查询", "同伴", "其他agent", "剩余",
    )
    return bool(_referenced_group_ids(task)) or any(term in text for term in coordination_terms)


def _task_has_spawnable_workstreams(task: str, min_streams: int = 3) -> bool:
    text = (task or "").lower()
    if _task_has_multi_phase_peer_protocol(task):
        return True
    if any(term in text for term in ("separable", "workstream", "workstreams", "peer agents", "multi-agent", "multiagent", "多agent")):
        return True
    stream_terms = {
        "audit", "patch", "implement", "implementation", "test", "review", "research",
        "generator", "judge", "coordinator", "scout", "integrator", "redteam",
        "评审", "测试", "实现", "审计", "生成", "汇总",
    }
    hits = sum(1 for term in stream_terms if term in text)
    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", task or ""))
    return hits >= max(2, min_streams) or bullet_count >= max(3, min_streams)


def _task_has_multi_phase_peer_protocol(task: str) -> bool:
    text = (task or "").lower()
    terms = (
        "opendeepthink",
        "open deepthink",
        "peer wave",
        "pairwise",
        "comparison",
        "comparisons",
        "mutation",
        "mutate",
        "generation",
        "generations",
        "round",
        "rounds",
        "protocol",
        "spawn comparison",
        "spawn/run comparison",
        "n=20",
        "k=4",
        "m=10",
        "t=3",
        "固定编排",
        "多阶段",
        "比较",
        "变异",
        "轮次",
    )
    return any(term in text for term in terms)


def _task_is_leaf_artifact_lane(
    *,
    task: str,
    role: str = "",
    group_id: str = "",
    current_task_tags: list[str] | None = None,
) -> bool:
    text = (task or "").lower()
    tags = [str(tag).lower() for tag in (current_task_tags or [])]
    context = " ".join([str(role).lower(), str(group_id).lower(), " ".join(tags)])
    if not text:
        return False
    write_terms = (
        "write", "create", "produce", "deliver", "generate", "implement",
        "输出", "产出", "写", "生成", "实现",
    )
    if not any(term in text for term in write_terms):
        return False
    coordinator_terms = (
        "spawn", "spawn_many", "create_agent", "coordinate", "coordinator",
        "query peers", "query group", "compare", "comparison", "pairwise",
        "judge", "aggregate", "bt_aggregate", "synthesis", "synthesize",
        "mutation", "mutate", "select final", "selection", "opendeepthink",
        "n=20", "k=4", "m=10", "t=3", "多阶段", "比较", "变异", "汇总",
    )
    if any(term in text for term in coordinator_terms):
        return False
    paths = _referenced_output_like_shared_paths(task)
    candidate_hint = bool(re.search(r"\bcandidate[_:-]?\d+\b", text)) or any(tag.startswith("candidate:") for tag in tags)
    role_hint = any(term in context for term in ("generator", "solver", "candidate", "role:generator", "role:solver"))
    if not (paths or candidate_hint):
        return False
    if not (role_hint or candidate_hint):
        return False
    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", task or ""))
    if bullet_count >= 4 and not candidate_hint:
        return False
    return len(paths) <= 3


def _task_requires_source_change(task: str) -> bool:
    text = (task or "").lower()
    if "shared/source" not in text:
        return False
    change_terms = (
        "patch", "implement", "implementation", "modify", "edit", "change", "fix",
        "source/test change", "diff", "add focused tests", "add tests", "新增", "修改", "修复",
    )
    return any(term in text for term in change_terms)


def _task_requires_test_run(task: str) -> bool:
    text = (task or "").lower()
    if "shared/source" not in text:
        return False
    test_terms = (
        "run the relevant tests", "run tests", "tests passed", "pytest",
        "do not claim tests passed", "测试通过", "运行测试",
    )
    return any(term in text for term in test_terms)


def _looks_like_final_report_path(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() not in {".md", ".txt"}:
        return False
    name = p.name.lower()
    return any(term in name for term in ("final", "report", "summary", "总结", "报告"))


def _looks_like_test_command(command: str) -> bool:
    lowered = (command or "").lower()
    return "pytest" in lowered or "unittest" in lowered


_TEXT_TOOL_ALIASES: dict[str, str] = {
    "read": "file_read",
    "read_file": "file_read",
    "list": "file_list",
    "list_files": "file_list",
    "write": "file_write",
    "write_file": "file_write",
    "replace": "file_replace",
    "replace_file": "file_replace",
    "create_agents": "spawn_many",
    "spawn_agents": "spawn_many",
    "spawn_batch": "spawn_many",
    "spaw_many": "spawn_many",
}

_LOOP_ACTIONS = {"create", "read", "message", "work", "compact", "stop"}


def _recent_tool_names(history: list[Message], lookback: int = 8) -> set[str]:
    names: set[str] = set()
    for msg in history[-lookback:]:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name"):
                names.add(str(fn["name"]))
    return names


_FENCED_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+.#-]*)\s*\n(.*?)```", re.DOTALL)
_LANGUAGE_SUFFIXES: dict[str, set[str]] = {
    ".cpp": {"cpp", "c++", "cxx", "cc"},
    ".py": {"py", "python"},
    ".json": {"json"},
    ".md": {"md", "markdown"},
    ".txt": {"text", "txt"},
}


def _extract_artifact_content_from_text(content: str, path: str) -> str | None:
    if "<｜DSML｜invoke" in (content or ""):
        return None
    suffix = Path(path).suffix.lower()
    blocks = [(lang.strip().lower(), body.strip()) for lang, body in _FENCED_BLOCK_RE.findall(content or "")]
    accepted_langs = _LANGUAGE_SUFFIXES.get(suffix, set())
    whole_content_is_process_chatter = suffix in {".md", ".txt"} and _looks_like_process_chatter(content)
    if blocks:
        for lang, body in blocks:
            if not body:
                continue
            if suffix in {".md", ".txt"} and whole_content_is_process_chatter:
                continue
            if lang in accepted_langs:
                if suffix in {".md", ".txt"} and not _looks_like_report_text(body):
                    continue
                return body + "\n"
        if suffix in {".md", ".txt"} and not whole_content_is_process_chatter and _looks_like_report_text(blocks[-1][1]):
            return blocks[-1][1] + "\n"
    if suffix in {".md", ".txt"} and content and content.strip() and _looks_like_report_text(content):
        return content.strip() + "\n"
    return None


def _looks_like_process_chatter(content: str) -> bool:
    text = " ".join((content or "").strip().lower().split())
    if not text:
        return True
    process_prefixes = (
        "i will ",
        "i'll ",
        "let me ",
        "i need to ",
        "i should ",
        "first, i ",
        "now i ",
    )
    process_terms = ("think", "analyze", "outline", "read", "inspect", "understand")
    if text.startswith(process_prefixes) and any(term in text[:160] for term in process_terms):
        return True
    if len(text) < 40 and any(term in text for term in ("not sure", "no code", "todo", "later")):
        return True
    return False


def _looks_like_report_text(content: str) -> bool:
    if _looks_like_process_chatter(content):
        return False
    stripped = (content or "").strip()
    lowered = stripped.lower()
    if "<｜" in stripped or "</think>" in stripped:
        return False
    task_echo_markers = (
        "your task:",
        "the user's task is:",
        "you are in a multi-agent system",
        "important: to avoid catastrophic race conditions",
        "do not provide other parameters",
    )
    if any(marker in lowered[:1000] for marker in task_echo_markers):
        return False
    first_meaningful = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    first_lower = first_meaningful.lower()
    if first_meaningful and first_meaningful.endswith((".", ":")) and first_meaningful.lower().startswith((
        "let me ",
        "i will ",
        "i'll ",
        "i need to ",
        "i should ",
        "now i ",
        "first i ",
        "first, i ",
        "the user's task is",
        "your task",
    )):
        return False
    if len(stripped) < 80:
        return lowered.startswith(("summary", "report", "status", "result", "candidate", "core idea", "complexity", "risks"))
    report_markers = (
        "summary",
        "report",
        "status",
        "result",
        "candidate",
        "core idea",
        "complexity",
        "risk",
        "artifact",
        "complete",
        "incomplete",
    )
    marker_hits = sum(1 for marker in report_markers if marker in lowered[:1200])
    structural_hits = sum(1 for marker in ("# ", "## ", "- ", "|", "```") if marker in stripped[:1600])
    heading_like = first_lower.startswith(("summary", "report", "status", "result", "candidate", "core idea", "complexity", "risks"))
    return marker_hits >= 2 and (structural_hits >= 1 or heading_like)

# ─── ID Generation (NATO phonetic) ──────────────────────────────────────────

_NATO = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
]


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
    message_type: str = "text"
    payload: dict[str, Any] | None = None
    requires_ack: bool = False
    urgency: str = "normal"
    priority: int = 0
    mode: Literal["immediate", "steer", "queue"] = "queue"
    transient: bool = False


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
    sandbox: SandboxSession | None = None
    enabled_work_tools: set[str] | None = None
    shell_mode: ShellMode = "controlled"
    controlled_shell_allowed_commands: set[str] | None = None
    shell_max_output: int = 10000
    file_read_max_chars: int = 50000
    file_list_max_entries: int = 500
    grep_max_results: int = 100


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    max_agents: int = 1000
    max_depth: int = 100
    max_concurrent_llm: int = 50
    budget: float = 10.0
    time_limit: float = 0.0
    max_turns: int = 200
    allowed_models: list[str] | None = None
    context_compress_ratio: float = 0.8
    default_model: str = "deepseek-v4-flash"
    log_dir: Path | None = field(default_factory=lambda: Path("./logs"))
    workspace_root: Path = field(default_factory=lambda: Path("./workspace"))
    shared_dir: str = "shared"
    sandbox_backend: str = "codex"          # codex, host
    sandbox_codex_bin: str = "codex"
    sandbox_network: bool = False
    retry: RetryConfig = field(default_factory=RetryConfig)
    # Resource notification thresholds (fraction consumed, e.g. 0.5 = 50%)
    notify_thresholds: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.70, 0.80, 0.90, 0.95])
    notify_parent_on_done: bool = False   # False favors query-driven summaries over parent-report fan-in
    # Compression / truncation settings
    compress_keep_recent: int = 6           # messages to keep verbatim during compression
    compress_max_messages: int = 40         # max old messages to include in summary
    compress_max_chars: int = 300           # max chars per message in summary (0 = unlimited)
    shell_max_output: int = 10000           # max chars for shell output (0 = unlimited)
    file_read_max_chars: int = 50000        # max chars for file_read (0 = unlimited)
    file_list_max_entries: int = 500        # max entries for file_list (0 = unlimited)
    grep_max_results: int = 100             # max grep results (0 = unlimited)
    artifact_write_max_tokens: int = 12000  # response budget for artifact-only file_write turns
    enabled_work_tools: set[str] | None = None  # None = all work tools enabled
    shell_mode: ShellMode = "controlled"        # disabled, controlled, unrestricted
    controlled_shell_allowed_commands: set[str] | None = None
    bootstrap_batch_file: str | None = None  # Optional JSON batch executed by root before its first LLM turn.
    orchestration_preference: OrchestrationPreference = "balanced"  # solo, balanced, parallel, aggressive
    child_orchestration_preference: ChildOrchestrationPreference = None  # None/task_adaptive lets children choose from assigned work
    min_spawnable_workstreams: int = 3
    spawn_before_turn: int = 2
    max_solo_tool_calls_before_spawn: int = 5
    peer_progress_check_after_turns: int = 3
    peer_progress_check_no_tool_turns: int = 2
    peer_progress_check_after_artifact_nudges: int = 2


# ─── Agent ───────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: str
    task: str
    model: str

    # Identity
    bio: str = ""  # mutable self-description, visible to all via query()
    role: str = ""
    create_type: str = ""
    relationship: str = ""
    created_by: str | None = None
    group_id: str = ""
    workflow_prior: str = ""
    orchestration_preference: OrchestrationPreference = "balanced"

    # State
    status: Literal["running", "idle", "done", "failed"] = "running"
    action_state: Literal["create", "stop", "read", "message", "work", "compact"] = "create"
    current_task_tags: list[str] = field(default_factory=list)
    work_outline: str = ""
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
    _compact_pending: dict | None = field(default=None, repr=False)
    _notified_thresholds: set = field(default_factory=set)  # resource thresholds already fired
    _orchestration_nudge_sent: bool = False
    _orchestration_nudge_turn: int = 0
    _artifact_nudge_count: int = 0
    _peer_progress_nudge_sent: bool = False
    _peer_progress_nudge_turn: int = 0
    _last_query_turn: int = 0
    _last_artifact_turn: int = 0
    _no_tool_turns: int = 0
    _tool_calls: int = 0
    _shell_commands: list[str] = field(default_factory=list)
    _successful_test_commands: list[str] = field(default_factory=list)
    _filtered_read_intent_count: int = 0
    _filtered_read_intent_seen_turn: int = 0
    _write_only_miss_count: int = 0
    _artifact_deferred_nudge_sent: bool = False
    _create_action_filtered_count: int = 0
    _create_resume_after_read: bool = False
    _loop_action_plan: dict[str, Any] | None = None
    _state_action_version: int = 0
    _state_action_consumed_version: int = 0
    memory: dict[str, Any] = field(default_factory=dict)


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
            enabled_work_tools=self.config.enabled_work_tools,
            shell_mode=self.config.shell_mode,
            controlled_shell_allowed_commands=self.config.controlled_shell_allowed_commands,
            shell_max_output=self.config.shell_max_output,
            file_read_max_chars=self.config.file_read_max_chars,
            file_list_max_entries=self.config.file_list_max_entries,
            grep_max_results=self.config.grep_max_results,
        )
        self.llm_call = llm_call or openai_compatible_call
        self.router = router
        self.scheduler = Scheduler(max_concurrent=self.config.max_concurrent_llm)
        self.sandbox = SandboxSession(
            SandboxConfig(
                backend=self.config.sandbox_backend,
                codex_bin=self.config.sandbox_codex_bin,
                network=self.config.sandbox_network,
            ),
            self.config.workspace_root,
        )
        self._tool_context.sandbox = self.sandbox
        self.on_event = on_event or (lambda e: None)
        self._start_time = time.time()
        self.memory = MemoryBroker()
        self.state_board: dict[str, dict[str, Any]] = {}
        self._events: list[dict] = []  # all events for post-hoc analysis
        self._messages_sent: list[tuple[str, str, int]] = []  # (from, to, tokens) for comm graph
        self._last_llm_messages: dict[str, list[Message]] = {}
        self._emit_lock = threading.Lock()  # protects events.jsonl writes

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
        role: str = "",
        create_type: str = "",
        relationship: str = "",
        created_by: str | None = None,
        group_id: str = "",
        workflow_prior: str = "",
        current_task_tags: list[str] | None = None,
        orchestration_preference: str | None = None,
    ) -> Agent:
        agent_id = self._id_gen.next()
        model = model or self.config.default_model
        agent_orchestration_preference, orchestration_resolution = self._resolve_agent_orchestration_preference(
            parent=parent,
            requested=orchestration_preference,
            task=task,
            role=role,
            create_type=create_type,
            relationship=relationship,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags or [],
        )
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

        system_prompt = self._build_system_prompt(
            agent_id,
            task,
            workspace,
            parent_context,
            orchestration_preference=agent_orchestration_preference,
        )

        agent = Agent(
            id=agent_id, task=task, model=model, quota=quota,
            parent=parent, depth=depth, workspace=workspace,
            role=role,
            create_type=create_type,
            relationship=relationship,
            created_by=created_by,
            group_id=group_id,
            workflow_prior=workflow_prior,
            orchestration_preference=agent_orchestration_preference,
            current_task_tags=list(current_task_tags or []),
            history=[{"role": "system", "content": system_prompt}],
        )

        # Context limit from model registry
        try:
            from nanoma.models import get_registry
            m = get_registry().get(model)
            if m:
                agent.context_limit = m.context_limit
        except Exception:
            pass

        self.agents[agent_id] = agent
        agent._created_at = time.time()
        if parent and parent in self.agents:
            self.agents[parent].children.add(agent_id)
        self.memory.init_agent(agent_id, task=task, tags=agent.current_task_tags)
        agent.memory = {"public_memory": self.memory.serialize(agent_id)}
        self.state_board_sync(agent_id)

        self._emit(agent_id, "agent_new", {
            "task": task, "model": model, "budget": quota.budget if math.isfinite(quota.budget) else None,
            "parent": parent, "depth": depth,
            "role": role,
            "create_type": create_type,
            "relationship": relationship,
            "created_by": created_by,
            "group_id": group_id,
            "workflow_prior": workflow_prior,
            "orchestration_preference": agent_orchestration_preference,
            "orchestration_resolution": orchestration_resolution,
        })
        return agent

    def start_agent(self, agent: Agent):
        agent._task = asyncio.ensure_future(self._agent_loop(agent))

    def _resolve_agent_orchestration_preference(
        self,
        *,
        parent: str | None,
        requested: str | None,
        task: str = "",
        role: str = "",
        create_type: str = "",
        relationship: str = "",
        group_id: str = "",
        workflow_prior: str = "",
        current_task_tags: list[str] | None = None,
    ) -> tuple[OrchestrationPreference, dict[str, Any]]:
        if requested in ORCHESTRATION_PREFERENCES:
            preference = requested  # type: ignore[assignment]
            if (
                preference != "solo"
                and _task_is_leaf_artifact_lane(
                    task=task,
                    role=role,
                    group_id=group_id,
                    current_task_tags=current_task_tags or [],
                )
            ):
                return "solo", {
                    "mode": "explicit_spawn_leaf_artifact_override",
                    "requested": requested,
                    "resolved": "solo",
                    "reasons": ["leaf artifact lane should complete assigned output instead of expanding"],
                }
            return preference, {"mode": "explicit_spawn", "requested": requested, "resolved": preference}
        if parent is None:
            preference = _normalize_orchestration_preference(str(self.config.orchestration_preference), "balanced")
            return preference, {"mode": "root_default", "resolved": preference}

        child_pref = self.config.child_orchestration_preference
        if child_pref in ORCHESTRATION_PREFERENCES:
            preference = child_pref  # type: ignore[assignment]
            return preference, {"mode": "runtime_child_override", "configured": child_pref, "resolved": preference}

        parent_pref = self.agents[parent].orchestration_preference if parent in self.agents else self.config.orchestration_preference
        if child_pref == "inherit_decayed":
            preference = _decay_orchestration_preference(parent_pref)
            return preference, {
                "mode": "inherit_decayed",
                "parent": parent,
                "parent_preference": parent_pref,
                "resolved": preference,
            }

        preference, reasons = _infer_task_orchestration_preference(
            parent_preference=parent_pref,
            task=task,
            role=role,
            create_type=create_type,
            relationship=relationship,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags or [],
        )
        return preference, {
            "mode": "task_adaptive",
            "parent": parent,
            "parent_preference": parent_pref,
            "resolved": preference,
            "reasons": reasons,
        }

    def state_board_sync(self, agent_id: str) -> dict[str, Any]:
        agent = self.agents[agent_id]
        board = {
            "action_state": agent.action_state,
            "status": agent.status,
            "current_task_tags": list(agent.current_task_tags),
            "work_outline": agent.work_outline,
        }
        self.state_board[agent_id] = board
        return board

    def state_board_get(self, agent_id: str) -> dict[str, Any]:
        return self.state_board_sync(agent_id)

    def state_board_list(self) -> dict[str, dict[str, Any]]:
        for agent_id in list(self.agents):
            self.state_board_sync(agent_id)
        return {agent_id: dict(board) for agent_id, board in sorted(self.state_board.items())}

    def state_board_update(
        self,
        agent_id: str,
        *,
        action_state: str | None = None,
        current_task_tags: list[str] | None = None,
        work_outline: str | None = None,
    ) -> dict[str, Any]:
        agent = self.agents[agent_id]
        if action_state is not None:
            agent.action_state = action_state
            agent._state_action_version += 1
        if current_task_tags is not None:
            agent.current_task_tags = list(current_task_tags)
        if work_outline is not None:
            agent.work_outline = work_outline
        memory = self.memory.get(agent_id)
        task_text = memory.active_task.task if memory and memory.active_task else agent.task
        self.memory.update(
            agent_id,
            active_task=ActiveTaskCard(task=task_text, tags=list(agent.current_task_tags), work_outline=agent.work_outline),
            tags=agent.current_task_tags,
        )
        agent.memory = {"public_memory": self.memory.serialize(agent_id)}
        return self.state_board_sync(agent_id)


    async def run(self, task: str, model: str | None = None) -> str:
        """Run a root agent, letting already-spawned peers finish before shutdown."""
        await self.sandbox.start()
        try:
            self._emit("system", "sandbox_start", {
                "backend": self.sandbox.backend,
                "workspace_root": str(self.config.workspace_root),
            })
            root = self.create_agent(task, model=model)
            if self.config.bootstrap_batch_file:
                await self._run_bootstrap_batch(root)
            self.start_agent(root)
            await root._task
            await self._wait_for_remaining_agents(root)
            return root.result or ""
        finally:
            await self.sandbox.stop()

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
        work_tools = self._available_work_tools()
        all_tools = {**work_tools, **META_TOOLS}
        tool_schemas = [t["schema"] for t in all_tools.values()]

        try:
            while agent.status == "running":
                agent._turns += 1
                agent._last_active = time.time()

                if agent._compact_pending:
                    self._execute_compact(agent)
                    if agent.status != "running":
                        break

                if agent._rebirth_pending:
                    self._execute_rebirth(agent)

                # Turn limits
                if agent._turns > agent.quota.max_turns:
                    if self._auto_complete_if_outputs_exist(agent, reason="max_turns"):
                        break
                    blockers = self.completion_blockers(agent)
                    if blockers:
                        agent.status = "failed"
                        agent.result = agent.result or "[Max turns reached with unresolved completion blockers]"
                        self._emit(agent.id, "completion_blocked", {
                            "reason": "max_turns",
                            "blockers": blockers,
                        })
                    else:
                        agent.status = "done"
                        agent.result = agent.result or "[Max turns reached]"
                    break

                # Time limit
                if agent.quota.time_limit > 0:
                    elapsed = time.time() - self._start_time
                    if elapsed >= agent.quota.time_limit:
                        blockers = self.completion_blockers(agent)
                        if blockers:
                            agent.status = "failed"
                            agent.result = agent.result or f"[Time limit at {elapsed:.0f}s with unresolved completion blockers]"
                            self._emit(agent.id, "completion_blocked", {
                                "reason": "time_limit",
                                "blockers": blockers,
                            })
                        else:
                            agent.status = "done"
                            agent.result = agent.result or f"[Time limit at {elapsed:.0f}s]"
                        break

                # Budget enforcement — global budget check
                if self.ledger.remaining() <= 0:
                    agent.status = "failed"
                    agent.result = agent.result or "[GLOBAL BUDGET EXHAUSTED]"
                    break

                # Time-based budget drain — DISABLED
                # (Previously drained budget over wall-clock time, but this punishes
                # agents for legitimately waiting on dependencies)

                # Rebirth
                if agent._rebirth_pending:
                    self._execute_rebirth(agent)

                # Runtime loop guidance is collected as transient context for this turn.
                # It should influence the next action without becoming permanent agent memory.
                self._check_thresholds(agent)
                self._check_orchestration_nudge(agent)
                self._check_artifact_nudge(agent)
                self._check_peer_progress_nudge(agent)

                # Inject queued messages
                self._inject_messages(agent, agent._queue_inbox)
                transient_messages = self._inject_messages(agent, agent._steer_inbox, persist_transient=False)

                # Context compression
                agent.context_tokens = count_message_tokens(agent.history)
                if agent.context_tokens > int(agent.context_limit * self.config.context_compress_ratio):
                    agent.history = await self._compress(agent.history)
                    agent.context_tokens = count_message_tokens(agent.history)

                missing_outputs = self._missing_expected_outputs(agent)
                self._refresh_loop_action_plan(agent, missing_outputs)
                loop_context = self._build_loop_action_context(agent, missing_outputs, transient_messages)
                loop_action = str((agent._loop_action_plan or {}).get("action", ""))
                children_before_turn = len(agent.children)
                create_miss_before_turn = agent._create_action_filtered_count
                peer_pre_query_pending = self._peer_progress_decision_pending(agent)
                peer_post_query_pending = self._peer_progress_post_query_pending(agent)
                turn_tool_scope = "all"
                if loop_action:
                    loop_tool_names, turn_tool_scope = self._tools_for_loop_turn(
                        agent,
                        loop_action,
                        missing_outputs,
                    )
                    turn_tool_schemas = self._artifact_scoped_tool_schemas(
                        missing_outputs,
                        sorted(loop_tool_names),
                    )
                else:
                    turn_tool_schemas = tool_schemas
                recommended_tool_choice: str | None = None
                if turn_tool_scope in {"retry_file_write_after_text_miss", "artifact_write_only"} and any(
                    (tool.get("function") or {}).get("name") == "file_write" for tool in turn_tool_schemas
                ):
                    recommended_tool_choice = "file_write"
                elif turn_tool_scope == "retry_spawn_after_create_miss" and any(
                    (tool.get("function") or {}).get("name") == "spawn_many" for tool in turn_tool_schemas
                ):
                    recommended_tool_choice = "spawn_many"
                turn_max_tokens = self._llm_max_tokens_for_turn(
                    loop_action=loop_action,
                    recommended_tool_choice=recommended_tool_choice,
                    missing_outputs=missing_outputs,
                )
                turn_history = agent.history + ([loop_context] if loop_context else [])
                self._last_llm_messages[agent.id] = [dict(m) for m in turn_history]
                await self.scheduler.acquire()
                try:
                    response = await self.llm_call(
                        turn_history, agent.model, turn_tool_schemas,
                        retry_config=self.config.retry,
                        max_tokens=turn_max_tokens,
                        tool_choice=(
                            {
                                "type": "function",
                                "function": {"name": recommended_tool_choice},
                            }
                            if recommended_tool_choice
                            else None
                        ),
                    )
                finally:
                    self.scheduler.release()
                if response is None:
                    agent.status = "failed"
                    agent.result = "[LLM returned no response]"
                    break

                # Record usage
                cost = self.ledger.record(agent.id, response.usage)
                agent.tokens_consumed += response.usage.total_tokens
                agent.quota.budget -= cost
                self._emit(agent.id, "llm_done", {
                    "tokens": response.usage.total_tokens,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cached": response.usage.cached_input_tokens,
                    "cost": round(cost, 6),
                    "model": agent.model,
                    "tool_calls": [tc.name for tc in response.tool_calls] if response.tool_calls else [],
                    "tool_choice": recommended_tool_choice or "auto",
                    "recommended_tool_choice": recommended_tool_choice,
                    "turn_max_tokens": turn_max_tokens,
                    "loop_action": loop_action or None,
                    "tool_scope": turn_tool_scope if loop_action else None,
                    "peer_progress_check": agent._peer_progress_nudge_sent,
                    "missing_outputs": missing_outputs[:10] if loop_action else [],
                    "has_content": bool(response.content),
                    "content_preview": (response.content or "")[:200],
                })
                if not response.tool_calls and response.content:
                    recovered_tool_calls = self._recover_text_tool_calls(response.content, turn_tool_schemas)
                    if recovered_tool_calls:
                        response.tool_calls = recovered_tool_calls
                        self._emit(agent.id, "text_tool_call_recovered", {
                            "tool_calls": [tc.name for tc in recovered_tool_calls],
                            "count": len(recovered_tool_calls),
                        })
                self._emit_filtered_text_tool_calls(agent, response.content or "", response.tool_calls, turn_tool_schemas)

                # Process response
                committed_from_text: list[str] = []
                executed_tool_calls: list[ToolCall] = []
                skipped_tool_calls: list[ToolCall] = []
                text_only_write_scope = (
                    not response.tool_calls
                    and response.content
                    and loop_action == "work"
                    and bool(missing_outputs)
                )
                write_only_content_note = (
                    "[write_only turn produced assistant text but no file_write tool call; "
                    "full text omitted from persistent history]"
                    if text_only_write_scope
                    else None
                )

                if response.tool_calls:
                    executed_tool_calls, skipped_tool_calls = self._partition_tool_calls_by_action(
                        agent,
                        response.tool_calls,
                    )
                    if executed_tool_calls:
                        # Append assistant message with executed tool calls only.
                        agent.history.append({
                            "role": "assistant", "content": response.content,
                            "tool_calls": [
                                {"id": tc.id, "type": "function",
                                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                for tc in executed_tool_calls
                            ],
                        })
                    elif write_only_content_note:
                        agent.history.append({"role": "assistant", "content": write_only_content_note})
                    elif response.content:
                        agent.history.append({"role": "assistant", "content": response.content})
                    else:
                        agent.history.append({"role": "assistant", "content": "(tool calls skipped by loop action)"})
                    # Execute tools
                    for i, tc in enumerate(executed_tool_calls):
                        # Immediate interrupt check
                        if not agent._immediate_inbox.empty():
                            msgs = self._inject_messages(agent, agent._immediate_inbox)
                            for skipped in executed_tool_calls[i:]:
                                agent.history.append({
                                    "role": "tool", "tool_call_id": skipped.id,
                                    "content": json.dumps({"interrupted": True}),
                                })
                            break

                        result = await self._execute_tool(tc, agent, all_tools)
                        if tc.name == "file_write":
                            result = self._annotate_file_write_result(agent, tc, result)
                            self._register_expected_artifact_write(agent, tc, result)
                        if tc.name == "shell":
                            command = str(tc.arguments.get("command", ""))
                            agent._shell_commands.append(command)
                            exit_code = result.get("returncode", result.get("exit_code", 1)) if isinstance(result, dict) else 1
                            if (
                                _looks_like_test_command(command)
                                and isinstance(result, dict)
                                and int(exit_code) == 0
                            ):
                                agent._successful_test_commands.append(command)
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
                        agent._tool_calls += 1
                        if tc.name == "query":
                            agent._last_query_turn = agent._turns
                            agent._filtered_read_intent_count = 0
                            agent._filtered_read_intent_seen_turn = 0
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") == "create":
                                agent._create_action_filtered_count = 0
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") in {"read", "query"}:
                                agent._loop_action_plan = None
                        if tc.name in {"spawn", "create_agent", "spawn_many"}:
                            created_count = len(agent.children) - children_before_turn
                            valid_spawn = created_count > 0
                            if isinstance(result, dict):
                                if tc.name == "spawn_many":
                                    valid_spawn = int(result.get("created") or 0) > 0
                                else:
                                    valid_spawn = bool(result.get("agent_id"))
                            if valid_spawn:
                                agent._create_action_filtered_count = 0
                                agent._create_resume_after_read = False
                                if agent._loop_action_plan and agent._loop_action_plan.get("action") == "create":
                                    agent._loop_action_plan = None
                            elif agent._loop_action_plan and agent._loop_action_plan.get("action") == "create":
                                agent._create_action_filtered_count += 1
                                self._emit(agent.id, "create_action_miss", {
                                    "count": agent._create_action_filtered_count,
                                    "reason": "spawn_tool_created_no_valid_children",
                                    "tool": tc.name,
                                    "result": result,
                                })
                        if tc.name in {"file_read", "file_list", "grep"}:
                            agent._filtered_read_intent_count = 0
                            agent._filtered_read_intent_seen_turn = 0
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") in {"read", "query"}:
                                agent._loop_action_plan = None
                        if tc.name in {"compact", "set_status"}:
                            if (
                                tc.name == "set_status"
                                and str(tc.arguments.get("action", "")) == "read"
                                and agent._loop_action_plan
                                and agent._loop_action_plan.get("action") == "create"
                                and _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
                            ):
                                agent._create_resume_after_read = True
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") in {"compact", "stop"}:
                                agent._loop_action_plan = None
                        if tc.name in {"file_write", "file_replace", "submit"}:
                            agent._filtered_read_intent_count = 0
                            agent._filtered_read_intent_seen_turn = 0
                            agent._last_artifact_turn = agent._turns
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") == "work":
                                agent._loop_action_plan = None
                            if not self._missing_expected_outputs(agent):
                                agent._artifact_nudge_count = 0

                    for skipped in skipped_tool_calls:
                        self._emit(agent.id, "tool_call_skipped", {
                            "tool": skipped.name,
                            "args": skipped.arguments,
                            "reason": "different_loop_action_requires_revalidation_next_turn",
                        })

                    committed_from_text = self._attempt_artifact_commit_from_text(agent, response.content or "")

                    # Steer messages before next LLM call
                    self._inject_messages(agent, agent._steer_inbox, persist_transient=False)

                elif response.content:
                    if write_only_content_note:
                        agent.history.append({"role": "assistant", "content": write_only_content_note})
                    else:
                        agent.history.append({"role": "assistant", "content": response.content})
                    committed_from_text = self._attempt_artifact_commit_from_text(agent, response.content)
                else:
                    agent.history.append({"role": "assistant", "content": "(empty)"})

                if (
                    loop_action == "create"
                    and agent.status == "running"
                    and len(agent.children) <= children_before_turn
                    and agent._create_action_filtered_count == create_miss_before_turn
                ):
                    explicit_next_action = any(
                        tc.name == "set_status"
                        and str(tc.arguments.get("action", "")) in {"read", "message", "work", "compact", "stop", "done", "self_stop"}
                        for tc in executed_tool_calls
                    )
                    if not explicit_next_action:
                        agent._create_action_filtered_count += 1
                        self._emit(agent.id, "create_action_miss", {
                            "count": agent._create_action_filtered_count,
                            "reason": "no_child_created_in_create_action",
                            "executed_tools": [tc.name for tc in executed_tool_calls],
                        })

                if committed_from_text:
                    agent._last_artifact_turn = agent._turns
                    if agent._loop_action_plan and agent._loop_action_plan.get("action") == "work":
                        if not self._missing_expected_outputs(agent):
                            agent._loop_action_plan = None

                if text_only_write_scope and not committed_from_text:
                    agent._write_only_miss_count += 1
                    self._emit(agent.id, "write_only_miss", {
                        "count": agent._write_only_miss_count,
                        "missing": missing_outputs[:10],
                        "content_preview": (response.content or "")[:200],
                    })
                else:
                    agent._write_only_miss_count = 0

                agent._no_tool_turns = 0 if response.tool_calls or committed_from_text else agent._no_tool_turns + 1

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
            # Only emit lifecycle completion for agents that actually finished (not just idle).
            if agent.status in ("done", "failed"):
                self.state_board_sync(agent.id)
                if self.config.notify_parent_on_done and agent.parent and agent.parent in self.agents:
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
                transient=True,
            )
            agent._steer_inbox.put_nowait(envelope)
            self._emit(agent.id, "send", {
                "from": "system", "to": agent.id,
                "mode": "steer", "tokens": envelope.tokens,
                "message": notice,
            })

    def _check_orchestration_nudge(self, agent: Agent):
        """Nudge non-solo agents once if they are still working alone past policy thresholds."""
        if agent._orchestration_nudge_sent or agent.children:
            return
        if agent.orchestration_preference == "solo":
            return
        if len(self.agents) >= self.config.max_agents:
            return

        spawnable = _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
        turn_trigger = agent._turns >= max(1, self.config.spawn_before_turn)
        tool_trigger = agent._tool_calls >= max(1, self.config.max_solo_tool_calls_before_spawn)
        if agent.orchestration_preference == "balanced":
            should_nudge = turn_trigger and (tool_trigger or spawnable)
        else:
            should_nudge = turn_trigger or tool_trigger
        if not should_nudge:
            return

        agent._orchestration_nudge_sent = True
        agent._orchestration_nudge_turn = agent._turns
        notice = (
            "[Orchestration Preference]\n"
            f"Preference: {agent.orchestration_preference}. You are still running without child agents "
            f"at turn {agent._turns} after {agent._tool_calls} tool calls.\n"
            f"If the task has at least {self.config.min_spawnable_workstreams} separable workstreams, "
            "choose action=create. Use query/read/work/compact/stop only when that action better matches the current state. "
            "Continue solo only if the task is truly atomic or spawning would add no useful independent work, "
            "and record that solo decision in memory with status:solo-decision."
        )
        envelope = Envelope(
            from_id="system", to_id=agent.id, content=notice,
            tokens=estimate_tokens(notice), timestamp=time.time(), mode="steer",
            transient=True,
        )
        agent._steer_inbox.put_nowait(envelope)
        data = {
            "preference": agent.orchestration_preference,
            "turns": agent._turns,
            "tool_calls": agent._tool_calls,
            "min_spawnable_workstreams": self.config.min_spawnable_workstreams,
            "spawnable_task": spawnable,
        }
        self._emit(agent.id, "orchestration_nudge", data)
        self._emit(agent.id, "send", {
            "from": "system", "to": agent.id,
            "mode": "steer", "tokens": envelope.tokens,
            "message": notice,
            "message_type": "orchestration_nudge",
        })

    def _spawn_create_action_pending(self, agent: Agent) -> bool:
        if agent.children:
            return False
        if "status:solo-decision" in set(agent.current_task_tags):
            return False
        if agent.action_state != "create":
            return False
        if agent._create_action_filtered_count > 0:
            return False
        if agent.orchestration_preference == "solo":
            return False
        if len(self.agents) >= self.config.max_agents:
            return False
        if not _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams):
            return False
        turn_trigger = agent._turns >= max(1, self.config.spawn_before_turn)
        tool_trigger = agent._tool_calls >= max(1, self.config.max_solo_tool_calls_before_spawn)
        if agent.orchestration_preference == "balanced" and not (turn_trigger and (tool_trigger or agent._orchestration_nudge_sent)):
            return False
        if agent.orchestration_preference in {"parallel", "aggressive"} and not (turn_trigger or tool_trigger or agent._orchestration_nudge_sent):
            return False
        if not agent._orchestration_nudge_sent:
            return False
        return True

    def _check_peer_progress_nudge(self, agent: Agent) -> None:
        """At bottlenecks, prompt peer discovery without constraining autonomy."""
        if agent._peer_progress_nudge_sent:
            return
        expected_outputs = _expected_output_paths(agent.task)
        if expected_outputs and self._missing_expected_outputs(agent):
            return
        if agent._turns < max(1, self.config.peer_progress_check_after_turns):
            return
        peers = self._peer_progress_candidates(agent)
        if not peers:
            return
        recent = _recent_tool_names(agent.history, lookback=10)
        if "query" in recent:
            return
        stalled_by_tools = agent._no_tool_turns >= max(1, self.config.peer_progress_check_no_tool_turns)
        stalled_by_artifacts = agent._artifact_nudge_count >= max(1, self.config.peer_progress_check_after_artifact_nudges)
        stalled_by_turns = agent._turns >= max(1, self.config.peer_progress_check_after_turns + 2) and agent._tool_calls <= 2
        if not (stalled_by_tools or stalled_by_artifacts or stalled_by_turns):
            return

        agent._peer_progress_nudge_sent = True
        agent._peer_progress_nudge_turn = agent._turns
        query_filter = self._peer_progress_query_filter(agent)
        tags = self._peer_progress_query_tags(agent)
        peer_lines = "\n".join(
            f"- {peer.id}: role={peer.role or '-'} status={peer.status} turns={peer._turns} "
            f"artifacts={len(peer.artifacts)} tags={','.join(peer.current_task_tags[:4]) or '-'}"
            for peer in peers[:8]
        )
        filter_text = json.dumps(query_filter, ensure_ascii=False) if query_filter else "{}"
        tags_text = json.dumps(tags, ensure_ascii=False) if tags else "[]"
        notice = (
            "[Peer Progress Check]\n"
            "You appear bottlenecked or at risk of duplicating peer work. You should usually call query() "
            "to inspect peer state_board/public_memory/artifacts before doing more solo work.\n"
            f"Recommended query: filter={filter_text}, tags={tags_text}.\n"
            "Use your judgment, but make the decision explicit:\n"
            "- query peers if their state may change your next action;\n"
            "- continue solo only if your output is still distinct and useful;\n"
            "- if peers are ahead or already cover your lane, compact with stop_after=true and tags including "
            "`status:pruned`, `reason:peer_ahead`, and your existing task tags;\n"
            "- summarize your partial insight, duplicate risk, and any artifacts so other agents can query it.\n"
            f"Known nearby peers:\n{peer_lines}"
        )
        envelope = Envelope(
            from_id="system",
            to_id=agent.id,
            content=notice,
            tokens=estimate_tokens(notice),
            timestamp=time.time(),
            mode="steer",
            message_type="peer_progress_nudge",
            urgency="high",
            priority=60,
            transient=True,
        )
        agent._steer_inbox.put_nowait(envelope)
        self._emit(agent.id, "peer_progress_nudge", {
            "turns": agent._turns,
            "tool_calls": agent._tool_calls,
            "no_tool_turns": agent._no_tool_turns,
            "artifact_nudge_count": agent._artifact_nudge_count,
            "filter": query_filter,
            "tags": tags,
            "peers": [peer.id for peer in peers[:20]],
        })
        self._emit(agent.id, "send", {
            "from": "system",
            "to": agent.id,
            "mode": "steer",
            "tokens": envelope.tokens,
            "message": notice,
            "message_type": "peer_progress_nudge",
        })

    def _peer_progress_candidates(self, agent: Agent) -> list[Agent]:
        candidates: list[Agent] = []
        for peer in self.agents.values():
            if peer.id == agent.id:
                continue
            if peer.status not in {"running", "idle", "done", "failed"}:
                continue
            same_group = bool(agent.group_id and peer.group_id and peer.group_id == agent.group_id)
            shared_tags = bool(set(agent.current_task_tags).intersection(peer.current_task_tags))
            same_creator = bool(agent.created_by and peer.created_by and peer.created_by == agent.created_by)
            same_parent = bool(agent.parent and peer.parent and peer.parent == agent.parent)
            if same_group or shared_tags or same_creator or same_parent:
                candidates.append(peer)
        candidates.sort(key=lambda p: (
            0 if p.status == "done" else 1,
            -len(p.artifacts),
            -p._turns,
            p.id,
        ))
        return candidates

    def _peer_progress_query_filter(self, agent: Agent) -> dict[str, Any]:
        if agent.group_id:
            return {"group_id": agent.group_id}
        if agent.created_by:
            return {"created_by": agent.created_by}
        if agent.parent:
            return {"parent": agent.parent}
        return {"status": "active"}

    def _peer_progress_query_tags(self, agent: Agent) -> list[str]:
        priority_prefixes = ("problem:", "phase:", "role:", "candidate:", "benchmark:", "feature:", "scope:")
        selected = [tag for tag in agent.current_task_tags if tag.startswith(priority_prefixes)]
        if selected:
            return selected[:8]
        return agent.current_task_tags[:8]

    def _check_artifact_nudge(self, agent: Agent) -> None:
        """Force artifact-oriented agents back to file tools instead of prose loops."""
        if agent._turns < 2:
            return
        missing = self._missing_expected_outputs(agent)
        if not missing:
            return
        if self._create_resume_after_read_pending(agent) or self._multi_phase_create_action_pending(agent):
            return
        source_pressure = self._artifact_pressure_deferred(agent, missing)
        active_child_plan = self._coordinator_child_dependency_plan(agent, missing, None)
        if not source_pressure and active_child_plan and active_child_plan.get("action") in {"message", "create"}:
            if agent._artifact_deferred_nudge_sent:
                return
            agent._artifact_deferred_nudge_sent = True
            unfinished = self.unfinished_child_agents(agent)
            child_lines = "\n".join(
                f"- {child.id}: role={child.role or '-'} group_id={child.group_id or '-'} "
                f"status={child.status} turns={child._turns} missing={self._missing_expected_outputs(child)[:4]}"
                for child in unfinished[:8]
            )
            if not child_lines:
                child_lines = "- no unfinished child details available"
            notice = (
                "[Coordinator Artifact Deferred]\n"
                "Final artifacts are missing, but this coordinator still has active or recoverable child dependencies. "
                "Do not switch into solo final artifact writing from the coordinator/root turn. "
                "Use action=message to query/wait on children and inspect their state_board, public memory, and artifacts. "
                "If query shows a stalled/missing lane or the next protocol wave is ready, leave that decision for the next create turn after runtime revalidation.\n"
                f"Coordinator action={active_child_plan.get('action')} reason={active_child_plan.get('reason')}.\n"
                f"Missing final files:\n" + "\n".join(f"- {path}" for path in missing[:8]) + "\n"
                f"Active child dependencies:\n{child_lines}"
            )
            envelope = Envelope(
                from_id="system",
                to_id=agent.id,
                content=notice,
                tokens=estimate_tokens(notice),
                timestamp=time.time(),
                mode="steer",
                message_type="artifact_nudge_deferred",
                urgency="high",
                priority=55,
                transient=True,
            )
            agent._steer_inbox.put_nowait(envelope)
            self._emit(agent.id, "artifact_nudge_deferred", {
                "turns": agent._turns,
                "missing": missing[:10],
                "coordinator_action": active_child_plan.get("action"),
                "coordinator_reason": active_child_plan.get("reason"),
                "unfinished_children": [
                    {
                        "id": child.id,
                        "role": child.role,
                        "group_id": child.group_id,
                        "status": child.status,
                        "turns": child._turns,
                        "missing_outputs": self._missing_expected_outputs(child)[:10],
                    }
                    for child in unfinished[:20]
                ],
            })
            self._emit(agent.id, "send", {
                "from": "system",
                "to": agent.id,
                "mode": "steer",
                "tokens": envelope.tokens,
                "message": notice,
                "message_type": "artifact_nudge_deferred",
            })
            return
        if source_pressure:
            if agent._artifact_deferred_nudge_sent:
                return
            agent._artifact_deferred_nudge_sent = True
            blockers = self.completion_blockers(agent, include_missing_outputs=False)
            blocker_lines = "\n".join(f"- {b['message']}" for b in blockers[:6])
            child_plan = self._coordinator_child_dependency_plan(agent, missing, blockers)
            if child_plan and child_plan.get("action") in {"message", "create"}:
                notice = (
                    "[Source/Test Evidence Required]\n"
                    "Your task names final artifacts, but it also requires source/test work in shared/source. "
                    "Do not switch into final-report-only delivery yet.\n"
                    "Coordinator boundary is active because you already have child agents. Follow the loop action card: "
                    "use action=message to query/wait on children, or action=create when runtime delegates integration/recovery work. "
                    "Do not inspect, edit, or diff shared/source yourself from the coordinator/root turn.\n"
                    f"Current blockers:\n{blocker_lines}"
                )
                envelope = Envelope(
                    from_id="system",
                    to_id=agent.id,
                    content=notice,
                    tokens=estimate_tokens(notice),
                    timestamp=time.time(),
                    mode="steer",
                    message_type="artifact_nudge_deferred",
                    urgency="high",
                    priority=55,
                    transient=True,
                )
                agent._steer_inbox.put_nowait(envelope)
                self._emit(agent.id, "artifact_nudge_deferred", {
                    "turns": agent._turns,
                    "missing": missing[:10],
                    "blockers": blockers,
                    "coordinator_action": child_plan.get("action"),
                    "coordinator_reason": child_plan.get("reason"),
                })
                self._emit(agent.id, "send", {
                    "from": "system",
                    "to": agent.id,
                    "mode": "steer",
                    "tokens": envelope.tokens,
                    "message": notice,
                    "message_type": "artifact_nudge_deferred",
                })
                return
            notice = (
                "[Source/Test Evidence Required]\n"
                "Your task names final artifacts, but it also requires source/test work in shared/source. "
                "Do not switch into final-report-only delivery yet. Choose action=work until source/test evidence exists.\n"
                f"Current blockers:\n{blocker_lines}"
            )
            envelope = Envelope(
                from_id="system",
                to_id=agent.id,
                content=notice,
                tokens=estimate_tokens(notice),
                timestamp=time.time(),
                mode="steer",
                message_type="artifact_nudge_deferred",
                urgency="high",
                priority=55,
                transient=True,
            )
            agent._steer_inbox.put_nowait(envelope)
            self._emit(agent.id, "artifact_nudge_deferred", {
                "turns": agent._turns,
                "missing": missing[:10],
                "blockers": blockers,
            })
            self._emit(agent.id, "send", {
                "from": "system",
                "to": agent.id,
                "mode": "steer",
                "tokens": envelope.tokens,
                "message": notice,
                "message_type": "artifact_nudge_deferred",
            })
            return
        recent_tool_names = _recent_tool_names(agent.history, lookback=8)
        if any(name in recent_tool_names for name in ("file_write", "file_replace", "submit")):
            return
        agent._artifact_nudge_count += 1
        targets = "\n".join(f"- {path}" for path in missing[:6])
        more = "" if len(missing) <= 6 else f"\n- ... {len(missing) - 6} more"
        force_line = ""
        if agent._artifact_nudge_count >= 2:
            force_line = (
                "\nRuntime will now narrow available tools toward artifact delivery because explicit output files "
                "are still missing after previous guidance."
            )
        if agent._artifact_nudge_count >= 3:
            force_line = (
                "\nRuntime will expose only file_write on this turn because explicit output files "
                "are still missing after repeated guidance."
            )
        notice = (
            "[Artifact Required]\n"
            "Your task names concrete output files, but they are not present yet. "
            "Do not leave the deliverable only in chat. Choose action=work if creating the missing artifacts is the next step. "
            "If the solution is uncertain, write the best current candidate and record uncertainty in the .md/report file.\n"
            f"Missing files:\n{targets}{more}{force_line}"
        )
        envelope = Envelope(
            from_id="system",
            to_id=agent.id,
            content=notice,
            tokens=estimate_tokens(notice),
            timestamp=time.time(),
            mode="steer",
            message_type="artifact_nudge",
            urgency="high",
            priority=50,
            transient=True,
        )
        agent._steer_inbox.put_nowait(envelope)
        self._emit(agent.id, "artifact_nudge", {
            "turns": agent._turns,
            "missing": missing[:10],
            "count": agent._artifact_nudge_count,
        })
        self._emit(agent.id, "send", {
            "from": "system",
            "to": agent.id,
            "mode": "steer",
            "tokens": envelope.tokens,
            "message": notice,
            "message_type": "artifact_nudge",
        })

    def _missing_expected_outputs(self, agent: Agent) -> list[str]:
        expected = _expected_output_paths(agent.task)
        if not expected:
            return []
        return [path for path in expected if not (self._tool_context.workspace_root / path).exists()]

    def expected_outputs(self, agent: Agent) -> list[str]:
        return _expected_output_paths(agent.task)

    def missing_expected_outputs(self, agent: Agent) -> list[str]:
        return self._missing_expected_outputs(agent)

    def _shared_source_dir(self) -> Path:
        return self.config.workspace_root / self.config.shared_dir / "source"

    def _source_change_status(self) -> dict[str, Any]:
        source_dir = self._shared_source_dir()
        status: dict[str, Any] = {
            "source_dir": str(source_dir),
            "exists": source_dir.exists(),
            "changed": False,
        }
        if not source_dir.exists():
            return status
        try:
            proc = subprocess.run(
                ["git", "-C", str(source_dir), "status", "--short"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except Exception as e:
            status["error"] = str(e)
            return status
        status["git_returncode"] = proc.returncode
        if proc.returncode != 0:
            status["stderr"] = proc.stderr[-500:]
            return status
        short = proc.stdout.strip()
        status["status_short"] = short[:1000]
        status["changed"] = bool(short)
        return status

    def _shared_source_has_changes(self) -> bool:
        return bool(self._source_change_status().get("changed"))

    def _descendant_agents(self, agent: Agent) -> list[Agent]:
        descendants: list[Agent] = []
        seen: set[str] = set()
        pending = sorted(agent.children)
        while pending:
            child_id = pending.pop(0)
            if child_id in seen:
                continue
            seen.add(child_id)
            child = self.agents.get(child_id)
            if not child:
                continue
            descendants.append(child)
            pending.extend(sorted(child.children))
        return descendants

    def _related_agents_for_completion_evidence(self, agent: Agent) -> list[Agent]:
        related: list[Agent] = [agent]
        seen = {agent.id}
        for child in self._descendant_agents(agent):
            if child.id in seen:
                continue
            seen.add(child.id)
            related.append(child)
        referenced_groups = set(_referenced_group_ids(agent.task))
        if referenced_groups:
            for peer in sorted(self.agents.values(), key=lambda a: a.id):
                if peer.id in seen or peer.group_id not in referenced_groups:
                    continue
                seen.add(peer.id)
                related.append(peer)
        return related

    def _successful_test_commands_for_completion(self, agent: Agent) -> list[str]:
        commands: list[str] = []
        seen: set[str] = set()
        for related in self._related_agents_for_completion_evidence(agent):
            for command in related._successful_test_commands:
                if command in seen:
                    continue
                seen.add(command)
                commands.append(command)
        return commands

    def _artifact_pressure_deferred(self, agent: Agent, missing_outputs: list[str]) -> bool:
        if not missing_outputs or not _task_requires_source_change(agent.task):
            return False
        if not self._shared_source_has_changes():
            return True
        return _task_requires_test_run(agent.task) and not self._successful_test_commands_for_completion(agent)

    def completion_blockers(
        self,
        agent: Agent,
        *,
        tags: list[str] | None = None,
        include_missing_outputs: bool = True,
    ) -> list[dict[str, Any]]:
        final_tags = set(tags or agent.current_task_tags)
        if "status:pruned" in final_tags and self._pruned_stop_is_allowed(agent):
            return []
        blockers: list[dict[str, Any]] = []
        if include_missing_outputs:
            missing = self._missing_expected_outputs(agent)
            if missing:
                blockers.append({
                    "kind": "missing_outputs",
                    "message": "Task names explicit output files that are still missing.",
                    "missing_outputs": missing[:20],
                })
        if _task_requires_source_change(agent.task):
            source_status = self._source_change_status()
            if not source_status.get("changed"):
                blockers.append({
                    "kind": "source_change",
                    "message": "Task requires changes under shared/source, but git status is clean or unavailable.",
                    "source_status": source_status,
                })
        successful_test_commands = self._successful_test_commands_for_completion(agent)
        if _task_requires_test_run(agent.task) and not successful_test_commands:
            blockers.append({
                "kind": "test_run",
                "message": "Task requires test execution, but no successful pytest/unittest shell command is recorded.",
                "shell_commands": [
                    command
                    for related in self._related_agents_for_completion_evidence(agent)
                    for command in related._shell_commands
                ][-10:],
            })
        return blockers

    def _child_marker_text(self, child: Agent) -> str:
        return "\n".join(
            str(part)
            for part in (
                child.role,
                child.group_id,
                child.workflow_prior,
                child.create_type,
                child.relationship,
                " ".join(child.current_task_tags),
            )
            if part
        ).lower()

    def _has_child_marker(self, agent: Agent, markers: tuple[str, ...]) -> bool:
        for child_id in agent.children:
            child = self.agents.get(child_id)
            if not child:
                continue
            text = self._child_marker_text(child)
            if any(marker in text for marker in markers):
                return True
        return False

    def _has_unfinished_child_marker(self, agent: Agent, markers: tuple[str, ...]) -> bool:
        for child in self.unfinished_child_agents(agent):
            text = self._child_marker_text(child)
            if any(marker in text for marker in markers):
                return True
        return False

    def _has_integration_child(self, agent: Agent) -> bool:
        return self._has_child_marker(
            agent,
            (
                "integration",
                "integrator",
                "phase:integration",
                "area:tests",
                "role:tester",
                "tester",
                "qa",
                "review",
                "reviewer",
                "validation",
                "verify",
            ),
        )

    def _has_recovery_child(self, agent: Agent) -> bool:
        return self._has_child_marker(
            agent,
            ("recovery", "fallback", "repair", "source-recovery", "source_recovery"),
        )

    def _coordinator_child_dependency_plan(
        self,
        agent: Agent,
        missing_outputs: list[str],
        blockers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if not agent.children:
            return None
        blockers = blockers if blockers is not None else self.completion_blockers(agent, include_missing_outputs=False)
        unfinished = self.unfinished_child_agents(agent)
        source_task = _task_requires_source_change(agent.task)
        blocker_kinds = {str(blocker.get("kind")) for blocker in blockers}
        can_create = len(self.agents) < self.config.max_agents
        integration_markers = (
            "integration",
            "integrator",
            "phase:integration",
            "area:tests",
            "role:tester",
            "tester",
            "qa",
            "review",
            "reviewer",
            "validation",
            "verify",
            "recovery",
            "fallback",
            "repair",
        )

        if source_task and blockers:
            if (
                can_create
                and "test_run" in blocker_kinds
                and self._shared_source_has_changes()
                and not self._has_unfinished_child_marker(agent, integration_markers)
            ):
                return {
                    "action": "create",
                    "reason": (
                        "delegate_integration_after_child_source_progress"
                        if not self._has_integration_child(agent)
                        else "delegate_test_recovery_after_integration_stall"
                    ),
                    "turn_added": agent._turns,
                    "blockers": blockers,
                }
            if (
                can_create
                and "source_change" in blocker_kinds
                and not unfinished
                and not self._has_unfinished_child_marker(agent, ("recovery", "fallback", "repair", "source-recovery", "source_recovery"))
            ):
                return {
                    "action": "create",
                    "reason": "delegate_source_recovery_after_child_stall",
                    "turn_added": agent._turns,
                    "blockers": blockers,
                }
            if unfinished:
                return {
                    "action": "message",
                    "reason": "coordinate_children_before_source_work",
                    "turn_added": agent._turns,
                    "blockers": blockers,
                }

        if unfinished and (missing_outputs or blockers or _task_prefers_query_before_artifact(agent.task)):
            return {
                "action": "message",
                "reason": "coordinate_unfinished_children_before_artifacts",
                "turn_added": agent._turns,
                "blockers": blockers,
            }
        return None

    def _pruned_stop_is_allowed(self, agent: Agent) -> bool:
        if agent._last_query_turn <= 0:
            return False
        return bool(self._peer_progress_candidates(agent))

    def unfinished_child_agents(self, agent: Agent) -> list[Agent]:
        return [
            self.agents[child_id]
            for child_id in sorted(agent.children)
            if child_id in self.agents and self.agents[child_id].status in {"running", "idle"}
        ]

    def expected_child_outputs_status(self, agent: Agent) -> dict[str, dict[str, Any]]:
        status: dict[str, dict[str, Any]] = {}
        for child in self.unfinished_child_agents(agent):
            expected = _expected_output_paths(child.task)
            if not expected:
                continue
            missing = self._missing_expected_outputs(child)
            status[child.id] = {
                "role": child.role,
                "group_id": child.group_id,
                "expected": expected[:20],
                "missing": missing[:20],
                "complete": not missing,
                "status": child.status,
                "turns": child._turns,
            }
        return status

    def stop_blockers(self, agent: Agent) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        seen: set[str] = set()
        for child in self.unfinished_child_agents(agent):
            expected = _expected_output_paths(child.task)
            missing = self._missing_expected_outputs(child) if expected else []
            seen.add(child.id)
            blockers.append({
                "kind": "child",
                "id": child.id,
                "role": child.role,
                "group_id": child.group_id,
                "status": child.status,
                "turns": child._turns,
                "missing_outputs": missing[:20],
            })
        for group_id in _referenced_group_ids(agent.task):
            for peer in sorted(self.agents.values(), key=lambda a: a.id):
                if peer.id == agent.id or peer.id in seen:
                    continue
                if peer.group_id != group_id or peer.status not in {"running", "idle"}:
                    continue
                expected = _expected_output_paths(peer.task)
                missing = self._missing_expected_outputs(peer) if expected else []
                seen.add(peer.id)
                blockers.append({
                    "kind": "referenced_group",
                    "id": peer.id,
                    "role": peer.role,
                    "group_id": peer.group_id,
                    "status": peer.status,
                    "turns": peer._turns,
                    "missing_outputs": missing[:20],
                })
        return blockers

    def can_stop_with_active_dependencies(self, agent: Agent, blockers: list[dict[str, Any]]) -> bool:
        if not blockers:
            return True
        if agent._last_query_turn <= 0:
            return False
        return (agent._turns - agent._last_query_turn) <= 4

    async def _wait_for_remaining_agents(self, root: Agent) -> None:
        """After root finishes, avoid truncating running children that are still making progress."""
        while True:
            running = [a for a in self.agents.values() if a.id != root.id and a.status in {"running", "idle"}]
            if not running:
                return
            tasks = [a._task for a in running if a._task and not a._task.done()]
            if not tasks:
                return
            self._emit(root.id, "join_remaining_agents", {
                "count": len(running),
                "agents": [
                    {
                        "id": a.id,
                        "role": a.role,
                        "group_id": a.group_id,
                        "status": a.status,
                        "turns": a._turns,
                        "missing_outputs": self._missing_expected_outputs(a)[:10],
                    }
                    for a in running[:20]
                ],
            })
            await asyncio.gather(*tasks, return_exceptions=True)

    def _artifact_tool_scope(self, agent: Agent, missing_outputs: list[str]) -> Literal["all", "read_write", "write_only"]:
        if not missing_outputs:
            return "all"
        if agent._turns < 2:
            return "all"
        if agent._artifact_nudge_count <= 0:
            return "all"
        if self._artifact_pressure_deferred(agent, missing_outputs):
            return "all"
        available = self._available_work_tools()
        if "file_write" not in available:
            return "all"
        coordination_task = _task_is_coordination_artifact_task(agent.task)
        if coordination_task and agent._last_query_turn <= 0:
            return "read_write"
        if coordination_task and agent._artifact_nudge_count <= 4:
            return "read_write"
        if coordination_task and agent._filtered_read_intent_count >= 1:
            return "read_write"
        if agent._artifact_nudge_count <= 2:
            return "read_write"
        if agent._artifact_nudge_count >= 3:
            if self._peer_progress_decision_pending(agent):
                return "read_write"
            return "write_only"
        if "file_read" in available:
            return "read_write"
        return "write_only"

    def _llm_max_tokens_for_turn(
        self,
        *,
        loop_action: str,
        recommended_tool_choice: str | None,
        missing_outputs: list[str],
    ) -> int | None:
        if loop_action != "work" or not missing_outputs:
            return None
        configured = int(self.config.artifact_write_max_tokens or 0)
        if configured <= 0:
            return None
        try:
            current = int(os.environ.get("NANOMA_MAX_TOKENS", "0") or "0")
        except ValueError:
            current = 0
        return max(current, configured) if current > 0 else configured

    def _artifact_scoped_tool_schemas(self, missing_outputs: list[str], names: list[str]) -> list[ToolDef]:
        from nanoma.meta import META_TOOLS

        available = {**self._available_work_tools(), **META_TOOLS}
        schemas: list[ToolDef] = []
        targets = ", ".join(missing_outputs[:8])
        more = "" if len(missing_outputs) <= 8 else f", ... {len(missing_outputs) - 8} more"
        for name in names:
            tool = available.get(name)
            if not tool:
                continue
            schema = json.loads(json.dumps(tool["schema"]))
            function = schema.get("function", {})
            if name == "file_write":
                function["description"] = (
                    "Write one required artifact file now. Write a path listed in the remaining required output files: "
                    f"{targets}{more}. Do not write placeholders or unrelated paths. "
                    "Put the full artifact in the content argument. Keep assistant message text empty or very short; "
                    "assistant prose alone does not create files."
                )
                params = function.get("parameters", {})
                props = params.get("properties", {})
                if "path" in props:
                    props["path"]["description"] = f"Must be exactly one of: {targets}{more}"
                    props["path"]["enum"] = missing_outputs[:50]
            elif name == "file_read":
                function["description"] = (
                    "Read only if you still need a specific detail before writing the required artifacts. "
                    "Do not call file_read repeatedly when the remaining output files are known."
                )
            schemas.append(schema)
        return schemas

    def _annotate_file_write_result(self, agent: Agent, tc: ToolCall, result: Any) -> Any:
        expected = _expected_output_paths(agent.task)
        if not expected or not isinstance(result, dict):
            return result
        path_arg = str(tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("filepath", ""))))
        normalized = self._normalize_workspace_relative_path(path_arg, agent.workspace)
        if normalized and normalized not in set(expected):
            annotated = dict(result)
            annotated["artifact_warning"] = "unexpected_artifact_path"
            annotated["expected_paths"] = expected[:20]
            annotated["written_relative_path"] = normalized
            return annotated
        return result

    def _register_expected_artifact_write(self, agent: Agent, tc: ToolCall, result: Any) -> None:
        expected = set(_expected_output_paths(agent.task))
        if not expected or not isinstance(result, dict) or result.get("error"):
            return
        path_arg = str(tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("filepath", ""))))
        normalized = self._normalize_workspace_relative_path(path_arg, agent.workspace)
        if normalized not in expected:
            return
        abs_path = (self.config.workspace_root / normalized).resolve()
        if not abs_path.exists() or not abs_path.is_file():
            return
        if normalized not in {artifact.path for artifact in agent.artifacts}:
            agent.artifacts.append(Artifact(path=normalized, absolute_path=abs_path, description="expected output", agent_id=agent.id))
        self.memory.update(agent.id, add_artifacts=[normalized], tags=agent.current_task_tags)
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        self.state_board_sync(agent.id)

    def _attempt_artifact_commit_from_text(self, agent: Agent, content: str) -> list[str]:
        """Commit explicit required artifacts from assistant text when tool calling fails."""
        if not content or "file_write" not in self._available_work_tools():
            return []
        missing = self._missing_expected_outputs(agent)
        if not missing:
            return []
        committed: list[str] = []
        for rel_path in missing:
            extracted = _extract_artifact_content_from_text(content, rel_path)
            if extracted is None:
                continue
            abs_path = (self._tool_context.workspace_root / rel_path).resolve()
            try:
                abs_path.relative_to(self._tool_context.workspace_root.resolve())
            except ValueError:
                continue
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(extracted)
            committed.append(rel_path)
            agent.artifacts.append(Artifact(path=rel_path, absolute_path=abs_path, description="committed from assistant text", agent_id=agent.id))
            self._emit(agent.id, "artifact_commit", {
                "path": rel_path,
                "bytes": len(extracted.encode()),
                "source": "assistant_text",
            })
        if committed:
            self.memory.update(agent.id, add_artifacts=committed, tags=agent.current_task_tags)
            agent.memory = {"public_memory": self.memory.serialize(agent.id)}
            if not self._missing_expected_outputs(agent):
                agent._artifact_nudge_count = 0
            self.state_board_sync(agent.id)
        return committed

    def _auto_complete_if_outputs_exist(self, agent: Agent, *, reason: str) -> bool:
        expected = _expected_output_paths(agent.task)
        if not expected or self._missing_expected_outputs(agent):
            return False
        blockers = self.completion_blockers(agent, include_missing_outputs=False)
        if blockers:
            self._emit(agent.id, "completion_blocked", {
                "reason": reason,
                "blockers": blockers,
            })
            return False
        files = [path for path in expected if (self.config.workspace_root / path).is_file()]
        for rel_path in files:
            abs_path = (self.config.workspace_root / rel_path).resolve()
            if rel_path not in {artifact.path for artifact in agent.artifacts}:
                agent.artifacts.append(Artifact(path=rel_path, absolute_path=abs_path, description="expected output", agent_id=agent.id))
        summary = f"Explicit output files exist; auto-completing at {reason}."
        self.memory.update(
            agent.id,
            public_summary=agent.result or summary,
            tags=agent.current_task_tags,
            add_artifacts=files,
            clear_active_task=True,
            add_experience=ExperienceCard(summary=agent.result or summary, tags=agent.current_task_tags, artifacts=files),
        )
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        agent.action_state = "stop"
        agent.status = "done"
        agent.result = agent.result or summary
        self.state_board_sync(agent.id)
        return True

    def _normalize_workspace_relative_path(self, path: str, agent_workspace: Path) -> str:
        raw = str(path or "").strip()
        if raw.startswith("$SHARED/"):
            raw = f"{self.config.shared_dir}/{raw[len('$SHARED/'):]}"
        p = Path(raw).expanduser()
        try:
            if p.is_absolute():
                resolved = p.resolve()
            elif raw.startswith(f"{self.config.shared_dir}/"):
                resolved = (self.config.workspace_root / raw).resolve()
            else:
                resolved = (agent_workspace / raw).resolve()
            return str(resolved.relative_to(self.config.workspace_root.resolve()))
        except Exception:
            return raw

    def _inject_messages(self, agent: Agent, queue: asyncio.Queue[Envelope], *, persist_transient: bool = True) -> list[Envelope]:
        msgs = []
        while not queue.empty():
            try:
                msgs.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        msgs.sort(key=lambda m: m.priority, reverse=True)
        for msg in msgs:
            if msg.transient and not persist_transient:
                continue
            header = f"[Message from {msg.from_id} | type={msg.message_type} | urgency={msg.urgency}]"
            payload = f"\nPayload: {json.dumps(msg.payload, ensure_ascii=False)}" if msg.payload is not None else ""
            ack = "\nAck requested." if msg.requires_ack else ""
            agent.history.append({"role": "user", "content": f"{header}: {msg.content}{payload}{ack}"})
        return msgs

    def _loop_message_content(self, msg: Envelope) -> str:
        header = f"[Message from {msg.from_id} | type={msg.message_type} | urgency={msg.urgency}]"
        payload = f"\nPayload: {json.dumps(msg.payload, ensure_ascii=False)}" if msg.payload is not None else ""
        ack = "\nAck requested." if msg.requires_ack else ""
        return f"{header}: {msg.content}{payload}{ack}"

    def _peer_progress_decision_pending(self, agent: Agent) -> bool:
        if not agent._peer_progress_nudge_sent:
            return False
        if agent._last_query_turn >= agent._peer_progress_nudge_turn:
            return False
        if not self._peer_progress_candidates(agent):
            return False
        return True

    def _peer_progress_post_query_pending(self, agent: Agent) -> bool:
        if not agent._peer_progress_nudge_sent:
            return False
        if agent._last_query_turn < agent._peer_progress_nudge_turn:
            return False
        if (agent._turns - agent._last_query_turn) > 4:
            return False
        if not self._missing_expected_outputs(agent):
            return False
        return bool(self._completed_peer_progress_candidates(agent))

    def _completed_peer_progress_candidates(self, agent: Agent) -> list[Agent]:
        candidates: list[Agent] = []
        for peer in self._peer_progress_candidates(agent):
            expected = _expected_output_paths(peer.task)
            outputs_complete = bool(expected) and not self._missing_expected_outputs(peer)
            if peer.status == "done" or peer.artifacts or outputs_complete:
                candidates.append(peer)
        return candidates

    def _create_resume_after_read_pending(self, agent: Agent) -> bool:
        if not agent._create_resume_after_read:
            return False
        if agent.children:
            agent._create_resume_after_read = False
            return False
        if len(self.agents) >= self.config.max_agents:
            agent._create_resume_after_read = False
            return False
        if "status:solo-decision" in set(agent.current_task_tags):
            agent._create_resume_after_read = False
            return False
        return _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)

    def _multi_phase_create_action_pending(self, agent: Agent) -> bool:
        if not agent.children:
            return False
        if self.unfinished_child_agents(agent):
            return False
        if len(self.agents) >= self.config.max_agents:
            return False
        if not self._missing_expected_outputs(agent):
            return False
        if "status:solo-decision" in set(agent.current_task_tags):
            return False
        if not _task_has_multi_phase_peer_protocol(agent.task):
            return False
        return True

    def _refresh_loop_action_plan(self, agent: Agent, missing_outputs: list[str]) -> None:
        state_action = agent.action_state if agent.action_state in _LOOP_ACTIONS else ""
        child_revalidation_blockers = self.completion_blockers(agent, include_missing_outputs=False) if agent.children else []
        child_revalidation_plan = self._coordinator_child_dependency_plan(
            agent,
            missing_outputs,
            child_revalidation_blockers,
        )
        if child_revalidation_plan and child_revalidation_plan.get("action") in {"message", "create"}:
            current = agent._loop_action_plan
            if current and current.get("action") == child_revalidation_plan.get("action") and current.get("reason") == child_revalidation_plan.get("reason"):
                return
            agent._loop_action_plan = child_revalidation_plan
            return

        if state_action and state_action != "create" and agent._state_action_version > agent._state_action_consumed_version:
            agent._state_action_consumed_version = agent._state_action_version
            agent._loop_action_plan = {
                "action": state_action,
                "reason": "state_board_action",
                "turn_added": agent._turns,
            }
            return

        if self._create_resume_after_read_pending(agent):
            agent._loop_action_plan = {
                "action": "create",
                "reason": "resume_create_after_read_orientation",
                "turn_added": agent._turns,
            }
            return

        if self._spawn_create_action_pending(agent):
            agent._loop_action_plan = {
                "action": "create",
                "reason": "spawnable_workstreams_before_solo_execution",
                "turn_added": agent._turns,
            }
            return

        if agent._create_action_filtered_count > 0:
            if (
                agent._filtered_read_intent_count <= 0
                and not agent.children
                and agent.action_state == "create"
                and _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
            ):
                agent._loop_action_plan = {
                    "action": "create",
                    "reason": "retry_spawn_after_create_miss",
                    "turn_added": agent._turns,
                }
                return
            action = "read" if agent._filtered_read_intent_count > 0 else "work"
            if action == "read":
                agent._create_resume_after_read = True
            self.state_board_update(agent.id, action_state=action)
            agent._state_action_consumed_version = agent._state_action_version
            agent._loop_action_plan = {
                "action": action,
                "reason": "create_scope_intent_released",
                "turn_added": agent._turns,
            }
            return

        expected_outputs = _expected_output_paths(agent.task)
        outputs_complete = bool(expected_outputs) and not missing_outputs
        if outputs_complete:
            current = agent._loop_action_plan
            if current and current.get("action") in {"compact", "stop"}:
                return
            blockers = self.completion_blockers(agent, include_missing_outputs=False)
            child_plan = self._coordinator_child_dependency_plan(agent, missing_outputs, blockers)
            if child_plan:
                if current and current.get("action") == child_plan.get("action") and current.get("reason") == child_plan.get("reason"):
                    return
                agent._loop_action_plan = child_plan
                return
            if blockers:
                agent._loop_action_plan = {
                    "action": "work",
                    "reason": "completion_evidence_required_before_compact",
                    "turn_added": agent._turns,
                    "blockers": blockers,
                }
                return
            if _task_prefers_query_before_artifact(agent.task) and agent._last_query_turn <= 0:
                agent._loop_action_plan = {
                    "action": "read",
                    "reason": "outputs_complete_but_peer_query_required",
                    "turn_added": agent._turns,
                }
                return
            agent._loop_action_plan = {
                "action": "compact",
                "reason": "outputs_complete_publish_memory",
                "turn_added": agent._turns,
            }
            return

        current = agent._loop_action_plan
        blockers_without_outputs = self.completion_blockers(agent, include_missing_outputs=False)
        child_plan = self._coordinator_child_dependency_plan(agent, missing_outputs, blockers_without_outputs)
        if child_plan:
            if current and current.get("action") == child_plan.get("action") and current.get("reason") == child_plan.get("reason"):
                return
            agent._loop_action_plan = child_plan
            return

        if self._multi_phase_create_action_pending(agent):
            agent._loop_action_plan = {
                "action": "create",
                "reason": "multi_phase_peer_wave_pending",
                "turn_added": agent._turns,
            }
            return

        if missing_outputs and self._artifact_pressure_deferred(agent, missing_outputs):
            if current and current.get("action") == "work" and current.get("reason") == "source_test_evidence_before_artifacts":
                return
            agent._loop_action_plan = {
                "action": "work",
                "reason": "source_test_evidence_before_artifacts",
                "turn_added": agent._turns,
                "blockers": blockers_without_outputs,
            }
            return

        query_before_artifact = (
            bool(missing_outputs)
            and _task_prefers_query_before_artifact(agent.task)
            and agent._last_query_turn <= 0
        )
        coordination_task = _task_is_coordination_artifact_task(agent.task)
        filtered_read_threshold = 1 if coordination_task else 2
        filtered_read_intent = bool(missing_outputs) and agent._filtered_read_intent_count >= filtered_read_threshold
        filtered_read_from_current_turn = (
            filtered_read_intent
            and coordination_task
            and agent._filtered_read_intent_seen_turn >= agent._turns
        )
        peer_post_query_pending = bool(missing_outputs) and self._peer_progress_post_query_pending(agent)
        if current:
            action = current.get("action")
            if filtered_read_from_current_turn and action == "work":
                current = None
            else:
                if action in {"compact", "stop"} and agent.status == "running":
                    return
                if peer_post_query_pending:
                    if action in {"work", "compact", "stop"}:
                        return
                if query_before_artifact or filtered_read_intent:
                    if action in {"read", "query"}:
                        return
                elif action == "work" and missing_outputs and agent._last_artifact_turn < int(current.get("turn_added", 0)):
                    return

        if missing_outputs:
            if peer_post_query_pending:
                if agent._write_only_miss_count >= 1:
                    agent._loop_action_plan = {
                        "action": "compact",
                        "reason": "peer_query_found_completed_peers_prune_or_summarize",
                        "turn_added": agent._turns,
                    }
                    return
                agent._loop_action_plan = {
                    "action": "work",
                    "reason": "peer_query_found_completed_peers_write_or_prune",
                    "turn_added": agent._turns,
                }
                return
            if query_before_artifact or filtered_read_intent:
                agent._loop_action_plan = {
                    "action": "read",
                    "reason": "task_requests_peer_query_before_artifact" if query_before_artifact else "filtered_read_intent_after_write_scope",
                    "turn_added": agent._turns,
                }
                return
            agent._loop_action_plan = {
                "action": "work",
                "reason": "missing_artifacts",
                "turn_added": agent._turns,
            }
            return
        default_action = self._default_loop_action(agent)
        agent._loop_action_plan = {
            "action": default_action,
            "reason": "default_task_progress",
            "turn_added": agent._turns,
        }

    def _default_loop_action(self, agent: Agent) -> str:
        if self.unfinished_child_agents(agent):
            return "message"
        if agent._turns <= 1 and agent.orchestration_preference != "solo" and len(self.agents) < self.config.max_agents:
            return "create"
        blockers = self.completion_blockers(agent, include_missing_outputs=False)
        if any(blocker.get("kind") in {"source_change", "test_run"} for blocker in blockers):
            return "work"
        if _task_prefers_query_before_artifact(agent.task) and agent._last_query_turn <= 0:
            return "read"
        return "work"

    def _partition_tool_calls_by_action(self, agent: Agent, tool_calls: list[ToolCall]) -> tuple[list[ToolCall], list[ToolCall]]:
        plan = agent._loop_action_plan
        if not plan:
            return tool_calls, []
        action = str(plan.get("action", ""))
        allowed, _scope = self._tools_for_loop_turn(
            agent,
            action,
            self._missing_expected_outputs(agent),
        )
        if not allowed:
            return tool_calls, []
        executed: list[ToolCall] = []
        skipped: list[ToolCall] = []
        for tc in tool_calls:
            if tc.name in allowed:
                executed.append(tc)
            else:
                skipped.append(tc)
        return executed, skipped

    def _is_missing_output_file_write(self, agent: Agent, tc: ToolCall) -> bool:
        if tc.name != "file_write":
            return False
        missing = set(self._missing_expected_outputs(agent))
        if not missing:
            return False
        path_arg = str(tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("filepath", ""))))
        normalized = self._normalize_workspace_relative_path(path_arg, agent.workspace)
        return normalized in missing

    def _emit_filtered_text_tool_calls(
        self,
        agent: Agent,
        content: str,
        accepted_tool_calls: list[ToolCall],
        turn_tool_schemas: list[ToolDef],
    ) -> None:
        if "<｜DSML｜invoke" not in (content or ""):
            return
        text_calls = _parse_text_tool_calls(content, None)
        if not text_calls:
            return
        schema_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in turn_tool_schemas
            if (tool.get("function") or {}).get("name")
        }
        accepted_names = [tc.name for tc in accepted_tool_calls]
        filtered_read_intent = False
        filtered_create_scope_intent = False
        for tc in text_calls:
            canonical_name = _TEXT_TOOL_ALIASES.get(tc.name, tc.name)
            if canonical_name in accepted_names or tc.name in accepted_names:
                continue
            reason = "not_in_current_tool_scope" if canonical_name not in schema_names else "not_returned_as_executable_tool_call"
            if canonical_name in {"query", "file_read", "file_list", "grep"}:
                filtered_read_intent = True
            if (agent._loop_action_plan or {}).get("action") == "create" and canonical_name not in self._tools_for_loop_action("create"):
                filtered_create_scope_intent = True
            self._emit(agent.id, "tool_call_filtered", {
                "tool": canonical_name,
                "raw_tool": tc.name,
                "args": tc.arguments,
                "reason": reason,
                "available_tools": sorted(schema_names),
            })
        if filtered_create_scope_intent:
            agent._create_action_filtered_count += 1
            self._emit(agent.id, "create_action_miss", {
                "count": agent._create_action_filtered_count,
                "available_tools": sorted(schema_names),
            })
        if filtered_read_intent:
            agent._filtered_read_intent_count += 1
            agent._filtered_read_intent_seen_turn = agent._turns
        elif text_calls:
            agent._filtered_read_intent_count = 0

    def _recover_text_tool_calls(self, content: str, turn_tool_schemas: list[ToolDef]) -> list[ToolCall]:
        if "<｜DSML｜invoke" not in (content or ""):
            return []
        schema_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in turn_tool_schemas
            if (tool.get("function") or {}).get("name")
        }
        if not schema_names:
            return []
        calls = _parse_text_tool_calls(content, schema_names)
        alias_allowed = set(schema_names).union(_TEXT_TOOL_ALIASES)
        if not calls:
            calls = _parse_text_tool_calls(content, alias_allowed)
        if not calls:
            calls = _parse_text_tool_calls(content, None)
        recovered: list[ToolCall] = []
        for tc in calls:
            name = _TEXT_TOOL_ALIASES.get(tc.name, tc.name)
            if name not in schema_names:
                matches = get_close_matches(name, schema_names, n=1, cutoff=0.88)
                if matches:
                    name = matches[0]
            if name not in schema_names:
                continue
            args = dict(tc.arguments)
            if name in {"file_read", "file_write", "file_replace"} and "path" not in args:
                for alias in ("file_path", "filepath", "filePath"):
                    if alias in args:
                        args["path"] = args[alias]
                        break
            recovered.append(ToolCall(id=tc.id, name=name, arguments=args))
        return recovered

    def _tools_for_loop_action(self, action: str) -> set[str]:
        if action in {"read", "query"}:
            return {"query", "file_read", "file_list", "grep", "set_status", "get_cost"}
        if action == "work":
            return {
                "file_read", "file_write", "file_replace", "file_list", "grep",
                "shell", "bt_aggregate", "submit", "query", "compact", "set_status", "get_cost",
            }
        if action == "message":
            return {"send", "wait", "query", "set_status", "get_cost"}
        if action == "compact":
            return {
                "compact", "set_status", "query", "file_read", "file_list", "grep",
                "file_write", "file_replace", "submit", "get_cost",
            }
        if action == "stop":
            return {
                "set_status", "compact", "wait", "query", "file_read", "file_list", "grep",
                "file_write", "file_replace", "submit", "get_cost",
            }
        if action == "create":
            return {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
        return set()

    def _tools_for_loop_turn(self, agent: Agent, action: str, missing_outputs: list[str]) -> tuple[set[str], str]:
        allowed = self._tools_for_loop_action(action)
        if (
            action == "create"
            and agent._create_action_filtered_count >= 1
            and not agent.children
            and _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
        ):
            spawn_tools = {"spawn", "create_agent", "spawn_many"} & allowed
            if spawn_tools:
                return spawn_tools, "retry_spawn_after_create_miss"
        if (
            action == "work"
            and missing_outputs
            and agent._write_only_miss_count >= 1
            and "file_write" in self._available_work_tools()
        ):
            return {"file_write"}, "retry_file_write_after_text_miss"
        if action == "work" and missing_outputs:
            artifact_scope = self._artifact_tool_scope(agent, missing_outputs)
            if artifact_scope == "read_write":
                scoped = {
                    "file_read", "file_list", "grep", "query", "file_write", "file_replace",
                    "compact", "set_status", "get_cost",
                } & allowed
                if scoped:
                    return scoped, "artifact_read_write"
            if artifact_scope == "write_only":
                scoped = {"file_write"} & allowed
                if scoped:
                    return scoped, "artifact_write_only"
        return allowed, "action_default"

    def _build_loop_action_context(
        self,
        agent: Agent,
        missing_outputs: list[str],
        messages: list[Envelope],
    ) -> Message | None:
        transient = [msg for msg in messages if msg.transient]
        if not transient and not agent._loop_action_plan:
            return None

        actions = "create, read, message, work, compact, stop"
        plan = agent._loop_action_plan
        parts = [
            "[Loop Action Card]",
            "This is transient runtime guidance for this LLM turn only; it is not agent memory.",
            f"Choose exactly one next action from: {actions}.",
            "Use tools only for the selected action. You may call multiple tools within that action, but follow-up actions must wait for the next loop turn so runtime can revalidate state.",
            "Use set_status(action=...) only to publish the next action/state after this turn; it does not execute that next action immediately.",
        ]
        if missing_outputs:
            targets = "\n".join(f"- {path}" for path in missing_outputs[:8])
            more = "" if len(missing_outputs) <= 8 else f"\n- ... {len(missing_outputs) - 8} more"
            parts.append(f"Missing required outputs:\n{targets}{more}")
            if plan and plan.get("action") == "work":
                parts.append(
                    "Current action is work. Do the work that best advances the task within the work tool scope. "
                    "If a different action is needed, leave it for the next loop turn so runtime can revalidate state."
                )
            if agent._write_only_miss_count:
                parts.append(
                    "Previous write-only artifact turns produced assistant text but no executable file_write call. "
                    "Plain prose does not create the missing files. Runtime will narrow this retry turn to file_write "
                    "so the selected work action creates the required output files."
                )
        elif _expected_output_paths(agent.task):
            parts.append(
                "All explicit output files named by your task currently exist. Prefer action=compact or action=stop: "
                "publish a short summary, tags, and artifact paths. Do not continue open-ended verification unless it changes the deliverable."
            )
        blockers = self.completion_blockers(agent, include_missing_outputs=False)
        if blockers:
            blocker_lines = "\n".join(f"- {b['message']}" for b in blockers[:6])
            parts.append(
                "Completion blockers are active. Do not claim done, compact with stop_after=true, or write a final-only report "
                "until these are resolved:\n"
                f"{blocker_lines}"
            )
        if plan and plan.get("action") == "create":
            parts.append(
                "Current action is create. Decide whether to form or adjust the agent structure. "
                "If you continue solo, compact/update memory with status:solo-decision and a concrete reason."
            )
            if plan.get("reason") == "resume_create_after_read_orientation":
                parts.append(
                    "You requested read orientation during a create turn and have now had that read turn. "
                    "Return to structure formation now: use spawn_many()/spawn() for the separable peer wave unless the task is truly atomic."
                )
            elif plan.get("reason") == "retry_spawn_after_create_miss":
                parts.append(
                    "Previous create turns did not create any child agents. Runtime will narrow this retry turn to spawn/create_agent/spawn_many. "
                    "Do not write directory placeholders or only set action=create again; create the peer wave now, or compact a solo-decision if spawning is truly wrong."
                )
            elif plan.get("reason") == "multi_phase_peer_wave_pending":
                parts.append(
                    "This task describes a multi-stage peer protocol and previous child waves are no longer active while final outputs remain missing. "
                    "Create the next comparison, mutation, review, or synthesis wave as appropriate. Do not collapse the remaining protocol into a solo final artifact unless budget exhaustion makes the incomplete protocol explicit."
                )
            if plan.get("reason") in {
                "delegate_integration_after_child_source_progress",
                "delegate_test_recovery_after_integration_stall",
            }:
                parts.append(
                    "Coordinator boundary: child agents have produced shared/source progress, but test/integration evidence is still missing. "
                    "Create one integration/test/review agent to query implementers, inspect the combined diff, run focused tests, and fix integration issues if needed. "
                    "Do not inspect implementation files or run the diff yourself in this root turn."
                )
            elif plan.get("reason") == "delegate_source_recovery_after_child_stall":
                parts.append(
                    "Coordinator boundary: child work did not produce the required shared/source change. "
                    "Create a focused recovery/implementation agent with the missing blocker details instead of doing the patch yourself in this root turn."
                )
        if plan and plan.get("action") == "message" and str(plan.get("reason", "")).startswith("coordinate_"):
            parts.append(
                "Coordinator boundary: you already have child agents. Use query() or wait() to inspect their state_board, public memory, artifacts, and progress. "
                "Do not inspect or edit implementation files yourself in this root/coordinator turn. "
                "If their progress makes integration or recovery work necessary, leave that for the next create turn after runtime revalidation."
            )
        if self._peer_progress_decision_pending(agent):
            query_filter = self._peer_progress_query_filter(agent)
            tags = self._peer_progress_query_tags(agent)
            parts.append(
                "Peer-progress risk is active. Choose action=read before more solo work if peer state may change your next step. "
                f"Relevant peer filter={json.dumps(query_filter, ensure_ascii=False)}, "
                f"tags={json.dumps(tags, ensure_ascii=False)}. "
                "If peers already cover your lane or your candidate is invalid/duplicate, use action=compact "
                "with stop_after=true and tags including status:pruned plus reason:peer_ahead, "
                "reason:duplicate_lane, or reason:invalid_candidate."
            )
        elif self._peer_progress_post_query_pending(agent):
            peers = self._completed_peer_progress_candidates(agent)
            peer_lines = "\n".join(
                f"- {peer.id}: status={peer.status} artifacts={','.join(a.path for a in peer.artifacts[:3]) or '-'} "
                f"summary={(peer.result or self.memory.serialize(peer.id).get('public_summary') or '')[:140]}"
                for peer in peers[:6]
            )
            parts.append(
                "Peer-progress query has already returned completed or artifact-bearing peers in your lane. "
                "Do not keep reading peer files just to re-verify the same point. Choose one action now: "
                "action=compact with stop_after=true and tags including status:pruned plus reason:peer_ahead "
                "if your lane is duplicate or no longer useful; or action=work "
                "if your output remains distinct. If unsure, prefer pruning with a useful summary over more analysis.\n"
                f"Completed/advanced peers:\n{peer_lines}"
            )
        if plan:
            parts.append(
                f"Next action candidate: action={plan.get('action')} reason={plan.get('reason', '-')}. "
                "Re-check this against current state before using it."
            )
        if transient:
            parts.append(
                "Recent runtime signals:\n"
                + "\n\n".join(self._loop_message_content(msg) for msg in transient[:4])
            )
        return {"role": "user", "content": "\n\n".join(parts)}

    def _execute_compact(self, agent: Agent):
        params = agent._compact_pending
        agent._compact_pending = None
        if params.get("stop_after"):
            guard_tags = list(params.get("tags", [])) or list(agent.current_task_tags)
            blockers = self.completion_blockers(agent, tags=guard_tags)
            if blockers:
                self._emit(agent.id, "completion_blocked", {
                    "reason": "compact_stop_after",
                    "blockers": blockers,
                })
                self.state_board_update(agent.id, action_state="work", current_task_tags=guard_tags)
                return
        summary = params["summary"]
        files = list(params.get("files", []))
        tags = list(params.get("tags", []))
        experience = params.get("experience")
        new_task = params.get("new_task")
        new_bio = params.get("new_bio")
        work_outline = params.get("work_outline", agent.work_outline)
        if new_bio:
            agent.bio = new_bio
        if new_task:
            agent.task = new_task
        self.state_board_update(
            agent.id,
            action_state="compact",
            current_task_tags=tags or agent.current_task_tags,
            work_outline=work_outline,
        )
        system_msg = agent.history[0] if agent.history else None
        compact_message = f"[Compact summary]\n\n{summary}"
        agent.history = ([system_msg] if system_msg else []) + [{"role": "system", "content": compact_message}]
        agent.context_tokens = count_message_tokens(agent.history)
        update_kwargs = {
            "public_summary": summary,
            "tags": tags or agent.current_task_tags,
            "add_artifacts": files,
            "active_task": ActiveTaskCard(task=new_task or agent.task, tags=tags or agent.current_task_tags, work_outline=work_outline),
        }
        if experience:
            update_kwargs["add_experience"] = ExperienceCard(summary=experience, tags=tags or agent.current_task_tags, artifacts=files)
        self.memory.update(agent.id, **update_kwargs)
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        if params.get("stop_after"):
            agent.action_state = "stop"
            agent.status = "done"
            agent.result = params.get("stop_result") or agent.result
            stop_experience = None if experience else ExperienceCard(summary=summary, tags=tags or agent.current_task_tags, artifacts=files)
            self.memory.update(
                agent.id,
                clear_active_task=True,
                add_experience=stop_experience,
            )
            agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        self.state_board_sync(agent.id)

    def _execute_rebirth(self, agent: Agent):
        params = agent._rebirth_pending
        agent._rebirth_pending = None
        summary = params["summary"]
        files = params.get("files", [])
        new_task = params.get("new_task")
        new_bio = params.get("new_bio")
        tags = list(params.get("tags", []))
        experience = params.get("experience")

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
        if tags:
            agent.current_task_tags = tags

        file_section = ""
        if files:
            file_section = "\n\nKey files:\n" + "\n".join(f"- {f}" for f in files)
        agent.history.append({"role": "user", "content": (
            f"[Rebirth — context reset]\n\n## Progress\n{summary}{file_section}"
        )})
        agent.context_tokens = count_message_tokens(agent.history)
        self.memory.update(
            agent.id,
            public_summary=summary,
            active_task=ActiveTaskCard(task=new_task or agent.task, tags=agent.current_task_tags, work_outline=agent.work_outline),
            tags=tags or agent.current_task_tags,
            add_artifacts=files,
            add_experience=ExperienceCard(summary=experience, tags=tags or agent.current_task_tags, artifacts=files) if experience else None,
        )
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        self.state_board_sync(agent.id)

    async def _execute_tool(self, tc: ToolCall, agent: Agent, tools: dict) -> Any:
        tool_info = tools.get(tc.name)
        if not tool_info:
            return {"error": f"Unknown tool: {tc.name}"}
        handler = tool_info["handler"]
        try:
            if tool_info.get("is_meta"):
                return await handler(tc.arguments, agent, self)
            else:
                return await handler(tc.arguments, agent.workspace, self._tool_context)
        except Exception as e:
            return {"error": str(e)}

    def _available_work_tools(self) -> dict[str, dict[str, Any]]:
        enabled = self._tool_context.enabled_work_tools
        if enabled is None:
            tools = dict(WORK_TOOLS)
        else:
            tools = {name: tool for name, tool in WORK_TOOLS.items() if name in enabled}
        if self._tool_context.shell_mode == "disabled":
            tools.pop("shell", None)
        return tools

    async def _run_bootstrap_batch(self, root: Agent) -> None:
        from nanoma.meta import meta_batch

        result = await meta_batch({"path": self.config.bootstrap_batch_file}, root, self)
        self._emit(root.id, "bootstrap_batch", {
            "path": self.config.bootstrap_batch_file,
            "executed": result.get("executed"),
            "error": result.get("error"),
        })
        if result.get("error"):
            root.history.append({
                "role": "user",
                "content": f"[Bootstrap batch failed]\n{json.dumps(result, ensure_ascii=False, default=str)}",
            })
            return
        root.history.append({
            "role": "user",
            "content": (
                "[Bootstrap batch executed before your first turn]\n"
                f"{json.dumps(result, ensure_ascii=False, default=str)[:4000]}\n\n"
                "Continue by querying spawned agents and coordinating integration. Do not respawn duplicates."
            ),
        })

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

    def _build_system_prompt(
        self,
        agent_id: str,
        task: str,
        workspace: Path,
        parent_context: dict | None = None,
        orchestration_preference: OrchestrationPreference | None = None,
    ) -> str:
        shared = self._tool_context.shared_dir
        orchestration_preference = orchestration_preference or self.config.orchestration_preference
        time_info = ""
        if self.config.time_limit > 0:
            time_info = f"\nTime limit: {self.config.time_limit:.0f}s total. Check with get_cost()."
        shell_guidance = self._shell_prompt_section()
        artifact_section = self._artifact_prompt_section(task)

        # Context about spawner (for sub-agents)
        context_section = ""
        if parent_context:
            if self.config.notify_parent_on_done:
                finish_guidance = (
                    '- When you finish, update your public memory/tags, send your spawner a concise '
                    'message with results, then call set_status("done") or compact(..., stop_after=true).'
                )
            else:
                finish_guidance = (
                    '- Parent-child lineage is structural. Do not report to the spawner by default; '
                    'keep public memory/tags current so peers can query your status and summarize from query(). '
                    'Call set_status("done") when complete, or compact(..., stop_after=true) when your final act is memory compaction.'
                )
            context_section = f"""
## Your Context
- Spawned by: agent "{parent_context['parent_id']}" (task: {parent_context['parent_task']})
- Peers working in parallel: {parent_context['siblings'] or 'none yet'}
- Depth: {parent_context['depth']}
{finish_guidance}
"""

        orchestration_section = self._orchestration_prompt_section(orchestration_preference)

        return f"""You are agent "{agent_id}" in a multi-agent system.

Task: {task}
Workspace: {workspace} (private to you)
Shared: {shared} (visible to all agents){time_info}
{context_section}
Coordination guidance: use query() to inspect other agents' state_board, role, group_id, tags, artifacts, progress, and public_memory before summarizing shared work. For peer waves, prefer query(filter={{"group_id": "<exact group_id>"}}) or query(tags=["feature:N"]); if query returns zero results, read query_help and retry with the advertised group_id/tags. Direct send() is for explicit coordination, not the default completion path. If you hit a bottleneck, have repeated no-tool/prose turns, or suspect duplicate work, proactively query nearby peers and compare progress/artifacts before continuing. If another peer is clearly ahead or already covers your lane, compact your partial findings with stop_after=true and tags such as status:pruned plus reason:peer_ahead/reason:duplicate_lane so the summary remains discoverable. If compacting memory is your last step, use compact(..., stop_after=true, result="...") or set_status(..., compact_before_stop=true); plain compact() means you intend to continue.
{orchestration_section}
Editing guidance: use file_replace for source edits. file_write is for new files or deliberate guarded overwrites; existing large files require overwrite=true or expected_sha256 and can otherwise be rejected to prevent accidental whole-file truncation.
{artifact_section}
{shell_guidance}
Workflow prior guidance: research_loop / critic_review_loop / synthesis_loop are prompt patterns only, not workflow engines.
"""

    def _artifact_prompt_section(self, task: str) -> str:
        expected = _expected_output_paths(task)
        if not expected:
            return ""
        targets = "\n".join(f"- {path}" for path in expected[:8])
        more = "" if len(expected) <= 8 else f"\n- ... {len(expected) - 8} more"
        return (
            "Artifact guidance: this task has explicit required output files. "
            "After reading the minimum necessary inputs, write those files with file_write; "
            "do not leave the deliverable only in chat. If several files are required, write all of them, "
            "for example source plus report/notes. If uncertain, write the best current artifact and explain uncertainty "
            "inside the required report file.\n"
            f"Required output files:\n{targets}{more}"
        )

    def _shell_prompt_section(self) -> str:
        mode = self.config.shell_mode
        if mode == "disabled":
            return "Shell guidance: shell is disabled. Do not try to execute commands by writing scripts; use file tools and clearly state when verification is static only."
        if mode == "controlled":
            allowed = self.config.controlled_shell_allowed_commands or {"python", "python3", "python3.10", "python3.11", "python3.12", "pytest", "git"}
            return (
                "Shell guidance: controlled shell is available for verification commands only. "
                f"Allowed command names: {', '.join(sorted(allowed))}. "
                "Use direct commands such as `python3 -m pytest tests -q`, `pytest tests -q`, or `git -C $SHARED/source status`. "
                "A single prefix of `cd $SHARED/source && <allowed command>` is also accepted and normalized to that working directory; for example `cd $SHARED/source && python3 -m pytest tests -q`. "
                "Other shell operators, redirection, package installs, network tools, or destructive filesystem operations are rejected."
            )
        return "Shell guidance: shell is available. Prefer focused verification commands with short timeouts; avoid broad destructive or network-heavy commands unless the task explicitly requires them."

    def _orchestration_prompt_section(self, preference: OrchestrationPreference) -> str:
        threshold = self.config.min_spawnable_workstreams
        if preference == "solo":
            behavior = (
                "Prefer single-agent execution. Do not spawn unless the user explicitly asks for multi-agent work, "
                "the task cannot be completed by one agent, or an independent review is required for correctness."
            )
        elif preference == "balanced":
            behavior = (
                f"Start solo for brief orientation, but form a peer wave early with spawn_many() when the task has "
                f"{threshold}+ separable workstreams or benefits from independent review."
            )
        elif preference == "parallel":
            behavior = (
                f"Bias toward early peer-wave creation. For {threshold}+ separable workstreams, call spawn_many() "
                "before detailed implementation and let the root agent coordinate/query/synthesize."
            )
        else:
            behavior = (
                "Strongly prefer a peer wave for nontrivial work. Use spawn_many() early unless the task is atomic; "
                "the root agent should mainly coordinate, query peer state, and synthesize."
            )
        return (
            "Orchestration preference: "
            f"{preference}. {behavior} Good peer waves include evidence checks, calculations, implementation, "
            "tests, critique, risk review, and final synthesis. Give peers role/group_id/current_task_tags."
        )

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
                aid: {"status": a.status, "task": a.task[:80], "bio": a.bio, "turns": a._turns, "action_state": a.action_state}
                for aid, a in self.agents.items()
            },
            "cost": self.ledger.summary(),
            "scheduler": self.scheduler.stats,
            "state_board": self.state_board_list(),
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

        # Peak concurrency (estimate from lifecycle events)
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
        query_events = 0
        for e in self._events:
            if e["event"] == "tool_call":
                name = e["data"].get("tool", "?")
                tool_counts[name] = tool_counts.get(name, 0) + 1
            elif e["event"] == "query":
                query_events += 1
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
                "query_count": tool_counts.get("query", 0) + query_events,
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
                    "orchestration_preference": a.orchestration_preference,
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
        await self.sandbox.stop()
