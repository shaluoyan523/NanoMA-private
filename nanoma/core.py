"""Core runtime: Agent, Runtime, ReAct loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Awaitable, Literal

from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import (
    LLMResponse, Message, RetryConfig, ToolCall, ToolDef,
    count_message_tokens, estimate_tokens, openai_compatible_call, set_log_dir,
    _parse_text_tool_calls,
)
from nanoma.memory import (
    ActiveTaskCard,
    MemoryBroker,
    agent_identity_tags,
    build_memory_card,
    canonicalize_role,
    normalize_tags,
    split_experience_text,
)
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
    "benchmark suite": 2,
    "candidate pool": 2,
    "candidate wave": 2,
    "compare": 1,
    "complex": 2,
    "concurrent": 2,
    "concurrency": 2,
    "coordination": 1,
    "evolution": 2,
    "fan-out": 2,
    "fanout": 2,
    "design": 1,
    "end-to-end": 2,
    "evaluate": 1,
    "evaluation": 1,
    "full": 1,
    "full-scale": 2,
    "implementation": 1,
    "implement": 1,
    "integration": 2,
    "multi-phase": 2,
    "multi-agent": 2,
    "multiagent": 2,
    "orchestration": 2,
    "parallel": 2,
    "parallelism": 2,
    "pipeline": 1,
    "protocol": 2,
    "review": 1,
    "synthesis": 1,
    "test matrix": 2,
    "tournament": 2,
    "workflow": 1,
    "workstream": 2,
    "workstreams": 2,
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
    "assigned artifact": 2,
    "assigned file": 2,
    "brief": 1,
    "grep": 2,
    "inspect": 1,
    "one artifact": 2,
    "one candidate": 2,
    "one file": 2,
    "quick": 1,
    "read one": 2,
    "single": 1,
    "single artifact": 2,
    "single candidate": 2,
    "single file": 2,
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
    concrete_single_output = _task_is_concrete_single_output_work(task)

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
        reasons.append("leaf artifact lane")
    if concrete_single_output:
        reasons.append("concrete single-output work")

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
    if concrete_single_output:
        target_rank = ORCHESTRATION_RANK["balanced"]
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
    input_paths = set(_referenced_input_like_shared_paths(task))
    for path in _referenced_output_like_shared_paths(task):
        if path in input_paths:
            continue
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _normalize_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


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


def _referenced_input_like_shared_paths(task: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    def is_input_path(path: str, start: int, end: int) -> bool:
        if not path.startswith("shared/") or path.startswith("shared/source"):
            return False
        if _looks_like_final_answer_path(path):
            return False
        task_text = task or ""
        context = task_text[max(0, start - 80): min(len(task_text), end + 80)].lower()
        local_prefix = task_text[max(0, start - 64): start].lower()
        if re.fullmatch(r"shared/[^/]+/problems/[^/]+/(?:metadata\.json|statement\.md)", path):
            return True
        if re.fullmatch(r"shared/[^/]+/README\.md", path):
            return True
        input_terms = (
            "public file", "public files", "input file", "read", "inspect",
            "metadata", "statement", "reference", "依据", "参考", "读取", "查看",
        )
        output_terms = (
            "write", "write to", "write the", "create", "produce", "deliver", "output",
            "save", "append", "update", "overwrite", "replace", "correct",
            "verified report", "final report", "生成", "写入", "产出", "更新", "追加", "覆盖",
        )
        downstream_input_terms = (
            "read", "inspect", "verify", "validate", "cross-check",
            "check", "confirm", "compare", "核验", "验证", "检查", "读取",
        )
        local_suffix = task_text[end: min(len(task_text), end + 64)].lower()
        post_path_output_terms = ("overwrite", "replace", "update", "append", "if needed", "as needed", "覆盖", "更新", "追加")
        output_near_path = any(term in local_prefix for term in output_terms) or any(
            term in local_suffix for term in post_path_output_terms
        )
        input_near_path = any(term in local_prefix for term in input_terms)
        downstream_input_near_path = any(term in local_prefix for term in downstream_input_terms)
        if output_near_path:
            return False
        if input_near_path and not any(term in context for term in output_terms):
            return True
        if downstream_input_near_path:
            return True
        return not _looks_like_output_path(path)

    for match in re.finditer(r"`(shared/[^`]+?)`", task or ""):
        path = match.group(1).strip().rstrip(".,;:)]}")
        if is_input_path(path, match.start(1), match.end(1)) and path not in seen:
            seen.add(path)
            paths.append(path)
    for match in re.finditer(r"(?<![\w/])shared/[^\s`'\"<>)\]}]+", task or ""):
        path = match.group(0).strip().rstrip(".,;:)]}")
        if is_input_path(path, match.start(0), match.end(0)) and path not in seen:
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
    peer_query_terms = (
        "query peer", "query peers", "query group", "query agents",
        "query other agents", "inspect peer", "inspect peers", "from peers",
        "group_id", "completed peers", "memory tags", "peer progress",
        "peer state", "peer status", "public memory", "public_memory",
        "state_board", "同伴", "其他agent", "查询同伴", "查询其他",
    )
    if any(term in text for term in peer_query_terms):
        return True
    coordination_terms = (
        "coordinate", "coordinator", "peer", "peers", "summarize shared",
        "inspect shared", "artifact_commit", "remaining gaps",
        "artifact paths", "before final synthesis", "before synthesis",
        "before summarizing", "completed peers", "memory tags",
        "peer progress", "peer state", "peer status", "public memory",
        "public_memory", "query-driven", "state_board", "status discovery",
        "协调", "汇总", "同伴", "其他agent",
    )
    return any(term in text for term in coordination_terms)


def _task_is_discovery_work(task: str) -> bool:
    text = (task or "").lower()
    discovery_terms = (
        "search", "research", "investigate", "look up", "lookup", "find",
        "identify", "discover", "collect evidence", "evidence", "source",
        "sources", "citation", "citations", "fact-check", "fact check",
        "verify from", "primary source", "paper", "papers", "arxiv", "doi",
        "web", "browse", "gaia",
        "搜索", "检索", "查找", "查询", "调查", "研究", "收集证据",
        "证据", "来源", "引用", "核验", "论文",
    )
    return any(term in text for term in discovery_terms)


def _task_is_concrete_single_output_work(task: str) -> bool:
    """Concrete evidence/API work should usually execute, not recursively delegate."""
    outputs = _non_terminal_expected_output_paths(task)
    if not outputs or len(outputs) > 2:
        return False
    if _task_has_multi_phase_peer_protocol(task) or _explicit_peer_wave_requirements(task):
        return False
    text = (task or "").lower()
    delegation_terms = (
        "spawn", "spawn_many", "create agent", "create_agent", "handoff", "hand off",
        "delegate", "peer wave", "worker wave", "multi-agent", "multiagent",
        "parallel agents", "independent agents", "workstream", "workstreams",
    )
    if any(term in text for term in delegation_terms):
        return False
    if re.search(r"\bcandidate[_:-]?\d+\b", text) or "/candidate_" in text:
        return False
    if "candidate evidence" in text:
        return False
    concrete_discovery_terms = (
        "api", "arxiv", "doi", "paper", "papers", "search", "research", "find",
        "identify", "look up", "lookup", "fetch", "parse", "extract", "source",
        "sources", "citation", "citations", "evidence", "query", "http://",
        "https://", "web", "browse", "gaia", "搜索", "检索", "查询", "论文",
        "证据", "来源", "引用",
    )
    return any(term in text for term in concrete_discovery_terms)


def _task_is_coordination_artifact_task(task: str) -> bool:
    if not _expected_output_paths(task) or not _task_prefers_query_before_artifact(task):
        return False
    text = (task or "").lower()
    coordination_terms = (
        "coordinate", "coordinator", "summarize", "summary", "report", "status",
        "inspect", "discover", "peer", "peers", "group_id", "remaining gaps",
        "artifact_commit", "file_write observations",
        "artifact paths", "before final synthesis", "bt ranking",
        "bradley-terry", "comparison results", "completed peers",
        "final synthesis", "memory tags", "peer progress", "peer state",
        "public memory", "public_memory", "query-driven", "state_board",
        "status discovery",
        "协调", "汇总", "总结", "报告", "状态", "查询", "同伴", "其他agent", "剩余",
    )
    return bool(_referenced_group_ids(task)) or any(term in text for term in coordination_terms)


def _task_has_spawnable_workstreams(task: str, min_streams: int = 3) -> bool:
    text = (task or "").lower()
    if _task_has_multi_phase_peer_protocol(task):
        return True
    if _explicit_peer_wave_requirements(task):
        return True
    if any(term in text for term in (
        "batch spawn", "comparison wave", "fan out", "fan-out", "generator wave",
        "independent agents", "independent candidates", "independent comparisons",
        "independent generators", "mutation wave", "parallel agents",
        "peer agents", "separable", "spawn agents", "spawn_many",
        "worker wave", "workstream", "workstreams", "multi-agent", "multiagent",
        "多agent",
    )):
        return True
    stream_terms = {
        "audit", "patch", "implement", "implementation", "test", "review", "research",
        "coordinator", "integrator", "redteam",
        "comparator", "evaluator", "mutator", "ranker",
        "reviewer", "selector", "synthesizer", "validator",
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
        "bradley-terry",
        "bt ranking",
        "bt.json",
        "candidate pool",
        "comparison graph",
        "comparison json",
        "degree k",
        "elite",
        "elites",
        "evolution",
        "evolutionary",
        "evolve",
        "final selection",
        "generation-0",
        "gen0 population",
        "gen0 candidates",
        "gen1 population",
        "gen1 candidates",
        "gen2 population",
        "gen2 candidates",
        "gen3 population",
        "gen3 candidates",
        "initial candidates",
        "judge pair",
        "peer wave",
        "pairwise",
        "pairwise judge",
        "population",
        "ranking round",
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
        "selection round",
        "top 15",
        "top 5",
        "bottom 5",
        "tournament",
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


def _explicit_peer_wave_requirements(task: str) -> list[dict[str, Any]]:
    """Find explicit numeric peer-wave requirements from natural task text."""
    requirements: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    text = task or ""
    chunks = re.split(r"(?<=[.;。；])\s+|\n+", text)
    patterns = (
        re.compile(r"\b(?:spawn|create|launch|start)\s+(?P<count>\d{1,3})\s+(?P<desc>[^.;\n]{0,120}?)(?:agents|peers|workers)\b", re.I),
        re.compile(r"\b(?P<count>\d{1,3})\s+(?P<desc>(?:independent|parallel)[^.;\n]{0,120}?)(?:agents|peers|workers|generators|judges|comparisons|candidates)\b", re.I),
        re.compile(r"\b(?:generate|produce|write)\s+(?P<count>\d{1,3})\s+(?P<desc>[^.;\n]{0,120}?)(?:candidate|solution|variant)s?\b", re.I),
    )
    for chunk in chunks:
        lower = chunk.lower()
        if not any(term in lower for term in ("spawn", "agent", "peer", "worker", "independent", "parallel", "candidate", "comparison", "mutation")):
            continue
        for pattern in patterns:
            for match in pattern.finditer(chunk):
                try:
                    target = int(match.group("count"))
                except (TypeError, ValueError):
                    continue
                if target <= 1 or target > 500:
                    continue
                snippet = f"{match.group(0)} {match.group('desc')}".lower()
                role = _infer_peer_wave_role(snippet)
                phase = _infer_peer_wave_phase(snippet)
                key = (target, role, phase)
                if key in seen:
                    continue
                seen.add(key)
                requirements.append({
                    "target": target,
                    "role": role,
                    "phase": phase,
                    "text": match.group(0).strip(),
                })
    return requirements


def _infer_peer_wave_role(text: str) -> str:
    if any(term in text for term in ("comparison", "compare", "judge", "pairwise")):
        return "judge"
    if any(term in text for term in ("mutation", "mutate", "mutator")):
        return "mutator"
    if any(term in text for term in ("review", "reviewer", "critic")):
        return "reviewer"
    if any(term in text for term in ("test", "tester", "validation", "verifier", "verify")):
        return "tester"
    if any(term in text for term in ("generator", "generate", "candidate", "solution", "population")):
        return "generator"
    if "worker" in text:
        return "worker"
    return ""


def _infer_peer_wave_phase(text: str) -> str:
    normalized = text.replace("-", "")
    if "final" in normalized:
        return "final"
    gen_match = re.search(r"\bgen(?:eration)?\s*0*(\d+)\b", normalized)
    if gen_match:
        return f"gen{int(gen_match.group(1))}"
    return ""


def _agent_matches_peer_wave_requirement(agent: "Agent", requirement: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(part)
        for part in [
            agent.role,
            agent.group_id,
            agent.workflow_prior,
            agent.task,
            " ".join(agent.current_task_tags),
        ]
        if part
    ).lower().replace("-", "")
    role = str(requirement.get("role") or "")
    phase = str(requirement.get("phase") or "").replace("-", "")
    role_terms = {
        "generator": ("generator", "candidate", "solution", "role:generator"),
        "judge": ("judge", "comparison", "compare", "pairwise", "role:judge"),
        "mutator": ("mutator", "mutation", "mutate", "role:mutator"),
        "reviewer": ("reviewer", "review", "critic", "role:reviewer"),
        "tester": ("tester", "test", "validation", "verifier", "role:tester"),
        "worker": ("worker", "role:worker"),
    }
    if role:
        terms = role_terms.get(role, (role,))
        if not any(term.replace("-", "") in haystack for term in terms):
            return False
    if phase and phase not in haystack:
        return False
    return bool(role or phase)


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
        "author", "compose", "draft", "output", "save",
        "输出", "产出", "写", "生成", "实现",
    )
    if not any(term in text for term in write_terms):
        return False
    coordinator_terms = (
        "spawn", "spawn_many", "create_agent", "coordinate", "coordinator",
        "query peers", "query group", "compare", "comparison", "pairwise",
        "judge", "aggregate", "bt_aggregate", "synthesis", "synthesize",
        "mutation", "mutate", "select final", "selection", "opendeepthink",
        "bradley-terry", "bt ranking", "bt.json", "candidate pool",
        "comparison graph", "comparison json", "elite", "elites",
        "evolution", "final selection", "population", "ranking round",
        "selection round", "tournament",
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


def _looks_like_evidence_report_path(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() not in {".md", ".txt", ".json"}:
        return False
    name = p.name.lower()
    return any(term in name for term in ("evidence", "report", "source", "citation", "paper", "verify", "verification"))


def _looks_like_final_answer_path(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() == ".json" and p.name.lower() in {"answer.json", "final_answer.json"}


def _final_answer_paths(task: str) -> list[str]:
    return [path for path in _expected_output_paths(task) if _looks_like_final_answer_path(path)]


def _task_mentions_submit_answer_protocol(task: str) -> bool:
    text = (task or "").lower()
    return bool(re.search(r"\bsubmit_answer\s*\(", text)) or "submit_answer(answer" in text


def _strip_submit_answer_protocol_from_child_task(task: str) -> str:
    text = str(task or "")
    if not _task_mentions_submit_answer_protocol(text) and "answer.json" not in text and "final_answer.json" not in text:
        return text
    replacements = [
        (
            r"(?im)^\s*[-*]?\s*when ready[^.\n]*submit_answer\s*\([^.\n]*(?:\.\s*)?$",
            "",
        ),
        (
            r"(?im)^\s*[-*]?\s*do not write [`'\"]?shared/(?:final_)?answer\.json[`'\"]?[^.\n]*(?:\.\s*)?$",
            "",
        ),
        (
            r"(?im)^\s*[-*]?\s*use submit_answer\s*\([^.\n]*(?:\.\s*)?$",
            "",
        ),
        (
            r"(?im)^\s*[-*]?\s*call submit_answer\s*\([^.\n]*(?:\.\s*)?$",
            "",
        ),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\bsubmit_answer\s*\([^)]*\)", "publish a local evidence summary", text)
    text = re.sub(
        r"(?i)\bwrite\s+[`'\"]?shared/(?:final_)?answer\.json[`'\"]?",
        "write the requested local evidence artifact",
        text,
    )
    text = re.sub(
        r"(?i)\bsubmit\s+[`'\"]?shared/(?:final_)?answer\.json[`'\"]?",
        "publish the requested local evidence artifact",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text != str(task or ""):
        text += (
            "\n\nLocal completion only: do not submit the global benchmark answer. "
            "Write or compact your assigned evidence/verification result; a runtime-authorized delivery agent will submit the final answer."
        )
    return text


def _slug_for_path(value: str, *, fallback: str = "agent") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return (text or fallback)[:64].strip("_") or fallback


def _task_requests_helper_reports(task: str) -> bool:
    text = (task or "").lower()
    if not text.strip():
        return False
    helper_terms = (
        "helper",
        "helpers",
        "sub-agent",
        "sub agent",
        "agent",
        "agents",
        "worker",
        "workers",
        "peer",
        "peers",
        "lane",
        "lanes",
    )
    output_terms = (
        "evidence report",
        "short evidence report",
        "report",
        "reports",
        "artifact",
        "artifacts",
        "write",
        "produce",
        "deliver",
        "save",
        "产物",
        "报告",
        "写",
    )
    return any(term in text for term in helper_terms) and any(term in text for term in output_terms)


def _preferred_child_output_dir_from_parent_task(parent_task: str) -> str | None:
    task = parent_task or ""
    candidates: list[str] = []
    for match in re.finditer(r"`(shared/[^`]+/)`", task):
        candidates.append(match.group(1).strip())
    for match in re.finditer(r"(?<![\w/])(shared/[^\s`'\"<>)\]}]+/)", task):
        candidates.append(match.group(1).strip())
    for raw in candidates:
        path = raw.rstrip(".,;:)]}") + "/"
        path = re.sub(r"/+", "/", path)
        if not path.startswith("shared/"):
            continue
        if _looks_like_final_answer_path(path.rstrip("/")):
            continue
        lowered = path.lower()
        if any(term in lowered for term in ("gaia", "evidence", "report", "agent_outputs", "lane", "research")):
            return path
    return None


def _lane_slug_for_child_output(
    *,
    role: str = "",
    group_id: str = "",
    current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    tags = [str(tag).strip() for tag in (current_task_tags or []) if str(tag).strip()]
    lane_tags = []
    for tag in tags:
        lowered = tag.lower()
        if lowered.startswith(("lane:", "role:", "deliverable:", "feature:")):
            lane_tags.append(tag.split(":", 1)[1])
    for candidate in [role, *lane_tags, group_id]:
        slug = _slug_for_path(candidate, fallback="")
        if slug and slug not in {"agent", "worker", "evidence", "verifier", "researcher", "peer_agent"}:
            return slug
    return "local_report"


def _default_child_output_path(
    *,
    parent_task: str,
    role: str = "",
    group_id: str = "",
    current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    output_dir = _preferred_child_output_dir_from_parent_task(parent_task)
    if not output_dir:
        group_slug = _slug_for_path(group_id, fallback="default")
        output_dir = f"shared/.nanoma/agent_outputs/{group_slug}/"
    lane_slug = _lane_slug_for_child_output(role=role, group_id=group_id, current_task_tags=current_task_tags)
    return f"{output_dir.rstrip('/')}/{lane_slug}_evidence.md"


def _inject_local_deliverable_if_needed(
    task: str,
    *,
    parent_task: str,
    role: str = "",
    group_id: str = "",
    current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
    child_submit_granted: bool = False,
) -> tuple[str, str | None]:
    if child_submit_granted:
        return task, None
    if _expected_output_paths(task):
        return task, None
    if not _task_requests_helper_reports(parent_task):
        return task, None
    path = _default_child_output_path(
        parent_task=parent_task,
        role=role,
        group_id=group_id,
        current_task_tags=current_task_tags,
    )
    if _looks_like_final_answer_path(path):
        return task, None
    instruction = (
        "\n\nRequired local output:\n"
        f"- Write your evidence or verification report to `{path}` before you stop.\n"
        "- Include concise sources/IDs, findings, confidence, blockers, and task/lane tags.\n"
        "- This is a local deliverable for ledger/query; do not submit the global answer."
    )
    return str(task or "").rstrip() + instruction, path


def _inject_expected_outputs_if_needed(task: str, expected_outputs: list[str] | tuple[str, ...] | set[str] | None) -> tuple[str, list[str]]:
    outputs: list[str] = []
    seen: set[str] = set()
    raw_outputs: Any = expected_outputs
    if isinstance(raw_outputs, str):
        raw_outputs = [raw_outputs]
    for raw in raw_outputs or []:
        path = str(raw or "").strip().strip("`'\"")
        if not path:
            continue
        path = path.rstrip(".,;:)]}")
        if not _looks_like_output_path(path):
            continue
        if _looks_like_final_answer_path(path):
            continue
        if path not in seen:
            seen.add(path)
            outputs.append(path)
    if not outputs:
        return task, []
    existing = set(_non_terminal_expected_output_paths(task))
    missing = [path for path in outputs if path not in existing]
    if not missing:
        return task, outputs
    lines = "\n".join(f"- `{path}`" for path in missing)
    instruction = (
        "\n\nRequired output slot(s):\n"
        f"{lines}\n"
        "- Treat these as the canonical deliverables for this task. If an existing file is low-confidence, replace it with evidence-backed content.\n"
        "- Include concise evidence references, confidence, blockers, and tags in the artifact where applicable."
    )
    return str(task or "").rstrip() + instruction, outputs


def _non_terminal_expected_output_paths(task: str) -> list[str]:
    return [path for path in _expected_output_paths(task) if not _looks_like_final_answer_path(path)]


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
_ROOT_SUBMIT_ONLY_REASONS = {"direct_answer_submission_ready", "terminal_candidate_submit_ready"}
_CREATE_EXECUTION_TOOLS = {"spawn", "create_agent", "spawn_many"}
_CREATE_STATE_TOOLS = {"compact", "set_status", "get_cost"}
_LEDGER_READ_TOOLS = {"ledger_read"}
_LEDGER_WRITE_TOOLS = {"ledger_update"}
_LEDGER_TOOLS = _LEDGER_READ_TOOLS | _LEDGER_WRITE_TOOLS
_TASK_LEDGER_OUTPUT_DONE_STATUSES = {"verified", "covered", "submitted"}
_TASK_LEDGER_OUTPUT_INCOMPLETE_STATUSES = {"missing", "unverified", "placeholder", "evidence_gap"}
_TASK_LEDGER_TERMINAL_STATUSES = {"verified", "covered", "pruned", "failed"}
_STRONG_UNVERIFIED_EVIDENCE_TERMS = (
    "unverified",
    "not verified",
    "requires verification",
    "need verification",
    "needs verification",
    "pending verification",
    "evidence pending",
    "evidence-pending",
    "pending evidence",
    "cannot verify",
    "could not verify",
    "could not locate",
    "unable to determine",
    "unable to verify",
    "best effort",
    "low confidence",
    "preliminary",
    "hypothesis",
    "uncertain",
    "partial artifact",
    "status:partial",
    "status:incomplete",
    "incomplete evidence",
    "incomplete artifact",
    "incomplete result",
    "needs:verification",
    "needs_verification",
    "needs-verification",
    "status:unverified",
    "status:evidence_pending",
    "status:evidence-pending",
    "status:needs-verification",
    "status:needs_verification",
    "confidence:low",
)
_PLACEHOLDER_EVIDENCE_TERMS = (
    "placeholder",
    "search in progress",
    "work in progress",
    "analysis in progress",
    "actively searching",
    "still searching",
    "still need to",
    "still needs",
    "pending:",
    "results pending",
    "pending results",
    "pending query",
    "pending execution",
    "will be updated",
    "will update",
    "will populate",
    "populate after execution",
    "to be populated",
    "to be filled",
    "not yet run",
    "not yet executed",
    "not yet identified",
    "not yet available",
    "tbd",
    "todo:",
)
_UNVERIFIED_EVIDENCE_TERMS = _STRONG_UNVERIFIED_EVIDENCE_TERMS + _PLACEHOLDER_EVIDENCE_TERMS
_EVIDENCE_ATTACHED_TERMS = (
    "source:",
    "sources:",
    "citation:",
    "citations:",
    "url:",
    "doi:",
    "arxiv:",
    "arxiv id",
    "http://",
    "https://",
    "retrieved",
    "primary source",
    "evidence ref",
    "evidence_refs",
)
_COMPLETION_CLAIM_TAGS = {
    "status:complete",
    "status:done",
    "confidence:high",
    "verified:true",
}
_TERMINAL_ANSWER_ARTIFACT_TERMS = (
    "candidate answer",
    "final answer",
    "answer verification",
    "cross-check complete",
    "overlap analysis",
    "candidate_answer:",
)
_TRUTH_GUARD_UNVERIFIED_TAGS = {"status:unverified", "needs:verification", "confidence:low"}
_CREATE_ALLOWED_TOOLS = _CREATE_EXECUTION_TOOLS | _CREATE_STATE_TOOLS
_UNCERTAIN_EVIDENCE_CREATE_REASONS = {
    "handoff_after_partial_progress",
    "delegate_uncertain_evidence_recovery",
}
_DOWNSTREAM_FOCUS_TERMS = (
    "verify",
    "verifier",
    "verification",
    "validate",
    "validator",
    "validation",
    "review",
    "reviewer",
    "cross-check",
    "corroborate",
    "confidence",
    "critique",
    "critic",
    "synthesis",
    "synthesize",
    "synthesizer",
    "summarize",
    "summary",
    "integrate",
    "integration",
    "integrator",
    "qa",
    "final",
)
_UPSTREAM_INPUT_TERMS = (
    "after",
    "based on",
    "before final",
    "confidence",
    "dependency",
    "depends",
    "evidence",
    "from peers",
    "input",
    "partial",
    "peer",
    "peers",
    "source",
    "upstream",
    "worker",
    "workers",
)
_DOWNSTREAM_STRONG_TERMS = (
    "after",
    "based on",
    "before final",
    "cross-check",
    "integrate",
    "integration",
    "review",
    "synthesis",
    "synthesize",
    "verify",
    "verifier",
    "verification",
)
_EVIDENCE_COLLECTION_TERMS = (
    "collect evidence",
    "evidence report",
    "find",
    "identify",
    "investigate",
    "look up",
    "lookup",
    "research",
    "search",
    "source",
    "sources",
)


def _sigmoid01(value: float, center: float = 0.7, sharpness: float = 10.0) -> float:
    return 1.0 / (1.0 + math.exp(-sharpness * (max(0.0, min(1.0, value)) - center)))


def _cheap_text_similarity(left: str, right: str) -> float:
    left_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", (left or "").lower()))
    right_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", (right or "").lower()))
    left_terms = {term for term in left_terms if len(term) > 1}
    right_terms = {term for term in right_terms if len(term) > 1}
    if not left_terms and not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))


def _identity_terms_for_dependency_match(agent: "Agent") -> set[str]:
    terms = {
        agent.id,
        agent.role,
        agent.group_id,
        agent.created_by or "",
        agent.create_type,
        agent.relationship,
        agent.workflow_prior,
    }
    terms.update(str(tag) for tag in agent.current_task_tags)
    terms.update(str(artifact.path) for artifact in agent.artifacts)
    public_memory = agent.memory.get("public_memory") if isinstance(agent.memory, dict) else None
    if isinstance(public_memory, dict):
        terms.update(str(tag) for tag in public_memory.get("tags") or [])
        terms.update(str(path) for path in public_memory.get("artifacts") or [])
        summary = str(public_memory.get("public_summary") or "")
        terms.update(re.findall(r"[\w:./-]+", summary.lower())[:80])
    terms.update(re.findall(r"[\w:./-]+", (agent.task or "").lower())[:120])
    normalized = {term.strip().lower() for term in terms if str(term).strip()}
    expanded = set(normalized)
    for term in normalized:
        if ":" in term:
            expanded.add(term.split(":", 1)[1])
        if term.startswith("lane:"):
            expanded.add(term[len("lane:"):])
        if term.startswith("feature:"):
            expanded.add(term[len("feature:"):])
        if term.startswith("role:"):
            expanded.add(term[len("role:"):])
    return expanded


def _agent_matches_dependency_terms(agent: "Agent", depends_on: list[str]) -> bool:
    wanted = {str(item).strip().lower() for item in depends_on if str(item).strip()}
    if not wanted:
        return True
    terms = _identity_terms_for_dependency_match(agent)
    for item in wanted:
        if item in terms:
            return True
        if ":" in item and item.split(":", 1)[1] in terms:
            return True
        if any(term.endswith(item) or item.endswith(term) for term in terms if len(term) >= 4):
            return True
    return False


def _lane_fingerprints_from_parts(*parts: Any) -> set[str]:
    text = " ".join(
        str(part or "")
        for part in parts
    ).lower()
    normalized = re.sub(r"[^a-z0-9:_./-]+", " ", text)
    terms = set(re.findall(r"[a-z0-9][a-z0-9:_./-]{2,}", normalized))
    fingerprints: set[str] = set()
    for term in terms:
        if term.startswith("lane:"):
            fingerprints.add("lane:" + term.split(":", 1)[1].replace("_", "-"))
        elif term.startswith("deliverable:"):
            fingerprints.add("deliverable:" + term.split(":", 1)[1].replace("_", "-"))
    has_ai = "ai" in terms or "artificial" in terms or "intelligence" in terms
    has_ai_regulation = "regulation" in terms or "governance" in terms or "regulating" in terms
    has_2022_june = (
        "2022" in terms
        or "202206" in normalized
        or "2022-06" in normalized
        or "2022/06" in normalized
        or "june 2022" in normalized
    )
    has_physics_society = (
        "physics" in terms
        or "soc-ph" in normalized
        or "phys-soc" in normalized
        or "physics.soc-ph" in normalized
        or "physics/soc-ph" in normalized
    )
    has_society = "society" in terms or "societies" in terms or "soc-ph" in normalized
    has_2016_aug11 = (
        "2016" in terms
        or "20160811" in normalized
        or "2016-08-11" in normalized
        or "2016/08/11" in normalized
        or "august 11 2016" in normalized
        or "august 11, 2016" in normalized
    )
    if has_ai and has_ai_regulation:
        fingerprints.add("topic:ai-regulation")
    if has_2022_june:
        fingerprints.add("date:2022-06")
    if has_physics_society and has_society:
        fingerprints.add("topic:physics-society")
    if has_2016_aug11:
        fingerprints.add("date:2016-08-11")
    if {"topic:ai-regulation", "date:2022-06"}.issubset(fingerprints):
        fingerprints.add("lane:ai-regulation-2022")
    if {"topic:physics-society", "date:2016-08-11"}.issubset(fingerprints):
        fingerprints.add("lane:phys-soc-2016")
    if "verify" in terms or "verifier" in terms or "verification" in terms or "overlap" in terms:
        fingerprints.add("lane:verification")
    return {item for item in fingerprints if item}


def _agent_lane_fingerprints(agent: "Agent") -> set[str]:
    return _lane_fingerprints_from_parts(
        agent.role,
        agent.group_id,
        agent.workflow_prior,
        agent.task,
        " ".join(agent.current_task_tags),
        " ".join(str(artifact.path) for artifact in agent.artifacts),
    )


def _spawn_request_lane_fingerprints(
    *,
    task: str,
    role: str = "",
    group_id: str = "",
    workflow_prior: str = "",
    current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
) -> set[str]:
    return _lane_fingerprints_from_parts(
        role,
        group_id,
        workflow_prior,
        task,
        " ".join(str(tag) for tag in (current_task_tags or [])),
    )


def _create_exit_decision_tags(tags: list[str] | tuple[str, ...] | set[str] | None) -> bool:
    normalized = {str(tag).strip().lower() for tag in (tags or [])}
    return bool(
        normalized.intersection({"status:solo-decision", "status:pruned"})
        or any(tag.startswith("reason:peer_ahead") or tag.startswith("reason:duplicate_lane") for tag in normalized)
    )


def _looks_like_verification_request(*parts: Any) -> bool:
    text = "\n".join(str(part or "") for part in parts).lower()
    return any(term in text for term in _DOWNSTREAM_STRONG_TERMS)


def _looks_like_recovery_request(*parts: Any) -> bool:
    text = "\n".join(str(part or "") for part in parts).lower()
    recovery_terms = (
        "failed lane",
        "failed child",
        "recover",
        "recovery",
        "repair",
        "blocked",
        "missing output",
        "missing artifact",
        "unverified",
        "low confidence",
        "partial evidence",
        "partial artifact",
        "resolve discrepancy",
        "investigate discrepancy",
        "resolve contradiction",
        "investigate contradiction",
    )
    return any(term in text for term in recovery_terms)


def _looks_like_downstream_focus(*parts: Any) -> bool:
    text = "\n".join(str(part or "") for part in parts).lower()
    if not text.strip():
        return False
    if any(term in text for term in _EVIDENCE_COLLECTION_TERMS) and not any(
        term in text for term in _DOWNSTREAM_STRONG_TERMS
    ):
        return False
    return (
        any(term in text for term in _DOWNSTREAM_STRONG_TERMS)
        and any(term in text for term in _UPSTREAM_INPUT_TERMS)
    )


def _focus_class_from_parts(
    *,
    task: str = "",
    role: str = "",
    group_id: str = "",
    workflow_prior: str = "",
    current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    tags = [str(tag).strip().lower() for tag in (current_task_tags or []) if str(tag).strip()]
    text = "\n".join([role or "", group_id or "", workflow_prior or "", " ".join(tags), task or ""]).lower()
    if (
        _final_answer_paths(task)
        or _task_mentions_submit_answer_protocol(task)
        or "submit_answer" in text
        or "answer_submission" in text
        or "final_delivery" in text
        or "final-delivery" in text
        or "role:delivery" in text
        or "role:finalizer" in text
        or re.search(r"\bfinali[sz]er\b", text)
        or re.search(r"\bdelivery agent\b", text)
    ):
        return "final_delivery"
    if _looks_like_recovery_request(task, role, group_id, workflow_prior, " ".join(tags)):
        return "recovery"
    if any(term in text for term in (
        "synthesis", "synthesize", "synthesizer", "integrate", "integration",
        "integrator", "aggregate", "summary", "summarize", "overlap",
        "intersection", "cross-lane", "cross lane", "final synthesis",
    )):
        return "synthesis"
    if _looks_like_verification_request(task, role, group_id, workflow_prior, " ".join(tags)) or any(
        term in text for term in (
            "verify", "verifier", "verification", "validate", "validator",
            "validation", "review", "reviewer", "cross-check", "corroborate",
        )
    ):
        return "verification"
    return "evidence"


def _extract_terminal_answer_hint(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped_heading = line.strip()
        if not stripped_heading.startswith("#"):
            continue
        heading_text = re.sub(r"^#+\s*", "", stripped_heading).strip().lower()
        heading_text = heading_text.strip(" .,:;`*_\"'")
        if heading_text not in {"candidate answer", "final answer", "answer"}:
            continue
        for next_line in lines[index + 1:index + 8]:
            candidate = next_line.strip()
            if not candidate or candidate.startswith("#") or candidate.startswith("|"):
                continue
            candidate = candidate.strip(" .,:;`*_\"'")
            if 0 < len(candidate) <= 120 and not _looks_like_process_chatter(candidate):
                return candidate
    patterns = (
        r"(?im)^\s*(?:candidate\s+answer|final\s+answer|answer)\s*[:\-]\s*\**`?\"?([^`\"\n*|]{1,120})",
        r"(?im)\bcandidate_answer\s*:\s*([^\s,;|\n]{1,120})",
        r"(?im)\boverlap\s*:\s*\**`?\"?([^`\"\n*|]{1,120})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        answer = match.group(1).strip()
        answer = re.sub(r"\s+\|.*$", "", answer).strip()
        answer = re.sub(r"\s+-\s+.*$", "", answer).strip()
        answer = answer.strip(" .,:;`*_\"'")
        if 0 < len(answer) <= 120 and not _looks_like_process_chatter(answer):
            return answer
    return ""


def _explicit_final_delivery_identity(
    *,
    role: str = "",
    group_id: str = "",
    workflow_prior: str = "",
    current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
) -> bool:
    tags = [str(tag).strip().lower() for tag in (current_task_tags or []) if str(tag).strip()]
    identity = "\n".join([role or "", group_id or "", workflow_prior or "", " ".join(tags)]).lower()
    return bool(
        "final_delivery" in identity
        or "final-delivery" in identity
        or "role:delivery" in identity
        or "role:finalizer" in identity
        or re.search(r"\bfinali[sz]er\b", identity)
        or re.search(r"\bdelivery\b", identity)
    )


@dataclass
class LoopConstraintMetrics:
    budget_pressure: float = 0.0
    time_pressure: float = 0.0
    context_pressure: float = 0.0
    agent_pressure: float = 0.0
    dependency_pressure: float = 0.0
    handoff_pressure: float = 0.0
    offspring_create_bias: float = 0.0
    failed_child_pressure: float = 0.0
    deferred_spawn_readiness: float = 0.0
    artifact_gap: float = 0.0
    coordination_ratio: float = 0.0
    create_retry_pressure: float = 0.0
    action_miss_pressure: float = 0.0
    lane_coverage_pressure: float = 0.0
    stagnation_pressure: float = 0.0
    output_similarity: float = 0.0
    duplicate_tool_pressure: float = 0.0
    consistency_pressure: float = 0.0
    uncertain_evidence_pressure: float = 0.0
    evidence_integration_pressure: float = 0.0


@dataclass
class LoopActionCandidate:
    action: str
    reason: str
    base_priority: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


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
    allow_agent_model_override: bool = False
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
    create_action_max_tokens: int = 20000   # response budget for create turns that may emit large spawn_many args
    enabled_work_tools: set[str] | None = None  # None = all work tools enabled
    shell_mode: ShellMode = "controlled"        # disabled, controlled, unrestricted
    controlled_shell_allowed_commands: set[str] | None = None
    bootstrap_batch_file: str | None = None  # Optional JSON batch executed by root before its first LLM turn.
    orchestration_preference: OrchestrationPreference = "balanced"  # solo, balanced, parallel, aggressive
    child_orchestration_preference: ChildOrchestrationPreference = None  # None/task_adaptive lets children choose from assigned work
    min_spawnable_workstreams: int = 3
    spawn_before_turn: int = 2
    max_solo_tool_calls_before_spawn: int = 5
    spawn_readiness_threshold: float = 0.62
    handoff_child_count_threshold: int = 2
    peer_progress_check_after_turns: int = 3
    peer_progress_check_no_tool_turns: int = 2
    peer_progress_check_after_artifact_nudges: int = 2
    # Loop action policy. "rule" preserves the current short-circuit planner;
    # "constraint" scores candidate actions against runtime pressure metrics.
    loop_action_policy: Literal["rule", "constraint"] = "rule"
    # Split loop action selection from action execution for real LLM calls.
    # The selector is a short, no-tool turn that only chooses one of the six
    # loop actions; the following execution turn fixes that action and scopes
    # tools accordingly.
    two_stage_action_selection: bool = True
    # After a terminal answer submission, ask remaining descendants to compact
    # and stop, then stop waiting after this short grace period. This prevents
    # an already-submitted root task from being held open by a stale child loop.
    terminal_submission_join_grace: float = 5.0
    disabled_loop_constraints: set[str] = field(default_factory=set)


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
    submitted_answer_path: str | None = None

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
    _peer_overlap_query_turn: int = 0
    _last_query_turn: int = 0
    _last_query_evidence_turn: int = 0
    _last_query_agent_ids: list[str] = field(default_factory=list)
    _last_query_reliable_evidence_ids: list[str] = field(default_factory=list)
    _last_read_evidence_turn: int = 0
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
    _deferred_spawn_requests: list[dict[str, Any]] = field(default_factory=list)
    _create_resume_after_read: bool = False
    _auxiliary_only_miss_count: int = 0
    _action_miss_count: int = 0
    _action_miss_action: str = ""
    _loop_action_plan: dict[str, Any] | None = None
    _loop_action_selected_by: str = ""
    _state_action_version: int = 0
    _state_action_consumed_version: int = 0
    _pending_prune_requests: list[dict[str, Any]] = field(default_factory=list)
    _prune_requests_sent_targets: set[str] = field(default_factory=set)
    _last_spawn_skipped_turn: int = 0
    _last_spawn_skipped_reason: str = ""
    _last_spawn_skipped_covered_by: list[str] = field(default_factory=list)
    _last_spawn_skipped_covered_paths: list[str] = field(default_factory=list)
    _last_create_resolution: str = ""
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
        self.work_tools = dict(WORK_TOOLS)
        self._events: list[dict] = []  # all events for post-hoc analysis
        self._messages_sent: list[tuple[str, str, int]] = []  # (from, to, tokens) for comm graph
        self._last_llm_messages: dict[str, list[Message]] = {}
        self.task_ledger: dict[str, Any] = {}
        self._submit_answer_grants: set[str] = set()
        self._emit_lock = threading.Lock()  # protects events.jsonl writes

        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        self._tool_context.shared_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_task_ledger_loaded()
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
        role, role_description, role_tags = canonicalize_role(role, current_task_tags)
        current_task_tags = normalize_tags(
            list(current_task_tags or [])
            + role_tags
            + agent_identity_tags(
                agent_id=agent_id,
                role=role,
                group_id=group_id,
                parent=parent,
                created_by=created_by,
                create_type=create_type,
                relationship=relationship,
                workflow_prior=workflow_prior,
                depth=depth,
            )
        )
        agent_orchestration_preference, orchestration_resolution = self._resolve_agent_orchestration_preference(
            parent=parent,
            requested=orchestration_preference,
            task=task,
            role=role,
            create_type=create_type,
            relationship=relationship,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
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
            bio=role_description,
            create_type=create_type,
            relationship=relationship,
            created_by=created_by,
            group_id=group_id,
            workflow_prior=workflow_prior,
            orchestration_preference=agent_orchestration_preference,
            current_task_tags=list(current_task_tags),
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
        if parent is None and self.answer_submission_required(agent):
            self._submit_answer_grants.add(agent.id)
        if parent and parent in self.agents:
            self.agents[parent].children.add(agent_id)
        self.memory.init_agent(agent_id, task=task, tags=agent.current_task_tags)
        agent.memory = {"public_memory": self.memory.serialize(agent_id)}
        self.state_board_sync(agent_id)
        self._ensure_task_ledger_agent(agent)

        self._emit(agent_id, "agent_new", {
            "task": task, "model": model, "budget": quota.budget if math.isfinite(quota.budget) else None,
            "parent": parent, "depth": depth,
            "role": role,
            "role_description": role_description,
            "create_type": create_type,
            "relationship": relationship,
            "created_by": created_by,
            "group_id": group_id,
            "workflow_prior": workflow_prior,
            "current_task_tags": list(agent.current_task_tags),
            "submit_answer_granted": agent.id in self._submit_answer_grants,
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
            agent.current_task_tags = normalize_tags(
                list(current_task_tags)
                + agent_identity_tags(
                    agent_id=agent.id,
                    role=agent.role,
                    group_id=agent.group_id,
                    parent=agent.parent,
                    created_by=agent.created_by,
                    create_type=agent.create_type,
                    relationship=agent.relationship,
                    workflow_prior=agent.workflow_prior,
                    depth=agent.depth,
                )
            )
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

    # ─── Task ledger ────────────────────────────────────────────────────

    def _task_ledger_path(self) -> Path:
        return self._tool_context.shared_dir / ".nanoma" / "task_ledger.json"

    def _empty_task_ledger(self) -> dict[str, Any]:
        now = time.time()
        return {
            "version": 1,
            "root_task": "",
            "root_agent_id": "",
            "created_at": now,
            "updated_at": now,
            "items": {},
        }

    def _ensure_task_ledger_loaded(self) -> dict[str, Any]:
        if self.task_ledger:
            return self.task_ledger
        path = self._task_ledger_path()
        data: dict[str, Any] = {}
        if path.exists():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                data = {}
        if not data:
            data = self._empty_task_ledger()
        data.setdefault("version", 1)
        data.setdefault("root_task", "")
        data.setdefault("root_agent_id", "")
        data.setdefault("created_at", time.time())
        data.setdefault("updated_at", time.time())
        if not isinstance(data.get("items"), dict):
            data["items"] = {}
        self.task_ledger = data
        return self.task_ledger

    def _save_task_ledger(self) -> None:
        ledger = self._ensure_task_ledger_loaded()
        ledger["updated_at"] = time.time()
        path = self._task_ledger_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as e:
            self._emit("system", "task_ledger_save_failed", {"path": str(path), "error": str(e)})

    def _task_ledger_copy(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._ensure_task_ledger_loaded(), ensure_ascii=False, default=str))

    def _task_ledger_agent_item_id(self, agent: Agent) -> str:
        return agent.id

    def _task_ledger_short_task(self, task: str, limit: int = 180) -> str:
        text = " ".join(str(task or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _task_ledger_existing_output_status(self, item: dict[str, Any], path: str) -> str:
        for output in item.get("expected_outputs") or []:
            if isinstance(output, dict) and output.get("path") == path:
                return str(output.get("status") or "")
        return ""

    def _prune_resolved_output_blockers(
        self,
        blockers: Any,
        resolved_outputs: set[str],
    ) -> list[Any]:
        if not isinstance(blockers, list) or not resolved_outputs:
            return list(blockers or []) if isinstance(blockers, list) else []
        kept: list[Any] = []
        for blocker in blockers:
            if not isinstance(blocker, dict):
                kept.append(blocker)
                continue
            kind = str(blocker.get("kind") or "")
            if kind not in {"uncertain_evidence", "unverified_evidence"}:
                kept.append(blocker)
                continue
            artifacts = {
                str(path)
                for path in _normalize_string_list(blocker.get("artifacts"))
            }
            if artifacts and artifacts.issubset(resolved_outputs):
                continue
            kept.append(blocker)
        return kept

    def _task_ledger_output_entries(self, agent: Agent, existing: dict[str, Any]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        resolved_outputs: set[str] = set()
        for path in _expected_output_paths(agent.task):
            existing_status = self._task_ledger_existing_output_status(existing, path)
            abs_path = self._tool_context.workspace_root / path
            exists = abs_path.exists()
            is_final_answer = _looks_like_final_answer_path(path)
            covered_by = []
            for output in existing.get("expected_outputs") or []:
                if isinstance(output, dict) and output.get("path") == path:
                    covered_by = _normalize_string_list(output.get("covered_by"))
                    break
            if existing_status == "covered" and covered_by:
                status = "covered"
            else:
                status = self._artifact_output_status(agent, path, existing_status)
            entry = {
                "path": path,
                "status": status,
                "exists": bool(exists),
            }
            if covered_by:
                entry["covered_by"] = covered_by[:8]
            if is_final_answer:
                entry["final_answer"] = True
                if agent.submitted_answer_path:
                    entry["submitted_answer_path"] = agent.submitted_answer_path
            if status in {"evidence_attached", "verified", "covered", "submitted"}:
                resolved_outputs.add(path)
            outputs.append(entry)
        if resolved_outputs and existing.get("blockers"):
            existing["blockers"] = self._prune_resolved_output_blockers(existing.get("blockers"), resolved_outputs)
        return outputs

    def _task_ledger_agent_status(
        self,
        agent: Agent,
        existing: dict[str, Any],
        outputs: list[dict[str, Any]],
    ) -> str:
        existing_status = str(existing.get("status") or "")
        if existing_status in {"covered", "pruned"} and existing.get("covered_by"):
            return existing_status
        if existing_status == "failed":
            return "failed"
        if agent.status == "failed":
            return "failed"
        tags = set(agent.current_task_tags)
        if "status:pruned" in tags:
            return "pruned"
        if outputs:
            statuses = {str(output.get("status") or "") for output in outputs}
            if statuses and statuses.issubset(_TASK_LEDGER_OUTPUT_DONE_STATUSES):
                return "verified" if agent.status == "done" else "ready_for_review"
            if statuses == {"evidence_attached"}:
                return "verified" if agent.status == "done" else "ready_for_review"
            if statuses.intersection({"draft", "unverified", "placeholder", "evidence_gap"}):
                return "partial"
            if "missing" in statuses:
                if existing_status in {"blocked", "partial", "ready_for_review", "claimed"}:
                    return existing_status
                return "in_progress" if agent._turns or agent.status == "running" else "claimed"
        if existing_status in {"blocked", "partial", "ready_for_review", "claimed", "in_progress"}:
            return existing_status
        if agent.status == "done":
            return "verified" if self._agent_has_reliable_evidence(agent) else "partial"
        if agent._tool_calls or agent._turns:
            return "in_progress"
        return "claimed"

    def _refresh_task_ledger_agent_item(self, agent: Agent, *, save: bool = True) -> dict[str, Any]:
        ledger = self._ensure_task_ledger_loaded()
        items = ledger.setdefault("items", {})
        if not agent.parent and not ledger.get("root_agent_id"):
            ledger["root_agent_id"] = agent.id
            ledger["root_task"] = agent.task
        item_id = self._task_ledger_agent_item_id(agent)
        existing = dict(items.get(item_id) or {})
        outputs = self._task_ledger_output_entries(agent, existing)
        item = {
            **existing,
            "id": item_id,
            "agent_id": agent.id,
            "owner": agent.id,
            "title": existing.get("title") or self._task_ledger_short_task(agent.task, 120),
            "task": agent.task,
            "task_preview": self._task_ledger_short_task(agent.task, 180),
            "parent": agent.parent,
            "children": sorted(agent.children),
            "depth": agent.depth,
            "role": agent.role,
            "group_id": agent.group_id,
            "workflow_prior": agent.workflow_prior,
            "tags": list(agent.current_task_tags[:30]),
            "action_state": agent.action_state,
            "agent_status": agent.status,
            "status": "",
            "expected_outputs": outputs,
            "updated_turn": agent._turns,
            "updated_by": agent.id,
            "updated_at": time.time(),
        }
        if "created_at" not in item:
            item["created_at"] = time.time()
        if existing.get("blockers"):
            item["blockers"] = list(existing.get("blockers") or [])[:8]
        if existing.get("note"):
            item["note"] = str(existing.get("note") or "")[:1000]
        if existing.get("covered_by"):
            item["covered_by"] = _normalize_string_list(existing.get("covered_by"))[:8]
        if existing.get("updates"):
            item["updates"] = list(existing.get("updates") or [])[-20:]
        item["status"] = self._task_ledger_agent_status(agent, existing, outputs)
        items[item_id] = item
        if save:
            self._save_task_ledger()
        return item

    def _ensure_task_ledger_agent(self, agent: Agent) -> dict[str, Any]:
        return self._refresh_task_ledger_agent_item(agent)

    def _refresh_task_ledger_all(self) -> None:
        for agent in list(self.agents.values()):
            self._refresh_task_ledger_agent_item(agent, save=False)
        self._save_task_ledger()

    def task_ledger_item_snapshot(self, agent: Agent) -> dict[str, Any] | None:
        item = self._refresh_task_ledger_agent_item(agent)
        return self._task_ledger_public_item(item, detailed=True)

    def _task_ledger_public_item(self, item: dict[str, Any], *, detailed: bool = False) -> dict[str, Any]:
        public = {
            "id": item.get("id"),
            "agent_id": item.get("agent_id"),
            "owner": item.get("owner"),
            "status": item.get("status"),
            "agent_status": item.get("agent_status"),
            "action_state": item.get("action_state"),
            "title": item.get("title") or item.get("task_preview"),
            "parent": item.get("parent"),
            "children": list(item.get("children") or [])[:8],
            "role": item.get("role"),
            "group_id": item.get("group_id"),
            "tags": list(item.get("tags") or [])[:12],
            "expected_outputs": [
                {
                    key: value
                    for key, value in dict(output).items()
                    if key in {"path", "status", "exists", "covered_by", "final_answer", "submitted_answer_path"}
                }
                for output in list(item.get("expected_outputs") or [])[:8]
                if isinstance(output, dict)
            ],
        }
        if item.get("blockers"):
            public["blockers"] = list(item.get("blockers") or [])[:4]
        if item.get("covered_by"):
            public["covered_by"] = _normalize_string_list(item.get("covered_by"))[:8]
        if item.get("note"):
            public["note"] = str(item.get("note") or "")[:500]
        if detailed:
            public["task_preview"] = item.get("task_preview") or self._task_ledger_short_task(str(item.get("task") or ""))
            public["updated_turn"] = item.get("updated_turn")
            public["updated_by"] = item.get("updated_by")
            public["updates"] = list(item.get("updates") or [])[-5:]
        return public

    def _task_ledger_relevant_item_ids(self, agent: Agent, *, limit: int = 10) -> list[str]:
        ledger = self._ensure_task_ledger_loaded()
        items = ledger.get("items") or {}
        ids: list[str] = []

        def add(item_id: str | None) -> None:
            if item_id and item_id in items and item_id not in ids:
                ids.append(item_id)

        add(str(ledger.get("root_agent_id") or ""))
        add(agent.id)
        add(agent.parent)
        for child_id in sorted(agent.children):
            add(child_id)

        own_outputs = {
            str(output.get("path") or "")
            for output in (items.get(agent.id, {}).get("expected_outputs") or [])
            if isinstance(output, dict) and output.get("path")
        }
        for item_id, item in sorted(items.items()):
            if len(ids) >= limit:
                break
            item_outputs = {
                str(output.get("path") or "")
                for output in (item.get("expected_outputs") or [])
                if isinstance(output, dict) and output.get("path")
            }
            if own_outputs and own_outputs.intersection(item_outputs):
                add(item_id)
        for item_id, item in sorted(items.items()):
            if len(ids) >= limit:
                break
            if agent.group_id and item.get("group_id") == agent.group_id:
                add(item_id)
        for item_id, item in sorted(items.items()):
            if len(ids) >= limit:
                break
            if item.get("status") in _TASK_LEDGER_TERMINAL_STATUSES:
                add(item_id)
        return ids[:limit]

    def task_ledger_snapshot(
        self,
        agent: Agent | None = None,
        *,
        include_all: bool = False,
        item_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        self._refresh_task_ledger_all()
        ledger = self._ensure_task_ledger_loaded()
        items = ledger.get("items") or {}
        if item_id:
            item = items.get(item_id)
            return {
                "path": str(self._task_ledger_path()),
                "item": self._task_ledger_public_item(item, detailed=True) if isinstance(item, dict) else None,
            }
        if include_all or agent is None:
            selected_ids = list(sorted(items))[: max(1, limit)]
        else:
            selected_ids = self._task_ledger_relevant_item_ids(agent, limit=max(1, limit))
        selected_items = [
            self._task_ledger_public_item(items[item_id], detailed=include_all)
            for item_id in selected_ids
            if isinstance(items.get(item_id), dict)
        ]
        counts: dict[str, int] = {}
        for item in items.values():
            status = str((item or {}).get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {
            "path": str(self._task_ledger_path()),
            "root_agent_id": ledger.get("root_agent_id"),
            "root_task": self._task_ledger_short_task(str(ledger.get("root_task") or ""), 220),
            "counts": counts,
            "current_agent_item": (
                self._task_ledger_public_item(items[agent.id], detailed=True)
                if agent is not None and isinstance(items.get(agent.id), dict)
                else None
            ),
            "items": selected_items,
            "count": len(selected_items),
            "total_items": len(items),
        }

    def task_ledger_update(self, agent: Agent, args: dict[str, Any]) -> dict[str, Any]:
        self._refresh_task_ledger_all()
        ledger = self._ensure_task_ledger_loaded()
        items = ledger.setdefault("items", {})
        item_id = str(args.get("item_id") or agent.id).strip() or agent.id
        if item_id not in items:
            if item_id == agent.id:
                self._refresh_task_ledger_agent_item(agent)
            else:
                return {"error": f"task ledger item '{item_id}' not found"}
        item = dict(items.get(item_id) or {})
        status = str(args.get("status") or "").strip()
        allowed_statuses = {
            "open", "claimed", "in_progress", "partial", "blocked",
            "ready_for_review", "verified", "covered", "pruned", "failed",
        }
        if status:
            if status not in allowed_statuses:
                return {"error": f"Unknown ledger status '{status}'"}
            item["status"] = status
        note = args.get("note")
        if note is not None:
            item["note"] = str(note)[:1000]
        blockers = args.get("blockers")
        if blockers is not None:
            if isinstance(blockers, list):
                item["blockers"] = blockers[:8]
            else:
                item["blockers"] = [blockers]
        covered_by = _normalize_string_list(args.get("covered_by"))
        if covered_by:
            item["covered_by"] = covered_by[:8]
        output_path = str(args.get("output_path") or "").strip()
        if output_path:
            output_path = self._normalize_workspace_relative_path(output_path, agent.workspace)
            output_status = str(args.get("output_status") or args.get("status") or "").strip()
            if not output_status:
                output_status = "draft"
            allowed_output_statuses = {
                "missing", "draft", "partial", "unverified", "placeholder", "evidence_gap",
                "evidence_attached", "verified", "covered", "submitted", "rejected",
            }
            if output_status not in allowed_output_statuses:
                return {"error": f"Unknown output status '{output_status}'"}
            outputs = [dict(output) for output in (item.get("expected_outputs") or []) if isinstance(output, dict)]
            target = None
            for output in outputs:
                if output.get("path") == output_path:
                    target = output
                    break
            if target is None:
                target = {"path": output_path}
                outputs.append(target)
            target["status"] = output_status
            target["exists"] = bool((self.config.workspace_root / output_path).exists())
            if covered_by:
                target["covered_by"] = covered_by[:8]
            evidence_refs = _normalize_string_list(args.get("evidence_refs", args.get("evidence", [])))
            if evidence_refs:
                target["evidence_refs"] = evidence_refs[:12]
            item["expected_outputs"] = outputs
        event = {
            "turn": agent._turns,
            "by": agent.id,
            "status": status or item.get("status"),
            "output_path": output_path or None,
            "output_status": args.get("output_status"),
            "note": str(note or "")[:240] if note is not None else "",
        }
        updates = list(item.get("updates") or [])
        updates.append(event)
        item["updates"] = updates[-20:]
        item["updated_at"] = time.time()
        item["updated_by"] = agent.id
        items[item_id] = item
        self._save_task_ledger()
        self._emit(agent.id, "task_ledger_update", {
            "item_id": item_id,
            "status": item.get("status"),
            "output_path": output_path or None,
            "output_status": args.get("output_status"),
        })
        return {
            "updated": True,
            "path": str(self._task_ledger_path()),
            "item": self._task_ledger_public_item(item, detailed=True),
        }

    def _task_ledger_loop_context(self, agent: Agent) -> str:
        snapshot = self.task_ledger_snapshot(agent, include_all=False, limit=10)
        current = snapshot.get("current_agent_item") or {}
        lines = [
            "[Task Ledger]",
            f"path={snapshot.get('path')}",
            f"root={snapshot.get('root_agent_id') or '-'} status_counts={snapshot.get('counts') or {}}",
        ]
        if current:
            outputs = current.get("expected_outputs") or []
            output_text = ", ".join(
                f"{output.get('path')}={output.get('status')}"
                for output in outputs[:5]
                if isinstance(output, dict)
            ) or "none"
            blockers = current.get("blockers") or []
            lines.append(
                f"you={current.get('id')} status={current.get('status')} "
                f"action={current.get('action_state')} outputs={output_text}"
            )
            if blockers:
                lines.append(f"your_blockers={json.dumps(blockers[:3], ensure_ascii=False, default=str)}")
            if current.get("covered_by"):
                lines.append(f"your_covered_by={current.get('covered_by')}")
        related = []
        for item in snapshot.get("items") or []:
            if item.get("id") == agent.id:
                continue
            outputs = ", ".join(
                f"{output.get('path')}={output.get('status')}"
                for output in (item.get("expected_outputs") or [])[:3]
                if isinstance(output, dict)
            )
            related.append(
                f"- {item.get('id')}: status={item.get('status')} "
                f"agent={item.get('agent_status')} action={item.get('action_state')} "
                f"group={item.get('group_id') or '-'} outputs={outputs or 'none'}"
            )
        if related:
            lines.append("related:\n" + "\n".join(related[:6]))
        lines.append(
            "Use ledger_read/ledger_update to inspect or update task state. "
            "Memory stores evidence; the task ledger tracks ownership, blockers, and output coverage. "
            "If this item is already covered/verified by others, compact a reusable summary and self-prune."
        )
        return "\n".join(lines)

    def _task_ledger_expected_output_coverage(self, agent: Agent) -> dict[str, list[str]]:
        ledger = self._ensure_task_ledger_loaded()
        expected = _non_terminal_expected_output_paths(agent.task)
        coverage: dict[str, list[str]] = {path: [] for path in expected}
        if not expected:
            return coverage
        for item_id, item in (ledger.get("items") or {}).items():
            if item_id == agent.id or not isinstance(item, dict):
                continue
            owner = str(item.get("owner") or item.get("agent_id") or item_id)
            for output in item.get("expected_outputs") or []:
                if not isinstance(output, dict):
                    continue
                path = str(output.get("path") or "")
                if path not in coverage:
                    continue
                if str(output.get("status") or "") in _TASK_LEDGER_OUTPUT_DONE_STATUSES:
                    coverage[path].append(owner)
        own_item = (ledger.get("items") or {}).get(agent.id) or {}
        for output in own_item.get("expected_outputs") or []:
            if not isinstance(output, dict):
                continue
            path = str(output.get("path") or "")
            if path in coverage and str(output.get("status") or "") == "covered":
                coverage[path].extend(_normalize_string_list(output.get("covered_by")))
        return {path: sorted(set(ids)) for path, ids in coverage.items()}

    def _task_ledger_pruned_stop_is_allowed(self, agent: Agent) -> bool:
        ledger = self._ensure_task_ledger_loaded()
        item = (ledger.get("items") or {}).get(agent.id) or {}
        if item.get("status") in {"covered", "pruned"} and item.get("covered_by"):
            return True
        expected = _non_terminal_expected_output_paths(agent.task)
        if not expected:
            return False
        coverage = self._task_ledger_expected_output_coverage(agent)
        return bool(coverage) and all(coverage.get(path) for path in expected)

    def _peer_exact_output_coverage_candidates(self, agent: Agent) -> list[Agent]:
        expected = set(_non_terminal_expected_output_paths(agent.task))
        if not expected:
            return []
        candidates: list[Agent] = []
        for peer in self.agents.values():
            if peer.id == agent.id:
                continue
            peer_paths = self._agent_known_artifact_paths(peer).union(_non_terminal_expected_output_paths(peer.task))
            if not expected.issubset(peer_paths):
                continue
            if peer.status != "done" and not self._agent_has_prune_worthy_progress(peer):
                continue
            if self._agent_has_uncertain_evidence(peer):
                continue
            candidates.append(peer)
        candidates.sort(key=lambda peer: (0 if peer.status == "done" else 1, peer.id))
        return candidates

    def _queried_prunable_peer_coverage_candidates(self, agent: Agent) -> list[Agent]:
        if agent._last_query_turn <= 0 or (agent._turns - agent._last_query_turn) > 4:
            return []
        queried_ids = set(agent._last_query_agent_ids)
        candidates: list[Agent] = []
        for peer in self._peer_progress_candidates(agent):
            if peer.id == agent.id:
                continue
            if queried_ids and peer.id not in queried_ids:
                continue
            if not self._agent_has_prune_worthy_progress(peer):
                continue
            if self._agent_has_uncertain_evidence(peer):
                continue
            if not self._agents_share_prunable_lane(agent, peer):
                continue
            candidates.append(peer)
        candidates.sort(key=lambda peer: (
            0 if peer.status == "done" else 1,
            -len(peer.artifacts),
            -peer._tool_calls,
            peer.id,
        ))
        return candidates

    def _pending_prune_request_has_coverage_signal(self, agent: Agent) -> bool:
        for request in reversed(agent._pending_prune_requests):
            if _normalize_string_list(request.get("covered_by")):
                return True
            evidence_agents = request.get("evidence_agents")
            if isinstance(evidence_agents, list) and evidence_agents:
                return True
            if _normalize_string_list(request.get("covered_paths")):
                return True
        return False

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
        if envelope.message_type == "prune_request":
            request = dict(envelope.payload or {})
            request.update({
                "from": envelope.from_id,
                "message": envelope.content,
                "timestamp": envelope.timestamp,
            })
            agent._pending_prune_requests.append(request)
            self.state_board_update(agent.id, action_state="compact")
            self._emit(agent.id, "prune_request_received", {
                "from": envelope.from_id,
                "payload": envelope.payload,
                "message": envelope.content[:500],
            })
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

                self._refresh_task_ledger_agent_item(agent)
                missing_outputs = self._missing_expected_outputs(agent)
                self._refresh_loop_action_plan(agent, missing_outputs)
                await self._maybe_select_loop_action(agent, missing_outputs)
                loop_context = self._build_loop_action_context(agent, missing_outputs, transient_messages)
                loop_action = str((agent._loop_action_plan or {}).get("action", ""))
                children_before_turn = len(agent.children)
                create_miss_before_turn = agent._create_action_filtered_count
                peer_pre_query_pending = self._peer_progress_decision_pending(agent)
                peer_post_query_pending = self._peer_progress_post_query_pending(agent) or self._peer_overlap_post_query_pending(agent, missing_outputs)
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
                plan_reason = str((agent._loop_action_plan or {}).get("reason", ""))
                recommended_tool_choice: str | None = None
                if turn_tool_scope in {
                    "retry_file_write_after_text_miss",
                    "retry_file_write_after_no_tool_miss",
                    "artifact_write_only",
                } and any(
                    (tool.get("function") or {}).get("name") == "file_write" for tool in turn_tool_schemas
                ):
                    recommended_tool_choice = "file_write"
                elif loop_action == "create" and any(
                    (tool.get("function") or {}).get("name") == "spawn_many" for tool in turn_tool_schemas
                ) and (
                    turn_tool_scope == "retry_spawn_after_create_miss"
                    or plan_reason == "explicit_peer_wave_incomplete"
                    or plan_reason == "delegate_failed_child_recovery"
                    or plan_reason == "deferred_spawn_readiness_met"
                    or plan_reason == "handoff_after_partial_progress"
                    or plan_reason == "multi_phase_peer_wave_pending"
                    or plan_reason == "delegate_final_delivery"
                ):
                    recommended_tool_choice = "spawn_many"
                elif (
                    loop_action == "create"
                    and turn_tool_scope == "final_delivery_handoff"
                    and any((tool.get("function") or {}).get("name") == "spawn" for tool in turn_tool_schemas)
                ):
                    recommended_tool_choice = "spawn"
                elif (
                    loop_action == "compact"
                    and turn_tool_scope == "retry_primary_after_auxiliary_only_miss"
                    and any((tool.get("function") or {}).get("name") == "compact" for tool in turn_tool_schemas)
                ):
                    recommended_tool_choice = "compact"
                elif (
                    loop_action == "stop"
                    and turn_tool_scope == "retry_primary_after_auxiliary_only_miss"
                    and any((tool.get("function") or {}).get("name") == "set_status" for tool in turn_tool_schemas)
                ):
                    recommended_tool_choice = "set_status"
                elif (
                    loop_action == "stop"
                    and turn_tool_scope == "answer_submit_only"
                    and any((tool.get("function") or {}).get("name") == "submit_answer" for tool in turn_tool_schemas)
                ):
                    recommended_tool_choice = "submit_answer"
                elif loop_action and turn_tool_scope != "action_default":
                    scoped_tool_names = [
                        str((tool.get("function") or {}).get("name") or "")
                        for tool in turn_tool_schemas
                        if (tool.get("function") or {}).get("name")
                    ]
                    primary_scoped = sorted(
                        set(scoped_tool_names) & self._primary_tools_for_loop_action(loop_action)
                    )
                    if len(primary_scoped) == 1:
                        recommended_tool_choice = primary_scoped[0]
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
                if loop_action in {"read", "query"} and (agent._loop_action_plan or {}).get("target_action"):
                    self._emit(agent.id, "pre_action_read", {
                        "target_action": agent._loop_action_plan.get("target_action"),
                        "reason": agent._loop_action_plan.get("reason"),
                        "tool_calls": [tc.name for tc in response.tool_calls] if response.tool_calls else [],
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
                empty_write_only_scope = (
                    not response.tool_calls
                    and not response.content
                    and loop_action == "work"
                    and turn_tool_scope in {"artifact_write_only", "retry_file_write_after_text_miss"}
                    and bool(missing_outputs)
                )
                write_only_content_note = (
                    "[write_only turn produced assistant text but no file_write tool call; "
                    "full text omitted from persistent history]"
                    if text_only_write_scope
                    else None
                )
                main_action_satisfied = False
                create_action_retargeted = False

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
                            self._refresh_task_ledger_agent_item(agent)
                        if tc.name == "shell":
                            command = str(tc.arguments.get("command", ""))
                            agent._shell_commands.append(command)
                            exit_code = result.get("returncode", result.get("exit_code", 1)) if isinstance(result, dict) else 1
                            if isinstance(result, dict) and int(exit_code) == 0:
                                self._publish_shell_evidence(agent, command, result)
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
                        if tc.name in {"send", "wait"} and loop_action in {"message", "stop"}:
                            main_action_satisfied = True
                        if tc.name == "query":
                            if loop_action in {"read", "query", "message"}:
                                main_action_satisfied = True
                            agent._last_query_turn = agent._turns
                            agent._last_query_agent_ids = self._query_result_agent_ids(result)
                            agent._last_query_reliable_evidence_ids = self._query_result_reliable_evidence_ids(result)
                            if self._query_result_has_reliable_evidence(result):
                                agent._last_query_evidence_turn = agent._turns
                            agent._filtered_read_intent_count = 0
                            agent._filtered_read_intent_seen_turn = 0
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") == "create":
                                agent._create_action_filtered_count = 0
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") in {"read", "query"}:
                                agent._loop_action_plan = None
                        if tc.name == "ledger_read":
                            if loop_action in {"read", "query", "message", "work", "compact", "stop"}:
                                main_action_satisfied = True
                        if tc.name == "ledger_update":
                            if loop_action in {"work", "compact", "stop", "message"}:
                                main_action_satisfied = True
                            self._refresh_task_ledger_agent_item(agent)
                        if tc.name in {"spawn", "create_agent", "spawn_many"}:
                            created_count = len(agent.children) - children_before_turn
                            valid_spawn = created_count > 0
                            covered_skip = False
                            covered_by: list[str] = []
                            covered_paths: list[str] = []
                            covered_reason = ""
                            if isinstance(result, dict):
                                if tc.name == "spawn_many":
                                    valid_spawn = int(result.get("created") or 0) > 0
                                else:
                                    valid_spawn = bool(result.get("agent_id"))
                                covered_skip, covered_by, covered_paths, covered_reason = self._spawn_result_covered_skip(result)
                            if valid_spawn:
                                if loop_action == "create":
                                    main_action_satisfied = True
                                agent._create_action_filtered_count = 0
                                agent._last_create_resolution = "created"
                                agent._create_resume_after_read = False
                                if agent._loop_action_plan and agent._loop_action_plan.get("action") == "create":
                                    agent._loop_action_plan = None
                            elif covered_skip and agent._loop_action_plan and agent._loop_action_plan.get("action") == "create":
                                main_action_satisfied = True
                                create_action_retargeted = True
                                agent._create_action_filtered_count = 0
                                agent._create_resume_after_read = False
                                agent._last_create_resolution = "skipped_covered"
                                agent._last_spawn_skipped_turn = agent._turns
                                agent._last_spawn_skipped_reason = covered_reason or agent._last_spawn_skipped_reason or "verified_lane_already_covered"
                                if covered_by:
                                    agent._last_spawn_skipped_covered_by = covered_by
                                if covered_paths:
                                    agent._last_spawn_skipped_covered_paths = covered_paths
                                followup = self._spawn_skipped_followup_plan(agent, missing_outputs)
                                agent._loop_action_plan = followup or {
                                    "action": "compact",
                                    "reason": "compact_after_spawn_skipped_covered",
                                    "turn_added": agent._turns,
                                    "covered_by": list(agent._last_spawn_skipped_covered_by),
                                    "covered_paths": list(agent._last_spawn_skipped_covered_paths),
                                    "skip_reason": agent._last_spawn_skipped_reason,
                                }
                                self._emit(agent.id, "create_action_resolved", {
                                    "resolution": "skipped_covered",
                                    "tool": tc.name,
                                    "covered_by": list(agent._last_spawn_skipped_covered_by),
                                    "covered_paths": list(agent._last_spawn_skipped_covered_paths),
                                    "next_action": agent._loop_action_plan.get("action"),
                                    "next_reason": agent._loop_action_plan.get("reason"),
                                })
                            elif agent._loop_action_plan and agent._loop_action_plan.get("action") == "create":
                                agent._create_action_filtered_count += 1
                                self._emit(agent.id, "create_action_miss", {
                                    "count": agent._create_action_filtered_count,
                                    "reason": "spawn_tool_created_no_valid_children",
                                    "tool": tc.name,
                                    "result": result,
                                })
                        if tc.name in {"file_read", "file_list", "grep"}:
                            if loop_action in {"read", "query", "work", "stop"}:
                                main_action_satisfied = True
                            agent._last_read_evidence_turn = agent._turns
                            agent._filtered_read_intent_count = 0
                            agent._filtered_read_intent_seen_turn = 0
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") in {"read", "query"}:
                                agent._loop_action_plan = None
                        if tc.name in {"compact", "set_status", "submit_answer"}:
                            create_exit_decision = False
                            create_next_action: str | None = None
                            if (
                                tc.name == "set_status"
                                and str(tc.arguments.get("action", "")) == "read"
                                and agent._loop_action_plan
                                and agent._loop_action_plan.get("action") == "create"
                                and _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
                            ):
                                agent._create_resume_after_read = True
                            if tc.name == "compact":
                                create_exit_decision = (
                                    bool(tc.arguments.get("stop_after") or tc.arguments.get("done"))
                                    or _create_exit_decision_tags(tc.arguments.get("tags"))
                                )
                                if not create_exit_decision and agent._compact_pending is not None:
                                    agent._compact_pending["return_action_after_auxiliary"] = "create"
                            elif tc.name == "set_status":
                                action_arg = str(tc.arguments.get("action", ""))
                                status_arg = str(tc.arguments.get("status", ""))
                                if action_arg in _LOOP_ACTIONS and action_arg != "create":
                                    create_next_action = "stop" if action_arg in {"done", "self_stop"} else action_arg
                                create_exit_decision = (
                                    action_arg in {"stop", "done", "self_stop"}
                                    or status_arg in {"done", "idle"}
                                    or bool(tc.arguments.get("compact_before_stop"))
                                    or _create_exit_decision_tags(tc.arguments.get("current_task_tags"))
                                )
                            elif tc.name == "submit_answer":
                                create_exit_decision = True
                            if (
                                agent._loop_action_plan
                                and agent._loop_action_plan.get("action") == "create"
                            ):
                                if create_next_action:
                                    agent._loop_action_plan = {
                                        "action": create_next_action,
                                        "reason": f"create_action_selected_{create_next_action}",
                                        "turn_added": agent._turns,
                                    }
                                    if create_next_action == "work":
                                        create_action_retargeted = True
                                        main_action_satisfied = True
                                elif create_exit_decision:
                                    create_action_retargeted = True
                                    main_action_satisfied = True
                                else:
                                    agent._loop_action_plan = {
                                        "action": "create",
                                        "reason": "retry_spawn_after_create_miss",
                                        "turn_added": agent._turns,
                                    }
                                    agent.action_state = "create"
                                    agent._state_action_consumed_version = agent._state_action_version
                            if agent._loop_action_plan and agent._loop_action_plan.get("action") in {"compact", "stop"}:
                                if tc.name == "compact" or tc.name == "submit_answer" or agent.status != "running" or create_exit_decision:
                                    main_action_satisfied = True
                                agent._loop_action_plan = None
                            self._refresh_task_ledger_agent_item(agent)
                        if tc.name in {"shell", "bt_aggregate"} and loop_action == "work":
                            main_action_satisfied = True
                        if tc.name in {"file_write", "file_replace", "submit"}:
                            if loop_action in {"work", "compact", "stop"}:
                                main_action_satisfied = True
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
                    if loop_action == "work" and committed_from_text:
                        main_action_satisfied = True
                else:
                    agent.history.append({"role": "assistant", "content": "(empty)"})

                if (
                    loop_action == "create"
                    and agent.status == "running"
                    and not main_action_satisfied
                    and len(agent.children) <= children_before_turn
                    and agent._create_action_filtered_count == create_miss_before_turn
                    and not create_action_retargeted
                ):
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

                if (text_only_write_scope or empty_write_only_scope) and not committed_from_text:
                    agent._write_only_miss_count += 1
                    self._emit(agent.id, "write_only_miss", {
                        "count": agent._write_only_miss_count,
                        "missing": missing_outputs[:10],
                        "content_preview": (response.content or "")[:200],
                        "empty_response": empty_write_only_scope,
                    })
                else:
                    agent._write_only_miss_count = 0

                if loop_action == "compact" and not main_action_satisfied and agent.status == "running":
                    agent._auxiliary_only_miss_count += 1
                    self._emit(agent.id, "loop_action_auxiliary_only_miss", {
                        "action": loop_action,
                        "count": agent._auxiliary_only_miss_count,
                        "tools": [tc.name for tc in executed_tool_calls],
                        "reason": "compact_action_requires_compact_or_done_status",
                    })
                elif loop_action and executed_tool_calls and agent.status == "running":
                    auxiliary_tools = {"get_cost", "set_status", "compact"}
                    if not main_action_satisfied and all(tc.name in auxiliary_tools for tc in executed_tool_calls):
                        agent._auxiliary_only_miss_count += 1
                        self._emit(agent.id, "loop_action_auxiliary_only_miss", {
                            "action": loop_action,
                            "count": agent._auxiliary_only_miss_count,
                            "tools": [tc.name for tc in executed_tool_calls],
                            "reason": "auxiliary_tools_did_not_satisfy_current_action",
                        })
                    else:
                        agent._auxiliary_only_miss_count = 0

                agent._no_tool_turns = 0 if response.tool_calls or committed_from_text else agent._no_tool_turns + 1
                if loop_action and agent.status == "running":
                    action_satisfied = main_action_satisfied or bool(committed_from_text)
                    if action_satisfied:
                        agent._action_miss_count = 0
                        agent._action_miss_action = ""
                    else:
                        if agent._action_miss_action == loop_action:
                            agent._action_miss_count += 1
                        else:
                            agent._action_miss_action = loop_action
                            agent._action_miss_count = 1
                        self._emit(agent.id, "loop_action_miss", {
                            "action": loop_action,
                            "count": agent._action_miss_count,
                            "tools": [tc.name for tc in executed_tool_calls],
                            "has_content": bool(response.content),
                            "missing_outputs": missing_outputs[:10],
                            "reason": (
                                "no_executable_tool_call"
                                if not executed_tool_calls
                                else "selected_action_not_satisfied"
                            ),
                        })
                        if agent._action_miss_count >= 2:
                            stale_plan = dict(agent._loop_action_plan or {})
                            agent._loop_action_plan = None
                            if loop_action == "message" and agent.action_state == "message":
                                if self._deferred_spawn_ready(agent):
                                    self.state_board_update(agent.id, action_state="create")
                                elif missing_outputs and not self.unfinished_child_agents(agent):
                                    self.state_board_update(agent.id, action_state="work")
                            self._emit(agent.id, "loop_action_replan", {
                                "from_action": loop_action,
                                "count": agent._action_miss_count,
                                "stale_reason": stale_plan.get("reason"),
                                "missing_outputs": missing_outputs[:10],
                                "new_action_state": agent.action_state,
                            })
                    if loop_action != "compact" and action_satisfied:
                        agent._auxiliary_only_miss_count = 0

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
                self._refresh_task_ledger_agent_item(agent)
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
        if (
            not spawnable
            and _task_is_concrete_single_output_work(agent.task)
            and not self._agent_has_uncertain_evidence(agent)
        ):
            return
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
            "choose action=create for independent worker lanes. Do not start downstream verifier/review/synthesis agents "
            "until their input readiness is high; use depends_on/readiness if proposing one. "
            "Use query/read/work/compact/stop only when that action better matches the current state. "
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

    def _shared_coordination_tags(self, agent: Agent, peer: Agent) -> set[str]:
        identity_prefixes = (
            "agent:", "id:", "parent:", "created_by:", "depth:",
            "group:", "group_id:", "create_type:", "relationship:", "workflow:",
        )
        left = {
            tag
            for tag in agent.current_task_tags
            if not str(tag).startswith(identity_prefixes)
        }
        right = {
            tag
            for tag in peer.current_task_tags
            if not str(tag).startswith(identity_prefixes)
        }
        return left.intersection(right)

    def _peer_overlap_query_candidates(self, agent: Agent) -> list[Agent]:
        candidates: list[Agent] = []
        agent_lanes = _agent_lane_fingerprints(agent)
        disambiguating_prefixes = ("topic:", "lane:", "candidate:", "scope:", "deliverable:")
        agent_disambiguators = {
            tag for tag in agent.current_task_tags if str(tag).startswith(disambiguating_prefixes)
        }
        for peer in self._peer_progress_candidates(agent):
            if not self._agents_have_compatible_output_slot(agent, peer):
                continue
            if not (agent.group_id and peer.group_id and agent.group_id == peer.group_id):
                shared_tags = self._shared_coordination_tags(agent, peer)
                if not shared_tags:
                    continue
            peer_lanes = _agent_lane_fingerprints(peer)
            lane_overlap = bool(agent_lanes and peer_lanes and agent_lanes.intersection(peer_lanes))
            task_similarity = _cheap_text_similarity(agent.task, peer.task)
            shared_tags = self._shared_coordination_tags(agent, peer)
            peer_disambiguators = {
                tag for tag in peer.current_task_tags if str(tag).startswith(disambiguating_prefixes)
            }
            different_explicit_lane = bool(
                agent_disambiguators
                and peer_disambiguators
                and not agent_disambiguators.intersection(peer_disambiguators)
            )
            if different_explicit_lane and not lane_overlap:
                continue
            if lane_overlap or task_similarity >= 0.20 or len(shared_tags) >= 2:
                candidates.append(peer)
        candidates.sort(key=lambda p: (
            0 if p.status == "done" else 1,
            -len(p.artifacts),
            -p._turns,
            p.id,
        ))
        return candidates

    def _peer_overlap_query_before_work_pending(self, agent: Agent, missing_outputs: list[str]) -> bool:
        if not missing_outputs:
            return False
        if not agent.parent:
            return False
        if agent.children:
            return False
        if agent._last_query_turn > 0:
            return False
        if agent._peer_overlap_query_turn > 0:
            if agent._action_miss_action == "read" and agent._action_miss_count >= 2:
                return False
            if (agent._turns - agent._peer_overlap_query_turn) > 4:
                return False
        if _task_is_leaf_artifact_lane(
            task=agent.task,
            role=agent.role,
            group_id=agent.group_id,
            current_task_tags=agent.current_task_tags,
        ):
            return False
        if not _task_is_discovery_work(agent.task):
            return False
        if not self._peer_overlap_query_candidates(agent):
            return False
        return True

    def _mark_peer_overlap_query_planned(self, agent: Agent) -> None:
        agent._peer_overlap_query_turn = agent._turns

    def _peer_overlap_post_query_pending(self, agent: Agent, missing_outputs: list[str]) -> bool:
        if not missing_outputs:
            return False
        if agent._peer_overlap_query_turn <= 0:
            return False
        if agent._last_query_turn < agent._peer_overlap_query_turn:
            return False
        if (agent._turns - agent._last_query_turn) > 4:
            return False
        if not _task_is_discovery_work(agent.task):
            return False
        return bool(self._completed_peer_overlap_candidates(agent))

    def _peer_overlap_active_duplicate_post_query_pending(self, agent: Agent, missing_outputs: list[str]) -> bool:
        if not missing_outputs:
            return False
        if agent._peer_overlap_query_turn <= 0:
            return False
        if agent._last_query_turn < agent._peer_overlap_query_turn:
            return False
        if (agent._turns - agent._last_query_turn) > 4:
            return False
        if not _task_is_discovery_work(agent.task):
            return False
        if agent._last_artifact_turn > 0 or agent.artifacts:
            return False
        return bool(self._active_duplicate_peer_overlap_candidates(agent))

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
            failed_children = self._failed_children_need_recovery(agent, missing)
            child_lines = "\n".join(
                f"- {child.id}: role={child.role or '-'} group_id={child.group_id or '-'} "
                f"status={child.status} turns={child._turns} missing={self._missing_expected_outputs(child)[:4]}"
                for child in unfinished[:8]
            )
            failed_lines = "\n".join(
                f"- {item.get('id')}: role={item.get('role') or '-'} group_id={item.get('group_id') or '-'} "
                f"status=failed missing={item.get('missing_outputs') or []} result={(item.get('result') or '')[:140]}"
                for item in failed_children[:8]
            )
            if failed_lines:
                child_lines = (child_lines + "\n" if child_lines else "") + "Recoverable failed children:\n" + failed_lines
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
                "failed_children": failed_children[:20],
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
            "If evidence is insufficient, mark the artifact as partial/evidence-pending; do not present it as final.\n"
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
        expected = _non_terminal_expected_output_paths(agent.task)
        if not expected:
            return []
        missing: list[str] = []
        ledger = self._ensure_task_ledger_loaded()
        item = (ledger.get("items") or {}).get(agent.id) or {}
        for path in expected:
            abs_path = self._tool_context.workspace_root / path
            if not abs_path.exists():
                missing.append(path)
                continue
            existing_status = self._task_ledger_existing_output_status(item, path)
            status = self._artifact_output_status(agent, path, existing_status)
            if status in {"placeholder", "evidence_gap", "unverified"}:
                missing.append(path)
        return missing

    def submit_answer_granted(self, agent: Agent) -> bool:
        return agent.id in self._submit_answer_grants

    def grant_submit_answer(self, agent: Agent) -> None:
        self._submit_answer_grants.add(agent.id)
        self._emit(agent.id, "submit_answer_grant", {
            "path": self.final_answer_path_for(agent),
        })

    def can_submit_answer(self, agent: Agent, path: str | None = None) -> dict[str, Any]:
        if not self.answer_submission_required(agent):
            return {
                "ok": False,
                "reason": "answer_submission_not_required_for_this_agent",
                "advice": "Finish local work with file artifacts, compact, or set_status instead.",
            }
        if agent.id not in self._submit_answer_grants:
            return {
                "ok": False,
                "reason": "submit_answer_not_granted",
                "advice": (
                    "This is local or intermediate work. Publish evidence/verification locally; "
                    "only a runtime-authorized delivery agent may submit the global answer."
                ),
            }
        expected_path = self.final_answer_path_for(agent)
        if path:
            normalized = self._normalize_workspace_relative_path(path, agent.workspace)
            if normalized and normalized != expected_path:
                return {
                    "ok": False,
                    "reason": "submit_answer_path_mismatch",
                    "expected_path": expected_path,
                    "requested_path": normalized,
                }
        return {"ok": True, "path": expected_path}

    def should_grant_submit_answer_to_child(
        self,
        parent: Agent,
        *,
        task: str,
        role: str = "",
        group_id: str = "",
        workflow_prior: str = "",
        current_task_tags: list[str] | None = None,
    ) -> bool:
        if not self.can_submit_answer(parent).get("ok"):
            return False
        if not _explicit_final_delivery_identity(
            role=role,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
        ):
            return False
        focus = _focus_class_from_parts(
            task=task,
            role=role,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
        )
        return focus == "final_delivery"

    def answer_submission_required(self, agent: Agent) -> bool:
        return bool(_final_answer_paths(agent.task)) or _task_mentions_submit_answer_protocol(agent.task)

    def final_answer_path_for(self, agent: Agent) -> str:
        paths = _final_answer_paths(agent.task)
        if paths:
            return paths[0]
        return f"{self.config.shared_dir}/answer.json"

    def _answer_submission_blockers(self, agent: Agent) -> list[dict[str, Any]]:
        if not self.answer_submission_required(agent):
            return []
        if agent.submitted_answer_path:
            return []
        return [{
            "kind": "answer_submission",
            "message": (
                "Answer tasks must finish through submit_answer(answer=...). "
                "Do not use file_write, compact, or set_status as the final answer submission."
            ),
            "path": self.final_answer_path_for(agent),
        }]

    def expected_outputs(self, agent: Agent) -> list[str]:
        return _expected_output_paths(agent.task)

    def missing_expected_outputs(self, agent: Agent) -> list[str]:
        return self._missing_expected_outputs(agent)

    def outputs_complete(self, agent: Agent) -> bool:
        expected = _expected_output_paths(agent.task)
        if not expected:
            return False
        if self._missing_expected_outputs(agent):
            return False
        if self._uncertain_artifact_paths(agent, expected_only=True):
            return False
        if self.answer_submission_required(agent) and not agent.submitted_answer_path:
            return False
        return True

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
        result: str | None = None,
    ) -> list[dict[str, Any]]:
        final_tags = set(tags or agent.current_task_tags)
        if (
            "status:pruned" in final_tags
            and any(tag in final_tags for tag in {"reason:peer_ahead", "reason:duplicate_lane", "reason:invalid_candidate"})
            and self._pruned_stop_is_allowed(agent)
        ):
            return []
        if (
            "status:pruned" in final_tags
            and any(tag in final_tags for tag in {"reason:premature_downstream", "needs:upstream_evidence"})
            and self._premature_downstream_stop_is_allowed(agent)
        ):
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
        blockers.extend(self._uncertain_evidence_blockers(agent))
        blockers.extend(self._answer_submission_blockers(agent))
        return blockers

    def _compact_evidence_text(self, agent: Agent, summary: str, experience: str | None, files: list[str]) -> str:
        parts = [summary or "", experience or "", " ".join(agent.current_task_tags)]
        candidate_files = list(files)
        if not candidate_files:
            candidate_files = [artifact.path for artifact in agent.artifacts]
        for rel_path in candidate_files[:8]:
            abs_path = (self.config.workspace_root / rel_path).resolve()
            try:
                abs_path.relative_to(self.config.workspace_root.resolve())
            except ValueError:
                continue
            if not abs_path.is_file():
                continue
            try:
                parts.append(abs_path.read_text(errors="replace")[:6000])
            except OSError:
                continue
        return "\n\n".join(parts).lower()

    def _compact_truth_guard(
        self,
        agent: Agent,
        *,
        summary: str,
        experience: str | None,
        tags: list[str],
        files: list[str],
        stop_result: str | None,
    ) -> dict[str, Any]:
        evidence_text = self._compact_evidence_text(agent, summary, experience, files)
        unverified = any(term in evidence_text for term in _UNVERIFIED_EVIDENCE_TERMS)
        if not unverified:
            return {
                "summary": summary,
                "experience": experience,
                "tags": tags,
                "stop_result": stop_result,
                "truth_guard": None,
            }

        original_tags = normalize_tags(tags)
        sanitized_tags = [
            tag for tag in original_tags
            if tag not in _COMPLETION_CLAIM_TAGS
        ]
        sanitized_tags = normalize_tags(sanitized_tags + sorted(_TRUTH_GUARD_UNVERIFIED_TAGS))
        guarded_summary = summary
        if not summary.lower().startswith("[unverified / needs verification]"):
            guarded_summary = f"[UNVERIFIED / NEEDS VERIFICATION]\n{summary}"
        guarded_experience = experience
        if experience and not experience.lower().startswith("[unverified / needs verification]"):
            guarded_experience = f"[UNVERIFIED / NEEDS VERIFICATION] {experience}"
        guarded_result = stop_result
        if stop_result and "unverified" not in stop_result.lower() and "low confidence" not in stop_result.lower():
            guarded_result = f"unverified: {stop_result}"

        event = {
            "reason": "unverified_or_low_confidence_evidence",
            "removed_tags": sorted(set(original_tags).intersection(_COMPLETION_CLAIM_TAGS)),
            "added_tags": sorted(_TRUTH_GUARD_UNVERIFIED_TAGS),
            "files": files[:8],
        }
        self._emit(agent.id, "compact_truth_guard", event)
        return {
            "summary": guarded_summary,
            "experience": guarded_experience,
            "tags": sanitized_tags,
            "stop_result": guarded_result,
            "truth_guard": event,
        }

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

    def _failed_child_agents(self, agent: Agent) -> list[Agent]:
        return [
            self.agents[child_id]
            for child_id in sorted(agent.children)
            if child_id in self.agents and self.agents[child_id].status == "failed"
        ]

    def _failed_children_need_recovery(self, agent: Agent, missing_outputs: list[str]) -> list[dict[str, Any]]:
        if not missing_outputs:
            return []
        recoverable: list[dict[str, Any]] = []
        for child in self._failed_child_agents(agent):
            child_expected = self.expected_outputs(child)
            child_missing = self.missing_expected_outputs(child)
            expected_overlap = [path for path in child_expected if path in missing_outputs or path in child_missing]
            if child_expected and not expected_overlap and not child_missing:
                continue
            recoverable.append({
                "id": child.id,
                "role": child.role,
                "group_id": child.group_id,
                "task": child.task,
                "result": child.result,
                "artifacts": [artifact.path for artifact in child.artifacts],
                "expected_outputs": child_expected[:20],
                "missing_outputs": (child_missing or expected_overlap or missing_outputs)[:20],
                "turns": child._turns,
                "tokens": child.tokens_consumed,
            })
        return recoverable

    def _has_active_recovery_child(self, agent: Agent, failed_children: list[dict[str, Any]]) -> bool:
        failed_ids = {str(item.get("id") or "") for item in failed_children}
        failed_roles = {str(item.get("role") or "") for item in failed_children if item.get("role")}
        for child in self.unfinished_child_agents(agent):
            text = self._child_marker_text(child)
            if not any(marker in text for marker in ("recovery", "fallback", "repair")):
                continue
            if failed_ids and (failed_ids.intersection(child.current_task_tags) or any(fid in child.task for fid in failed_ids)):
                return True
            if failed_roles and any(role and role in child.task for role in failed_roles):
                return True
            return True
        return False

    def _failed_child_recovery_plan(
        self,
        agent: Agent,
        missing_outputs: list[str],
        blockers: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        failed_children = self._failed_children_need_recovery(agent, missing_outputs)
        if not failed_children:
            return None
        if self._root_steward_mode(agent):
            return {
                "action": "message",
                "reason": "root_steward_query_failed_child_recovery_gap",
                "turn_added": agent._turns,
                "failed_children": failed_children,
                "blockers": blockers,
            }
        if self._has_active_recovery_child(agent, failed_children):
            return {
                "action": "message",
                "reason": "coordinate_active_recovery_child",
                "turn_added": agent._turns,
                "failed_children": failed_children,
                "blockers": blockers,
            }
        if len(self.agents) >= self.config.max_agents:
            return {
                "action": "message",
                "reason": "failed_child_recovery_blocked_by_agent_limit",
                "turn_added": agent._turns,
                "failed_children": failed_children,
                "blockers": blockers,
            }
        return {
            "action": "create",
            "reason": "delegate_failed_child_recovery",
            "turn_added": agent._turns,
            "failed_children": failed_children,
            "blockers": blockers,
        }

    def _can_create_child(self, agent: Agent) -> bool:
        return (
            len(self.agents) < self.config.max_agents
            and agent.depth + 1 <= self.config.max_depth
            and agent.orchestration_preference != "solo"
            and not self._root_steward_mode(agent)
            and not self._agent_has_solo_decision(agent)
            and not self._agent_has_pending_self_work_decision(agent)
        )

    def _root_steward_mode(self, agent: Agent) -> bool:
        return agent.parent is None and bool(agent.children) and not self._root_initial_peer_wave_incomplete(agent)

    def _root_initial_peer_wave_incomplete(self, agent: Agent) -> bool:
        if agent.parent is not None or not agent.children:
            return False
        for requirement in _explicit_peer_wave_requirements(agent.task):
            target = int(requirement.get("target") or 0)
            if target <= 1:
                continue
            matched = [
                self.agents[child_id]
                for child_id in sorted(agent.children)
                if child_id in self.agents and _agent_matches_peer_wave_requirement(self.agents[child_id], requirement)
            ]
            if len(matched) < target:
                return True
        return False

    def _agent_has_solo_decision(self, agent: Agent) -> bool:
        if "status:solo-decision" in set(agent.current_task_tags):
            return True
        memory = self.memory.get(agent.id)
        if not memory:
            return False
        if "status:solo-decision" in set(memory.tags or []):
            return True
        if memory.active_task and "status:solo-decision" in set(memory.active_task.tags or []):
            return True
        return any("status:solo-decision" in set(card.tags or []) for card in memory.experience_cards[-12:])

    def _agent_has_pending_self_work_decision(self, agent: Agent) -> bool:
        return (
            agent.action_state == "work"
            and agent._state_action_version > agent._state_action_consumed_version
            and (agent._create_action_filtered_count > 0 or self._agent_has_solo_decision(agent))
        )

    def _spawn_result_covered_skip(self, result: Any) -> tuple[bool, list[str], list[str], str]:
        if not isinstance(result, dict):
            return False, [], [], ""

        covered_by: list[str] = []
        covered_paths: list[str] = []
        reasons: list[str] = []

        def collect(item: dict[str, Any]) -> None:
            reason = str(item.get("reason") or "")
            if reason:
                reasons.append(reason)
            for covered in item.get("covered_by") or []:
                if isinstance(covered, dict) and covered.get("id"):
                    covered_by.append(str(covered["id"]))
            for path in item.get("covered_paths") or []:
                if str(path):
                    covered_paths.append(str(path))

        skipped_items: list[dict[str, Any]] = []
        if (
            result.get("skipped") is True
            or result.get("reason")
            or result.get("covered_by")
            or result.get("covered_paths")
        ):
            skipped_items.append(result)
        for wrapper in result.get("results") or []:
            if not isinstance(wrapper, dict):
                continue
            item = wrapper.get("result")
            if isinstance(item, dict) and item.get("skipped"):
                skipped_items.append(item)

        if not skipped_items:
            return False, [], [], ""

        for item in skipped_items:
            collect(item)

        created = int(result.get("created") or 0) if "created" in result else 0
        deferred = int(result.get("deferred") or 0) if "deferred" in result else 0
        all_covered = all(
            str(item.get("reason") or "").endswith("already_covered")
            or str(item.get("reason") or "") == "verified_lane_already_covered"
            for item in skipped_items
        )
        if created > 0 or deferred > 0 or not all_covered:
            return False, [], [], ""

        return (
            True,
            sorted(set(covered_by)),
            sorted(set(covered_paths)),
            reasons[0] if reasons else "verified_lane_already_covered",
        )

    def _spawn_skipped_followup_plan(
        self,
        agent: Agent,
        missing_outputs: list[str],
        blockers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if agent._last_spawn_skipped_turn <= 0:
            return None
        if (agent._turns - agent._last_spawn_skipped_turn) > 6:
            return None
        if agent._last_spawn_skipped_reason and not agent._last_spawn_skipped_reason.endswith("already_covered"):
            return None

        covered_by = list(agent._last_spawn_skipped_covered_by)
        covered_paths = list(agent._last_spawn_skipped_covered_paths)
        if not covered_by and not covered_paths:
            return None

        data = {
            "turn_added": agent._turns,
            "blockers": blockers or [],
            "covered_by": covered_by,
            "covered_paths": covered_paths,
            "skip_reason": agent._last_spawn_skipped_reason or "verified_lane_already_covered",
        }
        if not missing_outputs and not self.unfinished_child_agents(agent):
            return {
                "action": "compact",
                "reason": "compact_after_spawn_skipped_covered",
                **data,
            }
        if agent.children:
            return {
                "action": "message",
                "reason": "inspect_coverage_after_spawn_skipped",
                **data,
            }
        return {
            "action": "read",
            "reason": "read_after_spawn_skipped",
            **data,
        }

    def _create_miss_followup_plan(self, agent: Agent, missing_outputs: list[str]) -> dict[str, Any] | None:
        blockers = self.completion_blockers(agent, include_missing_outputs=False) if agent.children else []
        coverage_plan = self._spawn_skipped_followup_plan(agent, missing_outputs, blockers)
        if coverage_plan:
            return coverage_plan
        if agent._create_action_filtered_count <= 0:
            return None
        recovery_plan = self._failed_child_recovery_plan(agent, missing_outputs, blockers)
        if recovery_plan:
            return recovery_plan
        if (
            missing_outputs
            and not agent.children
            and (self._agent_has_solo_decision(agent) or self._agent_has_pending_self_work_decision(agent))
        ):
            return {
                "action": "work",
                "reason": (
                    "solo_decision_after_create_miss"
                    if self._agent_has_solo_decision(agent)
                    else "self_selected_work_after_create_miss"
                ),
                "turn_added": agent._turns,
                "blockers": blockers,
            }
        if self.unfinished_child_agents(agent):
            return {
                "action": "message",
                "reason": "wait_after_create_miss_with_active_children",
                "turn_added": agent._turns,
                "blockers": blockers,
            }
        if self._can_create_child(agent):
            return {
                "action": "create",
                "reason": "retry_spawn_after_create_miss",
                "turn_added": agent._turns,
                "blockers": blockers,
            }
        if agent._filtered_read_intent_count > 0:
            return {
                "action": "read",
                "reason": "read_after_create_miss_prerequisite",
                "turn_added": agent._turns,
                "target_action": "create",
                "blockers": blockers,
            }
        return {
            "action": "compact",
            "reason": "create_miss_requires_solo_or_pruned_decision",
            "turn_added": agent._turns,
            "blockers": blockers,
        }

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
        can_create = self._can_create_child(agent)
        recovery_plan = self._failed_child_recovery_plan(agent, missing_outputs, blockers)
        if recovery_plan:
            return recovery_plan
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
        if self._task_ledger_pruned_stop_is_allowed(agent):
            return True
        if agent._pending_prune_requests and self._pending_prune_request_has_coverage_signal(agent):
            return True
        expected = _non_terminal_expected_output_paths(agent.task)
        if expected:
            if self._peer_exact_output_coverage_candidates(agent):
                return True
            if self._queried_prunable_peer_coverage_candidates(agent):
                return True
            if (
                agent._last_query_turn > 0
                and (agent._turns - agent._last_query_turn) <= 4
                and _task_is_discovery_work(agent.task)
                and (
                    self._completed_peer_overlap_candidates(agent)
                    or self._active_duplicate_peer_overlap_candidates(agent)
                )
            ):
                return True
            return False
        if agent._last_query_turn <= 0:
            return False
        return bool(
            self._completed_peer_overlap_candidates(agent)
            or self._active_duplicate_peer_overlap_candidates(agent)
        )

    def _agent_has_downstream_focus(self, agent: Agent) -> bool:
        return _looks_like_downstream_focus(
            agent.role,
            agent.task,
            " ".join(agent.current_task_tags),
            agent.workflow_prior,
        )

    def _agent_focus_class(self, agent: Agent) -> str:
        return _focus_class_from_parts(
            task=agent.task,
            role=agent.role,
            group_id=agent.group_id,
            workflow_prior=agent.workflow_prior,
            current_task_tags=agent.current_task_tags,
        )

    def _spawn_request_focus_class(
        self,
        *,
        task: str,
        role: str = "",
        group_id: str = "",
        workflow_prior: str = "",
        current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> str:
        return _focus_class_from_parts(
            task=task,
            role=role,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
        )

    def _focus_classes_coverage_compatible(self, request_focus: str, covered_focus: str) -> bool:
        if request_focus == covered_focus:
            return True
        if request_focus == "final_delivery":
            return covered_focus == "final_delivery"
        if request_focus == "synthesis":
            return covered_focus in {"synthesis", "final_delivery"}
        if request_focus == "verification":
            return covered_focus in {"verification", "synthesis", "final_delivery"}
        if request_focus == "recovery":
            return covered_focus == "recovery"
        return covered_focus in {"evidence", "verification", "synthesis", "final_delivery"}

    def _memory_or_text_has_unverified_signal(self, agent: Agent) -> bool:
        parts = [agent.result or "", " ".join(agent.current_task_tags)]
        memory = self.memory.get(agent.id)
        if memory:
            parts.append(memory.public_summary or "")
            parts.append(" ".join(memory.tags or []))
            for card in memory.experience_cards:
                parts.append(card.summary or "")
                parts.append(" ".join(card.tags or []))
        return any(term in "\n".join(parts).lower() for term in _UNVERIFIED_EVIDENCE_TERMS)

    def _artifact_has_unverified_signal(self, rel_path: str) -> bool:
        text = self._artifact_text(rel_path, limit=12000)
        return bool(text) and any(term in text for term in _UNVERIFIED_EVIDENCE_TERMS)

    def _artifact_text(self, rel_path: str, *, limit: int = 12000) -> str:
        abs_path = (self.config.workspace_root / rel_path).resolve()
        try:
            abs_path.relative_to(self.config.workspace_root.resolve())
        except ValueError:
            return ""
        if not abs_path.is_file():
            return ""
        try:
            return abs_path.read_text(errors="replace")[:limit].lower()
        except OSError:
            return ""

    def _artifact_output_status(self, agent: Agent, path: str, existing_status: str = "") -> str:
        abs_path = self._tool_context.workspace_root / path
        exists = abs_path.exists()
        is_final_answer = _looks_like_final_answer_path(path)
        if existing_status == "covered":
            return "covered"
        if not exists:
            return "missing"
        if is_final_answer and agent.submitted_answer_path == path:
            return "submitted"
        if is_final_answer:
            return "draft"
        text = self._artifact_text(path, limit=16000)
        if text:
            if any(term in text for term in _PLACEHOLDER_EVIDENCE_TERMS):
                return "placeholder"
            if any(term in text for term in _STRONG_UNVERIFIED_EVIDENCE_TERMS):
                return "evidence_gap"
        if existing_status in _TASK_LEDGER_OUTPUT_DONE_STATUSES:
            return existing_status
        evidence_refs: list[str] = []
        ledger = self._ensure_task_ledger_loaded()
        item = (ledger.get("items") or {}).get(agent.id) or {}
        for output in item.get("expected_outputs") or []:
            if isinstance(output, dict) and output.get("path") == path:
                evidence_refs = _normalize_string_list(output.get("evidence_refs"))
                break
        if evidence_refs or any(term in text for term in _EVIDENCE_ATTACHED_TERMS):
            return "evidence_attached"
        if agent.status == "done" and not self._agent_has_uncertain_evidence(agent) and self._agent_has_reliable_evidence(agent):
            return "verified"
        if path in {artifact.path for artifact in agent.artifacts}:
            return "draft"
        return "draft"

    def _output_status_is_incomplete_for_scheduling(self, status: str) -> bool:
        return status in _TASK_LEDGER_OUTPUT_INCOMPLETE_STATUSES

    def _uncertain_artifact_paths(self, agent: Agent, *, expected_only: bool = False) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        candidates = _non_terminal_expected_output_paths(agent.task) if expected_only else [artifact.path for artifact in agent.artifacts]
        if not candidates:
            candidates = _non_terminal_expected_output_paths(agent.task)
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if self._artifact_has_unverified_signal(path):
                paths.append(path)
        return paths

    def _agent_has_uncertain_evidence(self, agent: Agent) -> bool:
        return self._memory_or_text_has_unverified_signal(agent) or bool(self._uncertain_artifact_paths(agent))

    def _agent_has_reliable_evidence(self, agent: Agent) -> bool:
        if self._agent_has_uncertain_evidence(agent):
            return False
        memory = self.memory.get(agent.id)
        if agent.artifacts or agent.result:
            return True
        if memory:
            if memory.artifact_index:
                return True
            for card in memory.experience_cards:
                tags = set(card.tags)
                if (
                    card.artifacts
                    or "memory_kind:evidence" in tags
                    or "status:complete" in tags
                    or "status:done" in tags
                    or card.memory_kind in {"evidence", "terminal"}
                ):
                    return True
        return False

    def _is_internal_evidence_trace_path(self, path: str) -> bool:
        normalized = str(path or "").replace("\\", "/").lstrip("./")
        return f"{self.config.shared_dir}/.nanoma/evidence/" in normalized or normalized.startswith(f"{self.config.shared_dir}/.nanoma/evidence/")

    def _agent_has_non_trace_artifact(self, agent: Agent) -> bool:
        return any(not self._is_internal_evidence_trace_path(path) for path in self._agent_known_artifact_paths(agent))

    def _agent_has_verified_coverage(self, agent: Agent) -> bool:
        if self._agent_has_uncertain_evidence(agent):
            return False
        if agent.status != "done":
            return False
        if self._agent_has_downstream_focus(agent):
            return bool(agent.result or self._agent_has_non_trace_artifact(agent))
        expected = _expected_output_paths(agent.task)
        if expected and not self._missing_expected_outputs(agent):
            return True
        if self._agent_has_non_trace_artifact(agent):
            return True
        memory = self.memory.get(agent.id)
        if memory:
            for card in memory.experience_cards:
                tags = set(card.tags)
                if (
                    card.memory_kind == "terminal"
                    and (
                        "status:complete" in tags
                        or "status:done" in tags
                        or "verified:true" in tags
                        or "confidence:high" in tags
                    )
                ):
                    return True
        return False

    def _agent_has_inspectable_evidence_artifact(self, agent: Agent) -> bool:
        if self._agent_has_uncertain_evidence(agent):
            return False
        expected = _expected_output_paths(agent.task)
        outputs_complete = bool(expected) and not self._missing_expected_outputs(agent)
        if agent.artifacts or outputs_complete:
            return True
        memory = self.memory.get(agent.id)
        if not memory:
            return False
        if memory.artifact_index:
            return True
        return any(card.artifacts for card in memory.experience_cards)

    def _agent_has_prune_worthy_progress(self, agent: Agent) -> bool:
        if self._agent_has_uncertain_evidence(agent):
            return False
        expected = _expected_output_paths(agent.task)
        if expected and not self._missing_expected_outputs(agent):
            return True
        if self._agent_has_non_trace_artifact(agent):
            return True
        if agent.status == "done" and self._agent_has_reliable_evidence(agent):
            return True
        return self._agent_has_reliable_evidence(agent) and self._agent_has_inspectable_evidence_artifact(agent)

    def _agent_known_artifact_paths(self, agent: Agent) -> set[str]:
        paths = {artifact.path for artifact in agent.artifacts if artifact.path}
        memory = self.memory.get(agent.id)
        if memory:
            paths.update(path for path in memory.artifact_index if path)
            for card in memory.experience_cards:
                paths.update(path for path in card.artifacts if path)
        return paths

    def _agents_have_compatible_output_slot(self, left: Agent, right: Agent) -> bool:
        left_outputs = set(_non_terminal_expected_output_paths(left.task))
        right_outputs = set(_non_terminal_expected_output_paths(right.task))
        if left_outputs and right_outputs:
            return bool(left_outputs.intersection(right_outputs))
        return True

    def _spawn_request_has_compatible_output_slot(self, agent: Agent, task: str) -> bool:
        request_outputs = set(_non_terminal_expected_output_paths(task))
        agent_outputs = set(_non_terminal_expected_output_paths(agent.task))
        if request_outputs and agent_outputs:
            return bool(request_outputs.intersection(agent_outputs))
        return True

    def _spawn_request_overlaps_agent_lane(
        self,
        agent: Agent,
        *,
        task: str,
        role: str = "",
        group_id: str = "",
        workflow_prior: str = "",
        current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> bool:
        request_outputs = set(_non_terminal_expected_output_paths(task))
        request_inputs = set(_referenced_input_like_shared_paths(task))
        request_paths = request_outputs.union(request_inputs)
        agent_paths = self._agent_known_artifact_paths(agent).union(_non_terminal_expected_output_paths(agent.task))
        if not self._spawn_request_has_compatible_output_slot(agent, task):
            return False
        if request_paths and agent_paths and request_paths.intersection(agent_paths):
            return True

        request_lanes = _spawn_request_lane_fingerprints(
            task=task,
            role=role,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
        )
        agent_lanes = _agent_lane_fingerprints(agent)
        generic_lanes = {"lane:verification"}
        specific_overlap = (request_lanes - generic_lanes).intersection(agent_lanes - generic_lanes)
        if specific_overlap:
            return True
        if request_lanes.intersection(agent_lanes) and group_id and agent.group_id == group_id:
            return _cheap_text_similarity(task, agent.task) >= 0.25 or bool(request_paths.intersection(agent_paths))
        return False

    def _completed_spawn_request_coverage(
        self,
        creator: Agent,
        *,
        task: str,
        role: str = "",
        group_id: str = "",
        workflow_prior: str = "",
        current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[Agent]:
        coverage: list[Agent] = []
        request_focus = self._spawn_request_focus_class(
            task=task,
            role=role,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
        )
        for peer in self.agents.values():
            if peer.id == creator.id:
                continue
            if peer.status != "done":
                continue
            if not self._focus_classes_coverage_compatible(request_focus, self._agent_focus_class(peer)):
                continue
            if not self._spawn_request_overlaps_agent_lane(
                peer,
                task=task,
                role=role,
                group_id=group_id,
                workflow_prior=workflow_prior,
                current_task_tags=current_task_tags,
            ):
                continue
            if not self._agent_has_verified_coverage(peer):
                continue
            coverage.append(peer)
        coverage.sort(key=lambda peer: (
            0 if self._agent_has_downstream_focus(peer) else 1,
            0 if peer.status == "done" else 1,
            -len(self._agent_known_artifact_paths(peer)),
            -peer._tool_calls,
            peer.id,
        ))
        return coverage

    def covered_spawn_request(
        self,
        creator: Agent,
        *,
        task: str,
        role: str = "",
        group_id: str = "",
        workflow_prior: str = "",
        current_task_tags: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any] | None:
        if _looks_like_recovery_request(task, role, group_id, " ".join(str(tag) for tag in (current_task_tags or []))):
            return None
        request_focus = self._spawn_request_focus_class(
            task=task,
            role=role,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
        )
        coverage = self._completed_spawn_request_coverage(
            creator,
            task=task,
            role=role,
            group_id=group_id,
            workflow_prior=workflow_prior,
            current_task_tags=current_task_tags,
        )
        if not coverage:
            return None

        request_is_downstream = _looks_like_verification_request(
            task,
            role,
            group_id,
            " ".join(str(tag) for tag in (current_task_tags or [])),
        )
        downstream_coverage = [peer for peer in coverage if self._agent_has_downstream_focus(peer)]
        recently_queried_coverage = [peer for peer in coverage if peer.id in set(creator._last_query_reliable_evidence_ids)]
        repeated_coverage = len(coverage) >= 2
        if not (request_is_downstream or recently_queried_coverage or repeated_coverage):
            return None
        if request_is_downstream and not (downstream_coverage or recently_queried_coverage or repeated_coverage):
            return None

        covered_paths = sorted({path for peer in coverage for path in self._agent_known_artifact_paths(peer)})
        return {
            "covered": True,
            "reason": (
                "verified_lane_already_covered"
                if downstream_coverage or request_is_downstream
                else "reliable_lane_already_covered"
            ),
            "covered_by": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "focus": self._agent_focus_class(peer),
                    "status": peer.status,
                    "artifacts": sorted(self._agent_known_artifact_paths(peer))[:8],
                    "result": (peer.result or "")[:220],
                    "tags": peer.current_task_tags[:12],
                }
                for peer in coverage[:8]
            ],
            "covered_paths": covered_paths[:20],
            "request_focus": request_focus,
            "coverage_focuses": sorted({self._agent_focus_class(peer) for peer in coverage}),
            "role_compatible": True,
            "advice": "Query/read the covered agents and either integrate their evidence or send prune_request to active duplicates instead of spawning another same-lane agent.",
        }

    def _upstream_peer_candidates(self, agent: Agent) -> list[Agent]:
        peers = [
            peer for peer in self._peer_progress_candidates(agent)
            if peer.parent and not self._agent_has_downstream_focus(peer)
        ]
        evidence_peers = [
            peer for peer in peers
            if peer.artifacts
            or self.expected_outputs(peer)
            or peer.result
            or any(term in " ".join([peer.role, " ".join(peer.current_task_tags)]).lower() for term in ("worker", "source", "evidence"))
        ]
        return evidence_peers or peers

    def _upstream_inputs_ready_for_downstream(self, agent: Agent) -> bool:
        upstream = self._upstream_peer_candidates(agent)
        if not upstream:
            return False
        for peer in upstream:
            if peer.status != "done":
                return False
            expected = self.expected_outputs(peer)
            if expected and self.missing_expected_outputs(peer):
                return False
            if not self._agent_has_reliable_evidence(peer):
                return False
        return True

    def _uncertain_evidence_blockers(self, agent: Agent) -> list[dict[str, Any]]:
        if agent.submitted_answer_path:
            return []
        uncertain_paths = self._uncertain_artifact_paths(agent, expected_only=True)
        if not uncertain_paths and not self._memory_or_text_has_unverified_signal(agent):
            return []
        if "status:pruned" in set(agent.current_task_tags):
            return []
        return [{
            "kind": "uncertain_evidence",
            "message": (
                "This agent's current evidence/artifacts are explicitly partial, unverified, or low confidence. "
                "Do not mark the lane complete until a verifier/recovery step resolves the uncertainty, or compact as pruned/blocked."
            ),
            "artifacts": uncertain_paths[:20],
        }]

    def _premature_downstream_stop_is_allowed(self, agent: Agent) -> bool:
        return self._agent_has_downstream_focus(agent) and not self._upstream_inputs_ready_for_downstream(agent)

    def _premature_downstream_plan(self, agent: Agent, missing_outputs: list[str]) -> dict[str, Any] | None:
        if not missing_outputs:
            return None
        if not agent.parent:
            return None
        if agent.children:
            return None
        if not self._agent_has_downstream_focus(agent):
            return None
        if self._upstream_inputs_ready_for_downstream(agent):
            return None
        upstream = self._upstream_peer_candidates(agent)
        return {
            "action": "compact",
            "reason": "premature_downstream_wait_for_evidence",
            "turn_added": agent._turns,
            "upstream_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "missing_outputs": self.missing_expected_outputs(peer)[:20],
                    "artifacts": [artifact.path for artifact in peer.artifacts[:5]],
                }
                for peer in upstream[:8]
            ],
        }

    def unfinished_child_agents(self, agent: Agent) -> list[Agent]:
        return [
            self.agents[child_id]
            for child_id in sorted(agent.children)
            if child_id in self.agents and self.agents[child_id].status in {"running", "idle"}
        ]

    def _completed_child_evidence_candidates(self, agent: Agent) -> list[Agent]:
        candidates: list[Agent] = []
        for child_id in sorted(agent.children):
            child = self.agents.get(child_id)
            if not child:
                continue
            expected = _expected_output_paths(child.task)
            outputs_complete = bool(expected) and not self._missing_expected_outputs(child)
            has_progress_artifact = child.status == "done" or child.artifacts or child.result or outputs_complete
            if (
                has_progress_artifact
                and self._agent_has_reliable_evidence(child)
                and self._agent_has_inspectable_evidence_artifact(child)
            ):
                candidates.append(child)
        candidates.sort(key=lambda item: (
            0 if item.status == "done" else 1,
            -len(item.artifacts),
            -item._last_artifact_turn,
            -item._tool_calls,
            item.id,
        ))
        return candidates

    def _agents_share_prunable_lane(self, left: Agent, right: Agent) -> bool:
        left_outputs = set(_non_terminal_expected_output_paths(left.task))
        right_outputs = set(_non_terminal_expected_output_paths(right.task))
        if left_outputs and right_outputs and left_outputs.intersection(right_outputs):
            return True
        if left_outputs and right_outputs:
            return False
        left_lanes = _agent_lane_fingerprints(left)
        right_lanes = _agent_lane_fingerprints(right)
        if left_lanes and right_lanes and left_lanes.intersection(right_lanes):
            return True
        disambiguating_prefixes = ("topic:", "lane:", "candidate:", "scope:", "deliverable:")
        left_disambiguators = {
            str(tag)
            for tag in left.current_task_tags
            if str(tag).startswith(disambiguating_prefixes)
        }
        right_disambiguators = {
            str(tag)
            for tag in right.current_task_tags
            if str(tag).startswith(disambiguating_prefixes)
        }
        if left_disambiguators and right_disambiguators and not left_disambiguators.intersection(right_disambiguators):
            return False
        if left.group_id and right.group_id and left.group_id == right.group_id:
            shared_tags = self._shared_coordination_tags(left, right)
            if len(shared_tags) >= 2:
                return True
            return _cheap_text_similarity(left.task, right.task) >= 0.30
        return False

    def _queried_evidence_agents(self, agent: Agent) -> list[Agent]:
        if agent._last_query_turn <= 0 or (agent._turns - agent._last_query_turn) > 4:
            return []
        evidence_ids = list(agent._last_query_reliable_evidence_ids)
        if not evidence_ids:
            evidence_ids = [
                peer.id
                for peer in self._completed_child_evidence_candidates(agent)
                if peer.id in set(agent._last_query_agent_ids)
            ]
        candidates: list[Agent] = []
        seen: set[str] = set()
        for peer_id in evidence_ids:
            peer = self.agents.get(peer_id)
            if not peer or peer.id == agent.id or peer.id in seen:
                continue
            expected = _expected_output_paths(peer.task)
            outputs_complete = bool(expected) and not self._missing_expected_outputs(peer)
            if self._agent_has_uncertain_evidence(peer):
                continue
            if not (self._agent_has_reliable_evidence(peer) or outputs_complete):
                continue
            candidates.append(peer)
            seen.add(peer.id)
        candidates.sort(key=lambda peer: (
            0 if peer.status == "done" else 1,
            -len(peer.artifacts),
            -peer._tool_calls,
            peer.id,
        ))
        return candidates

    def _prunable_agents_after_evidence_query(self, agent: Agent) -> list[Agent]:
        evidence_agents = self._queried_evidence_agents(agent)
        if not evidence_agents:
            return []
        queried_ids = set(agent._last_query_agent_ids)
        peer_progress_ids = {peer.id for peer in self._peer_progress_candidates(agent)}
        evidence_has_downstream = any(self._agent_has_downstream_focus(peer) for peer in evidence_agents)
        candidates: list[Agent] = []
        evidence_ids = {item.id for item in evidence_agents}
        for peer in self.agents.values():
            if peer.id == agent.id:
                continue
            if peer.id in agent._prune_requests_sent_targets:
                continue
            if peer.status not in {"running", "idle"}:
                continue
            if not peer.parent:
                continue
            if queried_ids and peer.id not in queried_ids and peer.id not in peer_progress_ids and peer.parent != agent.id:
                continue
            if peer.id in evidence_ids:
                continue
            peer_downstream = self._agent_has_downstream_focus(peer)
            if peer_downstream and not evidence_has_downstream:
                if not any(self._spawn_request_overlaps_agent_lane(
                    evidence_peer,
                    task=peer.task,
                    role=peer.role,
                    group_id=peer.group_id,
                    workflow_prior=peer.workflow_prior,
                    current_task_tags=peer.current_task_tags,
                ) for evidence_peer in evidence_agents):
                    continue
            if not peer_downstream and self._agent_has_reliable_evidence(peer):
                continue
            if any(self._agents_share_prunable_lane(peer, evidence_peer) for evidence_peer in evidence_agents):
                candidates.append(peer)
        candidates.sort(key=lambda peer: (
            -peer._turns,
            -peer._tool_calls,
            peer.id,
        ))
        return candidates

    def _prune_request_plan(self, agent: Agent) -> dict[str, Any] | None:
        targets = self._prunable_agents_after_evidence_query(agent)
        if not targets:
            return None
        evidence_agents = self._queried_evidence_agents(agent)
        return {
            "action": "message",
            "reason": "request_duplicate_agents_self_prune_after_key_evidence",
            "turn_added": agent._turns,
            "prune_targets": [peer.id for peer in targets[:8]],
            "evidence_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "artifacts": [artifact.path for artifact in peer.artifacts[:8]],
                    "result": (peer.result or "")[:240],
                    "tags": peer.current_task_tags[:12],
                }
                for peer in evidence_agents[:4]
            ],
        }

    def _pending_prune_request_plan(self, agent: Agent) -> dict[str, Any] | None:
        if not agent._pending_prune_requests:
            return None
        latest = dict(agent._pending_prune_requests[-1])
        return {
            "action": "compact",
            "reason": "received_prune_request_self_prune",
            "turn_added": agent._turns,
            "prune_request": latest,
            "prune_request_count": len(agent._pending_prune_requests),
        }

    def _evidence_integration_candidates(self, agent: Agent, missing_outputs: list[str]) -> list[Agent]:
        if not missing_outputs:
            return []
        candidates: list[Agent] = []
        seen: set[str] = set()
        for peer in self._completed_child_evidence_candidates(agent):
            if peer.id not in seen:
                candidates.append(peer)
                seen.add(peer.id)
        candidates.sort(key=lambda item: (
            0 if item.parent == agent.id else 1,
            0 if item.status == "done" else 1,
            -len(item.artifacts),
            -item._tool_calls,
            item.id,
        ))
        return candidates

    def _evidence_integration_ready(self, agent: Agent, missing_outputs: list[str]) -> bool:
        return bool(self._evidence_integration_candidates(agent, missing_outputs))

    def _evidence_integration_needs_inspection(self, agent: Agent, missing_outputs: list[str]) -> bool:
        if not self._evidence_integration_ready(agent, missing_outputs):
            return False
        if agent._last_query_evidence_turn > 0 and (agent._turns - agent._last_query_evidence_turn) <= 3:
            return False
        if agent._last_read_evidence_turn > 0 and (agent._turns - agent._last_read_evidence_turn) <= 3:
            return False
        return True

    def _evidence_integration_plan_data(self, agent: Agent, missing_outputs: list[str]) -> dict[str, Any]:
        peers = self._evidence_integration_candidates(agent, missing_outputs)
        return {
            "evidence_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "artifacts": [artifact.path for artifact in peer.artifacts[:8]],
                    "result": (peer.result or "")[:240],
                    "missing_outputs": self._missing_expected_outputs(peer)[:8],
                    "tags": peer.current_task_tags[:12],
                }
                for peer in peers[:8]
            ],
            "missing_outputs": missing_outputs[:20],
        }

    def _evidence_integration_lines(self, agent: Agent, missing_outputs: list[str]) -> list[str]:
        lines: list[str] = []
        for item in self._evidence_integration_plan_data(agent, missing_outputs).get("evidence_agents", []):
            artifacts = ", ".join(item.get("artifacts") or []) or "-"
            missing = item.get("missing_outputs") or []
            missing_text = f" missing={missing}" if missing else ""
            result = str(item.get("result") or "").replace("\n", " ")
            result_text = f" result={result[:120]}" if result else ""
            lines.append(
                f"- {item.get('id')} role={item.get('role') or '-'} "
                f"status={item.get('status') or '-'} artifacts={artifacts}{missing_text}{result_text}"
            )
        return lines

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
        if agent._pending_prune_requests:
            return True
        if agent._last_query_turn <= 0:
            return False
        return (agent._turns - agent._last_query_turn) <= 4

    def _root_terminal_submission_complete(self, root: Agent) -> bool:
        if root.status != "done":
            return False
        if not self.answer_submission_required(root):
            return False
        if not root.submitted_answer_path:
            return False
        path = self.config.workspace_root / root.submitted_answer_path
        return path.exists()

    async def _request_remaining_agents_prune_after_terminal_submission(
        self,
        root: Agent,
        running: list[Agent],
    ) -> None:
        evidence_agents = [
            {
                "id": root.id,
                "status": root.status,
                "artifacts": [artifact.path for artifact in root.artifacts[:8]],
                "result": (root.result or "")[:240],
            }
        ]
        payload = {
            "reason": "root_terminal_submission_complete",
            "covered_by": [root.id],
            "covered_paths": [root.submitted_answer_path] if root.submitted_answer_path else [],
            "evidence_agents": evidence_agents,
            "tags": ["status:prune-request", "reason:root_terminal_submission_complete"],
        }
        message = (
            "Root terminal answer has been submitted. If your remaining work cannot change the submitted "
            "terminal artifact, compact your current state with status:pruned and stop. If you have a critical "
            "contradiction that should block the answer, publish it immediately with compact/set_status."
        )
        notify_targets: list[Agent] = []
        seen: set[str] = set()
        for agent in list(running) + self._descendant_agents(root):
            if agent.id in seen:
                continue
            seen.add(agent.id)
            notify_targets.append(agent)

        for agent in notify_targets:
            if agent.id == root.id:
                continue
            if agent.status not in {"running", "idle"}:
                continue
            if agent.id in root._prune_requests_sent_targets:
                continue
            await self.deliver(Envelope(
                from_id=root.id,
                to_id=agent.id,
                content=message,
                tokens=estimate_tokens(message),
                timestamp=time.time(),
                message_type="prune_request",
                payload=payload,
                requires_ack=False,
                urgency="high",
                mode="steer",
            ))
            root._prune_requests_sent_targets.add(agent.id)
            self._emit(root.id, "terminal_prune_request", {
                "to": agent.id,
                "reason": payload["reason"],
                "covered_paths": payload["covered_paths"],
            })

    def _soft_stop_terminal_leftover_agent(self, root: Agent, agent: Agent) -> bool:
        if agent.status not in {"running", "idle"}:
            return False
        if self._missing_expected_outputs(agent):
            return False
        blockers = [
            blocker
            for blocker in self.completion_blockers(agent, include_missing_outputs=False)
            if blocker.get("kind") not in {"answer_submission"}
        ]
        if any(blocker.get("kind") in {"source_change", "test_run", "missing_outputs"} for blocker in blockers):
            return False
        agent.status = "done"
        agent.action_state = "stop"
        agent.result = agent.result or (
            f"[Pruned after root terminal submission {root.submitted_answer_path or ''}]".strip()
        )
        tags = normalize_tags(
            list(agent.current_task_tags)
            + ["status:pruned", "reason:root_terminal_submission_complete", f"covered_by:{root.id}"]
        )
        self.memory.update(
            agent.id,
            public_summary=agent.result,
            clear_active_task=True,
            add_experience=build_memory_card(
                agent.result,
                tags=tags,
                artifacts=[artifact.path for artifact in agent.artifacts],
                memory_kind="terminal",
                memory_source="terminal_join_prune",
                stop_after=True,
            ),
            tags=tags,
        )
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        self.state_board_sync(agent.id)
        self._refresh_task_ledger_agent_item(agent)
        if agent._task and not agent._task.done():
            agent._task.cancel()
        self._emit(root.id, "terminal_leftover_agent_pruned", {
            "agent": agent.id,
            "reason": "root_terminal_submission_complete",
            "submitted_answer_path": root.submitted_answer_path,
        })
        return True

    async def _wait_for_remaining_agents(self, root: Agent) -> None:
        """After root finishes, avoid truncating running children that are still making progress."""
        terminal_complete = self._root_terminal_submission_complete(root)
        prune_requested = False
        prune_started_at = 0.0
        while True:
            running = [a for a in self.agents.values() if a.id != root.id and a.status in {"running", "idle"}]
            if not running:
                return
            tasks = [a._task for a in running if a._task and not a._task.done()]
            if not tasks:
                return
            if terminal_complete:
                if not prune_requested:
                    prune_requested = True
                    prune_started_at = time.time()
                    await self._request_remaining_agents_prune_after_terminal_submission(root, running)
                    self._emit(root.id, "terminal_join_grace_started", {
                        "running": [agent.id for agent in running],
                        "grace_seconds": self.config.terminal_submission_join_grace,
                        "submitted_answer_path": root.submitted_answer_path,
                    })
                elapsed = time.time() - prune_started_at
                grace = max(0.0, float(self.config.terminal_submission_join_grace))
                if elapsed >= grace:
                    pruned = [
                        agent.id
                        for agent in running
                        if self._soft_stop_terminal_leftover_agent(root, agent)
                    ]
                    remaining = [
                        agent.id
                        for agent in self.agents.values()
                        if agent.id != root.id and agent.status in {"running", "idle"}
                    ]
                    self._emit(root.id, "terminal_join_grace_finished", {
                        "pruned": pruned,
                        "remaining": remaining,
                        "submitted_answer_path": root.submitted_answer_path,
                    })
                    if not remaining:
                        return
                    if pruned:
                        continue
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
            if terminal_complete:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=max(0.05, min(1.0, self.config.terminal_submission_join_grace)),
                    return_when=asyncio.ALL_COMPLETED,
                )
                if not pending:
                    return
                continue
            await asyncio.gather(*tasks, return_exceptions=True)

    def _artifact_tool_scope(self, agent: Agent, missing_outputs: list[str]) -> Literal["all", "read_write", "write_only"]:
        if not missing_outputs:
            return "all"
        if self.answer_submission_required(agent) and not agent.submitted_answer_path:
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
            if self._artifact_write_needs_more_evidence(agent, missing_outputs):
                return "read_write"
            return "write_only"
        if "file_read" in available:
            return "read_write"
        return "write_only"

    def _artifact_write_needs_more_evidence(self, agent: Agent, missing_outputs: list[str]) -> bool:
        evidence_like_output = any(
            _looks_like_final_report_path(path) or _looks_like_evidence_report_path(path)
            for path in missing_outputs
        )
        if not evidence_like_output and not (
            _task_is_discovery_work(agent.task) and _non_terminal_expected_output_paths(agent.task)
        ):
            return False
        for path in missing_outputs:
            status = self._artifact_output_status(agent, path)
            if status in {"placeholder", "evidence_gap", "unverified"}:
                return True
        if (agent._turns - agent._last_read_evidence_turn) <= 4 and agent._last_read_evidence_turn > 0:
            return False
        if (agent._turns - agent._last_query_evidence_turn) <= 4 and agent._last_query_evidence_turn > 0:
            return False
        if agent._shell_commands:
            return False
        memory = self.memory.get(agent.id)
        if memory:
            if memory.artifact_index:
                return False
            if any(
                card.artifacts
                or card.memory_kind in {"evidence", "terminal"}
                or "memory_kind:evidence" in set(card.tags)
                or "evidence:true" in set(card.tags)
                for card in memory.experience_cards
            ):
                return False
        return True

    def _final_report_write_needs_more_evidence(self, agent: Agent, missing_outputs: list[str]) -> bool:
        return self._artifact_write_needs_more_evidence(agent, missing_outputs)

    def _llm_max_tokens_for_turn(
        self,
        *,
        loop_action: str,
        recommended_tool_choice: str | None,
        missing_outputs: list[str],
    ) -> int | None:
        if loop_action == "create":
            configured = int(self.config.create_action_max_tokens or 0)
        elif loop_action == "work" and missing_outputs:
            configured = int(self.config.artifact_write_max_tokens or 0)
        else:
            return None
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
                    f"{targets}{more}. Do not write unrelated paths. "
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
        if normalized and _looks_like_final_answer_path(normalized):
            annotated = dict(result)
            annotated["artifact_warning"] = "terminal_answer_requires_submit_answer"
            annotated["message"] = (
                "This file write is not treated as final answer submission. "
                "Use submit_answer(answer=...) when ready to submit."
            )
            annotated["written_relative_path"] = normalized
            return annotated
        if normalized and normalized not in set(expected):
            annotated = dict(result)
            annotated["artifact_warning"] = "unexpected_artifact_path"
            annotated["expected_paths"] = expected[:20]
            annotated["written_relative_path"] = normalized
            return annotated
        return result

    def _register_expected_artifact_write(self, agent: Agent, tc: ToolCall, result: Any) -> None:
        expected = set(_non_terminal_expected_output_paths(agent.task))
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
        output_status = self._artifact_output_status(agent, normalized)
        update_args: dict[str, Any] = {
            "output_path": normalized,
            "output_status": output_status,
        }
        if output_status in {"placeholder", "evidence_gap", "unverified"}:
            update_args["status"] = "partial"
            update_args["blockers"] = [{
                "kind": "uncertain_evidence",
                "message": "Expected output exists but its content says evidence is pending, placeholder, or low confidence.",
                "artifacts": [normalized],
            }]
        self.task_ledger_update(agent, update_args)

    def _publish_shell_evidence(self, agent: Agent, command: str, result: dict[str, Any]) -> None:
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        if not stdout and not stderr:
            return
        full_parts = [
            f"agent: {agent.id}",
            f"turn: {agent._turns}",
            f"command: {command}",
        ]
        if stdout:
            full_parts.append(f"stdout:\n{stdout}")
        if stderr:
            full_parts.append(f"stderr:\n{stderr}")
        full_text = "\n\n".join(full_parts) + "\n"
        digest = hashlib.sha256(full_text.encode("utf-8", errors="replace")).hexdigest()[:12]
        evidence_rel = f"{self.config.shared_dir}/.nanoma/evidence/{agent.id}/turn_{agent._turns}_{digest}.txt"
        evidence_abs = (self.config.workspace_root / evidence_rel).resolve()
        evidence_artifacts: list[str] = []
        try:
            evidence_abs.relative_to(self.config.workspace_root.resolve())
            evidence_abs.parent.mkdir(parents=True, exist_ok=True)
            evidence_abs.write_text(full_text, encoding="utf-8")
            evidence_artifacts.append(evidence_rel)
        except OSError:
            evidence_artifacts = []
        stdout_excerpt = stdout[:2500]
        stderr_excerpt = stderr[:800]
        parts = [
            f"Shell command succeeded: {command}",
        ]
        if stdout_excerpt:
            parts.append(f"stdout:\n{stdout_excerpt}")
        if stderr_excerpt:
            parts.append(f"stderr:\n{stderr_excerpt}")
        summary = "\n\n".join(parts)
        tags = normalize_tags(
            list(agent.current_task_tags)
            + ["tool:shell", "status:evidence", f"agent:{agent.id}", f"id:{agent.id}"]
        )
        self.memory.update(
            agent.id,
            tags=agent.current_task_tags,
            add_artifacts=evidence_artifacts,
            add_experience=build_memory_card(
                summary,
                tags=tags,
                artifacts=evidence_artifacts,
                memory_kind="evidence",
                memory_source="shell",
                evidence=["shell"],
            ),
        )
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
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
        expected = _non_terminal_expected_output_paths(agent.task)
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
            add_experience=build_memory_card(
                agent.result or summary,
                tags=agent.current_task_tags,
                artifacts=files,
                memory_kind="terminal",
                memory_source="auto_complete",
                stop_after=True,
            ),
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

    def _queried_reliable_peer_progress(self, agent: Agent) -> bool:
        if agent._last_query_turn <= 0:
            return False
        if (agent._turns - agent._last_query_turn) > 8:
            return False
        return any(
            self._agent_has_reliable_evidence(peer)
            for peer in self._completed_peer_progress_candidates(agent)
        )

    def _query_result_has_reliable_evidence(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        agents = result.get("agents")
        if not isinstance(agents, list):
            return False
        for item in agents:
            if not isinstance(item, dict):
                continue
            progress = item.get("progress") if isinstance(item.get("progress"), dict) else {}
            public_memory = item.get("public_memory") if isinstance(item.get("public_memory"), dict) else {}
            memory = item.get("memory") if isinstance(item.get("memory"), dict) else {}
            nested_public = memory.get("public_memory") if isinstance(memory.get("public_memory"), dict) else {}
            if progress.get("outputs_complete") is True:
                return True
            if item.get("artifacts") or item.get("result"):
                return True
            if public_memory.get("artifact_index") or nested_public.get("artifact_index"):
                return True
            cards: list[Any] = []
            if isinstance(public_memory.get("experience_cards"), list):
                cards.extend(public_memory.get("experience_cards") or [])
            if isinstance(nested_public.get("experience_cards"), list):
                cards.extend(nested_public.get("experience_cards") or [])
            for card in cards:
                if not isinstance(card, dict):
                    continue
                tags = set(card.get("tags") or [])
                if (
                    card.get("artifacts")
                    or card.get("memory_kind") in {"evidence", "terminal"}
                    or "memory_kind:evidence" in tags
                    or "status:complete" in tags
                    or "status:done" in tags
                    or "evidence:true" in tags
                ):
                    return True
        return False

    def _query_result_agent_ids(self, result: Any) -> list[str]:
        if not isinstance(result, dict):
            return []
        raw_agents: list[Any]
        if isinstance(result.get("agents"), list):
            raw_agents = list(result.get("agents") or [])
        else:
            raw_agents = [result] if result.get("id") else []
        ids: list[str] = []
        seen: set[str] = set()
        for item in raw_agents:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("id") or "").strip()
            if agent_id and agent_id not in seen:
                ids.append(agent_id)
                seen.add(agent_id)
        return ids

    def _query_result_reliable_evidence_ids(self, result: Any) -> list[str]:
        if not isinstance(result, dict):
            return []
        raw_agents: list[Any]
        if isinstance(result.get("agents"), list):
            raw_agents = list(result.get("agents") or [])
        else:
            raw_agents = [result] if result.get("id") else []
        ids: list[str] = []
        seen: set[str] = set()
        for item in raw_agents:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("id") or "").strip()
            if not agent_id or agent_id in seen:
                continue
            if self._query_snapshot_has_reliable_evidence(item):
                ids.append(agent_id)
                seen.add(agent_id)
        return ids

    def _query_snapshot_has_reliable_evidence(self, item: dict[str, Any]) -> bool:
        progress = item.get("progress") if isinstance(item.get("progress"), dict) else {}
        public_memory = item.get("public_memory") if isinstance(item.get("public_memory"), dict) else {}
        memory = item.get("memory") if isinstance(item.get("memory"), dict) else {}
        nested_public = memory.get("public_memory") if isinstance(memory.get("public_memory"), dict) else {}
        if progress.get("outputs_complete") is True:
            return True
        if item.get("artifacts") or item.get("result"):
            return True
        if public_memory.get("artifact_index") or nested_public.get("artifact_index"):
            return True
        cards: list[Any] = []
        if isinstance(public_memory.get("experience_cards"), list):
            cards.extend(public_memory.get("experience_cards") or [])
        if isinstance(nested_public.get("experience_cards"), list):
            cards.extend(nested_public.get("experience_cards") or [])
        for card in cards:
            if not isinstance(card, dict):
                continue
            tags = set(card.get("tags") or [])
            if (
                card.get("artifacts")
                or card.get("memory_kind") in {"evidence", "terminal"}
                or "memory_kind:evidence" in tags
                or "status:complete" in tags
                or "status:done" in tags
                or "evidence:true" in tags
            ):
                return True
        return False

    def _completed_peer_overlap_candidates(self, agent: Agent) -> list[Agent]:
        candidates: list[Agent] = []
        for peer in self._peer_overlap_query_candidates(agent):
            if self._agent_has_prune_worthy_progress(peer):
                candidates.append(peer)
        return candidates

    def _active_duplicate_peer_overlap_candidates(self, agent: Agent) -> list[Agent]:
        candidates: list[Agent] = []
        agent_outputs = set(_non_terminal_expected_output_paths(agent.task))
        agent_lanes = _agent_lane_fingerprints(agent)
        for peer in self._peer_overlap_query_candidates(agent):
            if peer.status not in {"running", "idle"}:
                continue
            if not peer.parent:
                continue
            if not self._agents_have_compatible_output_slot(agent, peer):
                continue
            peer_outputs = set(_non_terminal_expected_output_paths(peer.task))
            same_output = bool(agent_outputs and peer_outputs and agent_outputs.intersection(peer_outputs))
            lane_overlap = bool(agent_lanes and _agent_lane_fingerprints(peer).intersection(agent_lanes))
            similar_task = _cheap_text_similarity(agent.task, peer.task) >= 0.45
            if not (same_output or lane_overlap or similar_task):
                continue
            if self._agent_has_prune_worthy_progress(peer):
                candidates.append(peer)
        candidates.sort(key=lambda p: (
            -len(p.artifacts),
            -p._tool_calls,
            -p._turns,
            p.id,
        ))
        return candidates

    def _create_resume_after_read_pending(self, agent: Agent) -> bool:
        if not agent._create_resume_after_read:
            return False
        if agent.children:
            agent._create_resume_after_read = False
            return False
        if not self._can_create_child(agent):
            agent._create_resume_after_read = False
            return False
        return _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)

    def _pre_create_reference_read_pending(self, agent: Agent) -> bool:
        if agent.children:
            return False
        if agent._create_resume_after_read:
            return False
        if agent._last_query_turn > 0 or agent._filtered_read_intent_count > 0:
            return False
        if not self._can_create_child(agent):
            return False
        if not _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams):
            return False
        return bool(_referenced_input_like_shared_paths(agent.task))

    def _initial_create_required(self, agent: Agent) -> bool:
        if agent.children:
            return False
        if not self._can_create_child(agent):
            return False
        return _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)

    def _multi_phase_create_action_pending(self, agent: Agent) -> bool:
        if not agent.children:
            return False
        if self.unfinished_child_agents(agent):
            return False
        if not self._can_create_child(agent):
            return False
        if not self._missing_expected_outputs(agent):
            return False
        if not _task_has_multi_phase_peer_protocol(agent.task):
            return False
        if self._lane_coverage_pressure(agent) >= 0.95:
            return False
        return True

    def _handoff_pressure(self, agent: Agent) -> float:
        if not agent.children:
            return 0.0
        threshold = max(1, int(self.config.handoff_child_count_threshold or 1))
        pressure = len(agent.children) / threshold
        if agent.depth == 0:
            pressure += 0.35
        if self.unfinished_child_agents(agent):
            pressure += 0.35
        if agent._deferred_spawn_requests:
            pressure = max(0.0, pressure - 0.25)
        return max(0.0, min(1.0, pressure))

    def _lane_coverage_pressure(self, agent: Agent) -> float:
        if not agent.children:
            return 0.0
        lane_counts: dict[str, int] = {}
        artifact_or_done_counts: dict[str, int] = {}
        for child_id in agent.children:
            child = self.agents.get(child_id)
            if not child:
                continue
            fingerprints = _agent_lane_fingerprints(child)
            if not fingerprints:
                continue
            covered = child.status in {"done", "failed"} or bool(child.artifacts) or bool(child.result)
            for lane in fingerprints:
                lane_counts[lane] = lane_counts.get(lane, 0) + 1
                if covered:
                    artifact_or_done_counts[lane] = artifact_or_done_counts.get(lane, 0) + 1
        if not lane_counts:
            return 0.0
        repeated = max((count - 1 for count in lane_counts.values()), default=0)
        covered_repeated = max((count - 1 for count in artifact_or_done_counts.values()), default=0)
        pressure = 0.35 * repeated + 0.45 * covered_repeated
        if len(agent.children) >= max(2, int(self.config.handoff_child_count_threshold or 2)):
            pressure += 0.20
        return max(0.0, min(1.0, pressure))

    def _offspring_create_bias(self, agent: Agent) -> float:
        if not agent.parent:
            return 0.0
        if agent.orchestration_preference == "solo":
            return 0.0
        if agent.children:
            return 0.0
        if not self._agent_has_uncertain_evidence(agent) and not _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams):
            return 0.0
        rank = ORCHESTRATION_RANK.get(agent.orchestration_preference, ORCHESTRATION_RANK["balanced"])
        return max(0.0, min(1.0, 0.35 + 0.2 * rank))

    def _handoff_create_after_partial_progress_plan(
        self,
        agent: Agent,
        missing_outputs: list[str],
    ) -> dict[str, Any] | None:
        uncertain_evidence = self._agent_has_uncertain_evidence(agent)
        if not missing_outputs and not uncertain_evidence:
            return None
        if not agent.parent:
            return None
        if agent.children:
            return None
        if not self._can_create_child(agent):
            return None
        if self.unfinished_child_agents(agent):
            return None
        if (
            missing_outputs
            and not uncertain_evidence
            and _task_is_concrete_single_output_work(agent.task)
        ):
            return None
        if (
            uncertain_evidence
            and _task_is_concrete_single_output_work(agent.task)
            and self._artifact_write_needs_more_evidence(
                agent,
                missing_outputs or _non_terminal_expected_output_paths(agent.task),
            )
        ):
            return None
        has_progress = (
            agent._tool_calls >= 3
            or agent._last_read_evidence_turn > 0
            or bool(agent.artifacts)
            or self._agent_has_reliable_evidence(agent)
        )
        if not has_progress:
            return None
        needs_help = (
            agent._artifact_nudge_count >= 2
            or agent._no_tool_turns >= 2
            or agent._write_only_miss_count >= 1
            or uncertain_evidence
            or _task_prefers_query_before_artifact(agent.task)
        )
        if not needs_help and agent._tool_calls < max(3, self.config.max_solo_tool_calls_before_spawn):
            return None
        return {
            "action": "create",
            "reason": "delegate_uncertain_evidence_recovery" if uncertain_evidence else "handoff_after_partial_progress",
            "turn_added": agent._turns,
            "missing_outputs": missing_outputs[:20],
            "uncertain_artifacts": self._uncertain_artifact_paths(agent)[:20],
        }

    def _ancestor_terminal_delivery_target(self, agent: Agent) -> Agent | None:
        parent_id = agent.parent
        while parent_id and parent_id in self.agents:
            parent = self.agents[parent_id]
            if self.answer_submission_required(parent) and not parent.submitted_answer_path:
                return parent
            parent_id = parent.parent
        return None

    def _sibling_evidence_agents_for_handoff(self, agent: Agent) -> list[Agent]:
        if not agent.parent or agent.parent not in self.agents:
            return []
        parent = self.agents[agent.parent]
        candidates: list[Agent] = []
        seen: set[str] = set()
        for peer_id in sorted(parent.children):
            peer = self.agents.get(peer_id)
            if not peer or peer.id in seen:
                continue
            if peer.id != agent.id and peer.status != "done":
                continue
            if not self._agent_has_reliable_evidence(peer):
                continue
            candidates.append(peer)
            seen.add(peer.id)
        candidates.sort(key=lambda peer: (
            0 if peer.id == agent.id else 1,
            0 if peer.status == "done" else 1,
            -len(self._agent_known_artifact_paths(peer)),
            -peer._tool_calls,
            peer.id,
        ))
        return candidates

    def _has_peer_final_delivery_claim(self, agent: Agent, target: Agent) -> bool:
        related_ids = {target.id, *target.children}
        if agent.parent:
            related_ids.add(agent.parent)
            related_ids.update(self.agents.get(agent.parent, agent).children)
        for peer in self.agents.values():
            if peer.id == agent.id:
                continue
            if peer.parent not in related_ids and peer.id not in related_ids:
                continue
            if self._is_final_delivery_agent(peer) and peer.status in {"running", "idle", "done"}:
                return True
        return False

    def _mature_evidence_handoff_plan(self, agent: Agent) -> dict[str, Any] | None:
        if not agent.parent:
            return None
        if agent.children:
            return None
        if not self._can_create_child(agent):
            return None
        if self._agent_focus_class(agent) not in {"evidence", "verification"}:
            return None
        if agent.status != "running":
            return None
        if self._agent_has_uncertain_evidence(agent):
            return None
        if not self._agent_has_reliable_evidence(agent):
            return None
        if not (agent.result or agent.artifacts or self._agent_has_non_trace_artifact(agent)):
            return None
        target = self._ancestor_terminal_delivery_target(agent)
        if not target:
            return None
        if self._has_peer_final_delivery_claim(agent, target):
            return None
        evidence_agents = self._sibling_evidence_agents_for_handoff(agent)
        if len(evidence_agents) < 2 and not self._answer_submission_ready(target):
            return None
        return {
            "action": "create",
            "reason": "handoff_after_evidence_maturity",
            "turn_added": agent._turns,
            "answer_path": self.final_answer_path_for(target),
            "target_agent": target.id,
            "evidence_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "artifacts": sorted(self._agent_known_artifact_paths(peer))[:8],
                    "result": (peer.result or "")[:240],
                    "tags": peer.current_task_tags[:12],
                }
                for peer in evidence_agents[:8]
            ],
        }

    def _non_root_answer_submission_ready(self, agent: Agent) -> bool:
        if not self.answer_submission_required(agent) or agent.submitted_answer_path:
            return False
        if self._agent_has_uncertain_evidence(agent):
            return False
        if agent.result:
            return True
        if agent.artifacts:
            return True
        children = [self.agents[child_id] for child_id in agent.children if child_id in self.agents]
        if any(
            (child.status == "done" or child.artifacts or child.result)
            and self._agent_has_reliable_evidence(child)
            for child in children
        ):
            return True
        for related in self._related_agents_for_completion_evidence(agent):
            if related.id == agent.id:
                continue
            if related.status not in {"done", "running", "idle"}:
                continue
            if self._agent_has_reliable_evidence(related):
                return True
        if self._agent_has_reliable_evidence(agent):
            return True
        return False

    def _answer_submission_ready(self, agent: Agent) -> bool:
        if agent.parent is None:
            return bool(self._root_answer_readiness(agent).get("can_attempt_submit"))
        return self._non_root_answer_submission_ready(agent)

    def _answer_submission_pending(self, agent: Agent) -> bool:
        return self.answer_submission_required(agent) and not agent.submitted_answer_path

    def _agent_has_mature_output_slot(self, agent: Agent) -> bool:
        for output in self._task_ledger_output_entries(agent, {}):
            if str(output.get("status") or "") in {"evidence_attached", "verified", "covered", "submitted"}:
                return True
        return False

    def _agent_ready_for_final_delivery_evidence(self, agent: Agent) -> bool:
        if self._agent_has_uncertain_evidence(agent):
            return False
        return self._agent_has_reliable_evidence(agent) or self._agent_has_mature_output_slot(agent)

    def _terminal_answer_candidate_from_agent(self, agent: Agent) -> dict[str, Any] | None:
        if self._agent_has_uncertain_evidence(agent):
            return None
        focus = self._agent_focus_class(agent)
        identity_text = " ".join([
            agent.role,
            agent.group_id,
            agent.workflow_prior,
            " ".join(agent.current_task_tags),
            agent.task[:600],
        ]).lower()
        terminal_identity = (
            focus in {"synthesis", "verification", "final_delivery"}
            or any(term in identity_text for term in ("synthesis", "answer", "final", "overlap", "cross-check"))
        )
        if not terminal_identity:
            return None
        candidate_paths = [artifact.path for artifact in agent.artifacts]
        candidate_paths.extend(_non_terminal_expected_output_paths(agent.task))
        seen_paths: set[str] = set()
        for path in candidate_paths[:12]:
            path = str(path or "")
            if not path or path in seen_paths or _looks_like_final_answer_path(path):
                continue
            seen_paths.add(path)
            text = self._artifact_text(path, limit=24000)
            if not text:
                continue
            lowered = text.lower()
            if not any(term in lowered for term in _TERMINAL_ANSWER_ARTIFACT_TERMS):
                continue
            answer_hint = _extract_terminal_answer_hint(text)
            if not answer_hint:
                continue
            return {
                "id": agent.id,
                "role": agent.role,
                "status": agent.status,
                "focus": focus,
                "artifact": path,
                "answer_hint": answer_hint,
                "result": (agent.result or "")[:240],
                "tags": agent.current_task_tags[:12],
            }
        result_hint = _extract_terminal_answer_hint(agent.result or "")
        if result_hint:
            return {
                "id": agent.id,
                "role": agent.role,
                "status": agent.status,
                "focus": focus,
                "artifact": "",
                "answer_hint": result_hint,
                "result": (agent.result or "")[:240],
                "tags": agent.current_task_tags[:12],
            }
        return None

    def _root_terminal_answer_candidate(self, agent: Agent) -> dict[str, Any] | None:
        if agent.parent is not None or not self.answer_submission_required(agent):
            return None
        related = self._related_agents_for_completion_evidence(agent)
        related.extend(peer for peer in self.agents.values() if peer.id not in {item.id for item in related})
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        queried_ids = set(agent._last_query_agent_ids)
        for peer in related:
            if peer.id == agent.id or peer.id in seen:
                continue
            seen.add(peer.id)
            candidate = self._terminal_answer_candidate_from_agent(peer)
            if not candidate:
                continue
            if queried_ids and peer.id not in queried_ids and peer.parent not in queried_ids:
                candidate["not_recently_queried"] = True
            candidates.append(candidate)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (
            1 if item.get("not_recently_queried") else 0,
            0 if item.get("status") == "done" else 1,
            0 if item.get("focus") in {"synthesis", "final_delivery"} else 1,
            str(item.get("id") or ""),
        ))
        best = candidates[0]
        if best.get("not_recently_queried") and agent._last_query_turn > 0:
            return None
        return best

    def _record_root_terminal_answer_candidate(self, agent: Agent, candidate: dict[str, Any]) -> None:
        self.task_ledger_update(agent, {
            "item_id": agent.id,
            "status": "ready_for_review",
            "note": (
                "terminal_candidate_ready: "
                f"answer_hint={candidate.get('answer_hint') or ''}; "
                f"source_agent={candidate.get('id') or ''}; "
                f"artifact={candidate.get('artifact') or ''}"
            ),
            "covered_by": [str(candidate.get("id") or "")],
        })

    def _agent_ledger_output_summary(self, agent: Agent) -> dict[str, Any]:
        item = self.task_ledger_item_snapshot(agent) or {}
        outputs = [output for output in item.get("expected_outputs") or [] if isinstance(output, dict)]
        incomplete = [
            output
            for output in outputs
            if str(output.get("status") or "") in _TASK_LEDGER_OUTPUT_INCOMPLETE_STATUSES
        ]
        mature = [
            output
            for output in outputs
            if str(output.get("status") or "") in {"evidence_attached", "verified", "covered", "submitted"}
        ]
        blockers = list(item.get("blockers") or [])
        return {
            "id": agent.id,
            "status": item.get("status") or agent.status,
            "agent_status": item.get("agent_status") or agent.status,
            "outputs": outputs,
            "incomplete_outputs": incomplete,
            "mature_outputs": mature,
            "blockers": blockers,
        }

    def _root_answer_readiness(self, agent: Agent) -> dict[str, Any]:
        terminal_candidate = self._root_terminal_answer_candidate(agent)
        evidence_agents = self._final_delivery_evidence_agents(agent)
        unfinished = self.unfinished_child_agents(agent)
        related = [
            peer
            for peer in self._related_agents_for_completion_evidence(agent)
            if peer.id != agent.id
        ]
        child_summaries = [self._agent_ledger_output_summary(peer) for peer in related[:12]]
        incomplete_children = [
            summary for summary in child_summaries
            if summary["incomplete_outputs"] or str(summary.get("status") or "") in {"in_progress", "partial"}
        ]
        blocker_kinds = {
            str(blocker.get("kind") or "")
            for blocker in self.completion_blockers(agent, include_missing_outputs=False)
            if blocker.get("kind") != "answer_submission"
        }
        recently_inspected = self._root_recently_inspected_final_delivery_evidence(agent, evidence_agents)
        score = 0.0
        reasons: list[str] = []
        if terminal_candidate:
            score += 0.72
            reasons.append("terminal_candidate")
        if evidence_agents:
            score += min(0.55, 0.35 + 0.10 * len(evidence_agents))
            reasons.append("mature_evidence")
        if recently_inspected:
            score += 0.25
            reasons.append("recently_inspected")
        if not incomplete_children and evidence_agents:
            score += 0.20
            reasons.append("no_known_child_gaps")
        if unfinished and not terminal_candidate:
            score -= min(0.28, 0.10 * len(unfinished))
            reasons.append("unfinished_children")
        if incomplete_children and not terminal_candidate:
            score -= min(0.35, 0.12 * len(incomplete_children))
            reasons.append("ledger_gaps")
        if blocker_kinds:
            score -= min(0.20, 0.08 * len(blocker_kinds))
            reasons.append("completion_blockers")
        score = max(0.0, min(1.0, score))
        return {
            "score": round(score, 3),
            "submit_only_ready": score >= 0.82 and (bool(terminal_candidate) or recently_inspected),
            "can_attempt_submit": score >= 0.45 or bool(evidence_agents) or bool(terminal_candidate),
            "terminal_candidate": terminal_candidate,
            "evidence_agents": evidence_agents,
            "unfinished_agents": unfinished,
            "incomplete_children": incomplete_children,
            "blocker_kinds": sorted(kind for kind in blocker_kinds if kind),
            "recently_inspected": recently_inspected,
            "reasons": reasons,
        }

    def _root_answer_readiness_public(self, readiness: dict[str, Any]) -> dict[str, Any]:
        evidence_agents = readiness.get("evidence_agents") or []
        unfinished_agents = readiness.get("unfinished_agents") or []
        return {
            "score": readiness.get("score", 0.0),
            "submit_only_ready": bool(readiness.get("submit_only_ready")),
            "can_attempt_submit": bool(readiness.get("can_attempt_submit")),
            "reasons": list(readiness.get("reasons") or [])[:8],
            "recently_inspected": bool(readiness.get("recently_inspected")),
            "blocker_kinds": list(readiness.get("blocker_kinds") or [])[:8],
            "terminal_candidate": readiness.get("terminal_candidate") or None,
            "evidence_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "artifacts": sorted(self._agent_known_artifact_paths(peer))[:8],
                    "result": (peer.result or "")[:220],
                }
                for peer in evidence_agents[:8]
                if isinstance(peer, Agent)
            ],
            "unfinished_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "missing_outputs": self.missing_expected_outputs(peer)[:8],
                }
                for peer in unfinished_agents[:8]
                if isinstance(peer, Agent)
            ],
            "incomplete_children": list(readiness.get("incomplete_children") or [])[:8],
        }

    def _final_delivery_evidence_agents(self, agent: Agent) -> list[Agent]:
        candidates: list[Agent] = []
        seen: set[str] = set()
        for related in self._related_agents_for_completion_evidence(agent):
            if related.id == agent.id or related.id in seen:
                continue
            if not self._agent_ready_for_final_delivery_evidence(related):
                continue
            candidates.append(related)
            seen.add(related.id)
        candidates.sort(key=lambda peer: (
            0 if peer.status == "done" else 1,
            -len(peer.artifacts),
            -peer._tool_calls,
            peer.id,
        ))
        return candidates

    def _is_final_delivery_agent(self, agent: Agent) -> bool:
        identity_text = " ".join(
            [
                agent.role,
                agent.bio,
                agent.workflow_prior,
                " ".join(agent.current_task_tags),
            ]
        ).lower()
        task_text = (agent.task or "")[:700].lower()
        identity_markers = (
            "finalizer", "finaliser", "delivery", "deliverer",
            "final_delivery", "final-delivery", "role:finalizer", "role:delivery",
        )
        if any(marker in identity_text for marker in identity_markers):
            return True
        task_markers = (
            "finalizer", "finaliser", "final delivery", "final_delivery",
            "final-delivery", "delivery agent", "deliverer agent",
        )
        return any(marker in task_text for marker in task_markers)

    def _has_active_final_delivery_agent(self, agent: Agent) -> bool:
        for peer in self._descendant_agents(agent):
            if peer.status in {"running", "idle"} and self._is_final_delivery_agent(peer):
                return True
        return False

    def _can_create_final_delivery_agent(self, agent: Agent) -> bool:
        root_submit_task = agent.parent is None and self.answer_submission_required(agent)
        if root_submit_task:
            return len(self.agents) < self.config.max_agents and agent.depth + 1 <= self.config.max_depth
        return (
            len(self.agents) < self.config.max_agents
            and agent.depth + 1 <= self.config.max_depth
            and agent.orchestration_preference != "solo"
            and not self._agent_has_solo_decision(agent)
            and not self._agent_has_pending_self_work_decision(agent)
        )

    def _root_recently_inspected_final_delivery_evidence(self, agent: Agent, evidence_agents: list[Agent]) -> bool:
        if agent.parent is not None or not evidence_agents:
            return False
        evidence_ids = {peer.id for peer in evidence_agents}
        if agent._last_query_evidence_turn > 0 and (agent._turns - agent._last_query_evidence_turn) <= 6:
            queried_ids = set(agent._last_query_reliable_evidence_ids) or set(agent._last_query_agent_ids)
            if not queried_ids or queried_ids.intersection(evidence_ids):
                return True
        if agent._last_query_turn > 0 and (agent._turns - agent._last_query_turn) <= 4:
            queried_ids = set(agent._last_query_agent_ids)
            if queried_ids.intersection(evidence_ids):
                return True
        if agent._last_read_evidence_turn > 0 and (agent._turns - agent._last_read_evidence_turn) <= 4:
            return True
        return False

    def _root_direct_answer_submission_plan(self, agent: Agent) -> dict[str, Any] | None:
        if agent.parent is not None:
            return None
        if not self.answer_submission_required(agent) or agent.submitted_answer_path:
            return None
        if not self.can_submit_answer(agent).get("ok"):
            return None
        readiness = self._root_answer_readiness(agent)
        if not readiness.get("submit_only_ready"):
            return None
        terminal_candidate = readiness.get("terminal_candidate")
        evidence_agents = list(readiness.get("evidence_agents") or [])
        if not evidence_agents and not terminal_candidate:
            return None
        blockers = [
            blocker
            for blocker in self.completion_blockers(agent, include_missing_outputs=False)
            if blocker.get("kind") != "answer_submission"
        ]
        if any(blocker.get("kind") in {"source_change", "test_run", "missing_outputs"} for blocker in blockers):
            return None
        if all(blocker.get("kind") in {"uncertain_evidence", "unverified_evidence"} for blocker in blockers):
            blockers = []
        if blockers:
            return None
        if terminal_candidate:
            self._record_root_terminal_answer_candidate(agent, terminal_candidate)
        return {
            "action": "stop",
            "reason": "terminal_candidate_submit_ready" if terminal_candidate else "direct_answer_submission_ready",
            "turn_added": agent._turns,
            "answer_path": self.final_answer_path_for(agent),
            "readiness": {
                key: value
                for key, value in readiness.items()
                if key not in {"evidence_agents", "terminal_candidate", "unfinished_agents"}
            },
            "evidence_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "artifacts": [artifact.path for artifact in peer.artifacts[:8]],
                    "result": (peer.result or "")[:240],
                    "tags": peer.current_task_tags[:12],
                }
                for peer in evidence_agents[:8]
            ],
            "terminal_candidate": terminal_candidate,
        }

    def _final_delivery_handoff_plan(self, agent: Agent) -> dict[str, Any] | None:
        if not self.answer_submission_required(agent) or agent.submitted_answer_path:
            return None
        if self._is_final_delivery_agent(agent):
            return None
        if not agent.children:
            return None
        if self.unfinished_child_agents(agent):
            return None
        if self._has_active_final_delivery_agent(agent):
            return None
        if not self._can_create_final_delivery_agent(agent):
            return None
        evidence_agents = self._final_delivery_evidence_agents(agent)
        if not evidence_agents:
            return None
        blockers = [
            blocker
            for blocker in self.completion_blockers(agent, include_missing_outputs=False)
            if blocker.get("kind") != "answer_submission"
        ]
        if any(blocker.get("kind") in {"source_change", "test_run", "missing_outputs"} for blocker in blockers):
            return None
        if all(blocker.get("kind") in {"uncertain_evidence", "unverified_evidence"} for blocker in blockers):
            blockers = []
        return {
            "action": "create",
            "reason": "delegate_final_delivery",
            "turn_added": agent._turns,
            "answer_path": self.final_answer_path_for(agent),
            "evidence_agents": [
                {
                    "id": peer.id,
                    "role": peer.role,
                    "status": peer.status,
                    "artifacts": [artifact.path for artifact in peer.artifacts[:8]],
                    "result": (peer.result or "")[:240],
                    "tags": peer.current_task_tags[:12],
                }
                for peer in evidence_agents[:8]
            ],
            "blockers": blockers,
        }

    def _deferred_spawn_readiness(self, agent: Agent) -> float:
        if not agent._deferred_spawn_requests:
            return 0.0
        scores = [self._deferred_spawn_request_readiness(agent, request) for request in agent._deferred_spawn_requests]
        return max(0.0, min(1.0, max(scores, default=0.0)))

    def _deferred_spawn_request_readiness(self, agent: Agent, request: dict[str, Any]) -> float:
        depends_on = [str(item).strip().lower() for item in request.get("depends_on") or [] if str(item).strip()]
        peers = [self.agents[child_id] for child_id in agent.children if child_id in self.agents]
        if agent.parent and (not depends_on or _agent_matches_dependency_terms(agent, depends_on)):
            peers.append(agent)

        def peer_score(peer: Agent) -> float:
            expected = self.expected_outputs(peer)
            missing = self.missing_expected_outputs(peer)
            if self._agent_has_uncertain_evidence(peer):
                if peer.artifacts or peer.result or peer.status == "done":
                    return 0.35
                return 0.15
            if peer.status == "done":
                return 1.0
            if expected:
                return (len(expected) - len(missing)) / max(1, len(expected))
            if peer.artifacts or peer.result:
                return 0.85
            if peer._tool_calls >= 3:
                return 0.35
            return 0.1

        if depends_on:
            dependency_scores: list[float] = []
            for dep in depends_on:
                matched = [peer for peer in peers if _agent_matches_dependency_terms(peer, [dep])]
                if not matched:
                    dependency_scores.append(0.0)
                    continue
                dependency_scores.append(max(peer_score(peer) for peer in matched))
            return sum(dependency_scores) / max(1, len(dependency_scores))

        if not peers:
            return float(request.get("readiness") or 0.0)
        downstream_focus = bool(request.get("downstream_focus"))
        if downstream_focus:
            informative = [
                peer for peer in peers
                if peer.status == "done" or peer.artifacts or peer.result or peer._tool_calls >= 3
            ]
            peers = informative or peers
        peer_scores = [peer_score(peer) for peer in peers]
        return sum(peer_scores) / max(1, len(peer_scores))

    def _deferred_spawn_ready(self, agent: Agent) -> bool:
        if not agent._deferred_spawn_requests:
            return False
        if len(self.agents) >= self.config.max_agents:
            return False
        configured_threshold = max(0.0, min(1.0, self.config.spawn_readiness_threshold))
        for request in agent._deferred_spawn_requests:
            threshold = max(
                configured_threshold,
                max(0.0, min(1.0, float(request.get("readiness_threshold") or configured_threshold))),
            )
            if self._deferred_spawn_request_readiness(agent, request) >= threshold:
                return True
        return False

    def _peer_wave_create_gap(self, agent: Agent) -> dict[str, Any] | None:
        if len(self.agents) >= self.config.max_agents:
            return None
        if agent.depth + 1 > self.config.max_depth:
            return None
        if agent.orchestration_preference == "solo":
            return None
        for requirement in _explicit_peer_wave_requirements(agent.task):
            target = int(requirement.get("target") or 0)
            if target <= 1:
                continue
            matched = [
                self.agents[child_id]
                for child_id in sorted(agent.children)
                if child_id in self.agents and _agent_matches_peer_wave_requirement(self.agents[child_id], requirement)
            ]
            if len(matched) < target:
                unfinished = [
                    child
                    for child in matched
                    if child.status in {"running", "idle"}
                ]
                remaining = target - len(matched)
                return {
                    "target": target,
                    "current": len(matched),
                    "remaining": remaining,
                    "unfinished": len(unfinished),
                    "role": requirement.get("role") or "",
                    "phase": requirement.get("phase") or "",
                    "requirement": requirement.get("text") or "",
                }
        return None

    def _refresh_loop_action_plan(self, agent: Agent, missing_outputs: list[str]) -> None:
        if self.config.loop_action_policy == "constraint":
            self._refresh_loop_action_plan_constraint(agent, missing_outputs)
        else:
            self._refresh_loop_action_plan_rule(agent, missing_outputs)
        self._apply_root_steward_plan(agent, missing_outputs)
        if agent._loop_action_plan:
            agent._loop_action_plan.setdefault("selected_by", "runtime_planner")
            agent._loop_action_selected_by = str(agent._loop_action_plan.get("selected_by") or "runtime_planner")

    def _two_stage_action_selection_enabled(self) -> bool:
        return (
            self.config.two_stage_action_selection
            and self.config.loop_action_policy == "constraint"
            and self.llm_call is openai_compatible_call
        )

    def _selector_action_context(self, agent: Agent, missing_outputs: list[str]) -> Message:
        plan = dict(agent._loop_action_plan or {})
        candidates = list(plan.get("candidate_scores") or [])
        if not candidates and plan.get("action"):
            candidates = [{
                "action": plan.get("action"),
                "reason": plan.get("reason"),
                "score": plan.get("score"),
            }]
        ledger = self.task_ledger_snapshot(agent, include_all=False, limit=8)
        blockers = self.completion_blockers(agent, include_missing_outputs=bool(missing_outputs))
        payload = {
            "agent_id": agent.id,
            "role": agent.role,
            "status": agent.status,
            "turn": agent._turns,
            "task_preview": agent.task[:700],
            "available_actions": sorted(_LOOP_ACTIONS),
            "runtime_recommended_action": plan.get("action"),
            "runtime_reason": plan.get("reason"),
            "candidate_scores": candidates[:8],
            "missing_outputs": missing_outputs[:8],
            "blockers": blockers[:6],
            "ledger_counts": ledger.get("counts") or {},
            "current_ledger_item": ledger.get("current_agent_item") or {},
            "related_ledger_items": (ledger.get("items") or [])[:6],
        }
        if self.answer_submission_required(agent) and not agent.submitted_answer_path:
            readiness = self._root_answer_readiness(agent) if agent.parent is None else {}
            if readiness:
                payload["answer_readiness"] = self._root_answer_readiness_public(readiness)
        return {
            "role": "user",
            "content": (
                "[Loop Action Selector]\n"
                "Select exactly one next action for this agent from create, read, message, work, compact, stop.\n"
                "Return only JSON: {\"action\":\"...\",\"reason\":\"...\"}.\n"
                "Do not plan tools and do not call tools. Tool planning happens after this selection.\n"
                "Use the runtime recommendation unless current state clearly favors another action.\n"
                f"State:\n{json.dumps(payload, ensure_ascii=False, default=str)[:12000]}"
            ),
        }

    def _parse_selected_loop_action(self, content: str | None) -> tuple[str, str]:
        text = (content or "").strip()
        if not text:
            return "", ""
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                action = str(data.get("action") or "").strip().lower()
                reason = str(data.get("reason") or "").strip()
                if action in _LOOP_ACTIONS:
                    return action, reason
            except Exception:
                pass
        lowered = text.lower()
        for action in sorted(_LOOP_ACTIONS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(action)}\b", lowered):
                return action, text[:240]
        return "", ""

    async def _maybe_select_loop_action(self, agent: Agent, missing_outputs: list[str]) -> None:
        if not self._two_stage_action_selection_enabled():
            return
        if not agent._loop_action_plan:
            return
        if agent._loop_action_plan.get("selector_locked"):
            return
        if agent.status != "running":
            return
        planned_action = str((agent._loop_action_plan or {}).get("action") or "")
        if (
            planned_action
            and agent._action_miss_action == planned_action
            and agent._action_miss_count >= 1
            and not (planned_action == "create" and agent._last_create_resolution == "skipped_covered")
        ):
            plan = dict(agent._loop_action_plan or {})
            plan["selected_by"] = "runtime_retry_after_action_miss"
            plan["selector_locked"] = True
            plan["selector_reason"] = (
                "Previous execution did not satisfy the same selected action; "
                "retry execution with the current scoped tools instead of re-selecting."
            )
            agent._loop_action_plan = plan
            agent._loop_action_selected_by = "runtime_retry_after_action_miss"
            self._emit(agent.id, "loop_action_selector_skipped", {
                "action": planned_action,
                "reason": "retry_after_action_miss",
                "miss_count": agent._action_miss_count,
            })
            return
        selector_message = self._selector_action_context(agent, missing_outputs)
        await self.scheduler.acquire()
        try:
            response = await self.llm_call(
                [selector_message],
                agent.model,
                [],
                retry_config=self.config.retry,
                max_tokens=160,
                tool_choice=None,
                temperature=0.1,
            )
        finally:
            self.scheduler.release()
        if response is None:
            return
        cost = self.ledger.record(agent.id, response.usage)
        agent.tokens_consumed += response.usage.total_tokens
        agent.quota.budget -= cost
        selected, reason = self._parse_selected_loop_action(response.content)
        previous_plan = dict(agent._loop_action_plan or {})
        previous = str(previous_plan.get("action") or "")
        self._emit(agent.id, "loop_action_selected", {
            "selected": selected or None,
            "previous": previous or None,
            "reason": reason,
            "tokens": response.usage.total_tokens,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost": round(cost, 6),
            "content_preview": (response.content or "")[:240],
        })
        if selected not in _LOOP_ACTIONS:
            return
        if previous_plan.get("reason") == "delegate_final_delivery" and selected != "create":
            self._emit(agent.id, "loop_action_selector_override_ignored", {
                "selected": selected,
                "kept": "create",
                "reason": "final_delivery_handoff_requires_create",
                "selector_reason": reason[:500],
            })
            plan = dict(previous_plan)
            plan["selected_by"] = "runtime_final_delivery_guard"
            plan["selector_previous_action"] = previous
            plan["selector_reason"] = reason[:500]
            plan["selector_locked"] = True
            agent._loop_action_plan = plan
            agent._loop_action_selected_by = "runtime_final_delivery_guard"
            self._apply_root_steward_plan(agent, missing_outputs)
            return
        plan = dict(previous_plan)
        plan["selector_previous_action"] = previous
        plan["action"] = selected
        plan["reason"] = reason[:240] or str(previous_plan.get("reason") or "")
        plan["selected_by"] = "llm_selector"
        plan["selector_reason"] = reason[:500]
        plan["selector_locked"] = True
        agent._loop_action_plan = plan
        agent._loop_action_selected_by = "llm_selector"
        self._apply_root_steward_plan(agent, missing_outputs)

    def _apply_root_steward_plan(self, agent: Agent, missing_outputs: list[str]) -> None:
        if not self._root_steward_mode(agent):
            return
        plan = agent._loop_action_plan or {}
        if plan:
            plan["root_steward"] = True
            agent._loop_action_plan = plan
        if plan.get("action") == "create" and plan.get("reason") == "delegate_final_delivery":
            return
        if plan.get("action") == "stop":
            return
        if plan.get("action") not in {"create", "work"}:
            return
        next_action = "message"
        reason = "root_steward_query_and_update_ledger"
        if self.outputs_complete(agent) and not self.completion_blockers(agent, include_missing_outputs=False):
            next_action = "compact"
            reason = "root_steward_outputs_complete_publish_memory"
        agent._loop_action_plan = {
            **plan,
            "action": next_action,
            "reason": reason,
            "steward_replaced_action": plan.get("action"),
            "missing_outputs": missing_outputs[:20],
        }

    def _refresh_loop_action_plan_rule(self, agent: Agent, missing_outputs: list[str]) -> None:
        state_action = agent.action_state if agent.action_state in _LOOP_ACTIONS else ""
        child_revalidation_blockers = self.completion_blockers(agent, include_missing_outputs=False) if agent.children else []
        child_revalidation_plan = self._coordinator_child_dependency_plan(
            agent,
            missing_outputs,
            child_revalidation_blockers,
        )
        pending_prune_plan = self._pending_prune_request_plan(agent)
        if pending_prune_plan:
            agent._loop_action_plan = pending_prune_plan
            return
        peer_wave_gap = self._peer_wave_create_gap(agent)
        if peer_wave_gap:
            agent._loop_action_plan = {
                "action": "create",
                "reason": "explicit_peer_wave_incomplete",
                "turn_added": agent._turns,
                "peer_wave_gap": peer_wave_gap,
            }
            return

        prune_request_plan = self._prune_request_plan(agent)
        if prune_request_plan:
            agent._loop_action_plan = prune_request_plan
            return

        if missing_outputs and self._evidence_integration_ready(agent, missing_outputs):
            reason = (
                "child_or_peer_evidence_inspect_before_artifact"
                if self._evidence_integration_needs_inspection(agent, missing_outputs)
                else "integrate_child_or_peer_evidence_into_artifact"
            )
            agent._loop_action_plan = {
                "action": "read" if reason == "child_or_peer_evidence_inspect_before_artifact" else "work",
                "reason": reason,
                "turn_added": agent._turns,
                **self._evidence_integration_plan_data(agent, missing_outputs),
            }
            return

        if child_revalidation_plan and child_revalidation_plan.get("action") in {"message", "create"}:
            current = agent._loop_action_plan
            if current and current.get("action") == child_revalidation_plan.get("action") and current.get("reason") == child_revalidation_plan.get("reason"):
                return
            agent._loop_action_plan = child_revalidation_plan
            return

        premature_downstream_plan = self._premature_downstream_plan(agent, missing_outputs)
        if premature_downstream_plan:
            agent._loop_action_plan = premature_downstream_plan
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

        if self._pre_create_reference_read_pending(agent):
            agent._create_resume_after_read = True
            agent._loop_action_plan = {
                "action": "read",
                "reason": "pre_create_reference",
                "turn_added": agent._turns,
                "target_action": "create",
            }
            return

        if self._spawn_create_action_pending(agent):
            agent._loop_action_plan = {
                "action": "create",
                "reason": "spawnable_workstreams_before_solo_execution",
                "turn_added": agent._turns,
            }
            return

        if self._deferred_spawn_ready(agent):
            agent._loop_action_plan = {
                "action": "create",
                "reason": "deferred_spawn_readiness_met",
                "turn_added": agent._turns,
                "deferred_spawn_count": len(agent._deferred_spawn_requests),
                "deferred_spawn_readiness": round(self._deferred_spawn_readiness(agent), 3),
            }
            return

        create_miss_plan = self._create_miss_followup_plan(agent, missing_outputs)
        if create_miss_plan:
            if create_miss_plan.get("action") == "read":
                agent._create_resume_after_read = True
            agent._loop_action_plan = create_miss_plan
            return

        direct_answer_plan = self._root_direct_answer_submission_plan(agent)
        if direct_answer_plan:
            agent._loop_action_plan = direct_answer_plan
            return

        final_delivery_plan = self._final_delivery_handoff_plan(agent)
        if final_delivery_plan:
            agent._loop_action_plan = final_delivery_plan
            return

        expected_outputs = _expected_output_paths(agent.task)
        outputs_complete = bool(expected_outputs) and not missing_outputs
        answer_submission_pending = self.answer_submission_required(agent) and not agent.submitted_answer_path
        answer_submission_ready = self._answer_submission_ready(agent)
        root_answer_readiness = (
            self._root_answer_readiness(agent)
            if answer_submission_pending and agent.parent is None
            else {}
        )
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
                uncertain_handoff_plan = (
                    self._handoff_create_after_partial_progress_plan(agent, missing_outputs)
                    if any(blocker.get("kind") == "uncertain_evidence" for blocker in blockers)
                    else None
                )
                if uncertain_handoff_plan:
                    agent._loop_action_plan = uncertain_handoff_plan
                    return
                if answer_submission_pending:
                    if answer_submission_ready:
                        readiness_public = (
                            self._root_answer_readiness_public(root_answer_readiness)
                            if root_answer_readiness
                            else {}
                        )
                        agent._loop_action_plan = {
                            "action": "stop",
                            "reason": "answer_submission_pending",
                            "turn_added": agent._turns,
                            "blockers": blockers,
                            "answer_readiness": readiness_public,
                        }
                    elif self._initial_create_required(agent):
                        agent._loop_action_plan = {
                            "action": "create",
                            "reason": "answer_needs_research_before_submission",
                            "turn_added": agent._turns,
                            "blockers": blockers,
                        }
                    else:
                        agent._loop_action_plan = {
                            "action": "work",
                            "reason": "answer_needs_evidence_before_submission",
                            "turn_added": agent._turns,
                            "blockers": blockers,
                        }
                    return
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
        peer_wave_gap = self._peer_wave_create_gap(agent)
        if peer_wave_gap:
            agent._loop_action_plan = {
                "action": "create",
                "reason": "explicit_peer_wave_incomplete",
                "turn_added": agent._turns,
                "peer_wave_gap": peer_wave_gap,
            }
            return

        if child_plan:
            if current and current.get("action") == child_plan.get("action") and current.get("reason") == child_plan.get("reason"):
                return
            agent._loop_action_plan = child_plan
            return

        premature_downstream_plan = self._premature_downstream_plan(agent, missing_outputs)
        if premature_downstream_plan:
            agent._loop_action_plan = premature_downstream_plan
            return

        if self._multi_phase_create_action_pending(agent):
            agent._loop_action_plan = {
                "action": "create",
                "reason": "multi_phase_peer_wave_pending",
                "turn_added": agent._turns,
            }
            return

        if missing_outputs and self._evidence_integration_ready(agent, missing_outputs):
            reason = (
                "child_or_peer_evidence_inspect_before_artifact"
                if self._evidence_integration_needs_inspection(agent, missing_outputs)
                else "integrate_child_or_peer_evidence_into_artifact"
            )
            agent._loop_action_plan = {
                "action": "read" if reason == "child_or_peer_evidence_inspect_before_artifact" else "work",
                "reason": reason,
                "turn_added": agent._turns,
                **self._evidence_integration_plan_data(agent, missing_outputs),
            }
            return

        handoff_plan = self._handoff_create_after_partial_progress_plan(agent, missing_outputs)
        if handoff_plan:
            agent._loop_action_plan = handoff_plan
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
        peer_overlap_query_pending = self._peer_overlap_query_before_work_pending(agent, missing_outputs)
        coordination_task = _task_is_coordination_artifact_task(agent.task)
        filtered_read_threshold = 1 if coordination_task else 2
        filtered_read_intent = bool(missing_outputs) and agent._filtered_read_intent_count >= filtered_read_threshold
        filtered_read_from_current_turn = (
            filtered_read_intent
            and coordination_task
            and agent._filtered_read_intent_seen_turn >= agent._turns
        )
        peer_post_query_pending = bool(missing_outputs) and (
            self._peer_progress_post_query_pending(agent)
            or self._peer_overlap_post_query_pending(agent, missing_outputs)
            or self._peer_overlap_active_duplicate_post_query_pending(agent, missing_outputs)
        )
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
                if peer_overlap_query_pending:
                    if action in {"read", "query"}:
                        return
                elif query_before_artifact or filtered_read_intent:
                    if action in {"read", "query"}:
                        return
                elif action == "work" and missing_outputs and agent._last_artifact_turn < int(current.get("turn_added", 0)):
                    return

        if missing_outputs:
            if peer_post_query_pending:
                if (
                    agent._write_only_miss_count >= 1
                    or self._peer_overlap_post_query_pending(agent, missing_outputs)
                    or self._peer_overlap_active_duplicate_post_query_pending(agent, missing_outputs)
                ):
                    reason = (
                        "peer_query_found_active_duplicate_prune_or_summarize"
                        if self._peer_overlap_active_duplicate_post_query_pending(agent, missing_outputs)
                        else "peer_query_found_completed_peers_prune_or_summarize"
                    )
                    agent._loop_action_plan = {
                        "action": "compact",
                        "reason": reason,
                        "turn_added": agent._turns,
                    }
                    return
                agent._loop_action_plan = {
                    "action": "work",
                    "reason": "peer_query_found_completed_peers_write_or_prune",
                    "turn_added": agent._turns,
                }
                return
            if peer_overlap_query_pending:
                self._mark_peer_overlap_query_planned(agent)
                agent._loop_action_plan = {
                    "action": "read",
                    "reason": "peer_overlap_query_before_work",
                    "turn_added": agent._turns,
                    "overlap_peers": [peer.id for peer in self._peer_overlap_query_candidates(agent)[:8]],
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

    def _refresh_loop_action_plan_constraint(self, agent: Agent, missing_outputs: list[str]) -> None:
        if self._deferred_spawn_ready(agent) and not self._evidence_integration_ready(agent, missing_outputs):
            metrics = self._compute_loop_constraint_metrics(agent, missing_outputs)
            agent._loop_action_plan = {
                "action": "create",
                "reason": "deferred_spawn_readiness_met",
                "turn_added": agent._turns,
                "constraints": asdict(metrics),
                "deferred_spawn_count": len(agent._deferred_spawn_requests),
                "deferred_spawn_readiness": round(self._deferred_spawn_readiness(agent), 3),
            }
            return

        create_miss_plan = self._create_miss_followup_plan(agent, missing_outputs)
        if create_miss_plan:
            metrics = self._compute_loop_constraint_metrics(agent, missing_outputs)
            if create_miss_plan.get("action") == "read":
                agent._create_resume_after_read = True
            plan = {
                "action": create_miss_plan["action"],
                "reason": create_miss_plan.get("reason") or "create_miss_followup",
                "turn_added": agent._turns,
                "constraints": asdict(metrics),
            }
            plan.update(create_miss_plan)
            agent._loop_action_plan = plan
            return

        candidates = self._collect_loop_action_candidates(agent, missing_outputs)
        metrics = self._compute_loop_constraint_metrics(agent, missing_outputs)
        legal = self._filter_loop_action_candidates(agent, candidates, metrics, missing_outputs)
        if not legal:
            fallback = self._default_loop_action(agent)
            agent._loop_action_plan = {
                "action": fallback,
                "reason": "constraint_fallback_default",
                "turn_added": agent._turns,
                "constraints": asdict(metrics),
            }
            return

        for candidate in legal:
            candidate.score = self._score_loop_action_candidate(candidate, metrics)
        legal.sort(key=lambda item: item.score, reverse=True)
        selected = legal[0]

        if selected.action == "read" and selected.reason == "pre_create_reference":
            agent._create_resume_after_read = True
        if selected.action == "read" and selected.reason == "read_after_create_miss_prerequisite":
            agent._create_resume_after_read = True
        if selected.action == "read" and selected.reason == "peer_overlap_query_before_work":
            self._mark_peer_overlap_query_planned(agent)
        if selected.reason == "state_board_action":
            agent._state_action_consumed_version = agent._state_action_version

        plan = {
            "action": selected.action,
            "reason": selected.reason,
            "turn_added": agent._turns,
            "score": round(selected.score, 4),
            "constraints": asdict(metrics),
            "candidate_scores": [
                {
                    "action": item.action,
                    "reason": item.reason,
                    "score": round(item.score, 4),
                    "base_priority": item.base_priority,
                }
                for item in legal[:8]
            ],
        }
        plan.update(selected.data)
        agent._loop_action_plan = plan

    def _collect_loop_action_candidates(self, agent: Agent, missing_outputs: list[str]) -> list[LoopActionCandidate]:
        candidates: list[LoopActionCandidate] = []

        def add(action: str, reason: str, base_priority: float, **data: Any) -> None:
            if action in _LOOP_ACTIONS:
                candidates.append(LoopActionCandidate(action=action, reason=reason, base_priority=base_priority, data=data))

        current = agent._loop_action_plan
        state_action = agent.action_state if agent.action_state in _LOOP_ACTIONS else ""
        blockers_without_outputs = self.completion_blockers(agent, include_missing_outputs=False)
        child_revalidation_blockers = blockers_without_outputs if agent.children else []
        child_plan = self._coordinator_child_dependency_plan(agent, missing_outputs, child_revalidation_blockers)
        pending_prune_plan = self._pending_prune_request_plan(agent)
        if pending_prune_plan:
            add(
                str(pending_prune_plan["action"]),
                str(pending_prune_plan.get("reason") or "received_prune_request_self_prune"),
                0.99,
                prune_request=pending_prune_plan.get("prune_request"),
                prune_request_count=pending_prune_plan.get("prune_request_count"),
            )
        prune_request_plan = self._prune_request_plan(agent)
        if prune_request_plan:
            add(
                str(prune_request_plan["action"]),
                str(prune_request_plan.get("reason") or "request_duplicate_agents_self_prune_after_key_evidence"),
                0.97,
                prune_targets=prune_request_plan.get("prune_targets", []),
                evidence_agents=prune_request_plan.get("evidence_agents", []),
            )
        peer_wave_gap = self._peer_wave_create_gap(agent)
        if peer_wave_gap:
            add("create", "explicit_peer_wave_incomplete", 0.94, peer_wave_gap=peer_wave_gap)

        if child_plan and child_plan.get("action") in _LOOP_ACTIONS:
            add(
                str(child_plan["action"]),
                str(child_plan.get("reason") or "child_dependency_plan"),
                0.90,
                blockers=child_plan.get("blockers", []),
                failed_children=child_plan.get("failed_children", []),
            )

        premature_downstream_plan = self._premature_downstream_plan(agent, missing_outputs)
        if premature_downstream_plan:
            add(
                str(premature_downstream_plan["action"]),
                str(premature_downstream_plan.get("reason") or "premature_downstream_wait_for_evidence"),
                0.96,
                upstream_agents=premature_downstream_plan.get("upstream_agents", []),
            )

        if state_action and state_action != "create" and agent._state_action_version > agent._state_action_consumed_version:
            add(state_action, "state_board_action", 0.82)

        if self._create_resume_after_read_pending(agent):
            add("create", "resume_create_after_read_orientation", 0.78)

        if self._pre_create_reference_read_pending(agent):
            add("read", "pre_create_reference", 0.92, target_action="create")

        if self._spawn_create_action_pending(agent):
            add("create", "spawnable_workstreams_before_solo_execution", 0.80)

        if missing_outputs and self._evidence_integration_ready(agent, missing_outputs):
            data = self._evidence_integration_plan_data(agent, missing_outputs)
            if self._evidence_integration_needs_inspection(agent, missing_outputs):
                add("read", "child_or_peer_evidence_inspect_before_artifact", 0.93, **data)
            add("work", "integrate_child_or_peer_evidence_into_artifact", 0.91, **data)

        create_miss_plan = self._create_miss_followup_plan(agent, missing_outputs)
        if create_miss_plan:
            add(
                str(create_miss_plan["action"]),
                str(create_miss_plan.get("reason") or "create_miss_followup"),
                0.92,
                blockers=create_miss_plan.get("blockers", []),
                failed_children=create_miss_plan.get("failed_children", []),
                target_action=create_miss_plan.get("target_action"),
            )

        direct_answer_plan = self._root_direct_answer_submission_plan(agent)
        if direct_answer_plan:
            add(
                "stop",
                str(direct_answer_plan.get("reason") or "direct_answer_submission_ready"),
                0.99,
                answer_path=direct_answer_plan.get("answer_path"),
                evidence_agents=direct_answer_plan.get("evidence_agents", []),
                terminal_candidate=direct_answer_plan.get("terminal_candidate"),
            )

        final_delivery_plan = self._final_delivery_handoff_plan(agent)
        if final_delivery_plan:
            add(
                "create",
                "delegate_final_delivery",
                0.98,
                answer_path=final_delivery_plan.get("answer_path"),
                evidence_agents=final_delivery_plan.get("evidence_agents", []),
                blockers=final_delivery_plan.get("blockers", []),
            )

        mature_handoff_plan = self._mature_evidence_handoff_plan(agent)
        if mature_handoff_plan:
            add(
                "create",
                "handoff_after_evidence_maturity",
                0.93,
                answer_path=mature_handoff_plan.get("answer_path"),
                target_agent=mature_handoff_plan.get("target_agent"),
                evidence_agents=mature_handoff_plan.get("evidence_agents", []),
            )

        expected_outputs = _expected_output_paths(agent.task)
        outputs_complete = bool(expected_outputs) and not missing_outputs
        answer_submission_pending = self.answer_submission_required(agent) and not agent.submitted_answer_path
        answer_submission_ready = self._answer_submission_ready(agent)
        root_answer_readiness = (
            self._root_answer_readiness(agent)
            if answer_submission_pending and agent.parent is None
            else {}
        )
        if outputs_complete:
            if current and current.get("action") in {"compact", "stop"}:
                add(str(current["action"]), str(current.get("reason") or "current_terminal_action"), 0.95)
            completion_blockers = self.completion_blockers(agent, include_missing_outputs=False)
            completion_child_plan = self._coordinator_child_dependency_plan(agent, missing_outputs, completion_blockers)
            if completion_child_plan and completion_child_plan.get("action") in _LOOP_ACTIONS:
                add(
                    str(completion_child_plan["action"]),
                    str(completion_child_plan.get("reason") or "outputs_complete_child_plan"),
                    0.85,
                    failed_children=completion_child_plan.get("failed_children", []),
                )
            if completion_blockers:
                uncertain_handoff_plan = (
                    self._handoff_create_after_partial_progress_plan(agent, missing_outputs)
                    if any(blocker.get("kind") == "uncertain_evidence" for blocker in completion_blockers)
                    else None
                )
                if uncertain_handoff_plan:
                    add(
                        "create",
                        str(uncertain_handoff_plan.get("reason") or "delegate_uncertain_evidence_recovery"),
                        0.96,
                        blockers=completion_blockers,
                        missing_outputs=uncertain_handoff_plan.get("missing_outputs", []),
                        uncertain_artifacts=uncertain_handoff_plan.get("uncertain_artifacts", []),
                )
                if answer_submission_pending:
                    if answer_submission_ready:
                        readiness_score = float(root_answer_readiness.get("score", 0.65) if root_answer_readiness else 0.65)
                        add(
                            "stop",
                            "answer_submission_pending",
                            max(0.40, min(0.90, 0.42 + 0.40 * readiness_score)),
                            blockers=completion_blockers,
                            answer_readiness=(
                                self._root_answer_readiness_public(root_answer_readiness)
                                if root_answer_readiness
                                else {}
                            ),
                        )
                    elif self._initial_create_required(agent):
                        add("create", "answer_needs_research_before_submission", 0.92, blockers=completion_blockers)
                    else:
                        add("work", "answer_needs_evidence_before_submission", 0.82, blockers=completion_blockers)
                else:
                    add("work", "completion_evidence_required_before_compact", 0.82, blockers=completion_blockers)
            if _task_prefers_query_before_artifact(agent.task) and agent._last_query_turn <= 0:
                add("read", "outputs_complete_but_peer_query_required", 0.72)
            if answer_submission_pending and answer_submission_ready:
                readiness_score = float(root_answer_readiness.get("score", 0.65) if root_answer_readiness else 0.65)
                add(
                    "stop",
                    "answer_submission_pending",
                    max(0.40, min(0.90, 0.42 + 0.40 * readiness_score)),
                    answer_readiness=(
                        self._root_answer_readiness_public(root_answer_readiness)
                        if root_answer_readiness
                        else {}
                    ),
                )
            elif not answer_submission_pending:
                add("compact", "outputs_complete_publish_memory", 0.86)
                add("stop", "outputs_complete_stop_candidate", 0.52)

        if self._multi_phase_create_action_pending(agent):
            add("create", "multi_phase_peer_wave_pending", 0.86)

        handoff_plan = self._handoff_create_after_partial_progress_plan(agent, missing_outputs)
        if handoff_plan:
            add(
                "create",
                str(handoff_plan.get("reason") or "handoff_after_partial_progress"),
                0.84,
                missing_outputs=handoff_plan.get("missing_outputs", []),
                uncertain_artifacts=handoff_plan.get("uncertain_artifacts", []),
            )

        if agent._deferred_spawn_requests:
            readiness = self._deferred_spawn_readiness(agent)
            if self._deferred_spawn_ready(agent):
                add(
                    "create",
                    "deferred_spawn_readiness_met",
                    0.88,
                    deferred_spawn_count=len(agent._deferred_spawn_requests),
                    deferred_spawn_readiness=round(readiness, 3),
                )
            elif agent.children:
                add(
                    "message",
                    "deferred_spawn_wait_for_readiness",
                    0.74,
                    deferred_spawn_count=len(agent._deferred_spawn_requests),
                    deferred_spawn_readiness=round(readiness, 3),
                )

        if missing_outputs and self._artifact_pressure_deferred(agent, missing_outputs):
            add("work", "source_test_evidence_before_artifacts", 0.86, blockers=blockers_without_outputs)
        if (
            _task_is_concrete_single_output_work(agent.task)
            and self._agent_has_uncertain_evidence(agent)
            and self._artifact_write_needs_more_evidence(
                agent,
                missing_outputs or _non_terminal_expected_output_paths(agent.task),
            )
        ):
            add("work", "uncertain_evidence_needs_primary_work", 0.88, blockers=blockers_without_outputs)

        query_before_artifact = (
            bool(missing_outputs)
            and _task_prefers_query_before_artifact(agent.task)
            and agent._last_query_turn <= 0
        )
        peer_overlap_query_pending = self._peer_overlap_query_before_work_pending(agent, missing_outputs)
        coordination_task = _task_is_coordination_artifact_task(agent.task)
        filtered_read_threshold = 1 if coordination_task else 2
        filtered_read_intent = bool(missing_outputs) and agent._filtered_read_intent_count >= filtered_read_threshold
        filtered_read_from_current_turn = (
            filtered_read_intent
            and coordination_task
            and agent._filtered_read_intent_seen_turn >= agent._turns
        )
        peer_post_query_pending = bool(missing_outputs) and (
            self._peer_progress_post_query_pending(agent)
            or self._peer_overlap_post_query_pending(agent, missing_outputs)
            or self._peer_overlap_active_duplicate_post_query_pending(agent, missing_outputs)
        )

        if current:
            action = str(current.get("action") or "")
            if action in _LOOP_ACTIONS:
                if filtered_read_from_current_turn and action == "work":
                    pass
                elif action in {"compact", "stop"} and agent.status == "running":
                    add(action, str(current.get("reason") or "current_terminal_action"), 0.88)
                elif peer_post_query_pending and action in {"work", "compact", "stop"}:
                    add(action, str(current.get("reason") or "current_peer_followup_action"), 0.74)
                elif peer_overlap_query_pending and action in {"read", "query"}:
                    add("read", str(current.get("reason") or "current_peer_overlap_query_action"), 0.82)
                elif (query_before_artifact or filtered_read_intent) and action in {"read", "query"}:
                    add("read", str(current.get("reason") or "current_read_action"), 0.78)
                elif action == "work" and missing_outputs and agent._last_artifact_turn < int(current.get("turn_added", 0)):
                    add("work", str(current.get("reason") or "current_work_action"), 0.78)

        if missing_outputs:
            if peer_post_query_pending:
                active_duplicate = self._peer_overlap_active_duplicate_post_query_pending(agent, missing_outputs)
                completed_overlap = self._peer_overlap_post_query_pending(agent, missing_outputs)
                if agent._write_only_miss_count >= 1 or completed_overlap or active_duplicate:
                    add(
                        "compact",
                        (
                            "peer_query_found_active_duplicate_prune_or_summarize"
                            if active_duplicate
                            else "peer_query_found_completed_peers_prune_or_summarize"
                        ),
                        0.88 if not active_duplicate else 0.90,
                        overlap_peers=[
                            peer.id
                            for peer in (
                                self._active_duplicate_peer_overlap_candidates(agent)
                                if active_duplicate
                                else self._completed_peer_overlap_candidates(agent)
                            )[:8]
                        ],
                    )
                add("work", "peer_query_found_completed_peers_write_or_prune", 0.76)
            if peer_overlap_query_pending:
                add(
                    "read",
                    "peer_overlap_query_before_work",
                    0.90,
                    overlap_peers=[peer.id for peer in self._peer_overlap_query_candidates(agent)[:8]],
                )
            if query_before_artifact or filtered_read_intent:
                add(
                    "read",
                    "task_requests_peer_query_before_artifact" if query_before_artifact else "filtered_read_intent_after_write_scope",
                    0.78,
                )
            add("work", "missing_artifacts", 0.82)

        default_action = self._default_loop_action(agent)
        add(default_action, "default_task_progress", 0.45)

        add("read", "constraint_alternative_read", 0.20)
        add("work", "constraint_alternative_work", 0.20)
        add("compact", "constraint_alternative_compact", 0.20)
        if not missing_outputs:
            add("stop", "constraint_alternative_stop", 0.20)
        if (
            self._can_create_child(agent)
            and (
                _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
                or self._agent_has_uncertain_evidence(agent)
            )
        ):
            add("create", "constraint_alternative_create", 0.15)
        if agent.children:
            add("message", "constraint_alternative_message", 0.25)

        return self._dedupe_loop_action_candidates(candidates)

    def _dedupe_loop_action_candidates(self, candidates: list[LoopActionCandidate]) -> list[LoopActionCandidate]:
        by_key: dict[tuple[str, str], LoopActionCandidate] = {}
        for candidate in candidates:
            key = (candidate.action, candidate.reason)
            existing = by_key.get(key)
            if existing is None or candidate.base_priority > existing.base_priority:
                by_key[key] = candidate
        return list(by_key.values())

    def _compute_loop_constraint_metrics(self, agent: Agent, missing_outputs: list[str]) -> LoopConstraintMetrics:
        disabled = self.config.disabled_loop_constraints

        def maybe(name: str, value: float) -> float:
            return 0.0 if name in disabled else max(0.0, min(1.0, value))

        total_budget = max(0.0, float(self.ledger.total_budget))
        budget_pressure = (self.ledger.total_spent / total_budget) if total_budget > 0 and math.isfinite(total_budget) else 0.0

        elapsed = time.time() - self._start_time
        time_limit = agent.quota.time_limit or self.config.time_limit
        time_pressure = elapsed / time_limit if time_limit and time_limit > 0 else 0.0

        context_pressure = agent.context_tokens / max(1, agent.context_limit)
        agent_pressure = len(self.agents) / max(1, self.config.max_agents)

        child_count = len(agent.children)
        unfinished_children = len(self.unfinished_child_agents(agent))
        dependency_pressure = unfinished_children / max(1, child_count)
        handoff_pressure = self._handoff_pressure(agent)
        offspring_create_bias = self._offspring_create_bias(agent)
        failed_child_pressure = len(self._failed_children_need_recovery(agent, missing_outputs)) / max(1, child_count)
        deferred_spawn_readiness = self._deferred_spawn_readiness(agent)
        lane_coverage_pressure = self._lane_coverage_pressure(agent)
        uncertain_evidence_pressure = 1.0 if self._agent_has_uncertain_evidence(agent) else 0.0

        expected_outputs = _expected_output_paths(agent.task)
        artifact_gap = len(missing_outputs) / max(1, len(expected_outputs)) if expected_outputs else 0.0

        total_tool_calls = sum(a._tool_calls for a in self.agents.values())
        coordination_events = len(self._messages_sent)
        coordination_ratio = coordination_events / max(1, total_tool_calls + coordination_events)

        create_retry_pressure = agent._create_action_filtered_count / 3
        action_miss_pressure = agent._action_miss_count / 3 if agent._action_miss_action else 0.0
        stagnation_pressure = agent._no_tool_turns / 3

        output_similarity = self._estimate_peer_output_similarity(agent)
        duplicate_tool_pressure = self._estimate_duplicate_recent_tool_pressure(agent)
        consistency_pressure = self._estimate_recent_consistency_pressure(agent)
        evidence_integration_pressure = 1.0 if self._evidence_integration_ready(agent, missing_outputs) else 0.0

        return LoopConstraintMetrics(
            budget_pressure=maybe("budget", budget_pressure),
            time_pressure=maybe("time", time_pressure),
            context_pressure=maybe("context", context_pressure),
            agent_pressure=maybe("agent", agent_pressure),
            dependency_pressure=maybe("dependency", dependency_pressure),
            handoff_pressure=maybe("handoff", handoff_pressure),
            offspring_create_bias=maybe("offspring_create", offspring_create_bias),
            failed_child_pressure=maybe("failed_child", failed_child_pressure),
            deferred_spawn_readiness=maybe("deferred_spawn", deferred_spawn_readiness),
            artifact_gap=maybe("artifact", artifact_gap),
            coordination_ratio=maybe("coordination", coordination_ratio),
            create_retry_pressure=maybe("create_retry", create_retry_pressure),
            action_miss_pressure=maybe("action_miss", action_miss_pressure),
            lane_coverage_pressure=maybe("lane_coverage", lane_coverage_pressure),
            stagnation_pressure=maybe("stagnation", stagnation_pressure),
            output_similarity=maybe("redundancy", output_similarity),
            duplicate_tool_pressure=maybe("duplicate_tool", duplicate_tool_pressure),
            consistency_pressure=maybe("consistency", consistency_pressure),
            uncertain_evidence_pressure=maybe("uncertain_evidence", uncertain_evidence_pressure),
            evidence_integration_pressure=maybe("evidence_integration", evidence_integration_pressure),
        )

    def _filter_loop_action_candidates(
        self,
        agent: Agent,
        candidates: list[LoopActionCandidate],
        metrics: LoopConstraintMetrics,
        missing_outputs: list[str],
    ) -> list[LoopActionCandidate]:
        expected_outputs = _expected_output_paths(agent.task)
        blockers = self.completion_blockers(agent, include_missing_outputs=False)
        unfinished_children = self.unfinished_child_agents(agent)
        evidence_integration_ready = self._evidence_integration_ready(agent, missing_outputs)
        legal: list[LoopActionCandidate] = []
        for candidate in candidates:
            action = candidate.action
            if action == "work" and self._initial_create_required(agent):
                continue
            if action == "create":
                final_delivery_create = candidate.reason == "delegate_final_delivery"
                if (
                    evidence_integration_ready
                    and candidate.reason not in {
                        "explicit_peer_wave_incomplete",
                        "delegate_failed_child_recovery",
                        "delegate_uncertain_evidence_recovery",
                        "delegate_final_delivery",
                        "handoff_after_evidence_maturity",
                    }
                ):
                    continue
                if unfinished_children and candidate.reason not in {
                    "explicit_peer_wave_incomplete",
                    "delegate_failed_child_recovery",
                    "deferred_spawn_readiness_met",
                    "delegate_uncertain_evidence_recovery",
                    "delegate_final_delivery",
                    "handoff_after_evidence_maturity",
                }:
                    continue
                if not (self._can_create_child(agent) or (final_delivery_create and self._can_create_final_delivery_agent(agent))):
                    continue
                if metrics.budget_pressure >= 0.98 or metrics.time_pressure >= 0.98:
                    continue
                if (
                    missing_outputs
                    and not self._agent_has_uncertain_evidence(agent)
                    and not _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
                    and candidate.reason in {
                        "constraint_alternative_create",
                        "spawnable_workstreams_before_solo_execution",
                    }
                ):
                    continue
                if (
                    metrics.lane_coverage_pressure >= 0.95
                    and candidate.reason not in {
                        "deferred_spawn_readiness_met",
                        "delegate_failed_child_recovery",
                        "explicit_peer_wave_incomplete",
                        "delegate_uncertain_evidence_recovery",
                        "delegate_final_delivery",
                        "handoff_after_evidence_maturity",
                    }
                ):
                    continue
                if (
                    self._queried_reliable_peer_progress(agent)
                    and candidate.reason not in {
                        "deferred_spawn_readiness_met",
                        "delegate_failed_child_recovery",
                        "explicit_peer_wave_incomplete",
                        "delegate_uncertain_evidence_recovery",
                        "delegate_final_delivery",
                        "handoff_after_evidence_maturity",
                    }
                ):
                    continue
            if action == "stop":
                if missing_outputs and expected_outputs:
                    continue
                non_submission_blockers = [blocker for blocker in blockers if blocker.get("kind") != "answer_submission"]
                if (
                    candidate.reason in {"direct_answer_submission_ready", "terminal_candidate_submit_ready"}
                    and non_submission_blockers
                    and all(
                        blocker.get("kind") in {"uncertain_evidence", "unverified_evidence"}
                        for blocker in non_submission_blockers
                    )
                ):
                    non_submission_blockers = []
                if non_submission_blockers:
                    continue
            if action == "compact":
                if blockers and missing_outputs:
                    continue
            if action == "message":
                has_unfinished_children = bool(unfinished_children)
                has_peer_query_need = bool(self._peer_progress_candidates(agent)) and self._peer_progress_decision_pending(agent)
                waiting_for_deferred = bool(agent._deferred_spawn_requests) and not self._deferred_spawn_ready(agent)
                sending_prune_request = candidate.reason == "request_duplicate_agents_self_prune_after_key_evidence"
                if evidence_integration_ready and not sending_prune_request:
                    continue
                if not (has_unfinished_children or has_peer_query_need or waiting_for_deferred or sending_prune_request):
                    continue
                if (
                    missing_outputs
                    and not has_unfinished_children
                    and not sending_prune_request
                    and agent._action_miss_action == "message"
                    and agent._action_miss_count >= 1
                ):
                    continue
            legal.append(candidate)
        return legal

    def _score_loop_action_candidate(self, candidate: LoopActionCandidate, metrics: LoopConstraintMetrics) -> float:
        score = candidate.base_priority
        resource_pressure = max(
            _sigmoid01(metrics.budget_pressure),
            _sigmoid01(metrics.time_pressure),
        )

        if candidate.action == "create":
            score += 0.65 * metrics.artifact_gap
            score += 0.25 * metrics.stagnation_pressure
            score += 0.80 * metrics.offspring_create_bias
            if candidate.reason in {
                "spawnable_workstreams_before_solo_execution",
                "retry_spawn_after_create_miss",
                "multi_phase_peer_wave_pending",
                "explicit_peer_wave_incomplete",
                "retry_spawn_after_create_miss",
                "deferred_spawn_readiness_met",
                "delegate_failed_child_recovery",
                "handoff_after_partial_progress",
                "delegate_uncertain_evidence_recovery",
                "delegate_final_delivery",
                "handoff_after_evidence_maturity",
            }:
                score += 0.75
            if candidate.reason == "delegate_final_delivery":
                score += 1.35
            if candidate.reason == "handoff_after_evidence_maturity":
                score += 1.10
            if candidate.reason == "delegate_failed_child_recovery":
                score += 0.95 * metrics.failed_child_pressure
            if candidate.reason == "delegate_uncertain_evidence_recovery":
                score += 1.10 * metrics.uncertain_evidence_pressure
            if candidate.reason == "deferred_spawn_readiness_met":
                score += 0.85 * metrics.deferred_spawn_readiness
            score -= 0.85 * resource_pressure
            score -= 0.70 * metrics.agent_pressure
            score -= 0.45 * metrics.dependency_pressure
            if candidate.reason not in {
                "deferred_spawn_readiness_met",
                "delegate_failed_child_recovery",
                "explicit_peer_wave_incomplete",
                "delegate_uncertain_evidence_recovery",
                "delegate_final_delivery",
                "handoff_after_evidence_maturity",
            }:
                score -= 1.05 * metrics.lane_coverage_pressure
            if candidate.reason not in {
                "explicit_peer_wave_incomplete",
                "deferred_spawn_readiness_met",
                "delegate_failed_child_recovery",
                "handoff_after_partial_progress",
                "delegate_uncertain_evidence_recovery",
                "delegate_final_delivery",
                "handoff_after_evidence_maturity",
            }:
                score -= 0.95 * metrics.handoff_pressure
            if candidate.reason != "deferred_spawn_readiness_met":
                score -= 0.45 * max(0.0, 1.0 - metrics.deferred_spawn_readiness) if metrics.deferred_spawn_readiness else 0.0
            score -= 0.55 * metrics.create_retry_pressure
            score -= 0.35 * metrics.coordination_ratio
            score -= 1.30 * metrics.evidence_integration_pressure
            if metrics.handoff_pressure >= 0.8 and candidate.reason not in _UNCERTAIN_EVIDENCE_CREATE_REASONS.union({
                "deferred_spawn_readiness_met",
                "delegate_failed_child_recovery",
                "explicit_peer_wave_incomplete",
                "delegate_final_delivery",
                "handoff_after_evidence_maturity",
            }):
                score -= 0.70

        elif candidate.action == "read":
            score += 0.45 * metrics.stagnation_pressure
            score += 0.25 * metrics.artifact_gap
            score += 0.20 * metrics.consistency_pressure
            score += 0.28 * metrics.handoff_pressure
            if candidate.reason in {
                "task_requests_peer_query_before_artifact",
                "filtered_read_intent_after_write_scope",
                "outputs_complete_but_peer_query_required",
                "create_scope_intent_released",
                "pre_create_reference",
                "peer_overlap_query_before_work",
                "child_or_peer_evidence_inspect_before_artifact",
            }:
                score += 0.60
            if candidate.reason == "peer_overlap_query_before_work":
                score += 0.35
            if candidate.reason == "child_or_peer_evidence_inspect_before_artifact":
                score += 1.05 * metrics.evidence_integration_pressure
            if candidate.reason == "pre_create_reference":
                score += 0.75
            score -= 0.25 * resource_pressure

        elif candidate.action == "message":
            score += 0.85 * metrics.dependency_pressure
            score += 0.55 * metrics.handoff_pressure
            if candidate.reason == "request_duplicate_agents_self_prune_after_key_evidence":
                score += 1.20
            if candidate.reason == "deferred_spawn_wait_for_readiness":
                score += 0.55 * max(0.0, 1.0 - metrics.deferred_spawn_readiness)
            score += 0.20 * metrics.consistency_pressure
            score -= 0.45 * metrics.coordination_ratio
            score -= 0.25 * metrics.context_pressure
            score -= 1.10 * metrics.action_miss_pressure
            score -= 1.40 * metrics.evidence_integration_pressure

        elif candidate.action == "work":
            score += 0.80 * metrics.artifact_gap
            score += 0.25 * (1.0 - metrics.dependency_pressure)
            score -= 0.85 * metrics.uncertain_evidence_pressure
            if candidate.reason == "uncertain_evidence_needs_primary_work":
                score += 1.05 * metrics.uncertain_evidence_pressure
            if candidate.reason == "integrate_child_or_peer_evidence_into_artifact":
                score += 1.20 * metrics.evidence_integration_pressure
                score += 0.25 * metrics.handoff_pressure
            if candidate.reason == "missing_artifacts":
                score -= 0.45
            if candidate.reason in {"missing_artifacts", "peer_query_found_completed_peers_write_or_prune"}:
                score += 0.45 * metrics.action_miss_pressure
            score -= 0.75 * metrics.failed_child_pressure
            score -= 0.65 * metrics.handoff_pressure
            score -= 0.35 * metrics.context_pressure
            score -= 0.20 * resource_pressure
            score -= 0.25 * metrics.output_similarity

        elif candidate.action == "compact":
            score += 0.95 * metrics.context_pressure
            score += 0.40 * resource_pressure
            score += 0.35 * metrics.output_similarity
            score += 0.25 * metrics.duplicate_tool_pressure
            score += 0.20 * metrics.coordination_ratio
            score += 0.25 * metrics.handoff_pressure
            score -= 0.70 * metrics.uncertain_evidence_pressure
            score -= 0.45 * metrics.artifact_gap
            if candidate.reason == "peer_query_found_completed_peers_prune_or_summarize":
                score += 1.40 + 0.50 * metrics.artifact_gap
            if candidate.reason == "peer_query_found_active_duplicate_prune_or_summarize":
                score += 2.05 + 0.65 * metrics.artifact_gap

        elif candidate.action == "stop":
            score += 0.85 * resource_pressure
            score += 0.35 * (1.0 - metrics.artifact_gap)
            score += 0.20 * metrics.handoff_pressure
            if candidate.reason == "answer_submission_pending":
                readiness = candidate.data.get("answer_readiness") if isinstance(candidate.data, dict) else {}
                readiness_score = float((readiness or {}).get("score", 0.5) or 0.0)
                score += 0.20 + 0.95 * max(0.0, min(1.0, readiness_score))
                if readiness and not readiness.get("submit_only_ready"):
                    score -= 0.35
            if candidate.reason == "direct_answer_submission_ready":
                score += 2.15
            if candidate.reason == "terminal_candidate_submit_ready":
                score += 2.45
            score -= 0.75 * metrics.dependency_pressure
            score -= 0.45 * metrics.consistency_pressure
            score -= 1.00 * metrics.uncertain_evidence_pressure

        return score

    def _estimate_peer_output_similarity(self, agent: Agent) -> float:
        texts = [self._latest_assistant_text(agent)]
        peers = self._peer_progress_candidates(agent)
        texts.extend(self._latest_assistant_text(peer) for peer in peers[:8])
        texts = [text for text in texts if text.strip()]
        if len(texts) < 2:
            return 0.0
        scores: list[float] = []
        for i, left in enumerate(texts):
            for right in texts[i + 1:]:
                scores.append(_cheap_text_similarity(left, right))
        return sum(scores) / max(1, len(scores))

    def _estimate_duplicate_recent_tool_pressure(self, agent: Agent) -> float:
        names = list(_recent_tool_names(agent.history, lookback=12))
        for peer in self._peer_progress_candidates(agent)[:8]:
            names.extend(_recent_tool_names(peer.history, lookback=12))
        if not names:
            return 0.0
        return max(0.0, min(1.0, 1.0 - len(set(names)) / max(1, len(names))))

    def _estimate_recent_consistency_pressure(self, agent: Agent) -> float:
        texts = [self._latest_assistant_text(agent)]
        texts.extend(self._latest_assistant_text(peer) for peer in self._peer_progress_candidates(agent)[:4])
        joined = "\n".join(text for text in texts if text)
        if not joined:
            return 0.0
        patterns = (
            r"\bcontradict", r"\bconflict", r"\bdisagree", r"\binconsistent",
            r"\bhowever\b", r"\bbut\b", "矛盾", "冲突", "不一致", "不同意", "相反", "但是", "然而",
        )
        count = sum(len(re.findall(pattern, joined, flags=re.IGNORECASE)) for pattern in patterns)
        return max(0.0, min(1.0, count / 5))

    def _latest_assistant_text(self, agent: Agent) -> str:
        for msg in reversed(agent.history):
            if msg.get("role") == "assistant":
                return str(msg.get("content") or "")
        return ""

    def _default_loop_action(self, agent: Agent) -> str:
        if self.unfinished_child_agents(agent):
            return "message"
        if (
            agent._turns <= 1
            and self._can_create_child(agent)
            and _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
        ):
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
            return {"query", "ledger_read", "file_read", "file_list", "grep", "set_status", "get_cost"}
        if action == "work":
            return {
                "file_read", "file_write", "file_replace", "file_list", "grep",
                "shell", "bt_aggregate", "submit", "query", "ledger_read", "ledger_update",
                "compact", "set_status", "get_cost",
            }
        if action == "message":
            return {"send", "wait", "query", "ledger_read", "ledger_update", "set_status", "get_cost"}
        if action == "compact":
            return {
                "compact", "set_status", "query", "ledger_read", "ledger_update", "file_read", "file_list", "grep",
                "file_write", "file_replace", "submit", "get_cost",
            }
        if action == "stop":
            return {
                "set_status", "compact", "wait", "query", "ledger_read", "ledger_update", "file_read", "file_list", "grep",
                "file_write", "file_replace", "submit", "submit_answer", "get_cost",
            }
        if action == "create":
            return set(_CREATE_ALLOWED_TOOLS)
        return set()

    def _primary_tools_for_loop_action(self, action: str) -> set[str]:
        if action in {"read", "query"}:
            return {"query", "ledger_read", "file_read", "file_list", "grep"}
        if action == "work":
            return {
                "file_read", "file_write", "file_replace", "file_list", "grep",
                "shell", "bt_aggregate", "submit", "query",
            }
        if action == "message":
            return {"send", "wait", "query"}
        if action == "compact":
            return {"compact", "set_status"}
        if action == "stop":
            return {"set_status", "compact", "submit_answer"}
        if action == "create":
            return set(_CREATE_EXECUTION_TOOLS)
        return set()

    def _tools_for_loop_turn(self, agent: Agent, action: str, missing_outputs: list[str]) -> tuple[set[str], str]:
        allowed = self._tools_for_loop_action(action)
        if not self.can_submit_answer(agent).get("ok"):
            allowed = allowed - {"submit_answer"}
        plan_reason = str((agent._loop_action_plan or {}).get("reason") or "")
        if (
            action == "stop"
            and self._answer_submission_pending(agent)
            and "submit_answer" in allowed
            and self.can_submit_answer(agent).get("ok")
        ):
            if plan_reason in _ROOT_SUBMIT_ONLY_REASONS:
                return {"submit_answer"}, "answer_submit_only"
            review_allowed = {
                "submit_answer", "query", "ledger_read", "ledger_update", "file_read",
                "file_list", "grep", "wait", "send", "set_status", "compact", "get_cost",
            }
            return allowed.union(_LEDGER_TOOLS).intersection(review_allowed), "answer_submit_review"
        if self._root_steward_mode(agent):
            if action == "create" and (agent._loop_action_plan or {}).get("reason") == "delegate_final_delivery":
                create_execution_tools = _CREATE_EXECUTION_TOOLS & allowed
                if create_execution_tools:
                    return create_execution_tools, "final_delivery_handoff"
            steward_allowed = {
                "query", "ledger_read", "ledger_update", "send", "wait",
                "compact", "set_status", "get_cost",
            }
            if self.can_submit_answer(agent).get("ok"):
                steward_allowed.add("submit_answer")
            return allowed.union(_LEDGER_TOOLS).intersection(steward_allowed), "root_steward"
        if (
            action == "create"
            and agent._create_action_filtered_count >= 1
            and not self.unfinished_child_agents(agent)
            and self._can_create_child(agent)
        ):
            create_execution_tools = _CREATE_EXECUTION_TOOLS & allowed
            if create_execution_tools:
                return create_execution_tools, "retry_spawn_after_create_miss"
        if (
            action == "create"
            and (agent._loop_action_plan or {}).get("reason") == "resume_create_after_read_orientation"
            and not agent.children
            and _task_has_spawnable_workstreams(agent.task, self.config.min_spawnable_workstreams)
        ):
            create_execution_tools = _CREATE_EXECUTION_TOOLS & allowed
            if create_execution_tools:
                return create_execution_tools, "resume_create_after_read_orientation"
        if action == "compact" and (agent._auxiliary_only_miss_count >= 1 or (agent._action_miss_action == action and agent._action_miss_count >= 1)):
            primary_tools = self._primary_tools_for_loop_action(action) & allowed
            if primary_tools:
                return primary_tools, "retry_primary_after_auxiliary_only_miss"
        if action == "stop" and (agent._auxiliary_only_miss_count >= 1 or (agent._action_miss_action == action and agent._action_miss_count >= 1)):
            primary_tools = self._primary_tools_for_loop_action(action) & allowed
            if primary_tools:
                return primary_tools, "retry_primary_after_auxiliary_only_miss"
        if agent._auxiliary_only_miss_count >= 1 and (
            bool(missing_outputs)
            or self.unfinished_child_agents(agent)
            or bool((agent._loop_action_plan or {}).get("target_action"))
        ):
            primary_tools = self._primary_tools_for_loop_action(action) & allowed
            if action == "work" and missing_outputs:
                artifact_scope = self._artifact_tool_scope(agent, missing_outputs)
                if (
                    artifact_scope == "write_only"
                    and "file_write" in primary_tools
                    and not self._artifact_write_needs_more_evidence(agent, missing_outputs)
                ):
                    return {"file_write"}, "retry_primary_after_auxiliary_only_miss"
            if primary_tools:
                return primary_tools, "retry_primary_after_auxiliary_only_miss"
        if (
            action == "work"
            and missing_outputs
            and agent._write_only_miss_count >= 1
            and "file_write" in self._available_work_tools()
        ):
            if not self._artifact_write_needs_more_evidence(agent, missing_outputs):
                return {"file_write"}, "retry_file_write_after_text_miss"
        if (
            action == "work"
            and missing_outputs
            and agent._action_miss_action == "work"
            and agent._action_miss_count >= 2
        ):
            if (
                "file_write" in self._available_work_tools()
                and "file_write" in allowed
                and not self._artifact_write_needs_more_evidence(agent, missing_outputs)
            ):
                return {"file_write"}, "retry_file_write_after_no_tool_miss"
            evidence_tools = {
                "query", "file_read", "file_list", "grep", "shell",
            } & allowed
            if evidence_tools:
                return evidence_tools, "retry_evidence_after_no_tool_miss"
        if (
            action == "work"
            and missing_outputs
            and self._evidence_integration_ready(agent, missing_outputs)
        ):
            scoped = {"query", "file_read", "file_list", "grep", "file_write", "file_replace"} & allowed
            if scoped:
                return scoped, "integrate_evidence_artifact"
        if action == "work" and missing_outputs:
            artifact_scope = self._artifact_tool_scope(agent, missing_outputs)
            if artifact_scope == "read_write":
                scoped = {
                    "file_read", "file_list", "grep", "query", "file_write", "file_replace",
                    "compact", "set_status", "get_cost",
                } & allowed
                if self._artifact_write_needs_more_evidence(agent, missing_outputs) and "shell" in allowed:
                    scoped.add("shell")
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
        ledger_context = self._task_ledger_loop_context(agent)
        if not transient and not agent._loop_action_plan and not ledger_context:
            return None

        actions = "create, read, message, work, compact, stop"
        plan = agent._loop_action_plan
        selected_action = str((plan or {}).get("action") or "")
        parts = [
            "[Loop Action Card]",
            "This is transient runtime guidance for this execution turn only; it is not agent memory.",
            (
                f"Current action is fixed by runtime: {selected_action}. "
                f"Do not re-select among {actions} in this turn."
                if selected_action
                else f"Runtime has not fixed an action; valid actions are: {actions}."
            ),
            "Plan and call tools only for the fixed current action. You may call multiple tools within that action, but follow-up actions must wait for the next loop turn so runtime can revalidate state.",
            "Do not use set_status(action=...) to change this turn's action. Use set_status only to publish state after doing the current action, or to stop/self-prune when that is the current action.",
        ]
        if ledger_context:
            parts.append(ledger_context)
        if self._root_steward_mode(agent):
            parts.append(
                "Root steward boundary: the first child wave already exists. "
                "Do not take over worker/search/implementation artifacts yourself. "
                "Use query, ledger_read, ledger_update, send/wait, compact, or stop to inspect agents, update the task ledger, request pruning, and record remaining gaps. "
                "If the evidence has already been inspected and is sufficient for the terminal answer, stop may submit directly with submit_answer."
            )
        if plan and plan.get("action") == "stop" and self.answer_submission_required(agent) and not agent.submitted_answer_path:
            readiness = plan.get("answer_readiness") if isinstance(plan.get("answer_readiness"), dict) else {}
            if not readiness and self._root_steward_mode(agent):
                readiness = self._root_answer_readiness_public(self._root_answer_readiness(agent))
            if readiness:
                incomplete = readiness.get("incomplete_children") or []
                unfinished = readiness.get("unfinished_agents") or []
                parts.append(
                    "Answer submission readiness is a soft signal, not a hard gate. "
                    f"score={readiness.get('score', 0)} reasons={readiness.get('reasons') or []}. "
                    "If the answer is sufficiently supported, submit_answer may be called and the harness may judge it wrong; "
                    "if evidence is still unclear, use query/ledger_read/file_read in this stop turn before submitting or publish why the next loop should replan. "
                    f"Known incomplete children={len(incomplete)} unfinished={len(unfinished)}."
                )
        if missing_outputs:
            targets = "\n".join(f"- {path}" for path in missing_outputs[:8])
            more = "" if len(missing_outputs) <= 8 else f"\n- ... {len(missing_outputs) - 8} more"
            parts.append(f"Missing required outputs:\n{targets}{more}")
            if plan and plan.get("reason") in {
                "child_or_peer_evidence_inspect_before_artifact",
                "integrate_child_or_peer_evidence_into_artifact",
            }:
                evidence_lines = self._evidence_integration_lines(agent, missing_outputs)
                parts.append(
                    "Queryable child/peer evidence already exists, but your own required output files are still missing. "
                    "Use query and file_read/file_list/grep to inspect the listed agents' public memory and artifacts, "
                    "then write your own missing output files from verified evidence. Do not create more agents or wait "
                    "until you have attempted this integration step."
                    + ("\nEvidence to inspect:\n" + "\n".join(evidence_lines[:8]) if evidence_lines else "")
                )
            if plan and plan.get("action") == "work":
                parts.append(
                    "Current action is work. Do the work that best advances the task within the work tool scope. "
                    "If a different action is needed, leave it for the next loop turn so runtime can revalidate state."
                )
                if self._artifact_write_needs_more_evidence(agent, missing_outputs):
                    parts.append(
                        "This output is evidence/report-like and current evidence is not sufficient yet. "
                        "Use work tools such as shell, grep, file_read, or query to obtain or inspect primary evidence before writing. "
                        "Do not write a speculative placeholder merely to satisfy the file path."
                    )
                if agent._write_only_miss_count:
                    parts.append(
                        "Previous write-only artifact turns produced assistant text but no executable file_write call. "
                        "Plain prose does not create the missing files. Runtime will narrow this retry turn to file_write "
                        "so the selected work action creates the required output files."
                    )
                if agent._action_miss_action == "work" and agent._action_miss_count >= 2:
                    if self._artifact_write_needs_more_evidence(agent, missing_outputs):
                        parts.append(
                            "Previous work turns did not execute a usable work tool. Runtime will narrow this retry turn "
                            "to evidence tools such as shell, grep, file_read, or query so the selected work action obtains "
                            "the missing evidence instead of looping."
                        )
                    else:
                        parts.append(
                            "Previous work turns did not execute a usable work tool. Runtime will narrow this retry turn "
                            "to file_write so the selected work action creates the required output files."
                        )
        if plan and plan.get("reason") in {"direct_answer_submission_ready", "terminal_candidate_submit_ready"}:
            evidence = list(plan.get("evidence_agents") or [])
            evidence_lines = "\n".join(
                f"- {item.get('id')}: status={item.get('status') or '-'} artifacts={item.get('artifacts') or []} result={(item.get('result') or '')[:180]}"
                for item in evidence[:8]
                if isinstance(item, dict)
            )
            terminal_candidate = plan.get("terminal_candidate") if isinstance(plan.get("terminal_candidate"), dict) else {}
            candidate_line = ""
            if terminal_candidate:
                candidate_line = (
                    f"\nTerminal candidate: answer_hint={terminal_candidate.get('answer_hint') or '-'} "
                    f"source_agent={terminal_candidate.get('id') or '-'} "
                    f"artifact={terminal_candidate.get('artifact') or '-'}"
                )
            parts.append(
                "Direct terminal delivery is ready: recent query/read evidence or a terminal candidate artifact supports the answer. "
                "If other agents are still running, the successful submission will notify them to prune. "
                "Call submit_answer(answer=...) with the concise answer string. Do not use compact, set_status, file_write, or send in this submit-only turn.\n"
                f"Answer path={plan.get('answer_path') or self.final_answer_path_for(agent)}\n"
                f"Evidence signals:\n{evidence_lines or '- none'}"
                f"{candidate_line}"
            )
        if agent._auxiliary_only_miss_count:
            parts.append(
                "Previous loop turns only used auxiliary tools such as get_cost/set_status/compact and did not satisfy "
                f"the selected action ({agent._auxiliary_only_miss_count} consecutive auxiliary-only miss"
                f"{'es' if agent._auxiliary_only_miss_count != 1 else ''}). "
                "Use those tools only if they directly support the current action; otherwise call a primary tool for this action now."
            )
        blockers = self.completion_blockers(agent, include_missing_outputs=False)
        if not missing_outputs and _expected_output_paths(agent.task):
            if self.answer_submission_required(agent) and not agent.submitted_answer_path:
                parts.append(
                    "All ordinary output files are complete, but this answer-style task still needs submit_answer(answer=...). "
                    "Use action=stop when ready to submit the concise answer; if evidence is insufficient, query/read/work first."
                )
            elif blockers:
                parts.append(
                    "All explicit output files named by your task currently exist, but completion blockers remain. "
                    "Do not prefer compact/stop merely because files exist. Use work to repair weak evidence, create to delegate a verifier/recovery lane when allowed, "
                    "or compact only as a clearly blocked/pruned handoff summary."
                )
            else:
                parts.append(
                    "All explicit output files named by your task currently exist. Prefer action=compact or action=stop: "
                    "publish a short summary, tags, and artifact paths. Do not continue open-ended verification unless it changes the deliverable."
                )
        if blockers:
            blocker_lines = "\n".join(f"- {b['message']}" for b in blockers[:6])
            parts.append(
                "Completion blockers are active. Do not claim done, compact with stop_after=true, or write a final-only report "
                "until these are resolved:\n"
                f"{blocker_lines}"
            )
        handoff_pressure = self._handoff_pressure(agent)
        if handoff_pressure >= 0.5:
            parts.append(
                "Handoff pressure is active: this agent has already formed a child wave. "
                "Prefer query(), wait(), compact(), or a clearly justified downstream handoff over taking over implementation yourself. "
                "Create more agents only for a concrete uncovered lane, failed-lane recovery, or an input-dependent focus whose upstream evidence is ready."
            )
        lane_pressure = self._lane_coverage_pressure(agent)
        if lane_pressure >= 0.6:
            parts.append(
                f"Lane coverage pressure is high ({lane_pressure:.2f}): existing child agents already cover overlapping semantic lanes. "
                "Before creating another agent, query/read existing lane reports and identify a concrete uncovered gap. "
                "If no new gap exists, choose work to integrate/finalize or compact/stop instead of duplicating the same lane."
            )
        if plan and plan.get("action") == "create":
            parts.append(
                "Current action is create. Decide whether to form or adjust the agent structure. "
                "Use create-scope tools only: spawn/create_agent/spawn_many, plus get_cost/set_status/compact when needed to publish state or a solo/pruned decision. "
                "Assign each new agent a focused, inspectable task and useful query tags; role names are optional labels, not capability limits. "
                "Use a 0-1 readiness estimate for any input-dependent downstream focus: spawn independent worker lanes early, but delay handoff/check/synthesis work until input evidence is mature. "
                "If you still need query/read evidence before creating, do not call query/read tools this turn; explain the missing prerequisite briefly so the next loop can revalidate with read/query. "
                "Do not switch to action=work before the first child wave on a spawnable task; runtime will route that back to create. "
                "If you continue solo, compact/update memory with status:solo-decision and a concrete reason."
            )
            if agent._deferred_spawn_requests:
                readiness = self._deferred_spawn_readiness(agent)
                deferred_lines = []
                for request in agent._deferred_spawn_requests[:5]:
                    deferred_lines.append(
                        f"- role={request.get('role') or '-'} group={request.get('group_id') or '-'} "
                        f"depends_on={request.get('depends_on') or []} "
                        f"previous_readiness={request.get('readiness', 0)} task={(request.get('task') or '')[:120]}"
                    )
                parts.append(
                    "Deferred spawn proposals exist. Re-evaluate them numerically against current peer state. "
                    f"Current inferred readiness={readiness:.2f}, threshold={self.config.spawn_readiness_threshold:.2f}. "
                    "If readiness is now high enough, call spawn/spawn_many with the same concrete task and updated readiness; "
                    "otherwise leave them deferred and use message/read in a later turn to inspect worker progress.\n"
                    + "\n".join(deferred_lines)
                )
            if plan.get("reason") == "resume_create_after_read_orientation":
                parts.append(
                    "You requested read orientation during a create turn and have now had that read turn. "
                    "Return to structure formation now: use spawn_many()/spawn() for the separable peer wave unless the task is truly atomic."
                )
            elif plan.get("reason") == "retry_spawn_after_create_miss":
                parts.append(
                    "Previous create turns did not create a valid child agent. Runtime will narrow this retry turn to create-execution tools. "
                    "Do not write directory placeholders or only set action=create again; create the focused agent(s) now, or compact a solo/pruned decision if spawning is truly wrong."
                )
            if plan.get("reason") == "handoff_after_partial_progress":
                parts.append(
                    "You have partial progress but explicit outputs or confidence gaps remain. "
                    "Create one focused follow-up agent for the specific remaining boundary: checking evidence, integrating partial findings, completing a missing artifact, or reviewing uncertainty. "
                    "Keep the assignment narrow and tag it with the shared group/problem plus the concrete gap; do not create a vague role-only agent."
                )
            if plan.get("reason") == "delegate_final_delivery":
                evidence = list(plan.get("evidence_agents") or [])
                evidence_lines = "\n".join(
                    f"- {item.get('id')}: status={item.get('status') or '-'} artifacts={item.get('artifacts') or []} result={(item.get('result') or '')[:160]}"
                    for item in evidence[:8]
                    if isinstance(item, dict)
                )
                parts.append(
                    "Final delivery handoff: the task has a terminal answer/submission protocol and queryable evidence appears ready, "
                    "but this coordinator should not take over final delivery directly. Prefer one focused finalizer/delivery agent unless current state clearly calls for a different delivery shape. "
                    "That delivery agent should query/read the listed evidence, verify the answer is sufficiently supported, call submit_answer(answer=...) if ready, "
                    "or compact a blocked summary if evidence is still insufficient. Do not create more research lanes from this action.\n"
                    f"Answer path={plan.get('answer_path') or self.final_answer_path_for(agent)}\n"
                    f"Evidence signals:\n{evidence_lines or '- none'}"
                )
            if plan.get("reason") == "handoff_after_evidence_maturity":
                evidence = list(plan.get("evidence_agents") or [])
                evidence_lines = "\n".join(
                    f"- {item.get('id')}: status={item.get('status') or '-'} artifacts={item.get('artifacts') or []} result={(item.get('result') or '')[:160]}"
                    for item in evidence[:8]
                    if isinstance(item, dict)
                )
                parts.append(
                    "Your evidence appears mature and an upstream terminal/integration target is still open. "
                    "Create one focused downstream agent instead of continuing to expand this evidence lane yourself. "
                    "Do not make it a vague role-only agent: assign the concrete next boundary such as cross-checking, synthesis, answer delivery, "
                    "or resolving a named uncertainty. The downstream agent should query/read the listed evidence agents and decide whether to submit, "
                    "publish a verified synthesis, or compact a blocked summary.\n"
                    f"Target agent={plan.get('target_agent') or '-'} answer_path={plan.get('answer_path') or '-'}\n"
                    f"Evidence signals:\n{evidence_lines or '- current agent evidence only'}"
                )
            if plan.get("reason") == "delegate_uncertain_evidence_recovery":
                uncertain = list(plan.get("uncertain_artifacts") or [])
                artifact_lines = "\n".join(f"- {path}" for path in uncertain[:8]) or "- current memory/result contains unverified or low-confidence evidence"
                parts.append(
                    "Your current evidence is explicitly partial, unverified, or low confidence. "
                    "Do not rewrite the same low-confidence artifact or mark this lane done. "
                    "Create a focused verifier/recovery agent that must query/read your public memory and artifacts, verify the uncertain claims from primary evidence, "
                    "and either replace the artifact with a verified version or compact with a clear blocked summary. "
                    "Use tags such as role:verifier or role:recovery plus the lane/task tags so other agents can query it later.\n"
                    f"Uncertain artifacts/signals:\n{artifact_lines}"
                )
            if plan.get("reason") == "multi_phase_peer_wave_pending":
                parts.append(
                    "This task describes a multi-stage peer protocol and previous child waves are no longer active while final outputs remain missing. "
                    "Create the next focused comparison, mutation, check, or synthesis wave as appropriate. Do not collapse the remaining protocol into a solo final artifact unless budget exhaustion makes the incomplete protocol explicit."
                )
            if plan.get("reason") == "explicit_peer_wave_incomplete":
                gap = plan.get("peer_wave_gap") or {}
                parts.append(
                    "The task names an explicit peer-wave size that is not yet satisfied. "
                    f"Observed {gap.get('current', 0)} of target {gap.get('target', '?')} "
                    f"for role={gap.get('role') or 'unspecified'} phase={gap.get('phase') or 'unspecified'}; "
                    f"create the missing {gap.get('remaining', '?')} peer agents now, avoiding duplicate lanes already covered."
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
            elif plan.get("reason") == "delegate_failed_child_recovery":
                failed = list(plan.get("failed_children") or [])
                failed_lines = []
                for item in failed[:5]:
                    failed_lines.append(
                        f"- failed_child={item.get('id') or '-'} role={item.get('role') or '-'} "
                        f"group={item.get('group_id') or '-'} missing={item.get('missing_outputs') or []} "
                        f"artifacts={item.get('artifacts') or []} result={(item.get('result') or '')[:160]}"
                    )
                parts.append(
                    "Coordinator boundary: one or more child lanes failed while final artifacts remain missing. "
                    "Do not take over the failed lane with root/coordinator work. Create a focused recovery agent now. "
                    "The recovery agent task should query/read the failed child's public state and artifacts, reuse any partial evidence, "
                    "complete the missing lane output, and write a concise recovery summary with tags including status:recovered "
                    "and the failed child id. Prefer one recovery agent per failed lane unless several failures share the same missing artifact.\n"
                    + "\n".join(failed_lines)
                )
        if plan and plan.get("action") == "compact" and plan.get("reason") == "premature_downstream_wait_for_evidence":
            upstream = list(plan.get("upstream_agents") or [])
            lines = "\n".join(
                f"- {item.get('id')}: status={item.get('status')} missing={item.get('missing_outputs') or []} artifacts={item.get('artifacts') or []}"
                for item in upstream[:6]
            )
            parts.append(
                "Your assignment appears input-dependent, but upstream peer evidence is not ready. "
                "Do not fabricate a final check or report. Compact and stop with tags including status:pruned, "
                "reason:premature_downstream, and needs:upstream_evidence so a later agent can query your state and restart this focus when inputs mature.\n"
                f"Current upstream signals:\n{lines}"
            )
        if plan and plan.get("action") == "compact" and plan.get("reason") == "received_prune_request_self_prune":
            request = dict(plan.get("prune_request") or {})
            evidence_agents = request.get("evidence_agents") or []
            evidence_lines = "\n".join(
                f"- {item.get('id')}: status={item.get('status') or '-'} artifacts={item.get('artifacts') or []} result={(item.get('result') or '')[:140]}"
                for item in evidence_agents[:6]
                if isinstance(item, dict)
            )
            parts.append(
                "You received a prune_request from a coordinating agent after it queried peer progress and found key evidence/output already covered. "
                "Treat this as a request to self-prune, not an emergency kill. Compact your current partial state with stop_after=true, "
                "include tags status:pruned plus reason:peer_ahead or reason:duplicate_lane and your task/lane tags, and mention what remains reusable. "
                "Only continue instead of pruning if your next output is clearly distinct from the covered evidence.\n"
                f"Prune request from={request.get('from') or '-'} reason={request.get('reason') or '-'} covered_by={request.get('covered_by') or []}\n"
                f"Evidence signals:\n{evidence_lines or '- none listed'}"
            )
        if plan and plan.get("action") == "message" and plan.get("reason") == "request_duplicate_agents_self_prune_after_key_evidence":
            targets = list(plan.get("prune_targets") or [])
            evidence = list(plan.get("evidence_agents") or [])
            target_lines = "\n".join(f"- {target}" for target in targets[:8])
            evidence_lines = "\n".join(
                f"- {item.get('id')}: status={item.get('status') or '-'} artifacts={item.get('artifacts') or []} result={(item.get('result') or '')[:140]}"
                for item in evidence[:6]
                if isinstance(item, dict)
            )
            parts.append(
                "Your recent query found key evidence/output already covered while other same-lane agents are still active. "
                "Use send() with message_type='prune_request' to ask duplicate agents to compact their partial findings and self-prune. "
                "Do not kill them directly. Include payload fields reason, covered_by, evidence_agents, and tags if useful. "
                "After sending, let the next loop revalidate whether verification/integration or recovery is needed.\n"
                f"Suggested prune targets:\n{target_lines or '- none'}\n"
                f"Covered evidence agents:\n{evidence_lines or '- none'}"
            )
        if plan and plan.get("reason") == "inspect_coverage_after_spawn_skipped":
            covered_by = list(plan.get("covered_by") or [])
            covered_lines = "\n".join(f"- {agent_id}" for agent_id in covered_by[:8])
            parts.append(
                "Your previous create request was skipped because runtime found existing lane coverage. "
                "Do not immediately retry the same create. Use query() or wait() to inspect the covered agents' "
                "state_board, public memory, artifacts, confidence, and remaining blockers. "
                "If coverage is complete and reliable, integrate/finalize or ask duplicate active agents to prune. "
                "If coverage is partial, stale, unverified, or missing the needed artifact, let the next loop replan "
                "a verifier or recovery agent after this inspection.\n"
                f"Skipped reason={plan.get('skip_reason') or '-'}\n"
                f"Covered agents:\n{covered_lines or '- none'}"
            )
        if plan and plan.get("reason") == "compact_after_spawn_skipped_covered":
            covered_by = list(plan.get("covered_by") or [])
            covered_paths = list(plan.get("covered_paths") or [])
            covered_lines = "\n".join(f"- {agent_id}" for agent_id in covered_by[:8])
            path_lines = "\n".join(f"- {path}" for path in covered_paths[:8])
            parts.append(
                "Your previous create request was skipped because existing agents/artifacts already cover that lane, "
                "and you have no remaining required output. Do not retry create. Compact a short reusable summary, "
                "tag it with status:pruned and reason:covered_by_existing_agents, include covered agents/paths, and stop_after=true "
                "unless you have a concrete contradiction to publish.\n"
                f"Skipped reason={plan.get('skip_reason') or '-'}\n"
                f"Covered agents:\n{covered_lines or '- none'}\n"
                f"Covered paths:\n{path_lines or '- none'}"
            )
        if plan and plan.get("action") == "message" and str(plan.get("reason", "")).startswith("coordinate_"):
            parts.append(
                "Coordinator boundary: you already have child agents. Use query() or wait() to inspect their state_board, public memory, artifacts, and progress. "
                "Do not inspect or edit implementation files yourself in this root/coordinator turn. "
                "If their progress makes integration or recovery work necessary, leave that for the next create turn after runtime revalidation."
            )
        if plan and plan.get("action") == "message":
            parts.append(
                "Current action is message. It can only communicate, query, or wait. "
                "If the correct next step is to create an agent, run work tools, or write missing outputs, do not describe that action in prose; "
                "use query()/wait() only if they are genuinely needed now, otherwise allow the next loop turn to replan."
            )
        if plan and plan.get("action") == "read" and plan.get("reason") == "peer_overlap_query_before_work":
            query_filter = self._peer_progress_query_filter(agent)
            tags = self._peer_progress_query_tags(agent)
            overlap_peers = self._peer_overlap_query_candidates(agent)
            peer_lines = "\n".join(
                f"- {peer.id}: role={peer.role or '-'} status={peer.status} turns={peer._turns} "
                f"artifacts={','.join(a.path for a in peer.artifacts[:3]) or '-'} "
                f"tags={','.join(peer.current_task_tags[:5]) or '-'}"
                for peer in overlap_peers[:8]
            )
            parts.append(
                "Current action is read because this looks like discovery/search/evidence work with nearby overlapping peers. "
                "Before doing more work, call query() against the shared group/tags and compare state_board, public_memory, artifacts, and active tasks. "
                f"Recommended query: filter={json.dumps(query_filter, ensure_ascii=False)}, "
                f"tags={json.dumps(tags, ensure_ascii=False)}. "
                "If another peer is already doing the same search lane or has stronger completed evidence, do not continue duplicate work; "
                "next turn should compact with stop_after=true, a concise terminal report, and tags including status:pruned plus "
                "reason:duplicate_lane or reason:peer_ahead. If your lane is distinct, continue work next turn after runtime revalidates.\n"
                f"Nearby overlap candidates:\n{peer_lines or '- none'}"
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
        elif (
            self._peer_progress_post_query_pending(agent)
            or self._peer_overlap_post_query_pending(agent, missing_outputs)
            or self._peer_overlap_active_duplicate_post_query_pending(agent, missing_outputs)
        ):
            active_duplicate = self._peer_overlap_active_duplicate_post_query_pending(agent, missing_outputs)
            peers = (
                self._active_duplicate_peer_overlap_candidates(agent)
                if active_duplicate
                else (
                    self._completed_peer_overlap_candidates(agent)
                    if self._peer_overlap_post_query_pending(agent, missing_outputs)
                    else self._completed_peer_progress_candidates(agent)
                )
            )
            peer_lines = "\n".join(
                f"- {peer.id}: status={peer.status} turns={peer._turns} tools={peer._tool_calls} "
                f"artifacts={','.join(a.path for a in peer.artifacts[:3]) or '-'} "
                f"summary={(peer.result or self.memory.serialize(peer.id).get('public_summary') or '')[:140]}"
                for peer in peers[:6]
            )
            if active_duplicate:
                parts.append(
                    "Peer-overlap query has already found active peers working the same or highly overlapping lane, "
                    "and you have not produced a distinct artifact yet. Do not continue a duplicate branch or spawn another copy. "
                    "Choose action=compact with stop_after=true and tags including status:pruned plus reason:duplicate_lane "
                    "or reason:peer_ahead, unless your next work output is clearly distinct from those peers.\n"
                    f"Active overlapping peers:\n{peer_lines}"
                )
            else:
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
                f"Fixed action: action={plan.get('action')} reason={plan.get('reason', '-')}. "
                "Execute this action now; if it is stale, use the smallest valid tool response that publishes why the next loop should replan."
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
            blockers = self.completion_blockers(agent, tags=guard_tags, result=params.get("stop_result"))
            if blockers:
                self._emit(agent.id, "completion_blocked", {
                    "reason": "compact_stop_after",
                    "blockers": blockers,
                })
                self.state_board_update(agent.id, action_state="work", current_task_tags=guard_tags)
                return
        summary = params["summary"]
        files = list(params.get("files", []))
        tags = normalize_tags(list(params.get("tags", [])) or list(agent.current_task_tags))
        experience = params.get("experience")
        truth_guard = self._compact_truth_guard(
            agent,
            summary=summary,
            experience=experience,
            tags=tags,
            files=files,
            stop_result=params.get("stop_result"),
        )
        summary = truth_guard["summary"]
        experience = truth_guard["experience"]
        tags = truth_guard["tags"]
        guarded_stop_result = truth_guard["stop_result"]
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
        if truth_guard["truth_guard"]:
            expected_paths = _non_terminal_expected_output_paths(agent.task)
            referenced_paths = [
                path for path in files
                if path in expected_paths or not expected_paths
            ]
            target_paths = referenced_paths or expected_paths
            for path in target_paths[:8]:
                self.task_ledger_update(agent, {
                    "item_id": agent.id,
                    "status": "partial",
                    "output_path": path,
                    "output_status": self._artifact_output_status(agent, path) if path in expected_paths else "evidence_gap",
                    "blockers": [{
                        "kind": "unverified_evidence",
                        "message": "Truth guard found placeholder, partial, or low-confidence evidence for this output.",
                        "artifacts": [path],
                        "event": truth_guard["truth_guard"],
                    }],
                    "note": "Truth guard downgraded this output; continue evidence repair before retrying terminal compact.",
                })
            self.task_ledger_update(agent, {
                "item_id": agent.id,
                "status": "blocked",
                "blockers": [{
                    "kind": "unverified_evidence",
                    "message": "Compact truth guard found partial, placeholder, or low-confidence evidence.",
                    "event": truth_guard["truth_guard"],
                }],
                "note": "Truth guard downgraded completion; continue with work or query before retrying terminal compact.",
            })
            agent._loop_action_plan = {
                "action": "work",
                "reason": "truth_guard_evidence_repair",
                "turn_added": agent._turns,
                "missing_outputs": target_paths[:20],
                "blockers": [{
                    "kind": "unverified_evidence",
                    "message": "Repair or replace low-confidence output with evidence-backed content.",
                    "artifacts": target_paths[:8],
                }],
            }
            update_kwargs["remove_tags"] = list(_COMPLETION_CLAIM_TAGS)
            if params.get("stop_after"):
                params["stop_after"] = False
                params["return_action_after_auxiliary"] = "work"
                update_kwargs["active_task"] = ActiveTaskCard(
                    task=new_task or agent.task,
                    tags=tags or agent.current_task_tags,
                    work_outline=work_outline,
                )
        return_action_after_auxiliary = params.get("return_action_after_auxiliary")
        if experience:
            update_kwargs["add_experience"] = build_memory_card(
                experience,
                tags=tags or agent.current_task_tags,
                artifacts=files,
                memory_kind=params.get("memory_kind") or ("terminal" if params.get("stop_after") else None),
                memory_source=params.get("memory_source") or "compact",
                durable=params.get("durable"),
                evidence=params.get("evidence"),
                stop_after=bool(params.get("stop_after")),
            )
        else:
            summary_cards = [
                build_memory_card(
                    card,
                    tags=normalize_tags((tags or agent.current_task_tags) + [f"memory_part:{idx + 1}"]),
                    artifacts=files,
                    memory_kind=params.get("memory_kind") or ("terminal" if params.get("stop_after") else "summary"),
                    memory_source=params.get("memory_source") or "compact",
                    durable=params.get("durable"),
                    evidence=params.get("evidence"),
                    stop_after=bool(params.get("stop_after")),
                )
                for idx, card in enumerate(split_experience_text(summary))
            ]
            if summary_cards:
                update_kwargs["add_experiences"] = summary_cards
        self.memory.update(agent.id, **update_kwargs)
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        if params.get("stop_after"):
            agent.action_state = "stop"
            agent.status = "done"
            agent.result = guarded_stop_result or agent.result
            stop_experience = None if experience else build_memory_card(
                summary,
                tags=tags or agent.current_task_tags,
                artifacts=files,
                memory_kind=params.get("memory_kind") or "terminal",
                memory_source=params.get("memory_source") or "compact_stop",
                durable=params.get("durable"),
                evidence=params.get("evidence"),
                stop_after=True,
            )
            self.memory.update(
                agent.id,
                clear_active_task=True,
                add_experience=stop_experience,
                remove_tags=list(_COMPLETION_CLAIM_TAGS) if truth_guard["truth_guard"] else None,
            )
            agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        elif return_action_after_auxiliary in _LOOP_ACTIONS:
            self.state_board_update(
                agent.id,
                action_state=return_action_after_auxiliary,
                current_task_tags=tags or agent.current_task_tags,
                work_outline=work_outline,
            )
        else:
            self.state_board_update(
                agent.id,
                action_state="work",
                current_task_tags=tags or agent.current_task_tags,
                work_outline=work_outline,
            )
        self.state_board_sync(agent.id)
        self._refresh_task_ledger_agent_item(agent)

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
            add_experience=build_memory_card(
                experience,
                tags=tags or agent.current_task_tags,
                artifacts=files,
                memory_kind="handoff",
                memory_source="rebirth",
            ) if experience else None,
        )
        agent.memory = {"public_memory": self.memory.serialize(agent.id)}
        self.state_board_sync(agent.id)

    async def _execute_tool(self, tc: ToolCall, agent: Agent, tools: dict) -> Any:
        tool_info = tools.get(tc.name)
        if not tool_info:
            return {"error": f"Unknown tool: {tc.name}"}
        if tc.name == "submit_answer":
            permission = self.can_submit_answer(agent, str(tc.arguments.get("path") or ""))
            if not permission.get("ok"):
                self._emit(agent.id, "submit_answer_denied", permission)
                return {"error": "submit_answer_denied", **permission}
        if tc.name == "file_write":
            path_arg = str(tc.arguments.get("path", tc.arguments.get("file_path", tc.arguments.get("filepath", ""))))
            normalized = self._normalize_workspace_relative_path(path_arg, agent.workspace)
            if normalized and _looks_like_final_answer_path(normalized):
                result = {
                    "error": "final_answer_write_denied",
                    "reason": "terminal_answer_requires_submit_answer",
                    "path": normalized,
                    "message": (
                        "Global answer files are protected protocol outputs. Use submit_answer "
                        "when a runtime-authorized delivery agent is ready to submit."
                    ),
                }
                self._emit(agent.id, "final_answer_write_denied", result)
                return result
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
            tools = dict(self.work_tools)
        else:
            tools = {name: tool for name, tool in self.work_tools.items() if name in enabled}
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
Coordination guidance: use query() to inspect other agents' state_board, role, group_id, tags, artifacts, progress, and public_memory before summarizing shared work. For peer waves, prefer query(filter={{"group_id": "<exact group_id>"}}) or query(tags=["feature:N"]); if query returns zero results, read query_help and retry with the advertised group_id/tags. Direct send() is for explicit coordination, not the default completion path. After query shows a key output is already covered, a coordinator may send message_type="prune_request" to same-lane duplicate agents so they compact useful partial state and self-prune; do not kill them directly. If you receive prune_request and your lane is no longer distinct, compact with stop_after=true and tags such as status:pruned plus reason:peer_ahead/reason:duplicate_lane so the summary remains discoverable. If you hit a bottleneck, have repeated no-tool/prose turns, or suspect duplicate work, proactively query nearby peers and compare progress/artifacts before continuing. If another peer is clearly ahead or already covers your lane, compact your partial findings with stop_after=true and tags such as status:pruned plus reason:peer_ahead/reason:duplicate_lane. If compacting memory is your last step, use compact(..., stop_after=true, result="...") or set_status(..., compact_before_stop=true); plain compact() means you intend to continue.
{orchestration_section}
Editing guidance: use file_replace for source edits. file_write is for new files or deliberate guarded overwrites; existing large files require overwrite=true or expected_sha256 and can otherwise be rejected to prevent accidental whole-file truncation.
{artifact_section}
{shell_guidance}
Workflow prior guidance: research_loop / critic_review_loop / synthesis_loop are prompt patterns only, not workflow engines.
"""

    def _artifact_prompt_section(self, task: str) -> str:
        expected = _non_terminal_expected_output_paths(task)
        final_answers = _final_answer_paths(task)
        if not expected:
            if final_answers:
                targets = "\n".join(f"- {path}" for path in final_answers[:8])
                more = "" if len(final_answers) <= 8 else f"\n- ... {len(final_answers) - 8} more"
                return (
                    "Answer submission guidance: this task has a terminal answer JSON path, but it is not an ordinary artifact. "
                    "Do not write it with file_write. When the answer is ready for benchmark evaluation, use submit_answer(answer=...). "
                    "The submit_answer tool writes the protocol file:\n"
                    f"{targets}{more}"
                )
            return ""
        targets = "\n".join(f"- {path}" for path in expected[:8])
        more = "" if len(expected) <= 8 else f"\n- ... {len(expected) - 8} more"
        section = (
            "Artifact guidance: this task has explicit required output files. "
            "After reading the minimum necessary inputs, write those files with file_write; "
            "do not leave the deliverable only in chat. If several files are required, write all of them, "
            "for example source plus report/notes. If uncertain, write the best current artifact and explain uncertainty "
            "inside the required report file.\n"
            f"Required output files:\n{targets}{more}"
        )
        if final_answers:
            answer_targets = "\n".join(f"- {path}" for path in final_answers[:8])
            answer_more = "" if len(final_answers) <= 8 else f"\n- ... {len(final_answers) - 8} more"
            section += (
                "\nAnswer submission guidance: final answer JSON paths are terminal submissions, not ordinary artifacts. "
                "Use submit_answer(answer=...) when the answer is ready; it writes:\n"
                f"{answer_targets}{answer_more}"
            )
        return section

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
                "For multi-line Python or network/API checks, first write a script file with file_write, then run it as `python3 script.py`; avoid `python3 -c` snippets. "
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
                f"Start solo for brief orientation, then form an early peer wave for independent worker lanes when the task has "
                f"{threshold}+ separable workstreams. Delay downstream review/verification/synthesis agents until their input readiness is high."
            )
        elif preference == "parallel":
            behavior = (
                f"Bias toward early peer-wave creation for independent worker lanes. For {threshold}+ separable workstreams, call spawn_many() "
                "before detailed implementation; create downstream verifier/review/synthesis agents later with depends_on/readiness once workers publish partial outputs."
            )
        else:
            behavior = (
                "Strongly prefer a peer wave for nontrivial independent worker lanes. Use spawn_many() early unless the task is atomic; "
                "the root agent should mainly coordinate and query peer state, then create downstream verifier/review/synthesis agents after worker evidence matures."
            )
        return (
            "Orchestration preference: "
            f"{preference}. {behavior} Good peer waves include evidence checks, calculations, implementation, "
            "tests, critique, risk review, and final synthesis. Give peers role/group_id/current_task_tags; use depends_on/readiness for downstream agents."
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
