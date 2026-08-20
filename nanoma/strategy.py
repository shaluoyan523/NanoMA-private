"""Runtime strategy primitives for tool-learning style coordination.

This module keeps policy labels and local decision rules separate from the
agent loop. The runtime owns observations and enforcement; strategies only map
state snapshots to structural tool decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


class StrategyAction:
    """Stable action labels for training/evaluation logs."""

    TOOL_OVERRIDE = "TOOL_OVERRIDE"
    SPAWN_PORTFOLIO = "SPAWN_PORTFOLIO"
    SELF_KILL = "SELF_KILL"
    QUERY_BEFORE_DELETE = "QUERY_BEFORE_DELETE"
    KILL_STALLED = "KILL_STALLED"
    KILL_REDUNDANT = "KILL_REDUNDANT"
    KILL_DUPLICATE_DEEPENING = "KILL_DUPLICATE_DEEPENING"
    KILL_QUERY_LOOP = "KILL_QUERY_LOOP"
    SPAWN_RECOVERY = "SPAWN_RECOVERY"
    QUERY_CANDIDATE_EVIDENCE = "QUERY_CANDIDATE_EVIDENCE"
    SPAWN_VERIFIER = "SPAWN_VERIFIER"
    FINAL_EVIDENCE_GATE = "FINAL_EVIDENCE_GATE"
    QUERY_VERIFICATION_EVIDENCE = "QUERY_VERIFICATION_EVIDENCE"
    FINALIZE = "FINALIZE"
    FINALIZE_CANDIDATE = "FINALIZE_CANDIDATE"
    IMPROVE_CANDIDATE = "IMPROVE_CANDIDATE"
    STOP_SPAWN = "STOP_SPAWN"


@dataclass(frozen=True)
class StrategyState:
    """A compact, serializable snapshot before an agent's next LLM call."""

    agent_id: str
    parent_id: str | None
    is_root: bool
    status: str
    turns: int
    tokens: int
    tool_calls: int
    no_tool_turns: int
    total_agents: int
    active_children: int
    shared_candidate_count: int
    has_local_candidate: bool
    query_calls: int
    kill_calls: int
    set_status_calls: int
    large_tool_results: int
    candidate_agent_count: int = 0
    candidate_query_coverage: int = 0
    shared_verification_count: int = 0
    verification_review_calls: int = 0
    best_shared_candidate_bytes: int | None = None
    best_shared_candidate_stable_observations: int = 0
    candidate_improve_attempts: int = 0
    candidate_improve_stalled: bool = False
    candidate_finalize_attempts: int = 0
    candidate_evidence_gate_attempts: int = 0
    root_recovery_spawn_attempts: int = 0
    last_finalized_candidate_bytes: int | None = None
    requested_auto_kills: int = 0
    already_auto_kill_requested: bool = False
    pending_override: bool = False
    resource_pressure: float = 0.0
    delivery_phase: str = "unknown"
    available_tools: tuple[str, ...] = ()

    def as_event(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "is_root": self.is_root,
            "status": self.status,
            "turns": self.turns,
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "no_tool_turns": self.no_tool_turns,
            "total_agents": self.total_agents,
            "active_children": self.active_children,
            "shared_candidate_count": self.shared_candidate_count,
            "candidate_agent_count": self.candidate_agent_count,
            "candidate_query_coverage": self.candidate_query_coverage,
            "shared_verification_count": self.shared_verification_count,
            "verification_review_calls": self.verification_review_calls,
            "best_shared_candidate_bytes": self.best_shared_candidate_bytes,
            "best_shared_candidate_stable_observations": self.best_shared_candidate_stable_observations,
            "has_local_candidate": self.has_local_candidate,
            "query_calls": self.query_calls,
            "kill_calls": self.kill_calls,
            "set_status_calls": self.set_status_calls,
            "large_tool_results": self.large_tool_results,
            "requested_auto_kills": self.requested_auto_kills,
            "candidate_improve_attempts": self.candidate_improve_attempts,
            "candidate_improve_stalled": self.candidate_improve_stalled,
            "candidate_finalize_attempts": self.candidate_finalize_attempts,
            "candidate_evidence_gate_attempts": self.candidate_evidence_gate_attempts,
            "root_recovery_spawn_attempts": self.root_recovery_spawn_attempts,
            "last_finalized_candidate_bytes": self.last_finalized_candidate_bytes,
            "already_auto_kill_requested": self.already_auto_kill_requested,
            "pending_override": self.pending_override,
            "resource_pressure": round(self.resource_pressure, 3),
            "delivery_phase": self.delivery_phase,
            "available_tools": list(self.available_tools),
        }


@dataclass(frozen=True)
class StrategyDecision:
    """A structural tool decision produced by a strategy module."""

    source: str
    action: str
    target_agent_id: str
    tools: tuple[str, ...]
    reason: str
    message: str = ""
    once: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_event(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "action": self.action,
            "target_agent_id": self.target_agent_id,
            "tools": list(self.tools),
            "reason": self.reason,
            "message": self.message[:500],
            "once": self.once,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LowValueAgentKillConfig:
    enabled: bool = False
    min_llm: int = 10
    max_llm_no_candidate: int = 14
    max_tokens_no_candidate: int = 160000
    max_llm_duplicate: int = 22
    max_agents: int = 3
    kill_only_tool: bool = False
    query_before_kill: bool = False


@dataclass(frozen=True)
class SpawnPortfolioConfig:
    enabled: bool = False
    min_root_turn: int = 0
    max_root_turn: int = 3
    min_total_agents: int = 1
    target_agents: int = 5
    require_no_children: bool = True
    require_no_candidate: bool = True


@dataclass(frozen=True)
class RootRecoverySpawnConfig:
    enabled: bool = False
    min_root_turns: int = 6
    target_agents: int = 4
    min_shared_candidates: int = 1
    max_attempts: int = 2
    require_no_active_children: bool = True


@dataclass(frozen=True)
class CandidateEvidenceGateConfig:
    enabled: bool = False
    min_root_turns: int = 4
    min_shared_candidates: int = 1
    required_verification_files: int = 1
    max_attempts: int = 2


@dataclass(frozen=True)
class FinalEvidenceGateConfig:
    enabled: bool = False
    min_root_turns: int = 4
    min_shared_candidates: int = 1
    min_candidate_agents: int = 1
    query_coverage_ratio: float = 1.0
    require_verification: bool = True
    require_verification_review: bool = True
    allow_spawn_verifier: bool = True
    max_verifier_spawns: int = 2
    trigger_phases: tuple[str, ...] = ("consolidate", "verify", "deliver")


@dataclass(frozen=True)
class CandidateFinalizeConfig:
    enabled: bool = False
    min_shared_candidates: int = 2
    finalize_after_turns: int = 8
    improve_after_turns: int = 6
    readable_bytes_threshold: int = 900
    max_improve_attempts: int = 2
    stop_improve_when_stalled: bool = True
    min_stable_observations: int = 0
    finalize_once_per_best: bool = False
    max_finalize_attempts: int = 0


class LowValueAgentKillStrategy:
    """Unsupervised delete-action rule for low-value non-root agents."""

    name = "low_value_agent_kill"

    def __init__(self, config: LowValueAgentKillConfig):
        self.config = config

    def decide(self, state: StrategyState) -> StrategyDecision | None:
        if not self.config.enabled:
            return None
        if state.is_root or state.status != "running":
            return None
        if state.pending_override or state.already_auto_kill_requested:
            return None
        if state.requested_auto_kills >= self.config.max_agents:
            return None
        if state.turns < self.config.min_llm:
            return None
        if state.set_status_calls or state.kill_calls:
            return None

        action = ""
        reason = ""
        if (
            not state.has_local_candidate
            and state.turns >= self.config.max_llm_no_candidate
            and state.tokens >= self.config.max_tokens_no_candidate
        ):
            action = StrategyAction.KILL_STALLED
            reason = (
                f"local policy inspection: {state.agent_id} used {state.turns} LLM turns / "
                f"{state.tokens} tokens without producing a candidate file."
            )
        elif (
            not state.has_local_candidate
            and state.shared_candidate_count >= 2
            and state.turns >= self.config.min_llm
        ):
            action = StrategyAction.KILL_REDUNDANT
            reason = (
                f"local policy inspection: {state.shared_candidate_count} shared candidates already exist; "
                f"{state.agent_id} has no candidate after {state.turns} LLM turns."
            )
        elif (
            state.has_local_candidate
            and state.shared_candidate_count >= 3
            and state.turns >= self.config.max_llm_duplicate
        ):
            action = StrategyAction.KILL_DUPLICATE_DEEPENING
            reason = (
                f"local policy inspection: {state.agent_id} has a candidate but continued for "
                f"{state.turns} LLM turns while {state.shared_candidate_count} shared candidates exist."
            )
        elif (
            state.query_calls >= 5
            and state.large_tool_results >= 2
            and state.shared_candidate_count >= 1
        ):
            action = StrategyAction.KILL_QUERY_LOOP
            reason = (
                f"local policy inspection: {state.agent_id} is spending context on query/large "
                "outputs after candidates exist."
            )

        if not action:
            return None

        if self.config.query_before_kill and state.query_calls <= 0:
            query_reason = (
                f"{reason} Query coordination state before deleting this branch so the "
                "delete action is based on current sibling/candidate evidence."
            )
            message = (
                "[Runtime strategy: query before delete]\n"
                f"{query_reason}\n"
                "Call query now. Inspect current agents/candidates only; do not continue "
                "local implementation on this turn. If the branch still has no unique "
                "value after query evidence, the next delete action may self-kill."
            )
            return StrategyDecision(
                source=self.name,
                action=StrategyAction.QUERY_BEFORE_DELETE,
                target_agent_id=state.agent_id,
                tools=("query", "get_cost"),
                reason=query_reason,
                message=message,
                once=True,
                metadata={
                    "tool_learning": {
                        "tool_family": "query",
                        "tool_name": "query",
                        "next_tool_family": "delete",
                        "next_tool_name": "kill",
                    },
                    "state": state.as_event(),
                },
            )

        message = (
            "Unsupervised runtime tool strategy triggered after query-equivalent "
            f"local inspection. {reason} Call kill(agent_id='{state.agent_id}') now."
        )
        tools = ("kill",) if self.config.kill_only_tool else ("kill", "get_cost")
        return StrategyDecision(
            source=self.name,
            action=action,
            target_agent_id=state.agent_id,
            tools=tools,
            reason=reason,
            message=message,
            once=False,
            metadata={
                "tool_learning": {
                    "tool_family": "delete",
                    "tool_name": "kill",
                    "requires_query_evidence": True,
                },
                "state": state.as_event(),
            },
        )


class SpawnPortfolioStrategy:
    """Unsupervised add-action rule for early complementary branching.

    The strategy does not hard-code a named agent per feature. It asks the
    root to create a compact portfolio of role slots that cover independent
    solution attempts, reproduction/localization, verification, and final
    integration. The model still decides the concrete subtask wording for the
    current problem.
    """

    name = "spawn_portfolio"

    def __init__(self, config: SpawnPortfolioConfig):
        self.config = config

    def decide(self, state: StrategyState) -> StrategyDecision | None:
        if not self.config.enabled:
            return None
        if not state.is_root or state.status != "running":
            return None
        if state.pending_override:
            return None
        if state.turns < self.config.min_root_turn or state.turns > self.config.max_root_turn:
            return None
        if state.total_agents < self.config.min_total_agents:
            return None
        if self.config.require_no_children and state.active_children > 0:
            return None
        if self.config.require_no_candidate and state.shared_candidate_count > 0:
            return None
        if state.total_agents >= self.config.target_agents:
            return None

        missing = max(1, self.config.target_agents - state.total_agents)
        reason = (
            f"early root has {state.total_agents} total agents, {state.active_children} active "
            f"children, and {state.shared_candidate_count} shared candidates; open a "
            f"{missing}-slot complementary portfolio before local deepening."
        )
        message = (
            "[Runtime strategy: spawn portfolio]\n"
            f"{reason}\n"
            "Call spawn now before doing more solo work. Create short-running complementary "
            "branches, not copies of the same plan. Use up to the missing slots for: "
            "1) direct baseline implementation, 2) independent alternative approach, "
            "3) issue reproduction/localization, 4) verification/adversarial testing, "
            "5) edge-case review, 6) final integration/selection. If fewer slots are available, merge adjacent "
            "roles. Each child must receive the full task context, a clear success artifact, "
            "and an instruction to report concise findings and stop. Omit model unless needed."
        )
        return StrategyDecision(
            source=self.name,
            action=StrategyAction.SPAWN_PORTFOLIO,
            target_agent_id=state.agent_id,
            tools=("spawn", "get_cost", "set_status"),
            reason=reason,
            message=message,
            once=True,
            metadata={
                "tool_learning": {
                    "tool_family": "add",
                    "tool_name": "spawn",
                    "portfolio_slots": missing,
                },
                "state": state.as_event(),
            },
        )


class RootRecoverySpawnStrategy:
    """Unsupervised add-action rule for roots stuck in solo deepening.

    This is a behavior-level rescue rule: if the root has spent several turns
    without enough shared candidates, it narrows the next turn to spawn/query
    coordination tools so the system adds short independent branches instead
    of letting one long root branch absorb the whole budget.
    """

    name = "root_recovery_spawn"

    def __init__(self, config: RootRecoverySpawnConfig):
        self.config = config

    def decide(self, state: StrategyState) -> StrategyDecision | None:
        if not self.config.enabled:
            return None
        if not state.is_root or state.status != "running":
            return None
        if state.pending_override:
            return None
        if state.turns < self.config.min_root_turns:
            return None
        if state.shared_candidate_count >= self.config.min_shared_candidates:
            return None
        if state.total_agents >= self.config.target_agents:
            return None
        if state.root_recovery_spawn_attempts >= self.config.max_attempts:
            return None
        if self.config.require_no_active_children and state.active_children > 0:
            return None

        missing = max(1, self.config.target_agents - state.total_agents)
        reason = (
            f"root has spent {state.turns} turns with {state.shared_candidate_count} shared "
            f"candidates and only {state.total_agents} agents; add up to {missing} short "
            "branches before further solo deepening."
        )
        message = (
            "[Runtime strategy: recovery spawn]\n"
            f"{reason}\n"
            "Call spawn now for short, non-duplicative branches. Each branch should have a "
            "distinct success artifact: a candidate, a localization report, a verification "
            "plan/result, or explicit no-candidate evidence. Keep the root as coordinator; "
            "do not use this turn for local editing or finalization."
        )
        return StrategyDecision(
            source=self.name,
            action=StrategyAction.SPAWN_RECOVERY,
            target_agent_id=state.agent_id,
            tools=("spawn", "query", "wait", "get_cost", "set_status"),
            reason=reason,
            message=message,
            once=True,
            metadata={
                "tool_learning": {
                    "tool_family": "add",
                    "tool_name": "spawn",
                    "purpose": "recover_from_solo_deepening",
                    "portfolio_slots": missing,
                },
                "state": state.as_event(),
            },
        )


class CandidateEvidenceGateStrategy:
    """Unsupervised query/add rule before candidate finalization.

    The strategy does not inspect task content. It only asks whether shared
    candidates exist without an accompanying verification/selection artifact.
    If so, the next root turn is narrowed to query/wait or spawn/query/wait,
    forcing coordination evidence before finalization.
    """

    name = "candidate_evidence_gate"

    def __init__(self, config: CandidateEvidenceGateConfig):
        self.config = config

    def decide(self, state: StrategyState) -> StrategyDecision | None:
        if not self.config.enabled:
            return None
        if not state.is_root or state.status != "running":
            return None
        if state.pending_override:
            return None
        if state.turns < self.config.min_root_turns:
            return None
        if state.shared_candidate_count < self.config.min_shared_candidates:
            return None
        if state.shared_verification_count >= self.config.required_verification_files:
            return None
        if state.candidate_evidence_gate_attempts >= self.config.max_attempts:
            return None

        if state.active_children > 0:
            reason = (
                f"{state.shared_candidate_count} shared candidates exist but only "
                f"{state.shared_verification_count} verification artifacts are visible; "
                f"{state.active_children} children are still active, so query/wait before "
                "creating more work or finalizing."
            )
            message = (
                "[Runtime strategy: candidate evidence gate]\n"
                f"{reason}\n"
                "Call query and/or wait now to collect current child status and artifacts. "
                "Do not edit or finalize on this turn. If no active child is producing "
                "verification evidence, the next coordination action should add a short "
                "verifier/selector branch."
            )
            return StrategyDecision(
                source=self.name,
                action=StrategyAction.QUERY_CANDIDATE_EVIDENCE,
                target_agent_id=state.agent_id,
                tools=("query", "wait", "get_cost"),
                reason=reason,
                message=message,
                once=True,
                metadata={
                    "tool_learning": {
                        "tool_family": "query",
                        "tool_name": "query",
                        "purpose": "candidate_evidence_collection",
                    },
                    "state": state.as_event(),
                },
            )

        reason = (
            f"{state.shared_candidate_count} shared candidates exist but only "
            f"{state.shared_verification_count} verification artifacts are visible; "
            "add a verifier/selector branch before finalization."
        )
        message = (
            "[Runtime strategy: spawn verifier]\n"
            f"{reason}\n"
            "Call spawn once for a short verifier/selector branch. Its job is to inspect "
            "shared candidates, apply or compare them when practical, run focused public "
            "checks when available, write a verification/ranking artifact to shared storage, "
            "report concise evidence, and stop. Do not use this turn to make a new final patch."
        )
        return StrategyDecision(
            source=self.name,
            action=StrategyAction.SPAWN_VERIFIER,
            target_agent_id=state.agent_id,
            tools=("spawn", "query", "wait", "get_cost"),
            reason=reason,
            message=message,
            once=True,
            metadata={
                "tool_learning": {
                    "tool_family": "add",
                    "tool_name": "spawn",
                    "purpose": "candidate_verification",
                },
                "state": state.as_event(),
            },
        )


class FinalEvidenceGateStrategy:
    """Hard behavior gate that blocks root finalization before coordination evidence.

    This rule is intentionally task-agnostic. It never judges candidate content;
    it only checks whether the root has queried candidate-producing branches and
    consumed verifier/selection evidence before entering a finish-like phase.
    """

    name = "final_evidence_gate"

    def __init__(self, config: FinalEvidenceGateConfig):
        self.config = config

    def decide(self, state: StrategyState) -> StrategyDecision | None:
        if not self.config.enabled:
            return None
        if not state.is_root or state.status != "running":
            return None
        if state.pending_override:
            return None
        if state.turns < self.config.min_root_turns:
            return None
        if state.shared_candidate_count < self.config.min_shared_candidates:
            return None
        if state.delivery_phase not in self.config.trigger_phases:
            return None

        candidate_agents = max(0, state.candidate_agent_count)
        required_candidate_queries = 0
        if candidate_agents >= self.config.min_candidate_agents:
            ratio = max(0.0, min(1.0, self.config.query_coverage_ratio))
            required_candidate_queries = max(1, math.ceil(candidate_agents * ratio))

        if required_candidate_queries and state.candidate_query_coverage < required_candidate_queries:
            reason = (
                f"root is in {state.delivery_phase} with {state.shared_candidate_count} shared candidates "
                f"from {candidate_agents} candidate agents, but has only queried "
                f"{state.candidate_query_coverage}/{required_candidate_queries} required candidate branches."
            )
            message = (
                "[Runtime strategy: final evidence gate]\n"
                f"{reason}\n"
                "Do not finalize on this turn. Call query on candidate-producing child agents, "
                "or wait if relevant children are still running. This is a tool-behavior gate only: "
                "collect coordination evidence without changing the task-specific answer directly."
            )
            return StrategyDecision(
                source=self.name,
                action=StrategyAction.FINAL_EVIDENCE_GATE,
                target_agent_id=state.agent_id,
                tools=("query", "wait", "get_cost"),
                reason=reason,
                message=message,
                once=True,
                metadata={
                    "tool_learning": {
                        "tool_family": "query",
                        "tool_name": "query",
                        "purpose": "candidate_query_coverage_before_final",
                    },
                    "state": state.as_event(),
                    "required_candidate_queries": required_candidate_queries,
                },
            )

        if self.config.require_verification and state.shared_verification_count <= 0:
            if (
                self.config.allow_spawn_verifier
                and state.candidate_evidence_gate_attempts < self.config.max_verifier_spawns
            ):
                reason = (
                    f"root is in {state.delivery_phase} with {state.shared_candidate_count} shared candidates "
                    "but no shared verification/selection artifact is visible."
                )
                message = (
                    "[Runtime strategy: final evidence gate]\n"
                    f"{reason}\n"
                    "Do not finalize on this turn. Call spawn for one short verifier/selector branch, "
                    "or query/wait if a verifier is already active. The branch should produce a generic "
                    "verification or ranking artifact and then stop."
                )
                return StrategyDecision(
                    source=self.name,
                    action=StrategyAction.SPAWN_VERIFIER,
                    target_agent_id=state.agent_id,
                    tools=("spawn", "query", "wait", "get_cost"),
                    reason=reason,
                    message=message,
                    once=True,
                    metadata={
                        "tool_learning": {
                            "tool_family": "add",
                            "tool_name": "spawn",
                            "purpose": "verification_before_final",
                        },
                        "state": state.as_event(),
                    },
                )

            reason = (
                f"root is in {state.delivery_phase} with candidates but no verification artifact; "
                "spawn attempts are exhausted, so collect existing coordination evidence before finalizing."
            )
            message = (
                "[Runtime strategy: final evidence gate]\n"
                f"{reason}\n"
                "Do not finalize on this turn. Call query or wait to collect any existing branch evidence."
            )
            return StrategyDecision(
                source=self.name,
                action=StrategyAction.FINAL_EVIDENCE_GATE,
                target_agent_id=state.agent_id,
                tools=("query", "wait", "get_cost"),
                reason=reason,
                message=message,
                once=True,
                metadata={
                    "tool_learning": {
                        "tool_family": "query",
                        "tool_name": "query",
                        "purpose": "verification_evidence_before_final",
                    },
                    "state": state.as_event(),
                },
            )

        if (
            self.config.require_verification_review
            and state.shared_verification_count > 0
            and state.verification_review_calls <= 0
        ):
            reason = (
                f"{state.shared_verification_count} verification/selection artifacts exist, "
                "but the root has not reviewed them before finalization."
            )
            message = (
                "[Runtime strategy: final evidence gate]\n"
                f"{reason}\n"
                "Do not finalize on this turn. Call ws_read_file for the shared verification/selection "
                "artifact, or query the verifier branch if the artifact path is not obvious."
            )
            return StrategyDecision(
                source=self.name,
                action=StrategyAction.QUERY_VERIFICATION_EVIDENCE,
                target_agent_id=state.agent_id,
                tools=("ws_read_file", "query", "wait", "get_cost"),
                reason=reason,
                message=message,
                once=True,
                metadata={
                    "tool_learning": {
                        "tool_family": "query",
                        "tool_name": "ws_read_file",
                        "purpose": "verification_review_before_final",
                    },
                    "state": state.as_event(),
                },
            )

        return None


class CandidateFinalizeStrategy:
    """Unsupervised rule for selecting or improving shared candidates.

    This strategy stays benchmark-agnostic: it looks at shared candidate count,
    root progress, and candidate byte size. For CodeGolf, byte size is a useful
    proxy; for normal coding tasks, the same rule behaves as a final selection
    nudge once several artifacts exist.
    """

    name = "candidate_finalize"

    def __init__(self, config: CandidateFinalizeConfig):
        self.config = config

    def decide(self, state: StrategyState) -> StrategyDecision | None:
        if not self.config.enabled:
            return None
        if not state.is_root or state.status != "running":
            return None
        if state.pending_override:
            return None
        if state.shared_candidate_count < self.config.min_shared_candidates:
            return None
        if state.active_children > 0 and state.turns < self.config.finalize_after_turns:
            return None

        best_bytes = state.best_shared_candidate_bytes

        if (
            best_bytes is not None
            and best_bytes >= self.config.readable_bytes_threshold
            and state.turns >= self.config.improve_after_turns
            and state.candidate_improve_attempts < self.config.max_improve_attempts
            and not (
                self.config.stop_improve_when_stalled
                and state.candidate_improve_stalled
            )
        ):
            reason = (
                f"{state.shared_candidate_count} shared candidates exist, but the best visible "
                f"candidate is still {best_bytes} bytes; add a short improvement branch before finalizing."
            )
            message = (
                "[Runtime strategy: improve candidate]\n"
                f"{reason}\n"
                "Call spawn once for a short candidate-improvement branch. Give it the current best "
                "candidate path, ask it to preserve correctness, reduce size/complexity, run local "
                "checks, write an improved candidate to shared storage, report bytes and evidence, "
                "then stop. Do not let this become open-ended exploration."
            )
            return StrategyDecision(
                source=self.name,
                action=StrategyAction.IMPROVE_CANDIDATE,
                target_agent_id=state.agent_id,
                tools=("spawn", "query", "wait", "get_cost"),
                reason=reason,
                message=message,
                once=True,
                metadata={
                    "tool_learning": {
                        "tool_family": "add",
                        "tool_name": "spawn",
                        "purpose": "candidate_improvement",
                    },
                    "state": state.as_event(),
                    "best_shared_candidate_bytes": best_bytes,
                    "candidate_improve_attempts": state.candidate_improve_attempts,
                    "candidate_improve_stalled": state.candidate_improve_stalled,
                },
            )

        if state.turns < self.config.finalize_after_turns:
            return None
        if not any(tool in state.available_tools for tool in ("shell", "ws_read_file", "ws_create_file", "set_status", "submit")):
            return None
        if (
            self.config.min_stable_observations > 1
            and best_bytes is not None
            and state.best_shared_candidate_stable_observations < self.config.min_stable_observations
        ):
            return None
        if (
            self.config.finalize_once_per_best
            and best_bytes is not None
            and state.last_finalized_candidate_bytes == best_bytes
        ):
            return None
        if (
            self.config.max_finalize_attempts > 0
            and state.candidate_finalize_attempts >= self.config.max_finalize_attempts
        ):
            return None

        reason = (
            f"{state.shared_candidate_count} shared candidates exist and root is at turn {state.turns}; "
            "select, verify, and submit the best available candidate instead of continuing exploration."
        )
        message = (
            "[Runtime strategy: finalize candidate]\n"
            f"{reason}\n"
            "Compare shared candidates using local evidence, prefer passing candidates, then prefer "
            "the smallest/simple correct artifact. Copy the selected artifact to the required final "
            "path, run a focused verification if possible, and finish with set_status(done, result=...). "
            "Before finalizing, compare against the smallest shared candidate observed in this run. "
            "Do not submit a longer candidate if a shorter candidate has passing local evidence. "
            "Do not start unrelated exploration."
        )
        return StrategyDecision(
            source=self.name,
            action=StrategyAction.FINALIZE_CANDIDATE,
            target_agent_id=state.agent_id,
            tools=("shell", "ws_read_file", "ws_create_file", "ws_replace_string", "set_status", "submit", "query", "wait", "get_cost"),
            reason=reason,
            message=message,
            once=True,
            metadata={
                "tool_learning": {
                    "tool_family": "finish",
                    "tool_name": "set_status",
                    "purpose": "candidate_selection",
                },
                "state": state.as_event(),
                "best_shared_candidate_bytes": best_bytes,
            },
        )


def decision_from_intervention_item(item: Mapping[str, Any]) -> StrategyDecision | None:
    """Convert a JSONL supervisor intervention row into a strategy decision."""
    agent_id = str(item.get("agent_id") or "").strip()
    if not agent_id:
        return None

    raw_action = str(item.get("action") or "tool_override")
    if raw_action not in {"tool_override", "self_kill"}:
        return None

    tools = item.get("tools")
    if not isinstance(tools, list) or not tools:
        tools = ["kill", "get_cost"] if raw_action == "self_kill" else []

    action = (
        StrategyAction.SELF_KILL
        if raw_action == "self_kill"
        else StrategyAction.TOOL_OVERRIDE
    )
    once = bool(item.get("once", raw_action != "self_kill"))
    return StrategyDecision(
        source="external_intervention",
        action=action,
        target_agent_id=agent_id,
        tools=tuple(str(tool) for tool in tools),
        reason=str(item.get("reason") or "").strip(),
        message=str(item.get("message") or "").strip(),
        once=once,
        metadata={
            "created_at": item.get("created_at"),
            "raw_action": raw_action,
            "tool_learning": {
                "tool_family": "delete" if raw_action == "self_kill" else "control",
                "tool_name": "kill" if raw_action == "self_kill" else "tool_override",
                "supervised": True,
            },
        },
    )
