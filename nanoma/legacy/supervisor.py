"""第 4 代 · LLM 拓扑监督者（13 个方法，1094 行）。

由一个 LLM 周期性审视全局状态并下达拓扑决策。supervisor_enabled 默认 False，
没有任何 benchmark adapter 打开它；设计上还与第 2 代固定拓扑互斥
（Runtime.__init__ 会在拓扑生效时强制关掉它）。

以 mixin 形式由 Runtime 继承，调用点无需改动。抽出时实测边界：
入向仅 1 个入口 <- core 中 2 个方法；出向依赖 core 中 10 个方法。
这是几个归档世代里边界最窄的一个。配置在 core.py 的
_ArchivedSupervisorConfig 里。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from nanoma.llm import RetryConfig, anthropic_compatible_call, openai_compatible_call
from nanoma.strategy import StrategyDecision
from nanoma.tool_groups import _DELIVERY_WRITE_TOOLS, _SHELL_TOOLS

if TYPE_CHECKING:
    from nanoma.core import Agent, Message, ToolPolicyState
    from nanoma.strategy import StrategyState

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


class SupervisorMixin:
    """Runtime 的 LLM 监督者行为。不可独立实例化，只作为 Runtime 的基类。"""

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
