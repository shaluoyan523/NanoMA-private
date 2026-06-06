"""Core runtime: Agent, Runtime, ReAct loop."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable, Literal

from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import (
    LLMResponse, Message, RetryConfig, ToolCall, ToolDef,
    count_message_tokens, estimate_tokens, openai_compatible_call, set_log_dir,
)
from nanoma.memory import ActiveTaskCard, ExperienceCard, MemoryBroker
from nanoma.sandbox import SandboxConfig, SandboxSession
from nanoma.scheduler import Scheduler
from nanoma.tools import WORK_TOOLS

logger = logging.getLogger("nanoma")

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
    # Compression / truncation settings
    compress_keep_recent: int = 6           # messages to keep verbatim during compression
    compress_max_messages: int = 40         # max old messages to include in summary
    compress_max_chars: int = 300           # max chars per message in summary (0 = unlimited)
    shell_max_output: int = 10000           # max chars for shell output (0 = unlimited)
    file_read_max_chars: int = 50000        # max chars for file_read (0 = unlimited)
    file_list_max_entries: int = 500        # max entries for file_list (0 = unlimited)
    grep_max_results: int = 100             # max grep results (0 = unlimited)


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
            role=role,
            create_type=create_type,
            relationship=relationship,
            created_by=created_by,
            group_id=group_id,
            workflow_prior=workflow_prior,
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
        })
        return agent

    def start_agent(self, agent: Agent):
        agent._task = asyncio.ensure_future(self._agent_loop(agent))

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
        """Run a single root agent to completion, then shut down all remaining agents."""
        await self.sandbox.start()
        try:
            self._emit("system", "sandbox_start", {
                "backend": self.sandbox.backend,
                "workspace_root": str(self.config.workspace_root),
            })
            root = self.create_agent(task, model=model)
            self.start_agent(root)
            await root._task
            # Clean up any still-running children
            running = [a for a in self.agents.values() if a.status in ("running", "idle") and a.id != root.id]
            if running:
                await self.shutdown()
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
        all_tools = {**WORK_TOOLS, **META_TOOLS}
        tool_schemas = [t["schema"] for t in all_tools.values()]

        try:
            while agent.status == "running":
                agent._turns += 1
                agent._last_active = time.time()

                if agent._compact_pending:
                    self._execute_compact(agent)

                if agent._rebirth_pending:
                    self._execute_rebirth(agent)

                # Turn limits
                if agent._turns > agent.quota.max_turns:
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
                self._emit(agent.id, "llm_done", {
                    "tokens": response.usage.total_tokens,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cached": response.usage.cached_input_tokens,
                    "cost": round(cost, 6),
                    "model": agent.model,
                    "tool_calls": [tc.name for tc in response.tool_calls] if response.tool_calls else [],
                    "has_content": bool(response.content),
                    "content_preview": (response.content or "")[:200],
                })

                # Process response
                if response.tool_calls:
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

                        result = await self._execute_tool(tc, agent, all_tools)
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
                    agent.history.append({"role": "assistant", "content": response.content})
                else:
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
                self.state_board_sync(agent.id)
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
            header = f"[Message from {msg.from_id} | type={msg.message_type} | urgency={msg.urgency}]"
            payload = f"\nPayload: {json.dumps(msg.payload, ensure_ascii=False)}" if msg.payload is not None else ""
            ack = "\nAck requested." if msg.requires_ack else ""
            agent.history.append({"role": "user", "content": f"{header}: {msg.content}{payload}{ack}"})
        return msgs

    def _execute_compact(self, agent: Agent):
        params = agent._compact_pending
        agent._compact_pending = None
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

        return f"""You are agent "{agent_id}" in a multi-agent system.

Task: {task}
Workspace: {workspace} (private to you)
Shared: {shared} (visible to all agents){time_info}
{context_section}
Workflow prior guidance: research_loop / critic_review_loop / synthesis_loop are prompt patterns only, not workflow engines.
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

        # Peak concurrency (estimate from events)
        running_at = {}  # agent_id -> (start_time, end_time)
        for e in self._events:
            aid = e["agent"]
            t = e["t"]
            if e["event"] == "spawn":
                running_at[aid] = [t, None]
            elif e["event"] == "done":
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
        await self.sandbox.stop()
