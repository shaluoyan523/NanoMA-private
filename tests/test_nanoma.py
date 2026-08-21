"""Tests for NanoMA framework — no real LLM calls, fully deterministic."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from nanoma.core import Agent, Envelope, ResourceQuota, Runtime, RuntimeConfig, ToolContext
from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import (
    LLMResponse,
    RetryConfig,
    ToolCall,
    openai_compatible_call,
    _openai_messages_to_anthropic,
    _openai_tool_to_anthropic,
    estimate_tokens,
    count_message_tokens,
)
from nanoma.meta import (
    meta_spawn, meta_kill, meta_send, meta_query, meta_wait,
    meta_transfer, meta_set_bio, meta_get_cost, meta_set_status,
    meta_rebirth, meta_submit, meta_batch, META_TOOLS,
)
from nanoma.tools import WORK_TOOLS
from nanoma.models import ModelRegistry, load_models
from nanoma.scheduler import Scheduler


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "shared").mkdir()
    return ws


@pytest.fixture
def runtime(tmp_workspace):
    """Runtime with a mock LLM that immediately calls set_status(done)."""
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "test done"})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_agents=50,
        log_dir=None,
    )
    return Runtime(config=config, llm_call=mock_llm)


@pytest.fixture
def runtime_multi_turn(tmp_workspace):
    """Runtime where LLM does 3 turns then quits."""
    call_count = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return LLMResponse(
                tool_calls=[ToolCall(id=f"tc{call_count['n']}", name="set_status",
                                     arguments={"status": "done", "result": f"done after {call_count['n']} turns"})],
                usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id=f"tc{call_count['n']}", name="get_cost", arguments={})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None)
    return Runtime(config=config, llm_call=mock_llm)


# ─── Test: Basic agent lifecycle ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_basic_run(runtime):
    """Agent starts, calls set_status(done), terminates."""
    result = await runtime.run("Say hello")
    assert result == "test done"
    assert len(runtime.agents) == 1
    agent = list(runtime.agents.values())[0]
    assert agent.status == "done"
    assert agent._turns >= 1


@pytest.mark.asyncio
async def test_multi_turn(runtime_multi_turn):
    """Agent runs multiple turns before completing."""
    result = await runtime_multi_turn.run("Do something complex")
    assert "done after 3 turns" in result


@pytest.mark.asyncio
async def test_zero_max_turns_disables_turn_limit(tmp_workspace):
    """max_turns=0 means unlimited turns."""
    calls = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 4:
            return LLMResponse(
                tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id=f"cost-{calls['n']}", name="get_cost", arguments={})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, max_turns=0, log_dir=None)
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("needs several turns")
    assert result == "finished"
    assert calls["n"] == 4


@pytest.mark.asyncio
async def test_worker_does_not_receive_direct_spawn(tmp_workspace):
    """Workers delegate only through the same-model planning decision."""
    captured = {}

    async def mock_llm(messages, model, tools=None, **kwargs):
        captured["tools"] = [t["function"]["name"] for t in tools]
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, max_agents=50, log_dir=None)
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Solve a generic task")
    assert result == "finished"
    assert "spawn" not in captured["tools"]
    assert "spawn_many" not in captured["tools"]
    assert "task_spawn" not in captured["tools"]
    assert "task_create" in captured["tools"]


@pytest.mark.asyncio
async def test_state_tool_policy_hides_spawn_without_prompt_injection(tmp_workspace):
    """State weights narrow schemas without adding policy text to the agent loop."""
    captured = {}

    async def mock_llm(messages, model, tools=None, **kwargs):
        captured["messages"] = [dict(m) for m in messages]
        captured["tools"] = [t["function"]["name"] for t in tools]
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, max_agents=1, log_dir=None)
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Solve the task and write `answer.json`")
    assert result == "finished"
    assert "spawn" not in captured["tools"]
    assert "set_status" in captured["tools"]

    visible_text = "\n".join(str(m.get("content") or "") for m in captured["messages"][1:])
    assert "tool_policy" not in visible_text.lower()
    assert "create weight" not in visible_text.lower()
    assert "spawn_closed" not in visible_text.lower()


@pytest.mark.asyncio
async def test_state_tool_policy_blocks_direct_spawn_when_dependency_active(tmp_workspace):
    """The spawn handler enforces the same state policy used for tool schemas."""
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_agents=50,
        tool_policy_dependency_window=1,
        log_dir=None,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("parent")
    child = rt.create_agent("active child", parent=parent.id)
    child.status = "running"

    result = await meta_spawn({"task": "second child"}, parent, rt)
    assert "error" in result
    assert "spawn blocked by tool policy" in result["error"]
    assert any(e["event"] == "tool_policy_block" for e in rt._events)


@pytest.mark.asyncio
async def test_state_tool_policy_prunes_tools_without_prompt_injection(tmp_workspace):
    """Optional pruning shrinks schemas structurally without adding policy text."""
    captured = {}

    async def mock_llm(messages, model, tools=None, **kwargs):
        captured["messages"] = [dict(m) for m in messages]
        captured["tools"] = [t["function"]["name"] for t in tools]
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_total_tokens=100,
        tool_policy_prune_tools=True,
        tool_policy_prune_pressure_start=0.0,
        tool_policy_prune_pressure_end=0.0,
        tool_policy_prune_min_tools=6,
        log_dir=None,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Solve the task and write `answer.json`")
    assert result == "finished"

    assert len(captured["tools"]) < len(rt._all_tools())
    assert "set_status" in captured["tools"]
    assert "submit" in captured["tools"]
    assert "ws_create_file" in captured["tools"]

    visible_text = "\n".join(str(m.get("content") or "") for m in captured["messages"][1:])
    assert "tool_policy" not in visible_text.lower()
    assert "prune_tools" not in visible_text.lower()


@pytest.mark.asyncio
async def test_soft_total_tokens_drives_pruning_without_hard_stop(tmp_workspace):
    """Policy-only token pressure can prune tools without ending the task."""
    captured = {}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls = captured.get("calls", 0) + 1
        captured["calls"] = calls
        captured.setdefault("tool_counts", []).append(len(tools))
        if calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="cost", name="get_cost", arguments={})],
                usage=UsageRecord(input_tokens=100, output_tokens=0, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_total_tokens=0,
        tool_policy_soft_total_tokens=50,
        tool_policy_prune_tools=True,
        tool_policy_prune_pressure_start=0.5,
        tool_policy_prune_pressure_end=1.0,
        tool_policy_prune_min_tools=6,
        log_dir=None,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Solve the task and write `answer.json`")

    assert result == "finished"
    assert captured["calls"] == 2
    assert captured["tool_counts"][1] < captured["tool_counts"][0]
    assert all("Max total tokens reached" not in (agent.result or "") for agent in rt.agents.values())


@pytest.mark.asyncio
async def test_shell_subcapability_pruning_keeps_shell_schema(tmp_workspace):
    """Constraint can prune shell sub-capabilities without creating new tools."""
    captured = {}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls = captured.get("calls", 0) + 1
        captured["calls"] = calls
        captured.setdefault("tool_names", []).append([
            t["function"]["name"] for t in tools
        ])
        if calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="cost", name="get_cost", arguments={})],
                usage=UsageRecord(input_tokens=100, output_tokens=0, model=model),
            )
        if calls == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="pip", name="shell", arguments={"command": "pip install requests"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_total_tokens=0,
        tool_policy_soft_total_tokens=50,
        tool_policy_prune_tools=True,
        tool_policy_prune_shell_capabilities=True,
        tool_policy_shell_capability_pressure_start=0.5,
        tool_policy_shell_capability_pressure_end=1.0,
        tool_policy_prune_pressure_start=0.5,
        tool_policy_prune_pressure_end=1.0,
        log_dir=None,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Solve the task")

    assert result == "finished"
    assert "shell" in captured["tool_names"][1]
    blocked = [
        e for e in rt._events
        if e["event"] == "tool_call" and e["data"]["tool"] == "shell"
    ][0]
    assert "Blocked shell capability by constraint: package" in blocked["data"]["result"]


@pytest.mark.asyncio
async def test_web_saturation_blocks_web_without_prompt_injection(tmp_workspace):
    """Runtime web saturation removes shell web capability without asking the LLM to compute it."""
    captured = {"messages": [], "tools": []}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls = captured.get("calls", 0) + 1
        captured["calls"] = calls
        captured["messages"].append([dict(m) for m in messages])
        captured["tools"].append([t["function"]["name"] for t in tools])
        if calls <= 3:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id=f"web-{calls}",
                        name="shell",
                        arguments={"command": f'printf "No results\\n" # https://example.com/search?q=repeat{calls % 2}'},
                    )
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if calls == 4:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="web-blocked",
                        name="shell",
                        arguments={"command": 'printf "No results\\n" # https://example.com/search?q=repeat0'},
                    )
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_total_tokens=0,
        tool_policy_soft_total_tokens=10,
        tool_policy_prune_shell_capabilities=True,
        tool_policy_shell_capability_pressure_start=0.0,
        tool_policy_shell_capability_pressure_end=1.0,
        tool_policy_web_saturation_min_calls=3,
        tool_policy_web_saturation_threshold=0.3,
        log_dir=None,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Search public evidence, then finish")

    assert result == "finished"
    blocked_shell_events = [
        e for e in rt._events
        if e["event"] == "tool_call"
        and e["data"]["tool"] == "shell"
        and "Blocked shell capability by constraint: web" in e["data"]["result"]
    ]
    assert blocked_shell_events
    assert rt.agents["alpha"]._shell_activity.web_saturation >= config.tool_policy_web_saturation_threshold
    assert all("shell" in names for names in captured["tools"])

    visible_text = "\n".join(
        str(m.get("content") or "")
        for call_messages in captured["messages"]
        for m in call_messages[1:]
    )
    assert "web_saturation" not in visible_text.lower()
    assert "saturation threshold" not in visible_text.lower()
    assert "repeated_web_queries" not in visible_text.lower()


@pytest.mark.asyncio
async def test_web_saturation_finalize_scope_removes_shell_schema(tmp_workspace):
    """Repeated web blocks after saturation force a finalization-only schema."""
    captured = {"messages": [], "tools": []}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls = captured.get("calls", 0) + 1
        captured["calls"] = calls
        names = [t["function"]["name"] for t in tools]
        captured["messages"].append([dict(m) for m in messages])
        captured["tools"].append(names)
        if calls <= 3:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id=f"web-{calls}",
                        name="shell",
                        arguments={"command": f'printf "No results\\n" # https://example.com/search?q=repeat{calls % 2}'},
                    )
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if calls <= 5:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id=f"blocked-{calls}",
                        name="shell",
                        arguments={"command": 'printf "No results\\n" # https://example.com/search?q=repeat0'},
                    )
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        assert "shell" not in names
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "finished"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_total_tokens=0,
        tool_policy_soft_total_tokens=10,
        tool_policy_prune_shell_capabilities=True,
        tool_policy_shell_capability_pressure_start=0.0,
        tool_policy_shell_capability_pressure_end=1.0,
        tool_policy_web_saturation_min_calls=3,
        tool_policy_web_saturation_threshold=0.3,
        tool_policy_web_saturation_finalize_after_blocks=2,
        log_dir=None,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Search public evidence, then finish")

    assert result == "finished"
    assert rt.agents["alpha"]._shell_activity.finalize_after_web_saturation is True
    assert "shell" in captured["tools"][4]
    assert "shell" not in captured["tools"][5]
    assert set(captured["tools"][5]) <= {"ws_create_file", "ws_append_file", "ws_read_file", "submit", "set_status", "get_cost"}

    visible_text = "\n".join(
        str(m.get("content") or "")
        for call_messages in captured["messages"]
        for m in call_messages[1:]
    )
    assert "web_saturation" not in visible_text.lower()
    assert "finalize_after_web_saturation" not in visible_text.lower()


@pytest.mark.asyncio
async def test_web_saturation_finalize_rejects_idle_and_empty_done(tmp_workspace):
    """Finalization scope keeps the agent running until it submits a non-empty answer."""
    captured = {"tool_results": []}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls = captured.get("calls", 0) + 1
        captured["calls"] = calls
        captured["tool_results"].extend([
            str(m.get("content") or "")
            for m in messages
            if m.get("role") == "tool"
        ])
        if calls <= 3:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id=f"web-{calls}",
                        name="shell",
                        arguments={"command": f'printf "No results\\n" # https://example.com/search?q=repeat{calls % 2}'},
                    )
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if calls <= 5:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id=f"blocked-{calls}",
                        name="shell",
                        arguments={"command": 'printf "No results\\n" # https://example.com/search?q=repeat0'},
                    )
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if calls == 6:
            return LLMResponse(
                tool_calls=[ToolCall(id="idle", name="set_status", arguments={"status": "idle"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if calls == 7:
            return LLMResponse(
                tool_calls=[ToolCall(id="empty", name="set_status", arguments={"status": "done", "result": ""})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"status": "done", "result": "142"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_total_tokens=0,
        tool_policy_soft_total_tokens=10,
        tool_policy_prune_shell_capabilities=True,
        tool_policy_shell_capability_pressure_start=0.0,
        tool_policy_shell_capability_pressure_end=1.0,
        tool_policy_web_saturation_min_calls=3,
        tool_policy_web_saturation_threshold=0.3,
        tool_policy_web_saturation_finalize_after_blocks=2,
        log_dir=None,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("Search public evidence, then finish")

    assert result == "142"
    assert captured["calls"] == 8
    joined_tool_results = "\n".join(captured["tool_results"])
    assert "idle is unavailable after web saturation" in joined_tool_results
    assert "empty done result is unavailable after web saturation" in joined_tool_results


# ─── Test: ID generation ─────────────────────────────────────────────────────

def test_id_generation():
    from nanoma.core import IdGenerator
    gen = IdGenerator()
    ids = [gen.next() for _ in range(30)]
    assert ids[0] == "alpha"
    assert ids[25] == "zulu"
    assert ids[26] == "alpha-1"
    assert len(set(ids)) == 30  # all unique


# ─── Test: ResourceQuota ─────────────────────────────────────────────────────

def test_quota_defaults():
    q = ResourceQuota(budget=10.0, time_limit=60.0, max_turns=100)
    assert q.budget == 10.0
    assert q.time_limit == 60.0
    assert q.max_turns == 100


# ─── Test: CostLedger ────────────────────────────────────────────────────────

def test_ledger():
    ledger = CostLedger(total_budget=5.0)
    assert ledger.remaining() == 5.0
    assert ledger.can_afford(3.0)
    usage = UsageRecord(input_tokens=1000, output_tokens=500, model="test")
    ledger.record("agent-1", usage)
    assert ledger.total_spent > 0
    assert "agent-1" in ledger.per_agent


# ─── Test: Scheduler ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler():
    s = Scheduler(max_concurrent=2)
    assert s.stats["available"] == 2
    await s.acquire()
    assert s.stats["active"] == 1
    assert s.stats["available"] == 1
    s.release()
    assert s.stats["active"] == 0


# ─── Test: Message delivery ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_message_delivery(runtime):
    agent = runtime.create_agent("test task")
    env = Envelope(from_id="other", to_id=agent.id, content="hello", tokens=5, timestamp=0.0, mode="queue")
    await runtime.deliver(env)
    assert not agent._queue_inbox.empty()


@pytest.mark.asyncio
async def test_idle_wake(runtime):
    """Idle agent wakes on message."""
    agent = runtime.create_agent("test")
    agent.status = "idle"
    env = Envelope(from_id="x", to_id=agent.id, content="wake up", tokens=5, timestamp=0.0, mode="queue")
    await runtime.deliver(env)
    assert agent.status == "running"


# ─── Test: Meta tools ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_spawn(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn({"task": "child task"}, parent, runtime)
    assert "agent_id" in result
    assert result["agent_id"] in runtime.agents
    # v0.9.0: no per-agent budget deduction on spawn (global budget model)
    child = runtime.agents[result["agent_id"]]
    assert child.parent == parent.id
    assert child.id in parent.children


@pytest.mark.asyncio
async def test_meta_spawn_max_depth(runtime):
    """Spawn fails when max depth is exceeded."""
    parent = runtime.create_agent("parent")
    parent.depth = runtime.config.max_depth  # already at max
    result = await meta_spawn({"task": "child"}, parent, runtime)
    assert "error" in result


@pytest.mark.asyncio
async def test_meta_kill(runtime):
    parent = runtime.create_agent("parent")
    child = runtime.create_agent("child", parent=parent.id)
    parent.children.add(child.id)
    result = await meta_kill({"agent_id": child.id}, parent, runtime)
    assert result["killed"] == child.id
    assert child.status == "done"


@pytest.mark.asyncio
async def test_meta_kill_permission(runtime):
    a = runtime.create_agent("a")
    b = runtime.create_agent("b")
    result = await meta_kill({"agent_id": b.id}, a, runtime)
    assert "error" in result  # can't kill non-descendant


@pytest.mark.asyncio
async def test_meta_send(runtime):
    a = runtime.create_agent("sender")
    b = runtime.create_agent("receiver")
    result = await meta_send({"to": b.id, "message": "hello", "mode": "queue"}, a, runtime)
    assert result["delivered"] == 1
    assert not b._queue_inbox.empty()


@pytest.mark.asyncio
async def test_meta_send_no_broadcast(runtime):
    """No broadcast support — must specify IDs."""
    a = runtime.create_agent("sender")
    # '*' is treated as a literal agent_id which won't exist
    result = await meta_send({"to": "*", "message": "hi"}, a, runtime)
    assert result["delivered"] == 0


@pytest.mark.asyncio
async def test_meta_query_all(runtime):
    runtime.create_agent("task A")
    runtime.create_agent("task B")
    a = runtime.create_agent("querier")
    result = await meta_query({}, a, runtime)
    assert result["count"] == 3
    assert all("bio" in x for x in result["agents"])


@pytest.mark.asyncio
async def test_meta_query_single(runtime):
    a = runtime.create_agent("target")
    a.bio = "I am a coder"
    b = runtime.create_agent("querier")
    result = await meta_query({"agent_id": a.id}, b, runtime)
    assert result["bio"] == "I am a coder"
    assert "messages" not in result  # messages=0 by default


@pytest.mark.asyncio
async def test_meta_query_with_messages(runtime):
    a = runtime.create_agent("target")
    a.history.append({"role": "user", "content": "msg1"})
    a.history.append({"role": "assistant", "content": "reply1"})
    a.history.append({"role": "user", "content": "msg2"})
    a.history.append({"role": "assistant", "content": "reply2"})
    b = runtime.create_agent("querier")
    # Last 2 messages
    result = await meta_query({"agent_id": a.id, "messages": 2}, b, runtime)
    assert "messages" in result
    assert len(result["messages"]) == 2


@pytest.mark.asyncio
async def test_meta_query_all_messages(runtime):
    a = runtime.create_agent("target")
    a.history.append({"role": "user", "content": "msg1"})
    a.history.append({"role": "assistant", "content": "reply1"})
    b = runtime.create_agent("querier")
    result = await meta_query({"agent_id": a.id, "messages": -1}, b, runtime)
    assert "messages" in result
    assert len(result["messages"]) >= 2


@pytest.mark.asyncio
async def test_meta_set_bio(runtime):
    a = runtime.create_agent("worker")
    result = await meta_set_bio({"bio": "I handle parsing"}, a, runtime)
    assert a.bio == "I handle parsing"


@pytest.mark.asyncio
async def test_meta_get_cost(runtime):
    a = runtime.create_agent("worker")
    a._turns = 5
    a.tokens_consumed = 1000
    result = await meta_get_cost({}, a, runtime)
    assert result["turns_used"] == 5
    assert result["tokens_consumed"] == 1000
    assert "budget_remaining" in result
    assert "budget_total" in result
    assert result["budget_total"] == runtime.ledger.total_budget
    assert "total_agents" in result


@pytest.mark.asyncio
async def test_meta_set_status(runtime):
    a = runtime.create_agent("worker")
    result = await meta_set_status({"status": "done", "result": "finished!"}, a, runtime)
    assert a.status == "done"
    assert a.result == "finished!"


@pytest.mark.asyncio
async def test_meta_rebirth(runtime):
    a = runtime.create_agent("worker")
    a.bio = "old bio"
    # Add some history
    for i in range(10):
        a.history.append({"role": "user", "content": f"msg {i}"})
    result = await meta_rebirth({"summary": "Did steps 1-5", "new_bio": "updated bio"}, a, runtime)
    assert result["scheduled"]
    # Execute rebirth
    runtime._execute_rebirth(a)
    assert a.bio == "updated bio"
    assert len(a.history) == 2  # system + rebirth message


@pytest.mark.asyncio
async def test_meta_submit(runtime):
    a = runtime.create_agent("worker")
    # Create a file in workspace
    test_file = a.workspace / "output.txt"
    test_file.write_text("result data")
    result = await meta_submit({"path": "output.txt", "description": "final output"}, a, runtime)
    assert result["submitted"] == "output.txt"
    assert len(a.artifacts) == 1
    # Check shared copy exists
    shared = runtime._tool_context.shared_dir / "output.txt"
    assert shared.exists()


@pytest.mark.asyncio
async def test_meta_batch(runtime):
    a = runtime.create_agent("worker")
    # Write a batch file
    batch_data = [
        {"tool": "ws_create_file", "args": {"path": "hello.txt", "content": "world"}},
        {"tool": "ws_read_file", "args": {"path": "hello.txt"}},
        {"tool": "nonexistent_tool", "args": {}},
    ]
    batch_file = a.workspace / "batch.json"
    batch_file.write_text(json.dumps(batch_data))
    result = await meta_batch({"path": "batch.json"}, a, runtime)
    assert result["executed"] == 3
    # First should succeed
    assert "error" not in result["results"][0]
    # Third should fail (unknown tool)
    assert "error" in result["results"][2]
    # Verify file was actually written
    assert (a.workspace / "hello.txt").read_text() == "world"


@pytest.mark.asyncio
async def test_meta_transfer_push(runtime):
    a = runtime.create_agent("sender")
    b = runtime.create_agent("receiver")
    (a.workspace / "data.txt").write_text("content")
    result = await meta_transfer({"src": "data.txt", "to": b.id}, a, runtime)
    assert "data.txt" in result["pushed"]
    assert (b.workspace / "data.txt").read_text() == "content"


@pytest.mark.asyncio
async def test_meta_transfer_pull(runtime):
    a = runtime.create_agent("puller")
    b = runtime.create_agent("source")
    (b.workspace / "info.txt").write_text("pulled content")
    result = await meta_transfer({"src": "info.txt", "from_agent": b.id}, a, runtime)
    assert "info.txt" in result["pulled"]
    assert (a.workspace / "info.txt").read_text() == "pulled content"


@pytest.mark.asyncio
async def test_meta_transfer_shared(runtime):
    a = runtime.create_agent("worker")
    (a.workspace / "shared_file.txt").write_text("shared!")
    result = await meta_transfer({"src": "shared_file.txt", "to": "shared"}, a, runtime)
    assert (runtime._tool_context.shared_dir / "shared_file.txt").exists()


@pytest.mark.asyncio
async def test_meta_wait_immediate(runtime):
    """Wait returns immediately when children already done."""
    parent = runtime.create_agent("parent")
    child = runtime.create_agent("child", parent=parent.id)
    parent.children.add(child.id)
    child.status = "done"
    child.result = "child result"
    result = await meta_wait({}, parent, runtime)
    assert len(result["completed"]) == 1
    assert result["completed"][0]["status"] == "done"


# ─── Test: Work tools ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_file_write_read(tmp_workspace):
    from nanoma.plugins.workspace_tools import tool_create_file, tool_read_file_advanced
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()

    # Write
    result = await tool_create_file({"path": "test.txt", "content": "hello world"}, ws, ctx)
    assert result["bytes_written"] == 11

    # Read
    result = await tool_read_file_advanced({"path": "test.txt"}, ws, ctx)
    assert result["content"] == "hello world"


@pytest.mark.asyncio
async def test_tool_file_read_sandbox(tmp_workspace):
    from nanoma.plugins.workspace_tools import tool_read_file_advanced
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_read_file_advanced({"path": "/etc/passwd"}, ws, ctx)
    assert "error" in result  # outside workspace


@pytest.mark.asyncio
async def test_tool_shell_file_list(tmp_workspace):
    from nanoma.tools import tool_shell
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (ws / "a.txt").write_text("a")
    (ws / "b.txt").write_text("b")
    result = await tool_shell({"command": "find . -maxdepth 1 -type f -printf '%f\\n'"}, ws, ctx)
    assert result["exit_code"] == 0
    assert "a.txt" in result["stdout"]
    assert "b.txt" in result["stdout"]


@pytest.mark.asyncio
async def test_tool_shell(tmp_workspace):
    from nanoma.tools import tool_shell
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "echo hello"}, ws, ctx)
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_tool_shell_timeout(tmp_workspace):
    from nanoma.tools import tool_shell
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "sleep 10", "timeout": 1}, ws, ctx)
    assert result["exit_code"] == -1
    assert "Timeout" in result["stderr"]


@pytest.mark.asyncio
async def test_tool_grep(tmp_workspace):
    from nanoma.plugins.workspace_tools import tool_grep_search
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (ws / "code.py").write_text("def hello():\n    return 42\n")
    result = await tool_grep_search({"query": "hello"}, ws, ctx)
    assert result["count"] >= 1
    assert any("hello" in m["content"] for m in result["matches"])


# ─── Test: Model registry ────────────────────────────────────────────────────

def test_model_registry(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("""
models:
  test-model:
    provider: test
    context_limit: 64000
    pricing:
      input: 1.0
      cached_input: 0.1
      output: 2.0
    tier: cheap
""")
    reg = load_models(config)
    m = reg.get("test-model")
    assert m is not None
    assert m.context_limit == 64000
    assert reg.pricing("test-model") == (1.0, 0.1, 2.0)
    assert reg.route(1.0) == "test-model"


# ─── Test: Token estimation ──────────────────────────────────────────────────

def test_estimate_tokens():
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("a" * 400) == 100


def test_count_message_tokens():
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello there"},
    ]
    tokens = count_message_tokens(msgs)
    assert tokens > 0


def test_anthropic_message_conversion():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant", "content": "I will use a tool.", "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":\"echo ok\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "{\"stdout\":\"ok\\n\"}"},
    ]
    system, converted = _openai_messages_to_anthropic(messages)
    assert system == "System prompt"
    assert converted[0]["role"] == "assistant"
    assert converted[0]["content"][1]["type"] == "tool_use"
    assert converted[0]["content"][1]["input"] == {"command": "echo ok"}
    assert converted[1]["role"] == "user"
    assert converted[1]["content"][0]["type"] == "tool_result"


def test_anthropic_tool_conversion():
    tool = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run shell",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
    converted = _openai_tool_to_anthropic(tool)
    assert converted["name"] == "shell"
    assert converted["input_schema"]["required"] == ["command"]


@pytest.mark.asyncio
async def test_openai_call_passes_top_p(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            captured["body"] = json
            return FakeResponse()

    monkeypatch.setattr("nanoma.llm._get_client", lambda timeout=180.0: FakeClient())
    response = await openai_compatible_call(
        [{"role": "user", "content": "hello"}],
        "test-model",
        base_url="https://example.invalid/v1",
        api_key="test",
        temperature=0.2,
        top_p=1.0,
        max_tokens=8192,
    )
    assert response.content == "ok"
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["top_p"] == 1.0
    assert captured["body"]["max_tokens"] == 8192


@pytest.mark.asyncio
@pytest.mark.parametrize("transient_status", [401, 403])
async def test_openai_call_retries_transient_gateway_auth(monkeypatch, transient_status):
    import httpx

    calls = {"count": 0}

    class FakeResponse:
        headers = {}
        text = "Unauthorized" if transient_status == 401 else "Forbidden"

        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            if self.status_code >= 400:
                return {"error": {"message": self.text}}
            return {
                "choices": [{"message": {"content": "recovered"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError(self.text, request=request, response=response)

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            calls["count"] += 1
            return FakeResponse(transient_status if calls["count"] == 1 else 200)

    monkeypatch.setattr("nanoma.llm._get_client", lambda timeout=180.0: FakeClient())
    monkeypatch.setattr("nanoma.llm.asyncio.sleep", AsyncMock())
    response = await openai_compatible_call(
        [{"role": "user", "content": "hello"}],
        "test-model",
        base_url="https://example.invalid/v1",
        api_key="test",
        retry_config=RetryConfig(max_retries=2, base_delay=0.1, max_delay=0.1),
    )
    assert response.content == "recovered"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_openai_call_does_not_retry_explicit_invalid_key(monkeypatch):
    import httpx

    calls = {"count": 0}

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            calls["count"] += 1
            request = httpx.Request("POST", url)
            return httpx.Response(
                401,
                request=request,
                json={"error": {"code": "invalid_api_key"}},
            )

    monkeypatch.setattr("nanoma.llm._get_client", lambda timeout=180.0: FakeClient())
    with pytest.raises(httpx.HTTPStatusError):
        await openai_compatible_call(
            [{"role": "user", "content": "hello"}],
            "test-model",
            base_url="https://example.invalid/v1",
            api_key="test",
            retry_config=RetryConfig(max_retries=2, base_delay=0.1, max_delay=0.1),
        )
    assert calls["count"] == 1


# ─── Test: Full integration (spawn + message + wait) ─────────────────────────

@pytest.mark.asyncio
async def test_same_model_judge_spawn_and_wait(tmp_workspace, monkeypatch):
    """Planning judge spawns a child; parent waits and receives its result."""
    monkeypatch.setenv("NANOMA_NODE_AUTONOMOUS_PLANNING", "1")
    turn_count = {"parent": 0, "child": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        if tools is None:
            return LLMResponse(
                content=(
                    '{"spawn":true,"reasoning":"parallel child useful",'
                    '"subagents":[{"subject":"child task","role":"worker",'
                    '"task":"child task"}]}'
                ),
                usage=UsageRecord(input_tokens=50, output_tokens=30, model=model),
            )
        # Detect if this is a child (task contains "child")
        system = messages[0]["content"] if messages else ""
        if "child task" in system:
            turn_count["child"] += 1
            return LLMResponse(
                tool_calls=[ToolCall(id="c1", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=50, output_tokens=30, model=model),
            )
        else:
            turn_count["parent"] += 1
            if turn_count["parent"] == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(
                        id="p1",
                        name="task_create",
                        arguments={"subject": "child task"},
                    )],
                    usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
                )
            elif turn_count["parent"] == 2:
                return LLMResponse(
                    tool_calls=[ToolCall(id="p2", name="wait", arguments={"timeout": 5})],
                    usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
                )
            else:
                return LLMResponse(
                    tool_calls=[ToolCall(id="p3", name="set_status", arguments={"status": "done", "result": "parent done"})],
                    usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
                )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None)
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("parent task")
    assert result == "parent done"
    assert len(rt.agents) == 2


@pytest.mark.asyncio
async def test_bio_discovery(tmp_workspace):
    """Agents can discover each other via bio."""
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="t1", name="set_status", arguments={"status": "done", "result": "ok"})],
            usage=UsageRecord(input_tokens=50, output_tokens=30, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None)
    rt = Runtime(config=config, llm_call=mock_llm)
    a = rt.create_agent("worker A")
    b = rt.create_agent("worker B")
    a.bio = "I handle file parsing"
    b.bio = "I do testing"

    # Agent C queries all
    c = rt.create_agent("coordinator")
    result = await meta_query({}, c, rt)
    bios = [x["bio"] for x in result["agents"]]
    assert "I handle file parsing" in bios
    assert "I do testing" in bios
