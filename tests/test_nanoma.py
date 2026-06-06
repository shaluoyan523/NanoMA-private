"""Tests for NanoMA framework — no real LLM calls, fully deterministic."""

import json

import pytest

from nanoma.core import Envelope, ResourceQuota, Runtime, RuntimeConfig, ToolContext
from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import LLMResponse, ToolCall, estimate_tokens, count_message_tokens
from nanoma.memory import ActiveTaskCard
from nanoma.meta import (
    META_TOOLS,
    meta_batch,
    meta_compact,
    meta_create_agent,
    meta_get_cost,
    meta_kill,
    meta_query,
    meta_rebirth,
    meta_send,
    meta_set_bio,
    meta_set_status,
    meta_spawn,
    meta_submit,
    meta_transfer,
    meta_wait,
)
from nanoma.scheduler import Scheduler
from nanoma.sandbox import SandboxConfig, SandboxSession


@pytest.fixture
def tmp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "shared").mkdir()
    return ws


@pytest.fixture
def runtime(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "test done"})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, max_agents=50, log_dir=None, sandbox_backend="host")
    return Runtime(config=config, llm_call=mock_llm)


@pytest.fixture
def runtime_multi_turn(tmp_workspace):
    call_count = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return LLMResponse(
                tool_calls=[ToolCall(id=f"tc{call_count['n']}", name="set_status", arguments={"status": "done", "result": f"done after {call_count['n']} turns"})],
                usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id=f"tc{call_count['n']}", name="get_cost", arguments={})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None, sandbox_backend="host")
    return Runtime(config=config, llm_call=mock_llm)


@pytest.mark.asyncio
async def test_basic_run(runtime):
    result = await runtime.run("Say hello")
    assert result == "test done"
    agent = list(runtime.agents.values())[0]
    assert agent.status == "done"
    assert agent.action_state == "stop"


@pytest.mark.asyncio
async def test_multi_turn(runtime_multi_turn):
    result = await runtime_multi_turn.run("Do something complex")
    assert "done after 3 turns" in result


def test_id_generation():
    from nanoma.core import IdGenerator

    gen = IdGenerator()
    ids = [gen.next() for _ in range(30)]
    assert ids[0] == "alpha"
    assert ids[25] == "zulu"
    assert ids[26] == "alpha-1"
    assert len(set(ids)) == 30


def test_quota_defaults():
    q = ResourceQuota(budget=10.0, time_limit=60.0, max_turns=100)
    assert q.budget == 10.0
    assert q.time_limit == 60.0
    assert q.max_turns == 100


def test_runtime_config_uses_codex_sandbox_by_default():
    assert RuntimeConfig().sandbox_backend == "codex"


def test_ledger():
    ledger = CostLedger(total_budget=5.0)
    usage = UsageRecord(input_tokens=1000, output_tokens=500, model="test")
    ledger.record("agent-1", usage)
    assert ledger.total_spent > 0
    assert "agent-1" in ledger.per_agent


@pytest.mark.asyncio
async def test_scheduler():
    s = Scheduler(max_concurrent=2)
    await s.acquire()
    assert s.stats["active"] == 1
    s.release()
    assert s.stats["active"] == 0


@pytest.mark.asyncio
async def test_message_delivery(runtime):
    agent = runtime.create_agent("test task")
    env = Envelope(from_id="other", to_id=agent.id, content="hello", tokens=5, timestamp=0.0, mode="queue")
    await runtime.deliver(env)
    assert not agent._queue_inbox.empty()


@pytest.mark.asyncio
async def test_idle_wake(runtime):
    agent = runtime.create_agent("test")
    agent.status = "idle"
    env = Envelope(from_id="x", to_id=agent.id, content="wake up", tokens=5, timestamp=0.0, mode="queue")
    await runtime.deliver(env)
    assert agent.status == "running"


@pytest.mark.asyncio
async def test_meta_spawn_peer_metadata_and_state_board(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn(
        {
            "task": "child task",
            "role": "researcher",
            "create_type": "peer_agent",
            "relationship": "peer",
            "group_id": "g1",
            "workflow_prior": "research_loop",
            "current_task_tags": ["analysis", "docs"],
        },
        parent,
        runtime,
    )
    child = runtime.agents[result["agent_id"]]
    assert child.parent == parent.id
    assert child.role == "researcher"
    assert child.create_type == "peer_agent"
    assert child.created_by == parent.id
    assert child.workflow_prior == "research_loop"
    assert runtime.state_board_get(child.id)["action_state"] == "create"
    assert runtime.memory.serialize(child.id)["active_task"]["tags"] == ["analysis", "docs"]


@pytest.mark.asyncio
async def test_meta_create_agent_alias(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_create_agent({"task": "child task", "workflow_prior": "critic_review_loop"}, parent, runtime)
    assert result["agent_id"] in runtime.agents


@pytest.mark.asyncio
async def test_meta_spawn_max_depth(runtime):
    parent = runtime.create_agent("parent")
    parent.depth = runtime.config.max_depth
    result = await meta_spawn({"task": "child"}, parent, runtime)
    assert "error" in result


@pytest.mark.asyncio
async def test_meta_kill_emergency_only(runtime):
    parent = runtime.create_agent("parent")
    child = runtime.create_agent("child", parent=parent.id)
    parent.children.add(child.id)
    result = await meta_kill({"agent_id": child.id}, parent, runtime)
    assert "error" in result
    result = await meta_kill({"agent_id": child.id, "emergency": True}, parent, runtime)
    assert result["killed"] == child.id
    assert child.status == "done"


@pytest.mark.asyncio
async def test_meta_kill_permission(runtime):
    a = runtime.create_agent("a")
    b = runtime.create_agent("b")
    result = await meta_kill({"agent_id": b.id, "emergency": True}, a, runtime)
    assert "error" in result


@pytest.mark.asyncio
async def test_meta_send_structured_and_stop_request(runtime):
    a = runtime.create_agent("sender")
    b = runtime.create_agent("receiver")
    result = await meta_send(
        {
            "to": b.id,
            "message": "please stop",
            "message_type": "stop_request",
            "payload": {"reason": "done"},
            "requires_ack": True,
            "urgency": "high",
            "mode": "queue",
        },
        a,
        runtime,
    )
    assert result["delivered"] == 1
    msg = b._queue_inbox.get_nowait()
    assert msg.message_type == "stop_request"
    assert msg.payload == {"reason": "done"}
    assert msg.requires_ack is True


@pytest.mark.asyncio
async def test_meta_send_no_broadcast(runtime):
    a = runtime.create_agent("sender")
    result = await meta_send({"to": "*", "message": "hi"}, a, runtime)
    assert result["delivered"] == 0


@pytest.mark.asyncio
async def test_meta_query_all_and_single(runtime):
    target = runtime.create_agent("target", role="builder", create_type="peer_agent", relationship="peer", created_by="alpha", group_id="g2", workflow_prior="synthesis_loop")
    target.bio = "I am a coder"
    runtime.state_board_update(target.id, action_state="work", current_task_tags=["backend"], work_outline="implement API")
    runtime.memory.update(target.id, public_summary="building API")
    target.memory = {"public_memory": runtime.memory.serialize(target.id)}
    q = runtime.create_agent("querier")
    result = await meta_query({"agent_id": target.id}, q, runtime)
    assert result["role"] == "builder"
    assert result["action_state"] == "work"
    assert result["state_board"]["work_outline"] == "implement API"
    assert result["public_memory"]["public_summary"] == "building API"
    all_result = await meta_query({}, q, runtime)
    assert all_result["count"] == len(runtime.agents)
    assert "state_board" in all_result
    assert target.id in all_result["public_memory"]


@pytest.mark.asyncio
async def test_meta_set_bio_and_get_cost(runtime):
    a = runtime.create_agent("worker")
    await meta_set_bio({"bio": "I handle parsing"}, a, runtime)
    a._turns = 5
    a.tokens_consumed = 1000
    result = await meta_get_cost({}, a, runtime)
    assert result["bio"] == "I handle parsing"
    assert result["action_state"] == a.action_state


@pytest.mark.asyncio
async def test_meta_set_status_state_board_work_and_unknown_action(runtime):
    a = runtime.create_agent("worker")
    result = await meta_set_status({"action": "work", "current_task_tags": ["parse"], "work_outline": "step1 -> step2"}, a, runtime)
    assert result["action_state"] == "work"
    assert runtime.state_board_get(a.id)["work_outline"] == "step1 -> step2"
    bad = await meta_set_status({"action": "weird"}, a, runtime)
    assert "error" in bad


@pytest.mark.asyncio
async def test_meta_set_status_self_stop(runtime):
    a = runtime.create_agent("worker")
    result = await meta_set_status({"action": "self_stop", "result": "finished"}, a, runtime)
    assert result["status"] == "done"
    assert a.action_state == "stop"
    assert runtime.memory.serialize(a.id)["active_task"] is None


@pytest.mark.asyncio
async def test_meta_compact_and_query_public_memory(runtime):
    a = runtime.create_agent("worker")
    a.history.append({"role": "user", "content": "long context"})
    result = await meta_compact({"summary": "condensed", "tags": ["research"], "experience": "did work", "work_outline": "outline"}, a, runtime)
    assert result["scheduled"] is True
    runtime._execute_compact(a)
    queried = await meta_query({"agent_id": a.id}, a, runtime)
    assert queried["public_memory"]["public_summary"] == "condensed"
    assert queried["public_memory"]["experience_cards"][0]["summary"] == "did work"
    assert queried["state_board"]["action_state"] == "compact"


@pytest.mark.asyncio
async def test_memory_broker_tag_read(runtime):
    a = runtime.create_agent("worker")
    runtime.memory.update(a.id, active_task=ActiveTaskCard(task="worker", tags=["python", "tests"]))
    read = runtime.memory.read(intent="find testers", seed_terms=["tests"])
    assert read["matches"][0]["agent_id"] == a.id
    assert "tests" in read["known_tags"]


@pytest.mark.asyncio
async def test_meta_rebirth_syncs_memory(runtime):
    a = runtime.create_agent("worker")
    result = await meta_rebirth({"summary": "Did steps 1-5", "new_bio": "updated bio", "tags": ["reborn"], "experience": "first pass"}, a, runtime)
    assert result["scheduled"]
    runtime._execute_rebirth(a)
    assert a.bio == "updated bio"
    assert runtime.memory.serialize(a.id)["public_summary"] == "Did steps 1-5"
    assert runtime.memory.serialize(a.id)["experience_cards"][0]["summary"] == "first pass"


@pytest.mark.asyncio
async def test_compact_before_stop(runtime):
    a = runtime.create_agent("worker")
    result = await meta_set_status({"action": "stop", "result": "finished", "compact_before_stop": True, "compact_summary": "final compact", "current_task_tags": ["done"]}, a, runtime)
    assert result["scheduled"] is True
    runtime._execute_compact(a)
    assert a.status == "done"
    assert runtime.memory.serialize(a.id)["public_summary"] == "final compact"
    assert runtime.memory.serialize(a.id)["active_task"] is None
    assert len(runtime.memory.serialize(a.id)["experience_cards"]) == 1


@pytest.mark.asyncio
async def test_meta_submit_and_batch(runtime):
    a = runtime.create_agent("worker")
    test_file = a.workspace / "output.txt"
    test_file.write_text("result data")
    result = await meta_submit({"path": "output.txt", "description": "final output"}, a, runtime)
    assert result["submitted"] == "output.txt"
    assert "output.txt" in runtime.memory.serialize(a.id)["artifact_index"]

    batch_data = [
        {"tool": "file_write", "args": {"path": "hello.txt", "content": "world"}},
        {"tool": "compact", "args": {"summary": "batched compact"}},
        {"tool": "nonexistent_tool", "args": {}},
    ]
    batch_file = a.workspace / "batch.json"
    batch_file.write_text(json.dumps(batch_data))
    batch_result = await meta_batch({"path": "batch.json"}, a, runtime)
    assert batch_result["executed"] == 3
    assert "error" in batch_result["results"][2]


@pytest.mark.asyncio
async def test_meta_transfer_push_pull_shared(runtime):
    a = runtime.create_agent("sender")
    b = runtime.create_agent("receiver")
    (a.workspace / "data.txt").write_text("content")
    push = await meta_transfer({"src": "data.txt", "to": b.id}, a, runtime)
    assert "data.txt" in push["pushed"]
    puller = runtime.create_agent("puller")
    pull = await meta_transfer({"src": "data.txt", "from_agent": b.id}, puller, runtime)
    assert "data.txt" in pull["pulled"]
    shared = await meta_transfer({"src": "data.txt", "to": "shared"}, a, runtime)
    assert shared["to"] == "shared"


@pytest.mark.asyncio
async def test_meta_wait_immediate(runtime):
    parent = runtime.create_agent("parent")
    child = runtime.create_agent("child", parent=parent.id)
    parent.children.add(child.id)
    child.status = "done"
    child.result = "child result"
    result = await meta_wait({}, parent, runtime)
    assert len(result["completed"]) == 1


@pytest.mark.asyncio
async def test_tool_file_write_read(tmp_workspace):
    from nanoma.tools import tool_file_read, tool_file_write

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_file_write({"path": "test.txt", "content": "hello world"}, ws, ctx)
    assert result["bytes"] == 11
    result = await tool_file_read({"path": "test.txt"}, ws, ctx)
    assert result["content"] == "hello world"


@pytest.mark.asyncio
async def test_tool_file_read_sandbox(tmp_workspace):
    from nanoma.tools import tool_file_read

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_file_read({"path": "/etc/passwd"}, ws, ctx)
    assert "error" in result


@pytest.mark.asyncio
async def test_tool_file_list(tmp_workspace):
    from nanoma.tools import tool_file_list

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (ws / "a.txt").write_text("a")
    (ws / "b.txt").write_text("b")
    result = await tool_file_list({"path": "."}, ws, ctx)
    names = [e["name"] for e in result["entries"]]
    assert "a.txt" in names
    assert "b.txt" in names


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
async def test_tool_shell_uses_runtime_sandbox(tmp_workspace):
    from nanoma.tools import tool_shell

    class FakeSandbox:
        def __init__(self):
            self.calls = []

        async def exec(self, cmd, workspace, shared_dir, timeout=30):
            self.calls.append((cmd, workspace, shared_dir, timeout))
            return {"exit_code": 0, "stdout": "sandboxed\n", "stderr": ""}

    fake = FakeSandbox()
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, sandbox=fake)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "echo host", "timeout": 7}, ws, ctx)
    assert result["stdout"] == "sandboxed\n"
    assert fake.calls == [("echo host", ws, tmp_workspace / "shared", 7)]


@pytest.mark.asyncio
async def test_codex_sandbox_command_shape(tmp_workspace, monkeypatch):
    calls = []

    async def fake_communicate(argv, *, timeout, cwd=None, env=None):
        calls.append({"argv": argv, "timeout": timeout, "cwd": cwd, "env": env})
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    import nanoma.sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "_communicate", fake_communicate)
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: "/usr/bin/codex")
    session = SandboxSession(SandboxConfig(backend="codex", codex_bin="codex"), tmp_workspace)
    session._started = True
    session.backend = "codex"
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await session.exec("echo ok", ws, tmp_workspace / "shared", 9)
    assert result["exit_code"] == 0
    assert calls[0]["argv"][:4] == ["codex", "sandbox", "--permissions-profile", ":workspace"]
    assert calls[0]["argv"][4:6] == ["-C", str(ws.resolve())]
    assert "/usr/bin/env" in calls[0]["argv"]
    assert "/bin/sh" in calls[0]["argv"]
    command_env = [part for part in calls[0]["argv"] if "=" in part]
    assert f"WORKSPACE={ws.resolve()}" in command_env
    assert f"SHARED={(tmp_workspace / 'shared').resolve()}" in command_env
    assert not any(part.startswith("OPENAI_API_KEY=") for part in command_env)
    assert "CODEX_HOME" in calls[0]["env"]


@pytest.mark.asyncio
async def test_tool_grep(tmp_workspace):
    from nanoma.tools import tool_grep

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (ws / "code.py").write_text("def hello():\n    return 42\n")
    result = await tool_grep({"pattern": "hello", "path": "."}, ws, ctx)
    assert result["count"] >= 1


def test_estimate_tokens():
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("a" * 400) == 100


def test_count_message_tokens():
    msgs = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello there"}]
    tokens = count_message_tokens(msgs)
    assert tokens > 0


@pytest.mark.asyncio
async def test_spawn_and_wait(tmp_workspace):
    turn_count = {"parent": 0, "child": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        system = messages[0]["content"] if messages else ""
        if "child task" in system:
            turn_count["child"] += 1
            return LLMResponse(
                tool_calls=[ToolCall(id="c1", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=50, output_tokens=30, model=model),
            )
        turn_count["parent"] += 1
        if turn_count["parent"] == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="p1", name="spawn", arguments={"task": "child task", "create_type": "peer_agent", "workflow_prior": "research_loop"})],
                usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
            )
        if turn_count["parent"] == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="p2", name="wait", arguments={"timeout": 5})],
                usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="p3", name="set_status", arguments={"status": "done", "result": "parent done"})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("parent task")
    assert result == "parent done"
    assert len(rt.agents) == 2


@pytest.mark.asyncio
async def test_bio_discovery(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="t1", name="set_status", arguments={"status": "done", "result": "ok"})],
            usage=UsageRecord(input_tokens=50, output_tokens=30, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config, llm_call=mock_llm)
    a = rt.create_agent("worker A")
    b = rt.create_agent("worker B")
    a.bio = "I handle file parsing"
    b.bio = "I do testing"
    c = rt.create_agent("coordinator")
    result = await meta_query({}, c, rt)
    bios = [x["bio"] for x in result["agents"]]
    assert "I handle file parsing" in bios
    assert "I do testing" in bios
    assert "compact" in META_TOOLS
