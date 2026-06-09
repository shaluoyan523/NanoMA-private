"""Tests for NanoMA framework — no real LLM calls, fully deterministic."""

import asyncio
import json
import subprocess

import pytest

from nanoma.core import Artifact, Envelope, ResourceQuota, Runtime, RuntimeConfig, ToolContext
from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import LLMResponse, ToolCall, estimate_tokens, count_message_tokens, _parse_text_tool_calls
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
    meta_spawn_many,
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


def _init_git_source(workspace, files: dict[str, str] | None = None):
    source = workspace / "shared" / "source"
    source.mkdir(parents=True, exist_ok=True)
    for rel_path, content in (files or {"README.md": "baseline\n"}).items():
        path = source / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init"], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return source


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
async def test_runtime_bootstrap_batch_runs_before_root_llm(tmp_workspace):
    seen = {"bootstrap_seen": False}

    async def mock_llm(messages, model, tools=None, **kwargs):
        seen["bootstrap_seen"] = any("Bootstrap batch executed" in (m.get("content") or "") for m in messages)
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "root done"})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    batch_file = tmp_workspace / "shared" / "bootstrap.json"
    batch_file.write_text(json.dumps([
        {"tool": "spawn", "args": {"task": "child 1", "role": "worker", "group_id": "g"}},
        {"tool": "spawn", "args": {"task": "child 2", "role": "worker", "group_id": "g"}},
    ]))
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_agents=10,
        log_dir=None,
        sandbox_backend="host",
        bootstrap_batch_file="shared/bootstrap.json",
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("root")
    assert result == "root done"
    assert len(rt.agents) == 3
    assert sum(1 for e in rt._events if e["event"] == "spawn") == 2
    assert seen["bootstrap_seen"] is True


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


def test_runtime_config_orchestration_defaults():
    config = RuntimeConfig()
    assert config.orchestration_preference == "balanced"
    assert config.child_orchestration_preference is None
    assert config.shell_mode == "controlled"
    assert config.spawn_before_turn == 2


def test_shared_source_paths_are_not_expected_output_artifacts(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Read `shared/source/examples/opendeepthink_fixed_runner.py` and write `shared/final.md`."
    )

    assert rt.expected_outputs(agent) == ["shared/final.md"]


def test_cf73_public_problem_files_are_not_expected_output_artifacts(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Public files: `shared/cf73_top10/problems/2161F/metadata.json`, "
        "`shared/cf73_top10/problems/2161F/statement.md`, and `shared/cf73_top10/README.md`. "
        "Write `shared/cf73_top10/2161F/final/bt.json`, "
        "`shared/cf73_top10/2161F/selected_solution.cpp`, and "
        "`shared/cf73_top10/2161F/report.md`."
    )

    assert rt.expected_outputs(agent) == [
        "shared/cf73_top10/2161F/final/bt.json",
        "shared/cf73_top10/2161F/selected_solution.cpp",
        "shared/cf73_top10/2161F/report.md",
    ]


def test_bare_shared_artifact_paths_are_expected_outputs(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Write shared/cf73_top10/2161F/gen0/candidate_13.cpp and "
        "shared/cf73_top10/2161F/gen0/candidate_13.md. "
        "Read public shared/cf73_top10/problems/2161F/statement.md first."
    )

    assert rt.expected_outputs(agent) == [
        "shared/cf73_top10/2161F/gen0/candidate_13.cpp",
        "shared/cf73_top10/2161F/gen0/candidate_13.md",
    ]


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
            "orchestration_preference": "parallel",
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
    assert child.orchestration_preference == "parallel"
    assert runtime.state_board_get(child.id)["action_state"] == "create"
    assert runtime.memory.serialize(child.id)["active_task"]["tags"] == ["analysis", "docs"]


@pytest.mark.asyncio
async def test_meta_create_agent_alias(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_create_agent({"task": "child task", "workflow_prior": "critic_review_loop"}, parent, runtime)
    assert result["agent_id"] in runtime.agents


@pytest.mark.asyncio
async def test_meta_spawn_many_creates_peer_wave(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn_many(
        {
            "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "wave-1", "orchestration_preference": "solo"},
            "agents": [
                {"task": "check requirements", "role": "requirements", "current_task_tags": ["phase:req"]},
                {"task": "write tests", "role": "tester", "current_task_tags": ["phase:test"]},
            ],
        },
        parent,
        runtime,
    )

    assert result["created"] == 2
    children = [runtime.agents[item["result"]["agent_id"]] for item in result["results"]]
    assert {child.role for child in children} == {"requirements", "tester"}
    assert all(child.create_type == "peer_agent" for child in children)
    assert all(child.group_id == "wave-1" for child in children)
    assert all(child.orchestration_preference == "solo" for child in children)


@pytest.mark.asyncio
async def test_meta_spawn_many_accepts_json_string_args(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn_many(
        {
            "defaults": json.dumps({"create_type": "peer_agent", "relationship": "peer", "group_id": "wave-json"}),
            "agents": json.dumps([
                {"task": "check requirements", "role": "requirements"},
                {"task": "write tests", "role": "tester"},
            ]),
        },
        parent,
        runtime,
    )

    assert result["created"] == 2
    children = [runtime.agents[item["result"]["agent_id"]] for item in result["results"]]
    assert {child.role for child in children} == {"requirements", "tester"}
    assert all(child.group_id == "wave-json" for child in children)


@pytest.mark.asyncio
async def test_meta_spawn_many_rejects_placeholder_tasks(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn_many(
        {"tasks": ["I'm sorry, but I cannot provide the content you're looking for."]},
        parent,
        runtime,
    )

    assert result["created"] == 0
    assert "placeholder" in result["results"][0]["result"]["error"]
    assert len(runtime.agents) == 1


@pytest.mark.asyncio
async def test_meta_spawn_many_accepts_top_level_defaults_when_defaults_malformed(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn_many(
        {
            "defaults": "<bad defaults string",
            "create_type": "peer_agent",
            "relationship": "peer",
            "group_id": "wave-top",
            "current_task_tags": ["phase:gen0"],
            "tasks": ["Generate candidate 00", "Generate candidate 01"],
        },
        parent,
        runtime,
    )

    assert result["created"] == 2
    children = [runtime.agents[item["result"]["agent_id"]] for item in result["results"]]
    assert all(child.group_id == "wave-top" for child in children)
    assert all(child.create_type == "peer_agent" for child in children)


@pytest.mark.asyncio
async def test_child_orchestration_preference_task_adaptive_keeps_parent_without_task_signal(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        child_orchestration_preference=None,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent("parent")

    result = await meta_spawn({"task": "child"}, parent, rt)
    child = rt.agents[result["agent_id"]]

    assert parent.orchestration_preference == "aggressive"
    assert child.orchestration_preference == "aggressive"
    event = [e for e in rt._events if e["event"] == "agent_new" and e["agent"] == child.id][0]
    assert event["data"]["orchestration_resolution"]["mode"] == "task_adaptive"


@pytest.mark.asyncio
async def test_child_orchestration_preference_can_explicitly_decay_from_parent(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        child_orchestration_preference="inherit_decayed",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("parent")

    result = await meta_spawn({"task": "child"}, parent, rt)
    child = rt.agents[result["agent_id"]]

    assert parent.orchestration_preference == "aggressive"
    assert child.orchestration_preference == "parallel"


@pytest.mark.asyncio
async def test_child_orchestration_preference_task_adaptive_can_increase_for_complex_work(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="balanced",
        child_orchestration_preference=None,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("parent")

    result = await meta_spawn(
        {
            "task": "Run a full-scale multi-agent benchmark with implementation, tests, review, synthesis, and integration.",
            "workflow_prior": "research_loop",
            "current_task_tags": ["benchmark:fixed", "scope:full"],
        },
        parent,
        rt,
    )
    child = rt.agents[result["agent_id"]]

    assert parent.orchestration_preference == "balanced"
    assert child.orchestration_preference == "aggressive"
    assert result["orchestration_preference"] == "aggressive"


@pytest.mark.asyncio
async def test_child_orchestration_preference_task_adaptive_can_decrease_for_atomic_work(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        child_orchestration_preference=None,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("parent")

    result = await meta_spawn({"task": "Quick grep for meta_spawn in one file."}, parent, rt)
    child = rt.agents[result["agent_id"]]

    assert parent.orchestration_preference == "aggressive"
    assert child.orchestration_preference == "solo"
    assert result["orchestration_preference"] == "solo"


@pytest.mark.asyncio
async def test_child_orchestration_preference_explicit_runtime_override(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        child_orchestration_preference="solo",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("parent")
    child = rt.create_agent("child", parent=parent.id)

    assert child.orchestration_preference == "solo"


@pytest.mark.asyncio
async def test_spawn_orchestration_preference_overrides_adaptive_default(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        child_orchestration_preference=None,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("parent")

    result = await meta_spawn({"task": "child", "orchestration_preference": "solo"}, parent, rt)
    child = rt.agents[result["agent_id"]]

    assert child.orchestration_preference == "solo"


@pytest.mark.asyncio
async def test_explicit_aggressive_leaf_artifact_lane_is_downgraded_to_solo(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        child_orchestration_preference=None,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate candidate generation.")

    result = await meta_spawn(
        {
            "task": (
                "Write a C++17 solution for Codeforces problem 2161F. "
                "Write shared/cf73_top10/2161F/gen0/candidate_13.cpp and "
                "shared/cf73_top10/2161F/gen0/candidate_13.md."
            ),
            "role": "generator",
            "group_id": "cf73-2161f-gen0",
            "current_task_tags": ["benchmark:cf73", "problem:2161F", "phase:gen0", "role:generator", "candidate:13"],
            "orchestration_preference": "aggressive",
        },
        parent,
        rt,
    )
    child = rt.agents[result["agent_id"]]

    assert result["requested_orchestration_preference"] == "aggressive"
    assert child.orchestration_preference == "solo"
    assert result["orchestration_preference"] == "solo"


def test_orchestration_preference_prompt_sections(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", orchestration_preference="parallel")
    rt = Runtime(config=config)
    agent = rt.create_agent("complex task")

    assert agent.orchestration_preference == "parallel"
    assert "Orchestration preference: parallel" in agent.history[0]["content"]
    assert "spawn_many()" in agent.history[0]["content"]


def test_system_prompt_encourages_autonomous_peer_query_on_bottleneck(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("complex task")
    prompt = agent.history[0]["content"]

    assert "If you hit a bottleneck" in prompt
    assert "proactively query nearby peers" in prompt
    assert "status:pruned" in prompt
    assert "reason:peer_ahead" in prompt


@pytest.mark.asyncio
async def test_orchestration_nudge_sent_once_for_parallel_agent(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="get_cost", arguments={})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=2,
        max_solo_tool_calls_before_spawn=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("complex task")
    agent._turns = 2
    agent._tool_calls = 2

    rt._check_orchestration_nudge(agent)
    rt._check_orchestration_nudge(agent)

    nudges = [e for e in rt._events if e["event"] == "orchestration_nudge"]
    assert len(nudges) == 1
    assert not agent._steer_inbox.empty()


@pytest.mark.asyncio
async def test_spawnable_parallel_task_gets_create_action_scope(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="get_cost", arguments={})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=1,
        min_spawnable_workstreams=3,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run(
        "Audit and patch shared/source with separable workstreams: requirements audit, implementation patch, "
        "focused tests, and risk review."
    )

    assert {"spawn_many", "spawn", "create_agent", "query", "compact", "set_status", "get_cost"} == set(calls[0]["tools"])
    assert "file_read" not in calls[0]["tools"]
    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "Current action is create" in card
    assert [e for e in rt._events if e["event"] == "orchestration_nudge" and e["data"].get("spawnable_task")]


@pytest.mark.asyncio
async def test_create_action_scope_persists_until_spawn_or_solo_decision(tmp_workspace):
    calls = []
    root_calls = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        task_prompt = messages[0].get("content", "")
        if "Task: audit source" in task_prompt or "Task: patch source" in task_prompt or "Task: test source" in task_prompt:
            return LLMResponse(
                tool_calls=[ToolCall(id="child-done", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        root_calls["n"] += 1
        calls.append([t["function"]["name"] for t in (tools or [])])
        if root_calls["n"] == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="get_cost", arguments={})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if root_calls["n"] == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="get_cost", arguments={})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if root_calls["n"] == 3:
            return LLMResponse(
                tool_calls=[ToolCall(
                    id="tc3",
                    name="spawn_many",
                    arguments={
                        "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "wave", "orchestration_preference": "solo"},
                        "agents": [
                            {"task": "audit source", "role": "auditor"},
                            {"task": "patch source", "role": "patcher"},
                            {"task": "test source", "role": "tester"},
                        ],
                    },
                )],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc4",
                name="set_status",
                arguments={"status": "done", "result": "root done"},
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=4,
        max_agents=10,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Patch shared/source with separable audit, patch, test, and review workstreams.")

    create_tools = {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
    assert set(calls[0]) == create_tools
    assert set(calls[1]) == {"spawn", "create_agent", "spawn_many"}
    assert set(calls[2]) == {"spawn", "create_agent", "spawn_many"}
    assert len(rt.agents) == 4


@pytest.mark.asyncio
async def test_create_miss_retries_with_spawn_tools_only(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        task_prompt = messages[0].get("content", "")
        if "Task: audit source" in task_prompt or "Task: patch source" in task_prompt:
            return LLMResponse(
                tool_calls=[ToolCall(id="child-done", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
            "tool_choice": kwargs.get("tool_choice"),
        })
        if len(calls) == 1:
            return LLMResponse(
                content="I should create a directory first, then spawn agents.",
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc2",
                name="spawn_many",
                arguments={
                    "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "wave"},
                    "agents": [
                        {"task": "audit source", "role": "auditor"},
                        {"task": "patch source", "role": "patcher"},
                    ],
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=4,
        max_agents=10,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Patch shared/source with separable audit and patch workstreams.")

    assert set(calls[0]["tools"]) == {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "spawn_many"}}
    card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "retry_spawn_after_create_miss" in card
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_set_status_create_counts_as_create_miss_until_spawn(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        task_prompt = messages[0].get("content", "")
        if "Task: audit source" in task_prompt or "Task: patch source" in task_prompt:
            return LLMResponse(
                tool_calls=[ToolCall(id="child-done", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"action": "create"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc2",
                name="spawn_many",
                arguments={
                    "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "wave"},
                    "agents": [
                        {"task": "audit source", "role": "auditor"},
                        {"task": "patch source", "role": "patcher"},
                    ],
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=4,
        max_agents=10,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Patch shared/source with separable audit and patch workstreams.")

    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "spawn_many"}}
    assert [e for e in rt._events if e["event"] == "create_action_miss" and e["data"].get("reason") == "no_child_created_in_create_action"]
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_degraded_spawn_many_result_keeps_create_retry(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(
                id=f"tc{len(calls)}",
                name="spawn_many",
                arguments={"tasks": ["I'm sorry, but I cannot provide the content you're looking for."]},
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=2,
        max_agents=10,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Patch shared/source with separable audit and patch workstreams.")

    assert len(rt.agents) == 1
    assert calls[0]["tool_choice"] is None
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "spawn_many"}}
    assert [e for e in rt._events if e["event"] == "create_action_miss" and e["data"].get("reason") == "spawn_tool_created_no_valid_children"]


@pytest.mark.asyncio
async def test_set_status_read_revalidates_create_scope_next_turn(tmp_workspace):
    (tmp_workspace / "shared" / "input.txt").write_text("hello")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(
                    id="tc1",
                    name="set_status",
                    arguments={"action": "read", "status": "reading", "current_task_tags": ["phase:audit"]},
                )],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 2:
            content = (
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"read_file\">\n"
                "<｜DSML｜parameter name=\"filePath\" string=\"true\">shared/input.txt</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            )
            return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        return LLMResponse(
            tool_calls=[ToolCall(id="tc3", name="set_status", arguments={"status": "done", "result": "read done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=4,
        min_spawnable_workstreams=3,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Audit `shared/input.txt` with separable research, review, and report workstreams.")

    create_tools = {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
    assert set(calls[0]["tools"]) == create_tools
    assert {"file_read", "file_list", "grep", "query", "set_status", "get_cost"} <= set(calls[1]["tools"])
    assert "spawn_many" not in calls[1]["tools"]
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered"]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_read"]


@pytest.mark.asyncio
async def test_filtered_read_intent_releases_create_scope_next_turn(tmp_workspace):
    (tmp_workspace / "shared" / "input.txt").write_text("hello")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        if len(calls) == 1:
            content = (
                "I need source context before deciding whether to spawn.\n"
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"read\">\n"
                "<｜DSML｜parameter name=\"file_path\" string=\"true\">shared/input.txt</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            )
            return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="file_read", arguments={"path": "shared/input.txt"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc3", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=4,
        min_spawnable_workstreams=3,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Coordinate separable research, review, and report workstreams. Inspect `shared/input.txt` before deciding.")

    create_tools = {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
    assert set(calls[0]) == create_tools
    assert {"file_read", "file_list", "grep", "query", "set_status", "get_cost"} <= set(calls[1])
    assert "spawn_many" not in calls[1]
    assert [e for e in rt._events if e["event"] == "create_action_miss"]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_read"]


@pytest.mark.asyncio
async def test_create_read_orientation_resumes_create_before_artifact_work(tmp_workspace):
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F").mkdir(parents=True)
    (tmp_workspace / "shared" / "cf73_top10" / "README.md").write_text("public benchmark\n")
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F" / "metadata.json").write_text("{}\n")
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F" / "statement.md").write_text("statement\n")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        if "Task: Generate candidate" in (messages[0].get("content") or ""):
            return LLMResponse(
                tool_calls=[ToolCall(id=f"child-{len(calls)}", name="set_status", arguments={"status": "done", "result": "candidate done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 1:
            content = (
                "I need to read public inputs first.\n"
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"read\">\n"
                "<｜DSML｜parameter name=\"filePath\" string=\"true\">shared/cf73_top10/problems/2161F/statement.md</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            )
            return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="file_read", arguments={"path": "shared/cf73_top10/problems/2161F/statement.md"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc3",
                name="spawn_many",
                arguments={
                    "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "cf73-2161F-gen0"},
                    "agents": [
                        {"task": "Generate candidate 00", "role": "generator-00"},
                        {"task": "Generate candidate 01", "role": "generator-01"},
                    ],
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        max_turns=3,
        max_agents=10,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    await rt.run(
        "Run OpenDeepThink n=20 K=4 T=3 M=10 for CF73 problem 2161F. "
        "Public files: `shared/cf73_top10/problems/2161F/metadata.json`, "
        "`shared/cf73_top10/problems/2161F/statement.md`, `shared/cf73_top10/README.md`. "
        "Spawn 20 independent gen-0 generators. Write `shared/cf73_top10/2161F/final/bt.json`, "
        "`shared/cf73_top10/2161F/selected_solution.cpp`, and `shared/cf73_top10/2161F/report.md`."
    )

    create_tools = {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
    assert set(calls[0]["tools"]) == create_tools
    assert "file_read" in calls[1]["tools"]
    assert set(calls[2]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    card = "\n".join(m.get("content") or "" for m in calls[2]["messages"])
    assert "resume_create_after_read_orientation" in card
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_create_action_recovers_spawn_many_typo_from_text(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        if "Task: Generate candidate" in (messages[0].get("content") or ""):
            return LLMResponse(
                tool_calls=[ToolCall(id=f"child-{len(calls)}", name="set_status", arguments={"status": "done", "result": "candidate done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        content = (
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"spaw_many\">\n"
            "<｜DSML｜parameter name=\"defaults\" json=\"true\">{\"create_type\":\"peer_agent\",\"relationship\":\"peer\",\"group_id\":\"wave\"}</｜DSML｜parameter>\n"
            "<｜DSML｜parameter name=\"agents\" json=\"true\">[{\"task\":\"Generate candidate 00\",\"role\":\"generator-00\"},{\"task\":\"Generate candidate 01\",\"role\":\"generator-01\"}]</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        max_turns=2,
        max_agents=10,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    await rt.run("Coordinate separable generator workstreams for candidates 00 and 01.")

    create_tools = {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
    assert set(calls[0]) == create_tools
    assert len(rt.agents) == 3
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered" and "spawn_many" in e["data"]["tool_calls"]]
    assert [e for e in rt._events if e["event"] == "spawn"]


@pytest.mark.asyncio
async def test_set_status_read_from_create_resumes_create_after_orientation(tmp_workspace):
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F").mkdir(parents=True)
    (tmp_workspace / "shared" / "cf73_top10" / "README.md").write_text("public benchmark\n")
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F" / "metadata.json").write_text("{}\n")
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F" / "statement.md").write_text("statement\n")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        if "Task: Generate candidate" in (messages[0].get("content") or ""):
            return LLMResponse(
                tool_calls=[ToolCall(id=f"child-{len(calls)}", name="set_status", arguments={"status": "done", "result": "candidate done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"action": "read"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="file_read", arguments={"path": "shared/cf73_top10/problems/2161F/statement.md"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc3",
                name="spawn_many",
                arguments={
                    "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "wave"},
                    "agents": [
                        {"task": "Generate candidate 00", "role": "generator-00"},
                        {"task": "Generate candidate 01", "role": "generator-01"},
                    ],
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        max_turns=3,
        max_agents=10,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    await rt.run(
        "Run OpenDeepThink n=20 K=4 T=3 M=10 for CF73 problem 2161F. "
        "Public files: `shared/cf73_top10/problems/2161F/metadata.json`, "
        "`shared/cf73_top10/problems/2161F/statement.md`, `shared/cf73_top10/README.md`. "
        "Spawn 20 independent gen-0 generators. Write `shared/cf73_top10/2161F/final/bt.json`, "
        "`shared/cf73_top10/2161F/selected_solution.cpp`, and `shared/cf73_top10/2161F/report.md`."
    )

    create_tools = {"spawn", "create_agent", "spawn_many", "query", "compact", "set_status", "get_cost"}
    assert set(calls[0]["tools"]) == create_tools
    assert "file_read" in calls[1]["tools"]
    assert set(calls[2]["tools"]) == create_tools
    card = "\n".join(m.get("content") or "" for m in calls[2]["messages"])
    assert "resume_create_after_read_orientation" in card
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_orchestration_nudge_skips_solo_agent(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", orchestration_preference="solo")
    rt = Runtime(config=config)
    agent = rt.create_agent("simple task")
    agent._turns = 99
    agent._tool_calls = 99

    rt._check_orchestration_nudge(agent)

    assert not [e for e in rt._events if e["event"] == "orchestration_nudge"]
    assert agent._steer_inbox.empty()


@pytest.mark.asyncio
async def test_peer_progress_nudge_prompts_query_without_tool_gate(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        return LLMResponse(
            tool_calls=[ToolCall(id=f"tc{len(calls)}", name="get_cost", arguments={})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=8,
        peer_progress_check_after_turns=3,
        peer_progress_check_no_tool_turns=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "worker",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer = rt.create_agent(
        "peer",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer.status = "done"
    peer.artifacts.append(Artifact("shared/out.cpp", tmp_workspace / "shared" / "out.cpp"))
    agent._turns = 3
    agent._no_tool_turns = 2

    await rt._agent_loop(agent)

    assert "query" in calls[0]
    assert "get_cost" in calls[0]
    nudges = [e for e in rt._events if e["event"] == "peer_progress_nudge"]
    assert nudges
    send = [e for e in rt._events if e["event"] == "send" and e["data"].get("message_type") == "peer_progress_nudge"][-1]
    assert "You should usually call query()" in send["data"]["message"]
    assert not any("Peer Progress Check" in (m.get("content") or "") for m in agent.history)
    assert any("Loop Action Card" in (m.get("content") or "") for m in rt._last_llm_messages[agent.id])


def test_peer_progress_nudge_waits_for_leaf_expected_outputs(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        peer_progress_check_after_turns=3,
        peer_progress_check_no_tool_turns=2,
    )
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Write shared/cf73_top10/2161F/gen0/candidate_02.cpp and "
        "shared/cf73_top10/2161F/gen0/candidate_02.md.",
        group_id="wave",
        created_by="root",
        role="generator",
        current_task_tags=["phase:gen0", "role:generator", "candidate:02"],
    )
    peer = rt.create_agent(
        "Write shared/cf73_top10/2161F/gen0/candidate_00.cpp and "
        "shared/cf73_top10/2161F/gen0/candidate_00.md.",
        group_id="wave",
        created_by="root",
        role="generator",
        current_task_tags=["phase:gen0", "role:generator", "candidate:00"],
    )
    peer.status = "done"
    peer.artifacts.append(Artifact("shared/cf73_top10/2161F/gen0/candidate_00.cpp", tmp_workspace / "shared" / "cf73_top10" / "2161F" / "gen0" / "candidate_00.cpp"))
    agent._turns = 4
    agent._no_tool_turns = 2

    rt._check_peer_progress_nudge(agent)

    assert not [e for e in rt._events if e["event"] == "peer_progress_nudge"]
    assert agent._steer_inbox.empty()


@pytest.mark.asyncio
async def test_peer_progress_query_then_pruned_compact_updates_memory(tmp_workspace):
    call_count = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="query", arguments={"filter": {"group_id": "wave"}})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="compact",
                    arguments={
                        "summary": "Pruned myself after peer query: peer already produced the same output lane.",
                        "tags": ["phase:gen0", "problem:X", "role:generator", "status:pruned", "reason:peer_ahead"],
                        "stop_after": True,
                        "result": "pruned duplicate lane",
                    },
                )
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=5,
        peer_progress_check_after_turns=3,
        peer_progress_check_no_tool_turns=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "worker",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer = rt.create_agent(
        "peer",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer.status = "done"
    agent._turns = 3
    agent._no_tool_turns = 2

    await rt._agent_loop(agent)

    memory = rt.memory.serialize(agent.id)
    assert agent.status == "done"
    assert agent.result == "pruned duplicate lane"
    assert memory["public_summary"].startswith("Pruned myself")
    assert "status:pruned" in memory["tags"]
    assert "reason:peer_ahead" in memory["tags"]
    assert memory["active_task"] is None


@pytest.mark.asyncio
async def test_artifact_scope_keeps_query_and_compact_when_peer_progress_pending(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        return LLMResponse(
            tool_calls=[ToolCall(id=f"tc{len(calls)}", name="query", arguments={"filter": {"group_id": "wave"}})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=8,
        peer_progress_check_after_turns=3,
        peer_progress_check_no_tool_turns=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Create `shared/out.cpp`.",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer = rt.create_agent(
        "peer",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer.status = "done"
    peer.artifacts.append(Artifact("shared/peer.cpp", tmp_workspace / "shared" / "peer.cpp"))
    agent._turns = 3
    agent._no_tool_turns = 2
    agent._artifact_nudge_count = 2
    agent._peer_progress_nudge_sent = True
    agent._peer_progress_nudge_turn = 3

    await rt._agent_loop(agent)

    assert {"file_read", "file_write", "query", "compact", "set_status"} <= set(calls[0]["tools"])
    assert any("Loop Action Card" in (m.get("content") or "") for m in calls[0]["messages"])
    assert not any("Artifact Required" in (m.get("content") or "") for m in agent.history)


@pytest.mark.asyncio
async def test_peer_progress_after_query_limits_more_peer_reading(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc1",
                name="compact",
                arguments={
                    "summary": "Pruned duplicate lane after peer query.",
                    "tags": ["phase:gen0", "problem:X", "role:generator", "status:pruned", "reason:peer_ahead"],
                    "stop_after": True,
                    "result": "pruned duplicate lane",
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=8,
        peer_progress_check_after_turns=3,
        peer_progress_check_no_tool_turns=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Create `shared/out.cpp` and `shared/out.md`.",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer = rt.create_agent(
        "Create `shared/peer.cpp` and `shared/peer.md`.",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer.status = "done"
    peer.artifacts.append(Artifact("shared/peer.cpp", tmp_workspace / "shared" / "peer.cpp"))
    agent._turns = 4
    agent._artifact_nudge_count = 1
    agent._peer_progress_nudge_sent = True
    agent._peer_progress_nudge_turn = 3
    agent._last_query_turn = 4

    await rt._agent_loop(agent)

    assert {"file_read", "file_list", "file_write", "compact", "set_status", "query"} <= set(calls[0]["tools"])
    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "Peer-progress query has already returned completed" in card
    assert agent.status == "done"
    assert agent.result == "pruned duplicate lane"


@pytest.mark.asyncio
async def test_loop_action_plan_allows_multiple_tools_in_one_action_only(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="tc1", name="query", arguments={"filter": {"group_id": "wave"}}),
                    ToolCall(id="tc2", name="query", arguments={"filter": {"status": "done"}}),
                    ToolCall(id="tc3", name="file_write", arguments={"path": "shared/out.cpp", "content": "int main(){}"}),
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="tc4", name="file_write", arguments={"path": "shared/out.cpp", "content": "int main(){}"}),
                    ToolCall(id="tc5", name="submit", arguments={"path": "shared/out.cpp"}),
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc6", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=6,
        peer_progress_check_after_turns=3,
        peer_progress_check_no_tool_turns=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Continue the current implementation lane.",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer = rt.create_agent(
        "peer",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer.status = "done"
    agent._turns = 3
    agent._no_tool_turns = 2
    agent._artifact_nudge_count = 2

    await rt._agent_loop(agent)

    tool_calls = [e for e in rt._events if e["event"] == "tool_call"]
    skipped = [e for e in rt._events if e["event"] == "tool_call_skipped"]
    assert tool_calls[0]["data"]["tool"] == "query"
    assert tool_calls[1]["data"]["tool"] == "query"
    assert tool_calls[2]["data"]["tool"] == "file_write"
    assert not [e for e in skipped if e["data"]["tool"] == "query"]
    assert (tmp_workspace / "shared" / "out.cpp").exists()


@pytest.mark.asyncio
async def test_peer_progress_nudge_does_not_override_work_action_plan(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append(messages)
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="query", arguments={"filter": {"group_id": "wave"}})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=4,
        peer_progress_check_after_turns=3,
        peer_progress_check_no_tool_turns=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Continue the current implementation lane.",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer = rt.create_agent(
        "peer",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer.status = "done"
    agent._turns = 3
    agent._no_tool_turns = 2
    agent._artifact_nudge_count = 2
    agent._loop_action_plan = {
        "action": "work",
        "reason": "manual_work_lane",
        "turn_added": agent._turns,
    }

    await rt._agent_loop(agent)

    card = "\n".join(m.get("content") or "" for m in calls[0])
    assert "Peer-progress risk is active" in card
    assert "Next action candidate: action=work reason=default_task_progress" in card
    assert "tool=" not in card
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "query"]
    assert not [e for e in rt._events if e["event"] == "tool_call_skipped" and e["data"]["tool"] == "query"]


@pytest.mark.asyncio
async def test_coordination_artifact_task_queries_before_writing_report(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc3", name="file_write", arguments={"path": "shared/report.md", "content": "Report"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) > 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc4", name="set_status", arguments={"status": "done", "result": "done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[
                ToolCall(id="tc1", name="query", arguments={"filter": {"group_id": "wave"}}),
                ToolCall(id="tc2", name="file_write", arguments={"path": "shared/report.md", "content": "Report"}),
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Coordinate peers. Query generator peers with group_id `wave`. Write `shared/report.md`.",
        group_id="coord",
        created_by="root",
        current_task_tags=["phase:coordination", "role:coordinator"],
    )
    agent._turns = 2
    agent._artifact_nudge_count = 1

    await rt._agent_loop(agent)

    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "Next action candidate: action=read reason=task_requests_peer_query_before_artifact" in card
    assert "tool=" not in card
    assert "query" in calls[0]["tools"]
    assert calls[0]["tool_choice"] is None
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "query"]
    assert not [e for e in rt._events if e["event"] == "tool_call_skipped" and e["data"]["tool"] == "query"]


@pytest.mark.asyncio
async def test_tool_calls_outside_current_action_are_skipped_until_next_loop(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/report.md", "content": "Report"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Inspect peers, then write `shared/report.md`.")
    agent._loop_action_plan = {
        "action": "read",
        "reason": "filtered_read_intent_after_write_scope",
        "turn_added": agent._turns,
    }

    await rt._agent_loop(agent)

    assert not (tmp_workspace / "shared" / "report.md").exists()
    assert [e for e in rt._events if e["event"] == "tool_call_skipped" and e["data"]["tool"] == "file_write"]


@pytest.mark.asyncio
async def test_work_action_exposes_work_tools_without_forced_tool_choice(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        if len(calls) > 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="set_status", arguments={"status": "done", "result": "done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 2
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}
    llm_events = [e for e in rt._events if e["event"] == "llm_done"]
    assert llm_events[0]["data"]["recommended_tool_choice"] == "file_write"
    assert llm_events[0]["data"]["tool_scope"] == "artifact_write_only"
    assert llm_events[0]["data"]["loop_action"] == "work"
    assert (tmp_workspace / "shared" / "out.txt").read_text() == "done"


def test_source_change_task_defers_final_artifact_pressure_until_evidence(tmp_workspace):
    _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n", "tests/test_mod.py": "def test_ok():\n    assert True\n"})
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Patch shared/source, add focused tests, and run tests. Write final report `shared/final_engineering_report.md`."
    )
    agent._turns = 3
    agent._artifact_nudge_count = 3

    missing = rt.missing_expected_outputs(agent)
    rt._check_artifact_nudge(agent)
    rt._refresh_loop_action_plan(agent, missing)
    card = rt._build_loop_action_context(agent, missing, [])

    assert agent._loop_action_plan["action"] == "work"
    assert agent._loop_action_plan["reason"] == "source_test_evidence_before_artifacts"
    assert "Completion blockers are active" in card["content"]
    assert [e for e in rt._events if e["event"] == "artifact_nudge_deferred"]
    assert not [e for e in rt._events if e["event"] == "artifact_nudge"]


def test_coordinator_with_active_children_does_not_take_source_work_tools(tmp_workspace):
    _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n", "tests/test_mod.py": "def test_ok():\n    assert True\n"})
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=10)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Patch shared/source and run tests. Coordinate peers and write final report `shared/final_engineering_report.md`.",
        orchestration_preference="parallel",
    )
    child = rt.create_agent(
        "Implement the source patch in shared/source.",
        parent=parent.id,
        role="implementer",
        group_id="wave",
        current_task_tags=["phase:implementation"],
    )
    child.status = "running"
    parent._turns = 3

    missing = rt.missing_expected_outputs(parent)
    rt._refresh_loop_action_plan(parent, missing)
    tools = rt._tools_for_loop_action(parent._loop_action_plan["action"])
    card = rt._build_loop_action_context(parent, missing, [])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["reason"] == "coordinate_children_before_source_work"
    assert {"query", "wait", "set_status", "get_cost"} <= tools
    assert "file_read" not in tools
    assert "shell" not in tools
    assert "Coordinator boundary" in card["content"]
    assert "Do not inspect or edit implementation files yourself" in card["content"]


def test_coordinator_child_plan_overrides_stale_work_state_action(tmp_workspace):
    _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n", "tests/test_mod.py": "def test_ok():\n    assert True\n"})
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=10)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Patch shared/source and run tests. Coordinate peers and write final report `shared/final_engineering_report.md`.",
        orchestration_preference="parallel",
    )
    child = rt.create_agent(
        "Implement the source patch in shared/source.",
        parent=parent.id,
        role="implementer",
        group_id="wave",
        current_task_tags=["phase:implementation"],
    )
    child.status = "running"
    parent._turns = 4
    rt.state_board_update(parent.id, action_state="work")

    missing = rt.missing_expected_outputs(parent)
    rt._refresh_loop_action_plan(parent, missing)

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["reason"] == "coordinate_children_before_source_work"
    assert parent._state_action_consumed_version == 0


def test_coordinator_delegates_integration_after_child_source_progress(tmp_workspace):
    source = _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n", "tests/test_mod.py": "def test_ok():\n    assert True\n"})
    (source / "pkg" / "mod.py").write_text("VALUE = 2\n")
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=10)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Patch shared/source and run tests. After implementers finish, create integration/test review. "
        "Write final report `shared/final_engineering_report.md`.",
        orchestration_preference="parallel",
    )
    child = rt.create_agent(
        "Implement feature work in shared/source.",
        parent=parent.id,
        role="implementer_feature",
        group_id="wave",
        current_task_tags=["phase:implementation"],
    )
    child.status = "running"
    parent._turns = 4

    missing = rt.missing_expected_outputs(parent)
    rt._refresh_loop_action_plan(parent, missing)
    tools = rt._tools_for_loop_action(parent._loop_action_plan["action"])
    card = rt._build_loop_action_context(parent, missing, [])

    assert parent._loop_action_plan["action"] == "create"
    assert parent._loop_action_plan["reason"] == "delegate_integration_after_child_source_progress"
    assert {"spawn", "create_agent", "spawn_many", "query", "set_status", "get_cost"} <= tools
    assert "file_read" not in tools
    assert "shell" not in tools
    assert "Create one integration/test/review agent" in card["content"]
    assert "Do not inspect implementation files or run the diff yourself" in card["content"]


def test_coordinator_artifact_nudge_uses_coordination_boundary(tmp_workspace):
    _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n", "tests/test_mod.py": "def test_ok():\n    assert True\n"})
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=10)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Patch shared/source and run tests. Coordinate peers and write final report `shared/final_engineering_report.md`.",
        orchestration_preference="parallel",
    )
    child = rt.create_agent(
        "Implement the source patch in shared/source.",
        parent=parent.id,
        role="implementer",
        group_id="wave",
        current_task_tags=["phase:implementation"],
    )
    child.status = "running"
    parent._turns = 3

    rt._check_artifact_nudge(parent)
    send = [e for e in rt._events if e["event"] == "send" and e["data"].get("message_type") == "artifact_nudge_deferred"][-1]

    assert "Coordinator boundary is active" in send["data"]["message"]
    assert "Choose action=work" not in send["data"]["message"]


def test_multiphase_coordinator_artifact_nudge_defers_with_active_children(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=30)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Run OpenDeepThink n=20 K=4 T=3 M=10. Query generator peers with group_id `wave`. "
        "Write `shared/final/bt.json`, `shared/selected_solution.cpp`, and `shared/report.md`.",
        orchestration_preference="aggressive",
    )
    child = rt.create_agent(
        "Write shared/gen0/candidate_00.cpp and shared/gen0/candidate_00.md.",
        parent=parent.id,
        role="generator",
        group_id="wave",
        current_task_tags=["phase:gen0", "role:generator", "candidate:00"],
    )
    child.status = "running"
    parent._turns = 12

    rt._check_artifact_nudge(parent)
    sends = [e for e in rt._events if e["event"] == "send" and e["data"].get("message_type") == "artifact_nudge_deferred"]

    assert sends
    message = sends[-1]["data"]["message"]
    assert "Coordinator Artifact Deferred" in message
    assert "Choose action=work" not in message
    assert "Use action=message to query/wait" in message
    assert not [e for e in rt._events if e["event"] == "artifact_nudge"]


def test_coordinator_uses_child_test_evidence_for_completion(tmp_workspace):
    source = _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n", "tests/test_mod.py": "def test_ok():\n    assert True\n"})
    (source / "pkg" / "mod.py").write_text("VALUE = 2\n")
    (tmp_workspace / "shared" / "final_engineering_report.md").write_text("final report\n")
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent("Patch shared/source and run tests. Write `shared/final_engineering_report.md`.")
    child = rt.create_agent(
        "Run focused tests for shared/source.",
        parent=parent.id,
        role="integration_tester",
        group_id="wave",
        current_task_tags=["phase:integration", "area:tests"],
    )
    child._successful_test_commands.append("cd $SHARED/source && python3 -m pytest tests/test_mod.py -q")

    blockers = rt.completion_blockers(parent)

    assert not [blocker for blocker in blockers if blocker["kind"] == "test_run"]


@pytest.mark.asyncio
async def test_done_blocked_without_required_source_diff_and_tests(tmp_workspace):
    _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n"})
    (tmp_workspace / "shared" / "final.md").write_text("final report\n")
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Patch shared/source and run tests. Write `shared/final.md`.")

    result = await meta_set_status({"status": "done", "result": "done"}, agent, rt)

    assert result["blocked"] == "completion_evidence"
    assert {b["kind"] for b in result["blockers"]} == {"source_change", "test_run"}
    assert agent.status == "running"
    assert agent.action_state == "work"


@pytest.mark.asyncio
async def test_compact_stop_blocked_without_required_source_diff_and_tests(tmp_workspace):
    _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n"})
    (tmp_workspace / "shared" / "final.md").write_text("final report\n")
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Patch shared/source and run tests. Write `shared/final.md`.")

    result = await meta_compact({"summary": "final", "stop_after": True, "result": "done"}, agent, rt)

    assert result["blocked"] == "completion_evidence"
    assert agent._compact_pending is None
    assert agent.status == "running"
    assert agent.action_state == "work"


@pytest.mark.asyncio
async def test_source_completion_allows_after_diff_and_successful_tests(tmp_workspace):
    source = _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n"})
    (source / "pkg" / "mod.py").write_text("VALUE = 2\n")
    (tmp_workspace / "shared" / "final.md").write_text("final report\n")
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Patch shared/source and run tests. Write `shared/final.md`.")
    agent._successful_test_commands.append("pytest -q tests/test_mod.py")

    result = await meta_set_status({"status": "done", "result": "done"}, agent, rt)

    assert result["status"] == "done"
    assert agent.status == "done"
    assert rt.memory.serialize(agent.id)["active_task"] is None


@pytest.mark.asyncio
async def test_shell_exit_code_zero_records_successful_test_command(tmp_workspace):
    source = _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n"})
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(
                    id="tc1",
                    name="file_write",
                    arguments={"path": "shared/source/pkg/mod.py", "content": "VALUE = 2\n", "overwrite": True},
                )],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="shell", arguments={"command": "pytest -q tests/test_mod.py"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc3", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    class FakeSandbox:
        async def exec(self, cmd, workspace, shared_dir, timeout=30):
            return {"exit_code": 0, "stdout": "1 passed\n", "stderr": ""}

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    rt._tool_context.sandbox = FakeSandbox()
    agent = rt.create_agent("Patch shared/source and run tests.")
    agent._turns = 3

    await rt._agent_loop(agent)

    assert (source / "pkg" / "mod.py").read_text() == "VALUE = 2\n"
    assert agent._successful_test_commands == ["pytest -q tests/test_mod.py"]
    assert not [b for b in rt.completion_blockers(agent) if b["kind"] == "test_run"]


@pytest.mark.asyncio
async def test_work_artifact_turn_raises_response_token_budget(tmp_workspace, monkeypatch):
    calls = []
    monkeypatch.setenv("NANOMA_MAX_TOKENS", "3072")

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "max_tokens": kwargs.get("max_tokens"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/out.cpp", "content": "int main(){}"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=5,
        artifact_write_max_tokens=9000,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.cpp`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["max_tokens"] == 9000
    llm_events = [e for e in rt._events if e["event"] == "llm_done"]
    assert llm_events[0]["data"]["turn_max_tokens"] == 9000


@pytest.mark.asyncio
async def test_filtered_text_tool_calls_are_logged(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        content = (
            "I want to inspect first.\n"
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"file_list\">\n"
            "<｜DSML｜parameter name=\"path\" string=\"true\">shared</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered"]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_list"]


@pytest.mark.asyncio
async def test_allowed_text_dsml_tool_call_is_recovered_and_executed(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        if len(calls) == 1:
            content = (
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"file_write\">\n"
                "<｜DSML｜parameter name=\"path\" string=\"true\">shared/out.txt</｜DSML｜parameter>\n"
                "<｜DSML｜parameter name=\"content\" string=\"true\">done</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            )
            return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        return LLMResponse(
            tool_calls=[ToolCall(id="tc2", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert calls[0] == ["file_write"]
    assert (tmp_workspace / "shared" / "out.txt").read_text() == "done"
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered"]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_write"]


@pytest.mark.asyncio
async def test_text_dsml_read_alias_recovers_when_file_read_is_in_scope(tmp_workspace):
    (tmp_workspace / "shared" / "input.txt").write_text("hello")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        if len(calls) == 1:
            content = (
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"read\">\n"
                "<｜DSML｜parameter name=\"file_path\" string=\"true\">shared/input.txt</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            )
            return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        return LLMResponse(
            tool_calls=[ToolCall(id="tc2", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=3)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Read `shared/input.txt` and summarize.")

    await rt._agent_loop(agent)

    assert "file_read" in calls[0]
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered"]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_read"]


@pytest.mark.asyncio
async def test_work_action_allows_read_intent_without_replanning_mid_turn(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if len(calls) <= 2:
            content = (
                "I need one more inspection.\n"
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"file_list\">\n"
                "<｜DSML｜parameter name=\"path\" string=\"true\">shared</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            )
            return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        if len(calls) == 3:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc3", name="file_list", arguments={"path": "shared"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc4", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=7)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/report.md`.")
    agent._turns = 3
    agent._artifact_nudge_count = 1

    await rt._agent_loop(agent)

    assert {"file_read", "file_write", "file_list", "grep"} <= set(calls[0]["tools"])
    assert calls[1]["tools"] == ["file_write"]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_list"]
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered"]


@pytest.mark.asyncio
async def test_repeated_artifact_nudge_for_leaf_work_limits_to_file_write(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/report.md", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=4)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/report.md`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}


@pytest.mark.asyncio
async def test_coordination_artifact_work_action_allows_read_intent(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if len(calls) == 1:
            content = (
                "I need to inspect peer outputs before writing the report.\n"
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"file_list\">\n"
                "<｜DSML｜parameter name=\"path\" string=\"true\">shared/gen0</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            )
            return LLMResponse(content=content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="file_list", arguments={"path": "shared/gen0"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc3", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Coordinate peers. Query generator peers with group_id `wave`. Inspect `shared/gen0/`. "
        "Write `shared/report.md` summarizing candidate file status and remaining gaps."
    )
    agent._turns = 3
    agent._artifact_nudge_count = 2
    agent._last_query_turn = 2

    await rt._agent_loop(agent)

    assert {"file_read", "file_write", "file_list", "grep", "query"} <= set(calls[0]["tools"])
    assert {"file_read", "file_write", "file_list", "grep", "query"} <= set(calls[1]["tools"])
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_list"]
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered"]


@pytest.mark.asyncio
async def test_coordination_artifact_read_action_includes_query_tools(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="query", arguments={"filter": {"group_id": "wave"}})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Coordinate peers. Query generator peers with group_id `wave`. Inspect `shared/gen0/`. "
        "Write `shared/report.md` summarizing candidate file status and remaining gaps."
    )
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert {"file_read", "query", "file_list", "grep"} <= set(calls[0])
    assert "file_write" not in calls[0]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "query"]


@pytest.mark.asyncio
async def test_outputs_complete_prefers_compact_stop_action(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc1",
                name="compact",
                arguments={
                    "summary": "artifact complete",
                    "files": ["shared/out.txt"],
                    "tags": ["status:complete"],
                    "stop_after": True,
                    "result": "done",
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=4)
    rt = Runtime(config=config, llm_call=mock_llm)
    (tmp_workspace / "shared" / "out.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_workspace / "shared" / "out.txt").write_text("done")
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3

    await rt._agent_loop(agent)

    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "All explicit output files named by your task currently exist" in card
    assert "Next action candidate: action=compact reason=outputs_complete_publish_memory" in card
    assert "tool=" not in card
    assert agent.status == "done"
    runtime_memory = rt.memory.serialize(agent.id)
    assert "shared/out.txt" in runtime_memory["artifact_index"]


@pytest.mark.asyncio
async def test_compact_action_allows_read_confirmation_tools(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[
                ToolCall(id="tc1", name="file_list", arguments={"path": "shared"}),
                ToolCall(
                    id="tc2",
                    name="compact",
                    arguments={
                        "summary": "confirmed and complete",
                        "files": ["shared/out.txt"],
                        "stop_after": True,
                        "result": "done",
                    },
                ),
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=4)
    rt = Runtime(config=config, llm_call=mock_llm)
    (tmp_workspace / "shared" / "out.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_workspace / "shared" / "out.txt").write_text("done")
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3

    await rt._agent_loop(agent)

    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_list"]
    assert not [e for e in rt._events if e["event"] == "tool_call_skipped"]
    assert agent.status == "done"


@pytest.mark.asyncio
async def test_compact_action_allows_final_artifact_write(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[
                ToolCall(id="tc1", name="file_write", arguments={"path": "shared/out.md", "content": "final", "overwrite": True}),
                ToolCall(
                    id="tc2",
                    name="compact",
                    arguments={
                        "summary": "final artifact fixed",
                        "files": ["shared/out.md"],
                        "stop_after": True,
                        "result": "done",
                    },
                ),
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=4)
    rt = Runtime(config=config, llm_call=mock_llm)
    (tmp_workspace / "shared" / "out.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_workspace / "shared" / "out.md").write_text("draft")
    agent = rt.create_agent("Create `shared/out.md`.")
    agent._turns = 3

    await rt._agent_loop(agent)

    assert (tmp_workspace / "shared" / "out.md").read_text() == "final"
    assert not [e for e in rt._events if e["event"] == "tool_call_skipped"]
    assert agent.status == "done"


@pytest.mark.asyncio
async def test_work_text_miss_omits_long_content_from_history(tmp_workspace):
    long_content = "I need to think more.\n" + ("analysis " * 1200)
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if len(calls) == 1:
            return LLMResponse(content=long_content, usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
        return LLMResponse(
            tool_calls=[ToolCall(id="tc2", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=4)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 2
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert [e for e in rt._events if e["event"] == "write_only_miss"]
    assert not any(long_content in (m.get("content") or "") for m in agent.history)
    assert calls[1]["tools"] == ["file_write"]
    second_card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "Previous write-only artifact turns produced assistant text" in second_card
    assert "Runtime will narrow this retry turn to file_write" in second_card
    llm_events = [e for e in rt._events if e["event"] == "llm_done"]
    assert llm_events[1]["data"]["tool_scope"] == "retry_file_write_after_text_miss"
    assert (tmp_workspace / "shared" / "out.txt").read_text() == "done"


@pytest.mark.asyncio
async def test_write_miss_retry_filters_non_file_write_tools(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        return LLMResponse(
            tool_calls=[
                ToolCall(id="tc1", name="file_list", arguments={"path": "shared"}),
                ToolCall(id="tc2", name="file_write", arguments={"path": "shared/out.txt", "content": "done"}),
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3
    agent._write_only_miss_count = 1

    await rt._agent_loop(agent)

    assert calls[0] == ["file_write"]
    assert [e for e in rt._events if e["event"] == "tool_call_skipped" and e["data"]["tool"] == "file_list"]
    assert (tmp_workspace / "shared" / "out.txt").read_text() == "done"


@pytest.mark.asyncio
async def test_explicit_artifact_task_enters_work_action_after_nudge(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tool_choice": kwargs.get("tool_choice"),
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if len(calls) == 1:
            return LLMResponse(
                content="I will think through the output first.",
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc3", name="set_status", arguments={"status": "done", "result": "artifact written"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)

    result = await rt.run("Create `shared/out.txt`.")

    assert result == "artifact written"
    assert calls[0]["tool_choice"] is None
    assert set(calls[0]["tools"]) > {"file_write"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}
    assert calls[1]["tools"] == ["file_write"]
    assert set(calls[2]["tools"]) > {"file_write"}
    assert (tmp_workspace / "shared" / "out.txt").read_text() == "done"
    assert [e for e in rt._events if e["event"] == "artifact_nudge"]


@pytest.mark.asyncio
async def test_artifact_work_action_persists_until_all_expected_outputs_exist(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tool_choice": kwargs.get("tool_choice"),
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if len(calls) == 1:
            return LLMResponse(
                content="I will outline the candidate first.",
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc2", name="file_write", arguments={"path": "shared/candidate.cpp", "content": "int main(){}"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 3:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc3", name="file_write", arguments={"path": "shared/candidate.md", "content": "notes"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc4", name="set_status", arguments={"status": "done", "result": "both artifacts written"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)

    result = await rt.run("Create `shared/candidate.cpp` and `shared/candidate.md`.")

    assert result == "both artifacts written"
    assert calls[0]["tool_choice"] is None
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}
    assert calls[1]["tools"] == ["file_write"]
    assert calls[2]["tool_choice"] is None
    assert {"file_read", "file_write", "file_replace", "file_list", "grep"} <= set(calls[2]["tools"])
    assert "shell" not in calls[2]["tools"]
    assert set(calls[3]["tools"]) > {"file_write"}
    assert (tmp_workspace / "shared" / "candidate.cpp").exists()
    assert (tmp_workspace / "shared" / "candidate.md").exists()


@pytest.mark.asyncio
async def test_artifact_write_warns_on_unexpected_path(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/placeholder.txt", "content": ""})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/required.txt`.")

    writes = [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_write"]
    assert writes
    assert "unexpected_artifact_path" in writes[-1]["data"]["result"]


@pytest.mark.asyncio
async def test_expected_file_write_registers_artifact_memory(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/out.txt`.")

    agent = rt.agents["alpha"]
    assert "shared/out.txt" in [artifact.path for artifact in agent.artifacts]
    assert "shared/out.txt" in rt.memory.serialize(agent.id)["artifact_index"]


@pytest.mark.asyncio
async def test_max_turn_auto_completes_when_expected_outputs_exist(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    result = await rt.run("Create `shared/out.txt`.")

    assert "auto-completing at max_turns" in result
    agent = rt.agents["alpha"]
    assert agent.status == "done"
    assert agent.result != "[Max turns reached]"
    assert "shared/out.txt" in [artifact.path for artifact in agent.artifacts]


@pytest.mark.asyncio
async def test_artifact_commit_extracts_required_cpp_from_assistant_text(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        if len([m for m in messages if m.get("role") == "assistant"]) == 0:
            return LLMResponse(
                content="Here is the solution:\n```cpp\n#include <bits/stdc++.h>\nint main(){return 0;}\n```",
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc2", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=3)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/solution.cpp`.")

    out = tmp_workspace / "shared" / "solution.cpp"
    assert out.exists()
    assert "#include <bits/stdc++.h>" in out.read_text()
    assert "shared/solution.cpp" in rt.memory.serialize("alpha")["artifact_index"]
    assert [e for e in rt._events if e["event"] == "artifact_commit"]


@pytest.mark.asyncio
async def test_artifact_commit_extracts_required_markdown_from_assistant_text(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        if len([m for m in messages if m.get("role") == "assistant"]) == 0:
            return LLMResponse(
                content="Summary\n\nThe implementation is complete.",
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc2", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=3)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/report.md`.")

    assert (tmp_workspace / "shared" / "report.md").read_text().startswith("Summary")


@pytest.mark.asyncio
async def test_artifact_commit_extracts_short_heading_markdown(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content="Core idea\n\nCompute the contribution of each tree edge.\n\nComplexity: O(n log n).\nRisks: unverified.",
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/candidate.md`.")

    assert (tmp_workspace / "shared" / "candidate.md").read_text().startswith("Core idea")


@pytest.mark.asyncio
async def test_artifact_commit_does_not_write_code_file_without_code_block(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content="I have an idea, but no code yet.",
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/solution.cpp`.")

    assert not (tmp_workspace / "shared" / "solution.cpp").exists()


@pytest.mark.asyncio
async def test_artifact_commit_does_not_write_report_from_process_chatter(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content="Let's start by reading the required files and exploring the shared directory structure.",
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/report.md`.")

    assert not (tmp_workspace / "shared" / "report.md").exists()


@pytest.mark.asyncio
async def test_artifact_commit_does_not_write_markdown_from_analysis_with_fence(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content=(
                "Let me analyze this problem before writing the report.\n\n"
                "```markdown\n# Draft\nThis is not final.\n```"
            ),
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/report.md`.")

    assert not (tmp_workspace / "shared" / "report.md").exists()
    assert not [e for e in rt._events if e["event"] == "artifact_commit"]


@pytest.mark.asyncio
async def test_artifact_commit_does_not_write_markdown_from_sample_block(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content=(
                "I need to verify the sample before writing the report.\n\n"
                "```text\n7\n3 1\n1 2\n3 5\n4 5\n3 6\n6 7\n```"
            ),
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/report.md`.")

    assert not (tmp_workspace / "shared" / "report.md").exists()
    assert not [e for e in rt._events if e["event"] == "artifact_commit"]


@pytest.mark.asyncio
async def test_artifact_commit_does_not_write_report_from_task_echo(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content=(
                'You are in a multi-agent system. Your unique agent id is "charlie".\n\n'
                "Your task: Create `shared/report.md`.\n\n"
                "Let me first read the problem statement and metadata."
            ),
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/report.md`.")

    assert not (tmp_workspace / "shared" / "report.md").exists()


@pytest.mark.asyncio
async def test_dsml_tool_text_recovers_file_write_without_artifact_commit(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content=(
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke name=\"file_write\">\n"
                "<｜DSML｜parameter name=\"path\" string=\"true\">shared/report.md</｜DSML｜parameter>\n"
                "<｜DSML｜parameter name=\"content\" string=\"true\">```markdown\n# Report\n```</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            ),
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/report.md`.")

    assert (tmp_workspace / "shared" / "report.md").exists()
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered"]
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_write"]
    assert not [e for e in rt._events if e["event"] == "artifact_commit"]


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
async def test_meta_query_filter_tags_and_trace_event(runtime):
    target = runtime.create_agent(
        "target",
        role="reviewer",
        group_id="reviewer_wave",
        current_task_tags=["phase:reviewer", "reviewer:01"],
    )
    other = runtime.create_agent("other", role="scout", group_id="scout_wave", current_task_tags=["phase:scout"])
    runtime.memory.update(target.id, public_summary="review done", tags=["security", "phase:reviewer"])
    runtime.memory.update(other.id, public_summary="scout done", tags=["runtime", "phase:scout"])
    q = runtime.create_agent("querier")

    result = await meta_query(
        {
            "filter": {"group_id": "reviewer_wave", "status": "running"},
            "tags": ["phase:reviewer"],
            "memory_intent": "find reviewers",
        },
        q,
        runtime,
    )

    assert result["count"] == 1
    assert result["agents"][0]["id"] == target.id
    assert result["total_agents"] == len(runtime.agents)
    assert result["memory_read"]["seed_terms"] == ["phase:reviewer"]
    query_events = [e for e in runtime._events if e["event"] == "query"]
    assert query_events[-1]["agent"] == q.id
    assert query_events[-1]["data"]["targets"] == [target.id]
    assert query_events[-1]["data"]["result_count"] == 1


@pytest.mark.asyncio
async def test_meta_query_exposes_progress_and_sorts_more_complete_peers(runtime, tmp_workspace):
    fast = runtime.create_agent("Create `shared/fast.txt`.", role="worker", group_id="wave")
    slow = runtime.create_agent("Create `shared/slow.txt`.", role="worker", group_id="wave")
    q = runtime.create_agent("querier")
    out = tmp_workspace / "shared" / "fast.txt"
    out.write_text("done")
    fast.artifacts.append(Artifact("shared/fast.txt", out))
    fast.status = "done"
    fast._turns = 2
    fast._tool_calls = 2
    slow._turns = 5
    slow._tool_calls = 1

    result = await meta_query({"filter": {"group_id": "wave"}}, q, runtime)

    assert [agent["id"] for agent in result["agents"][:2]] == [fast.id, slow.id]
    assert result["agents"][0]["progress"]["outputs_complete"] is True
    assert result["agents"][0]["progress"]["missing_outputs"] == []
    assert result["agents"][1]["progress"]["missing_outputs"] == ["shared/slow.txt"]


@pytest.mark.asyncio
async def test_meta_query_common_filter_mistakes_are_tolerated(runtime):
    target = runtime.create_agent(
        "target",
        role="implementer_feature7",
        create_type="peer_agent",
        group_id="cooperbench-click2068",
        current_task_tags=["feature:7"],
    )
    q = runtime.create_agent("querier")

    by_active = await meta_query({"filter": {"status": "active"}}, q, runtime)
    assert target.id in {a["id"] for a in by_active["agents"]}

    by_create_type = await meta_query({"filter": {"role": "peer_agent"}}, q, runtime)
    assert target.id in {a["id"] for a in by_create_type["agents"]}

    by_group_suffix = await meta_query({"filter": {"group_id": "click2068"}}, q, runtime)
    assert target.id in {a["id"] for a in by_group_suffix["agents"]}

    tagged_role = runtime.create_agent(
        "tagged generator",
        group_id="cooperbench-click2068",
        current_task_tags=["role:generator", "phase:gen0"],
    )
    by_role_tag = await meta_query({"filter": {"role": "generator"}}, q, runtime)
    assert tagged_role.id in {a["id"] for a in by_role_tag["agents"]}

    empty = await meta_query({"filter": {"group_id": "missing"}}, q, runtime)
    assert empty["count"] == 0
    assert "cooperbench-click2068" in empty["query_help"]["available_group_ids"]


@pytest.mark.asyncio
async def test_parent_done_notification_disabled_by_default(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent("parent")
    child = rt.create_agent("child", parent=parent.id)

    await rt._agent_loop(child)

    assert parent._steer_inbox.empty()
    assert not [e for e in rt._events if e["event"] == "send" and e["data"].get("from") == "system" and e["data"].get("to") == parent.id]


@pytest.mark.asyncio
async def test_parent_done_notification_can_be_enabled(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        log_dir=None,
        sandbox_backend="host",
        notify_parent_on_done=True,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent("parent")
    child = rt.create_agent("child", parent=parent.id)

    await rt._agent_loop(child)

    assert not parent._steer_inbox.empty()
    assert [e for e in rt._events if e["event"] == "send" and e["data"].get("from") == "system" and e["data"].get("to") == parent.id]


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
async def test_meta_compact_stop_after_marks_done(runtime):
    a = runtime.create_agent("worker")
    result = await meta_compact({"summary": "final compact", "tags": ["done"], "stop_after": True, "result": "finished"}, a, runtime)
    assert result["scheduled"] is True
    assert result["stop_after"] is True
    runtime._execute_compact(a)
    assert a.status == "done"
    assert a.result == "finished"
    memory = runtime.memory.serialize(a.id)
    assert memory["public_summary"] == "final compact"
    assert memory["active_task"] is None
    assert "done" in memory["tags"]


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
async def test_meta_batch_accepts_name_params_and_function_arguments(runtime):
    a = runtime.create_agent("worker")
    batch_data = [
        {"name": "file_write", "params": {"path": "from_params.txt", "content": "ok"}},
        {"function": {"name": "compact", "arguments": json.dumps({"summary": "function compact"})}},
    ]
    batch_file = a.workspace / "batch_alt.json"
    batch_file.write_text(json.dumps(batch_data))
    batch_result = await meta_batch({"path": "batch_alt.json"}, a, runtime)
    assert batch_result["executed"] == 2
    assert batch_result["results"][0]["tool"] == "file_write"
    assert (a.workspace / "from_params.txt").read_text() == "ok"
    assert batch_result["results"][1]["tool"] == "compact"
    assert a._compact_pending["summary"] == "function compact"


def test_stats_peak_concurrent_uses_agent_lifecycle(runtime):
    a = runtime.create_agent("first")
    b = runtime.create_agent("second")
    runtime._emit(a.id, "done", {})
    runtime._emit(b.id, "done", {})
    assert runtime.stats()["agents"]["peak_concurrent"] == 2


@pytest.mark.asyncio
async def test_meta_submit_shared_path(runtime):
    a = runtime.create_agent("worker")
    shared_file = runtime._tool_context.shared_dir / "final.md"
    shared_file.write_text("final data")
    result = await meta_submit({"path": "shared/final.md", "description": "final output"}, a, runtime)
    assert result["submitted"] == "shared/final.md"
    assert result["shared_copy"] == str(shared_file)
    assert "shared/final.md" in runtime.memory.serialize(a.id)["artifact_index"]


@pytest.mark.asyncio
async def test_meta_submit_workspace_root_prefixed_shared_path(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config, llm_call=mock_llm)
    a = rt.create_agent("worker")
    shared_file = rt._tool_context.shared_dir / "prefixed.md"
    shared_file.write_text("prefixed")

    result = await meta_submit(
        {"path": f"{tmp_workspace.name}/shared/prefixed.md", "description": "prefixed"},
        a,
        rt,
    )

    assert result["submitted"] == f"{tmp_workspace.name}/shared/prefixed.md"
    assert result["shared_copy"] == str(shared_file)


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

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_file_write({"path": "test.txt", "content": "hello world"}, ws, ctx)
    assert result["bytes"] == 11
    result = await tool_file_read({"path": "test.txt"}, ws, ctx)
    assert result["content"] == "hello world"
    assert "sha256" in result


@pytest.mark.asyncio
async def test_tool_file_path_alias_for_file_tools(tmp_workspace):
    from nanoma.tools import tool_file_read, tool_file_write

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()

    written = await tool_file_write({"file_path": "alias.txt", "content": "ok"}, ws, ctx)
    read = await tool_file_read({"file_path": "alias.txt"}, ws, ctx)

    assert "error" not in written
    assert read["content"] == "ok"


@pytest.mark.asyncio
async def test_tool_workspace_root_prefixed_shared_path(tmp_workspace):
    from nanoma.tools import tool_file_read, tool_file_write

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()

    prefixed = f"{tmp_workspace.name}/shared/root_prefixed.md"
    written = await tool_file_write({"path": prefixed, "content": "ok"}, ws, ctx)
    assert written["path"] == str(tmp_workspace / "shared" / "root_prefixed.md")

    read = await tool_file_read({"path": prefixed}, ws, ctx)
    assert read["content"] == "ok"


@pytest.mark.asyncio
async def test_tool_file_write_rejects_large_unguarded_overwrite(tmp_workspace):
    from nanoma.tools import tool_file_write

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    path = ws / "large.py"
    path.write_text("\n".join(f"line {i}" for i in range(220)))

    blocked = await tool_file_write({"path": "large.py", "content": "tiny"}, ws, ctx)
    assert "Refusing to overwrite" in blocked["error"]
    assert path.read_text().startswith("line 0")

    guarded = await tool_file_write(
        {"path": "large.py", "content": "tiny", "expected_sha256": blocked["sha256"]},
        ws,
        ctx,
    )
    assert guarded["bytes"] == 4
    assert path.read_text() == "tiny"


@pytest.mark.asyncio
async def test_tool_file_replace_and_env_paths(tmp_workspace):
    from nanoma.tools import tool_file_read, tool_file_replace

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (tmp_workspace / "shared" / "code.py").write_text("alpha\nbeta\n")

    read = await tool_file_read({"path": "$SHARED/code.py"}, ws, ctx)
    assert read["content"] == "alpha\nbeta\n"
    replaced = await tool_file_replace(
        {
            "path": "$SHARED/code.py",
            "old": "beta\n",
            "new": "gamma\n",
            "expected_sha256": read["sha256"],
        },
        ws,
        ctx,
    )
    assert replaced["replacements"] == 1
    assert (tmp_workspace / "shared" / "code.py").read_text() == "alpha\ngamma\n"


@pytest.mark.asyncio
async def test_tool_file_read_accepts_workspace_named_absolute_path(tmp_workspace):
    from nanoma.tools import tool_file_read

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (tmp_workspace / "shared" / "source" / "pkg").mkdir(parents=True)
    (tmp_workspace / "shared" / "source" / "pkg" / "labels.py").write_text("ok\n")
    pseudo_absolute = f"/{tmp_workspace.name}/shared/source/pkg/labels.py"

    result = await tool_file_read({"path": pseudo_absolute}, ws, ctx)

    assert result["content"] == "ok\n"


@pytest.mark.asyncio
async def test_tool_shared_path_maps_to_global_shared_dir(tmp_workspace):
    from nanoma.tools import tool_file_read, tool_file_write

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_file_write({"path": "shared/report.md", "content": "shared data"}, ws, ctx)
    assert result["path"] == str((tmp_workspace / "shared" / "report.md").resolve())
    assert not (ws / "shared" / "report.md").exists()
    result = await tool_file_read({"path": "shared/report.md"}, ws, ctx)
    assert result["content"] == "shared data"


@pytest.mark.asyncio
async def test_tool_file_read_sandbox(tmp_workspace):
    from nanoma.tools import tool_file_read

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
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

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "echo hello"}, ws, ctx)
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_tool_shell_can_be_disabled(tmp_workspace):
    from nanoma.tools import tool_shell

    ctx = ToolContext(
        shared_dir=tmp_workspace / "shared",
        workspace_root=tmp_workspace,
        enabled_work_tools={"file_read", "file_write", "file_list", "grep"},
    )
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "echo hello"}, ws, ctx)
    assert result["error"] == "Tool disabled by runtime policy: shell"


@pytest.mark.asyncio
async def test_tool_shell_controlled_allows_python_and_pytest(tmp_workspace):
    from nanoma.tools import tool_shell

    class FakeSandbox:
        def __init__(self):
            self.calls = []

        async def exec(self, cmd, workspace, shared_dir, timeout=30):
            self.calls.append((cmd, workspace, shared_dir, timeout))
            return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    fake = FakeSandbox()
    ctx = ToolContext(
        shared_dir=tmp_workspace / "shared",
        workspace_root=tmp_workspace,
        sandbox=fake,
        shell_mode="controlled",
    )
    ws = tmp_workspace / "agent"
    ws.mkdir()

    result = await tool_shell({"command": "python -m pytest tests -q", "timeout": 5}, ws, ctx)

    assert result["exit_code"] == 0
    assert fake.calls == [("python -m pytest tests -q", ws, tmp_workspace / "shared", 5)]


@pytest.mark.asyncio
async def test_tool_shell_controlled_allows_cd_shared_and_git(tmp_workspace):
    from nanoma.tools import tool_shell

    class FakeSandbox:
        def __init__(self):
            self.calls = []

        async def exec(self, cmd, workspace, shared_dir, timeout=30):
            self.calls.append((cmd, workspace, shared_dir, timeout))
            return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    fake = FakeSandbox()
    source = tmp_workspace / "shared" / "source"
    source.mkdir(parents=True)
    ctx = ToolContext(
        shared_dir=tmp_workspace / "shared",
        workspace_root=tmp_workspace,
        sandbox=fake,
        shell_mode="controlled",
    )
    ws = tmp_workspace / "agent"
    ws.mkdir()

    result = await tool_shell({"command": "cd $SHARED/source && git status", "timeout": 5}, ws, ctx)

    assert result["exit_code"] == 0
    assert fake.calls == [("git status", source.resolve(), tmp_workspace / "shared", 5)]


@pytest.mark.asyncio
async def test_tool_shell_controlled_strips_trailing_stderr_merge(tmp_workspace):
    from nanoma.tools import tool_shell

    class FakeSandbox:
        def __init__(self):
            self.calls = []

        async def exec(self, cmd, workspace, shared_dir, timeout=30):
            self.calls.append((cmd, workspace, shared_dir, timeout))
            return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    fake = FakeSandbox()
    source = tmp_workspace / "shared" / "source"
    source.mkdir(parents=True)
    ctx = ToolContext(
        shared_dir=tmp_workspace / "shared",
        workspace_root=tmp_workspace,
        sandbox=fake,
        shell_mode="controlled",
    )
    ws = tmp_workspace / "agent"
    ws.mkdir()

    result = await tool_shell(
        {"command": "cd $SHARED/source && python3 -m pytest tests/test_labels.py -v 2>&1", "timeout": 5},
        ws,
        ctx,
    )

    assert result["exit_code"] == 0
    assert fake.calls == [("python3 -m pytest tests/test_labels.py -v", source.resolve(), tmp_workspace / "shared", 5)]


@pytest.mark.asyncio
async def test_tool_shell_controlled_cd_still_rejects_inner_redirection(tmp_workspace):
    from nanoma.tools import tool_shell

    source = tmp_workspace / "shared" / "source"
    source.mkdir(parents=True)
    ctx = ToolContext(
        shared_dir=tmp_workspace / "shared",
        workspace_root=tmp_workspace,
        shell_mode="controlled",
    )
    ws = tmp_workspace / "agent"
    ws.mkdir()

    result = await tool_shell({"command": "cd $SHARED/source && python test.py > out.txt"}, ws, ctx)

    assert result["policy"] == "controlled_shell"
    assert "operators" in result["error"]


@pytest.mark.asyncio
async def test_tool_shell_controlled_rejects_shell_operators_and_pip(tmp_workspace):
    from nanoma.tools import tool_shell

    ctx = ToolContext(
        shared_dir=tmp_workspace / "shared",
        workspace_root=tmp_workspace,
        shell_mode="controlled",
    )
    ws = tmp_workspace / "agent"
    ws.mkdir()

    redirected = await tool_shell({"command": "python test.py > out.txt"}, ws, ctx)
    pip_install = await tool_shell({"command": "python -m pip install pytest"}, ws, ctx)

    assert redirected["policy"] == "controlled_shell"
    assert "operators" in redirected["error"]
    assert pip_install["policy"] == "controlled_shell"
    assert "package installation" in pip_install["error"]


@pytest.mark.asyncio
async def test_tool_shell_timeout(tmp_workspace):
    from nanoma.tools import tool_shell

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "sleep 10", "timeout": 1}, ws, ctx)
    assert result["exit_code"] == -1
    assert "Timeout" in result["stderr"]


@pytest.mark.asyncio
async def test_tool_shell_sets_absolute_shared_env(tmp_workspace):
    from nanoma.tools import tool_shell

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="controlled")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    source = tmp_workspace / "shared" / "source"
    source.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    result = await tool_shell({"command": "git -C $SHARED/source status"}, ws, ctx)

    assert result["exit_code"] == 0
    assert "On branch" in result["stdout"]


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
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, sandbox=fake, shell_mode="unrestricted")
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


@pytest.mark.asyncio
async def test_tool_bt_aggregate_from_comparison_directory(tmp_workspace):
    from nanoma.tools import tool_bt_aggregate

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace, shell_mode="unrestricted")
    ws = tmp_workspace / "agent"
    ws.mkdir()
    comparisons = tmp_workspace / "shared" / "comparisons"
    comparisons.mkdir(parents=True)
    results = [
        ("candidate_1", "candidate_2"),
        ("candidate_3", "candidate_1"),
        ("candidate_1", "candidate_4"),
        ("candidate_3", "candidate_2"),
        ("candidate_2", "candidate_4"),
        ("candidate_3", "candidate_4"),
    ]
    for idx, (winner, loser) in enumerate(results, start=1):
        (comparisons / f"judge_{idx}.json").write_text(json.dumps({"winner": winner, "loser": loser}))

    result = await tool_bt_aggregate(
        {"directory": "shared/comparisons", "output_path": "shared/bt_round0.json", "retain": 3},
        ws,
        ctx,
    )

    assert result["comparison_count"] == 6
    assert result["ranking"] == ["candidate_3", "candidate_1", "candidate_2", "candidate_4"]
    assert result["elite"] == "candidate_3"
    assert result["bottom"] == "candidate_4"
    assert result["retained"] == ["candidate_3", "candidate_1", "candidate_2"]
    assert result["discarded"] == ["candidate_4"]
    assert result["scores"]["candidate_3"] > result["scores"]["candidate_1"]
    assert result["scores"]["candidate_1"] > result["scores"]["candidate_2"]
    assert result["scores"]["candidate_2"] > result["scores"]["candidate_4"]

    output = json.loads((tmp_workspace / "shared" / "bt_round0.json").read_text())
    assert output["ranking"] == result["ranking"]


@pytest.mark.asyncio
async def test_tool_bt_aggregate_respects_enabled_tools(tmp_workspace):
    from nanoma.tools import tool_bt_aggregate

    ctx = ToolContext(
        shared_dir=tmp_workspace / "shared",
        workspace_root=tmp_workspace,
        enabled_work_tools={"file_read", "file_write", "file_list", "grep"},
    )
    ws = tmp_workspace / "agent"
    ws.mkdir()

    result = await tool_bt_aggregate({"directory": "shared/comparisons"}, ws, ctx)

    assert result["error"] == "Tool disabled by runtime policy: bt_aggregate"


def test_estimate_tokens():
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("a" * 400) == 100


def test_parse_dsml_text_tool_calls_filters_to_allowed_tools():
    content = """
<｜DSML｜tool_calls>
<｜DSML｜invoke name="file_read">
<｜DSML｜parameter name="path" string="true">shared/input.md</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="file_write">
<｜DSML｜parameter name="path" string="true">shared/out.txt</｜DSML｜parameter>
<｜DSML｜parameter name="content" string="true">hello &amp; goodbye</｜DSML｜parameter>
<｜DSML｜parameter name="overwrite" boolean="true">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
"""

    calls = _parse_text_tool_calls(content, {"file_write"})

    assert len(calls) == 1
    assert calls[0].name == "file_write"
    assert calls[0].arguments == {
        "path": "shared/out.txt",
        "content": "hello & goodbye",
        "overwrite": True,
    }


def test_count_message_tokens():
    msgs = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello there"}]
    tokens = count_message_tokens(msgs)
    assert tokens > 0


def test_stage_source_snapshot_initializes_git_baseline(tmp_path):
    from examples.high_concurrency_runner import stage_source_snapshot

    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    logs = tmp_path / "logs"
    source.mkdir()
    (source / "pkg.py").write_text("VALUE = 1\n")

    staged = stage_source_snapshot(source, workspace, logs)
    assert (staged / ".git").exists()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=staged,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert status.stdout == ""


def test_viewer_event_cache_resets_for_stale_offsets_and_rewrites(tmp_path):
    from nanoma.viewer import EventCache

    events_file = tmp_path / "events.jsonl"
    events_file.write_text(json.dumps({"type": "agent_new", "agent": "alpha", "data": {}}) + "\n")
    cache = EventCache(tmp_path)
    base_generation = cache.meta()["generation"]

    events, total, generation, reset = cache.get_since(99)
    assert reset is True
    assert total == 1
    assert events[0]["agent"] == "alpha"
    assert generation == base_generation

    events_file.write_text(json.dumps({"type": "agent_new", "agent": "bravo", "data": {}}) + "\n")
    events, total, generation, reset = cache.get_since(1)

    assert events == []
    assert total == 1
    assert reset is False
    assert generation > base_generation


def test_viewer_event_cache_distinguishes_append_from_new_trace(tmp_path):
    from nanoma.viewer import EventCache

    events_file = tmp_path / "events.jsonl"
    first = {"type": "agent_new", "agent": "alpha", "data": {}}
    second = {"type": "agent_new", "agent": "bravo", "data": {}}
    events_file.write_text(json.dumps(first) + "\n")
    cache = EventCache(tmp_path)
    base_generation = cache.meta()["generation"]

    events_file.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    events, total, generation, reset = cache.get_since(1, base_generation)

    assert reset is False
    assert total == 2
    assert generation == base_generation
    assert [event["agent"] for event in events] == ["bravo"]

    new_first = {"type": "agent_new", "agent": "charlie", "data": {"task": "new trace"}}
    new_second = {"type": "agent_new", "agent": "delta", "data": {"task": "new trace"}}
    events_file.write_text(json.dumps(new_first) + "\n" + json.dumps(new_second) + "\n")
    events, total, generation, reset = cache.get_since(2, base_generation)

    assert reset is True
    assert total == 2
    assert generation > base_generation
    assert [event["agent"] for event in events] == ["charlie", "delta"]


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
async def test_wait_accepts_string_timeout_and_agent_ids(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent("parent")
    child_a = rt.create_agent("child a", parent=parent.id)
    child_b = rt.create_agent("child b", parent=parent.id)
    child_a.status = "done"
    child_a.result = "a done"
    child_b.status = "done"
    child_b.result = "b done"

    result = await meta_wait({"agent_ids": f"{child_a.id},{child_b.id}", "timeout": "1", "mode": "all"}, parent, rt)

    assert {item["id"] for item in result["completed"]} == {child_a.id, child_b.id}
    assert result["pending"] == []


@pytest.mark.asyncio
async def test_parent_done_blocked_while_child_active(runtime):
    parent = runtime.create_agent("parent")
    child = runtime.create_agent("Create `shared/child.txt`.", parent=parent.id, role="worker", group_id="wave")
    child.status = "running"

    result = await meta_set_status({"status": "done", "result": "parent done"}, parent, runtime)

    assert result["blocked"] == "active_children"
    assert parent.status == "running"
    assert parent.action_state == "work"
    assert result["active_children"][0]["id"] == child.id
    assert result["active_children"][0]["missing_outputs"] == ["shared/child.txt"]

    await meta_query({"agent_id": child.id}, parent, runtime)
    result = await meta_set_status({"status": "done", "result": "parent done after query"}, parent, runtime)

    assert result["status"] == "done"
    assert parent.result == "parent done after query"


@pytest.mark.asyncio
async def test_coordinator_done_blocked_while_referenced_group_active(runtime):
    coordinator = runtime.create_agent("Query group_id `wave` and summarize after peers finish.")
    peer = runtime.create_agent("Create `shared/peer.txt`.", role="worker", group_id="wave")
    peer.status = "running"

    result = await meta_set_status({"status": "done", "result": "summary done"}, coordinator, runtime)

    assert result["blocked"] == "active_children"
    assert coordinator.status == "running"
    assert result["active_children"][0]["kind"] == "referenced_group"
    assert result["active_children"][0]["id"] == peer.id

    await meta_query({"filter": {"group_id": "wave"}}, coordinator, runtime)
    result = await meta_set_status({"status": "done", "result": "summary done after query"}, coordinator, runtime)

    assert result["status"] == "done"
    assert coordinator.result == "summary done after query"


@pytest.mark.asyncio
async def test_runtime_waits_for_running_children_after_root_finishes(tmp_workspace):
    calls = {"parent": 0, "child": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        system = messages[0]["content"] if messages else ""
        if "child writes file" in system:
            calls["child"] += 1
            if calls["child"] == 1:
                await asyncio.sleep(0.05)
                return LLMResponse(
                    content="I will write it on the next turn.",
                    usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
                )
            if calls["child"] == 2:
                return LLMResponse(
                    tool_calls=[ToolCall(id="c2", name="file_write", arguments={"path": "shared/child.txt", "content": "child done"})],
                    usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
                )
            return LLMResponse(
                tool_calls=[ToolCall(id="c3", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )

        calls["parent"] += 1
        if calls["parent"] == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="p1", name="spawn", arguments={"task": "child writes file `shared/child.txt`.", "role": "worker", "delegate": True})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)

    result = await rt.run("parent")

    assert result.startswith("[Delegated to ")
    assert (tmp_workspace / "shared" / "child.txt").read_text() == "child done"
    assert [e for e in rt._events if e["event"] == "join_remaining_agents"]


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
