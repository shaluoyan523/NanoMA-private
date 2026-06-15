"""Tests for NanoMA framework — no real LLM calls, fully deterministic."""

import asyncio
import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from nanoma.core import (
    Artifact,
    Envelope,
    ResourceQuota,
    Runtime,
    RuntimeConfig,
    ToolContext,
    _infer_task_orchestration_preference,
    _explicit_peer_wave_requirements,
    _task_has_multi_phase_peer_protocol,
    _task_has_spawnable_workstreams,
    _task_is_concrete_single_output_work,
    _task_is_coordination_artifact_task,
    _task_is_leaf_artifact_lane,
    _task_prefers_query_before_artifact,
)
from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import (
    LLMResponse,
    RetryConfig,
    ToolCall,
    TransientEmptyLLMResponse,
    TransientToolCallTransportMiss,
    _adapt_tool_schemas_for_model,
    _adapt_tool_choice_for_model,
    _body_with_tool_retry_repair,
    _model_needs_required_arg_tool_schema,
    _parse_text_tool_calls,
    _raise_for_transient_empty_response,
    _raise_for_transient_tool_call_transport_miss,
    _should_repair_tool_retry,
    count_message_tokens,
    estimate_tokens,
    openai_compatible_call,
)
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
    meta_submit_answer,
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


def test_model_registry_resolves_provider_and_tier_aliases():
    from nanoma.models import get_registry

    registry = get_registry()
    full = registry.get("deepseek/deepseek-v4-flash:nitro")
    provider_alias = registry.get("deepseek/deepseek-v4-flash")
    short_alias = registry.get("deepseek-v4-flash")

    assert full is not None
    assert provider_alias is full
    assert short_alias is full
    assert registry.context_limit("deepseek-v4-flash") == full.context_limit


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


def test_overwrite_verification_task_keeps_target_as_expected_output(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Verify the previous society article report, read current context if useful, "
        "and write the verified report to `shared/gaia_l2/c61d/society_articles.md` "
        "(overwrite existing file if needed).",
        model="test",
    )

    assert rt.expected_outputs(agent) == ["shared/gaia_l2/c61d/society_articles.md"]
    assert rt.missing_expected_outputs(agent) == ["shared/gaia_l2/c61d/society_articles.md"]


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
    active_tags = runtime.memory.serialize(child.id)["active_task"]["tags"]
    assert {"analysis", "docs"} <= set(active_tags)
    assert f"agent:{child.id}" in active_tags


@pytest.mark.asyncio
async def test_meta_spawn_canonicalizes_descriptive_role_and_indexes_identity_tags(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn(
        {
            "task": "Collect source evidence for lane one.",
            "role": "Evidence collector - Lane 1",
            "create_type": "peer_agent",
            "relationship": "peer",
            "group_id": "gaia-l2-c61d",
            "current_task_tags": ["Benchmark:GAIA", "task:C61D"],
        },
        parent,
        runtime,
    )
    child = runtime.agents[result["agent_id"]]

    assert child.role == "evidence"
    assert child.bio == "Evidence collector - Lane 1"
    assert result["role"] == "evidence"
    assert result["role_description"] == "Evidence collector - Lane 1"
    tags = runtime.memory.serialize(child.id)["tags"]
    assert f"agent:{child.id}" in tags
    assert "role:evidence" in tags
    assert "lane:1" in tags
    assert "group:gaia-l2-c61d" in tags
    assert "benchmark:gaia" in tags
    assert "task:c61d" in tags


@pytest.mark.asyncio
async def test_meta_create_agent_alias(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_create_agent({"task": "child task", "workflow_prior": "critic_review_loop"}, parent, runtime)
    assert result["agent_id"] in runtime.agents


@pytest.mark.asyncio
async def test_meta_spawn_rejects_agent_requested_unknown_model(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", default_model="deepseek-v4-flash")
    rt = Runtime(config=config)
    parent = rt.create_agent("parent task")

    result = await meta_spawn({"task": "child task", "model": "sonnet"}, parent, rt)
    child = rt.agents[result["agent_id"]]

    assert child.model == "deepseek-v4-flash"
    assert result["requested_model"] == "sonnet"
    assert result["model_fallback_reason"] == "agent_model_override_disabled"
    spawn_event = [e for e in rt._events if e["event"] == "spawn" and e["data"].get("child") == child.id][0]
    assert spawn_event["data"]["model_fallback_reason"] == "agent_model_override_disabled"


@pytest.mark.asyncio
async def test_meta_spawn_allows_configured_model_override(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        default_model="deepseek-v4-flash",
        allow_agent_model_override=True,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("parent task")

    result = await meta_spawn({"task": "child task", "model": "claude-sonnet-4-6"}, parent, rt)
    child = rt.agents[result["agent_id"]]

    assert child.model == "claude-sonnet-4-6"
    assert result["model_fallback_reason"] == ""


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
    assert all(item["result"]["requested_orchestration_preference"] is None for item in result["results"])
    assert all(child.orchestration_preference != "solo" for child in children)


@pytest.mark.asyncio
async def test_spawn_many_solo_default_does_not_freeze_research_workers(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        child_orchestration_preference=None,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA research lanes.")

    result = await meta_spawn_many(
        {
            "defaults": {
                "create_type": "peer_agent",
                "relationship": "peer",
                "group_id": "gaia_l2_c61d",
                "orchestration_preference": "solo",
            },
            "agents": [
                {
                    "task": (
                        "Find the paper about AI regulation originally submitted to arXiv.org in June 2022. "
                        "Report the arXiv ID, title, all six axis endpoint labels, and write "
                        "shared/gaia_l2/c61d/paper_2022_evidence.md."
                    ),
                    "role": "paper_finder_2022",
                    "current_task_tags": ["benchmark:gaia", "lane:2022_paper"],
                }
            ],
        },
        parent,
        rt,
    )

    child = rt.agents[result["results"][0]["result"]["agent_id"]]

    assert result["results"][0]["result"]["requested_orchestration_preference"] is None
    assert child.orchestration_preference != "solo"


@pytest.mark.asyncio
async def test_spawn_many_injects_local_output_for_gaia_helper_without_path(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate GAIA research lanes. If you spawn helpers, require each helper to write "
        "a short evidence report under `shared/gaia_l2/c61d/` before stopping."
    )

    result = await meta_spawn_many(
        {
            "defaults": {
                "create_type": "peer_agent",
                "relationship": "peer",
                "group_id": "gaia_l2_c61d",
            },
            "agents": [
                {
                    "task": "Find the AI regulation paper from June 2022 and identify all six axis-end words.",
                    "role": "paper_finder_2022",
                    "current_task_tags": ["benchmark:gaia", "lane:2022_paper"],
                },
                {
                    "task": "Find the Physics and Society article from August 11, 2016 and identify society descriptors.",
                    "role": "paper_finder_2016",
                    "current_task_tags": ["benchmark:gaia", "lane:2016_paper"],
                },
            ],
        },
        parent,
        rt,
    )

    assert result["created"] == 2
    first = rt.agents[result["results"][0]["result"]["agent_id"]]
    second = rt.agents[result["results"][1]["result"]["agent_id"]]

    assert result["results"][0]["result"]["auto_local_output"] == "shared/gaia_l2/c61d/paper_finder_2022_evidence.md"
    assert result["results"][1]["result"]["auto_local_output"] == "shared/gaia_l2/c61d/paper_finder_2016_evidence.md"
    assert rt.expected_outputs(first) == ["shared/gaia_l2/c61d/paper_finder_2022_evidence.md"]
    assert rt.expected_outputs(second) == ["shared/gaia_l2/c61d/paper_finder_2016_evidence.md"]
    assert "Required local output" in first.task
    assert [e for e in rt._events if e["event"] == "spawn_local_output_injected"]


@pytest.mark.asyncio
async def test_spawn_does_not_inject_local_output_when_child_has_explicit_output(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate helper agents and require reports under `shared/gaia_l2/c61d/`."
    )

    result = await meta_spawn(
        {
            "task": "Find evidence and write `shared/custom/report.md`.",
            "role": "researcher",
            "group_id": "wave",
        },
        parent,
        rt,
    )

    child = rt.agents[result["agent_id"]]
    assert result["auto_local_output"] is None
    assert rt.expected_outputs(child) == ["shared/custom/report.md"]
    assert child.task.count("Required local output") == 0


@pytest.mark.asyncio
async def test_spawn_expected_outputs_contract_injects_recovery_slot(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate helper agents and require reports under `shared/gaia_l2/c61d/`."
    )

    result = await meta_spawn(
        {
            "task": "Recover the society lane by replacing the previous low-confidence report.",
            "role": "recovery",
            "target_output_path": "shared/gaia_l2/c61d/society_articles.md",
            "readiness": 1.0,
        },
        parent,
        rt,
    )

    child = rt.agents[result["agent_id"]]
    assert result["expected_outputs"] == ["shared/gaia_l2/c61d/society_articles.md"]
    assert result["auto_local_output"] is None
    assert rt.expected_outputs(child) == ["shared/gaia_l2/c61d/society_articles.md"]
    assert "Required output slot(s)" in child.task


@pytest.mark.asyncio
async def test_spawn_does_not_inject_local_output_for_plain_parent(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent("parent task")

    result = await meta_spawn(
        {"task": "check requirements", "role": "requirements", "group_id": "wave"},
        parent,
        rt,
    )

    child = rt.agents[result["agent_id"]]
    assert result["auto_local_output"] is None
    assert rt.expected_outputs(child) == []


@pytest.mark.asyncio
async def test_meta_spawn_many_defers_low_readiness_downstream_agent(runtime):
    parent = runtime.create_agent("parent task")
    worker = runtime.create_agent("Find evidence and write shared/evidence.md", parent=parent.id, role="worker")
    parent.children.add(worker.id)

    result = await meta_spawn_many(
        {
            "agents": [
                {
                    "task": "Verify the worker evidence and summarize the answer",
                    "role": "verifier",
                    "depends_on": [worker.id],
                    "readiness": 0.1,
                }
            ],
        },
        parent,
        runtime,
    )

    assert result["created"] == 0
    assert result["deferred"] == 1
    assert parent._deferred_spawn_requests
    assert [e for e in runtime._events if e["event"] == "spawn_deferred"]


@pytest.mark.asyncio
async def test_meta_spawn_many_starts_independent_evidence_workers_without_readiness(runtime):
    parent = runtime.create_agent("parent task")

    result = await meta_spawn_many(
        {
            "agents": [
                {
                    "task": (
                        "Find the Physics and Society article submitted to arXiv on August 11, 2016. "
                        "Identify society descriptors and write shared/gaia/paper_2016_report.md with confidence."
                    ),
                    "role": "paper_2016_finder",
                    "current_task_tags": ["lane:2016_paper", "role:evidence"],
                },
                {
                    "task": (
                        "Find the AI regulation paper submitted to arXiv in June 2022. "
                        "Identify all figure axis words and write shared/gaia/paper_2022_report.md with confidence."
                    ),
                    "role": "paper_2022_finder",
                    "current_task_tags": ["lane:2022_paper", "role:evidence"],
                },
            ],
            "defaults": {"group_id": "gaia-workers", "create_type": "peer_agent"},
        },
        parent,
        runtime,
    )

    assert result["created"] == 2
    assert result["deferred"] == 0
    assert not parent._deferred_spawn_requests


@pytest.mark.asyncio
async def test_meta_spawn_defers_input_dependent_focus_without_explicit_depends_on(runtime):
    parent = runtime.create_agent("parent task")
    worker = runtime.create_agent("Find source evidence and write shared/evidence.md", parent=parent.id, role="source")
    parent.children.add(worker.id)

    result = await meta_spawn(
        {
            "task": "Cross-check the source evidence and produce a final confidence note.",
            "role": "open-focus",
            "current_task_tags": ["phase:confidence-check"],
            "readiness": 0.3,
        },
        parent,
        runtime,
    )

    assert result["deferred"] is True
    assert result["downstream_focus"] is True
    assert result["readiness"] < runtime.config.spawn_readiness_threshold


@pytest.mark.asyncio
async def test_meta_spawn_many_defers_downstream_agent_despite_model_lowered_threshold(runtime):
    parent = runtime.create_agent("parent task")
    worker = runtime.create_agent("Find evidence and write shared/evidence.md", parent=parent.id, role="worker")
    parent.children.add(worker.id)

    result = await meta_spawn_many(
        {
            "agents": [
                {
                    "task": "Verify worker evidence and produce final answer",
                    "role": "verifier",
                    "depends_on": [worker.id],
                    "readiness": 0.95,
                    "readiness_threshold": 0.2,
                }
            ],
        },
        parent,
        runtime,
    )

    assert result["created"] == 0
    assert result["deferred"] == 1
    deferred = result["results"][0]["result"]
    assert deferred["readiness"] < runtime.config.spawn_readiness_threshold
    assert deferred["readiness_threshold"] == runtime.config.spawn_readiness_threshold


@pytest.mark.asyncio
async def test_meta_spawn_allows_downstream_agent_when_dependency_ready(runtime):
    parent = runtime.create_agent("parent task")
    worker = runtime.create_agent("Find evidence and write shared/evidence.md", parent=parent.id, role="worker")
    parent.children.add(worker.id)
    worker.status = "done"
    worker.result = "evidence ready"

    result = await meta_spawn(
        {
            "task": "Verify the worker evidence and summarize the answer",
            "role": "verifier",
            "depends_on": [worker.id],
            "readiness": 0.1,
        },
        parent,
        runtime,
    )

    assert "agent_id" in result
    assert result["role"] == "verifier"
    assert result["readiness"] == 1.0


@pytest.mark.asyncio
async def test_meta_spawn_allows_self_verifier_when_current_agent_has_artifact(runtime, tmp_workspace):
    parent = runtime.create_agent("parent task")
    worker = runtime.create_agent(
        "Find evidence and write shared/evidence.md.",
        parent=parent.id,
        role="researcher",
        group_id="lane",
        current_task_tags=["lane:evidence"],
    )
    parent.children.add(worker.id)
    artifact_path = tmp_workspace / "shared" / "evidence.md"
    artifact_path.write_text("Status: evidence-pending\nCandidate: egalitarian\n")
    worker.artifacts.append(Artifact("shared/evidence.md", artifact_path, agent_id=worker.id))

    result = await meta_spawn(
        {
            "task": "Verify and correct delta evidence, then write shared/evidence.md.",
            "role": "verifier",
            "depends_on": [worker.id],
            "readiness": 0.9,
            "group_id": "lane",
            "current_task_tags": ["role:verifier", "lane:evidence"],
        },
        worker,
        runtime,
    )

    assert "agent_id" in result
    child = runtime.agents[result["agent_id"]]
    assert child.parent == worker.id
    assert child.role == "verifier"
    assert result["readiness"] >= runtime.config.spawn_readiness_threshold


@pytest.mark.asyncio
async def test_meta_spawn_many_skips_verified_duplicate_lane(runtime, tmp_workspace):
    parent = runtime.create_agent("Coordinate GAIA evidence and verification.", group_id="gaia-c61")
    report = tmp_workspace / "shared" / "gaia_l2" / "c61d" / "researcher2_evidence.md"
    report.parent.mkdir(parents=True)
    report.write_text("Status: verified\nTotal Results: 8\nConfidence: High\n")
    verifier = runtime.create_agent(
        "Verify shared/gaia_l2/c61d/researcher2_evidence.md against arXiv physics.soc-ph 2016-08-11.",
        parent=parent.id,
        role="verifier",
        group_id="verification_gaia_l2",
        current_task_tags=["benchmark:gaia", "lane:phys-soc-2016", "role:verifier"],
    )
    parent.children.add(verifier.id)
    verifier.status = "done"
    verifier.result = "Verified physics.soc-ph report."
    verifier.artifacts.append(Artifact("shared/gaia_l2/c61d/researcher2_evidence.md", report, agent_id=verifier.id))
    runtime.memory.update(verifier.id, add_artifacts=["shared/gaia_l2/c61d/researcher2_evidence.md"], tags=verifier.current_task_tags)

    result = await meta_spawn_many(
        {
            "defaults": {
                "create_type": "peer_agent",
                "relationship": "peer",
                "group_id": "verification_gaia_l2",
            },
            "agents": [
                {
                    "task": (
                        "Read shared/gaia_l2/c61d/researcher2_evidence.md. Verify completeness and accuracy "
                        "for all physics.soc-ph papers submitted on 2016-08-11."
                    ),
                    "role": "verifier",
                    "current_task_tags": ["benchmark:gaia", "lane:phys-soc-2016", "role:verifier"],
                }
            ],
        },
        parent,
        runtime,
    )

    assert result["created"] == 0
    assert result["skipped"] == 1
    skipped = result["results"][0]["result"]
    assert skipped["reason"] == "verified_lane_already_covered"
    assert skipped["covered_by"][0]["id"] == verifier.id
    assert len(runtime.agents) == 2
    assert any(e["event"] == "spawn_skipped" for e in runtime._events)


@pytest.mark.asyncio
async def test_meta_spawn_many_does_not_skip_final_delivery_due_to_evidence_lane(runtime, tmp_workspace):
    parent = runtime.create_agent(
        'Question requires submit_answer(answer="<answer-only string>").',
        group_id="gaia-c61",
        orchestration_preference="parallel",
    )
    report = tmp_workspace / "shared" / "gaia_l2" / "c61d" / "bravo_evidence.md"
    report.parent.mkdir(parents=True)
    report.write_text("Status: verified\nSix words: Standardized, Localized, Utilitarian, Egalitarian.\n")
    worker = runtime.create_agent(
        "Collect evidence for lane:ai-regulation-2022 and write shared/gaia_l2/c61d/bravo_evidence.md.",
        parent=parent.id,
        role="evidence",
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "lane:ai-regulation-2022", "role:evidence"],
    )
    parent.children.add(worker.id)
    worker.status = "done"
    worker.result = "Verified evidence: Egalitarian is one candidate word."
    worker.artifacts.append(Artifact("shared/gaia_l2/c61d/bravo_evidence.md", report, agent_id=worker.id))
    runtime.memory.update(
        worker.id,
        public_summary=worker.result,
        add_artifacts=["shared/gaia_l2/c61d/bravo_evidence.md"],
        tags=worker.current_task_tags + ["status:done", "confidence:high"],
    )

    result = await meta_spawn_many(
        {
            "defaults": {
                "create_type": "peer_agent",
                "relationship": "downstream",
                "group_id": "gaia-c61",
            },
            "agents": [
                {
                    "task": (
                        "You are the final delivery agent. Query/read the evidence reports, "
                        "verify the supported answer, and call submit_answer(answer=\"<answer-only string>\")."
                    ),
                    "role": "delivery",
                    "current_task_tags": ["final_delivery", "synthesis", "answer_submission"],
                    "depends_on": [worker.id],
                    "readiness": 1.0,
                }
            ],
        },
        parent,
        runtime,
    )

    assert result["created"] == 1
    assert result["skipped"] == 0
    child_id = result["results"][0]["result"]["agent_id"]
    assert runtime._agent_focus_class(runtime.agents[child_id]) == "final_delivery"


@pytest.mark.asyncio
async def test_meta_spawn_many_allows_verifier_when_existing_coverage_is_incomplete_trace_only(runtime, tmp_workspace):
    parent = runtime.create_agent("Coordinate evidence and verification.", group_id="gaia-c61")
    worker = runtime.create_agent(
        "Collect evidence for lane:phys-soc-2016 and write shared/researcher2_evidence.md.",
        parent=parent.id,
        role="researcher",
        group_id="verification_gaia_l2",
        current_task_tags=["benchmark:gaia", "lane:phys-soc-2016", "role:evidence", "status:incomplete", "needs_verification"],
    )
    parent.children.add(worker.id)
    trace = tmp_workspace / "shared" / ".nanoma" / "evidence" / worker.id / "turn_1.txt"
    trace.parent.mkdir(parents=True)
    trace.write_text("Candidate evidence found, needs verification.\n")
    worker.artifacts.append(Artifact(f"shared/.nanoma/evidence/{worker.id}/turn_1.txt", trace, agent_id=worker.id))
    runtime.memory.update(
        worker.id,
        add_artifacts=[f"shared/.nanoma/evidence/{worker.id}/turn_1.txt"],
        tags=worker.current_task_tags,
    )

    result = await meta_spawn_many(
        {
            "defaults": {
                "create_type": "peer_agent",
                "relationship": "peer",
                "group_id": "verification_gaia_l2",
                "readiness": 1.0,
            },
            "agents": [
                {
                    "task": (
                        "Independently verify lane:phys-soc-2016 evidence using primary sources "
                        "and write shared/verification.md."
                    ),
                    "role": "verifier",
                    "current_task_tags": ["benchmark:gaia", "lane:phys-soc-2016", "role:verifier"],
                }
            ],
        },
        parent,
        runtime,
    )

    assert result["created"] == 1
    assert result["skipped"] == 0
    assert runtime._agent_has_verified_coverage(worker) is False


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
async def test_meta_spawn_many_recovers_top_level_raw_json_args(runtime):
    parent = runtime.create_agent("parent task")
    raw_args = json.dumps({
        "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "wave-raw"},
        "agents": [
            {"task": "check requirements", "role": "requirements"},
            {"task": "write tests", "role": "tester"},
        ],
    })

    result = await meta_spawn_many({"_raw": raw_args}, parent, runtime)

    assert result["created"] == 2
    children = [runtime.agents[item["result"]["agent_id"]] for item in result["results"]]
    assert {child.role for child in children} == {"requirements", "tester"}
    assert all(child.group_id == "wave-raw" for child in children)


@pytest.mark.asyncio
async def test_meta_spawn_many_merges_agents_and_tasks(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn_many(
        {
            "defaults": {"create_type": "peer_agent", "group_id": "wave-merge"},
            "agents": [{"task": "candidate 00", "role": "generator"}],
            "tasks": ["candidate 01", "candidate 02"],
        },
        parent,
        runtime,
    )

    assert result["created"] == 3
    children = [runtime.agents[item["result"]["agent_id"]] for item in result["results"]]
    assert [child.task for child in children] == ["candidate 00", "candidate 01", "candidate 02"]
    assert {child.group_id for child in children} == {"wave-merge"}


@pytest.mark.asyncio
async def test_meta_spawn_many_recovers_agents_string_with_embedded_defaults(runtime):
    parent = runtime.create_agent("parent task")
    agents_string = (
        json.dumps([
            {"task": "Generate candidate 00", "role": "generator", "current_task_tags": ["candidate:00"]},
            {"task": "Generate candidate 01", "role": "generator", "current_task_tags": ["candidate:01"]},
        ])
        + ', "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "gen0-2161F"}'
    )

    result = await meta_spawn_many({"agents": agents_string}, parent, runtime)

    assert result["created"] == 2
    children = [runtime.agents[item["result"]["agent_id"]] for item in result["results"]]
    assert all(child.create_type == "peer_agent" for child in children)
    assert all(child.relationship == "peer" for child in children)
    assert all(child.group_id == "gen0-2161F" for child in children)
    child_tags = [set(child.current_task_tags) for child in children]
    assert {"candidate:00", "candidate:01"} == {tag for tags in child_tags for tag in tags if tag.startswith("candidate:")}
    assert all(f"agent:{child.id}" in child.current_task_tags for child in children)
    assert all("role:generator" in child.current_task_tags for child in children)


@pytest.mark.asyncio
async def test_meta_spawn_many_recovers_complete_prefix_from_truncated_agents_string(runtime):
    parent = runtime.create_agent("parent task")
    agents_string = (
        '['
        '{"task": "Generate candidate 00", "role": "generator-00"},'
        '{"task": "Generate candidate 01", "role": "generator-01"},'
        '{"task": "Generate candidate 02", "role": "generator-02"'
    )

    result = await meta_spawn_many(
        {
            "agents": agents_string,
            "defaults": {"create_type": "peer_agent", "relationship": "peer", "group_id": "gen0-prefix"},
        },
        parent,
        runtime,
    )

    assert result["created"] == 2
    children = [runtime.agents[item["result"]["agent_id"]] for item in result["results"]]
    assert {child.role for child in children} == {"generator-00", "generator-01"}
    assert all(child.group_id == "gen0-prefix" for child in children)


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
async def test_explicit_aggressive_leaf_artifact_lane_is_not_forced_to_solo(tmp_workspace):
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
    assert child.orchestration_preference == "aggressive"
    assert result["orchestration_preference"] == "aggressive"


def test_english_benchmark_protocol_keywords_trigger_spawnable_workstreams():
    task = (
        "Run an OpenDeepThink-style benchmark suite with an initial candidate pool, "
        "20 independent candidates, a pairwise comparison graph, Bradley-Terry ranking, "
        "top 5 elites, bottom 5 mutation wave, gen1 population, tournament selection, "
        "and final synthesis."
    )

    preference, reasons = _infer_task_orchestration_preference(
        parent_preference="balanced",
        task=task,
        workflow_prior="benchmark_protocol",
        current_task_tags=["benchmark:cf73", "scope:full"],
    )

    assert _task_has_multi_phase_peer_protocol(task)
    assert _task_has_spawnable_workstreams(task, min_streams=3)
    assert preference == "aggressive"
    assert any("parallel_score" in reason for reason in reasons)


def test_explicit_peer_wave_requirement_detects_gen0_generators():
    requirements = _explicit_peer_wave_requirements(
        "Spawn 20 independent gen-0 generator agents in parallel for candidate IDs 00..19."
    )

    assert requirements
    assert requirements[0]["target"] == 20
    assert requirements[0]["role"] == "generator"
    assert requirements[0]["phase"] == "gen0"


def test_single_english_candidate_artifact_keywords_stay_leaf_but_do_not_force_solo():
    task = (
        "Generate one candidate solution and save "
        "shared/cf73_top10/2161F/gen0/candidate_07.cpp and "
        "shared/cf73_top10/2161F/gen0/candidate_07.md."
    )

    preference, reasons = _infer_task_orchestration_preference(
        parent_preference="aggressive",
        task=task,
        role="generator",
        group_id="cf73-2161f-gen0",
        current_task_tags=["benchmark:cf73", "phase:gen0", "role:generator", "candidate:07"],
    )

    assert _task_is_leaf_artifact_lane(
        task=task,
        role="generator",
        group_id="cf73-2161f-gen0",
        current_task_tags=["benchmark:cf73", "phase:gen0", "role:generator", "candidate:07"],
    )
    assert not _task_has_multi_phase_peer_protocol(task)
    assert not _task_has_spawnable_workstreams(task, min_streams=3)
    assert preference == "aggressive"
    assert "leaf artifact lane" in reasons


def test_concrete_single_output_discovery_work_balances_without_forcing_solo():
    task = (
        "Query the arXiv API for the August 11 2016 physics.soc-ph article, "
        "extract the article title and relevant society descriptor, and write "
        "shared/gaia_l2/c61d/evidence_society_words.md."
    )

    preference, reasons = _infer_task_orchestration_preference(
        parent_preference="aggressive",
        task=task,
        role="researcher",
        group_id="gaia-l2-c61d",
        current_task_tags=["benchmark:gaia", "phase:evidence", "topic:physics_society"],
    )

    assert _task_is_concrete_single_output_work(task)
    assert not _task_has_spawnable_workstreams(task, min_streams=3)
    assert preference == "balanced"
    assert "concrete single-output work" in reasons


def test_english_peer_state_keywords_prefer_query_before_artifact():
    task = (
        "Before final synthesis, inspect peer state_board, public_memory, memory tags, "
        "completed peers, and artifact paths, then write shared/cf73_top10/2161F/report.md."
    )

    assert _task_prefers_query_before_artifact(task)
    assert _task_is_coordination_artifact_task(task)


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


def test_orchestration_nudge_skips_concrete_single_output_work(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        max_solo_tool_calls_before_spawn=1,
    )
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Query the arXiv API for the August 11 2016 physics.soc-ph article, "
        "extract the article title and relevant society descriptor, and write "
        "shared/gaia_l2/c61d/evidence_society_words.md.",
        model="test",
        orchestration_preference="aggressive",
    )
    agent._turns = 2
    agent._tool_calls = 2

    rt._check_orchestration_nudge(agent)

    assert not [e for e in rt._events if e["event"] == "orchestration_nudge"]
    assert agent._steer_inbox.empty()


@pytest.mark.asyncio
async def test_spawnable_parallel_task_gets_create_action_scope(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
            "tool_choice": kwargs.get("tool_choice"),
            "max_tokens": kwargs.get("max_tokens"),
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
        create_action_max_tokens=15000,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run(
        "Audit and patch shared/source with separable workstreams: requirements audit, implementation patch, "
        "focused tests, and risk review."
    )

    assert {"spawn_many", "spawn", "create_agent", "compact", "set_status", "get_cost"} == set(calls[0]["tools"])
    assert calls[0]["tool_choice"] is None
    assert calls[0]["max_tokens"] == 15000
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

    create_tools = {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"}
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

    assert set(calls[0]["tools"]) == {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"}
    assert calls[0]["tool_choice"] is None
    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "spawn_many"}}
    card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "retry_spawn_after_create_miss" in card
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_create_action_set_status_work_does_not_retry_spawn(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
            "messages": messages,
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(
                    id="tc1",
                    name="set_status",
                    arguments={
                        "action": "work",
                        "current_task_tags": ["status:solo-decision", "phase:artifact"],
                        "work_outline": "Handle this leaf artifact directly before any handoff.",
                    },
                )],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc2",
                name="file_write",
                arguments={"path": "shared/final.md", "content": "done\n"},
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_turns=3,
        max_agents=10,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Patch shared/source with separable audit and patch workstreams. Write shared/final.md.")

    assert set(calls[0]["tools"]) == {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"}
    assert "spawn_many" not in set(calls[1]["tools"])
    assert "file_write" in set(calls[1]["tools"])
    assert len(rt.agents) == 1
    root = rt.agents["alpha"]
    assert root._create_action_filtered_count == 0
    assert any(e["event"] == "tool_call" and e["data"]["tool"] == "file_write" for e in rt._events)


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

    assert set(calls[0]["tools"]) == {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"}
    assert calls[0]["tool_choice"] is None
    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "spawn_many"}}
    assert [e for e in rt._events if e["event"] == "create_action_miss" and e["data"].get("reason") == "no_child_created_in_create_action"]
    assert [e for e in rt._events if e["event"] == "loop_action_auxiliary_only_miss" and e["data"].get("action") == "create"]
    assert len(rt.agents) == 3


def test_constraint_loop_action_policy_scores_candidates(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "ok"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent(
        "Coordinate child work, then write shared/final.md",
        model="test",
        orchestration_preference="parallel",
    )
    child = rt.create_agent("Write shared/child.md", model="test", parent=parent.id)
    parent.children.add(child.id)

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["reason"] == "coordinate_unfinished_children_before_artifacts"
    assert "constraints" in parent._loop_action_plan
    assert parent._loop_action_plan["candidate_scores"][0]["action"] == "message"


def test_constraint_loop_action_keeps_structure_before_root_artifacts(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "ok"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        max_solo_tool_calls_before_spawn=1,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Run a benchmark with generator, comparator, mutation, synthesis, and report workstreams. "
        "Write shared/final.md.",
        model="test",
        orchestration_preference="aggressive",
    )
    agent._turns = 1
    agent._orchestration_nudge_sent = True

    rt._refresh_loop_action_plan(agent, ["shared/final.md"])

    assert agent._loop_action_plan["action"] == "create"
    assert agent._loop_action_plan["reason"] == "spawnable_workstreams_before_solo_execution"
    assert agent._loop_action_plan["candidate_scores"][0]["action"] == "create"


def test_worker_role_is_not_forced_solo_but_self_work_decision_blocks_create(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="parallel",
        spawn_before_turn=1,
        max_solo_tool_calls_before_spawn=1,
        max_agents=10,
    )
    rt = Runtime(config=config)
    worker = rt.create_agent(
        "Run a benchmark with independent search, implementation, validation, and synthesis workstreams. "
        "Write shared/final.md.",
        model="test",
        role="worker",
        orchestration_preference="parallel",
    )
    worker._turns = 1
    worker._orchestration_nudge_sent = True

    assert rt._can_create_child(worker)
    rt._refresh_loop_action_plan(worker, ["shared/final.md"])
    assert worker._loop_action_plan["action"] == "create"

    rt.state_board_update(
        worker.id,
        action_state="work",
        current_task_tags=["status:solo-decision", "phase:artifact"],
    )
    worker._create_action_filtered_count = 1
    rt._refresh_loop_action_plan(worker, ["shared/final.md"])

    assert not rt._can_create_child(worker)
    assert worker._loop_action_plan["action"] == "work"
    assert worker._loop_action_plan["reason"] == "solo_decision_after_create_miss"


def test_constraint_loop_action_preserves_query_before_artifact(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "ok"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Query peers by group_id before writing shared/final.md.",
        model="test",
        orchestration_preference="solo",
    )

    rt._refresh_loop_action_plan(agent, ["shared/final.md"])

    assert agent._loop_action_plan["action"] == "read"
    assert agent._loop_action_plan["reason"] == "task_requests_peer_query_before_artifact"


def test_constraint_loop_action_queries_overlap_before_discovery_work(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA evidence search.", model="test", orchestration_preference="parallel")
    agent = rt.create_agent(
        "Search arXiv and collect evidence about AI regulation papers; write shared/lane_alpha.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
        orchestration_preference="parallel",
    )
    rt.create_agent(
        "Search arXiv for AI regulation evidence and sources; write shared/lane_alpha.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
        orchestration_preference="parallel",
    )

    rt._refresh_loop_action_plan(agent, ["shared/lane_alpha.md"])

    assert agent._loop_action_plan["action"] == "read"
    assert agent._loop_action_plan["reason"] == "peer_overlap_query_before_work"
    assert agent._peer_overlap_query_turn == agent._turns
    assert agent._loop_action_plan["candidate_scores"][0]["action"] == "read"


def test_constraint_loop_action_does_not_query_before_leaf_candidate_work(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate candidates.", model="test", orchestration_preference="parallel")
    agent = rt.create_agent(
        "Write shared/cf73_top10/2161F/gen0/candidate_02.cpp and "
        "shared/cf73_top10/2161F/gen0/candidate_02.md.",
        model="test",
        parent=parent.id,
        group_id="wave",
        role="generator",
        current_task_tags=["phase:gen0", "role:generator", "candidate:02"],
        orchestration_preference="parallel",
    )
    rt.create_agent(
        "Write shared/cf73_top10/2161F/gen0/candidate_00.cpp and "
        "shared/cf73_top10/2161F/gen0/candidate_00.md.",
        model="test",
        parent=parent.id,
        group_id="wave",
        role="generator",
        current_task_tags=["phase:gen0", "role:generator", "candidate:00"],
        orchestration_preference="parallel",
    )

    rt._refresh_loop_action_plan(agent, [
        "shared/cf73_top10/2161F/gen0/candidate_02.cpp",
        "shared/cf73_top10/2161F/gen0/candidate_02.md",
    ])

    assert agent._loop_action_plan["action"] != "read"
    assert agent._loop_action_plan["reason"] != "peer_overlap_query_before_work"


def test_overlap_post_query_ignores_completed_peer_from_different_discovery_lane(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate broad evidence search.", model="test", orchestration_preference="parallel")
    agent = rt.create_agent(
        "Search arXiv for AI regulation papers from June 2022; write shared/ai_regulation.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
        orchestration_preference="parallel",
    )
    peer = rt.create_agent(
        "Search arXiv for physics and society papers from August 2016; write shared/physics_society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    peer.status = "done"
    peer.artifacts.append(Artifact("shared/physics_society.md", tmp_workspace / "shared" / "physics_society.md"))
    agent._peer_overlap_query_turn = 1
    agent._last_query_turn = 1
    agent._turns = 2

    rt._refresh_loop_action_plan(agent, ["shared/ai_regulation.md"])

    assert agent._loop_action_plan["action"] == "work"
    assert agent._loop_action_plan["reason"] != "peer_query_found_completed_peers_prune_or_summarize"


def test_overlap_post_query_does_not_prune_active_duplicate_without_artifact(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA evidence search.", model="test", orchestration_preference="parallel")
    agent = rt.create_agent(
        "Search arXiv for physics and society papers from August 11 2016; write shared/physics_society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    peer = rt.create_agent(
        "Search arXiv for physics and society papers from August 11 2016; write shared/physics_society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    peer._turns = 5
    peer._tool_calls = 3
    peer._last_query_turn = 4
    agent._peer_overlap_query_turn = 1
    agent._last_query_turn = 1
    agent._turns = 2
    agent._tool_calls = 1

    rt._refresh_loop_action_plan(agent, ["shared/physics_society.md"])

    assert agent._loop_action_plan["action"] == "work"
    assert agent._loop_action_plan["reason"] != "peer_query_found_active_duplicate_prune_or_summarize"
    assert not rt._active_duplicate_peer_overlap_candidates(agent)


def test_overlap_post_query_prunes_active_duplicate_with_real_artifact(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA evidence search.", model="test", orchestration_preference="parallel")
    agent = rt.create_agent(
        "Search arXiv for physics and society papers from August 11 2016; write shared/physics_society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    peer = rt.create_agent(
        "Search arXiv for physics and society papers from August 11 2016; write shared/physics_society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    artifact_path = tmp_workspace / "shared" / "physics_society.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("arXiv evidence collected from source page.\n")
    peer.artifacts.append(Artifact("shared/physics_society.md", artifact_path, agent_id=peer.id))
    rt.memory.update(peer.id, add_artifacts=["shared/physics_society.md"], tags=peer.current_task_tags)
    peer._turns = 5
    peer._tool_calls = 3
    peer._last_query_turn = 4
    peer._last_artifact_turn = 4
    agent._peer_overlap_query_turn = 1
    agent._last_query_turn = 1
    agent._turns = 2
    agent._tool_calls = 1

    rt._refresh_loop_action_plan(agent, ["shared/physics_society.md"])

    assert agent._loop_action_plan["action"] == "compact"
    assert agent._loop_action_plan["reason"] == "peer_query_found_active_duplicate_prune_or_summarize"
    assert agent._loop_action_plan["candidate_scores"][0]["action"] == "compact"


def test_overlap_post_query_does_not_prune_active_duplicate_with_placeholder_artifact(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA evidence search.", model="test", orchestration_preference="parallel")
    agent = rt.create_agent(
        "Search arXiv for physics and society papers from August 11 2016; write shared/physics_society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    peer = rt.create_agent(
        "Search arXiv for physics and society papers from August 11 2016; write shared/physics_society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    artifact_path = tmp_workspace / "shared" / "physics_society.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Results pending API execution.\n")
    peer.artifacts.append(Artifact("shared/physics_society.md", artifact_path, agent_id=peer.id))
    rt.memory.update(peer.id, add_artifacts=["shared/physics_society.md"], tags=peer.current_task_tags)
    peer._turns = 5
    peer._tool_calls = 3
    peer._last_query_turn = 4
    peer._last_artifact_turn = 4
    agent._peer_overlap_query_turn = 1
    agent._last_query_turn = 1
    agent._turns = 2
    agent._tool_calls = 1

    rt._refresh_loop_action_plan(agent, ["shared/physics_society.md"])

    assert agent._loop_action_plan["action"] == "work"
    assert agent._loop_action_plan["reason"] != "peer_query_found_active_duplicate_prune_or_summarize"
    assert not rt._active_duplicate_peer_overlap_candidates(agent)


def test_constraint_loop_action_creates_when_deferred_spawn_ready(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        spawn_readiness_threshold=0.6,
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate workers, then create downstream verifier when evidence is ready. Write shared/final.md.",
        model="test",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent("Find evidence", model="test", parent=parent.id, role="worker")
    parent.children.add(worker.id)
    worker.status = "done"
    worker.result = "evidence ready"
    parent._deferred_spawn_requests.append({
        "task": "Verify worker evidence",
        "role": "verifier",
        "depends_on": [worker.id],
        "readiness": 0.1,
        "readiness_threshold": 0.6,
    })

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["root_steward"] is True
    assert parent._loop_action_plan["steward_replaced_action"] == "create"
    assert parent._loop_action_plan["constraints"]["deferred_spawn_readiness"] == 1.0


def test_nested_constraint_loop_action_creates_when_deferred_spawn_ready(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        spawn_readiness_threshold=0.6,
        max_agents=10,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root bootstrap.", model="test", orchestration_preference="parallel")
    parent = rt.create_agent(
        "Coordinate workers, then create downstream verifier when evidence is ready. Write shared/final.md.",
        model="test",
        parent=root.id,
        role="coordinator",
        group_id="nested",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent("Find evidence", model="test", parent=parent.id, role="worker")
    parent.children.add(worker.id)
    worker.status = "done"
    worker.result = "evidence ready"
    parent._deferred_spawn_requests.append({
        "task": "Verify worker evidence",
        "role": "verifier",
        "depends_on": [worker.id],
        "readiness": 0.1,
        "readiness_threshold": 0.6,
    })

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "create"
    assert parent._loop_action_plan["reason"] == "deferred_spawn_readiness_met"
    assert parent._loop_action_plan["constraints"]["deferred_spawn_readiness"] == 1.0


def test_parent_checks_child_artifact_before_deferred_spawn(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        spawn_readiness_threshold=0.6,
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate workers, then create downstream verifier when evidence is ready. Write shared/final.md.",
        model="test",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent("Find evidence and write shared/evidence.md", model="test", parent=parent.id, role="worker")
    parent.children.add(worker.id)
    evidence_path = tmp_workspace / "shared" / "evidence.md"
    evidence_path.write_text("verified child evidence\n")
    worker.status = "done"
    worker.artifacts.append(Artifact("shared/evidence.md", evidence_path, agent_id=worker.id))
    rt.memory.update(worker.id, add_artifacts=["shared/evidence.md"], tags=worker.current_task_tags)
    parent._deferred_spawn_requests.append({
        "task": "Verify worker evidence",
        "role": "verifier",
        "depends_on": [worker.id],
        "readiness": 0.1,
        "readiness_threshold": 0.6,
    })

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "read"
    assert parent._loop_action_plan["reason"] == "child_or_peer_evidence_inspect_before_artifact"
    assert parent._loop_action_plan["constraints"]["evidence_integration_pressure"] == 1.0
    assert parent._loop_action_plan["evidence_agents"][0]["id"] == worker.id


def test_parent_integrates_child_artifact_after_recent_inspection(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate worker evidence and write shared/final.md.",
        model="test",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent("Find evidence and write shared/evidence.md", model="test", parent=parent.id, role="worker")
    parent.children.add(worker.id)
    evidence_path = tmp_workspace / "shared" / "evidence.md"
    evidence_path.write_text("verified child evidence\n")
    worker.status = "done"
    worker.artifacts.append(Artifact("shared/evidence.md", evidence_path, agent_id=worker.id))
    rt.memory.update(worker.id, add_artifacts=["shared/evidence.md"], tags=worker.current_task_tags)
    parent._turns = 6
    parent._last_query_evidence_turn = 5

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])
    allowed, scope = rt._tools_for_loop_turn(parent, parent._loop_action_plan["action"], ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["reason"] == "root_steward_query_and_update_ledger"
    assert parent._loop_action_plan["steward_replaced_action"] == "work"
    assert scope == "root_steward"
    assert {"query", "ledger_read", "ledger_update"} <= allowed
    assert "file_write" not in allowed
    assert "spawn" not in allowed
    assert "shell" not in allowed


def test_deferred_spawn_readiness_matches_lane_tags(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        spawn_readiness_threshold=0.6,
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate lane evidence, create verifier when both lanes are ready, and write shared/final.md.",
        model="test",
        orchestration_preference="parallel",
    )
    figure = rt.create_agent(
        "Find figure evidence.",
        model="test",
        parent=parent.id,
        role="lane1_figure",
        group_id="gaia_l2_c61d",
        current_task_tags=["lane:figure", "feature:2022_paper"],
    )
    society = rt.create_agent(
        "Find society evidence.",
        model="test",
        parent=parent.id,
        role="lane2_society",
        group_id="gaia_l2_c61d",
        current_task_tags=["lane:society", "feature:2016_article"],
    )
    parent.children.update({figure.id, society.id})
    figure.status = "done"
    society.status = "done"
    parent._deferred_spawn_requests.append({
        "task": "Verify overlap after lane evidence is ready.",
        "role": "verifier",
        "depends_on": ["lane:figure", "lane:society"],
        "readiness": 0.0,
        "readiness_threshold": 0.6,
        "downstream_focus": True,
    })

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["root_steward"] is True
    assert parent._loop_action_plan["steward_replaced_action"] == "create"
    assert parent._loop_action_plan["constraints"]["deferred_spawn_readiness"] == 1.0


def test_nested_deferred_spawn_readiness_matches_lane_tags(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        spawn_readiness_threshold=0.6,
        max_agents=10,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root bootstrap.", model="test", orchestration_preference="parallel")
    parent = rt.create_agent(
        "Coordinate lane evidence, create verifier when both lanes are ready, and write shared/final.md.",
        model="test",
        parent=root.id,
        role="coordinator",
        group_id="gaia_l2_c61d",
        orchestration_preference="parallel",
    )
    figure = rt.create_agent(
        "Find figure evidence.",
        model="test",
        parent=parent.id,
        role="lane1_figure",
        group_id="gaia_l2_c61d",
        current_task_tags=["lane:figure", "feature:2022_paper"],
    )
    society = rt.create_agent(
        "Find society evidence.",
        model="test",
        parent=parent.id,
        role="lane2_society",
        group_id="gaia_l2_c61d",
        current_task_tags=["lane:society", "feature:2016_article"],
    )
    parent.children.update({figure.id, society.id})
    figure.status = "done"
    society.status = "done"
    parent._deferred_spawn_requests.append({
        "task": "Verify overlap after lane evidence is ready.",
        "role": "verifier",
        "depends_on": ["lane:figure", "lane:society"],
        "readiness": 0.0,
        "readiness_threshold": 0.6,
        "downstream_focus": True,
    })

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "create"
    assert parent._loop_action_plan["reason"] == "deferred_spawn_readiness_met"
    assert parent._loop_action_plan["constraints"]["deferred_spawn_readiness"] == 1.0


def test_constraint_loop_action_avoids_duplicate_covered_lanes(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="aggressive",
        max_agents=12,
        handoff_child_count_threshold=2,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "GAIA research task with suggested lanes: identify June 2022 AI regulation paper, "
        "identify August 11 2016 physics society article, verify overlap, then write shared/answer.json.",
        model="test",
        orchestration_preference="aggressive",
    )
    for idx, lane in enumerate(["lane:ai-regulation-2022", "lane:phys-soc-2016", "lane:ai-regulation-2022"], start=1):
        child = rt.create_agent(
            f"Evidence lane {idx} for {lane}.",
            model="test",
            parent=parent.id,
            role="evidence",
            group_id="gaia-l2-c61d",
            current_task_tags=[lane, "benchmark:gaia", "role:evidence"],
        )
        parent.children.add(child.id)
        child.status = "done"
        child.result = f"{lane} evidence ready"

    rt._refresh_loop_action_plan(parent, ["shared/answer.json"])

    assert parent._loop_action_plan["action"] == "create"
    assert parent._loop_action_plan["reason"] == "delegate_final_delivery"
    assert parent._loop_action_plan["constraints"]["lane_coverage_pressure"] >= 0.95


def test_covered_spawn_request_requires_same_output_slot_when_explicit(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA lanes.", model="test")
    paper = rt.create_agent(
        "Find paper evidence and write `shared/gaia_l2/c61d/paper_figure_words.md`.",
        model="test",
        parent=parent.id,
        role="evidence",
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "lane:paper"],
    )
    parent.children.add(paper.id)
    artifact_path = tmp_workspace / "shared" / "gaia_l2" / "c61d" / "paper_figure_words.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Source: https://arxiv.org/abs/2206.00000\nVerified paper evidence.\n")
    paper.status = "done"
    paper.result = "paper evidence verified"
    paper.artifacts.append(Artifact("shared/gaia_l2/c61d/paper_figure_words.md", artifact_path, agent_id=paper.id))
    rt.memory.update(
        paper.id,
        public_summary="paper evidence verified",
        add_artifacts=["shared/gaia_l2/c61d/paper_figure_words.md"],
        tags=paper.current_task_tags + ["status:done", "confidence:high"],
    )

    covered = rt.covered_spawn_request(
        parent,
        task=(
            "Recover society evidence and write "
            "`shared/gaia_l2/c61d/society_articles.md`."
        ),
        role="recovery",
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "lane:society"],
    )

    assert covered is None


def test_constraint_loop_action_worker_handoffs_after_mature_evidence(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=12,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        'GAIA question. When ready call submit_answer(answer="<answer-only string>").',
        model="test",
        orchestration_preference="aggressive",
    )
    bravo = rt.create_agent(
        "Identify June 2022 AI regulation figure words.",
        model="test",
        parent=parent.id,
        role="evidence",
        group_id="gaia-c61",
        current_task_tags=["lane:ai-regulation-2022", "benchmark:gaia", "role:evidence"],
        orchestration_preference="aggressive",
    )
    charlie = rt.create_agent(
        "Identify August 11 2016 physics.soc-ph society descriptors.",
        model="test",
        parent=parent.id,
        role="evidence",
        group_id="gaia-c61",
        current_task_tags=["lane:phys-soc-2016", "benchmark:gaia", "role:evidence"],
        orchestration_preference="aggressive",
    )
    parent.children.update({bravo.id, charlie.id})
    for agent, result in [
        (bravo, "Axis words include Egalitarian."),
        (charlie, "Society descriptors include egalitarian."),
    ]:
        agent.status = "done" if agent is charlie else "running"
        agent.result = result
        agent._tool_calls = 6
        rt.memory.update(
            agent.id,
            public_summary=result,
            tags=agent.current_task_tags + ["status:done", "confidence:high"],
        )

    rt._refresh_loop_action_plan(bravo, [])

    assert bravo._loop_action_plan["action"] == "create"
    assert bravo._loop_action_plan["reason"] == "handoff_after_evidence_maturity"
    assert bravo._loop_action_plan["target_agent"] == parent.id
    assert {item["id"] for item in bravo._loop_action_plan["evidence_agents"]} == {bravo.id, charlie.id}


def test_constraint_loop_action_releases_create_when_child_running(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate worker evidence and write shared/final.md.",
        model="test",
        orchestration_preference="parallel",
    )
    child = rt.create_agent("Verify evidence", model="test", parent=parent.id, role="verifier")
    parent.children.add(child.id)
    parent.action_state = "create"
    parent._create_action_filtered_count = 2

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["reason"] in {
        "coordinate_unfinished_children_before_artifacts",
        "wait_after_create_miss_with_active_children",
        "default_task_progress",
    }


def test_create_skip_followup_inspects_coverage_before_retrying_create(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate worker evidence, verify coverage, and write shared/final.md.",
        model="test",
        orchestration_preference="parallel",
    )
    child = rt.create_agent("Collect evidence", model="test", parent=parent.id, role="evidence", group_id="wave")
    parent.children.add(child.id)
    parent.action_state = "create"
    parent._turns = 4
    parent._create_action_filtered_count = 1
    parent._last_spawn_skipped_turn = 4
    parent._last_spawn_skipped_reason = "verified_lane_already_covered"
    parent._last_spawn_skipped_covered_by = [child.id]

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["reason"] == "inspect_coverage_after_spawn_skipped"
    assert parent._loop_action_plan["covered_by"] == [child.id]


@pytest.mark.asyncio
async def test_spawn_skipped_covered_resolves_create_without_retry(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(
                    id="tc1",
                    name="spawn_many",
                    arguments={"agents": [{
                        "task": "Verify already covered evidence lane.",
                        "role": "verifier",
                        "group_id": "wave",
                        "readiness": 1.0,
                    }]},
                )],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc2",
                name="compact",
                arguments={
                    "summary": "Pruned after runtime reported the verifier lane was already covered.",
                    "tags": ["status:pruned", "reason:covered_by_existing_agents"],
                    "stop_after": True,
                    "result": "covered by existing agent",
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=4,
        max_agents=10,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    root = rt.create_agent("Root coordination.", model="test", orchestration_preference="parallel")
    parent = rt.create_agent(
        "Create independent verifier and reviewer agents for separable workstreams, then compact.",
        model="test",
        parent=root.id,
        depth=1,
        role="coordinator",
        group_id="wave",
        current_task_tags=["role:coordinator", "group:wave"],
        orchestration_preference="parallel",
    )
    root.children.add(parent.id)
    covered = rt.create_agent(
        "Verify already covered evidence lane.",
        model="test",
        parent=root.id,
        depth=1,
        role="verifier",
        group_id="wave",
        current_task_tags=["role:verifier", "status:verified"],
    )
    root.children.add(covered.id)
    artifact_path = tmp_workspace / "shared" / "coverage.md"
    artifact_path.write_text("verified coverage\n")
    covered.status = "done"
    covered.result = "verified coverage"
    covered.artifacts.append(Artifact("shared/coverage.md", artifact_path, agent_id=covered.id))
    rt.memory.update(
        covered.id,
        public_summary="verified coverage",
        add_artifacts=["shared/coverage.md"],
        tags=covered.current_task_tags,
    )

    await rt._agent_loop(parent)

    assert parent.status == "done"
    assert parent._create_action_filtered_count == 0
    assert parent._last_create_resolution == "skipped_covered"
    assert [e for e in rt._events if e["event"] == "create_action_resolved"]
    assert not [e for e in rt._events if e["event"] == "create_action_miss"]
    assert any(
        e["event"] == "llm_done" and e["data"].get("loop_action") == "compact"
        for e in rt._events
    )


def test_create_miss_with_completed_children_retries_create_instead_of_work(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate worker evidence, hand off remaining verification, and write shared/final.md.",
        model="test",
        orchestration_preference="parallel",
    )
    child = rt.create_agent("Collect evidence", model="test", parent=parent.id, role="evidence", group_id="wave")
    parent.children.add(child.id)
    child.status = "done"
    child.result = "partial evidence"
    parent.action_state = "create"
    parent._create_action_filtered_count = 1

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])
    allowed, scope = rt._tools_for_loop_turn(parent, parent._loop_action_plan["action"], ["shared/final.md"])

    assert parent._loop_action_plan["action"] in {"message", "compact"}
    assert parent._loop_action_plan["root_steward"] is True
    assert scope == "root_steward"
    assert {"query", "ledger_read", "ledger_update"} & allowed
    assert not {"spawn", "create_agent", "spawn_many"} & allowed


@pytest.mark.asyncio
async def test_premature_input_dependent_agent_compacts_and_stops(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate evidence and confidence check.", model="test")
    worker = rt.create_agent(
        "Find source evidence and write shared/evidence.md.",
        model="test",
        parent=parent.id,
        role="source",
        group_id="wave",
    )
    checker = rt.create_agent(
        "Cross-check source evidence and write shared/check.md.",
        model="test",
        parent=parent.id,
        role="open-focus",
        group_id="wave",
        current_task_tags=["phase:confidence-check"],
    )
    parent.children.update({worker.id, checker.id})

    rt._refresh_loop_action_plan(checker, ["shared/check.md"])
    result = await meta_compact(
        {
            "summary": "Started too early; upstream evidence is not ready.",
            "stop_after": True,
            "result": "pruned until upstream evidence exists",
            "tags": ["status:pruned", "reason:premature_downstream", "needs:upstream_evidence", "phase:confidence-check"],
        },
        checker,
        rt,
    )

    assert checker._loop_action_plan["action"] == "compact"
    assert checker._loop_action_plan["reason"] == "premature_downstream_wait_for_evidence"
    assert result["scheduled"] is True
    assert not rt.completion_blockers(
        checker,
        tags=["status:pruned", "reason:premature_downstream", "needs:upstream_evidence"],
    )


def test_input_dependent_agent_runs_after_upstream_evidence_ready(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate evidence and confidence check.", model="test")
    worker = rt.create_agent(
        "Find source evidence and write shared/evidence.md.",
        model="test",
        parent=parent.id,
        role="source",
        group_id="wave",
    )
    checker = rt.create_agent(
        "Cross-check source evidence and write shared/check.md.",
        model="test",
        parent=parent.id,
        role="open-focus",
        group_id="wave",
        current_task_tags=["phase:confidence-check"],
    )
    parent.children.update({worker.id, checker.id})
    worker.status = "done"
    worker.result = "source evidence ready"
    evidence_path = tmp_workspace / "shared" / "evidence.md"
    evidence_path.write_text("source evidence ready\n")
    worker.artifacts.append(Artifact(path="shared/evidence.md", absolute_path=evidence_path, agent_id=worker.id))

    rt._refresh_loop_action_plan(checker, ["shared/check.md"])

    assert checker._loop_action_plan["reason"] != "premature_downstream_wait_for_evidence"


def test_child_agent_can_create_handoff_after_partial_progress(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_solo_tool_calls_before_spawn=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root coordinates a research task.", model="test")
    child = rt.create_agent(
        "Investigate sources, then hand off remaining confidence check before writing shared/lane.md.",
        model="test",
        parent=root.id,
        role="open-focus",
        group_id="wave",
        current_task_tags=["phase:source-lane"],
        orchestration_preference="parallel",
    )
    root.children.add(child.id)
    child._tool_calls = 4
    child._last_read_evidence_turn = 2
    child._artifact_nudge_count = 2

    rt._refresh_loop_action_plan(child, ["shared/lane.md"])

    assert child._loop_action_plan["action"] == "create"
    assert child._loop_action_plan["reason"] == "handoff_after_partial_progress"


def test_reliable_peer_progress_after_query_suppresses_ordinary_create(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_solo_tool_calls_before_spawn=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root coordinates a research task.", model="test")
    child = rt.create_agent(
        "Investigate sources, then hand off remaining confidence check before writing shared/lane.md.",
        model="test",
        parent=root.id,
        role="open-focus",
        group_id="wave",
        current_task_tags=["phase:source-lane"],
        orchestration_preference="parallel",
    )
    peer = rt.create_agent(
        "Completed related source evidence.",
        model="test",
        parent=root.id,
        role="researcher",
        group_id="wave",
        current_task_tags=["phase:source-lane"],
    )
    root.children.update({child.id, peer.id})
    child._turns = 5
    child._tool_calls = 4
    child._last_read_evidence_turn = 2
    child._last_query_turn = 4
    child._artifact_nudge_count = 2
    peer.status = "done"
    peer.result = "Reliable source evidence ready"

    rt._refresh_loop_action_plan(child, ["shared/lane.md"])

    assert child._loop_action_plan["action"] != "create"
    reasons = {item["reason"] for item in child._loop_action_plan["candidate_scores"]}
    assert "handoff_after_partial_progress" not in reasons
    assert "constraint_alternative_create" not in reasons


def test_middle_agent_with_child_artifact_integrates_before_new_handoff(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_solo_tool_calls_before_spawn=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root coordinates a research task.", model="test")
    middle = rt.create_agent(
        "Investigate sources, delegate as useful, then write shared/lane.md.",
        model="test",
        parent=root.id,
        role="open-focus",
        group_id="wave",
        current_task_tags=["phase:source-lane"],
        orchestration_preference="parallel",
    )
    leaf = rt.create_agent(
        "Find source evidence and write shared/leaf-evidence.md.",
        model="test",
        parent=middle.id,
        role="researcher",
        group_id="wave",
        current_task_tags=["phase:source-lane", "role:evidence"],
    )
    root.children.add(middle.id)
    middle.children.add(leaf.id)
    evidence_path = tmp_workspace / "shared" / "leaf-evidence.md"
    evidence_path.write_text("verified leaf evidence\n")
    leaf.status = "done"
    leaf.artifacts.append(Artifact("shared/leaf-evidence.md", evidence_path, agent_id=leaf.id))
    rt.memory.update(leaf.id, add_artifacts=["shared/leaf-evidence.md"], tags=leaf.current_task_tags)
    middle._turns = 5
    middle._tool_calls = 4
    middle._last_read_evidence_turn = 2
    middle._artifact_nudge_count = 2

    rt._refresh_loop_action_plan(middle, ["shared/lane.md"])

    assert middle._loop_action_plan["action"] in {"read", "work"}
    assert middle._loop_action_plan["reason"] in {
        "child_or_peer_evidence_inspect_before_artifact",
        "integrate_child_or_peer_evidence_into_artifact",
    }
    reasons = {item["reason"] for item in middle._loop_action_plan["candidate_scores"]}
    assert "handoff_after_partial_progress" not in reasons


def test_peer_overlap_respects_distinct_lane_tags_even_with_shared_feature(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA evidence search.", model="test")
    figure = rt.create_agent(
        "Search AI regulation paper evidence and write shared/figure.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "feature:search", "lane:ai_regulation_paper"],
    )
    society = rt.create_agent(
        "Search physics.soc-ph society article evidence and write shared/society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "feature:search", "lane:physics_society_2016"],
    )
    parent.children.update({figure.id, society.id})

    assert society not in rt._peer_overlap_query_candidates(figure)
    assert not rt._peer_overlap_query_before_work_pending(figure, ["shared/figure.md"])


@pytest.mark.asyncio
async def test_pruned_compact_missing_artifact_requires_real_overlap(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    parent = rt.create_agent("Coordinate GAIA evidence search.", model="test")
    agent = rt.create_agent(
        "Search AI regulation paper evidence and write shared/figure.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "feature:search", "lane:ai_regulation_paper"],
    )
    other = rt.create_agent(
        "Search physics.soc-ph society article evidence and write shared/society.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "feature:search", "lane:physics_society_2016"],
    )
    parent.children.update({agent.id, other.id})
    agent._last_query_turn = 2
    other.status = "done"
    other.result = "different lane complete"

    result = await meta_compact(
        {
            "summary": "Pruning as duplicate, but peer is a different lane.",
            "tags": ["status:pruned", "reason:duplicate_lane", "lane:ai_regulation_paper"],
            "stop_after": True,
            "result": "pruned",
        },
        agent,
        rt,
    )

    assert result["blocked"] == "completion_evidence"
    assert any(blocker["kind"] == "missing_outputs" for blocker in result["blockers"])
    assert agent.status == "running"


def test_status_only_query_does_not_satisfy_discovery_report_evidence(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Search arXiv and write evidence report `shared/report.md`.", model="test")
    agent._turns = 4
    agent._last_query_turn = 3

    assert rt._artifact_write_needs_more_evidence(agent, ["shared/report.md"])


def test_query_with_reliable_peer_evidence_satisfies_discovery_report_evidence(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Search arXiv and write evidence report `shared/report.md`.", model="test")
    agent._turns = 4

    result = {
        "agents": [
            {
                "id": "peer",
                "status": "done",
                "result": "verified evidence",
                "artifacts": ["shared/evidence.md"],
                "progress": {"outputs_complete": True},
            }
        ]
    }

    assert rt._query_result_has_reliable_evidence(result)
    agent._last_query_turn = 3
    agent._last_query_evidence_turn = 3
    assert not rt._artifact_write_needs_more_evidence(agent, ["shared/report.md"])


def test_concrete_single_output_child_keeps_work_instead_of_recursive_handoff(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_solo_tool_calls_before_spawn=1,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root coordinates GAIA evidence lanes.", model="test")
    child = rt.create_agent(
        "Query the arXiv API for the August 11 2016 physics.soc-ph article, "
        "extract the article title and relevant society descriptor, and write "
        "shared/gaia_l2/c61d/evidence_society_words.md.",
        model="test",
        parent=root.id,
        role="researcher",
        group_id="gaia-l2-c61d",
        current_task_tags=["benchmark:gaia", "phase:evidence", "topic:physics_society"],
        orchestration_preference="aggressive",
    )
    root.children.add(child.id)
    child._tool_calls = 3
    child._last_read_evidence_turn = 2
    child._artifact_nudge_count = 2

    rt._refresh_loop_action_plan(child, ["shared/gaia_l2/c61d/evidence_society_words.md"])

    assert child._loop_action_plan["action"] == "work"
    reasons = {item["reason"] for item in child._loop_action_plan["candidate_scores"]}
    assert "handoff_after_partial_progress" not in reasons
    assert "constraint_alternative_create" not in reasons


def test_concrete_single_output_uncertain_placeholder_repairs_with_work_not_create(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root coordinates GAIA evidence lanes.", model="test")
    child = rt.create_agent(
        "Search arXiv for the Physics and Society article and write shared/gaia/paper_2016_report.md.",
        model="test",
        parent=root.id,
        role="researcher",
        group_id="gaia-l2-c61d",
        current_task_tags=["benchmark:gaia", "phase:evidence", "topic:physics_society"],
        orchestration_preference="parallel",
    )
    root.children.add(child.id)
    artifact_path = tmp_workspace / "shared" / "gaia" / "paper_2016_report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Status: EVIDENCE-PENDING\nConfidence: Low\nUnverified placeholder.\n")
    child.artifacts.append(Artifact(path="shared/gaia/paper_2016_report.md", absolute_path=artifact_path, agent_id=child.id))
    child._tool_calls = 5
    child._artifact_nudge_count = 3

    assert rt._agent_has_uncertain_evidence(child)

    rt._refresh_loop_action_plan(child, [])

    assert child._loop_action_plan["action"] == "work"
    assert child._loop_action_plan["reason"] == "uncertain_evidence_needs_primary_work"
    reasons = {item["reason"] for item in child._loop_action_plan["candidate_scores"]}
    assert "delegate_uncertain_evidence_recovery" not in reasons


def test_evidence_report_placeholder_content_blocks_completion(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
    )
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Search arXiv and write shared/gaia/paper_2022_findings.md.",
        model="test",
        current_task_tags=["benchmark:gaia", "role:evidence"],
    )
    artifact_path = tmp_workspace / "shared" / "gaia" / "paper_2022_findings.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "# Paper Findings\n\n"
        "## Search in Progress\n\n"
        "Status: Actively searching arXiv. This file will be updated once the paper is identified.\n"
    )
    agent.artifacts.append(Artifact(path="shared/gaia/paper_2022_findings.md", absolute_path=artifact_path, agent_id=agent.id))

    assert rt.missing_expected_outputs(agent) == ["shared/gaia/paper_2022_findings.md"]
    assert not rt.outputs_complete(agent)
    blockers = rt.completion_blockers(agent, include_missing_outputs=False)
    assert any(blocker["kind"] == "uncertain_evidence" for blocker in blockers)

    rt._refresh_loop_action_plan(agent, [])

    assert agent._loop_action_plan["action"] == "work"
    assert agent._loop_action_plan["reason"] == "uncertain_evidence_needs_primary_work"


@pytest.mark.asyncio
async def test_compact_stop_after_blocked_by_pending_evidence_report(runtime, tmp_workspace):
    a = runtime.create_agent("Search arXiv and write shared/gaia/paper_2016_findings.md.")
    artifact_path = tmp_workspace / "shared" / "gaia" / "paper_2016_findings.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "# Physics and Society Articles\n\n"
        "## Results\n"
        "*Pending: Python query to arXiv API needed. Will populate after execution.*\n"
    )
    a.artifacts.append(Artifact("shared/gaia/paper_2016_findings.md", artifact_path, agent_id=a.id))

    result = await meta_compact(
        {
            "summary": "Findings document is complete.",
            "files": ["shared/gaia/paper_2016_findings.md"],
            "tags": ["status:complete", "role:evidence"],
            "stop_after": True,
            "result": "ready",
        },
        a,
        runtime,
    )
    assert result["blocked"] == "completion_evidence"
    assert any(blocker["kind"] == "uncertain_evidence" for blocker in result["blockers"])
    assert a.status == "running"
    assert a.action_state == "work"
    assert a._compact_pending is None


def test_uncertain_leaf_child_can_create_verifier_recovery(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root coordinates evidence lanes.", model="test")
    child = rt.create_agent(
        "Write candidate evidence to shared/lane.md.",
        model="test",
        parent=root.id,
        role="generator",
        group_id="lane",
        current_task_tags=["candidate:1", "lane:evidence"],
        orchestration_preference="parallel",
    )
    root.children.add(child.id)
    artifact_path = tmp_workspace / "shared" / "lane.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Status: EVIDENCE-PENDING / PARTIAL\nConfidence: Low\n")
    child.artifacts.append(Artifact(path="shared/lane.md", absolute_path=artifact_path, agent_id=child.id))
    child._tool_calls = 3

    assert child.orchestration_preference == "parallel"
    assert not rt.outputs_complete(child)
    assert any(b["kind"] == "uncertain_evidence" for b in rt.completion_blockers(child, include_missing_outputs=False))

    rt._refresh_loop_action_plan(child, [])

    assert child._loop_action_plan["action"] == "create"
    assert child._loop_action_plan["reason"] == "delegate_uncertain_evidence_recovery"
    assert child._loop_action_plan["constraints"]["uncertain_evidence_pressure"] == 1.0


def test_explicit_solo_uncertain_child_still_cannot_create(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root coordinates evidence lanes.", model="test")
    child = rt.create_agent(
        "Write candidate evidence to shared/lane.md.",
        model="test",
        parent=root.id,
        role="generator",
        group_id="lane",
        current_task_tags=["candidate:1", "lane:evidence"],
        orchestration_preference="solo",
    )
    root.children.add(child.id)
    artifact_path = tmp_workspace / "shared" / "lane.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Status: EVIDENCE-PENDING / PARTIAL\nConfidence: Low\n")
    child.artifacts.append(Artifact(path="shared/lane.md", absolute_path=artifact_path, agent_id=child.id))
    child._tool_calls = 3

    rt._refresh_loop_action_plan(child, [])

    assert child.orchestration_preference == "solo"
    assert child._loop_action_plan["action"] != "create"


def test_constraint_loop_action_handoff_lowers_parent_create_after_child_wave(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="aggressive",
        handoff_child_count_threshold=2,
        max_agents=20,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate benchmark implementation, tests, review, synthesis, and write shared/final.md.",
        model="test",
        orchestration_preference="aggressive",
    )
    first = rt.create_agent("Implement lane A", model="test", parent=parent.id, role="worker", group_id="wave")
    second = rt.create_agent("Implement lane B", model="test", parent=parent.id, role="worker", group_id="wave")
    parent.children.update({first.id, second.id})
    first.status = "done"
    second.status = "done"
    parent.action_state = "create"
    parent._turns = 6

    rt._refresh_loop_action_plan(parent, ["shared/final.md"])

    assert parent._loop_action_plan["constraints"]["handoff_pressure"] == 1.0
    assert parent._loop_action_plan["action"] != "create"
    create_scores = [
        item for item in parent._loop_action_plan["candidate_scores"]
        if item["action"] == "create"
    ]
    assert not create_scores
    assert parent._loop_action_plan["root_steward"] is True


def test_constraint_loop_action_handoff_lowers_non_root_parent_create_after_child_wave(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="aggressive",
        child_orchestration_preference=None,
        handoff_child_count_threshold=2,
        max_agents=20,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root should bootstrap workers and avoid owning implementation.", model="test")
    parent = rt.create_agent(
        "Coordinate nested benchmark implementation, tests, review, synthesis, and write shared/nested.md.",
        model="test",
        parent=root.id,
        role="coordinator",
        group_id="nested",
        current_task_tags=["phase:nested", "role:coordinator"],
    )
    child_a = rt.create_agent("Implement nested lane A", model="test", parent=parent.id, role="worker", group_id="nested")
    child_b = rt.create_agent("Implement nested lane B", model="test", parent=parent.id, role="worker", group_id="nested")
    parent.children.update({child_a.id, child_b.id})
    child_a.status = "done"
    child_b.status = "done"
    parent.action_state = "create"
    parent._turns = 6

    rt._refresh_loop_action_plan(parent, ["shared/nested.md"])

    assert parent._loop_action_plan["constraints"]["handoff_pressure"] == 1.0
    assert parent._loop_action_plan["action"] != "create"


def test_constraint_loop_action_biases_complex_child_before_own_child_wave(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="balanced",
        child_orchestration_preference=None,
        spawn_before_turn=1,
        max_solo_tool_calls_before_spawn=1,
        max_agents=20,
    )
    rt = Runtime(config=config)
    root = rt.create_agent("Root bootstrap task.", model="test", orchestration_preference="balanced")
    child = rt.create_agent(
        "Run a full benchmark subtask with implementation, tests, review, synthesis, and integration. "
        "Write shared/child-final.md.",
        model="test",
        parent=root.id,
        role="coordinator",
        current_task_tags=["benchmark:subtask", "scope:full"],
    )
    child._turns = 1
    child._tool_calls = 1
    child._orchestration_nudge_sent = True

    rt._refresh_loop_action_plan(child, ["shared/child-final.md"])

    assert child.orchestration_preference == "aggressive"
    assert child._loop_action_plan["constraints"]["offspring_create_bias"] > 0
    assert child._loop_action_plan["action"] == "create"
    assert child._loop_action_plan["reason"] == "spawnable_workstreams_before_solo_execution"


def test_constraint_loop_action_delegates_failed_child_recovery(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate evidence lanes and write `shared/final.json`.",
        model="test",
        orchestration_preference="parallel",
    )
    child = rt.create_agent(
        "Find lane one evidence and write `shared/lane1.md`.",
        model="test",
        parent=parent.id,
        role="lane1",
        group_id="evidence",
    )
    child.status = "failed"
    child.result = "Crash: provider 400"
    child._turns = 4

    rt._refresh_loop_action_plan(parent, ["shared/final.json", "shared/lane1.md"])

    assert parent._loop_action_plan["action"] in {"message", "read"}
    assert parent._loop_action_plan["root_steward"] is True
    assert parent._loop_action_plan["constraints"]["failed_child_pressure"] == 1.0
    assert parent._loop_action_plan.get("steward_replaced_action") in {None, "create", "work"}


@pytest.mark.asyncio
async def test_failed_child_recovery_create_turn_recommends_spawn_many(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc1",
                name="spawn_many",
                arguments={
                    "agents": [{
                        "task": "Recover failed child bravo; write shared/lane1.md.",
                        "role": "recovery",
                        "group_id": "evidence",
                        "current_task_tags": ["status:recovery", "bravo"],
                    }],
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=1,
        max_agents=10,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent(
        "Coordinate evidence lanes and write `shared/final.json`.",
        model="test",
        orchestration_preference="parallel",
    )
    child = rt.create_agent(
        "Find lane one evidence and write `shared/lane1.md`.",
        model="test",
        parent=parent.id,
        role="lane1",
        group_id="evidence",
    )
    child.status = "failed"
    child.result = "Crash: provider 400"

    await rt._agent_loop(parent)

    assert {"query", "ledger_read", "ledger_update"} & set(calls[0]["tools"])
    assert not {"spawn_many", "spawn", "create_agent"} & set(calls[0]["tools"])
    assert calls[0]["tool_choice"] is None
    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "Root steward boundary" in card
    assert len(rt.agents) == 2


@pytest.mark.asyncio
async def test_nested_failed_child_recovery_create_turn_recommends_spawn_many(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc1",
                name="spawn_many",
                arguments={
                    "agents": [{
                        "task": "Recover failed child charlie; write shared/lane1.md.",
                        "role": "recovery",
                        "group_id": "evidence",
                        "current_task_tags": ["status:recovery", "charlie"],
                    }],
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=1,
        max_agents=10,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    root = rt.create_agent("Root bootstrap.", model="test", orchestration_preference="parallel")
    parent = rt.create_agent(
        "Coordinate evidence lanes and write `shared/final.json`.",
        model="test",
        parent=root.id,
        role="coordinator",
        group_id="evidence",
        orchestration_preference="parallel",
    )
    child = rt.create_agent(
        "Find lane one evidence and write `shared/lane1.md`.",
        model="test",
        parent=parent.id,
        role="lane1",
        group_id="evidence",
    )
    child.status = "failed"
    child.result = "Crash: provider 400"

    await rt._agent_loop(parent)

    assert set(calls[0]["tools"]) >= {"spawn_many", "spawn", "create_agent"}
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "spawn_many"}}
    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "delegate_failed_child_recovery" in card
    assert "failed_child=" in card
    assert "Do not take over the failed lane" in card
    assert len(rt.agents) == 4


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
async def test_set_status_read_is_allowed_but_does_not_satisfy_create(tmp_workspace):
    (tmp_workspace / "shared" / "input.txt").write_text("hello")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        return LLMResponse(
            tool_calls=[ToolCall(
                id="tc1",
                name="set_status",
                arguments={"action": "read", "status": "reading", "current_task_tags": ["phase:audit"]},
            )],
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

    await rt.run("Audit `shared/input.txt` with separable research, review, and report workstreams.")

    create_tools = {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"}
    assert set(calls[0]["tools"]) == create_tools
    assert [e for e in rt._events if e["event"] == "tool_call" and e["data"].get("tool") == "set_status"]
    assert [e for e in rt._events if e["event"] == "create_action_miss"]
    assert [e for e in rt._events if e["event"] == "loop_action_auxiliary_only_miss"]


@pytest.mark.asyncio
async def test_compact_in_create_scope_is_allowed_but_does_not_satisfy_create(tmp_workspace):
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
            "messages": messages,
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(
                    id="tc1",
                    name="compact",
                    arguments={"summary": "still deciding", "tags": ["status:planning"]},
                )],
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

    create_tools = {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"}
    assert set(calls[0]["tools"]) == create_tools
    assert calls[0]["tool_choice"] is None
    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "spawn_many"}}
    assert [e for e in rt._events if e["event"] == "loop_action_auxiliary_only_miss" and e["data"].get("tools") == ["compact"]]
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_work_auxiliary_only_turn_retries_with_primary_tools(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="get_cost", arguments={})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc2", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=3,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run("Create `shared/out.txt`.")

    assert "get_cost" in calls[0]["tools"]
    assert "get_cost" not in calls[1]["tools"]
    assert {"file_write", "file_read", "file_list", "grep"} <= set(calls[1]["tools"])
    second_card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "only used auxiliary tools" in second_card
    assert [e for e in rt._events if e["event"] == "loop_action_auxiliary_only_miss" and e["data"].get("action") == "work"]
    assert (tmp_workspace / "shared" / "out.txt").read_text() == "done"


@pytest.mark.asyncio
async def test_stale_message_action_replans_to_missing_output_work(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        tool_names = [t["function"]["name"] for t in (tools or [])]
        calls.append({
            "tools": tool_names,
            "messages": messages,
            "tool_choice": kwargs.get("tool_choice"),
        })
        if "file_write" in tool_names:
            return LLMResponse(
                tool_calls=[ToolCall(id="write", name="file_write", arguments={"path": "shared/final.md", "content": "done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            content="I need to create a verifier or write the final output, but this is a message turn.",
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=5,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent("Write `shared/final.md`.", model="test")
    child = rt.create_agent("Finished child", model="test", parent=parent.id)
    parent.children.add(child.id)
    child.status = "done"
    parent.action_state = "message"
    parent._state_action_version = 1
    parent._deferred_spawn_requests.append({
        "task": "Verify after missing dependency is ready.",
        "role": "verifier",
        "depends_on": ["lane:missing"],
        "readiness": 0.0,
        "readiness_threshold": 0.6,
        "downstream_focus": True,
    })

    await rt._agent_loop(parent)

    assert not (tmp_workspace / "shared" / "final.md").exists()
    assert [e for e in rt._events if e["event"] == "loop_action_replan" and e["data"].get("from_action") == "message"]
    assert all("file_write" not in call["tools"] for call in calls)
    assert any({"query", "ledger_read", "ledger_update"} & set(call["tools"]) for call in calls)


@pytest.mark.asyncio
async def test_stop_action_empty_turn_retries_with_set_status(tmp_workspace):
    (tmp_workspace / "shared" / "final.md").write_text("done")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        if len(calls) == 1:
            return LLMResponse(
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="done", name="set_status", arguments={"action": "done", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=4)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/final.md`.")
    agent.action_state = "stop"
    agent._state_action_version = 1

    await rt._agent_loop(agent)

    assert calls[0]["tool_choice"] is None
    assert set(calls[1]["tools"]) == {"set_status", "compact"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "set_status"}}
    assert agent.status == "done"
    assert [e for e in rt._events if e["event"] == "loop_action_miss" and e["data"].get("action") == "stop"]


@pytest.mark.asyncio
async def test_pre_create_reference_read_runs_before_create(tmp_workspace):
    (tmp_workspace / "shared" / "input.txt").write_text("hello")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tool_choice": kwargs.get("tool_choice"),
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if "Task: audit source" in (messages[0].get("content") or "") or "Task: patch source" in (messages[0].get("content") or ""):
            return LLMResponse(
                tool_calls=[ToolCall(id=f"child-{len(calls)}", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
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

    await rt.run(
        "Coordinate separable research, review, and report workstreams. "
        "Public files: `shared/input.txt`. Inspect it before deciding."
    )

    assert {"file_read", "file_list", "grep", "query", "set_status", "get_cost"} <= set(calls[0]["tools"])
    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    assert not [e for e in rt._events if e["event"] == "create_action_miss" and e["agent"] == "alpha"]
    assert [e for e in rt._events if e["event"] == "pre_action_read" and e["data"].get("target_action") == "create"]
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
                tool_calls=[ToolCall(
                id="tc2",
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
        return LLMResponse(
            tool_calls=[ToolCall(id="tc3", name="set_status", arguments={"status": "done", "result": "done"})],
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
        "Spawn independent generator workstreams for candidates 00 and 01."
    )

    assert {"file_read", "file_list", "grep", "query", "set_status", "get_cost"} <= set(calls[0]["tools"])
    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "resume_create_after_read_orientation" in card
    assert [e for e in rt._events if e["event"] == "pre_action_read" and e["data"].get("target_action") == "create"]
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_create_action_recovers_spawn_many_typo_from_text(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tool_choice": kwargs.get("tool_choice"),
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
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

    create_tools = {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"}
    assert set(calls[0]["tools"]) == create_tools
    assert len(rt.agents) == 3
    assert [e for e in rt._events if e["event"] == "text_tool_call_recovered" and "spawn_many" in e["data"]["tool_calls"]]
    assert [e for e in rt._events if e["event"] == "spawn"]


@pytest.mark.asyncio
async def test_pre_create_read_plan_resumes_create_after_orientation(tmp_workspace):
    (tmp_workspace / "shared" / "input.txt").write_text("benchmark notes\n")
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
                tool_calls=[ToolCall(id="tc2", name="file_read", arguments={"path": "shared/input.txt"})],
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
        "Coordinate separable generator workstreams for candidates 00 and 01. "
        "Public files: `shared/input.txt`. Inspect it before spawning."
    )

    assert "file_read" in calls[0]["tools"]
    assert set(calls[1]["tools"]) == {"spawn", "create_agent", "spawn_many"}
    card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "resume_create_after_read_orientation" in card
    assert len(rt.agents) == 3


@pytest.mark.asyncio
async def test_cf73_generator_leaf_reads_then_writes_without_create_retry(tmp_workspace):
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F").mkdir(parents=True)
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F" / "statement.md").write_text("statement\n")
    (tmp_workspace / "shared" / "cf73_top10" / "problems" / "2161F" / "metadata.json").write_text("{}\n")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
            "tool_choice": kwargs.get("tool_choice"),
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="r1", name="file_read", arguments={"path": "shared/cf73_top10/problems/2161F/statement.md"}),
                    ToolCall(id="r2", name="file_read", arguments={"path": "shared/cf73_top10/problems/2161F/metadata.json"}),
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[
                ToolCall(id="w1", name="file_write", arguments={"path": "shared/cf73_top10/2161F/gen0/candidate_01.cpp", "content": "// candidate\n"}),
                ToolCall(id="w2", name="file_write", arguments={"path": "shared/cf73_top10/2161F/gen0/candidate_01.md", "content": "candidate notes\n"}),
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        max_turns=3,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run(
        "You are generator candidate 01 for CF-73 problem 2161F. "
        "Read `shared/cf73_top10/problems/2161F/statement.md` and "
        "`shared/cf73_top10/problems/2161F/metadata.json`. "
        "Write `shared/cf73_top10/2161F/gen0/candidate_01.cpp` and "
        "`shared/cf73_top10/2161F/gen0/candidate_01.md`.",
    )

    assert {"file_read", "file_list", "grep", "query", "file_write", "file_replace"} & set(calls[0]["tools"])
    assert "spawn_many" not in calls[0]["tools"]
    assert "spawn_many" not in calls[1]["tools"]
    assert "file_write" in calls[1]["tools"]
    assert not [e for e in rt._events if e["event"] == "create_action_miss"]
    assert (tmp_workspace / "shared" / "cf73_top10" / "2161F" / "gen0" / "candidate_01.cpp").is_file()
    assert (tmp_workspace / "shared" / "cf73_top10" / "2161F" / "gen0" / "candidate_01.md").is_file()


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
        "Create `shared/out.cpp` and `shared/out.md`.",
        group_id="wave",
        created_by="root",
        current_task_tags=["phase:gen0", "problem:X", "role:generator"],
    )
    peer.status = "done"
    peer.artifacts.append(Artifact("shared/out.cpp", tmp_workspace / "shared" / "out.cpp"))
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
async def test_overlap_query_before_work_then_prunes_duplicate_discovery_lane(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="query", arguments={"filter": {"group_id": "gaia-c61"}})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="compact",
                    arguments={
                        "summary": "Stopping: peer bravo already has completed AI regulation evidence for this search lane.",
                        "tags": [
                            "benchmark:gaia",
                            "problem:c61",
                            "phase:evidence",
                            "topic:ai_regulation",
                            "status:pruned",
                            "reason:duplicate_lane",
                        ],
                        "stop_after": True,
                        "result": "pruned duplicate discovery lane",
                    },
                )
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=4,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent("Coordinate GAIA evidence search.", model="test", orchestration_preference="parallel")
    agent = rt.create_agent(
        "Search arXiv and collect evidence about AI regulation papers; write shared/lane_alpha.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
        orchestration_preference="parallel",
    )
    peer = rt.create_agent(
        "Search arXiv for AI regulation evidence and sources; write shared/lane_alpha.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
        orchestration_preference="parallel",
    )
    peer.status = "done"
    peer.result = "Completed AI regulation evidence lane."
    peer.artifacts.append(Artifact("shared/lane_alpha.md", tmp_workspace / "shared" / "lane_alpha.md"))

    await rt._agent_loop(agent)

    assert calls[0]["tools"]
    assert "query" in calls[0]["tools"]
    first_card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "discovery/search/evidence work with nearby overlapping peers" in first_card
    assert calls[1]["tools"]
    assert "compact" in calls[1]["tools"]
    second_card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "Peer-progress query has already returned completed" in second_card
    memory = rt.memory.serialize(agent.id)
    assert agent.status == "done"
    assert agent.result == "pruned duplicate discovery lane"
    assert "status:pruned" in memory["tags"]
    assert "reason:duplicate_lane" in memory["tags"]


@pytest.mark.asyncio
async def test_parent_query_can_request_duplicate_agents_self_prune(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="query", arguments={"filter": {"group_id": "gaia-c61"}})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        if len(calls) == 2:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="send",
                        arguments={
                            "to": "charlie",
                            "message_type": "prune_request",
                            "mode": "steer",
                            "urgency": "high",
                            "message": "bravo already produced the AI regulation evidence; compact useful partials and self-prune if your lane is duplicate.",
                            "payload": {
                                "reason": "peer_ahead",
                                "covered_by": ["bravo"],
                                "evidence_agents": [{"id": "bravo", "artifacts": ["shared/lane_bravo.md"]}],
                                "tags": ["benchmark:gaia", "problem:c61", "topic:ai_regulation"],
                            },
                        },
                    )
                ],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="tc3", name="set_status", arguments={"action": "stop", "result": "prune request sent"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=4,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    parent = rt.create_agent(
        "Coordinate GAIA evidence search; query worker state and prune duplicate work.",
        model="test",
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "problem:c61"],
        orchestration_preference="parallel",
    )
    producer = rt.create_agent(
        "Search arXiv for AI regulation evidence; write shared/lane_bravo.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
        orchestration_preference="parallel",
    )
    duplicate = rt.create_agent(
        "Search arXiv for the same AI regulation evidence; write shared/lane_bravo.md.",
        model="test",
        parent=parent.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
        orchestration_preference="parallel",
    )
    assert producer.id == "bravo"
    assert duplicate.id == "charlie"
    producer.status = "done"
    producer.result = "Completed AI regulation evidence lane."
    producer.artifacts.append(Artifact("shared/lane_bravo.md", tmp_workspace / "shared" / "lane_bravo.md"))

    await rt._agent_loop(parent)

    assert duplicate._pending_prune_requests
    assert duplicate.action_state == "compact"
    assert "charlie" in parent._prune_requests_sent_targets
    assert any(e["event"] == "send" and e["data"].get("message_type") == "prune_request" for e in rt._events)
    second_card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "message_type='prune_request'" in second_card


def test_prune_request_plan_covers_same_lane_active_agents_outside_last_query(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=12,
    )
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Coordinate GAIA society evidence verification.",
        model="test",
        group_id="gaia-c61",
        current_task_tags=["benchmark:gaia", "problem:c61"],
        orchestration_preference="parallel",
    )
    report = tmp_workspace / "shared" / "gaia_l2" / "c61d" / "researcher2_evidence.md"
    report.parent.mkdir(parents=True)
    report.write_text("Status: verified\nTotal Results: 8\nConfidence: High\n")
    completed = rt.create_agent(
        "Verify shared/gaia_l2/c61d/researcher2_evidence.md for physics.soc-ph on August 11 2016.",
        model="test",
        parent=parent.id,
        role="verifier",
        group_id="verification_gaia_l2",
        current_task_tags=["benchmark:gaia", "lane:phys-soc-2016", "role:verifier"],
    )
    duplicate = rt.create_agent(
        "Read shared/gaia_l2/c61d/researcher2_evidence.md and re-verify physics.soc-ph papers from 2016-08-11.",
        model="test",
        parent=parent.id,
        role="verifier",
        group_id="verification_gaia_l2",
        current_task_tags=["benchmark:gaia", "lane:phys-soc-2016", "role:verifier"],
    )
    parent.children.update({completed.id, duplicate.id})
    completed.status = "done"
    completed.result = "Verified the society report."
    completed.artifacts.append(Artifact("shared/gaia_l2/c61d/researcher2_evidence.md", report, agent_id=completed.id))
    rt.memory.update(completed.id, add_artifacts=["shared/gaia_l2/c61d/researcher2_evidence.md"], tags=completed.current_task_tags)
    duplicate.status = "running"
    duplicate._turns = 6
    duplicate._tool_calls = 4

    parent._turns = 2
    parent._last_query_turn = parent._turns
    parent._last_query_agent_ids = [completed.id]
    parent._last_query_reliable_evidence_ids = [completed.id]

    rt._refresh_loop_action_plan(parent, [])

    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["reason"] == "request_duplicate_agents_self_prune_after_key_evidence"
    assert parent._loop_action_plan["prune_targets"] == [duplicate.id]


@pytest.mark.asyncio
async def test_prune_request_receiver_compacts_and_self_prunes(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "messages": messages,
        })
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="compact",
                    arguments={
                        "summary": "Self-pruned after coordinator prune_request: bravo already covers this evidence lane. Reusable note: search terms and candidate source overlap the completed lane.",
                        "tags": [
                            "benchmark:gaia",
                            "problem:c61",
                            "phase:evidence",
                            "topic:ai_regulation",
                            "status:pruned",
                            "reason:peer_ahead",
                        ],
                        "stop_after": True,
                        "result": "self-pruned duplicate lane after prune_request",
                    },
                )
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=3,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    coordinator = rt.create_agent("coordinator", model="test")
    worker = rt.create_agent(
        "Search arXiv for AI regulation evidence and write shared/lane_delta.md.",
        model="test",
        parent=coordinator.id,
        group_id="gaia-c61",
        role="researcher",
        current_task_tags=["benchmark:gaia", "problem:c61", "phase:evidence", "topic:ai_regulation"],
    )

    await rt.deliver(Envelope(
        from_id=coordinator.id,
        to_id=worker.id,
        content="bravo already covers this lane; compact partials and self-prune if duplicate.",
        tokens=12,
        timestamp=0.0,
        message_type="prune_request",
        payload={
            "reason": "peer_ahead",
            "covered_by": ["bravo"],
            "evidence_agents": [{"id": "bravo", "artifacts": ["shared/lane_bravo.md"]}],
        },
        urgency="high",
        mode="steer",
    ))

    await rt._agent_loop(worker)

    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    memory = rt.memory.serialize(worker.id)
    assert "prune_request" in card
    assert worker.status == "done"
    assert worker.result == "self-pruned duplicate lane after prune_request"
    assert "status:pruned" in memory["tags"]
    assert "reason:peer_ahead" in memory["tags"]
    assert memory["active_task"] is None


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
    assert "Current action is fixed by runtime: work" in card
    assert "Fixed action: action=work reason=default_task_progress" in card
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
    assert "Current action is fixed by runtime: read" in card
    assert "Fixed action: action=read reason=task_requests_peer_query_before_artifact" in card
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

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
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


def test_explicit_peer_wave_gap_returns_coordinator_to_create(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=100)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Run benchmark. Spawn 20 independent gen-0 generator agents in parallel for candidate IDs 00..19. "
        "Write `shared/cf73_top10/2161F/report.md`.",
        orchestration_preference="aggressive",
    )
    for idx in range(4):
        child = rt.create_agent(
            f"Write shared/cf73_top10/2161F/gen0/candidate_{idx:02d}.cpp.",
            parent=parent.id,
            role="generator",
            group_id="cf73-gen0-2161F",
            current_task_tags=["benchmark:cf73", "phase:gen0", "role:generator", f"candidate:{idx:02d}"],
        )
        child.status = "running" if idx == 0 else "done"

    parent._turns = 6
    parent._last_query_turn = 5
    rt._refresh_loop_action_plan(parent, rt.missing_expected_outputs(parent))

    assert parent._loop_action_plan["action"] == "create"
    assert parent._loop_action_plan["reason"] == "explicit_peer_wave_incomplete"
    assert parent._loop_action_plan["peer_wave_gap"]["target"] == 20
    assert parent._loop_action_plan["peer_wave_gap"]["current"] == 4
    assert parent._loop_action_plan["peer_wave_gap"]["unfinished"] == 1


def test_explicit_peer_wave_gap_returns_initial_zero_wave(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=100)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Run benchmark. Spawn 20 independent gen-0 generator agents in parallel for candidate IDs 00..19. "
        "Write `shared/cf73_top10/2161F/report.md`.",
        orchestration_preference="aggressive",
    )

    rt._refresh_loop_action_plan(parent, rt.missing_expected_outputs(parent))
    card = rt._build_loop_action_context(parent, rt.missing_expected_outputs(parent), [])

    assert parent._loop_action_plan["action"] == "create"
    assert parent._loop_action_plan["reason"] == "explicit_peer_wave_incomplete"
    assert parent._loop_action_plan["peer_wave_gap"]["target"] == 20
    assert parent._loop_action_plan["peer_wave_gap"]["current"] == 0
    assert parent._loop_action_plan["peer_wave_gap"]["remaining"] == 20
    assert "Observed 0 of target 20" in card["content"]


def test_explicit_peer_wave_gap_still_creates_when_existing_wave_running(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=100)
    rt = Runtime(config=config)
    parent = rt.create_agent(
        "Spawn 20 independent gen-0 generator agents in parallel for candidate IDs 00..19. "
        "Write `shared/cf73_top10/2161F/report.md`.",
        orchestration_preference="aggressive",
    )
    for idx in range(13):
        rt.create_agent(
            f"Write shared/cf73_top10/2161F/gen0/candidate_{idx:02d}.cpp.",
            parent=parent.id,
            role="generator",
            group_id="cf73-2161f-gen0",
            current_task_tags=["benchmark:cf73", "phase:gen0", "role:generator", f"candidate:{idx:02d}"],
        )

    parent._turns = 10
    rt._refresh_loop_action_plan(parent, rt.missing_expected_outputs(parent))

    assert parent._loop_action_plan["action"] == "create"
    assert parent._loop_action_plan["reason"] == "explicit_peer_wave_incomplete"
    assert parent._loop_action_plan["peer_wave_gap"]["current"] == 13
    assert parent._loop_action_plan["peer_wave_gap"]["remaining"] == 7
    assert parent._loop_action_plan["peer_wave_gap"]["unfinished"] == 13


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

    allowed, scope = rt._tools_for_loop_turn(parent, parent._loop_action_plan["action"], missing)
    assert parent._loop_action_plan["action"] == "message"
    assert parent._loop_action_plan["root_steward"] is True
    assert parent._loop_action_plan.get("steward_replaced_action") in {None, "create", "work"}
    assert scope == "root_steward"
    assert {"query", "ledger_read", "ledger_update"} <= allowed
    assert "file_read" not in allowed
    assert "shell" not in allowed
    assert "Root steward boundary" in card["content"]


def test_nested_coordinator_delegates_integration_after_child_source_progress(tmp_workspace):
    source = _init_git_source(tmp_workspace, {"pkg/mod.py": "VALUE = 1\n", "tests/test_mod.py": "def test_ok():\n    assert True\n"})
    (source / "pkg" / "mod.py").write_text("VALUE = 2\n")
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_agents=10)
    rt = Runtime(config=config)
    root = rt.create_agent("Root bootstrap.", orchestration_preference="parallel")
    parent = rt.create_agent(
        "Patch shared/source and run tests. After implementers finish, create integration/test review. "
        "Write final report `shared/final_engineering_report.md`.",
        parent=root.id,
        role="coordinator",
        group_id="wave",
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
    assert {"spawn", "create_agent", "spawn_many", "compact", "set_status", "get_cost"} == tools
    assert "query" not in tools
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
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}


@pytest.mark.asyncio
async def test_repeated_artifact_nudge_keeps_final_report_read_write_without_evidence(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_list", arguments={"path": "shared"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create final report `shared/final_report.md`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert {"file_read", "file_list", "grep", "query", "file_write"} <= set(calls[0]["tools"])
    assert calls[0]["tool_choice"] is None
    assert "file_write" in calls[0]["tools"]
    llm_event = [e for e in rt._events if e["event"] == "llm_done"][0]
    assert llm_event["data"]["tool_scope"] == "artifact_read_write"


@pytest.mark.asyncio
async def test_repeated_artifact_nudge_keeps_discovery_report_read_write_without_evidence(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_list", arguments={"path": "shared"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Search arXiv for the Physics and Society article and write shared/gaia/paper_2016_report.md."
    )
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    assert {"file_read", "file_list", "grep", "query", "shell", "file_write"} <= set(calls[0]["tools"])
    assert calls[0]["tool_choice"] is None
    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "Do not write a speculative placeholder" in card
    llm_event = [e for e in rt._events if e["event"] == "llm_done"][0]
    assert llm_event["data"]["tool_scope"] == "artifact_read_write"


def test_plain_research_compact_returns_to_work_and_does_not_count_as_evidence(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Search arXiv for the Physics and Society article and write shared/gaia/paper_2016_report.md."
    )
    agent._turns = 4
    agent._artifact_nudge_count = 3
    agent._compact_pending = {
        "summary": "Researching; still need to search arXiv before writing the report.",
        "tags": ["status:researching", "lane:paper_2016"],
        "files": [],
        "stop_after": False,
    }

    rt._execute_compact(agent)
    allowed, scope = rt._tools_for_loop_turn(agent, "work", ["shared/gaia/paper_2016_report.md"])

    assert agent.action_state == "work"
    assert scope == "artifact_read_write"
    assert "shell" in allowed
    assert "file_write" in allowed


@pytest.mark.asyncio
async def test_repeated_artifact_nudge_allows_final_report_write_only_after_recent_evidence(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/final_report.md", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create final report `shared/final_report.md`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3
    agent._last_read_evidence_turn = 2

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}


@pytest.mark.asyncio
async def test_repeated_artifact_nudge_allows_discovery_report_write_only_after_shell_evidence(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/gaia/paper_2016_report.md", "content": "verified"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Search arXiv for the Physics and Society article and write shared/gaia/paper_2016_report.md."
    )
    agent._turns = 3
    agent._artifact_nudge_count = 3
    agent._shell_commands.append("python3 search_arxiv.py")

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}


@pytest.mark.asyncio
async def test_successful_shell_result_is_published_as_queryable_evidence(tmp_workspace):
    from nanoma.tools import WORK_TOOLS

    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append([t["function"]["name"] for t in (tools or [])])
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="shell", arguments={"command": "python3 search.py"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    async def fake_shell(args, workspace, ctx):
        return {
            "exit_code": 0,
            "stdout": "Found 8 entries; 1608.03637v1 mentions hierarchical and egalitarian societies.",
            "stderr": "",
        }

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=1,
        enabled_work_tools={"shell"},
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    rt.work_tools["shell"] = {"handler": fake_shell, "schema": WORK_TOOLS["shell"]["schema"]}
    agent = rt.create_agent(
        "Search arXiv for the Physics and Society article and write shared/gaia/paper_2016_report.md.",
        model="test",
        current_task_tags=["benchmark:gaia", "lane:society"],
    )

    await rt._agent_loop(agent)

    memory = rt.memory.serialize(agent.id)
    cards = memory["experience_cards"]
    assert any(card["memory_kind"] == "evidence" for card in cards)
    assert any("1608.03637v1" in card["summary"] for card in cards)
    artifacts = [path for card in cards for path in card["artifacts"]]
    assert artifacts
    evidence_path = tmp_workspace / artifacts[0]
    assert evidence_path.exists()
    assert "1608.03637v1" in evidence_path.read_text()
    queried = await meta_query({"target": agent.id, "include_memory": True}, agent, rt)
    assert "1608.03637v1" in json.dumps(queried, ensure_ascii=False)


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

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    (tmp_workspace / "shared" / "out.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_workspace / "shared" / "out.txt").write_text("done")
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3

    await rt._agent_loop(agent)

    card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "All explicit output files named by your task currently exist" in card
    assert "Current action is fixed by runtime: compact" in card
    assert "Fixed action: action=compact reason=outputs_complete_publish_memory" in card
    assert "tool=" not in card
    assert agent.status == "done"
    runtime_memory = rt.memory.serialize(agent.id)
    assert "shared/out.txt" in runtime_memory["artifact_index"]


def test_loop_action_card_does_not_prefer_compact_for_uncertain_outputs(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    out = tmp_workspace / "shared" / "out.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Status: partial / unverified\nConfidence: low\n")
    agent = rt.create_agent("Create `shared/out.txt`.", model="test")
    agent.artifacts.append(Artifact(path="shared/out.txt", absolute_path=out, agent_id=agent.id))
    agent._loop_action_plan = {
        "action": "work",
        "reason": "completion_evidence_required_before_compact",
        "turn_added": 3,
    }

    card = rt._build_loop_action_context(agent, [], [])
    assert card is not None
    text = card["content"]

    assert "All explicit output files named by your task currently exist, but completion blockers remain" in text
    assert "Do not prefer compact/stop merely because files exist" in text
    assert "Prefer action=compact or action=stop" not in text


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

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
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
async def test_compact_action_retries_with_compact_tools_after_read_only_miss(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tool_choice": kwargs.get("tool_choice"),
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if len(calls) == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="tc1", name="file_list", arguments={"path": "shared"})],
                usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
            )
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="compact",
                    arguments={"summary": "complete", "files": ["shared/out.txt"], "stop_after": True, "result": "done"},
                )
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    (tmp_workspace / "shared" / "out.txt").write_text("done")
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3

    await rt._agent_loop(agent)

    assert [e for e in rt._events if e["event"] == "loop_action_auxiliary_only_miss" and e["data"].get("action") == "compact"]
    assert set(calls[1]["tools"]) == {"compact", "set_status"}
    assert calls[1]["tool_choice"] == {"type": "function", "function": {"name": "compact"}}
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
async def test_empty_write_only_response_counts_as_write_miss(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
        })
        if len(calls) == 1:
            return LLMResponse(usage=UsageRecord(input_tokens=10, output_tokens=0, model=model))
        return LLMResponse(
            tool_calls=[ToolCall(id="tc2", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3

    await rt._agent_loop(agent)

    write_misses = [e for e in rt._events if e["event"] == "write_only_miss"]
    assert write_misses
    assert write_misses[0]["data"]["empty_response"] is True
    assert calls[1]["tools"] == ["file_write"]
    retry_card = "\n".join(m.get("content") or "" for m in calls[1]["messages"])
    assert "Previous write-only artifact turns produced assistant text" in retry_card
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
async def test_work_no_tool_miss_retries_with_file_write_when_evidence_exists(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/report.md", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Research the topic and write evidence report `shared/report.md`.")
    agent._turns = 3
    agent._action_miss_action = "work"
    agent._action_miss_count = 2
    agent._shell_commands.append("python3 collect_evidence.py")

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}
    retry_card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "Previous work turns did not execute a usable work tool" in retry_card
    llm_event = [e for e in rt._events if e["event"] == "llm_done"][0]
    assert llm_event["data"]["tool_scope"] == "retry_file_write_after_no_tool_miss"
    assert llm_event["data"]["recommended_tool_choice"] == "file_write"
    assert (tmp_workspace / "shared" / "report.md").read_text() == "done"


@pytest.mark.asyncio
async def test_work_no_tool_miss_retries_with_evidence_tools_when_evidence_missing(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "messages": messages,
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="query", arguments={"q": "peer evidence for report", "limit": 5})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Research the topic and write evidence report `shared/report.md`.")
    agent._turns = 3
    agent._action_miss_action = "work"
    agent._action_miss_count = 2

    await rt._agent_loop(agent)

    assert set(calls[0]["tools"]) == {"file_read", "file_list", "grep", "query", "shell"}
    assert calls[0]["tool_choice"] is None
    retry_card = "\n".join(m.get("content") or "" for m in calls[0]["messages"])
    assert "Runtime will narrow this retry turn to evidence tools" in retry_card
    llm_event = [e for e in rt._events if e["event"] == "llm_done"][0]
    assert llm_event["data"]["tool_scope"] == "retry_evidence_after_no_tool_miss"


@pytest.mark.asyncio
async def test_status_query_no_tool_miss_keeps_evidence_tools_for_discovery_report(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            usage=UsageRecord(input_tokens=10, output_tokens=0, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=6)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Search arXiv and write evidence report `shared/report.md`.")
    agent._turns = 3
    agent._last_query_turn = 3
    agent._action_miss_action = "work"
    agent._action_miss_count = 2

    await rt._agent_loop(agent)

    assert calls[0]["tool_choice"] is None
    assert set(calls[0]["tools"]) == {"file_read", "file_list", "grep", "query", "shell"}
    assert rt._artifact_write_needs_more_evidence(agent, ["shared/report.md"])


@pytest.mark.asyncio
async def test_auxiliary_miss_write_only_retry_recommends_file_write(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="file_write", arguments={"path": "shared/out.txt", "content": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3
    agent._artifact_nudge_count = 3
    agent._auxiliary_only_miss_count = 1

    await rt._agent_loop(agent)

    assert calls[0]["tools"] == ["file_write"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "file_write"}}
    llm_event = [e for e in rt._events if e["event"] == "llm_done"][0]
    assert llm_event["data"]["tool_scope"] == "retry_primary_after_auxiliary_only_miss"
    assert llm_event["data"]["recommended_tool_choice"] == "file_write"
    assert (tmp_workspace / "shared" / "out.txt").read_text() == "done"


@pytest.mark.asyncio
async def test_stop_auxiliary_miss_retries_with_terminal_tools(tmp_workspace):
    (tmp_workspace / "shared" / "out.txt").write_text("done")
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"action": "stop", "result": "done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Create `shared/out.txt`.")
    agent._turns = 3
    agent._loop_action_plan = {"action": "stop", "reason": "current_terminal_action", "turn_added": 3}
    agent._auxiliary_only_miss_count = 1

    await rt._agent_loop(agent)

    assert set(calls[0]["tools"]) == {"set_status", "compact"}
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "set_status"}}
    llm_event = [e for e in rt._events if e["event"] == "llm_done"][0]
    assert llm_event["data"]["tool_scope"] == "retry_primary_after_auxiliary_only_miss"
    assert llm_event["data"]["recommended_tool_choice"] == "set_status"
    assert agent.status == "done"


@pytest.mark.asyncio
async def test_answer_stop_retry_uses_submit_review_scope(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="submit_answer", arguments={"answer": "candidate"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=5)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent('Solve and submit `shared/answer.json` with {"answer": "<answer-only string>"}.')
    agent._turns = 3
    agent._loop_action_plan = {"action": "stop", "reason": "answer_submission_pending", "turn_added": 3}
    agent._auxiliary_only_miss_count = 1

    await rt._agent_loop(agent)

    assert "submit_answer" in calls[0]["tools"]
    assert "query" in calls[0]["tools"]
    assert "ledger_read" in calls[0]["tools"]
    assert calls[0]["tool_choice"] is None
    llm_event = [e for e in rt._events if e["event"] == "llm_done"][0]
    assert llm_event["data"]["tool_scope"] == "answer_submit_review"
    assert llm_event["data"]["recommended_tool_choice"] is None
    assert agent.status == "done"
    assert json.loads((tmp_workspace / "shared" / "answer.json").read_text()) == {"answer": "candidate"}


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
async def test_answer_only_task_does_not_stop_before_evidence_or_spawn(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            content="scope checked",
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        max_turns=1,
        loop_action_policy="constraint",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent(
        "Research three separable evidence lanes and when ready submit `shared/answer.json` "
        "with the final answer."
    )
    agent._turns = 1
    agent._orchestration_nudge_sent = True

    rt._refresh_loop_action_plan(agent, rt.missing_expected_outputs(agent))

    assert agent._loop_action_plan is not None
    assert agent._loop_action_plan["action"] == "create"
    assert agent._loop_action_plan["reason"] in {
        "spawnable_workstreams_before_solo_execution",
        "answer_needs_research_before_submission",
    }


def test_final_answer_path_uses_submit_answer_guidance_not_artifact_guidance(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent('Solve and submit `shared/answer.json` with {"answer": "<answer-only string>"}.')

    system_prompt = agent.history[0]["content"]

    assert "Answer submission guidance" in system_prompt
    assert "Do not write it with file_write" in system_prompt
    assert "Required output files:" not in system_prompt


def test_answer_submission_not_ready_after_only_orientation_read(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        min_spawnable_workstreams=2,
    )
    rt = Runtime(config=config)
    agent = rt.create_agent(
        "Research three separable evidence lanes, inspect shared references, and submit `shared/answer.json`."
    )
    agent._turns = 2
    agent._tool_calls = 2
    agent._last_read_evidence_turn = 1
    agent._create_resume_after_read = True
    agent._orchestration_nudge_sent = True

    assert not rt._answer_submission_ready(agent)

    rt._refresh_loop_action_plan(agent, [])

    assert agent._loop_action_plan["action"] == "create"
    assert agent._loop_action_plan["reason"] in {
        "resume_create_after_read_orientation",
        "spawnable_workstreams_before_solo_execution",
    }


@pytest.mark.asyncio
async def test_final_answer_file_write_does_not_submit_answer(tmp_workspace):
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="file_write",
                    arguments={"path": "shared/answer.json", "content": '{"answer": "placeholder"}'},
                )
            ],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=1)
    rt = Runtime(config=config, llm_call=mock_llm)

    await rt.run('Write `shared/answer.json` with {"answer": "<answer-only string>"}.')

    agent = rt.agents["alpha"]
    assert not (tmp_workspace / "shared" / "answer.json").exists()
    assert rt.missing_expected_outputs(agent) == []
    assert agent.submitted_answer_path is None
    assert agent.status == "failed"
    assert "shared/answer.json" not in [artifact.path for artifact in agent.artifacts]
    writes = [e for e in rt._events if e["event"] == "tool_call" and e["data"]["tool"] == "file_write"]
    assert writes
    write_result = json.loads(writes[-1]["data"]["result"])
    assert write_result["error"] == "final_answer_write_denied"
    assert write_result["reason"] == "terminal_answer_requires_submit_answer"


@pytest.mark.asyncio
async def test_final_answer_compact_stop_requires_submit_answer(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=2)
    rt = Runtime(config=config)
    agent = rt.create_agent('Write `shared/answer.json` with {"answer": "<answer-only string>"}.')
    answer_path = tmp_workspace / "shared" / "answer.json"
    answer_path.write_text('{"answer": "placeholder"}')

    assert rt.missing_expected_outputs(agent) == []
    result = await meta_compact(
        {
            "summary": "final answer appears to be placeholder",
            "files": ["shared/answer.json"],
            "stop_after": True,
            "result": "placeholder",
        },
        agent,
        rt,
    )

    assert result["blocked"] == "completion_evidence"
    assert any(b["kind"] == "answer_submission" for b in result["blockers"])
    assert agent.status == "running"


@pytest.mark.asyncio
async def test_submit_answer_writes_protocol_file_and_finishes_even_if_wrong(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=3)
    rt = Runtime(config=config)
    agent = rt.create_agent('Solve and submit `shared/answer.json` with {"answer": "<answer-only string>"}.')

    result = await meta_submit_answer(
        {
            "answer": "placeholder",
            "confidence": "low",
            "evidence_refs": ["shared/evidence.md"],
            "tags": ["benchmark:gaia", "task:demo"],
        },
        agent,
        rt,
    )

    assert result["submitted_answer"] == "shared/answer.json"
    assert agent.status == "done"
    assert agent.result == "placeholder"
    assert agent.submitted_answer_path == "shared/answer.json"
    assert rt.missing_expected_outputs(agent) == []
    assert rt.completion_blockers(agent) == []
    assert json.loads((tmp_workspace / "shared" / "answer.json").read_text()) == {"answer": "placeholder"}
    assert "shared/answer.json" in [artifact.path for artifact in agent.artifacts]


@pytest.mark.asyncio
async def test_child_verifier_cannot_submit_global_answer_without_grant(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=3)
    rt = Runtime(config=config)
    root = rt.create_agent('Solve GAIA and submit `shared/answer.json` with {"answer": "<answer-only string>"}.')
    child = rt.create_agent(
        'Verify one paper lane. When ready call submit_answer(answer="<answer-only string>").',
        parent=root.id,
        role="verifier",
        group_id="lane1_paper",
        current_task_tags=["role:verification", "lane:paper"],
    )
    root.children.add(child.id)

    result = await meta_submit_answer({"answer": "partial paper answer"}, child, rt)

    assert result["error"] == "submit_answer_denied"
    assert result["reason"] == "submit_answer_not_granted"
    assert child.status == "running"
    assert child.submitted_answer_path is None
    assert not (tmp_workspace / "shared" / "answer.json").exists()
    assert [e for e in rt._events if e["event"] == "submit_answer_denied" and e["agent"] == child.id]


@pytest.mark.asyncio
async def test_child_verifier_does_not_see_submit_answer_tool_without_grant(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "tools": [t["function"]["name"] for t in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "local done"})],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host", max_turns=2)
    rt = Runtime(config=config, llm_call=mock_llm)
    root = rt.create_agent('Solve GAIA and submit `shared/answer.json` with {"answer": "<answer-only string>"}.')
    child = rt.create_agent(
        'Verify one paper lane. When ready call submit_answer(answer="<answer-only string>").',
        parent=root.id,
        role="verifier",
        group_id="lane1_paper",
    )
    root.children.add(child.id)
    child._loop_action_plan = {"action": "stop", "reason": "answer_submission_pending", "turn_added": 1}

    await rt._agent_loop(child)

    assert calls
    assert "submit_answer" not in calls[0]["tools"]
    assert calls[0]["tool_choice"] != {"type": "function", "function": {"name": "submit_answer"}}


@pytest.mark.asyncio
async def test_spawn_strips_submit_answer_from_non_delivery_child(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    root = rt.create_agent('Solve GAIA and submit `shared/answer.json` with {"answer": "<answer-only string>"}.')

    result = await meta_spawn(
        {
            "task": 'Verify one paper lane. When ready call submit_answer(answer="<answer-only string>").',
            "role": "verifier",
            "group_id": "lane1_paper",
        },
        root,
        rt,
    )

    child = rt.agents[result["agent_id"]]
    assert result["submit_answer_granted"] is False
    assert "submit_answer(" not in child.task
    assert "Local completion only" in child.task
    assert [e for e in rt._events if e["event"] == "spawn_task_sanitized"]


@pytest.mark.asyncio
async def test_spawn_grants_submit_answer_to_delivery_child(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    root = rt.create_agent('Solve GAIA and submit `shared/answer.json` with {"answer": "<answer-only string>"}.')

    result = await meta_spawn(
        {
            "task": 'Final delivery: query evidence, verify the final answer, then call submit_answer(answer="<answer-only string>").',
            "role": "finalizer",
            "group_id": "final_delivery",
            "current_task_tags": ["role:finalizer", "final_delivery"],
            "readiness": 1.0,
        },
        root,
        rt,
    )

    child = rt.agents[result["agent_id"]]
    assert result["submit_answer_granted"] is True
    assert result["auto_local_output"] is None
    assert "submit_answer(" in child.task
    assert rt.can_submit_answer(child)["ok"] is True


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
    assert result["task_ledger"]["agent_id"] == target.id
    all_result = await meta_query({}, q, runtime)
    assert all_result["count"] == len(runtime.agents)
    assert "state_board" in all_result
    assert "task_ledger" in all_result
    assert target.id in all_result["public_memory"]


def test_task_ledger_visible_in_every_loop_card(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Research and write `shared/report.md`.", model="test")

    card = rt._build_loop_action_context(agent, ["shared/report.md"], [])

    assert card is not None
    assert "[Task Ledger]" in card["content"]
    assert f"you={agent.id}" in card["content"]
    assert "shared/report.md=missing" in card["content"]
    assert (tmp_workspace / "shared" / ".nanoma" / "task_ledger.json").exists()


def test_ledger_update_persists_output_state(runtime, tmp_workspace):
    agent = runtime.create_agent("Write `shared/report.md`.", model="test")

    result = runtime.task_ledger_update(agent, {
        "status": "blocked",
        "output_path": "shared/report.md",
        "output_status": "unverified",
        "blockers": [{"kind": "needs_evidence", "message": "Need primary source"}],
        "note": "Waiting for evidence.",
    })

    assert result["updated"] is True
    path = tmp_workspace / "shared" / ".nanoma" / "task_ledger.json"
    data = json.loads(path.read_text())
    item = data["items"][agent.id]
    assert item["status"] == "blocked"
    assert item["expected_outputs"][0]["path"] == "shared/report.md"
    assert item["expected_outputs"][0]["status"] == "unverified"
    assert item["blockers"][0]["kind"] == "needs_evidence"


def test_placeholder_artifact_remains_missing_for_scheduling(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Find primary evidence and write `shared/report.md`.", model="test")
    path = tmp_workspace / "shared" / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Status: PENDING VERIFICATION placeholder\n"
        "Primary source not yet available. Results to be confirmed.\n"
    )
    agent.artifacts.append(Artifact("shared/report.md", path, agent_id=agent.id))

    assert rt.expected_outputs(agent) == ["shared/report.md"]
    assert rt.missing_expected_outputs(agent) == ["shared/report.md"]

    item = rt.task_ledger_item_snapshot(agent)

    assert item["expected_outputs"][0]["status"] == "placeholder"
    assert item["status"] == "partial"
    assert any(blocker["kind"] == "uncertain_evidence" for blocker in rt.completion_blockers(agent))


def test_root_steward_mode_restricts_worker_tools(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    root = rt.create_agent("Coordinate evidence collection and final answer.", model="test")
    child = rt.create_agent("Collect evidence.", model="test", parent=root.id)
    root.children.add(child.id)

    tools, scope = rt._tools_for_loop_turn(root, "work", [])

    assert scope == "root_steward"
    assert {"query", "ledger_read", "ledger_update"}.issubset(tools)
    assert "shell" not in tools
    assert "file_write" not in tools
    assert not rt._can_create_child(root)


def test_loop_action_card_fixes_current_action(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)
    agent = rt.create_agent("Write `shared/report.md`.", model="test")
    agent._loop_action_plan = {"action": "work", "reason": "missing_artifacts"}

    card = rt._build_loop_action_context(agent, ["shared/report.md"], [])

    assert card is not None
    assert "Current action is fixed by runtime: work" in card["content"]
    assert "Do not re-select among create, read, message, work, compact, stop" in card["content"]
    assert "Choose exactly one next action" not in card["content"]


def test_parse_selected_loop_action_accepts_json_and_text(tmp_workspace):
    config = RuntimeConfig(workspace_root=tmp_workspace, log_dir=None, sandbox_backend="host")
    rt = Runtime(config=config)

    assert rt._parse_selected_loop_action('{"action":"message","reason":"coordinate"}') == ("message", "coordinate")
    assert rt._parse_selected_loop_action("I would compact now.") == ("compact", "I would compact now.")
    assert rt._parse_selected_loop_action('{"action":"invent","reason":"bad"}') == ("", "")


@pytest.mark.asyncio
async def test_selector_skips_reselection_after_same_action_miss(tmp_workspace, monkeypatch):
    calls = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls["n"] += 1
        return LLMResponse(
            content='{"action":"read","reason":"would reselect"}',
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        two_stage_action_selection=True,
    )
    monkeypatch.setattr("nanoma.core.openai_compatible_call", mock_llm)
    rt = Runtime(config=config, llm_call=mock_llm)
    agent = rt.create_agent("Write `shared/report.md`.", model="test")
    agent._loop_action_plan = {"action": "work", "reason": "missing_artifacts"}
    agent._action_miss_action = "work"
    agent._action_miss_count = 1

    await rt._maybe_select_loop_action(agent, ["shared/report.md"])

    assert calls["n"] == 0
    assert agent._loop_action_plan["action"] == "work"
    assert agent._loop_action_plan["selected_by"] == "runtime_retry_after_action_miss"
    assert [e for e in rt._events if e["event"] == "loop_action_selector_skipped"]


def test_root_steward_delegates_final_delivery_to_create_scope(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_depth=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent(
        "Coordinate evidence, then provide final answer using submit_answer(answer=...).",
        model="test",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent(
        "Find reliable answer evidence.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="gaia",
        current_task_tags=["role:evidence", "status:verified", "confidence:high"],
    )
    root.children.add(worker.id)
    evidence_path = tmp_workspace / "shared" / "evidence.md"
    evidence_path.write_text("Status: verified\nAnswer: Egalitarian\n")
    worker.status = "done"
    worker.result = "Verified answer: Egalitarian"
    worker.artifacts.append(Artifact("shared/evidence.md", evidence_path, agent_id=worker.id))
    rt.memory.update(
        worker.id,
        public_summary="Verified answer evidence: Egalitarian.",
        add_artifacts=["shared/evidence.md"],
        tags=worker.current_task_tags,
    )

    rt._refresh_loop_action_plan(root, [])
    tools, scope = rt._tools_for_loop_turn(root, root._loop_action_plan["action"], [])
    card = rt._build_loop_action_context(root, [], [])

    assert root._loop_action_plan["action"] == "create"
    assert root._loop_action_plan["reason"] == "delegate_final_delivery"
    assert root._loop_action_plan["root_steward"] is True
    assert root._loop_action_plan["candidate_scores"][0]["reason"] == "delegate_final_delivery"
    assert scope == "final_delivery_handoff"
    assert {"spawn", "create_agent", "spawn_many"} == tools
    assert card is not None
    assert "Final delivery handoff" in card["content"]
    assert "Prefer one focused finalizer/delivery agent" in card["content"]


def test_root_final_delivery_uses_mature_output_slots_and_prunes_old_blockers(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_depth=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent(
        'Coordinate evidence and submit `shared/answer.json` with {"answer": "<answer-only string>"} using submit_answer.',
        model="test",
        orchestration_preference="parallel",
    )
    paper = rt.create_agent(
        "Find paper words and write `shared/paper.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="gaia",
    )
    society = rt.create_agent(
        "Find society words and write `shared/society.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="gaia",
    )
    root.children.update({paper.id, society.id})
    for agent, path, content in [
        (paper, "shared/paper.md", "Source: https://arxiv.org/abs/2207.01510\nWords: Egalitarianism, Utilitarianism\n"),
        (society, "shared/society.md", "Source: https://arxiv.org/abs/1608.03637\nWords: egalitarian, hierarchical\n"),
    ]:
        abs_path = tmp_workspace / path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content)
        agent.status = "done"
        agent.artifacts.append(Artifact(path, abs_path, agent_id=agent.id))
        rt.task_ledger_update(agent, {
            "status": "partial",
            "output_path": path,
            "output_status": "placeholder",
            "blockers": [{"kind": "uncertain_evidence", "artifacts": [path], "message": "old blocker"}],
        })

    paper_item = rt.task_ledger_item_snapshot(paper)
    society_item = rt.task_ledger_item_snapshot(society)

    assert paper_item["expected_outputs"][0]["status"] == "evidence_attached"
    assert society_item["expected_outputs"][0]["status"] == "evidence_attached"
    assert not paper_item.get("blockers")
    assert not society_item.get("blockers")

    rt._refresh_loop_action_plan(root, rt.missing_expected_outputs(root))
    tools, scope = rt._tools_for_loop_turn(root, root._loop_action_plan["action"], [])

    assert root._loop_action_plan["action"] == "create"
    assert root._loop_action_plan["reason"] == "delegate_final_delivery"
    assert scope == "final_delivery_handoff"
    assert tools == {"spawn", "create_agent", "spawn_many"}


def test_root_direct_answer_submission_after_recent_evidence_query(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_depth=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent(
        'Coordinate evidence and submit `shared/answer.json` with {"answer": "<answer-only string>"} using submit_answer.',
        model="test",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent(
        "Find reliable answer evidence and write `shared/evidence.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="gaia",
        current_task_tags=["role:evidence", "status:verified", "confidence:high"],
    )
    root.children.add(worker.id)
    evidence_path = tmp_workspace / "shared" / "evidence.md"
    evidence_path.write_text("Source: https://example.test\nAnswer: egalitarian\n")
    worker.status = "done"
    worker.result = "Verified answer: egalitarian"
    worker.artifacts.append(Artifact("shared/evidence.md", evidence_path, agent_id=worker.id))
    rt.memory.update(
        worker.id,
        public_summary="Verified answer evidence: egalitarian.",
        add_artifacts=["shared/evidence.md"],
        tags=worker.current_task_tags,
    )
    root._turns = 5
    root._last_query_turn = 4
    root._last_query_evidence_turn = 4
    root._last_query_agent_ids = [worker.id]
    root._last_query_reliable_evidence_ids = [worker.id]

    rt._refresh_loop_action_plan(root, [])
    tools, scope = rt._tools_for_loop_turn(root, root._loop_action_plan["action"], [])
    card = rt._build_loop_action_context(root, [], [])

    assert root._loop_action_plan["action"] == "stop"
    assert root._loop_action_plan["reason"] == "direct_answer_submission_ready"
    assert root._loop_action_plan["root_steward"] is True
    assert root._loop_action_plan["candidate_scores"][0]["reason"] == "direct_answer_submission_ready"
    assert scope == "answer_submit_only"
    assert tools == {"submit_answer"}
    assert card is not None
    assert "Direct terminal delivery is ready" in card["content"]


def test_root_promotes_terminal_candidate_artifact_to_submit_even_with_running_descendants(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_depth=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent(
        'Coordinate evidence and submit `shared/answer.json` with {"answer": "<answer-only string>"} using submit_answer.',
        model="test",
        orchestration_preference="aggressive",
    )
    evidence = rt.create_agent(
        "Find evidence and write `shared/evidence.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="evidence",
    )
    synthesis = rt.create_agent(
        "Synthesize evidence and write `shared/answer_verification.md` with the candidate answer.",
        model="test",
        parent=evidence.id,
        role="synthesizer",
        group_id="synthesis",
        current_task_tags=["role:synthesis", "status:ready_for_review", "candidate_answer:egalitarian"],
    )
    extra = rt.create_agent("Still-running redundant verifier.", model="test", parent=root.id, role="verifier")
    root.children.update({evidence.id, extra.id})
    evidence.children.add(synthesis.id)

    artifact_path = tmp_workspace / "shared" / "answer_verification.md"
    artifact_path.write_text(
        "# Answer Verification Report\n\n"
        "## Candidate Answer\n\n"
        "**egalitarian**\n\n"
        "Cross-check complete. Sources: https://arxiv.org/abs/2207.01510 and https://arxiv.org/abs/1608.03637.\n",
        encoding="utf-8",
    )
    synthesis.artifacts.append(Artifact("shared/answer_verification.md", artifact_path, agent_id=synthesis.id))
    rt.memory.update(
        synthesis.id,
        public_summary="Cross-check complete. Candidate answer: egalitarian.",
        add_artifacts=["shared/answer_verification.md"],
        tags=synthesis.current_task_tags,
    )
    root._turns = 8
    root._last_query_turn = 7
    root._last_query_agent_ids = [synthesis.id]
    evidence.status = "running"
    extra.status = "running"
    synthesis.status = "running"

    rt._refresh_loop_action_plan(root, [])
    tools, scope = rt._tools_for_loop_turn(root, root._loop_action_plan["action"], [])
    card = rt._build_loop_action_context(root, [], [])

    assert root._loop_action_plan["action"] == "stop"
    assert root._loop_action_plan["reason"] == "terminal_candidate_submit_ready"
    assert root._loop_action_plan["terminal_candidate"]["id"] == synthesis.id
    assert root._loop_action_plan["terminal_candidate"]["answer_hint"] == "egalitarian"
    assert scope == "answer_submit_only"
    assert tools == {"submit_answer"}
    assert card is not None
    assert "answer_hint=egalitarian" in card["content"]

    root_item = rt.task_ledger_item_snapshot(root)
    assert root_item is not None
    assert root_item["status"] == "ready_for_review"
    assert "terminal_candidate_ready" in root_item["note"]


def test_root_partial_evidence_stop_uses_soft_submit_review_scope(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_agents=10,
        max_depth=4,
    )
    rt = Runtime(config=config)
    root = rt.create_agent(
        'Coordinate two evidence lanes and submit `shared/answer.json` with {"answer": "<answer-only string>"} using submit_answer.',
        model="test",
        orchestration_preference="aggressive",
    )
    complete = rt.create_agent(
        "Find one evidence lane and write `shared/lane1.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="evidence",
    )
    incomplete = rt.create_agent(
        "Find the other evidence lane and write `shared/lane2.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="evidence",
    )
    root.children.update({complete.id, incomplete.id})
    path = tmp_workspace / "shared" / "lane1.md"
    path.write_text("Source: https://example.test\nEvidence attached.\n", encoding="utf-8")
    complete.status = "done"
    complete.artifacts.append(Artifact("shared/lane1.md", path, agent_id=complete.id))
    incomplete.status = "running"
    root._loop_action_plan = {
        "action": "stop",
        "reason": "answer_submission_pending",
        "turn_added": 3,
        "answer_readiness": rt._root_answer_readiness_public(rt._root_answer_readiness(root)),
    }

    tools, scope = rt._tools_for_loop_turn(root, root._loop_action_plan["action"], [])
    card = rt._build_loop_action_context(root, [], [])

    assert root._loop_action_plan["action"] == "stop"
    assert scope == "answer_submit_review"
    assert "submit_answer" in tools
    assert "query" in tools
    assert "ledger_read" in tools
    assert card is not None
    assert "Answer submission readiness is a soft signal" in card["content"]


@pytest.mark.asyncio
async def test_selector_cannot_downgrade_final_delivery_to_message(tmp_workspace, monkeypatch):
    import nanoma.core as core

    calls = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls["n"] += 1
        return LLMResponse(
            content='{"action":"message","reason":"query one more time"}',
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        two_stage_action_selection=True,
    )
    monkeypatch.setattr(core, "openai_compatible_call", mock_llm)
    rt = Runtime(config=config, llm_call=mock_llm)
    root = rt.create_agent(
        'Coordinate evidence and submit `shared/answer.json` with {"answer": "<answer-only string>"} using submit_answer.',
        model="test",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent(
        "Find evidence and write `shared/evidence.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="gaia",
    )
    root.children.add(worker.id)
    evidence_path = tmp_workspace / "shared" / "evidence.md"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("Source: https://example.test\nAnswer: egalitarian\n")
    worker.status = "done"
    worker.artifacts.append(Artifact("shared/evidence.md", evidence_path, agent_id=worker.id))

    rt._refresh_loop_action_plan(root, [])
    await rt._maybe_select_loop_action(root, [])

    assert calls["n"] == 1
    assert root._loop_action_plan["action"] == "create"
    assert root._loop_action_plan["reason"] == "delegate_final_delivery"
    assert root._loop_action_selected_by == "runtime_final_delivery_guard"


@pytest.mark.asyncio
async def test_selector_can_soft_downgrade_direct_answer_submission(tmp_workspace, monkeypatch):
    import nanoma.core as core

    calls = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls["n"] += 1
        return LLMResponse(
            content='{"action":"message","reason":"ask one more child"}',
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        two_stage_action_selection=True,
    )
    monkeypatch.setattr(core, "openai_compatible_call", mock_llm)
    rt = Runtime(config=config, llm_call=mock_llm)
    root = rt.create_agent(
        'Coordinate evidence and submit `shared/answer.json` with {"answer": "<answer-only string>"} using submit_answer.',
        model="test",
        orchestration_preference="parallel",
    )
    worker = rt.create_agent(
        "Find evidence and write `shared/evidence.md`.",
        model="test",
        parent=root.id,
        role="evidence",
        group_id="gaia",
        current_task_tags=["role:evidence", "status:verified", "confidence:high"],
    )
    root.children.add(worker.id)
    evidence_path = tmp_workspace / "shared" / "evidence.md"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("Source: https://example.test\nAnswer: egalitarian\n")
    worker.status = "done"
    worker.result = "Verified answer: egalitarian"
    worker.artifacts.append(Artifact("shared/evidence.md", evidence_path, agent_id=worker.id))
    rt.memory.update(
        worker.id,
        public_summary="Verified answer evidence: egalitarian.",
        add_artifacts=["shared/evidence.md"],
        tags=worker.current_task_tags,
    )
    root._turns = 5
    root._last_query_turn = 4
    root._last_query_evidence_turn = 4
    root._last_query_agent_ids = [worker.id]
    root._last_query_reliable_evidence_ids = [worker.id]

    rt._refresh_loop_action_plan(root, [])
    await rt._maybe_select_loop_action(root, [])

    assert calls["n"] == 1
    assert root._loop_action_plan["action"] == "message"
    assert root._loop_action_plan["reason"] == "ask one more child"
    assert root._loop_action_selected_by == "llm_selector"
    assert root._loop_action_plan["selector_previous_action"] == "stop"


@pytest.mark.asyncio
async def test_root_submits_terminal_candidate_and_broadcasts_prune(tmp_workspace):
    calls = []

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls.append({
            "content": "\n".join(str(message.get("content") or "") for message in messages[-2:]),
            "tools": [tool["function"]["name"] for tool in (tools or [])],
            "tool_choice": kwargs.get("tool_choice"),
        })
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="submit_answer", arguments={"answer": "egalitarian"})],
            usage=UsageRecord(input_tokens=20, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        loop_action_policy="constraint",
        max_turns=5,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    root = rt.create_agent(
        'Coordinate evidence and submit `shared/answer.json` with {"answer": "<answer-only string>"} using submit_answer.',
        model="test",
        orchestration_preference="aggressive",
    )
    worker = rt.create_agent("Evidence lane.", model="test", parent=root.id, role="evidence")
    synthesis = rt.create_agent(
        "Synthesize evidence and write `shared/answer_verification.md`.",
        model="test",
        parent=worker.id,
        role="synthesizer",
        group_id="synthesis",
        current_task_tags=["role:synthesis", "candidate_answer:egalitarian"],
    )
    root.children.add(worker.id)
    worker.children.add(synthesis.id)
    artifact_path = tmp_workspace / "shared" / "answer_verification.md"
    artifact_path.write_text(
        "# Answer Verification Report\n\n## Candidate Answer\n\n**egalitarian**\n\nSources: https://example.test\n",
        encoding="utf-8",
    )
    synthesis.artifacts.append(Artifact("shared/answer_verification.md", artifact_path, agent_id=synthesis.id))
    rt.memory.update(
        synthesis.id,
        public_summary="Candidate answer: egalitarian.",
        add_artifacts=["shared/answer_verification.md"],
        tags=synthesis.current_task_tags,
    )
    root._turns = 3
    root._last_query_turn = 3
    root._last_query_agent_ids = [synthesis.id]

    await rt._agent_loop(root)

    assert calls[0]["tools"] == ["submit_answer"]
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "submit_answer"}}
    assert "answer_hint=egalitarian" in calls[0]["content"]
    assert json.loads((tmp_workspace / "shared" / "answer.json").read_text()) == {"answer": "egalitarian"}
    assert worker.id in root._prune_requests_sent_targets
    assert synthesis.id in root._prune_requests_sent_targets
    assert [e for e in rt._events if e["event"] == "terminal_prune_request"]


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
async def test_meta_query_ignores_unknown_agent_id_when_filter_is_present(runtime):
    target = runtime.create_agent("target", role="worker", group_id="diag")
    q = runtime.create_agent("querier")

    result = await meta_query({"agent_id": "made-up-id", "filter": {"group_id": "diag"}, "limit": 5}, q, runtime)

    assert result["count"] == 1
    assert result["agents"][0]["id"] == target.id
    query_events = [e for e in runtime._events if e["event"] == "query"]
    assert query_events[-1]["data"]["scope"] == "agents"
    assert query_events[-1]["data"]["filter"] == {"group_id": "diag"}


@pytest.mark.asyncio
async def test_meta_query_filter_agent_id_matches_direct_agent(runtime):
    target = runtime.create_agent("target", role="worker")
    q = runtime.create_agent("querier")

    result = await meta_query({"filter": {"agent_id": target.id}, "limit": 5}, q, runtime)

    assert result["count"] == 1
    assert result["agents"][0]["id"] == target.id


@pytest.mark.asyncio
async def test_meta_query_finds_agents_by_stable_identity_tags(runtime):
    target = runtime.create_agent(
        "target",
        role="Evidence collector - Lane 2",
        group_id="gaia-l2-c61d",
        current_task_tags=["Benchmark: GAIA", "Task:C61D"],
    )
    q = runtime.create_agent("querier")

    by_id_tag = await meta_query({"tags": [f"Agent:{target.id}"]}, q, runtime)
    by_role_tag = await meta_query({"tags": ["Role: Evidence"]}, q, runtime)
    by_lane_tag = await meta_query({"filter": {"tags": ["Lane: 2"]}}, q, runtime)

    assert target.id in {a["id"] for a in by_id_tag["agents"]}
    assert target.id in {a["id"] for a in by_role_tag["agents"]}
    assert target.id in {a["id"] for a in by_lane_tag["agents"]}


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
    assert queried["public_memory"]["experience_cards"][0]["memory_kind"] == "experience"
    assert "memory_kind:experience" in queried["public_memory"]["experience_cards"][0]["tags"]
    assert "memory_source:compact" in queried["public_memory"]["experience_cards"][0]["tags"]
    assert queried["state_board"]["action_state"] == "work"


@pytest.mark.asyncio
async def test_compact_memory_layers_preserve_kb_tags_for_query(runtime):
    a = runtime.create_agent(
        "GAIA worker",
        role="Evidence collector - Lane 3",
        group_id="gaia-l2-c61d",
        current_task_tags=["Benchmark: GAIA", "Task:C61D", "Source:PDF"],
    )

    result = await meta_compact(
        {
            "summary": "Found supporting PDF evidence for the answer.",
            "tags": ["Status:Complete", "Citation:Paper"],
            "files": ["shared/evidence.md"],
            "memory_kind": "evidence",
            "memory_source": "artifact",
            "evidence": ["PDF", "Cross Check"],
        },
        a,
        runtime,
    )
    assert result["scheduled"] is True

    runtime._execute_compact(a)

    memory = runtime.memory.serialize(a.id)
    tags = set(memory["tags"])
    card = memory["experience_cards"][0]
    card_tags = set(card["tags"])
    assert card["memory_kind"] == "evidence"
    assert card["memory_source"] == "artifact"
    assert "memory_kind:evidence" in card_tags
    assert "memory_source:artifact" in card_tags
    assert "evidence:pdf" in card_tags
    assert "agent:" + a.id in tags
    assert {"role:evidence", "lane:3", "benchmark:gaia", "task:c61d", "source:pdf"} <= tags

    read = runtime.memory.read(intent="find GAIA evidence", seed_terms=["memory_kind:evidence", "benchmark:gaia"])
    assert read["matches"]
    assert read["matches"][0]["matched_cards"]


@pytest.mark.asyncio
async def test_meta_compact_splits_summary_into_tagged_experience_cards(runtime):
    a = runtime.create_agent("worker", current_task_tags=["Benchmark:GAIA", "Task:C61D"])
    summary = (
        "# Evidence\nFound arXiv 2207.01510 and extracted Standardization vs Localization.\n\n"
        "# Physics\nFound arXiv 1608.03637 and extracted egalitarian societies.\n\n"
        "# Answer\nThe overlap should be egalitarian."
    )

    result = await meta_compact({"summary": summary, "tags": ["Role:Synthesizer"]}, a, runtime)
    assert result["scheduled"] is True
    runtime._execute_compact(a)

    memory = runtime.memory.serialize(a.id)
    cards = memory["experience_cards"]
    assert len(cards) >= 3
    assert any("2207.01510" in card["summary"] for card in cards)
    assert any("1608.03637" in card["summary"] for card in cards)
    assert all("role:synthesizer" in card["tags"] for card in cards)
    assert all("memory_source:compact" in card["tags"] for card in cards)
    assert any("memory_part:1" in card["tags"] for card in cards)

    read = runtime.memory.read(intent="physics evidence", seed_terms=["role:synthesizer"])
    assert read["matches"][0]["matched_cards"]


@pytest.mark.asyncio
async def test_meta_compact_truth_guard_downgrades_unverified_completion(runtime):
    a = runtime.create_agent("worker")
    result = await meta_compact(
        {
            "summary": "Status: UNVERIFIED. Low confidence; requires verification before final use.",
            "tags": ["status:complete", "role:evidence"],
            "stop_after": True,
            "result": "confirmed answer",
        },
        a,
        runtime,
    )
    assert result["scheduled"] is True

    runtime._execute_compact(a)

    memory = runtime.memory.serialize(a.id)
    assert "status:complete" not in memory["tags"]
    assert "status:unverified" in memory["tags"]
    assert "needs:verification" in memory["tags"]
    assert "confidence:low" in memory["tags"]
    assert memory["public_summary"].startswith("[UNVERIFIED / NEEDS VERIFICATION]")
    assert a.status == "running"
    assert a.action_state == "work"
    assert a.result is None
    assert memory["active_task"] is not None
    assert [e for e in runtime._events if e["event"] == "compact_truth_guard"]


@pytest.mark.asyncio
async def test_meta_compact_truth_guard_uses_artifact_uncertainty(runtime, tmp_workspace):
    a = runtime.create_agent("worker")
    artifact_path = tmp_workspace / "shared" / "report.md"
    artifact_path.write_text("Status: BEST EFFORT / UNVERIFIED\nConfidence: Low\n")
    a.artifacts.append(Artifact("shared/report.md", artifact_path))

    result = await meta_compact(
        {
            "summary": "Evidence complete and ready for final synthesis.",
            "files": ["shared/report.md"],
            "tags": ["status:complete", "role:evidence"],
            "stop_after": True,
            "result": "ready",
        },
        a,
        runtime,
    )
    assert result["scheduled"] is True

    runtime._execute_compact(a)

    memory = runtime.memory.serialize(a.id)
    assert "status:complete" not in memory["tags"]
    assert "status:unverified" in memory["tags"]
    assert memory["public_summary"].startswith("[UNVERIFIED / NEEDS VERIFICATION]")
    assert a.status == "running"
    assert a.action_state == "work"
    assert a.result is None
    assert memory["active_task"] is not None


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
    assert any(card["memory_kind"] == "terminal" for card in memory["experience_cards"])
    assert "memory_kind:terminal" in memory["tags"]


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
async def test_tool_file_list_expands_shared_alias(tmp_workspace):
    from nanoma.tools import tool_file_list

    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (tmp_workspace / "shared" / "evidence.txt").write_text("ok")

    result = await tool_file_list({"path": "$SHARED"}, ws, ctx)

    assert "error" not in result
    assert [entry["name"] for entry in result["entries"]] == ["evidence.txt"]


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


def test_mcp_schemas_keep_optional_arguments_model_compat_is_scoped():
    from nanoma.tools import WORK_TOOLS

    schemas = [
        WORK_TOOLS["file_list"]["schema"],
        META_TOOLS["spawn_many"]["schema"],
        META_TOOLS["query"]["schema"],
        META_TOOLS["wait"]["schema"],
        META_TOOLS["set_status"]["schema"],
        META_TOOLS["get_cost"]["schema"],
        WORK_TOOLS["bt_aggregate"]["schema"],
    ]

    assert [
        schema["function"]["parameters"].get("required")
        for schema in schemas
    ] == [None, None, None, None, None, None, None]
    assert "enum" not in META_TOOLS["set_status"]["schema"]["function"]["parameters"]["properties"]["action"]

    untouched = _adapt_tool_schemas_for_model(schemas, "deepseek-v4-flash")
    assert untouched is schemas
    assert [schema["function"]["parameters"].get("required") for schema in untouched] == [None, None, None, None, None, None, None]

    adapted = _adapt_tool_schemas_for_model(schemas, "deepseek-v4-pro")
    assert adapted is not schemas
    assert [schema["function"]["parameters"].get("required") for schema in adapted] == [
        ["path"],
        ["agents"],
        ["filter"],
        ["agent_ids"],
        ["action"],
        [],
        ["directory"],
    ]
    assert "enum" in adapted[4]["function"]["parameters"]["properties"]["action"]
    assert [schema["function"]["parameters"].get("required") for schema in schemas] == [None, None, None, None, None, None, None]


def test_v4_pro_meta_status_tools_are_adapted_without_removal():
    schemas = [
        META_TOOLS["get_cost"]["schema"],
        META_TOOLS["set_status"]["schema"],
        META_TOOLS["compact"]["schema"],
    ]

    adapted = _adapt_tool_schemas_for_model(schemas, "deepseek-v4-pro")

    assert [schema["function"]["name"] for schema in adapted] == ["get_cost", "set_status", "compact"]
    assert adapted[0]["function"]["parameters"].get("required") == []
    assert adapted[1]["function"]["parameters"].get("required") == ["action"]
    assert adapted[2]["function"]["parameters"].get("required") == ["summary"]
    assert "enum" in adapted[1]["function"]["parameters"]["properties"]["action"]


def test_v4_pro_schema_compat_matches_short_and_provider_versioned_names():
    assert _model_needs_required_arg_tool_schema("deepseek-v4-pro")
    assert _model_needs_required_arg_tool_schema("deepseek/deepseek-v4-pro")
    assert _model_needs_required_arg_tool_schema("deepseek/deepseek-v4-pro-20260423")
    assert not _model_needs_required_arg_tool_schema("deepseek-v4-flash")


def test_v4_pro_named_tool_choice_downgrades_to_schema_only_auto():
    named_choice = {"type": "function", "function": {"name": "file_write"}}

    assert _adapt_tool_choice_for_model(named_choice, "deepseek-v4-pro") == "auto"
    assert _adapt_tool_choice_for_model(named_choice, "deepseek/deepseek-v4-pro") == "auto"
    assert _adapt_tool_choice_for_model("auto", "deepseek-v4-pro") == "auto"
    assert _adapt_tool_choice_for_model(None, "deepseek-v4-pro") is None
    assert _adapt_tool_choice_for_model(named_choice, "deepseek-v4-flash") is named_choice


def test_empty_zero_usage_llm_response_is_transient():
    data = {
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    with pytest.raises(TransientEmptyLLMResponse):
        _raise_for_transient_empty_response(data)


def test_empty_content_with_real_usage_is_not_transient():
    data = {
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 0, "total_tokens": 123},
    }

    _raise_for_transient_empty_response(data)


def test_pro_tool_turn_text_zero_usage_is_transient_transport_miss():
    data = {
        "choices": [{"message": {"role": "assistant", "content": "I will call spawn_many now."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    with pytest.raises(TransientToolCallTransportMiss):
        _raise_for_transient_tool_call_transport_miss(
            data,
            model="deepseek-v4-pro",
            tools=[META_TOOLS["spawn_many"]["schema"]],
        )


def test_non_pro_tool_turn_text_zero_usage_is_not_transport_miss():
    data = {
        "choices": [{"message": {"role": "assistant", "content": "I will call spawn_many now."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    _raise_for_transient_tool_call_transport_miss(
        data,
        model="deepseek-v4-flash",
        tools=[META_TOOLS["spawn_many"]["schema"]],
    )


def test_tool_retry_repair_is_scoped_to_v4_pro_tool_transients():
    tools = [META_TOOLS["spawn_many"]["schema"]]

    assert _should_repair_tool_retry(TransientEmptyLLMResponse("empty"), model="deepseek-v4-pro", tools=tools)
    assert not _should_repair_tool_retry(TransientEmptyLLMResponse("empty"), model="deepseek-v4-flash", tools=tools)
    assert not _should_repair_tool_retry(RuntimeError("other"), model="deepseek-v4-pro", tools=tools)
    assert not _should_repair_tool_retry(TransientEmptyLLMResponse("empty"), model="deepseek-v4-pro", tools=[])


def test_body_with_tool_retry_repair_appends_transient_message_without_mutating_original():
    body = {"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "spawn"}]}

    repaired = _body_with_tool_retry_repair(body)

    assert repaired is not body
    assert len(body["messages"]) == 1
    assert len(repaired["messages"]) == 2
    assert repaired["messages"][-1]["role"] == "user"
    assert "valid tool_call" in repaired["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_openai_compatible_call_retries_empty_zero_usage_response(monkeypatch):
    import nanoma.llm as llm_mod

    responses = [
        {
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    ]

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.calls = 0

        async def post(self, *args, **kwargs):
            data = responses[self.calls]
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(200, json=data, request=request)

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)

    response = await openai_compatible_call(
        [{"role": "user", "content": "hello"}],
        "model-x",
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=1, base_delay=0, max_delay=0),
    )

    assert fake.calls == 2
    assert response.content == "ok"
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_openai_compatible_call_logs_http_status_error_details(monkeypatch, tmp_path):
    import nanoma.llm as llm_mod

    class FakeClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                418,
                json={"error": {"message": "bad tool schema", "type": "invalid_request_error"}},
                headers={"x-request-id": "req-test", "retry-after": "5"},
                request=request,
            )

    monkeypatch.setattr(llm_mod, "_shared_client", FakeClient())
    monkeypatch.setattr(llm_mod, "_log_dir", tmp_path)
    monkeypatch.setattr(llm_mod, "_log_counter", 0)

    with pytest.raises(httpx.HTTPStatusError):
        await openai_compatible_call(
            [{"role": "user", "content": "hello"}],
            "deepseek-v4-pro",
            tools=[META_TOOLS["spawn_many"]["schema"]],
            base_url="https://example.test/v1",
            api_key="test-key",
            retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
        )

    logs = list(tmp_path.glob("*_deepseek-v4-pro.jsonl"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text())
    assert payload["error"] == "HTTPStatusError"
    assert payload["response_status"] == 418
    assert payload["response_json"]["error"]["message"] == "bad tool schema"
    assert payload["response_text"]
    assert payload["response_headers"]["x-request-id"] == "req-test"
    assert payload["retry"] is False
    assert payload["request_url"] == "https://example.test/v1/chat/completions"


@pytest.mark.asyncio
async def test_openai_compatible_call_does_not_retry_non_retryable_http_status(monkeypatch, tmp_path):
    import nanoma.llm as llm_mod

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.calls = 0

        async def post(self, *args, **kwargs):
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                402,
                json={"error": {"message": "Insufficient Balance", "type": "invalid_request_error"}},
                request=request,
            )

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)
    monkeypatch.setattr(llm_mod, "_log_dir", tmp_path)
    monkeypatch.setattr(llm_mod, "_log_counter", 0)

    with pytest.raises(httpx.HTTPStatusError):
        await openai_compatible_call(
            [{"role": "user", "content": "hello"}],
            "deepseek-v4-pro",
            base_url="https://example.test/v1",
            api_key="test-key",
            retry_config=RetryConfig(max_retries=3, base_delay=0, max_delay=0),
        )

    logs = list(tmp_path.glob("*_deepseek-v4-pro.jsonl"))
    assert fake.calls == 1
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text())
    assert payload["response_status"] == 402
    assert payload["response_json"]["error"]["message"] == "Insufficient Balance"
    assert payload["retry"] is False


@pytest.mark.asyncio
async def test_openai_compatible_call_applies_required_arg_schema_only_for_v4_pro(monkeypatch):
    import nanoma.llm as llm_mod

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.bodies = []

        async def post(self, *args, **kwargs):
            self.bodies.append(kwargs["json"])
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                },
                request=request,
            )

    tool = META_TOOLS["spawn_many"]["schema"]
    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)

    await openai_compatible_call(
        [{"role": "user", "content": "hello"}],
        "deepseek-v4-flash",
        tools=[tool],
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
    )
    await openai_compatible_call(
        [{"role": "user", "content": "hello"}],
        "deepseek-v4-pro",
        tools=[tool],
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
    )

    assert fake.bodies[0]["tools"][0]["function"]["parameters"].get("required") is None
    assert fake.bodies[1]["tools"][0]["function"]["parameters"].get("required") == ["agents"]
    assert tool["function"]["parameters"].get("required") is None


@pytest.mark.asyncio
async def test_openai_compatible_call_uses_schema_only_for_v4_pro_named_tool_choice(monkeypatch):
    import nanoma.llm as llm_mod
    from nanoma.tools import WORK_TOOLS

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.bodies = []

        async def post(self, *args, **kwargs):
            self.bodies.append(kwargs["json"])
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "file_write",
                                            "arguments": json.dumps({"path": "shared/out.txt", "content": "ok"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                },
                request=request,
            )

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)

    await openai_compatible_call(
        [{"role": "user", "content": "write"}],
        "deepseek-v4-pro",
        tools=[WORK_TOOLS["file_write"]["schema"]],
        tool_choice={"type": "function", "function": {"name": "file_write"}},
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
    )

    assert fake.bodies[0]["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in fake.bodies[0]["tools"]] == ["file_write"]


@pytest.mark.asyncio
async def test_openai_compatible_call_sanitizes_old_tool_names_for_v4_pro_schema_only(monkeypatch):
    import nanoma.llm as llm_mod
    from nanoma.tools import WORK_TOOLS

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.bodies = []

        async def post(self, *args, **kwargs):
            self.bodies.append(kwargs["json"])
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "file_write",
                                            "arguments": json.dumps({"path": "shared/out.txt", "content": "ok"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                },
                request=request,
            )

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)
    messages = [
        {"role": "system", "content": "Use shell and file_list earlier, but now write."},
        {
            "role": "assistant",
            "content": "I will call shell.",
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "shell", "arguments": json.dumps({"command": "ls"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": '{"stdout": ""}'},
        {"role": "user", "content": "Ignore file_list, query_help, and call file_write."},
    ]

    await openai_compatible_call(
        messages,
        "deepseek-v4-pro",
        tools=[WORK_TOOLS["file_write"]["schema"]],
        tool_choice={"type": "function", "function": {"name": "file_write"}},
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
    )

    body_messages = fake.bodies[0]["messages"]
    encoded = json.dumps(body_messages)
    assert '"tool_calls"' not in encoded
    assert '"role": "tool"' not in encoded
    assert "shell" not in encoded
    assert "file_list" not in encoded
    assert "query_help" not in encoded
    assert "file_write" in encoded


@pytest.mark.asyncio
async def test_openai_compatible_call_sanitizes_target_tool_calls_for_v4_pro_schema_only(monkeypatch):
    import nanoma.llm as llm_mod

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.bodies = []

        async def post(self, *args, **kwargs):
            self.bodies.append(kwargs["json"])
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "spawn_many",
                                            "arguments": json.dumps({"agents": [{"task": "child"}]}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                },
                request=request,
            )

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)
    messages = [
        {"role": "system", "content": "spawn"},
        {
            "role": "assistant",
            "content": "I will spawn.",
            "tool_calls": [
                {
                    "id": "old_spawn",
                    "type": "function",
                    "function": {
                        "name": "spawn_many",
                        "arguments": json.dumps({"agents": [{"task": "old child"}]}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old_spawn", "content": '{"created": 0}'},
    ]

    await openai_compatible_call(
        messages,
        "deepseek-v4-pro",
        tools=[META_TOOLS["spawn_many"]["schema"]],
        tool_choice={"type": "function", "function": {"name": "spawn_many"}},
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
    )

    encoded = json.dumps(fake.bodies[0]["messages"])
    assert '"tool_calls"' not in encoded
    assert '"role": "tool"' not in encoded
    assert "Prior tool calls omitted" in encoded


@pytest.mark.asyncio
async def test_openai_compatible_call_parses_tool_arguments_with_raw_newlines(monkeypatch):
    import nanoma.llm as llm_mod

    class FakeClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "spawn_many",
                                            "arguments": '{"agents":[{"task":"line 1\nline 2","role":"worker"}]}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                },
                request=request,
            )

    monkeypatch.setattr(llm_mod, "_shared_client", FakeClient())

    response = await openai_compatible_call(
        [{"role": "user", "content": "spawn"}],
        "deepseek-v4-pro",
        tools=[META_TOOLS["spawn_many"]["schema"]],
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
    )

    assert response.tool_calls[0].arguments == {"agents": [{"task": "line 1\nline 2", "role": "worker"}]}


@pytest.mark.asyncio
async def test_openai_compatible_call_repairs_v4_pro_schema_only_named_tool_miss(monkeypatch):
    import nanoma.llm as llm_mod
    from nanoma.tools import WORK_TOOLS

    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_bad_1",
                                "type": "function",
                                "function": {"name": "shell", "arguments": json.dumps({"command": "ls"})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_bad_2",
                                "type": "function",
                                "function": {"name": "file_list", "arguments": json.dumps({"path": "shared"})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "file_write",
                                    "arguments": json.dumps({"path": "shared/out.txt", "content": "ok"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        },
    ]

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.calls = 0
            self.bodies = []

        async def post(self, *args, **kwargs):
            self.bodies.append(kwargs["json"])
            data = responses[self.calls]
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(200, json=data, request=request)

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)

    response = await openai_compatible_call(
        [{"role": "user", "content": "write"}],
        "deepseek-v4-pro",
        tools=[WORK_TOOLS["file_write"]["schema"]],
        tool_choice={"type": "function", "function": {"name": "file_write"}},
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=2, base_delay=0, max_delay=0),
    )

    assert fake.calls == 3
    assert fake.bodies[0]["tool_choice"] == "auto"
    assert fake.bodies[1]["tool_choice"] == "auto"
    assert fake.bodies[2]["tool_choice"] == "auto"
    assert "Schema-only tool retry" in fake.bodies[1]["messages"][-1]["content"]
    assert "Retry #2" in fake.bodies[2]["messages"][-1]["content"]
    assert response.tool_calls[0].name == "file_write"


@pytest.mark.asyncio
async def test_openai_compatible_call_keeps_named_tool_choice_for_non_pro(monkeypatch):
    import nanoma.llm as llm_mod
    from nanoma.tools import WORK_TOOLS

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.bodies = []

        async def post(self, *args, **kwargs):
            self.bodies.append(kwargs["json"])
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                },
                request=request,
            )

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)
    named_choice = {"type": "function", "function": {"name": "file_write"}}

    await openai_compatible_call(
        [{"role": "user", "content": "write"}],
        "deepseek-v4-flash",
        tools=[WORK_TOOLS["file_write"]["schema"]],
        tool_choice=named_choice,
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0, base_delay=0, max_delay=0),
    )

    assert fake.bodies[0]["tool_choice"] == named_choice


@pytest.mark.asyncio
async def test_openai_compatible_call_retries_pro_tool_transport_miss(monkeypatch):
    import nanoma.llm as llm_mod

    responses = [
        {
            "choices": [{"message": {"role": "assistant", "content": "I will spawn the agents now."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "spawn_many",
                                    "arguments": json.dumps({"agents": [{"task": "candidate 00"}]}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    ]

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.calls = 0

        async def post(self, *args, **kwargs):
            data = responses[self.calls]
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(200, json=data, request=request)

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)

    response = await openai_compatible_call(
        [{"role": "user", "content": "spawn"}],
        "deepseek-v4-pro",
        tools=[META_TOOLS["spawn_many"]["schema"]],
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=1, base_delay=0, max_delay=0),
    )

    assert fake.calls == 2
    assert response.tool_calls[0].name == "spawn_many"
    assert response.tool_calls[0].arguments == {"agents": [{"task": "candidate 00"}]}


@pytest.mark.asyncio
async def test_openai_compatible_call_repairs_pro_empty_tool_retry(monkeypatch):
    import nanoma.llm as llm_mod

    responses = [
        {
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "spawn_many",
                                    "arguments": json.dumps({"agents": [{"task": "candidate 00"}]}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    ]

    class FakeClient:
        is_closed = False

        def __init__(self):
            self.calls = 0
            self.bodies = []

        async def post(self, *args, **kwargs):
            self.bodies.append(kwargs["json"])
            data = responses[self.calls]
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(200, json=data, request=request)

    fake = FakeClient()
    monkeypatch.setattr(llm_mod, "_shared_client", fake)

    response = await openai_compatible_call(
        [{"role": "user", "content": "spawn"}],
        "deepseek-v4-pro",
        tools=[META_TOOLS["spawn_many"]["schema"]],
        base_url="https://example.test/v1",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=1, base_delay=0, max_delay=0),
    )

    assert fake.calls == 2
    assert len(fake.bodies[0]["messages"]) == 1
    assert len(fake.bodies[1]["messages"]) == 2
    assert "valid tool_call" in fake.bodies[1]["messages"][-1]["content"]
    assert response.tool_calls[0].name == "spawn_many"


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
    result = await rt.run("Spawn peer agents for parent coordination.")
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

    result = await rt.run("Spawn peer agents for delegated output coordination.")

    assert result.startswith("[Delegated to ")
    assert (tmp_workspace / "shared" / "child.txt").read_text() == "child done"
    assert [e for e in rt._events if e["event"] == "join_remaining_agents"]


@pytest.mark.asyncio
async def test_terminal_answer_submission_prunes_leftover_children_after_grace(tmp_workspace):
    calls = {"child": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        calls["child"] += 1
        await asyncio.sleep(0.2)
        return LLMResponse(
            content="Still trying to spawn another verifier.",
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        log_dir=None,
        sandbox_backend="host",
        orchestration_preference="aggressive",
        spawn_before_turn=1,
        min_spawnable_workstreams=2,
        loop_action_policy="constraint",
        max_turns=10,
        terminal_submission_join_grace=0.05,
    )
    rt = Runtime(config=config, llm_call=mock_llm)
    root = rt.create_agent(
        'Solve and submit `shared/answer.json` with {"answer": "<answer-only string>"}.',
        model="test",
    )
    child = rt.create_agent("leftover child with no required output.", model="test", parent=root.id, role="worker")
    root.children.add(child.id)
    rt.start_agent(child)

    await meta_submit_answer({"answer": "Egalitarian"}, root, rt)
    await rt._wait_for_remaining_agents(root)

    assert root.result == "Egalitarian"
    assert child.status == "done"
    assert "Pruned after root terminal submission" in (child.result or "")
    assert (tmp_workspace / "shared" / "answer.json").exists()
    assert [e for e in rt._events if e["event"] == "terminal_prune_request"]
    assert [e for e in rt._events if e["event"] == "terminal_join_grace_finished"]
    assert [e for e in rt._events if e["event"] == "terminal_leftover_agent_pruned"]


@pytest.mark.asyncio
async def test_root_submit_answer_immediately_notifies_descendants_to_prune(tmp_workspace):
    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        log_dir=None,
        sandbox_backend="host",
        terminal_submission_join_grace=0.05,
    )
    rt = Runtime(config=config)
    root = rt.create_agent(
        'Solve and submit `shared/answer.json` with {"answer": "<answer-only string>"}.',
        model="test",
    )
    child = rt.create_agent("continue searching lane A", model="test", parent=root.id, role="worker")
    grandchild = rt.create_agent("continue subsearch lane A1", model="test", parent=child.id, role="worker")
    root.children.add(child.id)
    child.children.add(grandchild.id)
    child.status = "running"
    grandchild.status = "idle"

    result = await meta_submit_answer({"answer": "egalitarian"}, root, rt)

    assert result["submitted_answer"] == "shared/answer.json"
    assert root._prune_requests_sent_targets == {child.id, grandchild.id}
    assert child._steer_inbox.qsize() == 1
    assert grandchild._steer_inbox.qsize() == 1
    assert [
        e for e in rt._events
        if e["event"] == "terminal_prune_request" and e["data"].get("to") in {child.id, grandchild.id}
    ]


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
