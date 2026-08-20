"""Tests for NanoMA's benchmark-independent API and CLI configuration."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

from nanoma.agent import prepare_general_config, run_agent
from nanoma.core import Runtime, RuntimeConfig
from nanoma.main import build_parser, config_from_args, resolve_task


def test_cli_accepts_argument_file_and_stdin(tmp_path):
    parser = build_parser()

    direct = parser.parse_args(["analyze this", "--quiet"])
    assert resolve_task(direct, stdin=io.StringIO("ignored")) == "analyze this"

    task_file = tmp_path / "task.md"
    task_file.write_text("task from file\n", encoding="utf-8")
    from_file = parser.parse_args(["--task-file", str(task_file), "--quiet"])
    assert resolve_task(from_file, stdin=io.StringIO("ignored")) == "task from file"

    piped = parser.parse_args(["-", "--quiet"])
    assert resolve_task(piped, stdin=io.StringIO("task from stdin\n")) == "task from stdin"


def test_generic_cli_enables_node_planning_without_benchmark_merge():
    args = build_parser().parse_args(["task", "--model", "worker-model", "--quiet"])
    config = config_from_args(args)

    assert config.default_model == "worker-model"
    assert config.node_autonomous_planning is True
    assert config.benchmark_merge_submit_enabled is False
    assert config.max_depth == 8
    assert config.max_concurrent_llm == 8


def test_generic_cli_can_disable_node_planning():
    args = build_parser().parse_args(["task", "--no-node-planning", "--quiet"])
    assert config_from_args(args).node_autonomous_planning is False


def test_general_runtime_exposes_only_general_coordination_tools():
    runtime = Runtime(RuntimeConfig(log_dir=None))
    tools = runtime._all_tools()

    assert all(name in tools for name in ("task_create", "task_update", "task_list"))
    assert all(name not in tools for name in ("spawn", "spawn_many", "task_spawn"))
    assert all(name not in tools for name in ("verify", "experiments", "deliveries"))


def test_prepare_general_config_keeps_project_separate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    base = RuntimeConfig(
        workspace_root=Path(".nanoma/workspace"),
        log_dir=Path(".nanoma/logs"),
        system_extra_instructions="Keep changes focused.",
    )

    prepared = prepare_general_config(
        base,
        project_dir=project,
        instructions="Run the relevant checks.",
    )

    assert prepared.workspace_root == (tmp_path / ".nanoma/workspace").resolve()
    assert prepared.log_dir == (tmp_path / ".nanoma/logs").resolve()
    assert project.resolve() in prepared.workspace_extra_roots
    assert str(project.resolve()) in prepared.system_extra_instructions
    assert "Keep changes focused." in prepared.system_extra_instructions
    assert "Run the relevant checks." in prepared.system_extra_instructions


def test_run_agent_is_a_general_wrapper(monkeypatch, tmp_path):
    seen = {}

    class FakeRuntime:
        def __init__(self, config, on_event=None):
            seen["config"] = config
            seen["on_event"] = on_event
            self.config = config

        async def run(self, task, model=None):
            seen["task"] = task
            seen["model"] = model
            return "finished"

        def stats(self):
            return {"agents": {"total_spawned": 1}, "overview": {"total_cost_usd": 0}}

    monkeypatch.setattr("nanoma.agent.Runtime", FakeRuntime)
    config = RuntimeConfig(
        default_model="worker-model",
        workspace_root=tmp_path / "work",
        log_dir=tmp_path / "logs",
    )

    run = asyncio.run(run_agent("do a normal task", config=config))

    assert run.result == "finished"
    assert seen["task"] == "do a normal task"
    assert seen["model"] == "worker-model"
    assert seen["config"].node_autonomous_planning is True
    assert run.workspace_root == (tmp_path / "work").resolve()
    assert run.as_dict()["result"] == "finished"
