"""Online feedback contracts and legacy delivery compatibility; no model calls."""
import ast
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest

from nanoma import Runtime, RuntimeConfig
from nanoma.online_objective import contract, from_result, parse_feedback, rank, runtime_time_limit, summarize


def policy(direction="maximize", selection="valid_then_score", **extra):
    return dict(visibility="agent_submission", metric_id="score",
                direction=direction, selection=selection, **extra)


def row(score, state, spec, **extra):
    return dict(source="official", metric=score, state=state, counts=True,
                valid=True, pass_rate=1.0,
                **{key: spec[key] for key in ("metric_id", "direction", "selection")}, **extra)


def feedback(score, round_id="agent-1", valid="yes"):
    return f"\n  {round_id} Results\nValid: {valid}\nPass rate: 100%\nScore: {score}\nAll tests passed!\n"


def runtime(tmp_path, monkeypatch, spec=None):
    target = tmp_path / "task"
    target.mkdir(exist_ok=True)
    (target / "main.py").write_text("original")
    monkeypatch.setenv("NANOMA_MERGE_SUBMIT_PATH", "1")
    rt = Runtime(config=RuntimeConfig(workspace_root=tmp_path / "work",
                                     workspace_extra_roots=[target], online_objective=spec, log_dir=None))
    agent = rt.create_agent("improve the output")
    agent.status = "running"
    return rt, agent, target


def test_contract_requires_explicit_online_visibility_and_direction():
    assert contract(None) is None
    for kwargs in ({"visibility": "host_only"}, {"direction": None}, {"metric_id": ""}):
        with pytest.raises(ValueError):
            contract({**policy(), **kwargs})


@pytest.mark.parametrize("direction,expected", [("maximize", 20), ("minimize", 10)])
def test_objective_ranking_keeps_metric_channels_separate(direction, expected):
    spec = policy(direction)
    entries = [row(10, "a", spec), row(20, "b", spec)]
    assert max(entries, key=lambda e: rank(e, spec))["metric"] == expected
    assert rank({**entries[0], "source": "verify", "metric": 999}, spec) is None
    assert rank({**entries[0], "metric_id": "latency"}, spec) is None
    assert rank({**entries[0], "valid": False}, spec) is None
    assert rank({**entries[0], "metric": float("nan")}, spec) is None


def test_partial_pass_ties_match_public_selection_contract():
    spec = policy(selection="pass_rate_first")
    a = {**row(1, "a", spec), "pass_rate": .5}
    b = {**row(999, "b", spec), "pass_rate": .5}
    assert rank(a, spec) == rank(b, spec)
    assert rank(row(0, "c", spec), spec) > rank(b, spec)


def test_parse_small_scores_and_invalid_feedback_without_optimality_claim():
    result = parse_feedback(feedback("2.7e-7", valid="   no"))
    assert result["score"] == 2.7e-7 and not result["valid"]
    assert result["pass_rate"] == 1
    assert parse_feedback("All tests passed!") is None
    assert parse_feedback(feedback("nan")) is None
    assert parse_feedback(feedback("inf")) is None
    text = summarize([row(11.04, "a", policy())], policy())
    assert "not an optimality claim" in text
    assert "satisfied=True" not in text


def test_nested_submission_and_truncated_output_keep_the_measurement():
    observed = parse_feedback(feedback("11.04"))
    result = {"submission": {"stdout": "tail without score", "objective_observation": observed},
              "attempts": [{"stdout": feedback("999")}]}
    assert from_result(result)["score"] == 11.04
    assert from_result({"submission": {"stdout": feedback("12")}})["score"] == 12


def test_plateau_counts_new_artifacts_and_resets_on_improvement():
    spec = policy(plateau_window=3)
    entries = [row(10, "a", spec)] + [row(10, "a", spec) for _ in range(10)]
    assert "Reconsider" not in summarize(entries, spec)
    entries += [row(9, x, spec) for x in ("b", "c", "d")]
    assert "3 distinct" in summarize(entries, spec)
    entries.append(row(11, "e", spec))
    assert "Reconsider" not in summarize(entries, spec)
    assert "Reconsider" not in summarize(entries[:-1], {**spec, "plateau_window": 0})


@pytest.mark.asyncio
async def test_submit_feedback_is_deduplicated_and_best_bytes_restore(tmp_path, monkeypatch):
    spec = policy("minimize")
    rt, agent, target = runtime(tmp_path, monkeypatch, spec)
    rt._note_state_being_submitted(agent, "submit", {})
    result = {"stdout": feedback(12)}
    await rt._calibrate_from_tool_result(agent, "submit", {}, result)
    assert result["objective_feedback"]["direction"] == "minimize"
    assert len(rt._ledger_entries()) == 1
    rt._note_state_being_submitted(agent, "submit", {})
    await rt._calibrate_from_tool_result(agent, "submit", {}, result)
    assert len(rt._ledger_entries()) == 1
    assert agent._steer_inbox.qsize() == 1
    (target / "main.py").write_text("worse")
    rt._note_state_being_submitted(agent, "submit", {})
    await rt._calibrate_from_tool_result(agent, "submit", {}, {"stdout": feedback(20, "agent-2")})
    rt._merge_restore_best()
    assert (target / "main.py").read_text() == "original"
    assert rt._ledger_best()["metric"] == 12
    # New iteration reuses its own permitted submission observations.
    other = Runtime(config=rt.config)
    assert "best=12" in other._online_objective_summary()


@pytest.mark.asyncio
async def test_changed_workspace_is_not_snapshotted_as_scored_bytes(tmp_path, monkeypatch):
    rt, agent, target = runtime(tmp_path, monkeypatch, policy())
    rt._note_state_being_submitted(agent, "submit", {})
    digest = rt._ledger_digest(rt._submitted_signature[agent.id])
    (target / "main.py").write_text("changed while judge was running")
    await rt._calibrate_from_tool_result(agent, "submit", {}, {"stdout": feedback(30)})
    assert not rt._ledger_snapshot_dir(digest).exists()
    assert any(e["event"] == "ledger_snapshot_skipped" for e in rt._events)


@pytest.mark.asyncio
async def test_low_objective_does_not_reject_feasible_check(tmp_path, monkeypatch):
    rt, agent, target = runtime(tmp_path, monkeypatch, policy())
    rt._verify_calibration = {"local": {"agreed": 2, "contradicted": 0}}
    rt._merge_best_rank = (1, 1)
    rt._note_state_being_submitted(agent, "submit", {})
    await rt._calibrate_from_tool_result(agent, "submit", {}, {"stdout": feedback(0)})
    assert rt._verify_calibration == {"local": {"agreed": 2, "contradicted": 0}}
    assert rt._merge_best_rank == (1, 1)


@pytest.mark.asyncio
async def test_explicit_invalid_verdict_still_contradicts_local_check(tmp_path, monkeypatch):
    rt, agent, target = runtime(tmp_path, monkeypatch, policy())
    agent._verify_spec = {"command": "local-check", "name": "check", "agent": agent.id}
    signature = rt._merge_scope_signature(target)
    rt._merge_scored_signature = signature
    rt._merge_best_signature = signature
    rt._merge_best_rank = (1, 1)
    rt._note_state_being_submitted(agent, "submit", {})
    await rt._calibrate_from_tool_result(agent, "submit", {}, {"stdout": feedback(20, valid="no")})
    assert rt._verify_calibration["local-check"]["contradicted"] == 1
    assert rt._merge_best_rank is None
    assert rt._ledger_best() is None


def test_opposite_local_and_official_directions_do_not_recurse(tmp_path, monkeypatch):
    spec = policy("minimize")
    rt, agent, target = runtime(tmp_path, monkeypatch, spec)
    check = {"command": "quality", "higher_is_better": True, "at": 1}
    agent._verify_spec = check
    rt._verify_metric_seen["quality"] = [1, 2]
    entries = [row(20, "a", spec), row(10, "b", spec),
               {"source": "verify", "metric": 1, "state": "a", "check": "quality"},
               {"source": "verify", "metric": 2, "state": "b", "check": "quality"}]
    monkeypatch.setattr(rt, "_ledger_entries", lambda: entries)
    assert rt._verify_metric_tracks_authority(check) is True
    assert rt._authoritative_spec() == check


def test_noise_does_not_mix_replaced_local_checks(tmp_path, monkeypatch):
    rt, agent, _ = runtime(tmp_path, monkeypatch, policy())
    agent._verify_spec = {"command": "accuracy", "higher_is_better": True}
    entries = [
        {"source": "verify", "state": "a", "metric": 1, "check": "accuracy", "direction": "maximize"},
        {"source": "verify", "state": "a", "metric": 1, "check": "accuracy", "direction": "maximize"},
        {"source": "verify", "state": "a", "metric": 9000, "check": "throughput", "direction": "maximize"},
    ]
    monkeypatch.setattr(rt, "_ledger_entries", lambda: entries)
    assert rt._ledger_state_metrics("a", "verify") == [1, 1]
    assert rt._measurement_spread("verify") == (0, 0)


@pytest.mark.asyncio
async def test_legacy_runtime_has_no_online_messages_or_extra_tool_fields(tmp_path, monkeypatch):
    rt, agent, _ = runtime(tmp_path, monkeypatch)
    rt._note_state_being_submitted(agent, "submit", {})
    result = {"stdout": feedback(1)}
    await rt._calibrate_from_tool_result(agent, "submit", {}, result)
    assert "objective_feedback" not in result
    assert agent._steer_inbox.empty()
    assert rt._online_objective_summary() == ""
    assert not any(e["event"] == "online_objective_feedback" for e in rt._events)


@pytest.mark.asyncio
async def test_list_and_unbound_feedback_cannot_be_used_as_live_best(tmp_path, monkeypatch):
    rt, agent, _ = runtime(tmp_path, monkeypatch, policy())
    await rt._calibrate_from_tool_result(agent, "shell", {"command": "sforge-submit --list"},
                                         {"stdout": feedback(999)})
    assert rt._ledger_entries() == []
    rt._note_state_being_submitted(agent, "shell", {"command": "sforge-submit -l"})
    await rt._calibrate_from_tool_result(agent, "shell", {"command": "sforge-submit -l"},
                                         {"stdout": feedback(999)})
    assert rt._ledger_entries() == []
    await rt._calibrate_from_tool_result(agent, "submit", {}, {"stdout": feedback(999)})
    assert rt._ledger_best() is None


def test_inner_time_budget_respects_remaining_time_and_resume_elapsed():
    assert runtime_time_limit(0, 8200, 7200, 90, now=1000) == 7020
    assert runtime_time_limit(6840, 8200, 7200, 90, now=7500) == 520
    assert runtime_time_limit(100, 8200, 7200, 90, now=7500, elapsed=600) == 700
    assert runtime_time_limit(42, 0, 0, 90, now=1000) == 42


@pytest.mark.asyncio
async def test_runner_wires_policy_and_deadline_into_runtime(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "benchmarks/edgebench/run_nanoma_edgebench.py"
    spec = importlib.util.spec_from_file_location("tested_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}
    class FakeRuntime:
        def __init__(self, config):
            captured["config"] = config
        async def run(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return "complete"
        async def continue_from_probe(self, probe):
            return "complete"
        def stats(self):
            return {"overview": {"llm_calls": 1}, "agents": {}}
        def _online_objective_summary(self):
            return "saved progress"
    async def empty(*args, **kwargs):
        return {}
    monkeypatch.setattr(module, "Runtime", FakeRuntime)
    monkeypatch.setattr(module, "_prepare_seed_archive", empty)
    monkeypatch.setattr(module, "_final_sforge_submit", empty)
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    monkeypatch.setenv("SFORGE_SCORE_DIRECTION", "minimize")
    monkeypatch.setenv("SFORGE_SELECTION_POLICY", "valid_then_score")
    monkeypatch.setenv("NANOMA_EDGE_FINAL_SAFETY_SECONDS", "90")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A public task")
    args = module._build_parser().parse_args([
        "--prompt-file", str(prompt), "--task-cwd", str(tmp_path),
        "--workspace", str(tmp_path / "work"), "--wall-seconds", "7200",
        "--deadline-epoch", "8200", "--time-limit", "0",
    ])
    await module._run_single_iteration(args)
    assert captured["config"].time_limit == 7020
    assert captured["config"].online_objective["direction"] == "minimize"
    assert captured["config"].max_agents == 1_000_000
    assert captured["config"].max_parallel_children == 0
    assert "saved progress" in captured["prompt"]
    probe = tmp_path / "state.pkl"
    probe.write_bytes(pickle.dumps({"runtime_resume_state": {"effective_elapsed_seconds": 200}}))
    args.resume_probe = str(probe)
    await module._run_single_iteration(args)
    assert captured["config"].time_limit == 7220


def test_embedded_fallback_runner_has_same_time_and_feedback_wiring():
    adapter = Path(__file__).parents[1] / "edgebench_adapter.py"
    if not adapter.exists():
        pytest.skip("external adapter is checked in the staging bundle")
    module = ast.parse(adapter.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_runner_source")
    ns = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fallback", "exec"), ns)
    code = ns["_runner_source"]()
    compile(code, "fallback_runner", "exec")
    assert "online_objective=objective" in code
    assert "time_limit=effective_time_limit" in code


@pytest.mark.asyncio
async def test_submission_preserves_score_before_truncation_and_cleans_up_timeout(tmp_path, monkeypatch):
    import os
    path = Path(__file__).parents[1] / "benchmarks/edgebench/run_nanoma_edgebench.py"
    spec = importlib.util.spec_from_file_location("submit_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = tmp_path / "sforge-submit"
    script.write_text(f"#!{sys.executable}\nprint({feedback('0.3142')!r})\nprint('x' * 8000)\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = await module._run_sforge_submit_once(tmp_path, timeout=3)
    assert result["exit_code"] == 0
    assert "Results" not in result["stdout"]
    assert result["objective_observation"]["score"] == .3142
    script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    captured = []
    create = module.asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        proc = await create(*args, **kwargs)
        captured.append(proc)
        return proc
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", capture)
    result = await module._run_sforge_submit_once(tmp_path, timeout=.1)
    assert result["exit_code"] == -1
    assert captured[0].returncode is not None
