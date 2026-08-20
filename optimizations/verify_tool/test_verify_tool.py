"""Tests for the verify tool and the nudge that keeps it from going unused.

Run:  python3 -m optimizations.verify_tool.test_verify_tool
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from optimizations.verify_tool import meta_verify

OK_CHECK = "python3 -c \"import json;print('building...');print(json.dumps({'ok': True, 'metric': 42.5}))\""


def _mk_runtime(tmp: Path):
    from nanoma.core import Runtime, RuntimeConfig

    submit_path = tmp / "task_cwd"
    workspace_root = submit_path / ".nanoma-task-work"
    submit_path.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        max_agents=16, max_depth=3, default_model="m", allowed_models=["m"],
        workspace_root=workspace_root, workspace_extra_roots=[submit_path],
    )
    rt = Runtime(config=config)
    rt.start_agent = lambda child: None
    return rt, submit_path


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_verdict_parsing():
    from nanoma.core import Runtime
    parse = Runtime._parse_verification_output
    assert parse('noise\n{"ok": true, "metric": 3760.0}') == (True, 3760.0)
    assert parse('{"ok": false, "metric": 12}') == (False, 12.0)
    assert parse("VERIFY: ok=true metric=9.5") == (True, 9.5)
    assert parse("VERIFY: ok=no metric=1") == (False, 1.0)
    assert parse('{"metric": 5}') == (None, 5.0)
    assert parse("all tests passed!") == (None, None), "prose is not a verdict"
    assert parse("") == (None, None)
    assert parse('{"ok": true}') == (True, None)
    # a trailing summary must not hide the verdict printed just above it
    assert parse('{"ok": true, "metric": 7}\nDone in 3s\n') == (True, 7.0)
    print("ok: verdicts parse, prose does not pass for one")


def test_metric_orientation():
    from nanoma.core import Runtime
    rank = Runtime._verify_rank
    assert rank(True, 10.0) > rank(True, 5.0), "higher is better by default"
    assert rank(True, 10.0, False) < rank(True, 5.0, False), "lower is better when asked"
    assert rank(False, 999.0) < rank(True, 0.0), "a failing check always loses"
    assert rank(None, 5.0) is None, "no verdict, no rank"
    print("ok: ranking respects orientation and never lets a failure win on metric")


def test_command_is_stored_portably():
    with tempfile.TemporaryDirectory() as d:
        os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
        rt, submit_path = _mk_runtime(Path(d))
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        copy = Path(child._merge_copy)
        (copy / "t.txt").write_text("x")
        out = _run(meta_verify(
            {"command": f"cd {copy} && {OK_CHECK}"}, child, rt,
        ))
        assert out["ok"] and out["metric"] == 42.5, out
        stored = child._verify_spec["command"]
        assert str(copy) not in stored, "the private copy path was abstracted away"
        assert rt.VERIFY_WORKDIR_TOKEN in stored, stored
        # and it still runs when pointed somewhere else
        result = _run(rt._run_verification(stored, submit_path, 60))
        assert result["ok"] and result["metric"] == 42.5
        os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: a registered check is re-runnable against a different directory")


def test_timeout_and_crash_are_failures():
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        slow = _run(meta_verify({"command": "sleep 5", "timeout": 1}, agent, rt))
        assert slow["ok"] is False and slow["timed_out"] is True
        crash = _run(meta_verify({"command": "exit 3"}, agent, rt))
        assert crash["ok"] is False and crash["exit_code"] == 3
        assert getattr(agent, "_verify_spec", None) is None, "neither was registered"
        missing = _run(meta_verify({"command": "  "}, agent, rt))
        assert "error" in missing
    print("ok: timeouts, crashes and empty commands are failures, not registrations")


def test_exit_code_overrides_a_cheerful_verdict():
    with tempfile.TemporaryDirectory() as d:
        rt, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        out = _run(meta_verify(
            {"command": "python3 -c \"import json;print(json.dumps({'ok': True, 'metric': 1}))\"; exit 1"},
            agent, rt,
        ))
        assert out["ok"] is False, "a non-zero exit is not a pass, whatever it printed"
    print("ok: a non-zero exit outranks a self-declared pass")


def test_nudge_fires_once_and_stops_after_registration():
    with tempfile.TemporaryDirectory() as d:
        os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
        rt, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        agent._turns = 1
        assert rt._verification_nudge(agent) is None, "silent early on"
        agent._turns = rt._VERIFY_NUDGE_AFTER_TURNS
        first = rt._verification_nudge(agent)
        assert first and "verify tool" in first
        assert rt._verification_nudge(agent) is None, "fires only once"

        other = rt.create_agent(task="b", parent=None, depth=0)
        other._turns = 20
        _run(meta_verify({"command": OK_CHECK}, other, rt))
        assert rt._verification_nudge(other) is None, "nothing to nag about once registered"
        os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: the nudge fires once, and not at all once a check is registered")


def test_workspace_paths_survive_the_rewrite():
    """The failure that killed the first check an agent ever tried to register.

    Its command teed output into .nanoma-task-work/<agent>/, the task-directory
    prefix got rewritten into the private copy — which deliberately has no such
    directory — and the whole thing died in a second.
    """
    with tempfile.TemporaryDirectory() as d:
        os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
        rt, submit_path = _mk_runtime(Path(d))
        (submit_path / "run.sh").write_text(
            "#!/bin/sh\necho '{\"ok\": true, \"metric\": 7}'\n"
        )
        (submit_path / "run.sh").chmod(0o755)
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        log = submit_path / ".nanoma-task-work" / child.id / "bench.txt"
        command = f"cd {submit_path} && ./run.sh | tee {log}"
        portable = rt._verify_portable_command(command, child)
        assert f"cd {rt.VERIFY_WORKDIR_TOKEN} &&" in portable, "the task dir is rewritten"
        assert str(log) in portable, "the workspace path is left alone"
        log.parent.mkdir(parents=True, exist_ok=True)
        out = _run(meta_verify({"command": command}, child, rt))
        assert out["ok"] and out["metric"] == 7, out
        assert log.is_file(), "the tee target was writable"
        os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: workspace paths are not rewritten into the private copy")


def test_failed_attempt_earns_another_reminder():
    with tempfile.TemporaryDirectory() as d:
        os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
        rt, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        agent._turns = 10
        assert "not registered a check" in rt._verification_nudge(agent)
        assert rt._verification_nudge(agent) is None, "quiet while nothing has failed"

        _run(meta_verify({"command": "echo no-verdict-here"}, agent, rt))
        follow_up = rt._verification_nudge(agent)
        assert follow_up and "did not register" in follow_up, follow_up
        assert "printed no verdict line" in follow_up, "it names the actual problem"

        _run(meta_verify({"command": "exit 9"}, agent, rt))
        third = rt._verification_nudge(agent)
        assert third and "exited with code 9" in third, third
        _run(meta_verify({"command": "exit 9"}, agent, rt))
        assert rt._verification_nudge(agent) is None, "stops after three, not endless"
        os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: a failed attempt earns a follow-up that names the problem")


def test_check_and_floor_carry_into_the_next_iteration():
    """Each EdgeBench iteration is a fresh process against the same task dir."""
    with tempfile.TemporaryDirectory() as d:
        os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        (submit_path / "sol.txt").write_text("solution\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _run(meta_verify({"command": OK_CHECK, "name": "bench"}, agent, rt))
        assert rt._merge_best_rank is not None and rt._merge_best_metric == 42.5

        later, _ = _mk_runtime(tmp)          # next iteration, fresh runtime
        fresh = later.create_agent(task="root", parent=None, depth=0)
        inherited = later._verification_spec_for(fresh)
        assert inherited and inherited["name"] == "bench", "the proven check carried over"
        assert later._merge_best_metric == 42.5, "and so did the ratchet floor"

        # Registering must not clobber the floor before it has been read. This
        # is what actually broke live: the first registration of each iteration
        # overwrote the file, so nothing was ever inherited. A higher stored
        # floor makes inheriting distinguishable from re-measuring.
        later._merge_best_rank = (1, 900.0)
        later._merge_best_metric = 900.0
        later._verify_save_state()
        third, _ = _mk_runtime(tmp)
        agent3 = third.create_agent(task="root", parent=None, depth=0)
        _run(meta_verify({"command": OK_CHECK, "name": "bench"}, agent3, third))
        assert third._merge_best_metric == 900.0, "the inherited floor survived registration"

        # a different check measures a different thing, so the old floor drops
        newest, _ = _mk_runtime(tmp)
        other = newest.create_agent(task="root", parent=None, depth=0)
        newest._verify_load_state()
        assert newest._merge_best_metric == 900.0
        _run(meta_verify(
            {"command": "python3 -c \"import json;print(json.dumps({'ok': True, 'metric': 3}))\""},
            other, newest,
        ))
        assert newest._merge_best_metric == 3, "the floor is rebased, not inherited blindly"
        os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: a proven check and its ratchet floor survive into the next iteration")


def test_nudge_silent_when_merge_is_off():
    with tempfile.TemporaryDirectory() as d:
        os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
        rt, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        agent._turns = 50
        assert rt._verification_nudge(agent) is None
    print("ok: no nudge when merging is off")


def test_tool_is_registered():
    from nanoma.meta import META_TOOLS
    assert "verify" in META_TOOLS
    schema = META_TOOLS["verify"]["schema"]["function"]
    assert schema["parameters"]["required"] == ["command"]
    assert '{"ok": true, "metric"' in schema["description"], "the contract is in the description"
    print("ok: verify is registered with the verdict contract in its description")


def main():
    tests = [
        test_verdict_parsing,
        test_metric_orientation,
        test_command_is_stored_portably,
        test_timeout_and_crash_are_failures,
        test_exit_code_overrides_a_cheerful_verdict,
        test_nudge_fires_once_and_stops_after_registration,
        test_workspace_paths_survive_the_rewrite,
        test_failed_attempt_earns_another_reminder,
        test_check_and_floor_carry_into_the_next_iteration,
        test_nudge_silent_when_merge_is_off,
        test_tool_is_registered,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} verify-tool tests passed.")


if __name__ == "__main__":
    main()
