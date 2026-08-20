"""Calibrating the local check against the judge's own verdict.

The judge is an agent-facing interface: sforge-submit prints back the score, the
pass rate and the names of what failed, and --list replays the agent's own
history for free. The host-side auto-evals are a separate admin view and are
deliberately hidden from the agent's token, so nothing here reads them.

What these cover is the failure this was written for: a registered check passed a
state the judge scored zero — it measured throughput while the task also required
a recall floor — and nothing in the run ever learned about the gap.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

SIZE_CHECK = (
    "python3 -c \"import json,glob;"
    "t=sum(len(open(f).read()) for f in sorted(glob.glob('*.txt')));"
    "print(json.dumps({'ok': True, 'metric': t}))\""
)

FAILING_VERDICT = """
========================================
  round-7 Results
========================================
  Valid:       no
  Score:       4419
  Pass rate:   61.9%
  Passed:      13/21

  Summary:
    recall below the required floor on 8 query sets

  Metrics:
    qps: 4419

  Failed checks:
    - recall@10 >= 0.95 on sift-128-euclidean
    - recall@10 >= 0.95 on glove-100-angular
========================================
"""

PASSING_VERDICT = """
========================================
  round-8 Results
========================================
  Score:       9100
  Pass rate:   100.0%
  Passed:      21/21

  All tests passed!
========================================
"""

COOLDOWN = "COOLDOWN: another submission is allowed in 42s\n"


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
    emitted: list = []
    inner = rt._emit
    rt._emit = lambda who, kind, data=None: (
        emitted.append((who, kind, data)), inner(who, kind, data))[-1]
    rt._emitted = emitted
    return rt, submit_path


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _drain(agent) -> str:
    out = []
    for inbox in (agent._steer_inbox, agent._queue_inbox, agent._immediate_inbox):
        while not inbox.empty():
            out.append(getattr(inbox.get_nowait(), "content", ""))
    return "\n".join(out)


def _verify(rt, agent, command=SIZE_CHECK, **kw):
    from optimizations.verify_tool import meta_verify
    return _run(meta_verify({"command": command, **kw}, agent, rt))


def _kinds(rt) -> list[str]:
    return [k for _, k, _ in rt._emitted]


def _enable():
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"


def _disable():
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)


def _passing_state(rt, submit_path):
    """A child whose work is merged and passed by its own registered check."""
    parent = rt.create_agent(task="root", parent=None, depth=0)
    child = rt.create_agent(task="c", parent=parent.id, depth=1)
    (Path(child._merge_copy) / "sol.txt").write_text("a-solution\n")
    assert _verify(rt, child)["ok"]
    assert _run(rt._merge_promote_child(child))["kept"]
    return parent, child


# ── reading the verdict ──────────────────────────────────────────────────────

def test_the_printed_verdict_is_read_back():
    from nanoma.core import Runtime

    v = Runtime._official_parse(FAILING_VERDICT)
    assert v["valid"] is False and v["score"] == 4419.0, v
    assert abs(v["pass_rate"] - 0.619) < 0.001, v
    assert v["round"] == "round-7", v
    assert any("recall@10" in f for f in v["failed"]), v

    ok = Runtime._official_parse(PASSING_VERDICT)
    assert ok["valid"] and ok["pass_rate"] == 1.0 and ok["score"] == 9100.0, ok

    assert Runtime._official_parse(COOLDOWN) is None, "a refusal is not a verdict"
    assert Runtime._official_parse("") is None
    print("ok: score, pass rate and the named failures are read out of the verdict")


# ── the disagreement ─────────────────────────────────────────────────────────

def test_a_check_that_passed_a_rejected_state_is_told_so():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        _drain(child)

        _run(rt._calibrate_with_official(child, FAILING_VERDICT))

        note = _drain(child)
        assert note, "the agent that submitted hears about it"
        assert "61%" in note or "62%" in note, note
        assert "recall@10" in note, "including what actually failed"
        assert "official_disagrees" in _kinds(rt)
    _disable()
    print("ok: a check that passed a state the judge rejects is told, with the failures")


def test_the_author_of_the_check_hears_it_too():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        other = rt.create_agent(task="o", parent=parent.id, depth=1)
        _drain(child)

        _run(rt._calibrate_with_official(other, FAILING_VERDICT))

        assert child.id in _drain(child), "the one whose check is gating merges"
        assert "Failing" in _drain(other), "and the one who submitted"
    _disable()
    print("ok: the author of the gating check hears it, not only the submitter")


def test_a_state_the_judge_rejects_stops_being_the_fallback():
    """It was recorded as the best state on the local metric alone. Rewinding to
    it at the end of the run would be rewinding to a zero."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        assert rt._merge_best_rank is not None, "precondition: it is the best state"

        _run(rt._calibrate_with_official(child, FAILING_VERDICT))

        assert rt._merge_best_rank is None and rt._merge_best_metric is None
        assert "merge_best_dropped" in _kinds(rt)
    _disable()
    print("ok: the judge's rejection drops that state as somewhere to rewind to")


def test_a_contradicted_check_stops_nominating_best_states():
    """Otherwise the next merge quietly re-records the state the judge threw out."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        _run(rt._calibrate_with_official(child, FAILING_VERDICT))
        assert rt._merge_best_rank is None

        second = rt.create_agent(task="c2", parent=parent.id, depth=1)
        (Path(second._merge_copy) / "sol.txt").write_text("another-try\n")
        kept = _run(rt._merge_promote_child(second))

        assert kept["kept"], "the check still gates merges — it is the only instrument"
        assert rt._merge_best_rank is None, "but it no longer nominates a state to keep"
        assert "merge_best_withheld" in _kinds(rt)
    _disable()
    print("ok: a contradicted check gates merges but stops naming the best state")


def test_agreement_is_recorded_and_says_nothing()  :
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        _drain(child)

        _run(rt._calibrate_with_official(child, PASSING_VERDICT))

        assert not _drain(child), "nothing to say when the two agree"
        spec = rt._authoritative_spec()
        assert rt._verify_calibration[spec["command"]]["agreed"] == 1
        assert rt._merge_best_rank is not None, "and the best state stands"
    _disable()
    print("ok: agreement is recorded quietly and the best state stands")


def test_a_verdict_on_someone_elses_state_is_not_evidence():
    """The judge scored a state, but the local check never passed *that* state —
    there is no disagreement to report."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        _drain(child)
        # the submission moved on after the check last approved it
        (submit_path / "sol.txt").write_text("edited-after-the-check\n")

        _run(rt._calibrate_with_official(child, FAILING_VERDICT))

        assert not _drain(child), "nothing is pinned on a check that never saw this"
        assert "official_disagrees" not in _kinds(rt)
        assert rt._merge_best_rank is not None, "and nothing is dropped on a guess"
    _disable()
    print("ok: a verdict on a state the check never passed is not held against it")


# ── how the verdict arrives ──────────────────────────────────────────────────

def test_it_is_caught_however_the_agent_submits():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        _drain(child)

        # through the submit tool
        _run(rt._calibrate_from_tool_result(
            child, "submit", {}, {"stdout": FAILING_VERDICT, "exit_code": 0}))
        assert "recall@10" in _drain(child), "the submit tool's output is read"

        # and through a plain shell call, which the prompt also offers
        rt._merge_best_rank = ("re-armed",)
        rt._merge_best_signature = rt._merge_scored_signature
        _run(rt._calibrate_from_tool_result(
            child, "shell", {"command": "cd /task && sforge-submit"},
            {"stdout": FAILING_VERDICT, "exit_code": 0}))
        assert "recall@10" in _drain(child), "so is a shell submission"
    _disable()
    print("ok: the verdict is picked up whether the agent used the tool or the shell")


def test_a_fragment_is_not_a_verdict():
    """A pass rate with no tally and no score once parsed as "invalid, 100% passed",
    which would blame a check for a rejection nobody made."""
    from nanoma.core import Runtime
    assert Runtime._official_parse(
        "  Results\n  Pass rate:   100.0%\n") is None
    assert Runtime._official_parse(FAILING_VERDICT) is not None
    assert Runtime._official_parse(PASSING_VERDICT) is not None
    print("ok: a fragment that looks like a verdict is not treated as one")


def test_reading_the_history_is_not_a_verdict():
    """--list is free and replays old rounds; treating it as a fresh verdict would
    pin an old failure on a check that never saw that state."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        _drain(child)

        _run(rt._calibrate_from_tool_result(
            child, "shell", {"command": "sforge-submit --list"},
            {"stdout": FAILING_VERDICT, "exit_code": 0}))

        assert not _drain(child) and "official_disagrees" not in _kinds(rt)
    _disable()
    print("ok: replaying the history is not treated as a verdict on the state at hand")


def test_calibration_carries_into_the_next_iteration():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        parent, child = _passing_state(rt, submit_path)
        command = rt._authoritative_spec()["command"]
        _run(rt._calibrate_with_official(child, FAILING_VERDICT))
        assert rt._verify_calibration[command]["contradicted"] == 1

        later, _ = _mk_runtime(Path(d))          # a fresh process, same task dir
        later._verify_load_state()
        assert later._verify_calibration.get(command, {}).get("contradicted") == 1, \
            "what the judge said outlives the iteration that heard it"
    _disable()
    print("ok: a contradicted check is still known to be contradicted next iteration")


def test_nothing_happens_when_merge_is_off():
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        assert _run(rt._calibrate_with_official(agent, FAILING_VERDICT)) is None
    print("ok: no calibration outside a merge run")


def main():
    tests = [
        test_the_printed_verdict_is_read_back,
        test_a_check_that_passed_a_rejected_state_is_told_so,
        test_the_author_of_the_check_hears_it_too,
        test_a_state_the_judge_rejects_stops_being_the_fallback,
        test_a_contradicted_check_stops_nominating_best_states,
        test_agreement_is_recorded_and_says_nothing,
        test_a_verdict_on_someone_elses_state_is_not_evidence,
        test_it_is_caught_however_the_agent_submits,
        test_a_fragment_is_not_a_verdict,
        test_reading_the_history_is_not_a_verdict,
        test_calibration_carries_into_the_next_iteration,
        test_nothing_happens_when_merge_is_off,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} official-calibration tests passed.")


if __name__ == "__main__":
    main()
