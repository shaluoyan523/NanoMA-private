"""Unit tests for verification-gated merging (NANOMA_MERGE_SUBMIT_PATH).

The runtime does not consult the benchmark's own judge to decide what to keep.
Agents declare a check with the `verify` tool, and the runtime re-runs that check
itself — against the *merged* submission path, which is the artifact that
actually ships and the one nobody thinks to measure. On ann_vector_search_qps
every party verified only its own copy, a config merged from three children
replaced a working solution unchecked, and the run ended at 0 with a peak of
4509 on record.

The checks here are real commands whose verdict is computed from the files on
disk, so the merged-state re-run is genuinely exercised rather than stubbed.

Run:  python3 -m optimizations.merge_submit.test_merge_submit
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

# A check that reads the submission and reports a verdict: "ok" while the total
# size stays within budget, "metric" = the size. Two changes that each verify on
# their own can therefore break once merged — the ann_vector_search_qps shape.
BUDGET_CHECK = (
    "python3 -c \"import json,glob;"
    "t=sum(len(open(f).read()) for f in sorted(glob.glob('*.txt')));"
    "print(json.dumps({'ok': t <= 12, 'metric': t}))\""
)
# Always-passing check whose metric is the size, for ranking tests.
SIZE_CHECK = (
    "python3 -c \"import json,glob;"
    "t=sum(len(open(f).read()) for f in sorted(glob.glob('*.txt')));"
    "print(json.dumps({'ok': True, 'metric': t}))\""
)
# Reads a benchmark artifact instead of regenerating it. This is the shape that
# broke ann_vector_search_qps: it passes in the copy that produced `results/`
# and crashes against the merged tree, which never carries that directory.
ARTIFACT_CHECK = (
    "python3 -c \"import json;"
    "print(json.dumps({'ok': True, 'metric': len(open('results/run.out').read())}))\""
)


def _mk_runtime(tmp: Path):
    """Runtime wired to temp dirs, with a submit tool that records its calls."""
    from nanoma.core import Runtime, RuntimeConfig

    workspace_root = tmp / "task_cwd" / ".nanoma-task-work"
    submit_path = tmp / "task_cwd"
    submit_path.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    submits: list[str] = []

    async def fake_submit(args, agent, runtime):
        submits.append(agent.id)
        return {"submitted": True}

    config = RuntimeConfig(
        max_agents=32, max_depth=4, default_model="m", allowed_models=["m"],
        workspace_root=workspace_root, workspace_extra_roots=[submit_path],
        extra_tools={"submit": {"handler": fake_submit, "is_meta": True, "schema": {}}},
    )
    rt = Runtime(config=config)
    rt.start_agent = lambda child: None
    return rt, submit_path, submits


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _read(root: Path, rel: str) -> str | None:
    p = root / rel
    return p.read_text() if p.is_file() else None


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _verify(rt, agent, command=SIZE_CHECK, **kwargs):
    from optimizations.verify_tool import meta_verify
    return _run(meta_verify({"command": command, **kwargs}, agent, rt))


def _enable():
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"


def _disable():
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)


# ── the verify tool itself ───────────────────────────────────────────────────

def test_verify_runs_in_the_agents_own_copy():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "extra.txt", "xx\n")
        out = _verify(rt, child)
        assert out["ok"] and out["metric"] == 8, out       # 5 + 3, the copy's view
        assert out["registered"] is True
        # stored portably, so the runtime can re-run it somewhere else
        assert rt.VERIFY_WORKDIR_TOKEN in child._verify_spec["command"] or \
            str(child._merge_copy) not in child._verify_spec["command"]
    _disable()
    print("ok: verify runs in the agent's own copy and registers a re-runnable check")


def test_a_check_that_only_runs_in_the_copy_is_refused():
    """The gate re-runs a check against the merged tree, so passing in the child's
    own copy is not enough. ann_vector_search_qps registered a check reading its
    own `results/*.hdf5`; every promotion for the rest of the run then answered
    `ok: false`, and nothing was ever kept."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        # the child generated a benchmark artifact that the merge will not carry
        _write(Path(child._merge_copy), "results/run.out", "measured\n")
        _write(Path(child._merge_copy), "sol.txt", "better\n")

        out = _verify(rt, child, ARTIFACT_CHECK)

        assert out["ok"] is True, "it does pass where the agent ran it"
        assert out["registered"] is False, out
        assert "shared submission path" in out["note"], out["note"]
        assert getattr(child, "_verify_spec", None) is None
        assert rt._authoritative_spec() is None, "and it cannot gate anything"
        # the agent is told what to fix, not just that it failed
        assert "regenerate" in out["note"]
    _disable()
    print("ok: a check that only runs in the agent's own copy is refused")


def test_a_check_that_regenerates_what_it_measures_is_registered():
    """The same shape, done right: nothing outside the submit scope is read, so
    the check survives the move to the merged tree."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "sol.txt", "a-longer-answer\n")

        out = _verify(rt, child, SIZE_CHECK)

        assert out["registered"] is True, out
        assert rt._authoritative_spec()["command"] == child._verify_spec["command"]
    _disable()
    print("ok: a check that regenerates what it measures is registered")


def test_the_probe_runs_once_per_command():
    """The probe costs a full run of the check, so it is not paid twice."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "sol.txt", "x\n")

        runs: list[Path] = []
        original = rt._run_verification

        async def counting(command, workdir, timeout):
            runs.append(Path(workdir))
            return await original(command, workdir, timeout)

        rt._run_verification = counting
        _verify(rt, child, SIZE_CHECK)
        _verify(rt, child, SIZE_CHECK)

        assert runs.count(submit_path) == 1, runs
    _disable()
    print("ok: the target probe runs once per command")


def test_a_metric_from_the_private_copy_is_not_evidence():
    """A number measured in the copy and a number measured at the target describe
    different trees. Counting both as 'this check moves' promoted a permanent
    floor of zero in one run."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)

        # two different numbers, both measured in the child's own copy
        _write(Path(child._merge_copy), "sol.txt", "aa\n")
        assert _verify(rt, child, SIZE_CHECK)["ok"]
        _write(Path(child._merge_copy), "sol.txt", "aaaaaaaa\n")
        assert _verify(rt, child, SIZE_CHECK)["ok"]

        spec = rt._authoritative_spec()
        assert not rt._verify_metric_is_proven(spec), \
            "moving inside one's own copy does not prove the number tracks the work"
    _disable()
    print("ok: a metric measured only in the private copy is not evidence")


def test_unreadable_verdict_is_not_registered():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        out = _verify(rt, parent, command="echo 'looks fine to me'")
        assert out["ok"] is False and out["registered"] is False, out
        assert getattr(parent, "_verify_spec", None) is None
        failing = _verify(rt, parent, command="python3 -c \"import json;print(json.dumps({'ok': False, 'metric': 1}))\"")
        assert failing["ok"] is False and failing["registered"] is False
    _disable()
    print("ok: a check with no readable verdict, or a failing one, is not registered")


# ── merge plumbing ───────────────────────────────────────────────────────────

def test_disabled_is_noop():
    _disable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        assert getattr(child, "_merge_copy", None) is None, "no copy when disabled"
        assert not (child.workspace / "_task_copy").exists()
    print("ok: disabled -> no-op")


def test_seed_and_baseline():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "src/main.py", "print(1)\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        copy = Path(child._merge_copy)
        assert copy.is_dir() and _read(copy, "src/main.py") == "print(1)\n"
        assert rt._merge_baseline_path and rt._merge_baseline_path.is_dir()
        assert str(copy) in child.task and "complete private copy" in child.task
        assert "verify tool" in child.task, "the child is told how to make work count"
    _disable()
    print("ok: seed + baseline + task anchor")


def test_diff_add_modify_delete():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "keep.txt", "keep\n")
        _write(submit_path, "mod.txt", "old\n")
        _write(submit_path, "del.txt", "bye\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        copy = Path(child._merge_copy)
        _write(copy, "mod.txt", "new\n")
        _write(copy, "added.txt", "fresh\n")
        (copy / "del.txt").unlink()
        changed, deleted = rt._merge_diff(copy, rt._merge_baseline_path)
        assert set(changed) == {"mod.txt", "added.txt"}, changed
        assert deleted == {"del.txt"}, deleted
    _disable()
    print("ok: diff add/modify/delete")


# ── the gate ─────────────────────────────────────────────────────────────────

def test_merged_result_is_what_gets_verified():
    """Each child verifies fine alone; the merge of the two does not.

    This is the ann_vector_search_qps failure in miniature, and the only way to
    catch it is to run the check against the merged submission rather than
    trusting each child's measurement of its own copy.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")            # 5
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=BUDGET_CHECK)           # shared task check
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        _write(Path(a._merge_copy), "a.txt", "aaaa\n")      # 5 -> 10, within budget
        _write(Path(b._merge_copy), "b.txt", "bbbb\n")      # 5 -> 10, within budget

        assert _verify(rt, a, command=BUDGET_CHECK)["ok"] is True, "A verifies alone"
        assert _verify(rt, b, command=BUDGET_CHECK)["ok"] is True, "B verifies alone"

        first = _run(rt._merge_promote_child(a))
        assert first["kept"] is True and _read(submit_path, "a.txt") == "aaaa\n"
        second = _run(rt._merge_promote_child(b))
        assert second["kept"] is False, second
        assert second["verified"] is False, "the merged state failed the check"
        assert _read(submit_path, "b.txt") is None, "the breaking merge was rolled back"
        assert _read(submit_path, "a.txt") == "aaaa\n", "the good state survived"
        assert _read(Path(b._merge_copy), "b.txt") == "bbbb\n", "B's copy left intact"
        assert submits == [], "no official submission was consulted"
    _disable()
    print("ok: the merged artifact is verified, not each child's own copy")


def test_first_merge_must_still_pass_its_check():
    """What a live round actually did: four merges in a row failed their check
    and all four were kept, because nothing better had been measured yet — and
    the last failing state was then snapshotted as the one to fall back to."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=BUDGET_CHECK)
        rt._merge_best_rank = None          # nothing measured yet
        rt._merge_best_metric = None
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "big.txt", "over-the-budget-by-far\n")

        out = _run(rt._merge_promote_child(child))
        assert out["kept"] is False, out
        assert _read(submit_path, "big.txt") is None, "the failing merge was rolled back"
        assert rt._merge_best_rank is None, "a failing state never becomes the fallback"
    _disable()
    print("ok: the first merge is rolled back too if its check fails")


def test_a_failing_rank_cannot_be_recorded_as_best():
    """Second line of defence: even called directly, a failed check must not
    become the fallback state. One live round snapshotted exactly that."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        rt.create_agent(task="root", parent=None, depth=0)
        rt._record_verified_best(None, rt._verify_rank(False, None))
        assert rt._merge_best_rank is None and not rt._merge_best_dir().exists()
        rt._record_verified_best(9.0, rt._verify_rank(True, 9.0))
        assert rt._merge_best_rank is not None and rt._merge_best_metric == 9.0
    _disable()
    print("ok: only a passing check can be recorded as the fallback state")


def test_failed_check_is_diagnosable():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        parent._verify_spec["command"] = "python3 -c \"import sys;sys.exit(4)\""
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "sol.txt", "changed\n")
        events = []
        original_emit = rt._emit
        rt._emit = lambda a, t, d=None: (events.append((t, d)), original_emit(a, t, d))[1]
        _run(rt._merge_promote_child(child))
        merged = [d for t, d in events if t == "verify_merged"]
        assert merged and merged[0]["exit_code"] == 4, merged
        assert "output_tail" in merged[0], "the reason a check failed is recorded"
    _disable()
    print("ok: a check that cannot run against the merged tree is diagnosable")


def test_worse_metric_is_rolled_back():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        _write(Path(a._merge_copy), "sol.txt", "a-longer-solution\n")
        _write(Path(b._merge_copy), "sol.txt", "short\n")
        _run(rt._merge_promote_child(a))
        assert _read(submit_path, "sol.txt") == "a-longer-solution\n"
        _run(rt._merge_promote_child(b))
        assert _read(submit_path, "sol.txt") == "a-longer-solution\n", "worse metric reverted"
        assert rt._merge_best_metric == 18
    _disable()
    print("ok: a merge that verifies worse is rolled back")


def test_equal_metric_is_kept():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        _write(Path(a._merge_copy), "sol.txt", "vAAA\n")
        _write(Path(b._merge_copy), "sol.txt", "vBBB\n")
        _run(rt._merge_promote_child(a))
        _run(rt._merge_promote_child(b))
        assert _read(submit_path, "sol.txt") == "vBBB\n", "equal is not a regression"
    _disable()
    print("ok: an equally good merge is kept")


def test_union_of_disjoint_children():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "base.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        _write(Path(a._merge_copy), "a.txt", "AAA\n")
        _write(Path(b._merge_copy), "b.txt", "BBB\n")
        _run(rt._merge_promote_child(a))
        _run(rt._merge_promote_child(b))
        assert _read(submit_path, "a.txt") == "AAA\n"
        assert _read(submit_path, "b.txt") == "BBB\n", "disjoint work unions"
        assert _read(submit_path, "base.txt") == "base\n"
    _disable()
    print("ok: disjoint children union when the merge keeps verifying")


def test_unverified_work_lands_but_never_becomes_best():
    """With no check registered the runtime has no signal — don't strand the
    work, but never let it claim to be the state worth restoring."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "sol.txt", "unchecked\n")
        out = _run(rt._merge_promote_child(child))
        assert out["kept"] is True and out["verified"] is False
        assert "verify tool" in out["note"], "the child is told how to fix this"
        assert rt._merge_best_rank is None, "unverified work is never the best state"
        assert _read(submit_path, "sol.txt") == "unchecked\n"
    _disable()
    print("ok: unverified work lands but can never be the best state")


def test_no_change_skips_everything():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "x.txt", "x\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        out = _run(rt._merge_promote_child(a))
        assert out["status"] == "nothing_to_deliver", out
        assert submits == []
    _disable()
    print("ok: a child with no changes triggers nothing")


# ── keep-best ────────────────────────────────────────────────────────────────

def test_parent_verification_protects_the_ending():
    """The 4509 -> 0 ending: a verified state, then unchecked damage on top."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "a-good-solution\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        out = _verify(rt, parent, command=SIZE_CHECK)
        assert out["ok"] and rt._merge_best_rank is not None, "parent's check is a measurement"

        _write(submit_path, "sol.txt", "broken\n")   # hand-edit, never verified
        rt._merge_restore_best()
        assert _read(submit_path, "sol.txt") == "a-good-solution\n", "verified state restored"
    _disable()
    print("ok: an unchecked edit at the end gives way to the last verified state")


def test_improving_run_is_left_alone():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "aa\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        _write(submit_path, "sol.txt", "a-longer-better-one\n")
        _verify(rt, parent, command=SIZE_CHECK)
        rt._merge_restore_best()
        assert _read(submit_path, "sol.txt") == "a-longer-better-one\n"
    _disable()
    print("ok: a run that keeps verifying better is left untouched")


def test_pending_children_are_folded_in_at_run_end():
    """Shutdown cancels children mid-flight; their work must still land, and
    the combination — which nobody has ever run — must be verified once."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "base.txt", "b\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        _write(Path(a._merge_copy), "from_a.txt", "a\n")
        _write(Path(b._merge_copy), "from_b.txt", "b\n")
        _run(rt._merge_promote_pending())
        assert _read(submit_path, "from_a.txt") == "a\n", "cancelled child's work landed"
        assert _read(submit_path, "from_b.txt") == "b\n", "and its sibling's too"
        assert rt._merge_best_metric == 6, "the union was verified as one artifact"
        assert submits == [], "no official submission spent"
    _disable()
    print("ok: work from cancelled children is folded in and verified once")


def test_broken_end_of_run_union_is_rewound():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=BUDGET_CHECK)   # verified best = the baseline
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        _write(Path(a._merge_copy), "a.txt", "aaaa\n")
        _write(Path(b._merge_copy), "b.txt", "bbbb\n")   # union busts the budget
        _run(rt._merge_promote_pending())
        rt._merge_restore_best()
        assert _read(submit_path, "a.txt") is None and _read(submit_path, "b.txt") is None
        assert _read(submit_path, "sol.txt") == "base\n", "rewound to the verified state"
    _disable()
    print("ok: an end-of-run union that fails its check is rewound")


# ── submission stays the agent's decision ────────────────────────────────────

def test_child_submit_delivers_then_submits():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=SIZE_CHECK)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "sol.txt", "a-better-solution\n")
        tools = rt._merge_wrap_submit(child, dict(rt.config.extra_tools))
        out = _run(tools["submit"]["handler"]({}, child, rt))
        assert out["delivery"]["kept"] is True, out
        assert _read(submit_path, "sol.txt") == "a-better-solution\n", "delivered first"
        assert submits == [child.id], "then the agent's submission went through"
    _disable()
    print("ok: a child's submit delivers its verified work, then submits")


def test_child_submit_withheld_when_the_merge_fails():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        _verify(rt, parent, command=BUDGET_CHECK)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "big.txt", "wayyyy-too-long\n")
        tools = rt._merge_wrap_submit(child, dict(rt.config.extra_tools))
        out = _run(tools["submit"]["handler"]({}, child, rt))
        assert out["kept"] is False, out
        assert submits == [], "a submission is not spent on a state that fails its check"
    _disable()
    print("ok: no submission is spent when the merged state fails verification")


def test_parent_submit_goes_straight_out_when_nothing_is_in_flight():
    """The parent's submit waits for outstanding children, so with none it must
    not hesitate: no delay, and nothing rewritten on the way through."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        parent = rt.create_agent(task="root", parent=None, depth=0)
        tools = rt._merge_wrap_submit(parent, dict(rt.config.extra_tools))
        started = time.time()
        out = _run(tools["submit"]["handler"]({}, parent, rt))
        assert time.time() - started < 1.0, "no wait when there is nobody to wait for"
        assert submits == [parent.id]
        assert "waited_for" not in out, out
    _disable()
    print("ok: the parent submits straight away when nothing is in flight")


FLAT_CHECK = (
    'python3 -c "import json;print(json.dumps({\'ok\': True, \'metric\': 0.0}))"'
)


def test_a_flat_metric_is_not_used_as_a_floor():
    """A check that always says `ok, metric 0.0` is a vacuous ratchet — nothing can
    be worse than zero — so its verdict counts and its number does not."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)

        assert _verify(rt, a, FLAT_CHECK)["ok"]
        _write(Path(a._merge_copy), "sol.txt", "aaa\n")
        first = _run(rt._merge_promote_child(a))
        assert first["kept"] and first["metric"] == 0.0, first
        assert not rt._verify_metric_is_proven(rt._verification_spec_for(a)), \
            "one flat reading proves nothing"

        # a later contribution measuring the same flat 0.0 is still let through
        _write(Path(b._merge_copy), "sol.txt", "bbb\n")
        second = _run(rt._merge_promote_child(b))
        assert second["kept"], second
        assert second["metric_proven"] is False, second
        assert _read(submit_path, "sol.txt") == "bbb\n"
    _disable()
    print("ok: a check whose number never moves gates on its verdict alone")


def test_a_moving_metric_earns_the_ratchet():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)

        c = rt.create_agent(task="c", parent=parent.id, depth=1)

        # SIZE_CHECK reports the file's length, so it moves with the work
        _write(Path(a._merge_copy), "sol.txt", "a-modest-answer\n")
        assert _verify(rt, a, SIZE_CHECK)["ok"]
        assert _run(rt._merge_promote_child(a))["kept"]

        # a second, different reading is what proves the number tracks the state
        _write(Path(b._merge_copy), "sol.txt", "b-a-considerably-longer-answer\n")
        assert _run(rt._merge_promote_child(b))["kept"]
        assert rt._verify_metric_is_proven(rt._verification_spec_for(b)), \
            "the check has now been seen to move"

        _write(Path(c._merge_copy), "sol.txt", "c\n")
        regressed = _run(rt._merge_promote_child(c))
        assert not regressed["kept"] and regressed["metric_proven"], regressed
        assert _read(submit_path, "sol.txt") == "b-a-considerably-longer-answer\n", \
            "the ratchet held once the metric had earned it"
    _disable()
    print("ok: a check whose number moves is trusted as a floor")


def test_a_flat_newcomer_cannot_displace_a_working_check():
    """The failure seen live: one agent registers a real QPS check, another then
    registers eight `ok, metric 0.0` ones. The gate must keep the useful one."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        c = rt.create_agent(task="c", parent=parent.id, depth=1)

        # a's check earns its keep by moving across two states
        _write(Path(a._merge_copy), "sol.txt", "a-modest\n")
        assert _verify(rt, a, SIZE_CHECK)["ok"]
        assert _run(rt._merge_promote_child(a))["kept"]
        _write(Path(b._merge_copy), "sol.txt", "b-considerably-longer-answer\n")
        assert _run(rt._merge_promote_child(b))["kept"]
        floor = rt._merge_best_metric

        # now a flurry of flat registrations
        for _ in range(3):
            assert _verify(rt, c, FLAT_CHECK)["ok"]
        assert rt._authoritative_spec()["command"] == rt._verify_portable_command(
            SIZE_CHECK, a), "the check that measures something stays in force"
        assert rt._merge_best_metric == floor, "and its floor is not thrown away"

        _write(Path(c._merge_copy), "sol.txt", "c\n")
        regressed = _run(rt._merge_promote_child(c))
        assert not regressed["kept"], "so a regression is still caught"
        assert _read(submit_path, "sol.txt") == "b-considerably-longer-answer\n"
    _disable()
    print("ok: flat checks cannot displace one that measures the objective")


def test_two_checks_never_share_one_floor():
    """Two agents registering different checks report on different scales. Ranking
    a 0.33 from one against a 1.0 from the other rolls back good work."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "sol.txt", "base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)

        # alpha proves a check that scores the file's size (high numbers)
        (Path(a._merge_copy) / "sol.txt").write_text("a-longer-and-better-answer\n")
        assert _verify(rt, a, SIZE_CHECK)["ok"]
        first = _run(rt._merge_promote_child(a))
        assert first["kept"], first
        assert rt._merge_best_metric == len("a-longer-and-better-answer\n")

        # bravo proves a different check, on a different scale
        (Path(b._merge_copy) / "sol.txt").write_text("b\n")
        assert _verify(rt, b, BUDGET_CHECK)["ok"]
        assert rt._merge_best_metric is None, \
            "a new scale cannot inherit the old scale's floor"

        # and bravo's own contribution is judged by the check now in force,
        # not measured on one scale and compared against the other
        second = _run(rt._merge_promote_child(b))
        assert second["kept"], f"a small metric on a fresh scale is not a regression: {second}"
        spec = rt._verification_spec_for(a)
        assert spec["command"] == rt._verification_spec_for(b)["command"], \
            "one authoritative check gates everybody"
    _disable()
    print("ok: a change of check resets the floor instead of mixing two scales")


# ── the gate in front of an official submission ──────────────────────────────

def _gate(**env):
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_a_failing_preflight_spends_no_submission():
    """Six submissions in one run scored zero because the algorithm had been
    renamed and the judge found nothing to run. A preflight sees that first."""
    _enable()
    _gate(NANOMA_SUBMIT_PREFLIGHT="grep -qx custom names.txt")
    try:
        with tempfile.TemporaryDirectory() as d:
            rt, submit_path, submits = _mk_runtime(Path(d))
            _write(submit_path, "names.txt", "custom-ivf\n")  # renamed away
            parent = rt.create_agent(task="root", parent=None, depth=0)
            tools = rt._merge_wrap_submit(parent, dict(rt.config.extra_tools))

            out = _run(tools["submit"]["handler"]({}, parent, rt))

            assert out["blocked"] == "preflight_failed", out
            assert submits == [], "the submission was never spent"
            assert "preflight failed" in out["reason"]

            # and once the name is back, the same submission goes through
            _write(submit_path, "names.txt", "custom\n")
            out = _run(tools["submit"]["handler"]({}, parent, rt))
            assert out.get("submitted") is True, out
            assert submits == [parent.id]
    finally:
        _gate(NANOMA_SUBMIT_PREFLIGHT=None)
        _disable()
    print("ok: a failing preflight spends no submission")


def test_an_unverified_state_is_refused_when_asked_for():
    """One run submitted 31 seconds after its aggregate wait timed out, and the
    verdict it got back recorded that no check had passed that state."""
    _enable()
    _gate(NANOMA_SUBMIT_REQUIRE_VERIFIED="1")
    try:
        with tempfile.TemporaryDirectory() as d:
            rt, submit_path, submits = _mk_runtime(Path(d))
            _write(submit_path, "sol.txt", "base\n")
            parent = rt.create_agent(task="root", parent=None, depth=0)
            assert _verify(rt, parent, SIZE_CHECK)["ok"], "a check is in force"
            tools = rt._merge_wrap_submit(parent, dict(rt.config.extra_tools))

            # a verified state ships
            out = _run(tools["submit"]["handler"]({}, parent, rt))
            assert out.get("submitted") is True, out

            # editing it afterwards makes it a state nobody measured
            _write(submit_path, "sol.txt", "changed-without-checking\n")
            out = _run(tools["submit"]["handler"]({}, parent, rt))
            assert out["blocked"] == "unverified_state", out
            assert submits == [parent.id], "only the verified submission was spent"

            # re-running the check clears it
            assert _verify(rt, parent, SIZE_CHECK)["ok"]
            out = _run(tools["submit"]["handler"]({}, parent, rt))
            assert out.get("submitted") is True, out
    finally:
        _gate(NANOMA_SUBMIT_REQUIRE_VERIFIED=None)
        _disable()
    print("ok: an unverified state is refused when the gate is asked for")


def test_the_gate_is_inert_without_a_check_to_judge_by():
    """A run that registered no working check must still be able to submit — a
    broken gate that blocks everything is a zero, not a safeguard."""
    _enable()
    _gate(NANOMA_SUBMIT_REQUIRE_VERIFIED="1")
    try:
        with tempfile.TemporaryDirectory() as d:
            rt, submit_path, submits = _mk_runtime(Path(d))
            _write(submit_path, "sol.txt", "base\n")
            parent = rt.create_agent(task="root", parent=None, depth=0)
            assert rt._authoritative_spec() is None
            tools = rt._merge_wrap_submit(parent, dict(rt.config.extra_tools))

            out = _run(tools["submit"]["handler"]({}, parent, rt))

            assert out.get("submitted") is True, out
            assert submits == [parent.id]
    finally:
        _gate(NANOMA_SUBMIT_REQUIRE_VERIFIED=None)
        _disable()
    print("ok: the gate is inert when there is no check to judge by")


def test_no_gate_configured_leaves_submit_untouched():
    """Nothing configured, merging off: the tool is not wrapped at all."""
    with tempfile.TemporaryDirectory() as d:
        rt, _submit_path, submits = _mk_runtime(Path(d))
        parent = rt.create_agent(task="root", parent=None, depth=0)
        tools = dict(rt.config.extra_tools)

        assert rt._merge_wrap_submit(parent, tools)["submit"] is tools["submit"]

        _run(tools["submit"]["handler"]({}, parent, rt))
        assert submits == [parent.id]
    print("ok: with no gate configured submit is left alone")


def test_a_preflight_gates_a_child_delivery_submit_too():
    """A child's submit spends the same scarce resource, so it passes the same gate
    — after its work is merged, since that is the state being scored."""
    _enable()
    _gate(NANOMA_SUBMIT_PREFLIGHT="grep -qx custom names.txt")
    try:
        with tempfile.TemporaryDirectory() as d:
            rt, submit_path, submits = _mk_runtime(Path(d))
            _write(submit_path, "names.txt", "custom\n")
            _write(submit_path, "sol.txt", "base\n")
            parent = rt.create_agent(task="root", parent=None, depth=0)
            _verify(rt, parent, SIZE_CHECK)
            child = rt.create_agent(task="c", parent=parent.id, depth=1)
            # the child's work renames the algorithm away
            _write(Path(child._merge_copy), "names.txt", "custom-ivf\n")
            tools = rt._merge_wrap_submit(child, dict(rt.config.extra_tools))

            out = _run(tools["submit"]["handler"]({}, child, rt))

            assert out["delivery"]["kept"] is True, "the work still lands"
            assert out["submission"]["blocked"] == "preflight_failed", out
            assert submits == [], "but no submission is spent on it"
    finally:
        _gate(NANOMA_SUBMIT_PREFLIGHT=None)
        _disable()
    print("ok: a child's delivery submit passes the same gate")


def main():
    tests = [
        test_verify_runs_in_the_agents_own_copy,
        test_a_check_that_only_runs_in_the_copy_is_refused,
        test_a_check_that_regenerates_what_it_measures_is_registered,
        test_the_probe_runs_once_per_command,
        test_a_metric_from_the_private_copy_is_not_evidence,
        test_a_failing_preflight_spends_no_submission,
        test_an_unverified_state_is_refused_when_asked_for,
        test_the_gate_is_inert_without_a_check_to_judge_by,
        test_no_gate_configured_leaves_submit_untouched,
        test_a_preflight_gates_a_child_delivery_submit_too,
        test_unreadable_verdict_is_not_registered,
        test_disabled_is_noop,
        test_seed_and_baseline,
        test_diff_add_modify_delete,
        test_merged_result_is_what_gets_verified,
        test_first_merge_must_still_pass_its_check,
        test_a_failing_rank_cannot_be_recorded_as_best,
        test_failed_check_is_diagnosable,
        test_worse_metric_is_rolled_back,
        test_equal_metric_is_kept,
        test_union_of_disjoint_children,
        test_unverified_work_lands_but_never_becomes_best,
        test_a_flat_metric_is_not_used_as_a_floor,
        test_a_moving_metric_earns_the_ratchet,
        test_a_flat_newcomer_cannot_displace_a_working_check,
        test_two_checks_never_share_one_floor,
        test_no_change_skips_everything,
        test_parent_verification_protects_the_ending,
        test_improving_run_is_left_alone,
        test_pending_children_are_folded_in_at_run_end,
        test_broken_end_of_run_union_is_rewound,
        test_child_submit_delivers_then_submits,
        test_child_submit_withheld_when_the_merge_fails,
        test_parent_submit_goes_straight_out_when_nothing_is_in_flight,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} verification-gated merge tests passed.")


if __name__ == "__main__":
    main()
