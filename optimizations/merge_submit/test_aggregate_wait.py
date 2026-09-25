"""Aggregating a fan-out only once all of it has arrived.

A parent that fans out and then aggregates the first thing to come back throws
away the parallelism it just paid for — and, in a merge run, submits a state no
check has passed. These cover the three places that has to hold: the wait itself,
what the parent is told when one child reports, and the official submission.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

SIZE_CHECK = (
    "python3 -c \"import json,glob;"
    "t=sum(len(open(f).read()) for f in sorted(glob.glob('*.txt')));"
    "print(json.dumps({'ok': True, 'metric': t}))\""
)


def _mk_runtime(tmp: Path):
    from nanoma.core import Runtime, RuntimeConfig

    submit_path = tmp / "task_cwd"
    workspace_root = submit_path / ".nanoma-task-work"
    submit_path.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    submits: list[dict] = []

    async def fake_submit(args, agent, runtime):
        submits.append({
            "by": agent.id,
            "state": (submit_path / "sol.txt").read_text()
            if (submit_path / "sol.txt").is_file() else "",
        })
        return {"submitted": True, "stdout": ""}

    config = RuntimeConfig(
        max_agents=16, max_depth=3, default_model="m", allowed_models=["m"],
        workspace_root=workspace_root, workspace_extra_roots=[submit_path],
        extra_tools={"submit": {"handler": fake_submit, "is_meta": True, "schema": {}}},
    )
    rt = Runtime(config=config)
    rt.start_agent = lambda child: None
    rt._AGGREGATE_WAIT_SECONDS = 10.0
    rt._AGGREGATE_POLL_SECONDS = 0.05
    emitted: list = []
    inner = rt._emit
    rt._emit = lambda who, kind, data=None: (
        emitted.append((who, kind, data)), inner(who, kind, data))[-1]
    rt._emitted = emitted
    return rt, submit_path, submits


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _fan_out(rt, n=3):
    parent = rt.create_agent(task="root", parent=None, depth=0)
    kids = [rt.create_agent(task=f"c{i}", parent=parent.id, depth=1) for i in range(n)]
    for k in kids:
        parent.children.add(k.id)
        k.status = "running"
    return parent, kids


async def _adeliver(rt, to, content, from_id="system"):
    from nanoma.core import Envelope
    await rt.deliver(Envelope(
        from_id=from_id, to_id=to.id, content=content,
        tokens=1, timestamp=time.time(), mode="steer",
    ))


def _deliver(rt, to, content, from_id="system"):
    _run(_adeliver(rt, to, content, from_id))


def _enable():
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"


def _disable():
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    os.environ.pop("NANOMA_SUBMIT_REQUIRE_COMPLETE", None)


def _kinds(rt):
    return [k for _, k, _ in rt._emitted]


# ── the wait ─────────────────────────────────────────────────────────────────

def test_a_childs_arrival_does_not_cut_short_the_wait_for_it():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, _, _ = _mk_runtime(Path(d))
        parent, kids = _fan_out(rt)
        ids = [k.id for k in kids]

        _deliver(rt, parent, f"[Agent {ids[0]} finished: done] here is my half")
        assert not rt._wait_is_interrupted(parent, ids), \
            "the report it is waiting for is progress, not an interruption"

        _deliver(rt, parent, "partial finding", from_id=ids[1])
        assert not rt._wait_is_interrupted(parent, ids), "nor is one of them writing in"
    _disable()
    print("ok: the arrivals a wait is for do not cut it short")


def test_an_unrelated_message_still_cuts_it_short():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, _, _ = _mk_runtime(Path(d))
        parent, kids = _fan_out(rt)
        ids = [k.id for k in kids]
        stranger = rt.create_agent(task="s", parent=None, depth=0)

        _deliver(rt, parent, "drop everything and look at this", from_id=stranger.id)
        assert rt._wait_is_interrupted(parent, ids), \
            "a message from outside the fan-out is still worth stopping for"
    _disable()
    print("ok: a message from outside the fan-out still interrupts")


def test_wait_all_returns_only_when_all_have_reported():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        from nanoma.meta import meta_wait
        rt, _, _ = _mk_runtime(Path(d))
        parent, kids = _fan_out(rt)

        async def scenario():
            waiting = asyncio.ensure_future(
                meta_wait({"mode": "all", "timeout": 8}, parent, rt)
            )
            await asyncio.sleep(0.2)
            for k in kids[:2]:                 # two of three report
                k.status = "done"
                k.result = f"{k.id} result"
                await _adeliver(rt, parent, f"[Agent {k.id} finished: done] {k.result}")
            await asyncio.sleep(0.5)
            assert not waiting.done(), "it kept waiting for the third"
            kids[2].status = "done"
            kids[2].result = "last result"
            return await waiting

        out = _run(scenario())
        assert not out.get("interrupted"), out
        assert len(out["completed"]) == 3 and not out["pending"], out
    _disable()
    print("ok: mode='all' comes back with the whole set, not the first arrival")


# ── what the parent is told ──────────────────────────────────────────────────

def test_the_first_report_says_what_is_still_coming():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt)
        (Path(kids[1]._merge_copy) / "sol.txt").write_text("c1 mid-edit\n")

        note = rt._outstanding_note(parent)
        assert kids[1].id in note and kids[2].id in note, note
        assert "sol.txt" in note, "including who is holding undelivered changes"
        assert "wait(mode='all')" in note, "and how to actually wait"

        for k in kids:
            k.status = "done"
        assert rt._outstanding_note(parent) == "", "silent once nothing is outstanding"
    _disable()
    print("ok: the parent is told what is still coming, and what is unfinished")


def test_a_parked_child_is_not_something_to_wait_for():
    """It owes nothing and is itself waiting to be woken — waiting on it is a standoff."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt)
        for k in kids:
            k.status = "idle"

        assert rt._outstanding_deliveries(parent) == {}, "nobody owes anything"

        # but a parked child sitting on undelivered work still counts
        (Path(kids[0]._merge_copy) / "sol.txt").write_text("parked but written\n")
        outstanding = rt._outstanding_deliveries(parent)
        assert list(outstanding) == [kids[0].id], outstanding
        assert outstanding[kids[0].id]["undelivered_files"] == ["sol.txt"]
    _disable()
    print("ok: a parked child owes nothing unless it is sitting on work")


# ── the submission ───────────────────────────────────────────────────────────

def test_submitting_waits_for_the_rest_and_merges_once():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        tools = rt._merge_wrap_submit(parent, {
            "submit": rt.config.extra_tools["submit"]})

        (Path(kids[0]._merge_copy) / "sol.txt").write_text("first-half\n")
        kids[0].status = "done"

        async def scenario():
            submitting = asyncio.ensure_future(
                tools["submit"]["handler"]({}, parent, rt)
            )
            await asyncio.sleep(0.3)
            assert not submits, "it did not spend a submission on half the work"
            # the straggler finishes and hands over
            (Path(kids[1]._merge_copy) / "other.txt").write_text("second-half\n")
            kids[1].status = "done"
            await rt._merge_promote_child(kids[1])
            return await submitting

        out = _run(scenario())
        assert submits, "the submission did happen"
        assert kids[1].id in (out.get("waited_for") or []), out
        assert (submit_path / "sol.txt").read_text() == "first-half\n"
        assert (submit_path / "other.txt").read_text() == "second-half\n", \
            "both halves are in the submitted workspace"
        assert "aggregate_wait" in _kinds(rt)
    _disable()
    print("ok: an official submission waits for the rest, then goes out once")


def test_a_straggler_that_never_arrives_does_not_block_forever():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.4
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        tools = rt._merge_wrap_submit(parent, {
            "submit": rt.config.extra_tools["submit"]})

        started = time.time()
        _run(tools["submit"]["handler"]({}, parent, rt))
        assert time.time() - started < 8, "the wait is bounded"
        assert submits, "and the submission still goes out"
        assert "aggregate_wait_timeout" in _kinds(rt)
    _disable()
    print("ok: a straggler delays the submission, it does not cancel it")


def test_the_wait_budget_is_spent_once_not_once_per_attempt():
    """Observed on the 2026-07-30 run: 2702 of 6240 seconds blocked.

    The deadline used to be recomputed on entry, so each submission attempt
    bought another full wait on the same children. Three attempts blocked the
    root for 45 minutes on the same three children holding the same two files,
    and the run reached its deadline having submitted once.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 1.0
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        _straggler_holding(rt, kids)

        first = time.time()
        _run(rt._await_outstanding_deliveries(parent))
        first_took = time.time() - first

        second = time.time()
        out = _run(rt._await_outstanding_deliveries(parent))
        second_took = time.time() - second

        assert first_took >= 0.9, f"the first attempt spends the budget ({first_took:.2f}s)"
        assert second_took < 0.3, f"the second must not buy another ({second_took:.2f}s)"
        assert out["timed_out"] is True, "and it still reports the state honestly"
        assert out["holding"], "including who is still sitting on changes"
    _disable()
    print("ok: the wait budget is spent once per run, not once per attempt")


def test_an_exhausted_budget_still_reports_what_is_outstanding():
    """The refusal downstream needs the holdings, so returning early must not
    hide them — otherwise skipping the wait would silently permit the submission
    the wait exists to stop."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.0
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        _straggler_holding(rt, kids)

        out = _run(rt._await_outstanding_deliveries(parent))
        assert out["timed_out"] is True, out
        assert out["holding"], out
        timeouts = [d for _, k, d in rt._emitted if k == "aggregate_wait_timeout"]
        assert timeouts and timeouts[-1]["exhausted"] is True, timeouts
    _disable()
    print("ok: an exhausted budget still reports what is outstanding")


# ── a timed-out wait is not permission to go ─────────────────────────────────
# The wait exists so a submission describes the whole workspace. When it timed
# out the submission went out anyway, because the timeout was invisible at the
# call site: the wait returned the names it had waited on and nothing about how
# it ended. One run waited the full 15 minutes on a child, submitted 31 seconds
# later, and the verdict it got back recorded that no local check had passed the
# state it sent.

def _straggler_holding(rt, kids, rel="mine.txt", text="half-written\n"):
    """A child still running and sitting on changes it has not handed over."""
    (Path(kids[1]._merge_copy) / rel).write_text(text)


def test_a_timed_out_wait_refuses_the_submission_once():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.4
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        tools = rt._merge_wrap_submit(parent, {
            "submit": rt.config.extra_tools["submit"]})
        _straggler_holding(rt, kids)

        out = _run(tools["submit"]["handler"]({}, parent, rt))
        assert out["blocked"] == "aggregate_incomplete", out
        assert kids[1].id in out["reason"], out
        assert "mine.txt" in out["reason"], out
        assert submits == [], "no submission was spent on the unfinished state"
        assert "aggregate_wait_timeout" in _kinds(rt)
    _disable()
    print("ok: a wait that timed out on a child holding work refuses the submission")


def test_submitting_again_goes_ahead():
    """A child can hang for the rest of the run, and refusing forever would turn
    that into a zero. The refusal makes the choice informed, not for the agent."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.4
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        tools = rt._merge_wrap_submit(parent, {
            "submit": rt.config.extra_tools["submit"]})
        _straggler_holding(rt, kids)

        first = _run(tools["submit"]["handler"]({}, parent, rt))
        assert first["blocked"] == "aggregate_incomplete", first
        second = _run(tools["submit"]["handler"]({}, parent, rt))
        assert second.get("submitted") is True, second
        assert len(submits) == 1, submits
        assert "submit_incomplete_allowed" in _kinds(rt)
    _disable()
    print("ok: submitting again after being told goes ahead")


def test_a_straggler_holding_nothing_is_not_refused():
    """Outstanding is not the same as owing something. A child that is running
    but has written nothing is not work this submission is missing."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.4
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        tools = rt._merge_wrap_submit(parent, {
            "submit": rt.config.extra_tools["submit"]})

        out = _run(tools["submit"]["handler"]({}, parent, rt))
        assert out.get("submitted") is True, out
        assert submits, submits
        assert "aggregate_wait_timeout" in _kinds(rt)
    _disable()
    print("ok: a straggler that owes nothing does not hold up the submission")


def test_a_clean_wait_rearms_the_refusal():
    """Being told once is about one timeout, not about the rest of the run."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.4
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        tools = rt._merge_wrap_submit(parent, {
            "submit": rt.config.extra_tools["submit"]})
        _straggler_holding(rt, kids)

        assert _run(tools["submit"]["handler"]({}, parent, rt))["blocked"] == \
            "aggregate_incomplete"

        # the stragglers finish, so the next wait ends cleanly
        for k in kids:
            k.status = "done"
        assert _run(tools["submit"]["handler"]({}, parent, rt)).get("submitted") is True

        # a fresh child starts and stalls holding work: refused again
        newcomer = rt.create_agent(task="c-new", parent=parent.id, depth=1)
        parent.children.add(newcomer.id)
        newcomer.status = "running"
        (Path(newcomer._merge_copy) / "later.txt").write_text("unfinished\n")
        out = _run(tools["submit"]["handler"]({}, parent, rt))
        assert out["blocked"] == "aggregate_incomplete", out
    _disable()
    print("ok: a clean wait re-arms the refusal for the next timeout")


def test_the_refusal_can_be_turned_off():
    _enable()
    os.environ["NANOMA_SUBMIT_REQUIRE_COMPLETE"] = "0"
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.4
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        tools = rt._merge_wrap_submit(parent, {
            "submit": rt.config.extra_tools["submit"]})
        _straggler_holding(rt, kids)

        out = _run(tools["submit"]["handler"]({}, parent, rt))
        assert out.get("submitted") is True, out
    _disable()
    print("ok: the refusal can be turned off")


def test_the_wait_reports_how_it_ended():
    """The defect itself: the caller could not tell the two endings apart."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 0.4
        parent, kids = _fan_out(rt, n=2)
        _straggler_holding(rt, kids)

        timed = _run(rt._await_outstanding_deliveries(parent))
        assert timed["timed_out"] is True, timed
        assert kids[1].id in timed["holding"], timed
        assert "mine.txt" in timed["holding"][kids[1].id], timed

        for k in kids:
            k.status = "done"
        clean = _run(rt._await_outstanding_deliveries(parent))
        assert clean["timed_out"] is False, clean
        assert clean["holding"] == {}, clean
    _disable()
    print("ok: the wait reports whether it finished or gave up, and on what")


def test_children_submit_as_before():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        child = kids[0]
        tools = rt._merge_wrap_submit(child, {
            "submit": rt.config.extra_tools["submit"]})
        (Path(child._merge_copy) / "sol.txt").write_text("mine\n")

        out = _run(tools["submit"]["handler"]({}, child, rt))
        assert out["delivery"]["kept"], out
        assert submits and submits[0]["by"] == child.id
    _disable()
    print("ok: a child's own submit still delivers its work first, unchanged")


def main():
    tests = [
        test_a_childs_arrival_does_not_cut_short_the_wait_for_it,
        test_an_unrelated_message_still_cuts_it_short,
        test_wait_all_returns_only_when_all_have_reported,
        test_the_first_report_says_what_is_still_coming,
        test_a_parked_child_is_not_something_to_wait_for,
        test_submitting_waits_for_the_rest_and_merges_once,
        test_a_straggler_that_never_arrives_does_not_block_forever,
        test_the_wait_reports_how_it_ended,
        test_the_wait_budget_is_spent_once_not_once_per_attempt,
        test_an_exhausted_budget_still_reports_what_is_outstanding,
        test_a_timed_out_wait_refuses_the_submission_once,
        test_submitting_again_goes_ahead,
        test_a_straggler_holding_nothing_is_not_refused,
        test_a_clean_wait_rearms_the_refusal,
        test_the_refusal_can_be_turned_off,
        test_children_submit_as_before,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} aggregate-wait tests passed.")


if __name__ == "__main__":
    main()
