"""Tests for the spawn decision taken at a delivery.

A planning moment is not the only point worth a spawn decision. A delivery is
when sibling contributions land on the same files and the submission becomes a
combination nobody has run — the state that, on ann_vector_search_qps, four
children built together and none of them ever executed.

The judge is given the orchestration picture (who else is in flight, which files
more than one agent changed, whether any check can run against the submission at
all) and decides whether to assign an agent to cross-check it. Roles are
assignments, not agent classes: what gets spawned is an ordinary agent whose task
happens to be verification, and whose deliverable is a registered check.

Run:  python3 -m optimizations.merge_submit.test_delivery_judge
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

SIZE_CHECK = (
    "python3 -c \"import json,glob;"
    "t=sum(len(open(f).read()) for f in sorted(glob.glob('*.txt')));"
    "print(json.dumps({'ok': True, 'metric': t}))\""
)

SPAWN_YES = json.dumps({
    "spawn": True,
    "reasoning": "two children rewrote the same file, the merge is untested",
    "subagents": [{
        "subject": "cross-check merged submission",
        "role": "verifier",
        "task": "Re-derive the benchmark result on the merged submission",
    }],
})
SPAWN_NO = json.dumps({"spawn": False, "reasoning": "already verified", "subagents": []})


def _mk_runtime(tmp: Path, judge_reply: str = SPAWN_YES):
    from nanoma.core import Runtime, RuntimeConfig

    submit_path = tmp / "task_cwd"
    workspace_root = submit_path / ".nanoma-task-work"
    submit_path.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        max_agents=16, max_depth=3, default_model="default-model",
        allowed_models=["default-model", "worker-model"],
        workspace_root=workspace_root, workspace_extra_roots=[submit_path],
    )
    rt = Runtime(config=config)
    rt.start_agent = lambda child: None
    seen: dict = {}

    class _Resp:
        content = judge_reply

    async def fake_llm(messages, model, tools=None, **kw):
        seen["model"] = model
        seen["system"] = messages[0]["content"]
        seen["user"] = messages[-1]["content"]
        seen["calls"] = seen.get("calls", 0) + 1
        return _Resp()

    rt.llm_call = fake_llm
    seen["emitted"] = emitted = []
    inner_emit = rt._emit
    rt._emit = lambda who, kind, data=None: (
        emitted.append((who, kind, data)), inner_emit(who, kind, data))[-1]
    return rt, submit_path, seen


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _drain(agent) -> str:
    """Everything sitting in an agent's inboxes, as text."""
    out = []
    for inbox in (agent._steer_inbox, agent._queue_inbox, agent._immediate_inbox):
        while not inbox.empty():
            out.append(getattr(inbox.get_nowait(), "content", ""))
    return "\n".join(out)


def _verify(rt, agent, command=SIZE_CHECK, **kw):
    from optimizations.verify_tool import meta_verify
    return _run(meta_verify({"command": command, **kw}, agent, rt))


def _enable():
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ["NANOMA_SPAWN_TODOLIST_JUDGE"] = "1"


def _disable():
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    os.environ.pop("NANOMA_SPAWN_TODOLIST_JUDGE", None)


def _fan_out(rt, submit_path, n=2):
    parent = rt.create_agent(
        task="make the benchmark faster", model="worker-model", parent=None, depth=0
    )
    _verify(rt, parent, command=SIZE_CHECK)
    children = [
        rt.create_agent(
            task=f"approach {i}", model="worker-model", parent=parent.id, depth=1
        )
        for i in range(n)
    ]
    return parent, children


def test_overlap_is_put_in_front_of_the_judge():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, seen = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, b) = _fan_out(rt, submit_path)
        # both children rewrite the same file, and each adds one of its own
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        (Path(a._merge_copy) / "a_only.txt").write_text("a\n")
        (Path(b._merge_copy) / "sol.txt").write_text("from-b\n")
        b.status = "working"

        delivery = _run(rt._merge_promote_child(a))
        a.status = "done"
        spawned = _run(rt._spawn_judge_at_delivery(a, delivery))

        assert spawned is True
        user = seen["user"]
        assert '"sol.txt"' in user and b.id in user, "the overlap is named for the judge"
        assert "a_only.txt" not in user.split("Files more than one agent changed")[1], \
            "a file only one agent touched is not reported as an overlap"
        assert f"- {b.id} [working]" in user, "siblings still in flight are listed"
        assert "OVERLAP" in seen["system"], "the criteria are stated"
        assert seen["model"] == "worker-model"
    _disable()
    print("ok: the judge is told which files several agents changed, and who is in flight")


def test_spawned_agent_owes_a_registered_check():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, b) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        before = set(rt.agents)
        delivery = _run(rt._merge_promote_child(a))
        assert _run(rt._spawn_judge_at_delivery(a, delivery)) is True
        new = [rt.agents[i] for i in set(rt.agents) - before]
        assert len(new) == 1
        verifier = new[0]
        assert verifier.parent == parent.id, "it is a sibling, not a grandchild"
        assert verifier.model == parent.model == "worker-model"
        assert getattr(verifier, "_spawned_at_delivery", False) is True
        assert "verify tool" in verifier.task, "its deliverable is a registered check"
        assert '{"ok": true, "metric":' in verifier.task, "with the verdict contract"
        assert f'query(agent_id="<id>"' in verifier.task, "and it can read others' context"
        assert verifier._merge_copy, "it works on its own copy of the merged submission"
    _disable()
    print("ok: what gets spawned is an ordinary agent that owes a reproducible check")


def test_only_one_cross_checker_at_a_time():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, b) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        (Path(b._merge_copy) / "sol.txt").write_text("from-b\n")
        first = _run(rt._merge_promote_child(a))
        assert _run(rt._spawn_judge_at_delivery(a, first)) is True
        count = len(rt.agents)

        second = _run(rt._merge_promote_child(b))
        assert _run(rt._spawn_judge_at_delivery(b, second)) is False
        assert len(rt.agents) == count, "no second cross-checker while one is running"

        for agent in rt.agents.values():
            if getattr(agent, "_spawned_at_delivery", False):
                agent.status = "done"
        third = _run(rt._merge_promote_child(b))
        if third and third.get("status") == "delivered":
            assert _run(rt._spawn_judge_at_delivery(b, third)) is True, \
                "once it finishes, a later delivery can get another"
    _disable()
    print("ok: cross-checkers do not pile up")


def test_nothing_delivered_means_no_decision():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, seen = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, _) = _fan_out(rt, submit_path)
        empty = _run(rt._merge_promote_child(a))       # child changed nothing
        assert empty["status"] == "nothing_to_deliver"
        assert _run(rt._spawn_judge_at_delivery(a, empty)) is False
        assert _run(rt._spawn_judge_at_delivery(a, None)) is False
        assert "user" not in seen, "the judge was not even consulted"
    _disable()
    print("ok: a delivery with nothing in it is not a decision point")


def test_judge_can_decline():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d), judge_reply=SPAWN_NO)
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, _) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        before = len(rt.agents)
        delivery = _run(rt._merge_promote_child(a))
        assert _run(rt._spawn_judge_at_delivery(a, delivery)) is False
        assert len(rt.agents) == before
    _disable()
    print("ok: a clean delivery can be left alone")


def test_judge_failure_is_harmless():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d), judge_reply="not json at all")
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, _) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        delivery = _run(rt._merge_promote_child(a))
        assert _run(rt._spawn_judge_at_delivery(a, delivery)) is False

        async def boom(*args, **kw):
            raise RuntimeError("judge unreachable")

        rt.llm_call = boom
        (Path(a._merge_copy) / "sol.txt").write_text("from-a-again\n")
        again = _run(rt._merge_promote_child(a))
        assert _run(rt._spawn_judge_at_delivery(a, again)) is False
        assert _read_ok(submit_path), "the delivery itself is unaffected"
    _disable()
    print("ok: an unreachable or babbling judge changes nothing")


def _read_ok(submit_path: Path) -> bool:
    return (submit_path / "sol.txt").is_file()


def test_missing_runnable_check_is_reported():
    """The judge needs to know when nothing can measure the submission — the
    condition that made a live round's promotes unverifiable."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, seen = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        a = rt.create_agent(task="approach", parent=parent.id, depth=1)
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        delivery = _run(rt._merge_promote_child(a))    # nobody registered a check
        _run(rt._spawn_judge_at_delivery(a, delivery))
        assert "NONE REGISTERED" in seen["user"], seen["user"][-400:]
        assert "own registered check: NONE" in seen["user"]
    _disable()
    print("ok: 'nothing can measure the submission' is stated plainly to the judge")


def test_a_delivery_reaches_the_peers_it_affects_directly():
    """The routing question: a sibling's delivery must reach whoever it affects
    without the parent having to notice and forward it."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, b) = _fan_out(rt, submit_path)
        watcher = rt.create_agent(task="w", parent=parent.id, depth=1)  # nothing of its own

        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        _run(rt._merge_promote_child(a))

        note = _drain(watcher)
        assert note, "the delivery reached it without a relay"
        assert a.id in note and "sol.txt" in note, "it is told who changed what"
        assert Path(watcher._merge_copy, "sol.txt").read_text() == "from-a\n", \
            "and it is looking at the delivered state, not the one it was spawned with"
        assert not any(rt._merge_own_changes(watcher)), \
            "the state it was handed is not mistaken for work of its own"
    _disable()
    print("ok: a delivery lands in the inbox of whoever it affects, not via the parent")


def test_an_agent_with_work_of_its_own_is_warned_not_overwritten():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, b) = _fan_out(rt, submit_path)
        (Path(b._merge_copy) / "sol.txt").write_text("bravo-is-mid-edit\n")

        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        _run(rt._merge_promote_child(a))

        assert Path(b._merge_copy, "sol.txt").read_text() == "bravo-is-mid-edit\n", \
            "its unfinished work is never overwritten by a refresh"
        note = _drain(b)
        assert "sol.txt" in note and a.id in note, "but it is told about the overlap"
    _disable()
    print("ok: an agent mid-edit keeps its work and is told who else touched those files")


def test_a_refreshed_agent_delivers_its_own_work_only():
    """Load-bearing: after a refresh its copy holds everyone else's work too, so a
    diff against the round baseline would re-apply contributions — including ones
    rolled back for failing. Its baseline moves with the refresh instead."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        (submit_path / "other.txt").write_text("untouched\n")
        parent, (a, b) = _fan_out(rt, submit_path)
        watcher = rt.create_agent(task="w", parent=parent.id, depth=1)

        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        _run(rt._merge_promote_child(a))   # watcher's copy is refreshed to this

        # nothing of its own yet, so nothing to deliver
        assert _run(rt._merge_promote_child(watcher))["status"] == "nothing_to_deliver"

        # now it does have something, and only that is delivered
        (Path(watcher._merge_copy) / "other.txt").write_text("fixed-by-the-watcher\n")
        out = _run(rt._merge_promote_child(watcher))
        assert out["changed_files"] == ["other.txt"], out
        assert (submit_path / "sol.txt").read_text() == "from-a\n", "a's work untouched"
        assert (submit_path / "other.txt").read_text() == "fixed-by-the-watcher\n"
    _disable()
    print("ok: a refreshed agent still delivers, and delivers only what is its own")


def test_what_verify_measures_follows_from_the_copy():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        from optimizations.verify_tool import meta_verify
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, _) = _fan_out(rt, submit_path)
        watcher = rt.create_agent(task="w", parent=parent.id, depth=1)
        (Path(a._merge_copy) / "sol.txt").write_text("a-much-longer-solution\n")
        _run(rt._merge_promote_child(a))

        # nothing of its own: what it means to measure is the submission
        out = _run(meta_verify({"command": SIZE_CHECK}, watcher, rt))
        assert out["measured"] == "submission", out
        assert out["metric"] == len("a-much-longer-solution\n"), out

        # once it is editing, its own copy is what it means
        (Path(watcher._merge_copy) / "sol.txt").write_text("tiny\n")
        own = _run(meta_verify({"command": SIZE_CHECK}, watcher, rt))
        assert own["measured"] == "own" and own["metric"] == len("tiny\n"), own
    _disable()
    print("ok: what verify measures follows from the copy, not from a label")


def test_staying_is_cheap_because_a_delivery_wakes_it():
    """Residency is only a real option if waiting costs nothing: an agent that
    parks itself must be brought back by the next delivery, not left asleep."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, _) = _fan_out(rt, submit_path)
        watcher = rt.create_agent(task="w", parent=parent.id, depth=1)
        watcher.status = "idle"

        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        _run(rt._merge_promote_child(a))

        assert watcher.status == "running", "the delivery woke it"
        assert a.id in _drain(watcher), "and it woke to the facts, not to nothing"
    _disable()
    print("ok: an agent that parks itself is woken by the next delivery")


def test_an_agent_finishing_is_told_who_is_still_in_flight():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, b) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("mine\n")
        (Path(b._merge_copy) / "sol.txt").write_text("also-mine\n")

        class _Call:
            name = "set_status"
            arguments = {"status": "done", "result": "finished"}

        held = rt._finish_time_situation(a, _Call())
        assert held and held["still_in_flight"] == {b.id: ["sol.txt"]}, held
        assert "verify" in held["detail"], "it is told its check outlives it"
        assert rt._finish_time_situation(a, _Call()) is None, \
            "said once, so nobody can be held in"
    _disable()
    print("ok: an agent deciding whether to stay is handed the state of play, once")



def test_delivery_survives_the_kill_that_follows_it():
    """A parent kills a child right after it delivers, so the fold-back runs with a
    cancellation pending. It must still land, and must not cost an Opus call."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, seen = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        _, (a, _) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("delivered-under-the-axe\n")
        a.status = "killed"

        async def kill_mid_finalize():
            async def child_loop():
                try:
                    await asyncio.sleep(3600)        # busy working
                finally:
                    rt._schedule_child_delivery(a)   # what the agent loop does

            loop_task = asyncio.ensure_future(child_loop())
            await asyncio.sleep(0.05)
            loop_task.cancel()       # the parent kills it right after it delivered
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            await rt._await_delivery_tasks(timeout=30)

        _run(kill_mid_finalize())
        assert (submit_path / "sol.txt").read_text() == "delivered-under-the-axe\n", \
            "a killed child's work still reaches the submission"
        assert seen.get("calls", 0) == 0, "and no judge round-trip is spent on a dead child"
    _disable()
    print("ok: work delivered just before a kill is not dropped on the floor")


def test_a_slow_judge_cannot_stall_the_ending():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, seen = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        _, (a, _) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")

        async def never_answers(*args, **kwargs):
            await asyncio.sleep(600)
        rt.llm_call = never_answers
        rt._DELIVERY_JUDGE_TIMEOUT = 0.2

        started = time.time()
        _run(rt._finalize_child_delivery(a))
        assert time.time() - started < 20, "the judge is bounded"
        assert (submit_path / "sol.txt").read_text() == "from-a\n", "delivery landed anyway"
        errs = [e for e in seen["emitted"] if e[1] == "spawn_judge_error"]
        assert errs and "timed out" in str(errs[-1]), errs
    _disable()
    print("ok: an unresponsive judge is abandoned instead of stalling the run")


def test_disabled_by_default():
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ.pop("NANOMA_SPAWN_TODOLIST_JUDGE", None)
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, seen = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, (a, _) = _fan_out(rt, submit_path)
        (Path(a._merge_copy) / "sol.txt").write_text("from-a\n")
        delivery = _run(rt._merge_promote_child(a))
        assert _run(rt._spawn_judge_at_delivery(a, delivery)) is False
        assert "user" not in seen
    _disable()
    print("ok: no delivery judging unless the judge is switched on")


def main():
    tests = [
        test_overlap_is_put_in_front_of_the_judge,
        test_spawned_agent_owes_a_registered_check,
        test_only_one_cross_checker_at_a_time,
        test_nothing_delivered_means_no_decision,
        test_judge_can_decline,
        test_judge_failure_is_harmless,
        test_missing_runnable_check_is_reported,
        test_a_delivery_reaches_the_peers_it_affects_directly,
        test_an_agent_with_work_of_its_own_is_warned_not_overwritten,
        test_a_refreshed_agent_delivers_its_own_work_only,
        test_what_verify_measures_follows_from_the_copy,
        test_staying_is_cheap_because_a_delivery_wakes_it,
        test_an_agent_finishing_is_told_who_is_still_in_flight,
        test_delivery_survives_the_kill_that_follows_it,
        test_a_slow_judge_cannot_stall_the_ending,
        test_disabled_by_default,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} delivery-judge tests passed.")


if __name__ == "__main__":
    main()
