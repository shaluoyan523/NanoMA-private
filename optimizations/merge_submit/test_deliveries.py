"""Unit tests for the `deliveries` tool.

The wait a parent does before submitting was invisible and unavoidable: it ran
inside the `submit` call, for up to fifteen minutes, and the parent had no way to
ask who it was for or to proceed without it. One run blocked its root for 2702 of
its 6240 seconds across three such waits on the same three children, and reached
its deadline having submitted once.

Run:  python3 -m optimizations.merge_submit.test_deliveries
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from optimizations.merge_submit.test_aggregate_wait import (
    _disable,
    _enable,
    _fan_out,
    _mk_runtime,
    _run,
)


def _deliveries(rt, agent, **args):
    from optimizations.merge_submit import meta_deliveries
    return _run(meta_deliveries(args, agent, rt))


def test_a_parent_can_see_who_is_holding_what():
    """The question the parent could not ask while it was being made to wait."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        (Path(kids[1]._merge_copy) / "mine.txt").write_text("half-written\n")

        out = _deliveries(rt, parent)
        assert kids[1].id in out["outstanding"], out
        assert "mine.txt" in out["outstanding"][kids[1].id]["undelivered_files"], out
        assert out["outstanding"][kids[1].id]["status"] == "running", out
    _disable()
    print("ok: a parent can see which children are holding which files")


def test_the_remaining_wait_budget_is_reported():
    """Whether `submit` will still block is the parent's decision input."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        rt._AGGREGATE_WAIT_SECONDS = 10.0
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)

        assert _deliveries(rt, parent)["wait_budget_remaining_seconds"] == 10.0
        rt._aggregate_wait_spent[parent.id] = 10.0
        out = _deliveries(rt, parent)
        assert out["wait_budget_remaining_seconds"] == 0.0
        assert "budget is spent" in out.get("note", ""), out
    _disable()
    print("ok: the remaining wait budget is reported, so the parent can decide")


def test_collect_folds_in_what_is_already_written_without_waiting():
    """The action the parent needed: take what exists now, do not wait for done."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        (Path(kids[0]._merge_copy) / "sol.txt").write_text("improved by kid 0\n")

        out = _deliveries(rt, parent, collect=True)
        assert "collected" in out, out
        assert (submit_path / "sol.txt").read_text() == "improved by kid 0\n"
        # The child is still running; only its written work was taken.
        assert kids[0].status == "running"
    _disable()
    print("ok: collect folds in what is written without waiting for anyone")


def test_collecting_twice_is_harmless():
    """A parent will call this repeatedly; unchanged work must not re-merge."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        (Path(kids[0]._merge_copy) / "sol.txt").write_text("improved\n")

        _deliveries(rt, parent, collect=True)
        second = _deliveries(rt, parent, collect=True)
        assert "collect_error" not in second, second
        assert (submit_path / "sol.txt").read_text() == "improved\n"
    _disable()
    print("ok: collecting twice changes nothing the second time")


def test_nothing_outstanding_says_so_plainly():
    """So a parent is not left guessing whether it may submit."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        (submit_path / "sol.txt").write_text("base\n")
        parent, kids = _fan_out(rt, n=2)
        for k in kids:
            k.status = "done"

        out = _deliveries(rt, parent)
        assert out["outstanding"] == {}, out
        assert "whole workspace" in out.get("note", ""), out
    _disable()
    print("ok: with nothing outstanding the parent is told it may submit")


def test_the_tool_stays_out_of_a_run_without_a_submission_path():
    """It must not appear to work where children do not deliver files."""
    _disable()
    with tempfile.TemporaryDirectory() as d:
        rt, _, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        assert "error" in _deliveries(rt, agent)
    print("ok: without a shared submission path the tool says so")


def main() -> None:
    tests = [
        test_a_parent_can_see_who_is_holding_what,
        test_the_remaining_wait_budget_is_reported,
        test_collect_folds_in_what_is_already_written_without_waiting,
        test_collecting_twice_is_harmless,
        test_nothing_outstanding_says_so_plainly,
        test_the_tool_stays_out_of_a_run_without_a_submission_path,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} deliveries tests passed.")


if __name__ == "__main__":
    main()
