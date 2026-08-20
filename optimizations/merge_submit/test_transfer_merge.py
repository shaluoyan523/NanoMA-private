"""Unit tests for transfer's diff-merge mode + uniqueness reminder.

Covers:
  * copy mode still works (regression guard)
  * merge mode with base='baseline': adds/modifies AND deletes by diff
  * merge mode without base: only differing files carried, no deletions
  * uniqueness check: overlapping active agents are reported + receiver reminded
  * push to a live agent injects a steer reminder into its mailbox

Run:  python3 -m optimizations.merge_submit.test_transfer_merge
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


def _mk_runtime(tmp: Path):
    from nanoma.core import Runtime, RuntimeConfig
    workspace_root = tmp / "ws"
    submit_path = tmp / "task_cwd"
    workspace_root.mkdir(parents=True, exist_ok=True)
    submit_path.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        max_agents=32, max_depth=4, default_model="m", allowed_models=["m"],
        workspace_root=workspace_root, workspace_extra_roots=[submit_path],
    )
    rt = Runtime(config=config)
    rt.start_agent = lambda child: None
    return rt, submit_path


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _read(root: Path, rel: str) -> str | None:
    p = root / rel
    return p.read_text() if p.is_file() else None


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _steer_contents(rt, agent_id):
    """Drain an agent's steer inbox (asyncio.Queue) and return message contents."""
    a = rt.agents[agent_id]
    q = getattr(a, "_steer_inbox", None)
    out = []
    if q is not None:
        while not q.empty():
            out.append(q.get_nowait().content)
    return out


def test_copy_mode_regression():
    import os
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    from nanoma.meta import meta_transfer
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, _ = _mk_runtime(tmp)
        a = rt.create_agent(task="a", parent=None, depth=0)
        b = rt.create_agent(task="b", parent=None, depth=0)
        _write(a.workspace, "note.txt", "hello\n")
        r = _run(meta_transfer({"src": "note.txt", "to": b.id}, a, rt))
        assert "pushed" in r and _read(b.workspace, "note.txt") == "hello\n", r
    print("ok: copy mode still works")


def test_merge_with_baseline_add_mod_delete():
    import os
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    from nanoma.meta import meta_transfer
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        # seed submit path (becomes the baseline for the child copy)
        _write(submit_path, "keep.txt", "keep\n")
        _write(submit_path, "mod.txt", "old\n")
        _write(submit_path, "del.txt", "bye\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        copy = Path(child._merge_copy)
        _write(copy, "mod.txt", "new\n")
        _write(copy, "added.txt", "fresh\n")
        (copy / "del.txt").unlink()
        # child pushes its diff (vs baseline) onto the shared submit path
        r = _run(meta_transfer({
            "src": str(copy), "to": "shared", "mode": "merge", "base": "baseline",
        }, child, rt))
        # 'shared' dest is runtime shared_dir; verify merge manifest instead of path
        assert set(r["merged"]) == {"mod.txt", "added.txt"}, r
        assert r["deleted"] == ["del.txt"], r
        shared = rt._tool_context.shared_dir
        assert _read(shared, "mod.txt") == "new\n"
        assert _read(shared, "added.txt") == "fresh\n"
        assert not (shared / "del.txt").exists(), "deletion applied"
    print("ok: merge with baseline (add/mod/delete)")


def test_merge_without_base_no_delete():
    import os
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    from nanoma.meta import meta_transfer
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "x.txt", "x\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        a = rt.create_agent(task="a", parent=parent.id, depth=1)
        b = rt.create_agent(task="b", parent=parent.id, depth=1)
        # b already has a pre-existing file the source doesn't know about
        _write(b.workspace, "existing.txt", "keepme\n")
        srcdir = Path(a._merge_copy)
        _write(srcdir, "feature.txt", "F\n")
        r = _run(meta_transfer({
            "src": str(srcdir), "to": b.id, "mode": "merge",  # no base
        }, a, rt))
        assert "feature.txt" in r["merged"], r
        assert r["deleted"] == [], "no deletions without a base"
        assert _read(b.workspace, "feature.txt") == "F\n"
        assert _read(b.workspace, "existing.txt") == "keepme\n", "unrelated file preserved"
    print("ok: merge without base (no deletions, preserves dest files)")


def test_uniqueness_overlap_detection_and_reminder():
    import os
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    from nanoma.meta import meta_transfer
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "shared_sol.py", "v0\n")
        _write(submit_path, "other.py", "o0\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        producer = rt.create_agent(task="p", parent=parent.id, depth=1)
        rival = rt.create_agent(task="r", parent=parent.id, depth=1)   # still active
        # both edit the SAME file in their private copies
        _write(Path(producer._merge_copy), "shared_sol.py", "vP\n")
        _write(Path(rival._merge_copy), "shared_sol.py", "vR\n")
        # producer pushes its diff to the parent, base=baseline
        r = _run(meta_transfer({
            "src": str(producer._merge_copy), "to": parent.id,
            "mode": "merge", "base": "baseline",
        }, producer, rt))
        assert "shared_sol.py" in r["merged"], r
        # rival is active and touches the same file -> reported as overlap
        assert "overlapping_agents" in r, r
        assert rival.id in r["overlapping_agents"], r["overlapping_agents"]
        assert "shared_sol.py" in r["overlapping_agents"][rival.id]
        assert "query" in r["uniqueness_check"].lower()
        assert "not delivered" in r["uniqueness_check"].lower() or "not deliver" in r["uniqueness_check"].lower()
        # parent (the receiver) got a steer reminder in its mailbox
        msgs = _steer_contents(rt, parent.id)
        assert any("merge" in m.lower() and rival.id in m for m in msgs), msgs
    print("ok: overlap detection + receiver reminder")


def test_uniqueness_unique_when_no_rivals():
    import os
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    from nanoma.meta import meta_transfer
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "sol.py", "v0\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        producer = rt.create_agent(task="p", parent=parent.id, depth=1)
        _write(Path(producer._merge_copy), "sol.py", "vP\n")
        r = _run(meta_transfer({
            "src": str(producer._merge_copy), "to": parent.id,
            "mode": "merge", "base": "baseline",
        }, producer, rt))
        assert "overlapping_agents" not in r, r
        assert "appears unique" in r["uniqueness_check"].lower(), r["uniqueness_check"]
    print("ok: reported unique when no rivals")


def main():
    tests = [
        test_copy_mode_regression,
        test_merge_with_baseline_add_mod_delete,
        test_merge_without_base_no_delete,
        test_uniqueness_overlap_detection_and_reminder,
        test_uniqueness_unique_when_no_rivals,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} transfer-merge tests passed.")


if __name__ == "__main__":
    main()
