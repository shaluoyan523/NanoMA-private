"""Unit tests for the child working copy, merge scoping, and spawn cap.

These cover both failure modes seen on ann_vector_search_qps (509MB task
directory, 12KB submission):
  * copying the whole tree per child -> OOM, run killed
  * copying only the submission subpath -> children get an unrunnable stub and
    every submission scores 0

The fix keeps two notions apart, which is what these tests pin down:
  * working copy: the COMPLETE tree; submitted files are always private while
    large out-of-scope payloads may be hardlinked
  * merge scope: only submission subpaths are snapshotted, diffed and applied
  * the budget guard measures duplicated bytes, not tree size
  * large files are compared without hashing
  * fan-out is capped by the container memory limit (runtime-enforced)

Run:  python3 -m optimizations.merge_submit.test_merge_scope
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path


def _mk_runtime(tmp: Path):
    from nanoma.core import Runtime, RuntimeConfig
    workspace_root = tmp / "task_cwd" / ".nanoma-task-work"
    submit_path = tmp / "task_cwd"
    submit_path.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        max_agents=64, max_depth=4, default_model="m", allowed_models=["m"],
        workspace_root=workspace_root, workspace_extra_roots=[submit_path],
    )
    rt = Runtime(config=config)
    rt.start_agent = lambda child: None
    return rt, submit_path


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _clear_env():
    for k in ("NANOMA_MERGE_PATHS", "SFORGE_SUBMIT_PATHS", "NANOMA_MERGE_MAX_MB",
              "NANOMA_MERGE_MAX_FILES", "NANOMA_MAX_PARALLEL_CHILDREN",
              "NANOMA_CHILD_MEM_MB", "NANOMA_RUNTIME_RESERVE_MB"):
        os.environ.pop(k, None)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_scope_roots_parsing():
    _clear_env()
    with tempfile.TemporaryDirectory() as d:
        rt, _ = _mk_runtime(Path(d))
        assert rt._merge_scope_roots() == (), "no env -> whole tree"
        os.environ["SFORGE_SUBMIT_PATHS"] = "ann_benchmarks/algorithms/custom/ other/dir"
        got = [p.as_posix() for p in rt._merge_scope_roots()]
        assert got == ["ann_benchmarks/algorithms/custom", "other/dir"], got
        # NANOMA_MERGE_PATHS overrides, and traversal/absolute are rejected
        os.environ["NANOMA_MERGE_PATHS"] = "src, ../evil, /abs"
        got = [p.as_posix() for p in rt._merge_scope_roots()]
        assert got == ["src"], got
        # "." means submit-everything -> whole tree
        os.environ["NANOMA_MERGE_PATHS"] = "."
        assert rt._merge_scope_roots() == ()
    _clear_env()
    print("ok: scope roots parsing (env, override, sanitization, '.')")


def test_child_copy_is_complete_and_cheap():
    """The ann-benchmarks shape: huge data/ dir, tiny submission subpath.

    The child must receive the whole task directory or it cannot build, run or
    score anything, while the bulk payload has to stay free (hardlinked).
    """
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ["SFORGE_SUBMIT_PATHS"] = "pkg/custom/"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "pkg/custom/module.py", "impl\n")
        _write(submit_path, "run.py", "harness entrypoint\n")
        _write(submit_path, "requirements.txt", "numpy\n")
        _write(submit_path, "data/huge.bin", "x" * (2 * 1024 * 1024))
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        copy = Path(child._merge_copy)
        # complete: everything needed to build/run/evaluate is there
        assert (copy / "pkg/custom/module.py").is_file(), "scoped file present"
        assert (copy / "run.py").is_file(), "harness entrypoint present"
        assert (copy / "requirements.txt").is_file(), "deps present"
        assert (copy / "data/huge.bin").is_file(), "dataset present -> child can run"
        # cheap: the bulk payload is a hardlink, not a duplicate
        assert (copy / "data/huge.bin").stat().st_ino == (submit_path / "data/huge.bin").stat().st_ino, \
            "large file hardlinked"
        # isolated: small files are duplicated, so edits never leak
        assert (copy / "run.py").stat().st_ino != (submit_path / "run.py").stat().st_ino, \
            "small file duplicated"
        (copy / "run.py").write_text("child scratch\n")
        assert (submit_path / "run.py").read_text() == "harness entrypoint\n", \
            "child edit did not leak into the shared task directory"
        # the runtime's own workspace must not be cloned into the copy
        assert not (copy / ".nanoma-task-work").exists(), "no recursive clone"
        # merge accounting stays scoped even though the copy is complete
        assert not (rt._merge_baseline_path / "data").exists(), "baseline excludes bulk data"
        assert set(rt._merge_rel_files(submit_path)) == {"pkg/custom/module.py"}
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: child copy is complete + runnable, bulk hardlinked, edits isolated")


def test_large_submitted_file_is_never_hardlinked():
    """An in-place write through a hardlink bypasses diff, gate and rollback."""
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        original = "A" * (2 * 1024 * 1024)
        _write(submit_path, "solution.bin", original)
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        private = Path(child._merge_copy) / "solution.bin"
        shared = submit_path / "solution.bin"
        assert private.stat().st_ino != shared.stat().st_ino, \
            "a large file in the whole-tree submission scope must be duplicated"
        with private.open("r+b") as handle:
            handle.write(b"child")
        assert shared.read_text() == original, \
            "an in-place child edit must not leak into the live submission"
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: large submitted files are private even when edited in place")


def test_child_local_dirs_do_not_collide():
    """Directories are real, so per-child build/benchmark output stays local."""
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ["SFORGE_SUBMIT_PATHS"] = "pkg/custom/"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "pkg/custom/module.py", "impl\n")
        _write(submit_path, "results/.keep", "")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        c1 = rt.create_agent(task="c1", parent=parent.id, depth=1)
        c2 = rt.create_agent(task="c2", parent=parent.id, depth=1)
        _write(Path(c1._merge_copy), "results/run.json", "c1\n")
        _write(Path(c2._merge_copy), "results/run.json", "c2\n")
        assert (Path(c1._merge_copy) / "results/run.json").read_text() == "c1\n"
        assert (Path(c2._merge_copy) / "results/run.json").read_text() == "c2\n"
        assert not (submit_path / "results/run.json").exists(), "output stayed local"
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: sibling children write benchmark output without colliding")


def test_scoped_promote_still_works():
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ["SFORGE_SUBMIT_PATHS"] = "pkg/custom/"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        submits = []

        async def fake_submit(args, agent, runtime):
            submits.append(agent.id)
            return {"score": 0.9}

        rt, submit_path = _mk_runtime(tmp)
        rt.config.extra_tools = {"submit": {"handler": fake_submit, "is_meta": True, "schema": {}}}
        _write(submit_path, "pkg/custom/module.py", "v0\n")
        # promotion is gated by the agents' own verification, so it must never
        # spend an official submission on its own
        _write(submit_path, "run.py", "harness\n")
        _write(submit_path, "data/huge.bin", "x" * 100_000)
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        _write(Path(child._merge_copy), "pkg/custom/module.py", "v1\n")
        # local scratch outside the submission scope must never be promoted
        _write(Path(child._merge_copy), "run.py", "child scratch\n")
        child.status = "done"
        _run(rt._merge_promote_child(child))
        assert (submit_path / "pkg/custom/module.py").read_text() == "v1\n", "scoped promote landed"
        assert (submit_path / "run.py").read_text() == "harness\n", "out-of-scope edit not promoted"
        assert (submit_path / "data/huge.bin").is_file(), "bulk data untouched"
        assert submits == [], "no official submission spent by a promote"
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: promote lands scoped changes only, without spending a submission")


def test_rollback_never_wipes_the_task_directory():
    """A rollback must overlay the snapshot, not clear the destination first.

    The submission path holds the harness, the datasets and — because EdgeBench
    puts the workspace at task_cwd/.nanoma-task-work — every child's working
    copy. Clearing it to restore a snapshot would delete the run itself. This
    stayed latent only because no promote ever reverted while the gate was
    inert; it would have detonated the moment the gate started working.
    """
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "sol.txt", "good\n")
        _write(submit_path, "run.py", "harness\n")
        _write(submit_path, "data/big.bin", "x" * 1000)
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        snapshot = rt._merge_root_dir() / "snap"
        assert rt._merge_copy_tree(submit_path, snapshot), "snapshot taken"

        _write(submit_path, "sol.txt", "damaged\n")
        rt._merge_restore_snapshot(snapshot, submit_path)

        assert (submit_path / "sol.txt").read_text() == "good\n", "snapshot restored"
        assert (submit_path / "run.py").is_file(), "harness survived the rollback"
        assert (submit_path / "data/big.bin").is_file(), "dataset survived"
        assert Path(child._merge_copy).is_dir(), "the child's working copy survived"
        # and the guard refuses the destructive direction outright
        assert rt._merge_copy_tree(snapshot, submit_path) is False, \
            "clearing the live submission path is refused"
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: a rollback overlays the snapshot and never clears the task directory")


def test_budget_guard_counts_duplicated_bytes_only():
    """A tree that is large only because of linkable payloads stays eligible."""
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ["NANOMA_MERGE_MAX_MB"] = "0.05"  # 50KB of duplication allowed
    os.environ["SFORGE_SUBMIT_PATHS"] = "src/"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "data/huge.bin", "x" * (4 * 1024 * 1024))  # hardlinked -> free
        _write(submit_path, "src/main.py", "code\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        assert child._merge_copy, "4MB of linkable payload must not trip a 50KB budget"
        assert rt._merge_disabled_reason is None
        assert (Path(child._merge_copy) / "data/huge.bin").is_file()
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: budget guard ignores hardlinkable payloads")


def test_budget_guard_disables_on_real_duplication():
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ["NANOMA_MERGE_MAX_MB"] = "0.05"  # 50KB
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        for i in range(20):  # 20 x 20KB small files -> 400KB duplicated per child
            _write(submit_path, f"src/f{i}.py", "y" * 20_000)
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        assert getattr(child, "_merge_copy", None) is None, "no copy when duplication too costly"
        assert rt._merge_disabled_reason and "too expensive" in rt._merge_disabled_reason
        assert rt._merge_active() is False, "merge stays disabled"
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: excessive duplication disables merge instead of copying gigabytes")


def _tc(name, args):
    from nanoma.core import ToolCall
    try:
        return ToolCall(id="t1", name=name, arguments=args)
    except TypeError:
        return ToolCall(name=name, arguments=args)


def test_child_filesystem_tools_are_redirected():
    """Isolation must not depend on the child obeying a prompt.

    The harness system prompt repeatedly names the shared task directory, and
    children follow it: on ann_vector_search_qps all four edited the shared dir
    while their copies stayed pristine. Filesystem tool arguments get rewritten.
    """
    _clear_env()
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"
    os.environ["SFORGE_SUBMIT_PATHS"] = "pkg/custom/"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, submit_path = _mk_runtime(tmp)
        _write(submit_path, "pkg/custom/module.py", "impl\n")
        _write(submit_path, "run.py", "harness\n")
        parent = rt.create_agent(task="root", parent=None, depth=0)
        child = rt.create_agent(task="c", parent=parent.id, depth=1)
        copy = str(child._merge_copy)
        shared = str(submit_path)

        call = _tc("shell", {"command": f"cd {shared} && python run.py"})
        rt._merge_redirect_tool_args(child, call)
        assert call.arguments["command"] == f"cd {copy} && python run.py", call.arguments

        call = _tc("ws_read_file", {"path": f"{shared}/pkg/custom/module.py"})
        rt._merge_redirect_tool_args(child, call)
        assert call.arguments["path"] == f"{copy}/pkg/custom/module.py"

        # the copy lives *inside* the task directory, so a naive replace would
        # corrupt paths that are already correct
        already = f"{copy}/run.py"
        call = _tc("shell", {"command": f"cat {already}"})
        rt._merge_redirect_tool_args(child, call)
        assert call.arguments["command"] == f"cat {already}", call.arguments

        # mixed references in one command: only the shared one moves
        call = _tc("shell", {"command": f"diff {shared}/run.py {copy}/run.py"})
        rt._merge_redirect_tool_args(child, call)
        assert call.arguments["command"] == f"diff {copy}/run.py {copy}/run.py"

        # nested structures are covered too
        call = _tc("ws_grep", {"paths": [f"{shared}/pkg", {"root": f"{shared}/x"}]})
        rt._merge_redirect_tool_args(child, call)
        assert call.arguments["paths"] == [f"{copy}/pkg", {"root": f"{copy}/x"}]

        # messages to other agents are left alone
        call = _tc("send", {"message": f"I edited {shared}/run.py"})
        rt._merge_redirect_tool_args(child, call)
        assert call.arguments["message"] == f"I edited {shared}/run.py"

        # and the parent, which legitimately owns the task directory, is exempt
        call = _tc("shell", {"command": f"cd {shared} && ls"})
        rt._merge_redirect_tool_args(parent, call)
        assert call.arguments["command"] == f"cd {shared} && ls"
    _clear_env()
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    print("ok: child filesystem tools are pinned to the private copy")


def test_large_file_compare_without_hashing():
    _clear_env()
    from nanoma.core import Runtime, _MERGE_HASH_MAX_BYTES
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        a = tmp / "a.bin"
        b = tmp / "b.bin"
        # same size, same mtime, different content: treated as equal past the cap
        big = _MERGE_HASH_MAX_BYTES + 1024
        a.write_bytes(b"a" * big)
        b.write_bytes(b"b" * big)
        sa = a.stat()
        os.utime(b, ns=(sa.st_atime_ns, sa.st_mtime_ns))  # copy2 preserves ns
        assert Runtime._merge_same_file(a, b) is True, "large files compared by size+mtime"
        # a rewrite in the same second is still detected (ns resolution)
        b.write_bytes(b"c" * big)
        os.utime(b, ns=(sa.st_atime_ns, sa.st_mtime_ns + 1))
        assert Runtime._merge_same_file(a, b) is False, "same-second rewrite detected"
        # small files still compared by content
        s1, s2 = tmp / "s1.txt", tmp / "s2.txt"
        s1.write_text("aaa")
        s2.write_text("bbb")
        os.utime(s2, (s1.stat().st_atime, s1.stat().st_mtime))
        assert Runtime._merge_same_file(s1, s2) is False, "small files compared by content"
    print("ok: large files skip hashing, small files hashed")


def test_memory_cap_limits_fanout():
    _clear_env()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, _ = _mk_runtime(tmp)
        os.environ["NANOMA_MAX_PARALLEL_CHILDREN"] = "2"
        assert rt._max_parallel_children() == 2
        parent = rt.create_agent(task="root", parent=None, depth=0)
        assert rt._spawn_memory_block_reason() is None, "no children yet -> allowed"
        c1 = rt.create_agent(task="c1", parent=parent.id, depth=1)
        assert rt._active_children_total() == 1
        assert rt._spawn_memory_block_reason() is None, "1 < cap 2 -> allowed"
        rt.create_agent(task="c2", parent=parent.id, depth=1)
        reason = rt._spawn_memory_block_reason()
        assert reason and "memory headroom" in reason, reason
        # finishing a child frees capacity again
        c1.status = "done"
        assert rt._spawn_memory_block_reason() is None, "capacity freed on completion"
    _clear_env()
    print("ok: memory cap bounds concurrent children and frees on completion")


def test_memory_cap_from_limit_math():
    _clear_env()
    with tempfile.TemporaryDirectory() as d:
        rt, _ = _mk_runtime(Path(d))
        os.environ["NANOMA_CHILD_MEM_MB"] = "1500"
        os.environ["NANOMA_RUNTIME_RESERVE_MB"] = "1500"
        rt._container_memory_limit_bytes = lambda: 8 * 1024 * 1024 * 1024  # 8g
        # (8192 - 1500) / 1500 = 4
        assert rt._max_parallel_children() == 4, rt._max_parallel_children()
        rt._container_memory_limit_bytes = lambda: None
        assert rt._max_parallel_children() is None, "no limit detected -> no cap"
    _clear_env()
    print("ok: cap derived from cgroup limit (8g -> 4 children)")


def test_spawn_blocked_when_at_cap():
    _clear_env()
    from nanoma.meta import meta_spawn
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rt, _ = _mk_runtime(tmp)
        os.environ["NANOMA_MAX_PARALLEL_CHILDREN"] = "1"
        parent = rt.create_agent(task="root", parent=None, depth=0)
        r1 = _run(meta_spawn({"task": "first"}, parent, rt))
        assert "agent_id" in r1, r1
        r2 = _run(meta_spawn({"task": "second"}, parent, rt))
        assert "error" in r2 and "memory headroom" in r2["error"], r2
    _clear_env()
    print("ok: meta_spawn blocked at cap with an actionable error")


def main():
    tests = [
        test_scope_roots_parsing,
        test_child_copy_is_complete_and_cheap,
        test_large_submitted_file_is_never_hardlinked,
        test_child_local_dirs_do_not_collide,
        test_scoped_promote_still_works,
        test_rollback_never_wipes_the_task_directory,
        test_budget_guard_counts_duplicated_bytes_only,
        test_budget_guard_disables_on_real_duplication,
        test_child_filesystem_tools_are_redirected,
        test_large_file_compare_without_hashing,
        test_memory_cap_limits_fanout,
        test_memory_cap_from_limit_math,
        test_spawn_blocked_when_at_cap,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} working-copy / merge-scope / memory-cap tests passed.")


if __name__ == "__main__":
    main()
