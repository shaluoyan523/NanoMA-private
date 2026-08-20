"""Standalone verification for the todo_tools add-on.

Run from the repo root:
    python -m optimizations.todo_tools.test_todo_tools

Checks:
  1. Handlers behave correctly against a minimal fake agent/runtime.
  2. The tools are registered into NanoMA's live tool library (`_all_tools`).
"""

from __future__ import annotations

import asyncio
import sys

from optimizations.todo_tools import (
    TODO_TOOLS,
    meta_task_create,
    meta_task_list,
    meta_task_spawn,
    meta_task_update,
    render_todo_reminder,
    spawn_window_open,
)


class _FakeAgent:
    def __init__(self) -> None:
        self.id = "test-agent"
        self._turns = 0


class _FakeRuntime:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def _emit(self, agent_id: str, event: str, payload: dict) -> None:
        self.events.append((agent_id, event, payload))


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


async def _run_handler_tests() -> None:
    print("[1] handler behavior")
    agent, rt = _FakeAgent(), _FakeRuntime()

    # create
    r1 = await meta_task_create({"subject": "Explore codebase", "description": "read core files"}, agent, rt)
    _check(r1["task_id"] == 1 and r1["status"] == "pending", "task_create returns id=1 pending")

    r2 = await meta_task_create({"subject": "Implement fix"}, agent, rt)
    _check(r2["task_id"] == 2 and r2["total"] == 2, "second task gets id=2, total=2")

    # missing subject
    err = await meta_task_create({"description": "no subject"}, agent, rt)
    _check("error" in err, "task_create without subject errors")

    # update status
    up = await meta_task_update({"task_id": 1, "status": "in_progress"}, agent, rt)
    _check(up["status"] == "in_progress" and "status" in up["changed"], "task_update sets in_progress")

    # invalid status
    bad = await meta_task_update({"task_id": 1, "status": "bogus"}, agent, rt)
    _check("error" in bad, "invalid status rejected")

    # unknown id
    missing = await meta_task_update({"task_id": 999, "status": "completed"}, agent, rt)
    _check("error" in missing, "unknown task_id errors")

    # complete task 1
    await meta_task_update({"task_id": 1, "status": "completed"}, agent, rt)

    # list + counts + filter
    lst = await meta_task_list({}, agent, rt)
    _check(lst["total"] == 2, "task_list total=2")
    _check(lst["counts"]["completed"] == 1 and lst["counts"]["pending"] == 1, "counts correct")

    filtered = await meta_task_list({"status": "pending"}, agent, rt)
    _check(len(filtered["tasks"]) == 1 and filtered["tasks"][0]["id"] == 2, "status filter works")

    # events emitted
    kinds = [e[1] for e in rt.events]
    _check("task_create" in kinds and "task_update" in kinds, "viewer events emitted")

    # state isolation across agents
    agent2 = _FakeAgent()
    lst2 = await meta_task_list({}, agent2, rt)
    _check(lst2["total"] == 0, "second agent starts with empty list")


async def _run_reminder_tests() -> None:
    print("[2] per-turn reminder / nudge (state re-injection)")
    agent, rt = _FakeAgent(), _FakeRuntime()

    # empty + early turn -> silent
    agent._turns = 1
    _check(render_todo_reminder(agent) is None, "empty list, early turn -> no reminder")

    # empty + past threshold -> one-time hint, then silent
    agent._turns = 5
    hint = render_todo_reminder(agent)
    _check(hint is not None and "task-list-hint" in hint, "empty list after threshold -> one-time hint")
    _check(render_todo_reminder(agent) is None, "empty-list hint fires only once")

    # create tasks -> list is re-injected with progress
    agent2, rt2 = _FakeAgent(), _FakeRuntime()
    agent2._turns = 1
    await meta_task_create({"subject": "Explore codebase"}, agent2, rt2)
    await meta_task_create({"subject": "Implement fix"}, agent2, rt2)
    agent2._turns = 2
    rem = render_todo_reminder(agent2)
    _check(rem is not None and "<task-list>" in rem, "active tasks -> list re-injected")
    _check("#1 Explore codebase" in rem and "0 completed / 2 total" in rem, "reminder shows tasks + progress")
    _check("REMINDER" not in rem, "no stale nudge immediately after mutation")

    # go idle several turns -> stale nudge appears
    agent2._turns = 6
    rem2 = render_todo_reminder(agent2)
    _check(rem2 is not None and "REMINDER" in rem2, "stale nudge after idle turns")

    # complete all -> reminder goes quiet
    await meta_task_update({"task_id": 1, "status": "completed"}, agent2, rt2)
    await meta_task_update({"task_id": 2, "status": "cancelled"}, agent2, rt2)
    agent2._turns = 20
    _check(render_todo_reminder(agent2) is None, "no open tasks -> reminder silent")


def _run_registration_test() -> None:
    print("[3] registration in NanoMA tool library")
    from nanoma.meta import META_TOOLS
    for name in TODO_TOOLS:
        _check(name in META_TOOLS, f"{name} present in META_TOOLS")

    from nanoma.core import Runtime, RuntimeConfig, Agent
    rt = Runtime(RuntimeConfig())
    tools = rt._all_tools()
    for name in TODO_TOOLS:
        _check(name in tools, f"{name} present in Runtime._all_tools()")
        _check(tools[name].get("is_meta") is True, f"{name} is_meta=True")
        schema_name = tools[name]["schema"]["function"]["name"]
        _check(schema_name == name, f"{name} schema name matches")

    # runtime injection: no todos -> history returned unchanged (same object)
    agent = Agent(id="a1", task="t", model="m")
    agent.history = [{"role": "user", "content": "task"}]
    out = rt._history_with_todo_reminder(agent)
    _check(out is agent.history, "no todos -> history passed through unchanged")

    # with an open task -> ephemeral reminder appended, history NOT mutated
    agent._todos = [{"id": 1, "subject": "Do X", "description": "", "activeForm": "", "status": "pending"}]
    agent._todo_last_mutation_turn = 1
    agent._turns = 2
    out2 = rt._history_with_todo_reminder(agent)
    _check(len(out2) == len(agent.history) + 1, "reminder appended to per-call history")
    _check(out2[-1]["role"] == "user" and "<task-list>" in out2[-1]["content"], "reminder is trailing user msg")
    _check(len(agent.history) == 1, "persisted history NOT mutated")


async def _run_task_spawn_tests() -> None:
    print("[4] task_spawn: spawn folded into the todolist")
    from nanoma.core import Runtime, RuntimeConfig

    rt = Runtime(RuntimeConfig(max_agents=4, max_depth=1, default_model="m", allowed_models=["m"]))
    rt.start_agent = lambda child: None  # don't actually run children in the test
    root = rt.create_agent(task="root", model="m", parent=None, depth=0)
    root._turns = 1

    await meta_task_create({"subject": "Independent subtask A", "description": "do A fully"}, root, rt)
    _check(spawn_window_open(root) is True, "pending task -> spawn window open")

    res = await meta_task_spawn({"task_id": 1}, root, rt)
    _check(res.get("delegated_to") and res.get("status") == "in_progress", "task_spawn delegates + sets in_progress")
    task = root._todos[0]
    _check(task["status"] == "in_progress" and task.get("child_id"), "task bound to child_id")
    _check(task["child_id"] in rt.agents, "child agent registered in runtime")
    _check(spawn_window_open(root) is False, "no more pending tasks -> window closes")

    err = await meta_task_spawn({"task_id": 1}, root, rt)
    _check("error" in err, "cannot re-delegate a non-pending task")

    # reflection: child finishing auto-completes the delegated task
    rt.agents[task["child_id"]].status = "done"
    root._turns = 2
    rem = render_todo_reminder(root, rt)
    _check(root._todos[0]["status"] == "completed", "delegated task auto-completed when child done")
    _check(rem is None, "reminder goes quiet once all tasks resolved")


def _run_gate_test() -> None:
    print("[5] runtime spawn/todolist gate")
    import os
    from nanoma.core import Runtime, RuntimeConfig

    os.environ["NANOMA_SPAWN_TODOLIST_GATE"] = "1"
    try:
        rt = Runtime(RuntimeConfig(max_agents=4, max_depth=1, default_model="m", allowed_models=["m"]))
        root = rt.create_agent(task="root", model="m", parent=None, depth=0)
        all_tools = rt._all_tools()
        subset = {k: all_tools[k] for k in ("spawn", "spawn_many", "task_spawn", "task_create", "shell") if k in all_tools}

        class _P:
            removed_tools: list = []
            reason = ""
            scoped_tools: list = []

        gated, _ = rt._apply_spawn_todolist_gate(root, dict(subset), _P())
        _check("spawn" not in gated and "spawn_many" not in gated, "raw spawn/spawn_many removed under gate")
        _check("task_spawn" not in gated, "task_spawn hidden when no pending tasks")

        root._todos = [{"id": 1, "subject": "X", "description": "", "activeForm": "", "status": "pending"}]
        gated2, _ = rt._apply_spawn_todolist_gate(root, dict(subset), _P())
        _check("task_spawn" in gated2, "task_spawn offered when a pending task exists")
        _check("spawn" not in gated2, "raw spawn still removed even in the window")

        # judge mode: the agent never gets any spawn tool (runtime/Opus decides)
        os.environ.pop("NANOMA_SPAWN_TODOLIST_GATE", None)
        os.environ["NANOMA_SPAWN_TODOLIST_JUDGE"] = "1"
        root._todos = [{"id": 1, "subject": "X", "description": "", "activeForm": "", "status": "pending"}]
        judged, _ = rt._apply_spawn_todolist_gate(root, dict(subset), _P())
        _check("spawn" not in judged and "spawn_many" not in judged, "judge: raw spawn removed")
        _check("task_spawn" not in judged, "judge: task_spawn removed even with a pending task")
        _check("shell" in judged and "task_create" in judged, "judge: normal work tools untouched")
        os.environ.pop("NANOMA_SPAWN_TODOLIST_JUDGE", None)
    finally:
        os.environ.pop("NANOMA_SPAWN_TODOLIST_GATE", None)
        os.environ.pop("NANOMA_SPAWN_TODOLIST_JUDGE", None)


async def _run_spawn_judge_tests() -> None:
    print("[6] Opus-judged spawn at the planning moment (decide-before-create)")
    import os
    from nanoma.core import Runtime, RuntimeConfig
    from optimizations.todo_tools.todo_tools import meta_task_create

    os.environ["NANOMA_SPAWN_TODOLIST_JUDGE"] = "1"
    try:
        # --- spawn=true: judge splits into 2 subagents; todolist is SUPPRESSED ---
        rt = Runtime(RuntimeConfig(max_agents=4, max_depth=1, default_model="m", allowed_models=["m"]))
        rt.start_agent = lambda child: None
        plan = ('{"spawn": true, "reasoning": "cross-check", "subagents": ['
                '{"subject": "A", "role": "solver", "task": "solve fully"},'
                '{"subject": "B", "role": "verifier", "task": "verify via a different method"}]}')

        class _Resp:
            content = plan

        async def _fake_llm(messages, model, tools=None, **kw):
            _Resp.seen_model = model
            _Resp.seen_user = messages[-1]["content"]  # judge input
            return _Resp()

        rt.llm_call = _fake_llm
        root = rt.create_agent(task="root task", model="m", parent=None, depth=0)
        # Parent working context that must be forwarded to the judge (not a checklist).
        root.history = [
            {"role": "user", "content": "root task"},
            {"role": "assistant", "content": "explored the codebase; found 3 bugs"},
        ]

        before = len(rt.agents)
        res = await meta_task_create({"subject": "plan step"}, root, rt)
        _check(res.get("delegated") is True, "spawn=true -> task_create returns delegated (todolist suppressed)")
        _check(len(getattr(root, "_todos", [])) == 0, "no local todo created on spawn (decide-before-create)")
        _check(len(rt.agents) - before == 2, "judge spawn=true -> 2 children created")
        _check(getattr(_Resp, "seen_model", None) == "claude-opus-4-8", "judge used the Opus model by default (hyphen slug)")
        _check("explored the codebase" in getattr(_Resp, "seen_user", ""),
               "judge is fed the parent's full working context (not just a checklist)")
        parsed = rt._parse_spawn_judge(plan)
        _check([s["role"] for s in parsed["subagents"]] == ["solver", "verifier"],
               "judge parses solver/verifier roles (parallel reasoning + verification)")

        # sibling task_create in the SAME turn is also suppressed
        res2 = await meta_task_create({"subject": "plan step 2"}, root, rt)
        _check(res2.get("delegated") is True and len(getattr(root, "_todos", [])) == 0,
               "sibling task_create in same turn also suppressed")

        # a later planning moment does NOT re-fan-out while children are still active;
        # with children in flight the judge declines and the todolist IS created.
        root._turns = 5
        n_active = len(rt.agents)
        res3 = await meta_task_create({"subject": "later plan"}, root, rt)
        _check(len(rt.agents) == n_active, "no re-spawn while children still active")
        _check(len(getattr(root, "_todos", [])) == 1 and res3.get("task_id"),
               "children active -> judge declines -> todolist IS created")

        # --- spawn=false: judge declines, todolist IS created ---
        rt2 = Runtime(RuntimeConfig(max_agents=4, max_depth=1, default_model="m", allowed_models=["m"]))
        rt2.start_agent = lambda child: None

        class _Resp2:
            content = '{"spawn": false, "reasoning": "single indivisible step", "subagents": []}'

        async def _fake_llm2(messages, model, tools=None, **kw):
            return _Resp2()

        rt2.llm_call = _fake_llm2
        root2 = rt2.create_agent(task="seq task", model="m", parent=None, depth=0)
        root2.history = [{"role": "user", "content": "seq task"}]
        n = len(rt2.agents)
        res_f = await meta_task_create({"subject": "step 1"}, root2, rt2)
        _check(len(rt2.agents) == n, "judge spawn=false -> no children")
        _check(res_f.get("task_id") and root2._todos[0]["status"] == "pending",
               "declined -> todolist created, task pending (self-execute)")

        # --- judge failure degrades to no-spawn + todolist still created ---
        rt3 = Runtime(RuntimeConfig(max_agents=4, max_depth=1, default_model="m", allowed_models=["m"]))
        rt3.start_agent = lambda child: None

        async def _boom(messages, model, tools=None, **kw):
            raise RuntimeError("opus unreachable")

        rt3.llm_call = _boom
        root3 = rt3.create_agent(task="t", model="m", parent=None, depth=0)
        root3.history = [{"role": "user", "content": "t"}]
        m0 = len(rt3.agents)
        res_b = await meta_task_create({"subject": "s"}, root3, rt3)  # must not raise
        _check(len(rt3.agents) == m0, "judge error -> graceful no-spawn")
        _check(res_b.get("task_id"), "judge error -> todolist still created")
    finally:
        os.environ.pop("NANOMA_SPAWN_TODOLIST_JUDGE", None)


def main() -> int:
    try:
        asyncio.run(_run_handler_tests())
        asyncio.run(_run_reminder_tests())
        asyncio.run(_run_task_spawn_tests())
        asyncio.run(_run_spawn_judge_tests())
        _run_gate_test()
        _run_registration_test()
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
