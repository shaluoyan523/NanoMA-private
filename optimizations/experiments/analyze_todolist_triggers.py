#!/usr/bin/env python3
"""Summarize todolist (task_create/update/list) triggering across a GAIA run.

Reads every <log_root>/<task_id>/events.jsonl produced by the NanoMA GAIA
harness and reports, per task and in aggregate:
  - total agent turns (llm_done events)
  - turn index of the FIRST task_create (when the todolist first "kicks in")
  - counts of task_create / task_update / task_list tool calls
  - number of ephemeral todo_reminder_injected re-injections

Usage:
    python3 optimizations/experiments/analyze_todolist_triggers.py <base_run_dir>
    # base_run_dir contains logs/<task_id>/events.jsonl and nanoma_l2/results.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _iter_events(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _tool_calls(ev: dict) -> list[str]:
    data = ev.get("data", {}) or {}
    tc = data.get("tool_calls")
    if isinstance(tc, list):
        return [str(x) for x in tc]
    return []


def analyze_task(events_path: Path) -> dict:
    turn = 0
    first_create_turn = None
    first_spawn_turn = None
    counts = {"task_create": 0, "task_update": 0, "task_list": 0}
    reminders = 0
    children = 0
    create_turns: list[int] = []
    spawn_turns: list[int] = []
    status = None
    for ev in _iter_events(events_path):
        name = ev.get("type") or ev.get("event")
        if name == "llm_done":
            turn += 1
            for call in _tool_calls(ev):
                if call in counts:
                    counts[call] += 1
                    if call == "task_create":
                        create_turns.append(turn)
                        if first_create_turn is None:
                            first_create_turn = turn
        elif name == "todo_reminder_injected":
            reminders += 1
        elif name == "spawn":
            # Same-model judge decisions create children through internal
            # meta_spawn, which emits the durable topology event.
            children += 1
            spawn_turns.append(turn)
            if first_spawn_turn is None:
                first_spawn_turn = turn
        elif name in ("done", "failed"):
            status = (ev.get("data", {}) or {}).get("status", name)
    return {
        "turns": turn,
        "first_create_turn": first_create_turn,
        "first_spawn_turn": first_spawn_turn,
        "create_turns": create_turns,
        "spawn_turns": spawn_turns,
        "children": children,
        **counts,
        "reminders": reminders,
        "status": status,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    base = Path(sys.argv[1])
    log_root = base / "logs"
    if not log_root.exists():
        log_root = base  # allow passing logs dir directly
    task_dirs = sorted(p for p in log_root.iterdir() if (p / "events.jsonl").exists())
    if not task_dirs:
        print(f"No events.jsonl found under {log_root}")
        return 1

    results = {}
    res_path = base / "nanoma_l2" / "results.jsonl"
    if res_path.exists():
        for line in open(res_path):
            try:
                r = json.loads(line)
                results[r.get("task_id")] = r
            except json.JSONDecodeError:
                pass

    header = (f"{'task_id':38} {'turns':>5} {'1stTC':>5} {'1stSP':>5} "
              f"{'TC':>3} {'TU':>3} {'kids':>4} {'remind':>6} {'pass':>4}")
    print(header)
    print("-" * len(header))
    agg = {"turns": 0, "task_create": 0, "task_update": 0, "task_list": 0,
           "children": 0, "reminders": 0}
    used_todolist = 0
    used_spawn = 0
    passed = 0
    for d in task_dirs:
        tid = d.name
        a = analyze_task(d / "events.jsonl")
        r = results.get(tid, {})
        p = r.get("passed")
        ptxt = "-" if p is None else ("Y" if p else "N")
        if p:
            passed += 1
        first = a["first_create_turn"] if a["first_create_turn"] is not None else "-"
        firstsp = a["first_spawn_turn"] if a["first_spawn_turn"] is not None else "-"
        print(f"{tid:38} {a['turns']:>5} {str(first):>5} {str(firstsp):>5} {a['task_create']:>3} "
              f"{a['task_update']:>3} {a['children']:>4} {a['reminders']:>6} {ptxt:>4}")
        for k in agg:
            agg[k] += a[k]
        if a["task_create"] > 0:
            used_todolist += 1
        if a["children"] > 0:
            used_spawn += 1

    n = len(task_dirs)
    print("-" * len(header))
    print(f"tasks={n} | used_todolist={used_todolist} ({100*used_todolist/n:.0f}%) | "
          f"used_judge_spawn={used_spawn} ({100*used_spawn/n:.0f}%) | passed={passed}")
    print(f"totals: turns={agg['turns']} task_create={agg['task_create']} "
          f"task_update={agg['task_update']} "
          f"children={agg['children']} reminders={agg['reminders']}")
    if agg["turns"]:
        print(f"task_create per 100 turns = {100*agg['task_create']/agg['turns']:.1f} | "
              f"judge-created children per 100 turns = {100*agg['children']/agg['turns']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
