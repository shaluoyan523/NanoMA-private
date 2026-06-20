#!/usr/bin/env python3
"""Run DRACO tasks with NanoMA and write raw outputs.

This runner intentionally does not judge results. It produces one JSONL record
per task with the model output, runtime stats, and workspace/log locations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanoma import Runtime, RuntimeConfig  # noqa: E402


DEFAULT_DATASET = ROOT / "external" / "valyu-benchmarks" / "datasets" / "draco.jsonl"
SINGLE_AGENT_DISABLED_TOOLS = {
    "spawn",
    "send",
    "query",
    "wait",
    "kill",
    "transfer",
    "set_bio",
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    from datasets import load_dataset

    ds = load_dataset("perplexity-ai/draco", split="test")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in ds]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def select_items(items: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.task_id:
        wanted = set(args.task_id)
        return [(i, item) for i, item in enumerate(items) if item.get("id") in wanted]
    start = args.offset
    stop = len(items) if args.limit <= 0 else min(len(items), start + args.limit)
    return list(enumerate(items[start:stop], start=start))


def task_prompt(item: dict[str, Any], *, single_agent: bool = False) -> str:
    problem = item["problem"]
    rubric = item.get("answer") or ""
    rubric_note = ""
    if rubric:
        rubric_note = (
            "\n\nThe benchmark rubric is included only to define expected coverage "
            "and presentation quality. Do not treat it as a finished answer. Use it "
            "to decide what evidence and details your final report must cover.\n"
            f"Rubric JSON:\n{rubric}\n"
        )

    agent_rule = (
        "- Work alone as one agent. Do not delegate to helper agents or rely on multi-agent coordination."
        if single_agent
        else "- This is a multi-agent benchmark. Freely spawn as many subagents as the task needs; do not artificially cap the team size.\n- Coordinate through shared files and messages when using subagents."
    )

    return f"""You are running a DRACO deep-research benchmark task.

Problem:
{problem}
{rubric_note}
Requirements:
- Produce a high-quality research answer for the problem, with concise citations or source names/URLs where useful.
- Use web retrieval, shell commands, and any public sources needed to solve the task.
{agent_rule}
- Write the final answer to `draco_answer.md` in your workspace and submit it.
- Also copy or write the final answer to `$SHARED/draco_answer.md`.
- When finished, call set_status(status="done", result="wrote draco_answer.md").
"""


async def run_one(index: int, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    tid = item["id"]
    short = tid[:8]
    task_dir = args.output_dir / f"{index:03d}_{short}"
    workspace = task_dir / "workspace"
    logs = task_dir / "logs"
    workspace.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config = RuntimeConfig(
        budget=args.budget,
        time_limit=args.time_limit,
        max_turns=args.max_turns,
        max_agents=args.max_agents,
        max_depth=args.max_depth,
        max_concurrent_llm=args.max_concurrent_llm,
        max_total_tokens=args.max_total_tokens,
        tool_policy_soft_total_tokens=args.tool_policy_soft_total_tokens,
        disabled_tools=set(args.disable_tool) | (SINGLE_AGENT_DISABLED_TOOLS if args.single_agent else set()),
        default_model=args.model,
        workspace_root=workspace,
        log_dir=logs,
        shell_max_output=args.shell_max_output,
        file_read_max_chars=args.file_read_max_chars,
        blocked_shell_patterns=args.blocked_shell_pattern or [],
        tool_policy_mode=args.tool_policy_mode,
        tool_policy_prune_tools=args.tool_policy_prune,
        tool_policy_prune_min_tools=args.tool_policy_prune_min_tools,
        tool_policy_prune_pressure_start=args.tool_policy_prune_pressure_start,
        tool_policy_prune_pressure_end=args.tool_policy_prune_pressure_end,
        tool_policy_prune_shell_capabilities=args.tool_policy_prune_shell_capabilities,
        tool_policy_shell_capability_pressure_start=args.tool_policy_shell_capability_pressure_start,
        tool_policy_shell_capability_pressure_end=args.tool_policy_shell_capability_pressure_end,
        tool_policy_web_saturation_enabled=args.tool_policy_web_saturation_enabled,
        tool_policy_web_saturation_min_calls=args.tool_policy_web_saturation_min_calls,
        tool_policy_web_saturation_threshold=args.tool_policy_web_saturation_threshold,
        tool_policy_web_saturation_finalize_after_blocks=args.tool_policy_web_saturation_finalize_after_blocks,
    )
    rt = Runtime(config=config)

    start = time.time()
    error = None
    result = ""
    try:
        result = await rt.run(task_prompt(item, single_agent=args.single_agent), model=args.model)
    except Exception as exc:  # keep batch alive
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - start

    answer_path: Path | None = None
    for candidate in (
        workspace / "shared" / "draco_answer.md",
        workspace / "shared" / "draco_answer.md" / "draco_answer.md",
        workspace / "alpha" / "draco_answer.md",
    ):
        if candidate.is_file():
            answer_path = candidate
            break
    output = answer_path.read_text(encoding="utf-8", errors="replace") if answer_path else result

    return {
        "id": tid,
        "index": index,
        "domain": item.get("domain"),
        "problem": item.get("problem"),
        "answer": item.get("answer"),
        "output": output,
        "elapsed": elapsed,
        "provider": "nanoma",
        "model": args.model,
        "tool_policy_mode": args.tool_policy_mode,
        "workspace": str(workspace),
        "logs": str(logs),
        "answer_file": str(answer_path) if answer_path else None,
        "stats": rt.stats(),
        "error": error,
    }


async def main_async(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = load_dataset(args.dataset)
    selected = select_items(items, args)
    if not selected:
        raise SystemExit("No DRACO tasks selected.")

    selected_path = args.output_dir / "selected_tasks.jsonl"
    with selected_path.open("w", encoding="utf-8") as f:
        for index, item in selected:
            row = dict(item)
            row["_index"] = index
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    results_path = args.output_dir / "nanoma_results.jsonl"
    summary_path = args.output_dir / "nanoma_summary.json"
    completed = 0
    records = []

    print(f"NanoMA DRACO run: {len(selected)} tasks", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    for index, item in selected:
        print(f"=== NanoMA DRACO {completed + 1}/{len(selected)} index={index} id={item['id']} ===", flush=True)
        row = await run_one(index, item, args)
        records.append(row)
        completed += 1
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        status = "ERR" if row.get("error") else "OK"
        print(
            f"[{completed:02d}/{len(selected):02d}] {status} {row['domain']} "
            f"{row['elapsed']:.1f}s {len(row.get('output') or '')}ch",
            flush=True,
        )

    total_tokens = sum(int((r.get("stats") or {}).get("overview", {}).get("total_tokens", 0)) for r in records)
    total_cost = sum(float((r.get("stats") or {}).get("overview", {}).get("total_cost_usd", 0.0)) for r in records)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "provider": "nanoma",
        "model": args.model,
        "tool_policy_mode": args.tool_policy_mode,
        "tasks": len(records),
        "errors": sum(1 for r in records if r.get("error")),
        "total_elapsed": sum(float(r.get("elapsed") or 0.0) for r in records),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "results": str(results_path),
        "selected_tasks": str(selected_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--task-id", action="append")
    ap.add_argument("--model", default=os.environ.get("NANOMA_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--budget", type=float, default=100000.0)
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--max-agents", type=int, default=1_000_000)
    ap.add_argument("--max-depth", type=int, default=1_000_000)
    ap.add_argument("--max-concurrent-llm", type=int, default=50)
    ap.add_argument("--max-total-tokens", type=int, default=0)
    ap.add_argument("--tool-policy-soft-total-tokens", type=int, default=0)
    ap.add_argument("--disable-tool", action="append", default=[])
    ap.add_argument("--single-agent", action="store_true")
    ap.add_argument("--shell-max-output", type=int, default=20000)
    ap.add_argument("--file-read-max-chars", type=int, default=100000)
    ap.add_argument("--blocked-shell-pattern", action="append")
    ap.add_argument("--tool-policy-mode", choices=["off", "adaptive", "enforce"], default="adaptive")
    ap.add_argument("--tool-policy-prune", action="store_true")
    ap.add_argument("--tool-policy-prune-min-tools", type=int, default=6)
    ap.add_argument("--tool-policy-prune-pressure-start", type=float, default=0.65)
    ap.add_argument("--tool-policy-prune-pressure-end", type=float, default=0.95)
    ap.add_argument("--tool-policy-prune-shell-capabilities", action="store_true")
    ap.add_argument("--tool-policy-shell-capability-pressure-start", type=float, default=0.55)
    ap.add_argument("--tool-policy-shell-capability-pressure-end", type=float, default=0.95)
    ap.add_argument("--tool-policy-web-saturation-enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tool-policy-web-saturation-min-calls", type=int, default=10)
    ap.add_argument("--tool-policy-web-saturation-threshold", type=float, default=0.85)
    ap.add_argument("--tool-policy-web-saturation-finalize-after-blocks", type=int, default=2)
    args = ap.parse_args()
    if args.single_agent:
        args.max_agents = 1
        args.max_depth = 0
        args.max_concurrent_llm = min(args.max_concurrent_llm, 1)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
