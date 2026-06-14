#!/usr/bin/env python3
"""Run GAIA Level 2 data with the NanoMA runtime from this checkout.

The GAIA benchmark files live in /data/workspace/NanoMA/benchmarks/gaia, but
this wrapper intentionally imports nanoma from the current repository first.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


CURRENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAIA_RUNNER = Path("/data/workspace/NanoMA/benchmarks/gaia/run_nanoma_gaia_l2.py")


def load_external_gaia(path: Path):
    spec = importlib.util.spec_from_file_location("external_gaia_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load GAIA runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefer_current_nanoma() -> None:
    current = str(CURRENT_ROOT)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != CURRENT_ROOT]
    sys.path.insert(0, current)


def prompt_for(gaia: Any, record: dict[str, Any], attachment: Path | None, task_id: str) -> str:
    attach_text = "No attachment is provided for this task."
    if attachment:
        attach_text = (
            f"Attachment path: {attachment}\n"
            "You may inspect it with shell tools. Use Python or command-line tools when useful."
        )
    return f"""You are evaluating one GAIA validation Level 2 task under a BenchAgent-style protocol.

Question:
{gaia.question(record)}

{attach_text}

Rules:
- Solve the task using available tools. Search the web through shell commands if needed.
- This is a research-style task with separable evidence trails. Prefer spawning helper agents for independent investigation, source verification, and answer checking when useful.
- If you spawn helpers, give each helper one concrete lane and require it to write a short evidence report under `shared/gaia_l2/{task_id}/` before stopping. Suggested lanes:
  - identify the June 2022 AI regulation paper and its three-axis figure words;
  - identify Physics and Society articles submitted on August 11, 2016 and society-type descriptors;
  - independently verify the overlap between the figure words and the 2016 article wording.
- Helper reports should include source URLs/IDs, concise evidence, candidate answer, confidence, and tags such as benchmark:gaia, task:{task_id}, role:evidence.
- Root should query helper state/memory or inspect `shared/gaia_l2/{task_id}/` before finalizing.
- When ready to submit the benchmark answer, call `submit_answer(answer="<answer-only string>")`.
  Do not write `shared/answer.json` directly; the submit_answer tool writes that protocol file.

Final answer requirements:
- Return only the concise answer expected by GAIA.
- If the answer is a list, preserve the order and use commas.
- If the answer is numeric, omit extra units unless the question requires them.
"""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "row_index",
        "passed",
        "answer",
        "gold",
        "elapsed_s",
        "tokens",
        "cost_usd",
        "agents",
        "peak_concurrent",
        "workspace",
        "logs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = row.get("stats", {})
            overview = stats.get("overview", {})
            agents = stats.get("agents", {})
            writer.writerow({
                "task_id": row.get("task_id"),
                "row_index": row.get("row_index"),
                "passed": row.get("passed"),
                "answer": row.get("answer"),
                "gold": row.get("gold"),
                "elapsed_s": row.get("elapsed_s"),
                "tokens": overview.get("total_tokens"),
                "cost_usd": overview.get("total_cost_usd"),
                "agents": agents.get("total_spawned"),
                "peak_concurrent": agents.get("peak_concurrent"),
                "workspace": row.get("workspace"),
                "logs": row.get("logs"),
            })


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    total_tokens = sum(row.get("stats", {}).get("overview", {}).get("total_tokens", 0) for row in rows)
    total_cost = sum(row.get("stats", {}).get("overview", {}).get("total_cost_usd", 0.0) for row in rows)
    return {
        "benchmark": "GAIA validation Level 2",
        "runtime_root": str(CURRENT_ROOT),
        "completed_tasks": total,
        "passed": passed,
        "pass_at_1": (passed / total) if total else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens": int(total_tokens / total) if total else 0,
        "total_cost_usd": round(total_cost, 4),
        "model": args.model,
        "runtime": {
            "max_agents": args.max_agents,
            "max_depth": args.max_depth,
            "max_concurrent_llm": args.max_concurrent_llm,
            "max_turns": args.max_turns,
            "time_limit": args.time_limit,
            "orchestration_preference": args.orchestration_preference,
            "child_orchestration_preference": args.child_orchestration_preference,
            "shell_mode": args.shell_mode,
        },
    }


async def run_one(args: argparse.Namespace, gaia: Any, record: dict[str, Any], index: int, attachment: Path | None) -> dict[str, Any]:
    prefer_current_nanoma()
    from nanoma.core import Runtime, RuntimeConfig
    from nanoma.env import load_dotenv

    load_dotenv()
    tid = gaia.task_id(record, index)
    workspace = args.workspace_root / tid
    log_dir = args.log_root / tid
    if args.clean:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)

    config = RuntimeConfig(
        max_agents=args.max_agents,
        max_depth=args.max_depth,
        max_concurrent_llm=args.max_concurrent_llm,
        budget=args.budget,
        time_limit=args.time_limit,
        max_turns=args.max_turns,
        default_model=args.model,
        workspace_root=workspace,
        log_dir=log_dir,
        shell_max_output=args.shell_max_output,
        file_read_max_chars=args.file_read_max_chars,
        shell_mode=args.shell_mode,
        sandbox_backend=args.sandbox_backend,
        sandbox_network=args.sandbox_network,
        orchestration_preference=args.orchestration_preference,
        child_orchestration_preference=args.child_orchestration_preference,
        spawn_before_turn=args.spawn_before_turn,
        max_solo_tool_calls_before_spawn=args.max_solo_tool_calls_before_spawn,
        min_spawnable_workstreams=args.min_spawnable_workstreams,
        loop_action_policy=args.loop_action_policy,
    )

    def on_event(e: dict[str, Any]) -> None:
        if args.verbose:
            print(json.dumps(e, ensure_ascii=False), flush=True)
            return
        if e["event"] in {
            "agent_new",
            "spawn",
            "orchestration_nudge",
            "create_action_miss",
            "done",
            "failed",
            "llm_done",
        }:
            data = e.get("data", {})
            compact = {
                key: data[key]
                for key in (
                    "task",
                    "child",
                    "preference",
                    "turns",
                    "tool_calls",
                    "loop_action",
                    "tool_scope",
                    "recommended_tool_choice",
                    "has_content",
                    "status",
                    "result",
                )
                if key in data
            }
            print(f"[{tid}][{e['event']}] {e['agent']} {compact}", flush=True)

    runtime = Runtime(config=config, on_event=on_event)
    started = time.time()
    result = await runtime.run(prompt_for(gaia, record, attachment, tid), model=args.model)
    answer, answer_path, answer_json = gaia.read_answer_json(workspace)
    if answer is None:
        answer = gaia.extract_answer(result)
    gold = gaia.final_answer(record)
    return {
        "task_id": tid,
        "row_index": index,
        "question": gaia.question(record),
        "file_name": gaia.file_name(record),
        "gold": gold,
        "answer": answer,
        "passed": gaia.gaia_score(answer, gold),
        "raw_result": result,
        "answer_path": str(answer_path) if answer_path else None,
        "answer_json": answer_json,
        "stats": runtime.stats(),
        "elapsed_s": round(time.time() - started, 3),
        "workspace": str(workspace),
        "logs": str(log_dir),
    }


async def run_selected(args: argparse.Namespace, gaia: Any, selected: list[tuple[int, dict[str, Any]]], token: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, (index, record) in enumerate(selected, start=1):
        tid = gaia.task_id(record, index)
        print(f"=== GAIA-L2 {ordinal}/{len(selected)} task_id={tid} ===", flush=True)
        attachment = gaia.ensure_attachment(
            record,
            args.data_dir,
            token=token,
            force=args.force_download,
            snapshot_dir=args.snapshot_dir,
        )
        try:
            row = await run_one(args, gaia, record, index, attachment)
        except Exception as exc:
            row = {
                "task_id": tid,
                "row_index": index,
                "question": gaia.question(record),
                "file_name": gaia.file_name(record),
                "gold": gaia.final_answer(record),
                "answer": "",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{tid}] ERROR {row['error']}", flush=True)
        rows.append(row)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.results_dir / "results.jsonl", rows)
        write_csv(args.results_dir / "results.csv", rows)
        (args.results_dir / "summary.json").write_text(
            json.dumps(summarize(rows, args), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaia-runner", type=Path, default=DEFAULT_GAIA_RUNNER)
    parser.add_argument("--model", default=os.environ.get("NANOMA_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--data-dir", type=Path, default=Path("/data/workspace/NanoMA/benchmarks/gaia/data"))
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--download-method", choices=["auto", "api", "repo"], default="auto")
    parser.add_argument("--results-dir", type=Path, default=CURRENT_ROOT / "runs" / "gaia_l2_current")
    parser.add_argument("--workspace-root", type=Path, default=CURRENT_ROOT / "runs" / "gaia_l2_workspace")
    parser.add_argument("--log-root", type=Path, default=CURRENT_ROOT / "runs" / "gaia_l2_logs")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--budget", type=float, default=200.0)
    parser.add_argument("--time-limit", type=float, default=7200)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-agents", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-concurrent-llm", type=int, default=6)
    parser.add_argument("--shell-max-output", type=int, default=50000)
    parser.add_argument("--file-read-max-chars", type=int, default=120000)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--shell-mode", choices=["disabled", "controlled", "unrestricted"], default="unrestricted")
    parser.add_argument("--sandbox-backend", choices=["codex", "host"], default="host")
    parser.add_argument("--sandbox-network", action="store_true")
    parser.add_argument("--orchestration-preference", choices=["solo", "balanced", "parallel", "aggressive"], default="aggressive")
    parser.add_argument(
        "--child-orchestration-preference",
        choices=["task_adaptive", "inherit_decayed", "solo", "balanced", "parallel", "aggressive"],
        default="task_adaptive",
    )
    parser.add_argument("--spawn-before-turn", type=int, default=1)
    parser.add_argument("--max-solo-tool-calls-before-spawn", type=int, default=1)
    parser.add_argument("--min-spawnable-workstreams", type=int, default=2)
    parser.add_argument("--loop-action-policy", choices=["rule", "constraint"], default="constraint")
    args = parser.parse_args()

    prefer_current_nanoma()
    from nanoma.env import load_dotenv

    load_dotenv()
    gaia = load_external_gaia(args.gaia_runner)
    prefer_current_nanoma()

    token = gaia.hf_token()
    if not token and not args.skip_download and not args.snapshot_dir:
        raise SystemExit("Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")

    metadata = gaia.snapshot_metadata_path(args.snapshot_dir) if args.snapshot_dir else args.data_dir / gaia.LEVEL2_PARQUET
    if not args.skip_download:
        metadata = gaia.ensure_level2_metadata(
            args.data_dir,
            token=token,
            force=args.force_download,
            download_method=args.download_method,
            snapshot_dir=args.snapshot_dir,
        )
    if not metadata.exists():
        raise SystemExit(f"Missing metadata parquet: {metadata}")

    records = gaia.table_to_records(metadata)
    selected = gaia.select_records(records, args)
    if not selected:
        raise SystemExit("No GAIA-L2 records selected.")

    print(f"current_nanoma={CURRENT_ROOT}", flush=True)
    print(f"external_gaia_runner={args.gaia_runner}", flush=True)
    print(f"GAIA-L2 records loaded: {len(records)}; selected: {len(selected)}", flush=True)
    rows = asyncio.run(run_selected(args, gaia, selected, token))
    summary = summarize(rows, args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.results_dir / "results.jsonl", rows)
    write_csv(args.results_dir / "results.csv", rows)
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"results={args.results_dir}", flush=True)
    return 0 if summary["completed_tasks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
