"""Command-line entry point for the general-purpose NanoMA agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from nanoma.agent import AgentRun, run_agent
from nanoma.core import RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoma",
        description=(
            "Run NanoMA as a general-purpose agent. Each node uses its own model "
            "for both task work and dynamic delegation decisions."
        ),
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task for the agent. Use '-' to read it from standard input.",
    )
    parser.add_argument("--task-file", type=Path, help="Read the task from a UTF-8 file.")
    parser.add_argument("--project", type=Path, help="Existing project/directory the agent may work on.")
    parser.add_argument("--model", default=os.environ.get("NANOMA_MODEL"))
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--max-agents", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-concurrent-llm", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("NANOMA_WORKSPACE", ".nanoma/workspace")),
        help="Internal per-agent workspace root (default: .nanoma/workspace).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("NANOMA_LOG_DIR", ".nanoma/logs")),
        help="Trace directory (default: .nanoma/logs).",
    )
    parser.add_argument("--no-logs", action="store_true", help="Disable trace-file output.")
    instruction_group = parser.add_mutually_exclusive_group()
    instruction_group.add_argument(
        "--instructions",
        default="",
        help="Additional task-independent instructions for all nodes.",
    )
    instruction_group.add_argument(
        "--instructions-file",
        type=Path,
        help="Read additional instructions from a UTF-8 file.",
    )
    parser.add_argument(
        "--node-planning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow every node to dynamically delegate through the same model (default: enabled).",
    )
    parser.add_argument(
        "--disable-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Hide a tool from all nodes; may be repeated.",
    )
    parser.add_argument("--json", action="store_true", help="Print result and stats as JSON.")
    parser.add_argument("--stats", action="store_true", help="Print a compact run summary.")
    parser.add_argument("--quiet", action="store_true", help="Suppress live event output.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def resolve_task(
    args: argparse.Namespace,
    *,
    stdin: TextIO = sys.stdin,
) -> str:
    if args.task_file is not None and args.task is not None:
        raise ValueError("provide either a positional task or --task-file, not both")
    if args.task_file is not None:
        task = args.task_file.expanduser().read_text(encoding="utf-8")
    elif args.task == "-":
        task = stdin.read()
    elif args.task is not None:
        task = args.task
    elif not stdin.isatty():
        task = stdin.read()
    else:
        raise ValueError("a task is required (argument, --task-file, or standard input)")
    task = str(task or "").strip()
    if not task:
        raise ValueError("task is empty")
    return task


def resolve_instructions(args: argparse.Namespace) -> str:
    if args.instructions_file is None:
        return str(args.instructions or "").strip()
    return args.instructions_file.expanduser().read_text(encoding="utf-8").strip()


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig(
        max_agents=args.max_agents,
        max_depth=args.max_depth,
        max_concurrent_llm=args.max_concurrent_llm,
        budget=args.budget,
        time_limit=args.time_limit,
        max_turns=args.max_turns,
        workspace_root=args.workspace,
        log_dir=None if args.no_logs else args.log_dir,
        disabled_tools=set(args.disable_tool),
        benchmark_merge_submit_enabled=False,
    )
    if args.model:
        config = replace(config, default_model=args.model)
    if args.node_planning is not None:
        config = replace(config, node_autonomous_planning=args.node_planning)
    return config


def _event_printer(event: dict) -> None:
    colors = {
        "agent_new": "\033[32m",
        "spawn": "\033[32m",
        "done": "\033[34m",
        "failed": "\033[31m",
        "tool_call": "\033[33m",
        "spawn_judge_decision": "\033[35m",
    }
    kind = str(event.get("event", "event"))
    color = colors.get(kind, "\033[0m")
    agent = event.get("agent", "?")
    data = event.get("data", {})
    print(f"{color}[{kind:>20}]\033[0m {agent}: {data}", file=sys.stderr)


def _print_summary(run: AgentRun) -> None:
    overview = run.stats.get("overview", {})
    agents = run.stats.get("agents", {})
    print(
        "NanoMA: "
        f"agents={agents.get('total_spawned', '?')} "
        f"cost=${overview.get('total_cost_usd', 0):.4f} "
        f"elapsed={overview.get('elapsed_seconds', 0)}s "
        f"shared={run.shared_dir}",
        file=sys.stderr,
    )


async def run_from_args(args: argparse.Namespace, task: str) -> AgentRun:
    return await run_agent(
        task,
        config=config_from_args(args),
        model=args.model,
        project_dir=args.project,
        instructions=resolve_instructions(args),
        on_event=None if args.quiet else _event_printer,
    )


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        task = resolve_task(args)
        run = asyncio.run(run_from_args(args, task))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(run.as_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print(run.result or "")
        if args.stats:
            _print_summary(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
