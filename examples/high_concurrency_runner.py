"""Run the high-concurrency NanoMA workflow stress test."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanoma import Runtime, RuntimeConfig
from nanoma.env import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the high-concurrency workflow stress test")
    parser.add_argument("--task-file", default="examples/high_concurrency_workflow_task.md")
    parser.add_argument("--model", default=None)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--max-agents", type=int, default=60)
    parser.add_argument("--allow-agent-model-override", action="store_true", help="Allow agents to request child models instead of inheriting the runner model")
    parser.add_argument("--max-concurrent-llm", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="Set NANOMA_MAX_TOKENS for each LLM response")
    parser.add_argument("--time-limit", type=float, default=1200)
    parser.add_argument("--workspace", default="./workspace-high-concurrency")
    parser.add_argument("--log-dir", default="./logs-high-concurrency")
    parser.add_argument("--sandbox-backend", choices=["codex", "host"], default="codex")
    parser.add_argument("--sandbox-network", action="store_true")
    parser.add_argument("--enable-shell", action="store_true", help="Allow unrestricted agent shell tool calls")
    parser.add_argument(
        "--shell-mode",
        choices=["disabled", "controlled", "unrestricted"],
        default=None,
        help="Agent shell policy. Omitted means controlled; --enable-shell means unrestricted.",
    )
    parser.add_argument("--notify-parent-on-done", action="store_true", help="Send automatic system completion notifications to parent agents")
    parser.add_argument("--source-root", default=None, help="Copy a source snapshot into workspace/shared/source before running")
    parser.add_argument("--file-read-max-chars", type=int, default=4000, help="Max chars returned by file_read when no line range is requested")
    parser.add_argument("--grep-max-results", type=int, default=40, help="Max grep results returned to agents")
    parser.add_argument("--bootstrap-batch-file", default=None, help="Execute this JSON batch file as the root agent before its first LLM turn")
    parser.add_argument("--orchestration-preference", choices=["solo", "balanced", "parallel", "aggressive"], default=None, help="Root agent preference for solo vs multi-agent orchestration")
    parser.add_argument(
        "--child-orchestration-preference",
        choices=["task_adaptive", "inherit_decayed", "solo", "balanced", "parallel", "aggressive"],
        default=None,
        help="Default strategy/preference for spawned child agents; omitted means task_adaptive",
    )
    parser.add_argument("--spawn-before-turn", type=int, default=None, help="Turn threshold for orchestration nudges when no children exist")
    parser.add_argument("--max-solo-tool-calls-before-spawn", type=int, default=None, help="Tool-call threshold for orchestration nudges when no children exist")
    parser.add_argument("--min-spawnable-workstreams", type=int, default=None, help="Workstream count referenced by orchestration guidance")
    parser.add_argument("--loop-action-policy", choices=["rule", "constraint"], default=os.environ.get("NANOMA_LOOP_ACTION_POLICY", "rule"), help="Loop planner policy")
    parser.add_argument("--disable-loop-constraint", action="append", default=[], help="Disable one constraint metric for the constraint loop planner")
    parser.add_argument("--viewer-port", type=int, default=8900, help="Port for the live trace viewer")
    parser.add_argument("--no-viewer", action="store_true", help="Do not start or rebind the live trace viewer")
    parser.add_argument("--fresh", action="store_true", help="Remove workspace and log directory before running")
    return parser.parse_args()


def stage_source_snapshot(source_root: Path, workspace: Path, log_dir: Path) -> Path:
    """Copy source into shared/source while avoiding recursive test artifacts."""
    source_root = source_root.resolve()
    target = (workspace / "shared" / "source").resolve()
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    excluded_roots = {".git", ".env", ".zgent", "__pycache__", ".pytest_cache", ".mypy_cache"}
    for generated_path in (workspace, log_dir, target):
        try:
            excluded_roots.add(generated_path.resolve().relative_to(source_root).parts[0])
        except (ValueError, IndexError):
            pass

    def ignore(dir_path: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            if name in excluded_roots:
                ignored.add(name)
            elif name.startswith(("logs-", "workspace-")):
                ignored.add(name)
            elif name.endswith((".pyc", ".pyo")):
                ignored.add(name)
        return ignored

    shutil.copytree(source_root, target, ignore=ignore)
    try:
        import subprocess

        subprocess.run(["git", "init"], cwd=target, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "add", "."], cwd=target, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=nanoma",
                "-c",
                "user.email=nanoma@example.invalid",
                "commit",
                "-m",
                "baseline source snapshot",
            ],
            cwd=target,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[source] warning: failed to initialize git baseline in shared/source: {e}")
    return target


def start_viewer(log_dir: Path, port: int) -> subprocess.Popen:
    """Start a viewer bound to this run's log directory."""
    log_dir.mkdir(parents=True, exist_ok=True)
    viewer_py = Path(__file__).parent.parent / "nanoma" / "viewer.py"
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    return subprocess.Popen(
        [sys.executable, str(viewer_py), str(log_dir), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def main() -> None:
    load_dotenv()
    args = parse_args()
    workspace = Path(args.workspace)
    log_dir = Path(args.log_dir)
    if args.fresh:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)
    if args.llm_max_tokens is not None:
        os.environ["NANOMA_MAX_TOKENS"] = str(args.llm_max_tokens)
    if args.source_root:
        staged = stage_source_snapshot(Path(args.source_root), workspace, log_dir)
        print(f"[source] staged snapshot: {staged}")
    if not args.no_viewer:
        start_viewer(log_dir, args.viewer_port)
        print(f"[viewer] http://localhost:{args.viewer_port} log_dir={log_dir}")
    shell_mode = args.shell_mode or ("unrestricted" if args.enable_shell else "controlled")

    task = Path(args.task_file).read_text()
    disabled_loop_constraints = set(args.disable_loop_constraint or [])
    if env_disabled := os.environ.get("NANOMA_DISABLED_LOOP_CONSTRAINTS"):
        disabled_loop_constraints.update(item.strip() for item in env_disabled.split(",") if item.strip())
    config = RuntimeConfig(
        budget=args.budget,
        max_agents=args.max_agents,
        allow_agent_model_override=args.allow_agent_model_override,
        max_concurrent_llm=args.max_concurrent_llm,
        max_turns=args.max_turns,
        time_limit=args.time_limit,
        default_model=args.model or os.environ.get("NANOMA_MODEL", "deepseek-v4-flash"),
        workspace_root=workspace,
        log_dir=log_dir,
        sandbox_backend=args.sandbox_backend,
        sandbox_network=args.sandbox_network,
        notify_parent_on_done=args.notify_parent_on_done,
        file_read_max_chars=args.file_read_max_chars,
        grep_max_results=args.grep_max_results,
        bootstrap_batch_file=args.bootstrap_batch_file,
        shell_mode=shell_mode,
        orchestration_preference=args.orchestration_preference or "balanced",
        child_orchestration_preference=args.child_orchestration_preference,
        spawn_before_turn=args.spawn_before_turn if args.spawn_before_turn is not None else 2,
        max_solo_tool_calls_before_spawn=(
            args.max_solo_tool_calls_before_spawn if args.max_solo_tool_calls_before_spawn is not None else 5
        ),
        min_spawnable_workstreams=args.min_spawnable_workstreams if args.min_spawnable_workstreams is not None else 3,
        loop_action_policy=args.loop_action_policy,
        disabled_loop_constraints=disabled_loop_constraints,
        enabled_work_tools=(
            None
            if shell_mode in {"controlled", "unrestricted"}
            else {"file_read", "file_write", "file_replace", "file_list", "grep", "bt_aggregate"}
        ),
    )

    def on_event(event: dict) -> None:
        ev = event["event"]
        data = event["data"]
        if ev == "agent_new":
            print(f'[agent] {event["agent"]:>10} depth={data.get("depth", 0)} {data.get("task", "")[:80]}')
        elif ev == "spawn":
            print(f'[spawn] {event["agent"]:>10} -> {data.get("child")}')
        elif ev == "query":
            print(f'[query] {event["agent"]:>10} count={data.get("result_count")} filter={data.get("filter") or data.get("tags") or ""}')
        elif ev == "orchestration_nudge":
            print(f'[nudge] {event["agent"]:>10} pref={data.get("preference")} turns={data.get("turns")} tools={data.get("tool_calls")}')
        elif ev == "llm_done":
            print(f'[llm]   {event["agent"]:>10} tokens={data.get("tokens")} tools={data.get("tool_calls")}')
        elif ev in {"done", "failed"}:
            print(f'[{ev}] {event["agent"]:>10} turns={data.get("turns")} result={(data.get("result") or "")[:100]}')

    runtime = Runtime(config=config, on_event=on_event)
    result = await runtime.run(task, model=config.default_model)
    stats = runtime.stats()

    print("\n=== Result ===")
    print(result[:1000])
    print("\n=== Stats ===")
    print(f"agents={stats['agents']['total_spawned']} peak={stats['agents']['peak_concurrent']}")
    print(f"tokens={stats['overview']['total_tokens']} cost=${stats['overview']['total_cost_usd']}")
    print(f"tool_calls={stats['tools']['total_calls']} messages={stats['communication']['total_messages']}")


if __name__ == "__main__":
    asyncio.run(main())
