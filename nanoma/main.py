"""CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def cli():
    from nanoma.env import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="nanoma", description="NanoMA multi-agent harness")
    parser.add_argument("task", help="Task for the root agent")
    parser.add_argument("--model", default=os.environ.get("NANOMA_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--time-limit", type=float, default=0)
    parser.add_argument("--max-agents", type=int, default=100)
    parser.add_argument("--allow-agent-model-override", action="store_true", help="Allow agents to request child models instead of inheriting the runtime default")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--sandbox-backend", default="codex", choices=["codex", "host"])
    parser.add_argument("--sandbox-codex-bin", default="codex", help="Codex CLI executable used for the codex sandbox backend")
    parser.add_argument("--sandbox-network", action="store_true", help="Allow network access inside Codex-sandboxed agent shell commands")
    parser.add_argument("--no-sandbox", action="store_true", help="Run agent shell commands directly on the host")
    parser.add_argument("--shell-mode", choices=["disabled", "controlled", "unrestricted"], default="controlled", help="Agent shell policy")
    parser.add_argument("--notify-parent-on-done", action="store_true", help="Send automatic system completion notifications to parent agents")
    parser.add_argument(
        "--loop-action-policy",
        choices=["rule", "constraint"],
        default=os.environ.get("NANOMA_LOOP_ACTION_POLICY", "constraint"),
        help="Loop planner policy: rule keeps existing short-circuit behavior; constraint scores candidate actions by runtime pressure",
    )
    parser.add_argument(
        "--disable-loop-constraint",
        action="append",
        default=[],
        help="Disable one constraint metric for --loop-action-policy constraint. Can be repeated.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    from nanoma.core import Runtime, RuntimeConfig

    disabled_loop_constraints = set(args.disable_loop_constraint or [])
    if env_disabled := os.environ.get("NANOMA_DISABLED_LOOP_CONSTRAINTS"):
        disabled_loop_constraints.update(item.strip() for item in env_disabled.split(",") if item.strip())

    config = RuntimeConfig(
        max_agents=args.max_agents,
        allow_agent_model_override=args.allow_agent_model_override,
        budget=args.budget,
        time_limit=args.time_limit,
        default_model=args.model,
        workspace_root=Path(args.workspace),
        log_dir=Path(args.log_dir),
        sandbox_backend="host" if args.no_sandbox else args.sandbox_backend,
        sandbox_codex_bin=args.sandbox_codex_bin,
        sandbox_network=args.sandbox_network,
        shell_mode=args.shell_mode,
        notify_parent_on_done=args.notify_parent_on_done,
        loop_action_policy=args.loop_action_policy,
        disabled_loop_constraints=disabled_loop_constraints,
    )

    def on_event(e):
        color = {"spawn": "\033[32m", "done": "\033[34m", "tool": "\033[33m"}.get(e["event"], "\033[0m")
        print(f"{color}[{e['event']:>6}]\033[0m {e['agent']}: {e['data']}", file=sys.stderr)

    runtime = Runtime(config=config, on_event=on_event)

    async def main():
        result = await runtime.run(args.task, model=args.model)
        print(result)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
