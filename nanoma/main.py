"""CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


def cli():
    parser = argparse.ArgumentParser(prog="nanoma", description="NanoMA multi-agent harness")
    parser.add_argument("task", help="Task for the root agent")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--time-limit", type=float, default=0)
    parser.add_argument("--max-agents", type=int, default=100)
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    from nanoma.core import Runtime, RuntimeConfig

    config = RuntimeConfig(
        max_agents=args.max_agents,
        budget=args.budget,
        time_limit=args.time_limit,
        default_model=args.model,
        workspace_root=Path(args.workspace),
        log_dir=Path(args.log_dir),
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
