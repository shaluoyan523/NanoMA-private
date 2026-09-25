"""
Example: Run any preset pattern with a task.

Usage:
    python examples/run_preset.py <preset_file> "<task>" [--budget 2.0] [--model deepseek/deepseek-v4-flash]
    python examples/run_preset.py presets/06_debate.md "Is TDD worth the overhead?"
    python examples/run_preset.py presets/13_mixture_of_agents.md "Best database for time-series data?"
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanoma.core import Runtime, RuntimeConfig


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run a NanoMA preset")
    p.add_argument("preset", help="Path to preset .md file")
    p.add_argument("task", help="Task to inject into {task} placeholder")
    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--max-agents", type=int, default=20)
    p.add_argument("--time-limit", type=int, default=180)
    p.add_argument("--viewer-port", type=int, default=8900)
    return p.parse_args()


async def main():
    args = parse_args()

    # Load and fill preset
    preset = Path(args.preset).read_text()
    # Strip comment header
    lines = preset.split("\n")
    prompt_lines = []
    past_header = False
    for line in lines:
        if not past_header and line.startswith("#"):
            continue
        past_header = True
        prompt_lines.append(line)
    prompt = "\n".join(prompt_lines).strip().replace("{task}", args.task)

    # Setup workspace
    workspace = Path("./workspace")
    logs = Path("./logs")
    workspace.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    # Start viewer
    viewer_py = Path(__file__).parent.parent / "nanoma" / "viewer.py"
    subprocess.run(["fuser", "-k", f"{args.viewer_port}/tcp"], capture_output=True)
    viewer = subprocess.Popen(
        [sys.executable, str(viewer_py), str(logs), str(args.viewer_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[viewer] http://localhost:{args.viewer_port}")

    # Configure runtime
    config = RuntimeConfig(
        budget=args.budget,
        max_agents=args.max_agents,
        max_depth=5,
        max_turns=40,
        max_concurrent_llm=8,
        time_limit=args.time_limit,
        default_model=args.model,
        workspace_root=workspace,
        log_dir=logs,
    )

    def on_event(e):
        ev, d = e["event"], e["data"]
        if ev == "spawn":
            print(f'  [spawn] {e["agent"]} → {d["child"]}')
        elif ev in ("done", "failed"):
            print(f'  [{ev:>6}] {e["agent"]} turns={d.get("turns", "?")}')

    rt = Runtime(config=config, on_event=on_event)

    print(f"\nPreset: {args.preset}")
    print(f"Task:   {args.task[:80]}")
    print(f"Model:  {args.model} | Budget: ${args.budget} | Time: {args.time_limit}s")
    print("=" * 60)

    result = await rt.run(prompt)

    print("=" * 60)
    stats = rt.stats()
    print(f"Agents: {stats['agents']['total_spawned']} | "
          f"Depth: {stats['agents']['max_depth']} | "
          f"Cost: ${stats['overview']['total_cost_usd']} | "
          f"Time: {stats['overview']['elapsed_seconds']}s")
    print(f"\nResult: {(result or '')[:500]}")
    print(f"\n[viewer] still running at http://localhost:{args.viewer_port}")

    # Keep viewer alive
    try:
        viewer.wait()
    except KeyboardInterrupt:
        viewer.terminate()


if __name__ == "__main__":
    asyncio.run(main())
