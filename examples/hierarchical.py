"""
Example: an autonomously formed hierarchy

Every node can reach a planning point and create another level through the
runtime's same-model planning decision. The final topology is observed rather
than prescribed by direct spawn calls.

Usage:
    python examples/hierarchical.py
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanoma.core import Runtime, RuntimeConfig

TASK = """Build a small Shopping Mall E-Commerce application with frontend,
backend, infrastructure, tests, and an architecture document.

At each genuinely multi-step phase, use task_create to make the next planning
decision. Parallelize independent implementation and verification work when it
helps. Any child may further decompose its own assignment. Coordinate results,
run the relevant tests, write the final architecture document, submit the main
artifact, and finish with a concise result.
"""


async def main():
    workspace = Path("/tmp/nanoma-hierarchical/workspace")
    logs = Path("/tmp/nanoma-hierarchical/logs")
    if workspace.exists():
        shutil.rmtree(workspace.parent)
    workspace.mkdir(parents=True)
    logs.mkdir(parents=True)

    # Start viewer
    viewer_py = Path(__file__).parent.parent / "nanoma" / "viewer.py"
    subprocess.run(["fuser", "-k", "8900/tcp"], capture_output=True)
    viewer = subprocess.Popen(
        [sys.executable, str(viewer_py), str(logs), "8900"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[viewer] http://localhost:8900")

    config = RuntimeConfig(
        budget=2.0,
        max_agents=25,
        max_depth=4,
        max_turns=30,
        max_concurrent_llm=8,
        time_limit=180,
        default_model=os.environ.get("NANOMA_MODEL", "deepseek/deepseek-v4-flash"),
        workspace_root=workspace,
        log_dir=logs,
    )

    def on_event(e):
        ev, d = e["event"], e["data"]
        if ev == "agent_new":
            print(f'  [+] {e["agent"]:>10}  depth={d.get("depth",0)}  {d.get("task","")[:50]}')
        elif ev == "spawn":
            print(f'  [→] {e["agent"]:>10} → {d["child"]}')
        elif ev in ("done", "failed"):
            print(f'  [✓] {e["agent"]:>10}  {d.get("status","")}  turns={d.get("turns","?")}')

    rt = Runtime(config=config, on_event=on_event)

    print("=" * 60)
    print("  Autonomous hierarchy: topology formed by per-node planning")
    print("=" * 60)
    result = await rt.run(TASK)
    print("=" * 60)

    stats = rt.stats()
    print(f"\nAgents: {stats['agents']['total_spawned']} | "
          f"Depth: {stats['agents']['max_depth']} | "
          f"Cost: ${stats['overview']['total_cost_usd']} | "
          f"Time: {stats['overview']['elapsed_seconds']}s")
    print(f"Result: {(result or '')[:200]}")
    print(f"\n[viewer] http://localhost:8900 — Ctrl+C to stop")

    try:
        viewer.wait()
    except KeyboardInterrupt:
        viewer.terminate()


if __name__ == "__main__":
    asyncio.run(main())
