"""
Example: 3-Layer Hierarchical Architecture

Demonstrates a fixed topology: 1 architect → 3 managers → 5 workers each = 19 agents.
This shows how NanoMA can implement rigid organizational structures via prompt alone.

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

TASK = """You are the TOP-LEVEL ARCHITECT. Build a Shopping Mall E-Commerce app.

You manage exactly 3 SUB-MANAGERS. Each sub-manager manages exactly 5 WORKERS.

1. spawn() FRONTEND MANAGER: "You are the FRONTEND MANAGER. Spawn 5 workers:
   W1: product listing page (shared/frontend/products.py)
   W2: shopping cart (shared/frontend/cart.py)
   W3: auth pages (shared/frontend/auth.py)
   W4: checkout flow (shared/frontend/checkout.py)
   W5: order history (shared/frontend/orders.py)
   wait() for all, write shared/frontend/README.md, set_status('done')."

2. spawn() BACKEND MANAGER: "You are the BACKEND MANAGER. Spawn 5 workers:
   W1: product API (shared/backend/api_products.py)
   W2: cart API (shared/backend/api_cart.py)
   W3: order API (shared/backend/api_orders.py)
   W4: auth API (shared/backend/api_auth.py)
   W5: payment module (shared/backend/payment.py)
   wait() for all, write shared/backend/README.md, set_status('done')."

3. spawn() INFRA MANAGER: "You are the INFRA MANAGER. Spawn 5 workers:
   W1: database models (shared/infra/models.py)
   W2: Dockerfile + docker-compose (shared/infra/Dockerfile, shared/infra/docker-compose.yml)
   W3: CI/CD pipeline (shared/infra/ci.yml)
   W4: nginx config (shared/infra/nginx.conf)
   W5: env config (shared/infra/config.py)
   wait() for all, write shared/infra/README.md, set_status('done')."

Process: spawn all 3 managers → wait() → read READMEs → write shared/ARCHITECTURE.md → submit → done.
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
    print("  3-Layer Hierarchy: 1 + 3 + 15 = 19 agents")
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
