"""
Example: run NanoMA as a general-purpose agent.

The agent decides its own strategy. With the neutral system prompt,
it will use whatever coordination it deems appropriate.

Usage:
    python examples/minimal.py "Build a snake game in Python"
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanoma import RuntimeConfig, run_agent


async def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "Write a Python function to compute fibonacci numbers"

    config = RuntimeConfig(
        budget=1.0,
        max_agents=10,
        max_turns=30,
        default_model=os.environ.get("NANOMA_MODEL", "deepseek/deepseek-v4-flash"),
        workspace_root=Path("./workspace"),
        log_dir=Path("./logs"),
    )

    print(f"Task: {task}")
    print(f"Model: {config.default_model} | Budget: ${config.budget}")
    print("-" * 40)

    run = await run_agent(task, config=config)
    stats = run.stats
    print("-" * 40)
    print(f"Agents: {stats['agents']['total_spawned']} | Cost: ${stats['overview']['total_cost_usd']}")
    print(f"Result: {(run.result or '')[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
