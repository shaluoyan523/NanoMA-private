"""Create a root bootstrap batch for the CF-73 top-10 OpenDeepThink run."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


PROBLEMS = [
    ("2174E1", 3100, "Game of Scientists (Version 1)"),
    ("2147G", 3100, "Modular Tetration"),
    ("2138E1", 3100, "Determinant Construction (Easy Version)"),
    ("2138E2", 3100, "Determinant Construction (Hard Version)"),
    ("2161F", 3000, "SubMST"),
    ("2158F2", 3000, "Distinct GCDs (Hard Version)"),
    ("2156F2", 3000, "Strange Operation (Hard Version)"),
    ("2164F2", 2900, "Chain Prefix Rank (Hard Version)"),
    ("2162H", 2900, "Beautiful Problem"),
    ("2146F", 2900, "Bubble Sort"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CF-73 top-10 bootstrap batch")
    parser.add_argument("--workspace", default="workspace-cf73-top10", help="NanoMA workspace containing shared/")
    args = parser.parse_args()

    agents = []
    for problem_id, rating, name in PROBLEMS:
        agents.append(
            {
                "role": f"problem-coordinator-{problem_id}",
                "task": problem_task(problem_id, rating, name),
                "group_id": "cf73-top10-opendeepthink-full",
                "relationship": "peer",
                "create_type": "peer_agent",
                "workflow_prior": "opendeepthink-n20-k4-t3-m10",
                "orchestration_preference": "aggressive",
                "current_task_tags": [
                    "benchmark:cf73",
                    "protocol:opendeepthink",
                    "n:20",
                    "k:4",
                    "t:3",
                    "m:10",
                    f"problem:{problem_id}",
                    "role:problem-coordinator",
                ],
            }
        )
        for idx in range(20):
            agents.append(
                {
                    "role": f"gen0-generator-{problem_id}-{idx:02d}",
                    "task": generator_task(problem_id, rating, name, idx),
                    "group_id": f"cf73-{problem_id}-gen0",
                    "relationship": "peer",
                    "create_type": "peer_agent",
                    "workflow_prior": "opendeepthink-gen0-sampling",
                    "orchestration_preference": "solo",
                    "current_task_tags": [
                        "benchmark:cf73",
                        "protocol:opendeepthink",
                        f"problem:{problem_id}",
                        "phase:gen0",
                        "role:generator",
                        f"candidate:{idx:02d}",
                    ],
                }
            )
    batch = [
        {
            "tool": "spawn_many",
            "args": {
                "defaults": {
                    "group_id": "cf73-top10-opendeepthink-full",
                    "relationship": "peer",
                    "create_type": "peer_agent",
                    "workflow_prior": "opendeepthink-n20-k4-t3-m10",
                    "orchestration_preference": "aggressive",
                },
                "agents": agents,
            },
        }
    ]
    output = Path(args.workspace) / "shared" / "cf73_top10_bootstrap_full.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(batch, indent=2, ensure_ascii=False) + "\n")
    print(output)


def problem_task(problem_id: str, rating: int, name: str) -> str:
    return f"""Run the full OpenDeepThink protocol for CF-73 problem {problem_id} ({name}, rating {rating}).

Public files:
- shared/cf73_top10/problems/{problem_id}/metadata.json
- shared/cf73_top10/problems/{problem_id}/statement.md
- shared/cf73_top10/README.md
- shared/cf73_top10/summary.md may be created or updated after you finish.

Rules:
- Use only public files in shared/cf73_top10 and your own/peer public memory.
- Do not use web search, local judge, hidden tests, Polygon packages, official solutions, checkers, interactors, or generated-test commands.
- Use C++17. For interactive problems, write a standard stdin/stdout interactive solution and flush after each query.
- Follow n=20, K=4, T=3, M=10 exactly unless runtime budget exhaustion forces you to record an incomplete step.
- Do not shrink the protocol into a small analysis/implementation split. Missing calls must be recorded as incomplete, not silently replaced.

Required workflow:
1. Read the statement and metadata.
2. The bootstrap has already launched 20 gen-0 generator peers with group_id `cf73-{problem_id}-gen0`. Query that group and inspect shared/cf73_top10/{problem_id}/gen0/ until 20 candidate files and rationales exist. If fewer than 20 finish after two coordinator turns, spawn replacement generator agents only for missing candidate IDs.
3. For t=1..3, form 40 unique K=4 pairwise comparisons, spawn/run comparison agents, aggregate with bt_aggregate, preserve top 5, discard bottom 5, and spawn/run 15 mutation agents for the top 15 into the next population.
4. Run the final M=10 comparison round with 100 unique comparisons, aggregate with bt_aggregate, select the top solution, and write shared/cf73_top10/{problem_id}/selected_solution.cpp.
5. Write shared/cf73_top10/{problem_id}/report.md with exact protocol completion counts, final BT ranking, and any uncertainty.

Coordination requirements:
- Use tags including benchmark:cf73, problem:{problem_id}, phase:<phase>, role:<role>.
- Query peer agents by group_id and problem:{problem_id} before each synthesis step; do not rely only on direct reports to your parent.
- Compact public memory with artifacts before stopping.
"""


def generator_task(problem_id: str, rating: int, name: str, idx: int) -> str:
    return f"""Generate one independent gen-0 C++17 candidate for CF-73 problem {problem_id} ({name}, rating {rating}).

Public files:
- shared/cf73_top10/problems/{problem_id}/metadata.json
- shared/cf73_top10/problems/{problem_id}/statement.md

Rules:
- Use only the public statement and metadata. Do not use web search, known solutions, local judge, hidden tests, Polygon packages, official solutions, checkers, interactors, or generated-test commands.
- Do not coordinate your algorithm with other generators before writing the candidate; diversity is useful.
- For interactive problems, produce a standard stdin/stdout interactive C++17 solution and flush after each query.
- Keep visible chat analysis brief. After reading the statement, immediately call `file_write` for both required files. Do not spend a turn writing a long prose analysis without tool calls.

Output:
- Write `shared/cf73_top10/{problem_id}/gen0/candidate_{idx:02d}.cpp`.
- Write `shared/cf73_top10/{problem_id}/gen0/candidate_{idx:02d}.md` with the core idea, complexity, and known risks.
- Compact/stop with tags `benchmark:cf73`, `problem:{problem_id}`, `phase:gen0`, `role:generator`, `candidate:{idx:02d}` and list both files as artifacts.

Completion rule:
- Your first turn may read files. Your next turn must write the `.cpp` and `.md` files, even if the solution is uncertain. Record uncertainty in the `.md` file instead of continuing analysis in chat.
"""


if __name__ == "__main__":
    main()
