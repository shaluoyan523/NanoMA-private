# OpenDeepThink Runner Evaluation Guardrails

You are working inside a staged source snapshot at `shared/source`.
Do not edit files outside `shared/source`.

Context:
We used `examples/opendeepthink_fixed_runner.py` to reproduce the OpenDeepThink
fixed protocol on CF-73 problems. Several runs failed with `deepseek-v4-flash`,
and then an external host sanity check showed the Polygon package's official
`solutions/ac1.cpp` passes the same evaluator. Future runs need clearer
guardrails so result summaries can distinguish model failure from evaluator
failure without changing the no-cheat LLM protocol.

Task:
Improve the runner and tests in `shared/source` so future post-hoc evaluation
is easier to trust.

Requirements:
1. Inspect `examples/opendeepthink_fixed_runner.py`,
   `examples/evaluate_cf73_solution.py`, and relevant tests before editing.
2. Add an optional official-AC sanity check flag to
   `examples/opendeepthink_fixed_runner.py`.
   - Suggested flag: `--evaluate-official-ac`.
   - It must require `--private-judge-dir` or otherwise record a clear error.
   - It must extract `solutions/ac1.cpp` from `polygon_package.zip`.
   - It must evaluate that official solution through the same evaluator path.
   - It must write a JSON result under `eval/` and include it in `summary.json`.
   - It must not change generation, mutation, judging prompts, BT ranking, or
     LLM-call protocol.
3. Add a compact `evaluation_summary` section to `summary.json` with:
   - selected solution status and pass counts,
   - gen0 status counts when gen0 evaluation is enabled,
   - official AC sanity status when enabled.
4. Add focused tests for the new pure/helper logic. Avoid tests that require
   real Polygon packages or LLM calls.
5. Run the relevant tests in `shared/source`.
6. Write `shared/final_engineering_report.md` summarizing:
   - what changed,
   - what tests passed or failed,
   - what agents queried from peers,
   - which workstreams were pruned or reused,
   - remaining risks.

Coordination expectations:
- Prefer dynamic multi-agent work when useful: auditing, implementation,
  testing, and review are separable streams.
- Agents should use `query()` to discover peer progress and avoid duplicate
  work.
- If another agent is clearly ahead on the same stream, compact a tagged
  summary and stop instead of duplicating effort.
- Use memory tags like `task:opendeepthink-runner`, `role:auditor`,
  `role:implementer`, `role:tester`, `status:candidate`, `status:reviewed`,
  and `status:pruned`.
- Final deliverables are the modified files in `shared/source` and
  `shared/final_engineering_report.md`.
