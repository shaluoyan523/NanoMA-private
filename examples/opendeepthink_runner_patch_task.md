# Runner Guardrail Patch Task

Work only inside the staged source tree. The private CF-73 judge packages are
not available to you and must not be referenced.

Objective:
Patch the local OpenDeepThink fixed runner so post-hoc evaluation summaries are
auditable without changing the LLM generation, mutation, judging, or BT protocol.

Source areas to inspect:
- `shared/source/examples/opendeepthink_fixed_runner.py`
- `shared/source/examples/evaluate_cf73_solution.py`
- `shared/source/tests/test_nanoma.py`

Required implementation:
1. Add an optional CLI flag named `--evaluate-official-ac`.
2. When the flag is used, the runner should evaluate `solutions/ac1.cpp` from
   `polygon_package.zip` through the same `evaluate_solution` path used for
   selected/gen0 solutions.
3. If no private judge directory is configured, the result must be a clear JSON
   error, not a crash.
4. Add an `evaluation_summary` object to `summary.json` that compactly reports:
   selected status/pass counts, gen0 status counts when present, and official
   AC sanity status when present.
5. Add focused tests in `shared/source/tests/test_opendeepthink_guardrails.py`.
   Tests must not call an LLM or require a real Polygon package.

Coordination:
- This task has separable audit, patch, test, and review streams. Prefer
  spawning peer agents if useful, then use `query()` to discover their progress.
- If you continue solo, record the reason in memory using tags
  `task:opendeepthink-runner` and `status:solo-decision`.
- Do not claim tests passed unless you actually ran them with shell.
- Do not submit or compact as done unless `git -C shared/source diff --stat`
  shows a source/test change.

Completion:
Submit the changed test file and stop with a concise result summary.
