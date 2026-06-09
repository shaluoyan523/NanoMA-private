# CooperBench Max-Agent Task: Click task2068, 12 Features

You are running a NanoMA multi-agent test based on CooperBench `pallets_click_task/task2068`.

Official task scale:
- CooperBench `coop` and `team` runners create `n_agents = len(features)`.
- This task has 12 feature directories, the largest feature count in the local CooperBench dataset.
- This run intentionally uses all 12 features to test whether NanoMA can build and use a large collaborative structure.

Workspace:
- Writable source tree: `shared/source`
- Feature specifications: `shared/cooperbench_task2068_features/feature1.md` through `feature12.md`
- Initial spawn batch: `shared/bootstrap_spawn_12.json`
- Do not read CooperBench `feature.patch`, `tests.patch`, or `combined.patch`. Solve from the feature specs and source code.
- For shell commands, start from your private workspace and use `cd ../shared/source`.
- For file tools, paths beginning with `shared/` are valid.

Required first action for the root coordinator:
- Call `batch` with path `shared/bootstrap_spawn_12.json`.
- Do this before inspecting source or planning.
- The batch file creates exactly 12 feature implementer agents, one per official feature.

Coordination requirements:
- Parent/child structure is allowed, but agents should not rely on parent reports.
- Each feature agent must set status with precise tags, keep public memory current, and query peer agents when it needs interface status, overlapping file changes, test assumptions, or implementation artifacts.
- Query calls must be need-driven, not a fixed quota.
- Direct `send` is allowed only for concrete conflicts or interface contracts.
- Agents should write notes under `shared/cooperbench_task2068_notes/featureN.md`.
- If a feature agent discovers cross-feature API conflict, it may spawn a focused reviewer or compatibility agent.

Implementation expectations:
- All 12 features modify Click editor behavior, mainly `src/click/_termui_impl.py` and `src/click/termui.py`.
- Preserve backwards compatibility for existing `click.edit()` behavior.
- Prefer small, composable changes to shared editor command construction rather than 12 isolated rewrites.
- Avoid whole-file overwrites of large source files. Use shell scripts, Python scripts, or small targeted edits against `shared/source` when needed.

Integration expectations:
- After feature agents are running, the root should query their public memory and artifacts.
- Spawn integration/review agents only after there is enough peer state to justify them.
- Integration should validate combined API signatures, editor command construction, environment handling, process options, cleanup behavior, and tests.
- Final report should be written to `shared/cooperbench_click2068_full_report.md`.

Done criteria:
- Feature agents have either implemented or clearly marked blocked with evidence.
- Integration has summarized which of the 12 official features are implemented, partially implemented, or failed.
- The report includes spawn count, peak concurrency, query graph observations, and whether NanoMA achieved the intended 12-agent official-scale structure.
