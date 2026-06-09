# Autonomous Real Engineering Task

You are the root coordinator for a real engineering investigation. The user wants to understand whether agents naturally use `query()` to discover peer state and public memory, not merely because a scripted stress test forced them to query once.

Source snapshot is available at `shared/source/`. Do not use `shell`.

## Objective

Produce an implementation-ready proposal for improving NanoMA's evaluation and visualization of autonomous query behavior.

The proposal should answer:

1. How can NanoMA distinguish scripted query calls from useful autonomous query calls?
2. What runtime events or metadata should be added to make query intent and usefulness observable?
3. How should the viewer/map show query edges over time without making the graph misleading?
4. What focused tests or example tasks should be added to verify autonomous query behavior?
5. What are the smallest safe code changes that should be implemented next?

## Autonomy Rules

- You may spawn peer agents if parallel work genuinely helps. There is no required number of agents.
- Do not create fixed waves or quotas.
- Do not tell every child agent to query. Instead, ask them to use `query()` only if they need information from peers, upstream work, public memory, or artifacts.
- Direct `send()` should be rare. Prefer public memory, shared files, and query when coordination is needed.
- Each child agent should expose useful public memory/tags before stopping, preferably with `compact(..., stop_after=true, result=...)`.
- Use tags that describe the actual work, for example `autonomy`, `query-intent`, `viewer-map`, `runtime-events`, `test-design`, `implementation-plan`.

## Suggested Work Areas

You do not have to cover these with fixed agents. Decide dynamically:

- Runtime/meta behavior around `query()`, `compact()`, `spawn()`, `wait()`, and events.
- Viewer/map behavior for query edges, live updates, edge density, and phase filtering.
- Test/example design for distinguishing scripted query from spontaneous query.
- Data model for query intent, trigger, result usage, and usefulness.
- Minimal code-change plan with risks.

## Evidence Expectations

- Read relevant source from `shared/source/`, but avoid open-ended source crawling.
- Use `grep` first, then small `file_read` ranges where useful.
- If another agent has already investigated something, discover that through `query()` or shared artifacts rather than duplicating the same work.
- It is acceptable for some agents not to query if they genuinely do not need peer state.

## Deliverables

Write these files:

- `shared/autonomy_real_task_report.md`
- `shared/autonomy_query_evidence.md`
- `shared/autonomy_next_patch_plan.md`

The final report should include:

1. Actual agent structure that emerged.
2. Which agents queried, why they queried, and whether the query result changed their next action.
3. Which agents did not query and whether that was reasonable.
4. Recommended telemetry fields for query usefulness.
5. Recommended viewer/map changes.
6. Minimal implementation plan and tests.

Finish with:

1. `submit(path="shared/autonomy_real_task_report.md")`
2. `set_status(status="done", result="autonomous real engineering task complete")`
