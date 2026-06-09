# Query-Driven Structure Smoke Test Without Compact

You are the root coordinator. Demonstrate query-driven coordination in a short run.

Rules:

1. Do not use shell.
2. Do not use compact.
3. Do not send completion messages to root.
4. Use `query()` to inspect agent state and tags before summarizing.

## Phase 1

Spawn exactly 3 workers in one assistant turn. Do not wait between spawns.

Worker tasks:

1. Worker 01 writes `shared/workers/worker_01.md` about runtime, tags `["phase:worker","worker:01","topic:runtime"]`.
2. Worker 02 writes `shared/workers/worker_02.md` about memory, tags `["phase:worker","worker:02","topic:memory"]`.
3. Worker 03 writes `shared/workers/worker_03.md` about query, tags `["phase:worker","worker:03","topic:query"]`.

Each worker must do exactly:

1. `set_status(action="work", current_task_tags=[...], work_outline="write note -> submit -> done")`
2. `file_write(path="shared/workers/worker_XX.md", content="120-160 words")`
3. `submit(path="shared/workers/worker_XX.md")`
4. `set_status(status="done", result="worker XX done", current_task_tags=[...])`

Workers must not call `send` and must not call `compact`.

Root then waits for all 3 workers and calls:

```json
{"filter":{"group_id":"worker_wave","status":"done"},"tags":["phase:worker"],"memory_intent":"root observes workers"}
```

## Phase 2

Spawn exactly 1 synth agent with `group_id="synth_wave"` and tags `["phase:synth","synth:01"]`.

Synth must do exactly:

1. `set_status(action="work", current_task_tags=["phase:synth","synth:01"], work_outline="query workers -> read files -> write synthesis -> done")`
2. `query(filter={"group_id":"worker_wave","status":"done"}, tags=["phase:worker"], memory_intent="synth observes workers")`
3. Read `shared/workers/worker_01.md`, `shared/workers/worker_02.md`, and `shared/workers/worker_03.md`
4. Write `shared/synth/synth_01.md`
5. `submit(path="shared/synth/synth_01.md")`
6. `set_status(status="done", result="synth 01 done", current_task_tags=["phase:synth","synth:01"])`

Synth must not call `send` and must not call `compact`.

Root then waits for synth and calls:

```json
{"filter":{"group_id":"synth_wave","status":"done"},"tags":["phase:synth"],"memory_intent":"root observes synth"}
```

## Final

Root writes `shared/query_structure_report.md` with:

1. Actual agent count.
2. Observed query events.
3. Whether `system -> alpha` completion-report fan-in occurred.
4. A short description of the final structure.

Then submit the report and set root status done.
