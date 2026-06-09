# Query-Driven Structure Smoke Test

You are the root coordinator. Run a short workflow that demonstrates query-driven coordination rather than parent-report fan-in.

Rules:

1. Do not use shell.
2. Do not ask child agents to send completion messages to root.
3. Child agents must expose progress through `set_status`, `compact`, `submit`, and fine-grained tags.
4. Root and synth agents must use `query()` to inspect other agents' status/public_memory before summarizing.

## Phase 1: Worker Wave

Use `batch` to spawn 4 workers with `group_id="worker_wave"` and tags `["phase:worker"]`.

Each worker writes one short file:

- worker 01: `shared/workers/worker_01.md`, tags `["phase:worker", "worker:01", "topic:runtime"]`
- worker 02: `shared/workers/worker_02.md`, tags `["phase:worker", "worker:02", "topic:memory"]`
- worker 03: `shared/workers/worker_03.md`, tags `["phase:worker", "worker:03", "topic:query"]`
- worker 04: `shared/workers/worker_04.md`, tags `["phase:worker", "worker:04", "topic:viewer"]`

Worker instructions:

- `set_status(action="work", current_task_tags=[...], work_outline="write short worker note -> submit -> compact memory and stop")`
- Write 120-180 words to the assigned file.
- `submit(path="shared/workers/worker_XX.md")`
- `compact(summary="...", tags=[...], files=[...], experience="...", stop_after=true, result="worker XX done")`
- Do not call `send`.

After all workers are done, root must call:

```json
{"filter":{"group_id":"worker_wave","status":"done"},"tags":["phase:worker"],"memory_intent":"root checks completed workers"}
```

## Phase 2: Synth Wave

Use `batch` to spawn 2 synth agents with `group_id="synth_wave"` and tags `["phase:synth"]`.

Each synth must:

- `set_status(action="work", current_task_tags=["phase:synth", "synth:XX"], work_outline="query workers -> read worker files -> write synthesis -> submit -> compact and stop")`
- First call `query(filter={"group_id":"worker_wave","status":"done"}, tags=["phase:worker"], memory_intent="synth observes workers")`
- Read all worker files under `shared/workers/`.
- Write `shared/synth/synth_XX.md` with a 180-250 word synthesis.
- `submit(path="shared/synth/synth_XX.md")`
- `compact(summary="...", tags=["phase:synth", "synth:XX"], files=[...], experience="...", stop_after=true, result="synth XX done")`
- Do not call `send`.

After both synth agents are done, root must call:

```json
{"filter":{"group_id":"synth_wave","status":"done"},"tags":["phase:synth"],"memory_intent":"root checks synth outputs"}
```

## Final

Root writes `shared/query_structure_report.md` with:

1. Actual agent count.
2. Which agents queried which groups.
3. Whether any completion messages were sent to root.
4. Whether the structure is still root-centered or query-observed.

Then:

- `submit(path="shared/query_structure_report.md")`
- `set_status(status="done", result="query structure smoke complete")`
