# NanoMA Optimizations

Evaluation and experimental enhancements kept separate from the reusable
`nanoma` package.

## `todo_tools` — compatibility path for core planning

The implementation now lives in `nanoma/planning.py` and ships with the general
agent. `optimizations.todo_tools` remains as a compatibility import for older
experiments and trajectories.

| Tool | Purpose |
|------|---------|
| `task_create` | Create a pending task (`subject`, `description`, optional `activeForm`). Returns its `task_id`. |
| `task_update` | Change a task's `status` (`pending` / `in_progress` / `completed` / `cancelled`) or revise its fields. |
| `task_list` | Read the current agent's task list; optional `status` filter. |

### Why

Workers do not receive direct spawn tools. They use `task_create` to
declare a fresh planning node. Before the task is materialized, the runtime
asks that worker's own model whether parallel children are useful and
what each child should do. A declined decision becomes an ordinary local task.

### Per-turn state re-injection + nudges

To match Claude Code's loop (not just the initial trigger), the runtime injects
an **ephemeral** task-list reminder before each LLM call:

- **State re-injection**: while any task is `pending`/`in_progress`, the current
  list (with `N completed / M total` progress) is appended as a trailing
  `user` message — so the model always sees its checklist and is nudged to
  `task_update` / add follow-ups.
- **Stale nudge**: if the list has not changed for `NANOMA_TODO_STALE_AFTER`
  turns (default 3) while tasks remain open, an "update your statuses" reminder
  is appended.
- **Empty-list hint**: if no tasks exist after `NANOMA_TODO_EMPTY_HINT_AFTER`
  turns (default 4), a one-time gentle hint suggests creating a list for
  multi-step work (fires once, then never again).

The reminder is **transient**: appended only to the per-call message list, never
to persisted `agent.history`, so it always reflects current state and never
accumulates context. Implemented via `Runtime._history_with_todo_reminder`
(single call-site change in the ReAct loop) + `render_todo_reminder(agent)`.
When all tasks are terminal, injection goes silent.

Tuning:

```bash
export NANOMA_TODO_STALE_AFTER=3        # turns idle before stale nudge
export NANOMA_TODO_EMPTY_HINT_AFTER=4   # turns before one-time empty-list hint
```

### Design

- **Naming**: snake_case to match NanoMA conventions (`task_create`, `set_bio`,
  `ws_read_file`). The tool *descriptions* keep Claude's "When to Use / When
  NOT to Use" policy text — proxy-log analysis of the planning-workflow runs
  showed that description text (not any runtime event) is what actually gates
  when a model emits these calls.
- **State**: stored per-agent on the `Agent` instance as `agent._todos`
  (list) + `agent._todo_seq` (counter), attached lazily. `Agent` is a plain
  dataclass without `__slots__`, so no core dataclass edit is required. State
  is per-agent (like Claude's per-session list); judge-created children
  start empty.
- **Events**: `task_create` / `task_update` emit viewer events through
  `runtime._emit`, like `send`; judge-created children still emit the internal
  `spawn` event used for topology accounting.

### Integration

Registered directly into NanoMA's tool library at the end of `nanoma/meta.py`:

```python
_register_optimization_tools()  # adds nanoma.planning.TODO_TOOLS to META_TOOLS
```

Because it augments `META_TOOLS`, the tools flow through every `_all_tools()`
merge site automatically and are present in installed packages without the
`optimizations` directory.

Alternatively, without touching core, inject them per-run:

```python
from nanoma.planning import TODO_TOOLS
config.extra_tools.update(TODO_TOOLS)  # RuntimeConfig.extra_tools is merged in _all_tools()
```

### Disabling

```python
config.disabled_tools |= {"task_create", "task_update", "task_list"}
```

### Test

```bash
cd /data/workspace/NanoMA
python -m optimizations.todo_tools.test_todo_tools
```

## `experiment_ledger` — what the run has measured, and the states it measured

Keeps every `(state, score)` pair the runtime observes, so an agent optimizing
something cannot lose or forget its own results.

| Tool | Purpose |
|------|---------|
| `experiments` | Read the run's measurements newest-first, with the current state marked; pass `restore` with a state id to put a recorded state back into the submission path. |

### Why

`merge_submit`'s keep-best ratchet is driven entirely by the agent's own
`verify` check. A run whose check never passes the submission path therefore has
no memory of what it measured at all — and that is not hypothetical: one run
failed its check on all five merges, then narrowed the scored config to fit a
local shell timeout, deleting the parameters behind the best score the judge had
already recorded for it. It spent its remaining submissions on the narrowed one.

Nothing extra has to be measured to prevent that. The judge's verdict arrives
through the submit tool, and `_calibrate_with_official` already parses it and
already fingerprints the state it applies to. It reduced that pair to one
boolean for calibration and discarded the score. The ledger keeps the pair.

This is deliberately independent of the agent authoring a working check: a run
that cannot measure itself is exactly the run that needs the record.

### What it does

- **Records** each verdict as `{metric, state, round, valid}` appended to
  `_merge/ledger.jsonl` — a file, so the record survives the process.
- **Snapshots** the state behind each new best into `_merge/ledger/<state>/`,
  subject to the same size limits as the keep-best snapshot
  (`NANOMA_MERGE_MAX_MB`, `NANOMA_MERGE_MAX_FILES`).
- **Refuses** a submission that would ship a state worse than one on record, or
  an unmeasured state while a measured better one exists, naming the score to
  beat and where the better state is kept. A verdict the judge marked invalid is
  recorded but never becomes something to fall back to.
- **Reports** the record to the agent through `experiments`, which is the part
  that addresses an agent never learning its own configuration's score.

Only ever refuses on evidence: with nothing measured there is nothing to be
worse than, so a first submission is never blocked.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `NANOMA_SUBMIT_REQUIRE_MEASURED` | `1` | Refuse submissions worse than the record. Inert without a merge target, since there is then no state to fingerprint. |
| `NANOMA_OFFICIAL_LOWER_IS_BETTER` | unset | Set to `1` when the judge's score is minimised, mirroring `verify`'s `higher_is_better`. |

### Test

```bash
cd /data/workspace/NanoMA
python3 -m optimizations.experiment_ledger.test_experiment_ledger
```
