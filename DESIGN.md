# NanoMA Design Principles

## The One Rule: Orthogonal Minimalism

> **Shell is the universal escape hatch. A dedicated tool exists only when it provides
> an interaction model that shell cannot reliably replicate.**

### Admission Criteria (satisfy ANY ONE to keep a tool)

1. **Reliability** — Shell cannot do this reliably due to escaping, quoting, or state issues.
   (e.g., `ws_create_file` — file content with backticks/dollars breaks heredocs)

2. **Atomicity** — The operation requires multi-step transactional semantics.
   (e.g., `ws_replace_string` — 4-tier matching cascade is impossible in sed)

3. **Coordination** — The operation requires access to runtime internals (other agents, budgets, queues).
   (e.g., `task_create`, `send`, `wait` — these operate on runtime state, not the filesystem)

### Elimination Criterion

If `shell("one-liner")` produces equally usable output → the tool is redundant → remove it.

### Cost Formula

```
Each tool ≈ 100 tokens of schema × N agents = N×100 tokens/turn
Fewer tools → less deliberation overhead → faster, cheaper, more reliable decisions
```

---

## Tool Architecture (3 Layers)

```
Layer 3: META (15 model-facing core tools)
  Per-node planning, coordination, lifecycle, and artifacts.
  task_create, task_update, task_list, kill, send, deliver_to_parent, query,
  wait, transfer, set_bio, get_cost, set_status, rebirth, submit, batch

  Child creation is internal. The model does not receive spawn/spawn_many.

Layer 2: WORKSPACE (9 tools)
  Structured operations where shell fails at reliability/atomicity.
  read_file, create_file, append_file, replace_string, multi_replace, apply_patch, grep, code_outline, read_symbol

Layer 1: SHELL (1 tool)
  Universal primitive. Anything not covered above.
  shell(command, timeout)
```

### Why these 9 workspace tools?

| Tool | Why shell can't | Layer justification |
|------|----------------|---------------------|
| `ws_read_file` | Paginated line-numbered reading with structural metadata | Structured I/O |
| `ws_create_file` | Content with `` ` ``, `$`, `\`, `EOF` breaks heredocs | Reliability |
| `ws_append_file` | Same escaping issues as create | Reliability |
| `ws_replace_string` | 4-tier cascading match (exact→trimmed→indent→normalized) | Atomicity |
| `ws_multi_replace` | All-or-nothing batch replacement | Atomicity |
| `ws_apply_patch` | V4A context-based multi-file patching | Atomicity |
| `ws_grep` | Structured JSON results with match positions, auto-ignore dirs | Structured I/O |
| `ws_code_outline` | Symbol tree with block boundaries (no CLI equivalent) | Structured I/O |
| `ws_read_symbol` | Precise extraction using outline knowledge | Structured I/O |

### What agents use shell for (explicitly)

These were previously dedicated tools, now delegated to shell:

```bash
mkdir -p path              # was: ws_create_directory
rm -rf path               # was: ws_delete_file
mv old new                # was: ws_rename_file
ls -la path               # was: ws_list_dir
find . -name '*.py'       # was: ws_file_search
tree -L 3                 # was: ws_project_structure
```

---

## Multi-Agent Coordination Design

The meta tools implement **uniform node infrastructure** — every node has the
same planning and coordination capabilities. The topology emerges from planning
decisions made at the nodes, not from a benchmark script or a central planner.

### Composability Examples

| Pattern | Composition |
|---------|-------------|
| Orchestrator-Workers | planning split → children → `wait(mode="all")` → aggregate |
| Map-Reduce | planning split → parallel map nodes → `wait` → reduce |
| Streaming Pipeline | successive planning nodes → `send(mode="steer")` chain |
| Debate/Tournament | method-portfolio split → `query(messages=N)` → compare → `kill` losers |
| Event-Driven Service | `set_status("idle")` → `send` wakes → process → `set_status("idle")` |
| Hierarchical Delegation | a child reaches `task_create` and makes another split |
| Iterative Refinement | planning split → `wait` → evaluate → `send` feedback → repeat |

### Key Insight

One recursively available planning unit plus coordination primitives can form
many dynamic topologies. The prompt supplies the task intent; each node decides
how its next phase should be structured, while the runtime enforces the actual
child-creation boundary.
