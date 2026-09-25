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
   (e.g., `spawn`, `send`, `wait` — these operate on the runtime graph, not the filesystem)

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
Layer 3: META (12 tools)
  Coordination primitives operating on runtime internals.
  spawn, kill, send, query, wait, transfer, set_bio, get_cost, set_status, rebirth, submit, batch

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

The 12 meta tools implement **uniform infrastructure** — every agent has identical capabilities.
The orchestration pattern emerges entirely from the task prompt.

### Composability Examples

| Pattern | Composition |
|---------|-------------|
| Orchestrator-Workers | `spawn` × N → `wait(mode="all")` → aggregate |
| Map-Reduce | `spawn` × N → `wait(mode="all")` → `transfer` → reduce |
| Streaming Pipeline | `spawn` → `send(mode="steer")` chain |
| Debate/Tournament | `spawn` × N → `query(messages=N)` → judge → `kill` losers |
| Event-Driven Service | `set_status("idle")` → `send` wakes → process → `set_status("idle")` |
| Hierarchical Delegation | `spawn(delegate=True)` cascading |
| Iterative Refinement | `spawn` → `wait` → evaluate → `send` feedback → repeat |

### Key Insight

12 primitives × free composition = unbounded topologies.
No framework code needed — the prompt IS the orchestration logic.
