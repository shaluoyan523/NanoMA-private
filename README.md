# NanoMA

[中文版](README_zh.md)

## Node-Autonomous Dynamic Multi-Agent Orchestration

> Anonymous artifact for a paper under double-blind review. Author names, affiliations, contact information, repository links, and citation metadata are intentionally withheld.

Long-horizon tasks rarely preserve the decomposition imagined at the beginning of a run. New dependencies appear, evidence from one branch changes the value of another, and some branches become redundant. NanoMA treats this changing collaboration structure as part of the problem-solving state rather than as a workflow fixed before execution.

NanoMA is a multi-agent ReAct framework in which **every agent acts as both planner and executor**. At each step, a node either advances its local task through an execution action or changes the active collaboration structure through one of four topology actions:

- `spawn`: insert a new agent node for a bounded subtask;
- `kill`: terminate a branch whose remaining value no longer justifies continued execution;
- `query`: read non-local execution state or evidence on demand; and
- `send`: deliver evidence, constraints, progress, or revised instructions to a selected agent.

The central idea is not merely recursive delegation. NanoMA separates the fixed provenance of a task from the collaboration relations that become useful later. A child remembers where its assignment came from, but that ancestry does not determine whom it may inspect, inform, or coordinate with during execution.

## From fixed workflows to an evolving execution graph

The paper distinguishes three organizational patterns:

| Pattern | Where planning lives | How collaboration changes |
| --- | --- | --- |
| Central planning | A controller decomposes work, routes dependencies, and aggregates results | Changes are mediated by the controller |
| Node recursion | Each node may decompose its own assignment and aggregate descendants | Expansion is distributed, but interaction remains largely parent-child |
| **NanoMA** | **Every active node plans and executes** | **Nodes can expand, inspect, update, and contract collaboration during execution** |

At time \(t\), NanoMA represents execution as a typed dynamic graph

$$
G_t = (V_t, E_t^{\mathrm{prov}}, E_t^{\mathrm{comm}}),
$$

where \(V_t\) is the set of instantiated agents, \(E_t^{\mathrm{prov}}\) contains immutable creation edges, and \(E_t^{\mathrm{comm}}\) contains directed information edges established during execution. Provenance edges record task origin; they do not restrict later communication.

Each node \(v_i\) maintains an independent state

$$
s_i^t = (\tau_i^t, h_i^t, Q_i^t, W_i^t, \ell_i^t, b_i^t, O_i^t, I_i^t),
$$

consisting of its current assignment, effective interaction history, dynamic local task queue, private workspace, lifecycle state, resource state, results and artifacts, and priority-ordered message inboxes. A newly created child receives its own context, workspace, and task queue, then decides locally how to execute or further decompose its assignment.

## Topology actions as graph operations

The four topology actions form a small data-structure interface over the execution graph.

| Data-structure role | Topology action | Direct effect |
| --- | --- | --- |
| Insertion | `spawn` | Adds a node to \(V_t\) and a provenance edge to \(E_t^{\mathrm{prov}}\) |
| Deletion | `kill` | Removes a selected branch from active execution while retaining its recorded history and delivered evidence |
| Lookup | `query` | Reads visible node state without modifying the target or graph |
| Update | `send` | Updates the recipient inbox and realizes a directed edge in \(E_t^{\mathrm{comm}}\) |

These actions compose into larger organizations. `spawn + wait` expresses recursive delegation; spawning without immediately waiting allows parent and child to work in parallel; `query + send` connects evidence across branches and levels; and `kill` contracts the active frontier after useful information has been preserved. The resulting graph may contain stars, chains, deep recursive paths, trees, dense subgraphs, wheels, or directed cycles without selecting a topology template in advance.

### How `spawn` is implemented in this artifact

`spawn` is the conceptual node-insertion action. In the current same-model profile it is not exposed as an unrestricted model-facing schema. Instead, an agent creates a bounded local work item with `task_create`; at that fresh planning point, **SpawnJudge uses the same node's model and context** to decide whether the item should remain local or become one or more child nodes. Runtime admission then checks executable constraints such as model availability, memory, depth, node count, and policy. The insertion primitive itself remains available at every eligible node, so planning authority is distributed even though node creation is guarded.

This scaffold addresses a current model limitation: LLMs often create broad role labels, duplicate branches, or recurse without a usable output contract. The guard constrains the quality and admissibility of an individual expansion; it does not generate a global plan or choose the full graph in advance.

## Per-agent execution loop

Every active node follows the same loop:

1. Observe local tool results, incoming messages, and any visible nodes needed for the current decision.
2. Maintain a node-local plan through `task_create`, `task_update`, and `task_list`.
3. Execute locally or, when SpawnJudge and Runtime admission agree, create a child with an independent local loop.
4. Use `query` and `send` when non-local evidence changes the current plan.
5. Continue independent work, synchronize through `wait`, or prune a redundant descendant through `kill`.
6. Deliver a supported result and terminate only after the local output contract is satisfied.

The four topology actions remain the method-level abstraction. Planning, synchronization, delivery, validation, lifecycle, and workspace interfaces are engineering adapters that make those actions executable and durable.

## Reliable collaboration and publication

NanoMA separates collaboration from final publication:

- **Targeted communication.** `query` reads state on demand; `send` changes a selected recipient's information state. Messages can be queued, injected between tool groups, or delivered immediately.
- **Durable answers.** `deliver_to_parent` records answer, evidence, confidence, method, and source in a candidate ledger instead of relying only on transient conversation history.
- **Private workspaces.** Parallel nodes work on isolated task trees. `transfer` and delivery diffs move changes across workspace boundaries without giving every child direct access to the final target.
- **Verified state promotion.** Validation checks assess merged candidates, roll back explicit regressions, and retain a Runtime-owned best-state snapshot.
- **Root-owned publication.** The root delivery contract stages a complete candidate, checks required files and validators, and replaces the official target atomically. Child completion is therefore not equivalent to final publication.

`kill` also includes a delivery grace path: if a descendant contains material but undelivered evidence, the Runtime first asks it to deliver and defers termination. This preserves useful work while still allowing the active graph to contract.

## Contributions

The paper makes three claims:

1. **Autonomous planning and execution at every node.** NanoMA distributes orchestration decisions and separates immutable task provenance from dynamically formed collaboration relations.
2. **A unified topology-action interface.** `spawn`, `kill`, `query`, and `send` provide insertion, deletion, lookup, and update operations over a runtime execution graph inside the same ReAct loop as task execution.
3. **Empirical evaluation of performance and emergent orchestration.** Experiments compare organizational patterns under matched model backbones, analyze the structures formed in complete traces, and ablate information visibility and message-recipient scope.

## Results

### RepoZero-Py2JS Hard

| Agent framework | Harness type | GPT-5.6 Luna High All-pass | GPT-5.6 Luna High Micro | Claude Sonnet 5 All-pass | Claude Sonnet 5 Micro | Qwen3.7-Plus All-pass | Qwen3.7-Plus Micro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenHands-bash | Single agent | 44.83% | 80.03% | 43.52% | 79.82% | 25.899% | 57.097% |
| CrewAI | Sequential | 53.24% | 75.02% | 36.69% | 77.11% | 28.777% | 67.102% |
| CC-Workflow | Central planning | 48.20% | 74.69% | 24.46% | 52.89% | 24.460% | 54.347% |
| Omnigent | Node recursive | 47.48% | 79.20% | 48.20% | 87.17% | **39.568%** | 72.260% |
| **NanoMA** | **Node autonomous** | **66.91%** | **93.32%** | **51.80%** | **87.30%** | 35.971% | **76.561%** |

### EdgeBench

| Model | Agent framework | Harness type | Overall score |
| --- | --- | --- | ---: |
| GPT-5.6 Luna | CC-Workflow | Central planning | 16.60 |
| GPT-5.6 Luna | CrewAI | Sequential | 19.36 |
| GPT-5.6 Luna | Codex baseline | Single agent | 20.04 |
| GPT-5.6 Luna | Omnigent | Node recursive | 14.31 |
| GPT-5.6 Luna | **NanoMA** | **Node autonomous** | **22.10** |
| Qwen3.7-Plus | CC-Workflow | Central planning | 10.95 |
| Qwen3.7-Plus | CrewAI | Sequential | 13.09 |
| Qwen3.7-Plus | Codex baseline | Single agent | 17.52 |
| Qwen3.7-Plus | Omnigent | Node recursive | 13.06 |
| Qwen3.7-Plus | **NanoMA** | **Node autonomous** | **18.58** |

## Limitations

Current LLMs do not reliably anticipate the delayed and cross-agent consequences of topology actions. They can underestimate coordination cost, destroy useful parallelism, terminate a branch too early, or create poorly scoped and duplicated work. Static prompt guidance is not sufficient to make these choices optimal. The paper therefore treats learned action-selection policies with Runtime feedback as an important next step.

The structural-convergence argument also depends on explicit assumptions: each node creates finitely many smaller obligations, accepted progress reduces unfinished work, and stagnant nodes are eventually returned to useful execution or terminated. It establishes termination of the active frontier under those conditions, not answer correctness.

## Repository map

| Path | Role in the method |
| --- | --- |
| `nanoma/core.py` | Node state, per-agent ReAct loop, Runtime state, planning gates, lifecycle transitions, and checkpointing |
| `nanoma/meta.py` | Topology actions and their communication, synchronization, and delivery adapters |
| `optimizations/todo_tools/` | Node-local task queues and the planning point used by SpawnJudge |
| `nanoma/delivery.py` | Delivery contracts, structured candidates, and frozen artifact trees |
| `nanoma/merge_submit.py` | Candidate ledger, differential merge, validation, best-state retention, and publication |
| `nanoma/online_objective.py` | Parsing and ranking benchmark-visible objective measurements |
| `nanoma/plugins/workspace_tools/` | Structured file creation, reading, editing, and code search |
| `benchmarks/edgebench/` | EdgeBench evaluation adapter |
| `benchmarks/repozero/` | RepoZero-Py2JS Hard evaluation adapter |

## Installation

NanoMA requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Configure a compatible chat-completions endpoint through environment variables. Keep credentials outside the repository.

```bash
export NANOMA_API_KEY="<provider-key>"
export NANOMA_LLM_BASE_URL="https://provider.example/v1"
```

Minimal programmatic use:

```python
import asyncio
from pathlib import Path

from nanoma import Runtime, RuntimeConfig


async def main() -> None:
    runtime = Runtime(
        config=RuntimeConfig(
            default_model="<model-id>",
            workspace_root=Path("./workspace"),
            log_dir=Path("./logs"),
        )
    )
    print(await runtime.run("Your task"))


asyncio.run(main())
```

## Evaluation adapters

The adapters require separately obtained benchmark installations and official task assets.

EdgeBench:

```bash
python benchmarks/edgebench/run_nanoma_edgebench.py \
  --prompt-file /path/to/task_prompt.txt \
  --task-cwd /path/to/edgebench/task \
  --workspace /path/to/workspace \
  --model <model-id> \
  --wall-seconds 7200
```

RepoZero-Py2JS Hard:

```bash
python benchmarks/repozero/run_nanoma_repozero_hard_small.py \
  --run-dir /path/to/output \
  --repozero-root /path/to/repozero \
  --model <model-id> \
  --tasks all
```

Neither adapter embeds model credentials, private endpoints, benchmark data, or judge outputs.

## Verification

```bash
pytest -q \
  tests/test_delivery_contract.py \
  tests/test_online_objective.py \
  tests/test_shell_memory_admission.py
```

The anonymous snapshot reports `57 passed, 1 skipped` for this regression subset.

## Anonymous-review note

The `anonymous` branch is a single parentless commit authored as `Anonymous Authors`. It omits author identities, citation metadata, credentials, private infrastructure addresses, and machine-specific launch scripts. For double-blind review, distribute an archive of this branch or mirror it through an anonymous repository service; a personally owned GitHub URL can reveal the account owner even when branch contents and history are anonymized.

## License

Released under the MIT License. Citation information will be restored after the anonymous review period.
