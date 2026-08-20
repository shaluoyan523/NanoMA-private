# NanoMA

NanoMA is a general-purpose agent runtime in which every node can plan its own
work and dynamically create further multi-agent structure. It can be used for
research, coding, analysis, and artifact-producing tasks without a benchmark
harness.

The model that works on a node also makes that node's planning decision. There
is no separate planner model and no model-specific routing rule.

## How autonomous planning works

1. A node works normally with shell, workspace, and coordination tools.
2. At a meaningful planning point it calls `task_create`.
3. NanoMA gives that node's current context to the same model and asks whether
   the next phase should stay local or be split into parallel child tasks.
4. If the work stays local, the task is added to the node's own task list.
5. If it is delegated, the runtime creates the approved children internally.
   Every child inherits the parent's model and can make the same kind of
   planning decision later.

The model never receives `spawn`, `spawn_many`, or `task_spawn`. Child creation
is a runtime operation resulting from node-level planning, rather than a second
model-facing orchestration API.

## Install

```bash
python -m pip install -e .
export NANOMA_API_KEY="your-api-key"
export NANOMA_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export NANOMA_MODEL="your-model"
```

Python 3.11 or newer is required. NanoMA accepts OpenAI-compatible endpoints;
the existing protocol settings in `nanoma.llm` remain available for other
supported routes.

## Run a general task

```bash
nanoma "Compare three database designs and recommend one" --budget 5
```

Work on an existing project:

```bash
nanoma --project . "Find and fix the failing tests, then explain the cause"
```

Read a longer task from a file or standard input:

```bash
nanoma --task-file task.md
printf '%s\n' "Audit this repository for unsafe file handling" | nanoma - --project .
```

The default internal state is written under `.nanoma/`:

- `.nanoma/workspace/`: private node workspaces and shared deliverables
- `.nanoma/logs/`: `events.jsonl` execution trace

These paths are independent from an optional `--project` target.

## Useful CLI options

```text
--model MODEL                 Working and planning model for the root node
--project PATH                Existing target directory
--budget USD                  Shared run budget
--time-limit SECONDS          Whole-run time limit; 0 means unlimited
--max-agents N                Maximum nodes created during the run
--max-depth N                 Maximum child depth
--max-concurrent-llm N        Concurrent model calls
--max-turns N                 Turn limit per node
--[no-]node-planning          Enable or disable node-level delegation
--instructions TEXT           Extra instructions for all nodes
--instructions-file PATH      Read extra instructions from a file
--disable-tool NAME           Hide a tool; may be repeated
--json                        Return result and run statistics as JSON
--stats                       Print a compact run summary
--quiet                       Suppress live events
```

`--no-node-planning` keeps `task_create` as a local checklist tool but prevents
it from creating children.

## Programmatic API

For most applications, use `run_agent`:

```python
import asyncio
from pathlib import Path

from nanoma import RuntimeConfig, run_agent


async def main():
    config = RuntimeConfig(
        default_model="your-model",
        budget=5.0,
        max_agents=12,
        max_depth=4,
        max_concurrent_llm=4,
        workspace_root=Path(".nanoma/workspace"),
        log_dir=Path(".nanoma/logs"),
        node_autonomous_planning=True,
    )
    run = await run_agent(
        "Improve the parser and add regression tests",
        config=config,
        project_dir=Path("."),
    )
    print(run.result)
    print(run.stats["agents"])


asyncio.run(main())
```

`run_agent` returns an `AgentRun` containing the result, statistics, artifact
paths, and the underlying `Runtime` for deeper inspection. Advanced users can
still instantiate `Runtime` directly.

## Model-facing tools

### Planning and coordination

| Tool | Purpose |
| --- | --- |
| `task_create` | Declare a new planning step; may become local work or an internal child split |
| `task_update` | Update the node's local task status |
| `task_list` | Inspect the node's local plan |
| `send` | Send a message to another node |
| `query` | Discover nodes, status, and available context |
| `wait` | Wait for selected children or peers |
| `kill` | Stop a node that is no longer useful |
| `deliver_to_parent` | Return a child's answer and evidence |
| `transfer` | Copy an artifact between node workspaces |

### Work and lifecycle

| Tool family | Purpose |
| --- | --- |
| `shell` | General system, development, and network commands |
| `ws_*` | Structured file reading, creation, editing, search, and code inspection |
| `submit` | Copy a completed artifact into NanoMA's shared directory |
| `set_status` | Finish or idle a node |
| `get_cost` | Inspect identity, resource use, and remaining budget |
| `set_bio` | Publish a node's role |
| `rebirth` | Restart a node with a compact carried-forward summary |

Optional benchmark adapters may add tools such as an official evaluator. Those
are not required by the general agent and are not part of its default task
contract.

## Architecture boundary

The reusable agent lives in:

- `nanoma/agent.py`: benchmark-independent API
- `nanoma/main.py`: general CLI
- `nanoma/planning.py`: per-node task planning and delegation trigger
- `nanoma/core.py`: runtime, node lifecycle, communication, and internal child creation

`benchmarks/` contains adapters only. In particular, the EdgeBench runner may
add SForge prompts, official submission tools, time guards, and merge policies;
none of those assumptions are present in `run_agent` or the default `nanoma`
command.

## Trace viewer

```bash
python nanoma/viewer.py .nanoma/logs 8900
```

Then open `http://localhost:8900`.

## Security

NanoMA's shell tool runs with the same permissions as the host process. Use a
container, virtual machine, or dedicated account for untrusted tasks. Structured
workspace tools validate paths, but shell commands are not a security boundary.

## Benchmark adapters

The GAIA, Draco, and EdgeBench runners remain under `benchmarks/` for
reproducibility and evaluation. They build on the same runtime but may inject
benchmark-specific prompts and tools. They are examples of adapters, not the
primary NanoMA interface.

## License

MIT. See `LICENSE`.
