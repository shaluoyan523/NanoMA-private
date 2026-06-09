# Dynamic DAG Replication Benchmark

You are the root coordinator. Replicate the benchmark topology below using dynamic agent orchestration.

Do not solve the five node tasks yourself. Your job is to infer the dependency graph, spawn only currently-unlocked nodes, wait for completions, query peer state, and synthesize the final report.

## Target Topology

- Node A: no dependencies.
- Node B: no dependencies.
- Node C: depends on A.
- Node D: depends on A and B.
- Node E: depends on C and D.

Expected execution structure:

1. Spawn A and B in the first wave, in parallel.
2. When A is done, C becomes eligible.
3. D must not start until both A and B are done.
4. E must not start until both C and D are done.
5. The root agent must query node state/memory before each unlock decision and before final synthesis.

## Node Work

Each node agent must:

- Use `set_status(action="work", current_task_tags=[...])` when it starts.
- Write one file under `shared/dag_replication/`:
  - A writes `a_requirements.md`
  - B writes `b_constraints.md`
  - C writes `c_design.md`
  - D writes `d_risk_review.md`
  - E writes `e_final_package.md`
- Include in its output which dependency files it read.
- Call `submit(path=...)`.
- Finish with `compact(..., stop_after=true, result="<NODE> done")`.
- Use tags that include `benchmark:dag-replication`, `node:<letter>`, and its phase.

## Root Final Work

The root must write `shared/dag_replication/root_assessment.md` with:

- The dependency graph it inferred.
- The actual launch order it used.
- Which query calls informed each unlock decision.
- Whether the observed structure matches the target topology.

Then submit `shared/dag_replication/root_assessment.md` and set status done.
