# CF-73 2161F Real OpenDeepThink Benchmark

Run the full OpenDeepThink-style protocol for CF-73 problem `2161F` (SubMST, rating 3000).

Work only under `shared/cf73_top10/`.
Use only public problem files:

- `shared/cf73_top10/problems/2161F/metadata.json`
- `shared/cf73_top10/problems/2161F/statement.md`
- `shared/cf73_top10/README.md`

Rules:

- Do not use web search, local judge, hidden tests, Polygon packages, official solutions, checkers, interactors, or generated-test commands during solution selection.
- Shell is disabled for agents. Do not try to simulate tests by writing helper scripts.
- Use C++17.
- Follow `n=20`, `K=4`, `T=3`, `M=10` unless runtime budget exhaustion forces an explicitly incomplete report.
- Do not shrink this into a mini/canary task. Missing protocol calls must be recorded as incomplete, not silently replaced.

Protocol:

1. Root/coordinator reads the public statement and metadata.
2. Spawn 20 independent gen-0 generator agents in parallel for candidate IDs `00..19`.
   - Each generator writes `shared/cf73_top10/2161F/gen0/candidate_<ID>.cpp`.
   - Each generator writes `shared/cf73_top10/2161F/gen0/candidate_<ID>.md` with core idea, complexity, and risks.
   - Generators compact/stop with tags `benchmark:cf73`, `problem:2161F`, `phase:gen0`, `role:generator`, and `candidate:<ID>`.
3. For each evolution generation `t=1..3`:
   - Query generator/mutation peers and inspect artifacts.
   - Build 40 unique pairwise comparisons over the current 20 candidates, with degree `K=4`.
   - Spawn comparison agents in parallel. Each comparison writes JSON under `shared/cf73_top10/2161F/gen<t>/comparisons/`.
   - Aggregate comparison JSON with `bt_aggregate` into `shared/cf73_top10/2161F/gen<t>/bt.json`.
   - Preserve top 5 candidates as elites, discard bottom 5, and spawn 15 mutation agents for the top 15 candidates.
   - Write the next population under `shared/cf73_top10/2161F/gen<t>/population/`.
4. Run final selection:
   - Build 100 unique pairwise comparisons over the generation-3 population, with degree `M=10`.
   - Spawn final comparison agents in parallel and write JSON under `shared/cf73_top10/2161F/final/comparisons/`.
   - Aggregate with `bt_aggregate` into `shared/cf73_top10/2161F/final/bt.json`.
5. Select the top solution and write:
   - `shared/cf73_top10/2161F/selected_solution.cpp`
   - `shared/cf73_top10/2161F/report.md`

Coordination requirements:

- Use tags including `benchmark:cf73`, `problem:2161F`, `phase:<phase>`, and `role:<role>`.
- Query peer state and memory by `group_id` and `problem:2161F` tags before each synthesis step.
- Prefer `spawn_many()` for peer waves. Root should coordinate/query/aggregate/synthesize, not write candidate algorithms itself unless recording an incomplete final report after budget exhaustion.
- Before stopping, submit the selected solution and report if they exist.
