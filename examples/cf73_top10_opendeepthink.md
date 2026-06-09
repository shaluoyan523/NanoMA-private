# CF-73 Top-10 Hardest OpenDeepThink Benchmark

Run NanoMA on the 10 hardest CF-73 problems by official Codeforces rating.

Work only under `shared/cf73_top10/`.
Use only public problem statements and public metadata. Do not inspect, request,
infer from, or depend on Polygon packages, official solutions, checkers,
interactors, hidden tests, generated-test commands, or private judge metadata.

## Problem Set

Use the problem statements and metadata staged in `shared/cf73_top10/problems/`.
Each problem directory is named `<contest><index>` and contains:

- `metadata.json`
- `statement.md`

The selected problems are:

1. `2174E1` rating 3100, Game of Scientists (Version 1)
2. `2147G` rating 3100, Modular Tetration
3. `2138E1` rating 3100, Determinant Construction (Easy Version)
4. `2138E2` rating 3100, Determinant Construction (Hard Version)
5. `2161F` rating 3000, SubMST
6. `2158F2` rating 3000, Distinct GCDs (Hard Version)
7. `2156F2` rating 3000, Strange Operation (Hard Version)
8. `2164F2` rating 2900, Chain Prefix Rank (Hard Version)
9. `2162H` rating 2900, Beautiful Problem
10. `2146F` rating 2900, Bubble Sort

## Required Protocol Per Problem

Use the paper setting `n=20`, `K=4`, `T=3`, `M=10`. This is about 285 LLM calls
per problem:

- 20 initial candidate generations.
- For each of 3 evolution generations: 40 pairwise comparisons plus mutation of
  the top 15 candidates after BT ranking.
- One final round of 100 pairwise comparisons over the generation-3 population.

All generation, comparison, and mutation calls inside the same round should be
issued in parallel whenever possible.

For each problem:

1. Read only the public `statement.md` and `metadata.json`.
2. Generate 20 independent C++17 candidate solutions in parallel.
   - Write candidates to `shared/cf73_top10/<problem>/gen0/candidate_<00-19>.cpp`.
   - Write a short rationale for each candidate.
3. For each evolution generation `t=1..3`:
   - Build a random undirected comparison graph where every candidate has degree
     `K=4`; this yields 40 unique pairwise comparisons and no self-pairs.
   - Run the 40 pairwise comparisons in parallel with randomized presentation
     order. Each comparison JSON must include candidate IDs, presentation order,
     winner (`a`, `b`, or `tie`), and critique text for both candidates.
   - Write comparison JSON files under
     `shared/cf73_top10/<problem>/gen<t>/comparisons/`.
   - Use `bt_aggregate` to write
     `shared/cf73_top10/<problem>/gen<t>/bt.json`.
   - Preserve the top 5 candidates unchanged as elites, discard the bottom 5,
     and mutate the top 15 candidates in parallel using their aggregated
     comparison critiques. The next population is the 5 elites plus 15 mutations.
   - Write the next population under
     `shared/cf73_top10/<problem>/gen<t>/population/`.
4. Run the final selection round over the generation-3 population:
   - Build a random comparison graph where every candidate has degree `M=10`;
     this yields 100 unique pairwise comparisons.
   - Run the 100 pairwise comparisons in parallel and write JSON files under
     `shared/cf73_top10/<problem>/final/comparisons/`.
   - Use `bt_aggregate` to write
     `shared/cf73_top10/<problem>/final/bt.json`.
5. Select one final C++17 solution.
   - Write `shared/cf73_top10/<problem>/selected_solution.cpp`.
   - Write `shared/cf73_top10/<problem>/report.md`.

Do not run local tests or use the judge to choose the winner. The judge is private
external scoring after selection. For interactive problems, produce a standard
interactive C++17 solution that communicates via stdin/stdout and flushes output
after every query.

For every `compact`, `set_status`, or `submit`, use problem-specific tags such as
`benchmark:cf73`, `problem:2147G`, `phase:gen1-compare`, `role:judge`, and
`artifact:selected-solution`. Before final synthesis, query peer/coordinator
state by `group_id` and `problem:<id>` tags rather than relying on parent reports.

## Completion

Write a top-level `shared/cf73_top10/summary.md` listing each problem, selected file, final BT ranking, and any incomplete protocol steps.
