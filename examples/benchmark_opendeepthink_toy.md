# OpenDeepThink Toy Benchmark

Run a scaled-down OpenDeepThink-style population search on one objective programming problem.

Work only under `shared/opendeepthink_toy/`.

## Programming Problem

Implement a Python module exposing:

```python
def count_stable_windows(values: list[int], limit: int) -> int:
    ...

def solve(inp: str) -> str:
    ...
```

`count_stable_windows(values, limit)` returns the number of contiguous non-empty subarrays whose maximum value minus minimum value is at most `limit`.

Constraints:

- `0 <= len(values) <= 200000`
- values may be negative or positive integers
- `limit >= 0`
- The return value may exceed 32-bit range
- Expected complexity: `O(n)` or `O(n log n)`; quadratic solutions are not acceptable

`solve(inp)` parses:

```text
n limit
v1 v2 ... vn
```

and returns the count plus a trailing newline. If `n == 0`, the second line may be absent and the answer is `0`.

## Required Mini OpenDeepThink Protocol

Do not use tests, brute force validation, or local execution to select the winner. Use local shell only for Bradley-Terry aggregation math if needed.

Use this reduced protocol:

1. Spawn four independent candidate agents in parallel.
   - Each candidate writes `shared/opendeepthink_toy/candidates/candidate_<N>.py`.
   - Each candidate also writes a brief rationale file.
   - Candidate tags must include `opendeepthink-toy`, `phase:generate`, and `candidate:<N>`.
2. Run pairwise comparison across the initial population.
   - Compare all six unordered candidate pairs.
   - Prefer separate judge agents where useful.
   - Each comparison writes `shared/opendeepthink_toy/comparisons/round0_<A>_vs_<B>.json` with winner, loser, and critiques.
3. Aggregate pairwise outcomes using Bradley-Terry or a clearly documented BT-compatible approximation.
   - Write `shared/opendeepthink_toy/bt_round0.json`.
   - Preserve the top-ranked candidate as an elite.
   - Discard the bottom-ranked candidate.
4. Mutate/refine the top three candidates in parallel using their pairwise critiques.
   - Each refiner writes `shared/opendeepthink_toy/refined/refined_<N>.py`.
   - Each refiner must state which critiques informed the mutation.
5. Run a final pairwise comparison round over the retained/refined pool.
   - Write final comparison JSON files and `shared/opendeepthink_toy/bt_final.json`.
6. Select one final module and copy it to `shared/opendeepthink_toy/selected_solution.py`.
   - Do not run tests before selection.
   - Write `shared/opendeepthink_toy/report.md` explaining population size, comparisons, BT ranking, mutation decisions, and selected candidate.

Completion:

- Submit `shared/opendeepthink_toy/selected_solution.py`.
- Submit `shared/opendeepthink_toy/report.md`.
- Finish with a concise result describing whether the mini OpenDeepThink protocol completed.
