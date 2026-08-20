# DeepSeek EdgeBench spawn-only fork

This branch derives from the NanoMA source and runner used by the most recent
completed DeepSeek EdgeBench trajectory batches. The three launch scripts stay
byte-for-byte exact; the runtime and runner are deliberately narrowed so the
DeepSeek judge is the only remaining model-facing spawn path.

## Provenance

The runtime source was recovered from the per-task `_nanoma_src` snapshot,
not from the server's later working tree. This branch is specialized to the
DeepSeek batches below; Luna's direct-spawn path is intentionally excluded.

| Batch | Model route | Task invocations | Started |
| --- | --- | ---: | --- |
| `edgebench_flt_branchstore_cpu4_20260802_103900` | `deepseek-v4-pro` (DeepSeek V4 Pro Preview) | 1 | 2026-08-02 |
| `edgebench_low11_branchstore_cpu4_20260802_210700` | `deepseek-v4-pro` (DeepSeek V4 Pro Preview) | 11 | 2026-08-02 |
| `edgebench_low7_branchstore_cpu4_20260803_230623` | `deepseek-v4-pro` (DeepSeek V4 Pro Preview) | 7 | 2026-08-03 |

There are 19 task invocations and 18 unique tasks because
`flt_regular_formalization` appears in both the dedicated FLT run and the
low7 run.

## DeepSeek spawn contract

1. A worker reaches a fresh planning node by calling `task_create`.
2. The runtime sends the parent's current context to the configured DeepSeek
   judge route.
3. A declined decision creates an ordinary local task. An accepted decision
   suppresses that local task and returns a structured child-task split.
4. The runtime invokes internal `meta_spawn` once per approved child, forcing
   the child onto the same DeepSeek route.

The worker never sees `spawn`, `spawn_many`, or `task_spawn`. A non-DeepSeek
judge route is rejected before it can make a topology decision.

## Files

- `run_nanoma_edgebench.py` is the trajectory runner with its worker prompt
  clarified for the DeepSeek judge-only spawn contract.
- `launch_snapshots/` contains the three unmodified server launch scripts.
  They intentionally retain their original run IDs, absolute server paths,
  resource limits, and environment-variable names for auditability.
- The repository's runtime began from the source snapshot staged into those
  containers, then removed direct-spawn tool registration, `spawn_many`,
  `task_spawn`, the alternative task-spawn gate, and Opus spawn experiments.

The launch scripts read credentials from `.env`; no `.env`, credential value,
trajectory log, benchmark data, judge output, or result file is included in
this branch.

## Integrity

Current runner SHA-256 followed by the unmodified launch-script SHA-256 values:

```text
20ef98af79419db91885e8ea825b7a79ef764fb2a1e8d5d868c946dd119437d4  run_nanoma_edgebench.py
38eb3597510a045d5ce904e46013a56562b2c0d83f5b5463bfd1573281e2ecc7  launch_snapshots/edgebench_flt_branchstore_cpu4_20260802_103900.sh
ef19e6f302ee366709f8436c6c32c1d578dc0b8cdc1a2c506af6f6df05a3ebb0  launch_snapshots/edgebench_low11_branchstore_cpu4_20260802_210700.sh
3cbbe16dd794d3762fa117b8f0d7efb063b2e8ce0ea6b76aee87e0a97b88f16c  launch_snapshots/edgebench_low7_branchstore_cpu4_20260803_230623.sh
```

For provenance, the deterministic digest of the unmodified 103-file source
snapshot in the parent commit is
`c75e819e90f1b3da55857d9594b1ae3f274af8a13eb627038f6c8bca9e794dd4`.
The current branch intentionally differs only to isolate DeepSeek spawn
behavior.

## Experimental settings captured by the launchers

- 7,200-second task wall time with a 90-second final safety margin
- 4 CPU / 16 GiB work container and 4 CPU / 8 GiB judge container
- spawn todolist judge enabled with the same DeepSeek model route
- direct `spawn`, `spawn_many`, and `task_spawn` tools excluded from the model
- internal `meta_spawn` retained only as the judge's child-creation executor
- branch delivery merge enabled
- verified submission required
- learned tool policy disabled
- SForge auto-evaluation every 300 seconds and submission cooldown of 120
  seconds
