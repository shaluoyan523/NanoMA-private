# EdgeBench trajectory snapshot

This directory preserves the exact NanoMA runner and launch scripts used for
the most recent completed EdgeBench trajectory batches on the experiment
server.

## Provenance

The runtime source was recovered from the per-task `_nanoma_src` snapshot,
not from the server's later working tree. The runner and source snapshot are
identical across the DeepSeek batches below and the Luna batch
`edgebench_luna_parallel5_2h_cpu4_20260805_201558`.

| Batch | Model route | Task invocations | Started |
| --- | --- | ---: | --- |
| `edgebench_flt_branchstore_cpu4_20260802_103900` | `deepseek-v4-pro` (DeepSeek V4 Pro Preview) | 1 | 2026-08-02 |
| `edgebench_low11_branchstore_cpu4_20260802_210700` | `deepseek-v4-pro` (DeepSeek V4 Pro Preview) | 11 | 2026-08-02 |
| `edgebench_low7_branchstore_cpu4_20260803_230623` | `deepseek-v4-pro` (DeepSeek V4 Pro Preview) | 7 | 2026-08-03 |

There are 19 task invocations and 18 unique tasks because
`flt_regular_formalization` appears in both the dedicated FLT run and the
low7 run.

## Files

- `run_nanoma_edgebench.py` is the exact runner installed in every SForge
  work container.
- `launch_snapshots/` contains the three unmodified server launch scripts.
  They intentionally retain their original run IDs, absolute server paths,
  resource limits, and environment-variable names for auditability.
- The repository's `nanoma/`, `optimizations/`, `models.yaml`, and
  `pyproject.toml` files have been updated to the exact source snapshot that
  those launch scripts staged into each container.

The launch scripts read credentials from `.env`; no `.env`, credential value,
trajectory log, benchmark data, judge output, or result file is included in
this branch.

## Integrity

SHA-256 values copied from and rechecked against the server:

```text
07c3916a0c2d6b00e738838b18377c86fd6281c0243bd2056fe3b72f5e56092f  run_nanoma_edgebench.py
38eb3597510a045d5ce904e46013a56562b2c0d83f5b5463bfd1573281e2ecc7  launch_snapshots/edgebench_flt_branchstore_cpu4_20260802_103900.sh
ef19e6f302ee366709f8436c6c32c1d578dc0b8cdc1a2c506af6f6df05a3ebb0  launch_snapshots/edgebench_low11_branchstore_cpu4_20260802_210700.sh
3cbbe16dd794d3762fa117b8f0d7efb063b2e8ce0ea6b76aee87e0a97b88f16c  launch_snapshots/edgebench_low7_branchstore_cpu4_20260803_230623.sh
```

The deterministic digest of the 103-file source snapshot is
`c75e819e90f1b3da55857d9594b1ae3f274af8a13eb627038f6c8bca9e794dd4`.
It was computed by sorting relative file names, hashing each non-bytecode
file, and hashing the resulting manifest.

## Experimental settings captured by the launchers

- 7,200-second task wall time with a 90-second final safety margin
- 4 CPU / 16 GiB work container and 4 CPU / 8 GiB judge container
- spawn todolist judge enabled with the same DeepSeek model route
- branch delivery merge enabled
- verified submission required
- learned tool policy disabled
- SForge auto-evaluation every 300 seconds and submission cooldown of 120
  seconds
