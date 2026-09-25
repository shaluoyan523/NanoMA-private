# RepoZero Py2JS Hard adapter

`run_nanoma_repozero_hard_small.py` evaluates NanoMA on the RepoZero Py2JS Hard tasks. The adapter prepares isolated task workspaces, constructs delivery contracts, runs the multi-agent runtime, validates complete candidate trees, and records benchmark scores and runtime statistics.

RepoZero itself, its task data, and its official evaluation dependencies must be installed separately. The adapter contains no provider endpoint or credential.

## Usage

```bash
python benchmarks/repozero/run_nanoma_repozero_hard_small.py \
  --run-dir /path/to/output \
  --repozero-root /path/to/repozero \
  --model <model-id> \
  --tasks all \
  --time-limit 7200 \
  --max-concurrent-llm 50
```

Use `--smoke-only` to validate model connectivity before starting an evaluation. Credentials are read from the normal NanoMA environment. The optional process-inheritance flag should only reference a process already authorized by the evaluator and never replaces repository-level secret hygiene.
