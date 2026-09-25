# EdgeBench adapter

`run_nanoma_edgebench.py` connects the NanoMA runtime to an official EdgeBench task workspace. It reads the task prompt, exposes benchmark-provided verification and submission commands to the agent, converts visible score feedback into an online objective, retains measured candidates, and performs a final submission before the task deadline.

The adapter configures effectively unbounded topology limits; provider concurrency remains controlled independently through `NANOMA_MAX_CONCURRENT_LLM`. Dynamic child creation follows the same-model planning gate implemented by the runtime. The adapter does not contain benchmark tasks, judge code, result files, model-provider endpoints, or credentials.

## Usage

```bash
python benchmarks/edgebench/run_nanoma_edgebench.py \
  --prompt-file /path/to/task_prompt.txt \
  --task-cwd /path/to/official/task_workspace \
  --workspace /path/to/nanoma_workspace \
  --log-dir /path/to/logs \
  --model <model-id> \
  --wall-seconds 7200
```

The official task workspace must provide the benchmark's own tools, including its verification and submission interface. Provider configuration is read from the normal NanoMA environment variables. Optional `NANOMA_EDGE_*`, `SFORGE_SCORE_DIRECTION`, and `SFORGE_SELECTION_POLICY` variables control deadline handling and candidate selection; consult the runner source for defaults.
