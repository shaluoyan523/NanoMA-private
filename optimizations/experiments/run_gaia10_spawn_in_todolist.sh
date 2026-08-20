#!/usr/bin/env bash
# Experiment: "spawn folded into the todolist".
#   - NANOMA_SPAWN_TODOLIST_GATE=1  -> raw spawn/spawn_many removed; delegation
#     happens ONLY via task_spawn(task_id), and task_spawn is offered ONLY while
#     the agent has a pending task (the planning moment).
#   - multi-agent allowed (max_agents>1, max_depth=1) so task_spawn can delegate.
#   - todolist stays strictly PER-AGENT (children get their own fresh list).
#   - same 10 GAIA L2 tasks / model as the spawn-banned baseline, for comparison.
#
# Usage:  bash optimizations/experiments/run_gaia10_spawn_in_todolist.sh [RUN_ID]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env
  set +a
fi

export NANOMA_SPAWN_TODOLIST_GATE=1

RUN_ID="${1:-gaia10_spawn_in_todolist_$(date +%Y%m%d-%H%M%S)}"
BASE_DIR="$ROOT/runs/$RUN_ID"
mkdir -p "$BASE_DIR"

MODEL="${NANOMA_MODEL:-deepseek-v4-pro}"
SNAPSHOT="/data_storage/nanoma/benchmarks/gaia/data"

{
  echo "run_id=$RUN_ID"
  echo "base_dir=$BASE_DIR"
  echo "model=$MODEL"
  echo "gate=NANOMA_SPAWN_TODOLIST_GATE=1 (raw spawn removed; delegate via task_spawn only in pending window)"
  echo "mode=multi_agent(max_agents=5,max_depth=1,max_concurrent_llm=4)"
  echo "todolist=per-agent (not shared)"
  echo "tasks=GAIA L2 offset=0 limit=10"
  echo "per_task_time_limit=600s max_turns=60 budget=50"
} | tee "$BASE_DIR/run.info"

# NOTE: spawn is NOT passed to --disable-tool; the gate removes it at runtime.
setsid nohup python3 benchmarks/gaia/run_nanoma_gaia_l2.py \
  --model "$MODEL" \
  --gaia-config 2023_level2 \
  --snapshot-dir "$SNAPSHOT" \
  --data-dir "$SNAPSHOT" \
  --offset 0 --limit 10 \
  --max-agents 5 --max-depth 1 --max-concurrent-llm 4 \
  --time-limit 600 --max-turns 60 --budget 50 \
  --results-dir "$BASE_DIR/nanoma_l2" \
  --workspace-root "$BASE_DIR/workspace" \
  --log-root "$BASE_DIR/logs" \
  --clean \
  > "$BASE_DIR/run.out" 2> "$BASE_DIR/run.err" < /dev/null &

echo $! > "$BASE_DIR/run.pid"
echo "Launched pid=$(cat "$BASE_DIR/run.pid")"
echo "logs: $BASE_DIR/run.out | events: $BASE_DIR/logs/<task_id>/events.jsonl"
