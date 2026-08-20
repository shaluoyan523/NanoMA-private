#!/usr/bin/env bash
# Experiment: "spawn decided by the worker model at the todolist node".
#   - NANOMA_SPAWN_TODOLIST_JUDGE=1: raw spawn AND task_spawn are removed from the
#     worker. When the worker creates pending tasks (the planning node), the
#     runtime makes ONE call with that same model to decide whether
#     to parallelize and how to split subagent tasks, then spawns children itself.
#     If the decision call fails, the node degrades gracefully to "no spawn".
#   - multi-agent allowed; todolist stays per-agent.
#
# Usage:  bash optimizations/experiments/run_gaia10_spawn_judge.sh [RUN_ID]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env
  set +a
fi

export NANOMA_SPAWN_TODOLIST_JUDGE=1

RUN_ID="${1:-gaia10_spawn_judge_$(date +%Y%m%d-%H%M%S)}"
BASE_DIR="$ROOT/runs/$RUN_ID"
mkdir -p "$BASE_DIR"

MODEL="${NANOMA_MODEL:-deepseek-v4-pro}"
SNAPSHOT="/data_storage/nanoma/benchmarks/gaia/data"

{
  echo "run_id=$RUN_ID"
  echo "base_dir=$BASE_DIR"
  echo "model=$MODEL"
  echo "mode=JUDGE (the same model decides spawn + task split at the planning node)"
  echo "mode_detail=multi_agent(max_agents=5,max_depth=1,max_concurrent_llm=4), todolist per-agent"
  echo "tasks=GAIA L2 offset=0 limit=10"
  echo "per_task_time_limit=600s max_turns=60 budget=50"
} | tee "$BASE_DIR/run.info"

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
echo "logs: $BASE_DIR/run.out | judge events: grep spawn_judge_decision $BASE_DIR/logs/*/events.jsonl"
