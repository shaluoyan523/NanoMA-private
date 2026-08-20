#!/usr/bin/env bash
# Experiment: observe todolist (task_create) triggering under a spawn-banned,
# single-agent NanoMA on 10 GAIA Level-2 tasks with deepseek-v4-pro.
#
#   - spawn / spawn_many banned  -> forces single-agent work, surfaces todolist use
#   - todo_tools registered       -> task_create/update/list + per-turn re-injection
#   - GAIA L2 data from local snapshot (no HF download)
#
# Usage:  bash optimizations/experiments/run_gaia10_todolist_spawnban.sh [RUN_ID]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Load API + HF creds
if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env
  set +a
fi

RUN_ID="${1:-gaia10_todolist_spawnban_$(date +%Y%m%d-%H%M%S)}"
BASE_DIR="$ROOT/runs/$RUN_ID"
mkdir -p "$BASE_DIR"

MODEL="${NANOMA_MODEL:-deepseek-v4-pro}"
SNAPSHOT="/data_storage/nanoma/benchmarks/gaia/data"

{
  echo "run_id=$RUN_ID"
  echo "base_dir=$BASE_DIR"
  echo "model=$MODEL"
  echo "snapshot=$SNAPSHOT"
  echo "banned_tools=spawn,spawn_many"
  echo "mode=single_agent(max_agents=1,max_depth=0,max_concurrent_llm=1)"
  echo "tasks=GAIA L2 offset=0 limit=10"
  echo "per_task_time_limit=600s max_turns=60 budget=50"
} | tee "$BASE_DIR/run.info"

setsid nohup python3 benchmarks/gaia/run_nanoma_gaia_l2.py \
  --model "$MODEL" \
  --gaia-config 2023_level2 \
  --snapshot-dir "$SNAPSHOT" \
  --data-dir "$SNAPSHOT" \
  --offset 0 --limit 10 \
  --disable-tool spawn --disable-tool spawn_many \
  --max-agents 1 --max-depth 0 --max-concurrent-llm 1 \
  --time-limit 600 --max-turns 60 --budget 50 \
  --results-dir "$BASE_DIR/nanoma_l2" \
  --workspace-root "$BASE_DIR/workspace" \
  --log-root "$BASE_DIR/logs" \
  --clean \
  > "$BASE_DIR/run.out" 2> "$BASE_DIR/run.err" < /dev/null &

echo $! > "$BASE_DIR/run.pid"
echo "Launched pid=$(cat "$BASE_DIR/run.pid")"
echo "logs: $BASE_DIR/run.out | events: $BASE_DIR/logs/<task_id>/events.jsonl"
