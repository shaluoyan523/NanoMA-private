#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

NANOMA_ROOT=/data/workspace/NanoMA
EDGEBENCH_ROOT=/data_storage/nanoma/repos/EdgeBench
TASKS_ROOT=/data_storage/nanoma/benchmarks/edgebench/tasks
RUN_ID=edgebench_low11_branchstore_cpu4_20260802_210700
RUN_ROOT=/data_storage/nanoma/runs/$RUN_ID
MODEL=deepseek-v4-pro
JUDGE_PORT=18089
TASKS=(
  apple_incremental_game
  bipedalwalker_locomotion_rl
  borden_source_inversion
  cta_risk_budget_optimization
  k12_math_recommendation
  nethack_dungeon_agent
  openrct2_theme_park_ai
  order_addition_permutation_optimization
  pfr_formalization
  trinity_text_adventure
  vibrating_path_graph_coloring
)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/config" "$RUN_ROOT/sforge_logs"
printf '%s\n' "${TASKS[@]}" >"$RUN_ROOT/config/tasks.txt"

cd "$NANOMA_ROOT"
set -a
source "$NANOMA_ROOT/.env"
set +a

BASE_URL=${NANOMA_LLM_BASE_URL:-${ANTHROPIC_BASE_URL:-}}
API_KEY=${NANOMA_API_KEY:-${ANTHROPIC_API_KEY:-}}
if [[ -z "$BASE_URL" || -z "$API_KEY" ]]; then
  printf '%s\n' 'NanoMA API configuration is unavailable' >&2
  exit 2
fi

export PYTHONPATH="$EDGEBENCH_ROOT:${PYTHONPATH:-}"
export SFORGE_TASKS_DIR="$TASKS_ROOT"
export SFORGE_LOG_DIR="$RUN_ROOT/sforge_logs"
export SFORGE_WORK_CPU_LIMIT=4
export SFORGE_WORK_MEM_LIMIT=16g
export SFORGE_JUDGE_CPU_LIMIT=4
export SFORGE_JUDGE_MEM_LIMIT=8g
export SFORGE_JUDGE_MAX_CONCURRENT=5
export SFORGE_DRAIN_WAIT=7200
export NANOMA_HOST_ROOT="$NANOMA_ROOT"

export SFORGE_AGENT_API_KEY="$API_KEY"
export SFORGE_AGENT_API_BASE_URL="$BASE_URL"
export SFORGE_AGENT_MODEL="$MODEL"

COMMON_AGENT_ENV="NANOMA_API_KEY=$API_KEY,NANOMA_LLM_BASE_URL=$BASE_URL,NANOMA_MODEL=$MODEL,NANOMA_LLM_PROTOCOL=openai,NANOMA_HTTP_TIMEOUT=7200,NANOMA_LLM_MAX_RETRIES=8,NANOMA_LLM_RETRY_BASE_DELAY=2,NANOMA_LLM_RETRY_MAX_DELAY=120,NANOMA_PARSE_TEXT_TOOL_CALLS=1,NANOMA_LLM_ADMISSION_CONTROL=1,NANOMA_LLM_MIN_START_SPACING=0.75,NANOMA_LLM_LARGE_CONTEXT_TOKENS=30000,NANOMA_LLM_LARGE_CONTEXT_SPACING=6,NANOMA_LLM_OVERLOAD_COOLDOWN_SECONDS=45,NANOMA_LLM_ADMISSION_MAX_DELAY=120,NANOMA_SPAWN_TODOLIST_JUDGE=1,NANOMA_SPAWN_JUDGE_MODEL=$MODEL,NANOMA_MERGE_SUBMIT_PATH=1,NANOMA_MERGE_MAX_MB=200,NANOMA_MERGE_MAX_FILES=40000,NANOMA_SUBMIT_REQUIRE_VERIFIED=1,NANOMA_TOOL_POLICY_MODE=off,NANOMA_EDGE_WALL_SECONDS=7200,NANOMA_EDGE_FINAL_SAFETY_SECONDS=90"

printf '%s\n' \
  "run_id=$RUN_ID" \
  "tasks=${TASKS[*]}" \
  "model=$MODEL" \
  "judge_port=$JUDGE_PORT" \
  'timeout_per_task=7200' \
  'eval_interval=300' \
  'submission_cooldown=120' \
  'work_cpu_limit=4' \
  'work_mem_limit=16g' \
  'judge_cpu_limit=4' \
  'judge_mem_limit=8g' \
  'judge_max_concurrent=5' \
  'candidate_scoring=post_official_drain' \
  'mode=sequential' \
  "started_at=$(date -Is)" \
  >"$RUN_ROOT/run.info"

serve_pid=''
cleanup() {
  if [[ -n "$serve_pid" ]] && kill -0 "$serve_pid" 2>/dev/null; then
    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$EDGEBENCH_ROOT"
python3 -m sforge \
  --tasks-dir "$TASKS_ROOT" \
  --log-dir "$RUN_ROOT/sforge_logs" \
  serve --host 0.0.0.0 --port "$JUDGE_PORT" \
  >"$RUN_ROOT/logs/sforge_serve.out" \
  2>"$RUN_ROOT/logs/sforge_serve.err" &
serve_pid=$!
printf '%s\n' "$serve_pid" >"$RUN_ROOT/sforge_serve.pid"

for _ in $(seq 1 60); do
  if ! kill -0 "$serve_pid" 2>/dev/null; then
    printf '%s\n' 'Judge server exited during startup' >&2
    exit 3
  fi
  if python3 - "$JUDGE_PORT" <<'PY'
import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
  then
    break
  fi
  sleep 1
done

overall_exit=0
for task in "${TASKS[@]}"; do
  if [[ "$task" == "pfr_formalization" ]]; then
    export SFORGE_AGENT_EXTRA_ENV="$COMMON_AGENT_ENV,NANOMA_MERGE_SHARED_PATHS=baseline se-bmk-intern/pfr/.lake/packages"
    shared_paths='baseline se-bmk-intern/pfr/.lake/packages'
  else
    export SFORGE_AGENT_EXTRA_ENV="$COMMON_AGENT_ENV"
    shared_paths='none'
  fi

  printf '%s\n' \
    "task_started=$task $(date -Is)" \
    "task_merge_shared_paths=$task $shared_paths" \
    >>"$RUN_ROOT/run.info"

  set +e
  python3 -m sforge \
    --tasks-dir "$TASKS_ROOT" \
    --log-dir "$RUN_ROOT/sforge_logs" \
    --silent run \
    --task "$task" \
    --agent nanoma \
    --model "$MODEL" \
    --timeout 7200 \
    --eval-interval 300 \
    --disable-auto-resume \
    --submission-cooldown 120 \
    --judge-url "http://host.docker.internal:$JUDGE_PORT" \
    --run-id "$RUN_ID" \
    --disable-internet \
    --work-cpu-limit 4 \
    --work-mem-limit 16g \
    --judge-cpu-limit 4 \
    --judge-mem-limit 8g \
    >"$RUN_ROOT/logs/$task.out" \
    2>"$RUN_ROOT/logs/$task.err"
  task_exit=$?
  set -e

  printf '%s\n' \
    "task_completed=$task exit_code=$task_exit $(date -Is)" \
    >>"$RUN_ROOT/run.info"
  if [[ "$task_exit" -ne 0 ]]; then
    overall_exit=$task_exit
  fi
done

printf '%s\n' \
  "completed_at=$(date -Is)" \
  "exit_code=$overall_exit" \
  >>"$RUN_ROOT/run.info"
exit "$overall_exit"
