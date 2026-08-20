#!/usr/bin/env bash
set -euo pipefail

ROOT="${NANOMA_ROOT:-/data/workspace/NanoMA}"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

RUN_ID="${1:-edgebench_nanoma_original_deepseek_full_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${EDGEBENCH_RUN_ROOT:-/data_storage/nanoma/runs}/${RUN_ID}"
EDGEBENCH_REPO="${EDGEBENCH_REPO:-/data_storage/nanoma/repos/EdgeBench}"
TASKS_DIR="${EDGEBENCH_TASKS_DIR:-/data_storage/nanoma/benchmarks/edgebench/tasks}"
TASKS_JSONL="$TASKS_DIR/tasks.jsonl"
MODEL="${EDGEBENCH_MODEL:-${NANOMA_MODEL:-deepseek-v4-pro}}"
BASE_URL="${NANOMA_LLM_BASE_URL:-${ANTHROPIC_BASE_URL:-}}"
API_KEY="${NANOMA_API_KEY:-${ANTHROPIC_API_KEY:-}}"
TIMEOUT="${EDGEBENCH_TIMEOUT:-7200}"
EVAL_INTERVAL="${EDGEBENCH_EVAL_INTERVAL:-300}"
MAX_SUBMISSIONS="${EDGEBENCH_MAX_SUBMISSIONS:-}"
SUBMISSION_COOLDOWN="${EDGEBENCH_SUBMISSION_COOLDOWN:-120}"
WORK_MEM_LIMIT="${EDGEBENCH_WORK_MEM_LIMIT:-8g}"
JUDGE_MEM_LIMIT="${EDGEBENCH_JUDGE_MEM_LIMIT:-8g}"
PORT="${EDGEBENCH_JUDGE_PORT:-18088}"
TASKS="${EDGEBENCH_TASKS:-ALL}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/config" "$TASKS_DIR" /workspace/nanoma-task-work

if [[ ! -d "$EDGEBENCH_REPO/sforge" ]]; then
  echo "Missing EdgeBench repo at $EDGEBENCH_REPO. Clone it first." >&2
  exit 1
fi
if [[ -z "$BASE_URL" ]]; then
  echo "Missing NANOMA_LLM_BASE_URL or ANTHROPIC_BASE_URL after loading .env" >&2
  exit 1
fi
if [[ -z "$API_KEY" ]]; then
  echo "Missing NANOMA_API_KEY or ANTHROPIC_API_KEY after loading .env" >&2
  exit 1
fi

export PYTHONPATH="$EDGEBENCH_REPO:${PYTHONPATH:-}"
export SFORGE_TASKS_DIR="$TASKS_DIR"
export SFORGE_LOG_DIR="$RUN_ROOT/sforge_logs"
export NANOMA_HOST_ROOT="$ROOT"
export NANOMA_MODEL="$MODEL"
export NANOMA_LLM_BASE_URL="$BASE_URL"
export NANOMA_API_KEY="$API_KEY"
export NANOMA_LLM_PROTOCOL="${EDGEBENCH_NANOMA_LLM_PROTOCOL:-${NANOMA_LLM_PROTOCOL:-openai}}"
export NANOMA_HTTP_TIMEOUT="${NANOMA_HTTP_TIMEOUT:-7200}"
export NANOMA_LLM_MAX_RETRIES="${NANOMA_LLM_MAX_RETRIES:-8}"
export NANOMA_LLM_RETRY_BASE_DELAY="${NANOMA_LLM_RETRY_BASE_DELAY:-2}"
export NANOMA_LLM_RETRY_MAX_DELAY="${NANOMA_LLM_RETRY_MAX_DELAY:-120}"
export NANOMA_PARSE_TEXT_TOOL_CALLS="${NANOMA_PARSE_TEXT_TOOL_CALLS:-1}"
export NANOMA_LLM_ADMISSION_CONTROL="${NANOMA_LLM_ADMISSION_CONTROL:-1}"
export NANOMA_LLM_MIN_START_SPACING="${NANOMA_LLM_MIN_START_SPACING:-0.75}"
export NANOMA_LLM_LARGE_CONTEXT_TOKENS="${NANOMA_LLM_LARGE_CONTEXT_TOKENS:-30000}"
export NANOMA_LLM_LARGE_CONTEXT_SPACING="${NANOMA_LLM_LARGE_CONTEXT_SPACING:-6}"
export NANOMA_LLM_OVERLOAD_COOLDOWN_SECONDS="${NANOMA_LLM_OVERLOAD_COOLDOWN_SECONDS:-45}"
export NANOMA_LLM_ADMISSION_MAX_DELAY="${NANOMA_LLM_ADMISSION_MAX_DELAY:-120}"

if ! python3 -c 'import fastapi, multipart' >/dev/null 2>&1; then
  echo "Installing missing SForge judge dependencies..." | tee -a "$RUN_ROOT/run.info"
  python3 -m pip install --user 'fastapi' 'python-multipart' >>"$RUN_ROOT/logs/pip_install.out" 2>>"$RUN_ROOT/logs/pip_install.err"
fi

if ! compgen -G "$TASKS_DIR/*.json" >/dev/null; then
  echo "Fetching EdgeBench task configs..." | tee -a "$RUN_ROOT/run.info"
  python3 -m sforge --tasks-dir "$TASKS_DIR" fetch-tasks edgebench >"$RUN_ROOT/logs/fetch_tasks.out" 2>"$RUN_ROOT/logs/fetch_tasks.err"
fi

mapfile -t ALL_TASKS < <(
  if [[ "$TASKS" == "ALL" ]]; then
    if [[ -f "$TASKS_JSONL" ]]; then
      python3 - <<'PY' "$TASKS_JSONL"
import json, sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    print(row.get("id") or row.get("task_id") or row.get("name"))
PY
    else
      python3 - <<'PY' "$TASKS_DIR"
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*.json")):
    if path.name != "tasks.jsonl":
        print(path.stem)
PY
    fi
  else
    printf '%s\n' $TASKS
  fi
)

if [[ "${#ALL_TASKS[@]}" -eq 0 ]]; then
  echo "No EdgeBench tasks selected." >&2
  exit 1
fi
printf '%s\n' "${ALL_TASKS[@]}" > "$RUN_ROOT/config/tasks.txt"

while ss -ltn | awk '{print $4}' | rg -q ":${PORT}$"; do
  PORT=$((PORT + 1))
done

{
  echo "run_id=$RUN_ID"
  echo "run_root=$RUN_ROOT"
  echo "edgebench_repo=$EDGEBENCH_REPO"
  echo "tasks_dir=$TASKS_DIR"
  echo "tasks_file=$RUN_ROOT/config/tasks.txt"
  echo "task_count=${#ALL_TASKS[@]}"
  echo "agent=nanoma_original"
  echo "model=$MODEL"
  echo "protocol=${EDGEBENCH_NANOMA_LLM_PROTOCOL:-${NANOMA_LLM_PROTOCOL:-openai}}"
  echo "judge_port=$PORT"
  echo "timeout=$TIMEOUT"
  echo "eval_interval=$EVAL_INTERVAL"
  echo "submission_cooldown=$SUBMISSION_COOLDOWN"
  echo "work_mem_limit=$WORK_MEM_LIMIT"
  echo "judge_mem_limit=$JUDGE_MEM_LIMIT"
  echo "mode=sequential Docker queue; official 0-100 summary generated after every task"
} | tee "$RUN_ROOT/run.info"

echo "Starting SForge judge server..." | tee -a "$RUN_ROOT/run.info"
setsid nohup python3 -m sforge \
  --tasks-dir "$TASKS_DIR" \
  --log-dir "$RUN_ROOT/sforge_logs" \
  serve --host 0.0.0.0 --port "$PORT" \
  >"$RUN_ROOT/logs/sforge_serve.out" 2>"$RUN_ROOT/logs/sforge_serve.err" < /dev/null &
SERVE_PID=$!
echo "$SERVE_PID" > "$RUN_ROOT/sforge_serve.pid"

sleep 3
if ! kill -0 "$SERVE_PID" 2>/dev/null; then
  echo "SForge judge server failed to start; see $RUN_ROOT/logs/sforge_serve.err" >&2
  exit 1
fi

QUEUE_SCRIPT="$RUN_ROOT/config/run_queue.sh"
cat > "$QUEUE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap "" HUP
cd "$ROOT"

export PYTHONPATH="$EDGEBENCH_REPO:\${PYTHONPATH:-}"
export SFORGE_TASKS_DIR="$TASKS_DIR"
export SFORGE_LOG_DIR="$RUN_ROOT/sforge_logs"
export NANOMA_HOST_ROOT="$ROOT"
export NANOMA_MODEL="$MODEL"
export NANOMA_LLM_BASE_URL="$BASE_URL"
export NANOMA_API_KEY="$API_KEY"
export NANOMA_LLM_PROTOCOL="${EDGEBENCH_NANOMA_LLM_PROTOCOL:-${NANOMA_LLM_PROTOCOL:-openai}}"
export NANOMA_HTTP_TIMEOUT="${NANOMA_HTTP_TIMEOUT:-7200}"
export NANOMA_LLM_MAX_RETRIES="${NANOMA_LLM_MAX_RETRIES:-8}"
export NANOMA_LLM_RETRY_BASE_DELAY="${NANOMA_LLM_RETRY_BASE_DELAY:-2}"
export NANOMA_LLM_RETRY_MAX_DELAY="${NANOMA_LLM_RETRY_MAX_DELAY:-120}"
export NANOMA_PARSE_TEXT_TOOL_CALLS="${NANOMA_PARSE_TEXT_TOOL_CALLS:-1}"

while IFS= read -r task; do
  [[ -z "\$task" ]] && continue
  echo "[\$(date -Is)] pulling \$task" | tee -a "$RUN_ROOT/logs/queue_progress.log"
  if ! python3 -m sforge --tasks-dir "$TASKS_DIR" --log-dir "$RUN_ROOT/sforge_logs" \\
      pull --task "\$task" --registry seededge \\
      >>"$RUN_ROOT/logs/pull_images.out" 2>>"$RUN_ROOT/logs/pull_images.err"; then
    echo "[\$(date -Is)] pull failed \$task" | tee -a "$RUN_ROOT/logs/queue_progress.log"
    continue
  fi

  RUN_ARGS=(
    --tasks-dir "$TASKS_DIR"
    --log-dir "$RUN_ROOT/sforge_logs"
    --silent
    run
    --task "\$task"
    --agent nanoma
    --model "$MODEL"
    --timeout "$TIMEOUT"
    --eval-interval "$EVAL_INTERVAL"
    --submission-cooldown "$SUBMISSION_COOLDOWN"
    --judge-url "http://host.docker.internal:$PORT"
    --run-id "$RUN_ID"
    --work-mem-limit "$WORK_MEM_LIMIT"
    --judge-mem-limit "$JUDGE_MEM_LIMIT"
  )
  if [[ -n "$MAX_SUBMISSIONS" ]]; then
    RUN_ARGS+=(--max-submissions "$MAX_SUBMISSIONS")
  fi

  echo "[\$(date -Is)] running \$task model=$MODEL" | tee -a "$RUN_ROOT/logs/queue_progress.log"
  set +e
  SFORGE_AGENT_API_KEY="$API_KEY" \\
  SFORGE_AGENT_API_BASE_URL="$BASE_URL" \\
  SFORGE_AGENT_MODEL="$MODEL" \\
  SFORGE_AGENT_EXTRA_ENV="NANOMA_API_KEY=$API_KEY,NANOMA_LLM_BASE_URL=$BASE_URL,NANOMA_MODEL=$MODEL,NANOMA_LLM_PROTOCOL=${EDGEBENCH_NANOMA_LLM_PROTOCOL:-${NANOMA_LLM_PROTOCOL:-openai}},NANOMA_HTTP_TIMEOUT=${NANOMA_HTTP_TIMEOUT:-7200},NANOMA_LLM_MAX_RETRIES=${NANOMA_LLM_MAX_RETRIES:-8},NANOMA_LLM_RETRY_BASE_DELAY=${NANOMA_LLM_RETRY_BASE_DELAY:-2},NANOMA_LLM_RETRY_MAX_DELAY=${NANOMA_LLM_RETRY_MAX_DELAY:-120},NANOMA_PARSE_TEXT_TOOL_CALLS=${NANOMA_PARSE_TEXT_TOOL_CALLS:-1},NANOMA_LLM_ADMISSION_CONTROL=${NANOMA_LLM_ADMISSION_CONTROL:-1},NANOMA_LLM_MIN_START_SPACING=${NANOMA_LLM_MIN_START_SPACING:-0.75},NANOMA_LLM_LARGE_CONTEXT_TOKENS=${NANOMA_LLM_LARGE_CONTEXT_TOKENS:-30000},NANOMA_LLM_LARGE_CONTEXT_SPACING=${NANOMA_LLM_LARGE_CONTEXT_SPACING:-6},NANOMA_LLM_OVERLOAD_COOLDOWN_SECONDS=${NANOMA_LLM_OVERLOAD_COOLDOWN_SECONDS:-45},NANOMA_LLM_ADMISSION_MAX_DELAY=${NANOMA_LLM_ADMISSION_MAX_DELAY:-120},NANOMA_SPAWN_TODOLIST_JUDGE=1,NANOMA_SPAWN_JUDGE_MODEL=${NANOMA_SPAWN_JUDGE_MODEL:-claude-opus-4-8},NANOMA_EDGE_WALL_SECONDS=${NANOMA_EDGE_WALL_SECONDS:-$TIMEOUT},NANOMA_EDGE_FINAL_SAFETY_SECONDS=${NANOMA_EDGE_FINAL_SAFETY_SECONDS:-90},NANOMA_MERGE_SUBMIT_PATH=${NANOMA_MERGE_SUBMIT_PATH:-1}" \\
  python3 -m sforge "\${RUN_ARGS[@]}" \\
    >"$RUN_ROOT/logs/\${task}.out" 2>"$RUN_ROOT/logs/\${task}.err"
  exit_code=\$?
  set -e
  echo "[\$(date -Is)] finished \$task exit=\$exit_code" | tee -a "$RUN_ROOT/logs/queue_progress.log"
  /data_storage/nanoma/benchmarks/edgebench/summarize_edgebench_official.py "$RUN_ROOT" \\
    >>"$RUN_ROOT/logs/summary_refresh.out" 2>>"$RUN_ROOT/logs/summary_refresh.err" || true
done < "$RUN_ROOT/config/tasks.txt"

/data_storage/nanoma/benchmarks/edgebench/summarize_edgebench_official.py "$RUN_ROOT" \\
  >>"$RUN_ROOT/logs/summary_refresh.out" 2>>"$RUN_ROOT/logs/summary_refresh.err" || true
echo "completed_at=\$(date -Is)" >> "$RUN_ROOT/run.info"
EOF
chmod +x "$QUEUE_SCRIPT"

setsid nohup "$QUEUE_SCRIPT" >"$RUN_ROOT/logs/queue.out" 2>"$RUN_ROOT/logs/queue.err" < /dev/null &
QUEUE_PID=$!
echo "$QUEUE_PID" > "$RUN_ROOT/nanoma_edgebench_full_queue.pid"

{
  echo "sforge_serve_pid=$SERVE_PID"
  echo "queue_pid=$QUEUE_PID"
  echo "queue_progress=$RUN_ROOT/logs/queue_progress.log"
  echo "summary_json=$RUN_ROOT/official_score_summary.json"
  echo "summary_md=$RUN_ROOT/official_score_summary.md"
  echo "sforge_run_dir=$RUN_ROOT/sforge_logs/runs/$RUN_ID"
} | tee -a "$RUN_ROOT/run.info"

echo "Started full EdgeBench NanoMA original deepseek run in $RUN_ROOT"
