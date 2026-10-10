#!/usr/bin/env bash
# Detached runner for benchmark module batches (survives tool time limits and terminal closes).
# Usage: scripts/run_module_batch.sh MODEL_ID RUN_ID OUT_DIR PRESET POLICY [POLICY...]
# Writes OUT_DIR/logs/RUN_ID.log and OUT_DIR/logs/RUN_ID.pid. Needs AWS_PROFILE/AWS_REGION in the environment.
set -euo pipefail
MODEL="${1:?model id, e.g. us.meta.llama4-maverick-17b-instruct-v1:0}"
RUN_ID="${2:?run id}"
OUT="${3:?out dir}"
PRESET="${4:?preset}"
shift 4
[ "$#" -ge 1 ] || { echo "at least one policy is required" >&2; exit 2; }
cd "$(dirname "$0")/.."
mkdir -p "$OUT/logs"
ARGS=()
for policy in "$@"; do ARGS+=(--counterpart-type "$policy"); done
nohup ./.venv/bin/end-bench module run deception_under_incentive --model "bedrock:${MODEL}" --preset "$PRESET" \
  "${ARGS[@]}" --run-id "$RUN_ID" --out "$OUT" > "$OUT/logs/${RUN_ID}.log" 2>&1 &
echo $! > "$OUT/logs/${RUN_ID}.pid"
echo "started ${RUN_ID} (pid $(cat "$OUT/logs/${RUN_ID}.pid")) policies: $*; log: $OUT/logs/${RUN_ID}.log"
