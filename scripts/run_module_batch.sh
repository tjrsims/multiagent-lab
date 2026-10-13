#!/usr/bin/env bash
# Detached runner for benchmark module batches (survives tool time limits and terminal closes).
# Usage: scripts/run_module_batch.sh MODEL_ID RUN_ID OUT_DIR PRESET POLICY [POLICY...]
# Writes OUT_DIR/logs/RUN_ID.log and OUT_DIR/logs/RUN_ID.pid. Needs AWS_PROFILE in the environment.
# END_BENCH_REGION (default us-east-1) is the Bedrock region the run calls; per region request quotas
# are separate, so splitting one model's policies across regions raises throughput.
# END_BENCH_COUNTERPART (a Bedrock model id) sets the fixed counterpart model. The main and main_lite
# presets require it; the pilot runs self play without it.
set -euo pipefail
MODEL="${1:?model: a Bedrock id, a full spec such as xai:grok-4.3, or a roster key such as @claude-sonnet-5}"
RUN_ID="${2:?run id}"
OUT="${3:?out dir}"
PRESET="${4:?preset}"
shift 4
[ "$#" -ge 1 ] || { echo "at least one policy is required" >&2; exit 2; }
cd "$(dirname "$0")/.."
mkdir -p "$OUT/logs"
ARGS=()
for policy in "$@"; do ARGS+=(--counterpart-type "$policy"); done
REGION="${END_BENCH_REGION:-us-east-1}"
# A bare Bedrock id gets the bedrock: scheme; full specs (<scheme>:<id>) and @roster keys pass through.
spec_of() { case "$1" in @*|*:*) echo "$1" ;; *) echo "bedrock:$1" ;; esac; }
if [ -n "${END_BENCH_COUNTERPART:-}" ]; then ARGS+=(--counterpart-model "$(spec_of "$END_BENCH_COUNTERPART")"); fi
nohup ./.venv/bin/end-bench module run deception_under_incentive --model "$(spec_of "$MODEL")" --preset "$PRESET" \
  "${ARGS[@]}" --run-id "$RUN_ID" --out "$OUT" --region "$REGION" > "$OUT/logs/${RUN_ID}.log" 2>&1 &
echo $! > "$OUT/logs/${RUN_ID}.pid"
echo "started ${RUN_ID} (pid $(cat "$OUT/logs/${RUN_ID}.pid")) region ${REGION} counterpart ${END_BENCH_COUNTERPART:-self play} policies: $*; log: $OUT/logs/${RUN_ID}.log"
