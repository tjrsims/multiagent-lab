#!/usr/bin/env bash
# Run the existing white-box capture path for an open model in the GPU container.
# Usage: run_whitebox_track.sh HF_MODEL SCENARIO FAMILIES ROLLOUTS RUN_ID [extra deception run args]
set -euo pipefail
MODEL="${1:?HF model id, e.g. google/gemma-2-9b-it}"
SCENARIO="${2:?scenario}"
FAMILIES="${3:?families}"
ROLLOUTS="${4:?rollouts}"
RUN_ID="${5:?run id}"
shift 5
cd "$(dirname "$0")/../.."
COMPOSE="docker compose -f end_bench/docker/docker-compose.yml"
REGION="${AWS_REGION:-us-east-1}"
if [ -z "${HF_TOKEN:-}" ]; then
  # Gated checkpoints (Gemma, Llama) need the token; it lives in SSM, not on disk.
  HF_TOKEN="$(aws ssm get-parameter --with-decryption --region "$REGION" --name /end-bench/hf_token \
    --query Parameter.Value --output text 2>/dev/null || true)"
fi
export HF_TOKEN
[ -n "$HF_TOKEN" ] || echo "warning: HF_TOKEN is empty (no SSM /end-bench/hf_token); gated models will fail to download" >&2
mkdir -p "data/runs/${RUN_ID}"
nohup $COMPOSE run --rm whitebox deception run \
  --model "$MODEL" --hybrid --device cuda --dtype bfloat16 \
  --experiment-track single_agent_white_box --ultrafast \
  --evaluator-type rule --skip-api-eval \
  --scenario-name "$SCENARIO" --trials "$FAMILIES" --rollouts-per-family "$ROLLOUTS" \
  --paired-facts --surface-variant default --max-rounds 5 \
  --checkpoint-dir "/data/runs/${RUN_ID}/ckpt" --output "/data/runs/${RUN_ID}" "$@" \
  > "data/runs/${RUN_ID}.log" 2>&1 &
echo $! > "data/runs/${RUN_ID}.pid"
echo "started ${RUN_ID}; log: data/runs/${RUN_ID}.log"
