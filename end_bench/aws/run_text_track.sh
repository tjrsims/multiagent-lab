#!/usr/bin/env bash
# Run one text-track design on Bedrock inside the CPU container and sync to S3.
# Usage: run_text_track.sh MODEL_ID RUN_ID [extra end-bench run-text args]
# Env:   END_BENCH_BUCKET (required), AWS_REGION (default us-east-1)
set -euo pipefail
MODEL_ID="${1:?Bedrock model id, e.g. anthropic.claude-sonnet-4-6}"
RUN_ID="${2:?run id}"
shift 2
: "${END_BENCH_BUCKET:?set END_BENCH_BUCKET=s3://<bucket>}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
cd "$(dirname "$0")/../.."
COMPOSE="docker compose -f end_bench/docker/docker-compose.yml"
mkdir -p data/runs
nohup $COMPOSE run --rm text run-text \
  --model "bedrock:${MODEL_ID}" --region "$AWS_REGION" --run-id "$RUN_ID" --out /data/runs "$@" \
  > "data/runs/${RUN_ID}.log" 2>&1 &
echo $! > "data/runs/${RUN_ID}.pid"
echo "started ${RUN_ID} (pid $(cat "data/runs/${RUN_ID}.pid")); log: data/runs/${RUN_ID}.log"
echo "sync when done: end_bench/aws/sync_s3.sh ${RUN_ID}"
