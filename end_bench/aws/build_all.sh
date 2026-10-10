#!/usr/bin/env bash
# Build a scorecard for every dataset under data/runs and sync builds to S3.
# Usage: build_all.sh [--n-bootstrap N]
set -euo pipefail
cd "$(dirname "$0")/../.."
COMPOSE="docker compose -f end_bench/docker/docker-compose.yml"
for manifest in data/runs/*/text_dataset.json data/runs/*/activations_*.json; do
  [ -f "$manifest" ] || continue
  run_dir="$(dirname "$manifest")"
  run_id="$(basename "$run_dir")"
  $COMPOSE run --rm text build --dataset "/data/runs/${run_id}/$(basename "$manifest")" \
    --out "/data/builds/${run_id}" "$@"
done
if [ -n "${END_BENCH_BUCKET:-}" ]; then
  aws s3 sync data/builds "${END_BENCH_BUCKET}/builds" --only-show-errors
fi
