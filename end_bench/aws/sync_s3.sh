#!/usr/bin/env bash
# Sync one run (or everything under data/) to the artifact bucket.
# Usage: sync_s3.sh [RUN_ID]      Env: END_BENCH_BUCKET=s3://<bucket>
set -euo pipefail
: "${END_BENCH_BUCKET:?set END_BENCH_BUCKET=s3://<bucket>}"
cd "$(dirname "$0")/../.."
if [ -n "${1:-}" ]; then
  aws s3 sync "data/runs/$1" "${END_BENCH_BUCKET}/runs/$1" --only-show-errors
  [ -f "data/runs/$1.log" ] && aws s3 cp "data/runs/$1.log" "${END_BENCH_BUCKET}/runs/$1/run.log" --only-show-errors
else
  aws s3 sync data "${END_BENCH_BUCKET}" --only-show-errors
fi
echo "synced to ${END_BENCH_BUCKET}"
