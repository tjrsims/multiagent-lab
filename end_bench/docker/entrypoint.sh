#!/usr/bin/env bash
# Dispatch: `end-bench ...` by default; `deception ...` for the research CLI;
# `shell` for an interactive shell. The flash_attn workaround used by the GPU
# runners is applied for `deception` invocations.
set -euo pipefail

# docker-compose forwards AWS_PROFILE even when the host left it unset, and
# botocore treats an empty string as a (missing) named profile. Drop it so the
# instance profile / mounted ~/.aws default chain is used instead.
if [ -z "${AWS_PROFILE:-}" ]; then
  unset AWS_PROFILE
fi

case "${1:-}" in
  deception)
    shift
    exec python -c "import sys; sys.modules['flash_attn'] = None; from interpretability.cli import cli; cli()" "$@"
    ;;
  shell|bash)
    shift || true
    exec bash "$@"
    ;;
  pytest)
    shift || true
    cd /workspace && exec python -m pytest "$@"
    ;;
  *)
    exec end-bench "$@"
    ;;
esac
