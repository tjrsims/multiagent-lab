#!/usr/bin/env bash
# Re-run the Concordia-facing suites against upstream Concordia ``main`` (or a
# ref) without touching the installed, pinned 2.4.0: the clone is put first on
# PYTHONPATH for the test run only.
# Usage: scripts/check_concordia_upstream.sh [REF]      (default: main)
set -euo pipefail
REF="${1:-main}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/concordia-upstream-check"
rm -rf "$WORK" && git clone -q --depth 1 --branch "$REF" https://github.com/google-deepmind/concordia.git "$WORK"
echo "upstream: $(git -C "$WORK" log -1 --format='%H %ad %s' --date=short)"
cd "$ROOT"
PYTHONPATH="$WORK" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib \
  python -c "import concordia; print('concordia from', concordia.__file__)"
PYTHONPATH="$WORK" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib \
  python -m pytest tests/negotiation tests/test_concordia_compat.py \
    tests/interpretability/test_runtime_executor.py tests/interpretability/test_turn_label_runtime.py \
    tests/end_bench -q -p no:cacheprovider "$@"
