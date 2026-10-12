"""The trial count backfill reads old results.json files and fills the manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "backfill_trial_counts.py"


def _old_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    failed = {
        "runtime_state": "failed",
        "runtime_checkpoint": {"trial_runner": {"state": "failed", "events": [
            {"from_state": "created", "to_state": "observed", "payload": {}},
            {"from_state": "observed", "to_state": "failed",
             "payload": {"error": "hard validation rejection retry limit exceeded", "error_type": "RetryLimitExceeded"}},
        ]}},
    }
    completed = {"runtime_state": "completed", "runtime_checkpoint": {"trial_runner": {"state": "completed", "events": []}}}
    results = {
        "ultimatum_bluff": {"conditions": {"high_incentive": {"trials": [completed, failed, completed]}}},
        "hidden_value": {"conditions": {"low_incentive": {"trials": [failed]}}},
    }
    (run_dir / "results.json").write_text(json.dumps(results))
    (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": "old", "n_rows": 7}))
    return run_dir


def test_backfill_adds_trial_counts_with_failure_types(tmp_path) -> None:
    run_dir = _old_run(tmp_path)
    proc = subprocess.run([sys.executable, str(SCRIPT), str(run_dir)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["run_id"] == "old" and manifest["n_rows"] == 7
    trials = manifest["trials"]
    assert (trials["planned"], trials["completed"], trials["failed"]) == (4, 2, 2)
    assert trials["failure_types"] == {"RetryLimitExceeded": 2}
    assert trials["by_cell"]["hidden_value/low_incentive"] == {"planned": 1, "completed": 0, "failed": 1}
    assert manifest["trials_backfilled_from"] == "results.json"
    assert "planned=4 completed=2 failed=2" in proc.stdout


def test_backfill_skips_directories_without_results(tmp_path) -> None:
    proc = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0 and "no results.json" in proc.stdout
    assert subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False).returncode == 2
