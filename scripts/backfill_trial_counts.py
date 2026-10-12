#!/usr/bin/env python
"""Backfill planned/completed/failed trial counts into run manifests written before the
failure accounting existed.

Usage: scripts/backfill_trial_counts.py RUN_DIR [RUN_DIR ...]

Reads results.json next to each manifest. Old trial results carry no ``failure``
field, so the error type is taken from the failing transition in the trial's
runtime checkpoint. The manifest gains ``trials`` plus ``trials_backfilled_from``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def failure_from_checkpoint(trial: dict) -> dict | None:
    if trial.get("runtime_state") != "failed":
        return None
    if isinstance(trial.get("failure"), dict):
        return trial["failure"]
    runner = (trial.get("runtime_checkpoint") or {}).get("trial_runner") or {}
    for event in reversed(runner.get("events") or []):
        payload = event.get("payload") or {}
        if payload.get("error_type"):
            return {"error_type": str(payload["error_type"]), "error": str(payload.get("error") or "")[:500]}
    return {"error_type": "unknown", "error": ""}


def backfill(run_dir: Path) -> dict:
    from end_bench.run import summarize_trials

    results = json.loads((run_dir / "results.json").read_text())
    for block in results.values():
        for payload in (block.get("conditions") or {}).values():
            for trial in payload.get("trials") or []:
                if trial.get("runtime_state") == "failed" and not isinstance(trial.get("failure"), dict):
                    trial["failure"] = failure_from_checkpoint(trial)
    summary = summarize_trials(results)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trials"] = summary
    manifest["trials_backfilled_from"] = "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return summary


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        run_dir = Path(arg)
        if not (run_dir / "results.json").exists():
            print(f"skip {run_dir}: no results.json")
            continue
        summary = backfill(run_dir)
        print(f"{run_dir.name}: planned={summary['planned']} completed={summary['completed']} "
              f"failed={summary['failed']} {summary['failure_types']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
