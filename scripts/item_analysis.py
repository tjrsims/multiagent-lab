#!/usr/bin/env python3
"""Item analysis over benchmark instances: find zero-information items.

For each trial (instance) in one or more saved datasets, compute the strict
pass (no false claim on any claim-bearing turn) and the per-turn contradiction
rate. Items that every dataset passes or every dataset fails carry no
discriminating information and are listed as retire candidates with the
diagnosis the analyst must complete (task defect / grader defect / genuinely
hard), following eval-forge Playbook 03's elimination order.

Usage:
    python scripts/item_analysis.py --dataset runs/a.json --name a \\
        --dataset runs/b.json --name b --out items.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interpretability.benchmark.views import build_views  # noqa: E402
from interpretability.script_artifacts import (  # noqa: E402
    add_legacy_trust_argument,
    load_activation_input,
)


def instance_table(dataset, *, allow_legacy: bool) -> dict[str, dict[str, Any]]:
    allowed = ("transactional", "legacy_compatibility") if allow_legacy else ("transactional",)
    views = build_views(dataset, allowed_runtime_paths=allowed)
    per_trial: dict[str, dict[str, Any]] = {}
    for index in range(views.n):
        trial = views.trial_ids[index]
        entry = per_trial.setdefault(trial, {
            "scenario": views.scenarios[index],
            "family": str(views.group_ids[index]),
            "incentive": views.incentive_conditions[index] if views.incentive_conditions else None,
            "pair_role": views.pair_roles[index] if views.pair_roles else None,
            "turns": 0,
            "false_claims": 0,
        })
        entry["turns"] += 1
        entry["false_claims"] += int(views.labels[index])
    for entry in per_trial.values():
        entry["strict_pass"] = entry["false_claims"] == 0
        entry["contradiction_rate"] = entry["false_claims"] / entry["turns"]
    return per_trial


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--name", action="append", default=None)
    parser.add_argument("--allow-legacy-runtime", action="store_true")
    parser.add_argument("--out", required=True)
    add_legacy_trust_argument(parser)
    args = parser.parse_args(argv)
    names = args.name or [Path(p).stem for p in args.dataset]
    if len(names) != len(args.dataset):
        parser.error("--name must be given once per --dataset")

    tables = {
        name: instance_table(
            load_activation_input(path, trust_legacy_pt=args.trust_legacy_pt),
            allow_legacy=args.allow_legacy_runtime,
        )
        for name, path in zip(names, args.dataset)
    }
    by_family: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    scenario_of: dict[str, str] = {}
    for name, table in tables.items():
        for trial, entry in table.items():
            by_family[entry["family"]][name].append(bool(entry["strict_pass"]))
            scenario_of[entry["family"]] = entry["scenario"]
    items = []
    for family, per_model in sorted(by_family.items()):
        rates = {name: sum(values) / len(values) for name, values in per_model.items()}
        all_pass = all(rate == 1.0 for rate in rates.values())
        all_fail = all(rate == 0.0 for rate in rates.values())
        items.append({
            "family": family,
            "scenario": scenario_of[family],
            "pass_rate_by_model": rates,
            "zero_information": all_pass or all_fail,
            "kind": "all_pass" if all_pass else "all_fail" if all_fail else "discriminating",
            "diagnosis": (
                None if not (all_pass or all_fail)
                else "TODO: task defect / grader defect / genuinely hard (Playbook 03 order)"
            ),
        })
    summary = {
        "n_items": len(items),
        "n_all_pass": sum(item["kind"] == "all_pass" for item in items),
        "n_all_fail": sum(item["kind"] == "all_fail" for item in items),
        "n_discriminating": sum(item["kind"] == "discriminating" for item in items),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "items": items, "per_model": tables}, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
