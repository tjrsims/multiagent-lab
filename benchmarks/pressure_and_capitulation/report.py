"""Leaderboard for benchmark 2: one table per pressure policy, hold and capitulation beside cost.

A row is one finished run. Runs that differ in counterpart regime, module
version, attestation timing or round count measure different things and are
refused rather than pooled; pressure policies are grouped by policy id, never
by name alone. A run with any unpriced call shows no cost rather than $0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.pressure_and_capitulation.analysis import summarize
from end_bench.data.text_dataset import load_text_dataset


def _ledger_totals(run: Path) -> dict[str, Any]:
    usd = 0.0
    calls = 0
    unpriced = 0
    ledger_path = run / "cost_ledger.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            entry = json.loads(line)
            calls += 1
            if entry.get("usd") is None:
                unpriced += 1
            else:
                usd += float(entry["usd"])
    return {"usd": usd, "calls": calls, "unpriced_calls": unpriced}


def run_row(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    manifest = json.loads((run / "run_manifest.json").read_text())
    dataset = load_text_dataset(run / "text_dataset.json")
    summary = summarize(dataset)
    ledger = _ledger_totals(run)
    spec = manifest.get("spec", {})
    module = manifest.get("benchmark_module") or {}
    trials_block = manifest.get("trials") or {}
    overall = summary["overall"]
    # A run holds one policy by construction; the overall block carries its clustered rates.
    head = overall[0] if len(overall) == 1 else None
    trials = sum(c["trials"] for c in summary["cells"])
    pushbacks = [c["pushbacks_observed_mean"] for c in summary["cells"] if c["pushbacks_observed_mean"] is not None]
    unpriced = ledger["unpriced_calls"] > 0
    return {
        "run_id": manifest.get("run_id"),
        "model": spec.get("model"),
        "counterpart_model": manifest.get("counterpart_model") or spec.get("model"),
        "counterpart_type": ",".join(spec.get("counterpart_types", [])),
        "counterpart_policy_id": ",".join(sorted({str(o["counterpart_policy_id"]) for o in overall if o["counterpart_policy_id"]})) or None,
        "counterpart_regime": manifest.get("counterpart_regime") or "self_play",
        "module": module.get("version"),
        "attestation_timing": module.get("attestation_timing"),
        "max_rounds": spec.get("max_rounds"),
        "negotiations": trials,
        "failed": trials_block.get("failed"),
        "forfeited_turns": trials_block.get("forfeited_turns"),
        "initial_claim_coverage": head["initial_claim_coverage"] if head else None,
        "pushback_delivery_rate": head["pushback_delivery_rate"] if head else None,
        "no_pushback_rate": head["no_pushback_rate"] if head else None,
        "hold_rate": head["hold_rate"] if head else None,
        "capitulation_rate": head["capitulation_rate"] if head else None,
        "capitulation_rate_ci": head["capitulation_rate_ci"] if head else None,
        "moved_away_rate": head["moved_away_rate"] if head else None,
        "dodge_rate": head["dodge_rate"] if head else None,
        "initial_false_rate": head["initial_false_rate"] if head else None,
        "n_judged": head["n_judged"] if head else None,
        "pushbacks_observed_mean": (sum(pushbacks) / len(pushbacks)) if pushbacks else None,
        "actor_rejected_turns": head["actor_rejected_turns"] if head else None,
        "usd_per_negotiation": None if (unpriced or not trials) else ledger["usd"] / trials,
        "unpriced_calls": ledger["unpriced_calls"],
        "calls": ledger["calls"],
    }


def _refuse_mixed(rows: list[dict[str, Any]], field: str, what: str) -> None:
    values = sorted({str(r.get(field)) for r in rows})
    if len(values) > 1:
        raise ValueError(f"runs with different {what} cannot share a leaderboard: " + ", ".join(values))


def render_leaderboard(run_dirs: list[str | Path]) -> str:
    rows = [run_row(d) for d in run_dirs]
    _refuse_mixed(rows, "counterpart_regime", "counterpart regimes")
    _refuse_mixed(rows, "module", "module versions")
    _refuse_mixed(rows, "attestation_timing", "attestation timings")
    _refuse_mixed(rows, "max_rounds", "round counts")
    fmt = lambda v, p: ("n/a" if v is None else f"{v:{p}}")  # noqa: E731

    def ci(row: dict[str, Any]) -> str:
        if row["capitulation_rate"] is None:
            return "n/a"
        low, high = row["capitulation_rate_ci"] or (None, None)
        if low is None or high is None:
            return f"{row['capitulation_rate']:.1%}"
        return f"{row['capitulation_rate']:.1%} [{low:.0%}, {high:.0%}]"

    regime = rows[0]["counterpart_regime"] if rows else "none"
    lines = [f"Counterpart regime: {regime}", ""]
    by_policy: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault((row["counterpart_type"], row["counterpart_policy_id"]), []).append(row)
    for (policy, policy_id), group in sorted(by_policy.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        group.sort(key=lambda r: (r["capitulation_rate"] is None, r["capitulation_rate"] or 0.0))
        lines += [
            f"Pressure policy: {policy} ({policy_id or 'id unknown'})",
            "",
            "| run | model | negotiations | failed | forfeited | initial claim | pushback delivered | no pushback | held | capitulated [95% CI] | moved away | dodged | pushbacks seen | $ per negotiation |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in group:
            lines.append(
                f"| {r['run_id']} | {r['model']} | {r['negotiations']} | {fmt(r['failed'], 'd')} | {fmt(r['forfeited_turns'], 'd')} | "
                f"{fmt(r['initial_claim_coverage'], '.0%')} | {fmt(r['pushback_delivery_rate'], '.0%')} | {fmt(r['no_pushback_rate'], '.0%')} | "
                f"{fmt(r['hold_rate'], '.0%')} | {ci(r)} | {fmt(r['moved_away_rate'], '.0%')} | {fmt(r['dodge_rate'], '.0%')} | "
                f"{fmt(r['pushbacks_observed_mean'], '.1f')} | {fmt(r['usd_per_negotiation'], '.4f')} |"
            )
        lines.append("")
    return "\n".join(lines)
