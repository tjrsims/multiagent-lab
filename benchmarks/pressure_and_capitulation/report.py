"""Leaderboard for benchmark 2: one row per run, hold and capitulation beside cost."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.pressure_and_capitulation.analysis import summarize
from end_bench.data.text_dataset import load_text_dataset


def run_row(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    manifest = json.loads((run / "run_manifest.json").read_text())
    dataset = load_text_dataset(run / "text_dataset.json")
    summary = summarize(dataset)
    ledger_path = run / "cost_ledger.jsonl"
    usd = 0.0
    calls = 0
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            entry = json.loads(line)
            calls += 1
            usd += entry.get("usd") or 0.0
    cells = summary["cells"]
    trials = sum(c["trials"] for c in cells)
    judged = sum(c["n_judged"] for c in cells)
    capitulated = sum(c["transitions"]["capitulated"] for c in cells)
    held = sum(c["transitions"]["held"] for c in cells)
    with_claim = sum(c["trials"] - c["transitions"]["no_initial_claim"] for c in cells)
    dodged = sum(c["transitions"]["dodged"] for c in cells)
    pushbacks = [c["pushbacks_observed_mean"] for c in cells if c["pushbacks_observed_mean"] is not None]
    return {
        "run_id": manifest.get("run_id"),
        "model": manifest.get("spec", {}).get("model"),
        "counterpart_type": ",".join(manifest.get("spec", {}).get("counterpart_types", [])),
        "counterpart_regime": manifest.get("counterpart_regime") or "self_play",
        "module": (manifest.get("benchmark_module") or {}).get("version"),
        "negotiations": trials,
        "initial_claim_coverage": (with_claim / trials) if trials else None,
        "hold_rate": (held / judged) if judged else None,
        "capitulation_rate": (capitulated / judged) if judged else None,
        "dodge_rate": (dodged / with_claim) if with_claim else None,
        "pushbacks_observed_mean": (sum(pushbacks) / len(pushbacks)) if pushbacks else None,
        "usd_per_negotiation": (usd / trials) if trials else None,
        "calls": calls,
    }


def render_leaderboard(run_dirs: list[str | Path]) -> str:
    rows = [run_row(d) for d in run_dirs]
    regimes = sorted({r["counterpart_regime"] for r in rows})
    if len(regimes) > 1:
        raise ValueError("runs with different counterpart regimes cannot share a leaderboard: " + ", ".join(regimes))
    rows.sort(key=lambda r: (r["capitulation_rate"] is None, r["capitulation_rate"] or 0.0))
    fmt = lambda v, p: ("n/a" if v is None else f"{v:{p}}")  # noqa: E731
    lines = [
        f"Counterpart regime: {regimes[0]}" if regimes else "Counterpart regime: none",
        "",
        "| run | model | pressure policy | negotiations | initial claim | held | capitulated | dodged | pushbacks seen | $ per negotiation |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['model']} | {r['counterpart_type']} | {r['negotiations']} | "
            f"{fmt(r['initial_claim_coverage'], '.0%')} | {fmt(r['hold_rate'], '.0%')} | {fmt(r['capitulation_rate'], '.1%')} | "
            f"{fmt(r['dodge_rate'], '.0%')} | {fmt(r['pushbacks_observed_mean'], '.1f')} | {fmt(r['usd_per_negotiation'], '.4f')} |"
        )
    return "\n".join(lines) + "\n"
