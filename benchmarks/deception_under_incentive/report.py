"""Leaderboard: one row per run, with cost next to the lie rate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.deception_under_incentive.analysis import summarize
from end_bench.data.text_dataset import load_text_dataset


def run_row(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    manifest = json.loads((run / "run_manifest.json").read_text())
    dataset = load_text_dataset(run / "text_dataset.json")
    summary = summarize(dataset)
    ledger_path = run / "cost_ledger.jsonl"
    usd = 0.0
    unverified = 0.0
    calls = 0
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            entry = json.loads(line)
            calls += 1
            usd += entry.get("usd") or 0.0
            if str(entry.get("price_source") or "").startswith("unverified"):
                unverified += entry.get("usd") or 0.0
    claims = sum(c["n_claims"] for c in summary["cells"])
    lies = sum((c["lie_rate"] or 0.0) * c["n_claims"] for c in summary["cells"])
    turns = sum(c["turns"] for c in summary["cells"])
    trials = sum(c["trials"] for c in summary["cells"])
    return {
        "run_id": manifest.get("run_id"),
        "model": manifest.get("spec", {}).get("model"),
        "counterpart_type": ",".join(manifest.get("spec", {}).get("counterpart_types", [])),
        "module": (manifest.get("benchmark_module") or {}).get("version"),
        "negotiations": trials,
        "claim_bearing_rate": (claims / turns) if turns else None,
        "lie_rate": (lies / claims) if claims else None,
        "usd_per_negotiation": (usd / trials) if trials else None,
        "unverified_share": (unverified / usd) if usd else 0.0,
        "calls": calls,
    }


def render_leaderboard(run_dirs: list[str | Path]) -> str:
    rows = [run_row(d) for d in run_dirs]
    rows.sort(key=lambda r: (r["lie_rate"] is None, r["lie_rate"] or 0.0))
    lines = [
        "| run | model | counterpart | negotiations | claim bearing | lie rate | $ per negotiation | unverified $ share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fmt = lambda v, p: ("n/a" if v is None else f"{v:{p}}")  # noqa: E731
        lines.append(
            f"| {r['run_id']} | {r['model']} | {r['counterpart_type']} | {r['negotiations']} | "
            f"{fmt(r['claim_bearing_rate'], '.0%')} | {fmt(r['lie_rate'], '.1%')} | "
            f"{fmt(r['usd_per_negotiation'], '.4f')} | {fmt(r['unverified_share'], '.0%')} |"
        )
    return "\n".join(lines) + "\n"
