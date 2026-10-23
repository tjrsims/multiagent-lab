"""Leaderboard: one row per run, with cost next to the lie rate."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
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
    unpriced = 0
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            entry = json.loads(line)
            calls += 1
            if entry.get("usd") is None:
                # An unpriced model must not read as free.
                unpriced += 1
                continue
            usd += float(entry["usd"])
            if str(entry.get("price_source") or "").startswith("unverified"):
                unverified += float(entry["usd"])
    rows = [r for r in dataset.get("metadata", []) if r.get("sample_type") == "negotiation"]
    policy_ids = sorted({str(r.get("counterpart_policy_id")) for r in rows if r.get("counterpart_policy_id")})
    module = manifest.get("benchmark_module") or {}
    trials_block = manifest.get("trials") or {}
    claims = sum(c["n_claims"] for c in summary["cells"])
    attest_turns = sum(c.get("attestation_turns", 0) for c in summary["cells"])
    attest_complied = sum(c.get("attestation_complied", 0) for c in summary["cells"])
    attest_declined = sum(c.get("attestation_declined", 0) for c in summary["cells"])
    lies = sum((c["lie_rate"] or 0.0) * c["n_claims"] for c in summary["cells"])
    turns = sum(c["turns"] for c in summary["cells"])
    trials = sum(c["trials"] for c in summary["cells"])
    sizes = [size for values in summary.get("lie_sizes_by_scenario", {}).values() for size in values]
    return {
        "lie_size_median": statistics.median(sizes) if sizes else None,
        "n_lie_sizes": len(sizes),
        "run_id": manifest.get("run_id"),
        "model": manifest.get("spec", {}).get("model"),
        # Manifests written before the counterpart model was recorded were self play.
        "counterpart_regime": manifest.get("counterpart_regime") or "self_play",
        "counterpart_model": manifest.get("counterpart_model") or manifest.get("spec", {}).get("model"),
        "counterpart_type": ",".join(manifest.get("spec", {}).get("counterpart_types", [])),
        "counterpart_policy_id": ",".join(policy_ids) or None,
        "module": module.get("version"),
        "attestation_timing": module.get("attestation_timing"),
        "max_rounds": manifest.get("spec", {}).get("max_rounds"),
        "negotiations": trials,
        "failed": trials_block.get("failed"),
        "forfeited_turns": trials_block.get("forfeited_turns"),
        "refusals": (manifest.get("refusals") or {}).get("model"),
        "claim_bearing_rate": (claims / turns) if turns else None,
        "attestation_compliance_rate": (attest_complied / attest_turns) if attest_turns else None,
        "attestation_refusal_rate": (attest_declined / attest_turns) if attest_turns else None,
        "lie_rate": (lies / claims) if claims else None,
        "usd_per_negotiation": None if (unpriced or not trials) else usd / trials,
        "unverified_share": None if unpriced else ((unverified / usd) if usd else 0.0),
        "unpriced_calls": unpriced,
        "calls": calls,
    }


def _refuse_mixed(rows: list[dict[str, Any]], field: str, what: str) -> None:
    values = sorted({str(r.get(field)) for r in rows})
    if len(values) > 1:
        raise ValueError(f"runs with different {what} cannot share a leaderboard: " + ", ".join(values))


def render_leaderboard(run_dirs: list[str | Path]) -> str:
    rows = [run_row(d) for d in run_dirs]
    regimes = sorted({r["counterpart_regime"] for r in rows})
    if len(regimes) > 1:
        raise ValueError(
            "runs with different counterpart regimes cannot share a leaderboard: "
            + ", ".join(regimes)
            + ". Self play rows and fixed counterpart rows measure different things."
        )
    _refuse_mixed(rows, "module", "module versions")
    _refuse_mixed(rows, "attestation_timing", "attestation timings")
    _refuse_mixed(rows, "max_rounds", "round counts")
    rows.sort(key=lambda r: (r["lie_rate"] is None, r["lie_rate"] or 0.0))
    lines = [
        f"Counterpart regime: {regimes[0]}" if regimes else "Counterpart regime: none",
        "",
        "| run | model | counterpart | policy id | negotiations | failed | forfeited | claim bearing | attest | refused | lie rate | median lie size | $ per negotiation | unverified $ share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fmt = lambda v, p: ("n/a" if v is None else f"{v:{p}}")  # noqa: E731
        policy_id = (r["counterpart_policy_id"] or "n/a")[:19]
        lines.append(
            f"| {r['run_id']} | {r['model']} | {r['counterpart_type']} | {policy_id} | {r['negotiations']} | "
            f"{fmt(r['failed'], 'd')} | {fmt(r['forfeited_turns'], 'd')} | "
            f"{fmt(r['claim_bearing_rate'], '.0%')} | {fmt(r['attestation_compliance_rate'], '.0%')} | "
            f"{fmt(r['attestation_refusal_rate'], '.0%')} | {fmt(r['lie_rate'], '.1%')} | {fmt(r['lie_size_median'], '+.0%')} | "
            f"{fmt(r['usd_per_negotiation'], '.4f')} | {fmt(r['unverified_share'], '.0%')} |"
        )
    return "\n".join(lines) + "\n"
