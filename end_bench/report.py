"""Aggregate builds, judge testbeds, adjudication, and outcomes into one report.

Every number printed here comes from a JSON artifact written by another step
(`scores.json`, `judges.json`, `adjudication.json`, `outcomes.json`) and
carries the interval that artifact recorded. The report never recomputes a
statistic; it renders.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_VERSION = "end-bench-report/1"


class ReportError(ValueError):
    """Raised when an input artifact is missing or malformed."""


def _load(path: Path) -> Any:
    if not path.exists():
        raise ReportError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(estimate: Mapping[str, Any] | None, digits: int = 3) -> str:
    if not estimate:
        return "n/a"
    value, low, high = estimate.get("value"), estimate.get("ci_low"), estimate.get("ci_high")
    if value is None:
        return "n/a"
    if low is None or high is None or low != low or high != high:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


@dataclass(frozen=True)
class BuildArtifacts:
    name: str
    manifest: Mapping[str, Any]
    scores: Mapping[str, Any]
    judges: Mapping[str, Any] | None = None
    adjudication: Mapping[str, Any] | None = None
    outcomes: Mapping[str, Any] | None = None


def collect_build(name: str, build_dir: str | Path) -> BuildArtifacts:
    base = Path(build_dir)
    judges = base / "judges.json"
    adjudication = base / "adjudication.json"
    outcomes = base / "outcomes.json"
    return BuildArtifacts(
        name=name,
        manifest=_load(base / "manifest.json"),
        scores=_load(base / "scores.json"),
        judges=_load(judges) if judges.exists() else None,
        adjudication=_load(adjudication) if adjudication.exists() else None,
        outcomes=_load(outcomes) if outcomes.exists() else None,
    )


def _behavior_table(builds: Sequence[BuildArtifacts]) -> list[str]:
    lines = ["## Behavior: contradiction rate on claim-bearing turns", "",
             "| model | scenario | condition | n turns | n trials | contradiction rate | strict instance pass |",
             "|---|---|---|---|---|---|---|"]
    for build in builds:
        for cell in build.scores.get("behavioral", []):
            keys = cell.get("keys", {})
            lines.append(
                f"| {build.name} | {keys.get('scenario')} | {keys.get('incentive_condition')} | "
                f"{cell.get('n_rows')} | {cell.get('n_trials')} | {_ci(cell.get('contradiction_rate'))} | "
                f"{_ci(cell.get('strict_instance_pass'))} |"
            )
    return lines


def _detector_table(builds: Sequence[BuildArtifacts]) -> list[str]:
    lines = ["## Detectors: AUROC on all rows vs matched pairs (test partition when available)", "",
             "Separation is max(AUROC, 1 - AUROC): a nuisance cue such as response length carries "
             "signal in either direction, and an AUROC of 0.35 means 0.65 with the sign flipped.", "",
             "| model | detector | subset | n | AUROC | separation | TPR@1%FPR | TPR@5%FPR |", "|---|---|---|---|---|---|---|---|"]
    for build in builds:
        extras = build.scores.get("extras", {})
        partition = extras.get("scoring_partition")
        reason = extras.get("scoring_partition_reason")
        if reason:
            lines.insert(4, f"Note for {build.name}: {reason}.")
        for score in build.scores.get("detector_scores", []):
            tpr = score.get("tpr_at_fpr", {})
            auroc = (score.get("auroc") or {}).get("value")
            separation = "n/a" if not isinstance(auroc, (int, float)) else f"{max(auroc, 1.0 - auroc):.3f}{' (flipped)' if auroc < 0.5 else ''}"
            lines.append(
                f"| {build.name} | {score.get('detector')} | {score.get('subset')}"
                f"{'' if partition is None else f' ({partition})'} | {score.get('n_rows')} | "
                f"{_ci(score.get('auroc'))} | {separation} | {_ci(tpr.get('0.01'))} | {_ci(tpr.get('0.05'))} |"
            )
    return lines


def _gap_table(builds: Sequence[BuildArtifacts]) -> list[str]:
    lines = ["## Paired detector gaps (A minus B on identical rows)", "",
             "| model | A | B | subset | AUROC difference | groups W/L/T |", "|---|---|---|---|---|---|"]
    for build in builds:
        for gap in build.scores.get("paired_gaps", []):
            lines.append(
                f"| {build.name} | {gap.get('detector_a')} | {gap.get('detector_b')} | {gap.get('subset')} | "
                f"{_ci(gap.get('auroc_difference'))} | {gap.get('group_wins')}/{gap.get('group_losses')}/{gap.get('group_ties')} |"
            )
    return lines


def _judge_tables(builds: Sequence[BuildArtifacts]) -> list[str]:
    lines: list[str] = []
    any_judges = [b for b in builds if b.judges]
    if not any_judges:
        return lines
    lines += ["## Judge testbed: error rates against computed truth", "",
              "| model | judge | access | scenario | n | FPR | FNR | undetermined | agreement |",
              "|---|---|---|---|---|---|---|---|---|"]
    for build in any_judges:
        for row in build.judges.get("error_rows", []):
            fpr = row.get("false_positive_rate"); fnr = row.get("false_negative_rate")
            lines.append(
                f"| {build.name} | {row.get('judge_name')} | {row.get('access')} | {row.get('scenario')} | {row.get('n')} | "
                f"{'n/a' if fpr is None else f'{fpr:.2f}'} | {'n/a' if fnr is None else f'{fnr:.2f}'} | "
                f"{row.get('undetermined_rate', 0.0):.2f} | {_ci(row.get('agreement'))} |"
            )
    lines += ["", "## Judge testbed: detector scores and access gaps", "",
              "| model | detector | subset | n | AUROC |", "|---|---|---|---|---|"]
    for build in any_judges:
        for score in build.judges.get("detector_scores", []):
            lines.append(f"| {build.name} | {score.get('detector')} | {score.get('subset')} | {score.get('n_rows')} | {_ci(score.get('auroc'))} |")
    lines += ["", "| model | fact-informed minus transcript-only | subset | difference |", "|---|---|---|---|"]
    for build in any_judges:
        for gap in build.judges.get("gaps", []):
            lines.append(f"| {build.name} | {gap.get('detector_a')} | {gap.get('subset')} | {_ci(gap.get('auroc_difference'))} |")
    return lines


def _adjudication_table(builds: Sequence[BuildArtifacts]) -> list[str]:
    lines: list[str] = []
    rows = [b for b in builds if b.adjudication]
    if not rows:
        return lines
    lines += ["## Human adjudication", "", "| model | items | raters | Fleiss kappa | rule-vs-human agreement | undetermined |", "|---|---|---|---|---|---|"]
    for build in rows:
        a = build.adjudication
        kappa = a.get("kappa")
        lines.append(
            f"| {build.name} | {a.get('n_items')} | {a.get('n_raters')} | "
            f"{'n/a' if kappa is None or kappa != kappa else f'{kappa:.3f}'} | "
            f"{_ci(a.get('rule_vs_human_agreement'))} | {a.get('undetermined_items')} |"
        )
    return lines


def _outcomes_table(builds: Sequence[BuildArtifacts]) -> list[str]:
    lines: list[str] = []
    rows = [b for b in builds if b.outcomes]
    if not rows:
        return lines
    lines += ["## Does lying pay? Mean ZOPA share, lying vs honest trials", "",
              "| model | scenario | n lying | n honest | share (lying) | share (honest) | difference | agreement rate lying/honest |",
              "|---|---|---|---|---|---|---|---|"]
    for build in rows:
        for cell in build.outcomes.get("lying_pays", []):
            def f(v): return "n/a" if v is None else f"{v:.3f}"
            lines.append(
                f"| {build.name} | {cell.get('scenario')} | {cell.get('n_lying')} | {cell.get('n_honest')} | "
                f"{f(cell.get('mean_share_lying'))} | {f(cell.get('mean_share_honest'))} | {_ci(cell.get('difference'))} | "
                f"{f(cell.get('agreement_rate_lying'))}/{f(cell.get('agreement_rate_honest'))} |"
            )
    return lines


def render_report(builds: Sequence[BuildArtifacts], *, title: str = "END-Bench results") -> str:
    """Markdown report over one or more builds."""
    if not builds:
        raise ReportError("at least one build is required")
    lines = [f"# {title}", "", f"Report version `{REPORT_VERSION}`. Every interval is a family-clustered bootstrap "
             "interval recorded by the producing step; nothing is recomputed here.", "",
             "## Provenance", "", "| model | benchmark manifest | rule | spec | label source | rows kept | pairs |", "|---|---|---|---|---|---|---|"]
    for build in builds:
        m = build.manifest
        report = m.get("views_report", {})
        pairs = (m.get("pairs_report") or {}).get("n_pairs")
        lines.append(
            f"| {build.name} | `{m.get('manifest_id')}` | {m.get('rule_version')} | {m.get('scenario_spec_version')} | "
            f"{m.get('label_source')} | {report.get('n_kept')}/{report.get('n_input_rows')} | {pairs} |"
        )
    lines.append("")
    lines += _behavior_table(builds) + [""] + _detector_table(builds) + [""] + _gap_table(builds)
    for block in (_judge_tables(builds), _adjudication_table(builds), _outcomes_table(builds)):
        if block:
            lines += [""] + block
    skipped = [(b.name, s) for b in builds for s in b.scores.get("extras", {}).get("skipped", [])]
    if skipped:
        lines += ["", "## Skipped scores", ""] + [f"- {name}: {s.get('detector')} ({s.get('subset')}): {s.get('reason')}" for name, s in skipped]
    return "\n".join(lines) + "\n"


__all__ = ["REPORT_VERSION", "BuildArtifacts", "ReportError", "collect_build", "render_report"]
