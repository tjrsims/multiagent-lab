"""Family-safe benchmark splits built from the views' own provenance.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
1. ``records_from_views`` turns each kept trial into a ``DyadTrialRecord``:
   its leakage family is the views' group (trial family, optionally with the
   counterpart assignment), so twins, rollouts, mirrored roles, and surface
   variants that already share a family stay together.
2. ``build_split`` allocates whole families to train/development/test with
   ``dyads.splits.assign_partitions`` (stratified by scenario, incentive,
   counterpart policy), returns a locked ``SplitManifest``, and runs
   ``audit_family_leakage`` so a manifest whose declared families hide a
   leak is rejected before anything is scored.
Where this fits: benchmark stage; the manifest id is stamped into the
``BenchmarkManifest`` and passed back to ``build_views(split_manifest=...)``
so every row carries its partition.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from interpretability.benchmark.views import BenchmarkViews
from interpretability.data.splits import PARTITIONS, SplitManifest
from interpretability.dyads.splits import (
    DEFAULT_FRACTIONS,
    DyadTrialRecord,
    FamilyLeakageAudit,
    StratificationReport,
    assign_partitions,
    audit_family_leakage,
)


BENCHMARK_SPLIT_VERSION = "benchmark-split/1"


class BenchmarkSplitError(ValueError):
    """Raised when views lack the provenance a safe split needs."""


def _identifier(value: object, *, field_name: str) -> str:
    text = str(value) if value is not None else ""
    if not text.strip():
        raise BenchmarkSplitError(f"{field_name} is required for every kept row")
    return text


def records_from_views(
    views: BenchmarkViews,
    *,
    model_pairing: str = "single_model",
) -> list[DyadTrialRecord]:
    """One record per trial; the leakage family is the views' group id."""
    if not views.trial_ids:
        raise BenchmarkSplitError("views carry no trial ids")
    seen: dict[str, DyadTrialRecord] = {}
    for index in range(views.n):
        trial_id = _identifier(views.trial_ids[index], field_name="trial_id")
        group = _identifier(views.group_ids[index], field_name="group_id")
        scenario = _identifier(views.scenarios[index], field_name="scenario")
        incentive = (
            views.incentive_conditions[index]
            if views.incentive_conditions else None
        ) or "unspecified"
        policy = (
            views.counterpart_types[index] if views.counterpart_types else None
        ) or "default"
        role = (views.pair_roles[index] if views.pair_roles else None) or "unpaired"
        record = DyadTrialRecord(
            trial_id=trial_id,
            dyad_id=group,
            base_instance_id=group,
            scenario_id=scenario,
            incentive_condition=str(incentive),
            counterpart_policy=str(policy),
            role_orientation=str(role),
            model_pairing=model_pairing,
        )
        existing = seen.get(trial_id)
        if existing is not None and existing != record:
            raise BenchmarkSplitError(
                f"trial {trial_id!r} appears with inconsistent provenance"
            )
        seen[trial_id] = record
    return [seen[key] for key in sorted(seen)]


def build_split(
    views: BenchmarkViews,
    *,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    seed: int = 42,
    tolerance: float = 0.1,
    lock: bool = True,
) -> tuple[SplitManifest, StratificationReport, FamilyLeakageAudit]:
    """Allocate whole families to partitions and audit the result."""
    records = records_from_views(views)
    manifest, report = assign_partitions(
        records, fractions=fractions, seed=seed, tolerance=tolerance
    )
    audit = audit_family_leakage(manifest, records)
    if audit.violations:
        raise BenchmarkSplitError(
            f"split manifest leaks families: {[v.kind for v in audit.violations]}"
        )
    populated = {assignment.partition for assignment in manifest.assignments}
    if lock and not manifest.locked and populated == set(PARTITIONS):
        # A locked manifest requires every partition to be non-empty; tiny
        # pilots stay unlocked and are scored on all rows (reported as such).
        manifest = replace(manifest, locked=True, manifest_id=None)
    return manifest, report, audit


__all__ = [
    "BENCHMARK_SPLIT_VERSION",
    "BenchmarkSplitError",
    "build_split",
    "records_from_views",
]
