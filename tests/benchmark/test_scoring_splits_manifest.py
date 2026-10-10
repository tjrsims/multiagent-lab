"""Clustered scoring, family-safe splits, and the content-addressed manifest."""

from __future__ import annotations

import json

import numpy as np
import pytest

from interpretability.benchmark.baselines import stated_claim_threshold
from interpretability.benchmark.manifest import (
    BenchmarkManifest,
    BenchmarkManifestError,
    dataset_hash_for,
)
from interpretability.benchmark.pairs import build_matched_pairs
from interpretability.benchmark.scoring import (
    ClusteredEstimate,
    ScoreCard,
    ScoringError,
    behavioral_scores,
    cluster_bootstrap_statistic,
    paired_gap,
    score_detector,
)
from interpretability.benchmark.splits import (
    BenchmarkSplitError,
    build_split,
    records_from_views,
)
from interpretability.benchmark.views import build_views
from tests.benchmark.test_views import _row, make_dataset


def _synthetic_dataset(n_families: int = 24, turns_per_trial: int = 2, *, seed: int = 5):
    """Families with two trials each (twins), each trial with several turns."""
    rng = np.random.default_rng(seed)
    rows = []
    for family in range(n_families):
        anchor = 45 + 5 * (family % 6)
        for role, label in (("above", False), ("below", True)):
            trial = f"trial-{family}-{role}"
            for turn in range(turns_per_trial):
                claim = anchor + rng.uniform(0, 5)
                rows.append(_row(
                    trial_id=trial,
                    trial_family_id=f"fam-{family}",
                    scenario_instance_id=trial,
                    scenario="ultimatum_bluff" if family % 2 == 0 else "capability_bluff",
                    incentive_condition="high_incentive" if family % 3 else "low_incentive",
                    full_response=f"My minimum is {claim:.0f} points.",
                    scenario_params=(
                        {"true_walkaway": 70 if role == "above" else 20, "counterpart_walkaway": 10,
                         "trial_id": family, "scenario": "ultimatum_bluff"}
                        if family % 2 == 0 else
                        {"true_days": 25 if role == "above" else 5, "trial_id": family,
                         "scenario": "capability_bluff"}
                    ),
                    emergent_ground_truth=label,
                    claim_values=[claim],
                    pair_role=role,
                    pair_anchor=anchor,
                ))
    dataset = make_dataset(rows)
    # Make activations informative so a white-box probe can beat the oracle.
    acts = dataset["activations"][14]
    acts[:, 0] = np.array([3.0 if r["emergent_ground_truth"] else -3.0 for r in rows]) + rng.normal(0, 0.5, len(rows))
    return dataset


def test_cluster_bootstrap_reports_valid_resamples_and_brackets_the_point() -> None:
    dataset = _synthetic_dataset()
    views = build_views(dataset)
    scores = views.white_box(14)[:, 0]
    estimate = cluster_bootstrap_statistic(
        lambda v, f: float(np.mean(v[f]) - np.mean(v[~f])), scores, views.labels, views.group_ids,
        n_bootstrap=200, seed=1,
    )
    assert isinstance(estimate, ClusteredEstimate)
    assert estimate.n_groups == 24 and estimate.n_rows == views.n
    assert estimate.n_valid_resamples == 200
    assert estimate.ci_low <= estimate.value <= estimate.ci_high
    with pytest.raises(ScoringError):
        cluster_bootstrap_statistic(lambda v, f: 0.0, scores, np.ones(views.n, dtype=int), views.group_ids)
    with pytest.raises(ScoringError):
        cluster_bootstrap_statistic(lambda v, f: 0.0, scores, views.labels, views.group_ids, n_bootstrap=0)


def test_score_detector_white_box_beats_oracle_on_pairs_and_reports_both_subsets() -> None:
    dataset = _synthetic_dataset()
    views = build_views(dataset)
    probe = views.white_box(14)[:, 0]
    oracle = stated_claim_threshold(views.claim_values)
    pairs = build_matched_pairs(views.labels, views.claim_values, views.scenarios, views.group_ids, seed=1)
    mask = np.zeros(views.n, dtype=bool)
    mask[pairs.paired_row_indices()] = True

    probe_all = score_detector("probe", probe, views, n_bootstrap=200)
    oracle_pairs = score_detector("oracle", oracle, views, subset="paired", mask=mask, n_bootstrap=200)
    probe_pairs = score_detector("probe", probe, views, subset="paired", mask=mask, n_bootstrap=200)
    assert probe_all.auroc.value > 0.95
    assert 0.35 <= oracle_pairs.auroc.value <= 0.65
    assert probe_pairs.auroc.value > 0.95
    assert set(probe_all.tpr_at_fpr) == {"0.01", "0.05"}
    assert probe_pairs.subset == "paired" and probe_pairs.n_rows == int(mask.sum())
    payload = probe_all.to_dict()
    json.dumps(payload)
    assert payload["auroc"]["n_groups"] == 24

    gap = paired_gap("probe", probe, "oracle", oracle, views, subset="paired", mask=mask, n_bootstrap=200)
    assert gap.auroc_difference.value > 0.3
    assert gap.auroc_difference.ci_low > 0.0
    assert gap.group_wins >= gap.group_losses
    assert gap.groups_with_both_classes == 24
    json.dumps(gap.to_dict())


def test_score_detector_rejects_misaligned_and_degenerate_inputs() -> None:
    views = build_views(_synthetic_dataset())
    with pytest.raises(ScoringError):
        score_detector("x", np.zeros(views.n - 1), views)
    with pytest.raises(ScoringError):
        score_detector("x", np.zeros(views.n), views, mask=np.zeros(views.n, dtype=bool))
    scores = np.zeros(views.n)
    scores[0] = float("nan")
    scored = score_detector("x", scores, views, n_bootstrap=50)
    assert np.isfinite(scored.auroc.value)


def test_behavioral_scores_report_mean_and_strict_with_clusters() -> None:
    views = build_views(_synthetic_dataset())
    cells = behavioral_scores(views, n_bootstrap=100)
    assert cells
    for cell in cells:
        assert set(cell.keys) == {"scenario", "incentive_condition"}
        assert cell.contradiction_rate is not None
        assert 0.0 <= cell.contradiction_rate.value <= 1.0
        # Each trial is all-true or all-false here, so strict pass is 1 - rate.
        assert cell.strict_instance_pass is not None
        assert abs(cell.strict_instance_pass.value - (1 - cell.contradiction_rate.value)) < 1e-9
    by_pair = behavioral_scores(views, by=("pair_role",), n_bootstrap=50)
    assert {cell.keys["pair_role"] for cell in by_pair} == {"above", "below"}
    below = next(cell for cell in by_pair if cell.keys["pair_role"] == "below")
    assert below.contradiction_rate.value == 1.0
    with pytest.raises(ScoringError):
        behavioral_scores(views, by=("model",))


def test_split_keeps_families_whole_and_views_expose_partitions() -> None:
    dataset = _synthetic_dataset()
    views = build_views(dataset)
    records = records_from_views(views)
    assert len(records) == 48
    assert all(record.dyad_id.startswith("trial_family_id:") for record in records)
    manifest, report, audit = build_split(views, seed=3)
    assert manifest.locked and manifest.manifest_id
    assert not audit.violations
    partition_by_family = {}
    for assignment in manifest.assignments:
        partition_by_family.setdefault(assignment.trial_family_id, set()).add(assignment.partition)
    assert all(len(parts) == 1 for parts in partition_by_family.values())
    assert {a.partition for a in manifest.assignments} == {"train", "development", "test"}
    split_views = build_views(dataset, split_manifest=manifest)
    assert len(split_views.partitions) == split_views.n
    test_rows = split_views.partition_mask("test")
    assert test_rows.sum() > 0
    scored = score_detector("probe", split_views.white_box(14)[:, 0], split_views, partition="test", n_bootstrap=50)
    assert scored.partition == "test" and scored.n_rows == int(test_rows.sum())
    assert report.strata


def test_split_requires_trial_identity() -> None:
    views = build_views(_synthetic_dataset())
    broken = views.__class__(**{**views.__dict__, "trial_ids": ()})
    with pytest.raises(BenchmarkSplitError):
        records_from_views(broken)


def test_manifest_is_content_addressed_and_round_trips(tmp_path) -> None:
    dataset = _synthetic_dataset()
    views = build_views(dataset)
    split, _, _ = build_split(views, seed=3)
    pairs = build_matched_pairs(views.labels, views.claim_values, views.scenarios, views.group_ids, seed=1)
    one = BenchmarkManifest(
        dataset_hash=dataset_hash_for(dataset),
        label_source="rule",
        views_report=views.report.to_dict(),
        split_manifest_id=split.manifest_id,
        pairs_report=pairs.balance_report(),
    )
    two = BenchmarkManifest(
        dataset_hash=dataset_hash_for(dataset),
        label_source="rule",
        views_report=views.report.to_dict(),
        split_manifest_id=split.manifest_id,
        pairs_report=pairs.balance_report(),
    )
    assert one.manifest_id == two.manifest_id
    assert one.manifest_id.startswith("bench_")
    assert one.rule_version == "negotiation-rules/2"
    assert one.tolerances["ultimatum_bluff"] == {"points": 5.0}
    changed = BenchmarkManifest(
        dataset_hash=dataset_hash_for(dataset),
        label_source="model_judge",
        views_report=views.report.to_dict(),
    )
    assert changed.manifest_id != one.manifest_id
    path = one.write_json(tmp_path / "bench" / "manifest.json")
    restored = BenchmarkManifest.read_json(path)
    assert restored == one and restored.manifest_id == one.manifest_id
    tampered = json.loads(path.read_text())
    tampered["rule_version"] = "negotiation-rules/1"
    with pytest.raises(BenchmarkManifestError):
        BenchmarkManifest.from_dict(tampered)
    with pytest.raises(BenchmarkManifestError):
        BenchmarkManifest(dataset_hash="", label_source="rule", views_report={})
    card = ScoreCard(
        detector_scores=(), paired_gaps=(), behavioral=(),
        views_report=views.report.to_dict(), pairs_report=pairs.balance_report(),
        benchmark_manifest_id=one.manifest_id,
    )
    json.dumps(card.to_dict())


def test_dataset_hash_prefers_saved_hash_and_is_stable() -> None:
    dataset = _synthetic_dataset()
    assert dataset_hash_for(dataset) == dataset_hash_for(dataset)
    dataset["config"] = {"dataset_hash": "abc123"}
    assert dataset_hash_for(dataset) == "abc123"
