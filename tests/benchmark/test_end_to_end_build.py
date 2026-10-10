"""Golden bundle through views -> split -> pairs -> scores -> manifest."""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from interpretability.benchmark.build import build_benchmark
from interpretability.benchmark.views import build_views
from interpretability.cli import cli
from interpretability.data.activation_dataset import load_activation_dataset
from tests.benchmark.golden import build_golden_dataset


@pytest.fixture(scope="module")
def golden_path(tmp_path_factory):
    return build_golden_dataset(tmp_path_factory.mktemp("golden") / "golden.json")


def test_golden_bundle_round_trips_with_rule_v2_provenance(golden_path) -> None:
    dataset = load_activation_dataset(golden_path)
    rows = dataset["metadata"]
    assert rows and all(row["runtime_path"] == "transactional" for row in rows)
    assert {row["pair_role"] for row in rows} == {"above", "below"}
    statuses = {row["claim_status"] for row in rows}
    assert "complete" in statuses and "no_relevant_action" in statuses
    rule_rows = [row for row in rows if row["label_source"] == "rule"]
    assert rule_rows and all(isinstance(row["emergent_ground_truth"], bool) for row in rule_rows)
    assert all(row["rule_version"] == "negotiation-rules/2" for row in rule_rows)
    assert all(row["label_source"] is None for row in rows if row["claim_status"] == "no_relevant_action")


def test_views_from_golden_bundle_keep_claim_bearing_twins(golden_path) -> None:
    dataset = load_activation_dataset(golden_path)
    views = build_views(dataset)
    # One claim-bearing turn per trial: 6+4 families x 2 twins.
    assert views.n == 20
    assert set(views.labels.tolist()) == {0, 1}
    assert views.labels.sum() == 10
    for role, label in zip(views.pair_roles, views.labels):
        assert (role == "below") == bool(label)
    assert views.report.excluded.get("label_source_missing") == 20  # the no-claim turns
    assert views.report.no_claim_rate_by_scenario["ultimatum_bluff"] == 0.5
    assert views.report.n_groups == 10
    for index in range(views.n):
        text = views.fact_informed_text(index)
        assert "counterpart_walkaway" not in text and "trial_id" not in text
    assert all(len(values) == 1 for values in views.claim_values)


def test_build_benchmark_writes_stable_manifest_and_oracle_sits_at_chance(golden_path, tmp_path) -> None:
    first = build_benchmark(golden_path, tmp_path / "one", n_bootstrap=60, split_seed=3)
    second = build_benchmark(golden_path, tmp_path / "two", n_bootstrap=60, split_seed=3)
    assert first.manifest.manifest_id == second.manifest.manifest_id
    for name in ("views_report", "split_manifest", "pairs", "scores", "manifest"):
        assert first.written[name].exists()
    manifest = json.loads(first.written["manifest"].read_text())
    assert manifest["manifest_id"] == first.manifest.manifest_id
    assert manifest["rule_version"] == "negotiation-rules/2"
    assert manifest["split_manifest_id"] == first.split_manifest.manifest_id
    assert first.pairs.parameters["pairs_over_all_rows"] == 10
    assert first.pairs.n_pairs == 2, "scoring pairs are built inside the test split"
    assert all(bal.twin_pairs == bal.pairs_formed for bal in first.pairs.bins)

    scores = {(s.detector, s.subset): s for s in first.scorecard.detector_scores}
    oracle = scores[("stated_claim_oracle", "paired")]
    assert abs(oracle.auroc.value - 0.5) < 1e-9, "same claim on both twins ties the oracle"
    rule = scores[("fact_informed_rule", "all")]
    assert rule.auroc.value == 1.0
    assert first.probe_selection is not None
    probe = scores[("linear_probe", "paired")]
    assert probe.n_rows == oracle.n_rows
    assert first.scorecard.extras["leakage_violations"] == 0
    assert first.scorecard.extras["scoring_partition"] == "test"
    payload = json.loads(first.written["scores"].read_text())
    assert payload["benchmark_manifest_id"] == first.manifest.manifest_id
    assert payload["behavioral"], "behavioral cells are reported"
    assert len(first.views.partitions) == first.views.n


def test_build_benchmark_label_source_changes_manifest(golden_path, tmp_path) -> None:
    rule = build_benchmark(golden_path, tmp_path / "rule", n_bootstrap=30, train_probe=False)
    assert rule.probe_selection is None
    assert all(s.detector != "linear_probe" for s in rule.scorecard.detector_scores)
    # model_judge labels were never produced by the fake runner: every row is
    # excluded and the build must refuse rather than guess.
    refused = CliRunner().invoke(
        cli, ["benchmark", "build", "--dataset", str(golden_path), "--out", str(tmp_path / "mj"),
              "--label-source", "model_judge"],
    )
    assert refused.exit_code == 1
    assert "BUILD FAILED" in refused.output
    assert not (tmp_path / "mj" / "manifest.json").exists()


def test_benchmark_cli_builds_and_reports(golden_path, tmp_path) -> None:
    result = CliRunner().invoke(
        cli,
        ["benchmark", "build", "--dataset", str(golden_path), "--out", str(tmp_path / "out"),
         "--n-bootstrap", "40", "--split-seed", "3"],
    )
    assert result.exit_code == 0, result.output
    assert "benchmark manifest bench_" in result.output
    assert "matched pairs: 2 inside the test split (10 over all rows)" in result.output
    assert "stated_claim_oracle" in result.output
    assert (tmp_path / "out" / "manifest.json").exists()
    bad = CliRunner().invoke(
        cli, ["benchmark", "build", "--dataset", str(tmp_path / "missing.json"), "--out", str(tmp_path / "x")],
    )
    assert bad.exit_code != 0


def test_paired_scoring_keeps_both_members_inside_the_scoring_partition(golden_path, tmp_path) -> None:
    """Pairs built over all rows and filtered to the test split left most scored
    rows without their partner, and the stated claim oracle read 0.89 on that
    unbalanced subset of a pooled live build while sitting at 0.50 over all pairs."""
    result = build_benchmark(golden_path, tmp_path / "scored", n_bootstrap=30, split_seed=3, train_probe=False)
    partitions = result.views.partitions
    pairs = result.pairs
    assert pairs.parameters["scoring_partition"] == "test"
    assert pairs.parameters["pairs_over_all_rows"] >= pairs.n_pairs
    for pos, neg in zip(pairs.positive_indices, pairs.negative_indices):
        assert partitions[int(pos)] == "test" and partitions[int(neg)] == "test"
    paired = [s for s in result.scorecard.detector_scores if s.subset == "paired"]
    assert paired and all(s.n_rows == 2 * pairs.n_pairs for s in paired)
    report = json.loads((tmp_path / "scored" / "pairs.json").read_text())
    assert report["parameters"]["scoring_partition"] == "test"
    unpartitioned = build_benchmark(golden_path, tmp_path / "all", n_bootstrap=30, split_seed=3, train_probe=False, scoring_partition=None)
    assert "scoring_partition" not in unpartitioned.pairs.parameters
    assert unpartitioned.pairs.n_pairs == pairs.parameters["pairs_over_all_rows"]
