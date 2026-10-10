"""Adjudication: blinded export, kappa, gold accuracy, agreement, label application."""

from __future__ import annotations

import csv
import json
import math

import pytest

from end_bench.adjudication import (
    AdjudicationError,
    apply_human_labels,
    export_adjudication_batch,
    fleiss_kappa,
    import_ratings,
)
from interpretability.benchmark.views import build_views


def _write_ratings(path, key, rater, *, flip_ids=(), undetermined_ids=()):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "rater_id", "label"])
        writer.writeheader()
        for item_id, info in key.items():
            truth = info["gold_label"] if info["is_gold"] else info["rule_label"]
            label = truth
            if item_id in flip_ids:
                label = "false" if truth == "true" else "true"
            if item_id in undetermined_ids:
                label = "cannot_determine"
            writer.writerow({"item_id": item_id, "rater_id": rater, "label": label})


def test_export_is_blinded_stratified_and_keyed(session_dataset, tmp_path) -> None:
    manifest = export_adjudication_batch(session_dataset, tmp_path / "batch", n=8, seed=1, gold_fraction=0.25)
    assert manifest.n_items >= 8 and manifest.n_gold >= 1
    blind = list(csv.DictReader((tmp_path / "batch" / "blind.csv").open()))
    informed = list(csv.DictReader((tmp_path / "batch" / "informed.csv").open()))
    assert len(blind) == len(informed) == manifest.n_items
    assert all(row["label"] == "" for row in blind)
    assert "private_facts" not in blind[0]
    assert all(json.loads(row["private_facts"]) for row in informed)
    key = json.loads((tmp_path / "batch" / "private" / "KEY.json").read_text())
    assert set(key) == {row["item_id"] for row in blind}
    gold = [k for k, v in key.items() if v["is_gold"]]
    assert len(gold) == manifest.n_gold
    assert all(key[k]["gold_label"] in {"true", "false"} for k in gold)
    assert (tmp_path / "batch" / "INSTRUCTIONS.txt").exists()
    with pytest.raises(AdjudicationError):
        export_adjudication_batch(session_dataset, tmp_path / "bad", n=0)


def test_fleiss_kappa_known_values() -> None:
    assert fleiss_kappa([["true", "true"], ["false", "false"]]) == 1.0
    disagree = fleiss_kappa([["true", "false"], ["false", "true"], ["true", "false"], ["false", "true"]])
    assert disagree < 0
    assert math.isnan(fleiss_kappa([]))
    with pytest.raises(AdjudicationError):
        fleiss_kappa([["true"]])


def test_import_computes_kappa_gold_and_agreement_then_labels_apply(session_dataset, tmp_path) -> None:
    batch = tmp_path / "batch"
    export_adjudication_batch(session_dataset, batch, n=10, seed=2, gold_fraction=0.2)
    key = json.loads((batch / "private" / "KEY.json").read_text())
    non_gold = [k for k, v in key.items() if not v["is_gold"]]
    gold = [k for k, v in key.items() if v["is_gold"]]
    r1, r2, r3 = (tmp_path / f"r{i}.csv" for i in (1, 2, 3))
    _write_ratings(r1, key, "alice")
    _write_ratings(r2, key, "bob")
    _write_ratings(r3, key, "carol", flip_ids={non_gold[0]}, undetermined_ids={gold[0]} if gold else ())
    report = import_ratings(batch, [r1, r2, r3], n_bootstrap=50)
    assert report.n_raters == 3 and report.n_items == len(key) and report.n_rated_by_all == len(key)
    assert 0.5 < report.kappa <= 1.0
    assert report.gold_accuracy_by_rater["alice"] == 1.0
    if gold:
        assert report.gold_accuracy_by_rater["carol"] < 1.0
    # Majority label wins: two honest raters outvote one flip.
    assert report.majority_labels[non_gold[0]] == key[non_gold[0]]["rule_label"]
    assert report.rule_vs_human_agreement.value == 1.0
    assert report.undetermined_items == 0
    assert set(report.human_labels_by_row) == {key[k]["row_index"] for k in non_gold}

    labeled = apply_human_labels(session_dataset, report)
    human_rows = [row for row in labeled["metadata"] if "human_label" in row]
    assert len(human_rows) == len(non_gold)
    views = build_views(labeled, label_source="human")
    assert views.n == len(non_gold)
    assert views.label_sources == tuple(["rule"] * views.n) or all(views.labels.tolist())  # source column is the projection
    assert set(views.report.excluded) <= {"label_unavailable"}

    with pytest.raises(AdjudicationError):
        import_ratings(batch, [])
    bad = tmp_path / "bad.csv"
    bad.write_text("item_id,rater_id,label\nnot-an-item,x,true\n")
    with pytest.raises(AdjudicationError):
        import_ratings(batch, [bad])


def test_ties_become_undetermined_and_are_excluded_from_agreement(session_dataset, tmp_path) -> None:
    batch = tmp_path / "batch"
    export_adjudication_batch(session_dataset, batch, n=6, seed=3, gold_fraction=0.0)
    key = json.loads((batch / "private" / "KEY.json").read_text())
    first = next(iter(key))
    r1, r2 = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_ratings(r1, key, "a")
    _write_ratings(r2, key, "b", flip_ids={first})
    report = import_ratings(batch, [r1, r2], n_bootstrap=20)
    assert report.majority_labels[first] == "cannot_determine"
    assert report.undetermined_items == 1
    assert first not in {k for k, v in key.items() if v["row_index"] in report.human_labels_by_row}


def test_import_writes_a_loadable_activation_bundle_for_white_box_datasets(tmp_path) -> None:
    """Human labels on an activation dataset must round-trip through the safe JSON+NPZ format."""
    import importlib.util
    from pathlib import Path

    from click.testing import CliRunner

    from end_bench.cli import cli
    from interpretability.data.activation_dataset import load_activation_dataset

    golden_path = Path(__file__).resolve().parents[1] / "benchmark" / "golden.py"
    spec = importlib.util.spec_from_file_location("end_bench_test_golden", golden_path)
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)

    manifest = golden.build_golden_dataset(tmp_path / "golden.json", ultimatum_families=4, capability_families=2)
    dataset = load_activation_dataset(manifest)
    export_adjudication_batch(dataset, tmp_path / "batch", n=6, seed=3, gold_fraction=0.25)
    key = json.loads((tmp_path / "batch" / "private" / "KEY.json").read_text())
    r1, r2 = tmp_path / "r1.csv", tmp_path / "r2.csv"
    _write_ratings(r1, key, "r1")
    _write_ratings(r2, key, "r2")

    result = CliRunner().invoke(cli, [
        "adjudicate", "import", "--batch", str(tmp_path / "batch"),
        "--ratings", str(r1), "--ratings", str(r2),
        "--dataset", str(manifest), "--out", str(tmp_path / "out"), "--n-bootstrap", "20",
    ])
    assert result.exit_code == 0, result.output

    relabeled = load_activation_dataset(tmp_path / "out" / "dataset_with_human_labels.json")
    assert any(row.get("human_label") is not None for row in relabeled["metadata"])
    assert relabeled["activations"], "activation arrays must survive the relabel"
    human_views = build_views(relabeled, label_source="human")
    assert human_views.n >= 1
