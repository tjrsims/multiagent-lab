"""Pooled text datasets: ids never collide, families stay shared, pairs never cross a run."""

from __future__ import annotations

import copy
import json

import pytest
from click.testing import CliRunner

from end_bench.adjudication import export_adjudication_batch
from end_bench.cli import cli
from end_bench.data.pooled import (
    POOLED_DATASET_VERSION,
    PooledDatasetError,
    is_pooled_dataset,
    pool_text_datasets,
    source_id_for,
    split_source_id,
)
from end_bench.data.text_dataset import load_text_dataset, save_text_dataset
from interpretability.benchmark.pairs import build_matched_pairs, pair_scopes_for_views, twin_ids_for_views
from interpretability.benchmark.views import build_views


def _renamed_copy(dataset, run_id):
    other = copy.deepcopy(dataset)
    other["config"]["run_manifest"]["run_id"] = run_id
    other["config"]["model_name"] = f"bedrock:other-model-{run_id}"
    return other


def test_pool_prefixes_ids_keeps_families_and_round_trips(session_dataset, tmp_path) -> None:
    other = _renamed_copy(session_dataset, "copy-b")
    pooled = pool_text_datasets([session_dataset, other])

    n = len(session_dataset["metadata"])
    assert pooled["config"]["n_rows"] == 2 * n
    assert len(pooled["metadata"]) == 2 * n
    for column, values in pooled["labels"].items():
        assert len(values) == 2 * n, column
    assert is_pooled_dataset(pooled) and not is_pooled_dataset(session_dataset)
    info = pooled["config"]["run_manifest"]["pooled"]
    assert info["version"] == POOLED_DATASET_VERSION
    assert [item["run_id"] for item in info["sources"]] == ["session-smoke", "copy-b"]

    keys = {(row["trial_id"], row["generation_record_id"]) for row in pooled["metadata"]}
    assert len(keys) == 2 * n, "every (trial, call) pair is unique after pooling"
    sources = {row["source_run_id"] for row in pooled["metadata"]}
    assert sources == {"session-smoke", "copy-b"}
    first, mirrored = pooled["metadata"][0], pooled["metadata"][n]
    assert split_source_id(first["trial_id"]) == ("session-smoke", session_dataset["metadata"][0]["trial_id"])
    assert split_source_id(mirrored["trial_id"]) == ("copy-b", session_dataset["metadata"][0]["trial_id"])
    # Families and scenario instances are the same facts in both runs: shared on purpose.
    assert first["trial_family_id"] == mirrored["trial_family_id"] == session_dataset["metadata"][0]["trial_family_id"]
    assert first["scenario_instance_id"] == mirrored["scenario_instance_id"]
    assert first["source_model"] != mirrored["source_model"]

    path = save_text_dataset(tmp_path / "pooled.json", pooled)
    loaded = load_text_dataset(path)
    assert loaded["config"]["dataset_hash"] == pooled["config"]["dataset_hash"]


def test_pooled_views_pairs_and_export_never_cross_a_source(session_dataset, tmp_path) -> None:
    pooled = pool_text_datasets([session_dataset, _renamed_copy(session_dataset, "copy-b")])
    single = build_views(session_dataset)
    views = build_views(pooled)
    assert views.n == 2 * single.n
    assert views.report.source_runs == ("copy-b", "session-smoke")
    assert "source_runs" in views.report.to_dict()
    assert "source_runs" not in single.report.to_dict(), "single-run reports keep their exact shape"
    assert single.sources == ()
    assert set(views.sources) == {"session-smoke", "copy-b"}
    # One family from two runs stays one leakage group.
    assert views.report.n_groups == single.report.n_groups

    scopes = pair_scopes_for_views(views)
    assert scopes is not None and pair_scopes_for_views(single) is None
    pairs = build_matched_pairs(
        views.labels, views.claim_values, views.scenarios, views.group_ids,
        twin_ids=twin_ids_for_views(views), scopes=scopes,
    )
    single_pairs = build_matched_pairs(
        single.labels, single.claim_values, single.scenarios, single.group_ids,
        twin_ids=twin_ids_for_views(single),
    )
    assert pairs.n_pairs == 2 * single_pairs.n_pairs
    for pos, neg in zip(pairs.positive_indices, pairs.negative_indices):
        assert views.sources[pos] == views.sources[neg]
    assert pairs.balance_report()["parameters"] == {"twin_ids_supplied": True, "scoped": True}
    assert all("scope" in item for item in pairs.balance_report()["bins"])
    assert all("scope" not in item for item in single_pairs.balance_report()["bins"])

    manifest = export_adjudication_batch(pooled, tmp_path / "batch", n=12, seed=3, gold_fraction=0.0)
    key = json.loads((tmp_path / "batch" / "private" / "KEY.json").read_text())
    assert manifest.n_items == 12
    assert {item["source_run_id"] for item in key.values()} <= {"session-smoke", "copy-b"}
    assert all(item["source_run_id"] is not None for item in key.values())


def test_pool_refuses_unsafe_inputs(session_dataset) -> None:
    with pytest.raises(PooledDatasetError, match="distinct"):
        pool_text_datasets([session_dataset, copy.deepcopy(session_dataset)])
    other_track = _renamed_copy(session_dataset, "copy-c")
    other_track["config"]["experiment_track"] = "single_agent_white_box"
    with pytest.raises(PooledDatasetError, match="tracks"):
        pool_text_datasets([session_dataset, other_track])
    other_registry = _renamed_copy(session_dataset, "copy-d")
    other_registry["config"]["schema_registry_checksum"] = "0" * 64
    with pytest.raises(PooledDatasetError, match="schema registries"):
        pool_text_datasets([session_dataset, other_registry])
    pooled = pool_text_datasets([session_dataset, other_registry], allow_mixed_registry=True)
    assert len(pooled["config"]["run_manifest"]["pooled"]["schema_registry_checksums"]) == 2
    with pytest.raises(PooledDatasetError, match="at least one"):
        pool_text_datasets([])
    nameless = copy.deepcopy(session_dataset)
    nameless["config"]["run_manifest"] = {}
    assert source_id_for(nameless).startswith("dataset_")


def test_cli_pool_writes_a_loadable_dataset(session_smoke_run, tmp_path) -> None:
    source = session_smoke_run.text_dataset_path
    other = _renamed_copy(load_text_dataset(source), "copy-e")
    from end_bench.data.text_dataset import _content_hash

    other["config"]["dataset_hash"] = _content_hash(other)
    other_path = save_text_dataset(tmp_path / "other.json", other)
    runner = CliRunner()
    duplicate = runner.invoke(cli, ["pool", "--dataset", str(source), "--dataset", str(source), "--out", str(tmp_path / "dup.json")])
    assert duplicate.exit_code != 0 and "distinct" in duplicate.output
    result = runner.invoke(cli, ["pool", "--dataset", str(source), "--dataset", str(other_path), "--out", str(tmp_path / "pooled.json")])
    assert result.exit_code == 0, result.output
    assert "pooled 2 datasets" in result.output
    loaded = load_text_dataset(tmp_path / "pooled.json")
    assert loaded["config"]["model_name"] == "pooled"
    judged = runner.invoke(cli, ["judge", "--dataset", str(tmp_path / "pooled.json"), "--out", str(tmp_path / "judge"), "--judge", "oracle", "--n-bootstrap", "20"])
    assert judged.exit_code == 0, judged.output
    assert "matched pairs:" in judged.output
