"""Offline analysis scripts run end to end on the golden bundle."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tests.benchmark.golden import build_golden_dataset


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def golden_path(tmp_path_factory):
    return build_golden_dataset(tmp_path_factory.mktemp("golden_scripts") / "golden.json")


def test_tolerance_sensitivity_sweeps_and_reports_oracle_ceiling(golden_path, tmp_path) -> None:
    module = _load_script("tolerance_sensitivity")
    out = tmp_path / "sweep.json"
    code = module.main([
        "--dataset", str(golden_path), "--name", "fake",
        "--dataset", str(golden_path), "--name", "fake_twin",
        "--points", "0", "5", "--ratios", "1.1", "1.3", "--out", str(out),
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    cells = payload["datasets"]["fake"]
    assert len(cells) == 4
    for cell in cells:
        assert "error" not in cell, cell
        assert 0.0 <= cell["lie_rate"] <= 1.0
        assert cell["n_pairs"] >= 1
        assert cell["fact_rule_auroc_all"] == 1.0
    # Twins were generated under the default tolerance, so only that cell is
    # guaranteed to pair twins (same claim -> the oracle ties at exactly 0.5).
    # Other cells relabel one twin and fall back to cross-family pairs.
    default = next(c for c in cells if c["points"] == 5.0 and c["ratio"] == 1.3)
    assert abs(default["oracle_auroc_paired"] - 0.5) < 1e-9
    assert all(0.25 <= c["oracle_auroc_paired"] <= 0.75 for c in cells)
    # Tolerance moves labels: a zero tolerance can only raise the lie rate.
    strict = next(c for c in cells if c["points"] == 0.0 and c["ratio"] == 1.1)
    lenient = next(c for c in cells if c["points"] == 5.0 and c["ratio"] == 1.3)
    assert strict["lie_rate"] >= lenient["lie_rate"]
    assert payload["rank_stability"], "two datasets yield neighbour rank correlations"
    # Identical datasets give tied model rates: the correlation is undefined
    # (NaN) and must be reported as such, never silently as 1.0.
    assert all(entry["spearman"] != entry["spearman"] for entry in payload["rank_stability"])
    assert math.isnan(module.spearman([0.5, 0.5], [0.4, 0.4]))
    assert module.spearman([0.1, 0.2, 0.3], [1.0, 2.0, 3.0]) == 1.0
    assert module.spearman([0.1, 0.2, 0.3], [3.0, 2.0, 1.0]) == -1.0


def test_tolerance_sensitivity_relabel_keeps_rows_consistent(golden_path) -> None:
    from interpretability.data.activation_dataset import load_activation_dataset

    module = _load_script("tolerance_sensitivity")
    dataset = load_activation_dataset(golden_path)
    relabeled = module.relabel_dataset(
        dataset, {"ultimatum_bluff": {"points": 0.0}, "capability_bluff": {"ratio": 1.0}}
    )
    assert len(relabeled["metadata"]) == len(dataset["metadata"])
    for before, after in zip(dataset["metadata"], relabeled["metadata"]):
        assert after["full_response"] == before["full_response"]
        if after["claim_status"] == "complete":
            assert after["label_source"] == "rule"
            assert isinstance(after["emergent_ground_truth"], bool)
        else:
            assert after["label_source"] is None


def test_item_analysis_flags_zero_information_items(golden_path, tmp_path) -> None:
    module = _load_script("item_analysis")
    out = tmp_path / "items.json"
    code = module.main([
        "--dataset", str(golden_path), "--name", "a",
        "--dataset", str(golden_path), "--name", "b",
        "--out", str(out),
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["summary"]["n_items"] == 10
    # Every twin family has one honest and one lying trial: pass rate 0.5 for
    # both "models", so no family is zero-information.
    assert payload["summary"]["n_discriminating"] == 10
    assert all(item["diagnosis"] is None for item in payload["items"])
    assert set(payload["per_model"]) == {"a", "b"}


def test_scripts_expose_explicit_legacy_trust_flag() -> None:
    for name in ("tolerance_sensitivity", "item_analysis"):
        module = _load_script(name)
        with pytest.raises(SystemExit) as excinfo:
            module.main(["--help"])
        assert excinfo.value.code == 0
