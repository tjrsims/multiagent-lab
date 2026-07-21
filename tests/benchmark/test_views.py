"""Tests for the v1 benchmark detector-view extractor."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability.benchmark.views import BenchmarkViews, build_views


def make_dataset(n_negotiation: int = 3, include_probe: bool = True):
    """A minimal saved-dataset dict matching save_dataset's shape."""
    rows = []
    for i in range(n_negotiation):
        rows.append({
            "sample_type": "negotiation",
            "scenario": "ultimatum_bluff" if i % 2 == 0 else "capability_bluff",
            "trial_id": i // 2,
            "trial_family_id": f"fam-{i // 2}",
            "full_response": f"my final offer is {40 + i}",
            "scenario_params": {"true_walkaway": 35 + i, "trial_id": i},
            "emergent_ground_truth": bool(i % 2),
        })
    if include_probe:
        rows.append({
            "sample_type": "pre_verification",
            "scenario": "ultimatum_bluff",
            "trial_id": 99,
            "trial_family_id": "fam-99",
            "full_response": "belief probe",
            "scenario_params": {"true_walkaway": 50},
            "emergent_ground_truth": False,
        })
    total = len(rows)
    return {
        "activations": {
            14: np.arange(total * 4, dtype=float).reshape(total, 4),
            "14_mean": np.ones((total, 4), dtype=float),
        },
        "labels": {"gm_labels": [float(r["emergent_ground_truth"]) for r in rows]},
        "metadata": rows,
    }


def test_build_views_aligns_and_filters_probes():
    views = build_views(make_dataset())
    assert isinstance(views, BenchmarkViews)
    assert views.n == 3  # the pre_verification probe is dropped
    assert views.labels.tolist() == [0, 1, 0]
    assert views.responses[0] == "my final offer is 40"
    assert views.scenarios == (
        "ultimatum_bluff", "capability_bluff", "ultimatum_bluff",
    )
    assert set(views.layers) == {14, "14_mean"}
    assert views.white_box(14).shape == (3, 4)


def test_group_ids_use_family_and_survive_split_isolation():
    views = build_views(make_dataset())
    # rows 0,1 share fam-0; row 2 is fam-1 (trial 1)
    assert views.group_ids.tolist() == ["fam-0"[:0] + "trial_family_id:fam-0",
                                        "trial_family_id:fam-0",
                                        "trial_family_id:fam-1"]


def test_fact_informed_text_is_deterministic_and_carries_facts():
    views = build_views(make_dataset())
    text = views.fact_informed_text(0)
    assert "my final offer is 40" in text
    assert "true_walkaway=35" in text
    # deterministic: sorted keys, stable across calls
    assert views.fact_informed_text(0) == text


def test_white_box_rows_track_the_kept_samples():
    views = build_views(make_dataset())
    # original activation row i == i*4 .. i*4+3; kept rows are 0,1,2 (probe=3 dropped)
    assert views.white_box(14)[2].tolist() == [8.0, 9.0, 10.0, 11.0]
    with pytest.raises(KeyError, match="not captured"):
        views.white_box(99)


def test_missing_scenario_params_fails_closed():
    dataset = make_dataset(include_probe=False)
    del dataset["metadata"][0]["scenario_params"]
    with pytest.raises(ValueError, match="missing 'scenario_params'"):
        build_views(dataset)


def test_label_falls_back_to_gm_when_no_emergent_truth():
    dataset = make_dataset(include_probe=False)
    for row in dataset["metadata"]:
        row["emergent_ground_truth"] = None
    dataset["labels"]["gm_labels"] = [0.9, 0.1, 0.7]
    views = build_views(dataset)
    assert views.labels.tolist() == [1, 0, 1]


def test_negotiation_only_can_be_disabled():
    views = build_views(make_dataset(), negotiation_only=False)
    assert views.n == 4  # probe retained


def test_scenario_mask_selects_rows():
    views = build_views(make_dataset())
    mask = views.scenario_mask("capability_bluff")
    assert mask.tolist() == [False, True, False]


def test_misaligned_activations_fail_closed():
    dataset = make_dataset(include_probe=False)
    dataset["activations"][14] = np.zeros((2, 4))  # wrong row count
    with pytest.raises(ValueError, match="rows"):
        build_views(dataset)
