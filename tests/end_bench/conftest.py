"""Shared offline fixtures: one scripted text-track run per test session."""

from __future__ import annotations

import pytest

from end_bench.run import RunSpec, run_text_track


@pytest.fixture(scope="session")
def session_smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("session_smoke")
    spec = RunSpec(
        model="scripted",
        scenarios=("hidden_value", "ultimatum_bluff", "capability_bluff"),
        conditions=("high_incentive",),
        families=3,
        max_rounds=2,
        rollouts_per_family=1,
        paired_facts=True,
        run_id="session-smoke",
    )
    return run_text_track(spec, out)


@pytest.fixture(scope="session")
def session_dataset(session_smoke_run):
    from end_bench.data.text_dataset import load_text_dataset

    return load_text_dataset(session_smoke_run.text_dataset_path)
