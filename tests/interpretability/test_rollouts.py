"""Repeated rollouts: schedule expansion, plan accounting, and provenance."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from interpretability.cli import cli
from interpretability.core.dataset_builder import DatasetBuilder
from interpretability.data import ActivationSample
from interpretability.launch import resolve_execution_surface
from interpretability.runtime.runner import (
    build_counterbalance_schedule,
    expand_rollouts,
)


def test_expand_rollouts_nests_rollouts_inside_each_assignment() -> None:
    schedule = build_counterbalance_schedule(surface_variants=("default",))
    expanded = expand_rollouts(schedule, 3)
    assert len(expanded) == 3 * len(schedule)
    assert [index for _, index, _ in expanded[:3]] == [0, 1, 2]
    assert all(role is None for _, _, role in expanded)
    assert all(a is schedule[0] for a, _, _ in expanded[:3])
    assert [a for a, _, _ in expand_rollouts(schedule, 1)] == list(schedule)
    assert expand_rollouts((), 2) == ()
    twins = expand_rollouts(schedule[:1], 2, pair_roles=("above", "below"))
    assert [(index, role) for _, index, role in twins] == [
        (0, "above"), (0, "below"), (1, "above"), (1, "below"),
    ]
    with pytest.raises(ValueError):
        expand_rollouts(schedule, 1, pair_roles=("sideways",))
    with pytest.raises(ValueError):
        expand_rollouts(schedule, 1, pair_roles=("above", "above"))
    with pytest.raises(TypeError):
        expand_rollouts(schedule, 1, pair_roles=())
    with pytest.raises(ValueError):
        expand_rollouts(schedule, 0)
    with pytest.raises(ValueError):
        expand_rollouts(schedule, True)
    with pytest.raises(TypeError):
        expand_rollouts("abc", 2)


def test_execution_plan_counts_rollouts_and_rejects_them_for_legacy_runs() -> None:
    base = resolve_execution_surface(experiment_track="single_agent_white_box")
    repeated = resolve_execution_surface(
        experiment_track="single_agent_white_box", rollouts_per_family=5
    )
    assert base.rollouts_per_family == 1
    assert repeated.rollouts_per_family == 5
    assert repeated.executions_per_family == 5 * base.executions_per_family
    with pytest.raises(ValueError):
        resolve_execution_surface(experiment_track="single_agent_white_box", rollouts_per_family=0)
    with pytest.raises(ValueError):
        resolve_execution_surface(
            experiment_track="single_agent_white_box", mode="instructed", rollouts_per_family=2
        )
    paired = resolve_execution_surface(
        experiment_track="single_agent_white_box", paired_facts=True, rollouts_per_family=2
    )
    assert paired.paired_facts is True
    assert paired.executions_per_family == 2 * 2 * base.executions_per_family
    with pytest.raises(ValueError):
        resolve_execution_surface(
            experiment_track="single_agent_white_box", mode="instructed", paired_facts=True
        )
    with pytest.raises(TypeError):
        resolve_execution_surface(experiment_track="single_agent_white_box", paired_facts=1)


def test_cli_exposes_rollouts_option() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--rollouts-per-family" in result.output
    assert "--paired-facts" in result.output
    rejected = CliRunner().invoke(cli, ["run", "--rollouts-per-family", "0", "--trials", "1"])
    assert rejected.exit_code != 0


def test_sample_metadata_carries_rollout_index() -> None:
    import torch

    sample = ActivationSample(
        trial_id="t", round_num=0, agent_name="Negotiator",
        activations={"blocks.1.hook_resid_post": torch.zeros(4)},
        actual_deception=None, perceived_deception=None,
        rollout_index=3, pair_role="above", pair_anchor=60.0,
    )
    row = DatasetBuilder._sample_metadata(sample, "ultimatum_bluff")
    assert row["rollout_index"] == 3
    assert row["pair_role"] == "above" and row["pair_anchor"] == 60.0
    assert row["runtime_path"] is None



def test_planned_executions_doubles_only_twin_capable_scenarios() -> None:
    from interpretability.launch import planned_executions

    plan = resolve_execution_surface(
        experiment_track="single_agent_white_box", paired_facts=True
    )
    out = planned_executions(
        plan,
        scenarios=["ultimatum_bluff", "info_withholding"],
        conditions=2,
        families_per_scenario_condition=3,
    )
    per = out["per_scenario"]
    assert per["ultimatum_bluff"]["paired_facts_effective"] is True
    assert per["info_withholding"]["paired_facts_effective"] is False
    assert per["ultimatum_bluff"]["executions_per_family"] == plan.executions_per_family
    assert per["info_withholding"]["executions_per_family"] == plan.executions_per_family // 2
    assert out["total"] == sum(v["planned_executions"] for v in per.values())
    assert out["total"] == (2 + 1) * out["unpaired_executions_per_family"] * 2 * 3

    unpaired = resolve_execution_surface(experiment_track="single_agent_white_box")
    flat = planned_executions(
        unpaired, scenarios=["hidden_value"], conditions=["high_incentive"],
        families_per_scenario_condition=4,
    )
    assert flat["paired_executions_per_family"] is None
    assert flat["total"] == unpaired.executions_per_family * 4
