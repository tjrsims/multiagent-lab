"""Build a small, real, offline activation dataset for benchmark tests.

The bundle is produced by the transactional executor with the fake scoped
model from ``tests.interpretability.test_runtime_executor`` and saved through
``DatasetBuilder.save`` as JSON+NPZ, so every downstream benchmark test runs
against the same contract the real pipeline writes.
"""

from __future__ import annotations

from pathlib import Path

from interpretability.core.dataset_builder import DatasetBuilder
from interpretability.scenarios.compiled import compute_instance_turn_label
from interpretability.scenarios.emergent_prompts import (
    generate_paired_scenario_params,
    pair_anchor_for,
)
from tests.interpretability.test_runtime_executor import _executor, _run


ROLES = {"actor": "Negotiator", "counterpart": "Counterpart"}
COUNTERPART_LINES = [
    "I offer you 45 points and keep the rest for myself this round.",
    "I could stretch to 50 points for you if we close today.",
    "Understood, let us keep talking about the remaining terms here.",
]
CLIENT_LINES = [
    "How long will this project take your team to complete in full?",
    "That is longer than we hoped for, but we understand the constraints.",
]


def _rule_evaluator(instance, resolution, _events):
    return compute_instance_turn_label(
        instance,
        resolution.event.action.raw_text,
        actor_role_id=resolution.event.action.actor_id,
        round_index=resolution.event.round_index,
    ).evaluator_result()


def _ultimatum_lines(anchor: int) -> list[str]:
    return [
        f"My minimum is {anchor} points, and I would like to settle today.",
        f"I offer {anchor + 5} points to me and the rest to you, as a fair split.",
    ]


def _capability_lines(anchor: int) -> list[str]:
    return [
        f"Our team needs {anchor} days to deliver this scope with good quality.",
        "We can discuss milestones once the overall timeline is agreed upon.",
    ]


def build_golden_dataset(
    path: Path,
    *,
    ultimatum_families: int = 6,
    capability_families: int = 4,
    rollouts: int = 1,
    model_name: str = "fake-golden-model",
) -> Path:
    """Run twin trials through the executor and save a JSON+NPZ dataset."""
    builder = DatasetBuilder(
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        split_seed=7,
    )
    trial_seed = 100
    plan = [("ultimatum_bluff", seed) for seed in range(1, ultimatum_families + 1)]
    plan += [("capability_bluff", seed) for seed in range(1, capability_families + 1)]
    for scenario, family_seed in plan:
        anchor = pair_anchor_for(scenario, family_seed)
        for pair_role in ("above", "below"):
            # Sanity: the generator guarantees the anchor claim flips truth.
            generate_paired_scenario_params(scenario, family_seed, pair_role)
            for _ in range(rollouts):
                lines = (
                    _ultimatum_lines(anchor)
                    if scenario == "ultimatum_bluff" else _capability_lines(anchor)
                )
                counterpart = (
                    COUNTERPART_LINES if scenario == "ultimatum_bluff" else CLIENT_LINES
                )
                executor, _, _ = _executor(lines, counterpart, rule_evaluator=_rule_evaluator)
                result = _run(
                    executor,
                    scenario=scenario,
                    condition="high_incentive",
                    family_seed=family_seed,
                    trial_seed=trial_seed,
                    max_rounds=2,
                    role_assignment=ROLES,
                    pair_role=pair_role,
                )
                trial_seed += 1
                assert result.completed and not result.failed
                builder.generation_records.extend(result.generation_records)
                builder.label_records.extend(result.label_records)
                builder.interaction_events.extend(
                    result.adjudicator_state.get("events", ())
                )
                for sample in result.activation_samples:
                    builder.add_sample(sample)
    target = Path(path)
    builder.save(target, model_name=model_name)
    return target.with_suffix(".json")
