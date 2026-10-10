"""Run presets. One run covers one counterpart policy; cross policies by running each."""
from __future__ import annotations

from typing import Callable

from benchmarks.deception_under_incentive.manifest import CONDITIONS, SCENARIOS
from end_bench.run import RunSpec


def pilot_spec(model: str, *, counterpart_type: str = "interrogative", run_id: str = "") -> RunSpec:
    """Week one elicitation pilot: small, every scenario, five rounds, twins on."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=("high_incentive",),
        families=3, max_rounds=5, rollouts_per_family=1, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True,
        run_id=run_id, closing_attestation=True,
    )


def main_spec(model: str, *, counterpart_type: str = "interrogative", run_id: str = "") -> RunSpec:
    """Main design: all conditions, 30 families, five rollouts, twins on."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=CONDITIONS,
        families=30, max_rounds=5, rollouts_per_family=5, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True,
        run_id=run_id, closing_attestation=True,
    )


def main_lite_spec(model: str, *, counterpart_type: str = "interrogative", run_id: str = "") -> RunSpec:
    """Main design with one rollout. Same families, conditions, and twins; a fifth of the cost and time."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=CONDITIONS,
        families=30, max_rounds=5, rollouts_per_family=1, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True,
        run_id=run_id, closing_attestation=True,
    )


PRESETS: dict[str, Callable[..., RunSpec]] = {
    "pilot": pilot_spec,
    "main_lite": main_lite_spec,
    "main": main_spec,
}
