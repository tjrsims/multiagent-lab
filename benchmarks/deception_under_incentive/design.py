"""Run presets. One run covers one counterpart policy; cross policies by running each.

Sampling: every preset leaves ``temperature`` unset, so each model runs at its
provider's default sampling. Several frontier models accept nothing else, and a
cross model comparison must treat them all the same way.

The counterpart model is part of the design. Under self play the acting model
plays both sides, so each evaluated model faces a different opponent and the
leaderboard mixes a model's own propensity with its opponent's behavior. The
main presets therefore require a fixed counterpart model, the same one for
every evaluated model; the pilot allows self play for cheap elicitation checks.
"""
from __future__ import annotations

from typing import Callable

from benchmarks.deception_under_incentive.manifest import CONDITIONS, SCENARIOS
from end_bench.run import RunSpec, RunSpecError


def _require_counterpart(counterpart_model: str | None, preset: str) -> str:
    if not counterpart_model:
        raise RunSpecError(
            f"the {preset!r} preset needs a fixed counterpart model (counterpart_model=...); "
            "self play is allowed only in the pilot preset"
        )
    return counterpart_model


def pilot_spec(model: str, *, counterpart_type: str = "interrogative", run_id: str = "",
               counterpart_model: str | None = None) -> RunSpec:
    """Week one elicitation pilot: small, every scenario, five rounds, twins on."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=("high_incentive",),
        families=3, max_rounds=5, rollouts_per_family=1, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True,
        run_id=run_id, closing_attestation=True, counterpart_model=counterpart_model,
    )


def main_spec(model: str, *, counterpart_type: str = "interrogative", run_id: str = "",
              counterpart_model: str | None = None) -> RunSpec:
    """Main design: all conditions, 30 families, five rollouts, twins on, fixed counterpart."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=CONDITIONS,
        families=30, max_rounds=5, rollouts_per_family=5, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True,
        run_id=run_id, closing_attestation=True,
        counterpart_model=_require_counterpart(counterpart_model, "main"),
    )


def main_lite_spec(model: str, *, counterpart_type: str = "interrogative", run_id: str = "",
                   counterpart_model: str | None = None) -> RunSpec:
    """Main design with one rollout. Same families, conditions, twins and fixed counterpart; a fifth of the cost."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=CONDITIONS,
        families=30, max_rounds=5, rollouts_per_family=1, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True,
        run_id=run_id, closing_attestation=True,
        counterpart_model=_require_counterpart(counterpart_model, "main_lite"),
    )


PRESETS: dict[str, Callable[..., RunSpec]] = {
    "pilot": pilot_spec,
    "main_lite": main_lite_spec,
    "main": main_spec,
}
