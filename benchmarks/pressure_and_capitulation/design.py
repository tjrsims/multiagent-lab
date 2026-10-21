"""Run presets for benchmark 2. One run covers one pressure policy; cross policies by running each.

Sampling is the provider default and the reply cap is 400 tokens, as in
benchmark 1. The main presets require a fixed counterpart model so every
evaluated model faces the same pusher.
"""
from __future__ import annotations

from typing import Callable

from benchmarks.pressure_and_capitulation.manifest import CONDITIONS, DEFAULT_POLICY, SCENARIOS
from end_bench.run import RunSpec, RunSpecError

MAX_TOKENS = 400


def _require_counterpart(counterpart_model: str | None, preset: str) -> str:
    if not counterpart_model:
        raise RunSpecError(
            f"the {preset!r} preset needs a fixed counterpart model (counterpart_model=...); "
            "self play is allowed only in the pilot preset"
        )
    return counterpart_model


def pilot_spec(model: str, *, counterpart_type: str = DEFAULT_POLICY, run_id: str = "",
               counterpart_model: str | None = None) -> RunSpec:
    """Elicitation pilot: small, every scenario, five rounds, twins on, high incentive."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=("high_incentive",),
        families=3, max_rounds=5, rollouts_per_family=1, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True, max_tokens=MAX_TOKENS,
        run_id=run_id, closing_attestation=True, counterpart_model=counterpart_model,
    )


def main_spec(model: str, *, counterpart_type: str = DEFAULT_POLICY, run_id: str = "",
              counterpart_model: str | None = None) -> RunSpec:
    """Main design: all conditions, 30 families, five rollouts, twins on, fixed counterpart."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=CONDITIONS,
        families=30, max_rounds=5, rollouts_per_family=5, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True, max_tokens=MAX_TOKENS,
        run_id=run_id, closing_attestation=True,
        counterpart_model=_require_counterpart(counterpart_model, "main"),
    )


def main_lite_spec(model: str, *, counterpart_type: str = DEFAULT_POLICY, run_id: str = "",
                   counterpart_model: str | None = None) -> RunSpec:
    """Main design with one rollout; a fifth of the cost."""
    return RunSpec(
        model=model, scenarios=SCENARIOS, conditions=CONDITIONS,
        families=30, max_rounds=5, rollouts_per_family=1, paired_facts=True,
        counterpart_types=(counterpart_type,), counterbalance=False, ultrafast=True, max_tokens=MAX_TOKENS,
        run_id=run_id, closing_attestation=True,
        counterpart_model=_require_counterpart(counterpart_model, "main_lite"),
    )


PRESETS: dict[str, Callable[..., RunSpec]] = {
    "pilot": pilot_spec,
    "main_lite": main_lite_spec,
    "main": main_spec,
}
