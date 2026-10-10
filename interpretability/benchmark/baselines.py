"""Reference detector baselines for the v1 benchmark views.

Every baseline maps rows to a score where higher means "more likely a false
claim". None of them loads a model; model-backed judges plug in through the
same ``scores`` interface in ``benchmark.scoring``.

- ``random_scores``: seeded noise (chance control).
- ``response_length_scores``: text length (nuisance control).
- ``stated_claim_threshold``: the largest amount the actor asserted. This is
  the "exploit" a transcript-only detector can learn when fact ranges do not
  straddle the claims; on a balanced matched-pair subset it must sit near
  AUROC 0.5. Reporting it on both the full and paired subsets makes the
  benchmark's central contrast auditable.
- ``fact_informed_rule_scores``: the labeling rule itself re-run with the
  actor's authorized facts (the reference ceiling; it is the grader, not a
  detector, and is reported as such).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def random_scores(n: int, *, seed: int = 0) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.random.default_rng(seed).random(n)


def response_length_scores(responses: Sequence[str]) -> np.ndarray:
    return np.asarray([float(len(text or "")) for text in responses], dtype=float)


def stated_claim_threshold(
    claim_values: Sequence[Sequence[float] | None],
    *,
    missing: float = float("-inf"),
) -> np.ndarray:
    """Score = largest asserted amount; rows without a claim get ``missing``."""
    scores = []
    for values in claim_values:
        numbers = [
            float(v) for v in (values or ())
            if v is not None and math.isfinite(float(v))
        ]
        scores.append(max(numbers) if numbers else missing)
    return np.asarray(scores, dtype=float)


def fact_informed_rule_scores(
    scenarios: Sequence[str],
    responses: Sequence[str],
    private_facts: Sequence[Mapping[str, Any]],
    *,
    tolerance_by_scenario: Mapping[str, Mapping[str, float]] | None = None,
    tolerances: Sequence[Mapping[str, float] | None] | None = None,
) -> np.ndarray:
    """Re-run the labeling rule with authorized facts: 1.0 TRUE, 0.0 FALSE, 0.5 UNKNOWN.

    ``tolerances`` is the per-row tolerance frozen into each instance's rule
    (``BenchmarkViews.tolerances``); it takes precedence over
    ``tolerance_by_scenario`` so the baseline relabels with the same tolerance
    that produced the label it is scored against.
    """
    from interpretability.scenarios.turn_labels import compute_turn_label

    if not (len(scenarios) == len(responses) == len(private_facts)):
        raise ValueError("inputs must be index-aligned")
    if tolerances is not None and len(tolerances) != len(scenarios):
        raise ValueError("tolerances must be index-aligned with scenarios")
    scores = []
    for index, (scenario, response, facts) in enumerate(zip(scenarios, responses, private_facts)):
        tolerance = tolerances[index] if tolerances is not None else None
        if tolerance is None:
            tolerance = (tolerance_by_scenario or {}).get(scenario)
        label = compute_turn_label(scenario, response, dict(facts), tolerance=tolerance)
        if label.is_deceptive is None:
            scores.append(0.5)
        else:
            scores.append(1.0 if label.is_deceptive else 0.0)
    return np.asarray(scores, dtype=float)


__all__ = [
    "fact_informed_rule_scores",
    "random_scores",
    "response_length_scores",
    "stated_claim_threshold",
]
