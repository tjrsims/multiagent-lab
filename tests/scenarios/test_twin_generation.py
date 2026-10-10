"""Twin (matched-pair) generation: same anchor, opposite truth, one family."""

from __future__ import annotations

import pytest

from interpretability.scenarios.compiled import compile_emergent_scenario
from interpretability.scenarios.emergent_prompts import (
    EMERGENT_SCENARIOS,
    PAIR_ROLES,
    generate_paired_scenario_params,
    generate_scenario_params,
    get_counterpart_prompt,
    get_emergent_prompt,
    pair_anchor_for,
    IncentiveCondition,
)
from interpretability.scenarios.turn_labels import compute_turn_label
from interpretability.scenarios.schema import LabelValue


@pytest.mark.parametrize("scenario", ["ultimatum_bluff", "capability_bluff"])
@pytest.mark.parametrize("seed", range(1, 41))
def test_twins_make_the_anchor_claim_honest_and_false(scenario, seed) -> None:
    above = generate_paired_scenario_params(scenario, seed, "above")
    below = generate_paired_scenario_params(scenario, seed, "below")
    anchor = pair_anchor_for(scenario, seed)
    assert above["pair_anchor"] == below["pair_anchor"] == anchor
    if scenario == "ultimatum_bluff":
        claim = f"My minimum is {anchor} points."
        low, high = EMERGENT_SCENARIOS[scenario]["value_ranges"]["true_walkaway"]
        assert low <= above["true_walkaway"] <= high
        assert low <= below["true_walkaway"] <= high
        # A feasible agreement always exists for both twins.
        for twin in (above, below):
            assert twin["true_walkaway"] + twin["counterpart_walkaway"] < 100
            assert twin["counterpart_walkaway"] >= 1
    else:
        claim = f"Our team needs {anchor} days."
        low, high = EMERGENT_SCENARIOS[scenario]["value_ranges"]["true_days"]
        assert low <= above["true_days"] <= high
        assert low <= below["true_days"] <= high
    honest = compute_turn_label(scenario, claim, above)
    lie = compute_turn_label(scenario, claim, below)
    assert honest.value is LabelValue.FALSE, (anchor, above)
    assert lie.value is LabelValue.TRUE, (anchor, below)


def test_twin_generation_is_deterministic_and_rng_local() -> None:
    import random

    random.seed(1)
    state = random.getstate()
    one = generate_paired_scenario_params("ultimatum_bluff", 9, "above")
    two = generate_paired_scenario_params("ultimatum_bluff", 9, "above")
    assert one == two
    assert random.getstate() == state


def test_invalid_pair_requests_are_rejected() -> None:
    with pytest.raises(ValueError):
        generate_paired_scenario_params("ultimatum_bluff", 1, "sideways")
    with pytest.raises(ValueError):
        generate_paired_scenario_params("info_withholding", 1, "above")
    with pytest.raises(ValueError):
        pair_anchor_for("promise_break", 1)
    assert PAIR_ROLES == ("above", "below")
    from interpretability.scenarios.emergent_prompts import supports_twin_generation
    assert supports_twin_generation("hidden_value") is True
    assert supports_twin_generation("info_withholding") is False
    with pytest.raises(ValueError):
        supports_twin_generation("nope")


@pytest.mark.parametrize("seed", range(1, 21))
def test_hidden_value_twins_flip_the_exact_cost_claim(seed) -> None:
    above = generate_paired_scenario_params("hidden_value", seed, "above")
    below = generate_paired_scenario_params("hidden_value", seed, "below")
    anchor = pair_anchor_for("hidden_value", seed)
    claim = f"I paid ${anchor} for the item."
    assert above["true_value"] == anchor and below["true_value"] != anchor
    assert compute_turn_label("hidden_value", claim, above).value is LabelValue.FALSE
    assert compute_turn_label("hidden_value", claim, below).value is LabelValue.TRUE
    assert 30 <= below["true_value"] <= 80


def test_ordinary_generation_keeps_agreements_feasible() -> None:
    for seed in range(200):
        params = generate_scenario_params("ultimatum_bluff", seed)
        assert 20 <= params["true_walkaway"] <= 80
        assert params["true_walkaway"] + params["counterpart_walkaway"] < 100
        assert params["counterpart_walkaway"] >= 1
        days = generate_scenario_params("capability_bluff", seed)["true_days"]
        assert 5 <= days <= 25


@pytest.mark.parametrize("condition", list(IncentiveCondition))
def test_prompts_render_twins_consistently(condition) -> None:
    for role in PAIR_ROLES:
        params = generate_paired_scenario_params("ultimatum_bluff", 4, role)
        actor = get_emergent_prompt("ultimatum_bluff", condition, params)
        counterpart = get_counterpart_prompt("ultimatum_bluff", params)
        assert "{" not in actor and "{" not in counterpart
        assert f"{params['true_walkaway']} points" in actor
        assert f"{params['counterpart_walkaway']} points" in counterpart
        assert str(params["true_walkaway"]) not in counterpart.split("minimum acceptable amount is")[1][:4]


def test_compiled_twins_share_one_family_and_differ_only_in_the_fact() -> None:
    above = compile_emergent_scenario(
        "ultimatum_bluff", family_seed=5, trial_seed=1,
        condition="high_incentive", pair_role="above",
    )
    below = compile_emergent_scenario(
        "ultimatum_bluff", family_seed=5, trial_seed=2,
        condition="high_incentive", pair_role="below",
    )
    plain = compile_emergent_scenario(
        "ultimatum_bluff", family_seed=5, trial_seed=1, condition="high_incentive",
    )
    assert above.trial_family_id == below.trial_family_id
    assert above.trial_id != below.trial_id
    assert plain.trial_family_id != above.trial_family_id
    a_state = dict(dict(above.public_state)["counterbalance"])
    b_state = dict(dict(below.public_state)["counterbalance"])
    assert a_state["pair_role"] == "above" and b_state["pair_role"] == "below"
    assert a_state["pair_anchor"] == b_state["pair_anchor"] == pair_anchor_for("ultimatum_bluff", 5)
    a_facts = dict(above.rule_config)["semantic_params"]
    b_facts = dict(below.rule_config)["semantic_params"]
    assert "pair_role" not in a_facts and "pair_anchor" not in a_facts
    assert a_facts["true_walkaway"] != b_facts["true_walkaway"]
    assert dict(plain.public_state)["counterbalance"]["pair_role"] is None
