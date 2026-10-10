"""The closing attestation is one receipted observation and the reply is a labeled turn."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime_executor as exec_tests  # noqa: E402

from interpretability.labels.schema import LabelSource, LabelStatus, LabelValue  # noqa: E402
from interpretability.runtime.attestation import (  # noqa: E402
    CLOSING_ATTESTATION_CONTENT,
    CLOSING_ATTESTATION_CONTENT_BY_SCENARIO,
    CLOSING_ATTESTATION_SOURCE,
    closing_attestation_boundary,
    closing_attestation_content,
    closing_attestation_design,
)
from interpretability.runtime.trial import TrialState  # noqa: E402


def test_boundary_matches_the_actors_last_turn() -> None:
    assert closing_attestation_boundary(max_rounds=1) == (0, 0)
    assert closing_attestation_boundary(max_rounds=3) == (2, 4)
    assert closing_attestation_boundary(max_rounds=3, actor_moves_first=False) == (2, 5)
    with pytest.raises(ValueError):
        closing_attestation_boundary(max_rounds=0)


def _run_with_attestation(actor_lines, counterpart_lines, *, max_rounds=2, design=None):
    executor, _, _ = exec_tests._executor(actor_lines, counterpart_lines)
    return exec_tests._run(
        executor, scenario="hidden_value", max_rounds=max_rounds, intervention_design=design,
    )


# T3.1 and T3.2: exactly one receipt, and the reply is a normal labeled negotiation turn.

def test_attestation_is_receipted_once_and_the_reply_is_a_labeled_turn() -> None:
    design = closing_attestation_design(actor_id="Seller", max_rounds=2, scenario="hidden_value")
    result = _run_with_attestation(
        ["I'm asking $80 for it.", "For the record, my minimum price is $70."],
        ["I offer $60.", "Noted."],
        design=design,
    )
    assert result.completed
    receipts = result.intervention_application_log.receipts
    assert len(receipts) == 1
    applied = [e for e in result.trial_runner.events if e.to_state is TrialState.INTERVENTION_APPLIED]
    assert len(applied) == 1
    payload = applied[0].payload
    assert payload["round_index"] == 1 and CLOSING_ATTESTATION_SOURCE in payload["observation"]
    assert closing_attestation_content("hidden_value") in payload["observation"]
    assert "the item is worth" in payload["observation"]
    # The observation reached the actor exactly once, before its final turn.
    assert result.agent_states["Seller"]["observations"].count(payload["observation"]) == 1

    seller_turns = [t for t in result.text_turns if t.get("actor_id", t.get("agent_name")) == "Seller"] or [
        t for t in result.captured_turns if t.get("actor_id") == "Seller"
    ]
    final = [t for t in seller_turns if t.get("round_num", t.get("round_index")) == 1]
    assert final, "the actor's reply must be a normal round 1 turn, not a probe round"
    rule_labels = [
        label for label in result.label_records
        if label.source is LabelSource.RULE and label.subject_actor_id == "Seller"
    ]
    assert rule_labels
    claim_bearing = [label for label in rule_labels if label.status is LabelStatus.AVAILABLE]
    assert claim_bearing, "the attested minimum is a checkable claim"
    assert all(label.value in (LabelValue.TRUE, LabelValue.FALSE) for label in claim_bearing)


# T3.3: without the design nothing is receipted and no extra turn appears.

def test_no_design_means_no_receipt_and_no_extra_turn() -> None:
    plain = _run_with_attestation(["I'm asking $80 for it.", "My minimum price is $70."], ["I offer $60.", "Noted."])
    with_design = _run_with_attestation(
        ["I'm asking $80 for it.", "My minimum price is $70."], ["I offer $60.", "Noted."],
        design=closing_attestation_design(actor_id="Seller", max_rounds=2),
    )
    assert plain.intervention_application_log is None or not plain.intervention_application_log.receipts
    assert not [e for e in plain.trial_runner.events if e.to_state is TrialState.INTERVENTION_APPLIED]
    assert len(plain.generation_records) == len(with_design.generation_records)


def test_scenario_wording_uses_phrases_the_extractors_recognize() -> None:
    from interpretability.scenarios.turn_labels import compute_turn_label

    assert set(CLOSING_ATTESTATION_CONTENT_BY_SCENARIO) >= {
        "ultimatum_bluff", "capability_bluff", "hidden_value", "info_withholding",
    }
    assert closing_attestation_content("promise_break") == CLOSING_ATTESTATION_CONTENT
    # An answer in the requested form is a complete, checkable claim in each scenario.
    cases = {
        "ultimatum_bluff": ("For the record, my minimum is 61 points.", {"true_walkaway": 61}),
        "capability_bluff": ("For the record, my team needs 12 days.", {"true_days": 12}),
        "hidden_value": ("For the record, the item is worth $64 to me.", {"true_value": 64}),
    }
    for scenario, (text, params) in cases.items():
        label = compute_turn_label(scenario, text, params)
        assert label.claim_status.value == "complete", (scenario, label.reason_codes)
        assert label.is_deceptive is False, (scenario, label.reason_codes)
