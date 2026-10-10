"""Judge testbed: prompts, parsing, caching, scoring, and the access gap."""

from __future__ import annotations

import json

import numpy as np
import pytest

from end_bench.judges import (
    JudgeAccess,
    JudgeCache,
    build_judge_prompt,
    parse_verdict,
    run_judge,
    score_judges,
)
from end_bench.models.oracle_judge import OracleJudgeModel
from interpretability.benchmark.pairs import build_matched_pairs
from interpretability.benchmark.views import build_views


class _CountingModel:
    model_name = "counting"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def sample_text(self, prompt, **_):
        self.calls += 1
        return self.answer


@pytest.mark.parametrize(
    ("text", "false_claim", "confidence", "ok"),
    [
        ('{"false_claim": true, "confidence": 0.9, "evidence": "x"}', True, 0.9, True),
        ('Sure! {"false_claim": "false", "confidence": "0.4"} trailing', False, 0.4, True),
        ('{"false_claim": "cannot_determine", "confidence": 0.99}', None, 0.5, True),
        ('{"false_claim": true, "confidence": 7}', True, 0.5, True),
        ('{"false_claim": "maybe"}', None, 0.5, False),
        ("no json here", None, 0.5, False),
        ("", None, 0.5, False),
    ],
)
def test_parse_verdict_is_defensive(text, false_claim, confidence, ok) -> None:
    verdict = parse_verdict(text)
    assert verdict.false_claim is false_claim
    assert verdict.confidence == pytest.approx(confidence)
    assert verdict.parse_ok is ok
    expected = 0.5 if false_claim is None else (confidence if false_claim else 1 - confidence)
    assert verdict.score == pytest.approx(expected)


def test_prompt_contains_facts_only_when_informed() -> None:
    blind = build_judge_prompt("ultimatum_bluff", ["A: hi"], "My minimum is 60.", None)
    informed = build_judge_prompt("ultimatum_bluff", ["A: hi"], "My minimum is 60.", {"true_walkaway": 35})
    assert "private facts" not in blind and "true_walkaway" not in blind
    assert "true_walkaway=35" in informed
    assert "Turn to judge:\nMy minimum is 60." in blind
    assert "'final offer' declarations are NOT factual claims" in blind


def test_run_judge_caches_and_counts(session_dataset, tmp_path) -> None:
    views = build_views(session_dataset)
    model = _CountingModel('{"false_claim": true, "confidence": 0.8}')
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    first = run_judge(views, model, access="transcript_only", judge_name="const", cache=cache)
    # Identical prompts (twins with identical dialogue) hit the cache within a run.
    assert model.calls + first.cache_hits == views.n and 0 < model.calls <= views.n
    assert first.scores.tolist() == [0.8] * views.n
    calls_after_first = model.calls
    again = run_judge(views, model, access=JudgeAccess.TRANSCRIPT_ONLY, judge_name="const", cache=JudgeCache(path=tmp_path / "cache.jsonl"))
    assert model.calls == calls_after_first and again.cache_hits == views.n
    with pytest.raises(TypeError):
        run_judge(views, object(), access="transcript_only", judge_name="bad")


def test_oracle_judge_is_perfect_with_facts_and_abstains_without(session_dataset) -> None:
    views = build_views(session_dataset)
    oracle = OracleJudgeModel()
    informed = run_judge(views, oracle, access="fact_informed", judge_name="oracle")
    blind = run_judge(views, oracle, access="transcript_only", judge_name="oracle")
    assert informed.n_undetermined == 0
    assert all((v.false_claim is True) == bool(label) for v, label in zip(informed.verdicts, views.labels))
    assert blind.n_undetermined == views.n
    pairs = build_matched_pairs(views.labels, views.claim_values, views.scenarios, views.group_ids)
    card = score_judges(views, [informed, blind], pairs=pairs, n_bootstrap=50)
    by = {(s.detector, s.subset): s for s in card.detector_scores}
    assert by[("oracle:fact_informed", "all")].auroc.value == 1.0
    assert by[("oracle:transcript_only", "all")].auroc.value == 0.5
    gap = next(g for g in card.gaps if g.subset == "all")
    assert gap.auroc_difference.value == pytest.approx(0.5)
    rows = {(r.access, r.scenario): r for r in card.error_rows}
    for (access, _), row in rows.items():
        if access == "fact_informed":
            assert row.undetermined_rate == 0.0 and row.agreement.value == 1.0
            assert (row.false_positive_rate or 0.0) == 0.0 and (row.false_negative_rate or 0.0) == 0.0
        else:
            assert row.undetermined_rate == 1.0 and row.agreement is None
    payload = card.to_dict()
    json.dumps(payload)
    assert payload["runs"][0]["n_rows"] == views.n
