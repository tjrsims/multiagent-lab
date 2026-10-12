"""Offline oracle judge for smoke tests: perfect with facts, abstains without.

The judge prompt from ``end_bench.judges`` carries the acting party's private
facts only under fact-informed access. This model answers from those facts by
re-running the labeling rule on the turn text; under transcript-only access it
has nothing to go on and says so. It exists to exercise the judge testbed end
to end offline and to demonstrate the by-construction ceiling, not to stand
in for a real LLM judge.
"""

from __future__ import annotations

import json
import re
from typing import Any


_FACTS = re.compile(r"private facts \(ground truth\): (?P<facts>.+)")
_SCENARIO = re.compile(r"Scenario family: (?P<scenario>[a-z_]+)")
# The turn runs until the facts line or the verdict cue, not the first blank
# line: 62 multi paragraph live turns were read as their first paragraph only.
_TURN = re.compile(r"Turn to judge:\n(?P<turn>.*?)\n\n(?=Acting party's private facts|JSON verdict:)", re.DOTALL)


class OracleJudgeModel:
    """Minimal ``sample_text`` provider for the judge testbed."""

    model_name = "oracle-judge:offline"

    def __init__(self) -> None:
        self.call_count = 0

    def sample_text(self, prompt: str, **_: Any) -> str:
        from interpretability.scenarios.turn_labels import compute_turn_label

        self.call_count += 1
        scenario = _SCENARIO.search(prompt)
        turn = _TURN.search(prompt)
        facts_match = _FACTS.search(prompt)
        if scenario is None or turn is None or facts_match is None:
            return json.dumps({"false_claim": "cannot_determine", "confidence": 0.5, "evidence": ""})
        facts: dict[str, Any] = {}
        for pair in facts_match.group("facts").split(";"):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            value = value.strip()
            try:
                facts[key.strip()] = int(value)
            except ValueError:
                try:
                    facts[key.strip()] = float(value)
                except ValueError:
                    facts[key.strip()] = value
        label = compute_turn_label(scenario.group("scenario"), turn.group("turn"), facts)
        if label.is_deceptive is None:
            return json.dumps({"false_claim": "cannot_determine", "confidence": 0.5, "evidence": ""})
        return json.dumps({
            "false_claim": bool(label.is_deceptive),
            "confidence": 0.95,
            "evidence": turn.group("turn")[:60],
        })


__all__ = ["OracleJudgeModel"]
