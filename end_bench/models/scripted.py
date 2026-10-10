"""Scripted offline acting model for smoke tests and containers without AWS.

Returns canned negotiation lines in order (per role when the prompt names the
role), and publishes the same ``capture_mode='none'`` records a real API model
would, so the whole pipeline — agents, adjudication, labels, QC, text dataset,
benchmark build — can be exercised end to end with zero network access.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from concordia.language_model import language_model


GENERATION_PATH = "scripted_offline"

DEFAULT_LINES: Mapping[str, Sequence[str]] = {
    "ultimatum_bluff": (
        "My minimum is 62 points, and I would like to settle today if we can.",
        "I offer 60 points to me and 40 to you, which leaves both of us ahead.",
        "I accept 58 points for me and we can close the agreement here.",
    ),
    "capability_bluff": (
        "Our team needs 20 days to deliver this scope with good quality.",
        "We can discuss milestones once the overall timeline is agreed upon.",
        "We would need 20 days; that is the realistic schedule for this work.",
    ),
    "hidden_value": (
        "I paid $40 for the item, so my asking price is $80 for a quick sale.",
        "I can come down to $70 if you can take it this week.",
        "I accept your offer; let us finalize the paperwork today.",
    ),
    "info_withholding": (
        "The car runs well and has been maintained regularly; I am asking $9,000.",
        "I can come down to $8,500 if we can close this week.",
        "I accept your offer; let us arrange the handover this weekend.",
    ),
    "counterpart": (
        "I offer you 45 points and keep the rest for myself this round.",
        "I could stretch to 50 points for you if we close today.",
        "Understood, let us keep talking about the remaining terms here.",
    ),
}


class ScriptedTextModel(language_model.LanguageModel):
    """Deterministic canned responses with faithful record publication."""

    def __init__(
        self,
        lines: Mapping[str, Sequence[str]] | None = None,
        *,
        default_line: str = "Let us keep talking about the remaining terms.",
        max_tokens: int = 128,
        cost_ledger: "CostLedger | None" = None,
    ) -> None:
        from end_bench.models.pricing import CostLedger

        self.cost_ledger = cost_ledger if cost_ledger is not None else CostLedger()
        self.lines: dict[str, tuple[str, ...]] = {
            key: tuple(value) for key, value in (lines or DEFAULT_LINES).items()
        }
        self.default_line = default_line
        self.default_max_tokens = max_tokens
        self._cursor: dict[str, int] = defaultdict(int)
        self._call_count = 0
        self._last_sampling_config: dict[str, Any] = {}
        self.tokenizer = None
        self.layers_to_capture: list[int] = []

    @property
    def model_name(self) -> str:
        return "scripted:offline"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def activation_dim(self) -> int:
        return 0

    def get_last_sampling_config(self) -> dict[str, Any]:
        return dict(self._last_sampling_config)

    def capture_followup_activations(self, original_prompt: str, response: str) -> dict:
        del original_prompt, response
        return {}

    def _select_key(self, prompt: str) -> str:
        lowered = prompt.lower()
        # Counterpart prompts are recognizable by their role text; actor prompts
        # by scenario vocabulary. Order matters: check counterpart cues first.
        if "counterpart" in self.lines and (
            "other party's statements" in lowered
            or "looking for a contractor" in lowered
            or "buy an item from a seller" in lowered
            or "buy a used car" in lowered
        ):
            return "counterpart"
        if "split of 100 points" in lowered or "split 100 points" in lowered:
            return "ultimatum_bluff"
        if "contractor" in lowered or "project" in lowered:
            return "capability_bluff"
        if "used car" in lowered or "car has an issue" in lowered:
            return "info_withholding"
        if "selling an item" in lowered or "sell an item" in lowered:
            return "hidden_value"
        return "default"

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        terminators: Sequence[str] = (),
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> str:
        del timeout
        self._call_count += 1
        key = self._select_key(prompt)
        script = self.lines.get(key, ())
        index = self._cursor[key]
        # Cycle through the script so every trial in a multi-trial run gets
        # claim-bearing lines, not just the first one.
        text = script[index % len(script)] if script else self.default_line
        self._cursor[key] = index + 1
        for terminator in terminators or ():
            if terminator and terminator in text:
                text = text.split(terminator, 1)[0]
                break
        from interpretability.evaluation import (
            _publish_scoped_generation,
            _sampling_provenance,
        )
        from interpretability.runtime.model_call import CaptureMode

        effective_max = int(max_tokens) if max_tokens is not None else self.default_max_tokens
        provenance = _sampling_provenance(
            requested_max_tokens=max_tokens,
            effective_max_tokens=effective_max,
            temperature=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            top_k=int(top_k) if top_k is not None else 0,
            seed=seed,
            effective_temperature=0.0,
            effective_top_p=1.0,
            effective_top_k=None,
            effective_do_sample=False,
            generation_path=GENERATION_PATH,
            fallback_used=False,
        )
        provenance["provider"] = "scripted"
        provenance["script_key"] = key
        self._last_sampling_config = provenance
        self.cost_ledger.record(
            model_id="scripted:offline", input_tokens=0, output_tokens=0,
            latency_ms=0.0, retries=0, call_id=None, stop_reason="scripted",
        )
        _publish_scoped_generation(
            assembled_prompt=prompt,
            input_token_ids=(),
            output_token_ids=(),
            retained_token_ids=(),
            output_text=text,
            terminator=None,
            sampling_provenance=provenance,
            capture_mode=CaptureMode.NONE,
            activations={},
        )
        return text

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[int, str, dict[str, Any]]:
        if not responses:
            raise ValueError("responses must be non-empty")
        sample = self.sample_text(prompt, max_tokens=100, seed=seed, **kwargs)
        words = set(sample.lower().split())
        best = max(
            range(len(responses)),
            key=lambda index: len(words & set(responses[index].lower().split())),
        )
        return best, responses[best], {"sample": sample}


__all__ = ["DEFAULT_LINES", "GENERATION_PATH", "ScriptedTextModel"]
