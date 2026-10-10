"""Contract tests for the pinned upstream Concordia runtime."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from concordia.associative_memory import basic_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity

from config import concordia_runtime


class RecordingModel(language_model.LanguageModel):
    """Small deterministic model that records the v2.4 sampling contract."""

    def __init__(self) -> None:
        self.last_text_call: dict[str, object] = {}

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        self.last_text_call = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "terminators": tuple(terminators),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "timeout": timeout,
            "seed": seed,
        }
        return "recorded"

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        del prompt, seed
        return 0, responses[0], {}


def test_pinned_concordia_distribution_is_installed() -> None:
    concordia_runtime.require_supported_version()
    assert concordia_runtime.installed_version() == "2.4.0"


def test_action_spec_uses_json_safe_round_trip() -> None:
    original = entity.ActionSpec(
        call_to_action="What does {name} do?",
        output_type=entity.OutputType.CHOICE,
        options=("accept", "decline"),
        tag="negotiation",
    )

    serialized = original.to_dict()
    restored = entity.action_spec_from_dict(serialized)

    assert serialized["output_type"] == "choice"
    assert serialized["options"] == ["accept", "decline"]
    assert restored == original


def test_interactive_document_forwards_sampling_controls() -> None:
    model = RecordingModel()
    document = interactive_document.InteractiveDocument(model)

    result = document.open_question(
        "Choose a move",
        temperature=0.2,
        top_p=0.7,
        top_k=11,
    )

    assert result == "recorded"
    assert model.last_text_call["temperature"] == 0.2
    assert model.last_text_call["top_p"] == 0.7
    assert model.last_text_call["top_k"] == 11


def test_game_master_memory_can_retain_repeated_events() -> None:
    memory = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=lambda _: np.ones(3),
        allow_duplicates=True,
    )

    memory.add("[putative_event] Alice: repeats offer")
    memory.add("[putative_event] Alice: repeats offer")

    assert memory.get_all_memories_as_text() == [
        "[putative_event] Alice: repeats offer",
        "[putative_event] Alice: repeats offer",
    ]



def test_repository_never_imports_symbols_removed_upstream() -> None:
    """Upstream main dropped these; our code must not depend on them."""
    import re
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    offenders = []
    for package in ("negotiation", "interpretability", "end_bench", "config"):
        for path in (root / package).rglob("*.py"):
            if path.name == "concordia_runtime.py":
                continue  # it lists the removed symbols by design
            text = path.read_text(encoding="utf-8")
            for symbol in concordia_runtime.REMOVED_UPSTREAM_SYMBOLS:
                if re.search(rf"\b{re.escape(symbol)}\b", text):
                    offenders.append((str(path.relative_to(root)), symbol))
    assert offenders == []


def test_act_component_constructor_contract_is_the_stable_subset() -> None:
    """We pass only kwargs that exist in both 2.4.0 and upstream main."""
    import inspect

    from concordia.components.agent import concat_act_component

    from negotiation.components import scoped_act

    upstream = inspect.signature(concat_act_component.ConcatActComponent.__init__)
    assert {"model", "component_order", "prefix_entity_name", "randomize_choices"} <= set(
        upstream.parameters
    )
    ours = inspect.signature(scoped_act.ScopedConcatActComponent.__init__)
    assert {"action_scope_factory", "prefix_entity_name"} <= set(ours.parameters)


def test_audit_metadata_is_well_formed() -> None:
    assert len(concordia_runtime.UPSTREAM_TAG_COMMIT) == 40
    assert len(concordia_runtime.AUDITED_MAIN_COMMIT) == 40
    assert concordia_runtime.AUDITED_MAIN_DATE <= concordia_runtime.AUDIT_PERFORMED
