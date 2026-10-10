"""Act component with an injectable scope around final action generation.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
This file is the white-box activation-capture hook for negotiation agents.
1. ``ScopedConcatActComponent`` subclasses Concordia's ``ConcatActComponent`` (the
   component that concatenates every context component's ``pre_act`` text into one prompt
   and asks the model for the agent's action). It overrides exactly one method.
2. ``__init__`` accepts an optional ``action_scope_factory``: a zero-argument callable
   returning a fresh context manager. The caller (the interpretability runner) supplies a
   factory whose context manager installs forward-pass hooks on the model.
3. ``get_action_attempt`` wraps ONLY the final decision-producing model call in that
   context manager (or a no-op ``nullcontext`` when no factory is given), then delegates
   to the parent implementation unchanged.
Key property: auxiliary model calls made earlier by individual components during
``pre_act`` (e.g. the reflection questions in ``base_negotiator.py``) happen OUTSIDE this
scope. Activations captured inside the scope therefore come from the same single model
call that produced the stored response text -- a scientific invariant of this project.
Where this fits: this is the seam between the "negotiation" and "capture" stages of the
negotiation -> capture -> label pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from concordia.components.agent import concat_act_component
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


ActionScopeFactory = Callable[[], AbstractContextManager[Any]]


class ScopedConcatActComponent(concat_act_component.ConcatActComponent):
    """Scope only the final acting-model call, after component context reads."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        component_order: Sequence[str] | None = None,
        prefix_entity_name: bool = True,
        randomize_choices: bool = True,
        action_scope_factory: ActionScopeFactory | None = None,
    ) -> None:
        super().__init__(
            model=model,
            component_order=component_order,
            prefix_entity_name=prefix_entity_name,
            randomize_choices=randomize_choices,
        )
        self._action_scope_factory = action_scope_factory

    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Run final generation in the injected call scope, if configured."""
        # Step 1: build a fresh scope for THIS call only. With no factory configured,
        # nullcontext() makes the wrapper behave exactly like the stock component.
        scope = (
            self._action_scope_factory()
            if self._action_scope_factory is not None
            else nullcontext()
        )
        # Step 2: enter the scope (e.g. activation-capture hooks attach here), then let
        # the parent class assemble the prompt from `contexts` and query the model. The
        # component pre_act calls already happened before this method -- outside the scope.
        with scope:
            return super().get_action_attempt(contexts, action_spec)
