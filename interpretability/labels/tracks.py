"""Label track registry, the plug in point for how a scenario's turns get labeled.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
A label track is one rule that turns a committed negotiation turn into a
``LabelRecord``. The base ships four: the false claim rule for the numeric
scenarios, the disclosure rule for withholding, the commitment rule for
promises, and the "final offer" hold rule that runs beside the false claim
rule for ultimatum turns. The runtime executor used to pick them with
``if instance.scenario == ...`` switches, so a benchmark module could not add
a track without editing the base. This module makes the choice data:

1. ``LabelTrack`` names the rule, its version string, the behavior construct it
   measures, and the callable that produces the record. ``secondary`` tracks
   run beside the primary one and are never pooled with it.
2. ``LabelTrackRegistry`` binds tracks to scenario names. Every scenario needs
   exactly one primary track. A scenario with no track fails closed; the
   runtime never falls back to a label of "honest".
3. The runner builds the built in registry from its own evaluators (see
   ``runner.builtin_label_track_registry``) and a module may pass its own.

Pipeline position: scenarios -> agents -> capture -> **labels** -> QC.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Iterable

from interpretability.labels.schema import BehaviorTarget, LabelRecord

__all__ = ["LabelTrack", "LabelTrackError", "LabelTrackRegistry", "TrackEvaluator"]

# (executor, instance, resolution, events, *, evaluation_event_id, evidence) -> LabelRecord
TrackEvaluator = Callable[..., LabelRecord]
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class LabelTrackError(ValueError):
    """A track or binding is invalid, or a scenario has no track."""


@dataclass(frozen=True)
class LabelTrack:
    """One versioned labeling rule."""

    name: str
    version: str
    behavior_target: BehaviorTarget
    evaluate: TrackEvaluator
    secondary: bool = False
    # Commitment style rules: when the rule is not AVAILABLE the model judge
    # record mirrors its status, and the judge sees the rule's evidence ids.
    judge_mirrors_rule_status: bool = False
    judge_uses_rule_evidence: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_PATTERN.match(self.name):
            raise LabelTrackError(f"track name must match {_NAME_PATTERN.pattern}: {self.name!r}")
        if not isinstance(self.version, str) or not self.version.strip():
            raise LabelTrackError("track version must be non empty text")
        if not isinstance(self.behavior_target, BehaviorTarget):
            raise LabelTrackError("behavior_target must be a BehaviorTarget")
        if not callable(self.evaluate):
            raise LabelTrackError("evaluate must be callable")

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.version}"


class LabelTrackRegistry:
    """Explicit, thread safe binding of tracks to scenario names."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._primary: dict[str, LabelTrack] = {}
        self._secondary: dict[str, list[LabelTrack]] = {}

    def register(self, track: LabelTrack, *, scenarios: Iterable[str]) -> LabelTrack:
        if not isinstance(track, LabelTrack):
            raise TypeError("only LabelTrack instances can be registered")
        names = tuple(scenarios)
        if not names or any(not isinstance(s, str) or not s for s in names):
            raise LabelTrackError("a track must be bound to at least one scenario name")
        with self._lock:
            for scenario in names:
                if track.secondary:
                    existing = self._secondary.setdefault(scenario, [])
                    if any(t.name == track.name for t in existing):
                        raise LabelTrackError(
                            f"secondary track {track.name!r} is already bound to {scenario!r}"
                        )
                    existing.append(track)
                else:
                    if scenario in self._primary:
                        raise LabelTrackError(
                            f"scenario {scenario!r} already has primary track "
                            f"{self._primary[scenario].qualified_name}"
                        )
                    self._primary[scenario] = track
        return track

    def primary_for(self, scenario: str) -> LabelTrack:
        with self._lock:
            track = self._primary.get(scenario)
        if track is None:
            raise LabelTrackError(
                f"no primary label track is registered for scenario {scenario!r}; "
                "labels never default to honest, so the run cannot proceed"
            )
        return track

    def secondary_for(self, scenario: str) -> tuple[LabelTrack, ...]:
        with self._lock:
            return tuple(self._secondary.get(scenario, ()))

    def behavior_target_for(self, scenario: str) -> BehaviorTarget:
        return self.primary_for(scenario).behavior_target

    def scenarios(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(set(self._primary) | set(self._secondary)))

    def bindings(self) -> dict[str, dict[str, Any]]:
        """Provenance: which track versions label which scenario."""
        with self._lock:
            return {
                scenario: {
                    "primary": self._primary[scenario].qualified_name if scenario in self._primary else None,
                    "secondary": [t.qualified_name for t in self._secondary.get(scenario, ())],
                }
                for scenario in self.scenarios()
            }
