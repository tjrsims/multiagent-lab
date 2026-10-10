"""Benchmark modules: a named bundle of registries the base runs without knowing it.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
The base packages (negotiation, interpretability, end_bench) never name a
benchmark. A benchmark lives in its own package under ``benchmarks/`` and
hands the base one ``BenchmarkModule``: which counterpart policies it uses
(``policy_registry``), which rule labels which scenario (``label_tracks``),
which scenarios and incentive conditions it runs, and whether it asks for the
closing attestation. END Bench passes those registries down to launch, the
study runner, and the executor. ``provenance()`` is what the run manifest and
the dataset record, so a result always says which module version produced it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from interpretability.labels.tracks import LabelTrackRegistry
from interpretability.scenarios.policy_registry import CounterpartPolicyRegistry

__all__ = ["BenchmarkModule", "BenchmarkModuleError"]

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class BenchmarkModuleError(ValueError):
    """A module declaration is incomplete or inconsistent."""


@dataclass(frozen=True)
class BenchmarkModule:
    name: str
    version: str
    policy_registry: CounterpartPolicyRegistry
    label_tracks: LabelTrackRegistry
    scenarios: tuple[str, ...]
    conditions: tuple[str, ...] = ()
    closing_attestation: bool = False
    attestation_timing: str = "closing"
    description: str = ""
    # SHA-256 of the module's preregistration file; recorded in every run manifest.
    prereg_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise BenchmarkModuleError(f"module name must match {_NAME.pattern}: {self.name!r}")
        if not isinstance(self.version, str) or not self.version.strip():
            raise BenchmarkModuleError("module version must be non empty text")
        if not isinstance(self.policy_registry, CounterpartPolicyRegistry):
            raise BenchmarkModuleError("policy_registry must be a CounterpartPolicyRegistry")
        if not isinstance(self.label_tracks, LabelTrackRegistry):
            raise BenchmarkModuleError("label_tracks must be a LabelTrackRegistry")
        if not self.scenarios or any(not isinstance(s, str) or not s for s in self.scenarios):
            raise BenchmarkModuleError("a module must name at least one scenario")
        for scenario in self.scenarios:
            # Fail closed at declaration time: every scenario needs a label track.
            self.label_tracks.primary_for(scenario)
        if type(self.closing_attestation) is not bool:
            raise BenchmarkModuleError("closing_attestation must be a boolean")
        from interpretability.runtime.attestation import ATTESTATION_TIMINGS
        if self.attestation_timing not in ATTESTATION_TIMINGS:
            raise BenchmarkModuleError(f"attestation_timing must be one of {ATTESTATION_TIMINGS}")

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "scenarios": list(self.scenarios),
            "conditions": list(self.conditions),
            "closing_attestation": self.closing_attestation,
            "attestation_timing": self.attestation_timing,
            "prereg_sha256": self.prereg_sha256,
            "counterpart_policy_ids": self.policy_registry.policy_ids(),
            "label_tracks": self.label_tracks.bindings(),
        }
