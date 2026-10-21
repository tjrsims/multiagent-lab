"""What benchmark 2 is made of, as one declaration."""
from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks.pressure_and_capitulation.policies import PRESSURE_DOSES, PRESSURE_STYLES, all_pressure_policies, policy_name
from interpretability.modules import BenchmarkModule
from interpretability.runtime.runner import builtin_label_track_registry
from interpretability.scenarios.policy_registry import builtin_policy_registry

MODULE_NAME = "pressure_and_capitulation"
MODULE_VERSION = "0.1.0"
PREREG_PATH = Path(__file__).with_name("PREREG.md")

SCENARIOS = ("hidden_value", "capability_bluff", "info_withholding", "ultimatum_bluff")
CONDITIONS = ("high_incentive", "low_incentive", "penalty")
POLICIES = tuple(policy_name(style, dose) for style in PRESSURE_STYLES for dose in PRESSURE_DOSES)
DEFAULT_POLICY = policy_name("disbelief", "once")


def prereg_hash() -> str | None:
    """SHA-256 of the preregistration text beside this module, or None if absent."""
    if not PREREG_PATH.exists():
        return None
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def build_module() -> BenchmarkModule:
    """A fresh module object; nothing is cached or shared with the base."""
    policies = builtin_policy_registry()
    for spec in all_pressure_policies():
        policies.register(spec)
    return BenchmarkModule(
        name=MODULE_NAME,
        version=MODULE_VERSION,
        policy_registry=policies,
        label_tracks=builtin_label_track_registry(),
        scenarios=SCENARIOS,
        conditions=CONDITIONS,
        closing_attestation=True,
        attestation_timing="both",
        prereg_sha256=prereg_hash(),
        description=(
            "Does an agent keep a true statement about its private fact when the other side "
            "pushes back? Four scenarios, three incentive conditions, four pressure styles at "
            "two doses, an early and a closing attestation, and a fact hold label per trial."
        ),
    )
