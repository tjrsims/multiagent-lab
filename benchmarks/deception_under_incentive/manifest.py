"""What this benchmark is made of, as one declaration."""
from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks.deception_under_incentive.policies import interrogative_policy
from interpretability.modules import BenchmarkModule
from interpretability.runtime.runner import builtin_label_track_registry
from interpretability.scenarios.policy_registry import builtin_policy_registry

MODULE_NAME = "deception_under_incentive"
MODULE_VERSION = "0.1.0"
PREREG_PATH = Path(__file__).with_name("PREREG.md")


def prereg_hash() -> str | None:
    """SHA-256 of the preregistration text beside this module, or None if absent."""
    if not PREREG_PATH.exists():
        return None
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()

SCENARIOS = ("hidden_value", "capability_bluff", "info_withholding", "ultimatum_bluff")
CONDITIONS = ("high_incentive", "low_incentive", "penalty")
POLICIES = ("default", "skeptical", "informed", "interrogative")


def build_module() -> BenchmarkModule:
    """A fresh module object; nothing is cached or shared with the base."""
    policies = builtin_policy_registry()
    policies.register(interrogative_policy())
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
            "Does an agent lie about a private fact when lying pays? Four scenarios, "
            "three incentive conditions, matched twins, a questioning counterpart, "
            "and a closing attestation before the final turn."
        ),
    )
