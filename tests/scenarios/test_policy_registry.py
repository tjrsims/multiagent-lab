"""Counterpart policy registry: built ins unchanged, module policies pluggable."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretability.scenarios.compiled import (
    CounterpartPolicy,
    compile_emergent_scenario,
    policy_spec_for,
    render_actor_prompt,
    render_counterpart_prompt,
    validate_counterpart_policy,
    validate_counterpart_policy_contract,
)
from interpretability.scenarios.policy_registry import (
    BUILTIN_POLICY_NAMES,
    CounterpartPolicyError,
    CounterpartPolicyRegistry,
    CounterpartPolicySpec,
    builtin_policy_registry,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "counterpart_prompts_v1.json").read_text()
)
ROLES = {"actor": "Negotiator", "counterpart": "Counterpart"}


def _compile(scenario, policy, registry=None, **kwargs):
    return compile_emergent_scenario(
        scenario, family_seed=11, trial_seed=5, condition="high_incentive",
        role_assignment=ROLES, counterpart_type=policy, policy_registry=registry, **kwargs,
    )


# --- T1.1 the built ins render exactly the prompts they rendered before ---------------

@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_builtin_policies_render_golden_prompts(key: str) -> None:
    scenario, policy = key.split("|")
    instance = _compile(scenario, policy)
    assert render_counterpart_prompt(instance, "Counterpart") == GOLDEN[key]["counterpart_prompt"]
    assert render_actor_prompt(instance, "Negotiator") == GOLDEN[key]["actor_prompt"]


def test_builtin_registry_names_match_the_enum() -> None:
    registry = builtin_policy_registry()
    assert registry.names() == BUILTIN_POLICY_NAMES
    assert tuple(p.value for p in CounterpartPolicy) == BUILTIN_POLICY_NAMES
    informed = registry.get("informed")
    assert informed.grant_keys_for("hidden_value") == ("true_value",)
    assert informed.grant_keys_for("ultimatum_bluff") == ("true_walkaway",)
    assert registry.get("skeptical").grant_keys_for("hidden_value") == ()


# --- spec validation ------------------------------------------------------------------

def test_spec_rejects_bad_names_and_ungranted_placeholders() -> None:
    with pytest.raises(CounterpartPolicyError, match="name must match"):
        CounterpartPolicySpec(name="Bad Name", version="1")
    with pytest.raises(CounterpartPolicyError, match="protocol"):
        CounterpartPolicySpec(name="absent", version="1")
    with pytest.raises(CounterpartPolicyError, match="placeholders"):
        CounterpartPolicySpec(name="leaky", version="1", prompt_modifier="Their minimum is {true_walkaway}.")
    with pytest.raises(CounterpartPolicyError, match="not .*granted"):
        CounterpartPolicySpec(
            name="leaky", version="1",
            prompt_modifiers_by_scenario={"hidden_value": "Value is {true_value}."},
        )
    ok = CounterpartPolicySpec(
        name="peek", version="1",
        prompt_modifiers_by_scenario={"hidden_value": "Value is {true_value}."},
        grant_keys_by_scenario={"hidden_value": ("true_value",)},
    )
    assert ok.modifier_for("hidden_value", {"true_value": 42}) == "Value is 42."
    with pytest.raises(CounterpartPolicyError, match="needs granted facts"):
        ok.modifier_for("hidden_value", {})


def test_policy_id_is_a_content_hash() -> None:
    a = CounterpartPolicySpec(name="probe", version="1", prompt_modifier="Ask once.")
    b = CounterpartPolicySpec(name="probe", version="1", prompt_modifier="Ask once.")
    c = CounterpartPolicySpec(name="probe", version="1", prompt_modifier="Ask twice.")
    assert a.policy_id == b.policy_id != c.policy_id
    assert a.qualified_name == "probe/1"
    assert a.to_dict()["policy_id"] == a.policy_id


# --- registry behavior ----------------------------------------------------------------

def test_registry_rejects_duplicates_absent_and_unknown_names() -> None:
    registry = CounterpartPolicyRegistry()
    spec = registry.register(CounterpartPolicySpec(name="probe", version="1"))
    with pytest.raises(CounterpartPolicyError, match="already registered"):
        registry.register(spec)
    with pytest.raises(CounterpartPolicyError, match="solo protocol"):
        registry.get("absent")
    with pytest.raises(CounterpartPolicyError, match="Unsupported counterpart policy: 'nope'"):
        registry.get("nope")
    assert "probe" in registry and "nope" not in registry
    assert registry.policy_ids() == {"probe": spec.policy_id}


# --- T1.2 / T1.3 a module policy flows through compile and render ------------------------

def _module_registry() -> CounterpartPolicyRegistry:
    registry = builtin_policy_registry()
    registry.register(CounterpartPolicySpec(
        name="interrogative", version="1",
        prompt_modifier="\n\nCOUNTERPART POLICY: Ask one direct question about their bottom line every turn.",
        description="Elicits a checkable claim each turn.",
    ))
    return registry


def test_module_policy_is_rejected_without_its_registry() -> None:
    with pytest.raises(ValueError, match="Unsupported counterpart policy: 'interrogative'"):
        validate_counterpart_policy("interrogative")
    with pytest.raises(ValueError, match="interrogative"):
        _compile("hidden_value", "interrogative")


def test_module_policy_reaches_the_counterpart_prompt_only() -> None:
    registry = _module_registry()
    assert validate_counterpart_policy("interrogative", registry) == "interrogative"
    assert validate_counterpart_policy("skeptical", registry) is CounterpartPolicy.SKEPTICAL
    custom = _compile("hidden_value", "interrogative", registry)
    default = _compile("hidden_value", "default")
    prompt = render_counterpart_prompt(custom, "Counterpart", registry)
    assert prompt.endswith("Ask one direct question about their bottom line every turn.")
    assert render_actor_prompt(custom, "Negotiator") == render_actor_prompt(default, "Negotiator")
    assert validate_counterpart_policy_contract(custom, "Counterpart", registry) == "interrogative"
    assert custom.trial_family_id == default.trial_family_id  # same family, different policy
    assert policy_spec_for("interrogative", registry).policy_id == registry.get("interrogative").policy_id


def test_module_policy_with_a_grant_gets_an_authorized_knowledge_grant() -> None:
    registry = builtin_policy_registry()
    registry.register(CounterpartPolicySpec(
        name="appraiser", version="1",
        prompt_modifiers_by_scenario={"hidden_value": "\n\nAn appraisal puts their value at ${true_value}."},
        grant_keys_by_scenario={"hidden_value": ("true_value",)},
    ))
    instance = _compile("hidden_value", "appraiser", registry)
    private = instance.view_for("Counterpart").private_state
    grant = dict(private["policy_knowledge_grant"])
    assert grant["policy"] == "appraiser" and set(grant["parameters"]) == {"true_value"}
    prompt = render_counterpart_prompt(instance, "Counterpart", registry)
    assert f"An appraisal puts their value at ${grant['parameters']['true_value']}." in prompt
    # The same spec without the grant can never render the fact.
    with pytest.raises(CounterpartPolicyError):
        CounterpartPolicySpec(
            name="appraiser", version="2",
            prompt_modifiers_by_scenario={"hidden_value": "value {true_value}"},
        )



# --- T1.3 / T1.4 the registry flows through the executor and its checkpoints ------------

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interpretability"))
import test_runtime_executor as _exec_tests  # noqa: E402  (scripted executor helpers)

from interpretability.launch import resolve_execution_surface  # noqa: E402
from interpretability.runtime.runner import (  # noqa: E402
    CounterbalanceAssignment,
    EmergentTrialExecutor,
    build_counterbalance_schedule,
)


def _executor_with_registry(registry, actor_lines, counterpart_lines, prompts=None):
    """Scripted executor whose counterpart builder records the prompt it was given."""
    executor, actor_model, counterpart_model = _exec_tests._executor(actor_lines, counterpart_lines)
    if prompts is not None:
        original = executor.counterpart_builder

        def recording_builder(role_id, prompt, scope_factory):
            prompts.append(prompt)
            return original(role_id, prompt, scope_factory)

        executor.counterpart_builder = recording_builder
    executor.policy_registry = registry
    return executor, actor_model, counterpart_model


def test_executor_renders_a_module_policy_and_stamps_its_hash() -> None:
    registry = _module_registry()
    prompts: list[str] = []
    executor, _, _ = _executor_with_registry(
        registry, ["I'm asking $80 for it."], ["What is the lowest you would take?"], prompts
    )
    result = _exec_tests._run(executor, scenario="hidden_value", counterpart_type="interrogative", max_rounds=1)
    expected_id = registry.get("interrogative").policy_id
    assert result.assignment.counterpart_type == "interrogative"
    assert result.assignment.counterpart_policy_id == expected_id
    assert prompts and all("Ask one direct question about their bottom line" in p for p in prompts)
    assert result.completed


def test_executor_without_the_registry_rejects_the_module_policy_before_any_call() -> None:
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("builder must not be called")

    executor = EmergentTrialExecutor(
        run_id="no-registry", actor_builder=forbidden, counterpart_builder=forbidden,
        model_revision="fake-model@1", tokenizer_revision="fake-tokenizer@1",
    )
    with pytest.raises(ValueError, match="Unsupported counterpart policy: 'interrogative'"):
        _exec_tests._run(executor, counterpart_type="interrogative")
    assert calls == []


def test_checkpoint_round_trips_the_policy_hash_and_detects_changed_policy_text() -> None:
    registry = _module_registry()
    executor, _, _ = _executor_with_registry(registry, ["I offer $70."], ["I accept your offer."])
    interrupted = _exec_tests._run(executor, counterpart_type="interrogative", stop_after_adjudications=1)
    checkpoint = json.loads(json.dumps(interrupted.to_checkpoint()))
    assert checkpoint["assignment"]["counterpart_policy_id"] == registry.get("interrogative").policy_id
    restored = CounterbalanceAssignment.from_dict(checkpoint["assignment"])
    assert restored == interrupted.assignment

    # A registry whose policy text changed under the same name must refuse to resume.
    changed = builtin_policy_registry()
    changed.register(CounterpartPolicySpec(name="interrogative", version="2", prompt_modifier="Different text."))
    resumed, _, _ = _executor_with_registry(changed, ["I offer $70."], ["I accept your offer."])
    with pytest.raises(ValueError, match="policy text changed"):
        _exec_tests._run(resumed, counterpart_type="interrogative", resume_from=checkpoint,
                         activation_lookup=interrupted.activation_snapshots)


def test_schedule_and_launch_carry_policy_ids_and_reject_unregistered_names() -> None:
    registry = _module_registry()
    schedule = build_counterbalance_schedule(
        participant_ids=("Alice", "Bob"), counterpart_types=("default", "interrogative"),
        surface_variants=("default",), policy_registry=registry,
    )
    ids = {a.counterpart_type if isinstance(a.counterpart_type, str) else a.counterpart_type.value: a.counterpart_policy_id
           for a in schedule}
    assert ids == {"default": registry.get("default").policy_id, "interrogative": registry.get("interrogative").policy_id}
    with pytest.raises(ValueError, match="interrogative"):
        build_counterbalance_schedule(participant_ids=("Alice", "Bob"), counterpart_types=("interrogative",),
                                      surface_variants=("default",))
    plan = resolve_execution_surface(experiment_track="text_only", counterpart_types=["interrogative"],
                                     counterbalance=False, policy_registry=registry)
    assert plan.counterpart_types == ("interrogative",)
    assert plan.counterpart_policy_ids == (registry.get("interrogative").policy_id,)
    with pytest.raises(ValueError, match="interrogative"):
        resolve_execution_surface(experiment_track="text_only", counterpart_types=["interrogative"], counterbalance=False)
    builtin_plan = resolve_execution_surface(experiment_track="text_only", counterpart_types=["skeptical"], counterbalance=False)
    assert builtin_plan.counterpart_policy_ids == (builtin_policy_registry().get("skeptical").policy_id,)
