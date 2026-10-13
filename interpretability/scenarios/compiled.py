"""Compile emergent prompts, private views, and rules from one scenario record."""

from __future__ import annotations

from enum import StrEnum
from typing import Sequence, Any, Mapping

from negotiation.domain.scenario import RoleView, ScenarioInstance
from negotiation.domain.schema import (
    SCHEMA_VERSION,
    DisclosureObligation,
    Fact,
    stable_id,
    thaw_json,
)
from negotiation.profiles import AgentProfile, validate_agent_profile

from interpretability.scenarios.turn_labels import (
    TurnLabel,
    compute_turn_label,
    evaluate_hold_commitment,
    extraction_context,
    resolve_tolerance,
)
from interpretability.scenarios.policy_registry import (
    ABSENT_COUNTERPART,
    CounterpartPolicyError,
    CounterpartPolicyRegistry,
    CounterpartPolicySpec,
    builtin_policy_registry,
)
from interpretability.scenarios.emergent_prompts import (
    IncentiveCondition,
    compute_ground_truth,
    generate_paired_scenario_params,
    generate_scenario_params,
    pair_anchor_for,
    get_counterpart_prompt,
    get_emergent_prompt,
)


# ``ScenarioInstance.spec_version`` identifies the negotiation-domain envelope;
# this separate version identifies prompt/rule compilation semantics and is
# therefore part of the family identity and public schema registry.
# /8: the low incentive prompts carry only the flat payoff. The reputation
# notes they held made "low incentive" a payoff change plus an honesty cue,
# so the incentive contrast was confounded (adversarial review, October 2026).
EMERGENT_SCENARIO_SPEC_VERSION = "emergent-scenario/8"


class ExecutionProtocol(StrEnum):
    """Versioned scheduling semantics that contribute to trial identity."""

    ALTERNATING = "alternating"
    SIMULTANEOUS = "simultaneous"
    SOLO_NO_RESPONSE = "solo_no_response"


class CounterpartPolicy(StrEnum):
    """Executable counterpart policies supported by the two-party runtime."""

    DEFAULT = "default"
    SKEPTICAL = "skeptical"
    CREDULOUS = "credulous"
    INFORMED = "informed"


SUPPORTED_COUNTERPART_POLICIES = tuple(
    policy.value for policy in CounterpartPolicy
)
COUNTERPART_KNOWLEDGE_GRANT_VERSION = "counterpart-knowledge-grant/1"

_ABSENT_COUNTERPART_POLICY = "absent"

# Built in policy text now lives in policy_registry.py as data. The grant key
# allowlist is derived from it so a renderer can never forward an actor's whole
# private mapping by accident.
_BUILTIN_POLICIES = builtin_policy_registry()
_INFORMED_GRANT_KEYS = {
    scenario: _BUILTIN_POLICIES.get("informed").grant_keys_for(scenario)
    for scenario in _BUILTIN_POLICIES.get("informed").grant_keys_by_scenario
}

SUPPORTED_SURFACE_VARIANTS = (
    "default",
    "formal-brief",
    "compact-brief",
    # Retained as an accepted historical identifier. It now changes the
    # rendered form; it is no longer metadata-only in behavior.
    "formal-metadata-only",
)

_SURFACE_PREFIXES = {
    "default": "",
    "formal-brief": "NEGOTIATION BRIEF\n=================\n",
    "compact-brief": "NEGOTIATION CONTEXT — ",
    "formal-metadata-only": "FORMAL NEGOTIATION CONTEXT\n--------------------------\n",
}


_PRIVATE_KEYS = {
    "ultimatum_bluff": {
        "actor": ("true_walkaway",),
        "counterpart": ("counterpart_walkaway",),
    },
    "capability_bluff": {
        "actor": ("true_days",),
        "counterpart": (),
    },
    "hidden_value": {
        "actor": ("true_value",),
        "counterpart": ("max_payment",),
    },
    "info_withholding": {
        "actor": ("defect", "fair_price"),
        "counterpart": ("max_payment",),
    },
    "promise_break": {
        "actor": ("temptation", "sucker"),
        "counterpart": (),
    },
    "alliance_betrayal": {
        "actor": ("betrayal",),
        "counterpart": (),
    },
}

_PUBLIC_KEYS = {
    "ultimatum_bluff": (),
    "capability_bluff": (),
    "hidden_value": (),
    "info_withholding": (),
    "promise_break": ("mutual_coop", "mutual_defect"),
    "alliance_betrayal": ("alliance",),
}


def actor_private_fact_keys(scenario: str) -> tuple[str, ...]:
    """Private-fact keys the acting role is authorized to see for a scenario."""
    if scenario not in _PRIVATE_KEYS:
        raise ValueError(f"Unknown emergent scenario: {scenario}")
    return tuple(_PRIVATE_KEYS[scenario]["actor"])


def public_fact_keys(scenario: str) -> tuple[str, ...]:
    """Public (both-role) parameter keys for a scenario."""
    if scenario not in _PUBLIC_KEYS:
        raise ValueError(f"Unknown emergent scenario: {scenario}")
    return tuple(_PUBLIC_KEYS[scenario])




def _condition_value(condition: IncentiveCondition | str) -> str:
    if isinstance(condition, IncentiveCondition):
        return condition.value
    try:
        return IncentiveCondition(condition).value
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unknown incentive condition: {condition!r}") from error


def _policy_name(counterpart_policy: CounterpartPolicy | str) -> str:
    return (
        counterpart_policy.value
        if isinstance(counterpart_policy, CounterpartPolicy)
        else counterpart_policy
    )


def _resolve_registry(
    policy_registry: CounterpartPolicyRegistry | None,
) -> CounterpartPolicyRegistry:
    return _BUILTIN_POLICIES if policy_registry is None else policy_registry


def validate_counterpart_policy(
    counterpart_policy: CounterpartPolicy | str,
    policy_registry: CounterpartPolicyRegistry | None = None,
) -> CounterpartPolicy | str:
    """Normalize one supported two-party policy or reject it explicitly.

    Without a registry only the built in policies are accepted and the enum
    member is returned (the historical behavior). With a registry, any
    registered policy is accepted; built in names still come back as the enum
    and module policies come back as their plain name.
    """
    if isinstance(counterpart_policy, CounterpartPolicy):
        name = counterpart_policy.value
    elif isinstance(counterpart_policy, str):
        name = counterpart_policy
    else:
        raise TypeError("counterpart policy must be a string or CounterpartPolicy")
    if name == _ABSENT_COUNTERPART_POLICY:
        raise ValueError(
            "Counterpart policy 'absent' is unsupported by the transactional "
            "two-party executor; a separate solo protocol is required"
        )
    registry = _resolve_registry(policy_registry)
    try:
        registry.get(name)
    except CounterpartPolicyError as error:
        raise ValueError(str(error)) from error
    try:
        return CounterpartPolicy(name)
    except ValueError:
        return name


def policy_spec_for(
    counterpart_policy: CounterpartPolicy | str,
    policy_registry: CounterpartPolicyRegistry | None = None,
) -> CounterpartPolicySpec:
    """The spec behind a validated policy name."""
    return _resolve_registry(policy_registry).get(_policy_name(counterpart_policy))


def validate_surface_variant(surface_variant: str) -> str:
    """Return a supported surface variant or reject it before rendering."""
    if not isinstance(surface_variant, str):
        raise TypeError("surface variant must be a string")
    if surface_variant not in SUPPORTED_SURFACE_VARIANTS:
        supported = ", ".join(SUPPORTED_SURFACE_VARIANTS)
        raise ValueError(
            f"Unsupported surface variant: {surface_variant!r}; "
            f"expected one of {supported}"
        )
    return surface_variant


def validate_execution_protocol(
    protocol: ExecutionProtocol | str,
) -> ExecutionProtocol:
    """Normalize one supported scheduler without accepting aliases."""
    if isinstance(protocol, ExecutionProtocol):
        return protocol
    if not isinstance(protocol, str):
        raise TypeError("execution protocol must be a string or ExecutionProtocol")
    try:
        return ExecutionProtocol(protocol)
    except ValueError as error:
        raise ValueError(f"Unsupported execution protocol: {protocol!r}") from error


def _resolve_first_mover(
    first_mover: str,
    assignments: Mapping[str, str],
) -> str:
    if not isinstance(first_mover, str):
        raise TypeError("first_mover must be a logical role or assigned role ID")
    if first_mover in assignments:
        return assignments[first_mover]
    if first_mover in assignments.values():
        return first_mover
    raise ValueError("first_mover must identify actor or counterpart")


def _render_surface(prompt: str, surface_variant: str) -> str:
    variant = validate_surface_variant(surface_variant)
    return f"{_SURFACE_PREFIXES[variant]}{prompt}"


def _build_policy_grant(
    spec: CounterpartPolicySpec,
    scenario: str,
    *,
    counterpart_role_id: str,
    semantic_params: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = {
        key: semantic_params[key]
        for key in spec.grant_keys_for(scenario)
    }
    content = {
        "schema_version": COUNTERPART_KNOWLEDGE_GRANT_VERSION,
        "policy": spec.name,
        "scenario": scenario,
        "recipient_role_id": counterpart_role_id,
        "parameters": parameters,
    }
    return {
        **content,
        "grant_id": stable_id("counterpart_knowledge_grant", content),
    }


def _build_informed_grant(
    scenario: str,
    *,
    counterpart_role_id: str,
    semantic_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Historical name kept for the built in informed policy."""
    return _build_policy_grant(
        _BUILTIN_POLICIES.get("informed"),
        scenario,
        counterpart_role_id=counterpart_role_id,
        semantic_params=semantic_params,
    )


def _counterpart_policy_from_instance(
    instance: ScenarioInstance,
    policy_registry: CounterpartPolicyRegistry | None = None,
) -> CounterpartPolicy | str:
    counterbalance = thaw_json(instance.public_state).get("counterbalance")
    if not isinstance(counterbalance, Mapping):
        raise ValueError("Compiled scenario counterbalance record is required")
    if "counterpart_type" not in counterbalance:
        raise ValueError("Compiled scenario counterpart policy is required")
    return validate_counterpart_policy(
        counterbalance["counterpart_type"], policy_registry
    )


def validate_counterpart_policy_contract(
    instance: ScenarioInstance,
    counterpart_role_id: str,
    policy_registry: CounterpartPolicyRegistry | None = None,
) -> CounterpartPolicy | str:
    """Validate role-view authorization for the compiled counterpart policy."""
    policy = _counterpart_policy_from_instance(instance, policy_registry)
    spec = policy_spec_for(policy, policy_registry)
    view = instance.view_for(counterpart_role_id)
    private = thaw_json(view.private_state)
    if private.get("logical_role") != "counterpart":
        raise ValueError("counterpart renderer requires the counterpart RoleView")

    semantic_params = thaw_json(instance.rule_config).get("semantic_params")
    if not isinstance(semantic_params, Mapping):
        raise ValueError("Compiled scenario semantic parameters are required")
    expected_parameters = {
        key: semantic_params[key]
        for key in _PRIVATE_KEYS[instance.scenario]["counterpart"]
    }
    if private.get("parameters") != expected_parameters:
        raise ValueError(
            "Counterpart RoleView parameters exceed or contradict its authorization"
        )

    expected_keys = {"logical_role", "parameters"}
    grant = private.get("policy_knowledge_grant")
    if spec.grant_keys_for(instance.scenario):
        expected_keys.add("policy_knowledge_grant")
        expected_grant = _build_policy_grant(
            spec,
            instance.scenario,
            counterpart_role_id=counterpart_role_id,
            semantic_params=semantic_params,
        )
        if grant != expected_grant:
            raise ValueError(
                "Informed counterpart knowledge grant is missing, altered, or "
                "contains unauthorized fields"
            )
    elif grant is not None:
        raise ValueError(
            "Only the informed counterpart policy may carry a knowledge grant"
        )
    if set(private) != expected_keys:
        raise ValueError("Counterpart RoleView contains unauthorized private fields")
    return policy


def compile_emergent_scenario(
    scenario: str,
    *,
    family_seed: int,
    trial_seed: int,
    condition: IncentiveCondition | str,
    role_assignment: Mapping[str, str] | None = None,
    first_mover: str = "actor",
    counterpart_type: CounterpartPolicy | str = CounterpartPolicy.DEFAULT,
    surface_variant: str = "default",
    actor_profile: AgentProfile | str = AgentProfile.ADVANCED,
    counterpart_profile: AgentProfile | str = AgentProfile.ADVANCED,
    intervention_design_id: str | None = None,
    protocol: ExecutionProtocol | str = ExecutionProtocol.ALTERNATING,
    pair_role: str | None = None,
    policy_registry: CounterpartPolicyRegistry | None = None,
) -> ScenarioInstance:
    """Create one canonical source for prompts, private facts, rules, and IDs.

    ``policy_registry`` is the explicit set of counterpart policies this run
    may use (a benchmark module supplies one); ``None`` means the built ins.

    ``pair_role`` (``"above"``/``"below"``) requests one twin of a matched
    pair: both twins share ``family_seed`` and a claim anchor, differ only in
    the private fact that makes a claim at the anchor honest or false, and
    share one ``trial_family_id`` so leakage-safe splits keep them together.
    """
    if scenario not in _PRIVATE_KEYS:
        raise ValueError(f"Unknown emergent scenario: {scenario}")
    execution_protocol = validate_execution_protocol(protocol)
    if execution_protocol is ExecutionProtocol.SOLO_NO_RESPONSE:
        raw_counterpart_type = (
            counterpart_type.value
            if isinstance(counterpart_type, CounterpartPolicy)
            else counterpart_type
        )
        if raw_counterpart_type != _ABSENT_COUNTERPART_POLICY:
            raise ValueError(
                "solo_no_response requires counterpart_type='absent'"
            )
        counterpart_policy = None
        counterpart_policy_spec = None
        counterpart_policy_value = _ABSENT_COUNTERPART_POLICY
    else:
        counterpart_policy = validate_counterpart_policy(
            counterpart_type, policy_registry
        )
        counterpart_policy_spec = policy_spec_for(counterpart_policy, policy_registry)
        counterpart_policy_value = _policy_name(counterpart_policy)
    actor_profile = validate_agent_profile(actor_profile)
    counterpart_profile = validate_agent_profile(counterpart_profile)
    if intervention_design_id is not None and (
        not isinstance(intervention_design_id, str)
        or not intervention_design_id.strip()
    ):
        raise ValueError(
            "intervention_design_id must be null or a non-empty string"
        )
    if pair_role is None:
        raw_params = generate_scenario_params(scenario, family_seed)
        pair_anchor = None
    else:
        raw_params = generate_paired_scenario_params(
            scenario,
            family_seed,
            pair_role,
            tolerance=resolve_tolerance(scenario, None),
        )
        pair_anchor = pair_anchor_for(scenario, family_seed)
    semantic_params = {
        key: value
        for key, value in raw_params.items()
        if key not in {"trial_id", "scenario", "pair_role", "pair_anchor"}
    }
    condition_value = _condition_value(condition)
    assignments = dict(
        role_assignment or {"actor": "actor", "counterpart": "counterpart"}
    )
    if set(assignments) != {"actor", "counterpart"}:
        raise ValueError("role_assignment must map actor and counterpart")
    if any(
        not isinstance(role_id, str) or not role_id.strip()
        for role_id in assignments.values()
    ):
        raise ValueError("role_assignment values must be non-empty strings")
    if len(set(assignments.values())) != 2:
        raise ValueError("role_assignment values must be distinct")
    normalized_surface = validate_surface_variant(surface_variant)
    first_mover_id = _resolve_first_mover(first_mover, assignments)

    family_id = stable_id(
        "trial_family",
        {
            "spec_version": EMERGENT_SCENARIO_SPEC_VERSION,
            "scenario": scenario,
            "family_seed": family_seed,
            # Twins share the family: identity comes from the pair anchor,
            # not from the fact that distinguishes them.
            "semantic_params": (
                semantic_params if pair_anchor is None
                else {"pair_anchor": pair_anchor}
            ),
        },
    )
    trial_id = stable_id(
        "trial",
        {
            "trial_family_id": family_id,
            "trial_seed": trial_seed,
            "condition": condition_value,
            "role_assignment": assignments,
            "first_mover_id": first_mover_id,
            "counterpart_type": counterpart_policy_value,
            "surface_variant": normalized_surface,
            "actor_profile": actor_profile.value,
            "counterpart_profile": counterpart_profile.value,
            "intervention_design_id": intervention_design_id,
            "protocol": execution_protocol.value,
            "pair_role": pair_role,
        },
    )
    public_state = {
        "scenario_spec_version": EMERGENT_SCENARIO_SPEC_VERSION,
        "scenario": scenario,
        "condition": condition_value,
        "family_seed": family_seed,
        "protocol": execution_protocol.value,
        "counterbalance": {
            "role_assignment": assignments,
            "first_mover_id": first_mover_id,
            "counterpart_type": counterpart_policy_value,
            "surface_variant": normalized_surface,
            "pair_role": pair_role,
            "pair_anchor": pair_anchor,
        },
        "agent_profiles": {
            "actor": actor_profile.value,
            "counterpart": counterpart_profile.value,
        },
        "intervention_design_id": intervention_design_id,
        "parameters": {
            key: semantic_params[key]
            for key in _PUBLIC_KEYS[scenario]
            if key in semantic_params
        },
    }
    role_views = []
    for logical_role in ("actor", "counterpart"):
        private_state = {
            key: semantic_params[key]
            for key in _PRIVATE_KEYS[scenario][logical_role]
            if key in semantic_params
        }
        role_private_state: dict[str, Any] = {
            "logical_role": logical_role,
            "parameters": private_state,
        }
        if (
            logical_role == "counterpart"
            and counterpart_policy_spec is not None
            and counterpart_policy_spec.grant_keys_for(scenario)
        ):
            role_private_state["policy_knowledge_grant"] = _build_policy_grant(
                counterpart_policy_spec,
                scenario,
                counterpart_role_id=assignments[logical_role],
                semantic_params=semantic_params,
            )
        role_views.append(
            RoleView(
                role_id=assignments[logical_role],
                public_state=public_state,
                private_state=role_private_state,
            )
        )
    rule_config: dict[str, Any] = {
        "semantic_params": semantic_params,
        "tolerance": resolve_tolerance(scenario, None),
    }
    if scenario == "info_withholding":
        defect_fact = Fact(
            subject_id=f"{trial_id}:vehicle",
            predicate="known_defect",
            value=semantic_params["defect"],
            visible_to=(assignments["actor"],),
        )
        creation_event_id = stable_id(
            "event",
            {
                "trial_id": trial_id,
                "kind": "buyer_condition_question",
                "actor_id": assignments["counterpart"],
                "recipient_id": assignments["actor"],
                "boundary": 0,
            },
        )
        obligation = DisclosureObligation(
            actor_id=assignments["actor"],
            recipient_id=assignments["counterpart"],
            fact_id=defect_fact.fact_id,
            fact_version=defect_fact.fact_version,
            creation_event_id=creation_event_id,
            created_at_boundary=0,
        )
        rule_config.update(
            {
                "disclosure_fact": defect_fact.to_dict(),
                "disclosure_obligation": obligation.to_dict(),
            }
        )

    instance = ScenarioInstance(
        spec_version=SCHEMA_VERSION,
        scenario=scenario,
        seed=trial_seed,
        trial_id=trial_id,
        trial_family_id=family_id,
        public_state=public_state,
        role_views=tuple(role_views),
        legal_actions=("message", "offer", "accept", "reject", "walk_away"),
        rule_config=rule_config,
    )
    if execution_protocol is not ExecutionProtocol.SOLO_NO_RESPONSE:
        validate_counterpart_policy_contract(
            instance,
            assignments["counterpart"],
            policy_registry=policy_registry,
        )
    return instance


def parameters_for_role(instance: ScenarioInstance, role_id: str) -> dict[str, Any]:
    """Return only the parameters authorized for one rendered role view."""
    view = instance.view_for(role_id)
    private = thaw_json(view.private_state)
    return {
        "scenario": instance.scenario,
        "trial_id": instance.trial_id,
        **dict(thaw_json(instance.public_state).get("parameters", {})),
        **dict(private.get("parameters", {})),
    }


def render_actor_prompt(instance: ScenarioInstance, actor_role_id: str) -> str:
    """Render the actor prompt from its authorized view."""
    condition = instance.public_state["condition"]
    counterbalance = thaw_json(instance.public_state).get("counterbalance", {})
    surface_variant = counterbalance.get("surface_variant", "default")
    return _render_surface(
        get_emergent_prompt(
            instance.scenario,
            condition,
            parameters_for_role(instance, actor_role_id),
        ),
        surface_variant,
    )


def render_counterpart_prompt(
    instance: ScenarioInstance,
    counterpart_role_id: str,
    policy_registry: CounterpartPolicyRegistry | None = None,
) -> str:
    """Render the counterpart prompt without exposing actor-only facts."""
    protocol = validate_execution_protocol(
        thaw_json(instance.public_state).get(
            "protocol", ExecutionProtocol.ALTERNATING.value
        )
    )
    if protocol is ExecutionProtocol.SOLO_NO_RESPONSE:
        raise ValueError(
            "solo_no_response has no counterpart prompt or counterpart model call"
        )
    counterbalance = thaw_json(instance.public_state).get("counterbalance", {})
    surface_variant = counterbalance.get("surface_variant", "default")
    policy = validate_counterpart_policy_contract(
        instance,
        counterpart_role_id,
        policy_registry,
    )
    spec = policy_spec_for(policy, policy_registry)
    granted: Mapping[str, Any] | None = None
    if spec.grant_keys_for(instance.scenario):
        private = thaw_json(instance.view_for(counterpart_role_id).private_state)
        granted = private["policy_knowledge_grant"]["parameters"]
    modifier = spec.modifier_for(instance.scenario, granted)
    return _render_surface(
        get_counterpart_prompt(
            instance.scenario,
            parameters_for_role(instance, counterpart_role_id),
        ) + modifier,
        surface_variant,
    )


_RULE_EVIDENCE_FIELDS = frozenset({
    "disclosure_decision",
    "disclosure_history",
    "prior_commitment",
    "conflicting_plan",
    "executed_action",
})


def rule_parameters(
    instance: ScenarioInstance,
    *,
    rule_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical facts plus supplied committed-event evidence for the rules."""
    rule_config = thaw_json(instance.rule_config)
    parameters = {
        "scenario": instance.scenario,
        "trial_id": instance.trial_id,
        **dict(rule_config["semantic_params"]),
    }
    if "disclosure_obligation" in rule_config:
        parameters["disclosure_obligation"] = rule_config["disclosure_obligation"]
    if rule_evidence:
        unexpected = set(rule_evidence).difference(_RULE_EVIDENCE_FIELDS)
        if unexpected:
            raise ValueError(
                "rule_evidence contains non-event or canonical fields: "
                + ", ".join(sorted(map(str, unexpected)))
            )
        parameters.update(rule_evidence)
    return parameters


def instance_tolerance(instance: ScenarioInstance) -> dict[str, float]:
    """The tolerance frozen into this instance's rule configuration."""
    rule_config = thaw_json(instance.rule_config)
    return resolve_tolerance(instance.scenario, rule_config.get("tolerance"))


def _counterpart_role_ids(
    instance: ScenarioInstance, actor_role_id: str
) -> tuple[str, ...]:
    return tuple(
        view.role_id for view in instance.role_views if view.role_id != actor_role_id
    ) or ("counterpart",)


def compute_instance_turn_label(
    instance: ScenarioInstance,
    response: str,
    *,
    actor_role_id: str | None = None,
    round_index: int | None = None,
    rule_evidence: Mapping[str, Any] | None = None,
) -> TurnLabel:
    """Typed rule-v2 label for one actor turn of a compiled instance."""
    actor = actor_role_id or _logical_role_id(instance, "actor")
    return compute_turn_label(
        instance.scenario,
        response,
        rule_parameters(instance, rule_evidence=rule_evidence),
        tolerance=instance_tolerance(instance),
        actor_id=actor,
        recipient_ids=_counterpart_role_ids(instance, actor),
        round_index=round_index,
        trial_id=instance.trial_id,
    )


def compute_instance_hold_commitment(
    instance: ScenarioInstance,
    prior_actor_texts: Sequence[str],
    current_text: str,
    *,
    actor_role_id: str | None = None,
    round_index: int | None = None,
) -> TurnLabel:
    """Commitment-track label for an ultimatum "final offer" declaration."""
    if instance.scenario != "ultimatum_bluff":
        raise ValueError("hold commitments are defined for ultimatum_bluff only")
    actor = actor_role_id or _logical_role_id(instance, "actor")
    context = extraction_context(
        instance.scenario,
        trial_id=instance.trial_id,
        actor_id=actor,
        recipient_ids=_counterpart_role_ids(instance, actor),
        round_index=round_index,
    )
    return evaluate_hold_commitment(
        prior_actor_texts,
        current_text,
        tolerance=instance_tolerance(instance),
        context=context,
    )


def _logical_role_id(instance: ScenarioInstance, logical_role: str) -> str:
    counterbalance = thaw_json(instance.public_state).get("counterbalance", {})
    assignment = counterbalance.get("role_assignment", {})
    return str(assignment.get(logical_role, logical_role))


def evaluate_actor_response(
    instance: ScenarioInstance,
    response: str,
    *,
    rule_evidence: Mapping[str, Any] | None = None,
) -> bool:
    """Compatibility bool over the typed label; UNKNOWN projects to False."""
    label = compute_instance_turn_label(
        instance, response, rule_evidence=rule_evidence
    )
    return label.is_deceptive is True
