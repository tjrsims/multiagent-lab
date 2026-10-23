"""Orchestrate a text-track run: model adapter -> plan -> runner -> text dataset.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
1. ``RunSpec`` is the whole experimental design as data (model, scenarios,
   conditions, families, rollouts, twins, counterpart policies, surfaces,
   agent profile). It is written verbatim into the run manifest.
2. ``run_text_track`` builds the acting model (Bedrock, or the scripted
   offline model), resolves the fail-closed ``ExecutionSurfacePlan`` exactly as
   the `deception` CLI does, injects the model into ``InterpretabilityRunner``
   on the ``text_only`` track, and runs the emergent study through the
   canonical transactional runtime.
3. Afterwards it exports an END-Bench text dataset, the study results, the
   cost ledger totals, and a run manifest (plan, spec, versions, hash) into
   ``out_dir``. Nothing here resumes a run; a crash is a new run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from end_bench import END_BENCH_VERSION
from end_bench.data.text_dataset import export_text_dataset, save_text_dataset
from end_bench.models.pricing import CostLedger, load_price_table
from interpretability.scenarios.policy_registry import CounterpartPolicyRegistry
from interpretability.labels.tracks import LabelTrackRegistry
from interpretability.modules import BenchmarkModule
from interpretability.runtime.attestation import CLOSING_ATTESTATION_SOURCE
from interpretability.launch import resolve_execution_surface
from interpretability.schema_registry import schema_registry_checksum


DEFAULT_CONDITIONS = ("high_incentive", "low_incentive", "penalty")
SUPPORTED_CONDITIONS = frozenset({"high_incentive", "low_incentive", "penalty", "minimal"})


class RunSpecError(ValueError):
    """Raised for an inconsistent run specification."""


# Model spec schemes a run accepts beside "scripted" come from the provider
# registry (end_bench.models.providers); adding a provider is one row there.
from end_bench.models.providers import ModelSpecError, build_text_model, known_schemes, parse_model_spec

MODEL_PREFIXES = tuple(f"{scheme}:" for scheme in known_schemes() if scheme != "scripted")


@dataclass(frozen=True)
class RunSpec:
    """One text-track experimental design."""

    model: str                                  # "bedrock:<model_id>", "openai:<model>" or "scripted"
    scenarios: tuple[str, ...] = ("hidden_value", "capability_bluff", "info_withholding", "ultimatum_bluff")
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS
    families: int = 30
    max_rounds: int = 5
    rollouts_per_family: int = 3
    paired_facts: bool = True
    counterpart_types: tuple[str, ...] = ("default",)
    surface_variants: tuple[str, ...] = ("default",)
    counterbalance: bool = False
    counterbalance_seed: int = 0
    ultrafast: bool = True
    agent_modules: tuple[str, ...] = ()
    max_tokens: int = 160
    # None sends no temperature or top_p: the provider's default sampling. Several
    # frontier models reject any temperature (OpenAI GPT-6 on Converse and on the
    # OpenAI API, Claude Fable 5.1) and others advise their default (Gemini 3,
    # DeepSeek V4), so a cross model comparison runs every model this way.
    temperature: float | None = None
    # Model specific request fields (Converse additionalModelRequestFields, or
    # extra body keys on OpenAI compatible endpoints), recorded in the manifest.
    request_extras: Mapping[str, Any] | None = None
    region: str = "us-east-1"
    run_id: str = ""
    price_table: str | None = None
    family_seed_start: int = 0
    # Ask the actor to state its private number before its final turn
    # (interpretability.runtime.attestation); a module may force this on.
    closing_attestation: bool = False
    attestation_timing: str = "closing"
    # Pace the model under a known daily token cap (Bedrock's hidden account
    # level limit); None sends as fast as the request quota allows.
    token_budget_per_day: int | None = None
    # The counterpart's model. None is self play (the acting model plays both
    # sides). A fixed counterpart makes the opponent the same stimulus for
    # every evaluated model, which a leaderboard needs; runs with different
    # counterpart regimes are never pooled.
    counterpart_model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise RunSpecError("model is required")
        try:
            parse_model_spec(self.model)
        except ModelSpecError as error:
            raise RunSpecError(f"unsupported model: {error}") from error
        if self.counterpart_model is not None:
            if not isinstance(self.counterpart_model, str) or not self.counterpart_model.strip():
                raise RunSpecError("unsupported counterpart_model: it must be a non empty string when given")
            try:
                parse_model_spec(self.counterpart_model)
            except ModelSpecError as error:
                raise RunSpecError(f"unsupported counterpart_model: {error}") from error
        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not 0.0 <= float(self.temperature) <= 2.0:
                raise RunSpecError("temperature must be within [0, 2], or None for the provider default")
            from end_bench.models.roster import sampling_constraint

            for role, name in (("model", self.model), ("counterpart_model", self.counterpart_model)):
                if name is not None and sampling_constraint(name) == "provider_default_only":
                    raise RunSpecError(
                        f"{role} {name!r} accepts only its provider default sampling; leave temperature unset"
                    )
        if self.request_extras is not None:
            if not isinstance(self.request_extras, Mapping) or not all(isinstance(k, str) for k in self.request_extras):
                raise RunSpecError("request_extras must be a mapping with string keys")
            object.__setattr__(self, "request_extras", dict(self.request_extras))
        if not self.scenarios:
            raise RunSpecError("at least one scenario is required")
        unknown = set(self.conditions) - SUPPORTED_CONDITIONS
        if unknown:
            raise RunSpecError(f"unsupported conditions: {sorted(unknown)}")
        for name, value in (("families", self.families), ("max_rounds", self.max_rounds),
                            ("rollouts_per_family", self.rollouts_per_family),
                            ("max_tokens", self.max_tokens)):
            if type(value) is not int or value < 1:
                raise RunSpecError(f"{name} must be a positive integer")
        if self.ultrafast and self.agent_modules:
            raise RunSpecError("ultrafast agents take no optional modules")
        if self.token_budget_per_day is not None and (
            type(self.token_budget_per_day) is not int or self.token_budget_per_day < 1
        ):
            raise RunSpecError("token_budget_per_day must be a positive integer when given")
        if not self.run_id:
            object.__setattr__(self, "run_id", f"run-{time.strftime('%Y%m%d-%H%M%S')}")

    @property
    def provider(self) -> str:
        return self.model.split(":", 1)[0] if ":" in self.model else self.model

    @property
    def counterpart_regime(self) -> str:
        """``self_play`` or ``fixed:<model>``; runs are pooled only within one regime."""
        return "self_play" if self.counterpart_model is None else f"fixed:{self.counterpart_model}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_model(spec: RunSpec, ledger: CostLedger, model_spec: str | None = None):
    """The acting model, or the model named by ``model_spec`` (the counterpart).

    Dispatch lives in the provider registry. The run spec's sampling values,
    not Concordia's per call defaults, are what every adapter sends.
    """
    name = spec.model if model_spec is None else model_spec
    try:
        return build_text_model(
            name,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            cost_ledger=ledger,
            token_budget_per_day=spec.token_budget_per_day,
            honor_caller_sampling=False,
            region=spec.region,
            request_extras=spec.request_extras,
        )
    except ModelSpecError as error:
        raise RunSpecError(f"unsupported model: {error}") from error


def summarize_trials(results: Mapping[str, Any]) -> dict[str, Any]:
    """Planned, completed and failed trial counts from the study results, with failure types.

    A failed trial keeps its generation records but produces no dataset rows, so
    a dataset can look complete while whole blocks are missing. The manifest
    therefore states the counts and the error types explicitly.
    """
    planned = completed = failed = 0
    forfeited_turns = 0
    failure_types: dict[str, int] = {}
    by_cell: dict[str, dict[str, int]] = {}
    for scenario, block in (results or {}).items():
        conditions = block.get("conditions") if isinstance(block, Mapping) else None
        if not isinstance(conditions, Mapping):
            continue
        for condition, payload in conditions.items():
            trials = payload.get("trials") if isinstance(payload, Mapping) else None
            if not isinstance(trials, list):
                continue
            cell = by_cell.setdefault(f"{scenario}/{condition}", {"planned": 0, "completed": 0, "failed": 0})
            for trial in trials:
                planned += 1
                cell["planned"] += 1
                if trial.get("runtime_state") == "failed":
                    failed += 1
                    cell["failed"] += 1
                    failure = trial.get("failure") or {}
                    key = str(failure.get("error_type") or "unknown") if isinstance(failure, Mapping) else "unknown"
                    failure_types[key] = failure_types.get(key, 0) + 1
                else:
                    completed += 1
                    cell["completed"] += 1
                # Turns the game master forfeited because a participant kept
                # submitting an acceptance that broke its own private limit.
                checkpoint = trial.get("runtime_checkpoint") if isinstance(trial, Mapping) else None
                events = ((checkpoint or {}).get("adjudicator") or {}).get("events") or []
                forfeited_turns += sum(
                    1 for event in events
                    if isinstance(event, Mapping)
                    and str(((event.get("action") or {}).get("raw_text")) or "").startswith("[FORFEITED_TURN_ENVIRONMENT]")
                )
    return {
        "planned": planned,
        "completed": completed,
        "failed": failed,
        "forfeited_turns": forfeited_turns,
        "failure_types": dict(sorted(failure_types.items())),
        "by_cell": by_cell,
    }


@dataclass(frozen=True)
class RunResult:
    run_id: str
    out_dir: Path
    text_dataset_path: Path
    manifest_path: Path
    results_path: Path
    n_rows: int
    cost_totals: Mapping[str, Any]
    results: Mapping[str, Any] = field(repr=False)
    trials: Mapping[str, Any] = field(default_factory=dict)


def run_text_track(
    spec: RunSpec,
    out_dir: str | Path,
    *,
    policy_registry: "CounterpartPolicyRegistry | None" = None,
    label_tracks: "LabelTrackRegistry | None" = None,
    module: "BenchmarkModule | None" = None,
    preflight: bool = True,
) -> RunResult:
    """Run the design and write the text dataset, results, and manifest.

    A benchmark module supplies its registries through ``module``; the two
    registry arguments exist for callers without a module. ``None`` everywhere
    means the built in policies and tracks.
    """
    if module is not None:
        if policy_registry is not None or label_tracks is not None:
            raise RunSpecError("pass either module or explicit registries, not both")
        policy_registry = module.policy_registry
        label_tracks = module.label_tracks
    closing_attestation = bool(spec.closing_attestation or (module is not None and module.closing_attestation))
    attestation_timing = module.attestation_timing if module is not None else spec.attestation_timing
    target = Path(out_dir) / spec.run_id
    target.mkdir(parents=True, exist_ok=True)
    price_table = load_price_table(spec.price_table) if spec.price_table else None
    ledger = CostLedger(path=target / "cost_ledger.jsonl", table=price_table)
    model = _build_model(spec, ledger)
    counterpart = None
    if spec.counterpart_model is not None:
        counterpart = _build_model(spec, ledger, spec.counterpart_model)
    if preflight:
        # One tiny call before the study. The executor records a model failure
        # as a FAILED trial and keeps going, so without this a model the
        # account cannot use (missing use case form, sales gated id) would
        # burn the whole design and end with "zero rows" and no reason.
        for name, candidate in ((spec.model, model), (spec.counterpart_model, counterpart)):
            if candidate is None:
                continue
            try:
                candidate.sample_text("Reply with the single word: ready")
            except Exception as error:  # noqa: BLE001 - surfaced as the run's reason
                raise RunSpecError(
                    f"preflight call to {name!r} failed, so the run was not started: {error}"
                ) from error

    plan = resolve_execution_surface(
        experiment_track="text_only",
        protocol="alternating",
        counterpart_types=list(spec.counterpart_types),
        counterbalance=spec.counterbalance,
        counterbalance_seed=spec.counterbalance_seed,
        surface_variants=list(spec.surface_variants),
        ultrafast=spec.ultrafast,
        mode="emergent",
        probe_mode="off",
        agent_modules=list(spec.agent_modules) or None,
        rollouts_per_family=spec.rollouts_per_family,
        paired_facts=spec.paired_facts,
        policy_registry=policy_registry,
    )
    manifest_head = {
        "end_bench_version": END_BENCH_VERSION,
        "run_id": spec.run_id,
        "spec": spec.to_dict(),
        "execution_plan": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(plan).items()},
            "experiment_track": plan.experiment_track.value,
            "protocol": plan.protocol.value,
        },
        "schema_registry_checksum": schema_registry_checksum(),
        "benchmark_module": module.provenance() if module is not None else None,
        "closing_attestation": closing_attestation,
        "closing_attestation_source": CLOSING_ATTESTATION_SOURCE if closing_attestation else None,
        "attestation_timing": attestation_timing if closing_attestation else None,
        # Who the actor negotiated against. Self play means the acting model
        # played both sides; a leaderboard pools runs within one regime only.
        "counterpart_model": spec.counterpart_model or spec.model,
        "counterpart_regime": spec.counterpart_regime,
        # Endpoint hosts for API models (never keys); None for Bedrock and scripted.
        "model_endpoints": {
            "actor": getattr(model, "endpoint_description", None),
            "counterpart": getattr(counterpart, "endpoint_description", None) if counterpart is not None else None,
        },
        # What the adapter sends on every acting call. Concordia asks for its
        # own defaults per call; the adapter ignores them and sends these.
        "sampling": {
            "temperature": spec.temperature if spec.temperature is not None else "provider default (not sent)",
            "max_tokens": spec.max_tokens,
            "top_p": (
                "not sent (provider default)" if spec.temperature is None
                else "adapter default 0.9; not sent to Anthropic models"
            ),
            "request_extras": dict(spec.request_extras or {}),
            "policy": "run spec values override caller values; the generation records hold requested and effective",
        },
        "started_at": time.time(),
    }
    (target / "run_manifest.json").write_text(json.dumps(manifest_head, indent=2, sort_keys=True, default=str))

    from interpretability.evaluation import InterpretabilityRunner

    runner = InterpretabilityRunner(
        model_name=model.model_name,
        device="cpu",
        max_tokens=spec.max_tokens,
        evaluator_api=None,
        evaluator_type="rule",
        trial_id_offset=spec.family_seed_start,
        experiment_track="text_only",
        captured_actor_ids=(),
        model=model,
        run_id=spec.run_id,
        counterpart_model=counterpart,
    )
    results = runner.run_all_emergent_scenarios(
        scenarios=list(spec.scenarios),
        trials_per_scenario=spec.families,
        conditions=list(spec.conditions),
        agent_modules=list(spec.agent_modules),
        max_rounds=spec.max_rounds,
        ultrafast=spec.ultrafast,
        counterpart_types=plan.counterpart_types,
        protocol=plan.protocol,
        counterbalance=plan.counterbalance,
        counterbalance_seed=plan.counterbalance_seed,
        surface_variants=plan.surface_variants,
        run_probes=False,
        rollouts_per_family=plan.rollouts_per_family,
        paired_facts=plan.paired_facts,
        policy_registry=policy_registry,
        label_tracks=label_tracks,
        closing_attestation=closing_attestation,
        attestation_timing=attestation_timing,
    )
    totals = ledger.totals()
    text_rows = list(getattr(runner, 'text_samples', ()))
    if not text_rows and not runner.activation_samples:
        last_error = getattr(model, "last_error", None)
        raise RunSpecError(
            "the run produced zero rows; every trial failed before a label could be made"
            + (f". Last model error: {last_error}" if last_error else "")
            + f". Ledger: {totals['calls']} successful calls."
        )
    dataset = export_text_dataset(
        samples=text_rows if text_rows else runner.activation_samples,
        generation_records=runner.generation_records,
        label_records=runner.label_records,
        interaction_events=runner.interaction_events,
        model_name=model.model_name,
        provider=spec.provider,
        experiment_track="text_only",
        run_manifest={**manifest_head, "cost_totals": totals},
    )
    dataset_path = save_text_dataset(target / "text_dataset.json", dataset)
    results_path = target / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
    trial_summary = summarize_trials(results)
    manifest = {
        **manifest_head,
        "finished_at": time.time(),
        "cost_totals": totals,
        "n_rows": len(dataset["metadata"]),
        "dataset_hash": dataset["config"]["dataset_hash"],
        "text_dataset": dataset_path.name,
        "trials": trial_summary,
        # Provider refusals (a refusal stop reason with HTTP 200) per side. The
        # adapters count them; the refused turn itself reaches the dataset as
        # an empty response that QC drops, so without this the run could not
        # tell a refusing model from a silent one.
        "refusals": {
            "model": int(getattr(model, "refusals", 0) or 0),
            "counterpart": (int(getattr(counterpart, "refusals", 0) or 0) if counterpart is not None else None),
        },
    }
    manifest_path = target / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return RunResult(
        run_id=spec.run_id,
        out_dir=target,
        text_dataset_path=dataset_path,
        manifest_path=manifest_path,
        results_path=results_path,
        n_rows=len(dataset["metadata"]),
        trials=trial_summary,
        cost_totals=totals,
        results=results,
    )


__all__ = [
    "DEFAULT_CONDITIONS",
    "RunResult",
    "RunSpec",
    "RunSpecError",
    "run_text_track",
]
