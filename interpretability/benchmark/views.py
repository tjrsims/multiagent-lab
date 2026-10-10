"""Build the three v1 detector views from a saved activation dataset.

The v1 benchmark (benchmark_planning/03_V1_TARGET.md) scores three detectors
over the SAME negotiation turns:

- **transcript-blind** — the response text only;
- **fact-informed** — the response text + the acting agent's *authorized*
  private facts;
- **white-box** — the captured residual-stream activations.

The contrast that defines the benchmark: on turns where the deception label
depends on a private fact absent from the transcript, a transcript-blind
detector is information-limited while fact-informed and white-box detectors
are not.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
This module is pure: it consumes the dict written by ``save_dataset`` and
never loads a model. It fails closed everywhere the earlier version fell back:
1. **One label source per build.** ``label_source`` is required; a row is kept
   only when its own selected source matches and its value is known. There is
   no substitution of a GM/judge score for a missing rule label.
2. **Claim-bearing rows by default.** ``rows="claim_bearing"`` keeps only turns
   whose rule found a complete, checkable claim. A turn with no claim is not
   honest; its rate is reported per scenario in ``ViewsReport`` instead.
3. **QC applied.** ``qc_rejected`` rows and rows with QC flags are excluded and
   counted.
4. **Actor-authorized facts only.** The fact-informed view renders exactly the
   keys the acting role was authorized to see (``actor_private_fact_keys``),
   never counterpart facts, payoffs, or identifiers.
5. **Grouping fails closed.** The chosen ``group_key`` must be present on every
   kept row; with more than one counterpart policy the group also carries the
   counterpart assignment so dyads never straddle a split. An optional
   ``SplitManifest`` is audited: every group must sit in one partition.
6. **Runtime path.** Rows from the legacy inline runtime are rejected outright
   for structured-evidence scenarios (their labels were never computed) and
   excluded elsewhere unless explicitly allowed.
Every exclusion is counted in ``BenchmarkViews.report`` so the manifest can
hash the audit trail.

Where this fits: the final, benchmark stage of the scenarios -> agents ->
capture -> labels -> QC -> dataset -> probes/causal/benchmark pipeline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from interpretability.scenarios.compiled import actor_private_fact_keys


VIEWS_VERSION = "benchmark-views/2"

LabelSourceName = Literal["rule", "model_judge", "gm", "human"]
RowsPolicy = Literal["claim_bearing", "all"]
QcPolicy = Literal["exclude", "keep"]
GroupKey = Literal["trial_family_id", "scenario_instance_id", "trial_id"]

STRUCTURED_EVIDENCE_SCENARIOS = frozenset(
    {"info_withholding", "promise_break", "alliance_betrayal"}
)
_LEGACY_RUNTIME_PATH = "legacy_compatibility"
_IDENTIFIER_KEYS = frozenset({"trial_id", "scenario", "pair_role", "pair_anchor"})


class ViewsError(ValueError):
    """Raised when a dataset cannot be projected without guessing."""


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like to a float numpy array."""
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=float)


def _require(dataset: Mapping[str, Any], key: str) -> Any:
    if key not in dataset:
        raise ViewsError(f"dataset is missing required top-level key: {key!r}")
    return dataset[key]


@dataclass(frozen=True)
class ViewsReport:
    """Audit trail of what the views kept and why rows were excluded."""

    label_source: str
    rows_policy: str
    qc_policy: str
    group_key: str
    n_input_rows: int
    n_kept: int
    excluded: Mapping[str, int]
    excluded_by_scenario: Mapping[str, Mapping[str, int]]
    no_claim_rate_by_scenario: Mapping[str, float]
    claim_status_counts_by_scenario: Mapping[str, Mapping[str, int]]
    scenarios: tuple[str, ...]
    rule_versions: tuple[str, ...]
    qc_versions: tuple[str, ...]
    runtime_paths: tuple[str, ...]
    counterpart_types: tuple[str, ...]
    n_groups: int
    dyad_aware_groups: bool
    split_manifest_id: str | None
    version: str = VIEWS_VERSION
    # Distinct ``source_run_id`` values when the dataset is a pooled one
    # (``end_bench.data.pooled``); empty for a single-run dataset, and then
    # left out of ``to_dict`` so existing reports keep their exact shape.
    source_runs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = {
            "version": self.version,
            "label_source": self.label_source,
            "rows_policy": self.rows_policy,
            "qc_policy": self.qc_policy,
            "group_key": self.group_key,
            "n_input_rows": self.n_input_rows,
            "n_kept": self.n_kept,
            "excluded": dict(sorted(self.excluded.items())),
            "excluded_by_scenario": {
                scenario: dict(sorted(counts.items()))
                for scenario, counts in sorted(self.excluded_by_scenario.items())
            },
            "no_claim_rate_by_scenario": dict(
                sorted(self.no_claim_rate_by_scenario.items())
            ),
            "claim_status_counts_by_scenario": {
                scenario: dict(sorted(counts.items()))
                for scenario, counts in sorted(
                    self.claim_status_counts_by_scenario.items()
                )
            },
            "scenarios": list(self.scenarios),
            "rule_versions": list(self.rule_versions),
            "qc_versions": list(self.qc_versions),
            "runtime_paths": list(self.runtime_paths),
            "counterpart_types": list(self.counterpart_types),
            "n_groups": self.n_groups,
            "dyad_aware_groups": self.dyad_aware_groups,
            "split_manifest_id": self.split_manifest_id,
        }
        if self.source_runs:
            out["source_runs"] = list(self.source_runs)
        return out


@dataclass(frozen=True)
class BenchmarkViews:
    """Three aligned detector views over N negotiation turns.

    All arrays/tuples are index-aligned: row ``i`` of every field describes the
    same turn. ``activations`` maps a layer key to an ``(N, d_model)`` array.
    """

    labels: np.ndarray  # (N,) int {0, 1}
    responses: tuple[str, ...]  # transcript-blind text
    private_facts: tuple[Mapping[str, Any], ...]  # actor-authorized facts only
    scenarios: tuple[str, ...]
    group_ids: np.ndarray  # (N,) str — leakage-safe split key
    activations: Mapping[Any, np.ndarray]
    claim_status: tuple[str | None, ...] = ()
    claim_values: tuple[tuple[float, ...], ...] = ()
    reason_codes: tuple[tuple[str, ...], ...] = ()
    # Per-row tolerance frozen into the labeling rule (``label_tolerance``);
    # the fact-informed baseline must relabel with exactly this, not defaults.
    tolerances: tuple[Mapping[str, float] | None, ...] = ()
    rule_versions: tuple[str | None, ...] = ()
    label_sources: tuple[str, ...] = ()
    incentive_conditions: tuple[str | None, ...] = ()
    counterpart_types: tuple[str | None, ...] = ()
    trial_ids: tuple[str, ...] = ()
    runtime_paths: tuple[str | None, ...] = ()
    pair_roles: tuple[str | None, ...] = ()
    pair_anchors: tuple[float | None, ...] = ()
    partitions: tuple[str | None, ...] = ()
    dialogue_history: tuple[tuple[str, ...], ...] = ()
    generation_record_ids: tuple[str | None, ...] = ()
    # The run each row came from (``source_run_id``) in a pooled dataset.
    # Twins and matched pairs never cross a source; leakage groups still may,
    # because two runs of one family share the same private facts.
    sources: tuple[str | None, ...] = ()
    report: ViewsReport | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        n = len(self.labels)
        aligned = (
            self.responses, self.private_facts, self.scenarios, self.group_ids,
        )
        if any(len(item) != n for item in aligned):
            raise ViewsError("view fields are not index-aligned")
        optional = (
            self.claim_status, self.claim_values, self.reason_codes,
            self.tolerances, self.rule_versions, self.label_sources, self.incentive_conditions,
            self.counterpart_types, self.trial_ids, self.runtime_paths,
            self.pair_roles, self.pair_anchors, self.partitions,
            self.dialogue_history, self.generation_record_ids, self.sources,
        )
        if any(item and len(item) != n for item in optional):
            raise ViewsError("optional view fields are not index-aligned")
        for layer, matrix in self.activations.items():
            if matrix.shape[0] != n:
                raise ViewsError(
                    f"activation layer {layer!r} has {matrix.shape[0]} rows, "
                    f"expected {n}"
                )

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def layers(self) -> tuple[Any, ...]:
        return tuple(self.activations)

    def white_box(self, layer: Any) -> np.ndarray:
        """Return the ``(N, d_model)`` activation matrix for one layer."""
        if layer not in self.activations:
            raise KeyError(
                f"layer {layer!r} not captured; available: {self.layers}"
            )
        return self.activations[layer]

    def fact_informed_text(self, index: int) -> str:
        """Serialize response + the actor's authorized facts (sorted keys)."""
        facts = self.private_facts[index]
        rendered = "; ".join(f"{key}={facts[key]}" for key in sorted(facts))
        return f"{self.responses[index]}\n[private: {rendered}]"

    def scenario_mask(self, scenario: str) -> np.ndarray:
        """Boolean mask selecting rows from one scenario."""
        return np.array([name == scenario for name in self.scenarios], dtype=bool)

    def claim_bearing_mask(self) -> np.ndarray:
        """Rows whose rule found a complete, checkable claim."""
        if not self.claim_status:
            return np.ones(self.n, dtype=bool)
        return np.array([status == "complete" for status in self.claim_status], dtype=bool)

    def partition_mask(self, partition: str) -> np.ndarray:
        if not self.partitions:
            raise ViewsError("views were built without a split manifest")
        return np.array([item == partition for item in self.partitions], dtype=bool)


# --------------------------------------------------------------------------
# Row resolution helpers
# --------------------------------------------------------------------------


def _label_from_row(
    row: Mapping[str, Any], label_source: str
) -> tuple[int | None, str | None]:
    """Return ``(label, exclusion_reason)`` for one row under one source."""
    if label_source == "human":
        value = row.get("human_label")
        if isinstance(value, bool):
            return int(value), None
        return None, "label_unavailable"
    declared = row.get("label_source")
    if declared is None:
        return None, "label_source_missing"
    if declared != label_source:
        return None, f"label_source:{declared}"
    if label_source == "rule":
        value = row.get("emergent_ground_truth")
        if isinstance(value, bool):
            return int(value), None
        return None, "label_unavailable"
    key = "gt_llm_rules" if label_source == "model_judge" else "actual_deception"
    value = row.get(key)
    if isinstance(value, bool):
        return int(value), None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(float(value) > 0.5), None
    return None, "label_unavailable"


def _qc_rejected(row: Mapping[str, Any]) -> bool:
    if row.get("sample_type") == "qc_rejected":
        return True
    flags = row.get("qc_flags")
    return bool(flags)


def _actor_facts(row: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    params = row.get("scenario_params")
    if params is None:
        raise ViewsError(
            "sample is missing 'scenario_params' — regenerate data with the "
            "scenario_params provenance fix, or the fact-informed view cannot "
            "be built"
        )
    try:
        allowed = actor_private_fact_keys(scenario)
    except ValueError as error:
        raise ViewsError(f"row has an unknown scenario {scenario!r}: {error}") from error
    missing = [key for key in allowed if key not in params]
    if missing:
        raise ViewsError(
            f"scenario_params for {scenario!r} lacks authorized actor facts: "
            f"{missing}"
        )
    return {key: params[key] for key in allowed}


def _claim_values(row: Mapping[str, Any]) -> tuple[float, ...]:
    values = row.get("claim_values") or ()
    return tuple(
        float(value) for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------



def _partition_lookup(split_manifest: Any) -> tuple[dict[str, str], str | None]:
    assignments = getattr(split_manifest, "assignments", None)
    if assignments is None:
        raise ViewsError("split_manifest must expose .assignments")
    partition_of = {str(item.trial_id): str(item.partition) for item in assignments}
    return partition_of, getattr(split_manifest, "manifest_id", None)


def _resolve_partitions(
    trial_ids: Sequence[str],
    group_ids: Sequence[str],
    partition_of: Mapping[str, str],
) -> tuple[str, ...]:
    """Map every kept trial to its partition; fail closed on a straddling group."""
    resolved: list[str] = []
    by_group: dict[str, set[str]] = defaultdict(set)
    for trial_id, group in zip(trial_ids, group_ids):
        if trial_id not in partition_of:
            raise ViewsError(f"trial {trial_id!r} is missing from the split manifest")
        resolved.append(partition_of[trial_id])
        by_group[str(group)].add(partition_of[trial_id])
    leaks = {group: sorted(parts) for group, parts in by_group.items() if len(parts) > 1}
    if leaks:
        raise ViewsError(f"groups straddle split partitions: {leaks}")
    return tuple(resolved)


def _row_tolerance(row: Mapping[str, Any]) -> Mapping[str, float] | None:
    value = row.get("label_tolerance")
    if not isinstance(value, Mapping) or not value:
        return None
    return {str(key): float(item) for key, item in value.items()}


def attach_partitions(views: BenchmarkViews, split_manifest: Any) -> BenchmarkViews:
    """Return ``views`` with partitions from ``split_manifest`` attached.

    Equivalent to rebuilding the views with ``split_manifest=...`` but without
    re-filtering rows or re-converting activation layers; the same
    missing-trial and straddle checks apply.
    """
    partition_of, manifest_id = _partition_lookup(split_manifest)
    partitions = _resolve_partitions(
        views.trial_ids, [str(group) for group in views.group_ids], partition_of
    )
    report = (
        replace(views.report, split_manifest_id=manifest_id)
        if views.report is not None else None
    )
    return replace(views, partitions=partitions, report=report)


def build_views(
    dataset: Mapping[str, Any],
    *,
    label_source: LabelSourceName = "rule",
    rows: RowsPolicy = "claim_bearing",
    qc: QcPolicy = "exclude",
    group_key: GroupKey = "trial_family_id",
    allowed_runtime_paths: frozenset[str] | Sequence[str] = frozenset({"transactional"}),
    split_manifest: Any | None = None,
) -> BenchmarkViews:
    """Project a loaded dataset dict into the three aligned detector views.

    Args:
        dataset: the mapping written by ``save_dataset`` (keys ``activations``,
            ``labels``, ``metadata``, ...).
        label_source: the one label source every kept row must carry.
        rows: ``"claim_bearing"`` keeps only rows with a complete claim;
            ``"all"`` keeps every labeled negotiation row.
        qc: ``"exclude"`` drops QC-rejected/flagged rows; ``"keep"`` retains them.
        group_key: the leakage-safe grouping key, required on every kept row.
        allowed_runtime_paths: runtime paths whose rows may enter the views.
            Legacy rows are always rejected for structured-evidence scenarios.
        split_manifest: optional ``SplitManifest``; when given, every group must
            sit in exactly one partition and ``partitions`` is populated.
    """
    if label_source not in ("rule", "model_judge", "gm", "human"):
        raise ViewsError(f"unsupported label_source: {label_source!r}")
    if rows not in ("claim_bearing", "all"):
        raise ViewsError(f"unsupported rows policy: {rows!r}")
    if qc not in ("exclude", "keep"):
        raise ViewsError(f"unsupported qc policy: {qc!r}")
    if group_key not in ("trial_family_id", "scenario_instance_id", "trial_id"):
        raise ViewsError(f"unsupported group_key: {group_key!r}")
    allowed_paths = frozenset(str(item) for item in allowed_runtime_paths)
    if not allowed_paths:
        raise ViewsError("allowed_runtime_paths must not be empty")

    metadata: Sequence[Mapping[str, Any]] = list(_require(dataset, "metadata"))
    labels_block: Mapping[str, Any] = _require(dataset, "labels")
    activations_block: Mapping[Any, Any] = _require(dataset, "activations")
    gm_labels = list(labels_block.get("gm_labels", [None] * len(metadata)))
    if len(gm_labels) != len(metadata):
        raise ViewsError("labels.gm_labels and metadata are not aligned")
    stacked = {
        layer: _to_numpy(matrix) for layer, matrix in activations_block.items()
    }
    for layer, matrix in stacked.items():
        if matrix.shape[0] != len(metadata):
            raise ViewsError(
                f"activation layer {layer!r} has {matrix.shape[0]} rows, "
                f"metadata has {len(metadata)}"
            )

    partition_of: dict[str, str] | None = None
    manifest_id: str | None = None
    if split_manifest is not None:
        partition_of, manifest_id = _partition_lookup(split_manifest)

    excluded: Counter[str] = Counter()
    excluded_by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    claim_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    keep: list[int] = []
    resolved_labels: list[int] = []

    def exclude(index: int, reason: str) -> None:
        excluded[reason] += 1
        excluded_by_scenario[str(metadata[index].get("scenario", "unknown"))][reason] += 1

    for index, row in enumerate(metadata):
        scenario = str(row.get("scenario", "unknown"))
        sample_type = row.get("sample_type", "negotiation")
        runtime_path = row.get("runtime_path")
        if (
            scenario in STRUCTURED_EVIDENCE_SCENARIOS
            and runtime_path == _LEGACY_RUNTIME_PATH
        ):
            raise ViewsError(
                f"{scenario} rows from the legacy runtime carry dead labels "
                "(structured disclosure/commitment evidence is never supplied "
                "on that path); regenerate on the transactional runtime"
            )
        if sample_type == "qc_rejected":
            # The transactional runtime files QC failures under their own
            # sample type; they are negotiation turns that failed QC.
            if qc == "exclude":
                exclude(index, "qc_rejected")
                continue
        elif sample_type != "negotiation":
            exclude(index, f"sample_type:{sample_type}")
            continue
        if runtime_path is None:
            exclude(index, "runtime_path_missing")
            continue
        if runtime_path not in allowed_paths:
            exclude(index, f"runtime_path:{runtime_path}")
            continue
        if qc == "exclude" and _qc_rejected(row):
            exclude(index, "qc_rejected")
            continue
        status = row.get("claim_status")
        claim_status_counts[scenario][str(status) if status is not None else "missing"] += 1
        label, reason = _label_from_row(row, label_source)
        if reason is not None:
            exclude(index, reason)
            continue
        if rows == "claim_bearing":
            if status is None:
                exclude(index, "claim_status_missing")
                continue
            if status != "complete":
                exclude(index, f"claim_status:{status}")
                continue
        keep.append(index)
        resolved_labels.append(int(label))

    if not keep:
        excluded_summary = ", ".join(f"{k}={v}" for k, v in sorted(excluded.items())) or "none"
        claim_summary = "; ".join(
            f"{scn}: " + ", ".join(f"{st}={n}" for st, n in sorted(counts.items()))
            for scn, counts in sorted(claim_status_counts.items())
        ) or "none"
        raise ViewsError(
            "no negotiation turns remain after filtering "
            f"(rows={rows!r}, qc={qc!r}, label_source={label_source!r}; "
            f"input_rows={len(metadata)}; excluded: {excluded_summary}; "
            f"claim_status by scenario: {claim_summary}). "
            "A dataset whose turns carry no checkable claim cannot be scored; "
            "elicit claims (skeptical/informed counterparts, more rounds) or pass rows='all' "
            "for behavioral rates only."
        )

    kept_rows = [metadata[i] for i in keep]
    scenarios = tuple(str(row.get("scenario", "unknown")) for row in kept_rows)
    for row, scenario in zip(kept_rows, scenarios):
        if row.get(group_key) is None:
            raise ViewsError(
                f"kept row is missing the grouping key {group_key!r}; refusing to "
                "fall back to a different key (a family could straddle a split)"
            )
    counterpart_types = tuple(
        (str(row["counterpart_type"]) if row.get("counterpart_type") is not None else None)
        for row in kept_rows
    )
    distinct_policies = {item for item in counterpart_types if item is not None}
    # Several counterpart policies make the design dyad-aware, but the leakage
    # unit stays the trial family: twins, rollouts, mirrored roles, and every
    # counterpart variant of one family share one group so none can straddle a
    # split. A per-dyad suffix would carve one family into several groups and
    # let its private facts appear on both sides of a locked split.
    dyad_aware = len(distinct_policies) > 1
    group_ids_list: list[str] = [f"{group_key}:{row[group_key]}" for row in kept_rows]

    trial_ids = tuple(str(row.get("trial_id")) for row in kept_rows)
    partitions: tuple[str | None, ...] = ()
    if partition_of is not None:
        partitions = _resolve_partitions(trial_ids, group_ids_list, partition_of)

    private_facts = tuple(_actor_facts(row, scenario) for row, scenario in zip(kept_rows, scenarios))
    responses = tuple(str(row.get("full_response", "")) for row in kept_rows)
    sources: tuple[str | None, ...] = tuple(
        (str(row["source_run_id"]) if row.get("source_run_id") is not None else None)
        for row in kept_rows
    )
    if all(item is None for item in sources):
        sources = ()  # a single-run dataset carries no sources at all
    kept_activations = {layer: matrix[keep] for layer, matrix in stacked.items()}

    no_claim_rate = {}
    for scenario, counts in claim_status_counts.items():
        total = sum(counts.values())
        no_claim_rate[scenario] = (
            counts.get("no_relevant_action", 0) / total if total else float("nan")
        )
    report = ViewsReport(
        label_source=label_source,
        rows_policy=rows,
        qc_policy=qc,
        group_key=group_key,
        n_input_rows=len(metadata),
        n_kept=len(keep),
        excluded=dict(excluded),
        excluded_by_scenario={k: dict(v) for k, v in excluded_by_scenario.items()},
        no_claim_rate_by_scenario=no_claim_rate,
        claim_status_counts_by_scenario={k: dict(v) for k, v in claim_status_counts.items()},
        scenarios=tuple(sorted(set(scenarios))),
        rule_versions=tuple(sorted({str(row.get("rule_version")) for row in kept_rows if row.get("rule_version")})),
        qc_versions=tuple(sorted({str(row.get("qc_version")) for row in kept_rows if row.get("qc_version")})),
        runtime_paths=tuple(sorted({str(row.get("runtime_path")) for row in kept_rows})),
        counterpart_types=tuple(sorted(distinct_policies)),
        n_groups=len(set(group_ids_list)),
        dyad_aware_groups=dyad_aware,
        split_manifest_id=manifest_id,
        source_runs=tuple(sorted({item for item in sources if item is not None})),
    )
    return BenchmarkViews(
        labels=np.asarray(resolved_labels, dtype=int),
        responses=responses,
        private_facts=private_facts,
        scenarios=scenarios,
        group_ids=np.asarray(group_ids_list, dtype=object),
        activations=kept_activations,
        claim_status=tuple(
            (str(row["claim_status"]) if row.get("claim_status") is not None else None)
            for row in kept_rows
        ),
        claim_values=tuple(_claim_values(row) for row in kept_rows),
        reason_codes=tuple(
            tuple(str(code) for code in (row.get("label_reason_codes") or ()))
            for row in kept_rows
        ),
        tolerances=tuple(_row_tolerance(row) for row in kept_rows),
        rule_versions=tuple(row.get("rule_version") for row in kept_rows),
        label_sources=tuple(str(row.get("label_source")) for row in kept_rows),
        incentive_conditions=tuple(row.get("incentive_condition") for row in kept_rows),
        counterpart_types=counterpart_types,
        trial_ids=trial_ids,
        runtime_paths=tuple(row.get("runtime_path") for row in kept_rows),
        pair_roles=tuple(row.get("pair_role") for row in kept_rows),
        pair_anchors=tuple(_optional_float(row.get("pair_anchor")) for row in kept_rows),
        partitions=partitions,
        dialogue_history=tuple(
            tuple(str(line) for line in (row.get("dialogue_history") or ()))
            for row in kept_rows
        ),
        generation_record_ids=tuple(
            (str(row["generation_record_id"]) if row.get("generation_record_id") is not None else None)
            for row in kept_rows
        ),
        sources=sources,
        report=report,
    )


def subset_views(views: BenchmarkViews, indices: Sequence[int]) -> BenchmarkViews:
    """The same views restricted to ``indices`` (in the given order).

    Every index-aligned field is re-indexed; ``report`` still describes the
    full views it came from, so callers that persist a subset must record the
    selection themselves. Duplicate or out-of-range indices are refused.
    """
    import dataclasses

    chosen = [int(item) for item in indices]
    if len(set(chosen)) != len(chosen):
        raise ViewsError("subset indices must be unique")
    if any(item < 0 or item >= views.n for item in chosen):
        raise ViewsError("subset index out of range")
    replacements: dict[str, Any] = {}
    for item in dataclasses.fields(views):
        value = getattr(views, item.name)
        if item.name == "activations":
            replacements[item.name] = {layer: matrix[chosen] for layer, matrix in value.items()}
        elif item.name == "report":
            continue
        elif isinstance(value, np.ndarray) and value.shape[:1] == (views.n,):
            replacements[item.name] = value[chosen]
        elif isinstance(value, tuple) and len(value) == views.n:
            replacements[item.name] = tuple(value[i] for i in chosen)
    return dataclasses.replace(views, **replacements)


__all__ = [
    "STRUCTURED_EVIDENCE_SCENARIOS",
    "attach_partitions",
    "subset_views",
    "VIEWS_VERSION",
    "BenchmarkViews",
    "ViewsError",
    "ViewsReport",
    "build_views",
]
