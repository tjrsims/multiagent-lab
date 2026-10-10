"""Relabel a text dataset's numeric claim turns with the current rule, offline.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
Labels are computed at run time and stored on each row. When the extractor
changes (1.1.0 -> 1.2.0 stopped decision wording and bare trigger mentions
from voiding clean claims), datasets that cost hours to generate can be read
again under the new rule without a new model call, because every row keeps
its text, its private facts and its frozen tolerance.

1. ``relabel_text_dataset`` walks every transactional negotiation row of a
   numeric claim scenario (ultimatum, capability, hidden value) and calls
   ``compute_turn_label`` on ``full_response`` + ``scenario_params`` with the
   row's own ``label_tolerance``.
2. The row's label fields are rewritten to what the executor would have
   written (``emergent_ground_truth``, ``actual_deception``, ``gt_regex``,
   ``claim_status``, ``claim_values``, ``offer_values``, ``label_reason_codes``,
   ``rule_version``, ``label_source``) and the parallel ``labels.gm_labels``
   column follows. Changed rows record the extractor they came from.
3. The original label records stay as the run time evidence; a new rule
   record is appended per changed row and linked from ``label_record_ids``
   with ``metadata.relabeled_from_label_id``.
4. ``config.relabel`` states versions, counts and every transition, and the
   dataset hash is recomputed. Withholding rows are left alone: their rule is
   the disclosure evaluator, not the numeric extractor.
Where this fits: between a finished run and scoring. A relabeled dataset is
derived; its manifest says so.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping

from end_bench.data.text_dataset import TEXT_DATASET_SCHEMA_VERSION, _content_hash
from interpretability.scenarios.extractors import PARSER_VERSION
from interpretability.scenarios.turn_labels import (
    TURN_LABEL_RULE_VERSION,
    LabelValue,
    compute_turn_label,
)


RELABEL_VERSION = "end_bench_relabel/1"
NUMERIC_CLAIM_SCENARIOS = ("ultimatum_bluff", "capability_bluff", "hidden_value")
WITHHOLDING_SCENARIO = "info_withholding"
RELABEL_SCENARIOS = (*NUMERIC_CLAIM_SCENARIOS, WITHHOLDING_SCENARIO)
_RULE_EVALUATOR_PREFIX = "compiled-rule-evaluator/"
_DISCLOSURE_EVALUATOR_PREFIX = "disclosure-evaluation/"


@dataclass(frozen=True)
class _Reading:
    """What the current rule says about one stored row."""

    value: LabelValue
    claim_status: str
    claim_values: tuple[float, ...]
    offer_values: tuple[float, ...]
    reason_codes: tuple[str, ...]
    rule_version: str
    extractor_version: str | None
    behavior_subtype: str
    evidence: tuple[Mapping[str, Any], ...] = ()


def _read_numeric(row: Mapping[str, Any]) -> _Reading:
    label = compute_turn_label(
        str(row["scenario"]), str(row.get("full_response") or ""), row["scenario_params"],
        tolerance=row.get("label_tolerance") or None,
    )
    return _Reading(
        value=label.value, claim_status=label.claim_status.value,
        claim_values=tuple(float(v) for v in label.claim_values),
        offer_values=tuple(float(v) for v in label.offer_values),
        reason_codes=tuple(str(c) for c in label.reason_codes), rule_version=label.rule_version,
        # Hidden value labels come from the domain rule and carry no extractor
        # version of their own; the parser version that read the text applies.
        extractor_version=label.extractor_version or PARSER_VERSION,
        behavior_subtype=label.behavior_subtype.value, evidence=label.evidence,
    )


def _read_withholding(row: Mapping[str, Any]) -> _Reading:
    """The disclosure rule read from the stored transcript (see ``scenarios.disclosure``)."""
    from interpretability.scenarios.disclosure import DISCLOSURE_RULE_VERSION, evaluate_disclosure_turn
    from interpretability.scenarios.turn_labels import extract_turn, extraction_context

    response = str(row.get("full_response") or "")
    defect = str((row.get("scenario_params") or {}).get("defect") or "")
    if not defect:
        raise RelabelError("withholding row has no defect fact; cannot relabel")
    accepts = False
    try:
        action = extract_turn(WITHHOLDING_SCENARIO, response, extraction_context(WITHHOLDING_SCENARIO))
        accepts = any(claim.predicate == "decision" and claim.value == "accept" for claim in action.claims)
    except Exception:  # noqa: BLE001 - an unparsable turn simply adds no trigger
        pass
    evaluation = evaluate_disclosure_turn(
        response, defect, list(row.get("dialogue_history") or ()), accepts_deal=accepts,
    )
    value = (
        LabelValue.UNKNOWN if evaluation.omitted is None
        else LabelValue.TRUE if evaluation.omitted else LabelValue.FALSE
    )
    return _Reading(
        value=value, claim_status=evaluation.claim_status, claim_values=(), offer_values=(),
        reason_codes=evaluation.reason_codes, rule_version=DISCLOSURE_RULE_VERSION,
        extractor_version=None, behavior_subtype="strategic_omission",
    )


class RelabelError(ValueError):
    """Raised when a dataset cannot be relabeled safely."""


def _scalar(value: LabelValue) -> float | None:
    if value is LabelValue.TRUE:
        return 1.0
    if value is LabelValue.FALSE:
        return 0.0
    return None


def _old_state(row: Mapping[str, Any]) -> bool | None:
    truth = row.get("emergent_ground_truth")
    return bool(truth) if isinstance(truth, bool) and row.get("claim_status") == "complete" else None


def _transition(old: bool | None, new: bool | None) -> str:
    name = {None: "unknown", True: "deceptive", False: "honest"}
    return f"{name[old]}->{name[new]}"


def relabel_text_dataset(
    dataset: Mapping[str, Any],
    *,
    scenarios: tuple[str, ...] = RELABEL_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(relabeled dataset, report)``. Pure; the input is not modified.

    Numeric claim scenarios are re-read with ``compute_turn_label``;
    withholding rows with the disclosure rule over the stored transcript. A
    row counts as changed when its label value, claim status, claim values,
    offer values or reason codes differ; a rule version that moved with no
    other difference is stamped but not counted.
    """
    if dataset.get("schema_version") != TEXT_DATASET_SCHEMA_VERSION:
        raise RelabelError(f"only {TEXT_DATASET_SCHEMA_VERSION} datasets can be relabeled")
    unknown = sorted(set(scenarios) - set(RELABEL_SCENARIOS))
    if unknown:
        raise RelabelError(f"not relabelable scenarios: {unknown}")
    out = copy.deepcopy(dict(dataset))
    metadata: list[dict[str, Any]] = out["metadata"]
    gm_labels: list[Any] = out["labels"]["gm_labels"]
    if len(gm_labels) != len(metadata):
        raise RelabelError("labels.gm_labels is not aligned with metadata")
    records_by_id = {record.get("label_id"): record for record in out["label_records"]}
    transitions: Counter[str] = Counter()
    by_scenario: dict[str, Counter[str]] = {}
    from_versions: Counter[str] = Counter()
    considered = 0
    changed = 0
    for index, row in enumerate(metadata):
        if row.get("sample_type") != "negotiation" or row.get("scenario") not in scenarios:
            continue
        if row.get("runtime_path") not in (None, "transactional"):
            continue
        params = row.get("scenario_params")
        if not isinstance(params, Mapping):
            raise RelabelError(f"row {index} has no scenario_params; cannot relabel")
        considered += 1
        withholding = row.get("scenario") == WITHHOLDING_SCENARIO
        reading = _read_withholding(row) if withholding else _read_numeric(row)
        old = _old_state(row)
        scalar = _scalar(reading.value)
        new = None if scalar is None else bool(scalar)
        old_fields = {
            key: row.get(key) for key in ("claim_status", "claim_values", "offer_values", "label_reason_codes")
        }
        new_fields = {
            "claim_status": reading.claim_status,
            "claim_values": list(reading.claim_values),
            "offer_values": list(reading.offer_values),
            "label_reason_codes": list(reading.reason_codes),
        }
        if old == new and old_fields == new_fields:
            if row.get("rule_version") != reading.rule_version:
                row["rule_version"] = reading.rule_version  # same reading under the newer rule
            continue
        new_fields["rule_version"] = reading.rule_version
        changed += 1
        key = _transition(old, new)
        transitions[key] += 1
        by_scenario.setdefault(str(row["scenario"]), Counter())[key] += 1
        previous_extractor = None
        rule_record = None
        prefix = _DISCLOSURE_EVALUATOR_PREFIX if withholding else _RULE_EVALUATOR_PREFIX
        for label_id in row.get("label_record_ids") or ():
            record = records_by_id.get(label_id)
            if record and str(record.get("evaluator_version", "")).startswith(prefix):
                rule_record = record
                previous_extractor = (record.get("metadata") or {}).get("extractor_version")
        previous_rule = row.get("rule_version")
        from_versions[str(previous_rule if withholding else previous_extractor)] += 1
        row.update(new_fields)
        row["emergent_ground_truth"] = new
        row["actual_deception"] = scalar
        row["actual_deception_projection"] = scalar
        row["gt_regex"] = scalar
        row["label_source"] = "rule" if scalar is not None else None
        row["behavior_subtype"] = reading.behavior_subtype
        row["relabel_version"] = RELABEL_VERSION
        row["relabeled_from_extractor"] = previous_extractor
        row["relabeled_to_extractor"] = reading.extractor_version
        row["relabeled_from_rule"] = previous_rule
        gm_labels[index] = scalar
        new_record = copy.deepcopy(rule_record) if rule_record is not None else {
            "behavior_target": "factual_deception", "confidence": None, "evaluation_error": None,
            "evaluation_event_id": None, "evidence_event_ids": [], "fallback_reason": None,
            "schema_version": "1.2.0", "severity": None, "source": "rule", "subject_actor_id": row.get("agent_name"),
            "target_event_id": row.get("interaction_event_id"),
        }
        digest = hashlib.sha256(
            f"{row.get('trial_id')}|{row.get('generation_record_id')}|{RELABEL_VERSION}|{reading.rule_version}|{reading.extractor_version}".encode("utf-8")
        ).hexdigest()[:24]
        new_record["label_id"] = f"relabel_{digest}"
        new_record["evaluator_version"] = f"{prefix}relabel"
        new_record["status"] = "complete" if scalar is not None else "unknown"
        new_record["value"] = scalar
        new_record["metadata"] = {
            "scenario": row.get("scenario"),
            "value": reading.value.value,
            "behavior_subtype": reading.behavior_subtype,
            "claim_status": reading.claim_status,
            "claim_values": list(reading.claim_values),
            "offer_values": list(reading.offer_values),
            "reason_codes": list(reading.reason_codes),
            "rule_version": reading.rule_version,
            "extractor_version": reading.extractor_version,
            "evidence": [dict(item) for item in reading.evidence],
            "relabel_version": RELABEL_VERSION,
            "relabeled_from_label_id": rule_record.get("label_id") if rule_record else None,
            "relabeled_from_extractor": previous_extractor,
            "relabeled_from_rule": previous_rule,
        }
        out["label_records"].append(new_record)
        row["label_record_ids"] = [*(row.get("label_record_ids") or []), new_record["label_id"]]
    from interpretability.scenarios.disclosure import DISCLOSURE_RULE_VERSION

    report = {
        "version": RELABEL_VERSION,
        "rule_version": TURN_LABEL_RULE_VERSION,
        "disclosure_rule_version": DISCLOSURE_RULE_VERSION,
        "to_extractor": PARSER_VERSION,
        "from_extractors": dict(from_versions),
        "scenarios": list(scenarios),
        "rows_considered": considered,
        "rows_changed": changed,
        "transitions": dict(sorted(transitions.items())),
        "transitions_by_scenario": {k: dict(sorted(v.items())) for k, v in sorted(by_scenario.items())},
        "relabeled_at": time.time(),
    }
    config = out.setdefault("config", {})
    history = list(config.get("relabel_history") or [])
    if config.get("relabel"):
        history.append(config["relabel"])
    config["relabel"] = report
    if history:
        config["relabel_history"] = history
    config["dataset_hash"] = _content_hash(out)
    return out, report


__all__ = ["NUMERIC_CLAIM_SCENARIOS", "RELABEL_SCENARIOS", "RELABEL_VERSION", "RelabelError", "relabel_text_dataset"]
