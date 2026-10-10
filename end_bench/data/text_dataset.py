"""END-Bench text dataset: rows, labels, and canonical records without arrays.

The research pipeline deliberately refuses to publish an *activation dataset*
for text-only runs (it would be an empty-array dataset). Text tracks need a
first-class artifact of their own: the same per-turn metadata rows the
activation dataset carries (so `benchmark.views.build_views` works unchanged),
the label block, and the canonical generation / label / interaction records
for lineage — plus a content hash and the schema registry snapshot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from interpretability.core.dataset_builder import DatasetBuilder, _serialize_records
from interpretability.schema_registry import (
    schema_registry,
    schema_registry_checksum,
)


TEXT_DATASET_SCHEMA_VERSION = "end_bench_text_dataset/1"
_REQUIRED_TOP_LEVEL = frozenset({
    "schema_version", "config", "labels", "metadata", "activations",
    "generation_records", "label_records", "interaction_events",
})


class TextDatasetError(ValueError):
    """Raised when a text dataset is malformed or its hash does not match."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    """Convert tuples/mappings to JSON-safe structures; reject NaN/inf."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TextDatasetError("text dataset values must be finite")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value  # enums
    return str(value)


def _content_hash(dataset: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("labels", "metadata", "generation_records", "label_records", "interaction_events"):
        digest.update(_canonical(dataset[key]))
    return f"sha256:{digest.hexdigest()}"


def export_text_dataset(
    *,
    samples: Sequence[Any],
    generation_records: Sequence[Any],
    label_records: Sequence[Any],
    interaction_events: Sequence[Any],
    model_name: str,
    provider: str,
    experiment_track: str = "text_only",
    run_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the text dataset mapping from runner state (no files touched)."""
    if not samples:
        raise TextDatasetError("cannot export a text dataset with zero rows")
    scenarios = [
        sample.emergent_scenario or sample.scenario_type or "unknown" for sample in samples
    ]
    metadata = [
        _json_safe(DatasetBuilder._sample_metadata(sample, scenario))
        for sample, scenario in zip(samples, scenarios)
    ]
    gm_labels = [
        (1.0 if sample.emergent_ground_truth else 0.0)
        if sample.emergent_ground_truth is not None
        else sample.actual_deception
        for sample in samples
    ]
    labels = {
        "gm_labels": _json_safe(gm_labels),
        "agent_labels": _json_safe([sample.perceived_deception for sample in samples]),
        "scenario": scenarios,
        "mode_labels": [sample.experiment_mode or "unknown" for sample in samples],
        "round_nums": [sample.round_num for sample in samples],
        "sample_type": [sample.sample_type for sample in samples],
    }
    generation = _json_safe(_serialize_records(list(generation_records), id_key="call_id"))
    label_rows = _json_safe(_serialize_records(list(label_records), id_key="label_id"))
    events = _json_safe(_serialize_records(list(interaction_events), id_key="event_id"))
    generation_ids = {row["call_id"] for row in generation}
    label_ids = {row["label_id"] for row in label_rows}
    for row in metadata:
        if row.get("generation_record_id") is not None and row["generation_record_id"] not in generation_ids:
            raise TextDatasetError(
                f"row references a missing generation record: {row['generation_record_id']}"
            )
        missing = set(row.get("label_record_ids") or ()) - label_ids
        if missing:
            raise TextDatasetError(f"row references missing label records: {sorted(missing)}")
    dataset: dict[str, Any] = {
        "schema_version": TEXT_DATASET_SCHEMA_VERSION,
        "config": {
            "model_name": model_name,
            "provider": provider,
            "experiment_track": experiment_track,
            "n_rows": len(metadata),
            "schema_registry": schema_registry(),
            "schema_registry_checksum": schema_registry_checksum(),
            "run_manifest": _json_safe(dict(run_manifest or {})),
        },
        "labels": labels,
        "metadata": metadata,
        "activations": {},
        "generation_records": generation,
        "label_records": label_rows,
        "interaction_events": events,
    }
    dataset["config"]["dataset_hash"] = _content_hash(dataset)
    return dataset


def save_text_dataset(path: str | Path, dataset: Mapping[str, Any]) -> Path:
    target = Path(path)
    if target.suffix != ".json":
        raise TextDatasetError("text datasets are JSON files (.json)")
    missing = _REQUIRED_TOP_LEVEL - set(dataset)
    if missing:
        raise TextDatasetError(f"text dataset is missing keys: {sorted(missing)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dataset, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    return target


def is_text_dataset(path: str | Path) -> bool:
    """Whether ``path`` is a JSON object declaring the text dataset schema."""
    source = Path(path)
    if source.suffix != ".json" or not source.is_file():
        return False
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == TEXT_DATASET_SCHEMA_VERSION


def load_text_dataset(path: str | Path) -> dict[str, Any]:
    """Load and verify a text dataset; returns the mapping ``build_views`` expects."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TextDatasetError("text dataset must be a JSON object")
    if payload.get("schema_version") != TEXT_DATASET_SCHEMA_VERSION:
        raise TextDatasetError(
            f"unsupported text dataset schema: {payload.get('schema_version')!r}"
        )
    missing = _REQUIRED_TOP_LEVEL - set(payload)
    if missing:
        raise TextDatasetError(f"text dataset is missing keys: {sorted(missing)}")
    if payload["activations"] not in ({}, []):
        raise TextDatasetError("text datasets carry no activation arrays")
    expected = payload["config"].get("dataset_hash")
    actual = _content_hash(payload)
    if expected != actual:
        raise TextDatasetError("text dataset hash does not match its contents")
    payload["activations"] = {}
    return payload


__all__ = [
    "TEXT_DATASET_SCHEMA_VERSION",
    "TextDatasetError",
    "export_text_dataset",
    "is_text_dataset",
    "load_text_dataset",
    "save_text_dataset",
]
