"""Pool several END-Bench text datasets into one derived dataset.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
Some checks in the benchmark look across runs rather than within one: the
human adjudication batch samples 300 claim-bearing turns across every model
and counterpart policy, and the judge audit scores judges on the same pooled
turns. The research pipeline writes one text dataset per run, and the
executor stamps call ids from a run identity that does not include the model,
so two runs of the same design produce identical trial ids and call ids. A
naive concatenation would therefore collide.

1. ``pool_text_datasets`` takes loaded text datasets (``load_text_dataset``)
   and derives one source id per dataset from its run manifest ``run_id``.
2. Every row and record id that must be unique inside a dataset is prefixed
   with ``<source>::`` (trial ids, generation record ids, interaction event
   ids, label record ids, receipts). ``trial_family_id`` and
   ``scenario_instance_id`` are left alone on purpose: two runs of one family
   share the same private facts, so they must stay one leakage group.
3. Each row gains ``source_run_id``, ``source_model`` and
   ``source_dataset_hash``; ``benchmark.views`` carries ``source_run_id`` as
   ``BenchmarkViews.sources`` and the pair builder never pairs across sources.
4. The pooled dataset keeps the ``end_bench_text_dataset/1`` shape, so
   ``build_views``, ``adjudicate export`` and ``judge`` work unchanged. Its
   ``config.run_manifest.pooled`` lists the sources. It is a derived audit
   artifact, not a run: never a leaderboard entry, never replayable.
Where this fits: after the runs, before the human check and the judge audit.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from end_bench.data.text_dataset import (
    TEXT_DATASET_SCHEMA_VERSION,
    _content_hash,
)


POOLED_DATASET_VERSION = "end_bench_pooled_text_dataset/1"
SOURCE_SEPARATOR = "::"

# Row fields that name one thing inside a dataset and would collide across runs.
_ROW_SCALAR_IDS = ("trial_id", "generation_record_id", "interaction_event_id")
_ROW_LIST_IDS = ("label_record_ids", "intervention_application_receipt_ids")
# Record fields, per record list.
_GENERATION_IDS = ("call_id", "replay_call_id", "trial_id")
_LABEL_SCALAR_IDS = ("label_id", "evaluation_event_id", "target_event_id")
_LABEL_LIST_IDS = ("evidence_event_ids",)
_EVENT_IDS = ("event_id", "negotiation_id")


class PooledDatasetError(ValueError):
    """Raised when datasets cannot be pooled safely."""


def source_id_for(dataset: Mapping[str, Any]) -> str:
    """The run id from the dataset's run manifest, else a hash prefix."""
    config = dataset.get("config") or {}
    manifest = config.get("run_manifest") or {}
    run_id = manifest.get("run_id") if isinstance(manifest, Mapping) else None
    if run_id:
        return str(run_id)
    digest = str(config.get("dataset_hash") or "")
    if digest.startswith("sha256:") and len(digest) > 23:
        return f"dataset_{digest[7:23]}"
    raise PooledDatasetError("dataset has neither a run id nor a dataset hash to name it by")


def prefix_id(source: str, value: Any) -> Any:
    """``<source>::<value>`` for a non-empty string; other values pass through."""
    if value is None or value == "":
        return value
    return f"{source}{SOURCE_SEPARATOR}{value}"


def split_source_id(value: str) -> tuple[str | None, str]:
    """Undo ``prefix_id``: ``(source, original)``; ``(None, value)`` if unprefixed."""
    if SOURCE_SEPARATOR not in value:
        return None, value
    source, original = value.split(SOURCE_SEPARATOR, 1)
    return source, original


def _prefix_fields(record: dict[str, Any], source: str, scalars: Sequence[str], lists: Sequence[str] = ()) -> None:
    for key in scalars:
        if key in record:
            record[key] = prefix_id(source, record[key])
    for key in lists:
        if record.get(key):
            record[key] = [prefix_id(source, item) for item in record[key]]


def pool_text_datasets(
    datasets: Sequence[Mapping[str, Any]],
    *,
    allow_mixed_registry: bool = False,
) -> dict[str, Any]:
    """Concatenate text datasets with source-prefixed ids. Pure; returns a new mapping."""
    if len(datasets) < 1:
        raise PooledDatasetError("at least one dataset is required")
    sources: list[str] = []
    for dataset in datasets:
        if dataset.get("schema_version") != TEXT_DATASET_SCHEMA_VERSION:
            raise PooledDatasetError(
                f"only {TEXT_DATASET_SCHEMA_VERSION} datasets can be pooled, got {dataset.get('schema_version')!r}"
            )
        if dataset.get("activations") not in ({}, [], None):
            raise PooledDatasetError("pooled datasets carry no activation arrays")
        sources.append(source_id_for(dataset))
    duplicates = sorted({item for item in sources if sources.count(item) > 1})
    if duplicates:
        raise PooledDatasetError(f"source run ids must be distinct, duplicated: {duplicates}")
    if SOURCE_SEPARATOR in "".join(sources):
        raise PooledDatasetError(f"source run ids must not contain {SOURCE_SEPARATOR!r}")

    first_config = datasets[0]["config"]
    track = first_config.get("experiment_track")
    checksums = {str(dataset["config"].get("schema_registry_checksum")) for dataset in datasets}
    if len(checksums) > 1 and not allow_mixed_registry:
        raise PooledDatasetError(
            "datasets were written under different schema registries; pass allow_mixed_registry=True "
            "to pool them anyway (the pooled manifest records every checksum)"
        )
    for dataset in datasets:
        if dataset["config"].get("experiment_track") != track:
            raise PooledDatasetError("datasets from different experiment tracks are never pooled")

    metadata: list[dict[str, Any]] = []
    labels: dict[str, list[Any]] = {}
    generation: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    for source, dataset in zip(sources, datasets):
        rows = copy.deepcopy(list(dataset["metadata"]))
        dataset_labels = dataset["labels"]
        for key, values in dataset_labels.items():
            if len(values) != len(rows):
                raise PooledDatasetError(f"{source}: label column {key!r} is not aligned with its rows")
        if labels and set(labels) != set(dataset_labels):
            raise PooledDatasetError(f"{source}: label columns differ from the first dataset")
        model_name = str(dataset["config"].get("model_name"))
        dataset_hash = str(dataset["config"].get("dataset_hash"))
        for row in rows:
            _prefix_fields(row, source, _ROW_SCALAR_IDS, _ROW_LIST_IDS)
            row["source_run_id"] = source
            row["source_model"] = model_name
            row["source_dataset_hash"] = dataset_hash
        metadata.extend(rows)
        for key, values in dataset_labels.items():
            labels.setdefault(key, []).extend(copy.deepcopy(list(values)))
        for record in copy.deepcopy(list(dataset["generation_records"])):
            _prefix_fields(record, source, _GENERATION_IDS)
            generation.append(record)
        for record in copy.deepcopy(list(dataset["label_records"])):
            _prefix_fields(record, source, _LABEL_SCALAR_IDS, _LABEL_LIST_IDS)
            label_rows.append(record)
        for record in copy.deepcopy(list(dataset["interaction_events"])):
            _prefix_fields(record, source, _EVENT_IDS)
            events.append(record)
        manifest_sources.append({
            "run_id": source,
            "model_name": model_name,
            "dataset_hash": dataset_hash,
            "n_rows": len(rows),
            "schema_registry_checksum": dataset["config"].get("schema_registry_checksum"),
        })

    generation_ids = {record.get("call_id") for record in generation}
    label_ids = {record.get("label_id") for record in label_rows}
    seen: set[tuple[Any, Any]] = set()
    for row in metadata:
        key = (row.get("trial_id"), row.get("generation_record_id"))
        if key in seen:
            raise PooledDatasetError(f"pooled rows are not unique by (trial_id, generation_record_id): {key}")
        seen.add(key)
        if row.get("generation_record_id") is not None and row["generation_record_id"] not in generation_ids:
            raise PooledDatasetError(f"row references a missing generation record: {row['generation_record_id']}")
        missing = set(row.get("label_record_ids") or ()) - label_ids
        if missing:
            raise PooledDatasetError(f"row references missing label records: {sorted(missing)}")

    pooled: dict[str, Any] = {
        "schema_version": TEXT_DATASET_SCHEMA_VERSION,
        "config": {
            "model_name": "pooled",
            "provider": "pooled",
            "experiment_track": track,
            "n_rows": len(metadata),
            "schema_registry": copy.deepcopy(first_config.get("schema_registry")),
            "schema_registry_checksum": first_config.get("schema_registry_checksum"),
            "run_manifest": {
                "pooled": {
                    "version": POOLED_DATASET_VERSION,
                    "sources": manifest_sources,
                    "schema_registry_checksums": sorted(checksums),
                    "note": (
                        "Derived audit dataset. Ids are prefixed '<run_id>::'; trial families are "
                        "shared across sources. Not a run, not replayable, never a leaderboard entry."
                    ),
                },
            },
        },
        "labels": labels,
        "metadata": metadata,
        "activations": {},
        "generation_records": generation,
        "label_records": label_rows,
        "interaction_events": events,
    }
    pooled["config"]["dataset_hash"] = _content_hash(pooled)
    return pooled


def is_pooled_dataset(dataset: Mapping[str, Any]) -> bool:
    manifest = (dataset.get("config") or {}).get("run_manifest") or {}
    return isinstance(manifest, Mapping) and "pooled" in manifest


__all__ = [
    "POOLED_DATASET_VERSION",
    "SOURCE_SEPARATOR",
    "PooledDatasetError",
    "is_pooled_dataset",
    "pool_text_datasets",
    "prefix_id",
    "source_id_for",
    "split_source_id",
]
