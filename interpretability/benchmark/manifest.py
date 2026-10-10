"""Content-addressed benchmark manifest: what produced every reported number.

A ``BenchmarkManifest`` binds the benchmark version to the exact label rule,
extractor, QC, scenario-spec, and schema-registry versions, the dataset hash,
the views audit report, the split manifest, and the matched-pair
construction. Its ``manifest_id`` is a SHA-256 over the canonical JSON of
those fields, so two builds with the same inputs produce the same id and any
silent change in grading or data shows up as a different id. Scores carry the
id; results across different ids are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from interpretability.benchmark.pairs import PAIR_BUILDER_VERSION
from interpretability.benchmark.scoring import SCORING_VERSION
from interpretability.benchmark.views import VIEWS_VERSION
from interpretability.core.qc_filter import QC_VERSION
from interpretability.scenarios.compiled import EMERGENT_SCENARIO_SPEC_VERSION
from interpretability.scenarios.extractors import PARSER_VERSION
from interpretability.scenarios.turn_labels import (
    DEFAULT_TOLERANCES,
    TURN_LABEL_RULE_VERSION,
)


BENCHMARK_VERSION = "end_bench/0.1"
BENCHMARK_MANIFEST_VERSION = "benchmark-manifest/1"


class BenchmarkManifestError(ValueError):
    """Raised when a manifest is malformed or tampered."""


def _registry_checksum() -> str:
    # Imported lazily: the schema registry itself records this module's
    # versions, so a module-level import would be circular.
    from interpretability.schema_registry import schema_registry_checksum

    return schema_registry_checksum()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def dataset_hash_for(dataset: Mapping[str, Any]) -> str:
    """Prefer the saved dataset hash; otherwise hash labels + metadata."""
    config = dataset.get("config")
    if isinstance(config, Mapping) and isinstance(config.get("dataset_hash"), str):
        return str(config["dataset_hash"])
    digest = hashlib.sha256()
    digest.update(_canonical({"labels": dataset.get("labels", {})}))
    digest.update(_canonical({"metadata": list(dataset.get("metadata", []))}))
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class BenchmarkManifest:
    """Immutable, content-addressed description of one benchmark build."""

    dataset_hash: str
    label_source: str
    views_report: Mapping[str, Any]
    split_manifest_id: str | None = None
    pairs_report: Mapping[str, Any] | None = None
    tolerances: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_TOLERANCES.items()}
    )
    benchmark_version: str = BENCHMARK_VERSION
    manifest_version: str = BENCHMARK_MANIFEST_VERSION
    rule_version: str = TURN_LABEL_RULE_VERSION
    extractor_version: str = PARSER_VERSION
    qc_version: str = QC_VERSION
    scenario_spec_version: str = EMERGENT_SCENARIO_SPEC_VERSION
    views_version: str = VIEWS_VERSION
    pairs_version: str = PAIR_BUILDER_VERSION
    scoring_version: str = SCORING_VERSION
    schema_registry_checksum: str = field(default_factory=_registry_checksum)
    manifest_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_hash, str) or not self.dataset_hash:
            raise BenchmarkManifestError("dataset_hash is required")
        if not isinstance(self.label_source, str) or not self.label_source:
            raise BenchmarkManifestError("label_source is required")
        expected = self._compute_id()
        if self.manifest_id and self.manifest_id != expected:
            raise BenchmarkManifestError("manifest_id does not match manifest contents")
        object.__setattr__(self, "manifest_id", expected)

    def _content(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "manifest_version": self.manifest_version,
            "dataset_hash": self.dataset_hash,
            "label_source": self.label_source,
            "views_report": dict(self.views_report),
            "split_manifest_id": self.split_manifest_id,
            "pairs_report": dict(self.pairs_report) if self.pairs_report is not None else None,
            "tolerances": {k: dict(v) for k, v in self.tolerances.items()},
            "rule_version": self.rule_version,
            "extractor_version": self.extractor_version,
            "qc_version": self.qc_version,
            "scenario_spec_version": self.scenario_spec_version,
            "views_version": self.views_version,
            "pairs_version": self.pairs_version,
            "scoring_version": self.scoring_version,
            "schema_registry_checksum": self.schema_registry_checksum,
        }

    def _compute_id(self) -> str:
        return f"bench_{hashlib.sha256(_canonical(self._content())).hexdigest()[:32]}"

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "manifest_id": self.manifest_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkManifest":
        data = dict(payload)
        manifest_id = str(data.pop("manifest_id", ""))
        try:
            return cls(**data, manifest_id=manifest_id)
        except TypeError as error:
            raise BenchmarkManifestError(f"malformed manifest payload: {error}") from error

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "BenchmarkManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "BENCHMARK_MANIFEST_VERSION",
    "BENCHMARK_VERSION",
    "BenchmarkManifest",
    "BenchmarkManifestError",
    "dataset_hash_for",
]
