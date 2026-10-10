"""End-to-end benchmark build: dataset -> views -> split -> pairs -> scores -> manifest.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
``build_benchmark`` is the one function the CLI calls. It never loads a
language model; the only "detector" it trains is a linear probe on stored
activations, and that probe's layer is chosen on the development partition
only. The locked test partition is scored once per build.
1. Load the saved dataset (JSON+NPZ; legacy ``.pt`` only with explicit trust).
2. ``build_views`` with the requested label source and policies (first pass,
   no split) -> ``build_split`` -> ``build_views`` again with the locked
   manifest so every row carries its partition.
3. ``build_matched_pairs`` on claim-bearing rows (twins first).
4. Score the reference baselines and the probe on ``all`` and ``paired``
   subsets for the test partition; compute the probe-vs-oracle and
   probe-vs-blind gaps; behavioral cells by scenario x incentive.
5. Write ``views_report.json``, ``split_manifest.json``, ``pairs.json``,
   ``scores.json``, and ``manifest.json`` (content-addressed) to ``out_dir``.
Where this fits: the benchmark stage's entry point; everything upstream is
read-only here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from interpretability.benchmark.baselines import (
    fact_informed_rule_scores,
    random_scores,
    response_length_scores,
    stated_claim_threshold,
)
from interpretability.benchmark.manifest import BenchmarkManifest, dataset_hash_for
from interpretability.benchmark.pairs import (
    MatchedPairs,
    build_matched_pairs,
    pair_scopes_for_views,
    twin_ids_for_views,
)
from interpretability.benchmark.scoring import (
    BehavioralCell,
    DetectorScore,
    PairedGap,
    ScoreCard,
    ScoringError,
    behavioral_scores,
    paired_gap,
    score_detector,
)
from interpretability.benchmark.splits import build_split
from interpretability.benchmark.views import BenchmarkViews, build_views, attach_partitions
from interpretability.data.activation_dataset import load_activation_dataset
from interpretability.dyads.evaluation import roc_auc


BUILD_VERSION = "benchmark-build/1"


class BenchmarkBuildError(ValueError):
    """Raised when a build cannot proceed without guessing."""


@dataclass(frozen=True)
class ProbeSelection:
    """Layer chosen on development AUROC; test scored once."""

    layer: str
    development_auroc_by_layer: Mapping[str, float]
    n_train: int
    n_development: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "development_auroc_by_layer": dict(self.development_auroc_by_layer),
            "n_train": self.n_train,
            "n_development": self.n_development,
        }


def _fit_probe(train_x: np.ndarray, train_y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    )
    model.fit(train_x, train_y)
    return model


def train_probe_with_dev_selection(
    views: BenchmarkViews,
    *,
    layers: Sequence[Any] | None = None,
) -> tuple[np.ndarray, ProbeSelection]:
    """Train per layer on ``train``, pick the layer on ``development``, score all rows.

    Returns scores for every row (so ``test`` can be masked downstream) and the
    selection record. Rows of the test partition never influence the choice.
    """
    if not views.partitions:
        raise BenchmarkBuildError("probe training requires views built with a split")
    train = views.partition_mask("train")
    development = views.partition_mask("development")
    if train.sum() < 4 or development.sum() < 2:
        raise BenchmarkBuildError("too few train/development rows for probe selection")
    if len(set(views.labels[train].tolist())) < 2:
        raise BenchmarkBuildError("train partition lacks both classes")
    if len(set(views.labels[development].tolist())) < 2:
        raise BenchmarkBuildError("development partition lacks both classes")
    candidates = list(views.layers) if layers is None else list(layers)
    if not candidates:
        raise BenchmarkBuildError("no activation layers available for the probe")
    dev_auroc: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    for layer in candidates:
        matrix = views.white_box(layer)
        if matrix.ndim != 2:
            matrix = matrix.reshape(matrix.shape[0], -1)
        model = _fit_probe(matrix[train], views.labels[train])
        probabilities = model.predict_proba(matrix[development])[:, 1]
        dev_auroc[str(layer)] = roc_auc(
            probabilities.tolist(), views.labels[development].astype(bool).tolist()
        )
        fitted[str(layer)] = (layer, model)
    best = max(sorted(dev_auroc), key=lambda key: dev_auroc[key])
    layer, model = fitted[best]
    matrix = views.white_box(layer)
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    scores = model.predict_proba(matrix)[:, 1]
    return np.asarray(scores, dtype=float), ProbeSelection(
        layer=best,
        development_auroc_by_layer=dev_auroc,
        n_train=int(train.sum()),
        n_development=int(development.sum()),
    )


def _pair_mask(views: BenchmarkViews, pairs: MatchedPairs) -> np.ndarray:
    mask = np.zeros(views.n, dtype=bool)
    mask[pairs.paired_row_indices()] = True
    return mask


def _try_score(
    name: str,
    scores: np.ndarray,
    views: BenchmarkViews,
    *,
    subset: str,
    mask: np.ndarray | None,
    partition: str | None,
    n_bootstrap: int,
    seed: int,
    skipped: list[dict[str, str]],
) -> DetectorScore | None:
    try:
        return score_detector(
            name, scores, views, subset=subset, mask=mask, partition=partition,
            n_bootstrap=n_bootstrap, seed=seed,
        )
    except (ScoringError, ValueError) as error:
        skipped.append({"detector": name, "subset": subset, "partition": partition or "all",
                        "reason": str(error)})
        return None


@dataclass(frozen=True)
class BenchmarkBuild:
    """Everything one build produced, in memory."""

    views: BenchmarkViews
    pairs: MatchedPairs
    manifest: BenchmarkManifest
    scorecard: ScoreCard
    probe_selection: ProbeSelection | None
    split_manifest: Any
    written: Mapping[str, Path]


def build_benchmark(
    dataset_path: str | Path | None,
    out_dir: str | Path,
    *,
    dataset: Mapping[str, Any] | None = None,
    label_source: str = "rule",
    rows: str = "claim_bearing",
    qc: str = "exclude",
    group_key: str = "trial_family_id",
    allow_legacy_runtime: bool = False,
    bin_width: float = 5.0,
    split_seed: int = 42,
    pair_seed: int = 0,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
    scoring_partition: str | None = "test",
    train_probe: bool = True,
    trusted_legacy: bool = False,
) -> BenchmarkBuild:
    """Run the full offline benchmark build and write its artifacts.

    ``dataset`` may be a preloaded mapping (for example an END-Bench text
    dataset, which has no activation arrays); otherwise ``dataset_path`` is
    loaded as a safe activation dataset.
    """
    if dataset is None:
        if dataset_path is None:
            raise BenchmarkBuildError("either dataset_path or dataset is required")
        dataset = load_activation_dataset(dataset_path, trusted_legacy=trusted_legacy)
    allowed = ("transactional", "legacy_compatibility") if allow_legacy_runtime else ("transactional",)
    common = dict(
        label_source=label_source, rows=rows, qc=qc, group_key=group_key,
        allowed_runtime_paths=allowed,
    )
    views = build_views(dataset, **common)
    split_manifest, stratification, audit = build_split(views, seed=split_seed)
    views = attach_partitions(views, split_manifest)
    populated_partitions = {a.partition for a in split_manifest.assignments}
    split_complete = split_manifest.locked and populated_partitions == {"train", "development", "test"}
    if scoring_partition is not None and (
        not split_complete or scoring_partition not in populated_partitions
    ):
        # Too few families for a held-out partition: score every row and say
        # so in the scorecard rather than silently scoring an empty subset.
        scoring_partition = None
        train_probe = False
    pairs = build_matched_pairs(
        views.labels, views.claim_values, views.scenarios, views.group_ids,
        bin_width=bin_width, seed=pair_seed, twin_ids=twin_ids_for_views(views),
        scopes=pair_scopes_for_views(views),
    )
    pair_mask = _pair_mask(views, pairs)

    detectors: dict[str, np.ndarray] = {
        "random": random_scores(views.n, seed=bootstrap_seed),
        "response_length": response_length_scores(views.responses),
        "stated_claim_oracle": stated_claim_threshold(views.claim_values),
        "fact_informed_rule": fact_informed_rule_scores(
            views.scenarios, views.responses, views.private_facts,
            tolerances=views.tolerances or None,
            dialogue_history=views.dialogue_history or None,
        ),
    }
    probe_selection: ProbeSelection | None = None
    skipped: list[dict[str, str]] = []
    if train_probe:
        try:
            probe_scores, probe_selection = train_probe_with_dev_selection(views)
            detectors["linear_probe"] = probe_scores
        except BenchmarkBuildError as error:
            skipped.append({"detector": "linear_probe", "subset": "all",
                            "partition": scoring_partition or "all", "reason": str(error)})

    scores: list[DetectorScore] = []
    for name, values in detectors.items():
        for subset, mask in (("all", None), ("paired", pair_mask)):
            result = _try_score(
                name, values, views, subset=subset, mask=mask,
                partition=scoring_partition, n_bootstrap=n_bootstrap,
                seed=bootstrap_seed, skipped=skipped,
            )
            if result is not None:
                scores.append(result)
    gaps: list[PairedGap] = []
    if "linear_probe" in detectors:
        for other in ("stated_claim_oracle", "response_length"):
            for subset, mask in (("all", None), ("paired", pair_mask)):
                try:
                    gaps.append(paired_gap(
                        "linear_probe", detectors["linear_probe"], other, detectors[other],
                        views, subset=subset, mask=mask, partition=scoring_partition,
                        n_bootstrap=n_bootstrap, seed=bootstrap_seed,
                    ))
                except (ScoringError, ValueError) as error:
                    skipped.append({"detector": f"linear_probe-{other}", "subset": subset,
                                    "partition": scoring_partition or "all", "reason": str(error)})
    behavioral: tuple[BehavioralCell, ...] = behavioral_scores(
        views, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )

    manifest = BenchmarkManifest(
        dataset_hash=dataset_hash_for(dataset),
        label_source=label_source,
        views_report=views.report.to_dict(),
        split_manifest_id=split_manifest.manifest_id,
        pairs_report=pairs.balance_report(),
    )
    scorecard = ScoreCard(
        detector_scores=tuple(scores),
        paired_gaps=tuple(gaps),
        behavioral=behavioral,
        views_report=views.report.to_dict(),
        pairs_report=pairs.balance_report(),
        benchmark_manifest_id=manifest.manifest_id,
        extras={
            "build_version": BUILD_VERSION,
            "scoring_partition": scoring_partition,
            "split_complete": split_complete,
            "split_locked": split_manifest.locked,
            "probe_selection": probe_selection.to_dict() if probe_selection else None,
            "skipped": skipped,
            "stratification_unbalanced": list(stratification.unbalanced_strata),
            "stratification_tolerance": stratification.tolerance,
            "leakage_violations": len(audit.violations),
        },
    )

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written = {
        "views_report": target / "views_report.json",
        "split_manifest": target / "split_manifest.json",
        "pairs": target / "pairs.json",
        "scores": target / "scores.json",
        "manifest": target / "manifest.json",
    }
    written["views_report"].write_text(json.dumps(views.report.to_dict(), indent=2, sort_keys=True))
    written["split_manifest"].write_text(json.dumps(split_manifest.to_dict(), indent=2, sort_keys=True))
    written["pairs"].write_text(json.dumps(pairs.balance_report(), indent=2, sort_keys=True))
    written["scores"].write_text(json.dumps(scorecard.to_dict(), indent=2, sort_keys=True))
    manifest.write_json(written["manifest"])
    return BenchmarkBuild(
        views=views,
        pairs=pairs,
        manifest=manifest,
        scorecard=scorecard,
        probe_selection=probe_selection,
        split_manifest=split_manifest,
        written=written,
    )


__all__ = [
    "BUILD_VERSION",
    "BenchmarkBuild",
    "BenchmarkBuildError",
    "ProbeSelection",
    "build_benchmark",
    "train_probe_with_dev_selection",
]
