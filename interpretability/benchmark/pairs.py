"""Matched pairs: same claim, opposite truth, balanced priors per claim bin.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
The v1 benchmark's defining contrast needs turns where the *same* kind of claim
is a false claim in one instance and honest in another, because only the
private fact differs. On such pairs a transcript-only detector has no
information to use. This module builds that subset from already-labeled rows.
1. ``build_matched_pairs`` groups claim-bearing rows by ``(scenario, bin)``
   where ``bin`` is the claimed amount quantized by ``bin_width``. Within a bin
   it pairs TRUE rows with FALSE rows one-to-one, never pairing two rows from
   the same leakage group, and prefers generator twins (rows sharing a
   ``twin_id``) when they are available.
2. The result is a ``MatchedPairs`` record: index arrays into the input rows,
   the per-bin balance report (how many of each class were available and how
   many were dropped to balance), and the construction parameters, so the
   subset is reproducible and auditable.
3. ``paired_row_indices`` gives the flat row subset for scoring "paired" vs
   "all" results side by side.
Where this fits: the benchmark stage; consumes the arrays produced by
``benchmark.views.build_views`` (labels, claim values, scenarios, group ids)
and feeds ``benchmark.scoring``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np


PAIR_BUILDER_VERSION = "matched-pairs/1"


class MatchedPairError(ValueError):
    """Raised when pair construction inputs are inconsistent."""


@dataclass(frozen=True)
class BinBalance:
    """Availability and usage counts for one ``(scenario, bin)`` cell."""

    scenario: str
    bin_index: int
    bin_low: float
    bin_high: float
    positives_available: int
    negatives_available: int
    pairs_formed: int
    twin_pairs: int
    # Pairing scope (the source run in a pooled dataset); None for one run.
    scope: str | None = None

    @property
    def dropped(self) -> int:
        return (
            self.positives_available + self.negatives_available
            - 2 * self.pairs_formed
        )


@dataclass(frozen=True)
class MatchedPairs:
    """Index-aligned matched pairs over the caller's rows."""

    pair_ids: tuple[str, ...]
    positive_indices: np.ndarray  # (P,) row index of the TRUE member
    negative_indices: np.ndarray  # (P,) row index of the FALSE member
    bins: tuple[BinBalance, ...]
    bin_width: float
    seed: int
    version: str = PAIR_BUILDER_VERSION
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (
            len(self.pair_ids) == len(self.positive_indices) == len(self.negative_indices)
        ):
            raise MatchedPairError("pair fields are not index-aligned")
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise MatchedPairError("pair_ids must be unique")

    @property
    def n_pairs(self) -> int:
        return len(self.pair_ids)

    def paired_row_indices(self) -> np.ndarray:
        """Sorted unique row indices covered by the pairs."""
        if self.n_pairs == 0:
            return np.array([], dtype=int)
        return np.unique(
            np.concatenate([self.positive_indices, self.negative_indices])
        )

    def balance_report(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "bin_width": self.bin_width,
            "seed": self.seed,
            "n_pairs": self.n_pairs,
            "n_twin_pairs": int(sum(item.twin_pairs for item in self.bins)),
            "bins": [
                {
                    "scenario": item.scenario,
                    "bin_index": item.bin_index,
                    "bin_low": item.bin_low,
                    "bin_high": item.bin_high,
                    "positives_available": item.positives_available,
                    "negatives_available": item.negatives_available,
                    "pairs_formed": item.pairs_formed,
                    "twin_pairs": item.twin_pairs,
                    "dropped": item.dropped,
                    **({"scope": item.scope} if item.scope is not None else {}),
                }
                for item in self.bins
            ],
            "parameters": dict(self.parameters),
        }


def _row_claim(values: Sequence[float] | None) -> float | None:
    if values is None:
        return None
    numbers = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not numbers:
        return None
    # The operative claim is the largest stated amount (the overstatement).
    return max(numbers)


def build_matched_pairs(
    labels: Sequence[int],
    claim_values: Sequence[Sequence[float] | None],
    scenarios: Sequence[str],
    group_ids: Sequence[str],
    *,
    bin_width: float = 5.0,
    seed: int = 0,
    twin_ids: Sequence[str | None] | None = None,
    scopes: Sequence[str | None] | None = None,
) -> MatchedPairs:
    """Pair TRUE and FALSE claim-bearing rows within claim bins.

    Rows without a claim value are skipped (they are not claim-bearing). Two
    rows from the same ``group_id`` are never paired, so a pair always spans
    distinct leakage groups. When ``twin_ids`` is given, rows sharing a twin id
    with opposite labels are paired first. When ``scopes`` is given (the source
    run of each row in a pooled dataset), a pair never crosses a scope: a
    model's turn is only ever matched against another turn of the same model.
    """
    n = len(labels)
    if not (len(claim_values) == len(scenarios) == len(group_ids) == n):
        raise MatchedPairError("inputs must be index-aligned")
    if twin_ids is not None and len(twin_ids) != n:
        raise MatchedPairError("twin_ids must align with rows")
    if scopes is not None and len(scopes) != n:
        raise MatchedPairError("scopes must align with rows")
    if not (isinstance(bin_width, (int, float)) and math.isfinite(bin_width)
            and bin_width > 0):
        raise MatchedPairError("bin_width must be a positive finite number")
    if type(seed) is not int:
        raise MatchedPairError("seed must be an integer")

    cells: dict[tuple[str, int, str | None], dict[str, list[int]]] = {}
    for index in range(n):
        label = int(labels[index])
        if label not in (0, 1):
            raise MatchedPairError("labels must be 0/1")
        claim = _row_claim(claim_values[index])
        if claim is None:
            continue
        bin_index = int(math.floor(claim / bin_width))
        scope = None
        if scopes is not None:
            scope = str(scopes[index]) if scopes[index] is not None else ""
        cell = cells.setdefault(
            (str(scenarios[index]), bin_index, scope), {"pos": [], "neg": []}
        )
        cell["pos" if label == 1 else "neg"].append(index)

    rng = random.Random(seed)
    pair_ids: list[str] = []
    positives: list[int] = []
    negatives: list[int] = []
    balances: list[BinBalance] = []
    for (scenario, bin_index, scope) in sorted(cells, key=lambda key: (key[0], key[1], key[2] or "")):
        cell = cells[(scenario, bin_index, scope)]
        pos = sorted(cell["pos"], key=lambda i: (str(group_ids[i]), i))
        neg = sorted(cell["neg"], key=lambda i: (str(group_ids[i]), i))
        formed = 0
        twin_formed = 0
        used_pos: set[int] = set()
        used_neg: set[int] = set()

        if twin_ids is not None:
            by_twin: dict[str, list[int]] = {}
            for i in neg:
                twin = twin_ids[i]
                if twin is not None:
                    by_twin.setdefault(str(twin), []).append(i)
            for p in pos:
                twin = twin_ids[p]
                if twin is None:
                    continue
                # Twins intentionally share a leakage group (one family, one
                # anchor, opposite truth); the only constraint is that the
                # negative is still unused. The distinct-group rule applies to
                # ordinary pairs only.
                candidates = [
                    i for i in by_twin.get(str(twin), ()) if i not in used_neg
                ]
                if not candidates:
                    continue
                q = candidates[0]
                used_pos.add(p)
                used_neg.add(q)
                positives.append(p)
                negatives.append(q)
                pair_ids.append(f"pair:{scenario}:{bin_index}:{p}:{q}")
                formed += 1
                twin_formed += 1

        remaining_pos = [i for i in pos if i not in used_pos]
        remaining_neg = [i for i in neg if i not in used_neg]
        rng.shuffle(remaining_pos)
        rng.shuffle(remaining_neg)
        for p in remaining_pos:
            match = next(
                (
                    i for i in remaining_neg
                    if i not in used_neg and str(group_ids[i]) != str(group_ids[p])
                ),
                None,
            )
            if match is None:
                continue
            used_pos.add(p)
            used_neg.add(match)
            positives.append(p)
            negatives.append(match)
            pair_ids.append(f"pair:{scenario}:{bin_index}:{p}:{match}")
            formed += 1
        balances.append(BinBalance(
            scenario=scenario,
            bin_index=bin_index,
            bin_low=bin_index * bin_width,
            bin_high=(bin_index + 1) * bin_width,
            positives_available=len(pos),
            negatives_available=len(neg),
            pairs_formed=formed,
            twin_pairs=twin_formed,
            scope=scope,
        ))

    parameters: dict[str, Any] = {"twin_ids_supplied": twin_ids is not None}
    if scopes is not None:
        parameters["scoped"] = True
    return MatchedPairs(
        pair_ids=tuple(pair_ids),
        positive_indices=np.asarray(positives, dtype=int),
        negative_indices=np.asarray(negatives, dtype=int),
        bins=tuple(balances),
        bin_width=float(bin_width),
        seed=seed,
        parameters=parameters,
    )


def twin_ids_for_views(views: Any) -> list[str | None]:
    """One twin id per view row: leakage group plus anchor, plus the source run when pooled.

    Twins are the two trials of one family that share a claim anchor and carry
    opposite truth. In a pooled dataset the same family appears once per source
    run, so the source is part of the id and a twin never crosses models.
    """
    sources = tuple(views.sources) if getattr(views, "sources", None) else (None,) * views.n
    twins: list[str | None] = []
    for group, role, anchor, source in zip(views.group_ids, views.pair_roles, views.pair_anchors, sources):
        if role is None:
            twins.append(None)
            continue
        twin = f"{group}|{anchor}"
        if source is not None:
            twin = f"{twin}|{source}"
        twins.append(twin)
    return twins


def pair_scopes_for_views(views: Any) -> tuple[str, ...] | None:
    """Pairing scopes for ``build_matched_pairs``: the source run per row, or None when not pooled."""
    sources = tuple(getattr(views, "sources", None) or ())
    if not sources or all(item is None for item in sources):
        return None
    return tuple(str(item) if item is not None else "" for item in sources)


__all__ = [
    "PAIR_BUILDER_VERSION",
    "BinBalance",
    "MatchedPairError",
    "MatchedPairs",
    "build_matched_pairs",
    "pair_scopes_for_views",
    "twin_ids_for_views",
]
