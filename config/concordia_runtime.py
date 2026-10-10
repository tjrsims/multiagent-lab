"""Pinned upstream Concordia runtime metadata and validation."""

from __future__ import annotations

from importlib import metadata


UPSTREAM_DISTRIBUTION = "gdm-concordia"
UPSTREAM_VERSION = "2.4.0"
UPSTREAM_TAG = "v2.4.0"
UPSTREAM_TAG_COMMIT = "702998f57da71f87bf4e607abc1325ee51cca21f"
# Upstream ``main`` audited against this repository's Concordia usage. On the
# audit date the latest PyPI release was still 2.4.0; main had moved on (act
# component refactor, ``LastNObservations`` retrieval rewrite, prefab acting
# injection, removal of ``ContextProcessorComponent``). Our Concordia-facing
# suites pass against that commit via PYTHONPATH (see
# ``scripts/check_concordia_upstream.sh``).
AUDITED_MAIN_COMMIT = "b71d202750e23444ce1bf28d9f114bcd88aa41db"
AUDITED_MAIN_DATE = "2026-10-01"
AUDIT_PERFORMED = "2026-10-05"
# Symbols upstream main removed that this repository must never import.
REMOVED_UPSTREAM_SYMBOLS = (
    "ContextProcessorComponent",
    "no_op_context_processor",
)


def installed_version() -> str:
    """Return the installed Concordia distribution version."""
    return metadata.version(UPSTREAM_DISTRIBUTION)


def require_supported_version() -> None:
    """Fail early when runtime code and the installed API can diverge."""
    version = installed_version()
    if version != UPSTREAM_VERSION:
        raise RuntimeError(
            f"Unsupported {UPSTREAM_DISTRIBUTION} version {version}; "
            f"expected {UPSTREAM_VERSION} ({UPSTREAM_TAG_COMMIT})."
        )
