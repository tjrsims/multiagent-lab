"""Bedrock model catalog: which ids this account can invoke, and how.

Pure functions over a boto3 ``bedrock`` client (or any object with the same
three methods) so the listing is testable offline. Nothing here invokes a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from end_bench.models.pricing import resolve_price

__all__ = ["CatalogRow", "list_bedrock_models", "format_catalog", "TARGET_PROVIDERS"]

TARGET_PROVIDERS = ("Anthropic", "Meta", "Mistral AI", "Amazon")


@dataclass(frozen=True)
class CatalogRow:
    model_id: str
    provider: str
    inference_types: tuple[str, ...]
    inference_profiles: tuple[str, ...]
    priced: bool
    access: str  # 'granted' | 'not granted' | 'pending' | 'unknown'

    @property
    def invoke_id(self) -> str:
        """The id to pass as ``bedrock:<id>``: a profile when the model needs one."""
        if "ON_DEMAND" in self.inference_types or not self.inference_profiles:
            return self.model_id
        return self.inference_profiles[0]


def _access_status(client: Any, model_id: str) -> str:
    getter = getattr(client, "get_foundation_model_availability", None)
    if getter is None:
        return "unknown"
    try:
        info = getter(modelId=model_id)
    except Exception:  # noqa: BLE001 - permission or API shape differences
        return "unknown"
    agreement = str((info.get("agreementAvailability") or {}).get("status", "")).upper()
    authorized = str(info.get("authorizationStatus", "")).upper()
    entitled = str(info.get("entitlementAvailability", "")).upper()
    if agreement == "PENDING":
        return "pending"
    if authorized == "AUTHORIZED" and entitled == "AVAILABLE":
        return "granted"
    if authorized or entitled or agreement:
        return "not granted"
    return "unknown"


def list_bedrock_models(
    client: Any,
    *,
    provider: str | None = None,
    check_access: bool = True,
) -> list[CatalogRow]:
    kwargs: dict[str, Any] = {}
    if provider:
        kwargs["byProvider"] = provider
    summaries = client.list_foundation_models(**kwargs).get("modelSummaries", [])

    profiles: dict[str, set[str]] = {}
    try:
        for page in client.get_paginator("list_inference_profiles").paginate():
            for item in page.get("inferenceProfileSummaries", []):
                for ref in item.get("models", []):
                    base = str(ref.get("modelArn", "")).rsplit("/", 1)[-1]
                    profiles.setdefault(base, set()).add(str(item.get("inferenceProfileId")))
    except Exception:  # noqa: BLE001 - optional listing
        pass

    rows: list[CatalogRow] = []
    for item in summaries:
        if "TEXT" not in item.get("outputModalities", []):
            continue
        serving = set(item.get("inferenceTypesSupported", []))
        if serving and not serving & {"ON_DEMAND", "INFERENCE_PROFILE"}:
            continue  # provisioned-throughput-only variants are not pay-per-token
        model_id = str(item.get("modelId", ""))
        rows.append(
            CatalogRow(
                model_id=model_id,
                provider=str(item.get("providerName", "")),
                inference_types=tuple(sorted(item.get("inferenceTypesSupported", []))),
                inference_profiles=tuple(sorted(profiles.get(model_id, ()))),
                priced=resolve_price(model_id) is not None,
                access=_access_status(client, model_id) if check_access else "unknown",
            )
        )
    return sorted(rows, key=lambda r: (r.provider, r.model_id))


def format_catalog(rows: Iterable[CatalogRow]) -> str:
    lines = [f"{'invoke with bedrock:<id>':58s} {'provider':11s} {'access':12s} {'priced':6s} serving"]
    for row in rows:
        serving = "on-demand" if "ON_DEMAND" in row.inference_types else (
            "inference-profile" if row.inference_profiles else ",".join(row.inference_types).lower() or "-")
        lines.append(f"{row.invoke_id:58s} {row.provider[:11]:11s} {row.access:12s} {'yes' if row.priced else 'no':6s} {serving}")
    return "\n".join(lines)
