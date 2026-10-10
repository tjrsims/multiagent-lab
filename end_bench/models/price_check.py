"""Verify the price table against the AWS Price List API (pure over a client).

``fetch_pricelist`` walks ``get_products`` for ``AmazonBedrock`` in one region
and keeps standard on-demand token prices; ``verify_price_table`` compares them
with ``DEFAULT_PRICES_PER_MTOK`` for the ids we know how to name. Anthropic
models are not in the Price List and are reported as ``not_in_pricelist``.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from end_bench.models.pricing import DEFAULT_PRICES_PER_MTOK, PRICE_SOURCES

__all__ = ["PRICELIST_MODEL_NAMES", "fetch_pricelist", "verify_price_table"]

# table prefix -> ``model`` attribute as the Price List spells it
PRICELIST_MODEL_NAMES: Mapping[str, str] = {
    "meta.llama4-maverick": "Llama 4 Maverick 17B",
    "meta.llama4-scout": "Llama 4 Scout 17B",
    "mistral.mistral-large-3": "Mistral Large 3",
    "amazon.nova-pro": "Nova Pro",
    "amazon.nova-lite": "Nova Lite",
    "amazon.nova-micro": "Nova Micro",
}
_SKIP = ("batch", "cache", "flex", "priority", "latency")


def fetch_pricelist(client: Any, region: str = "us-east-1") -> dict[tuple[str, str], float]:
    """``{(model name, 'input'|'output'): usd per million tokens}`` for standard on-demand."""
    prices: dict[tuple[str, str], float] = {}
    paginator = client.get_paginator("get_products")
    filters = [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}]
    for page in paginator.paginate(ServiceCode="AmazonBedrock", Filters=filters):
        for raw in page.get("PriceList", []):
            product = json.loads(raw) if isinstance(raw, str) else raw
            attrs = product.get("product", {}).get("attributes", {})
            model = attrs.get("model") or attrs.get("titanModel")
            inference = str(attrs.get("inferenceType", "")).lower()
            if not model or any(token in inference for token in _SKIP):
                continue
            if inference.startswith("input tokens"):
                direction = "input"
            elif inference.startswith("output tokens"):
                direction = "output"
            else:
                continue
            for term in product.get("terms", {}).get("OnDemand", {}).values():
                for dim in term.get("priceDimensions", {}).values():
                    usd = dim.get("pricePerUnit", {}).get("USD")
                    unit = str(dim.get("unit", "")).lower()
                    if usd is None or float(usd) <= 0:
                        continue
                    per_mtok = float(usd) * (1000.0 if "1k" in unit else 1.0)
                    prices.setdefault((str(model), direction), per_mtok)
    return prices


def verify_price_table(
    fetched: Mapping[tuple[str, str], float],
    table: Mapping[str, tuple[float, float]] | None = None,
    *,
    rel_tol: float = 0.005,
) -> list[dict[str, Any]]:
    """One row per table entry: verified, mismatch, or not_in_pricelist."""
    prices = DEFAULT_PRICES_PER_MTOK if table is None else table
    rows: list[dict[str, Any]] = []
    for prefix, (table_in, table_out) in prices.items():
        name = PRICELIST_MODEL_NAMES.get(prefix)
        api_in = fetched.get((name, "input")) if name else None
        api_out = fetched.get((name, "output")) if name else None
        if api_in is None or api_out is None:
            status = "offline" if prefix == "scripted" else "not_in_pricelist"
        else:
            close = (
                abs(api_in - table_in) <= rel_tol * max(table_in, 1e-9)
                and abs(api_out - table_out) <= rel_tol * max(table_out, 1e-9)
            )
            status = "verified" if close else "mismatch"
        rows.append({
            "model_id_prefix": prefix,
            "status": status,
            "table_usd_per_mtok": [table_in, table_out],
            "pricelist_usd_per_mtok": None if api_in is None else [api_in, api_out],
            "source": PRICE_SOURCES.get(prefix),
        })
    return rows
