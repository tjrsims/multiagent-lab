"""Offline tests for price provenance and Price List verification."""
from __future__ import annotations

import json

from end_bench.models.price_check import fetch_pricelist, verify_price_table
from end_bench.models.pricing import (
    CostLedger,
    DEFAULT_PRICES_PER_MTOK,
    PRICE_SOURCES,
    estimate_design_cost,
    price_is_verified,
    price_source,
)


def _product(model, inference, usd_per_1k, feature="On-demand Inference"):
    return json.dumps({
        "product": {"attributes": {"model": model, "inferenceType": inference, "feature": feature,
                                   "regionCode": "us-east-1"}},
        "terms": {"OnDemand": {"x": {"priceDimensions": {"y": {"unit": "1K tokens",
                                                                "pricePerUnit": {"USD": str(usd_per_1k)}}}}}},
    })


class _Paginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **kwargs):
        assert kwargs["ServiceCode"] == "AmazonBedrock"
        return iter(self.pages)


class FakePricingClient:
    def __init__(self, pages):
        self.pages = pages

    def get_paginator(self, name):
        assert name == "get_products"
        return _Paginator(self.pages)


def test_every_table_entry_has_a_recorded_source() -> None:
    assert set(DEFAULT_PRICES_PER_MTOK) == set(PRICE_SOURCES)
    assert price_source("bedrock:us.meta.llama4-maverick-17b-instruct-v1:0").startswith("verified")
    assert price_source("global.anthropic.claude-sonnet-4-6").startswith("unverified")
    assert price_is_verified("meta.llama4-maverick-17b-instruct-v1:0")
    assert not price_is_verified("anthropic.claude-opus-4-8")
    assert price_source("unknown.model") is None
    assert estimate_design_cost("anthropic.claude-sonnet-4-6", negotiations=10)["price_source"].startswith("unverified")


def test_ledger_tracks_dollars_resting_on_unverified_prices(tmp_path) -> None:
    ledger = CostLedger(path=tmp_path / "ledger.jsonl")
    ledger.record(model_id="meta.llama4-maverick-17b-instruct-v1:0", input_tokens=1_000_000, output_tokens=0,
                  latency_ms=1.0, retries=0, call_id="a", stop_reason="end_turn")
    entry = ledger.record(model_id="global.anthropic.claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=0,
                          latency_ms=1.0, retries=0, call_id="b", stop_reason="end_turn")
    assert entry["price_source"].startswith("unverified")
    totals = ledger.totals()
    assert totals["usd"] == 0.24 + 3.0
    assert totals["unverified_usd"] == 3.0


def test_pricelist_fetch_and_verification() -> None:
    pages = [{"PriceList": [
        _product("Llama 4 Maverick 17B", "Input tokens", 0.00024),
        _product("Llama 4 Maverick 17B", "Output tokens", 0.00097),
        _product("Llama 4 Maverick 17B", "Input tokens batch", 0.00012),   # skipped
        _product("Nova Pro", "Input tokens", 0.0008),
        _product("Nova Pro", "Output tokens", 0.0099),                     # deliberately wrong
        _product("Nova Pro", "Input tokens priority", 0.0014),            # skipped
    ]}]
    fetched = fetch_pricelist(FakePricingClient(pages))
    assert abs(fetched[("Llama 4 Maverick 17B", "input")] - 0.24) < 1e-9
    assert ("Nova Pro", "input") in fetched and len(fetched) == 4
    rows = {row["model_id_prefix"]: row for row in verify_price_table(fetched)}
    assert rows["meta.llama4-maverick"]["status"] == "verified"
    assert rows["amazon.nova-pro"]["status"] == "mismatch"
    assert rows["anthropic.claude-sonnet-4-6"]["status"] == "not_in_pricelist"
    assert rows["scripted"]["status"] == "offline"
