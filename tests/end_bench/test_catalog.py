"""Offline coverage for the Bedrock catalog listing (no network)."""
from __future__ import annotations

from end_bench.models.catalog import format_catalog, list_bedrock_models


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self):
        return iter(self._pages)


class FakeBedrockClient:
    def __init__(self, *, availability=None, profiles=True):
        self.availability = availability or {}
        self.profiles = profiles
        self.calls = []

    def list_foundation_models(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"modelSummaries": [
            {"modelId": "anthropic.claude-sonnet-4-6", "providerName": "Anthropic",
             "outputModalities": ["TEXT"], "inferenceTypesSupported": ["INFERENCE_PROFILE"]},
            {"modelId": "mistral.mistral-large-3-675b-instruct", "providerName": "Mistral AI",
             "outputModalities": ["TEXT"], "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "amazon.titan-image-generator-v1", "providerName": "Amazon",
             "outputModalities": ["IMAGE"], "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "amazon.nova-pro-v1:0:300k", "providerName": "Amazon",
             "outputModalities": ["TEXT"], "inferenceTypesSupported": ["PROVISIONED"]},
        ]}

    def get_paginator(self, name):
        assert name == "list_inference_profiles"
        if not self.profiles:
            raise RuntimeError("no permission")
        return _Paginator([{"inferenceProfileSummaries": [
            {"inferenceProfileId": "us.anthropic.claude-sonnet-4-6", "models": [
                {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6"},
                {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-6"},
            ]},
        ]}])

    def get_foundation_model_availability(self, modelId):
        return self.availability[modelId]


def test_rows_use_inference_profile_when_model_is_not_on_demand() -> None:
    client = FakeBedrockClient(availability={
        "anthropic.claude-sonnet-4-6": {"agreementAvailability": {"status": "AVAILABLE"},
                                        "authorizationStatus": "AUTHORIZED", "entitlementAvailability": "AVAILABLE"},
        "mistral.mistral-large-3-675b-instruct": {"agreementAvailability": {"status": "PENDING"},
                                                  "authorizationStatus": "NOT_AUTHORIZED",
                                                  "entitlementAvailability": "NOT_AVAILABLE"},
    })
    rows = list_bedrock_models(client)
    assert [r.model_id for r in rows] == ["anthropic.claude-sonnet-4-6", "mistral.mistral-large-3-675b-instruct"]
    claude, mistral = rows
    assert claude.invoke_id == "us.anthropic.claude-sonnet-4-6"          # deduplicated, profile required
    assert claude.access == "granted" and claude.priced
    assert mistral.invoke_id == "mistral.mistral-large-3-675b-instruct"  # on-demand keeps the base id
    assert mistral.access == "pending"
    text = format_catalog(rows)
    assert "us.anthropic.claude-sonnet-4-6" in text and "inference-profile" in text and "pending" in text


def test_listing_degrades_without_profile_or_access_permissions() -> None:
    client = FakeBedrockClient(profiles=False)
    rows = list_bedrock_models(client, check_access=False)
    assert all(r.access == "unknown" for r in rows)
    assert rows[0].invoke_id == "anthropic.claude-sonnet-4-6"  # no profile known -> base id
    assert client.calls == [("list", {})]
    list_bedrock_models(client, provider="Anthropic", check_access=False)
    assert client.calls[-1] == ("list", {"byProvider": "Anthropic"})
