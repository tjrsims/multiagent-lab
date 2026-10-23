"""Provider registry and model roster: pure, offline, no keys read beyond presence."""

from __future__ import annotations

import json

import pytest

from end_bench.models.bedrock import BedrockTextModel
from end_bench.models.openai_compat import OpenAICompatTextModel
from end_bench.models.pricing import (
    DEFAULT_PRICES_PER_MTOK,
    PRICE_SOURCES,
    CostLedger,
    normalize_model_id,
    price_is_verified,
    price_source,
    resolve_price,
)
from end_bench.models.providers import (
    PROVIDERS,
    ModelSpecError,
    ProviderSpec,
    build_text_model,
    known_schemes,
    parse_model_spec,
)
from end_bench.models.roster import (
    MODEL_ROSTER,
    MODEL_ROSTER_VERSION,
    PEER_BENCHMARKS,
    SAMPLING_MODES,
    TIERS,
    RosterEntry,
    Route,
    format_roster,
    paper_roster,
    resolve_model_spec,
    roster_json,
    sampling_constraint,
    standard_roster,
)
from end_bench.models.scripted import ScriptedTextModel


def test_registry_rows_are_well_formed_and_parse() -> None:
    assert {"bedrock", "scripted", "openai", "xai", "gemini", "deepseek", "mistral", "moonshot", "qwen", "zai", "meta"} <= set(known_schemes())
    for provider in PROVIDERS.values():
        if provider.kind == "openai_compat":
            assert provider.base_url.startswith("https://") and provider.key_env.endswith("_API_KEY")
            assert provider.base_url_env == f"{provider.scheme.upper()}_BASE_URL"
    xai, model_id = parse_model_spec("xai:grok-4.3")
    assert xai.display == "xAI" and model_id == "grok-4.3"
    assert parse_model_spec("scripted")[0].kind == "scripted"
    with pytest.raises(ModelSpecError, match="unsupported model spec"):
        parse_model_spec("azure:gpt")
    with pytest.raises(ModelSpecError, match="no model id"):
        parse_model_spec("xai:")
    with pytest.raises(ModelSpecError):
        parse_model_spec("scripted:x")
    with pytest.raises(ValueError, match="base_url and key_env"):
        ProviderSpec("foo", "Foo", "openai_compat")


def test_factory_builds_each_kind_with_provider_defaults(monkeypatch) -> None:
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    model = build_text_model("xai:grok-4.3", max_tokens=160, temperature=0.7, cost_ledger=CostLedger(), honor_caller_sampling=False)
    assert isinstance(model, OpenAICompatTextModel)
    assert model.model_name == "xai:grok-4.3" and model.endpoint_host == "api.x.ai"
    assert model._api_key_env == "XAI_API_KEY" and model.honor_caller_sampling is False
    gemini = build_text_model("gemini:gemini-3.1-pro-preview", max_tokens=160, temperature=0.7)
    assert gemini.base_url.endswith("/v1beta/openai") and gemini.endpoint_host == "generativelanguage.googleapis.com"
    assert gemini.model_name == "gemini:gemini-3.1-pro-preview" and gemini._api_key_env == "GEMINI_API_KEY"
    qwen = build_text_model("qwen:qwen3.6-plus", max_tokens=160, temperature=0.7)
    assert qwen._api_key_env == "DASHSCOPE_API_KEY" and "dashscope-intl" in qwen.base_url
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://proxy.example.test/v1")
    deepseek = build_text_model("deepseek:deepseek-v4-pro", max_tokens=160, temperature=0.7)
    assert deepseek.endpoint_host == "proxy.example.test", "a <SCHEME>_BASE_URL override wins"
    bedrock = build_text_model("bedrock:us.amazon.nova-pro-v1:0", max_tokens=160, temperature=0.7, region="us-west-2")
    assert isinstance(bedrock, BedrockTextModel) and bedrock.region == "us-west-2" and bedrock.model_id == "us.amazon.nova-pro-v1:0"
    assert isinstance(build_text_model("scripted", max_tokens=160, temperature=0.7), ScriptedTextModel)
    with pytest.raises(ModelSpecError):
        build_text_model("azure:gpt", max_tokens=160, temperature=0.7)


def test_scheme_aware_adapter_publishes_its_scheme_in_records_and_ledger(tmp_path) -> None:
    from tests.end_bench.test_openai_compat_adapter import FakeTransport, _scoped

    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    model = OpenAICompatTextModel("xai:grok-4.3", scheme="xai", base_url="https://api.x.ai/v1", api_key="k",
                                  transport=FakeTransport(usage=(1_000_000, 1_000_000)), cost_ledger=ledger)
    _, records = _scoped(model, "Negotiate.")
    assert model.model_name == "xai:grok-4.3" and model.model_id == "grok-4.3"
    assert model.get_last_sampling_config()["scheme"] == "xai"
    entry = json.loads((tmp_path / "cost.jsonl").read_text().splitlines()[0])
    assert entry["model_id"] == "xai:grok-4.3" and entry["usd"] == pytest.approx(1.25 + 2.50)
    assert entry["price_source"].startswith("verified")
    assert records[0].to_dict()["effective_sampling"] and records[0].to_dict()["model_revision"] == "xai:grok-4.3"
    with pytest.raises(ValueError, match="scheme"):
        OpenAICompatTextModel("x", scheme="X-AI", transport=FakeTransport(), api_key="k")


def test_price_table_keys_other_providers_by_scheme() -> None:
    assert set(DEFAULT_PRICES_PER_MTOK) == set(PRICE_SOURCES)
    assert normalize_model_id("xai:grok-4.3") == "xai:grok-4.3"
    assert normalize_model_id("bedrock:us.amazon.nova-pro-v1:0") == "amazon.nova-pro-v1:0", "Bedrock colons are not schemes"
    assert resolve_price("xai:grok-4.3") == (1.25, 2.50) and price_is_verified("xai:grok-4.3")
    assert resolve_price("gemini:gemini-3.1-pro-preview") == (2.00, 12.00), "prefix match on the dated id"
    assert resolve_price("gemini:gemini-3.5-flash-lite") == (0.30, 2.50), "the longer prefix wins over 3.5 Flash"
    assert not price_is_verified("openai:gpt-5.6-sol") and price_source("openai:gpt-5.6-sol").startswith("unverified")
    assert resolve_price("moonshot:kimi-k2.6") is None, "no price is invented"
    assert resolve_price("bedrock:global.openai.gpt-5.6-sol") is None, "Bedrock prices for OpenAI ids are not in the table yet"
    assert resolve_price("xai:anthropic.claude-sonnet-4-6") is None, "a scheme id never matches a Bedrock price"


def test_roster_entries_are_consistent() -> None:
    assert MODEL_ROSTER_VERSION.startswith("end-bench-roster/")
    assert len(MODEL_ROSTER) >= 25
    for key, entry in MODEL_ROSTER.items():
        assert key == entry.key and entry.tier in TIERS
        assert entry.routes and entry.preferred is entry.routes[0]
        for route in entry.routes:
            assert route.scheme in known_schemes() and route.scheme != "scripted"
            parse_model_spec(route.spec)
        for route in entry.routes:
            if not route.verified:
                assert route.note, f"{key}: an unverified route must say how to confirm it"
        assert set(entry.peer_benchmarks) <= set(PEER_BENCHMARKS)
    standard = standard_roster()
    assert 15 <= len(standard) <= 24
    vendors = {entry.vendor for entry in standard}
    assert {"OpenAI", "Anthropic", "Google", "xAI", "DeepSeek", "Meta", "Mistral", "Amazon", "Z.ai"} <= vendors
    assert {entry.tier for entry in standard} == set(TIERS)
    # Every standard entry either has an id read from the provider's own listing or says what to confirm.
    for entry in standard:
        assert entry.preferred.verified or entry.preferred.note
    # The paper set: the user's eleven, each the latest of its line, all callable, all in the standard set.
    paper = paper_roster()
    assert [e.key for e in paper] == [
        "claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "gpt-6-astra", "gpt-6.1-sol",
        "gemini-3.1-pro", "gemini-3.8-flash", "deepseek-v4-pro", "grok-4.7", "glm-5.3", "muse-wonder-1.0",
    ]
    assert all(e.available and e.standard for e in paper)
    assert all(e.preferred.verified for e in paper), "every paper route id was read from the provider's own listing"
    assert {e.sampling for e in paper} == set(SAMPLING_MODES)
    # Announced but not callable entries fail closed and are never paper entries.
    argon = MODEL_ROSTER["gemini-4-argon"]
    assert not argon.available and not argon.paper
    with pytest.raises(ValueError, match="must be available"):
        RosterEntry("x", "X", "V", "mid", routes=(Route("xai:x", True, "t"),), standard=True, paper=True, available=False)
    with pytest.raises(ValueError, match="standard set"):
        RosterEntry("x", "X", "V", "mid", routes=(Route("xai:x", True, "t"),), paper=True)
    with pytest.raises(ValueError, match="sampling"):
        RosterEntry("x", "X", "V", "mid", routes=(Route("xai:x", True, "t"),), sampling="hot")
    assert sampling_constraint("bedrock:global.anthropic.claude-fable-5-1") == "provider_default_only"
    assert sampling_constraint("gemini:gemini-3.8-flash") == "provider_default_recommended"
    assert sampling_constraint("xai:grok-4.7") == "free" and sampling_constraint("bedrock:unknown.model") == "free"
    with pytest.raises(ValueError, match="tier"):
        RosterEntry("x", "X", "V", "ultra", routes=(Route("xai:x", True, "t"),))
    with pytest.raises(ValueError, match="unknown peer benchmarks"):
        RosterEntry("x", "X", "V", "mid", routes=(Route("xai:x", True, "t"),), peer_benchmarks=("Nope",))
    with pytest.raises(ModelSpecError):
        Route("azure:x", True, "t")


def test_roster_keys_resolve_to_specs() -> None:
    assert resolve_model_spec("@nova-pro") == "bedrock:us.amazon.nova-pro-v1:0"
    assert resolve_model_spec("@grok-4.7") == "xai:grok-4.7"
    assert resolve_model_spec("@grok-4.7/bedrock") == "bedrock:global.xai.grok-4.7"
    assert resolve_model_spec("@gpt-6-astra") == "bedrock:global.openai.gpt-6-astra"
    assert resolve_model_spec("@gpt-6-astra/openai") == "openai:gpt-6-astra"
    with pytest.raises(ModelSpecError, match="not callable yet"):
        resolve_model_spec("@gemini-4-argon")
    assert resolve_model_spec("xai:grok-4.3") == "xai:grok-4.3" and resolve_model_spec("scripted") == "scripted"
    with pytest.raises(ModelSpecError, match="unknown roster key"):
        resolve_model_spec("@nope")
    with pytest.raises(ModelSpecError, match="no 'xai' route"):
        resolve_model_spec("@nova-pro/xai")
    with pytest.raises(ModelSpecError, match="unsupported model spec"):
        resolve_model_spec("azure:gpt")


def test_roster_rendering_and_json(monkeypatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    text = format_roster(MODEL_ROSTER.values())
    assert "Pgpt-6-astra" in text and "*nova-pro" in text and "!gemini-4-argon" in text
    assert "CHECK" in text and "missing" in text and "aws" in text and "DEFAULT" in text
    payload = json.loads(roster_json(MODEL_ROSTER.values()))
    assert payload["roster_version"] == MODEL_ROSTER_VERSION
    assert {e["key"] for e in payload["entries"]} == set(MODEL_ROSTER)
    assert payload["peer_benchmarks"]["DecepEval"]["url"].startswith("https://arxiv.org/")
    nova = next(e for e in payload["entries"] if e["key"] == "nova-pro")
    assert nova["routes"][0]["priced"] is True and nova["routes"][0]["verified"] is True
    fable = next(e for e in payload["entries"] if e["key"] == "claude-fable-5-1")
    assert fable["paper"] and fable["sampling"] == "provider_default_only" and fable["reasoning"] == "always_on"


def test_cli_roster_and_model_resolution(tmp_path) -> None:
    from click.testing import CliRunner

    from end_bench.cli import cli

    runner = CliRunner()
    listing = runner.invoke(cli, ["roster", "--standard"])
    assert listing.exit_code == 0, listing.output
    assert "gpt-6-astra" in listing.output and "deepseek-v4-pro" in listing.output and "gpt-5.6-sol" not in listing.output
    paper = runner.invoke(cli, ["roster", "--paper", "--json"])
    assert paper.exit_code == 0 and len(json.loads(paper.output)["entries"]) == 11
    tier = runner.invoke(cli, ["roster", "--tier", "budget", "--json"])
    assert tier.exit_code == 0 and all(e["tier"] == "budget" for e in json.loads(tier.output)["entries"])
    providers = runner.invoke(cli, ["roster", "--providers"])
    assert providers.exit_code == 0 and "GEMINI_API_KEY" in providers.output and "api.x.ai" in providers.output
    bad = runner.invoke(cli, ["module", "run", "deception_under_incentive", "--model", "@nope", "--out", str(tmp_path)])
    assert bad.exit_code != 0 and "unknown roster key" in bad.output
    bad_cp = runner.invoke(cli, ["smoke", "--families", "1", "--counterpart-model", "azure:x", "--out", str(tmp_path)])
    assert bad_cp.exit_code != 0 and "--counterpart-model" in bad_cp.output
    offline = runner.invoke(cli, ["run-text", "--model", "scripted", "--families", "1"])
    assert offline.exit_code != 0 and "smoke" in offline.output
    info = runner.invoke(cli, ["info"])
    assert info.exit_code == 0 and json.loads(info.output)["model_roster_version"] == MODEL_ROSTER_VERSION


def test_run_spec_validates_specs_through_the_registry() -> None:
    from end_bench.run import RunSpec, RunSpecError

    assert RunSpec(model="gemini:gemini-3.1-pro-preview", counterpart_model="bedrock:us.amazon.nova-pro-v1:0").counterpart_regime == "fixed:bedrock:us.amazon.nova-pro-v1:0"
    with pytest.raises(RunSpecError, match="unsupported model"):
        RunSpec(model="azure:gpt")
    with pytest.raises(RunSpecError, match="unsupported counterpart_model"):
        RunSpec(model="scripted", counterpart_model="@nova-pro")  # roster keys resolve at the CLI, specs stay literal
    # Provider default sampling is the default; a fixed temperature is refused for models that reject it.
    assert RunSpec(model="scripted").temperature is None
    assert RunSpec(model="xai:grok-4.7", temperature=0.7).temperature == 0.7
    with pytest.raises(RunSpecError, match="provider default sampling"):
        RunSpec(model="bedrock:global.anthropic.claude-fable-5-1", temperature=0.7)
    with pytest.raises(RunSpecError, match="counterpart_model .* provider default"):
        RunSpec(model="scripted", counterpart_model="bedrock:global.openai.gpt-6-astra", temperature=1.0)
    with pytest.raises(RunSpecError, match="temperature"):
        RunSpec(model="scripted", temperature=3.0)
    with pytest.raises(RunSpecError, match="request_extras"):
        RunSpec(model="scripted", request_extras=[("effort", "low")])
    assert RunSpec(model="scripted", request_extras={"effort": "low"}).to_dict()["request_extras"] == {"effort": "low"}


def test_adapters_send_nothing_for_provider_default_sampling_and_forward_extras(tmp_path) -> None:
    from tests.end_bench.test_bedrock_adapter import FakeConverseClient
    from tests.end_bench.test_openai_compat_adapter import FakeTransport

    client = FakeConverseClient()
    bedrock = build_text_model("bedrock:global.openai.gpt-6-astra", max_tokens=400, temperature=None,
                               honor_caller_sampling=False, request_extras={"reasoning_effort": "low"})
    bedrock._client = client
    bedrock.sample_text("hi", temperature=1.0, top_p=0.95, max_tokens=2200)
    request = client.requests[0]
    assert request["inferenceConfig"] == {"maxTokens": 400}, "no temperature and no topP leave the adapter"
    assert request["additionalModelRequestFields"] == {"reasoning_effort": "low"}
    prov = bedrock.get_last_sampling_config()
    assert prov["sampling_policy"] == "provider_default" and prov["effective"]["temperature"] is None
    assert prov["effective"]["top_p"] is None and prov["requested"]["temperature"] == 1.0
    assert prov["additional_request_fields"] == {"reasoning_effort": "low"} and prov["refusal"] is False

    refusing = FakeConverseClient()
    refusing.converse = lambda **req: {"output": {"message": {"content": [{"text": ""}]}},
                                       "usage": {"inputTokens": 10, "outputTokens": 0}, "stopReason": "refusal"}
    fable = BedrockTextModel("bedrock:global.anthropic.claude-fable-5-1", client=refusing, temperature=None)
    assert fable.sample_text("hi") == "" and fable.refusals == 1 and fable.get_last_sampling_config()["refusal"] is True

    transport = FakeTransport()
    openai = build_text_model("openai:gpt-6-astra", max_tokens=400, temperature=None, honor_caller_sampling=False,
                              request_extras={"reasoning_effort": "low"})
    openai._api_key = "k"
    openai._transport = transport  # never the real network in tests
    openai.sample_text("hi", temperature=0.7, top_p=0.9)
    body = transport.calls[0][2]
    assert "temperature" not in body and "top_p" not in body and body["reasoning_effort"] == "low"
    assert openai.get_last_sampling_config()["sampling_policy"] == "provider_default"
    # A configured temperature still goes out, and direct callers still win when honored.
    configured = BedrockTextModel("amazon.nova-pro-v1:0", client=FakeConverseClient(), temperature=0.5, honor_caller_sampling=False)
    configured.sample_text("hi", temperature=1.0)
    assert configured._client.requests[0]["inferenceConfig"]["temperature"] == 0.5
    assert configured.get_last_sampling_config()["sampling_policy"] == "configured"


def test_price_prefixes_stop_at_a_version_or_date_boundary() -> None:
    """Audit: Grok 4.7 Fast, GPT 5.5 Mini and Gemini 2.5 Flash Lite inherited a neighbour's price."""
    assert resolve_price("xai:grok-4.7-fast") is None
    assert resolve_price("openai:gpt-5.5-mini") is None
    assert resolve_price("gemini:gemini-2.5-flash-lite") is None
    assert resolve_price("bedrock:anthropic.claude-sonnet-4-5x") is None
    assert resolve_price("bedrock:us.amazon.nova-pro-v1:0") == resolve_price("amazon.nova-pro")
    assert resolve_price("bedrock:meta.llama4-maverick-17b-instruct-v1:0") == resolve_price("meta.llama4-maverick")
    assert resolve_price("bedrock:anthropic.claude-sonnet-4-6-20260301-v1:0") == resolve_price("anthropic.claude-sonnet-4-6")
    assert resolve_price("scripted:offline") == resolve_price("scripted")
    assert resolve_price("gemini:gemini-3.1-pro-preview") == resolve_price("gemini:gemini-3.1-pro")
    assert price_source("xai:grok-4.7-fast") is None


def test_sampling_constraint_ignores_the_regional_inference_profile() -> None:
    from end_bench.models.roster import canonical_spec, paper_roster

    fable = next(entry for entry in paper_roster() if entry.key == "claude-fable-5-1")
    route = fable.routes[0].spec
    assert sampling_constraint(route) == fable.sampling
    bare = route[len("bedrock:"):]
    for prefix in ("us.", "eu.", "global."):
        for known in ("us.", "eu.", "apac.", "global.", "us-gov.", "jp.", "au.", "ca."):
            if bare.startswith(known):
                bare = bare[len(known):]
        assert sampling_constraint(f"bedrock:{prefix}{bare}") == fable.sampling, prefix
    assert canonical_spec("bedrock:us.anthropic.claude-fable-5-1") == "bedrock:anthropic.claude-fable-5-1"
    assert canonical_spec("openai:gpt-6-astra") == "openai:gpt-6-astra"
