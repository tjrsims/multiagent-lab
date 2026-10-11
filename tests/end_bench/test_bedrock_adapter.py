"""Bedrock adapter: records, provenance, retries, cost ledger — with a fake client."""

from __future__ import annotations

import json

import pytest

from end_bench.models.bedrock import (
    GENERATION_PATH,
    BedrockGenerationError,
    BedrockTextModel,
)
from end_bench.models.pricing import (
    PRICE_TABLE_VERSION,
    CostLedger,
    estimate_cost,
    estimate_design_cost,
    load_price_table,
    resolve_price,
)
from interpretability.runtime.model_call import (
    CallPurpose,
    CaptureMode,
    GenerationCallSpec,
    GenerationRecorder,
    active_generation_recorder,
    generation_call,
)


class _FakeClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


class FakeConverseClient:
    def __init__(self, *, text="My minimum is 60 points.", fail_codes=(), usage=(1200, 40)):
        self.text = text
        self.fail_codes = list(fail_codes)
        self.usage = usage
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        if self.fail_codes:
            raise _FakeClientError(self.fail_codes.pop(0))
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": self.text}]}},
            "usage": {"inputTokens": self.usage[0], "outputTokens": self.usage[1]},
            "stopReason": "end_turn",
        }


def _scoped(model, prompt, **kwargs):
    """Call sample_text inside an explicit generation scope and return the records."""
    recorder = GenerationRecorder("test-run")
    spec = GenerationCallSpec(
        run_id="test-run",
        trial_id="trial-1",
        attempt=0,
        sequence=0,
        actor_id="Negotiator",
        purpose=CallPurpose.ACTOR_ACTION,
        model_revision=model.model_name,
        tokenizer_revision=model.model_name,
        concordia_version="2.4.0",
        capture_mode=CaptureMode.NONE,
    )
    with active_generation_recorder(recorder):
        with generation_call(spec):
            text = model.sample_text(prompt, **kwargs)
    return text, recorder.records


def test_sample_text_publishes_faithful_text_only_record(tmp_path) -> None:
    client = FakeConverseClient()
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    model = BedrockTextModel(
        "bedrock:anthropic.claude-sonnet-4-6-20260301-v1:0", client=client,
        cost_ledger=ledger, max_tokens=120, temperature=0.5,
    )
    text, records = _scoped(model, "Negotiate.", max_tokens=64, seed=7)
    assert text == "My minimum is 60 points."
    assert model.model_name == "bedrock:anthropic.claude-sonnet-4-6-20260301-v1:0"
    assert model.call_count == 1
    assert len(records) == 1
    record = records[0]
    assert record.capture_mode is CaptureMode.NONE
    assert record.input_token_ids == () and record.output_token_ids == ()
    assert record.retained_token_ids == () and record.retained_token_index is None
    assert record.activation_artifacts == ()
    assert record.generation_path == GENERATION_PATH
    assert record.output_text == text and record.assembled_prompt == "Negotiate."
    assert record.requested_sampling.max_tokens == 64
    assert record.requested_sampling.seed == 7
    assert record.effective_sampling.seed is None, "Bedrock exposes no seed"
    request = client.requests[0]
    assert request["modelId"] == "anthropic.claude-sonnet-4-6-20260301-v1:0"
    # Anthropic ids get temperature only; Bedrock rejects temperature plus topP for them.
    assert request["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.5}
    assert request["messages"][0]["content"][0]["text"] == "Negotiate."
    provenance = model.get_last_sampling_config()
    assert provenance["provider"] == "bedrock" and provenance["usage"] == {"input_tokens": 1200, "output_tokens": 40}
    totals = ledger.totals()
    assert totals["calls"] == 1 and totals["input_tokens"] == 1200 and totals["output_tokens"] == 40
    assert totals["usd"] == pytest.approx(1200 / 1e6 * 3.0 + 40 / 1e6 * 15.0)
    line = json.loads((tmp_path / "cost.jsonl").read_text().splitlines()[0])
    assert line["call_id"] == record.call_id and line["price_table_version"] == PRICE_TABLE_VERSION


def test_outside_a_scope_no_record_is_published_but_cost_is_logged() -> None:
    model = BedrockTextModel("amazon.nova-pro-v1:0", client=FakeConverseClient())
    assert model.sample_text("hello") == "My minimum is 60 points."
    assert model.cost_ledger.calls == 1
    assert model.cost_ledger.totals()["usd"] == pytest.approx(1200 / 1e6 * 0.8 + 40 / 1e6 * 3.2)


def test_terminators_truncate_and_record_the_terminator() -> None:
    model = BedrockTextModel("amazon.nova-pro-v1:0", client=FakeConverseClient(text="Offer 60.\nCounterpart: no"))
    text, records = _scoped(model, "p", terminators=("\nCounterpart:",))
    assert text == "Offer 60."
    assert records[0].terminator == "\nCounterpart:"


def test_retries_on_throttling_then_succeeds_and_counts_retries() -> None:
    sleeps = []
    client = FakeConverseClient(fail_codes=["ThrottlingException", "ServiceUnavailableException"])
    model = BedrockTextModel("amazon.nova-lite-v1:0", client=client, sleep=sleeps.append, max_retries=5)
    assert model.sample_text("p") == "My minimum is 60 points."
    assert len(client.requests) == 3
    assert len(sleeps) == 2 and all(0 < s <= 30 for s in sleeps)
    assert model.get_last_sampling_config()["retries"] == 2


def test_retries_exhausted_raise_and_non_retryable_errors_raise_immediately() -> None:
    client = FakeConverseClient(fail_codes=["ThrottlingException"] * 3)
    model = BedrockTextModel("amazon.nova-lite-v1:0", client=client, sleep=lambda s: None, max_retries=2)
    with pytest.raises(BedrockGenerationError):
        model.sample_text("p")
    assert len(client.requests) == 3
    client = FakeConverseClient(fail_codes=["ValidationException"])
    model = BedrockTextModel("amazon.nova-lite-v1:0", client=client, sleep=lambda s: None)
    with pytest.raises(BedrockGenerationError):
        model.sample_text("p")
    assert len(client.requests) == 1
    assert model.cost_ledger.calls == 0, "failed calls are never billed as successes"


def test_sample_choice_picks_lexical_match() -> None:
    model = BedrockTextModel("amazon.nova-lite-v1:0", client=FakeConverseClient(text="I will accept the offer"))
    index, choice, debug = model.sample_choice("Choose", ["reject it", "accept the offer"])
    assert index == 1 and choice == "accept the offer" and "sample" in debug


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": ""}, {"model_id": "m", "max_tokens": 0}, {"model_id": "m", "temperature": 1.5},
        {"model_id": "m", "top_p": 0}, {"model_id": "m", "max_retries": -1},
    ],
)
def test_constructor_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        BedrockTextModel(client=FakeConverseClient(), **kwargs)


def test_pricing_prefix_match_estimates_and_unknown_models(tmp_path) -> None:
    assert resolve_price("anthropic.claude-sonnet-4-6-20260301-v1:0") == (3.0, 15.0)
    assert resolve_price("bedrock:meta.llama4-maverick-17b-instruct-v1:0") == (0.24, 0.97)
    # Regional inference-profile ids price as their foundation model.
    assert resolve_price("us.anthropic.claude-sonnet-4-6-20260301-v1:0") == (3.0, 15.0)
    assert resolve_price("bedrock:eu.amazon.nova-pro-v1:0") == (0.8, 3.2)
    from end_bench.models.pricing import normalize_model_id
    assert normalize_model_id("bedrock:us.anthropic.claude-sonnet-4-6") == "anthropic.claude-sonnet-4-6"
    assert normalize_model_id("anthropic.claude-sonnet-4-6") == "anthropic.claude-sonnet-4-6"
    assert resolve_price("unknown.model") is None
    assert estimate_cost("unknown.model", 10, 10) is None
    assert estimate_cost("amazon.nova-micro-v1:0", 1_000_000, 1_000_000) == pytest.approx(0.035 + 0.14)
    with pytest.raises(ValueError):
        estimate_cost("amazon.nova-micro-v1:0", -1, 0)
    design = estimate_design_cost("anthropic.claude-sonnet-4-6", negotiations=3240)
    assert design["calls"] == 32400 and design["usd_total"] == pytest.approx(218.7)
    custom = tmp_path / "prices.json"
    custom.write_text(json.dumps({"my.model": [1.0, 2.0]}))
    table = load_price_table(custom)
    assert estimate_cost("my.model-v1", 1_000_000, 0, table) == 1.0
    ledger = CostLedger(table=table)
    ledger.record(model_id="other", input_tokens=5, output_tokens=5, latency_ms=1.0, retries=0, call_id=None, stop_reason=None)
    assert ledger.totals()["unpriced_calls"] == 1 and ledger.totals()["usd"] == 0.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"x": [1.0]}))
    with pytest.raises(ValueError):
        load_price_table(bad)



def test_anthropic_models_get_temperature_only_and_others_keep_top_p() -> None:
    """Bedrock rejects Anthropic requests that set both temperature and top_p."""
    from end_bench.models.bedrock import BedrockTextModel

    class Capture:
        def __init__(self):
            self.requests = []

        def converse(self, **request):
            self.requests.append(request)
            return {"output": {"message": {"content": [{"text": "ready"}]}}, "stopReason": "end_turn",
                    "usage": {"inputTokens": 3, "outputTokens": 1}}

    claude = Capture()
    BedrockTextModel("global.anthropic.claude-sonnet-4-6", client=claude, temperature=0.5).sample_text("hi")
    assert "topP" not in claude.requests[0]["inferenceConfig"]
    assert claude.requests[0]["inferenceConfig"]["temperature"] == 0.5
    nova = Capture()
    BedrockTextModel("amazon.nova-pro-v1:0", client=nova, temperature=0.5).sample_text("hi")
    assert nova.requests[0]["inferenceConfig"]["topP"] == 0.9
    explicit = Capture()
    BedrockTextModel("global.anthropic.claude-sonnet-4-6", client=explicit, top_p=0.8).sample_text("hi", top_p=0.8)
    assert "topP" not in explicit.requests[0]["inferenceConfig"]  # never sent, even when asked for


class _DailyCapError(Exception):
    def __init__(self):
        super().__init__("ThrottlingException")
        self.response = {"Error": {"Code": "ThrottlingException",
                                   "Message": "Too many tokens per day, please wait before trying again."}}


class _CapThenRecover(FakeConverseClient):
    def __init__(self, cap_hits, **kwargs):
        super().__init__(**kwargs)
        self.cap_hits = cap_hits

    def converse(self, **request):
        if self.cap_hits:
            self.cap_hits -= 1
            self.requests.append(request)
            raise _DailyCapError()
        return super().converse(**request)


def test_daily_cap_throttling_waits_long_instead_of_failing_the_call() -> None:
    sleeps = []
    client = _CapThenRecover(2)
    model = BedrockTextModel("global.anthropic.claude-sonnet-4-6", client=client, sleep=sleeps.append,
                             max_retries=2, daily_cap_backoff_seconds=900.0)
    assert model.sample_text("p") == "My minimum is 60 points."
    assert len(client.requests) == 3
    assert len(sleeps) == 2 and all(675 <= s <= 1125 for s in sleeps), "long waits, not the 30 second schedule"
    assert model.daily_cap_waits == 2 and model.daily_cap_wait_seconds == pytest.approx(sum(sleeps))
    config = model.get_last_sampling_config()
    assert config["retries"] == 0, "cap waits do not count against the short retries"
    assert config["daily_cap_waits"] == 2
    assert model.failed_calls == 0 and model.cost_ledger.calls == 1


def test_daily_cap_wait_has_a_ceiling() -> None:
    client = _CapThenRecover(5)
    model = BedrockTextModel("global.anthropic.claude-sonnet-4-6", client=client, sleep=lambda s: None,
                             daily_cap_backoff_seconds=900.0, daily_cap_max_wait_seconds=1000.0)
    with pytest.raises(BedrockGenerationError, match="daily token cap"):
        model.sample_text("p")
    assert len(client.requests) == 2, "one wait of about 900s, then the second hit exceeds the ceiling"
    assert model.failed_calls == 1 and "per day" in model.last_error


def test_plain_throttling_keeps_the_short_schedule() -> None:
    sleeps = []
    client = FakeConverseClient(fail_codes=["ThrottlingException"])
    model = BedrockTextModel("amazon.nova-lite-v1:0", client=client, sleep=sleeps.append)
    model.sample_text("p")
    assert len(sleeps) == 1 and sleeps[0] <= 30 and model.daily_cap_waits == 0


def test_token_budget_paces_calls_under_a_rolling_daily_window() -> None:
    now = [1_000_000.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    client = FakeConverseClient(usage=(1200, 40))  # 1,240 tokens per call
    model = BedrockTextModel("global.anthropic.claude-sonnet-4-6", client=client, sleep=sleep,
                             clock=lambda: now[0], token_budget_per_day=2000)
    model.sample_text("one")
    now[0] += 60
    model.sample_text("two")  # 1,240 used, under budget, allowed; 2,480 after
    assert sleeps == []
    now[0] += 60
    model.sample_text("three")  # 2,480 used: wait until the first call leaves the 24h window
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(24 * 3600 - 120)
    assert model.budget_waits == 1 and len(client.requests) == 3
    assert model.get_last_sampling_config()["token_budget_per_day"] == 2000
    with pytest.raises(ValueError, match="token_budget_per_day"):
        BedrockTextModel("amazon.nova-lite-v1:0", client=client, token_budget_per_day=0)
