"""OpenAI compatible adapter: records, provenance, retries, key handling, cost ledger. Fake transport only."""

from __future__ import annotations

import json

import pytest

from end_bench.models.openai_compat import (
    DEFAULT_BASE_URL,
    GENERATION_PATH,
    OpenAICompatGenerationError,
    OpenAICompatTextModel,
)
from end_bench.models.pricing import CostLedger
from interpretability.runtime.model_call import (
    CallPurpose,
    CaptureMode,
    GenerationCallSpec,
    GenerationRecorder,
    active_generation_recorder,
    generation_call,
)


class FakeTransport:
    """Scripted (status, payload) answers; records every request body and header set."""

    def __init__(self, answers=None, *, text="My minimum is 60 points.", usage=(900, 30)):
        self.answers = list(answers or [])
        self.text = text
        self.usage = usage
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, dict(headers), json.loads(json.dumps(body))))
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        return 200, {
            "model": body["model"] + "-2026-01-01",
            "choices": [{"message": {"role": "assistant", "content": self.text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": self.usage[0], "completion_tokens": self.usage[1]},
        }


def _scoped(model, prompt, **kwargs):
    recorder = GenerationRecorder("test-run")
    spec = GenerationCallSpec(
        run_id="test-run", trial_id="trial-1", attempt=0, sequence=0, actor_id="Negotiator",
        purpose=CallPurpose.ACTOR_ACTION, model_revision=model.model_name, tokenizer_revision=model.model_name,
        concordia_version="2.4.0", capture_mode=CaptureMode.NONE,
    )
    with active_generation_recorder(recorder):
        with generation_call(spec):
            text = model.sample_text(prompt, **kwargs)
    return text, recorder.records


def _model(transport, **kwargs):
    kwargs.setdefault("api_key", "sk-test-not-real")
    kwargs.setdefault("sleep", lambda _s: None)
    return OpenAICompatTextModel("openai:gpt-5", transport=transport, **kwargs)


def test_sample_text_publishes_a_faithful_text_only_record(tmp_path) -> None:
    transport = FakeTransport()
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    model = _model(transport, cost_ledger=ledger, max_tokens=120, temperature=0.5, base_url="https://api.x.ai/v1/")
    text, records = _scoped(model, "Negotiate.", max_tokens=64, seed=7)
    assert text == "My minimum is 60 points."
    assert model.model_name == "openai:gpt-5" and model.endpoint_host == "api.x.ai"
    assert model.endpoint_description == "api.x.ai openai_chat_completions"
    url, headers, body = transport.calls[0]
    assert url == "https://api.x.ai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test-not-real"
    assert body["messages"] == [{"role": "user", "content": "Negotiate."}]
    # Direct callers mean the values they pass.
    assert body["max_tokens"] == 64 and body["temperature"] == 0.5 and "top_p" not in body
    [record] = records
    assert record.capture_mode is CaptureMode.NONE and record.generation_path == GENERATION_PATH
    assert record.input_token_ids == () and record.output_token_ids == () and record.activation_artifacts == ()
    assert record.output_text == text and record.assembled_prompt == "Negotiate."
    provenance = model.get_last_sampling_config()
    assert provenance["provider"] == "openai_compat" and provenance["endpoint_host"] == "api.x.ai"
    assert provenance["usage"] == {"input_tokens": 900, "output_tokens": 30}
    assert provenance["stop_reason"] == "stop" and provenance["retries"] == 0
    assert provenance["effective"]["seed"] is None and provenance["response_model"] == "gpt-5-2026-01-01"
    assert "sk-test" not in json.dumps(provenance) and "sk-test" not in json.dumps(record.to_dict())
    entry = json.loads((tmp_path / "cost.jsonl").read_text().splitlines()[0])
    assert entry["model_id"] == "openai:gpt-5" and entry["usd"] is None and entry["price_source"] is None
    assert ledger.totals()["unpriced_calls"] == 1, "no price is invented for an endpoint not in the table"


def test_custom_price_table_prices_openai_ids_without_touching_bedrock_prices(tmp_path) -> None:
    ledger = CostLedger(path=tmp_path / "cost.jsonl", table={"openai:gpt-5": (1.25, 10.0)})
    model = _model(FakeTransport(usage=(1_000_000, 1_000_000)), cost_ledger=ledger)
    model.sample_text("x")
    assert ledger.totals()["usd"] == pytest.approx(11.25) and ledger.unpriced_calls == 0
    from end_bench.models.pricing import resolve_price

    assert resolve_price("openai:anthropic.claude-sonnet-4-6") is None, "an openai: id never matches a Bedrock price"


def test_run_settings_win_over_caller_values_when_configured() -> None:
    transport = FakeTransport()
    model = _model(transport, max_tokens=160, temperature=0.7, honor_caller_sampling=False)
    model.sample_text("x", max_tokens=2200, temperature=1.0, top_p=0.95)
    body = transport.calls[0][2]
    assert body["max_tokens"] == 160 and body["temperature"] == 0.7 and "top_p" not in body
    provenance = model.get_last_sampling_config()
    assert provenance["requested"]["max_tokens"] == 2200 and provenance["effective"]["max_tokens"] == 160


def test_retries_rate_limits_and_server_errors_then_fails_closed() -> None:
    transport = FakeTransport(answers=[(429, {"error": {"type": "rate_limit_error", "message": "slow down"}}),
                                       (503, "upstream unavailable")])
    slept = []
    model = _model(transport, sleep=slept.append, max_retries=3)
    assert model.sample_text("x") == "My minimum is 60 points."
    assert len(transport.calls) == 3 and len(slept) == 2 and model.get_last_sampling_config()["retries"] == 2

    exhausted = _model(FakeTransport(answers=[(500, {})] * 5), max_retries=2)
    with pytest.raises(OpenAICompatGenerationError, match="after 2 retries"):
        exhausted.sample_text("x")
    assert exhausted.failed_calls == 1 and "HTTP 500" in exhausted.last_error

    quota = _model(FakeTransport(answers=[(429, {"error": {"type": "insufficient_quota", "message": "You exceeded your current quota"}})]))
    with pytest.raises(OpenAICompatGenerationError, match="insufficient_quota"):
        quota.sample_text("x")
    assert len(quota._transport.calls) == 1, "a quota stop is not retried"

    auth = _model(FakeTransport(answers=[(401, {"error": {"message": "bad key"}})]))
    with pytest.raises(OpenAICompatGenerationError, match="HTTP 401"):
        auth.sample_text("x")

    network = _model(FakeTransport(answers=[ConnectionError("reset")]), max_retries=1)
    assert network.sample_text("x") == "My minimum is 60 points."


def test_switches_to_max_completion_tokens_once_when_the_server_demands_it() -> None:
    transport = FakeTransport(answers=[(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}})])
    model = _model(transport, max_tokens=99)
    model.sample_text("x")
    assert transport.calls[0][2].get("max_tokens") == 99
    assert transport.calls[1][2].get("max_completion_tokens") == 99 and "max_tokens" not in transport.calls[1][2]
    assert model.get_last_sampling_config()["max_tokens_field"] == "max_completion_tokens"
    assert model.get_last_sampling_config()["retries"] == 0, "the field switch spends no retry"
    model.sample_text("y")
    assert "max_completion_tokens" in transport.calls[2][2]


def test_key_comes_from_the_environment_and_is_required(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    model = OpenAICompatTextModel("openai:gpt-5", transport=FakeTransport())
    assert model.base_url == DEFAULT_BASE_URL and model.endpoint_host == "api.openai.com"
    with pytest.raises(OpenAICompatGenerationError, match="no API key"):
        model.sample_text("x")
    assert model.call_count == 1
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model = OpenAICompatTextModel("openai:meta-llama/llama-4-maverick", transport=FakeTransport())
    assert model.endpoint_host == "openrouter.ai"
    model.sample_text("x")
    assert model._transport.calls[0][1]["Authorization"] == "Bearer sk-env"
    with pytest.raises(ValueError, match="https"):
        OpenAICompatTextModel("openai:x", base_url="http://example.com/v1", transport=FakeTransport())


def test_system_prompt_terminators_content_parts_and_choice() -> None:
    transport = FakeTransport(answers=[(200, {"choices": [{"message": {"content": [{"type": "text", "text": "Deal. STOP more"}]},
                                                           "finish_reason": "length"}],
                                              "usage": {"prompt_tokens": 5, "completion_tokens": 3}})])
    model = _model(transport, system_prompt="Be brief.")
    assert model.sample_text("x", terminators=("STOP",)) == "Deal. "
    assert transport.calls[0][2]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert model.get_last_sampling_config()["stop_reason"] == "length"
    index, choice, info = _model(FakeTransport(text="I accept the deal")).sample_choice("q", ["walk away", "accept the deal"])
    assert (index, choice) == (1, "accept the deal") and info["sample"] == "I accept the deal"
    empty = _model(FakeTransport(answers=[(200, {"choices": [], "usage": {}})]))
    assert empty.sample_text("x") == "" and empty.get_last_sampling_config()["stop_reason"] is None


def test_token_budget_paces_calls_across_a_sliding_day() -> None:
    now = [1_000_000.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    model = _model(FakeTransport(usage=(600, 0)), token_budget_per_day=1000, clock=lambda: now[0], sleep=sleep)
    model.sample_text("a")
    model.sample_text("b")
    assert slept == [] and model.budget_waits == 0
    model.sample_text("c")  # 1200 tokens in the window: wait until the first call ages out
    assert model.budget_waits == 1 and slept and slept[0] == pytest.approx(24 * 3600.0)


def test_run_spec_and_builder_accept_openai_models(tmp_path, monkeypatch) -> None:
    import end_bench.run as run_module
    from end_bench.models.openai_compat import OpenAICompatTextModel as Adapter
    from end_bench.run import RunSpec, RunSpecError, _build_model

    spec = RunSpec(model="openai:gpt-5", counterpart_model="openai:grok-4", families=1)
    assert spec.provider == "openai" and spec.counterpart_regime == "fixed:openai:grok-4"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.test/v1")
    actor = _build_model(spec, CostLedger())
    opponent = _build_model(spec, CostLedger(), spec.counterpart_model)
    assert isinstance(actor, Adapter) and actor.model_id == "gpt-5" and actor.honor_caller_sampling is False
    assert isinstance(opponent, Adapter) and opponent.model_id == "grok-4" and opponent.endpoint_host == "api.example.test"
    with pytest.raises(RunSpecError, match="unsupported counterpart_model"):
        RunSpec(model="scripted", counterpart_model="anthropic:claude")
    with pytest.raises(RunSpecError, match="unsupported model"):
        _build_model(RunSpec(model="azure:gpt"), CostLedger())
    del run_module, tmp_path


def test_cli_run_text_and_judge_accept_openai_specs() -> None:
    from click.testing import CliRunner

    from end_bench.cli import _judge_model, cli

    result = CliRunner().invoke(cli, ["run-text", "--model", "azure:gpt", "--families", "1"])
    assert result.exit_code != 0 and "'bedrock:' or 'openai:'" in result.output
    judge = _judge_model("openai:gpt-5", "us-east-1")
    assert judge.model_name == "openai:gpt-5" and judge.default_max_tokens == 400 and judge.default_temperature == 0.0
