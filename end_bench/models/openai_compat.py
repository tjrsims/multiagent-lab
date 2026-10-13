"""OpenAI compatible chat completions adapter with the same provenance contract as Bedrock.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
1. ``OpenAICompatTextModel`` is a Concordia ``LanguageModel`` over the
   ``POST {base_url}/chat/completions`` API that OpenAI, xAI, DeepSeek,
   Mistral's platform, Together, Fireworks and OpenRouter all serve. The run
   names it ``openai:<model>``; ``OPENAI_BASE_URL`` picks the endpoint and
   ``OPENAI_API_KEY`` the key. The key is read at call time and never stored,
   logged, or written to provenance; only the endpoint host is recorded.
2. Every ``sample_text`` call inside an active generation scope publishes a
   ``GenerationRecord`` with ``capture_mode='none'`` and empty token id tuples,
   exactly like the Bedrock adapter. Requested sampling values are recorded
   as requested; the effective values are what was sent. A run builds the
   adapter with ``honor_caller_sampling=False`` so the run spec's values win
   over Concordia's per call defaults.
3. Transport is a plain function (default: ``urllib``), so tests inject a
   fake and no network or SDK is needed. Rate limits (429), 408, 409 and 5xx
   and connection errors retry with exponential backoff and jitter; a quota
   or billing error, an auth error, or a malformed request fails at once so
   the executor records a failed attempt instead of fabricated text.
4. Usage feeds the ``CostLedger``. Prices for these endpoints are not in the
   Bedrock table, so calls are recorded as unpriced unless the run passes a
   custom price table keyed ``openai:<model prefix>``. No price is invented.
Where this fits: the generation stage of the text track; identical scenario
specs, labels, QC, and benchmark views apply downstream.
"""

from __future__ import annotations

from collections import deque
import json
import os
import random
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from concordia.language_model import language_model

from end_bench.models.pricing import CostLedger


GENERATION_PATH = "openai_chat_completions"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
# A 429 with one of these in the error body is a quota or billing stop, not a rate limit.
NON_RETRYABLE_429_MARKERS = ("insufficient_quota", "billing", "exceeded your current quota")

# transport(url, headers, body, timeout) -> (status, parsed json or text)
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], tuple[int, Any]]


class OpenAICompatGenerationError(RuntimeError):
    """A chat completion failed after retries (or was not retryable)."""


class _HTTPStatusError(Exception):
    def __init__(self, status: int, body: Any):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


def _urllib_transport(url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float) -> tuple[int, Any]:
    data = json.dumps(dict(body)).encode("utf-8")
    req = urllib_request.Request(url, data=data, method="POST", headers=dict(headers))
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - https endpoint chosen by the operator
            raw = response.read().decode("utf-8")
            return int(response.status), (json.loads(raw) if raw else {})
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else ""
        try:
            parsed: Any = json.loads(raw) if raw else {}
        except ValueError:
            parsed = raw
        return int(exc.code), parsed


def _error_text(body: Any) -> str:
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            return " ".join(str(error.get(key)) for key in ("type", "code", "message") if error.get(key))
        if error:
            return str(error)
        return json.dumps(body, sort_keys=True)[:300]
    return str(body)[:300]


class OpenAICompatTextModel(language_model.LanguageModel):
    """Text only acting model over an OpenAI compatible chat completions endpoint."""

    def __init__(
        self,
        model_id: str,
        *,
        scheme: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        max_tokens: int = 128,
        temperature: float | None = 0.7,
        top_p: float | None = None,
        system_prompt: str | None = None,
        transport: Transport | None = None,
        cost_ledger: CostLedger | None = None,
        max_retries: int = 8,
        backoff_base_seconds: float = 1.0,
        backoff_cap_seconds: float = 30.0,
        read_timeout_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        token_budget_per_day: int | None = None,
        clock: Callable[[], float] = time.time,
        honor_caller_sampling: bool = True,
        max_tokens_field: str = "max_tokens",
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if temperature is not None and not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be within [0, 2] (None sends the provider default)")
        if top_p is not None and not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be within (0, 1]")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if token_budget_per_day is not None and (type(token_budget_per_day) is not int or token_budget_per_day < 1):
            raise ValueError("token_budget_per_day must be a positive integer when given")
        if max_tokens_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("max_tokens_field must be 'max_tokens' or 'max_completion_tokens'")
        if not isinstance(scheme, str) or not scheme.isalpha() or not scheme.islower():
            raise ValueError("scheme must be lowercase letters, e.g. 'openai', 'xai', 'gemini'")
        # The scheme names the provider in every id this adapter publishes
        # (``xai:grok-4.3``) and picks the default environment variable names.
        self.scheme = scheme
        self.model_id = model_id.removeprefix(f"{scheme}:")
        api_key_env = api_key_env or f"{scheme.upper()}_API_KEY"
        base_url_env = base_url_env or f"{scheme.upper()}_BASE_URL"
        resolved_base = (base_url or os.environ.get(base_url_env) or DEFAULT_BASE_URL).rstrip("/")
        if not resolved_base.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("base_url must use https (plain http is allowed for localhost only)")
        self.base_url = resolved_base
        self.endpoint_host = urllib_parse.urlparse(self.base_url).netloc
        self._api_key = api_key
        self._api_key_env = api_key_env
        self.default_max_tokens = max_tokens
        # None: provider default sampling, no temperature or top_p in the body.
        # OpenAI reasoning models reject any temperature; Gemini 3 and DeepSeek
        # V4 advise their defaults.
        self.default_temperature = None if temperature is None else float(temperature)
        self.default_top_p = None if top_p is None else float(top_p)
        self.refusals = 0
        self.system_prompt = system_prompt
        self.max_tokens_field = max_tokens_field
        self.extra_body = dict(extra_body or {})
        self.last_error: str | None = None
        self.failed_calls = 0
        self._transport = transport or _urllib_transport
        self._read_timeout = float(read_timeout_seconds)
        self.cost_ledger = cost_ledger if cost_ledger is not None else CostLedger()
        self.max_retries = max_retries
        self.backoff_base = float(backoff_base_seconds)
        self.backoff_cap = float(backoff_cap_seconds)
        self.token_budget_per_day = token_budget_per_day
        self._clock = clock
        self.honor_caller_sampling = bool(honor_caller_sampling)
        self._token_window: deque[tuple[float, int]] = deque()
        self.budget_waits = 0
        self.budget_wait_seconds = 0.0
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self._call_count = 0
        self._last_sampling_config: dict[str, Any] = {}
        self._last_usage: dict[str, int] = {}
        # Attributes the research runner reads for provenance.
        self.tokenizer = None
        self.layers_to_capture: list[int] = []

    # -- identity -------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"{self.scheme}:{self.model_id}"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def activation_dim(self) -> int:
        return 0

    @property
    def endpoint_description(self) -> str:
        """Host and generation path, for manifests. Never includes the key."""
        return f"{self.endpoint_host} {GENERATION_PATH}"

    def get_last_sampling_config(self) -> dict[str, Any]:
        return dict(self._last_sampling_config)

    def capture_followup_activations(self, original_prompt: str, response: str) -> dict:
        del original_prompt, response
        return {}

    # -- transport --------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        key = self._api_key or os.environ.get(self._api_key_env)
        if not key:
            raise OpenAICompatGenerationError(
                f"no API key: pass api_key or set ${self._api_key_env} (the key is never stored in run artifacts)"
            )
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _body(self, prompt: str, *, max_tokens: int, temperature: float | None, top_p: float | None) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            self.max_tokens_field: int(max_tokens),
        }
        if temperature is not None:
            body["temperature"] = float(temperature)
        if top_p is not None:
            body["top_p"] = float(top_p)
        body.update(self.extra_body)
        return body

    def _post(self, prompt: str, *, max_tokens: int, temperature: float | None, top_p: float | None) -> tuple[dict[str, Any], int]:
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        attempt = 0
        while True:
            body = self._body(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            try:
                status, payload = self._transport(url, headers, body, self._read_timeout)
            except (urllib_error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if attempt >= self.max_retries:
                    return self._fail(f"{type(exc).__name__}: {exc}", attempt, exc)
                self._backoff(attempt)
                attempt += 1
                continue
            if 200 <= status < 300 and isinstance(payload, Mapping):
                return dict(payload), attempt
            message = _error_text(payload)
            # Newer OpenAI models take max_completion_tokens and reject max_tokens;
            # switch the field once and record the switch, no retry budget spent.
            if status == 400 and self.max_tokens_field == "max_tokens" and "max_completion_tokens" in message:
                self.max_tokens_field = "max_completion_tokens"
                continue
            quota_stop = status == 429 and any(marker in message.lower() for marker in NON_RETRYABLE_429_MARKERS)
            if status not in RETRYABLE_STATUS or quota_stop or attempt >= self.max_retries:
                return self._fail(f"HTTP {status}: {message}", attempt, _HTTPStatusError(status, payload))
            self._backoff(attempt)
            attempt += 1

    def _fail(self, message: str, attempt: int, exc: BaseException) -> tuple[dict[str, Any], int]:
        self.failed_calls += 1
        self.last_error = message
        raise OpenAICompatGenerationError(
            f"chat completion failed for {self.model_id} at {self.endpoint_host} after {attempt} retries: {message}"
        ) from exc

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
        self._sleep(delay * (0.5 + self._rng.random()))  # jitter in [0.5, 1.5)

    def _respect_token_budget(self) -> None:
        """Sleep until the last 24 hours of usage fall under ``token_budget_per_day``."""
        budget = self.token_budget_per_day
        if budget is None:
            return
        window = 24 * 3600.0
        while True:
            now = self._clock()
            while self._token_window and self._token_window[0][0] <= now - window:
                self._token_window.popleft()
            used = sum(tokens for _, tokens in self._token_window)
            if used < budget or not self._token_window:
                return
            wait = max(1.0, self._token_window[0][0] + window - now)
            self.budget_waits += 1
            self.budget_wait_seconds += wait
            self._sleep(wait)

    @staticmethod
    def _extract(response: Mapping[str, Any]) -> tuple[str, str | None]:
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], Mapping):
            return "", None
        first = choices[0]
        message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
        content = message.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
        text = content if isinstance(content, str) else ""
        finish = first.get("finish_reason")
        return text, (str(finish) if finish is not None else None)

    # -- Concordia interface ------------------------------------------------------

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        terminators: Sequence[str] = (),
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> str:
        del timeout
        self._call_count += 1
        use_caller = self.honor_caller_sampling
        effective_max_tokens = int(max_tokens) if (use_caller and max_tokens is not None) else self.default_max_tokens
        if use_caller and temperature is not None:
            effective_temperature: float | None = min(max(float(temperature), 0.0), 2.0)
            sampling_policy = "caller"
        elif self.default_temperature is None:
            effective_temperature = None
            sampling_policy = "provider_default"
        else:
            effective_temperature = min(max(self.default_temperature, 0.0), 2.0)
            sampling_policy = "configured"
        requested_top_p = float(top_p) if top_p is not None else (self.default_top_p if self.default_top_p is not None else 1.0)
        effective_top_p = None if effective_temperature is None else (
            float(top_p) if (use_caller and top_p is not None) else self.default_top_p
        )
        self._respect_token_budget()
        started = time.monotonic()
        response, retries = self._post(
            prompt, max_tokens=effective_max_tokens, temperature=effective_temperature, top_p=effective_top_p,
        )
        latency_ms = (time.monotonic() - started) * 1000.0
        text, stop_reason = self._extract(response)
        refused = str(stop_reason).lower() in ("refusal", "content_filter")
        if refused:
            self.refusals += 1
        terminator_hit: str | None = None
        for terminator in terminators or ():
            if terminator and terminator in text:
                text = text.split(terminator, 1)[0]
                terminator_hit = terminator
                break
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        self._last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        from interpretability.evaluation import (  # heavy import, kept lazy
            _publish_scoped_generation,
            _sampling_provenance,
        )
        from interpretability.runtime.model_call import CaptureMode

        requested_temperature = (
            float(temperature) if temperature is not None
            else (effective_temperature if effective_temperature is not None else 1.0)
        )
        provenance = _sampling_provenance(
            requested_max_tokens=max_tokens,
            effective_max_tokens=effective_max_tokens,
            temperature=requested_temperature,
            top_p=requested_top_p,
            top_k=int(top_k) if top_k is not None else 0,
            seed=seed,
            effective_temperature=effective_temperature,
            effective_top_p=effective_top_p,
            effective_top_k=None,
            effective_do_sample=True if effective_temperature is None else effective_temperature > 0.0,
            generation_path=GENERATION_PATH,
            fallback_used=False,
            fallback_reason=None,
        )
        provenance["effective"]["seed"] = None  # not sent; servers differ on seed support
        provenance["provider"] = "openai_compat"
        provenance["sampling_policy"] = sampling_policy
        provenance["extra_body"] = dict(self.extra_body)
        provenance["refusal"] = refused
        provenance["scheme"] = self.scheme
        provenance["endpoint_host"] = self.endpoint_host
        provenance["model_id"] = self.model_id
        provenance["max_tokens_field"] = self.max_tokens_field
        provenance["usage"] = dict(self._last_usage)
        provenance["stop_reason"] = stop_reason
        provenance["retries"] = retries
        provenance["token_budget_per_day"] = self.token_budget_per_day
        provenance["response_model"] = response.get("model")
        self._token_window.append((self._clock(), input_tokens + output_tokens))
        self._last_sampling_config = provenance

        record = _publish_scoped_generation(
            assembled_prompt=prompt,
            input_token_ids=(),
            output_token_ids=(),
            retained_token_ids=(),
            output_text=text,
            terminator=terminator_hit,
            sampling_provenance=provenance,
            capture_mode=CaptureMode.NONE,
            activations={},
        )
        self.cost_ledger.record(
            model_id=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            retries=retries,
            call_id=record.call_id if record is not None else None,
            stop_reason=stop_reason,
        )
        return text

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[int, str, dict[str, Any]]:
        """Pick the response with the most lexical overlap with a sampled answer."""
        if not responses:
            raise ValueError("responses must be non-empty")
        sample = self.sample_text(prompt, max_tokens=100, seed=seed, **kwargs)
        sample_words = set(sample.lower().split())
        best = max(
            range(len(responses)),
            key=lambda index: len(sample_words & set(responses[index].lower().split())),
        )
        return best, responses[best], {"sample": sample}


__all__ = [
    "DEFAULT_BASE_URL",
    "GENERATION_PATH",
    "OpenAICompatGenerationError",
    "OpenAICompatTextModel",
]
