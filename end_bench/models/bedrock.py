"""Amazon Bedrock acting-model adapter that publishes scoped generation records.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
1. ``BedrockTextModel`` is a Concordia ``LanguageModel`` (``sample_text`` /
   ``sample_choice``) over the Bedrock Converse API. The research runner
   accepts it through ``InterpretabilityRunner(model=...)`` on the
   ``text_only`` track; API models cannot provide activations, so any other
   track is refused upstream.
2. Every ``sample_text`` call inside an active generation scope publishes a
   ``GenerationRecord`` with ``capture_mode='none'`` and *empty* token-id
   tuples (Bedrock does not expose token ids); the record contract allows
   that. Requested sampling settings are recorded faithfully; the effective
   seed is ``None`` because Bedrock exposes none — stated, not invented.
3. Throttling and transient 5xx errors are retried with exponential backoff
   and jitter through an injectable ``sleep``; exhausted retries raise
   ``BedrockGenerationError`` so the executor records a failed attempt rather
   than fabricated text.
4. Usage (input/output tokens) feeds a ``CostLedger`` so every run manifest
   can state its spend and the price table that produced it.
Where this fits: the generation stage of the text tracks; identical scenario
specs, labels, QC, and benchmark views apply downstream.
"""

from __future__ import annotations

import random
from collections import deque
import time
from typing import Any, Callable, Mapping, Sequence

from concordia.language_model import language_model

from end_bench.models.pricing import CostLedger


GENERATION_PATH = "bedrock_converse"
RETRYABLE_ERROR_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ServiceQuotaExceededException",
})
RETRYABLE_EXCEPTION_NAMES = frozenset({
    "ReadTimeoutError",
    "ConnectTimeoutError",
    "EndpointConnectionError",
    "ConnectionClosedError",
})


class BedrockGenerationError(RuntimeError):
    """Raised when a Converse call cannot be completed after retries."""


# Bedrock's hidden account level daily token cap answers with a
# ThrottlingException whose message says "Too many tokens per day". That is
# not a transient blip: short retries burn out within minutes and the trial
# fails, which silently removed 238 of 630 trials from a live Sonnet 4.6 run.
DAILY_CAP_MARKERS = ("per day",)


def _error_message(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping) and error.get("Message") is not None:
            return str(error.get("Message"))
    return str(exc)


def _is_daily_cap(exc: BaseException) -> bool:
    if not _is_retryable(exc):
        return False
    message = _error_message(exc).lower()
    return any(marker in message for marker in DAILY_CAP_MARKERS)


def _error_code(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping):
            code = error.get("Code")
            return str(code) if code is not None else None
    return None


def _is_retryable(exc: BaseException) -> bool:
    code = _error_code(exc)
    if code is not None:
        return code in RETRYABLE_ERROR_CODES
    return type(exc).__name__ in RETRYABLE_EXCEPTION_NAMES


def _default_client(region: str, *, read_timeout: float):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:  # pragma: no cover - exercised only without boto3
        raise ImportError(
            "boto3 is required for Bedrock runs: pip install 'boto3>=1.34' "
            "(or install the repo with the [aws] extra)"
        ) from error
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=read_timeout, retries={"max_attempts": 0}),
    )


class BedrockTextModel(language_model.LanguageModel):
    """Text-only acting model over Bedrock Converse with faithful provenance."""

    def __init__(
        self,
        model_id: str,
        *,
        region: str = "us-east-1",
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float | None = 0.9,
        system_prompt: str | None = None,
        client: Any | None = None,
        cost_ledger: CostLedger | None = None,
        max_retries: int = 8,
        backoff_base_seconds: float = 1.0,
        backoff_cap_seconds: float = 30.0,
        read_timeout_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        daily_cap_backoff_seconds: float = 900.0,
        daily_cap_max_wait_seconds: float = 6 * 3600.0,
        token_budget_per_day: int | None = None,
        clock: Callable[[], float] = time.time,
        honor_caller_sampling: bool = True,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not 0.0 <= float(temperature) <= 1.0:
            raise ValueError("Bedrock temperature must be within [0, 1]")
        if top_p is not None and not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be within (0, 1]")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not (float(daily_cap_backoff_seconds) > 0 and float(daily_cap_max_wait_seconds) >= 0):
            raise ValueError("daily cap backoff must be positive and the max wait non-negative")
        if token_budget_per_day is not None and (type(token_budget_per_day) is not int or token_budget_per_day < 1):
            raise ValueError("token_budget_per_day must be a positive integer when given")
        self.model_id = model_id.removeprefix("bedrock:")
        self.region = region
        self.default_max_tokens = max_tokens
        self.default_temperature = float(temperature)
        # Anthropic models on Bedrock reject a request that sets both
        # temperature and top_p ("cannot both be specified for this model"),
        # so they never get topP. The requested value is still recorded in
        # the sampling provenance; the effective value is recorded as not sent.
        self.anthropic = "anthropic." in self.model_id
        self.default_top_p = None if (self.anthropic or top_p is None) else float(top_p)
        self.last_error: str | None = None
        self.failed_calls = 0
        self.system_prompt = system_prompt
        self._client = client
        self._read_timeout = read_timeout_seconds
        self.cost_ledger = cost_ledger if cost_ledger is not None else CostLedger()
        self.max_retries = max_retries
        self.backoff_base = float(backoff_base_seconds)
        self.backoff_cap = float(backoff_cap_seconds)
        # Daily cap handling: wait in long steps instead of failing the trial.
        self.daily_cap_backoff = float(daily_cap_backoff_seconds)
        self.daily_cap_max_wait = float(daily_cap_max_wait_seconds)
        self.daily_cap_waits = 0
        self.daily_cap_wait_seconds = 0.0
        # Optional pacing under a known cap: a sliding 24 hour window of tokens.
        self.token_budget_per_day = token_budget_per_day
        self._clock = clock
        # Direct callers (the judge, tests) mean the values they pass. Inside an
        # END-Bench run the caller is Concordia's act component asking for its
        # own defaults on every call (temperature 1.0, top_p 0.95, 2,200
        # tokens); the run builds the adapter with honor_caller_sampling=False
        # so the run spec's settings are what gets sent and the caller's values
        # are recorded only as requested.
        self.honor_caller_sampling = bool(honor_caller_sampling)
        self._token_window: deque[tuple[float, int]] = deque()
        self.budget_waits = 0
        self.budget_wait_seconds = 0.0
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self._call_count = 0
        self._last_sampling_config: dict[str, Any] = {}
        self._last_prompt: str | None = None
        self._last_usage: dict[str, int] = {}
        # Attributes the research runner reads for provenance.
        self.tokenizer = None
        self.layers_to_capture: list[int] = []

    # -- identity -------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"bedrock:{self.model_id}"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def activation_dim(self) -> int:
        return 0

    def get_last_sampling_config(self) -> dict[str, Any]:
        return dict(self._last_sampling_config)

    def capture_followup_activations(self, original_prompt: str, response: str) -> dict:
        del original_prompt, response
        return {}

    # -- transport --------------------------------------------------------------

    def _client_or_default(self):
        if self._client is None:
            self._client = _default_client(self.region, read_timeout=self._read_timeout)
        return self._client

    def _converse(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float | None) -> tuple[dict[str, Any], int]:
        inference: dict[str, Any] = {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if top_p is not None:
            inference["topP"] = float(top_p)
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": inference,
        }
        if self.system_prompt:
            request["system"] = [{"text": self.system_prompt}]
        attempt = 0
        waited_for_cap = 0.0
        while True:
            try:
                return self._client_or_default().converse(**request), attempt
            except Exception as exc:  # noqa: BLE001 - classified below
                if _is_daily_cap(exc):
                    if waited_for_cap >= self.daily_cap_max_wait:
                        self.failed_calls += 1
                        self.last_error = f"{type(exc).__name__}: {_error_message(exc)}"
                        raise BedrockGenerationError(
                            f"Bedrock converse failed for {self.model_id}: the daily token cap "
                            f"did not lift within {self.daily_cap_max_wait:.0f}s: {_error_message(exc)}"
                        ) from exc
                    delay = self.daily_cap_backoff * (0.75 + 0.5 * self._rng.random())  # jitter [0.75, 1.25)
                    self.daily_cap_waits += 1
                    self.daily_cap_wait_seconds += delay
                    waited_for_cap += delay
                    self._sleep(delay)
                    continue  # a cap wait does not count against the short retries
                if not _is_retryable(exc) or attempt >= self.max_retries:
                    self.failed_calls += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    raise BedrockGenerationError(
                        f"Bedrock converse failed for {self.model_id} after "
                        f"{attempt} retries: {type(exc).__name__}: {exc}"
                    ) from exc
                delay = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
                delay = delay * (0.5 + self._rng.random())  # jitter in [0.5, 1.5)
                self._sleep(delay)
                attempt += 1

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
    def _extract_text(response: Mapping[str, Any]) -> str:
        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, Mapping) else {}
        content = message.get("content", []) if isinstance(message, Mapping) else []
        parts = [
            str(block["text"]) for block in content
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        return "".join(parts)

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
        del timeout  # read timeout is a client property
        self._call_count += 1
        self._last_prompt = prompt
        requested_max_tokens = max_tokens
        use_caller = self.honor_caller_sampling
        effective_max_tokens = (
            int(max_tokens) if (use_caller and max_tokens is not None) else self.default_max_tokens
        )
        effective_temperature = (
            float(temperature) if (use_caller and temperature is not None) else self.default_temperature
        )
        effective_temperature = min(max(effective_temperature, 0.0), 1.0)
        requested_top_p = float(top_p) if top_p is not None else (self.default_top_p if self.default_top_p is not None else 1.0)
        effective_top_p = (
            None if self.anthropic  # None omits topP
            else (float(top_p) if (use_caller and top_p is not None) else self.default_top_p)
        )
        self._respect_token_budget()
        cap_waits_before = self.daily_cap_waits
        started = time.monotonic()
        response, retries = self._converse(
            prompt,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
            top_p=effective_top_p,
        )
        latency_ms = (time.monotonic() - started) * 1000.0
        text = self._extract_text(response)
        terminator_hit: str | None = None
        for terminator in terminators or ():
            if terminator and terminator in text:
                text = text.split(terminator, 1)[0]
                terminator_hit = terminator
                break
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        input_tokens = int(usage.get("inputTokens", 0) or 0)
        output_tokens = int(usage.get("outputTokens", 0) or 0)
        stop_reason = response.get("stopReason") if isinstance(response, Mapping) else None
        self._last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        # Provenance: requested values are what the caller asked for; the
        # effective record states what Bedrock was sent. Seeds and top_k are
        # not exposed by Converse and are recorded as unavailable.
        from interpretability.evaluation import (  # heavy import, kept lazy
            _publish_scoped_generation,
            _sampling_provenance,
        )
        from interpretability.runtime.model_call import CaptureMode

        provenance = _sampling_provenance(
            requested_max_tokens=requested_max_tokens,
            effective_max_tokens=effective_max_tokens,
            temperature=effective_temperature if temperature is None else float(temperature),
            # When topP is not sent (Anthropic), Bedrock's default of no
            # nucleus truncation applies: requested 1.0, effective recorded as
            # None (not sent).
            top_p=requested_top_p,
            top_k=int(top_k) if top_k is not None else 0,
            seed=seed,
            effective_temperature=effective_temperature,
            effective_top_p=effective_top_p,
            effective_top_k=None,
            effective_do_sample=effective_temperature > 0.0,
            generation_path=GENERATION_PATH,
            fallback_used=False,
            fallback_reason=None,
        )
        provenance["effective"]["seed"] = None
        provenance["provider"] = "bedrock"
        provenance["model_id"] = self.model_id
        provenance["usage"] = dict(self._last_usage)
        provenance["stop_reason"] = stop_reason
        provenance["retries"] = retries
        provenance["daily_cap_waits"] = self.daily_cap_waits - cap_waits_before
        provenance["token_budget_per_day"] = self.token_budget_per_day
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
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            retries=retries,
            call_id=record.call_id if record is not None else None,
            stop_reason=str(stop_reason) if stop_reason is not None else None,
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
    "GENERATION_PATH",
    "RETRYABLE_ERROR_CODES",
    "BedrockGenerationError",
    "BedrockTextModel",
]
