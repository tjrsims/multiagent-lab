"""Bedrock prices, cost estimation, and a per-call cost ledger.

Prices are USD per one million tokens (input, output) for on-demand, standard
(non-batch, non-priority) inference in us-east-1. They are a *table*,
versioned, and overridable from a JSON file so a run manifest can state exactly
which prices produced its cost figures. Unknown model ids yield ``None``; the
ledger records the tokens and leaves the dollar figure empty rather than
guessing.

Provenance (``PRICE_SOURCES``): Meta, Mistral and Amazon Nova entries were
read from the AWS Price List API (ServiceCode ``AmazonBedrock``, regionCode
``us-east-1``, feature "On-demand Inference", inferenceType "Input tokens" /
"Output tokens") on 2026-10-09; ``end-bench prices --verify`` re-checks them.
The Price List API and the static pricing page publish no numbers for current
Anthropic models on Bedrock, so those entries carry Anthropic's first-party
list prices and are marked ``unverified``; the ledger reports how many dollars
rest on unverified prices so a cost claim can say so.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping


# Bedrock entries date from 2026-10-09; the other providers were added 2026-10-13.
PRICE_TABLE_VERSION = "end-bench-prices/2026-10-13"

_PRICELIST = "verified: AWS Price List API, AmazonBedrock us-east-1 on-demand standard, 2026-10-09"
_ANTHROPIC = (
    "unverified: Anthropic first-party list price; Bedrock does not publish current "
    "Anthropic prices via the Price List API or the static pricing page"
)
PRICE_SOURCES: Mapping[str, str] = {
    "anthropic.claude-sonnet-4-6": _ANTHROPIC,
    "anthropic.claude-sonnet-4-5": _ANTHROPIC,
    "anthropic.claude-haiku-4-5": _ANTHROPIC,
    "anthropic.claude-opus-4-5": _ANTHROPIC,
    "anthropic.claude-opus-4-6": _ANTHROPIC,
    "anthropic.claude-opus-4-8": _ANTHROPIC,
    "meta.llama4-maverick": _PRICELIST,
    "meta.llama4-scout": _PRICELIST,
    "mistral.mistral-large-3": _PRICELIST,
    "amazon.nova-pro": _PRICELIST,
    "amazon.nova-lite": _PRICELIST,
    "amazon.nova-micro": _PRICELIST,
    "amazon.nova-premier": "unverified: not present in the us-east-1 Price List dump",
    "scripted": "offline: free by definition",
    # Other providers, keyed by scheme. "verified" means read from the vendor's
    # own pricing page on the date given; "unverified" means a third party summary.
    "xai:grok-4.3": "verified: docs.x.ai models page read 2026-10-13 (prompts under 200k tokens)",
    "xai:grok-4.5": "verified: docs.x.ai models page read 2026-10-13 (prompts under 200k tokens)",
    "xai:grok-4.6": "verified: docs.x.ai models page read 2026-10-13 (prompts under 200k tokens)",
    "xai:grok-4.7": "verified: docs.x.ai models page read 2026-10-13 (prompts under 200k tokens)",
    "gemini:gemini-3.1-pro": "verified: ai.google.dev pricing page read 2026-10-13 (prompts up to 200k tokens)",
    "gemini:gemini-3.1-flash-lite": "verified: ai.google.dev pricing page read 2026-10-13",
    "gemini:gemini-3.5-flash-lite": "verified: ai.google.dev pricing page read 2026-10-13",
    "gemini:gemini-3.5-flash": "unverified: third party summaries dated 2026-08-25; absent from the Google pricing page read 2026-10-13",
    "gemini:gemini-2.5-pro": "verified: ai.google.dev pricing page read 2026-10-13 (prompts up to 200k tokens)",
    "gemini:gemini-2.5-flash": "verified: ai.google.dev pricing page read 2026-10-13",
    "deepseek:deepseek-v4-pro": "verified: api-docs.deepseek.com pricing read 2026-10-13, peak rate for cache misses (off peak is half)",
    "deepseek:deepseek-flash": "verified: api-docs.deepseek.com pricing read 2026-10-13, peak rate for cache misses (off peak is half)",
    "openai:gpt-5.6-sol": "unverified: third party summaries of the OpenAI list price read 2026-10-13 disagree (4/20 vs 5/30); the higher is kept",
    "openai:gpt-5.6-terra": "unverified: third party summaries of the OpenAI list price read 2026-10-13",
    "openai:gpt-5.6-luna": "unverified: third party summaries of the OpenAI list price read 2026-10-13",
    "openai:gpt-5.5": "unverified: third party summaries of the OpenAI list price read 2026-10-13",
    "mistral:mistral-large-latest": "unverified: third party summaries read 2026-10-13; equals the Bedrock Price List for Mistral Large 3",
    "mistral:mistral-medium-latest": "unverified: third party summaries read 2026-10-13",
    "mistral:mistral-small-latest": "unverified: third party summaries read 2026-10-13",
    # Paper set additions, 2026-10-13.
    "anthropic.claude-fable-5-1": "unverified: Anthropic list price (platform.claude.com read 2026-10-13); the Bedrock us. profile is reported 10% higher",
    "anthropic.claude-opus-5-5": "unverified: Anthropic list price (platform.claude.com read 2026-10-13)",
    "anthropic.claude-sonnet-5-5": "unverified: Anthropic list price (platform.claude.com read 2026-10-13)",
    "anthropic.claude-haiku-5-5": "unverified: Anthropic list price 'from' rate (platform.claude.com read 2026-10-13)",
    "openai:gpt-6-astra": "unverified: OpenAI docs listing and third party summaries read 2026-10-13",
    "openai:gpt-6.1-sol": "unverified: third party summaries of the 2026 price cut read 2026-10-13",
    "openai:gpt-6-sol": "unverified: third party summaries of the 2026 price cut read 2026-10-13",
    "openai:gpt-6-luna": "unverified: third party summaries of the 2026 price cut read 2026-10-13",
    "gemini:gemini-3.8-flash": "verified: ai.google.dev pricing page read 2026-10-13 (rate through 2026-12-31; doubles on 2027-01-01)",
    "gemini:gemini-3.6-flash": "verified: ai.google.dev pricing page read 2026-10-13 (rate through 2026-12-31; doubles on 2027-01-01)",
    "zai:glm-5.3": "unverified: third party summaries read 2026-10-13",
    "meta:muse-wonder-1.0": "unverified: Meta newsroom post as summarized by third parties, read 2026-10-13",
    "meta:muse-spark-1.3": "unverified: OpenRouter and third party listings read 2026-10-13",
    "openrouter:meta/muse-spark-1.3": "unverified: OpenRouter listing read 2026-10-13",
}

DEFAULT_PRICES_PER_MTOK: Mapping[str, tuple[float, float]] = {
    "anthropic.claude-sonnet-4-6": (3.00, 15.00),
    "anthropic.claude-sonnet-4-5": (3.00, 15.00),
    "anthropic.claude-haiku-4-5": (1.00, 5.00),
    "anthropic.claude-opus-4-5": (5.00, 25.00),
    "anthropic.claude-opus-4-6": (5.00, 25.00),
    "anthropic.claude-opus-4-8": (5.00, 25.00),
    "meta.llama4-maverick": (0.24, 0.97),
    "meta.llama4-scout": (0.17, 0.66),
    "mistral.mistral-large-3": (0.50, 1.50),
    "amazon.nova-pro": (0.80, 3.20),
    "amazon.nova-lite": (0.06, 0.24),
    "amazon.nova-micro": (0.035, 0.14),
    "amazon.nova-premier": (2.50, 12.50),
    # Offline scripted model: free by definition, so smoke runs are "priced".
    "scripted": (0.0, 0.0),
    # Other providers (USD per 1M tokens, input, output); see PRICE_SOURCES.
    "xai:grok-4.3": (1.25, 2.50),
    "xai:grok-4.5": (2.00, 6.00),
    "xai:grok-4.6": (2.00, 6.00),
    "xai:grok-4.7": (2.00, 6.00),
    "gemini:gemini-3.1-pro": (2.00, 12.00),
    "gemini:gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini:gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini:gemini-3.5-flash": (1.50, 9.00),
    "gemini:gemini-2.5-pro": (1.25, 10.00),
    "gemini:gemini-2.5-flash": (0.30, 2.50),
    "deepseek:deepseek-v4-pro": (1.32, 3.96),
    "deepseek:deepseek-flash": (0.30, 1.20),
    "openai:gpt-5.6-sol": (5.00, 30.00),
    "openai:gpt-5.6-terra": (2.00, 12.00),
    "openai:gpt-5.6-luna": (0.20, 1.20),
    "openai:gpt-5.5": (5.00, 30.00),
    "mistral:mistral-large-latest": (0.50, 1.50),
    "mistral:mistral-medium-latest": (1.50, 7.50),
    "mistral:mistral-small-latest": (0.15, 0.60),
    # Paper set additions, 2026-10-13 (USD per 1M tokens, input, output).
    "anthropic.claude-fable-5-1": (10.00, 50.00),
    "anthropic.claude-opus-5-5": (4.00, 20.00),
    "anthropic.claude-sonnet-5-5": (2.00, 10.00),
    "anthropic.claude-haiku-5-5": (0.10, 0.50),
    "openai:gpt-6-astra": (10.00, 50.00),
    "openai:gpt-6.1-sol": (2.00, 10.00),
    "openai:gpt-6-sol": (2.00, 10.00),
    "openai:gpt-6-luna": (0.10, 0.50),
    "gemini:gemini-3.8-flash": (0.75, 3.75),
    "gemini:gemini-3.6-flash": (0.75, 3.75),
    "zai:glm-5.3": (1.40, 4.40),
    "meta:muse-wonder-1.0": (4.00, 20.00),
    "meta:muse-spark-1.3": (1.25, 4.25),
    "openrouter:meta/muse-spark-1.3": (1.25, 4.25),
}


def load_price_table(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load ``{model_id_prefix: [input, output]}`` from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, tuple[float, float]] = {}
    for key, value in payload.items():
        if (
            not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in value)
        ):
            raise ValueError(f"price table entry {key!r} must be [input, output] >= 0")
        table[str(key)] = (float(value[0]), float(value[1]))
    return table


INFERENCE_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "global.", "us-gov.", "jp.", "au.", "ca.")


def normalize_model_id(model_id: str) -> str:
    """Strip our ``bedrock:`` scheme and Bedrock's regional inference-profile prefix.

    On-demand Claude (and some other) models are invoked through inference
    profiles such as ``us.anthropic.claude-sonnet-4-6-...``; the price is the
    foundation model's, so pricing matches on the id without that prefix.
    """
    normalized = model_id.strip()
    if normalized.startswith("bedrock:"):
        normalized = normalized[len("bedrock:"):]
    scheme, sep, _rest = normalized.partition(":")
    if sep and scheme.isalpha() and scheme != "bedrock":
        # Another provider's id keeps its scheme ("xai:grok-4.3") so the table
        # keys on it; it is never matched against the Bedrock prices. Bedrock
        # ids with a colon ("amazon.nova-pro-v1:0") have a dot before it.
        return normalized
    for prefix in INFERENCE_PROFILE_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


def resolve_price(
    model_id: str,
    table: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[float, float] | None:
    """Longest-prefix match of a Bedrock model id against the table."""
    prices = DEFAULT_PRICES_PER_MTOK if table is None else table
    normalized = normalize_model_id(model_id)
    candidates = [key for key in prices if normalized.startswith(key)]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return prices[best]


def price_source(
    model_id: str,
    table: Mapping[str, tuple[float, float]] | None = None,
) -> str | None:
    """Provenance string for the price a model id resolves to (``None`` if unpriced)."""
    if table is not None:
        return "custom table" if resolve_price(model_id, table) is not None else None
    normalized = normalize_model_id(model_id)
    candidates = [key for key in DEFAULT_PRICES_PER_MTOK if normalized.startswith(key)]
    if not candidates:
        return None
    return PRICE_SOURCES.get(max(candidates, key=len), "unverified: no source recorded")


def price_is_verified(model_id: str, table: Mapping[str, tuple[float, float]] | None = None) -> bool:
    source = price_source(model_id, table)
    return bool(source) and source.startswith(("verified", "offline"))


def estimate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    table: Mapping[str, tuple[float, float]] | None = None,
) -> float | None:
    """USD for one call, or ``None`` when the model is not in the table."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    price = resolve_price(model_id, table)
    if price is None:
        return None
    return input_tokens / 1_000_000 * price[0] + output_tokens / 1_000_000 * price[1]


def estimate_design_cost(
    model_id: str,
    *,
    negotiations: int,
    calls_per_negotiation: int = 10,
    input_tokens_per_call: int = 1500,
    output_tokens_per_call: int = 150,
    table: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Back-of-envelope budget for an experimental design."""
    if negotiations < 0 or calls_per_negotiation < 0:
        raise ValueError("counts must be non-negative")
    calls = negotiations * calls_per_negotiation
    per_call = estimate_cost(model_id, input_tokens_per_call, output_tokens_per_call, table)
    return {
        "model_id": model_id,
        "negotiations": negotiations,
        "calls": calls,
        "input_tokens": calls * input_tokens_per_call,
        "output_tokens": calls * output_tokens_per_call,
        "usd_per_negotiation": (
            None if per_call is None else per_call * calls_per_negotiation
        ),
        "usd_total": None if per_call is None else per_call * calls,
        "price_table_version": PRICE_TABLE_VERSION if table is None else "custom",
        "price_source": price_source(model_id, table),
    }


@dataclass
class CostLedger:
    """Append-only JSONL ledger of model calls with running totals."""

    path: Path | None = None
    table: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usd = 0.0
        self.unverified_usd = 0.0
        self.unpriced_calls = 0
        if self.path is not None:
            self.path = Path(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        retries: int,
        call_id: str | None,
        stop_reason: str | None,
    ) -> dict[str, Any]:
        usd = estimate_cost(model_id, input_tokens, output_tokens, self.table)
        source = price_source(model_id, self.table)
        entry = {
            "price_source": source,
            "ts": time.time(),
            "model_id": model_id,
            "call_id": call_id,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "usd": usd,
            "latency_ms": float(latency_ms),
            "retries": int(retries),
            "stop_reason": stop_reason,
            "price_table_version": PRICE_TABLE_VERSION if self.table is None else "custom",
        }
        with self._lock:
            self.calls += 1
            self.input_tokens += int(input_tokens)
            self.output_tokens += int(output_tokens)
            if usd is None:
                self.unpriced_calls += 1
            else:
                self.usd += usd
                if not price_is_verified(model_id, self.table):
                    self.unverified_usd += usd
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def totals(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": self.usd,
            "unverified_usd": self.unverified_usd,
            "unpriced_calls": self.unpriced_calls,
            "price_table_version": PRICE_TABLE_VERSION if self.table is None else "custom",
        }


__all__ = [
    "DEFAULT_PRICES_PER_MTOK",
    "INFERENCE_PROFILE_PREFIXES",
    "PRICE_SOURCES",
    "PRICE_TABLE_VERSION",
    "CostLedger",
    "price_is_verified",
    "price_source",
    "estimate_cost",
    "estimate_design_cost",
    "load_price_table",
    "normalize_model_id",
    "resolve_price",
]
