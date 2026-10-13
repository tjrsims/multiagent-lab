"""Model roster: which models the benchmark runs on, with routes, tiers and peer evidence.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
The roster is data, versioned, and separate from the adapters. Each entry names
a model once (``key``), its vendor and tier, whether its weights are open, the
peer benchmarks that evaluated it (so the choice is defensible in a paper), and
one or more *routes*: full model specs such as ``bedrock:global.openai.gpt-5.6-sol``
or ``xai:grok-4.3``. The first route is preferred; ``@key`` in any CLI model
option resolves to it, ``@key/<scheme>`` picks another route. A route is marked
``verified`` only when its id was read from the provider's own listing on the
date in ``MODEL_ROSTER_VERSION``; unverified ids must be checked in the console
before a paid run. The ``standard`` flag marks the headline leaderboard set.

Peer evidence (October 2026): DecepEval (agent deception, arXiv 2610.07967),
SWE-DeceptionBench (arXiv 2610.13626), KnownLieBench (arXiv 2608.26372),
TERMS-Bench (bilateral price negotiation, arXiv 2605.13909), PieArena
(negotiation, arXiv 2602.05302) and the MASK honesty leaderboard
(labs.scale.com/leaderboard/mask). The standard set is the union of models two
or more of those evaluate, plus one model per tier for every major vendor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Mapping

from end_bench.models.pricing import resolve_price
from end_bench.models.providers import PROVIDERS, ModelSpecError, parse_model_spec

__all__ = [
    "MODEL_ROSTER",
    "MODEL_ROSTER_VERSION",
    "PEER_BENCHMARKS",
    "TIERS",
    "RosterEntry",
    "Route",
    "format_roster",
    "resolve_model_spec",
    "roster_json",
    "standard_roster",
]

MODEL_ROSTER_VERSION = "end-bench-roster/2026-10-13"
TIERS = ("frontier", "mid", "budget")

# name -> (url, note). Which models each evaluated is recorded on the entries.
PEER_BENCHMARKS: Mapping[str, tuple[str, str]] = {
    "DecepEval": ("https://arxiv.org/abs/2610.07967", "deception propensity of LLM agents, 1,532 paired tasks, 9 models"),
    "SWE-DeceptionBench": ("https://arxiv.org/abs/2610.13626", "deception across the software lifecycle, 7 models"),
    "KnownLieBench": ("https://arxiv.org/abs/2608.26372", "incentivized lies by customer service agents, 18 models"),
    "TERMS-Bench": ("https://arxiv.org/abs/2605.13909", "bilateral price negotiation diagnosis, 13 agents"),
    "PieArena": ("https://arxiv.org/abs/2602.05302", "MBA course negotiation scenarios, 13 agents"),
    "MASK": ("https://labs.scale.com/leaderboard/mask", "honesty under pressure, 67 models"),
}

_BEDROCK_SNAPSHOT = "id read from the Bedrock model id listing dated 2026-10-10"


@dataclass(frozen=True)
class Route:
    spec: str                 # full model spec, e.g. "bedrock:global.openai.gpt-5.6-sol"
    verified: bool            # id read from the provider's own listing on the roster date
    note: str = ""

    def __post_init__(self) -> None:
        parse_model_spec(self.spec)  # fails closed on an unknown scheme

    @property
    def scheme(self) -> str:
        return parse_model_spec(self.spec)[0].scheme

    @property
    def priced(self) -> bool:
        return resolve_price(self.spec) is not None


@dataclass(frozen=True)
class RosterEntry:
    key: str
    display: str
    vendor: str
    tier: str
    routes: tuple[Route, ...]
    standard: bool = False
    open_weights: bool = False
    peer_benchmarks: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if not self.routes:
            raise ValueError(f"roster entry {self.key!r} needs at least one route")
        unknown = set(self.peer_benchmarks) - set(PEER_BENCHMARKS)
        if unknown:
            raise ValueError(f"unknown peer benchmarks on {self.key!r}: {sorted(unknown)}")

    @property
    def preferred(self) -> Route:
        return self.routes[0]

    def route_for(self, scheme: str) -> Route:
        for route in self.routes:
            if route.scheme == scheme:
                return route
        raise ModelSpecError(
            f"roster entry {self.key!r} has no {scheme!r} route; routes: {[r.spec for r in self.routes]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "display": self.display, "vendor": self.vendor, "tier": self.tier,
            "standard": self.standard, "open_weights": self.open_weights,
            "peer_benchmarks": list(self.peer_benchmarks), "notes": self.notes,
            "routes": [{"spec": r.spec, "scheme": r.scheme, "verified": r.verified, "priced": r.priced, "note": r.note}
                       for r in self.routes],
        }


_ENTRIES = (
    # ---- frontier -----------------------------------------------------------
    RosterEntry("gpt-5.6-sol", "GPT-5.6 Sol", "OpenAI", "frontier", standard=True,
                peer_benchmarks=("DecepEval",),
                routes=(Route("bedrock:global.openai.gpt-5.6-sol", True, _BEDROCK_SNAPSHOT),
                        Route("openai:gpt-5.6-sol", True, "id from OpenAI pricing summaries, 2026-10-13"))),
    RosterEntry("claude-opus-5-5", "Claude Opus 5.5", "Anthropic", "frontier",
                routes=(Route("bedrock:global.anthropic.claude-opus-5-5", True, _BEDROCK_SNAPSHOT),),
                notes="Newest Claude on Bedrock in the snapshot; no peer benchmark has used it yet."),
    RosterEntry("claude-fable-5", "Claude Fable 5", "Anthropic", "frontier",
                routes=(Route("bedrock:global.anthropic.claude-fable-5", False,
                              "not in the 2026-10-10 Bedrock snapshot; confirm the id in the console first"),),
                notes="Anthropic's most capable public model per the planning note; access and id unconfirmed."),
    RosterEntry("grok-4.3", "Grok 4.3", "xAI", "frontier", standard=True,
                peer_benchmarks=("KnownLieBench",),
                routes=(Route("xai:grok-4.3", True, "id and price from docs.x.ai, 2026-10-13"),),
                notes="1M context; Bedrock carries 4.6 and 4.7, not 4.3."),
    RosterEntry("grok-4.7", "Grok 4.7", "xAI", "frontier",
                routes=(Route("bedrock:global.xai.grok-4.7", True, _BEDROCK_SNAPSHOT),
                        Route("xai:grok-4.7", True, "docs.x.ai, 2026-10-13"))),
    RosterEntry("gemini-3.1-pro", "Gemini 3.1 Pro", "Google", "frontier", standard=True,
                peer_benchmarks=("TERMS-Bench", "MASK"),
                routes=(Route("gemini:gemini-3.1-pro-preview", True, "id as listed on the MASK leaderboard and Google docs; price from ai.google.dev, 2026-10-13"),)),
    RosterEntry("nova-premier", "Amazon Nova Premier", "Amazon", "frontier",
                routes=(Route("bedrock:us.amazon.nova-premier-v1:0", False,
                              "not in the 2026-10-10 Bedrock snapshot; confirm with end-bench bedrock-models"),)),
    # ---- mid ----------------------------------------------------------------
    RosterEntry("gpt-5.6-terra", "GPT-5.6 Terra", "OpenAI", "mid", standard=True,
                peer_benchmarks=("DecepEval",),
                routes=(Route("bedrock:global.openai.gpt-5.6-terra", True, _BEDROCK_SNAPSHOT),
                        Route("openai:gpt-5.6-terra", True, "OpenAI pricing summaries, 2026-10-13"))),
    RosterEntry("gpt-5.5", "GPT-5.5", "OpenAI", "mid",
                peer_benchmarks=("SWE-DeceptionBench", "KnownLieBench", "TERMS-Bench"),
                routes=(Route("openai:gpt-5.5", False, "id as the peer benchmarks spell it; not in the Bedrock snapshot"),)),
    RosterEntry("claude-opus-4-8", "Claude Opus 4.8", "Anthropic", "mid", standard=True,
                peer_benchmarks=("SWE-DeceptionBench", "KnownLieBench"),
                routes=(Route("bedrock:global.anthropic.claude-opus-4-8", True, _BEDROCK_SNAPSHOT),)),
    RosterEntry("claude-sonnet-5", "Claude Sonnet 5", "Anthropic", "mid", standard=True,
                peer_benchmarks=("DecepEval", "KnownLieBench"),
                routes=(Route("bedrock:global.anthropic.claude-sonnet-5", True, _BEDROCK_SNAPSHOT),)),
    RosterEntry("gemini-3.5-flash", "Gemini 3.5 Flash", "Google", "mid", standard=True,
                peer_benchmarks=("SWE-DeceptionBench", "KnownLieBench"),
                routes=(Route("gemini:gemini-3.5-flash", False, "id as the peer benchmarks spell it; the Google pricing page read 2026-10-13 listed 3.6 and 3.8 Flash, confirm 3.5 is still served"),)),
    RosterEntry("deepseek-v4-pro", "DeepSeek V4 Pro", "DeepSeek", "mid", standard=True, open_weights=True,
                peer_benchmarks=("DecepEval", "SWE-DeceptionBench", "KnownLieBench", "TERMS-Bench"),
                routes=(Route("deepseek:deepseek-v4-pro", True, "id and price from api-docs.deepseek.com, 2026-10-13"),),
                notes="Bedrock carries only DeepSeek V3.1."),
    RosterEntry("mistral-large-3", "Mistral Large 3", "Mistral", "mid", standard=True, open_weights=True,
                routes=(Route("bedrock:mistral.mistral-large-3-675b-instruct", True, _BEDROCK_SNAPSHOT + "; priced"),
                        Route("mistral:mistral-large-latest", False, "platform alias; pin the dated id before a paid run"))),
    RosterEntry("nova-pro", "Amazon Nova Pro", "Amazon", "mid", standard=True,
                routes=(Route("bedrock:us.amazon.nova-pro-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),),
                notes="Recommended fixed counterpart: cheap, verified price, outside the families under test."),
    # ---- budget -------------------------------------------------------------
    RosterEntry("gpt-5.6-luna", "GPT-5.6 Luna", "OpenAI", "budget", standard=True,
                routes=(Route("bedrock:global.openai.gpt-5.6-luna", True, _BEDROCK_SNAPSHOT),
                        Route("openai:gpt-5.6-luna", True, "OpenAI pricing summaries, 2026-10-13"))),
    RosterEntry("claude-haiku-4-5", "Claude Haiku 4.5", "Anthropic", "budget", standard=True,
                peer_benchmarks=("DecepEval",),
                routes=(Route("bedrock:global.anthropic.claude-haiku-4-5", False,
                              "profiles only in the snapshot; confirm the exact dated id with end-bench bedrock-models"),)),
    RosterEntry("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "Google", "budget",
                peer_benchmarks=("MASK",),
                routes=(Route("gemini:gemini-3.1-flash-lite-preview", True, "id as on the MASK leaderboard; price from ai.google.dev, 2026-10-13"),)),
    RosterEntry("deepseek-flash", "DeepSeek V4 Flash", "DeepSeek", "budget", standard=True, open_weights=True,
                peer_benchmarks=("DecepEval",),
                routes=(Route("deepseek:deepseek-flash", True, "api-docs.deepseek.com, 2026-10-13 (served by V4.1 Flash)"),)),
    RosterEntry("mistral-small-4", "Mistral Small 4", "Mistral", "budget", open_weights=True,
                routes=(Route("mistral:mistral-small-latest", False, "platform alias; pin the dated id before a paid run"),)),
    RosterEntry("nova-lite", "Amazon Nova Lite", "Amazon", "budget",
                routes=(Route("bedrock:us.amazon.nova-lite-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),)),
    RosterEntry("nova-micro", "Amazon Nova Micro", "Amazon", "budget",
                routes=(Route("bedrock:us.amazon.nova-micro-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),)),
    # ---- open weights (tier by price) -----------------------------------------
    RosterEntry("glm-5.3", "GLM 5.3", "Z.ai", "mid", standard=True, open_weights=True,
                peer_benchmarks=("DecepEval",),
                routes=(Route("bedrock:global.zai.glm-5.3", True, _BEDROCK_SNAPSHOT + "; enterprise eligibility gate"),
                        Route("zai:glm-5.2", True, "api.z.ai id from the Z.ai docs summaries, 2026-10-13; GLM 5.2 on the vendor API")),
                notes="DecepEval used GLM 5.3 Flash; SWE-DeceptionBench and KnownLieBench used GLM 5.2."),
    RosterEntry("kimi-k2.6", "Kimi K2.6", "Moonshot AI", "mid", standard=True, open_weights=True,
                peer_benchmarks=("SWE-DeceptionBench", "TERMS-Bench"),
                routes=(Route("moonshot:kimi-k2.6", True, "api.moonshot.ai id from the Moonshot docs summaries, 2026-10-13"),
                        Route("bedrock:moonshot.kimi-k2-thinking", True, _BEDROCK_SNAPSHOT + "; older K2 Thinking"))),
    RosterEntry("qwen3.6-plus", "Qwen 3.6 Plus", "Alibaba", "mid", standard=True, open_weights=True,
                peer_benchmarks=("TERMS-Bench",),
                routes=(Route("qwen:qwen3.6-plus", True, "DashScope compatible mode id from the Alibaba docs summaries, 2026-10-13"),
                        Route("bedrock:qwen.qwen3-235b-a22b-2507-v1:0", True, _BEDROCK_SNAPSHOT + "; older Qwen3 235B"))),
    RosterEntry("llama-4-maverick", "Llama 4 Maverick", "Meta", "mid", standard=True, open_weights=True,
                peer_benchmarks=("MASK",),
                routes=(Route("bedrock:us.meta.llama4-maverick-17b-instruct-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),)),
    RosterEntry("llama-4-scout", "Llama 4 Scout", "Meta", "budget", open_weights=True,
                routes=(Route("bedrock:us.meta.llama4-scout-17b-instruct-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),)),
    RosterEntry("gemma-3-12b", "Gemma 3 12B", "Google", "budget", open_weights=True,
                peer_benchmarks=("KnownLieBench",),
                routes=(Route("bedrock:google.gemma-3-12b-it", True, _BEDROCK_SNAPSHOT),),
                notes="Gemma 4 was not in the Bedrock snapshot."),
)
MODEL_ROSTER: Mapping[str, RosterEntry] = {entry.key: entry for entry in _ENTRIES}
if len(MODEL_ROSTER) != len(_ENTRIES):
    raise RuntimeError("roster keys must be unique")


def standard_roster() -> tuple[RosterEntry, ...]:
    return tuple(entry for entry in _ENTRIES if entry.standard)


def resolve_model_spec(value: str) -> str:
    """``@key`` or ``@key/<scheme>`` -> the route's full spec; anything else passes through.

    A plain spec is validated against the provider registry so a typo fails
    before a run directory is created.
    """
    if not isinstance(value, str) or not value.strip():
        raise ModelSpecError("model spec must be non empty text")
    text = value.strip()
    if not text.startswith("@"):
        parse_model_spec(text)
        return text
    key, _, scheme = text[1:].partition("/")
    entry = MODEL_ROSTER.get(key)
    if entry is None:
        raise ModelSpecError(f"unknown roster key {key!r}; known keys: {', '.join(MODEL_ROSTER)}")
    route = entry.route_for(scheme) if scheme else entry.preferred
    return route.spec


def format_roster(entries: Iterable[RosterEntry]) -> str:
    """One line per entry: key, tier, vendor, preferred route, id status, price, credentials."""
    lines = [f"{'key':22s} {'tier':9s} {'vendor':12s} {'preferred route':56s} {'id':10s} {'priced':6s} creds  peer evidence"]
    for entry in entries:
        route = entry.preferred
        provider = PROVIDERS[route.scheme]
        star = "*" if entry.standard else " "
        lines.append(
            f"{star}{entry.key:21s} {entry.tier:9s} {entry.vendor[:12]:12s} {route.spec[:56]:56s} "
            f"{'verified' if route.verified else 'CHECK':10s} {'yes' if route.priced else 'no':6s} "
            f"{provider.credentials_status():6s} {', '.join(entry.peer_benchmarks) or '-'}"
        )
    lines.append("")
    lines.append("* = standard set. id CHECK = the id was not read from the provider's own listing; confirm it before a paid run.")
    lines.append("creds: set/missing = the provider's key variable is present; aws = boto3 credentials apply. Values are never read.")
    return "\n".join(lines)


def roster_json(entries: Iterable[RosterEntry]) -> str:
    return json.dumps({
        "roster_version": MODEL_ROSTER_VERSION,
        "peer_benchmarks": {name: {"url": url, "note": note} for name, (url, note) in PEER_BENCHMARKS.items()},
        "entries": [entry.to_dict() for entry in entries],
    }, indent=2, sort_keys=True)
