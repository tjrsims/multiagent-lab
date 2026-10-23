"""Model roster: which models the benchmark runs on, with routes, tiers and peer evidence.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
The roster is data, versioned, and separate from the adapters. Each entry names
a model once (``key``), its vendor and tier, whether its weights are open, the
peer benchmarks that evaluated it (so the choice is defensible in a paper), and
one or more *routes*: full model specs such as ``bedrock:global.openai.gpt-6-astra``
or ``xai:grok-4.7``. The first route is preferred; ``@key`` in any CLI model
option resolves to it, ``@key/<scheme>`` picks another route. A route is marked
``verified`` only when its id was read from the provider's own listing on the
date in ``MODEL_ROSTER_VERSION``; unverified ids must be checked in the console
before a paid run. The ``standard`` flag marks the living leaderboard set and ``paper`` the
conference run; ``sampling`` says whether the API accepts a temperature at all.

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

from end_bench.models.pricing import INFERENCE_PROFILE_PREFIXES, resolve_price
from end_bench.models.providers import PROVIDERS, ModelSpecError, parse_model_spec

__all__ = [
    "MODEL_ROSTER",
    "MODEL_ROSTER_VERSION",
    "PEER_BENCHMARKS",
    "TIERS",
    "RosterEntry",
    "Route",
    "REASONING_MODES",
    "SAMPLING_MODES",
    "format_roster",
    "paper_roster",
    "resolve_model_spec",
    "roster_json",
    "sampling_constraint",
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
_BEDROCK_CARD = "id read from the AWS Bedrock model card, 2026-10-13"
SAMPLING_MODES = ("free", "provider_default_recommended", "provider_default_only")
REASONING_MODES = ("none", "optional", "always_on")


@dataclass(frozen=True)
class Route:
    spec: str                 # full model spec, e.g. "bedrock:global.openai.gpt-6-astra"
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
    standard: bool = False        # the living leaderboard set
    paper: bool = False           # the conference run (a subset of standard)
    available: bool = True        # False: announced but not callable yet
    open_weights: bool = False
    # "provider_default_only": the API rejects any temperature or top_p, so a
    # run must leave sampling unset. "provider_default_recommended": the vendor
    # documents that other values degrade the model. Both are run the same way.
    sampling: str = "free"
    # "always_on": the model reasons before every answer and bills those tokens
    # as output; max_tokens must leave room for them or the visible reply is empty.
    reasoning: str = "none"
    released: str = ""            # ISO date when known
    peer_benchmarks: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if not self.routes:
            raise ValueError(f"roster entry {self.key!r} needs at least one route")
        if self.sampling not in SAMPLING_MODES:
            raise ValueError(f"sampling must be one of {SAMPLING_MODES}")
        if self.reasoning not in REASONING_MODES:
            raise ValueError(f"reasoning must be one of {REASONING_MODES}")
        if self.paper and not self.standard:
            raise ValueError(f"paper entry {self.key!r} must also be in the standard set")
        if self.paper and not self.available:
            raise ValueError(f"paper entry {self.key!r} must be available")
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
            "standard": self.standard, "paper": self.paper, "available": self.available,
            "open_weights": self.open_weights, "sampling": self.sampling, "reasoning": self.reasoning,
            "released": self.released, "peer_benchmarks": list(self.peer_benchmarks), "notes": self.notes,
            "routes": [{"spec": r.spec, "scheme": r.scheme, "verified": r.verified, "priced": r.priced, "note": r.note}
                       for r in self.routes],
        }


_ENTRIES = (
    # ================= the paper set (latest model of each line, October 13, 2026) =================
    RosterEntry("claude-fable-5-1", "Claude Fable 5.1", "Anthropic", "frontier", standard=True, paper=True,
                sampling="provider_default_only", reasoning="always_on", released="2026-09-01",
                routes=(Route("bedrock:global.anthropic.claude-fable-5-1", True, _BEDROCK_CARD + "; us. profile costs 10% more"),),
                notes="Bedrock requires the account data retention mode aws_review before any call. Converse accepts "
                      "temperature 1.0 or unset only. Classifier refusals return HTTP 200 with stopReason refusal and "
                      "are materially more frequent than on earlier Claude models; the adapter counts them. "
                      "Adaptive thinking is always on and billed as output; give max_tokens room for it."),
    RosterEntry("claude-opus-5-5", "Claude Opus 5.5", "Anthropic", "frontier", standard=True, paper=True,
                sampling="provider_default_only", reasoning="always_on", released="2026-09-22",
                routes=(Route("bedrock:global.anthropic.claude-opus-5-5", True, _BEDROCK_SNAPSHOT),),
                notes="Anthropic's most capable Opus; adaptive thinking always on (default effort medium)."),
    RosterEntry("claude-sonnet-5-5", "Claude Sonnet 5.5", "Anthropic", "mid", standard=True, paper=True,
                sampling="provider_default_only", reasoning="optional", released="2026-09-28",
                routes=(Route("bedrock:global.anthropic.claude-sonnet-5-5", True, _BEDROCK_SNAPSHOT),)),
    RosterEntry("gpt-6-astra", "GPT-6 Astra", "OpenAI", "frontier", standard=True, paper=True,
                sampling="provider_default_only", reasoning="optional", released="2026-09-03",
                routes=(Route("bedrock:global.openai.gpt-6-astra", True, "AWS Bedrock model card page and the inference profile id named in several SDK issues, 2026-10-13; profile only, the bare id is rejected"),
                        Route("openai:gpt-6-astra", True, "developers.openai.com model page, 2026-10-13")),
                notes="OpenAI's flagship. Converse and the OpenAI API reject temperature, top_p and stop sequences for it. "
                      "$10/$50 per million; long context pricing above 272k input tokens does not apply here."),
    RosterEntry("gpt-6.1-sol", "GPT-6.1 Sol", "OpenAI", "mid", standard=True, paper=True,
                sampling="provider_default_only", reasoning="optional", released="2026-10-01",
                routes=(Route("bedrock:global.openai.gpt-6.1-sol", True, "AWS weekly roundup of 2026-10-05 and an aws-samples issue naming the global profile"),
                        Route("openai:gpt-6.1-sol", True, "third party pricing pages naming the id, 2026-10-13")),
                notes="Upgrade of GPT-6 Sol; now the mid tier at $2/$10. Rejects temperature and top_p."),
    RosterEntry("gemini-3.1-pro", "Gemini 3.1 Pro", "Google", "frontier", standard=True, paper=True,
                sampling="provider_default_recommended", reasoning="optional",
                peer_benchmarks=("TERMS-Bench", "MASK"),
                routes=(Route("gemini:gemini-3.1-pro-preview", True, "id as listed on the MASK leaderboard and Google docs; price from ai.google.dev, 2026-10-13"),),
                notes="The current callable Pro. Gemini 3.5 Pro is partner only and Gemini 4 Argon is not callable (see gemini-4-argon). "
                      "Google advises leaving temperature at its default for Gemini 3."),
    RosterEntry("gemini-3.8-flash", "Gemini 3.8 Flash", "Google", "mid", standard=True, paper=True,
                sampling="provider_default_recommended", reasoning="optional", released="2026-09-02",
                routes=(Route("gemini:gemini-3.8-flash", True, "stable id in Google's OpenAI compatibility docs and pricing page, 2026-10-13"),),
                notes="Google's most capable Flash; $0.75/$3.75 through 2026-12-31, then double."),
    RosterEntry("deepseek-v4-pro", "DeepSeek V4 Pro", "DeepSeek", "mid", standard=True, paper=True, open_weights=True,
                sampling="provider_default_recommended", reasoning="optional",
                peer_benchmarks=("DecepEval", "SWE-DeceptionBench", "KnownLieBench", "TERMS-Bench"),
                routes=(Route("deepseek:deepseek-v4-pro", True, "id and price from api-docs.deepseek.com, 2026-10-13"),),
                notes="Bedrock carries only DeepSeek V3.1. Vendor recommends temperature 1.0. Peak price is used; off peak is half."),
    RosterEntry("grok-4.7", "Grok 4.7", "xAI", "frontier", standard=True, paper=True,
                reasoning="optional", released="2026-09-21",
                routes=(Route("xai:grok-4.7", True, "docs.x.ai models page, 2026-10-13"),
                        Route("bedrock:global.xai.grok-4.7", True, _BEDROCK_SNAPSHOT)),
                notes="Current Grok; 500k context; $2/$6."),
    RosterEntry("glm-5.3", "GLM 5.3", "Z.ai", "mid", standard=True, paper=True, open_weights=True,
                reasoning="optional", released="2026-08-14",
                peer_benchmarks=("DecepEval",),
                routes=(Route("zai:glm-5.3", True, "official API id per Z.ai release coverage, 2026-10-13"),
                        Route("bedrock:global.zai.glm-5.3", True, _BEDROCK_SNAPSHOT + "; eligible enterprise customers only")),
                notes="Coding first post training on the GLM 5.2 base; still a general chat model. DecepEval used GLM 5.3 Flash."),
    RosterEntry("muse-wonder-1.0", "Muse Wonder 1.0", "Meta", "frontier", standard=True, paper=True,
                reasoning="optional", released="2026-10-12",
                routes=(Route("meta:muse-wonder-1.0", True, "id from Meta's newsroom post and Model API docs, 2026-10-13"),),
                notes="Meta Superintelligence Labs' frontier agentic model, one day old at roster time. Meta Model API at "
                      "api.meta.ai; Muse Wonder Full is subscription only and not an API model."),
    # ================= standard set beyond the paper =================
    RosterEntry("gpt-6-luna", "GPT-6 Luna", "OpenAI", "budget", standard=True,
                sampling="provider_default_only", reasoning="optional",
                routes=(Route("bedrock:global.openai.gpt-6-luna", True, "AWS Bedrock model card page, 2026-10-13"),
                        Route("openai:gpt-6-luna", True, "third party pricing pages naming the id, 2026-10-13"))),
    RosterEntry("claude-haiku-5-5", "Claude Haiku 5.5", "Anthropic", "budget", standard=True,
                sampling="provider_default_only", reasoning="optional", released="2026-10-07",
                routes=(Route("bedrock:global.anthropic.claude-haiku-5-5", False, "released after the 2026-10-10 Bedrock snapshot; confirm the id with end-bench bedrock-models"),)),
    RosterEntry("deepseek-flash", "DeepSeek V4 Flash", "DeepSeek", "budget", standard=True, open_weights=True,
                sampling="provider_default_recommended", peer_benchmarks=("DecepEval",),
                routes=(Route("deepseek:deepseek-flash", True, "api-docs.deepseek.com, 2026-10-13 (served by V4.1 Flash)"),)),
    RosterEntry("mistral-large-3", "Mistral Large 3", "Mistral", "mid", standard=True, open_weights=True,
                routes=(Route("bedrock:mistral.mistral-large-3-675b-instruct", True, _BEDROCK_SNAPSHOT + "; priced"),
                        Route("mistral:mistral-large-latest", False, "platform alias; pin the dated id before a paid run"))),
    RosterEntry("nova-pro", "Amazon Nova Pro", "Amazon", "mid", standard=True,
                routes=(Route("bedrock:us.amazon.nova-pro-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),),
                notes="The fixed counterpart: cheap, verified price, outside the families under test."),
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
    # ================= announced, not callable =================
    RosterEntry("gemini-4-argon", "Gemini 4 Argon", "Google", "frontier", available=False,
                sampling="provider_default_recommended", reasoning="optional", released="2026-09-30",
                routes=(Route("gemini:gemini-4-argon", False, "announced 2026-09-30; callable only by Fairwind Program cyber defenders; paid API customers next with no date"),),
                notes="Not runnable yet. Gemini 3.1 Pro stands in for Google's Pro line until Argon reaches the API."),
    # ================= superseded, kept for comparability with peer papers =================
    RosterEntry("gpt-5.6-sol", "GPT-5.6 Sol", "OpenAI", "frontier", sampling="provider_default_only", reasoning="optional",
                peer_benchmarks=("DecepEval",),
                routes=(Route("bedrock:global.openai.gpt-5.6-sol", True, _BEDROCK_SNAPSHOT),
                        Route("openai:gpt-5.6-sol", True, "OpenAI pricing summaries, 2026-10-13")),
                notes="Superseded by GPT-6 Astra and GPT-6.1 Sol."),
    RosterEntry("gpt-5.6-terra", "GPT-5.6 Terra", "OpenAI", "mid", sampling="provider_default_only", reasoning="optional",
                peer_benchmarks=("DecepEval",),
                routes=(Route("bedrock:global.openai.gpt-5.6-terra", True, _BEDROCK_SNAPSHOT),
                        Route("openai:gpt-5.6-terra", True, "OpenAI pricing summaries, 2026-10-13"))),
    RosterEntry("gpt-5.5", "GPT-5.5", "OpenAI", "mid", sampling="provider_default_only", reasoning="optional",
                peer_benchmarks=("SWE-DeceptionBench", "KnownLieBench", "TERMS-Bench"),
                routes=(Route("openai:gpt-5.5", False, "id as the peer benchmarks spell it; not in the Bedrock snapshot"),)),
    RosterEntry("claude-opus-4-8", "Claude Opus 4.8", "Anthropic", "mid",
                peer_benchmarks=("SWE-DeceptionBench", "KnownLieBench"),
                routes=(Route("bedrock:global.anthropic.claude-opus-4-8", True, _BEDROCK_SNAPSHOT),),
                notes="Superseded by Opus 5.5."),
    RosterEntry("claude-sonnet-5", "Claude Sonnet 5", "Anthropic", "mid",
                peer_benchmarks=("DecepEval", "KnownLieBench"),
                routes=(Route("bedrock:global.anthropic.claude-sonnet-5", True, _BEDROCK_SNAPSHOT),),
                notes="Superseded by Sonnet 5.5."),
    RosterEntry("claude-haiku-4-5", "Claude Haiku 4.5", "Anthropic", "budget",
                peer_benchmarks=("DecepEval",),
                routes=(Route("bedrock:global.anthropic.claude-haiku-4-5", False, "profiles only in the snapshot; confirm the exact dated id"),),
                notes="Superseded by Haiku 5.5."),
    RosterEntry("gemini-3.5-flash", "Gemini 3.5 Flash", "Google", "mid",
                sampling="provider_default_recommended",
                peer_benchmarks=("SWE-DeceptionBench", "KnownLieBench"),
                routes=(Route("gemini:gemini-3.5-flash", False, "id as the peer benchmarks spell it; absent from the Google pricing page read 2026-10-13"),),
                notes="Superseded by Gemini 3.8 Flash."),
    RosterEntry("grok-4.3", "Grok 4.3", "xAI", "mid", peer_benchmarks=("KnownLieBench",),
                routes=(Route("xai:grok-4.3", True, "docs.x.ai, 2026-10-13"),),
                notes="Superseded by Grok 4.7; still served at $1.25/$2.50."),
    RosterEntry("muse-spark-1.3", "Muse Spark 1.3", "Meta", "mid", released="2026-09-02",
                routes=(Route("meta:muse-spark-1.3", True, "Meta Model API docs example id, 2026-10-13"),
                        Route("openrouter:meta/muse-spark-1.3", True, "OpenRouter listing, 2026-10-13")),
                notes="Superseded by Muse Wonder 1.0."),
    RosterEntry("llama-4-scout", "Llama 4 Scout", "Meta", "budget", open_weights=True,
                routes=(Route("bedrock:us.meta.llama4-scout-17b-instruct-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),)),
    RosterEntry("gemma-3-12b", "Gemma 3 12B", "Google", "budget", open_weights=True,
                peer_benchmarks=("KnownLieBench",),
                routes=(Route("bedrock:google.gemma-3-12b-it", True, _BEDROCK_SNAPSHOT),),
                notes="Gemma 4 was not in the Bedrock snapshot."),
    RosterEntry("nova-lite", "Amazon Nova Lite", "Amazon", "budget",
                routes=(Route("bedrock:us.amazon.nova-lite-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),)),
    RosterEntry("nova-micro", "Amazon Nova Micro", "Amazon", "budget",
                routes=(Route("bedrock:us.amazon.nova-micro-v1:0", True, _BEDROCK_SNAPSHOT + "; priced"),)),
    RosterEntry("nova-premier", "Amazon Nova Premier", "Amazon", "frontier",
                routes=(Route("bedrock:us.amazon.nova-premier-v1:0", False, "not in the 2026-10-10 Bedrock snapshot; confirm with end-bench bedrock-models"),)),
    RosterEntry("mistral-small-4", "Mistral Small 4", "Mistral", "budget", open_weights=True,
                routes=(Route("mistral:mistral-small-latest", False, "platform alias; pin the dated id before a paid run"),)),
    RosterEntry("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "Google", "budget", peer_benchmarks=("MASK",),
                sampling="provider_default_recommended",
                routes=(Route("gemini:gemini-3.1-flash-lite-preview", True, "id as on the MASK leaderboard; price from ai.google.dev, 2026-10-13"),)),
)
MODEL_ROSTER: Mapping[str, RosterEntry] = {entry.key: entry for entry in _ENTRIES}
if len(MODEL_ROSTER) != len(_ENTRIES):
    raise RuntimeError("roster keys must be unique")

def canonical_spec(spec: str) -> str:
    """A Bedrock spec without its regional inference profile prefix; other specs unchanged.

    ``bedrock:us.anthropic.claude-fable-5-1`` and ``bedrock:global.anthropic.claude-fable-5-1``
    are the same model, so they share the roster entry and its sampling constraint.
    """
    text = str(spec).strip()
    if not text.startswith("bedrock:"):
        return text
    rest = text[len("bedrock:"):]
    for prefix in INFERENCE_PROFILE_PREFIXES:
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break
    return "bedrock:" + rest


_ROUTE_INDEX: Mapping[str, RosterEntry] = {
    route.spec: entry for entry in _ENTRIES for route in entry.routes
}
_CANONICAL_ROUTE_INDEX: Mapping[str, RosterEntry] = {
    canonical_spec(route.spec): entry for entry in _ENTRIES for route in entry.routes
}


def standard_roster() -> tuple[RosterEntry, ...]:
    return tuple(entry for entry in _ENTRIES if entry.standard)


def paper_roster() -> tuple[RosterEntry, ...]:
    return tuple(entry for entry in _ENTRIES if entry.paper)


def sampling_constraint(spec: str) -> str:
    """The sampling mode for a full model spec, ``"free"`` when the roster does not know it."""
    entry = _ROUTE_INDEX.get(str(spec).strip()) or _CANONICAL_ROUTE_INDEX.get(canonical_spec(spec))
    return entry.sampling if entry is not None else "free"


def resolve_model_spec(value: str) -> str:
    """``@key`` or ``@key/<scheme>`` -> the route's full spec; anything else passes through.

    A plain spec is validated against the provider registry so a typo fails
    before a run directory is created. An entry that is announced but not
    callable yet fails closed with its note.
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
    if not entry.available:
        raise ModelSpecError(f"roster entry {key!r} is not callable yet: {entry.preferred.note}")
    route = entry.route_for(scheme) if scheme else entry.preferred
    return route.spec


def format_roster(entries: Iterable[RosterEntry]) -> str:
    """One line per entry: key, tier, vendor, preferred route, id status, price, credentials, sampling."""
    lines = [f"{'key':22s} {'tier':9s} {'vendor':12s} {'preferred route':52s} {'id':9s} {'priced':6s} {'creds':8s} {'sampling':10s} peer evidence"]
    for entry in entries:
        route = entry.preferred
        provider = PROVIDERS[route.scheme]
        mark = "P" if entry.paper else ("*" if entry.standard else ("!" if not entry.available else " "))
        sampling = {"free": "free", "provider_default_recommended": "default+", "provider_default_only": "DEFAULT"}[entry.sampling]
        lines.append(
            f"{mark}{entry.key:21s} {entry.tier:9s} {entry.vendor[:12]:12s} {route.spec[:52]:52s} "
            f"{'verified' if route.verified else 'CHECK':9s} {'yes' if route.priced else 'no':6s} "
            f"{provider.credentials_status():8s} {sampling:10s} {', '.join(entry.peer_benchmarks) or '-'}"
        )
    lines.append("")
    lines.append("P = paper set, * = standard set, ! = announced but not callable. id CHECK = not read from the provider's own listing; confirm before a paid run.")
    lines.append("sampling DEFAULT = the API rejects any temperature (leave it unset); default+ = the vendor advises its default. creds: set/missing = key variable present; aws = boto3 credentials.")
    return "\n".join(lines)


def roster_json(entries: Iterable[RosterEntry]) -> str:
    return json.dumps({
        "roster_version": MODEL_ROSTER_VERSION,
        "peer_benchmarks": {name: {"url": url, "note": note} for name, (url, note) in PEER_BENCHMARKS.items()},
        "entries": [entry.to_dict() for entry in entries],
    }, indent=2, sort_keys=True)
