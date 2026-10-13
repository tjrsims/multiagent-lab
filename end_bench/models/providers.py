"""Provider registry: one table maps a model scheme to an adapter, endpoint and key.

WALKTHROUGH (a guided tour for new readers)
-------------------------------------------
A run names its model as ``<scheme>:<model id>``. This module is the only place
that knows what a scheme means: ``bedrock`` builds the Bedrock adapter,
``scripted`` the offline model, and every other scheme builds the OpenAI
compatible adapter pointed at that provider's endpoint with that provider's key
variable. Adding a provider is one ``ProviderSpec`` row; nothing else in the
run path changes. Keys are never read here, only their variable names; the
adapters read the variable at call time.

``build_text_model`` is the single factory the run, the judge and the fixed
counterpart use. ``parse_model_spec`` fails closed on an unknown scheme and
names the known ones.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

__all__ = [
    "ModelSpecError",
    "PROVIDERS",
    "ProviderSpec",
    "build_text_model",
    "known_schemes",
    "parse_model_spec",
]


class ModelSpecError(ValueError):
    """A model spec names an unknown scheme or an empty id."""


@dataclass(frozen=True)
class ProviderSpec:
    scheme: str
    display: str
    kind: str                       # "bedrock" | "openai_compat" | "scripted"
    base_url: str | None = None     # openai_compat only
    key_env: str | None = None      # openai_compat only; the variable, never the value
    docs_url: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.scheme.isalpha() or not self.scheme.islower():
            raise ValueError("scheme must be lowercase letters")
        if self.kind not in ("bedrock", "openai_compat", "scripted"):
            raise ValueError(f"unknown provider kind {self.kind!r}")
        if self.kind == "openai_compat" and not (self.base_url and self.key_env):
            raise ValueError("openai_compat providers need base_url and key_env")

    @property
    def base_url_env(self) -> str | None:
        """Environment variable that overrides ``base_url`` (proxies, regional hosts)."""
        return f"{self.scheme.upper()}_BASE_URL" if self.kind == "openai_compat" else None

    def credentials_status(self) -> str:
        """``set`` / ``missing`` for key based providers; ``aws`` or ``none`` otherwise.

        Reads only whether the variable exists, never its value.
        """
        if self.kind == "openai_compat":
            return "set" if os.environ.get(self.key_env or "") else "missing"
        if self.kind == "bedrock":
            return "aws"
        return "none"


_ROWS = (
    ProviderSpec("bedrock", "Amazon Bedrock", "bedrock",
                 docs_url="https://docs.aws.amazon.com/bedrock/",
                 notes="Converse API; boto3 credentials; ids may need a regional or global inference profile."),
    ProviderSpec("scripted", "Offline scripted model", "scripted",
                 notes="Deterministic canned lines for smoke tests; free."),
    ProviderSpec("openai", "OpenAI", "openai_compat", "https://api.openai.com/v1", "OPENAI_API_KEY",
                 docs_url="https://platform.openai.com/docs"),
    ProviderSpec("xai", "xAI", "openai_compat", "https://api.x.ai/v1", "XAI_API_KEY",
                 docs_url="https://docs.x.ai/docs/models"),
    ProviderSpec("gemini", "Google Gemini API", "openai_compat",
                 "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
                 docs_url="https://ai.google.dev/api/compatibility",
                 notes="Google's OpenAI compatible endpoint; keys from AI Studio."),
    ProviderSpec("deepseek", "DeepSeek", "openai_compat", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
                 docs_url="https://api-docs.deepseek.com/",
                 notes="Off peak prices are half of peak; the ledger uses peak."),
    ProviderSpec("mistral", "Mistral platform", "openai_compat", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                 docs_url="https://docs.mistral.ai/"),
    ProviderSpec("moonshot", "Moonshot AI (Kimi)", "openai_compat", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY",
                 docs_url="https://platform.moonshot.ai/docs"),
    ProviderSpec("qwen", "Alibaba Model Studio (Qwen)", "openai_compat",
                 "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY",
                 docs_url="https://www.alibabacloud.com/help/en/model-studio/",
                 notes="International endpoint; mainland endpoint differs."),
    ProviderSpec("zai", "Z.ai (GLM)", "openai_compat", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY",
                 docs_url="https://docs.z.ai/"),
    ProviderSpec("meta", "Meta Model API (Muse)", "openai_compat", "https://api.meta.ai/v1", "META_API_KEY",
                 docs_url="https://dev.meta.ai/docs/api-reference",
                 notes="Meta's docs call the key MODEL_API_KEY; export it as META_API_KEY here."),
    ProviderSpec("openrouter", "OpenRouter (many vendors)", "openai_compat",
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                 docs_url="https://openrouter.ai/docs",
                 notes="Routes to third party hosts; the serving host is not under your control, record it in the prereg."),
    ProviderSpec("together", "Together AI (open weights)", "openai_compat",
                 "https://api.together.xyz/v1", "TOGETHER_API_KEY",
                 docs_url="https://docs.together.ai/"),
)
PROVIDERS: Mapping[str, ProviderSpec] = {row.scheme: row for row in _ROWS}


def known_schemes() -> tuple[str, ...]:
    return tuple(PROVIDERS)


def parse_model_spec(spec: str) -> tuple[ProviderSpec, str]:
    """``"xai:grok-4.3"`` -> (xAI provider, ``"grok-4.3"``); ``"scripted"`` has no id."""
    if not isinstance(spec, str) or not spec.strip():
        raise ModelSpecError("model spec must be non empty text")
    text = spec.strip()
    if text == "scripted":
        return PROVIDERS["scripted"], "scripted"
    scheme, sep, model_id = text.partition(":")
    if not sep or scheme not in PROVIDERS or scheme == "scripted":
        raise ModelSpecError(
            f"unsupported model spec {spec!r}; use <scheme>:<model id> with a scheme in "
            f"{', '.join(s for s in PROVIDERS if s != 'scripted')}, or 'scripted'"
        )
    if not model_id.strip():
        raise ModelSpecError(f"model spec {spec!r} has no model id after the scheme")
    return PROVIDERS[scheme], model_id.strip()


def build_text_model(
    spec: str,
    *,
    max_tokens: int,
    temperature: float | None,
    cost_ledger: Any = None,
    token_budget_per_day: int | None = None,
    honor_caller_sampling: bool = True,
    region: str = "us-east-1",
    request_extras: Mapping[str, Any] | None = None,
) -> Any:
    """The acting model for ``spec``; adapters publish records with the scheme in their name.

    ``temperature=None`` means provider default sampling (nothing sent).
    ``request_extras`` are model specific request fields: Converse
    ``additionalModelRequestFields`` on Bedrock, extra body keys elsewhere.
    """
    provider, model_id = parse_model_spec(spec)
    if provider.kind == "scripted":
        from end_bench.models.scripted import ScriptedTextModel

        return ScriptedTextModel(max_tokens=max_tokens, cost_ledger=cost_ledger)
    if provider.kind == "bedrock":
        from end_bench.models.bedrock import BedrockTextModel

        return BedrockTextModel(
            model_id,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
            cost_ledger=cost_ledger,
            token_budget_per_day=token_budget_per_day,
            honor_caller_sampling=honor_caller_sampling,
            additional_request_fields=request_extras,
        )
    from end_bench.models.openai_compat import OpenAICompatTextModel

    return OpenAICompatTextModel(
        model_id,
        scheme=provider.scheme,
        base_url=os.environ.get(provider.base_url_env or "") or provider.base_url,
        api_key_env=provider.key_env,
        base_url_env=provider.base_url_env,
        max_tokens=max_tokens,
        temperature=temperature,
        cost_ledger=cost_ledger,
        token_budget_per_day=token_budget_per_day,
        honor_caller_sampling=honor_caller_sampling,
        extra_body=request_extras,
    )
