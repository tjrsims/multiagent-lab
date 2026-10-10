"""Acting-model adapters that publish scoped generation records."""

from end_bench.models.bedrock import BedrockGenerationError, BedrockTextModel
from end_bench.models.pricing import (
    DEFAULT_PRICES_PER_MTOK,
    PRICE_TABLE_VERSION,
    CostLedger,
    estimate_cost,
    estimate_design_cost,
    resolve_price,
)
from end_bench.models.scripted import ScriptedTextModel

__all__ = [
    "DEFAULT_PRICES_PER_MTOK",
    "PRICE_TABLE_VERSION",
    "BedrockGenerationError",
    "BedrockTextModel",
    "CostLedger",
    "ScriptedTextModel",
    "estimate_cost",
    "estimate_design_cost",
    "resolve_price",
]
