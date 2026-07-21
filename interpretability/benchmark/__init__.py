"""v1 benchmark analysis: transcript-underdetermined negotiation deception.

See `benchmark_planning/03_V1_TARGET.md`. This package reads the `.pt`
activation dataset produced by the experiment runner and exposes the three
aligned detector views used by v1 — transcript-blind, fact-informed, and
white-box — over the same negotiation turns, with leakage-safe grouping keys.
"""
