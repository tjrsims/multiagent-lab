"""END-Bench: the benchmark product built on top of the research framework.

This package is additive. It never modifies the research pipeline's contracts;
it supplies the pieces the AWS runbook needs — an API acting-model adapter
(Amazon Bedrock), a scripted offline model for smoke tests, a text-dataset
contract for text-only tracks, an orchestrator that drives the existing
`InterpretabilityRunner`, and a small CLI (`end-bench`). Container and cloud
scripts live in `end_bench/docker/` and `end_bench/aws/`.
"""

END_BENCH_VERSION = "0.1.0"

__all__ = ["END_BENCH_VERSION"]
