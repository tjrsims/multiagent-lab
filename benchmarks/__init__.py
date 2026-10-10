"""Benchmark modules built on the base packages.

Each subpackage is one benchmark. It exposes ``build_module()`` returning an
``interpretability.modules.BenchmarkModule`` and ``PRESETS`` mapping preset
names to functions that build an END Bench ``RunSpec``. The base never imports
this package; END Bench loads a module by name on request.
"""
