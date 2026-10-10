"""`end-bench`: run text tracks on Bedrock, smoke-test offline, build scorecards.

Commands
--------
- ``end-bench run-text``   run a text-track design on Amazon Bedrock
- ``end-bench smoke``      same pipeline with the scripted offline model (no AWS)
- ``end-bench build``      views -> split -> pairs -> scores -> manifest for a
                           text dataset or an activation dataset
- ``end-bench estimate``   back-of-envelope Bedrock cost for a design
- ``end-bench info``       versions and the schema registry checksum
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def cli():
    """END-Bench: benchmark runs and builds on top of the research framework."""


def _common_design_options(f):
    f = click.option('--scenario', 'scenarios', multiple=True,
                     default=("hidden_value", "capability_bluff", "info_withholding", "ultimatum_bluff"),
                     show_default=True, help='Scenario (repeatable).')(f)
    f = click.option('--condition', 'conditions', multiple=True,
                     default=("high_incentive", "low_incentive", "penalty"), show_default=True,
                     type=click.Choice(["high_incentive", "low_incentive", "penalty", "minimal"]),
                     help='Incentive condition (repeatable).')(f)
    f = click.option('--families', type=click.IntRange(min=1), default=30, show_default=True,
                     help='Semantic families per scenario per condition.')(f)
    f = click.option('--max-rounds', type=click.IntRange(min=1), default=5, show_default=True)(f)
    f = click.option('--rollouts-per-family', type=click.IntRange(min=1), default=3, show_default=True)(f)
    f = click.option('--paired-facts/--no-paired-facts', default=True, show_default=True,
                     help='Generate matched twins (same claim anchor, opposite truth).')(f)
    f = click.option('--counterpart-type', 'counterpart_types', multiple=True, default=("default",),
                     type=click.Choice(["default", "skeptical", "credulous", "informed"]),
                     show_default=True, help='Counterpart policy (repeatable for a crossed design).')(f)
    f = click.option('--surface-variant', 'surface_variants', multiple=True, default=("default",),
                     show_default=True, help='Prompt surface variant (repeatable).')(f)
    f = click.option('--counterbalance/--no-counterbalance', default=False, show_default=True,
                     help='Cross roles x first mover (4x executions); off keeps the actor fixed.')(f)
    f = click.option('--ultrafast/--advanced', 'ultrafast', default=True, show_default=True,
                     help='One model call per action (default) vs the ToM-enabled advanced agent.')(f)
    f = click.option('--max-tokens', type=click.IntRange(min=16), default=160, show_default=True)(f)
    f = click.option('--family-seed-start', type=click.IntRange(min=0), default=0, show_default=True)(f)
    f = click.option('--run-id', default="", help='Run identifier (default: timestamp).')(f)
    f = click.option('--out', 'out_dir', type=click.Path(file_okay=False), default="runs", show_default=True)(f)
    return f


def _spec_from_options(model, **options):
    from end_bench.run import RunSpec, RunSpecError

    try:
        return RunSpec(
            model=model,
            scenarios=tuple(options["scenarios"]),
            conditions=tuple(options["conditions"]),
            families=options["families"],
            max_rounds=options["max_rounds"],
            rollouts_per_family=options["rollouts_per_family"],
            paired_facts=options["paired_facts"],
            counterpart_types=tuple(options["counterpart_types"]),
            surface_variants=tuple(options["surface_variants"]),
            counterbalance=options["counterbalance"],
            ultrafast=options["ultrafast"],
            agent_modules=() if options["ultrafast"] else ("theory_of_mind",),
            max_tokens=options["max_tokens"],
            temperature=options.get("temperature", 0.7),
            region=options.get("region", "us-east-1"),
            run_id=options["run_id"],
            price_table=options.get("price_table"),
            family_seed_start=options["family_seed_start"],
        )
    except RunSpecError as error:
        raise click.UsageError(str(error)) from error


def _echo_result(result) -> None:
    click.echo(click.style(f"run {result.run_id} complete", fg='green', bold=True))
    click.echo(f"  rows: {result.n_rows}")
    totals = result.cost_totals
    usd = totals.get("usd")
    click.echo(
        f"  calls: {totals.get('calls')}  tokens in/out: "
        f"{totals.get('input_tokens')}/{totals.get('output_tokens')}  "
        f"usd: {usd:.4f}" if isinstance(usd, (int, float)) else f"  calls: {totals.get('calls')}"
    )
    if totals.get("unpriced_calls"):
        click.echo(click.style(f"  unpriced calls: {totals['unpriced_calls']}", fg='yellow'))
    click.echo(f"  text dataset: {result.text_dataset_path}")
    click.echo(f"  manifest:     {result.manifest_path}")


@cli.command('run-text')
@click.option('--model', required=True,
              help="Bedrock model id, e.g. 'bedrock:anthropic.claude-sonnet-4-6'.")
@click.option('--region', default="us-east-1", show_default=True)
@click.option('--temperature', type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option('--price-table', type=click.Path(exists=True, dir_okay=False), default=None,
              help='JSON {model_id_prefix: [usd_in, usd_out]} per 1M tokens.')
@_common_design_options
def run_text(model, region, temperature, price_table, **options):
    """Run a text-track design on Amazon Bedrock and export a text dataset."""
    if not model.startswith("bedrock:"):
        raise click.UsageError("--model must start with 'bedrock:' (use `smoke` for offline runs)")
    from end_bench.run import run_text_track

    spec = _spec_from_options(model, region=region, temperature=temperature,
                              price_table=price_table, **options)
    out_dir = options["out_dir"]
    click.echo(f"design: {json.dumps(spec.to_dict(), sort_keys=True)}")
    result = run_text_track(spec, out_dir)
    _echo_result(result)


@cli.command('smoke')
@_common_design_options
def smoke(**options):
    """Run the whole pipeline offline with the scripted model (no AWS, no GPU)."""
    from end_bench.run import run_text_track

    overrides = dict(options)
    spec = _spec_from_options("scripted", **overrides)
    result = run_text_track(spec, options["out_dir"])
    _echo_result(result)


@cli.command('build')
@click.option('--dataset', 'dataset_path', required=True, type=click.Path(exists=True, dir_okay=False),
              help='END-Bench text dataset (.json) or safe activation dataset manifest.')
@click.option('--out', 'out_dir', required=True, type=click.Path(file_okay=False))
@click.option('--label-source', type=click.Choice(['rule', 'model_judge', 'gm', 'human']), default='rule', show_default=True)
@click.option('--rows', type=click.Choice(['claim_bearing', 'all']), default='claim_bearing', show_default=True)
@click.option('--bin-width', type=float, default=5.0, show_default=True)
@click.option('--split-seed', type=int, default=42, show_default=True)
@click.option('--n-bootstrap', type=int, default=1000, show_default=True)
@click.option('--no-probe', is_flag=True, help='Skip the activation probe (text datasets have none).')
@click.option('--trust-legacy-pt', is_flag=True,
              help='Allow pickle-capable .pt loading only for a reviewed artifact')
def build(dataset_path, out_dir, label_source, rows, bin_width, split_seed, n_bootstrap, no_probe, trust_legacy_pt):
    """Project, split, pair, score, and stamp one benchmark build."""
    from end_bench.data.text_dataset import TextDatasetError, is_text_dataset, load_text_dataset
    from interpretability.benchmark.build import BenchmarkBuildError, build_benchmark
    from interpretability.benchmark.splits import BenchmarkSplitError
    from interpretability.benchmark.views import ViewsError

    preloaded = None
    train_probe = not no_probe
    try:
        if is_text_dataset(dataset_path):
            preloaded = load_text_dataset(dataset_path)
            train_probe = False
        result = build_benchmark(
            None if preloaded is not None else dataset_path,
            out_dir,
            dataset=preloaded,
            label_source=label_source,
            rows=rows,
            bin_width=bin_width,
            split_seed=split_seed,
            n_bootstrap=n_bootstrap,
            train_probe=train_probe,
            trusted_legacy=trust_legacy_pt,
        )
    except (TextDatasetError, ViewsError, BenchmarkSplitError, BenchmarkBuildError, ValueError) as error:
        click.echo(click.style(f"BUILD FAILED: {error}", fg='red', bold=True))
        raise SystemExit(1)
    report = result.views.report
    click.echo(click.style(f"benchmark manifest {result.manifest.manifest_id}", fg='blue', bold=True))
    click.echo(f"  rows kept: {report.n_kept}/{report.n_input_rows} (groups={report.n_groups})")
    if report.excluded:
        click.echo("  excluded: " + ", ".join(f"{k}={v}" for k, v in sorted(report.excluded.items())))
    click.echo(f"  matched pairs: {result.pairs.n_pairs}")
    for score in result.scorecard.detector_scores:
        a = score.auroc
        click.echo(f"  {score.detector:22s} {score.subset:7s} n={score.n_rows:4d} "
                   f"AUROC={a.value:.3f} [{a.ci_low:.3f}, {a.ci_high:.3f}]")
    for name, path in result.written.items():
        click.echo(f"  wrote {name}: {path}")


@cli.command('pool')
@click.option('--dataset', 'dataset_paths', multiple=True, required=True,
              type=click.Path(exists=True, dir_okay=False), help='END-Bench text dataset (.json). Repeatable.')
@click.option('--out', 'out_path', required=True, type=click.Path(dir_okay=False),
              help='Where to write the pooled text dataset (.json).')
@click.option('--allow-mixed-registry', is_flag=True, default=False,
              help='Pool datasets whose schema registry checksums differ (records it in the manifest).')
def pool(dataset_paths, out_path, allow_mixed_registry):
    """Pool several text datasets into one derived dataset for cross-run audits.

    Row and record ids are prefixed with the source run id so nothing collides;
    trial families stay shared because two runs of one family share the same
    private facts. The result feeds `adjudicate export` and `judge`; it is not
    a run and is never a leaderboard entry.
    """
    from end_bench.data.pooled import PooledDatasetError, pool_text_datasets
    from end_bench.data.text_dataset import load_text_dataset, save_text_dataset

    try:
        pooled = pool_text_datasets(
            [load_text_dataset(path) for path in dataset_paths], allow_mixed_registry=allow_mixed_registry,
        )
    except PooledDatasetError as error:
        raise click.UsageError(str(error)) from error
    save_text_dataset(out_path, pooled)
    info = pooled["config"]["run_manifest"]["pooled"]
    click.echo(f"pooled {len(info['sources'])} datasets -> {pooled['config']['n_rows']} rows: {out_path}")
    for source in info["sources"]:
        click.echo(f"  {source['run_id']:45s} {source['model_name']:50s} rows={source['n_rows']}")


@cli.command('judge')
@click.option('--dataset', 'dataset_path', required=True, type=click.Path(exists=True, dir_okay=False),
              help='END-Bench text dataset or activation dataset manifest.')
@click.option('--out', 'out_dir', required=True, type=click.Path(file_okay=False))
@click.option('--judge', 'judges', multiple=True, required=True,
              help="Judge model: 'bedrock:<model_id>' or 'oracle' (offline). Repeatable.")
@click.option('--access', type=click.Choice(['both', 'transcript_only', 'fact_informed']), default='both', show_default=True)
@click.option('--region', default="us-east-1", show_default=True)
@click.option('--n-bootstrap', type=int, default=1000, show_default=True)
@click.option('--bin-width', type=float, default=5.0, show_default=True)
def judge(dataset_path, out_dir, judges, access, region, n_bootstrap, bin_width):
    """Score LLM judges against computed truth, transcript-only and fact-informed."""
    from end_bench.judges import JudgeAccess, JudgeCache, run_judge, score_judges
    from interpretability.benchmark.pairs import build_matched_pairs, pair_scopes_for_views, twin_ids_for_views

    dataset, views = _load_views(dataset_path)
    target = Path(out_dir); target.mkdir(parents=True, exist_ok=True)
    cache = JudgeCache(path=target / "judge_cache.jsonl")
    accesses = [JudgeAccess.TRANSCRIPT_ONLY, JudgeAccess.FACT_INFORMED] if access == 'both' else [JudgeAccess(access)]
    runs = []
    for spec in judges:
        model = _judge_model(spec, region)
        for mode in accesses:
            runs.append(run_judge(views, model, access=mode, judge_name=spec, cache=cache))
            click.echo(f"  judged {spec} [{mode.value}]: undetermined={runs[-1].n_undetermined}/{views.n} cache_hits={runs[-1].cache_hits}")
    pairs = build_matched_pairs(
        views.labels, views.claim_values, views.scenarios, views.group_ids, bin_width=bin_width,
        twin_ids=twin_ids_for_views(views), scopes=pair_scopes_for_views(views),
    )
    click.echo(f"  matched pairs: {pairs.n_pairs} (twins first; pairs never cross a source run)")
    card = score_judges(views, runs, pairs=pairs, n_bootstrap=n_bootstrap)
    (target / "judges.json").write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True))
    for score in card.detector_scores:
        a = score.auroc
        click.echo(f"  {score.detector:40s} {score.subset:7s} n={score.n_rows:4d} AUROC={a.value:.3f} [{a.ci_low:.3f}, {a.ci_high:.3f}]")
    for gap in card.gaps:
        d = gap.auroc_difference
        click.echo(f"  gap {gap.detector_a} - {gap.detector_b} ({gap.subset}): {d.value:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]")
    click.echo(f"  wrote {target / 'judges.json'}")


@cli.group()
def adjudicate():
    """Human adjudication: export blinded batches, import ratings, apply labels."""


@adjudicate.command('export')
@click.option('--dataset', 'dataset_path', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--out', 'out_dir', required=True, type=click.Path(file_okay=False))
@click.option('--n', type=click.IntRange(min=1), default=300, show_default=True)
@click.option('--seed', type=int, default=0, show_default=True)
@click.option('--gold-fraction', type=click.FloatRange(0.0, 0.9), default=0.1, show_default=True)
def adjudicate_export(dataset_path, out_dir, n, seed, gold_fraction):
    """Write blind.csv, informed.csv, and private/KEY.json for raters."""
    from end_bench.adjudication import AdjudicationError, export_adjudication_batch

    dataset, _ = _load_views(dataset_path)
    try:
        manifest = export_adjudication_batch(dataset, out_dir, n=n, seed=seed, gold_fraction=gold_fraction)
    except AdjudicationError as error:
        raise click.UsageError(str(error)) from error
    click.echo(f"batch: {manifest.n_items} items ({manifest.n_gold} gold) -> {manifest.batch_dir}")
    click.echo("  rater sheets: blind.csv, informed.csv; key: private/KEY.json (do not share)")


@adjudicate.command('import')
@click.option('--batch', 'batch_dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--ratings', 'rating_files', multiple=True, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--dataset', 'dataset_path', default=None, type=click.Path(exists=True, dir_okay=False),
              help='When given, also write <out>/dataset_with_human_labels.json.')
@click.option('--out', 'out_dir', required=True, type=click.Path(file_okay=False))
@click.option('--n-bootstrap', type=int, default=1000, show_default=True)
def adjudicate_import(batch_dir, rating_files, dataset_path, out_dir, n_bootstrap):
    """Compute kappa, gold accuracy, and rule-vs-human agreement; apply labels."""
    from end_bench.adjudication import AdjudicationError, apply_human_labels, import_ratings
    from end_bench.data.text_dataset import save_text_dataset

    target = Path(out_dir); target.mkdir(parents=True, exist_ok=True)
    try:
        report = import_ratings(batch_dir, list(rating_files), n_bootstrap=n_bootstrap)
    except AdjudicationError as error:
        raise click.UsageError(str(error)) from error
    (target / "adjudication.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    kappa = report.kappa
    click.echo(f"items={report.n_items} raters={report.n_raters} rated_by_all={report.n_rated_by_all} "
               f"kappa={'n/a' if kappa != kappa else f'{kappa:.3f}'} undetermined={report.undetermined_items}")
    if report.rule_vs_human_agreement is not None:
        a = report.rule_vs_human_agreement
        click.echo(f"rule-vs-human agreement: {a.value:.3f} [{a.ci_low:.3f}, {a.ci_high:.3f}] (n={a.n_rows}, groups={a.n_groups})")
    for rater, acc in report.gold_accuracy_by_rater.items():
        click.echo(f"  gold accuracy {rater}: {'n/a' if acc is None else f'{acc:.2f}'}")
    if dataset_path:
        dataset, _ = _load_views(dataset_path)
        labeled = apply_human_labels(dataset, report)
        if labeled.get("schema_version"):
            labeled["config"]["dataset_hash"] = _rehash_text_dataset(labeled)
            path = save_text_dataset(target / "dataset_with_human_labels.json", labeled)
        else:
            # Activation datasets keep their arrays: write the safe JSON+NPZ
            # bundle so `end-bench build --label-source human` can load it.
            from interpretability.data.activation_dataset import save_activation_dataset

            saved = save_activation_dataset(target / "dataset_with_human_labels.json", labeled)
            path = saved[0] if isinstance(saved, tuple) else saved
        click.echo(f"  wrote {path} (build it with --label-source human)")


def _load_benchmark_module(name: str):
    import importlib

    try:
        package = importlib.import_module(f"benchmarks.{name}")
    except ModuleNotFoundError as error:
        raise click.UsageError(f"no benchmark module named {name!r} under benchmarks/") from error
    for attr in ("build_module", "PRESETS"):
        if not hasattr(package, attr):
            raise click.UsageError(f"benchmark module {name!r} must define {attr}")
    return package


@cli.group()
def module():
    """Run a benchmark module (a package under benchmarks/) on the text track."""


@module.command('run')
@click.argument('name')
@click.option('--model', required=True, help="'scripted' or bedrock:<model id>.")
@click.option('--preset', default="pilot", show_default=True)
@click.option('--counterpart-type', 'counterpart_types', multiple=True, default=("interrogative",),
              show_default=True, help="One run per policy (repeatable).")
@click.option('--run-id', default="", help="Run id prefix (default: timestamp).")
@click.option('--out', 'out_dir', type=click.Path(file_okay=False), default="runs", show_default=True)
@click.option('--region', default="us-east-1", show_default=True)
def module_run(name, model, preset, counterpart_types, run_id, out_dir, region):
    """Run one preset of a module, once per counterpart policy."""
    import dataclasses
    import time as _time

    from end_bench.run import RunSpecError, run_text_track

    package = _load_benchmark_module(name)
    if preset not in package.PRESETS:
        raise click.UsageError(f"unknown preset {preset!r}; choose from {sorted(package.PRESETS)}")
    benchmark = package.build_module()
    prefix = run_id or _time.strftime("%Y%m%d-%H%M%S")
    failures: list[tuple[str, str]] = []
    for policy in counterpart_types:
        spec = package.PRESETS[preset](model, counterpart_type=policy, run_id=f"{prefix}-{preset}-{policy}")
        spec = dataclasses.replace(spec, region=region)
        try:
            result = run_text_track(spec, out_dir, module=benchmark)
        except RunSpecError as error:
            # One policy failing (a network blip at preflight, a model the
            # account cannot use) must not lose the rest of the batch.
            failures.append((policy, str(error)))
            click.echo(f"FAILED policy={policy}: {error}", err=True)
            continue
        click.echo(f"{benchmark.name} {benchmark.version} preset={preset} policy={policy}")
        _echo_result(result)
    if failures:
        summary = "; ".join(f"{policy}: {reason[:160]}" for policy, reason in failures)
        raise click.ClickException(f"{len(failures)} of {len(counterpart_types)} policy runs failed. {summary}")


@module.command('pilot')
@click.argument('name')
@click.argument('run_dirs', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--out', 'out_dir', type=click.Path(file_okay=False), required=True)
@click.option('--n-bootstrap', type=int, default=200, show_default=True)
def module_pilot(name, run_dirs, out_dir, n_bootstrap):
    """Score pilot runs, apply the module's keep rule, and write the report and selection."""
    import importlib

    _load_benchmark_module(name)
    pilot = importlib.import_module(f"benchmarks.{name}.pilot")
    table = pilot.cell_table(list(run_dirs), n_bootstrap=n_bootstrap)
    selection = pilot.select_cells(table)
    target = Path(out_dir); target.mkdir(parents=True, exist_ok=True)
    (target / "pilot_cells.json").write_text(json.dumps({"table": table, "selection": selection}, indent=2, sort_keys=True, default=str))
    report = pilot.render_pilot_report(table, selection)
    (target / "PILOT_REPORT.md").write_text(report)
    click.echo(report)
    click.echo(f"kept {len(selection['kept'])} cells, dropped {len(selection['dropped'])}; wrote {target / 'PILOT_REPORT.md'}")


@module.command('leaderboard')
@click.argument('name')
@click.argument('run_dirs', nargs=-1, required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--out', 'out_path', type=click.Path(dir_okay=False), default=None)
def module_leaderboard(name, run_dirs, out_path):
    """Render the module's leaderboard over finished runs."""
    import importlib

    _load_benchmark_module(name)
    report = importlib.import_module(f"benchmarks.{name}.report")
    text = report.render_leaderboard(list(run_dirs))
    if out_path:
        Path(out_path).write_text(text)
        click.echo(f"wrote {out_path}")
    else:
        click.echo(text)


@cli.command('outcomes')
@click.option('--dataset', 'dataset_path', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--out', 'out_dir', required=True, type=click.Path(file_okay=False))
@click.option('--n-bootstrap', type=int, default=1000, show_default=True)
def outcomes(dataset_path, out_dir, n_bootstrap):
    """Agreement prices, ZOPA shares, and the lying-pays comparison."""
    from end_bench.outcomes import lying_pays, trial_outcomes

    dataset, _ = _load_views(dataset_path)
    items = trial_outcomes(dataset)
    cells = lying_pays(items, n_bootstrap=n_bootstrap)
    target = Path(out_dir); target.mkdir(parents=True, exist_ok=True)
    payload = {"trial_outcomes": [i.to_dict() for i in items], "lying_pays": [c.to_dict() for c in cells]}
    (target / "outcomes.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    agreed = sum(1 for i in items if i.agreed)
    click.echo(f"trials={len(items)} agreed={agreed}")
    for cell in cells:
        d = cell.difference
        diff = "n/a" if d is None else f"{d.value:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]"
        click.echo(f"  {cell.scenario:16s} lying n={cell.n_lying} honest n={cell.n_honest} share diff={diff}")
    click.echo(f"  wrote {target / 'outcomes.json'}")


@cli.command('report')
@click.option('--build', 'builds', multiple=True, required=True,
              help="NAME=DIR of a build directory (scores.json, manifest.json; optional judges/adjudication/outcomes.json).")
@click.option('--out', 'out_path', required=True, type=click.Path(dir_okay=False))
@click.option('--title', default="END-Bench results", show_default=True)
def report(builds, out_path, title):
    """Render one markdown report over several builds."""
    from end_bench.report import ReportError, collect_build, render_report

    artifacts = []
    for item in builds:
        if "=" not in item:
            raise click.UsageError("--build expects NAME=DIR")
        name, directory = item.split("=", 1)
        try:
            artifacts.append(collect_build(name, directory))
        except ReportError as error:
            raise click.UsageError(str(error)) from error
    text = render_report(artifacts, title=title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    click.echo(f"wrote {out_path} ({len(text.splitlines())} lines)")


def _load_views(dataset_path):
    from end_bench.data.text_dataset import is_text_dataset, load_text_dataset
    from interpretability.benchmark.views import build_views
    from interpretability.data.activation_dataset import load_activation_dataset

    dataset = load_text_dataset(dataset_path) if is_text_dataset(dataset_path) else load_activation_dataset(dataset_path)
    return dataset, build_views(dataset)


def _judge_model(spec: str, region: str):
    if spec == "oracle":
        from end_bench.models.oracle_judge import OracleJudgeModel

        return OracleJudgeModel()
    if spec.startswith("bedrock:"):
        from end_bench.models.bedrock import BedrockTextModel

        return BedrockTextModel(spec, region=region, max_tokens=200, temperature=0.0)
    raise click.UsageError(f"unsupported judge {spec!r}; use 'oracle' or 'bedrock:<model_id>'")


def _rehash_text_dataset(dataset):
    from end_bench.data.text_dataset import _content_hash

    return _content_hash(dataset)


@cli.command('estimate')
@click.option('--model', 'model_id', required=True, help='Bedrock model id (prefix match).')
@click.option('--negotiations', type=click.IntRange(min=0), required=True)
@click.option('--calls-per-negotiation', type=click.IntRange(min=0), default=10, show_default=True)
@click.option('--input-tokens', type=click.IntRange(min=0), default=1500, show_default=True)
@click.option('--output-tokens', type=click.IntRange(min=0), default=150, show_default=True)
@click.option('--price-table', type=click.Path(exists=True, dir_okay=False), default=None)
def estimate(model_id, negotiations, calls_per_negotiation, input_tokens, output_tokens, price_table):
    """Estimate Bedrock spend for a design from the versioned price table."""
    from end_bench.models.pricing import estimate_design_cost, load_price_table

    table = load_price_table(price_table) if price_table else None
    payload = estimate_design_cost(
        model_id.removeprefix("bedrock:"), negotiations=negotiations,
        calls_per_negotiation=calls_per_negotiation,
        input_tokens_per_call=input_tokens, output_tokens_per_call=output_tokens, table=table,
    )
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
    if payload["usd_total"] is None:
        click.echo(click.style("model id not in the price table; no dollar estimate", fg='yellow'))


@cli.command('prices')
@click.option('--verify', is_flag=True, help="Compare the table with the AWS Price List API (needs credentials).")
@click.option('--region', default="us-east-1", show_default=True)
def prices(verify, region):
    """Show the price table with each entry's provenance; --verify re-checks it live."""
    from end_bench.models.pricing import DEFAULT_PRICES_PER_MTOK, PRICE_SOURCES, PRICE_TABLE_VERSION

    click.echo(f"price table {PRICE_TABLE_VERSION} (USD per 1M tokens: input, output)")
    rows = None
    if verify:
        try:
            import boto3
        except ImportError as error:
            raise click.UsageError("boto3 is required: pip install -e '.[aws]'") from error
        from end_bench.models.price_check import fetch_pricelist, verify_price_table

        fetched = fetch_pricelist(boto3.client("pricing", region_name="us-east-1"), region)
        rows = {row["model_id_prefix"]: row for row in verify_price_table(fetched)}
    for prefix, (usd_in, usd_out) in DEFAULT_PRICES_PER_MTOK.items():
        status = f" [{rows[prefix]['status']}]" if rows else ""
        extra = ""
        if rows and rows[prefix]["pricelist_usd_per_mtok"]:
            api_in, api_out = rows[prefix]["pricelist_usd_per_mtok"]
            extra = f"  pricelist={api_in:.4g}/{api_out:.4g}"
        click.echo(f"  {prefix:32s} {usd_in:8.4g} {usd_out:8.4g}{status}{extra}")
        click.echo(f"      {PRICE_SOURCES.get(prefix, 'unverified: no source recorded')}")
    if rows and any(row["status"] == "mismatch" for row in rows.values()):
        raise click.ClickException("price table disagrees with the Price List API; update pricing.py")


@cli.command('bedrock-models')
@click.option('--region', default="us-east-1", show_default=True)
@click.option('--provider', default=None, help="Filter by provider name, e.g. Anthropic, Meta, 'Mistral AI', Amazon.")
@click.option('--all-providers', is_flag=True, help="Include providers outside the END-Bench target set.")
@click.option('--check-access/--no-check-access', default=True, show_default=True,
              help="Query per-model access status (one API call per model).")
def bedrock_models(region, provider, all_providers, check_access):
    """List Bedrock text models, the id to invoke them with, and whether access is granted."""
    try:
        import boto3
    except ImportError as error:
        raise click.UsageError("boto3 is required: pip install -e '.[aws]'") from error
    from end_bench.models.catalog import TARGET_PROVIDERS, format_catalog, list_bedrock_models

    client = boto3.client("bedrock", region_name=region)
    rows = list_bedrock_models(client, provider=provider, check_access=check_access)
    if not provider and not all_providers:
        rows = [r for r in rows if r.provider in TARGET_PROVIDERS]
    click.echo(format_catalog(rows))
    click.echo("\naccess 'not granted'/'pending': Bedrock console -> Model access (us-east-1). "
               "'priced: no': tokens are counted but dollars are not until the id is in a --price-table JSON.")


@cli.command('info')
def info():
    """Print versions that stamp every run and build."""
    from end_bench import END_BENCH_VERSION
    from end_bench.data.text_dataset import TEXT_DATASET_SCHEMA_VERSION
    from end_bench.models.pricing import PRICE_TABLE_VERSION
    from interpretability.schema_registry import schema_registry, schema_registry_checksum

    payload = {
        "end_bench_version": END_BENCH_VERSION,
        "text_dataset_schema": TEXT_DATASET_SCHEMA_VERSION,
        "price_table_version": PRICE_TABLE_VERSION,
        "schema_registry_checksum": schema_registry_checksum(),
        "schema_registry": schema_registry(),
    }
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:  # pragma: no cover - console entry point
    cli()


__all__ = ["cli", "main"]
