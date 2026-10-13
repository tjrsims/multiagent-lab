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
    f = click.option('--token-budget-per-day', type=click.IntRange(min=1), default=None,
                     help='Pace the model under a daily token cap (tokens per rolling 24 hours).')(f)
    f = click.option('--counterpart-model', default=None,
                     help="Fixed counterpart model (<scheme>:<id>, @<roster key>, or 'scripted'); default is self play.")(f)
    return f


def _resolve_model(value, option):
    """``@key`` roster lookups and scheme validation, as a usage error on failure."""
    if value is None:
        return None
    from end_bench.models.providers import ModelSpecError
    from end_bench.models.roster import resolve_model_spec

    try:
        return resolve_model_spec(value)
    except ModelSpecError as error:
        raise click.UsageError(f"{option}: {error}") from error


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
            token_budget_per_day=options.get("token_budget_per_day"),
            counterpart_model=_resolve_model(options.get("counterpart_model"), "--counterpart-model"),
            price_table=options.get("price_table"),
            family_seed_start=options["family_seed_start"],
        )
    except RunSpecError as error:
        raise click.UsageError(str(error)) from error


def _echo_result(result) -> None:
    click.echo(click.style(f"run {result.run_id} complete", fg='green', bold=True))
    click.echo(f"  rows: {result.n_rows}")
    trials = getattr(result, "trials", None) or {}
    if trials:
        line = f"  trials: {trials.get('completed')} completed, {trials.get('failed')} failed of {trials.get('planned')} planned"
        if trials.get("failed"):
            types = ", ".join(f"{k}={v}" for k, v in (trials.get("failure_types") or {}).items())
            click.echo(click.style(f"{line}  ({types})", fg='yellow'))
        else:
            click.echo(line)
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
              help="<scheme>:<model id> (schemes: `end-bench roster --providers`) or @<roster key>, e.g. "
                   "bedrock:global.openai.gpt-5.6-sol, xai:grok-4.3, @claude-sonnet-5.")
@click.option('--region', default="us-east-1", show_default=True)
@click.option('--temperature', type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option('--price-table', type=click.Path(exists=True, dir_okay=False), default=None,
              help='JSON {model_id_prefix: [usd_in, usd_out]} per 1M tokens.')
@_common_design_options
def run_text(model, region, temperature, price_table, **options):
    """Run a text-track design on an API model and export a text dataset."""
    model = _resolve_model(model, "--model")
    if model == "scripted":
        raise click.UsageError("--model scripted is the offline model; use `end-bench smoke` for offline runs")
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
@click.option('--min-scoring-families', type=click.IntRange(min=1), default=8, show_default=True,
              help='Fewest trial families the held out scoring split may have; below it every row is scored and the manifest says why.')
@click.option('--n-bootstrap', type=int, default=1000, show_default=True)
@click.option('--no-probe', is_flag=True, help='Skip the activation probe (text datasets have none).')
@click.option('--trust-legacy-pt', is_flag=True,
              help='Allow pickle-capable .pt loading only for a reviewed artifact')
def build(dataset_path, out_dir, label_source, rows, bin_width, split_seed, n_bootstrap, no_probe, trust_legacy_pt, min_scoring_families):
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
            min_scoring_families=min_scoring_families,
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
    pair_params = dict(result.pairs.parameters)
    if pair_params.get("scoring_partition"):
        click.echo(f"  matched pairs: {result.pairs.n_pairs} inside the {pair_params['scoring_partition']} split "
                   f"({pair_params.get('pairs_over_all_rows')} over all rows)")
    else:
        click.echo(f"  matched pairs: {result.pairs.n_pairs}")
    for score in result.scorecard.detector_scores:
        a = score.auroc
        click.echo(f"  {score.detector:22s} {score.subset:7s} n={score.n_rows:4d} "
                   f"AUROC={a.value:.3f} [{a.ci_low:.3f}, {a.ci_high:.3f}]")
    for name, path in result.written.items():
        click.echo(f"  wrote {name}: {path}")


@cli.command('relabel')
@click.option('--run-dir', 'run_dirs', multiple=True, type=click.Path(exists=True, file_okay=False),
              help='Finished run directory (text_dataset.json, run_manifest.json, cost_ledger.jsonl). Repeatable.')
@click.option('--dataset', 'dataset_path', default=None, type=click.Path(exists=True, dir_okay=False),
              help='A single text dataset instead of run directories.')
@click.option('--out', 'out_path', required=True, type=click.Path(),
              help='Output root for --run-dir (one sub-directory per run) or the output file for --dataset.')
def relabel(run_dirs, dataset_path, out_path):
    """Re-read numeric claim turns under the current extractor and rule, offline.

    Labels are computed at run time and stored. After an extractor fix the
    stored text and facts are read again; the original label records stay as
    evidence, changed rows record the extractor they came from, and the
    manifest says the dataset is relabeled.
    """
    import shutil

    from end_bench.data.relabel import RelabelError, relabel_text_dataset
    from end_bench.data.text_dataset import load_text_dataset, save_text_dataset

    if bool(run_dirs) == bool(dataset_path):
        raise click.UsageError("pass either --run-dir (repeatable) or --dataset")

    def _one(source_file, target_file):
        relabeled, report = relabel_text_dataset(load_text_dataset(source_file))
        save_text_dataset(target_file, relabeled)
        click.echo(f"  {Path(source_file).parent.name}: {report['rows_changed']}/{report['rows_considered']} rows changed "
                   f"{report['transitions']} (extractor {report['from_extractors']} -> {report['to_extractor']})")
        return report

    try:
        if dataset_path:
            _one(dataset_path, out_path)
            return
        root = Path(out_path); root.mkdir(parents=True, exist_ok=True)
        for run_dir in run_dirs:
            run = Path(run_dir)
            if not (run / "text_dataset.json").exists():
                click.echo(f"skipped {run}: no text_dataset.json", err=True)
                continue
            target = root / run.name
            target.mkdir(parents=True, exist_ok=True)
            report = _one(run / "text_dataset.json", target / "text_dataset.json")
            for name in ("cost_ledger.jsonl", "results.json"):
                if (run / name).exists():
                    shutil.copyfile(run / name, target / name)
            manifest = json.loads((run / "run_manifest.json").read_text()) if (run / "run_manifest.json").exists() else {}
            manifest["relabel"] = report
            manifest["relabeled_from_run_dir"] = str(run)
            (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    except RelabelError as error:
        raise click.UsageError(str(error)) from error


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
@click.option('--subset', default='all', show_default=True,
              help="Rows to judge: 'all', 'paired' (members of matched pairs only), or 'batch:<dir>' "
                   "(the rows of an adjudication batch exported from this same dataset).")
@click.option('--workers', type=click.IntRange(min=1), default=1, show_default=True,
              help='Parallel judge calls per judge; answers are cached as they arrive.')
@click.option('--rate-per-minute', type=click.FloatRange(min=0.1), default=None,
              help='Cap on judge call submissions per minute, to stay under the Bedrock request quota.')
def judge(dataset_path, out_dir, judges, access, region, n_bootstrap, bin_width, subset, workers, rate_per_minute):
    """Score LLM judges against computed truth, transcript-only and fact-informed."""
    from end_bench.judges import JudgeAccess, JudgeCache, restrict_pairs, run_judge, score_judges
    from end_bench.models.pricing import CostLedger
    from interpretability.benchmark.pairs import build_matched_pairs, pair_scopes_for_views, twin_ids_for_views
    from interpretability.benchmark.views import subset_views

    dataset, full_views = _load_views(dataset_path)
    target = Path(out_dir); target.mkdir(parents=True, exist_ok=True)
    full_pairs = build_matched_pairs(
        full_views.labels, full_views.claim_values, full_views.scenarios, full_views.group_ids, bin_width=bin_width,
        twin_ids=twin_ids_for_views(full_views), scopes=pair_scopes_for_views(full_views),
    )
    selection = {"subset": subset, "n_rows_total": full_views.n, "pairs_total": full_pairs.n_pairs,
                 "workers": workers, "rate_per_minute": rate_per_minute}
    if subset == 'all':
        views, pairs = full_views, full_pairs
    elif subset == 'paired':
        indices = [int(i) for i in full_pairs.paired_row_indices()]
        views, pairs = subset_views(full_views, indices), restrict_pairs(full_pairs, indices)
    elif subset.startswith('batch:'):
        key_path = Path(subset[len('batch:'):]) / "private" / "KEY.json"
        if not key_path.exists():
            raise click.UsageError(f"{key_path} not found; 'batch:<dir>' needs an exported adjudication batch")
        key = json.loads(key_path.read_text(encoding="utf-8"))
        indices = sorted({int(item["row_index"]) for item in key.values() if item.get("row_index") is not None})
        this_hash = (dataset.get("config") or {}).get("dataset_hash")
        key_hashes = {item.get("dataset_hash") for item in key.values() if item.get("dataset_hash")}
        if key_hashes and key_hashes != {this_hash}:
            raise click.UsageError("the batch was exported from a different dataset (dataset hash differs)")
        for item in key.values():
            row = item.get("row_index")
            if row is None:
                continue
            if item.get("generation_record_id") and full_views.generation_record_ids and \
                    str(full_views.generation_record_ids[int(row)]) != str(item["generation_record_id"]):
                raise click.UsageError("the batch's rows do not line up with this dataset (record ids differ)")
            if str(full_views.group_ids[int(row)]) != str(item.get("group_id")):
                raise click.UsageError("the batch was exported from a different dataset: group ids do not line up")
        views, pairs = subset_views(full_views, indices), restrict_pairs(full_pairs, indices)
        selection["batch"] = str(Path(subset[len('batch:'):]))
    else:
        raise click.UsageError("--subset must be 'all', 'paired', or 'batch:<dir>'")
    selection["n_rows"] = views.n
    selection["pairs"] = pairs.n_pairs
    click.echo(f"  rows judged: {views.n}/{full_views.n} (subset={subset}); matched pairs: {pairs.n_pairs}/{full_pairs.n_pairs}")
    cache = JudgeCache(path=target / "judge_cache.jsonl")
    accesses = [JudgeAccess.TRANSCRIPT_ONLY, JudgeAccess.FACT_INFORMED] if access == 'both' else [JudgeAccess(access)]
    runs = []
    costs = {}
    for spec in judges:
        ledger = CostLedger(path=target / "judge_cost_ledger.jsonl") if spec != "oracle" else None
        model = _judge_model(spec, region, ledger)
        for mode in accesses:
            def _progress(done, total, _spec=spec, _mode=mode.value):
                if done == total or done % 200 == 0:
                    click.echo(f"    {_spec} [{_mode}] {done}/{total} calls", err=True)
            runs.append(run_judge(views, model, access=mode, judge_name=spec, cache=cache, workers=workers,
                                  progress=_progress, rate_per_minute=rate_per_minute))
            click.echo(f"  judged {spec} [{mode.value}]: undetermined={runs[-1].n_undetermined}/{views.n} "
                       f"parse_failures={runs[-1].n_parse_failures} recovered={runs[-1].n_recovered} cache_hits={runs[-1].cache_hits}")
        if ledger is not None:
            costs[spec] = ledger.totals()
            click.echo(f"  cost {spec}: ${costs[spec].get('usd', 0.0):.2f} over {costs[spec].get('calls', 0)} calls")
    # Views already restricted to the pairs would print the same rows twice as
    # "all" and "paired"; a batch subset has no complete pairs to speak of.
    score_subsets = ("paired",) if subset == 'paired' else ("all",) if subset.startswith('batch:') else ("all", "paired")
    card = score_judges(views, runs, pairs=pairs, n_bootstrap=n_bootstrap, subsets=score_subsets)
    payload = card.to_dict()
    payload["selection"] = selection
    payload["costs"] = costs
    payload["dataset"] = {"path": str(dataset_path), "dataset_hash": (dataset.get("config") or {}).get("dataset_hash")}
    (target / "judges.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
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
@click.option('--model', required=True, help="'scripted', <scheme>:<model id>, or @<roster key>.")
@click.option('--preset', default="pilot", show_default=True)
@click.option('--counterpart-type', 'counterpart_types', multiple=True, default=("interrogative",),
              show_default=True, help="One run per policy (repeatable).")
@click.option('--run-id', default="", help="Run id prefix (default: timestamp).")
@click.option('--out', 'out_dir', type=click.Path(file_okay=False), default="runs", show_default=True)
@click.option('--region', default="us-east-1", show_default=True)
@click.option('--token-budget-per-day', type=click.IntRange(min=1), default=None,
              help='Pace the model under a daily token cap (tokens per rolling 24 hours, per process).')
@click.option('--counterpart-model', default=None,
              help="Fixed counterpart model ('bedrock:<id>' or 'scripted'). The main presets require one; "
                   "the pilot defaults to self play.")
def module_run(name, model, preset, counterpart_types, run_id, out_dir, region, token_budget_per_day,
               counterpart_model):
    """Run one preset of a module, once per counterpart policy."""
    import dataclasses
    import time as _time

    from end_bench.run import RunSpecError, run_text_track

    package = _load_benchmark_module(name)
    if preset not in package.PRESETS:
        raise click.UsageError(f"unknown preset {preset!r}; choose from {sorted(package.PRESETS)}")
    model = _resolve_model(model, "--model")
    counterpart_model = _resolve_model(counterpart_model, "--counterpart-model")
    benchmark = package.build_module()
    prefix = run_id or _time.strftime("%Y%m%d-%H%M%S")
    failures: list[tuple[str, str]] = []
    for policy in counterpart_types:
        try:
            spec = package.PRESETS[preset](model, counterpart_type=policy, run_id=f"{prefix}-{preset}-{policy}",
                                           counterpart_model=counterpart_model)
        except RunSpecError as error:
            raise click.UsageError(str(error)) from error
        spec = dataclasses.replace(spec, region=region, token_budget_per_day=token_budget_per_day)
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
    finished = [d for d in run_dirs if (Path(d) / "text_dataset.json").exists()]
    for skipped in sorted(set(run_dirs) - set(finished)):
        click.echo(f"skipped {skipped}: no text_dataset.json yet (run unfinished or failed)", err=True)
    if not finished:
        raise click.UsageError("no finished runs among the given directories")
    text = report.render_leaderboard(finished)
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


def _judge_model(spec: str, region: str, cost_ledger=None):
    if spec == "oracle":
        from end_bench.models.oracle_judge import OracleJudgeModel

        return OracleJudgeModel()
    if spec.startswith("bedrock:"):
        from end_bench.models.bedrock import BedrockTextModel

        return BedrockTextModel(spec, region=region, max_tokens=400, temperature=0.0, cost_ledger=cost_ledger)
    from end_bench.models.providers import ModelSpecError, build_text_model
    from end_bench.models.roster import resolve_model_spec

    try:
        resolved = resolve_model_spec(spec)
        if resolved == "scripted":
            raise ModelSpecError("the scripted model is not a judge")
        return build_text_model(resolved, max_tokens=400, temperature=0.0, cost_ledger=cost_ledger, region=region)
    except ModelSpecError as error:
        raise click.UsageError(f"unsupported judge {spec!r}: {error}; use 'oracle', <scheme>:<model id> or @<roster key>") from error


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


@cli.command('roster')
@click.option('--tier', type=click.Choice(["frontier", "mid", "budget"]), default=None, help="Only this tier.")
@click.option('--standard/--all', 'standard_only', default=False, show_default=True,
              help="Only the standard leaderboard set, or every entry.")
@click.option('--scheme', default=None, help="Only entries whose preferred route uses this scheme.")
@click.option('--providers', 'show_providers', is_flag=True, help="List the provider registry instead.")
@click.option('--json', 'as_json', is_flag=True, help="Machine readable output.")
def roster(tier, standard_only, scheme, show_providers, as_json):
    """The model roster (which models the benchmark runs on) and the provider registry."""
    from end_bench.models.providers import PROVIDERS
    from end_bench.models.roster import MODEL_ROSTER, MODEL_ROSTER_VERSION, format_roster, roster_json

    if show_providers:
        if as_json:
            click.echo(json.dumps([{**vars(p), "base_url_env": p.base_url_env, "credentials": p.credentials_status()}
                                   for p in PROVIDERS.values()], indent=2))
            return
        click.echo(f"{'scheme':11s} {'provider':30s} {'kind':13s} {'key variable':18s} {'creds':8s} base url")
        for p in PROVIDERS.values():
            click.echo(f"{p.scheme:11s} {p.display[:30]:30s} {p.kind:13s} {(p.key_env or '-'):18s} "
                       f"{p.credentials_status():8s} {p.base_url or '-'}")
        click.echo("\nUse <scheme>:<model id> as --model; <SCHEME>_BASE_URL overrides the base url for a proxy.")
        return
    entries = [e for e in MODEL_ROSTER.values()
               if (not standard_only or e.standard) and (tier is None or e.tier == tier)
               and (scheme is None or e.preferred.scheme == scheme)]
    if as_json:
        click.echo(roster_json(entries))
        return
    click.echo(f"model roster {MODEL_ROSTER_VERSION}: {len(entries)} entries")
    click.echo(format_roster(entries))
    click.echo("Use @<key> (or @<key>/<scheme>) anywhere a --model is accepted.")


@cli.command('info')
def info():
    """Print versions that stamp every run and build."""
    from end_bench import END_BENCH_VERSION
    from end_bench.data.text_dataset import TEXT_DATASET_SCHEMA_VERSION
    from end_bench.models.pricing import PRICE_TABLE_VERSION
    from end_bench.models.roster import MODEL_ROSTER_VERSION
    from interpretability.schema_registry import schema_registry, schema_registry_checksum

    payload = {
        "end_bench_version": END_BENCH_VERSION,
        "text_dataset_schema": TEXT_DATASET_SCHEMA_VERSION,
        "price_table_version": PRICE_TABLE_VERSION,
        "model_roster_version": MODEL_ROSTER_VERSION,
        "schema_registry_checksum": schema_registry_checksum(),
        "schema_registry": schema_registry(),
    }
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:  # pragma: no cover - console entry point
    cli()


__all__ = ["cli", "main"]
