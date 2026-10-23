# END-Bench (`end_bench/`)

The benchmark product layer, built additively on top of the research framework
in `interpretability/` and `negotiation/`. Nothing in the research packages was
deleted; this directory adds what the AWS runbook needs and ships it in
containers.

## What is here

| Path | Purpose |
|---|---|
| `models/bedrock.py` | Amazon Bedrock acting-model adapter (Converse API) that publishes faithful `GenerationRecord`s (`capture_mode='none'`), with retries and a cost ledger |
| `models/pricing.py`, `models/price_check.py` | Versioned Bedrock price table with per-entry provenance (`PRICE_SOURCES`), cost estimates, JSONL cost ledger that tracks dollars resting on unverified prices; `end-bench prices --verify` diffs the table against the AWS Price List API |
| `models/scripted.py` | Deterministic offline model so the whole pipeline runs with no network |
| `data/text_dataset.py` | `end_bench_text_dataset/1`: rows, labels, canonical records, content hash — the text-track artifact `benchmark.views` consumes |
| `run.py` | `RunSpec` + `run_text_track`: model → fail-closed execution plan → runner (injected model, text-only track) → text dataset + run manifest |
| `cli.py` | `end-bench run-text, smoke, build, pool, judge, adjudicate export/import, outcomes, report, estimate, prices, bedrock-models, module run/pilot/leaderboard, info` |
| `data/relabel.py` | `end-bench relabel`: re-read numeric claim turns under the current extractor and rule offline, with the original label records kept and every transition counted |
| `data/pooled.py` | `end-bench pool`: merge several runs' text datasets into one derived audit dataset (ids prefixed by run id, families shared, pairs never cross a run) for the pooled human batch and judge audit |
| `judges.py` | Judge testbed: transcript-only vs fact-informed judges scored against computed truth (AUROC, FPR/FNR, access gap); `oracle` judge for offline smoke |
| `adjudication.py` | Blinded, stratified rater batches with hidden gold items; Fleiss' kappa, gold accuracy, rule-vs-human agreement; `human_label` application so builds can run with `--label-source human` |
| `outcomes.py` | Agreement price and ZOPA share per trial from committed events; lying-vs-honest comparison with clustered CI |
| `report.py` | One markdown report over builds, judges, adjudication, and outcomes; renders recorded intervals, recomputes nothing |
| `docker/` | CPU image (text tracks, builds), GPU image (white-box capture via `deception run --hybrid`), compose file, entrypoint |
| `aws/` | `SETUP.md` first-time AWS guide, `bootstrap_aws.sh` (bucket, IAM, budget, SSM), `launch_ec2.sh`, `setup_ec2.sh`, run/sync/build scripts |

## Quick start

```bash
# offline smoke (no AWS): agents, adjudication, labels, QC, text dataset, build
end-bench smoke --families 2 --rollouts-per-family 1 --max-rounds 2 --out runs
end-bench build --dataset runs/<run_id>/text_dataset.json --out builds/<run_id> --n-bootstrap 200

# cost estimate for a Tier A design (3,240 negotiations)
end-bench estimate --model anthropic.claude-sonnet-4-6 --negotiations 3240

# Bedrock run (needs AWS credentials with bedrock:InvokeModel and model access enabled)
end-bench run-text --model bedrock:meta.llama4-maverick-17b-instruct-v1:0 \
  --scenario hidden_value --condition high_incentive --condition penalty \
  --families 30 --rollouts-per-family 3 --paired-facts --out runs
```

Containers:

```bash
docker compose -f end_bench/docker/docker-compose.yml build text
docker compose -f end_bench/docker/docker-compose.yml run --rm text smoke --families 1 --rollouts-per-family 1 --max-rounds 2 --out /data/runs
```

After a run (all offline):

```bash
end-bench relabel    --run-dir runs/<a> --run-dir runs/<b> --out relabeled/      # re-read stored turns under the current extractor; original label records stay
end-bench pool       --dataset runs/<a>/text_dataset.json --dataset runs/<b>/text_dataset.json --out pooled/text_dataset.json
end-bench judge      --dataset runs/<id>/text_dataset.json --out builds/<id> --judge oracle          # or --judge bedrock:<model_id>; a pooled dataset works too
end-bench judge      --dataset pooled/text_dataset.json --out judges/paired --judge bedrock:<id> --subset paired --workers 8   # twins only, 8 parallel calls; --subset batch:<dir> judges an exported human batch
end-bench outcomes   --dataset runs/<id>/text_dataset.json --out builds/<id>
end-bench adjudicate export --dataset runs/<id>/text_dataset.json --out batches/<id> --n 300
end-bench adjudicate import --batch batches/<id> --ratings r1.csv --ratings r2.csv --ratings r3.csv \
                            --dataset runs/<id>/text_dataset.json --out builds/<id>
end-bench build      --dataset builds/<id>/dataset_with_human_labels.json --out builds/<id>-human --label-source human
end-bench report     --build model_a=builds/<id> --out reports/tier_a.md
```

New to AWS: read `end_bench/aws/SETUP.md`, run `bootstrap_aws.sh` once, request Bedrock model access in the console, then
`end-bench bedrock-models` to see invocable ids. On EC2: `launch_ec2.sh`, `setup_ec2.sh`, then `run_text_track.sh`,
`run_whitebox_track.sh`, `sync_s3.sh`, `build_all.sh` (see each script's header).

## Providers and the model roster

A model is named `<scheme>:<model id>`. The scheme picks the provider from one registry (`end_bench/models/providers.py`): `bedrock` (Converse API, boto3 credentials), `scripted` (offline), and OpenAI compatible endpoints `openai`, `xai`, `gemini`, `deepseek`, `mistral`, `moonshot` (Kimi), `qwen` (Alibaba Model Studio), `zai` (GLM), `openrouter`, `together`. Each of those reads its key from `<SCHEME>_API_KEY` at call time and never writes it anywhere; `<SCHEME>_BASE_URL` overrides the endpoint for a proxy. Adding a provider is one row in the registry. `end-bench roster --providers` lists them with whether the key variable is present.

Sampling defaults to the provider's own (no temperature or top_p sent) because several frontier models reject any other value; `--temperature` forces one where the model allows it, and `--request-extra KEY=JSON` passes model specific fields such as a thinking effort level. The roster (`end_bench/models/roster.py`, versioned) says which models the benchmark runs on: tier, vendor, open weights, the peer deception and negotiation benchmarks that evaluated the model, and one or more routes (full specs). `@<key>` works anywhere a model is accepted and resolves to the preferred route; `@<key>/<scheme>` picks another route (`@grok-4.7/xai`). `end-bench roster --standard` prints the headline set; a route marked `CHECK` has an id that was not read from the provider's own listing and must be confirmed before a paid run. Prices for other providers live in the same table as Bedrock's, keyed by scheme, each with its source; unpriced models are counted in tokens, never guessed in dollars.

```bash
end-bench roster --paper                         # the conference set; --standard is the wider leaderboard set
end-bench module run deception_under_incentive --model @gpt-5.6-terra --counterpart-model @nova-pro --preset main_lite
END_BENCH_COUNTERPART=@nova-pro scripts/run_module_batch.sh @grok-4.3 grok43 data/main_b main_lite interrogative
```

## Benchmark modules

A benchmark lives under `benchmarks/<name>/` and passes a `BenchmarkModule` (policy registry, label tracks, closing attestation) into `run_text_track(spec, out, module=...)`. Run one with `end-bench module run <name> --model bedrock:<id> --preset pilot --counterpart-type interrogative`, then `end-bench module leaderboard <name> runs/<id>...`. The second module, `benchmarks/pressure_and_capitulation/`, adds eight pressure policies (four styles, two doses) and a per trial fact hold analysis; run it with `end-bench module run pressure_and_capitulation --model @key --preset pilot --counterpart-type pressure_expert_every --counterpart-model @nova-pro`, score pilots with `end-bench module pilot pressure_and_capitulation runs/<id>... --out pilot/` (keeps a cell only when 80% of trials hold a claim and 80% of those saw a pushback), and render `end-bench module leaderboard pressure_and_capitulation runs/<id>...` (one table per policy id; runs with different module versions, attestation timings, round counts or counterpart regimes are refused; an unpriced model shows no cost). Without `--counterpart-type` a module runs its `DEFAULT_POLICY`.

## Contracts added to the research packages (all additive)

- `InterpretabilityRunner(model=...)` injects a ready acting model on the
  `text_only` track only.
- `run_transactional_emergent_trial(pair_role=...)`,
  `run_emergent_study(paired_facts=...)`, `expand_rollouts(..., pair_roles=...)`,
  `ExecutionSurfacePlan.paired_facts`, and `deception run --paired-facts`.
- `benchmark.build.build_benchmark(dataset=...)` accepts a preloaded mapping.

## Honest limits

- Bedrock exposes neither token ids nor sampling seeds; records say so.
- Anthropic prices on Bedrock cannot be verified programmatically; the ledger's `unverified_usd` says how much of a cost figure rests on them.
- Containers drop an empty `AWS_PROFILE` so the EC2 instance role is used; the HF token is read from SSM at launch, never written to disk.
- No resume. A crashed run is a new run; the cost ledger and recovery
  snapshots remain for audit.
- The scripted model exercises the pipeline, not model behavior.
