# AWS setup for END-Bench (first-time guide)

You need four AWS things: a way to **pay** (credits + a budget alarm), a way to
**authenticate** without pasting keys into code (IAM), permission to call the
models (**Bedrock model access**), and somewhere durable to keep results (**S3**).
EC2 is optional for the text track and required only for the GPU white-box track.

Everything below is one-time. Allow about 45 minutes, most of it waiting.

## 0. Vocabulary

| Term | What it is | Where it shows up here |
|---|---|---|
| Region | A physical AWS location; resources and prices are per region | `us-east-1` everywhere (`AWS_REGION`) |
| IAM | Users, roles and policies that say who may call what | `iam-policy.json`, `EndBenchEC2Role`, `end-bench-local` |
| Access key | A key id + secret for a *user* (your laptop) | `~/.aws/credentials`, profile `end-bench` |
| Instance profile | A *role* attached to an EC2 instance so it needs no keys | `--iam-instance-profile Name=EndBenchEC2Role` |
| Bedrock | Pay-per-token hosted models (Claude, Llama, Mistral, Nova) | `end-bench run-text --model bedrock:<id>` |
| Inference profile | A routing id like `us.anthropic.claude-...` that on-demand Claude requires | `end-bench bedrock-models` prints them |
| S3 | Object storage, cents per GB-month | `END_BENCH_BUCKET=s3://<bucket>` |
| SSM Parameter Store | Encrypted key/value store | HF token at `/end-bench/hf_token` |
| EC2 | Rented VMs billed per second while running | `launch_ec2.sh`, `setup_ec2.sh` |

## 0b. Finding your way around the console

The console has three fixed anchors; everything below is reached from them.

- **Search bar**, top centre. Type a service name (Billing, IAM, Bedrock, S3,
  EC2) and press Enter. This beats hunting through menus every time.
- **Region picker**, top right, next to your account name. It must read
  **N. Virginia (us-east-1)** whenever you open Bedrock or EC2; resources in
  other regions are invisible from here and are billed separately.
- **Account menu**, top right (your account id or name). It holds *Billing and
  Cost Management* and *Security credentials* (root MFA lives there).

Inside a service the **left navigation pane** is often collapsed behind the
three-line icon at the top left. The breadcrumb trail above the page title tells
you where you are.

Console paths used in this guide (labels as of October 2026; if one has moved,
search for the bold term on the page):

| Goal | Path |
|---|---|
| Check credits | Search *Billing* → **Credits** in the left pane |
| Root MFA | Account menu → **Security credentials** → **Multi-factor authentication** → Assign MFA device |
| Admin user | Search *IAM* → **Users** → **Create user** → name `admin` → tick console access if you want it → **Next** → **Attach policies directly** → search `AdministratorAccess` → tick → **Next** → **Create user** |
| Admin access key | IAM → Users → `admin` → **Security credentials** tab → **Access keys** → **Create access key** → choose **Command Line Interface (CLI)** → tick the confirmation → **Next** → **Create access key** → copy both values or download the CSV now; the secret is never shown again |
| Bedrock model access | Search *Bedrock* → region N. Virginia → left pane, scroll to **Bedrock configurations** → **Model access** → **Modify model access** → tick models → **Next** → fill the Anthropic use-case form → **Submit** |
| G-instance quota (GPU only) | Search *Service Quotas* → **AWS services** → *Amazon EC2* → search **Running On-Demand G and VT instances** → **Request increase at account level** → 4 |
| Watch spend | Search *Billing* → **Cost Explorer**; or **Budgets** to see the alerts the bootstrap script created |

## 1. Account and credits (console, 10 min)

1. Create the account at aws.amazon.com (credit card required even with credits).
2. Billing console → **Credits** → redeem the code. Note the expiry date and
   which services the credits cover (Bedrock, EC2 and S3 are normally covered).
3. Turn on MFA for the root user, then **stop using root**. Everything below runs
   as an IAM identity.
4. IAM → Users → create an admin user for yourself (attach `AdministratorAccess`),
   give it console access and MFA, and create one access key "for CLI".
5. On your laptop:

   ```bash
   brew install awscli            # or: pip install awscli
   aws configure --profile admin   # paste the admin key id + secret, region us-east-1, output json
   aws sts get-caller-identity --profile admin
   ```

   The last command printing your account id means authentication works.

## 2. Bootstrap the project resources (one command, 2 min)

```bash
export AWS_PROFILE=admin AWS_REGION=us-east-1
END_BENCH_BUCKET_NAME=<yourname>-end-bench BUDGET_USD=300 ALERT_EMAIL=you@example.com \
  bash end_bench/aws/bootstrap_aws.sh
```

It creates the bucket (private, versioned), the least-privilege policy from
`iam-policy.json` with your bucket name filled in, the EC2 role + instance
profile, a laptop user `end-bench-local`, a monthly budget with 50/80/100 %
e-mail alerts, and (if `HF_TOKEN` is exported) the SSM parameter. The laptop
user's access key is written straight into `~/.aws/credentials` as profile
`end-bench` and is never printed, so it cannot end up in scrollback or shell
history. Docker mounts `~/.aws` read-only. Never paste a key anywhere in the repo.

From now on use the narrow profile, not admin:

```bash
export AWS_PROFILE=end-bench AWS_REGION=us-east-1 END_BENCH_BUCKET=s3://<yourname>-end-bench
```

## 3. Bedrock model access (console only, 5 min + approval wait)

Bedrock console → region **us-east-1** → **Model access** → *Modify model access*.
Tick the models you plan to run and judge with:

- Anthropic: Claude Sonnet 4.6, Claude Opus 4.8 (first Anthropic request asks for a
  short use-case form; approval is usually minutes, occasionally a day)
- Meta: Llama 4 Maverick
- Mistral AI: Mistral Large 3
- Amazon: Nova Pro

Then verify from your laptop:

```bash
end-bench bedrock-models --region us-east-1 --provider Anthropic
```

The last column lists inference-profile ids. If a Claude model shows one,
invoke it with that id (`bedrock:us.anthropic.claude-sonnet-4-6-...`); the
price table already strips the `us.` prefix when computing cost. `priced: no`
means tokens are still counted but dollars are not; add the id to a JSON passed
with `--price-table`.

## 4. First live call (about $1, 3 min)

```bash
end-bench estimate --model meta.llama4-maverick-17b-instruct-v1:0 --negotiations 8
end-bench run-text --model bedrock:meta.llama4-maverick-17b-instruct-v1:0 \
  --scenario hidden_value --condition high_incentive \
  --families 2 --rollouts-per-family 1 --max-rounds 2 --run-id live_smoke --out runs
end-bench build --dataset runs/live_smoke/text_dataset.json --out builds/live_smoke --n-bootstrap 200
cat runs/live_smoke/cost_ledger.jsonl | tail -1
aws s3 sync runs/live_smoke ${END_BENCH_BUCKET}/runs/live_smoke
```

If `run-text` fails with `AccessDeniedException`, model access (step 3) is not
approved yet or you are in the wrong region. `ValidationException ... inference
profile` means use the `us.` id from `bedrock-models`. `ExpiredTokenException`
means the shell has a stale profile; re-export `AWS_PROFILE`.

The same commands run inside the container (credentials come from the mounted `~/.aws`):

```bash
docker compose -f end_bench/docker/docker-compose.yml run --rm text run-text --model bedrock:... --out /data/runs
```

## 5. EC2 (only for long runs or the GPU track)

Text-track runs make thousands of sequential API calls; a laptop that sleeps
kills the run and there is deliberately no resume. Run Tier A from a small
always-on instance instead:

```bash
export AWS_PROFILE=admin                       # launching instances is outside the narrow policy
bash end_bench/aws/launch_ec2.sh text          # c7i.xlarge, ~$0.18/h
aws ssm start-session --target <instance id>   # browser-free shell, no SSH key or open port needed
```

On the instance:

```bash
sudo su - ubuntu
curl -fsSL https://raw.githubusercontent.com/Gadfly-Research/multiagent-lab/main/end_bench/aws/setup_ec2.sh | bash
cd ~/work/multiagent-lab
export END_BENCH_BUCKET=s3://<yourname>-end-bench AWS_REGION=us-east-1
end_bench/aws/run_text_track.sh meta.llama4-maverick-17b-instruct-v1:0 tierA_llama \
  --families 30 --rollouts-per-family 3 --paired-facts
tail -f data/runs/tierA_llama.log           # ctrl-c leaves the run going (it is under nohup)
end_bench/aws/sync_s3.sh tierA_llama        # when the log says it finished
```

The instance authenticates through its role; no keys exist on it. Docker can
reach those credentials only because `launch_ec2.sh` sets the metadata hop limit
to 2; if you launch by hand, set that or the container gets `NoCredentialsError`.

**Stop the instance when idle** (`aws ec2 stop-instances`); a stopped instance
costs only its disk (~$5/month for 60 GB). Terminate when the project is done.

For the white-box track: `launch_ec2.sh whitebox` picks a `g6e.xlarge` with the
NVIDIA driver preinstalled; `setup_ec2.sh` adds the container toolkit and builds
the GPU image. `run_whitebox_track.sh` reads the Hugging Face token from SSM at
launch time; it is never written to disk. New accounts often have a
**vCPU quota of 0 for G instances**; request a raise under Service Quotas →
EC2 → "Running On-Demand G and VT instances" to 4 before launching, and expect
a day for approval.

## 6. Cost control checklist

- Budget alerts are set by `bootstrap_aws.sh`; also open Billing → Cost Explorer
  weekly and filter by the `project=end-bench` tag.
- `end-bench estimate` before every design; `cost_ledger.jsonl` after. `end-bench prices --verify` checks the
  table against the AWS Price List API; Anthropic entries show as `not_in_pricelist` and are marked unverified.
- Tier A on Llama 4 Maverick is on the order of $15; on Claude Sonnet 4.6 about $150;
  judges on Opus 4.8 add roughly $50 per 3,000 turns. Credits usually cover all of it.
- Stopped EC2 still bills disk; terminated bills nothing. S3 at this scale is cents.
- Rotate the `end-bench-local` key if you ever suspect it leaked (IAM → Users → Security credentials).

## What never goes in the repo

Access keys, `~/.aws`, `.env.aws`, the HF token, raw transcripts, or unreviewed
datasets. `.gitignore` blocks `data/`; keep results in S3.
