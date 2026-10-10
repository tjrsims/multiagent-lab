"""The AWS helper assets must stay parseable and least-privilege."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

AWS_DIR = Path(__file__).resolve().parents[2] / "end_bench" / "aws"


def test_iam_policy_is_least_privilege_json() -> None:
    policy = json.loads((AWS_DIR / "iam-policy.json").read_text())
    statements = {s["Sid"]: s for s in policy["Statement"]}
    invoke = statements["BedrockInvoke"]
    assert {"bedrock:InvokeModel", "bedrock:Converse"} <= set(invoke["Action"])
    # Invoke is scoped to model and inference-profile ARNs, never '*'.
    assert invoke["Resource"] != "*" and all(r.startswith("arn:aws:bedrock:") for r in invoke["Resource"])
    assert any("inference-profile/*" in r for r in invoke["Resource"])
    catalog = set(statements["BedrockCatalog"]["Action"])
    assert {"bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles",
            "bedrock:GetFoundationModelAvailability"} <= catalog
    assert all(a.startswith(("bedrock:List", "bedrock:Get")) for a in catalog)
    bucket = statements["ArtifactBucket"]
    assert "s3:DeleteObject" not in bucket["Action"]  # sync never deletes
    assert all(r.startswith("arn:aws:s3:::REPLACE_ME-end-bench") for r in bucket["Resource"])
    assert statements["SecretsForHuggingFace"]["Resource"].endswith(":parameter/end-bench/*")
    text = json.dumps(policy)
    assert not any(a.endswith(":*") or a == "*" for s in statements.values() for a in s["Action"])
    assert "iam:" not in text and "ec2:" not in text


def test_ec2_trust_policy_only_trusts_ec2() -> None:
    trust = json.loads((AWS_DIR / "ec2-trust-policy.json").read_text())
    [statement] = trust["Statement"]
    assert statement["Principal"] == {"Service": "ec2.amazonaws.com"}
    assert statement["Action"] == "sts:AssumeRole"


@pytest.mark.parametrize("script", ["bootstrap_aws.sh", "launch_ec2.sh", "setup_ec2.sh",
                                    "run_text_track.sh", "run_whitebox_track.sh", "sync_s3.sh", "build_all.sh"])
def test_shell_scripts_parse_and_fail_closed(script: str) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    path = AWS_DIR / script
    subprocess.run([bash, "-n", str(path)], check=True)
    assert "set -euo pipefail" in path.read_text()


def test_bootstrap_substitutes_bucket_placeholder_and_never_prints_the_secret() -> None:
    text = (AWS_DIR / "bootstrap_aws.sh").read_text()
    assert "REPLACE_ME-end-bench" in text and "END_BENCH_BUCKET_NAME" in text
    assert "SecureString" in text
    # The access key is piped straight into `aws configure set`; no echo/print of it.
    assert "aws configure set aws_secret_access_key \"$secret_key\"" in text
    assert "print \"aws configure set" not in text and "echo \"$secret" not in text
    code_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    redirects = [
        target
        for line in code_lines
        for target in re.findall(r"(?:^|\s)[12]?>{1,2}\s*(\S+)", line)
        if not target.startswith(("/dev/null", "&"))
    ]
    assert redirects == [], redirects


def test_launch_script_handles_empty_key_array_on_bash_3() -> None:
    text = (AWS_DIR / "launch_ec2.sh").read_text()
    assert "HttpPutResponseHopLimit=2" in text
    assert '${KEY_ARG[@]+"${KEY_ARG[@]}"}' in text


def test_hf_token_is_fetched_at_run_time_not_parked_on_disk() -> None:
    setup = (AWS_DIR / "setup_ec2.sh").read_text()
    assert ".env.aws" not in setup
    whitebox = (AWS_DIR / "run_whitebox_track.sh").read_text()
    assert "ssm get-parameter --with-decryption" in whitebox and "/end-bench/hf_token" in whitebox
    assert "export HF_TOKEN" in whitebox


def test_entrypoint_drops_empty_aws_profile() -> None:
    entrypoint = (AWS_DIR.parent / "docker" / "entrypoint.sh").read_text()
    assert 'if [ -z "${AWS_PROFILE:-}" ]; then' in entrypoint and "unset AWS_PROFILE" in entrypoint
