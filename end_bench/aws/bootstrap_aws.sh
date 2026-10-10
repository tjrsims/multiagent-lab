#!/usr/bin/env bash
# One-time account bootstrap for END-Bench. Run it ONCE from your laptop after
# `aws configure` with your admin (or IAM-user-with-admin) credentials.
#
# It creates, idempotently:
#   1. an S3 bucket for runs/builds            (END_BENCH_BUCKET)
#   2. an IAM policy from iam-policy.json      (least privilege: Bedrock invoke, that bucket, SSM /end-bench/*)
#   3. an IAM role + instance profile for EC2  (so instances need no pasted keys)
#   4. an IAM user for your laptop with the same policy, plus one access key (printed once)
#   5. an SSM SecureString /end-bench/hf_token (optional, for the white-box GPU track)
#   6. a monthly cost budget with e-mail alerts at 50% / 80% / 100%
#
# Usage:
#   END_BENCH_BUCKET_NAME=<globally-unique-name> BUDGET_USD=300 ALERT_EMAIL=you@example.com \
#     bash end_bench/aws/bootstrap_aws.sh
# Optional env: AWS_REGION (default us-east-1), HF_TOKEN (stored in SSM if set), SKIP_IAM_USER=1
set -euo pipefail
cd "$(dirname "$0")"

: "${END_BENCH_BUCKET_NAME:?set END_BENCH_BUCKET_NAME to a globally unique bucket name, e.g. yourname-end-bench}"
REGION="${AWS_REGION:-us-east-1}"
BUDGET_USD="${BUDGET_USD:-300}"
ALERT_EMAIL="${ALERT_EMAIL:-}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
echo "account ${ACCOUNT_ID}, region ${REGION}, bucket ${END_BENCH_BUCKET_NAME}"

# 1. bucket (private, versioned so an accidental overwrite is recoverable)
if ! aws s3api head-bucket --bucket "$END_BENCH_BUCKET_NAME" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$END_BENCH_BUCKET_NAME" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$END_BENCH_BUCKET_NAME" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
fi
aws s3api put-public-access-block --bucket "$END_BENCH_BUCKET_NAME" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-versioning --bucket "$END_BENCH_BUCKET_NAME" --versioning-configuration Status=Enabled
for prefix in runs builds adjudication reports; do
  aws s3api put-object --bucket "$END_BENCH_BUCKET_NAME" --key "${prefix}/" >/dev/null
done

# 2. policy
POLICY_NAME="EndBenchRunner"
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
POLICY_DOC="$(sed "s/REPLACE_ME-end-bench/${END_BENCH_BUCKET_NAME}/g" iam-policy.json)"
if aws iam get-policy --policy-arn "$POLICY_ARN" >/dev/null 2>&1; then
  # keep at most one non-default version, then publish the current file as default
  for v in $(aws iam list-policy-versions --policy-arn "$POLICY_ARN" --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text); do
    aws iam delete-policy-version --policy-arn "$POLICY_ARN" --version-id "$v"
  done
  aws iam create-policy-version --policy-arn "$POLICY_ARN" --policy-document "$POLICY_DOC" --set-as-default >/dev/null
else
  aws iam create-policy --policy-name "$POLICY_NAME" --policy-document "$POLICY_DOC" >/dev/null
fi

# 3. EC2 role + instance profile
ROLE_NAME="EndBenchEC2Role"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://ec2-trust-policy.json >/dev/null
fi
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN"
# SSM Session Manager lets you open a shell from the console without SSH keys or open ports.
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
if ! aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE_NAME" --role-name "$ROLE_NAME"
fi

# 4. laptop user (skip if you only ever run from EC2)
if [ "${SKIP_IAM_USER:-0}" != "1" ]; then
  USER_NAME="end-bench-local"
  if ! aws iam get-user --user-name "$USER_NAME" >/dev/null 2>&1; then
    aws iam create-user --user-name "$USER_NAME" >/dev/null
  fi
  aws iam attach-user-policy --user-name "$USER_NAME" --policy-arn "$POLICY_ARN"
  if [ "$(aws iam list-access-keys --user-name "$USER_NAME" --query 'length(AccessKeyMetadata)')" = "0" ]; then
    # The secret goes straight into ~/.aws/credentials (mode 600) via the CLI;
    # it is never echoed, so it cannot land in scrollback or shell history.
    aws iam create-access-key --user-name "$USER_NAME" \
      --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text \
      | {
          read -r key_id secret_key
          aws configure set aws_access_key_id "$key_id" --profile end-bench
          aws configure set aws_secret_access_key "$secret_key" --profile end-bench
          aws configure set region "$REGION" --profile end-bench
          aws configure set output json --profile end-bench
        }
    echo "wrote AWS profile 'end-bench' to ~/.aws/credentials (the secret was not printed)"
  else
    echo "IAM user ${USER_NAME} already has an access key; not creating another"
  fi
fi

# 5. HF token (only the GPU white-box track needs it)
if [ -n "${HF_TOKEN:-}" ]; then
  aws ssm put-parameter --region "$REGION" --name /end-bench/hf_token --type SecureString --value "$HF_TOKEN" --overwrite >/dev/null
  echo "stored HF token in SSM /end-bench/hf_token"
fi

# 6. budget
if [ -n "$ALERT_EMAIL" ]; then
  BUDGET_NAME="end-bench-monthly"
  if ! aws budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name "$BUDGET_NAME" >/dev/null 2>&1; then
    aws budgets create-budget --account-id "$ACCOUNT_ID" --budget \
      "{\"BudgetName\":\"${BUDGET_NAME}\",\"BudgetLimit\":{\"Amount\":\"${BUDGET_USD}\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}" \
      --notifications-with-subscribers "$(for pct in 50 80 100; do
        printf '{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":%s,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"%s"}]},' "$pct" "$ALERT_EMAIL"
      done | sed 's/,$//; s/^/[/; s/$/]/')"
    echo "budget ${BUDGET_NAME}: \$${BUDGET_USD}/month, alerts to ${ALERT_EMAIL}"
  fi
fi

cat <<EOF

bootstrap complete.
  export END_BENCH_BUCKET=s3://${END_BENCH_BUCKET_NAME}
  export AWS_REGION=${REGION}
  EC2 instance profile to attach at launch: ${ROLE_NAME}
Still manual (console only): Bedrock -> Model access -> request the models you want in ${REGION}.
Then verify:  end-bench bedrock-models --region ${REGION}
EOF
