#!/usr/bin/env bash
# Launch one EC2 instance ready for END-Bench and print how to connect.
# Usage: launch_ec2.sh [text|whitebox] [NAME]
#   text     -> c7i.xlarge (4 vCPU, 8 GB), CPU container, Bedrock calls only          (~$0.18/h)
#   whitebox -> g6e.xlarge (1x L40S 48 GB), GPU container for `deception run --hybrid` (~$1.86/h)
# Env: AWS_REGION (default us-east-1), KEY_NAME (optional SSH key pair; without it use SSM Session Manager)
# Requires bootstrap_aws.sh to have created the EndBenchEC2Role instance profile.
set -euo pipefail
TRACK="${1:-text}"
NAME="${2:-end-bench-${TRACK}}"
REGION="${AWS_REGION:-us-east-1}"
case "$TRACK" in
  text)     TYPE="c7i.xlarge"; DISK=60
            AMI="$(aws ssm get-parameter --region "$REGION" --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)" ;;
  whitebox) TYPE="g6e.xlarge"; DISK=200
            # Deep Learning Base AMI (Ubuntu 24.04) ships the NVIDIA driver; setup_ec2.sh adds the container toolkit.
            AMI="$(aws ec2 describe-images --region "$REGION" --owners amazon \
                    --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*' 'Name=state,Values=available' \
                    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)" ;;
  *) echo "track must be text or whitebox" >&2; exit 2 ;;
esac
KEY_ARG=()
[ -n "${KEY_NAME:-}" ] && KEY_ARG=(--key-name "$KEY_NAME")

# HttpPutResponseHopLimit=2 lets Docker containers reach the instance-profile credentials (IMDSv2).
INSTANCE_ID="$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
  --iam-instance-profile Name=EndBenchEC2Role ${KEY_ARG[@]+"${KEY_ARG[@]}"} \
  --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${DISK},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=project,Value=end-bench}]" \
  --query 'Instances[0].InstanceId' --output text)"
echo "launched ${INSTANCE_ID} (${TYPE}) as ${NAME}; waiting for it to start..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
cat <<EOF
ready.
  open a shell (no SSH key needed):  aws ssm start-session --region ${REGION} --target ${INSTANCE_ID}
  then on the instance:              curl -fsSL https://raw.githubusercontent.com/Gadfly-Research/multiagent-lab/main/end_bench/aws/setup_ec2.sh | bash
  STOP when idle (billing stops):    aws ec2 stop-instances  --region ${REGION} --instance-ids ${INSTANCE_ID}
  TERMINATE when finished:           aws ec2 terminate-instances --region ${REGION} --instance-ids ${INSTANCE_ID}
EOF
