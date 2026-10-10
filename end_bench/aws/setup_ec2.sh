#!/usr/bin/env bash
# One-time EC2 setup (Ubuntu 24.04). Installs Docker (and the NVIDIA container
# toolkit when a GPU is present), clones the repo, builds the images, and pulls
# the Hugging Face token from SSM Parameter Store (/end-bench/hf_token).
# Usage: bash setup_ec2.sh [REPO_URL] [BRANCH]
set -euo pipefail
REPO_URL="${1:-https://github.com/Gadfly-Research/multiagent-lab.git}"
BRANCH="${2:-main}"

sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl git unzip jq
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi
if ! command -v aws >/dev/null 2>&1; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  (cd /tmp && unzip -q -o awscliv2.zip && sudo ./aws/install)
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  fi
fi

mkdir -p "$HOME/work" && cd "$HOME/work"
if [ -d multiagent-lab ]; then (cd multiagent-lab && git fetch -q && git checkout -q "$BRANCH" && git pull -q); else git clone -q --branch "$BRANCH" "$REPO_URL"; fi
cd multiagent-lab
mkdir -p data/runs data/builds

# The Hugging Face token stays in SSM (/end-bench/hf_token); run_whitebox_track.sh
# reads it into the environment at launch time, so nothing secret is written here.
sudo docker compose -f end_bench/docker/docker-compose.yml build text
if command -v nvidia-smi >/dev/null 2>&1; then
  sudo docker compose -f end_bench/docker/docker-compose.yml build whitebox
fi
sudo docker compose -f end_bench/docker/docker-compose.yml run --rm text info
echo "setup complete; next: end_bench/aws/run_text_track.sh <MODEL_ID> <RUN_ID> [extra end-bench args]"
