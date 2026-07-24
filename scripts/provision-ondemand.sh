#!/usr/bin/env bash
# On-demand L4 (no preemption) reusing the same pre-baked tooling image + GCS-staged model as
# the spot script from github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4. For stable multi-turn experiments.
set -euo pipefail
INSTANCE="${INSTANCE:-qwen36-mtp-l4-od}"
MACHINE="${MACHINE:-g2-standard-8}"
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-c us-west1-a us-west1-b europe-west4-a asia-east1-a asia-east1-b asia-east1-c asia-southeast1-b}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
GCS_MODEL="${GCS_MODEL:-gs://jinaai-dev-qwen36-mtp-l4/model.gguf}"
TOOLING_FAMILY="${TOOLING_FAMILY:-qwen36-mtp-l4-tooling}"

if gcloud compute images describe-from-family "$TOOLING_FAMILY" --project="$PROJECT" >/dev/null 2>&1; then
  IMG=(--image-family="$TOOLING_FAMILY" --image-project="$PROJECT")
else
  IMG=(--image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release)
fi

STARTUP="$(mktemp)"; cat > "$STARTUP" <<SH
#!/bin/bash
set -e
exec >>/var/log/qwen-startup.log 2>&1
if nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | grep -qi Enabled; then
  nvidia-smi -e 0 || true; reboot; exit 0
fi
mkdir -p /opt/models; MODEL=/opt/models/model.gguf
( command -v docker >/dev/null || { apt-get update -qq; apt-get install -y -qq docker.io; nvidia-ctk runtime configure --runtime=docker; systemctl restart docker; }
  docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda >/dev/null 2>&1 || docker pull ghcr.io/ggml-org/llama.cpp:server-cuda ) &
DPID=\$!
( [ -f "\$MODEL" ] || gcloud storage cp "$GCS_MODEL" "\$MODEL" ) &
MPID=\$!
wait \$DPID; wait \$MPID
docker rm -f llama-server 2>/dev/null || true
docker run -d --name llama-server --restart unless-stopped --gpus all -p 8080:8080 \
  -v /opt/models:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/model.gguf --alias Qwen3.6-35B-A3B-Q4KXL-MTP --host 0.0.0.0 --port 8080 --jinja --tools all \
  --ctx-size 56320 --parallel 1 --flash-attn on -ngl 99 --n-cpu-moe 0 -ub 64 -b 512 \
  --no-mmap --threads 8 --spec-type draft-mtp --spec-draft-n-max 2 --no-warmup --metrics
SH

gcloud compute firewall-rules create allow-llama-8080 \
  --allow=tcp:8080 --target-tags=llama-server --source-ranges=0.0.0.0/0 2>/dev/null || true

ZONE=""
for z in $ZONES; do
  echo "Trying on-demand $MACHINE in $z ..."
  if gcloud compute instances create "$INSTANCE" --zone="$z" --machine-type="$MACHINE" \
      --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
      "${IMG[@]}" --boot-disk-size=80GB --boot-disk-type=pd-ssd --tags=llama-server \
      --metadata-from-file=startup-script="$STARTUP"; then ZONE="$z"; break; fi
  echo "No capacity in $z, next ..."
done
rm -f "$STARTUP"
[ -n "$ZONE" ] || { echo "No on-demand L4 capacity in any zone"; exit 1; }

IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
echo "Created on-demand $INSTANCE in $ZONE (IP $IP). Waiting for readiness..."
until curl -fsS --max-time 5 "http://$IP:8080/health" 2>/dev/null | grep -q '"status":"ok"'; do printf '.'; sleep 10; done
echo; echo "READY  API http://$IP:8080/v1/chat/completions  (model Qwen3.6-35B-A3B-Q4KXL-MTP)"
echo "IP=$IP"
