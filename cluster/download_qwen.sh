#!/bin/bash
set -euo pipefail
# Pre-download the Qwen ceiling model to the PVC HF cache (CPU-only, GPU_REQUEST=0 →
# no GPU quota used; runs in parallel with wave-1 generation). Needed before wave-2
# Qwen serving. ~400 GB FP8.
#
# Submit:
#   GPU_REQUEST=0 CPU_REQUEST=8 MEMORY_REQUEST=32G RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-dl-qwen397 download_qwen.sh

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

apt-get update -qq && apt-get install -y -qq python3.10 python3-pip > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
# Version-stable Python API (the `huggingface-cli download` command was removed
# in hf-hub 1.20+ and renamed to `hf download`); hf_transfer is a separate pkg.
pip3 install -q --retries 10 huggingface_hub hf_transfer > /dev/null

mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "=== downloading $MODEL → $HF_HOME/hub ==="
df -h "$HF_CACHE_DIR" | tail -1
python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], max_workers=16)
print("SNAPSHOT:", p)
PY
echo "=== DONE ==="
du -sh "$HF_HOME/hub/models--Qwen--Qwen3.5-397B-A17B-FP8" 2>/dev/null || true
