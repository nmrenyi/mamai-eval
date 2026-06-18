#!/bin/bash
set -euo pipefail
# R2c Phase 3 (generate) — build the EmbeddingGemma arm from its Phase-2 retrievals and
# generate kenya SAQ answers with the deployed generator (gemma4-e4b GGUF). Judge after
# with run_cluster_value_gate_judge.sh (RUN_DIRS=$OUT_DIR/embeddinggemma/run).
#
# Submit (1x H200):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-vg-eg-gen run_cluster_value_gate_eg.sh DATASETS=kenya

CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma4-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
DATASETS="${DATASETS:-kenya}"
ARM="${ARM:-embeddinggemma}"
RETR="${RETR:-/lightscratch/users/yiren/eval_output/screen_embedder/retrievals_embeddinggemma_300m_d768.json}"
ARMS_DIR="${ARMS_DIR:-/lightscratch/users/yiren/eval_output/rag_arms_eg}"
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/value_gate_eg}"
REPO_REF="${REPO_REF:-main}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git cmake build-essential ninja-build > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir -q --retries 10 \
  llama-cpp-python numpy datasets huggingface_hub tqdm > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$ARMS_DIR/$ARM" "$OUT_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

echo "=== BUILD ARM ($ARM) from $RETR ==="
python3 -m retrieval_eval.screen_embedder arm_format --retrievals "$RETR" --out-dir "$ARMS_DIR/$ARM" --top-k 3
ls -la "$ARMS_DIR/$ARM"

echo "=== GENERATE $MODEL over arm=$ARM datasets=$DATASETS ==="
python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" \
  --model-dir "$MODEL_DIR" --datasets "$DATASETS" \
  --rag "$ARMS_DIR/$ARM" --output-dir "$OUT_DIR/$ARM" --run-dir "$OUT_DIR/$ARM/run"

echo "=== DONE — generated (judge next with run_cluster_value_gate_judge.sh RUN_DIRS=$OUT_DIR/$ARM/run) ==="
ls -la "$OUT_DIR/$ARM/run"