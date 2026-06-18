#!/bin/bash
set -euo pipefail
# No-RAG generation — run_eval WITHOUT --rag, for the generator×retriever matrix's
# no-RAG baseline row. Judge after with run_cluster_value_gate_judge.sh.
#
# Submit (1x GPU):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-norag-3n run_cluster_norag_gen.sh \
#     MODEL=gemma3n-e4b DATASETS=kenya,healthbench_oss_eval \
#     OUT_DIR=/lightscratch/users/yiren/eval_output/norag_gemma3n \
#     REPO_REF=feat/r2-retriever-upgrade-20260613

CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma4-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
DATASETS="${DATASETS:-kenya,healthbench_oss_eval}"
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/norag_${MODEL}}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git cmake build-essential ninja-build > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir -q --retries 10 \
  llama-cpp-python numpy datasets huggingface_hub tqdm > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$OUT_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

echo "=== GENERATE (NO-RAG) $MODEL over datasets=$DATASETS ==="
python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" \
  --model-dir "$MODEL_DIR" --datasets "$DATASETS" \
  --output-dir "$OUT_DIR" --run-dir "$OUT_DIR/run"

echo "=== DONE — no-RAG generated (judge: run_cluster_value_gate_judge.sh RUN_DIRS=$OUT_DIR/run) ==="
ls -la "$OUT_DIR/run"
