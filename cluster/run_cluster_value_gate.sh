#!/bin/bash
set -euo pipefail

# R2c P1 (step 2/3) — generate the 3-arm answers for the end-to-end value gate.
# For each arm (gecko / hybrid / hybrid_rerank) runs gemma4-e4b (the DEPLOYED
# on-device generator; app_config llm_model = gemma-4-E4B-it) over the SAQ + MCQ
# datasets with that arm's RAG context.
# MCQ is auto-scored inline; SAQ answers are judged in step 3 (gpt-oss-120b).
#
# Reads arm contexts from $ARMS_DIR/{gecko,hybrid,hybrid_rerank}/<dataset>.json
# (produced by run_cluster_rerank_arms.sh). Writes results to $OUT_DIR/<arm>/.
#
# Submit (GPU; llama.cpp CUDA):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-vg-gen run_cluster_value_gate.sh

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip ninja-build git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3
echo "=== INSTALLING PYTHON PACKAGES ==="
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir \
  llama-cpp-python pandas "openai>=1.0.0" tqdm datasets huggingface_hub > /dev/null 2>&1
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma4-e4b}"   # the on-device generator (app_config llm_model = gemma-4-E4B-it)
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
ARMS_DIR="${ARMS_DIR:-/lightscratch/users/yiren/eval_output/rag_arms}"
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/value_gate}"
# Use ${VAR-default} (not :-) so an explicitly-passed empty value stays empty
# (SAQ_DS="" / MCQ_DS="" genuinely disables that block instead of re-defaulting).
SAQ_DS="${SAQ_DS-kenya,afrimedqa_saq}"
MCQ_DS="${MCQ_DS-afrimedqa}"
MCQ_MAX="${MCQ_MAX:-600}"
ARMS="${ARMS:-gecko hybrid hybrid_rerank}"   # override to run one arm per GPU job
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

mkdir -p "$OUT_DIR" "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

run_arm() {
  local arm="$1"; local datasets="$2"; local extra="$3"
  [ -z "$datasets" ] && { echo "=== skip arm=$arm (no datasets) ==="; return 0; }
  echo "=== GENERATE arm=$arm datasets=$datasets ==="
  python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" \
    --model-dir "$MODEL_DIR" --datasets "$datasets" \
    --rag "$ARMS_DIR/$arm" --output-dir "$OUT_DIR/$arm" --run-dir "$OUT_DIR/$arm/run" \
    $extra
}

for ARM in $ARMS; do
  run_arm "$ARM" "$SAQ_DS" ""                       # SAQ — generation only, judged later
  run_arm "$ARM" "$MCQ_DS" "--max-questions $MCQ_MAX" # MCQ — auto-scored inline
done

echo "=== GENERATION COMPLETE ==="
find "$OUT_DIR" -name "*.json" | sort
