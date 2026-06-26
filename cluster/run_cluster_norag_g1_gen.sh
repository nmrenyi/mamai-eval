#!/bin/bash
set -euo pipefail
# No-RAG generation WITH the G1 (deployed) system prompt — for the G1 no-RAG row of
# the Qwen-vs-Gemma capability ablation. Same as run_cluster_norag_gen.sh but injects
# G1 via --system-prompt (arm2_system_en.txt), so the Gemma no-RAG arms match the
# prompt used by the Qwen no-RAG job and the matrix's +RAG G1 cells.
#
# Submit (1 GPU each):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-g4-g1-norag run_cluster_norag_g1_gen.sh \
#     MODEL=gemma4-e4b DATASETS=kenya,healthbench_oss_eval REPO_REF=main
#   (gemma3n only needs healthbench — kenya G1 no-RAG already exists at g1-ab-3n arm2.)

CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma4-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
DATASETS="${DATASETS:-kenya,healthbench_oss_eval}"
PROMPT="${PROMPT:-g1}"                  # g1 | g1g2 | baseline
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/norag_g1_${MODEL}_20260624}"
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

mkdir -p "$HF_CACHE_DIR" "$OUT_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"
PFX="$WORKTREE/configs/$CONFIG/results/end_to_end_eval/g1-ab-3n-20260619/prompts"
case "$PROMPT" in
  g1)   SPF="$PFX/arm2_system_en.txt";;
  g1g2) SPF="$PFX/arm3_system_en.txt";;
  *)    SPF="";;
esac
SP=(); [ -n "$SPF" ] && SP=(--system-prompt "$SPF")
echo "=== prompt=$PROMPT  system_prompt_file=${SPF:-<config default>} ==="

echo "=== GENERATE (NO-RAG, $PROMPT) $MODEL over datasets=$DATASETS ==="
python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" \
  --model-dir "$MODEL_DIR" --datasets "$DATASETS" ${SP[@]+"${SP[@]}"} \
  --output-dir "$OUT_DIR" --run-dir "$OUT_DIR/run"

echo "=== DONE — no-RAG G1 generated ==="
ls -la "$OUT_DIR/run"
