#!/bin/bash
set -euo pipefail

# Generator faithfulness — stage 3 (Lynx variant).
# Runs Patronus Lynx 70B (open Llama-3-70B RAG hallucination detector) on
# Gemma's saved oracle_responses.json, producing a holistic PASS/FAIL
# faithfulness verdict + reasoning per response. Writes lynx_scores.json.
#
# Lynx 70B in fp16 is ~140 GB — needs 2 GPUs (tensor parallel). Submit with:
#   GPU_REQUEST=2 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-lynx-smoke run_cluster_lynx.sh MAX_QUESTIONS=5
#   GPU_REQUEST=2 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-lynx-gemma4 run_cluster_lynx.sh MAX_QUESTIONS=

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES ==="
# vllm bundles a CUDA-matched torch. --timeout/--retries for slow PyPI mirrors.
# stderr/stdout intentionally NOT silenced so install errors surface in logs.
pip3 install --no-cache-dir --timeout 600 --retries 5 vllm tqdm
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-main}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
INPUT_DIR="${INPUT_DIR:-/lightscratch/users/yiren/eval_output/generator/gemma4-e4b/20260520T094749}"
LYNX_MODEL="${LYNX_MODEL:-PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
OVERWRITE="${OVERWRITE:-}"     # set to 1 to ignore stale output (smoke iterations)
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

if [ ! -f "$INPUT_DIR/oracle_responses.json" ]; then
  echo "ERROR: $INPUT_DIR/oracle_responses.json not found"
  exit 1
fi

CMD_ARGS=(
  "$INPUT_DIR"
  --lynx-model "$LYNX_MODEL"
  --tensor-parallel "$TENSOR_PARALLEL"
)
if [ -n "$MAX_QUESTIONS" ]; then
  CMD_ARGS+=(--max-questions "$MAX_QUESTIONS")
fi
if [ -n "$OVERWRITE" ]; then
  CMD_ARGS+=(--overwrite)
fi

echo "=== STARTING LYNX SCORING ==="
echo "REPO_REF=$REPO_REF"
echo "INPUT_DIR=$INPUT_DIR"
echo "LYNX_MODEL=$LYNX_MODEL"
echo "TENSOR_PARALLEL=$TENSOR_PARALLEL"
echo "MAX_QUESTIONS=${MAX_QUESTIONS:-<all>}"

python3 -m generator_eval.score_lynx "${CMD_ARGS[@]}"

echo "=== RUN COMPLETE ==="
ls -la "$INPUT_DIR/"
