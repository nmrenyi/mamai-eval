#!/bin/bash
set -euo pipefail
# Generator×prompt matrix — Track A (end-to-end, +RAG) for ONE cell (generator × prompt).
# Generates kenya + healthbench_oss_eval WITH the deployed EmbeddingGemma top-3 retrievals
# (precomputed in rag_arms_eg / rag_arms_eg_hb). Judge after with run_cluster_g1_judge.sh.
#
# Submit (1x GPU per cell, raced H200>H100>A100):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=48G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-mx-a-3n-baseline run_cluster_matrix_e2e.sh \
#     MODEL=gemma3n-e4b OUT_DIR=/lightscratch/users/yiren/eval_output/gen_prompt_matrix_20260622/A/3n_baseline
#   ... add SYSTEM_PROMPT=configs/.../prompts/arm2_system_en.txt for the +G1 / +G1+G2 cells.

CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma3n-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
EG_BASE="${EG_BASE:-/lightscratch/users/yiren/eval_output}"
EG_KENYA="${EG_KENYA:-$EG_BASE/rag_arms_eg/embeddinggemma}"
EG_HB="${EG_HB:-$EG_BASE/rag_arms_eg_hb/embeddinggemma}"
REPO_REF="${REPO_REF:-feat/g1-prompt-fix-20260611}"
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

SP=()
if [ -n "$SYSTEM_PROMPT" ]; then
  [ -f "$SYSTEM_PROMPT" ] || { echo "ERROR: SYSTEM_PROMPT not in repo: $SYSTEM_PROMPT"; exit 1; }
  SP=(--system-prompt "$SYSTEM_PROMPT"); echo "=== prompt override: $SYSTEM_PROMPT ==="
else
  echo "=== baseline prompt (no override) ==="
fi

echo "=== GENERATE kenya (+RAG EmbeddingGemma) $MODEL ==="
python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" --model-dir "$MODEL_DIR" \
  --datasets kenya --rag "$EG_KENYA" ${SP[@]+"${SP[@]}"} \
  --output-dir "$OUT_DIR" --run-dir "$OUT_DIR/run"

echo "=== GENERATE healthbench_oss_eval (+RAG EmbeddingGemma) $MODEL ==="
python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" --model-dir "$MODEL_DIR" \
  --datasets healthbench_oss_eval --rag "$EG_HB" ${SP[@]+"${SP[@]}"} \
  --output-dir "$OUT_DIR" --run-dir "$OUT_DIR/run"

echo "=== DONE — Track A cell generated ==="
ls -la "$OUT_DIR/run"
