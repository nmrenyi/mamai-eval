#!/bin/bash
set -euo pipefail
# G1/G2 prompt A/B — no-RAG generation for ONE arm on Gemma 3n E4B.
# Only the open-ended system prompt varies across arms (--system-prompt override);
# everything else (model, params, datasets) is held constant. Judge after with
# rescore_saq.sh (SAQ sets) + rescore_rubric.sh (healthbench).
#
# Submit one job per arm (1x GPU each). Arm 1 = config baseline (no override):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=48G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-g1-arm1 run_cluster_g1_arm.sh ARM=arm1
#   NODE_POOL=h100 ... ./submit_job.sh mamai-g1-arm2 run_cluster_g1_arm.sh \
#     ARM=arm2 SYSTEM_PROMPT=configs/config-v0.2.0/results/end_to_end_eval/g1-ab-3n-20260619/prompts/arm2_system_en.txt
#   NODE_POOL=default ... ./submit_job.sh mamai-g1-arm3 run_cluster_g1_arm.sh \
#     ARM=arm3 SYSTEM_PROMPT=configs/config-v0.2.0/results/end_to_end_eval/g1-ab-3n-20260619/prompts/arm3_system_en.txt

CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma3n-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
DATASETS="${DATASETS:-kenya,afrimedqa_saq,whb,healthbench_oss_eval}"
ARM="${ARM:-arm1}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
OUT_BASE="${OUT_BASE:-/lightscratch/users/yiren/eval_output/g1_ab_3n_20260619}"
OUT_DIR="$OUT_BASE/$ARM"
REPO_REF="${REPO_REF:-feat/g1-prompt-fix-20260611}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code_$ARM}"
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

SP_ARG=()
if [ -n "$SYSTEM_PROMPT" ]; then
  if [ ! -f "$SYSTEM_PROMPT" ]; then echo "ERROR: SYSTEM_PROMPT not found in repo: $SYSTEM_PROMPT"; exit 1; fi
  SP_ARG=(--system-prompt "$SYSTEM_PROMPT")
  echo "=== ARM $ARM — system-prompt OVERRIDE: $SYSTEM_PROMPT ==="
  sha256sum "$SYSTEM_PROMPT" || true
else
  echo "=== ARM $ARM — config baseline prompt (no override) ==="
fi

echo "=== GENERATE (NO-RAG) $MODEL over datasets=$DATASETS ==="
python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" \
  --model-dir "$MODEL_DIR" --datasets "$DATASETS" \
  ${SP_ARG[@]+"${SP_ARG[@]}"} \
  --output-dir "$OUT_DIR" --run-dir "$OUT_DIR/run"

echo "=== DONE — $ARM generated ==="
ls -la "$OUT_DIR/run"
