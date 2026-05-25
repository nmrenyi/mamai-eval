#!/bin/bash
set -euo pipefail

# gpt-oss-120b judge — v0.1.0 validation pass.
#
# Runs gpt-oss-120b in BOTH modes against the v0.1.0 run dir:
#   1. categorize  — 145 Lynx FAIL cases → lynx_fail_categories_gpt_oss.json
#   2. calibrate   — 100 blinded cases   → calibration_verdicts_gpt_oss.json
#
# Both outputs land alongside Claude's existing v0.1.0 labels so the gate
# check can compare them in place. Set MODE=categorize or MODE=calibrate to
# run just one stage.
#
# gpt-oss-120b ships with MXFP4-native weights and fits a single 80GB GPU,
# but tensor-parallel across 2 GPUs is faster. Submit with:
#   GPU_REQUEST=2 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-gptoss-v01-val run_cluster_gpt_oss_judge.sh
#
# Smoke (5 cases per mode, single GPU):
#   GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-gptoss-smoke run_cluster_gpt_oss_judge.sh \
#     MAX_QUESTIONS=5 TENSOR_PARALLEL=1

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES ==="
# vLLM ≥0.10 has the kernels for gpt-oss MXFP4 weights + harmony chat template.
pip3 install --no-cache-dir --timeout 600 --retries 5 \
    'vllm>=0.10.0' 'transformers>=4.55' tqdm
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/faithfulness-eval}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
INPUT_DIR="${INPUT_DIR:-/lightscratch/users/yiren/eval_output/generator/gemma4-e4b/20260520T094749}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
OVERWRITE="${OVERWRITE:-}"
MODE="${MODE:-both}"   # categorize | calibrate | both
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

# Sync inputs from the cloned repo to the persistent scratch path. The repo
# has all v0.1.0 artifacts committed; scratch may only have the older
# oracle_responses.json + lynx_scores.json from the original Lynx run.
REPO_RUN_DIR="$WORKTREE/configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749"
mkdir -p "$INPUT_DIR"
echo "=== SYNCING INPUTS ==="
for f in lynx_scores.json oracle_responses.json calibration_blind.json lynx_fail_categories.json; do
  if [ ! -f "$INPUT_DIR/$f" ] && [ -f "$REPO_RUN_DIR/$f" ]; then
    echo "  copying $f from repo → $INPUT_DIR"
    cp "$REPO_RUN_DIR/$f" "$INPUT_DIR/$f"
  fi
done
# lynx_fail_cases.json is deterministic from lynx_scores + oracle_responses.
if [ ! -f "$INPUT_DIR/lynx_fail_cases.json" ]; then
  echo "  regenerating lynx_fail_cases.json via analyze_lynx_fails extract"
  python3 -m generator_eval.analyze_lynx_fails extract "$INPUT_DIR"
fi

run_mode() {
  local mode="$1"
  local input_name="$2"
  local output_name="$3"

  local input_path="$INPUT_DIR/$input_name"
  local output_path="$INPUT_DIR/$output_name"

  if [ ! -f "$input_path" ]; then
    echo "ERROR: $input_path not found"
    return 1
  fi

  local args=(
    "$mode"
    --input "$input_path"
    --output "$output_path"
    --model "$JUDGE_MODEL"
    --tensor-parallel "$TENSOR_PARALLEL"
    --reasoning-effort "$REASONING_EFFORT"
  )
  [ -n "$MAX_QUESTIONS" ] && args+=(--max-questions "$MAX_QUESTIONS")
  [ -n "$OVERWRITE" ] && args+=(--overwrite)

  echo
  echo "=== $mode ==="
  echo "INPUT_DIR=$INPUT_DIR"
  echo "JUDGE_MODEL=$JUDGE_MODEL"
  echo "TENSOR_PARALLEL=$TENSOR_PARALLEL  REASONING_EFFORT=$REASONING_EFFORT"
  echo "MAX_QUESTIONS=${MAX_QUESTIONS:-<all>}"

  python3 -m generator_eval.score_gpt_oss_judge "${args[@]}"
}

case "$MODE" in
  categorize)
    run_mode categorize lynx_fail_cases.json lynx_fail_categories_gpt_oss.json
    ;;
  calibrate)
    run_mode calibrate calibration_blind.json calibration_verdicts_gpt_oss.json
    ;;
  both)
    run_mode categorize lynx_fail_cases.json lynx_fail_categories_gpt_oss.json
    run_mode calibrate  calibration_blind.json calibration_verdicts_gpt_oss.json
    ;;
  *)
    echo "ERROR: unknown MODE=$MODE (categorize|calibrate|both)" ; exit 1 ;;
esac

echo
echo "=== RUN COMPLETE ==="
ls -la "$INPUT_DIR/"
