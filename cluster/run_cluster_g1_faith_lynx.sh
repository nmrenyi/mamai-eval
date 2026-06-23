#!/bin/bash
set -euo pipefail
# G1/G2 faithfulness — Lynx scoring for all 3 arms in ONE job (2 GPUs, TP=2).
# Scores each arm's oracle_responses.json sequentially → lynx_scores.json in place.
# One deps install + clone; the Lynx-70B weight load repeats per arm (acceptable
# vs. 3 separate jobs each doing deps+clone+load).
#
# Submit (2x GPU, 80GB-class each):
#   NODE_POOL=h100 GPU_REQUEST=2 MEMORY_REQUEST=96G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-g1f-lynx run_cluster_g1_faith_lynx.sh

LYNX_MODEL="${LYNX_MODEL:-PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
OVERWRITE="${OVERWRITE:-}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/g1-prompt-fix-20260611}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
FBASE="${FBASE:-/lightscratch/users/yiren/eval_output/g1_faith_3n_20260619}"
RUN_DIRS="${RUN_DIRS:-$FBASE/arm1 $FBASE/arm2 $FBASE/arm3}"

echo "=== DEPS ==="
apt-get update && apt-get install -y python3.10 python3-pip git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir --timeout 600 --retries 5 vllm tqdm
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

# shellcheck disable=SC2206
DIRS=($RUN_DIRS)
echo "=== LYNX scoring ${#DIRS[@]} arm dir(s) ==="; printf '  %s\n' "${DIRS[@]}"
for d in "${DIRS[@]}"; do
  if [ ! -f "$d/oracle_responses.json" ]; then echo "WARN: $d/oracle_responses.json missing; skip"; continue; fi
  echo "##### LYNX over $d #####"
  CMD_ARGS=("$d" --lynx-model "$LYNX_MODEL" --tensor-parallel "$TENSOR_PARALLEL")
  [ -n "$MAX_QUESTIONS" ] && CMD_ARGS+=(--max-questions "$MAX_QUESTIONS")
  [ -n "$OVERWRITE" ] && CMD_ARGS+=(--overwrite)
  python3 -m generator_eval.score_lynx "${CMD_ARGS[@]}"
done

echo "=== DONE — lynx_scores.json written per arm ==="
for d in "${DIRS[@]}"; do echo "--- $d ---"; ls -la "$d" 2>/dev/null | grep -iE 'lynx|oracle' || true; done
