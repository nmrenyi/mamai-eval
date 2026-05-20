#!/bin/bash
set -euo pipefail

# Generator faithfulness — stage 3: MiniCheck scoring (response-level / solution A).
# Loads Gemma's saved oracle_responses.json from scratch and runs the Bespoke
# MiniCheck-7B model to compute P(supported) per response. Writes
# minicheck_scores_A.json alongside the input.
#
# Smoke test:
#   ./submit_job.sh mamai-minicheck-smoke run_cluster_minicheck.sh MAX_QUESTIONS=5
# Full run (uses default INPUT_DIR pointing at the v0.2 gemma4-e4b run):
#   ./submit_job.sh mamai-minicheck-gemma4 run_cluster_minicheck.sh MAX_QUESTIONS=
#
# To target a different generator run, pass INPUT_DIR=/path/to/<ts>/.

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES ==="
# We call Bespoke-MiniCheck-7B directly via transformers — the upstream
# `minicheck` package isn't on PyPI and its GitHub pyproject.toml is missing
# a `name` field, breaking pip installs. Direct transformers also drops the
# vllm dependency entirely; per-call latency on A100 is ~100ms which is fine
# for the ~2.7K-call scale of this stage.
# stderr/stdout intentionally NOT silenced so install errors surface in logs.
pip3 install --no-cache-dir \
    torch transformers accelerate huggingface_hub tqdm
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-main}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
INPUT_DIR="${INPUT_DIR:-/lightscratch/users/yiren/eval_output/generator/gemma4-e4b/20260520T094749}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
MINICHECK_CACHE="${MINICHECK_CACHE:-/lightscratch/users/yiren/minicheck_cache}"

mkdir -p "$HF_CACHE_DIR" "$MINICHECK_CACHE"
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
  --threshold "$THRESHOLD"
  --cache-dir "$MINICHECK_CACHE"
  --batch-size "$BATCH_SIZE"
)
if [ -n "$MAX_QUESTIONS" ]; then
  CMD_ARGS+=(--max-questions "$MAX_QUESTIONS")
fi

echo "=== STARTING MINICHECK SCORING ==="
echo "REPO_REF=$REPO_REF"
echo "INPUT_DIR=$INPUT_DIR"
echo "THRESHOLD=$THRESHOLD"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_QUESTIONS=${MAX_QUESTIONS:-<all>}"

python3 -m generator_eval.score_minicheck "${CMD_ARGS[@]}"

echo "=== RUN COMPLETE ==="
ls -la "$INPUT_DIR/"
