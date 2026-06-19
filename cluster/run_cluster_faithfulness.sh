#!/bin/bash
set -euo pipefail

# Generator faithfulness — stage 2 (generation pass).
# Runs `python -m generator_eval.eval_faithfulness` against the committed
# mamaretrieval oracle and writes oracle_responses.json to scratch.
#
# Stage 3 (MiniCheck scoring) is post-hoc on the saved JSON — not done here.
#
# Smoke test:
#   ./submit_job.sh mamai-faithfulness-smoke run_cluster_faithfulness.sh MAX_QUESTIONS=5
# Full run:
#   ./submit_job.sh mamai-faithfulness-gemma4 run_cluster_faithfulness.sh

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip ninja-build git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3
echo "=== INSTALLING PYTHON PACKAGES ==="
# llama-cpp-python compiled with CUDA. `datasets`+`huggingface_hub` aren't
# strictly required here (the oracle is a committed JSONL) but are kept so
# shared.dataset_loader still imports cleanly if any future helper touches it.
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir \
  llama-cpp-python numpy pandas tqdm datasets huggingface_hub \
  > /dev/null 2>&1
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-main}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma4-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lightscratch/users/yiren/eval_output}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
ORACLE="${ORACLE:-}"          # default resolved after checkout (in-tree path)
TOP_K="${TOP_K:-3}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-}"
RUN_DIR="${RUN_DIR:-}"
LOG_DIR="${LOG_DIR:-}"
RESUME="${RESUME:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"   # A/B prompt arms: repo-relative path to an arm's system_en.txt

mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

# Default oracle: committed file from the checked-out tree (v0.2.0 top-20
# union, score>=5 — the current canonical revision). To rerun on v0.1.0,
# pass ORACLE=…/mamaretrieval-v0.1.0-score5.jsonl explicitly.
if [ -z "$ORACLE" ]; then
  ORACLE="$WORKTREE/configs/$CONFIG/oracle/mamaretrieval-v0.2.0-score5.jsonl"
fi
if [ ! -f "$ORACLE" ]; then
  echo "ERROR: oracle file not found at $ORACLE"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
if [ -z "$RUN_DIR" ]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%S)"
  RUN_DIR="$OUTPUT_ROOT/generator/$MODEL/$RUN_ID"
fi
mkdir -p "$RUN_DIR"
if [ -z "$LOG_DIR" ]; then
  LOG_DIR="$RUN_DIR/logs"
fi
mkdir -p "$LOG_DIR"

CMD_ARGS=(
  --config "$CONFIG"
  --model "$MODEL"
  --model-dir "$MODEL_DIR"
  --oracle "$ORACLE"
  --top-k "$TOP_K"
  --output-dir "$OUTPUT_ROOT/generator"
  --run-dir "$RUN_DIR"
)

if [ -n "$MAX_QUESTIONS" ]; then
  CMD_ARGS+=(--max-questions "$MAX_QUESTIONS")
fi
if [ -n "$MAX_TOKENS" ]; then
  CMD_ARGS+=(--max-tokens "$MAX_TOKENS")
fi
if [ -n "$N_GPU_LAYERS" ]; then
  CMD_ARGS+=(--n-gpu-layers "$N_GPU_LAYERS")
fi
if [ -n "$RESUME" ]; then
  CMD_ARGS+=(--resume "$RESUME")
fi
if [ -n "$SYSTEM_PROMPT" ]; then
  if [ ! -f "$SYSTEM_PROMPT" ]; then echo "ERROR: SYSTEM_PROMPT not found in repo: $SYSTEM_PROMPT"; exit 1; fi
  CMD_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
  echo "SYSTEM_PROMPT override: $SYSTEM_PROMPT"
fi

echo "=== STARTING FAITHFULNESS GENERATION ==="
echo "CONFIG=$CONFIG"
echo "MODEL=$MODEL"
echo "REPO_REF=$REPO_REF"
echo "RUN_DIR=$RUN_DIR"
echo "ORACLE=$ORACLE"
echo "TOP_K=$TOP_K"
echo "MAX_QUESTIONS=${MAX_QUESTIONS:-<all>}"

LOG_PATH="$LOG_DIR/faithfulness_${MODEL}.log"
python3 -m generator_eval.eval_faithfulness "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo "=== RUN COMPLETE ==="
find "$RUN_DIR" -maxdepth 2 -type f | sort
