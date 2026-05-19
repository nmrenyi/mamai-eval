#!/bin/bash
set -euo pipefail

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip ninja-build git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES ==="
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir llama-cpp-python pandas "openai>=1.0.0" tqdm > /dev/null 2>&1
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-main}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-/lightscratch/users/yiren/eval_code/datasets}"
CONFIG="${CONFIG:-config-v0.1.0}"
MODEL="${MODEL:-gemma4-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
OUTPUT_DIR="${OUTPUT_DIR:-/lightscratch/users/yiren/eval_output}"
JUDGE="${JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-}"

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"
rm -rf datasets
ln -s "$DATA_SOURCE_DIR" datasets

mkdir -p "$OUTPUT_DIR"

OPEN_DS=(kenya_vignettes whb_stumps afrimedqa_saq)

COMMON_ARGS=(
  --config "$CONFIG"
  --model "$MODEL"
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
)

if [ "$JUDGE" = "1" ]; then
  COMMON_ARGS+=(--judge)
  if [ -n "$JUDGE_MODEL" ]; then
    COMMON_ARGS+=(--judge-model "$JUDGE_MODEL")
  fi
fi

if [ -n "$N_GPU_LAYERS" ]; then
  COMMON_ARGS+=(--n-gpu-layers "$N_GPU_LAYERS")
fi

echo "=== STARTING GEMMA 4 E4B OPEN-ENDED EVALUATIONS (NO RAG) ==="
echo "CONFIG=$CONFIG"
echo "MODEL=$MODEL"
echo "REPO_REF=$REPO_REF"

PIDS=()
for DS in "${OPEN_DS[@]}"; do
  LOG_PATH="${OUTPUT_DIR}/eval_${MODEL}_open_${DS}.log"
  python3 run_eval.py "${COMMON_ARGS[@]}" --datasets "$DS" > "$LOG_PATH" 2>&1 &
  PID="$!"
  PIDS+=("$PID")
  echo "Started no-RAG $DS (PID $PID) -> $LOG_PATH"
done

echo "=== WAITING FOR RUNS ==="
FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAIL=1
  fi
done

echo "=== ALL DONE ==="
find "$OUTPUT_DIR/$MODEL" -maxdepth 2 -type f 2>/dev/null | sort

if [ "$FAIL" -ne 0 ]; then
  echo "One or more evaluation jobs failed."
  exit 1
fi
