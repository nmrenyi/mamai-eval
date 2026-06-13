#!/bin/bash
set -euo pipefail

# R2c P1 (step 3/3) — judge the 3-arm SAQ answers with the pinned gpt-oss-120b.
# Serves gpt-oss-120b via vLLM, then runs rescore_open_v2 (key-fact recall +
# safety + axis scores) on each arm's SAQ result files in place.
#
# Submit (2x H200, gpt-oss-120b TP=2):
#   NODE_POOL=h200 GPU_REQUEST=2 MEMORY_REQUEST=128G CPU_REQUEST=12 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-vg-judge run_cluster_value_gate_judge.sh

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
CONFIG="${CONFIG:-config-v0.2.0}"
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/value_gate}"
SAQ_DS="${SAQ_DS:-kenya afrimedqa_saq}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai huggingface_hub pandas tqdm > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
VLLM_LOG=/tmp/vllm.log
echo "=== STARTING vLLM ($JUDGE_MODEL TP=$TP_SIZE) ==="
python3 -m vllm.entrypoints.openai.api_server --model "$JUDGE_MODEL" \
  --tensor-parallel-size "$TP_SIZE" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() { kill -TERM "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "=== WAITING FOR /health (up to 30 min) ==="
for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "healthy after ~$((i*10))s"; break; }
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm died:"; tail -60 "$VLLM_LOG"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "not healthy"; tail -60 "$VLLM_LOG"; exit 1; }

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"

for ARM in gecko hybrid hybrid_rerank; do
  RUN="$OUT_DIR/$ARM/run"
  for DS in $SAQ_DS; do
    f="$RUN/$DS.json"
    [ -f "$f" ] || { echo "SKIP missing $f"; continue; }
    echo "=== JUDGE arm=$ARM ds=$DS ==="
    python3 -m end_to_end_eval.rescore_open_v2 --config "$CONFIG" "$f"
  done
done

echo "=== JUDGE COMPLETE ==="
find "$OUT_DIR" -name "*.json" | sort
