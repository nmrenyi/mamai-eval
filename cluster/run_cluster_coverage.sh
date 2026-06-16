#!/bin/bash
set -euo pipefail
# R2c Phase 1 (refinement) — top-20 UNION coverage across retrievers. Judges the
# gecko ∪ EmbeddingGemma top-20 union per kenya query with Qwen3-32B + V2 rubric,
# reports union coverage + per-arm + the corpus-absent vs ranking-fixable(buried) split.
#
# Submit (2x H200):
#   ... ./submit_job.sh mamai-cov20 run_cluster_coverage.sh

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
TOPK="${TOPK:-20}"
SCR="${SCR:-/lightscratch/users/yiren/eval_output/screen_embedder}"
RETRIEVALS="${RETRIEVALS:-$SCR/retrievals_gecko_d768.json,$SCR/retrievals_embeddinggemma_300m_d768.json}"
OUT="${OUT:-$SCR/kenya_coverage_top20.json}"
REPO_REF="${REPO_REF:-feat/r2c-embedder-bakeoff-20260615}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WT=/tmp/eval_code
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai numpy > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WT"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WT"; cd "$WT"

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

echo "=== COVERAGE (top-$TOPK union) ==="
python3 -m retrieval_eval.screen_embedder coverage \
  --retrievals "$RETRIEVALS" --base-url "http://localhost:$PORT/v1" \
  --model "$JUDGE_MODEL" --top-k "$TOPK" --out "$OUT"
echo "=== DONE ==="; cat "$OUT"