#!/bin/bash
set -euo pipefail
# Thresholdability support: re-judge EmbeddingGemma kenya top-20 retrievals with Qwen3-32B
# (V2 rubric) and DUMP PER-PAIR grades [qid, idx, rank, sim, grade] so we can compute the
# chunk-level AUC of the EmbeddingGemma cosine score vs relevance (R1's thresholdability bar).
# No embedding needed — reuses the existing depth-20 retrievals file.
#
# Submit (2x H200):
#   NODE_POOL=h200 GPU_REQUEST=2 MEMORY_REQUEST=128G CPU_REQUEST=12 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-eg-grades run_cluster_eg_grades.sh

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
WORKERS="${WORKERS:-32}"
RETR="${RETR:-/lightscratch/users/yiren/eval_output/screen_embedder/retrievals_embeddinggemma_300m_d768.json}"
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/eg_thresh}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 vllm openai numpy datasets huggingface_hub > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$OUT_SCRATCH"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

echo "=== STARTING vLLM ($JUDGE_MODEL TP=$TP_SIZE) ==="
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
VLLM_LOG=/tmp/vllm.log
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

echo "=== JUDGE EmbeddingGemma kenya top-20 (dump per-pair grades) ==="
python3 -m retrieval_eval.screen_embedder judge_score \
  --retrievals "$RETR" --base-url "http://localhost:$PORT/v1" \
  --model "$JUDGE_MODEL" --workers "$WORKERS" \
  --out "$OUT_SCRATCH/screen_eg_kenya_rejudge.json" \
  --grades-out "$OUT_SCRATCH/eg_kenya_pairgrades.json"

echo "=== DONE ==="; ls -la "$OUT_SCRATCH"