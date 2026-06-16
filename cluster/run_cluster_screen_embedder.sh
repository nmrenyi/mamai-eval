#!/bin/bash
set -euo pipefail
# R2c Phase 2 — offline retrieval screen for a candidate embedder.
# Embed corpus+queries (sentence-transformers, GPU) -> retrieve top-k -> serve Qwen3-32B
# (vLLM) -> judge with V2 rubric -> score P@3 / HR@k vs the Gecko baseline.
#
# Submit (2x H200; needs HF_TOKEN with EmbeddingGemma access exported locally):
#   HF_TOKEN=hf_xxx NODE_POOL=h200 GPU_REQUEST=2 MEMORY_REQUEST=128G CPU_REQUEST=12 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-screen-eg run_cluster_screen_embedder.sh CANDIDATE=google/embeddinggemma-300m DIM=768

CANDIDATE="${CANDIDATE:-google/embeddinggemma-300m}"
DIM="${DIM:-768}"
DATASETS="${DATASETS:-kenya}"
TOP_K="${TOP_K:-20}"
TAG="${TAG:-$(echo "$CANDIDATE" | sed 's#.*/##; s/[^a-zA-Z0-9]/_/g')_d${DIM}}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
REPO_REF="${REPO_REF:-feat/r2c-embedder-bakeoff-20260615}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
ASSETS="${ASSETS:-/lightscratch/users/yiren/rag_assets}"
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/screen_embedder}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS (embed env) ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch 'sentence-transformers>=5.0' 'transformers>=4.56' datasets numpy huggingface_hub > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$OUT_SCRATCH"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

RETR="$OUT_SCRATCH/retrievals_${TAG}.json"
OUT="$OUT_SCRATCH/screen_${TAG}.json"

echo "=== PHASE A: embed + retrieve ($CANDIDATE, dim=$DIM, top_k=$TOP_K) ==="
python3 -m retrieval_eval.screen_embedder embed_retrieve \
  --candidate "$CANDIDATE" --db-path "$ASSETS/embeddings.sqlite" \
  --datasets "$DATASETS" --top-k "$TOP_K" --dim "$DIM" --out "$RETR"

echo "=== DEPS (vllm) ==="
pip3 install --no-cache-dir -q --retries 10 vllm openai > /dev/null
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

echo "=== PHASE B: judge + score ==="
python3 -m retrieval_eval.screen_embedder judge_score \
  --retrievals "$RETR" --base-url "http://localhost:$PORT/v1" \
  --model "$JUDGE_MODEL" --out "$OUT"

echo "=== DONE ==="; cat "$OUT"
