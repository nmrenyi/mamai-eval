#!/bin/bash
set -euo pipefail
# Qwen-397B NO-RAG ceiling — Track A only (kenya + healthbench_oss_eval), in ONE 8-GPU job.
# Research question: how much of Qwen's matrix advantage is parametric medical knowledge
# vs. its use of retrieved context? This ablates RAG: serve Qwen/Qwen3.5-397B-A17B-FP8 via
# vLLM (TP=8) ONCE, then generate the G1 (deployed) prompt arm with retrieval DISABLED
# (no --rag). Directly comparable to the matrix's Qwen+G1 +RAG cell (only RAG differs),
# and to the existing Gemma 4 / 3n no-RAG arms (capability ladder, retrieval held out).
#
# Track B (faithfulness) is intentionally omitted — it requires oracle context, so a
# no-RAG faithfulness arm is ill-defined.
#
# Judge separately AFTER (gpt-oss-120b): healthbench via cluster/rescore_rubric.sh
# (RUBRIC_DIRS_OVERRIDE=<this run dir>), kenya via the Track-A value-gate judge.
#
# Submit (needs 8 GPUs free; project quota ≤15):
#   NODE_POOL=h200 GPU_REQUEST=8 MEMORY_REQUEST=256G CPU_REQUEST=16 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-qwen-norag run_cluster_qwen_norag.sh REPO_REF=main
#   (PROMPTS defaults to "g1"; pass PROMPTS="baseline g1" to add the no-prompt control.)

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-397B-A17B-FP8}"
TP_SIZE="${TP_SIZE:-8}"
WORKERS="${WORKERS:-32}"
PROMPTS="${PROMPTS:-g1}"               # space-separated subset of: baseline g1 g1g2
DATASETS="${DATASETS:-kenya healthbench_oss_eval}"
CONFIG="${CONFIG:-config-v0.2.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OUT_BASE="${OUT_BASE:-/lightscratch/users/yiren/eval_output/qwen_norag_tracka_20260624}"
REPO_REF="${REPO_REF:-main}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
export PIP_ONLY_BINARY="numpy,pyarrow"   # avoid from-source numpy/pyarrow builds (killed 397B deps)
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai datasets huggingface_hub tqdm numpy > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"
PFX="$WORKTREE/configs/$CONFIG/results/end_to_end_eval/g1-ab-3n-20260619/prompts"

VLLM_LOG=/tmp/vllm.log
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
# pip-only container has runtime CUDA but no nvcc → DeepGEMM JIT FP8 kernels fail to
# compile. Disable DeepGEMM; use vLLM's prebuilt FP8 kernels (CUTLASS/Triton). Hopper FP8 OK.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
echo "=== SERVE Qwen ($MODEL_ID, TP=$TP_SIZE, VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM) ==="
python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_ID" --tensor-parallel-size "$TP_SIZE" \
  --host 0.0.0.0 --port "$PORT" --gpu-memory-utilization "$GPU_MEMORY_UTIL" --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" --reasoning-parser qwen3 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup(){ echo "=== stopping vLLM ==="; kill -TERM "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "=== wait /health (up to 60 min) ==="
HEALTHY=0
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; echo "healthy ~$((i*10))s"; break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vllm exited:"; tail -80 "$VLLM_LOG"; exit 1; }
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "ERROR: not healthy"; tail -80 "$VLLM_LOG"; exit 1; }
curl -s "http://localhost:$PORT/v1/models" | head -c 300; echo

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export MAMAI_EVAL_CONFIG="$CONFIG"

sp_for(){ case "$1" in g1) echo "$PFX/arm2_system_en.txt";; g1g2) echo "$PFX/arm3_system_en.txt";; *) echo "";; esac; }

for p in $PROMPTS; do
  SPF=$(sp_for "$p"); SP=(); [ -n "$SPF" ] && SP=(--system-prompt "$SPF")
  OA="$OUT_BASE/qwen_${p}_norag"
  for ds in $DATASETS; do
    echo "########## QWEN no-RAG prompt=$p dataset=$ds ##########"
    # NO --rag → retrieval disabled (k=0); the model answers from parametric knowledge.
    python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model qwen-397b --datasets "$ds" \
      --workers "$WORKERS" ${SP[@]+"${SP[@]}"} --output-dir "$OA" --run-dir "$OA/run"
  done
done

echo "=== DONE — Qwen no-RAG Track-A cells generated ==="
for p in $PROMPTS; do echo "--- qwen_${p}_norag ---"; ls -la "$OUT_BASE/qwen_${p}_norag/run" 2>/dev/null | grep -iE 'json'; done
