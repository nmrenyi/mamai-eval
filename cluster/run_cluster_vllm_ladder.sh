#!/bin/bash
set -euo pipefail
# Model-size ladder — serve ONE vLLM model (TP configurable), generate G1 prompt arm on
# kenya + healthbench_oss_eval, for the requested retrieval modes. Track A only.
#   RAG_MODES="norag"        -> no-RAG only (Qwen capability-ceiling rungs)
#   RAG_MODES="norag rag"    -> both (Gemma 4 rungs: does RAG still help at this size?)
# +RAG reuses the deployed EmbeddingGemma retrieval arms already on PVC.
#
# Submit (Qwen 27/35B → H100, 122B → H200 TP=2; Gemma 26/31B → A100/H100):
#   NODE_POOL=h100 GPU_REQUEST=1 MEMORY_REQUEST=128G CPU_REQUEST=12 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-lad-qwen36-27b run_cluster_vllm_ladder.sh \
#     MODEL=qwen36-27b MODEL_ID=Qwen/Qwen3.6-27B REASONING_PARSER=qwen3 RAG_MODES=norag REPO_REF=feat/qwen-norag-tracka-20260624

MODEL="${MODEL:?registry name, e.g. qwen36-27b}"
MODEL_ID="${MODEL_ID:?served HF id, e.g. Qwen/Qwen3.6-27B}"
TP_SIZE="${TP_SIZE:-1}"
WORKERS="${WORKERS:-32}"
REASONING_PARSER="${REASONING_PARSER:-}"   # e.g. qwen3 for Qwen; empty for Gemma
RAG_MODES="${RAG_MODES:-norag}"            # "norag" or "norag rag"
PROMPT="${PROMPT:-g1}"
DATASETS="${DATASETS:-kenya healthbench_oss_eval}"
CONFIG="${CONFIG:-config-v0.2.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
OUT_BASE="${OUT_BASE:-/lightscratch/users/yiren/eval_output/size_ladder_20260624/$MODEL}"
EG_KENYA="${EG_KENYA:-/lightscratch/users/yiren/eval_output/rag_arms_eg/embeddinggemma}"
EG_HB="${EG_HB:-/lightscratch/users/yiren/eval_output/rag_arms_eg_hb/embeddinggemma}"
REPO_REF="${REPO_REF:-feat/qwen-norag-tracka-20260624}"
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
case "$PROMPT" in g1) SPF="$PFX/arm2_system_en.txt";; g1g2) SPF="$PFX/arm3_system_en.txt";; *) SPF="";; esac
SP=(); [ -n "$SPF" ] && SP=(--system-prompt "$SPF")

VLLM_LOG=/tmp/vllm.log
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"   # no nvcc → DeepGEMM JIT fails; use prebuilt FP8 kernels
RP=(); [ -n "$REASONING_PARSER" ] && RP=(--reasoning-parser "$REASONING_PARSER")
echo "=== SERVE $MODEL ($MODEL_ID, TP=$TP_SIZE, parser=${REASONING_PARSER:-none}) ==="
python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_ID" --tensor-parallel-size "$TP_SIZE" \
  --host 0.0.0.0 --port "$PORT" --gpu-memory-utilization "$GPU_MEMORY_UTIL" --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" ${RP[@]+"${RP[@]}"} > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup(){ echo "=== stopping vLLM ==="; kill -TERM "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "=== wait /health (up to 60 min) ==="
HEALTHY=0
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; echo "healthy ~$((i*10))s"; break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vllm exited:"; tail -100 "$VLLM_LOG"; exit 1; }
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "ERROR: not healthy"; tail -100 "$VLLM_LOG"; exit 1; }
curl -s "http://localhost:$PORT/v1/models" | head -c 300; echo

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export MAMAI_EVAL_CONFIG="$CONFIG"

rag_dir_for(){ case "$1" in kenya) echo "$EG_KENYA";; healthbench_oss_eval) echo "$EG_HB";; *) echo "";; esac; }

for mode in $RAG_MODES; do
  OUT="$OUT_BASE/${PROMPT}_${mode}"
  for ds in $DATASETS; do
    echo "########## $MODEL prompt=$PROMPT mode=$mode dataset=$ds ##########"
    RAG=()
    if [ "$mode" = "rag" ]; then RAG=(--rag "$(rag_dir_for "$ds")"); fi
    python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" --datasets "$ds" \
      --workers "$WORKERS" ${SP[@]+"${SP[@]}"} ${RAG[@]+"${RAG[@]}"} \
      --output-dir "$OUT" --run-dir "$OUT/run"
  done
done

echo "=== DONE — $MODEL ladder cells generated ==="
for mode in $RAG_MODES; do echo "--- ${PROMPT}_${mode} ---"; ls -la "$OUT_BASE/${PROMPT}_${mode}/run" 2>/dev/null | grep -iE 'json'; done
