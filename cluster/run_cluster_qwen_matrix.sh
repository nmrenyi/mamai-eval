#!/bin/bash
set -euo pipefail
# Generator×prompt matrix WAVE 2 — Qwen-397B ceiling, both tracks, in ONE 8-GPU job.
# Serves Qwen/Qwen3.5-397B-A17B-FP8 via vLLM (TP=8) ONCE, then generates each prompt
# arm for Track A (+RAG EmbeddingGemma) and Track B (oracle faithfulness) via
# run_eval / eval_faithfulness with --model qwen-397b --workers N (concurrent against
# the local endpoint — sequential would take days on a 397B).
#
# Submit AFTER wave 1 frees its GPUs (project quota ≤15; this needs 8):
#   NODE_POOL=h200 GPU_REQUEST=8 MEMORY_REQUEST=256G CPU_REQUEST=16 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-mx-qwen run_cluster_qwen_matrix.sh PROMPTS="baseline"
#   (PROMPTS="baseline g1 g1g2" for the full 3-prompt sweep — ~3× longer.)

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-397B-A17B-FP8}"
TP_SIZE="${TP_SIZE:-8}"
WORKERS="${WORKERS:-32}"
PROMPTS="${PROMPTS:-baseline}"        # space-separated subset of: baseline g1 g1g2
CONFIG="${CONFIG:-config-v0.2.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OUT_BASE="${OUT_BASE:-/lightscratch/users/yiren/eval_output/gen_prompt_matrix_20260622}"
EG_KENYA="${EG_KENYA:-/lightscratch/users/yiren/eval_output/rag_arms_eg/embeddinggemma}"
EG_HB="${EG_HB:-/lightscratch/users/yiren/eval_output/rag_arms_eg_hb/embeddinggemma}"
REPO_REF="${REPO_REF:-feat/g1-prompt-fix-20260611}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai datasets huggingface_hub tqdm numpy > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"
ORACLE="$WORKTREE/configs/$CONFIG/oracle/mamaretrieval-v0.2.0-score5.jsonl"
PFX="$WORKTREE/configs/$CONFIG/results/end_to_end_eval/g1-ab-3n-20260619/prompts"

VLLM_LOG=/tmp/vllm.log
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
echo "=== SERVE Qwen ($MODEL_ID, TP=$TP_SIZE) ==="
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
  echo "########## QWEN prompt=$p ##########"
  OA="$OUT_BASE/A/qwen_$p"
  python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model qwen-397b --datasets kenya \
    --rag "$EG_KENYA" --workers "$WORKERS" ${SP[@]+"${SP[@]}"} --output-dir "$OA" --run-dir "$OA/run"
  python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model qwen-397b --datasets healthbench_oss_eval \
    --rag "$EG_HB" --workers "$WORKERS" ${SP[@]+"${SP[@]}"} --output-dir "$OA" --run-dir "$OA/run"
  OB="$OUT_BASE/B/qwen_$p"
  python3 generator_eval/eval_faithfulness.py --config "$CONFIG" --model qwen-397b \
    --oracle "$ORACLE" --top-k 3 --workers "$WORKERS" ${SP[@]+"${SP[@]}"} --run-dir "$OB"
done

echo "=== DONE — Qwen matrix cells generated ==="
for p in $PROMPTS; do echo "--- qwen_$p ---"; ls -la "$OUT_BASE/A/qwen_$p/run" "$OUT_BASE/B/qwen_$p" 2>/dev/null | grep -iE 'json|oracle'; done
