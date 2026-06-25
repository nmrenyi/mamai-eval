#!/bin/bash
set -euo pipefail
# Judge the model-size/precision ladder with the pinned gpt-oss-120b. Serves the judge
# ONCE, then scores every ladder run-dir twice: rescore_open_v2 (kenya SAQ key-fact recall
# + safety) and rescore_rubric (healthbench weighted_met +/-). Writes verdicts in-place,
# then copies scored JSONs to $OUTPUT_DIR for rsync. Methodology matches the matrix.
#
# Submit AFTER all ladder generation jobs finish (A100 default avoids H200 preemption):
#   NODE_POOL=default GPU_REQUEST=2 MEMORY_REQUEST=160G CPU_REQUEST=12 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-ladder-judge run_cluster_ladder_judge.sh REPO_REF=feat/qwen-norag-tracka-20260624

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
CONFIG="${CONFIG:-config-v0.2.0}"
REPO_REF="${REPO_REF:-feat/qwen-norag-tracka-20260624}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
EVAL_OUT="${EVAL_OUT:-/lightscratch/users/yiren/eval_output}"
LADDER="${LADDER:-$EVAL_OUT/size_ladder_20260624}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_OUT/ladder_judge_20260624}"

# All ladder run-dirs (each holds kenya.json and/or healthbench_oss_eval.json).
# Override RUN_DIRS to judge a subset.
RUN_DIRS="${RUN_DIRS:-
  $EVAL_OUT/norag_g1_gemma4-e4b_20260624/run
  $EVAL_OUT/qwen_norag_tracka_20260624/qwen_g1_norag/run
  $LADDER/gemma4-e2b-q4/g1_norag/run   $LADDER/gemma4-e2b-q4/g1_rag/run
  $LADDER/gemma4-e4b-q8/g1_norag/run   $LADDER/gemma4-e4b-q8/g1_rag/run
  $LADDER/gemma4-e4b-bf16/g1_norag/run $LADDER/gemma4-e4b-bf16/g1_rag/run
  $LADDER/gemma4-26b-a4b/g1_norag/run  $LADDER/gemma4-26b-a4b/g1_rag/run
  $LADDER/gemma4-31b/g1_norag/run      $LADDER/gemma4-31b/g1_rag/run
  $LADDER/qwen36-27b/g1_norag/run
  $LADDER/qwen36-35b-a3b/g1_norag/run
  $LADDER/qwen35-122b/g1_norag/run
}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl rsync > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
export PIP_ONLY_BINARY="numpy,pyarrow"
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai huggingface_hub pandas tqdm numpy > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$OUTPUT_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

VLLM_LOG="$OUTPUT_DIR/vllm.log"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
echo "=== SERVE judge $JUDGE_MODEL (TP=$TP_SIZE) ==="
python3 -m vllm.entrypoints.openai.api_server --model "$JUDGE_MODEL" --tensor-parallel-size "$TP_SIZE" \
  --host 0.0.0.0 --port "$PORT" --gpu-memory-utilization "$GPU_MEMORY_UTIL" --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" > "$VLLM_LOG" 2>&1 &
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
[ "$HEALTHY" = 1 ] || { echo "ERROR: judge not healthy"; tail -100 "$VLLM_LOG"; exit 1; }
curl -s "http://localhost:$PORT/v1/models" | head -c 300; echo

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export MAMAI_EVAL_CONFIG="$CONFIG"

# shellcheck disable=SC2206
DIRS=($RUN_DIRS)
echo "=== SCORE — kenya SAQ (rescore_open_v2) over ${#DIRS[@]} run-dirs ==="
python3 -m end_to_end_eval.rescore_open_v2 --config "$CONFIG" "${DIRS[@]}" 2>&1 | tee "$OUTPUT_DIR/rescore_open.log"
echo "=== SCORE — healthbench rubric (rescore_rubric) over ${#DIRS[@]} run-dirs ==="
python3 -m end_to_end_eval.rescore_rubric --config "$CONFIG" "${DIRS[@]}" 2>&1 | tee "$OUTPUT_DIR/rescore_rubric.log"

echo "=== COPY scored JSONs to $OUTPUT_DIR ==="
for d in "${DIRS[@]}"; do
  [ -d "$d" ] || continue
  rel=$(echo "$d" | sed "s#$EVAL_OUT/##; s#/#__#g")
  mkdir -p "$OUTPUT_DIR/$rel"
  cp "$d"/kenya.json "$d"/healthbench_oss_eval.json "$OUTPUT_DIR/$rel/" 2>/dev/null || true
done
echo "=== DONE ==="
ls -R "$OUTPUT_DIR" | head -60
