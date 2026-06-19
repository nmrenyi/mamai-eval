#!/bin/bash
set -euo pipefail
# G1/G2 A/B judging — boot the pinned gpt-oss-120b judge ONCE and score both
# tracks for the given arm run-dir(s):
#   - SAQ (open_ended): rescore_open_v2  -> 4-value behavior + key-fact recall + safety
#   - rubric:           rescore_rubric   -> weighted_met / completeness / penalty
# Rescorers write verdicts IN-PLACE into the PVC result JSONs (RUN_DIRS point at
# the PVC, mounted at /lightscratch), so scored files persist after the job ends.
#
# Submit (1x 80GB+ GPU; gpt-oss-120b fits on one H200/H100/A100-80GB):
#   NODE_POOL=h100 GPU_REQUEST=1 MEMORY_REQUEST=96G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-g1-judge-arm2 run_cluster_g1_judge.sh \
#     RUN_DIRS=/lightscratch/users/yiren/eval_output/g1_ab_3n_20260619/arm2/run

MODEL="${MODEL:-openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-1}"
CONFIG="${CONFIG:-config-v0.2.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
EXTRA_VLLM_FLAGS="${EXTRA_VLLM_FLAGS:-}"
# Space-separated PVC run-dirs to score (each holds kenya/afrimedqa_saq/whb/
# healthbench_oss_eval.json). Default: all three G1 arms.
RUN_DIRS="${RUN_DIRS:-/lightscratch/users/yiren/eval_output/g1_ab_3n_20260619/arm1/run /lightscratch/users/yiren/eval_output/g1_ab_3n_20260619/arm2/run /lightscratch/users/yiren/eval_output/g1_ab_3n_20260619/arm3/run}"

REPO_REF="${REPO_REF:-feat/g1-prompt-fix-20260611}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai huggingface_hub > /dev/null
python3 - <<'PY'
import sys
try:
    import vllm, torch, transformers, openai
    print(f"  torch={torch.__version__} vllm={vllm.__version__} openai={openai.__version__}")
except Exception as e:
    print(f"DEP-CHECK FAILED: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
PY
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

VLLM_LOG=/tmp/vllm.log
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
echo "=== STARTING vLLM ($MODEL, TP=$TP_SIZE, port=$PORT) ==="
VLLM_ARGS=( --model "$MODEL" --tensor-parallel-size "$TP_SIZE" --host 0.0.0.0 --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" --trust-remote-code --max-model-len "$MAX_MODEL_LEN" )
# shellcheck disable=SC2206
VLLM_ARGS+=($EXTRA_VLLM_FLAGS)
python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup(){ echo "=== STOPPING vLLM ($VLLM_PID) ==="; kill -TERM "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

HEALTH_WAIT_S="${HEALTH_WAIT_S:-3600}"; HEALTH_POLL_S=10
echo "=== WAITING FOR vLLM /health (up to $((HEALTH_WAIT_S/60)) min) ==="
HEALTHY=0
for i in $(seq 1 $((HEALTH_WAIT_S/HEALTH_POLL_S))); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; echo "vLLM healthy after ~$((i*HEALTH_POLL_S))s"; break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "ERROR: vllm exited before healthy:"; tail -80 "$VLLM_LOG"; exit 1; fi
  sleep "$HEALTH_POLL_S"
done
[ "$HEALTHY" -ne 1 ] && { echo "ERROR: vllm not healthy in time"; tail -80 "$VLLM_LOG"; exit 1; }

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export MAMAI_EVAL_CONFIG="$CONFIG"
curl -s "http://localhost:$PORT/v1/models" | head -c 400; echo

# shellcheck disable=SC2206
DIRS=($RUN_DIRS)
echo "=== JUDGING ${#DIRS[@]} run-dir(s) ==="; printf '  %s\n' "${DIRS[@]}"
for d in "${DIRS[@]}"; do
  echo "##### SAQ behavior-judge (rescore_open_v2) over $d #####"
  # Directory arg: rescore_open_v2 auto-selects the open_ended files
  # (kenya/afrimedqa_saq/whb) and skips the rubric file.
  python3 -m end_to_end_eval.rescore_open_v2 --config "$CONFIG" "$d"
  HB="$d/healthbench_oss_eval.json"
  if [ -f "$HB" ]; then
    echo "##### rubric-judge (rescore_rubric) over $HB #####"
    python3 -m end_to_end_eval.rescore_rubric --config "$CONFIG" "$HB"
  else
    echo "WARN: $HB not found; skipping rubric for $d"
  fi
done

echo "=== DONE — scored verdicts written in-place ==="
for d in "${DIRS[@]}"; do echo "--- $d ---"; ls -la "$d"; done
