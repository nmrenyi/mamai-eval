#!/bin/bash
# Cluster recipe: serve the pinned judge with vLLM and run the v0.2 SAQ
# (open_ended) production rescore over the existing Gemma 4 E4B response
# files (both ±RAG arms × 3 datasets — kenya, whb, afrimedqa_saq).
#
# Counterpart to cluster/rescore_rubric.sh — same vLLM-boot discipline,
# different rescore script (`end_to_end_eval.rescore_open_v2`) and
# different target dirs (the `*-openended/` dirs, not the `*-rubric/`).
#
# Required env vars: none — sensible defaults for the pinned setup below.
#
# Optional env vars (most you'll never touch):
#   MODEL              HF model id (default: openai/gpt-oss-120b, the pinned judge)
#   TP_SIZE            tensor parallel size (default: 1; gpt-oss-120b fits on 1× H200)
#   REPO_REF           git ref to clone (default: feat/phase-b-rubric-20260608)
#   CONFIG             config dir name (default: config-v0.2.0)
#   PORT               vLLM listen port (default: 8000)
#   GPU_MEMORY_UTIL    vLLM --gpu-memory-utilization (default: 0.92)
#   MAX_MODEL_LEN      vLLM --max-model-len (default: 16384 — judge calls
#                      are ~1-2K input + ~500 output; ample KV headroom)
#   EXTRA_VLLM_FLAGS   extra flags appended to `vllm serve`
#   HEALTH_WAIT_S      max seconds to wait for /health (default: 3600)
#   OPENENDED_DIRS_OVERRIDE  space-separated list of files/dirs to score,
#                            replaces the default both-arms sweep. Use for
#                            probe / partial / smoke runs.
#   SAMPLE_PER_FILE    integer; if set, passes `--sample-per-file N` to the
#                      rescorer (stratified smoke runs).
#   SAMPLE_SEED        integer (default 42, deterministic) — only applied
#                      when SAMPLE_PER_FILE is also set.
#
# Output layout on PVC: $OUTPUT_DIR/{vllm.log, rescore.dryrun.log,
# rescore.log, <arm-dir>/*.json}. Same shape as rescore_rubric.sh's output.

set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-1}"
REPO_REF="${REPO_REF:-feat/phase-b-rubric-20260608}"
CONFIG="${CONFIG:-config-v0.2.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
EXTRA_VLLM_FLAGS="${EXTRA_VLLM_FLAGS:-}"

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lightscratch/users/yiren/phase_b_saq}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl rsync > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES (vllm + openai) ==="
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai huggingface_hub > /dev/null

python3 - <<'PY'
import sys
try:
    import vllm, torch, transformers, openai
    print(f"  torch={torch.__version__} transformers={transformers.__version__} "
          f"vllm={vllm.__version__} openai={openai.__version__}")
except Exception as e:
    print(f"DEP-CHECK FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== CHECKOUT $REPO_REF ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

mkdir -p "$OUTPUT_DIR"
VLLM_LOG="$OUTPUT_DIR/vllm.log"
RESCORE_LOG="$OUTPUT_DIR/rescore.log"
DRYRUN_LOG="$OUTPUT_DIR/rescore.dryrun.log"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

echo "=== STARTING vLLM SERVER ==="
echo "  MODEL=$MODEL  TP=$TP_SIZE  PORT=$PORT"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN  GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
echo "  EXTRA_VLLM_FLAGS=$EXTRA_VLLM_FLAGS"
echo "  VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S"
echo "  log -> $VLLM_LOG"

VLLM_ARGS=(
  --model "$MODEL"
  --tensor-parallel-size "$TP_SIZE"
  --host 0.0.0.0
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
)
# shellcheck disable=SC2206
VLLM_ARGS+=($EXTRA_VLLM_FLAGS)

python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "=== SHUTTING DOWN vLLM (pid $VLLM_PID) ==="
  kill -TERM "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

HEALTH_WAIT_S="${HEALTH_WAIT_S:-3600}"
HEALTH_POLL_S=10
HEALTH_MAX_ITERS=$(( HEALTH_WAIT_S / HEALTH_POLL_S ))
echo "=== WAITING FOR vLLM /health (up to $(( HEALTH_WAIT_S / 60 )) min) ==="
HEALTHY=0
for i in $(seq 1 "$HEALTH_MAX_ITERS"); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    HEALTHY=1
    echo "vLLM healthy after ~$((i*HEALTH_POLL_S))s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vllm exited before becoming healthy. Tail of log:"
    tail -80 "$VLLM_LOG" || true
    exit 1
  fi
  sleep "$HEALTH_POLL_S"
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "ERROR: vllm did not become healthy within $(( HEALTH_WAIT_S / 60 )) min"
  tail -80 "$VLLM_LOG" || true
  exit 1
fi

echo "=== MODELS ENDPOINT ==="
curl -s "http://localhost:$PORT/v1/models" | head -200 || true
echo

# Point the OpenAI client (used by rescore_open_v2._call_openai) at vLLM.
# OPENAI_API_KEY must be set (SDK enforces it) but vLLM ignores the value.
# MAMAI_EVAL_CONFIG drives which params.json provides the judge ensemble.
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export MAMAI_EVAL_CONFIG="$CONFIG"

# Default sweep: both ±RAG arms × 3 SAQ datasets = 6 files.
# Override via OPENENDED_DIRS_OVERRIDE for probe / smoke runs.
if [ -n "${OPENENDED_DIRS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  OPENENDED_DIRS=($OPENENDED_DIRS_OVERRIDE)
  echo "=== OPENENDED_DIRS overridden — probe / partial run ==="
  printf '  %s\n' "${OPENENDED_DIRS[@]}"
else
  OPENENDED_DIRS=(
    configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/20260520T104610-cluster-rag-openended
    configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/20260520T104611-cluster-norag-openended
  )
fi

# Build the rescorer arg list. SAMPLE_PER_FILE turns this into a stratified
# smoke (random N rows from each file, seed-deterministic).
RESCORER_ARGS=(-m end_to_end_eval.rescore_open_v2 --config "$CONFIG")
if [ -n "${SAMPLE_PER_FILE:-}" ]; then
  RESCORER_ARGS+=(--sample-per-file "$SAMPLE_PER_FILE")
  if [ -n "${SAMPLE_SEED:-}" ]; then
    RESCORER_ARGS+=(--sample-seed "$SAMPLE_SEED")
  fi
  echo "=== STRATIFIED SMOKE: $SAMPLE_PER_FILE row(s) per file (seed=${SAMPLE_SEED:-42}) ==="
fi

echo "=== DRY RUN — scope + config check ==="
python3 "${RESCORER_ARGS[@]}" --dry-run "${OPENENDED_DIRS[@]}" 2>&1 | tee "$DRYRUN_LOG"

echo
echo "=== PRODUCTION RESCORE ==="
echo "(SAQ track: ±RAG × {kenya, whb, afrimedqa_saq}."
echo " gpt-oss-120b judge writes judge_ensemble.{judges,aggregate} into each JSON."
echo " Internal CoT is captured as reasoning_content in each judge's output entry.)"
python3 "${RESCORER_ARGS[@]}" "${OPENENDED_DIRS[@]}" 2>&1 | tee "$RESCORE_LOG"

# Copy scored result JSONs OUT of the /tmp worktree to PVC. Same shape as
# rescore_rubric.sh's copy-out so the rsync-back convention is identical.
echo
echo "=== COPYING SCORED FILES TO $OUTPUT_DIR ==="
for d in "${OPENENDED_DIRS[@]}"; do
  if [ -f "$d" ]; then
    arm=$(basename "$(dirname "$d")")
    mkdir -p "$OUTPUT_DIR/$arm"
    cp "$d" "$OUTPUT_DIR/$arm/"
  elif [ -d "$d" ]; then
    arm=$(basename "$d")
    mkdir -p "$OUTPUT_DIR/$arm"
    # Skip eval_input.json / eval_output.json side-files; only the result JSONs.
    cp "$d"/kenya.json "$d"/whb.json "$d"/afrimedqa_saq.json \
      "$OUTPUT_DIR/$arm/" 2>/dev/null || true
  else
    echo "WARN: $d is neither file nor dir; skipping copy-out"
  fi
done

echo "=== DONE ==="
ls -la "$OUTPUT_DIR"
echo
echo "Tail of rescore log:"
tail -25 "$RESCORE_LOG" || true
