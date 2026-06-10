#!/bin/bash
# Cluster recipe: serve the pinned judge with vLLM and run the v0.2
# HealthBench rubric-track production rescore over the existing Gemma 4 E4B
# response files (both ±RAG arms).
#
# Counterpart to cluster/serve_judge.sh: serve_judge.sh was for the judge
# bake-off (one candidate model per job, write verdicts + metric report to
# PVC). This script runs the pinned judge over the production result JSONs
# from feat/phase-b-rubric-20260608 — Phase B of the v0.2 evaluation plan.
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
#   MAX_MODEL_LEN      vLLM --max-model-len (default: 16384 — judge calls are
#                      ~1-2K input + ~500 output, 16K leaves ample KV headroom)
#   EXTRA_VLLM_FLAGS   extra flags appended to `vllm serve` (e.g. reasoning
#                      parser, if/when needed for a new model)
#   HEALTH_WAIT_S      max seconds to wait for /health (default: 3600)
#   RUBRIC_DIRS_OVERRIDE  space-separated list of files/dirs to score, replaces
#                      the default two-arm sweep. Use for probe runs:
#                      RUBRIC_DIRS_OVERRIDE=path/to/one/healthbench_hard.json
#                      Each entry may be a single .json file (smallest unit)
#                      or a directory (rescores every healthbench_*.json in it).
#
# Output layout on PVC:
#   $OUTPUT_DIR/vllm.log                — vLLM server logs
#   $OUTPUT_DIR/rescore.dryrun.log      — scope check before the real run
#   $OUTPUT_DIR/rescore.log             — the actual scoring run's stdout
#   $OUTPUT_DIR/<arm-dir>/*.json        — scored result files copied out for
#                                          download (rescore_rubric.py writes
#                                          in-place inside the worktree;
#                                          worktree is destroyed at job end)
#
# Retrieval: after the job, rsync from haas001:
#   $OUTPUT_DIR/20260521T122626-cluster-rag-rubric/
#   $OUTPUT_DIR/20260521T123051-cluster-norag-rubric/
# back into your local checkout's matching paths, then commit + PR.
#
# Run mode: this script is invoked inside a runai job (see submit_job.sh).

set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-1}"
REPO_REF="${REPO_REF:-feat/phase-b-rubric-20260608}"
CONFIG="${CONFIG:-config-v0.2.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
# Same rationale as serve_judge.sh: 16K is safe for judge calls (~1-2K input,
# ~500 output) and leaves plenty of KV cache headroom for batching. Increase
# only if a future judge model needs longer context.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
EXTRA_VLLM_FLAGS="${EXTRA_VLLM_FLAGS:-}"

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lightscratch/users/yiren/phase_b_rubric}"
# Per-run subdir; ISO-ish UTC timestamp so re-runs don't clobber each other.
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl rsync > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES (vllm + openai) ==="
# Same wheel-pinning rationale as serve_judge.sh: cu124 index lets torch
# resolve to a CUDA-12.4 build compatible with the transformers vllm pulls in.
# --retries 10 absorbs slow-link partial-wheel timeouts seen on EPFL <-> PyPI.
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai huggingface_hub > /dev/null

# Sanity-check the install BEFORE we try to serve — fail fast on broken installs.
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

# Persist HF model cache across job runs via PVC (gpt-oss-120b is ~120GB).
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

# vLLM engine-startup timeout. gpt-oss-120b loads from PVC cache; budget 30 min.
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
  # trust-remote-code is a no-op for first-party orgs (openai/, meta-llama/);
  # leaving it on so a future judge swap from a less-official org Just Works.
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
)
# shellcheck disable=SC2206
VLLM_ARGS+=($EXTRA_VLLM_FLAGS)

# Module form rather than `vllm serve` console-script: partial pip installs
# occasionally don't put `vllm` in PATH (mid-wheel network timeouts).
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

# Sanity ping: confirm vLLM exposes the judge model under the expected id.
echo "=== MODELS ENDPOINT ==="
curl -s "http://localhost:$PORT/v1/models" | head -200 || true
echo

# Point the OpenAI client (used internally by rescore_rubric._call_openai) at
# the local vLLM. OPENAI_API_KEY must be set (the SDK enforces it) but its
# value is ignored by vLLM. MAMAI_EVAL_CONFIG drives which params.json to
# load (judge.rubric block lives there).
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export MAMAI_EVAL_CONFIG="$CONFIG"

# The two ±RAG arms with Gemma 4 E4B's HealthBench responses.
# Each directory holds three result JSONs: healthbench_{oss_eval,consensus,hard}.json.
# rescore_rubric.py writes scored verdicts in-place into each JSON.
# Override at submission time via RUBRIC_DIRS_OVERRIDE for probe runs.
if [ -n "${RUBRIC_DIRS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  RUBRIC_DIRS=($RUBRIC_DIRS_OVERRIDE)
  echo "=== RUBRIC_DIRS overridden — probe / partial run ==="
  printf '  %s\n' "${RUBRIC_DIRS[@]}"
else
  RUBRIC_DIRS=(
    configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/20260521T122626-cluster-rag-rubric
    configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/20260521T123051-cluster-norag-rubric
  )
fi

# Dry-run first: confirms the judge config resolves to the pinned model +
# extra_body, and reports the unscored row count. Cheap, surfaces config
# wiring problems before any token spend.
echo "=== DRY RUN — scope + config check ==="
python3 -m end_to_end_eval.rescore_rubric --config "$CONFIG" --dry-run \
  "${RUBRIC_DIRS[@]}" 2>&1 | tee "$DRYRUN_LOG"

echo
echo "=== PRODUCTION RESCORE ==="
echo "(Two arms × 3 HealthBench datasets = 6 files, ~38,308 criterion calls."
echo " Scored verdicts written in-place to each JSON.)"
python3 -m end_to_end_eval.rescore_rubric --config "$CONFIG" \
  "${RUBRIC_DIRS[@]}" 2>&1 | tee "$RESCORE_LOG"

# Copy the scored result JSONs OUT of the worktree (which is /tmp/ and will be
# destroyed when the job ends) and into the PVC output dir so they can be
# rsync'd back to a workstation for commit.
echo
echo "=== COPYING SCORED FILES TO $OUTPUT_DIR ==="
for d in "${RUBRIC_DIRS[@]}"; do
  if [ -f "$d" ]; then
    # Single-file entry (probe mode): preserve its arm-dir as the bucket name.
    arm=$(basename "$(dirname "$d")")
    mkdir -p "$OUTPUT_DIR/$arm"
    cp "$d" "$OUTPUT_DIR/$arm/"
  elif [ -d "$d" ]; then
    # Directory entry (full sweep): copy out all healthbench_*.json,
    # skipping any eval_input.json / eval_output.json side-files.
    arm=$(basename "$d")
    mkdir -p "$OUTPUT_DIR/$arm"
    cp "$d"/healthbench_*.json "$OUTPUT_DIR/$arm/"
  else
    echo "WARN: $d is neither file nor dir; skipping copy-out"
  fi
done

echo "=== DONE ==="
ls -la "$OUTPUT_DIR"
echo
echo "Tail of rescore log:"
tail -25 "$RESCORE_LOG" || true
