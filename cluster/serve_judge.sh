#!/bin/bash
# Cluster recipe: serve a candidate judge with vLLM, then run the
# judge-validation bake-off against it, then compute the metric report.
#
# One job per candidate. Outputs land on the PVC under
# /lightscratch/users/yiren/judge_validation/<JUDGE_NAME>/.
#
# Required env vars:
#   JUDGE_NAME     short label, used as output subdir name
#   MODEL          HuggingFace model id (passed to `vllm serve`)
#   TP_SIZE        tensor parallel size (matches GPU_REQUEST in submit_job.sh)
#
# Optional env vars (sensible defaults):
#   EXTRA_BODY            JSON forwarded to OpenAI extra_body (per-candidate reasoning kwargs)
#   EXTRA_VLLM_FLAGS      extra flags appended to `vllm serve` (e.g. --enable-reasoning ...)
#   JUDGE_TEMP            judge sampling temperature (default 0.0)
#   JUDGE_MAX_TOKENS      max output tokens per judge call (default 1024)
#   JUDGE_WORKERS         ThreadPoolExecutor width (default 20)
#   MAX_MODEL_LEN         vLLM --max-model-len (default unset; uses model config)
#   GPU_MEMORY_UTIL       vLLM --gpu-memory-utilization (default 0.92)
#   PORT                  vLLM listen port (default 8000)
#   REPO_REF              git ref to clone (default: feat/judge-validation-20260522)
#
# Run mode: this script is invoked inside a runai job (see submit_job.sh).

set -euo pipefail

JUDGE_NAME="${JUDGE_NAME:?required: short label for this candidate}"
MODEL="${MODEL:?required: HF model id, e.g. openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:?required: tensor parallel size}"

EXTRA_BODY="${EXTRA_BODY:-}"
EXTRA_VLLM_FLAGS="${EXTRA_VLLM_FLAGS:-}"
JUDGE_TEMP="${JUDGE_TEMP:-0.0}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
JUDGE_WORKERS="${JUDGE_WORKERS:-20}"
# vLLM serves the model's full context by default, which on Nemotron-Ultra
# (128K) and Maverick (1M) eats so much GPU memory for KV cache that the
# engine refuses to start (saw: KV cache needs 8 GiB but only 0.95 GiB free).
# The judge calls are ~1-2K input + ~150 output, so capping at 16K is safe
# and leaves plenty of KV headroom for batching. Override via env if needed.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
PORT="${PORT:-8000}"

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/judge-validation-20260522}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/lightscratch/users/yiren/judge_validation}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$JUDGE_NAME}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES (vllm + openai) ==="
# vLLM ships prebuilt CUDA wheels. Use --extra-index-url so torch resolves
# to a CUDA-12.4 build consistent with the transformers vllm pulls in
# (we've seen transformers latest need torch>=2.6's torch.distributed.tensor
# .device_mesh — fails Llama-4 Maverick model load if torch is older).
# --retries 10 handles the slow EPFL <-> PyPI link (seen mid-wheel timeouts).
pip3 install --no-cache-dir -q --retries 10 --upgrade pip > /dev/null
pip3 install --no-cache-dir -q --retries 10 \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  vllm openai huggingface_hub > /dev/null

# Sanity-check the install BEFORE we try to serve — fail fast with a clean
# error if a partial / net-interrupted install left things broken.
python3 - <<'PY'
import sys
try:
    import vllm, torch, transformers
    print(f"  torch={torch.__version__} transformers={transformers.__version__} vllm={vllm.__version__}")
    # The exact import chain that broke Maverick's job:
    from transformers.models.auto.image_processing_auto import ImageProcessingMixin  # noqa: F401
except Exception as e:
    print(f"DEP-CHECK FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
echo "=== DEPS DONE ==="

# Persist HF model cache across job runs via PVC.
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
VERDICTS="$OUTPUT_DIR/verdicts.jsonl"

# vLLM's engine-startup timeout defaults to 600s; the 253B Nemotron and 400G
# Maverick FP8 checkpoints both take well over that to load from PVC HF
# cache. Raise to 30 min so big models can finish weight loading.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

echo "=== STARTING vLLM SERVER ==="
echo "  MODEL=$MODEL  TP=$TP_SIZE  PORT=$PORT"
echo "  EXTRA_VLLM_FLAGS=$EXTRA_VLLM_FLAGS"
echo "  VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S"
echo "  log -> $VLLM_LOG"

VLLM_ARGS=(
  --model "$MODEL"
  --tensor-parallel-size "$TP_SIZE"
  --host 0.0.0.0
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  # Required for Nemotron-Ultra (custom modeling code on the HF repo);
  # safe / no-op for the other candidates from official orgs.
  --trust-remote-code
)
if [ -n "$MAX_MODEL_LEN" ]; then
  VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
# shellcheck disable=SC2206
VLLM_ARGS+=($EXTRA_VLLM_FLAGS)

# Use module form rather than `vllm serve`: partial pip installs occasionally
# don't place the `vllm` console-scripts entry in PATH (e.g. after a mid-wheel
# network timeout, which we hit on the first attempt for this bake-off).
python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "=== SHUTTING DOWN vLLM (pid $VLLM_PID) ==="
  kill -TERM "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== WAITING FOR vLLM /health (up to 20 min) ==="
HEALTHY=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    HEALTHY=1
    echo "vLLM healthy after ~$((i*10))s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vllm exited before becoming healthy. Tail of log:"
    tail -80 "$VLLM_LOG" || true
    exit 1
  fi
  sleep 10
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "ERROR: vllm did not become healthy within 20 min"
  tail -80 "$VLLM_LOG" || true
  exit 1
fi

# Sanity ping: list models the server is offering.
echo "=== MODELS ENDPOINT ==="
curl -s "http://localhost:$PORT/v1/models" | head -200 || true
echo

echo "=== RUNNING BAKE-OFF JUDGE ==="
EXTRA_BODY_ARGS=()
if [ -n "$EXTRA_BODY" ]; then
  EXTRA_BODY_ARGS=(--extra-body "$EXTRA_BODY")
fi

python3 -m calibration.judge_validation judge \
  --base-url "http://localhost:$PORT/v1" \
  --model "$MODEL" \
  --output "$VERDICTS" \
  --temperature "$JUDGE_TEMP" \
  --max-tokens "$JUDGE_MAX_TOKENS" \
  --max-workers "$JUDGE_WORKERS" \
  "${EXTRA_BODY_ARGS[@]}"

echo "=== COMPUTING METRICS ==="
python3 -m calibration.judge_validation metrics \
  --verdicts "$VERDICTS" \
  --report-md "$OUTPUT_DIR/report.md" \
  --report-json "$OUTPUT_DIR/report.json" \
  --bootstrap

echo "=== DONE ==="
ls -la "$OUTPUT_DIR"
echo
echo "Report preview:"
sed -n '1,40p' "$OUTPUT_DIR/report.md" || true
