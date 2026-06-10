#!/bin/bash
set -euo pipefail

# Submit eval job to EPFL RunAI cluster
# Usage: ./submit_job.sh [job-name] [script-name] [KEY=VALUE ...]
#
# Examples:
#   ./submit_job.sh mamai-eval-v5                         # runs run_cluster.sh (GGUF models)
#   ./submit_job.sh mamai-gemma4-smoke run_cluster.sh MODEL=gemma4-e4b DATASETS=afrimedqa_mcq MAX_QUESTIONS=5
#   ./submit_job.sh mamai-precompute run_cluster_precompute.sh CONFIG=config-v0.1.0
#
# Prerequisites:
#   - SSH alias "light" configured for haas001
#   - PVC permissions: chmod -R g+w /mnt/light/scratch/users/yiren/
#   - API keys on haas001 under /mnt/light/scratch/users/yiren/keys/
#       - openai_key.txt   (required: legacy --judge fallback + GPT-5 generation)
#       - hf_key.txt       (required for v0.2: gated dataset auth)
#       - gemini_key.txt   (optional: Gemini-as-generator runs)
#       - anthropic_key.txt (optional: Claude-as-generator runs)

JOB_NAME="${1:-mamai-eval-run}"
SCRIPT_NAME="${2:-run_cluster.sh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$SCRIPT_NAME" = /* ]]; then
  SCRIPT_PATH="$SCRIPT_NAME"
elif [[ "$SCRIPT_NAME" == */* ]]; then
  SCRIPT_PATH="$(cd "$(dirname "$SCRIPT_NAME")" && pwd)/$(basename "$SCRIPT_NAME")"
else
  SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"
fi
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
EXTRA_ENV=("$@")

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: $SCRIPT_NAME not found at $SCRIPT_PATH"
  exit 1
fi

# Base64 encode the script to avoid quoting issues
B64=$(base64 < "$SCRIPT_PATH" | tr -d '\n')

# Read API keys from the cluster (best-effort: only OpenAI + HF are required for
# v0.2 generation runs; the others are passed through if present).
_read_key() {
  # `|| true` so a missing key file (e.g. anthropic_key.txt) doesn't trip
  # `set -e` via command substitution under bash 3.2 (macOS default shell).
  ssh light "cat /mnt/light/scratch/users/yiren/keys/$1 2>/dev/null" 2>/dev/null || true
}

# Each key may come from (in order of preference):
#   (a) a matching env var already set in the local shell (lets you submit
#       without SSH-to-cluster — useful when off the EPFL network or for
#       vLLM-only jobs that don't actually need real provider keys; pass
#       OPENAI_API_KEY=EMPTY explicitly)
#   (b) the file under /mnt/light/scratch/users/yiren/keys/ on haas001
#       (the canonical source for cluster runs that talk to real providers)
OPENAI_KEY="${OPENAI_API_KEY:-$(_read_key openai_key.txt)}"
if [ -z "$OPENAI_KEY" ]; then
  echo "Error: Could not read OpenAI API key (set OPENAI_API_KEY locally, or"
  echo "       ensure ssh light can reach /mnt/light/scratch/users/yiren/keys/openai_key.txt)"
  exit 1
fi

HF_TOKEN="${HF_TOKEN:-$(_read_key hf_key.txt)}"
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN not set and hf_key.txt not on cluster — gated HF datasets / model downloads will fail"
fi

GEMINI_KEY="${GOOGLE_API_KEY:-$(_read_key gemini_key.txt)}"
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-$(_read_key anthropic_key.txt)}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HAS_REPO_REF=0
EFFECTIVE_REPO_REF="$CURRENT_BRANCH"
# bash 3.2 (macOS) + set -u errors on "${arr[@]}" when arr is empty; the
# `${arr[@]+"${arr[@]}"}` form expands to nothing when unset/empty instead of
# erroring. This block (and the matching one below) iterates extra env-pairs
# passed as positional args; empty is a valid case (no extras supplied).
for ENV_PAIR in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do
  if [[ "$ENV_PAIR" != *=* ]]; then
    echo "Error: extra arguments must be KEY=VALUE pairs, got: $ENV_PAIR"
    exit 1
  fi
  if [[ "$ENV_PAIR" == REPO_REF=* ]]; then
    HAS_REPO_REF=1
    EFFECTIVE_REPO_REF="${ENV_PAIR#REPO_REF=}"
  fi
done

# Node pool. `default` is the A100-80GB pool on light cluster; override via
# NODE_POOL env var (e.g. NODE_POOL=h100, NODE_POOL=h200) for larger GPUs.
NODE_POOL="${NODE_POOL:-default}"
# CPU + memory requests. RunAI's silent defaults are stingy (~4 CPU / 16G);
# CPU-heavy workloads (e.g. multiprocessed precompute) need explicit headroom.
CPU_REQUEST="${CPU_REQUEST:-4}"
MEMORY_REQUEST="${MEMORY_REQUEST:-16G}"
# Number of GPUs. Defaults to 1 (the eval/generation workload needs one A100).
# Set GPU_REQUEST=0 for CPU-only work (e.g. RAG context precompute) — frees the
# A100 for other users and lets us land on whatever node has free CPU.
GPU_REQUEST="${GPU_REQUEST:-1}"

RUNAI_ARGS=(
  runai submit
  --name "$JOB_NAME"
  --image nvidia/cuda:12.4.1-devel-ubuntu22.04
  --pvc light-scratch:/lightscratch
  --large-shm
  --node-pool "$NODE_POOL"
  -e "OPENAI_API_KEY=$OPENAI_KEY"
  --gpu "$GPU_REQUEST"
  --cpu "$CPU_REQUEST"
  --cpu-limit "$CPU_REQUEST"
  --memory "$MEMORY_REQUEST"
  --memory-limit "$MEMORY_REQUEST"
  --backoff-limit 0
  --run-as-gid 84257
)

if [ -n "$HF_TOKEN" ]; then
  RUNAI_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
fi
if [ -n "$GEMINI_KEY" ]; then
  RUNAI_ARGS+=(-e "GOOGLE_API_KEY=$GEMINI_KEY")
fi
if [ -n "$ANTHROPIC_KEY" ]; then
  RUNAI_ARGS+=(-e "ANTHROPIC_API_KEY=$ANTHROPIC_KEY")
fi

if [ "$HAS_REPO_REF" -eq 0 ]; then
  RUNAI_ARGS+=(-e "REPO_REF=$CURRENT_BRANCH")
fi

for ENV_PAIR in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do
  RUNAI_ARGS+=(-e "$ENV_PAIR")
done

RUNAI_ARGS+=(--command -- bash -c "echo $B64 | base64 -d | bash")

echo "Submitting job: $JOB_NAME"
echo "Script: $SCRIPT_NAME"
echo "Repo ref: ${EFFECTIVE_REPO_REF}"
if [ "${#EXTRA_ENV[@]}" -gt 0 ]; then
  echo "Extra env:"
  printf '  %s\n' "${EXTRA_ENV[@]}"
fi

# Submit target. Default: run `runai` on haas001 over SSH. Set RUNAI_LOCAL=1 to
# run `runai` on this machine instead — useful when the cluster-side RunAI
# session has expired but the local CLI is authenticated to the same cluster.
# API keys are still read from the cluster over SSH either way.
RUNAI_LOCAL="${RUNAI_LOCAL:-0}"
if [ "$RUNAI_LOCAL" = "1" ]; then
  echo "Submit mode: local runai CLI"
  "${RUNAI_ARGS[@]}"
else
  echo "Submit mode: ssh light"
  printf -v REMOTE_CMD '%q ' "${RUNAI_ARGS[@]}"
  ssh light "$REMOTE_CMD"
fi

echo ""
echo "Monitor with:"
if [ "$RUNAI_LOCAL" = "1" ]; then
  echo "  runai logs $JOB_NAME -f"
  echo "  runai describe job $JOB_NAME -p light-yiren"
else
  echo "  ssh light 'runai logs $JOB_NAME -f'"
  echo "  ssh light 'runai describe job $JOB_NAME -p light-yiren'"
fi
echo ""
echo "Download results:"
echo "  scp -r 'light:/mnt/light/scratch/users/yiren/eval_output/*' ~/Downloads/mamai-eval/results/"
