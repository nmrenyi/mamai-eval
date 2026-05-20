#!/bin/bash
set -euo pipefail

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3
echo "=== INSTALLING PYTHON PACKAGES ==="
# v0.2 adds `datasets` + `huggingface_hub` (HF replaces local TSV) and keeps
# `ai-edge-litert` for the Gecko embedder and `sentencepiece` for the tokenizer.
pip3 install --no-cache-dir \
  numpy pandas tqdm sentencepiece ai-edge-litert datasets huggingface_hub \
  > /dev/null 2>&1
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-main}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
CONFIG="${CONFIG:-config-v0.2.0}"
# Container paths (PVC `light-scratch` mounts at /lightscratch — see submit_job.sh).
# RAG assets are scp'd from the local mamai/device_push/ tree by the submitter.
DB_PATH="${DB_PATH:-/lightscratch/users/yiren/rag_assets/embeddings.sqlite}"
GECKO_MODEL="${GECKO_MODEL:-/lightscratch/users/yiren/rag_assets/Gecko_1024_quant.tflite}"
TOKENIZER="${TOKENIZER:-/lightscratch/users/yiren/rag_assets/sentencepiece.model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lightscratch/users/yiren/eval_output/rag_contexts}"
CONTEXT_VERSION="${CONTEXT_VERSION:-ragctx-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$CONTEXT_VERSION}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
DATASETS="${DATASETS:-all}"
TOP_K="${TOP_K:-3}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
MEDMCQA_MAX_QUESTIONS="${MEDMCQA_MAX_QUESTIONS-500}"
ROW_IDS="${ROW_IDS:-}"

# Persist HF dataset cache across job runs (survives container lifecycle via PVC).
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

mkdir -p "$OUTPUT_DIR"

if [ "$DATASETS" = "all" ]; then
  # v0.2 default: MCQ + open-ended only. open_ended_rubric configs (HealthBench)
  # are multi-turn and quite different — opt in explicitly if you want their
  # contexts precomputed.
  DATASET_LIST=(
    "afrimedqa"
    "medqa_usmle"
    "medmcqa"
    "kenya"
    "whb"
    "afrimedqa_saq"
  )
else
  IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
fi

echo "=== STARTING RAG PRECOMPUTATION ==="
echo "CONFIG=$CONFIG"
echo "REPO_REF=$REPO_REF"
echo "CONTEXT_VERSION=$CONTEXT_VERSION"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HF_HOME=$HF_HOME"
echo "TOP_K=$TOP_K"
echo "DATASETS=${DATASET_LIST[*]}"

for RAW_DS in "${DATASET_LIST[@]}"; do
  DS="$(echo "$RAW_DS" | xargs)"
  if [ -z "$DS" ]; then
    continue
  fi

  if [ -f "$OUTPUT_DIR/${DS}.json" ]; then
    echo "SKIP $DS: already exists at $OUTPUT_DIR/${DS}.json"
    continue
  fi

  DATASET_ARGS=(
    --config "$CONFIG"
    --db-path "$DB_PATH"
    --gecko-model "$GECKO_MODEL"
    --tokenizer "$TOKENIZER"
    --output-dir "$OUTPUT_DIR"
    --context-version "$CONTEXT_VERSION"
    --top-k "$TOP_K"
    --datasets "$DS"
  )

  if [ -n "$MAX_QUESTIONS" ]; then
    DATASET_ARGS+=(--max-questions "$MAX_QUESTIONS")
  elif [ "$DS" = "medmcqa" ] && [ -n "$MEDMCQA_MAX_QUESTIONS" ]; then
    DATASET_ARGS+=(--max-questions "$MEDMCQA_MAX_QUESTIONS")
  fi

  if [ -n "$ROW_IDS" ]; then
    # Same path discipline as run_cluster.sh's ROW_IDS — typically points at a
    # committed manifest like $WORKTREE/configs/<cfg>/calibration/<manifest>.json.
    DATASET_ARGS+=(--row-ids "$ROW_IDS")
  fi

  echo "Processing $DS..."
  python3 precompute_retrieval.py "${DATASET_ARGS[@]}"
done

echo "=== PRECOMPUTE COMPLETE ==="
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort
