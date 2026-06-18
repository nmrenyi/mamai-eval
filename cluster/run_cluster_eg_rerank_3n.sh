#!/bin/bash
set -euo pipefail
# EmbeddingGemma + reranker (MiniLM-L6-ft, mxbai-base-ft) on Gemma 3n, kenya + healthbench.
# Closes the one untested cell: EmbeddingGemma candidates re-scored by a fine-tuned CE.
#   1. regenerate EmbeddingGemma healthbench retrievals at depth-20 (kenya depth-20 already exists)
#   2. rerank the depth-20 candidates with each CE -> top-3 arm
#   3. generate gemma3n answers per (reranker) arm over kenya+healthbench
# Judge after with run_cluster_value_gate_judge.sh (kenya recall + hb rubric).
#
# Submit (1x H200):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-eg-rr-3n run_cluster_eg_rerank_3n.sh

CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma3n-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
DATASETS="${DATASETS:-kenya,healthbench_oss_eval}"
ASSETS="${ASSETS:-/lightscratch/users/yiren/rag_assets}"
FT="${FT:-/lightscratch/users/yiren/eval_output/rerank_finetune}"
KENYA_D20="${KENYA_D20:-/lightscratch/users/yiren/eval_output/screen_embedder/retrievals_embeddinggemma_300m_d768.json}"
ARMS_DIR="${ARMS_DIR:-/lightscratch/users/yiren/eval_output/rag_arms_eg_rr_3n}"
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/value_gate_eg_rr_3n}"
EG_MODEL="${EG_MODEL:-google/embeddinggemma-300m}"
EG_DIM="${EG_DIM:-768}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git cmake build-essential ninja-build > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir -q --retries 10 \
  llama-cpp-python numpy datasets huggingface_hub tqdm \
  'transformers>=4.51' 'sentence-transformers>=3.0' \
  torch --extra-index-url https://download.pytorch.org/whl/cu124 > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$ARMS_DIR" "$OUT_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

# 1. EmbeddingGemma healthbench retrievals at depth-20 (kenya depth-20 already exists)
HB_D20=/tmp/eg_hb_d20.json
echo "=== RETRIEVE (EmbeddingGemma depth-20) healthbench ==="
python3 -m retrieval_eval.screen_embedder embed_retrieve \
  --candidate "$EG_MODEL" --dim "$EG_DIM" --db-path "$ASSETS/embeddings.sqlite" \
  --datasets healthbench_oss_eval --top-k 20 --out "$HB_D20"
ls -la "$HB_D20" "$KENYA_D20"

# 2. rerank depth-20 candidates with each CE, then format top-3 arms
for RR in minilm_ft mxbai_ft; do
  case "$RR" in
    minilm_ft) RRPATH="$FT/minilm-l6-finetuned-model" ;;
    mxbai_ft)  RRPATH="$FT/mxbai-base-finetuned-model" ;;
  esac
  echo "=== RERANK ($RR) from $RRPATH ==="
  python3 -m retrieval_eval.rerank_retrievals \
    --retrievals "$KENYA_D20,$HB_D20" --reranker "$RRPATH" \
    --out "/tmp/eg_${RR}.json" --seq-len 256
  echo "=== ARM_FORMAT ($RR) top-3 ==="
  python3 -m retrieval_eval.screen_embedder arm_format \
    --retrievals "/tmp/eg_${RR}.json" --out-dir "$ARMS_DIR/$RR" --top-k 3
  ls -la "$ARMS_DIR/$RR"
done

# 3. generate gemma3n per reranker arm over kenya + healthbench
for RR in minilm_ft mxbai_ft; do
  echo "=== GENERATE $MODEL  arm=$RR  datasets=$DATASETS ==="
  python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" \
    --model-dir "$MODEL_DIR" --datasets "$DATASETS" \
    --rag "$ARMS_DIR/$RR" --output-dir "$OUT_DIR/$RR" --run-dir "$OUT_DIR/$RR/run"
  ls -la "$OUT_DIR/$RR/run"
done

echo "=== DONE — judge with run_cluster_value_gate_judge.sh RUN_DIRS=$OUT_DIR/minilm_ft/run $OUT_DIR/mxbai_ft/run ==="
