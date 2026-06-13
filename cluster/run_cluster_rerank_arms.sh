#!/bin/bash
set -euo pipefail

# R2c P1 (step 1/3) — precompute the 3-arm RAG contexts (gecko / hybrid /
# hybrid+rerank) for the end-to-end value gate. CPU-only (Gecko TFLite + BM25 +
# a cross-encoder rerank over the hybrid top-20). Writes <out>/{gecko,hybrid,
# hybrid_rerank}/<dataset>.json to scratch.
#
# Submit (CPU, no GPU needed):
#   GPU_REQUEST=0 CPU_REQUEST=8 MEMORY_REQUEST=32G RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-arms run_cluster_rerank_arms.sh \
#     RERANKER=/lightscratch/users/yiren/eval_output/rerank_finetune/minilm-l6-finetuned-model \
#     DATASETS=kenya,afrimedqa_saq

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3
echo "=== INSTALLING PYTHON PACKAGES ==="
pip3 install --no-cache-dir numpy pandas tqdm sentencepiece ai-edge-litert \
  datasets huggingface_hub rank_bm25 'transformers>=4.51' \
  torch --extra-index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
CONFIG="${CONFIG:-config-v0.2.0}"
DB_PATH="${DB_PATH:-/lightscratch/users/yiren/rag_assets/embeddings.sqlite}"
GECKO_MODEL="${GECKO_MODEL:-/lightscratch/users/yiren/rag_assets/Gecko_1024_quant.tflite}"
TOKENIZER="${TOKENIZER:-/lightscratch/users/yiren/rag_assets/sentencepiece.model}"
RERANKER="${RERANKER:-mixedbread-ai/mxbai-rerank-base-v1}"
RERANKER_SEQ_LEN="${RERANKER_SEQ_LEN:-256}"
DATASETS="${DATASETS:-kenya,afrimedqa_saq}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/rag_arms}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

mkdir -p "$OUT_SCRATCH" "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

ARGS=(
  --config "$CONFIG" --db-path "$DB_PATH" --gecko-model "$GECKO_MODEL"
  --tokenizer "$TOKENIZER" --reranker "$RERANKER"
  --reranker-seq-len "$RERANKER_SEQ_LEN" --datasets "$DATASETS"
  --out-dir "$OUT_SCRATCH"
)
[ -n "$MAX_QUESTIONS" ] && ARGS+=(--max-questions "$MAX_QUESTIONS")

echo "=== PRECOMPUTE ARMS (reranker=$RERANKER) ==="
python3 -m retrieval_eval.precompute_rerank_arms "${ARGS[@]}"
echo "=== DONE ==="
ls -laR "$OUT_SCRATCH" | head -40
