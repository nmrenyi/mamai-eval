#!/bin/bash
set -euo pipefail
# Thresholdability follow-up: dump per-pair cross-encoder scores for EmbeddingGemma's
# kenya top-20 candidates (MiniLM-L6-ft & mxbai-base-ft). rerank_retrievals writes each
# candidate's ce_score, which we join with the existing relevance grades to ask whether the
# RERANKER score is more thresholdable than EmbeddingGemma cosine (R1's 0.80 chunk-AUC bar).
# No generator / no vLLM — just the CE. Fast.
#
# Submit (1 GPU):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=48G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-eg-rrscore run_cluster_eg_rerank_scores.sh

FT="${FT:-/lightscratch/users/yiren/eval_output/rerank_finetune}"
RETR="${RETR:-/lightscratch/users/yiren/eval_output/screen_embedder/retrievals_embeddinggemma_300m_d768.json}"
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/eg_thresh}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q --retries 10 numpy 'transformers>=4.51' \
  torch --extra-index-url https://download.pytorch.org/whl/cu124 > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$OUT_SCRATCH"; export HF_HOME="$HF_CACHE_DIR"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"

for RR in minilm_ft mxbai_ft; do
  case "$RR" in
    minilm_ft) RRPATH="$FT/minilm-l6-finetuned-model" ;;
    mxbai_ft)  RRPATH="$FT/mxbai-base-finetuned-model" ;;
  esac
  echo "=== RERANK SCORES ($RR) over EmbeddingGemma kenya top-20 ==="
  python3 -m retrieval_eval.rerank_retrievals \
    --retrievals "$RETR" --reranker "$RRPATH" \
    --out "$OUT_SCRATCH/eg_kenya_reranked_${RR}.json" --seq-len 256
done
echo "=== DONE ==="; ls -la "$OUT_SCRATCH"