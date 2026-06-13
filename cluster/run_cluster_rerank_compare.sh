#!/bin/bash
set -euo pipefail

# R2c P2 — score every candidate reranker on the held-out test split (GPU).
#
# Scores the deployable cross-encoders (MiniLM-L6/L12, electra-base,
# bge-reranker-base, mxbai-rerank-base, MedCPT, jina tiny/turbo) plus the
# Qwen3-Reranker 4B/8B offline references over the 9 820 test pairs, at the
# deployed seq-len (256). Emits one JSON per model under the repo's r2c-rerank
# results dir, then copies them to scratch for scp-back.
#
# Submit (H200 preferred; 8B reference fits one 80GB GPU):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=96G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-rerank-compare run_cluster_rerank_compare.sh
#
# CE-only smoke (no Qwen3, faster, single GPU):
#   NODE_POOL=h100 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-rerank-ce run_cluster_rerank_compare.sh GROUP=ce

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip python3.10-venv git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES ==="
pip3 install --no-cache-dir --timeout 600 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cu124
pip3 install --no-cache-dir --timeout 600 --retries 5 \
    'transformers>=4.51' 'huggingface_hub>=0.23' pandas pyarrow scikit-learn einops accelerate
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
GROUP="${GROUP:-all}"          # all | ce | qwen3
ONLY="${ONLY:-}"               # comma-separated keys (overrides GROUP)
MAX_LEN="${MAX_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/rerank_compare}"

mkdir -p "$HF_CACHE_DIR" "$OUT_SCRATCH"
export HF_HOME="$HF_CACHE_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

FEATURES_DIR="configs/config-v0.2.0/results/retrieval_eval/r2c-rerank"
OUT_DIR="$FEATURES_DIR/candidates"

ARGS=(
  --features-dir "$FEATURES_DIR"
  --out-dir "$OUT_DIR"
  --max-len "$MAX_LEN"
  --batch-size "$BATCH_SIZE"
)
if [ -n "$ONLY" ]; then
  ARGS+=(--only "$ONLY")
else
  ARGS+=(--group "$GROUP")
fi

echo "=== SCORING (GROUP=$GROUP ONLY=${ONLY:-<none>} MAX_LEN=$MAX_LEN) ==="
nvidia-smi || true
python3 -m retrieval_eval.score_candidates "${ARGS[@]}"

echo "=== COPYING RESULTS TO SCRATCH ==="
cp -v "$OUT_DIR"/*.json "$OUT_SCRATCH/" || true
ls -la "$OUT_SCRATCH/"
echo "=== RUN COMPLETE ==="
