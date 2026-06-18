#!/bin/bash
set -euo pipefail

# R2c P2.5 — fine-tune cross-encoder reranker(s) on the 230k graded pairs (GPU).
#
# Fine-tunes one or more candidates (MODELS as "key=hf_id" space-separated) on
# the TRAIN split, selects on DEV, reports held-out TEST reranking quality +
# the Stage-1 gate. Emits <key>-finetuned.json under the repo's finetune dir,
# copied to scratch.
#
# Submit (H200; both hero + best-zero-shot):
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=96G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-rerank-ft run_cluster_finetune.sh \
#     MODELS="minilm-l6=cross-encoder/ms-marco-MiniLM-L6-v2 mxbai-base=mixedbread-ai/mxbai-rerank-base-v1"

echo "=== INSTALLING DEPENDENCIES ==="
apt-get update && apt-get install -y python3.10 python3-pip python3.10-venv git > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

echo "=== INSTALLING PYTHON PACKAGES ==="
pip3 install --no-cache-dir --timeout 600 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cu124
pip3 install --no-cache-dir --timeout 600 --retries 5 \
    'transformers>=4.51' 'datasets>=2.18' 'huggingface_hub>=0.23' pandas pyarrow \
    scikit-learn accelerate
echo "=== DEPS DONE ==="

REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
MODELS="${MODELS:-minilm-l6=cross-encoder/ms-marco-MiniLM-L6-v2}"
EPOCHS="${EPOCHS:-2}"
SEQ_LEN="${SEQ_LEN:-256}"
SAVE_MODEL="${SAVE_MODEL:-1}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/rerank_finetune}"

mkdir -p "$HF_CACHE_DIR" "$OUT_SCRATCH"
export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "=== CHECKOUT ==="
rm -rf "$WORKTREE"
git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"
cd "$WORKTREE"

FEATURES_DIR="configs/config-v0.2.0/results/retrieval_eval/r2c-rerank"
OUT_DIR="$FEATURES_DIR/finetune"

nvidia-smi || true
# Copy per-model right after each finishes (with `|| true` around training) so
# one model's failure can't lose another's results/weights.
for spec in $MODELS; do
  KEY="${spec%%=*}"; MODEL="${spec#*=}"
  echo "=== FINE-TUNE $KEY ($MODEL) ==="
  EXTRA=()
  [ "$SAVE_MODEL" = "1" ] && EXTRA+=(--save-model)
  python3 -m retrieval_eval.finetune_reranker \
    --model "$MODEL" --key "$KEY" \
    --features-dir "$FEATURES_DIR" --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS" --seq-len "$SEQ_LEN" "${EXTRA[@]}" || echo "WARN: $KEY failed"
  cp -v "$OUT_DIR/$KEY-finetuned.json" "$OUT_SCRATCH/" 2>/dev/null || true
  cp -rv "$OUT_DIR/$KEY-finetuned-model" "$OUT_SCRATCH/" 2>/dev/null || true
done

echo "=== RUN COMPLETE ==="
ls -la "$OUT_SCRATCH/"
