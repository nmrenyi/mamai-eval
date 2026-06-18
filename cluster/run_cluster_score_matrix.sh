#!/bin/bash
set -euo pipefail
# R2c table J1 — offline retriever×reranker matrix on the mamaretrieval test split.
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-matrix-offline run_cluster_score_matrix.sh

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q torch --index-url https://download.pytorch.org/whl/cu124
pip3 install --no-cache-dir -q 'transformers>=4.51' 'huggingface_hub>=0.23' pandas pyarrow scikit-learn
echo "=== DEPS DONE ==="

REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
FT=/lightscratch/users/yiren/eval_output/rerank_finetune
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/matrix}"
export HF_HOME=/lightscratch/users/yiren/hf_cache
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p "$OUT_SCRATCH"
rm -rf /tmp/eval_code; git clone --branch "$REPO_REF" --depth 1 https://github.com/nmrenyi/mamai-eval.git /tmp/eval_code; cd /tmp/eval_code

python3 -m retrieval_eval.score_pool_matrix \
  --features-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \
  --models "minilm_ft=$FT/minilm-l6-finetuned-model,mxbai_ft=$FT/mxbai-base-finetuned-model" \
  --out "$OUT_SCRATCH/matrix_offline.json"
echo "=== DONE ==="; cat "$OUT_SCRATCH/matrix_offline.json"
