#!/bin/bash
set -euo pipefail
# R2c table J2/J3 — precompute all retriever×reranker arm contexts for DATASETS.
#   NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=48G CPU_REQUEST=12 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-arms-kenya run_cluster_arms_matrix.sh DATASETS=kenya
#   (healthbench): ... ./submit_job.sh mamai-arms-hb run_cluster_arms_matrix.sh DATASETS=healthbench_oss_eval

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
pip3 install --no-cache-dir -q numpy pandas tqdm sentencepiece ai-edge-litert datasets \
  huggingface_hub rank_bm25 'transformers>=4.51' \
  torch --extra-index-url https://download.pytorch.org/whl/cu124
echo "=== DEPS DONE ==="

REPO_REF="${REPO_REF:-feat/r2-retriever-upgrade-20260613}"
CONFIG="${CONFIG:-config-v0.2.0}"
DATASETS="${DATASETS:-kenya}"
ASSETS=/lightscratch/users/yiren/rag_assets
FT=/lightscratch/users/yiren/eval_output/rerank_finetune
OUT_SCRATCH="${OUT_SCRATCH:-/lightscratch/users/yiren/eval_output/arms_matrix}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
export HF_HOME=/lightscratch/users/yiren/hf_cache
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p "$OUT_SCRATCH"
rm -rf /tmp/eval_code; git clone --branch "$REPO_REF" --depth 1 https://github.com/nmrenyi/mamai-eval.git /tmp/eval_code; cd /tmp/eval_code

ARGS=(--config "$CONFIG" --db-path "$ASSETS/embeddings.sqlite"
  --gecko-model "$ASSETS/Gecko_1024_quant.tflite" --tokenizer "$ASSETS/sentencepiece.model"
  --rerankers "minilm_ft=$FT/minilm-l6-finetuned-model,mxbai_ft=$FT/mxbai-base-finetuned-model"
  --datasets "$DATASETS" --out-dir "$OUT_SCRATCH")
[ -n "$MAX_QUESTIONS" ] && ARGS+=(--max-questions "$MAX_QUESTIONS")
python3 -m retrieval_eval.precompute_arms_matrix "${ARGS[@]}"
echo "=== DONE ==="; ls -la "$OUT_SCRATCH"
