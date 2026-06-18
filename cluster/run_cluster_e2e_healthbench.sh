#!/bin/bash
set -euo pipefail
# R2c Phase 3 (healthbench) — per arm: retrieve healthbench_oss_eval, build arm, generate
# gemma4-e4b. Judge after with rescore_rubric.sh (RUN_DIRS=$OUT_DIR/$ARM/run).
#
# Submit one job per arm (1x H200 each):
#   ... ./submit_job.sh mamai-hb-eg    run_cluster_e2e_healthbench.sh CANDIDATE=google/embeddinggemma-300m ARM=embeddinggemma DIM=768
#   ... ./submit_job.sh mamai-hb-gecko run_cluster_e2e_healthbench.sh CANDIDATE=gecko ARM=gecko

CANDIDATE="${CANDIDATE:-google/embeddinggemma-300m}"
DIM="${DIM:-768}"
DATASETS="${DATASETS:-healthbench_oss_eval}"
MAXQ="${MAXQ:-500}"
TOPK="${TOPK:-3}"
ARM="${ARM:-embeddinggemma}"
CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma4-e4b}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
ASSETS="${ASSETS:-/lightscratch/users/yiren/rag_assets}"
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/value_gate_eg_hb}"
ARMS_DIR="${ARMS_DIR:-/lightscratch/users/yiren/eval_output/rag_arms_eg_hb}"
REPO_REF="${REPO_REF:-main}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"
WT=/tmp/eval_code

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git cmake build-essential ninja-build > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
if [ "$CANDIDATE" = "gecko" ]; then
  pip3 install --no-cache-dir -q --retries 10 numpy ai-edge-litert sentencepiece datasets huggingface_hub tqdm > /dev/null
else
  pip3 install --no-cache-dir -q --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch 'sentence-transformers>=5.0' 'transformers>=4.56' datasets numpy huggingface_hub tqdm > /dev/null
fi
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir -q --retries 10 llama-cpp-python > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$OUT_DIR" "$ARMS_DIR/$ARM"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WT"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WT"; cd "$WT"

RETR="$OUT_DIR/retrievals_${ARM}.json"
echo "=== RETRIEVE ($CANDIDATE, top-$TOPK) ==="
if [ "$CANDIDATE" = "gecko" ]; then
  python3 -m retrieval_eval.screen_embedder embed_retrieve --candidate gecko \
    --gecko-model "$ASSETS/Gecko_1024_quant.tflite" --tokenizer "$ASSETS/sentencepiece.model" \
    --db-path "$ASSETS/embeddings.sqlite" --datasets "$DATASETS" --top-k "$TOPK" --out "$RETR"
else
  python3 -m retrieval_eval.screen_embedder embed_retrieve --candidate "$CANDIDATE" --dim "$DIM" \
    --db-path "$ASSETS/embeddings.sqlite" --datasets "$DATASETS" --top-k "$TOPK" --out "$RETR"
fi
echo "=== BUILD ARM ==="
python3 -m retrieval_eval.screen_embedder arm_format --retrievals "$RETR" --out-dir "$ARMS_DIR/$ARM" --top-k "$TOPK"
echo "=== GENERATE $MODEL over $DATASETS (max $MAXQ) ==="
python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" --model-dir "$MODEL_DIR" \
  --datasets "$DATASETS" --rag "$ARMS_DIR/$ARM" --max-questions "$MAXQ" \
  --output-dir "$OUT_DIR/$ARM" --run-dir "$OUT_DIR/$ARM/run"
echo "=== DONE (judge: rescore_rubric.sh RUN_DIRS=$OUT_DIR/$ARM/run) ==="
ls -la "$OUT_DIR/$ARM/run"