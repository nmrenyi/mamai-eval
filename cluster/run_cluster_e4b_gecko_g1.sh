#!/bin/bash
set -euo pipefail
# Gecko-retriever conversion arm: Gemma 4 E4B (int4/Q4_0, deployed) + G1 prompt + GECKO RAG,
# on kenya + healthbench_oss_eval. Built to be directly comparable to the deployed
# EmbeddingGemma arm (E4B-int4 + G1 + EG, config-v0.3.0) so we can see whether the offline
# retrieval win (Gecko->EmbeddingGemma, +12-13pp P@3) converts to online open-ended gains.
#
# Fair comparison: gecko arm built with the SAME pure-dense embed_retrieve top-3 pipeline
# as the EG arm (not the hybrid alpha/RRF arms_matrix arm).
#
# Submit (1 GPU; gecko retrieval is CPU-tflite, gemma gen needs the GPU):
#   NODE_POOL=h100 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-e4b-gecko-g1 run_cluster_e4b_gecko_g1.sh REPO_REF=feat/qwen-norag-tracka-20260624

CONFIG="${CONFIG:-config-v0.2.0}"
MODEL="${MODEL:-gemma4-e4b}"            # int4 Q4_0 GGUF, already in MODEL_DIR
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
DATASETS="${DATASETS:-kenya,healthbench_oss_eval}"   # comma-sep for screen_embedder
TOPK="${TOPK:-3}"
ASSETS="${ASSETS:-/lightscratch/users/yiren/rag_assets}"
OUT_DIR="${OUT_DIR:-/lightscratch/users/yiren/eval_output/e4b_gecko_g1_20260624}"
ARM_DIR="${ARM_DIR:-$OUT_DIR/gecko_arm}"
REPO_REF="${REPO_REF:-feat/qwen-norag-tracka-20260624}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git cmake build-essential ninja-build > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
export PIP_ONLY_BINARY="numpy,pyarrow"
pip3 install --no-cache-dir -q --retries 10 numpy ai-edge-litert sentencepiece datasets huggingface_hub tqdm > /dev/null
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir -q --retries 10 llama-cpp-python > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$ARM_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"
PFX="$WORKTREE/configs/$CONFIG/results/end_to_end_eval/g1-ab-3n-20260619/prompts"
SP=(--system-prompt "$PFX/arm2_system_en.txt")   # G1 (deployed) prompt

RETR="$OUT_DIR/retrievals_gecko.json"
echo "=== RETRIEVE gecko (pure-dense, top-$TOPK) over $DATASETS ==="
python3 -m retrieval_eval.screen_embedder embed_retrieve --candidate gecko \
  --gecko-model "$ASSETS/Gecko_1024_quant.tflite" --tokenizer "$ASSETS/sentencepiece.model" \
  --db-path "$ASSETS/embeddings.sqlite" --datasets "$DATASETS" --top-k "$TOPK" --out "$RETR"
echo "=== BUILD ARM ==="
python3 -m retrieval_eval.screen_embedder arm_format --retrievals "$RETR" --out-dir "$ARM_DIR" --top-k "$TOPK"

echo "=== GENERATE $MODEL + G1 + gecko RAG ==="
for ds in ${DATASETS//,/ }; do
  echo "########## $MODEL G1 gecko-RAG dataset=$ds ##########"
  python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" --model-dir "$MODEL_DIR" \
    --datasets "$ds" --rag "$ARM_DIR" "${SP[@]}" \
    --output-dir "$OUT_DIR" --run-dir "$OUT_DIR/run"
done

echo "=== DONE — E4B-int4 + G1 + gecko RAG generated ==="
ls -la "$OUT_DIR/run"
