#!/bin/bash
set -euo pipefail
# Model-size/precision ladder — GGUF rung via llama.cpp. Downloads one quant, generates the
# G1 prompt arm on kenya + healthbench_oss_eval for the requested retrieval modes (Track A).
# Used for the Gemma 4 GGUF rungs (E2B-Q4, and the E4B precision axis Q8_0 / BF16), all ±RAG.
#
# Submit (A100 default pool is plenty for these):
#   NODE_POOL=default GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
#     ./submit_job.sh mamai-lad-e4b-q8 run_cluster_gguf_ladder.sh \
#     MODEL=gemma4-e4b-q8 GGUF_REPO=unsloth/gemma-4-E4B-it-GGUF GGUF_FILE=gemma-4-E4B-it-Q8_0.gguf \
#     DEST_DIR=gemma-4-E4B-gguf RAG_MODES="norag rag" REPO_REF=feat/qwen-norag-tracka-20260624

MODEL="${MODEL:?registry name, e.g. gemma4-e4b-q8}"
GGUF_REPO="${GGUF_REPO:?HF GGUF repo, e.g. unsloth/gemma-4-E4B-it-GGUF}"
GGUF_FILE="${GGUF_FILE:?GGUF filename, e.g. gemma-4-E4B-it-Q8_0.gguf}"
DEST_DIR="${DEST_DIR:?subdir under MODEL_DIR matching the registry path prefix}"
RAG_MODES="${RAG_MODES:-norag rag}"
PROMPT="${PROMPT:-g1}"
DATASETS="${DATASETS:-kenya healthbench_oss_eval}"
CONFIG="${CONFIG:-config-v0.2.0}"
MODEL_DIR="${MODEL_DIR:-/lightscratch/users/yiren/models}"
OUT_BASE="${OUT_BASE:-/lightscratch/users/yiren/eval_output/size_ladder_20260624/$MODEL}"
EG_KENYA="${EG_KENYA:-/lightscratch/users/yiren/eval_output/rag_arms_eg/embeddinggemma}"
EG_HB="${EG_HB:-/lightscratch/users/yiren/eval_output/rag_arms_eg_hb/embeddinggemma}"
REPO_REF="${REPO_REF:-feat/qwen-norag-tracka-20260624}"
REPO_URL="${REPO_URL:-https://github.com/nmrenyi/mamai-eval.git}"
WORKTREE="${WORKTREE:-/tmp/eval_code}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/lightscratch/users/yiren/hf_cache}"

echo "=== DEPS ==="
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git cmake build-essential ninja-build > /dev/null
ln -sf /usr/bin/python3.10 /usr/bin/python3
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip3 install --no-cache-dir -q --retries 10 \
  llama-cpp-python numpy datasets huggingface_hub tqdm > /dev/null
echo "=== DEPS DONE ==="

mkdir -p "$HF_CACHE_DIR" "$MODEL_DIR/$DEST_DIR"; export HF_HOME="$HF_CACHE_DIR"
[ -n "${HF_TOKEN:-}" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "=== DOWNLOAD $GGUF_REPO/$GGUF_FILE -> $MODEL_DIR/$DEST_DIR ==="
python3 - "$GGUF_REPO" "$GGUF_FILE" "$MODEL_DIR/$DEST_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, fname, dest = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo, filename=fname, local_dir=dest)
print("downloaded:", p)
PY
ls -la "$MODEL_DIR/$DEST_DIR/$GGUF_FILE"

rm -rf "$WORKTREE"; git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$WORKTREE"; cd "$WORKTREE"
PFX="$WORKTREE/configs/$CONFIG/results/end_to_end_eval/g1-ab-3n-20260619/prompts"
case "$PROMPT" in g1) SPF="$PFX/arm2_system_en.txt";; g1g2) SPF="$PFX/arm3_system_en.txt";; *) SPF="";; esac
SP=(); [ -n "$SPF" ] && SP=(--system-prompt "$SPF")
rag_dir_for(){ case "$1" in kenya) echo "$EG_KENYA";; healthbench_oss_eval) echo "$EG_HB";; *) echo "";; esac; }

for mode in $RAG_MODES; do
  OUT="$OUT_BASE/${PROMPT}_${mode}"
  for ds in $DATASETS; do
    echo "########## $MODEL prompt=$PROMPT mode=$mode dataset=$ds ##########"
    RAG=()
    if [ "$mode" = "rag" ]; then RAG=(--rag "$(rag_dir_for "$ds")"); fi
    python3 end_to_end_eval/run_eval.py --config "$CONFIG" --model "$MODEL" --model-dir "$MODEL_DIR" \
      --datasets "$ds" ${SP[@]+"${SP[@]}"} ${RAG[@]+"${RAG[@]}"} \
      --output-dir "$OUT" --run-dir "$OUT/run"
  done
done

echo "=== DONE — $MODEL ($GGUF_FILE) ladder cells generated ==="
for mode in $RAG_MODES; do echo "--- ${PROMPT}_${mode} ---"; ls -la "$OUT_BASE/${PROMPT}_${mode}/run" 2>/dev/null | grep -iE 'json'; done
