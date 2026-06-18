#!/bin/bash
set -euo pipefail
# Wrapper to run mamaretrieval LateOn (ColBERT, lightonai/GTE-ModernColBERT-v1) on KENYA via
# mamai-eval's local-runai path. Repo dir is read-only to the pod → redirect all writes.
#   ... ./submit_job.sh mamai-mr-lateon run_cluster_mr_lateon.sh

# base CUDA image has no python — install it first
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
WORK="/lightscratch/users/yiren/eval_output/mr_kenya"
mkdir -p "$WORK/out"

export REPO_DIR
export CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
export QUERIES_PATH="${QUERIES_PATH:-$REPO_DIR/data/kenya_queries.jsonl}"
export QUERY_IDS_PATH="$WORK/kenya_ids.txt"
export TOP_K="${TOP_K:-20}"
export DEVICE="${DEVICE:-cuda}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MODEL="${MODEL:-lightonai/GTE-ModernColBERT-v1}"
# writable redirects:
export RUNAI_HOME="$WORK/home_lateon"
export PYTHONUSERBASE="$WORK/pyuser_lateon"
export HF_HOME="/lightscratch/users/yiren/hf_cache"
export INDEX_FOLDER="$WORK/lateon_plaid_index_kenya"
export OUTPUT_PATH="$WORK/out/kenya_lateon_top20.jsonl"
export HF_API_KEY_FILE_AT="/lightscratch/users/yiren/keys/hf_key.txt"
mkdir -p "$RUNAI_HOME" "$PYTHONUSERBASE" "$HF_HOME" "$INDEX_FOLDER"

# build kenya_ids.txt in a writable spot
python3 -c "import json; ids=[json.loads(l)['query_id'] for l in open('$QUERIES_PATH')]; open('$QUERY_IDS_PATH','w').write('\n'.join(ids)+'\n'); print('ids',len(ids))"

echo "=== mamaretrieval LateOn (ColBERT) on kenya — model=$MODEL ==="
bash "$REPO_DIR/scripts/run_lateon_audit_job.sh"
echo "=== DONE ==="; ls -la "$OUTPUT_PATH" && wc -l "$OUTPUT_PATH"