#!/bin/bash
set -euo pipefail
# Wrapper to run mamaretrieval LateOn (ColBERT, lightonai/GTE-ModernColBERT-v1) retrieval on
# the KENYA queries, via mamai-eval's local-runai path. Builds a PLAID index over the 63k corpus.
#   ... ./submit_job.sh mamai-mr-lateon run_cluster_mr_lateon.sh

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
export REPO_DIR
export CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
export QUERIES_PATH="${QUERIES_PATH:-$REPO_DIR/data/kenya_queries.jsonl}"
export QUERY_IDS_PATH="${QUERY_IDS_PATH:-$REPO_DIR/data/kenya_ids.txt}"
export TOP_K="${TOP_K:-20}"
export DEVICE="${DEVICE:-cuda}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MODEL="${MODEL:-lightonai/GTE-ModernColBERT-v1}"
export INDEX_FOLDER="${INDEX_FOLDER:-$REPO_DIR/.cache/lateon_plaid_index_kenya}"
export OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/data/kenya_lateon_top20.jsonl}"
export HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"

# ensure kenya_ids.txt exists on the cluster (built from kenya_queries.jsonl)
python3 -c "import json,os; p='$QUERIES_PATH'; ids=[json.loads(l)['query_id'] for l in open(p)]; open('$QUERY_IDS_PATH','w').write('\n'.join(ids)+'\n'); print('ids',len(ids))" || true

echo "=== mamaretrieval LateOn (ColBERT) retrieval on kenya ==="
bash "$REPO_DIR/scripts/run_lateon_audit_job.sh"
echo "=== DONE ==="; ls -la "$OUTPUT_PATH" && wc -l "$OUTPUT_PATH"