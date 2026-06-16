#!/bin/bash
set -euo pipefail
# Wrapper to run the mamaretrieval pool_candidates retrievers (bm25/medcpt/octen) on the
# KENYA queries, submitted via mamai-eval's local-runai path (the cluster-side runai token
# is expired). The mamaretrieval repo + corpus already live on the PVC.
#
# Submit (1x H200 per retriever set; run octen and medcpt,bm25 as separate parallel jobs):
#   ... ./submit_job.sh mamai-mr-octen run_cluster_mr_pool.sh RETRIEVERS=octen
#   ... ./submit_job.sh mamai-mr-medcpt run_cluster_mr_pool.sh RETRIEVERS=medcpt,bm25

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
export REPO_DIR
export CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
export QUERIES_PATH="${QUERIES_PATH:-$REPO_DIR/data/kenya_queries.jsonl}"
export RETRIEVERS="${RETRIEVERS:-octen}"
export TOP_K="${TOP_K:-20}"
export SHARD_INDEX="${SHARD_INDEX:-0}"
export SHARD_COUNT="${SHARD_COUNT:-1}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export DEVICE="${DEVICE:-cuda}"
export CACHE_DIR="${CACHE_DIR:-$REPO_DIR/.cache}"
TAG="$(echo "$RETRIEVERS" | tr ',' '_')"
export OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/data/kenya_candidates_${TAG}.jsonl}"
export HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"

echo "=== mamaretrieval pool_candidates: RETRIEVERS=$RETRIEVERS TOP_K=$TOP_K on kenya ==="
echo "REPO_DIR=$REPO_DIR  OUTPUT=$OUTPUT_PATH"
bash "$REPO_DIR/scripts/run_pool_candidates_job.sh"
echo "=== DONE ==="; ls -la "$OUTPUT_PATH" && wc -l "$OUTPUT_PATH"