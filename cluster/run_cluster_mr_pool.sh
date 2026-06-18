#!/bin/bash
set -euo pipefail
# Wrapper to run mamaretrieval pool_candidates (bm25/medcpt/octen) on the KENYA queries via
# mamai-eval's local-runai path. The mamaretrieval repo dir is READ-ONLY to the pod, so all
# writes (deps, caches, output) are redirected to the writable eval_output PVC area, per-job.
#   ... ./submit_job.sh mamai-mr-octen  run_cluster_mr_pool.sh RETRIEVERS=octen
#   ... ./submit_job.sh mamai-mr-medcpt run_cluster_mr_pool.sh RETRIEVERS=medcpt,bm25

# base CUDA image has no python — install it before the mamaretrieval job script runs
apt-get update -qq && apt-get install -y -qq python3.10 python3-pip git curl > /dev/null 2>&1
ln -sf /usr/bin/python3.10 /usr/bin/python3

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
RETRIEVERS="${RETRIEVERS:-octen}"
TAG="$(echo "$RETRIEVERS" | tr ',' '_')"
WORK="/lightscratch/users/yiren/eval_output/mr_kenya"
mkdir -p "$WORK/out"

export REPO_DIR
export CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
export QUERIES_PATH="${QUERIES_PATH:-$REPO_DIR/data/kenya_queries.jsonl}"
export RETRIEVERS
export TOP_K="${TOP_K:-20}"
export SHARD_INDEX="${SHARD_INDEX:-0}"
export SHARD_COUNT="${SHARD_COUNT:-1}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export DEVICE="${DEVICE:-cuda}"
# writable redirects (repo dir is read-only to the pod):
export RUNAI_HOME="$WORK/home_${TAG}"
export PYTHONUSERBASE="$WORK/pyuser_${TAG}"
export HF_HOME="/lightscratch/users/yiren/hf_cache"
export CACHE_DIR="$WORK/cache"            # corpus embedding cache (persists, shared)
export OUTPUT_PATH="$WORK/out/kenya_candidates_${TAG}.jsonl"
export HF_API_KEY_FILE_AT="/lightscratch/users/yiren/keys/hf_key.txt"
mkdir -p "$RUNAI_HOME" "$PYTHONUSERBASE" "$HF_HOME" "$CACHE_DIR"

echo "=== mamaretrieval pool_candidates: RETRIEVERS=$RETRIEVERS TOP_K=$TOP_K on kenya ==="
echo "OUTPUT=$OUTPUT_PATH  CACHE=$CACHE_DIR  PYUSER=$PYTHONUSERBASE"
bash "$REPO_DIR/scripts/run_pool_candidates_job.sh"
echo "=== DONE ==="; ls -la "$OUTPUT_PATH" && wc -l "$OUTPUT_PATH"