"""
Pre-compute RAG retrieval contexts for all mamabench v0.2 evaluation datasets.

Embeds each question using the Gecko TFLite model and retrieves the top-k
most similar chunks from the app's vector store. Results are saved as JSON
files that run_eval.py can load with --rag.

For HealthBench-style multi-turn rows, retrieval is run on the latest user
turn — mirroring how the on-device app retrieves per-message.

Usage:
  python precompute_retrieval.py --config config-v0.2.0 \\
      --db-path /path/to/embeddings.sqlite \\
      --gecko-model /path/to/Gecko_1024_quant.tflite \\
      --tokenizer /path/to/sentencepiece.model

  python precompute_retrieval.py --config config-v0.2.0 \\
      --db-path ... --gecko-model ... --tokenizer ... \\
      --top-k 5 --datasets afrimedqa,whb
"""

import argparse
import hashlib
import json
import os
import subprocess

# ── Resolve --config before any prompts imports ──────────────────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_pre_args, _ = _pre.parse_known_args()
os.environ["MAMAI_EVAL_CONFIG"] = _pre_args.config
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from prompts import RETRIEVAL_TOP_K, CONFIG_VERSION, DATASET_HF_REPO, DATASET_REVISION
from retrieval import (
    GeckoEmbedder,
    build_index,
    format_app_context_chunks,
    load_vector_store,
    retrieve,
)

_REPO_ROOT = Path(__file__).parent

# Dataset registry: name → hf_config (set_type doesn't matter for retrieval)
HF_CONFIGS = {
    "medmcqa":               "medmcqa",
    "medqa_usmle":           "medqa_usmle",
    "afrimedqa":             "afrimedqa",
    "kenya":                 "kenya",
    "whb":                   "whb",
    "afrimedqa_saq":         "afrimedqa_saq",
    "healthbench_oss_eval":  "healthbench_oss_eval",
    "healthbench_consensus": "healthbench_consensus",
    "healthbench_hard":      "healthbench_hard",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_output(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), *args], text=True,
        ).strip()
    except Exception:
        return ""


def _question_text(question) -> str:
    """Pick the text to embed: latest user turn for multi-turn, else the string."""
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        for turn in reversed(question):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return str(turn.get("content", ""))
        if question and isinstance(question[-1], dict):
            return str(question[-1].get("content", ""))
    return str(question or "")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute RAG retrieval contexts")
    parser.add_argument("--config", required=True,
                        help="Config version to use (e.g. config-v0.2.0)")
    parser.add_argument("--db-path", required=True, help="Path to embeddings.sqlite")
    parser.add_argument("--gecko-model", required=True,
                        help="Path to Gecko TFLite model (Gecko_1024_quant.tflite)")
    parser.add_argument("--tokenizer", required=True, help="Path to sentencepiece.model")
    parser.add_argument("--revision", default=None,
                        help="HF dataset revision (default: dataset.revision from params.json, else v0.2)")
    parser.add_argument("--hf-repo", default=None,
                        help="HF dataset repo (default: dataset.hf_repo from params.json)")
    parser.add_argument("--output-dir", default="rag_contexts",
                        help="Output directory for JSON context files")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K,
                        help="Number of chunks to retrieve per question")
    parser.add_argument("--datasets", default="all",
                        help="Comma-separated dataset names, or 'all'")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions per dataset")
    parser.add_argument("--context-version", default=None,
                        help="Version label for this retrieval context set")
    parser.add_argument("--rag-lock", default=None,
                        help="Path to rag_assets.lock.json for bundle provenance metadata (optional)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    context_version = args.context_version or datetime.now(timezone.utc).strftime(
        "ragctx-%Y%m%dT%H%M%SZ"
    )

    revision = args.revision or DATASET_REVISION or "v0.2"
    hf_repo = args.hf_repo or DATASET_HF_REPO or "nmrenyi/mamabench"

    if args.datasets == "all":
        dataset_names = list(HF_CONFIGS.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        for name in dataset_names:
            if name not in HF_CONFIGS:
                parser.error(f"Unknown dataset: {name}. Available: {list(HF_CONFIGS.keys())}")

    lock_data = {}
    if args.rag_lock and Path(args.rag_lock).exists():
        lock_data = json.loads(Path(args.rag_lock).read_text())

    db_path = Path(args.db_path)
    gecko_path = Path(args.gecko_model)
    tokenizer_path = Path(args.tokenizer)
    manifest_path = Path(args.output_dir) / "manifest.json"

    run_manifest = {
        "schema_version": 2,
        "context_version": context_version,
        "config_version": CONFIG_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_ref": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "repo_commit": _git_output("rev-parse", "HEAD"),
        "source_lock": {
            "bundle_version": lock_data.get("bundle_version"),
            "manifest_sha256": lock_data.get("manifest_sha256"),
            "producer_repo": lock_data.get("producer_repo"),
            "producer_commit": lock_data.get("producer_commit"),
            "chunk_count": lock_data.get("chunk_count"),
            "source_count": lock_data.get("source_count"),
        },
        "dataset_source": {"hf_repo": hf_repo, "hf_revision": revision},
        "retrieval_config": {
            "top_k": args.top_k,
            "datasets": dataset_names,
            "max_questions": args.max_questions,
        },
        "artifacts": {
            "db_path": str(db_path.resolve()),
            "db_sha256": _sha256(db_path),
            "gecko_model_path": str(gecko_path.resolve()),
            "gecko_model_sha256": _sha256(gecko_path),
            "tokenizer_path": str(tokenizer_path.resolve()),
            "tokenizer_sha256": _sha256(tokenizer_path),
        },
        "datasets": {},
    }

    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest.get("context_version") == context_version:
            run_manifest["created_at_utc"] = existing_manifest.get(
                "created_at_utc", run_manifest["created_at_utc"],
            )
            existing_datasets = existing_manifest.get("datasets", {})
            if isinstance(existing_datasets, dict):
                run_manifest["datasets"] = existing_datasets
            existing_requested = existing_manifest.get("retrieval_config", {}).get("datasets", [])
            run_manifest["retrieval_config"]["datasets"] = sorted(
                set(existing_requested) | set(dataset_names)
            )

    store = load_vector_store(args.db_path)
    texts, normed_matrix = build_index(store)
    embedder = GeckoEmbedder(args.gecko_model, args.tokenizer)

    from datasets import load_dataset

    for ds_name in dataset_names:
        hf_config = HF_CONFIGS[ds_name]

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  ({hf_repo}/{hf_config}@{revision})")
        print(f"{'='*60}")

        try:
            ds = load_dataset(hf_repo, hf_config, revision=revision, split="test")
        except Exception as e:
            print(f"SKIP: failed to load {ds_name}: {e}")
            continue

        rows = list(ds)
        if args.max_questions:
            rows = rows[:args.max_questions]
        print(f"Processing {len(rows)} questions")

        retrievals = []
        for raw in tqdm(rows, total=len(rows), desc=ds_name):
            text = _question_text(raw.get("question", ""))
            if not text:
                retrievals.append({"id": raw.get("id", ""), "question": "",
                                   "chunks": [], "similarities": []})
                continue

            query_emb = embedder.embed(text)
            results = retrieve(query_emb, texts, normed_matrix, top_k=args.top_k)
            raw_chunks = [chunk for chunk, _ in results]
            context_chunks, retrieved_docs = format_app_context_chunks(raw_chunks)

            retrievals.append({
                "id": raw.get("id", ""),
                "question": text,
                "chunks": context_chunks,
                "retrieved_docs": retrieved_docs,
                "similarities": [round(score, 4) for _, score in results],
            })

        output = {
            "metadata": {
                "context_version": context_version,
                "config_version": CONFIG_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dataset": ds_name,
                "hf_repo": hf_repo,
                "hf_revision": revision,
            },
            "config": {
                "context_version": context_version,
                "top_k": args.top_k,
                "embedding_model": "Gecko_1024_quant",
                "n_chunks_in_store": len(store),
                "n_questions": len(retrievals),
            },
            "retrievals": retrievals,
        }

        output_path = os.path.join(args.output_dir, f"{ds_name}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")
        run_manifest["datasets"][ds_name] = {
            "n_questions": len(retrievals),
            "output_file": f"{ds_name}.json",
        }

        if retrievals and retrievals[0]["chunks"]:
            r = retrievals[0]
            print(f"\nSample — Q: {r['question'][:100]}...")
            for i, (chunk, sim) in enumerate(zip(r["chunks"], r["similarities"])):
                print(f"  [{i+1}] sim={sim:.4f}: {chunk[:80]}...")

    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
