#!/usr/bin/env python3
"""Build the HealthBench outcome table (Figure 2b of the R1 threshold plan).

HealthBench analogue of build_mcq_outcome_table.py: joins the no-RAG and +RAG
rubric-track runs row-by-row on the judge-scored `weighted_met`, and recomputes
the deployed Gecko retrieval locally (latest user turn, top-3 — mirroring
precompute_retrieval.py) to recover the injected bundle's cosine scores. The
outcome here is continuous: delta = weighted_met(+RAG) − weighted_met(no-RAG).

Parity caveat: rubric result rows do not store the injected rag_context, so the
per-row context check used for the MCQ table is impossible here. The recompute
uses the same assets and code path that achieved 97.4% row-level parity on the
MCQ table; the manifest records this as inherited, not verified.

Usage (from the repo root):
  python -m retrieval_eval.build_rubric_outcome_table \\
      --norag-dir configs/.../20260521T123051-cluster-norag-rubric \\
      --rag-dir   configs/.../20260521T122626-cluster-rag-rubric \\
      --output-dir configs/config-v0.2.0/results/retrieval_eval/r1-threshold
"""

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASETS = ["healthbench_oss_eval", "healthbench_consensus", "healthbench_hard"]
DEFAULT_SQLITE = "/Users/renyi/Downloads/mamai-medical-guidelines/releases/rag-bundle-v0.2.0/runtime/embeddings.sqlite"
DEFAULT_MODEL = "/Users/renyi/Downloads/mamai/device_push/models/Gecko_1024_quant.tflite"
DEFAULT_TOKENIZER = "/Users/renyi/Downloads/mamai/device_push/models/sentencepiece.model"

_worker = {}


def question_text(question) -> str:
    """Latest user turn for multi-turn rows (same rule as precompute_retrieval)."""
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        for turn in reversed(question):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return str(turn.get("content", ""))
        if question and isinstance(question[-1], dict):
            return str(question[-1].get("content", ""))
    return str(question or "")


def _init_worker(db_path: str, model_path: str, tokenizer_path: str, top_k: int):
    from retrieval_eval.retrieval import (GeckoEmbedder, build_index,
                                          load_vector_store)
    store = load_vector_store(db_path)
    texts, normed = build_index(store)
    _worker["texts"] = texts
    _worker["normed"] = normed
    _worker["embedder"] = GeckoEmbedder(model_path, tokenizer_path)
    _worker["top_k"] = top_k


def _score_row(task):
    from retrieval_eval.retrieval import retrieve
    idx, text = task
    emb = _worker["embedder"].embed(text)
    results = retrieve(emb, _worker["texts"], _worker["normed"],
                       top_k=_worker["top_k"])
    return idx, [round(s, 6) for _, s in results]


def split_of(row_id: str) -> str:
    return "tune" if int(hashlib.md5(row_id.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--norag-dir", required=True)
    ap.add_argument("--rag-dir", required=True)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--db-path", default=DEFAULT_SQLITE)
    ap.add_argument("--gecko-model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--per-embed-cpu-s", type=float, default=2.9,
                    help="measured per-embed CPU cost, for the upfront ETA print")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joined = []  # (id, dataset, text, met_norag, met_rag)
    for ds in args.datasets.split(","):
        norag = {r["id"]: r for r in json.load(open(Path(args.norag_dir) / f"{ds}.json"))["results"]}
        rag = {r["id"]: r for r in json.load(open(Path(args.rag_dir) / f"{ds}.json"))["results"]}
        for rid in rag:
            if rid not in norag:
                continue
            joined.append((rid, ds, question_text(rag[rid]["question"]),
                           float(norag[rid]["rubric_score"]["weighted_met"]),
                           float(rag[rid]["rubric_score"]["weighted_met"])))
        print(f"{ds}: joined {sum(1 for j in joined if j[1] == ds):,} rows", flush=True)

    eta_min = len(joined) * args.per_embed_cpu_s / args.n_workers / 60
    print(f"\n{len(joined):,} embeds to compute, ~{args.per_embed_cpu_s} CPU-s each "
          f"on {args.n_workers} workers -> ETA ~{eta_min:.0f} min", flush=True)

    tasks = [(i, j[2]) for i, j in enumerate(joined)]
    scored = {}
    t0 = time.time()
    with mp.Pool(processes=args.n_workers, initializer=_init_worker,
                 initargs=(args.db_path, args.gecko_model, args.tokenizer,
                           args.top_k)) as pool:
        for n_done, (idx, cosines) in enumerate(
                pool.imap_unordered(_score_row, tasks, chunksize=16), 1):
            scored[idx] = cosines
            if n_done % 100 == 0 or n_done == len(tasks):
                rate = n_done / (time.time() - t0)
                remaining = (len(tasks) - n_done) / rate / 60
                print(f"  progress: {n_done}/{len(tasks)} "
                      f"({n_done / len(tasks):.0%}), ~{remaining:.1f} min left",
                      flush=True)

    rows_out = []
    for i, (rid, ds, _text, met_norag, met_rag) in enumerate(joined):
        rows_out.append({
            "id": rid, "dataset": ds, "split": split_of(rid),
            "weighted_met_norag": round(met_norag, 4),
            "weighted_met_rag": round(met_rag, 4),
            "delta": round(met_rag - met_norag, 4),
            "cosines": scored[i],
        })

    table_path = out_dir / "table_b2_healthbench.jsonl.gz"
    with gzip.open(table_path, "wt") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "schema_version": 1,
        "name": "r1-threshold-table-b2-healthbench",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_runs": {"norag": args.norag_dir, "rag": args.rag_dir},
        # Record asset basenames only — absolute paths are machine-specific and
        # leak the developer's filesystem layout into committed manifests.
        "assets": {"db": Path(args.db_path).name,
                   "gecko_model": Path(args.gecko_model).name,
                   "tokenizer": Path(args.tokenizer).name},
        "retrieval": {"top_k": args.top_k,
                      "query": "latest user turn (precompute_retrieval parity)"},
        "parity": "not verifiable per-row (rubric results store no rag_context); "
                  "same pipeline scored 97.4% on the MCQ table",
        "counts": {"n_rows": len(rows_out),
                   "per_dataset": {ds: sum(1 for r in rows_out if r["dataset"] == ds)
                                   for ds in args.datasets.split(",")}},
        "split_rule": "md5(id) % 2 — tune if even",
        "data_file": table_path.name,
    }
    with open(out_dir / "table_b2_healthbench.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(rows_out):,} rows -> {table_path}", flush=True)


if __name__ == "__main__":
    main()
