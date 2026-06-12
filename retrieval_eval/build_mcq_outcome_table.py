#!/usr/bin/env python3
"""Build Table B (the R1 MCQ outcome table) from existing ±RAG result runs.

Joins the committed no-RAG and +RAG MCQ result JSONs row-by-row, labels each
row's outcome (hurt = right→wrong under RAG, helped = wrong→right, unchanged),
and recomputes the deployed Gecko retrieval locally to recover the per-chunk
cosine scores of the injected bundle (the result files store the injected text
but not the scores). Figure 2 of docs/r1-threshold-tuning-plan.md is drawn
from this table.

Parity: for every row, the locally recomputed top-3 context (formatted as
``Document N:`` blocks joined by blank lines, exactly as run_eval.py injected
it) is string-compared against the ``rag_context`` stored in the +RAG result
row. The match rate is reported in the manifest; a low rate would mean the
local assets differ from the ones used in the cluster run and the scores
cannot be trusted.

Tune/holdout split: deterministic per-row md5(id) parity. Figure 2 and all
threshold *selection* use split == "tune" only; "holdout" is reserved for
acceptance (see the plan doc).

Usage (from the repo root):
  python -m retrieval_eval.build_mcq_outcome_table \\
      --norag-dir configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/20260520T032705-cluster-norag \\
      --rag-dir   configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/20260520T082028-cluster-rag \\
      --output-dir configs/config-v0.2.0/results/retrieval_eval/r1-threshold
"""

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASETS = ["medmcqa", "medqa_usmle", "afrimedqa"]
DEFAULT_SQLITE = "/Users/renyi/Downloads/mamai-medical-guidelines/releases/rag-bundle-v0.2.0/runtime/embeddings.sqlite"
DEFAULT_MODEL = "/Users/renyi/Downloads/mamai/device_push/models/Gecko_1024_quant.tflite"
DEFAULT_TOKENIZER = "/Users/renyi/Downloads/mamai/device_push/models/sentencepiece.model"

_worker = {}


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
    from retrieval_eval.retrieval import format_app_context_chunks, retrieve
    idx, question, stored_context = task
    emb = _worker["embedder"].embed(question)
    results = retrieve(emb, _worker["texts"], _worker["normed"],
                       top_k=_worker["top_k"])
    chunks, _ = format_app_context_chunks([c for c, _ in results])
    recomputed_context = "\n\n".join(chunks)
    # run_eval.py:144 stores only context_str[:200] + "..." — compare that form.
    return idx, {
        "cosines": [round(s, 6) for _, s in results],
        "context_match": (recomputed_context[:200] + "...") == stored_context,
    }


def _is_correct(row: dict) -> bool:
    v = row.get("correct")
    return v if isinstance(v, bool) else str(v).lower() == "true"


def outcome_label(norag_correct: bool, rag_correct: bool) -> str:
    if norag_correct and not rag_correct:
        return "hurt"
    if not norag_correct and rag_correct:
        return "helped"
    return "unchanged"


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
    ap.add_argument("--max-rows", type=int, default=None, help="debug limit per dataset")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    join_stats = {}
    for ds in args.datasets.split(","):
        norag = {r["id"]: r for r in json.load(open(Path(args.norag_dir) / f"{ds}.json"))["results"]}
        rag = {r["id"]: r for r in json.load(open(Path(args.rag_dir) / f"{ds}.json"))["results"]}
        common = [i for i in rag if i in norag]
        if args.max_rows:
            common = common[:args.max_rows]
        join_stats[ds] = {"norag": len(norag), "rag": len(rag), "joined": len(common)}
        print(f"{ds}: {len(common):,} joined rows "
              f"(norag {len(norag):,}, rag {len(rag):,})")

        tasks = [(i, rag[rid]["question"], rag[rid].get("rag_context", ""))
                 for i, rid in enumerate(common)]
        with mp.Pool(processes=args.n_workers, initializer=_init_worker,
                     initargs=(args.db_path, args.gecko_model, args.tokenizer,
                               args.top_k)) as pool:
            scored = dict(pool.imap_unordered(_score_row, tasks, chunksize=64))

        for i, rid in enumerate(common):
            s = scored[i]
            rows_out.append({
                "id": rid,
                "dataset": ds,
                "split": split_of(rid),
                "outcome": outcome_label(_is_correct(norag[rid]), _is_correct(rag[rid])),
                "cosines": s["cosines"],
                "context_match": s["context_match"],
            })
        n_match = sum(1 for r in rows_out if r["dataset"] == ds and r["context_match"])
        n_ds = sum(1 for r in rows_out if r["dataset"] == ds)
        print(f"  context parity: {n_match}/{n_ds} = {n_match / n_ds:.1%}")

    table_path = out_dir / "table_b.jsonl.gz"
    with gzip.open(table_path, "wt") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    n_match = sum(1 for r in rows_out if r["context_match"])
    outcomes = {}
    for r in rows_out:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    manifest = {
        "schema_version": 1,
        "name": "r1-threshold-table-b",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_runs": {"norag": args.norag_dir, "rag": args.rag_dir},
        # Record asset basenames only — absolute paths are machine-specific and
        # leak the developer's filesystem layout into committed manifests.
        "assets": {"db": Path(args.db_path).name,
                   "gecko_model": Path(args.gecko_model).name,
                   "tokenizer": Path(args.tokenizer).name},
        "retrieval": {"top_k": args.top_k},
        "counts": {"n_rows": len(rows_out), "per_dataset": join_stats,
                   "outcomes": outcomes,
                   "context_parity_rate": round(n_match / len(rows_out), 4)},
        "split_rule": "md5(id) % 2 — tune if even; selection uses tune only",
        "data_file": table_path.name,
    }
    with open(out_dir / "table_b.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(rows_out):,} rows -> {table_path}")
    print(f"Outcomes: {outcomes}")
    print(f"Context parity overall: {n_match / len(rows_out):.1%}")


if __name__ == "__main__":
    main()
