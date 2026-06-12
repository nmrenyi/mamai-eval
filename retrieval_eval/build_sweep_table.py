#!/usr/bin/env python3
"""Build Table A (the R1 threshold sweep table) from mamaretrieval.

Joins Gecko's published audit rankings (data/rankings.parquet on
nmrenyi/mamaretrieval — top-20 chunks with cosine scores per query, produced
on-cluster by mamaretrieval's retrieve_gecko_audit.py using the deployed
Gecko_1024_quant.tflite + sentencepiece.model + v0.2.0 bundle sqlite) against
the 0–6 LLM relevance judgments. Output is the per-pair table that Figures 1
and 3 of docs/r1-threshold-tuning-plan.md are drawn from:

    {query_id, chunk_id, grade, cosine, gecko_rank}

Score parity: the rankings were computed with the exact deployed artifacts, so
the cosine scale matches what the app sees at runtime. As a cross-check, the
scores are compared against mamaretrieval's local 100-query smoke run of the
same script (gecko_top20.jsonl, Apple-silicon build of the same TFLite model);
the manifest records the max/mean absolute difference. A local full recompute
was abandoned — ~4 s/query on a laptop CPU for identical numbers.

Scope note: the table covers Gecko's top-20 per query. That is the entire
population a deployment-time threshold can act on (the app injects top-3;
top-20 is the analysis margin) — judged pairs outside Gecko's top-20 are
unreachable by per-chunk filtering and are deliberately excluded.

Usage (from the repo root):
  python -m retrieval_eval.build_sweep_table \\
      --output-dir configs/config-v0.2.0/results/retrieval_eval/r1-threshold
"""

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_HF_REPO = "nmrenyi/mamaretrieval"
DEFAULT_REVISION = "v0.2.0"
DEFAULT_PARITY_FILE = "/Users/renyi/Downloads/mamaretrieval/data/audit/gecko_top20.jsonl"


def load_gecko_rankings(hf_repo: str, revision: str):
    """Gecko top-20 rows from rankings.parquet: list of dicts with
    query_id, chunk_id, rank, score."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(hf_repo, "data/rankings.parquet",
                           repo_type="dataset", revision=revision)
    df = pd.read_parquet(path)
    return df[df["retriever"] == "gecko"].to_dict("records")


def load_grades(hf_repo: str, revision: str) -> dict[tuple[str, str], int]:
    """(query_id, chunk_id) -> grade from the judgments config."""
    from datasets import load_dataset
    judgments = load_dataset(hf_repo, "judgments", revision=revision, split="test")
    return {(r["query_id"], r["chunk_id"]): r["score"] for r in judgments}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--parity-file", default=DEFAULT_PARITY_FILE,
                    help="local gecko_top20.jsonl smoke file ('' to skip)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rankings = load_gecko_rankings(args.hf_repo, args.revision)
    grades = load_grades(args.hf_repo, args.revision)
    print(f"{len(rankings):,} gecko ranking rows, {len(grades):,} judged pairs")

    rows_out = []
    unjudged = {1: 0, 2: 0, 3: 0, "4-20": 0}
    queries = set()
    for r in rankings:
        queries.add(r["query_id"])
        grade = grades.get((r["query_id"], r["chunk_id"]))
        if grade is None:
            unjudged[r["rank"] if r["rank"] <= 3 else "4-20"] += 1
            continue
        rows_out.append({
            "query_id": r["query_id"], "chunk_id": r["chunk_id"],
            "grade": int(grade), "cosine": round(float(r["score"]), 6),
            "gecko_rank": int(r["rank"]),
        })

    # Parity vs the local smoke run of the same retrieval script.
    parity = {"n_pairs_checked": 0, "max_abs_diff": None, "mean_abs_diff": None}
    if args.parity_file and Path(args.parity_file).exists():
        ref = {}
        with open(args.parity_file) as f:
            for line in f:
                r = json.loads(line)
                for x in r["results"]:
                    ref[(r["query_id"], x["chunk_id"])] = x["score"]
        score_of = {(r["query_id"], r["chunk_id"]): r["cosine"] for r in rows_out}
        diffs = [abs(score_of[k] - v) for k, v in ref.items() if k in score_of]
        if diffs:
            parity = {"n_pairs_checked": len(diffs),
                      "max_abs_diff": round(max(diffs), 6),
                      "mean_abs_diff": round(sum(diffs) / len(diffs), 6)}

    table_path = out_dir / "table_a.jsonl.gz"
    with gzip.open(table_path, "wt") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    n_top3 = sum(1 for r in rows_out if r["gecko_rank"] <= 3)
    manifest = {
        "schema_version": 2,
        "name": "r1-threshold-table-a",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision,
                   "rankings_file": "data/rankings.parquet (retriever=gecko)",
                   "judgments": "judgments config"},
        "counts": {
            "n_queries": len(queries),
            "n_pairs": len(rows_out),
            "n_top3_pairs": n_top3,
            "top3_judged_coverage": round(n_top3 / (3 * len(queries)), 4),
            "unjudged_by_rank": unjudged,
        },
        "parity_vs_local_smoke": parity,
        "data_file": table_path.name,
    }
    with open(out_dir / "table_a.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(rows_out):,} pairs -> {table_path}")
    print(f"Top-3 judged coverage: {manifest['counts']['top3_judged_coverage']:.1%}")
    print(f"Unjudged by rank: {unjudged}")
    print(f"Parity vs local smoke: {parity}")


if __name__ == "__main__":
    main()
