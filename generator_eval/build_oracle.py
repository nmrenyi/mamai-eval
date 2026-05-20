#!/usr/bin/env python3
"""Build oracle context for faithfulness evaluation from mamaretrieval.

For each query in nmrenyi/mamaretrieval, selects all chunks judged at
`score >= threshold` (default 5) under the four-dimension rubric, sorted
by score descending then chunk_id ascending. Chunk text is inlined so
the downstream faithfulness runner consumes a single self-contained
JSONL — no chunk-table lookup at eval time.

Rubric: `score = d1 * (d2 + d3 + d4) ∈ [0..6]`. See the mamaretrieval
dataset card for the full definition. v0.1.0 ships top-3 union across
6 retrievers (3,185 queries, 36,418 (q,c) judgments). Calibration on a
62-pair pilot vs Claude Opus 4.7 reports 95% threshold agreement at
score >= 3 and 85% at score >= 5 — score >= 5 is the validated cut.

Outputs under `--output-dir`:
  <name>.jsonl           one row per query (the oracle)
  <name>.manifest.json   provenance + counts

Usage:
    python -m generator_eval.build_oracle
    python -m generator_eval.build_oracle --threshold 6
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def build_oracle(hf_repo: str, revision: str, threshold: int) -> tuple[list[dict], dict]:
    """Pull mamaretrieval from HF and assemble per-query oracle rows.

    Returns (oracle_rows, counts) where each oracle row is
    {query_id, query_text, chunks: [{chunk_id, score, text}, ...]}.
    """
    from datasets import load_dataset

    print(f"Loading {hf_repo}@{revision} (queries, judgments, chunks)")
    queries = load_dataset(hf_repo, "queries", revision=revision, split="test")
    judgments = load_dataset(hf_repo, "judgments", revision=revision, split="test")
    chunks_ds = load_dataset(hf_repo, "chunks", revision=revision, split="test")

    chunk_text = {row["chunk_id"]: row["text"] for row in chunks_ds}
    query_text = {row["query_id"]: row["query_text"] for row in queries}
    print(f"  {len(query_text):,} queries, {len(judgments):,} judgments, "
          f"{len(chunk_text):,} chunks")

    by_query: dict[str, list[tuple[str, int]]] = {}
    for row in judgments:
        if row["score"] >= threshold:
            by_query.setdefault(row["query_id"], []).append(
                (row["chunk_id"], row["score"])
            )

    oracle_rows: list[dict] = []
    chunks_per_query: list[int] = []
    for query_id in sorted(query_text.keys()):
        pairs = by_query.get(query_id)
        if not pairs:
            continue
        # Score desc, chunk_id asc — deterministic and puts strongest first.
        pairs.sort(key=lambda x: (-x[1], x[0]))
        oracle_rows.append({
            "query_id": query_id,
            "query_text": query_text[query_id],
            "chunks": [
                {"chunk_id": cid, "score": score, "text": chunk_text[cid]}
                for cid, score in pairs
            ],
        })
        chunks_per_query.append(len(pairs))

    counts = {
        "n_queries_total": len(query_text),
        "n_queries_with_oracle": len(oracle_rows),
        "n_qc_pairs": sum(chunks_per_query),
        "chunks_per_query_histogram": dict(sorted(Counter(chunks_per_query).items())),
    }
    return oracle_rows, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    p.add_argument("--revision", default="v0.1.0")
    p.add_argument("--threshold", type=int, default=5,
                   help="Minimum judgment score (0..6) for a chunk to count as oracle. Default 5.")
    p.add_argument("--output-dir", default="configs/config-v0.2.0/oracle",
                   help="Directory for the JSONL + manifest.")
    p.add_argument("--name", default=None,
                   help="Artifact basename. Default mamaretrieval-<revision>-score<threshold>.")
    args = p.parse_args()

    oracle_rows, counts = build_oracle(args.hf_repo, args.revision, args.threshold)

    name = args.name or f"mamaretrieval-{args.revision}-score{args.threshold}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"{name}.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for row in oracle_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "name": name,
        "source": {"hf_repo": args.hf_repo, "revision": args.revision},
        "threshold": args.threshold,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_file": data_path.name,
        "counts": counts,
    }
    manifest_path = out_dir / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {data_path} ({data_path.stat().st_size:,} bytes)")
    print(f"Wrote {manifest_path}")
    print()
    print(f"  Queries with >=1 qualifying chunk: "
          f"{counts['n_queries_with_oracle']:,}/{counts['n_queries_total']:,}")
    print(f"  Total (query, chunk) pairs:        {counts['n_qc_pairs']:,}")


if __name__ == "__main__":
    main()
