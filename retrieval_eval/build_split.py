#!/usr/bin/env python3
"""Freeze the by-query train/dev/test split for all R2c reranker work.

70/15/15 over the 3,185 mamaretrieval queries, split BY QUERY (all of a query's
graded pairs stay on one side — no leakage), STRATIFIED on a single binary label
(does the query have >=1 strict-relevant grade>=5 chunk anywhere in its judged
pool), assigned by DETERMINISTIC hash within each stratum so the proportions are
exact and the split is reproducible. Frozen once here and reused by every later
reranker (feature-LTR, zero-shot cross-encoders, fine-tuned models) so all
numbers are reported on the same untouched test queries.

Why stratify on strict only, and against the HYBRID TOP-20 pool: the strict
metric is computed by reranking the hybrid top-20, so the variable that decides
whether strict P@3/HR@3 is even achievable for a query is "is there a grade>=5
chunk in its hybrid top-20" — ~78% yes (genuine 78/22 variance). Lenient
presence in the pool is ~99% (near-constant, nothing to balance), and strict
presence in the *union* pool of all 6 retrievers is ~94% (also near-constant) —
neither is worth stratifying on. We balance the one variable that is both
variable and drives the noisy metric. See the R2b report and
docs/r2c-reranker-literature-review-20260613.md.

Usage:
  python -m retrieval_eval.build_split \\
      --output-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

TRAIN_FRAC, DEV_FRAC = 0.70, 0.15  # test = remainder
ALPHA, K = 0.5, 60  # hybrid config (R2a) the stratum pool is defined against
TOP_K_POOL = 20     # the pool the reranker reorders / metric is computed on


def hash_unit(query_id: str) -> float:
    """Deterministic uniform value in [0,1) from the query id."""
    return int(hashlib.md5(query_id.encode()).hexdigest(), 16) / (1 << 128)


def assign(query_id: str) -> str:
    u = hash_unit(query_id)
    if u < TRAIN_FRAC:
        return "train"
    if u < TRAIN_FRAC + DEV_FRAC:
        return "dev"
    return "test"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from huggingface_hub import hf_hub_download

    from retrieval_eval.gate_hybrid import (load_rankings_and_grades, rank_map,
                                            build_hybrid_rows)

    queries = pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/queries.parquet", repo_type="dataset",
        revision=args.revision))

    # Stratum label: query has >=1 strict-relevant (grade>=5) chunk in its
    # HYBRID top-20 pool (the pool the reranker reorders and the metric scores).
    rk, grades = load_rankings_and_grades(args.hf_repo, args.revision)
    fused = build_hybrid_rows(rank_map(rk, "gecko"), rank_map(rk, "bm25"),
                              grades, ALPHA, K)
    strict_set = set()
    by_q: dict[str, list] = {}
    for r in fused:
        if r["rank"] <= TOP_K_POOL:
            by_q.setdefault(r["query_id"], []).append(r["grade"])
    for qid, gr in by_q.items():
        if any(g >= 5 for g in gr):
            strict_set.add(qid)

    rows = []
    for qid in sorted(queries["query_id"]):
        rows.append({"query_id": qid, "split": assign(qid),
                     "has_strict": qid in strict_set})

    split_of = {r["query_id"]: r["split"] for r in rows}
    with open(out_dir / "split.json", "w") as f:
        json.dump(split_of, f)

    # Verify stratification: each split should hold ~same strict-rate.
    def counts(pred):
        sub = [r for r in rows if pred(r)]
        n = len(sub)
        strict = sum(1 for r in sub if r["has_strict"])
        return {"n": n, "strict_rate": round(strict / n, 4) if n else None}

    manifest = {
        "schema_version": 1,
        "name": "r2c-query-split",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision},
        "rule": "by-query 70/15/15, stratified on has_strict (grade>=5 in the "
                f"hybrid top-{TOP_K_POOL} pool, RRF alpha={ALPHA} k={K}), "
                "deterministic md5(query_id) hash within stratum",
        "stratum_pool": {"hybrid_alpha": ALPHA, "rrf_k": K, "top_k": TOP_K_POOL},
        "fractions": {"train": TRAIN_FRAC, "dev": DEV_FRAC,
                      "test": round(1 - TRAIN_FRAC - DEV_FRAC, 2)},
        "overall": counts(lambda r: True),
        "by_split": {s: counts(lambda r: r["split"] == s)
                     for s in ("train", "dev", "test")},
        "data_file": "split.json",
    }
    with open(out_dir / "split.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Split frozen -> {out_dir}/split.json")
    print(f"overall: {manifest['overall']}")
    for s, c in manifest["by_split"].items():
        print(f"  {s:5s}: {c}")


if __name__ == "__main__":
    main()
