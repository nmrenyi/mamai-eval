#!/usr/bin/env python3
"""Build the feature-LTR feature table over the hybrid top-20 pool.

For each (query, chunk) in the hybrid (Gecko+BM25 RRF, alpha=0.5/k=60) top-20
pool, computes the cheap features a feature-LTR reranker would use — all derived
from signals already available at retrieval time plus light lexical overlap —
attaches the judge grade and the frozen train/dev/test split, and writes one
parquet row per pair. No embedding, no model: pure arithmetic + string ops.

Features (per query, chunk):
  gecko_score, gecko_rank, bm25_score, bm25_rank   -- retrieval signals (NaN if
                                                      chunk absent from that
                                                      retriever's top-20)
  rrf_score, rrf_rank, in_both                      -- fusion signals
  q_len, c_len, overlap_count, jaccard, q_coverage, num_overlap  -- lexical
Label: grade (0-6). Group key: query_id. Split: train/dev/test (frozen).

The lexical features are the bet: they carry signal the embedding/RRF scores
miss. Feature importances downstream show whether that bet pays off (lexical
features matter) or whether the model is just a learned RRF (only score/rank
features matter).

Usage:
  python -m retrieval_eval.build_ltr_features \\
      --split-dir  configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \\
      --output-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ALPHA, K = 0.5, 60
TOP_K_POOL = 20
_TOKEN = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"^\d+(?:\.\d+)?$")
# Tiny stopword set so overlap reflects content words, not function words.
_STOP = set("a an the of to in for and or is are be on with as at by from this "
            "that these those it its which what when how who whom whose do does "
            "did can could should would will may might must not no".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def lexical_features(q_text: str, c_text: str) -> dict:
    q, c = set(tokens(q_text)), set(tokens(c_text))
    inter = q & c
    union = q | c
    q_nums = {t for t in q if _NUM.match(t)}
    c_nums = {t for t in c if _NUM.match(t)}
    return {
        "q_len": len(q),
        "c_len": len(c),
        "overlap_count": len(inter),
        "jaccard": len(inter) / len(union) if union else 0.0,
        "q_coverage": len(inter) / len(q) if q else 0.0,
        "num_overlap": len(q_nums & c_nums),
    }


def rank_score_maps(rk, retriever: str) -> dict:
    """{query_id: {chunk_id: (rank, score)}}."""
    sub = rk[rk["retriever"] == retriever]
    m: dict[str, dict[str, tuple[int, float]]] = {}
    for r in sub.itertuples():
        m.setdefault(r.query_id, {})[r.chunk_id] = (int(r.rank), float(r.score))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    rk = pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/rankings.parquet", repo_type="dataset",
        revision=args.revision))
    q_text = {r.query_id: r.query_text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/queries.parquet", repo_type="dataset",
        revision=args.revision)).itertuples()}
    c_text = {r.chunk_id: r.text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/chunks.parquet", repo_type="dataset",
        revision=args.revision)).itertuples()}
    judg = load_dataset(args.hf_repo, "judgments", revision=args.revision,
                        split="test")
    grades = {(r["query_id"], r["chunk_id"]): int(r["score"]) for r in judg}

    with open(Path(args.split_dir) / "split.json") as f:
        split_of = json.load(f)

    gecko = rank_score_maps(rk, "gecko")
    bm25 = rank_score_maps(rk, "bm25")

    rows = []
    n_missing_text = 0
    for qid in gecko:
        g, b = gecko[qid], bm25.get(qid, {})
        cands = set(g) | set(b)
        scored = {}
        for c in cands:
            gr = g[c][0] if c in g else None
            br = b[c][0] if c in b else None
            rrf = (ALPHA * (1.0 / (K + gr) if gr else 0.0)
                   + (1 - ALPHA) * (1.0 / (K + br) if br else 0.0))
            scored[c] = rrf
        fused = sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:TOP_K_POOL]
        for frank, (cid, rrf) in enumerate(fused, 1):
            gr, gs = g.get(cid, (np.nan, np.nan))
            br, bs = b.get(cid, (np.nan, np.nan))
            ctext = c_text.get(cid)
            if ctext is None:
                n_missing_text += 1
            lex = lexical_features(q_text.get(qid, ""), ctext or "")
            rows.append({
                "query_id": qid, "chunk_id": cid,
                "split": split_of.get(qid, "train"),
                "grade": grades.get((qid, cid), 0),
                "gecko_score": gs, "gecko_rank": gr,
                "bm25_score": bs, "bm25_rank": br,
                "rrf_score": rrf, "rrf_rank": frank,
                "in_both": int(cid in g and cid in b),
                **lex,
            })

    df = pd.DataFrame(rows)
    out_path = out_dir / "ltr_features.parquet"
    df.to_parquet(out_path, index=False)

    feature_cols = ["gecko_score", "gecko_rank", "bm25_score", "bm25_rank",
                    "rrf_score", "rrf_rank", "in_both", "q_len", "c_len",
                    "overlap_count", "jaccard", "q_coverage", "num_overlap"]
    manifest = {
        "schema_version": 1,
        "name": "r2c-ltr-features",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision},
        "pool": {"hybrid_alpha": ALPHA, "rrf_k": K, "top_k": TOP_K_POOL},
        "feature_cols": feature_cols,
        "label": "grade (0-6)", "group_key": "query_id",
        "counts": {
            "n_rows": len(df), "n_queries": df["query_id"].nunique(),
            "n_missing_chunk_text": n_missing_text,
            "by_split": {s: int((df["split"] == s).sum())
                         for s in ("train", "dev", "test")},
            "grade_ge3_rate": round(float((df["grade"] >= 3).mean()), 4),
            "grade_ge5_rate": round(float((df["grade"] >= 5).mean()), 4),
        },
        "data_file": out_path.name,
    }
    with open(out_dir / "ltr_features.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(df):,} rows ({df['query_id'].nunique()} queries) -> {out_path}")
    print(f"by split: {manifest['counts']['by_split']}")
    print(f"grade>=3 rate {manifest['counts']['grade_ge3_rate']}, "
          f">=5 rate {manifest['counts']['grade_ge5_rate']}, "
          f"missing chunk text: {n_missing_text}")


if __name__ == "__main__":
    main()
