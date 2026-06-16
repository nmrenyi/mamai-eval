#!/usr/bin/env python3
"""R2c table — offline retriever×reranker matrix on the mamaretrieval test split.

For each first-stage pool (gecko / bm25 / hybrid top-20, built from the published
rankings.parquet) and each reranker (none / a fine-tuned cross-encoder), reports
P@3 (lenient grade>=3, strict grade>=5) + the Stage-1 gate (chunk AUC etc.). This
fills the OFFLINE half of the 2D table — i.e. rerank gecko's / bm25's own pool,
not just the hybrid pool the earlier P2 run used.

Usage (cluster GPU, fine-tuned models on scratch):
  python -m retrieval_eval.score_pool_matrix \\
      --features-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \\
      --models minilm_ft=/lightscratch/.../minilm-l6-finetuned-model,mxbai_ft=/lightscratch/.../mxbai-base-finetuned-model \\
      --out matrix_offline.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_eval.compare_retriever_gates import gate_stats

ALPHA, RRF_K, POOL = 0.5, 60, 20


def rerank_metrics_from_pool(pool, score_key, cut):
    """pool: {qid: [{id, grade, <score_key>}]}. P@3 over top-3 by score."""
    p = hr = n = 0
    for qid, rows in pool.items():
        top3 = sorted(rows, key=lambda r: (-r[score_key], r["id"]))[:3]
        rel = sum(r["grade"] >= cut for r in top3)
        p += min(rel, 3) / 3.0; hr += int(rel > 0); n += 1
    return round(p / n, 4), round(hr / n, 4)


def gate_from_pool(pool, score_key):
    rows = []
    for qid, rs in pool.items():
        g = sorted(rs, key=lambda r: (-r[score_key], r["id"]))
        for rank, r in enumerate(g, 1):
            rows.append({"query_id": qid, "rank": rank, "score": float(r[score_key]), "grade": int(r["grade"])})
    return gate_stats(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--models", default="", help="comma list key=path of fine-tuned CE rerankers")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(Path(args.features_dir) / "split.json"))
    test_q = {q for q, s in split.items() if s == "test"}

    rk = pd.read_parquet(hf_hub_download(args.hf_repo, "data/rankings.parquet", repo_type="dataset", revision=args.revision))
    rk = rk[rk["query_id"].isin(test_q)]
    jg = pd.read_parquet(hf_hub_download(args.hf_repo, "data/judgments.parquet", repo_type="dataset", revision=args.revision))
    grade = {(r.query_id, r.chunk_id): int(r.score) for r in jg.itertuples()}
    q_text = {r.query_id: r.query_text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/queries.parquet", repo_type="dataset", revision=args.revision)).itertuples()}
    c_text = {r.chunk_id: r.text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/chunks.parquet", repo_type="dataset", revision=args.revision)).itertuples()}

    # per-retriever rank maps
    def rank_map(retr):
        d = {}
        for r in rk[rk["retriever"] == retr].itertuples():
            d.setdefault(r.query_id, {})[r.chunk_id] = (r.rank, r.score)
        return d
    g_rank, b_rank = rank_map("gecko"), rank_map("bm25")
    queries = sorted(test_q)

    # build the three pools (top-20 chunk_ids) per query
    pools = {"gecko": {}, "bm25": {}, "hybrid": {}}
    for q in queries:
        gr, br = g_rank.get(q, {}), b_rank.get(q, {})
        gtop = sorted(gr, key=lambda c: gr[c][0])[:POOL]
        btop = sorted(br, key=lambda c: br[c][0])[:POOL]
        pools["gecko"][q] = [{"id": c, "grade": grade.get((q, c), 0), "own": float(-gr[c][0])} for c in gtop]
        pools["bm25"][q] = [{"id": c, "grade": grade.get((q, c), 0), "own": float(-br[c][0])} for c in btop]
        # hybrid: RRF over gecko+bm25 top-50
        g50 = {c: r for c, (r, _) in sorted(gr.items(), key=lambda kv: kv[1][0])[:50]}
        b50 = {c: r for c, (r, _) in sorted(br.items(), key=lambda kv: kv[1][0])[:50]}
        rrf = {}
        for c in set(g50) | set(b50):
            s = 0.0
            if c in g50: s += ALPHA / (RRF_K + g50[c])
            if c in b50: s += (1 - ALPHA) / (RRF_K + b50[c])
            rrf[c] = s
        htop = sorted(rrf, key=lambda c: (-rrf[c], c))[:POOL]
        pools["hybrid"][q] = [{"id": c, "grade": grade.get((q, c), 0), "own": rrf[c]} for c in htop]

    cuts = {"lenient_ge3": 3, "strict_ge5": 5}
    out = {"created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_test_queries": len(queries), "device": device, "cells": {}}

    def record(retr, rname, scored_pool, score_key):
        rec = {"by_cut": {}}
        for ck, cv in cuts.items():
            p, hr = rerank_metrics_from_pool(scored_pool, score_key, cv)
            rec["by_cut"][ck] = {"p_at_3": p, "hr_at_3": hr}
        g = gate_from_pool(scored_pool, score_key)
        rec["stage1_gate"] = {k: g[k] for k in ("chunk_auc_grade3", "chunk_auc_grade5",
                              "within_bundle_concordance", "bundle_any_relevant_auc_top1")}
        out["cells"][f"{retr}__{rname}"] = rec
        print(f"  {retr:7} + {rname:10} P@3(>=3)={rec['by_cut']['lenient_ge3']['p_at_3']} "
              f"strict={rec['by_cut']['strict_ge5']['p_at_3']} AUC3={rec['stage1_gate']['chunk_auc_grade3']}", flush=True)

    # reranker none: use the retriever's own order (the "own" score)
    for retr in pools:
        record(retr, "none", pools[retr], "own")

    # fine-tuned rerankers: score each pool row, store under "rr"
    models = [kv for kv in args.models.split(",") if kv.strip()]
    for kv in models:
        rname, path = kv.split("=", 1)
        print(f"=== reranker {rname} ({path}) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(device)
        nlab = model.config.num_labels
        for retr in pools:
            scored_pool = {}
            for q in queries:
                lst = pools[retr][q]
                qt = q_text.get(q, "")
                pairs_txt = [c_text.get(r["id"], "") for r in lst]
                sc = np.zeros(len(lst), np.float32)
                with torch.no_grad():
                    for s in range(0, len(lst), args.batch_size):
                        enc = tok([qt] * len(pairs_txt[s:s + args.batch_size]), pairs_txt[s:s + args.batch_size],
                                  truncation=True, max_length=args.seq_len, padding=True, return_tensors="pt").to(device)
                        lg = model(**enc).logits
                        sc[s:s + args.batch_size] = (lg.squeeze(-1) if nlab == 1 else lg[:, -1]).float().cpu().numpy()
                scored_pool[q] = [{"id": lst[i]["id"], "grade": lst[i]["grade"], "rr": float(sc[i])}
                                  for i in range(len(lst))]
            record(retr, rname, scored_pool, "rr")

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("Written:", args.out, flush=True)


if __name__ == "__main__":
    main()
