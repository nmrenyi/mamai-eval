#!/usr/bin/env python3
"""Stage-1 score-quality gate applied to the R2a hybrid (Gecko + BM25, RRF).

The R1 negative result killed thresholding on Gecko's *cosine* scores. The
hybrid produces a different score — the RRF fused score — which is rank-based
and bounded, so it might be more thresholdable. This runs the same three
Stage-1 statistics (R1 §6) on the fused score and compares against Gecko alone:

  - chunk-level AUC (grade >= 3)            -- absolute cutoffs
  - within-bundle concordance              -- relative rules
  - bundle any-relevant AUC (top-1 score)  -- gated abstention

Uses the deployable config from R2a (alpha=0.5, k=60). Fused rankings are built
over the gecko∪bm25 top-20 union per query (all judged); the fused top-3 is the
population scored, mirroring the gate's top-3 focus.

Usage:
  python -m retrieval_eval.gate_hybrid \\
      --results-dir configs/config-v0.2.0/results/retrieval_eval/r2-hybrid \\
      --report-dir  configs/config-v0.2.0/reports/r2-hybrid
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from retrieval_eval.compare_retriever_gates import gate_stats


def load_rankings_and_grades(hf_repo: str, revision: str):
    """Return (rankings DataFrame, grades dict)."""
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    rk = pd.read_parquet(hf_hub_download(
        hf_repo, "data/rankings.parquet", repo_type="dataset", revision=revision))
    judgments = load_dataset(hf_repo, "judgments", revision=revision, split="test")
    grades = {(r["query_id"], r["chunk_id"]): int(r["score"]) for r in judgments}
    return rk, grades


def single_retriever_rows(rk, retriever: str, grades: dict) -> list[dict]:
    """Rows {query_id, rank, score, grade} from a retriever's own ranking +
    its native score (cosine for gecko, BM25 score for bm25)."""
    sub = rk[rk["retriever"] == retriever]
    rows = []
    for r in sub.itertuples():
        g = grades.get((r.query_id, r.chunk_id))
        if g is not None:
            rows.append({"query_id": r.query_id, "rank": int(r.rank),
                         "score": float(r.score), "grade": g})
    return rows


def rank_map(rk, retriever: str) -> dict:
    """{query_id: {chunk_id: rank}} for RRF fusion."""
    sub = rk[rk["retriever"] == retriever]
    m: dict[str, dict[str, int]] = {}
    for r in sub.itertuples():
        m.setdefault(r.query_id, {})[r.chunk_id] = int(r.rank)
    return m


def build_hybrid_rows(gecko: dict, bm25: dict, grades: dict,
                      alpha: float, k: int) -> list[dict]:
    """Fused rows {query_id, rank, score, grade} over the union pool per query.

    Only chunks with a known grade are kept (the fused top-3 always is, since
    it lies within each retriever's judged top-20)."""
    rows = []
    for qid in gecko:
        g, b = gecko[qid], bm25.get(qid, {})
        cands = set(g) | set(b)
        scored = {
            c: alpha * (1.0 / (k + g[c]) if c in g else 0.0)
               + (1 - alpha) * (1.0 / (k + b[c]) if c in b else 0.0)
            for c in cands
        }
        # Tie-break by chunk_id: RRF ties are common (a gecko-only and a
        # bm25-only chunk at the same rank score identically), so a stable
        # secondary key is needed for reproducibility.
        ranked = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
        for rank, (cid, sc) in enumerate(ranked, 1):
            grade = grades.get((qid, cid))
            if grade is None:
                continue
            rows.append({"query_id": qid, "rank": rank, "score": sc,
                         "grade": grade})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    rk, grades = load_rankings_and_grades(args.hf_repo, args.revision)

    # All three gates computed from data on their native score:
    #   gecko -> cosine, bm25 -> BM25 score, hybrid -> RRF fused score.
    gecko = gate_stats(single_retriever_rows(rk, "gecko", grades))
    bm25 = gate_stats(single_retriever_rows(rk, "bm25", grades))
    voyage = gate_stats(single_retriever_rows(rk, "voyage", grades))
    hybrid = gate_stats(build_hybrid_rows(
        rank_map(rk, "gecko"), rank_map(rk, "bm25"), grades, args.alpha, args.k))
    hybrid["config"] = {"alpha": args.alpha, "k": args.k}

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision},
        "note": "each gate computed on the retriever's native score: gecko cosine, "
                "bm25 BM25 score, voyage cosine (API-only ceiling), hybrid RRF "
                "fused score. gecko chunk AUC should reproduce R1 §3 (0.572).",
        "gecko_cosine": gecko,
        "bm25": bm25,
        "voyage_cosine_ceiling": voyage,
        "hybrid_rrf": hybrid,
    }
    with open(results_dir / "hybrid_gate.json", "w") as f:
        json.dump(out, f, indent=2)

    # Comparison figure: gecko cosine vs bm25 vs hybrid RRF on the 3 stats.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["chunk AUC\n(grade>=3)\nabsolute cutoffs",
              "within-bundle\nconcordance\nrelative rules",
              "bundle any-relevant\nAUC\ngated abstention"]
    series = [
        ("Gecko cosine", "#b3372f", [gecko["chunk_auc_grade3"],
         gecko["within_bundle_concordance"], gecko["bundle_any_relevant_auc_top1"]]),
        ("BM25 score", "#92510a", [bm25["chunk_auc_grade3"],
         bm25["within_bundle_concordance"], bm25["bundle_any_relevant_auc_top1"]]),
        (f"Hybrid RRF (α={args.alpha}, k={args.k})", "#4a6fa5",
         [hybrid["chunk_auc_grade3"], hybrid["within_bundle_concordance"],
          hybrid["bundle_any_relevant_auc_top1"]]),
    ]
    voyage_vals = [voyage["chunk_auc_grade3"], voyage["within_bundle_concordance"],
                   voyage["bundle_any_relevant_auc_top1"]]
    x = range(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for j, (name, color, vals) in enumerate(series):
        offs = (j - 1) * width
        ax.bar([i + offs for i in x], vals, width, label=name, color=color, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i + offs, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5)
    # voyage (API-only ceiling) as a per-stat dotted reference marker.
    for i, v in enumerate(voyage_vals):
        ax.hlines(v, i - 0.45, i + 0.45, color="#6b21a8", ls=":", lw=2.2,
                  label="voyage (API-only ceiling)" if i == 0 else None)
        ax.text(i + 0.45, v + 0.006, f"{v:.3f}", ha="left", va="bottom",
                fontsize=7.5, color="#6b21a8")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.axhline(0.8, ls="--", color="#14532d", lw=1, label="viability bar")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("statistic")
    ax.set_title("Stage-1 score-quality gate: Gecko cosine vs BM25 vs hybrid RRF "
                 "(dotted=chance, dashed=viability bar)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "fig_hybrid_gate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def line(name, g):
        print(f"  {name:14s} chunkAUC={g['chunk_auc_grade3']}  "
              f"conc={g['within_bundle_concordance']}  "
              f"bundleAUC={g['bundle_any_relevant_auc_top1']}")
    print("Stage-1 gate on native scores:")
    line("gecko cosine", gecko)
    line("bm25 score", bm25)
    line("hybrid RRF", hybrid)
    line("voyage ceiling", voyage)
    print(f"(gecko chunk AUC should reproduce R1's 0.572)")
    print(f"Written: {results_dir}/hybrid_gate.json")


if __name__ == "__main__":
    main()
