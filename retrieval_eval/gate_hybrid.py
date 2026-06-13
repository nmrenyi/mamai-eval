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
from retrieval_eval.simulate_hybrid import load_rank_maps


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
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        for rank, (cid, sc) in enumerate(ranked, 1):
            grade = grades.get((qid, cid))
            if grade is None:
                continue
            rows.append({"query_id": qid, "rank": rank, "score": sc,
                         "grade": grade})
    return rows


def gecko_rows(gecko: dict, grades: dict) -> list[dict]:
    rows = []
    for qid, ranks in gecko.items():
        for cid, rk in ranks.items():
            g = grades.get((qid, cid))
            if g is not None:
                # cosine score is irrelevant to rank-based stats; use -rank so
                # within-bundle / pooled ordering matches gecko's own ranking.
                rows.append({"query_id": qid, "rank": rk, "score": -rk, "grade": g})
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

    rank_maps, grades = load_rank_maps(args.hf_repo, args.revision)
    gecko, bm25 = rank_maps["gecko"], rank_maps["bm25"]

    hybrid = gate_stats(build_hybrid_rows(gecko, bm25, grades, args.alpha, args.k))
    hybrid["config"] = {"alpha": args.alpha, "k": args.k}

    # Gecko reference: rank-based stats reproduce R1's cosine numbers because
    # within gecko's own top-3 the cosine order IS the rank order; the chunk-AUC
    # here uses -rank, so it is NOT the cosine AUC (that was 0.572 in R1). We
    # cite R1's cosine AUC explicitly instead of recomputing it from ranks.
    gecko_g = gate_stats(gecko_rows(gecko, grades))

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision},
        "note": "hybrid score = RRF fused score; gecko_rank_based uses -rank so "
                "its chunk_auc is a rank proxy, NOT gecko's cosine AUC (0.572, "
                "see R1 §3). Compare the hybrid's RRF-score AUC against 0.572.",
        "gecko_cosine_auc_grade3_from_r1": 0.572,
        "hybrid_rrf": hybrid,
        "gecko_rank_based_reference": gecko_g,
    }
    with open(results_dir / "hybrid_gate.json", "w") as f:
        json.dump(out, f, indent=2)

    # Comparison figure: gecko cosine (from R1) vs hybrid RRF on the 3 stats.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["chunk AUC\n(grade>=3)\nabsolute cutoffs",
              "within-bundle\nconcordance\nrelative rules",
              "bundle any-relevant\nAUC\ngated abstention"]
    gecko_vals = [0.572, 0.623, 0.574]   # R1 §3 cosine numbers
    hybrid_vals = [hybrid["chunk_auc_grade3"], hybrid["within_bundle_concordance"],
                   hybrid["bundle_any_relevant_auc_top1"]]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar([i - 0.2 for i in x], gecko_vals, 0.4, label="Gecko cosine (R1)",
           color="#b3372f", alpha=0.85)
    ax.bar([i + 0.2 for i in x], hybrid_vals, 0.4,
           label=f"Hybrid RRF (α={args.alpha}, k={args.k})", color="#4a6fa5",
           alpha=0.85)
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.axhline(0.8, ls="--", color="#14532d", lw=1, label="viability bar")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("statistic")
    for i, (g, h) in enumerate(zip(gecko_vals, hybrid_vals)):
        ax.text(i - 0.2, g + 0.008, f"{g:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, h + 0.008, f"{h:.3f}", ha="center", fontsize=8)
    ax.set_title("Stage-1 score-quality gate: Gecko cosine vs hybrid RRF "
                 "(dotted=chance, dashed=viability bar)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "fig_hybrid_gate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Hybrid (alpha={args.alpha}, k={args.k}) Stage-1 gate on RRF score:")
    print(f"  chunk AUC (grade>=3)        : {hybrid['chunk_auc_grade3']}  "
          f"(gecko cosine was 0.572)")
    print(f"  within-bundle concordance   : {hybrid['within_bundle_concordance']}  "
          f"(gecko cosine was 0.623)")
    print(f"  bundle any-relevant AUC     : {hybrid['bundle_any_relevant_auc_top1']}  "
          f"(gecko cosine was 0.574)")
    print(f"Written: {results_dir}/hybrid_gate.json")


if __name__ == "__main__":
    main()
