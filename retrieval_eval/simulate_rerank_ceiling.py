#!/usr/bin/env python3
"""R2b — oracle rerank-ceiling simulation.

A reranker can only reorder a retriever's top-k pool, never add new chunks. So
the maximum any reranker can achieve is the *oracle*: reorder the pool by the
true judge grade and take the top-3. This computes that ceiling for the
deployable pools (Gecko, the R2a hybrid, BM25), against the actual top-3 floor,
as a function of rerank depth. Pure offline arithmetic on the mamaretrieval
audit — no model, no GPU, no new judging.

The oracle uses the same 0-6 grades that define P@3/HR@3, so it is tautologically
optimal by construction — that IS the definition of the ceiling, not leakage. A
real reranker captures only a fraction of it; R2b decides whether scoping that
(non-free) follow-up is worth it.

Metrics per pool, per rerank depth d (rerank the top-d, keep top-3):
    P@3  = (# relevant in reranked top-3) / 3      at grade>=3 and grade>=5
    HR@3 = 1 if >=1 relevant in reranked top-3
Floor = the retriever's own (non-reranked) top-3. Oracle HR@3 at depth d equals
the pool's HR@d (a perfect reranker surfaces any relevant chunk in the top-d).

Usage:
  python -m retrieval_eval.simulate_rerank_ceiling \\
      --results-dir configs/config-v0.2.0/results/retrieval_eval/r2-rerank \\
      --report-dir  configs/config-v0.2.0/reports/r2-rerank
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_eval.gate_hybrid import (load_rankings_and_grades, rank_map,
                                        build_hybrid_rows)

DEPTHS = [3, 5, 10, 20]
ALPHA, K = 0.5, 60   # deployable hybrid config from R2a


def pool_from_ranking(rank_map_one: dict, grades: dict) -> dict:
    """{query_id: [(grade, chunk_id), ... ] ordered by the retriever's rank}."""
    pool = {}
    for qid, ranks in rank_map_one.items():
        ordered = sorted(ranks.items(), key=lambda x: x[1])  # by rank asc
        pool[qid] = [(grades.get((qid, cid), 0), cid) for cid, _ in ordered]
    return pool


def hybrid_pool(rk, grades: dict) -> dict:
    """Fused top-k pool as {query_id: [(grade, chunk_id), ...]} in RRF order."""
    rows = build_hybrid_rows(rank_map(rk, "gecko"), rank_map(rk, "bm25"),
                             grades, ALPHA, K)
    by_q: dict[str, list] = {}
    for r in rows:  # build_hybrid_rows already emits rows in fused-rank order
        by_q.setdefault(r["query_id"], []).append((r["grade"], r["rank"], None))
    # rows carry rank; reconstruct grade list ordered by fused rank
    pool = {}
    for qid in by_q:
        pool[qid] = [(g, None) for g, _rk, _ in sorted(by_q[qid], key=lambda x: x[1])]
    return pool


def floor_p_hr(pool: dict, cut: int) -> tuple[float, float]:
    """Actual top-3 (no rerank): P@3, HR@3 at a grade cut."""
    p, hr, n = 0.0, 0, 0
    for chunks in pool.values():
        top3 = chunks[:3]
        rel = sum(1 for g, _ in top3 if g >= cut)
        p += min(rel, 3) / 3.0; hr += int(rel > 0); n += 1
    return round(p / n, 4), round(hr / n, 4)


def oracle_p_hr(pool: dict, depth: int, cut: int) -> tuple[float, float]:
    """Oracle: reorder top-`depth` by grade, keep top-3. P@3, HR@3 at a cut."""
    p, hr, n = 0.0, 0, 0
    for chunks in pool.values():
        window = chunks[:depth]
        rel_in_window = sum(1 for g, _ in window if g >= cut)
        # perfect rerank puts the highest-grade chunks first; top-3 captures
        # min(rel_in_window, 3) relevant.
        p += min(rel_in_window, 3) / 3.0; hr += int(rel_in_window > 0); n += 1
    return round(p / n, 4), round(hr / n, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    rk, grades = load_rankings_and_grades(args.hf_repo, args.revision)
    pools = {
        "gecko": pool_from_ranking(rank_map(rk, "gecko"), grades),
        "bm25": pool_from_ranking(rank_map(rk, "bm25"), grades),
        f"hybrid (α={ALPHA}, k={K})": hybrid_pool(rk, grades),
    }
    cuts = {"lenient_ge3": 3, "strict_ge5": 5}

    out_pools = {}
    for name, pool in pools.items():
        rec = {"n_queries": len(pool), "floor": {}, "oracle_by_depth": {}}
        for ck, cv in cuts.items():
            p, hr = floor_p_hr(pool, cv)
            rec["floor"][ck] = {"p_at_3": p, "hr_at_3": hr}
        for d in DEPTHS:
            rec["oracle_by_depth"][d] = {}
            for ck, cv in cuts.items():
                p, hr = oracle_p_hr(pool, d, cv)
                rec["oracle_by_depth"][d][ck] = {"p_at_3": p, "hr_at_3": hr}
        out_pools[name] = rec
        f3 = rec["floor"]["lenient_ge3"]
        o20 = rec["oracle_by_depth"][20]["lenient_ge3"]
        o20s = rec["oracle_by_depth"][20]["strict_ge5"]
        print(f"{name:22s} floor P@3≥3={f3['p_at_3']:.3f} HR@3={f3['hr_at_3']:.3f} | "
              f"oracle@20 P@3≥3={o20['p_at_3']:.3f} HR@3={o20['hr_at_3']:.3f} | "
              f"oracle@20 strict P@3={o20s['p_at_3']:.3f} HR@3={o20s['hr_at_3']:.3f}",
              flush=True)

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision},
        "method": "oracle rerank: reorder top-d pool by judge grade, keep top-3. "
                  "Upper bound for any reranker; uses the grades that define the "
                  "metric (definition of the ceiling, not leakage).",
        "hybrid_config": {"alpha": ALPHA, "k": K},
        "depths": DEPTHS,
        "pools": out_pools,
    }
    with open(results_dir / "rerank_ceiling.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure: P@3 and HR@3 vs rerank depth, lenient + strict, one line per pool,
    # floor drawn as a dotted marker at depth 3.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gecko": "#b3372f", "bm25": "#92510a",
              f"hybrid (α={ALPHA}, k={K})": "#4a6fa5"}
    panels = [("lenient_ge3", "p_at_3", "P@3 (grade>=3)"),
              ("lenient_ge3", "hr_at_3", "HR@3 (grade>=3)"),
              ("strict_ge5", "p_at_3", "P@3 (grade>=5)"),
              ("strict_ge5", "hr_at_3", "HR@3 (grade>=5)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (ck, stat, title) in zip(axes.flat, panels):
        for name, rec in out_pools.items():
            ys = [rec["oracle_by_depth"][d][ck][stat] for d in DEPTHS]
            ax.plot(DEPTHS, ys, "o-", color=colors[name], label=f"{name} oracle")
            ax.axhline(rec["floor"][ck][stat], ls=":", color=colors[name], lw=1.2,
                       label=f"{name} floor (top-3)")
        ax.set_xlabel("rerank depth (top-d pool reordered)")
        ax.set_ylabel(title); ax.set_title(title)
        ax.set_xticks(DEPTHS)
        ax.legend(fontsize=7)
    fig.suptitle("R2b — oracle rerank ceiling vs depth: max achievable by "
                 "reordering each pool (mamaretrieval, n=3,185)", y=1.0)
    fig.tight_layout()
    fig.savefig(report_dir / "fig_rerank_ceiling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWritten: {results_dir}/rerank_ceiling.json + "
          f"{report_dir}/fig_rerank_ceiling.png")


if __name__ == "__main__":
    main()
