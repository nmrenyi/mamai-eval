#!/usr/bin/env python3
"""R2a — offline hybrid (Gecko + BM25) retrieval simulation via RRF.

Fuses Gecko's and BM25's published audit top-20 rankings with weighted
reciprocal-rank fusion, takes the fused top-3, and scores it against the
mamaretrieval 0-6 judgments. Sweeps the RRF constant k and the Gecko/BM25
weight alpha. Everything is offline arithmetic on rankings.parquet + the
judgments — no embedding, no GPU, no new judging (both retrievers already
ran in the audit; 100% of each one's top-20 is judged).

Weighted RRF score of chunk c for a query:
    score(c) = alpha * [c in gecko]/(k + rank_gecko(c))
             + (1-alpha) * [c in bm25]/(k + rank_bm25(c))
alpha=1 -> Gecko only, alpha=0 -> BM25 only. Ranks are 1-based.

Metrics (per query, averaged), at lenient (grade>=3) and strict (grade>=5):
    P@3  = (# relevant in fused top-3) / 3       -- precision of the bundle
    HR@3 = 1 if >=1 relevant in fused top-3      -- hit rate
Baselines for context: Gecko-alone, BM25-alone, voyage-4-large (API ceiling).

Usage (from repo root):
  python -m retrieval_eval.simulate_hybrid \\
      --results-dir configs/config-v0.2.0/results/retrieval_eval/r2-hybrid \\
      --report-dir  configs/config-v0.2.0/reports/r2-hybrid
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_HF_REPO = "nmrenyi/mamaretrieval"
DEFAULT_REVISION = "v0.2.0"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0 = BM25 only, 1 = Gecko only
KS = [10, 30, 60, 100]


def load_rank_maps(hf_repo: str, revision: str):
    """Return {retriever: {query_id: {chunk_id: rank}}} and grades dict."""
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    rk = pd.read_parquet(hf_hub_download(
        hf_repo, "data/rankings.parquet", repo_type="dataset", revision=revision))
    rank_maps: dict[str, dict[str, dict[str, int]]] = {}
    for retriever in ("gecko", "bm25", "voyage"):
        sub = rk[rk["retriever"] == retriever]
        m: dict[str, dict[str, int]] = {}
        for row in sub.itertuples():
            m.setdefault(row.query_id, {})[row.chunk_id] = int(row.rank)
        rank_maps[retriever] = m

    judgments = load_dataset(hf_repo, "judgments", revision=revision, split="test")
    grades = {(r["query_id"], r["chunk_id"]): int(r["score"]) for r in judgments}
    return rank_maps, grades


def score_topk(top_chunks: list[str], qid: str, grades: dict, cut: int) -> tuple[float, int]:
    """Return (relevant_count / 3, hit) for a query's chosen chunks at a grade cut."""
    rels = [grades.get((qid, c), 0) >= cut for c in top_chunks]
    return sum(rels) / 3.0, int(any(rels))


def eval_ranking(rank_map: dict, grades: dict, cut: int) -> tuple[float, float]:
    """P@3, HR@3 for a single retriever's own ranking at a grade cut."""
    p, hr, n = 0.0, 0, 0
    for qid, ranks in rank_map.items():
        top3 = [c for c, _ in sorted(ranks.items(), key=lambda x: x[1])[:3]]
        pq, hq = score_topk(top3, qid, grades, cut)
        p += pq; hr += hq; n += 1
    return round(p / n, 4), round(hr / n, 4)


def eval_hybrid(gecko: dict, bm25: dict, grades: dict, alpha: float, k: int,
                cut: int) -> tuple[float, float]:
    p, hr, n = 0.0, 0, 0
    for qid in gecko:
        g, b = gecko[qid], bm25.get(qid, {})
        cands = set(g) | set(b)
        scores = {
            c: alpha * (1.0 / (k + g[c]) if c in g else 0.0)
               + (1 - alpha) * (1.0 / (k + b[c]) if c in b else 0.0)
            for c in cands
        }
        top3 = [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        pq, hq = score_topk(top3, qid, grades, cut)
        p += pq; hr += hq; n += 1
    return round(p / n, 4), round(hr / n, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    rank_maps, grades = load_rank_maps(args.hf_repo, args.revision)
    gecko, bm25 = rank_maps["gecko"], rank_maps["bm25"]
    n_queries = len(gecko)
    print(f"{n_queries} queries; {len(grades):,} judged pairs", flush=True)

    cuts = {"lenient_ge3": 3, "strict_ge5": 5}
    baselines = {
        name: {ck: dict(zip(("p_at_3", "hr_at_3"), eval_ranking(rank_maps[name], grades, cv)))
               for ck, cv in cuts.items()}
        for name in ("gecko", "bm25", "voyage")
    }
    for name, b in baselines.items():
        print(f"baseline {name:7s}  P@3(≥3)={b['lenient_ge3']['p_at_3']:.3f}  "
              f"HR@3(≥3)={b['lenient_ge3']['hr_at_3']:.3f}  "
              f"P@3(≥5)={b['strict_ge5']['p_at_3']:.3f}", flush=True)

    grid = []  # one record per (alpha, k)
    for alpha in ALPHAS:
        for k in KS:
            rec = {"alpha": alpha, "k": k}
            for ck, cv in cuts.items():
                p, hr = eval_hybrid(gecko, bm25, grades, alpha, k, cv)
                rec[f"{ck}_p_at_3"] = p
                rec[f"{ck}_hr_at_3"] = hr
            grid.append(rec)
        print(f"  swept alpha={alpha}", flush=True)

    # Best config by lenient P@3, with strict P@3 alongside.
    best = max(grid, key=lambda r: r["lenient_ge3_p_at_3"])
    gecko_p = baselines["gecko"]["lenient_ge3"]["p_at_3"]
    voyage_p = baselines["voyage"]["lenient_ge3"]["p_at_3"]
    gap_closed = ((best["lenient_ge3_p_at_3"] - gecko_p) / (voyage_p - gecko_p)
                  if voyage_p > gecko_p else None)

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision,
                   "rankings": "data/rankings.parquet (gecko + bm25 fused; voyage ref)"},
        "method": "weighted RRF over gecko+bm25 top-20; fused top-3 scored vs judgments",
        "n_queries": n_queries,
        "sweep": {"alphas": ALPHAS, "ks": KS,
                  "alpha_note": "1=gecko only, 0=bm25 only"},
        "baselines": baselines,
        "grid": grid,
        "best_by_lenient_p_at_3": best,
        "best_vs_baselines": {
            "gecko_p_at_3": gecko_p, "voyage_p_at_3": voyage_p,
            "best_p_at_3": best["lenient_ge3_p_at_3"],
            "fraction_of_gecko_to_voyage_gap_closed": (
                round(gap_closed, 4) if gap_closed is not None else None)},
    }
    with open(results_dir / "hybrid_sweep.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure: 2x2 metric panels; x=alpha, one line per k; baseline hlines.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("lenient_ge3_p_at_3", "P@3 (grade>=3)"),
              ("lenient_ge3_hr_at_3", "HR@3 (grade>=3)"),
              ("strict_ge5_p_at_3", "P@3 (grade>=5)"),
              ("strict_ge5_hr_at_3", "HR@3 (grade>=5)")]
    base_key = {"lenient_ge3_p_at_3": ("lenient_ge3", "p_at_3"),
                "lenient_ge3_hr_at_3": ("lenient_ge3", "hr_at_3"),
                "strict_ge5_p_at_3": ("strict_ge5", "p_at_3"),
                "strict_ge5_hr_at_3": ("strict_ge5", "hr_at_3")}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (mkey, title) in zip(axes.flat, panels):
        for k in KS:
            ys = [r[mkey] for r in grid if r["k"] == k]
            ax.plot(ALPHAS, ys, "o-", label=f"k={k}", alpha=0.85)
        ck, stat = base_key[mkey]
        for name, color in (("gecko", "#b3372f"), ("voyage", "#14532d"),
                            ("bm25", "#92510a")):
            ax.axhline(baselines[name][ck][stat], ls="--", color=color, lw=1,
                       label=f"{name} alone")
        ax.set_xlabel("alpha  (0 = BM25 only … 1 = Gecko only)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("R2a — hybrid Gecko+BM25 (RRF) vs alpha and k, scored on "
                 "mamaretrieval audit (n=3,185)", y=1.0)
    fig.tight_layout()
    fig.savefig(report_dir / "fig_hybrid_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nBest by P@3(≥3): alpha={best['alpha']}, k={best['k']} -> "
          f"P@3={best['lenient_ge3_p_at_3']:.3f} "
          f"(gecko {gecko_p:.3f}, voyage {voyage_p:.3f}; "
          f"gap closed {gap_closed:.0%})" if gap_closed is not None else "")
    print(f"Written: {results_dir}/hybrid_sweep.json + {report_dir}/fig_hybrid_sweep.png")


if __name__ == "__main__":
    main()
