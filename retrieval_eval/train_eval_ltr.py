#!/usr/bin/env python3
"""Train + evaluate the feature-LTR reranker (LightGBM LambdaMART).

Trains on the frozen train split, early-stops on dev (nDCG@3), and reports on
the held-out TEST split only. Places feature-LTR on the curve we already have —
hybrid floor (RRF order) -> feature-LTR -> oracle (reorder by grade) — at
lenient (grade>=3) and strict (grade>=5) cuts, runs the R1 Stage-1 gate on the
LTR score (threshold-revival test), and reports feature importances (is the
signal from lexical overlap, or just a learned RRF?).

Deterministic: fixed seed, no bagging/feature subsampling, so numbers reproduce.

Usage:
  python -m retrieval_eval.train_eval_ltr \\
      --features-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \\
      --report-dir   configs/config-v0.2.0/reports/r2c-rerank
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_eval.compare_retriever_gates import gate_stats

FEATURES = ["gecko_score", "gecko_rank", "bm25_score", "bm25_rank",
            "rrf_score", "rrf_rank", "in_both", "q_len", "c_len",
            "overlap_count", "jaccard", "q_coverage", "num_overlap"]


def rerank_metrics(df, score_col: str, cut: int) -> tuple[float, float]:
    """P@3, HR@3 after sorting each query's candidates by score_col desc."""
    p, hr, n = 0.0, 0, 0
    for _, grp in df.groupby("query_id"):
        g = grp.sort_values([score_col, "chunk_id"], ascending=[False, True])
        top3 = g.head(3)
        rel = int((top3["grade"] >= cut).sum())
        p += min(rel, 3) / 3.0; hr += int(rel > 0); n += 1
    return round(p / n, 4), round(hr / n, 4)


def gate_on_score(df, score_col: str) -> dict:
    """R1 Stage-1 gate on score_col: rank each query's candidates by it, feed
    the gate (which scores the top-3 population)."""
    rows = []
    for _, grp in df.groupby("query_id"):
        g = grp.sort_values([score_col, "chunk_id"], ascending=[False, True])
        for rank, r in enumerate(g.itertuples(), 1):
            rows.append({"query_id": r.query_id, "rank": rank,
                         "score": getattr(r, score_col), "grade": int(r.grade)})
    return gate_stats(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import lightgbm as lgb
    import pandas as pd

    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    feats_dir = Path(args.features_dir)
    df = pd.read_parquet(feats_dir / "ltr_features.parquet")

    def split_df(s):
        return df[df["split"] == s].sort_values("query_id").reset_index(drop=True)
    tr, dv, te = split_df("train"), split_df("dev"), split_df("test")

    def groups(d):
        return d.groupby("query_id", sort=False).size().to_numpy()

    train_set = lgb.Dataset(tr[FEATURES], label=tr["grade"], group=groups(tr))
    dev_set = lgb.Dataset(dv[FEATURES], label=dv["grade"], group=groups(dv),
                          reference=train_set)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3],
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 50,
        "feature_fraction": 1.0, "bagging_fraction": 1.0, "seed": args.seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1,
    }
    model = lgb.train(params, train_set, num_boost_round=500,
                      valid_sets=[dev_set], valid_names=["dev"],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(0)])

    te = te.copy()
    te["ltr_score"] = model.predict(te[FEATURES], num_iteration=model.best_iteration)

    cuts = {"lenient_ge3": 3, "strict_ge5": 5}
    # Floor = RRF order (the deployed hybrid); oracle = reorder by grade.
    results = {"n_test_queries": int(te["query_id"].nunique()),
               "best_iteration": int(model.best_iteration), "by_cut": {}}
    for ck, cv in cuts.items():
        fp, fhr = rerank_metrics(te, "rrf_score", cv)
        lp, lhr = rerank_metrics(te, "ltr_score", cv)
        op, ohr = rerank_metrics(te, "grade", cv)
        results["by_cut"][ck] = {
            "floor_rrf": {"p_at_3": fp, "hr_at_3": fhr},
            "feature_ltr": {"p_at_3": lp, "hr_at_3": lhr},
            "oracle": {"p_at_3": op, "hr_at_3": ohr},
        }

    gate = gate_on_score(te, "ltr_score")
    results["stage1_gate_on_ltr_score"] = {
        "chunk_auc_grade3": gate["chunk_auc_grade3"],
        "within_bundle_concordance": gate["within_bundle_concordance"],
        "bundle_any_relevant_auc_top1": gate["bundle_any_relevant_auc_top1"],
    }
    importances = sorted(
        zip(FEATURES, model.feature_importance(importance_type="gain")),
        key=lambda x: -x[1])
    total = sum(v for _, v in importances) or 1
    results["feature_importance_gain"] = [
        {"feature": f, "gain_frac": round(v / total, 4)} for f, v in importances]

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "LightGBM LambdaMART (lambdarank, ndcg@3 early-stop)",
        "params": params, "features": FEATURES, "results": results,
    }
    with open(report_dir / "ltr_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure: metrics (floor/LTR/oracle) + feature importances.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axm, axi) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    panels = [("lenient_ge3", "p_at_3", "P@3 ≥3"), ("lenient_ge3", "hr_at_3", "HR@3 ≥3"),
              ("strict_ge5", "p_at_3", "P@3 ≥5"), ("strict_ge5", "hr_at_3", "HR@3 ≥5")]
    x = range(len(panels)); w = 0.26
    series = [("floor (RRF)", "#92510a", "floor_rrf"),
              ("feature-LTR", "#4a6fa5", "feature_ltr"),
              ("oracle", "#14532d", "oracle")]
    for j, (name, color, key) in enumerate(series):
        vals = [results["by_cut"][ck][key][stat] for ck, stat, _ in panels]
        offs = (j - 1) * w
        axm.bar([i + offs for i in x], vals, w, label=name, color=color, alpha=0.85)
        for i, v in enumerate(vals):
            axm.text(i + offs, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    axm.set_xticks(list(x)); axm.set_xticklabels([t for _, _, t in panels])
    axm.set_ylim(0, 1.0); axm.set_ylabel("score")
    axm.set_title("feature-LTR vs hybrid floor vs oracle (held-out test, "
                  f"n={results['n_test_queries']})", fontsize=10)
    axm.legend(fontsize=8)

    imp = results["feature_importance_gain"][::-1]
    axi.barh([d["feature"] for d in imp], [d["gain_frac"] for d in imp],
             color="#4a6fa5", alpha=0.85)
    axi.set_xlabel("gain importance (fraction)")
    axi.set_title("feature importance", fontsize=10)
    fig.suptitle("R2c-A — feature-LTR (LightGBM LambdaMART) reranking the hybrid "
                 "top-20", y=1.02)
    fig.tight_layout()
    fig.savefig(report_dir / "fig_ltr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"test queries: {results['n_test_queries']}, best_iter {model.best_iteration}")
    for ck in cuts:
        c = results["by_cut"][ck]
        print(f"{ck}: floor P@3={c['floor_rrf']['p_at_3']} -> "
              f"LTR {c['feature_ltr']['p_at_3']} -> oracle {c['oracle']['p_at_3']} "
              f"| HR floor {c['floor_rrf']['hr_at_3']} -> LTR {c['feature_ltr']['hr_at_3']}")
    print("Stage-1 gate on LTR score:", results["stage1_gate_on_ltr_score"])
    print("top features:", [(f, round(v / total, 3)) for f, v in importances[:5]])
    print(f"Written: {report_dir}/ltr_results.json + fig_ltr.png")


if __name__ == "__main__":
    main()
