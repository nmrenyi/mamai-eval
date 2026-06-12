#!/usr/bin/env python3
"""Figures 1–2 of the R1 threshold plan: does Gecko's cosine score carry signal?

Figure 1 (from Table A, build_sweep_table.py): cosine distributions for
relevant (grade >= 3) vs junk (grade < 3) chunks, ROC/AUC and PR curves.
Primary population is Gecko's top-3 (the bundle actually injected in
deployment); top-20 is shown as a secondary line.

Figure 2 (from Table B, build_mcq_outcome_table.py): bundle-score
distributions for rows where RAG hurt (right -> wrong), helped
(wrong -> right) or left the outcome unchanged. Tune split only; rows whose
recomputed context failed the parity check are excluded from score stats.

Reading guide (see docs/r1-threshold-tuning-plan.md): Fig 1 AUC >= ~0.80 means
an absolute cutoff is viable; ~0.65-0.80 lean on relative rules; <= ~0.60 stop
and file a negative result. Fig 2 is the end-to-end mirror: if hurt rows do
not sit at lower scores than helped rows, filtering cannot fix the -1.8 pp gap.

Usage:
  python -m retrieval_eval.plot_threshold_signal \\
      --table-dir configs/config-v0.2.0/results/retrieval_eval/r1-threshold \\
      --report-dir configs/config-v0.2.0/reports/r1-threshold
"""

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Continuous-outcome deadband: rubric-score deltas with |delta| below this are
# binned 'neutral' in the violin panel. weighted_met is a sum of ~5-20 graded
# criteria, so 0.05 ≈ one small criterion flipping — below that is judge noise.
# Visual grouping only; the headline Spearman statistic uses no deadband.
NEUTRAL_DEADBAND = 0.05


def load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def roc_pr(scores: np.ndarray, labels: np.ndarray):
    from sklearn.metrics import (auc, average_precision_score,
                                 precision_recall_curve, roc_curve)
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr),
            "prec": prec, "rec": rec,
            "ap": average_precision_score(labels, scores)}


def fig1(table_a: list[dict], report_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {}
    pops = {
        "top3": [r for r in table_a if r["gecko_rank"] <= 3],
        "top20": [r for r in table_a if r["gecko_rank"] <= 20],
    }
    fig, axes2d = plt.subplots(2, 2, figsize=(12, 9))
    axes = [axes2d[0, 0], axes2d[1, 0], axes2d[1, 1]]

    rows3 = pops["top3"]
    cos3 = np.array([r["cosine"] for r in rows3])
    bins = np.linspace(min(cos3), max(cos3), 60)
    for ax, grade_cut in ((axes2d[0, 0], 3), (axes2d[0, 1], 5)):
        rel = np.array([r["grade"] >= grade_cut for r in rows3])
        ax.hist(cos3[rel], bins=bins, density=True, alpha=0.55,
                label=f"relevant (grade>={grade_cut}), n={rel.sum():,}",
                color="#2c7a3f")
        ax.hist(cos3[~rel], bins=bins, density=True, alpha=0.55,
                label=f"junk (grade<{grade_cut}), n={(~rel).sum():,}",
                color="#b3372f")
        ax.set_xlabel("Gecko cosine score")
        ax.set_ylabel("density")
        ax.set_title(f"top-3 chunks: score by relevance (grade>={grade_cut})")
        ax.legend(fontsize=8)

    for pop_name, rows in pops.items():
        cos = np.array([r["cosine"] for r in rows])
        for grade_cut, style in ((3, "-"), (5, "--")):
            rel = np.array([r["grade"] >= grade_cut for r in rows])
            if rel.sum() == 0 or (~rel).sum() == 0:
                continue
            m = roc_pr(cos, rel)
            key = f"{pop_name}_grade{grade_cut}"
            metrics[key] = {
                "n_pairs": len(rows), "n_relevant": int(rel.sum()),
                "base_rate_precision": round(float(rel.mean()), 4),
                "auc": round(float(m["auc"]), 4),
                "average_precision": round(float(m["ap"]), 4),
            }
            axes[1].plot(m["fpr"], m["tpr"], style,
                         label=f"{pop_name}, grade>={grade_cut}: AUC={m['auc']:.3f}")
            axes[2].plot(m["rec"], m["prec"], style,
                         label=f"{pop_name}, grade>={grade_cut}: AP={m['ap']:.3f}")
    axes[1].plot([0, 1], [0, 1], ":", color="gray", label="coin flip")
    axes[1].set_xlabel("false positive rate (junk kept)")
    axes[1].set_ylabel("true positive rate (relevant kept)")
    axes[1].set_title("ROC — cosine as relevance classifier")
    axes[1].legend(fontsize=8)
    axes[2].set_xlabel("recall (relevant kept)")
    axes[2].set_ylabel("precision of kept chunks")
    axes[2].set_title("Precision-recall")
    axes[2].legend(fontsize=8)

    # Supplementary statistics cited in the report (not plotted):
    # (a) within-query concordance — among (relevant, junk) pairs drawn from the
    #     SAME top-3 bundle, how often does the relevant chunk score higher?
    #     This is the signal available to relative rules (margin-from-top-1).
    # (b) bundle-level AUC — does a bundle's top-1/mean score predict whether
    #     the bundle contains ANY relevant chunk? The signal available to
    #     gated-abstention rules.
    from sklearn.metrics import roc_auc_score
    by_q: dict[str, list[dict]] = {}
    for r in pops["top3"]:
        by_q.setdefault(r["query_id"], []).append(r)
    conc, tot = 0, 0
    b_top1, b_mean, b_hasrel = [], [], []
    for ch in by_q.values():
        rel_ch = [c for c in ch if c["grade"] >= 3]
        junk_ch = [c for c in ch if c["grade"] < 3]
        for a in rel_ch:
            for b in junk_ch:
                tot += 1
                conc += a["cosine"] > b["cosine"]
        cs = [c["cosine"] for c in ch]
        b_top1.append(max(cs))
        b_mean.append(float(np.mean(cs)))
        b_hasrel.append(bool(rel_ch))
    metrics["top3_within_query_concordance"] = {
        "value": round(conc / tot, 4), "n_pairs": tot,
        "definition": "share of same-bundle (relevant, junk) pairs where the "
                      "relevant chunk has the higher cosine",
    }
    metrics["bundle_any_relevant_auc"] = {
        "top1": round(float(roc_auc_score(b_hasrel, b_top1)), 4),
        "mean3": round(float(roc_auc_score(b_hasrel, b_mean)), 4),
        "n_bundles": len(b_hasrel),
        "base_rate": round(float(np.mean(b_hasrel)), 4),
    }
    fig.suptitle("Figure 1 — does Gecko's cosine separate relevant from junk? "
                 "(Table A: mamaretrieval grades)", y=1.02)
    fig.tight_layout()
    fig.savefig(report_dir / "fig1_score_separation.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return metrics


def fig2(table_b: list[dict], report_dir: Path,
         table_b2: list[dict] | None = None) -> dict:
    """2x2 when table_b2 (HealthBench) is given: MCQ violins on the top row,
    HealthBench delta-scatter + violins on the bottom row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    tune = [r for r in table_b if r["split"] == "tune"]
    usable = [r for r in tune if r["context_match"] and r["cosines"]]
    groups = {o: [r for r in usable if r["outcome"] == o]
              for o in ("hurt", "helped", "unchanged")}

    def top1(rows):
        return np.array([r["cosines"][0] for r in rows])

    def mean3(rows):
        return np.array([float(np.mean(r["cosines"])) for r in rows])

    if table_b2 is not None:
        fig, axes2d = plt.subplots(2, 2, figsize=(13, 9.5))
        axes = axes2d[0]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"hurt": "#b3372f", "helped": "#2c7a3f", "unchanged": "#888888"}
    for ax, fn, title in ((axes[0], top1, "MCQ: top-1 cosine"),
                          (axes[1], mean3, "MCQ: mean cosine of injected bundle")):
        data = [fn(groups[o]) for o in ("hurt", "helped", "unchanged")]
        parts = ax.violinplot(data, showmedians=True)
        for body, o in zip(parts["bodies"], ("hurt", "helped", "unchanged")):
            body.set_facecolor(colors[o])
            body.set_alpha(0.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{o}\nn={len(groups[o]):,}"
                            for o in ("hurt", "helped", "unchanged")])
        ax.set_ylabel("Gecko cosine")
        ax.set_title(title)

    hb_metrics = None
    if table_b2 is not None:
        hb_metrics = _fig2_healthbench_row(table_b2, axes2d[1], plt)
        fig.suptitle("Figure 2 — bundle scores vs RAG outcome: MCQ accuracy flips "
                     "(top, tune half, parity-matched) and HealthBench oss_eval "
                     "rubric-score deltas (bottom)", y=1.0)
    else:
        fig.suptitle("Figure 2 — bundle scores by RAG outcome on MCQ "
                     "(Table B, tune half, parity-matched rows)", y=1.02)
    fig.tight_layout()
    fig.savefig(report_dir / "fig2_outcome_separation.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "n_tune": len(tune),
        "n_usable": len(usable),
        "n_excluded_parity": len(tune) - len(usable),
        "groups": {o: {
            "n": len(rows),
            "top1_mean": round(float(np.mean(top1(rows))), 4) if rows else None,
            "mean3_mean": round(float(np.mean(mean3(rows))), 4) if rows else None,
        } for o, rows in groups.items()},
    }
    if groups["hurt"] and groups["helped"]:
        h, p = top1(groups["hurt"]), top1(groups["helped"])
        u, pval = mannwhitneyu(h, p, alternative="less")
        # P(random helped row scores higher than random hurt row) — outcome AUC
        metrics["helped_vs_hurt"] = {
            "top1_auc": round(float(1 - u / (len(h) * len(p))), 4),
            "mannwhitney_p_hurt_lower": float(pval),
        }
    if hb_metrics is not None:
        metrics = {"mcq": metrics, "healthbench": hb_metrics}
    return metrics


def _fig2_healthbench_row(table_b2: list[dict], axes, plt) -> dict:
    """Bottom row of Figure 2: continuous weighted_met delta vs bundle score.

    'Unchanged' has no exact analogue for a continuous outcome; the violin
    panel uses a |delta| < 0.05 deadband (see NEUTRAL_DEADBAND) purely for
    visual comparability with the MCQ panel. The primary statistic is the
    deadband-free Spearman correlation in the left panel.
    """
    from scipy.stats import mannwhitneyu, spearmanr

    rows = [r for r in table_b2 if r["cosines"]]
    top1_all = np.array([r["cosines"][0] for r in rows])
    delta = np.array([r["delta"] for r in rows])
    rho, pval = spearmanr(top1_all, delta)

    groups = {"hurt": [r for r in rows if r["delta"] <= -NEUTRAL_DEADBAND],
              "neutral": [r for r in rows if abs(r["delta"]) < NEUTRAL_DEADBAND],
              "helped": [r for r in rows if r["delta"] >= NEUTRAL_DEADBAND]}
    colors = {"hurt": "#b3372f", "helped": "#2c7a3f", "neutral": "#888888"}
    order = ["hurt", "helped", "neutral"]

    def top1(rows_):
        return np.array([r["cosines"][0] for r in rows_])

    def mean3(rows_):
        return np.array([float(np.mean(r["cosines"])) for r in rows_])

    for ax, fn, title in (
            (axes[0], top1, "HealthBench: top-1 cosine"),
            (axes[1], mean3, "HealthBench: mean cosine of injected bundle")):
        data = [fn(groups[o]) for o in order]
        parts = ax.violinplot(data, showmedians=True)
        for body, o in zip(parts["bodies"], order):
            body.set_facecolor(colors[o])
            body.set_alpha(0.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{o}\nn={len(groups[o]):,}" for o in order])
        ax.set_ylabel("Gecko cosine")
        ax.set_title(f"{title} (|delta| >= {NEUTRAL_DEADBAND})")

    metrics = {
        "n_rows": len(rows),
        "spearman_top1_vs_delta": {"rho": round(float(rho), 4), "p": float(pval)},
        "mean_delta": round(float(delta.mean()), 4),
        "groups": {o: {"n": len(groups[o]),
                       "top1_mean": round(float(np.mean([r["cosines"][0] for r in groups[o]])), 4)
                       if groups[o] else None}
                   for o in order},
    }
    if groups["hurt"] and groups["helped"]:
        h = np.array([r["cosines"][0] for r in groups["hurt"]])
        p_ = np.array([r["cosines"][0] for r in groups["helped"]])
        u, pv = mannwhitneyu(h, p_, alternative="less")
        metrics["helped_vs_hurt"] = {
            "top1_auc": round(float(1 - u / (len(h) * len(p_))), 4),
            "mannwhitney_p_hurt_lower": float(pv),
        }
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--figures", default="1,2")
    args = ap.parse_args()

    table_dir = Path(args.table_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = args.figures.split(",")

    out = {"created_at_utc":
           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "table_dir": str(table_dir)}
    if "1" in figures:
        table_a = load_jsonl_gz(table_dir / "table_a.jsonl.gz")
        out["fig1"] = fig1(table_a, report_dir)
        print("Figure 1 metrics:", json.dumps(out["fig1"], indent=2))
    if "2" in figures:
        table_b = load_jsonl_gz(table_dir / "table_b.jsonl.gz")
        b2_path = table_dir / "table_b2_healthbench.jsonl.gz"
        table_b2 = load_jsonl_gz(b2_path) if b2_path.exists() else None
        out["fig2"] = fig2(table_b, report_dir, table_b2=table_b2)
        print("Figure 2 metrics:", json.dumps(out["fig2"], indent=2))

    with open(report_dir / "threshold_signal_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFigures + metrics written to {report_dir}/")


if __name__ == "__main__":
    main()
