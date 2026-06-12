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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    rows3 = pops["top3"]
    cos3 = np.array([r["cosine"] for r in rows3])
    rel3 = np.array([r["grade"] >= 3 for r in rows3])
    bins = np.linspace(min(cos3), max(cos3), 60)
    axes[0].hist(cos3[rel3], bins=bins, density=True, alpha=0.55,
                 label=f"relevant (grade>=3), n={rel3.sum():,}", color="#2c7a3f")
    axes[0].hist(cos3[~rel3], bins=bins, density=True, alpha=0.55,
                 label=f"junk (grade<3), n={(~rel3).sum():,}", color="#b3372f")
    axes[0].set_xlabel("Gecko cosine score")
    axes[0].set_ylabel("density")
    axes[0].set_title("Gecko top-3 chunks: score by relevance")
    axes[0].legend(fontsize=8)

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
    fig.suptitle("Figure 1 — does Gecko's cosine separate relevant from junk? "
                 "(Table A: mamaretrieval grades)", y=1.02)
    fig.tight_layout()
    fig.savefig(report_dir / "fig1_score_separation.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return metrics


def fig2(table_b: list[dict], report_dir: Path) -> dict:
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"hurt": "#b3372f", "helped": "#2c7a3f", "unchanged": "#888888"}
    for ax, fn, title in ((axes[0], top1, "top-1 cosine"),
                          (axes[1], mean3, "mean cosine of injected bundle")):
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
        out["fig2"] = fig2(table_b, report_dir)
        print("Figure 2 metrics:", json.dumps(out["fig2"], indent=2))

    with open(report_dir / "threshold_signal_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFigures + metrics written to {report_dir}/")


if __name__ == "__main__":
    main()
