#!/usr/bin/env python3
"""Stage-1 score-quality gate, run for every audit retriever.

Applies the Figure-1 gate of docs/r1-threshold-tuning-plan.md to all six
retrievers in the mamaretrieval audit (voyage-4-large, octen-8B, lateon,
medcpt, bm25, gecko), using their own ranking scores from rankings.parquet
against the shared 0-6 judgments. Per retriever, on its top-3 population:

  - chunk-level AUC (grade >= 3 and >= 5)  — ability needed by absolute cutoffs
  - within-bundle concordance              — ability needed by relative rules
  - bundle any-relevant AUC (top-1 score)  — ability needed by gated abstention
  - P@3 / HR@3 base rates                  — context

AUC and concordance are rank-based, so the retrievers' incomparable score
scales (BM25 ~29 vs cosine ~0.7) do not matter. Only Stage 1 is computable
here: no end-to-end runs exist with these retrievers' contexts, so the
Figure-2 outcome gate does not apply.

Usage:
  python -m retrieval_eval.compare_retriever_gates \\
      --report-dir configs/config-v0.2.0/reports/r1-threshold
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_HF_REPO = "nmrenyi/mamaretrieval"
DEFAULT_REVISION = "v0.2.0"
RETRIEVER_NOTES = {
    "voyage": "API-only — reference ceiling, not deployable",
    "octen": "cluster-only (8B) — not deployable on-device",
    "lateon": "audit comparator",
    "medcpt": "audit comparator",
    "bm25": "lexical — on-device feasible via SQLite FTS5 (R2a)",
    "gecko": "deployed on-device retriever",
}


def gate_stats(rows: list[dict]) -> dict:
    """Stage-1 statistics for one retriever. rows: top-20 ranking rows joined
    with grades: {query_id, rank, score, grade}."""
    from sklearn.metrics import roc_auc_score

    top3 = [r for r in rows if r["rank"] <= 3]
    cos = np.array([r["score"] for r in top3])
    rel3 = np.array([r["grade"] >= 3 for r in top3])
    rel5 = np.array([r["grade"] >= 5 for r in top3])

    by_q: dict[str, list[dict]] = {}
    for r in top3:
        by_q.setdefault(r["query_id"], []).append(r)

    conc, tot = 0, 0
    b_top1, b_hasrel = [], []
    hr3 = 0
    for ch in by_q.values():
        rel_ch = [c for c in ch if c["grade"] >= 3]
        junk_ch = [c for c in ch if c["grade"] < 3]
        for a in rel_ch:
            for b in junk_ch:
                tot += 1
                conc += a["score"] > b["score"]
        b_top1.append(max(c["score"] for c in ch))
        b_hasrel.append(bool(rel_ch))
        hr3 += bool(rel_ch)

    return {
        "n_top3_pairs": len(top3),
        "n_queries": len(by_q),
        "p_at_3": round(float(rel3.mean()), 4),
        "hr_at_3": round(hr3 / len(by_q), 4),
        "chunk_auc_grade3": round(float(roc_auc_score(rel3, cos)), 4),
        "chunk_auc_grade5": round(float(roc_auc_score(rel5, cos)), 4),
        "within_bundle_concordance": round(conc / tot, 4) if tot else None,
        "concordance_n_pairs": tot,
        "bundle_any_relevant_auc_top1": round(
            float(roc_auc_score(b_hasrel, b_top1)), 4)
        if 0 < sum(b_hasrel) < len(b_hasrel) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rk = pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/rankings.parquet", repo_type="dataset",
        revision=args.revision))
    judgments = load_dataset(args.hf_repo, "judgments",
                             revision=args.revision, split="test")
    grade = {(r["query_id"], r["chunk_id"]): r["score"] for r in judgments}

    results = {}
    skipped = {}
    for retriever in sorted(rk["retriever"].unique()):
        sub = rk[rk["retriever"] == retriever]
        if sub["score"].isna().all():
            skipped[retriever] = ("rank-only in rankings.parquet (no scores "
                                  "published) — score gate not computable")
            print(f"{retriever:8s}  SKIPPED: rank-only, no scores", flush=True)
            continue
        rows, missing = [], 0
        for r in sub.itertuples():
            g = grade.get((r.query_id, r.chunk_id))
            if g is None:
                missing += 1
                continue
            rows.append({"query_id": r.query_id, "rank": int(r.rank),
                         "score": float(r.score), "grade": int(g)})
        stats = gate_stats(rows)
        stats["n_unjudged_top20"] = missing
        stats["note"] = RETRIEVER_NOTES.get(retriever, "")
        results[retriever] = stats
        print(f"{retriever:8s}  AUC3={stats['chunk_auc_grade3']:.3f}  "
              f"conc={stats['within_bundle_concordance']:.3f}  "
              f"bundleAUC={stats['bundle_any_relevant_auc_top1']}  "
              f"P@3={stats['p_at_3']:.3f}  HR@3={stats['hr_at_3']:.3f}",
              flush=True)

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"hf_repo": args.hf_repo, "revision": args.revision},
        "gate": "Stage 1 (label gate) only — no end-to-end runs exist for "
                "non-deployed retrievers, so Stage 2 (outcome gate) is not "
                "computable here",
        "retrievers": results,
        "skipped": skipped,
    }
    with open(report_dir / "retriever_gate_comparison.json", "w") as f:
        json.dump(out, f, indent=2)

    # Comparison figure: the three gate statistics per retriever.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(results, key=lambda n: -results[n]["chunk_auc_grade3"])
    stats_to_plot = [
        ("chunk_auc_grade3", "chunk-level AUC (grade>=3)\nabsolute cutoffs"),
        ("within_bundle_concordance", "within-bundle concordance\nrelative rules"),
        ("bundle_any_relevant_auc_top1", "bundle any-relevant AUC\ngated abstention"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, stats_to_plot):
        vals = [results[n][key] for n in names]
        colors = ["#b3372f" if n == "gecko" else "#4a6fa5" for n in names]
        ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.axvline(0.5, ls=":", color="gray", lw=1)
        ax.axvline(0.8, ls="--", color="#14532d", lw=1)
        ax.set_xlim(0.4, 1.0)
        ax.set_title(title, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle("Stage-1 score-quality gate across audit retrievers "
                 "(top-3 population; dotted = chance, dashed = viability bar)",
                 y=1.04)
    fig.tight_layout()
    fig.savefig(report_dir / "fig_retriever_gate_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nWritten: {report_dir}/retriever_gate_comparison.json + "
          f"fig_retriever_gate_comparison.png")


if __name__ == "__main__":
    main()
