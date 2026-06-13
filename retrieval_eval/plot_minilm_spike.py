#!/usr/bin/env python3
"""Assemble the MiniLM-L6 spike figure from the committed JSON results.

Left: rerank quality (P@3/HR@3, lenient+strict) for floor / feature-LTR /
MiniLM-L6 / oracle. Right: the Stage-1 score-quality gate across every score
measured in R1-R2 — gecko, bm25, hybrid RRF, feature-LTR, MiniLM-L6 CE — vs the
0.80 viability bar, the through-line of the whole thresholdability arc.

Usage:
  python -m retrieval_eval.plot_minilm_spike \\
      --report-dir   configs/config-v0.2.0/reports/r2c-rerank \\
      --hybrid-gate  configs/config-v0.2.0/results/retrieval_eval/r2-hybrid/hybrid_gate.json
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", default="configs/config-v0.2.0/reports/r2c-rerank")
    ap.add_argument("--hybrid-gate",
                    default="configs/config-v0.2.0/results/retrieval_eval/r2-hybrid/hybrid_gate.json")
    ap.add_argument("--split-dir",
                    default="configs/config-v0.2.0/results/retrieval_eval/r2c-rerank")
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    args = ap.parse_args()

    rd = Path(args.report_dir)
    spike = json.load(open(rd / "minilm_spike.json"))
    ltr = json.load(open(rd / "ltr_results.json"))
    hg = json.load(open(args.hybrid_gate))

    # voyage (API-only retriever) top-3 P@3/HR@3 on the SAME test queries — a
    # retrieval ceiling reference for the quality panel. Not a reranker; shown
    # because oracle (perfect rerank of our pool) and voyage are different
    # ceilings worth seeing together.
    voyage_quality = None
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        split_of = json.load(open(Path(args.split_dir) / "split.json"))
        test_qs = {q for q, s in split_of.items() if s == "test"}
        rk = pd.read_parquet(hf_hub_download(args.hf_repo, "data/rankings.parquet",
                                             repo_type="dataset", revision=args.revision))
        vo = rk[(rk["retriever"] == "voyage") & (rk["rank"] <= 3) & (rk["query_id"].isin(test_qs))]
        jd = pd.read_parquet(hf_hub_download(args.hf_repo, "data/judgments.parquet",
                                             repo_type="dataset", revision=args.revision))
        grades = {(r.query_id, r.chunk_id): int(r.score) for r in jd.itertuples()}
        voyage_quality = {}
        for ck, cut in (("lenient_ge3", 3), ("strict_ge5", 5)):
            p = hr = n = 0
            for qid, grp in vo.groupby("query_id"):
                rel = sum(1 for c in grp["chunk_id"] if grades.get((qid, c), 0) >= cut)
                p += min(rel, 3) / 3.0; hr += int(rel > 0); n += 1
            voyage_quality[ck] = {"p_at_3": p / n, "hr_at_3": hr / n}
    except Exception as e:
        print(f"voyage quality skipped: {e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axq, axg) = plt.subplots(1, 2, figsize=(15, 5),
                                   gridspec_kw={"width_ratios": [1.1, 1.2]})

    # --- quality panel ---
    panels = [("lenient_ge3", "p_at_3", "P@3 ≥3"), ("lenient_ge3", "hr_at_3", "HR@3 ≥3"),
              ("strict_ge5", "p_at_3", "P@3 ≥5"), ("strict_ge5", "hr_at_3", "HR@3 ≥5")]
    qx = range(len(panels)); w = 0.16
    sq = spike["quality"]["by_cut"]
    lq = ltr["results"]["by_cut"]
    series = [
        ("floor (RRF)", "#92510a", lambda ck, st: sq[ck]["floor_rrf"][st]),
        ("feature-LTR", "#888888", lambda ck, st: lq[ck]["feature_ltr"][st]),
        ("MiniLM-L6 fp32", "#4a6fa5", lambda ck, st: sq[ck]["minilm_ce"][st]),
        ("MiniLM-L6 int8", "#9ecae1", lambda ck, st: sq[ck]["minilm_ce_int8"][st]),
        ("oracle", "#14532d", lambda ck, st: sq[ck]["oracle"][st]),
    ]
    for j, (name, color, fn) in enumerate(series):
        vals = [fn(ck, st) for ck, st, _ in panels]
        offs = (j - 2) * w
        axq.bar([i + offs for i in qx], vals, w, label=name, color=color, alpha=0.85)
        for i, v in enumerate(vals):
            axq.text(i + offs, v + 0.008, f"{v:.2f}", ha="center", fontsize=5.5)
    if voyage_quality:
        for i, (ck, st, _) in enumerate(panels):
            v = voyage_quality[ck][st]
            axq.hlines(v, i - 0.42, i + 0.42, color="#6b21a8", ls=":", lw=2,
                       label="voyage retrieval (ceiling)" if i == 0 else None)
    axq.set_xticks(list(qx)); axq.set_xticklabels([t for _, _, t in panels])
    axq.set_ylim(0, 1.0); axq.set_ylabel("score")
    axq.set_title("Rerank quality (held-out test): MiniLM-L6 zero-shot vs "
                  "feature-LTR / floor / oracle", fontsize=9)
    axq.legend(fontsize=7, ncol=2)

    # --- gate panel ---
    gstats = [("chunk_auc_grade3", "chunk AUC\n(>=3)"),
              ("within_bundle_concordance", "concordance"),
              ("bundle_any_relevant_auc_top1", "bundle AUC")]
    gseries = [
        ("Gecko", "#b3372f", hg["gecko_cosine"]),
        ("BM25", "#92510a", hg["bm25"]),
        ("Hybrid RRF", "#888888", hg["hybrid_rrf"]),
        ("feature-LTR", "#c08a3e", ltr["results"]["stage1_gate_on_ltr_score"]),
        ("MiniLM-L6 fp32", "#4a6fa5", spike["quality"]["stage1_gate_on_ce_score"]),
        ("MiniLM-L6 int8", "#9ecae1", spike["quality"]["stage1_gate_on_ce_int8_score"]),
    ]
    gx = range(len(gstats)); bw = 0.13
    for j, (name, color, vals) in enumerate(gseries):
        offs = (j - 2.5) * bw
        ys = [vals[k] for k, _ in gstats]
        axg.bar([i + offs for i in gx], ys, bw, label=name, color=color, alpha=0.85)
        for i, v in enumerate(ys):
            axg.text(i + offs, v + 0.006, f"{v:.2f}", ha="center", fontsize=5)
    voy = hg.get("voyage_cosine_ceiling")
    if voy:
        for i, (k, _) in enumerate(gstats):
            axg.hlines(voy[k], i - 0.42, i + 0.42, color="#6b21a8", ls=":", lw=2,
                       label="voyage (ceiling)" if i == 0 else None)
    axg.axhline(0.5, ls=":", color="gray", lw=1)
    axg.axhline(0.8, ls="--", color="#14532d", lw=1, label="viability bar")
    axg.set_xticks(list(gx)); axg.set_xticklabels([t for _, t in gstats], fontsize=8)
    axg.set_ylim(0.4, 1.0); axg.set_ylabel("statistic")
    axg.set_title("Stage-1 score-quality gate: full progression to MiniLM-L6", fontsize=9)
    axg.legend(fontsize=6.5, ncol=2)

    fig.suptitle("R2c Phase-0 — MiniLM-L6 cross-encoder: quality + thresholdability",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(rd / "fig_minilm_spike.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {rd}/fig_minilm_spike.png")


if __name__ == "__main__":
    main()
