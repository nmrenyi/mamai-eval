#!/usr/bin/env python3
"""R2c — reranker comparison figure: lenient P@3 + Stage-1 gate AUC for every
candidate (zero-shot + fine-tuned), against floor / oracle / Qwen3 reference.

Usage:
  python -m retrieval_eval.plot_reranker_comparison \\
      --cand-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank/candidates \\
      --ft-dir   configs/config-v0.2.0/results/retrieval_eval/r2c-rerank/finetune \\
      --out      configs/config-v0.2.0/reports/r2c-rerank/fig_reranker_compare.png
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand-dir", required=True)
    ap.add_argument("--ft-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cd = Path(args.cand_dir)
    fl = json.load(open(cd / "_floor_oracle.json"))["by_cut"]
    floor = fl["lenient_ge3"]["floor_rrf"]["p_at_3"]
    oracle = fl["lenient_ge3"]["oracle"]["p_at_3"]

    # (label, p@3, auc3, kind)  kind in {deploy, deploy_ft, ref}
    items = []
    order = ["medcpt", "minilm-l12", "minilm-l6", "electra-base", "bge-base", "mxbai-base"]
    nice = {"medcpt": "MedCPT", "minilm-l12": "MiniLM-L12", "minilm-l6": "MiniLM-L6",
            "electra-base": "ELECTRA-base", "bge-base": "bge-base", "mxbai-base": "mxbai-base"}
    for k in order:
        p = cd / f"{k}.json"
        if p.exists():
            r = json.load(open(p))
            items.append((nice[k], r["by_cut"]["lenient_ge3"]["p_at_3"],
                          r["stage1_gate"]["chunk_auc_grade3"], "deploy"))
    if args.ft_dir:
        for k, lab in (("minilm-l6", "MiniLM-L6 (fine-tuned)"),
                       ("mxbai-base", "mxbai-base (fine-tuned)")):
            p = Path(args.ft_dir) / f"{k}-finetuned.json"
            if p.exists():
                r = json.load(open(p))
                items.append((lab, r["by_cut"]["lenient_ge3"]["p_at_3"],
                              r["stage1_gate"]["chunk_auc_grade3"], "deploy_ft"))
    for k, lab in (("qwen3-rr-4b", "Qwen3-4B (ref)"), ("qwen3-rr-8b", "Qwen3-8B (ref)")):
        p = cd / f"{k}.json"
        if p.exists():
            r = json.load(open(p))
            items.append((lab, r["by_cut"]["lenient_ge3"]["p_at_3"],
                          r["stage1_gate"]["chunk_auc_grade3"], "ref"))

    items.sort(key=lambda x: x[1])
    labels = [x[0] for x in items]
    p3 = [x[1] for x in items]
    auc = [x[2] for x in items]
    colors = {"deploy": "#4b6584", "deploy_ft": "#16a085", "ref": "#b3541e"}
    cols = [colors[x[3]] for x in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))
    y = range(len(items))
    ax1.barh(list(y), p3, color=cols)
    ax1.axvline(floor, color="#999", ls="--", lw=1); ax1.text(floor, -0.8, f"floor {floor:.2f}", fontsize=8, color="#666")
    ax1.axvline(oracle, color="#2d6a2d", ls="--", lw=1); ax1.text(oracle, -0.8, f"oracle {oracle:.2f}", fontsize=8, color="#2d6a2d")
    ax1.set_yticks(list(y)); ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("rerank P@3 (lenient, grade≥3)"); ax1.set_xlim(0.45, 0.95)
    ax1.set_title("Reranking quality on held-out test split")
    for i, v in enumerate(p3):
        ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    ax2.barh(list(y), auc, color=cols)
    ax2.axvline(0.80, color="#b00", ls="--", lw=1); ax2.text(0.80, -0.8, "0.80 bar", fontsize=8, color="#b00")
    ax2.axvline(0.5, color="#999", ls=":", lw=1)
    ax2.set_yticks(list(y)); ax2.set_yticklabels([])
    ax2.set_xlabel("Stage-1 gate: chunk AUC (grade≥3)"); ax2.set_xlim(0.5, 0.9)
    ax2.set_title("Score thresholdability (R1 gate)")
    for i, v in enumerate(auc):
        ax2.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#4b6584", label="deployable (zero-shot)"),
                        Patch(color="#16a085", label="deployable (fine-tuned)"),
                        Patch(color="#b3541e", label="offline reference (GPU)")],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
