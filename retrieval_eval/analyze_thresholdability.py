#!/usr/bin/env python3
"""Thresholdability of the EmbeddingGemma retrieval (cosine) score for EG × Gemma 3n, kenya.

Two questions, both about whether a cosine-similarity threshold can drive an
abstention/confidence gate (R1's idea, bar = chunk AUC 0.80):

  (A) CHUNK level  — does cosine separate RELEVANT from IRRELEVANT chunks?
      data: eg_kenya_pairgrades.json  [{qid, idx, rank, sim, grade(0-6)}] (Qwen3-32B V2 rubric)
      label relevant = grade>=5 (strict) / grade>=3 (lenient); score = cosine sim; metric = ROC-AUC + sweep.

  (B) ANSWER level — does the per-query retrieval score predict ANSWER quality?
      data: eg3n_kenya_answerlevel_thresh.json  [{qid, recall, top1, mean3, max}]
      label good = recall>0 / recall>=median; score = top1/mean3 cosine; metric = ROC-AUC.

Usage: python -m retrieval_eval.analyze_thresholdability [results_dir]
"""
import json, sys, statistics as st
from pathlib import Path

RES = Path(sys.argv[1] if len(sys.argv) > 1 else
           "configs/config-v0.2.0/results/retrieval_eval/r2c-embedder")


def auc(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]: j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j): ranks[order[k]] = avg
        i = j
    npos = sum(labels); nneg = len(labels) - npos
    if not npos or not nneg: return None
    sp = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (sp - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    print("== (A) CHUNK-level: cosine vs relevance grade ==")
    d = [x for x in json.loads((RES / "eg_kenya_pairgrades.json").read_text())
         if x["grade"] is not None and x["sim"] is not None]
    sims = [x["sim"] for x in d]; grades = [x["grade"] for x in d]
    for desc, thr in [("strict grade>=5", 5), ("lenient grade>=3", 3)]:
        lab = [g >= thr for g in grades]
        print(f"  {desc}: AUC={auc(sims, lab):.3f}  pos={sum(lab)}/{len(d)}  [R1 bar 0.80]")
        for cut in (0.50, 0.55, 0.60):
            kept = [(s, l) for s, l in zip(sims, lab) if s >= cut]
            prec = sum(l for _, l in kept) / len(kept) if kept else 0
            rec = sum(l for _, l in kept) / max(1, sum(lab))
            print(f"     sim>={cut}: keep {100*len(kept)/len(d):.0f}%  precision={prec:.3f}  recall={rec:.3f}")

    print("\n== (B) ANSWER-level: cosine vs key-fact recall ==")
    a = json.loads((RES / "eg3n_kenya_answerlevel_thresh.json").read_text())
    rec = [r["recall"] for r in a]; med = st.median(rec)
    for sk in ("top1", "mean3"):
        s = [r[sk] for r in a]
        for desc, lab in [("recall>0", [r > 0 for r in rec]), ("recall>=median", [r >= med for r in rec])]:
            print(f"  score={sk} label={desc}: AUC={auc(s, lab):.3f}")


if __name__ == "__main__":
    main()
