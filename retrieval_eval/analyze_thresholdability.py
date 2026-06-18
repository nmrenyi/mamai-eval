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
import json, sys, random, statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path(sys.argv[1] if len(sys.argv) > 1 else
           "configs/config-v0.2.0/results/retrieval_eval/r2c-embedder")
random.seed(0)


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
    sims = [x["sim"] for x in d]; grades = [x["grade"] for x in d]; qids = [x["qid"] for x in d]
    byq = defaultdict(list)
    for s, l, q in zip(sims, grades, qids): byq[q].append((s, l))
    for desc, thr in [("strict grade>=5", 5), ("lenient grade>=3", 3)]:
        lab = [g >= thr for g in grades]
        base = sum(lab) / len(d)
        pooled = auc(sims, lab)
        # cluster bootstrap CI (resample queries)
        qs = list(byq.keys()); boots = []
        for _ in range(300):
            samp = [random.choice(qs) for _ in qs]; ss = []; ll = []
            for q in samp:
                for s, g in byq[q]: ss.append(s); ll.append(g >= thr)
            a = auc(ss, ll)
            if a is not None: boots.append(a)
        boots.sort(); lo, hi = boots[int(.025 * len(boots))], boots[int(.975 * len(boots))]
        # per-query within-query AUC
        pq = [a for q in byq for a in [auc([s for s, _ in byq[q]], [g >= thr for _, g in byq[q]])] if a is not None]
        # effect size
        sr = [s for s, l in zip(sims, lab) if l]; si = [s for s, l in zip(sims, lab) if not l]
        cohen = (st.mean(sr) - st.mean(si)) / st.pstdev(sims)
        print(f"  {desc}: pooled AUC={pooled:.3f} (95% CI [{lo:.3f},{hi:.3f}])  per-query AUC={st.mean(pq):.3f}  "
              f"Cohen d={cohen:.2f}  base-rate={base:.3f}  pos={sum(lab)}/{len(d)}  [R1 bar 0.80]")
        for cut in (0.50, 0.55, 0.60):
            kept = [(s, l) for s, l in zip(sims, lab) if s >= cut]
            prec = sum(l for _, l in kept) / len(kept) if kept else 0
            rec = sum(l for _, l in kept) / max(1, sum(lab))
            print(f"     sim>={cut}: keep {100*len(kept)/len(d):.0f}%  precision={prec:.3f}  recall={rec:.3f}")

    print("\n== (A2) R1-style Stage-1 gate (EmbeddingGemma top-3, the injected bundle) ==")
    t3 = [x for x in d if x["rank"] < 3]
    s3 = [x["sim"] for x in t3]; g3 = [x["grade"] for x in t3]
    nq3 = len({x["qid"] for x in t3})
    for thr in (3, 5):
        lab = [g >= thr for g in g3]
        print(f"  [absolute] chunk-level AUC grade>={thr}: {auc(s3, lab):.3f}  (P@3={sum(lab)/nq3/3:.3f})  [bar 0.80, stop 0.60]")
    bq = defaultdict(list)
    for x in t3: bq[x["qid"]].append((x["sim"], x["grade"]))
    num = den = 0.0
    for rows in bq.values():
        rel = [s for s, g in rows if g >= 3]; junk = [s for s, g in rows if g < 3]
        for r in rel:
            for j in junk:
                den += 1; num += 1.0 if r > j else (0.5 if r == j else 0.0)
    print(f"  [relative]  within-query concordance grade>=3: {num/den:.3f}  (over {int(den)} co-bundle rel/junk pairs)  [0.5=chance]")
    bs = [max(s for s, _ in rows) for rows in bq.values()]
    bl = [any(g >= 3 for _, g in rows) for rows in bq.values()]
    print(f"  [gated]     bundle top-1 AUC (any grade>=3): {auc(bs, bl):.3f}  (HR@3={sum(bl)/len(bl):.3f})  [bar 0.80]")

    print("\n== (B) ANSWER-level: cosine vs key-fact recall ==")
    a = json.loads((RES / "eg3n_kenya_answerlevel_thresh.json").read_text())
    rec = [r["recall"] for r in a]; med = st.median(rec)
    for sk in ("top1", "mean3"):
        s = [r[sk] for r in a]
        for desc, lab in [("recall>0", [r > 0 for r in rec]), ("recall>=median", [r >= med for r in rec])]:
            print(f"  score={sk} label={desc}: AUC={auc(s, lab):.3f}")


if __name__ == "__main__":
    main()
