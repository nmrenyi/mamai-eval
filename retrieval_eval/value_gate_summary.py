#!/usr/bin/env python3
"""R2c P1 — summarize the 3-arm end-to-end value gate.

Reads the gecko / hybrid / hybrid_rerank arm result dirs (each
<arms-root>/<arm>/run/<dataset>.json) and prints the acceptance-gate table:
SAQ key-fact recall, refusal rate, harm rate per arm (A=gecko deployed,
B=hybrid isolates fusion, C=hybrid+rerank isolates reranking), and MCQ accuracy.

A->B = the R2a fusion effect; B->C = the reranking effect (the decision).

Usage:
  python -m retrieval_eval.value_gate_summary --arms-root results/value_gate \\
      --saq kenya,afrimedqa_saq --mcq afrimedqa --out value_gate_summary.json
"""

import argparse
import json
from pathlib import Path

ARMS = [("gecko", "A gecko (deployed)"), ("hybrid", "B hybrid (RRF)"),
        ("hybrid_rerank", "C hybrid+rerank")]


def load_agg(arms_root, arm, ds):
    p = Path(arms_root) / arm / "run" / f"{ds}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("aggregate_scores", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms-root", required=True)
    ap.add_argument("--saq", default="kenya,afrimedqa_saq")
    ap.add_argument("--mcq", default="afrimedqa")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    saq = [d.strip() for d in args.saq.split(",") if d.strip()]
    mcq = [d.strip() for d in args.mcq.split(",") if d.strip()]
    out = {"saq": {}, "mcq": {}}

    print("\n=== SAQ value gate (key-fact recall / refusal / harm) ===")
    for ds in saq:
        print(f"\n{ds}:")
        print(f"  {'arm':22} {'recall':>7} {'refusal':>8} {'harm':>6} {'n':>5}")
        out["saq"][ds] = {}
        for arm, label in ARMS:
            a = load_agg(args.arms_root, arm, ds)
            if a is None:
                print(f"  {label:22}   (missing)"); continue
            out["saq"][ds][arm] = a
            print(f"  {label:22} {a.get('mean_key_fact_recall',0):>7.4f} "
                  f"{a.get('refusal_rate',0):>8.4f} {a.get('harm_rate',0):>6.4f} "
                  f"{a.get('n_judged',0):>5}")

    print("\n=== MCQ value gate (accuracy) ===")
    for ds in mcq:
        print(f"\n{ds}:")
        print(f"  {'arm':22} {'accuracy':>9} {'n':>5}")
        out["mcq"][ds] = {}
        for arm, label in ARMS:
            a = load_agg(args.arms_root, arm, ds)
            if a is None:
                print(f"  {label:22}   (missing)"); continue
            out["mcq"][ds][arm] = a
            print(f"  {label:22} {a.get('accuracy',0):>9.4f} {a.get('total',0):>5}")

    # deltas: B->C reranking effect on SAQ recall
    print("\n=== reranking effect (B hybrid -> C hybrid+rerank) ===")
    for ds in saq:
        b = out["saq"].get(ds, {}).get("hybrid", {})
        c = out["saq"].get(ds, {}).get("hybrid_rerank", {})
        if b and c:
            dr = c.get("mean_key_fact_recall", 0) - b.get("mean_key_fact_recall", 0)
            print(f"  {ds}: key-fact recall {b.get('mean_key_fact_recall',0):.4f} -> "
                  f"{c.get('mean_key_fact_recall',0):.4f}  (delta {dr:+.4f})")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
