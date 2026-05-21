#!/usr/bin/env python3
"""Calibrate a Lynx faithfulness run against an independent judge.

Lynx is a high-recall / low-precision hallucination detector — its raw FAIL
rate must be calibrated before it can be trusted. This tool draws a stratified
blinded sample, has an independent judge re-judge it, and measures agreement
(precision, miss rate, population estimate of true hallucinations).

Pipeline (deterministic sample -> external LLM/human judging -> deterministic
score). The independent judging is a semantic step; its backend is intentionally
NOT wired here. `sample` writes the blinded cases plus the exact pinned rubric;
an operator runs any judge (a second model, or a human) against that rubric to
produce a verdicts file; `score` consumes it.

  python -m generator_eval.calibrate sample <run-dir>
  # ... produce <run-dir>/calibration_verdicts.json with a judge + the rubric ...
  python -m generator_eval.calibrate score  <run-dir> \
      --verdicts <run-dir>/calibration_verdicts.json
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# ── Sampling parameters — frozen for reproducibility ─────────────────────────
SAMPLE_SEED = 42
N_PASS_SAMPLE = 50
N_PER_FAIL_CATEGORY = 10
FAIL_CATEGORIES = ["contradiction", "unsupported_addition",
                   "omission", "refusal", "unclear"]

# ── Pinned calibration rubric ────────────────────────────────────────────────
# The exact instruction the independent judge must be given. Frozen so the
# calibration applies an identical faithfulness definition every run.
CALIBRATION_RUBRIC = """\
Independently judge whether each answer is FAITHFUL to its context.

A medical assistant (Gemma 4) was given clinical-guideline context and answered
a clinical question. Judge from first principles — do NOT try to match any
other model's verdict. For each case you are given: query_id, query_text,
context, answer.

Judge FAIL if and ONLY if the answer does one of these:
- contradiction        — asserts something that conflicts with the context
                         (wrong dose, wrong threshold, states X where the
                         context says Y).
- unsupported addition — asserts a clinical claim NOT supported by the context
                         (information introduced beyond the document).

Judge PASS in every other case. These are NOT faithfulness failures and must
be PASS:
- The answer is incomplete — correct as far as it goes but omits things the
  context contains. Incompleteness is not unfaithfulness.
- The answer declines / refuses ("consult a doctor", "I cannot provide this",
  "out of scope"). A refusal asserts no false claim, so it is faithful (PASS).

Edge guidance:
- Trivial paraphrase / equivalent wording ("≥50%" vs "50% or more"; a drug's
  two valid names) is NOT a contradiction.
- If the context itself contains conflicting figures and the answer picks one
  of them, that is PASS — the answer is supported by the context.
- Judge only against the provided context, not outside medical knowledge.

Output one JSON object per case:
  {"query_id": "...", "verdict": "PASS | FAIL",
   "reasoning": "one or two sentences"}
"""


def _load(path: Path):
    if not path.exists():
        sys.exit(f"ERROR: required file not found: {path}")
    return json.loads(path.read_text())


def _bootstrap_ci(values, n_resamples=1000, alpha=0.05):
    """Percentile bootstrap CI on the mean of a 0/1 list. Deterministic."""
    rng = random.Random(1234)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_resamples):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return (round(means[int(alpha / 2 * n_resamples)], 4),
            round(means[int((1 - alpha / 2) * n_resamples)], 4))


def cmd_sample(run_dir: Path) -> None:
    """Deterministic, seeded: draw a stratified blinded calibration sample.

    50 PASS + N_PER_FAIL_CATEGORY from each of the 5 FAIL categories. The
    random-call order is fixed so the same inputs always yield the same sample.
    """
    lynx = {r["query_id"]: r
            for r in _load(run_dir / "lynx_scores.json")["results"]}
    cats = {r["query_id"]: r
            for r in _load(run_dir / "lynx_fail_analysis.json")}
    oracle = {r["query_id"]: r
              for r in _load(run_dir / "oracle_responses.json")["results"]}

    rng = random.Random(SAMPLE_SEED)

    # PASS stratum — random sample of all Lynx-PASS rows.
    pass_ids = sorted(q for q, r in lynx.items() if r.get("score") == "PASS")
    if len(pass_ids) < N_PASS_SAMPLE:
        sys.exit(f"ERROR: only {len(pass_ids)} PASS rows, need {N_PASS_SAMPLE}")
    pass_sample = sorted(rng.sample(pass_ids, N_PASS_SAMPLE))

    # FAIL strata — N per category, in the fixed category order.
    by_cat: dict[str, list[str]] = {}
    for qid, c in cats.items():
        by_cat.setdefault(c["category"], []).append(qid)
    fail_sample: list[str] = []
    for cat in FAIL_CATEGORIES:
        ids = sorted(by_cat.get(cat, []))
        pick = sorted(rng.sample(ids, min(N_PER_FAIL_CATEGORY, len(ids))))
        fail_sample += pick

    sample_ids = pass_sample + fail_sample
    rng.shuffle(sample_ids)  # so the judge can't infer strata from order

    blind, key = [], []
    for qid in sample_ids:
        o = oracle[qid]
        blind.append({
            "query_id": qid,
            "query_text": o["query_text"],
            "context": o["context"],
            "answer": o["model_response"],
        })
        key.append({
            "query_id": qid,
            "lynx_score": lynx[qid]["score"],
            "lynx_category": cats[qid]["category"] if qid in cats else None,
        })

    blind_path = run_dir / "calibration_blind.json"
    key_path = run_dir / "calibration_key.json"
    rubric_path = run_dir / "calibration_rubric.txt"
    blind_path.write_text(json.dumps(blind, indent=2, ensure_ascii=False))
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False))
    rubric_path.write_text(CALIBRATION_RUBRIC)

    strata = Counter(k["lynx_score"] for k in key)
    print(f"Wrote {blind_path}  ({len(blind)} blinded cases)")
    print(f"Wrote {key_path}    (lynx verdict/category, held aside for scoring)")
    print(f"Wrote {rubric_path} (pinned calibration rubric)")
    print(f"  strata: {dict(strata)}  (seed={SAMPLE_SEED})")
    print()
    print("NEXT STEP (independent judging — backend intentionally not wired):")
    print(f"  Run an independent judge over {blind_path.name} using the rubric")
    print(f"  in {rubric_path.name} — a second model OR a human. It must emit a")
    print("  JSON array, one object per case (keys: query_id, verdict,")
    print("  reasoning) to:")
    print(f"      {run_dir / 'calibration_verdicts.json'}")
    print("  Then run:  calibrate score <run-dir> --verdicts <that file>")


def cmd_score(run_dir: Path, verdicts_path: Path) -> None:
    """Deterministic: confusion matrix + precision/recall + population estimate."""
    key = {r["query_id"]: r
           for r in _load(run_dir / "calibration_key.json")}
    verdicts = _load(verdicts_path)
    if not isinstance(verdicts, list):
        sys.exit(f"ERROR: {verdicts_path} must be a JSON array")
    ind = {r["query_id"]: r for r in verdicts}

    # Validation.
    problems = []
    for qid in key:
        if qid not in ind:
            problems.append(f"{qid}: no independent verdict")
        elif ind[qid].get("verdict") not in ("PASS", "FAIL"):
            problems.append(f"{qid}: bad verdict {ind[qid].get('verdict')!r}")
    if problems:
        print("VALIDATION FAILED:")
        for p in problems[:10]:
            print(f"  - {p}")
        sys.exit(1)

    # Population base rates from the full Lynx run.
    lynx = _load(run_dir / "lynx_scores.json")["results"]
    n_pass = sum(1 for r in lynx if r.get("score") == "PASS")
    n_fail = sum(1 for r in lynx if r.get("score") == "FAIL")
    n_total = len(lynx)

    # Confusion matrix over the sample.
    cm = Counter((key[q]["lynx_score"], ind[q]["verdict"]) for q in key)

    fail_stratum = [q for q in key if key[q]["lynx_score"] == "FAIL"]
    pass_stratum = [q for q in key if key[q]["lynx_score"] == "PASS"]
    f_in_fail = [1 if ind[q]["verdict"] == "FAIL" else 0 for q in fail_stratum]
    f_in_pass = [1 if ind[q]["verdict"] == "FAIL" else 0 for q in pass_stratum]

    precision = sum(f_in_fail) / len(fail_stratum) if fail_stratum else 0.0
    miss_rate = sum(f_in_pass) / len(pass_stratum) if pass_stratum else 0.0

    # Population estimate of true hallucinations.
    est_from_fail = n_fail * precision
    est_from_pass = n_pass * miss_rate
    est_total = est_from_fail + est_from_pass

    # Per-category confirmation within the FAIL stratum.
    per_cat = {}
    for cat in FAIL_CATEGORIES:
        qs = [q for q in fail_stratum if key[q]["lynx_category"] == cat]
        per_cat[cat] = {
            "n": len(qs),
            "confirmed_fail": sum(1 for q in qs if ind[q]["verdict"] == "FAIL"),
        }

    report = {
        "verdicts_file": verdicts_path.name,
        "n_sample": len(key),
        "confusion_matrix": {
            "lynx_PASS_ref_PASS": cm[("PASS", "PASS")],
            "lynx_PASS_ref_FAIL": cm[("PASS", "FAIL")],  # Lynx missed
            "lynx_FAIL_ref_PASS": cm[("FAIL", "PASS")],  # Lynx false alarm
            "lynx_FAIL_ref_FAIL": cm[("FAIL", "FAIL")],
        },
        "lynx_precision": round(precision, 4),
        "lynx_precision_95ci": _bootstrap_ci(f_in_fail),
        "lynx_miss_rate": round(miss_rate, 4),
        "lynx_miss_rate_95ci": _bootstrap_ci(f_in_pass),
        "fail_stratum_n": len(fail_stratum),
        "pass_stratum_n": len(pass_stratum),
        "per_fail_category_confirmation": per_cat,
        "population": {"n_total": n_total, "n_lynx_pass": n_pass,
                       "n_lynx_fail": n_fail},
        "estimated_true_hallucinations": round(est_total, 1),
        "estimated_true_hallucination_rate": round(est_total / n_total, 4),
    }
    out = run_dir / "calibration_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Validated {len(key)} verdicts — OK")
    print(f"Wrote {out}\n")
    print(f"  confusion (n={len(key)}):")
    print(f"    Lynx PASS / ref PASS : {cm[('PASS','PASS')]}")
    print(f"    Lynx PASS / ref FAIL : {cm[('PASS','FAIL')]}   (Lynx missed)")
    print(f"    Lynx FAIL / ref PASS : {cm[('FAIL','PASS')]}   (false alarm)")
    print(f"    Lynx FAIL / ref FAIL : {cm[('FAIL','FAIL')]}")
    print(f"  Lynx precision : {precision:.1%}  95% CI {report['lynx_precision_95ci']}")
    print(f"  Lynx miss rate : {miss_rate:.1%}  95% CI {report['lynx_miss_rate_95ci']}")
    print(f"  est. true hallucinations: ~{est_total:.0f}/{n_total} "
          f"= {est_total/n_total:.2%}")
    print(f"  per FAIL category (confirmed/n): "
          + ", ".join(f"{c} {per_cat[c]['confirmed_fail']}/{per_cat[c]['n']}"
                      for c in FAIL_CATEGORIES))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sample", help="draw the stratified blinded sample")
    ps.add_argument("run_dir")

    pc = sub.add_parser("score", help="score independent verdicts vs Lynx")
    pc.add_argument("run_dir")
    pc.add_argument("--verdicts", required=True,
                    help="independent judge's verdicts JSON (query_id + verdict)")

    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"ERROR: run dir not found: {run_dir}")

    if args.cmd == "sample":
        cmd_sample(run_dir)
    elif args.cmd == "score":
        cmd_score(run_dir, Path(args.verdicts))


if __name__ == "__main__":
    main()
