#!/usr/bin/env python3
"""Gate check: does gpt-oss-120b match Claude on v0.1.0 well enough to trust?

Compares gpt-oss outputs against Claude's v0.1.0 baseline on two pre-committed
gates. Pass both ⇒ trust gpt-oss for v0.2.0 categorization + calibration.
Fail either ⇒ stop and report.

Gates (set before looking at results):

  Categorization (per-bucket count):
      |gpt_oss[c] - claude[c]| / claude[c] ≤ 0.20  for each c in
      [contradiction, unsupported_addition, omission, refusal, unclear]

  Calibration:
      row-level verdict agreement ≥ 90/100
      AND population true-hallucination rate within ±0.2pp of Claude's

Reads:
  <run-dir>/lynx_fail_categories.json            Claude v0.1.0 categorization
  <run-dir>/lynx_fail_categories_gpt_oss.json    gpt-oss categorization
  <run-dir>/calibration_independent.json         Claude v0.1.0 verdicts
  <run-dir>/calibration_verdicts_gpt_oss.json    gpt-oss verdicts
  <run-dir>/calibration_key.json                 lynx verdict per sampled id
  <run-dir>/lynx_scores.json                     population PASS/FAIL counts

Usage:
  python -m generator_eval.validate_gpt_oss_v0_1 \\
      configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator_eval.analyze_lynx_fails import VALID_CATEGORIES

# ── Pre-committed thresholds (locked before validation) ──────────────────────
CATEGORIZATION_TOLERANCE = 0.20         # ±20% per bucket
CALIBRATION_MIN_AGREEMENT = 90          # ≥90 / 100 verdict matches
CALIBRATION_RATE_TOLERANCE_PP = 0.20    # ±0.2 percentage points on population rate


def _load(path: Path):
    if not path.exists():
        sys.exit(f"ERROR: required file not found: {path}")
    return json.loads(path.read_text())


def _calibrated_rate(verdicts_by_id: dict[str, str],
                     key: list[dict],
                     n_total: int,
                     n_lynx_pass: int,
                     n_lynx_fail: int) -> tuple[float, float, float]:
    """Replicate calibrate.cmd_score's population estimate.

    Returns (precision, miss_rate, calibrated_rate). Uses the same formula:
        est_total = n_fail * precision + n_pass * miss_rate
    """
    fail_stratum = [k for k in key if k["lynx_score"] == "FAIL"]
    pass_stratum = [k for k in key if k["lynx_score"] == "PASS"]
    if not fail_stratum or not pass_stratum:
        return 0.0, 0.0, 0.0

    precision = sum(1 for k in fail_stratum
                    if verdicts_by_id.get(k["query_id"]) == "FAIL") / len(fail_stratum)
    miss = sum(1 for k in pass_stratum
               if verdicts_by_id.get(k["query_id"]) == "FAIL") / len(pass_stratum)
    est = n_lynx_fail * precision + n_lynx_pass * miss
    return precision, miss, est / n_total


def gate_categorization(claude_cats: list[dict], gptoss_cats: list[dict]) -> tuple[bool, list[str]]:
    """±20% per-bucket count check."""
    claude_dist = Counter(r["category"] for r in claude_cats)
    gptoss_dist = Counter(r["category"] for r in gptoss_cats)
    lines = ["", "── Categorization gate (±20% per bucket) ─────────────────────"]
    lines.append(f"  {'category':22s} {'claude':>8s} {'gpt-oss':>8s} "
                 f"{'delta':>8s} {'tol':>8s} {'PASS?':>6s}")
    passed_all = True
    for c in VALID_CATEGORIES:
        cl = claude_dist.get(c, 0)
        go = gptoss_dist.get(c, 0)
        delta = abs(go - cl)
        tol = max(1, int(round(cl * CATEGORIZATION_TOLERANCE)))
        passed = delta <= tol
        passed_all &= passed
        lines.append(f"  {c:22s} {cl:>8d} {go:>8d} "
                     f"{delta:>+8d} {tol:>8d} {'pass' if passed else 'FAIL':>6s}")
    lines.append(f"  TOTAL gpt-oss labels: {len(gptoss_cats)} (claude: {len(claude_cats)})")
    lines.append(f"  Categorization gate: {'PASS' if passed_all else 'FAIL'}")
    return passed_all, lines


def gate_calibration(claude_verdicts: list[dict],
                     gptoss_verdicts: list[dict],
                     key: list[dict],
                     lynx_scores: list[dict]) -> tuple[bool, list[str]]:
    """Verdict-agreement ≥90/100 AND calibrated rate within ±0.2pp."""
    claude_by_id = {r["query_id"]: r["verdict"] for r in claude_verdicts}
    gptoss_by_id = {r["query_id"]: r["verdict"] for r in gptoss_verdicts}

    sample_ids = [k["query_id"] for k in key]
    agree = sum(1 for q in sample_ids
                if claude_by_id.get(q) == gptoss_by_id.get(q))
    n_total = len(lynx_scores)
    n_lynx_pass = sum(1 for r in lynx_scores if r.get("score") == "PASS")
    n_lynx_fail = sum(1 for r in lynx_scores if r.get("score") == "FAIL")

    cl_prec, cl_miss, cl_rate = _calibrated_rate(claude_by_id, key, n_total,
                                                 n_lynx_pass, n_lynx_fail)
    go_prec, go_miss, go_rate = _calibrated_rate(gptoss_by_id, key, n_total,
                                                 n_lynx_pass, n_lynx_fail)
    rate_delta_pp = abs(go_rate - cl_rate) * 100

    agree_pass = agree >= CALIBRATION_MIN_AGREEMENT
    rate_pass = rate_delta_pp <= CALIBRATION_RATE_TOLERANCE_PP

    lines = ["", "── Calibration gate ──────────────────────────────────────────"]
    lines.append(f"  Row-level verdict agreement: {agree}/{len(sample_ids)}  "
                 f"(≥{CALIBRATION_MIN_AGREEMENT} required) "
                 f"→ {'pass' if agree_pass else 'FAIL'}")
    lines.append(f"  Disagreement breakdown:")
    cm = Counter((claude_by_id.get(q), gptoss_by_id.get(q)) for q in sample_ids)
    for (c, g), n in sorted(cm.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        marker = "  " if c == g else " *"
        lines.append(f"   {marker} claude={c} / gpt-oss={g}: {n}")
    missing_claude = sum(1 for q in sample_ids if q not in claude_by_id)
    missing_gptoss = sum(1 for q in sample_ids if q not in gptoss_by_id)
    if missing_claude or missing_gptoss:
        lines.append(f"  WARNING: missing verdicts — claude={missing_claude}, "
                     f"gpt-oss={missing_gptoss}")
    lines.append(f"  Claude precision={cl_prec:.2%}  miss={cl_miss:.2%}  "
                 f"rate={cl_rate:.4f} ({cl_rate*100:.2f}%)")
    lines.append(f"  gpt-oss precision={go_prec:.2%}  miss={go_miss:.2%}  "
                 f"rate={go_rate:.4f} ({go_rate*100:.2f}%)")
    lines.append(f"  Rate delta: {rate_delta_pp:.3f}pp  "
                 f"(≤{CALIBRATION_RATE_TOLERANCE_PP}pp required) "
                 f"→ {'pass' if rate_pass else 'FAIL'}")

    passed = agree_pass and rate_pass
    lines.append(f"  Calibration gate: {'PASS' if passed else 'FAIL'}")
    return passed, lines


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="v0.1.0 run dir with both Claude + gpt-oss outputs")
    args = p.parse_args()
    rd = Path(args.run_dir)
    if not rd.is_dir():
        sys.exit(f"ERROR: not a directory: {rd}")

    claude_cats = _load(rd / "lynx_fail_categories.json")
    gptoss_cats = _load(rd / "lynx_fail_categories_gpt_oss.json")
    claude_verdicts = _load(rd / "calibration_independent.json")
    gptoss_verdicts = _load(rd / "calibration_verdicts_gpt_oss.json")
    key = _load(rd / "calibration_key.json")
    lynx = _load(rd / "lynx_scores.json")["results"]

    cat_passed, cat_lines = gate_categorization(claude_cats, gptoss_cats)
    cal_passed, cal_lines = gate_calibration(claude_verdicts, gptoss_verdicts,
                                             key, lynx)

    print("\n".join(cat_lines))
    print("\n".join(cal_lines))

    print("\n══════════════════════════════════════════════════════════════")
    if cat_passed and cal_passed:
        print("  OVERALL: PASS — trust gpt-oss for v0.2.0 categorization + calibration")
        sys.exit(0)
    else:
        which = []
        if not cat_passed: which.append("categorization")
        if not cal_passed: which.append("calibration")
        print(f"  OVERALL: FAIL — stop; gpt-oss failed gate(s): {', '.join(which)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
