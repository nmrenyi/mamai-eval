#!/usr/bin/env python3
"""Categorize the FAIL verdicts of a Lynx faithfulness run.

Lynx's `FAIL` conflates several distinct things — genuine hallucination,
incompleteness, refusal. This tool splits them so the true hallucination
rate can be read off.

Pipeline (deterministic prep -> external LLM categorization -> deterministic
aggregate). The categorization itself is a semantic judgement done by an LLM;
the backend for that step is intentionally NOT wired here. `extract` writes
the cases plus the exact pinned rubric; an operator runs any judge against
that rubric to produce a categories file; `aggregate` consumes it.

  python -m generator_eval.analyze_lynx_fails extract  <run-dir>
  # ... produce <run-dir>/lynx_fail_categories.json with an LLM + the rubric ...
  python -m generator_eval.analyze_lynx_fails aggregate <run-dir> \
      --categories <run-dir>/lynx_fail_categories.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ── Pinned categorization rubric ─────────────────────────────────────────────
# Frozen so every run (incl. future oracle revisions) categorizes identically.
# This is the exact instruction the LLM categorizer must be given.
CATEGORIZE_RUBRIC = """\
Categorize WHY each faithfulness FAIL failed.

A medical assistant (Gemma 4) was given clinical-guideline context and answered a clinical question. A judge model (Patronus Lynx) marked the answer FAIL with bullet-point reasoning. For each case you are given: query_id, query_text, context (the guideline text the model was given), answer (the model's response), lynx_reasoning (Lynx's reasoning bullets).

Assign exactly one PRIMARY category:

1. contradiction — the answer asserts something that conflicts with / is inconsistent with the context (wrong dose, wrong threshold, states X where the context says Y). A genuine faithfulness failure.
2. unsupported_addition — the answer introduces clinical claims NOT present in the context ("information beyond the document"). A genuine faithfulness failure.
3. omission — the answer is correct as far as it goes but leaves out content the context contains; Lynx faults it as incomplete. NOT a hallucination.
4. refusal — the answer declines to give clinical substance ("I cannot provide a treatment", "consult a doctor") when the context did contain relevant info. NOT a hallucination.
5. unclear — Lynx's reasoning does not let you confidently determine the failure type.

Rules:
- Judge by what Lynx's reasoning actually says is wrong, cross-checked against the answer and the context.
- A case may have multiple issues — pick the PRIMARY one (usually the operative reason in Lynx's concluding bullet); list any others in `secondary`.
- `contradiction` and `unsupported_addition` are the two TRUE faithfulness failures; the other three are not.
- Also judge whether Lynx's FAIL verdict itself looks `sound` or `questionable`.

Output one JSON object per case:
{"query_id": "...", "category": "<one of the five>", "secondary": ["...other issues, or empty..."], "justification": "one sentence", "lynx_verdict": "sound | questionable", "lynx_verdict_note": "one sentence if questionable, else empty"}
"""

VALID_CATEGORIES = ["contradiction", "unsupported_addition",
                    "omission", "refusal", "unclear"]
TRUE_HALLUCINATION = {"contradiction", "unsupported_addition"}


def _load(path: Path):
    if not path.exists():
        sys.exit(f"ERROR: required file not found: {path}")
    return json.loads(path.read_text())


def cmd_extract(run_dir: Path) -> None:
    """Deterministic: join Lynx FAILs with their oracle context/answer."""
    lynx = _load(run_dir / "lynx_scores.json")["results"]
    oracle = {r["query_id"]: r
              for r in _load(run_dir / "oracle_responses.json")["results"]}

    fails = [r for r in lynx if r.get("score") == "FAIL"]
    cases = []
    for r in fails:
        o = oracle.get(r["query_id"], {})
        cases.append({
            "query_id": r["query_id"],
            "query_text": o.get("query_text", ""),
            "context": o.get("context", ""),
            "answer": o.get("model_response", ""),
            "lynx_reasoning": r.get("reasoning"),
            "lynx_note": r.get("note"),
        })

    cases_path = run_dir / "lynx_fail_cases.json"
    rubric_path = run_dir / "lynx_categorize_rubric.txt"
    cases_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    rubric_path.write_text(CATEGORIZE_RUBRIC)

    print(f"Wrote {cases_path}  ({len(cases)} FAIL cases)")
    print(f"Wrote {rubric_path}  (pinned categorization rubric)")
    print()
    print("NEXT STEP (LLM categorization — backend intentionally not wired):")
    print(f"  Run a judge over {cases_path.name} using the rubric in")
    print(f"  {rubric_path.name}. It must emit one JSON object per case with")
    print("  keys: query_id, category, secondary, justification, lynx_verdict,")
    print("  lynx_verdict_note — as a JSON array — to:")
    print(f"      {run_dir / 'lynx_fail_categories.json'}")
    print("  Then run:  analyze_lynx_fails aggregate <run-dir> "
          "--categories <that file>")


def cmd_aggregate(run_dir: Path, categories_path: Path) -> None:
    """Deterministic: validate the categories file and report the breakdown."""
    lynx = _load(run_dir / "lynx_scores.json")["results"]
    n_responses = len(lynx)
    fail_ids = {r["query_id"] for r in lynx if r.get("score") == "FAIL"}

    labels = _load(categories_path)
    if not isinstance(labels, list):
        sys.exit(f"ERROR: {categories_path} must be a JSON array of label objects")

    # ── Validation ──────────────────────────────────────────────────────────
    seen, problems = set(), []
    for i, lab in enumerate(labels):
        qid = lab.get("query_id")
        cat = lab.get("category")
        if qid is None:
            problems.append(f"row {i}: missing query_id")
            continue
        if qid in seen:
            problems.append(f"{qid}: duplicate label")
        seen.add(qid)
        if cat not in VALID_CATEGORIES:
            problems.append(f"{qid}: invalid category {cat!r}")
    missing = fail_ids - seen
    extra = seen - fail_ids
    if missing:
        problems.append(f"{len(missing)} FAIL ids have no label "
                        f"(e.g. {sorted(missing)[:3]})")
    if extra:
        problems.append(f"{len(extra)} labels are not FAIL rows "
                        f"(e.g. {sorted(extra)[:3]})")
    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    # ── Aggregate ───────────────────────────────────────────────────────────
    dist = Counter(lab["category"] for lab in labels)
    true_h = sum(dist[c] for c in TRUE_HALLUCINATION)
    verdict_tally = Counter(lab.get("lynx_verdict") for lab in labels
                            if lab.get("lynx_verdict"))

    summary = {
        "categories_file": categories_path.name,
        "n_fail": len(fail_ids),
        "n_responses": n_responses,
        "distribution": {c: dist.get(c, 0) for c in VALID_CATEGORIES},
        "true_hallucination_count": true_h,
        "true_hallucination_rate_of_all": round(true_h / n_responses, 4),
        "raw_fail_rate_of_all": round(len(fail_ids) / n_responses, 4),
        "lynx_verdict_tally": dict(verdict_tally) or None,
    }
    out = run_dir / "lynx_fail_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Validated {len(labels)} labels against {len(fail_ids)} FAIL rows — OK")
    print(f"Wrote {out}\n")
    for c in VALID_CATEGORIES:
        tag = "  (true hallucination)" if c in TRUE_HALLUCINATION else ""
        print(f"  {c:22s}: {dist.get(c, 0):3d}{tag}")
    print(f"  {'TOTAL':22s}: {len(labels):3d}")
    print()
    print(f"  true hallucination: {true_h}/{n_responses} = "
          f"{100*true_h/n_responses:.2f}%   "
          f"(raw FAIL: {len(fail_ids)}/{n_responses} = "
          f"{100*len(fail_ids)/n_responses:.2f}%)")
    if verdict_tally:
        print(f"  Lynx verdict soundness: {dict(verdict_tally)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="join Lynx FAILs with oracle context")
    pe.add_argument("run_dir")

    pa = sub.add_parser("aggregate", help="validate categories file + report")
    pa.add_argument("run_dir")
    pa.add_argument("--categories", required=True,
                    help="LLM-produced categories JSON (query_id + category per row)")

    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"ERROR: run dir not found: {run_dir}")

    if args.cmd == "extract":
        cmd_extract(run_dir)
    elif args.cmd == "aggregate":
        cmd_aggregate(run_dir, Path(args.categories))


if __name__ == "__main__":
    main()
