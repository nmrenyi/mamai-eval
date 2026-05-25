"""CLI for the judge-validation bake-off.

Usage:
    python -m calibration.judge_validation fetch
    python -m calibration.judge_validation judge \\
        --base-url http://gpu013.rcp.epfl.ch:8000/v1 \\
        --model openai/gpt-oss-120b \\
        --output calibration/judge_validation/verdicts/gpt-oss-120b.jsonl
    python -m calibration.judge_validation metrics \\
        --verdicts calibration/judge_validation/verdicts/gpt-oss-120b.jsonl \\
        --report-md calibration/judge_validation/reports/gpt-oss-120b.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import fetch, judge, metrics


def cmd_fetch(args: argparse.Namespace) -> int:
    p = fetch.fetch_calibration(
        cache_dir=args.cache_dir,
        revision=args.revision,
        force=args.force,
    )
    size = p.stat().st_size
    print(f"Calibration cached at: {p} ({size:,} bytes)")
    return 0


def _resolve_response_format(spec: str | None) -> dict | None:
    if not spec:
        return None
    if spec == "criterion_verdict":
        return judge.CRITERION_VERDICT_SCHEMA
    return json.loads(spec)


def cmd_judge(args: argparse.Namespace) -> int:
    cal_path = Path(args.input) if args.input else fetch.fetch_calibration()
    rows = metrics.load_rows(cal_path)
    if args.limit:
        rows = rows[: args.limit]

    extra_body = json.loads(args.extra_body) if args.extra_body else None
    response_format = _resolve_response_format(args.json_schema)
    grader = judge.make_openai_grader(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
        seed=args.seed,
        response_format=response_format,
    )
    n = judge.run_judge(
        rows, args.output, grader, args.model,
        max_workers=args.max_workers,
    )
    summary = judge.count_verdicts(args.output)
    print(json.dumps({"new_verdicts_written": n, **summary}, indent=2))
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    cal_path = Path(args.input) if args.input else fetch.fetch_calibration()
    rows = metrics.load_rows(cal_path)
    verdicts = judge.load_verdicts(args.verdicts)

    judge_model = args.model
    if not judge_model:
        # Recover from the first record's judge_model field if not given.
        with open(args.verdicts, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    judge_model = json.loads(line).get("judge_model", "?")
                    break

    report = metrics.full_report(
        rows, verdicts,
        judge_model=judge_model or "?",
        bootstrap=args.bootstrap,
        n_resamples=args.n_resamples,
    )

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.report_json}", file=sys.stderr)

    md = metrics.render_markdown(report)
    if args.report_md:
        Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {args.report_md}", file=sys.stderr)
    else:
        print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m calibration.judge_validation")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="Download obgyn_meta_eval.jsonl")
    pf.add_argument("--cache-dir", default=fetch.DEFAULT_CACHE)
    pf.add_argument("--revision", default=fetch.DEFAULT_REVISION)
    pf.add_argument("--force", action="store_true")
    pf.set_defaults(func=cmd_fetch)

    pj = sub.add_parser("judge", help="Run a candidate judge over the calibration set")
    pj.add_argument("--base-url", required=True,
                    help="vLLM OpenAI-compatible endpoint, e.g. http://host:8000/v1")
    pj.add_argument("--model", required=True,
                    help="Model name as known to the endpoint")
    pj.add_argument("--output", required=True,
                    help="JSONL path for per-row verdicts (append-only, resumable)")
    pj.add_argument("--input", default=None,
                    help="Override calibration JSONL path")
    pj.add_argument("--api-key", default="EMPTY")
    pj.add_argument("--temperature", type=float, default=0.0)
    pj.add_argument("--max-tokens", type=int, default=None,
                    help="Output (completion) token cap. UNSET (default) lets "
                         "vLLM use the remaining context — safer for reasoning "
                         "models since max_tokens caps reasoning+final together.")
    pj.add_argument("--seed", type=int, default=None,
                    help="Sampling seed for reproducibility.")
    pj.add_argument("--json-schema", default=None,
                    help='Built-in preset "criterion_verdict" (recommended), '
                         'or an inline JSON dict for response_format.')
    pj.add_argument("--max-workers", type=int, default=judge.DEFAULT_MAX_WORKERS)
    pj.add_argument("--extra-body", default=None,
                    help='JSON dict forwarded to OpenAI extra_body, e.g. '
                         '{"reasoning_effort":"high"} for gpt-oss.')
    pj.add_argument("--limit", type=int, default=None,
                    help="Only judge first N rows (for smoke tests)")
    pj.set_defaults(func=cmd_judge)

    pm = sub.add_parser("metrics", help="Compute the metric report from a verdicts file")
    pm.add_argument("--verdicts", required=True)
    pm.add_argument("--input", default=None,
                    help="Override calibration JSONL path")
    pm.add_argument("--model", default=None,
                    help="Override judge name in the report (else taken from the verdicts file)")
    pm.add_argument("--report-json", default=None)
    pm.add_argument("--report-md", default=None)
    pm.add_argument("--bootstrap", action="store_true",
                    help="Compute 95%% bootstrap CIs (slower)")
    pm.add_argument("--n-resamples", type=int, default=1000)
    pm.set_defaults(func=cmd_metrics)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
