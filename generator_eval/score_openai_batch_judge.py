"""OpenAI Batch API judge — categorize + calibrate at 50% off standard rates.

Mirrors generator_eval/score_gpt_oss_judge.py but targets the OpenAI API
via Batch — designed for one-shot eval workloads where 24h turnaround is
acceptable in exchange for half the per-token cost.

Two modes:
  categorize  — applies the pinned CATEGORIZE_RUBRIC (from analyze_lynx_fails)
                to lynx_fail_cases.json. Default reasoning_effort=medium.
  calibrate   — applies the pinned CALIBRATION_RUBRIC (from calibrate.py) to
                calibration_blind.json. Default reasoning_effort=high.

Rubrics are imported, not redefined — single source of truth in their owning
scripts so v0.2.0 numbers stay comparable across judges.

Structured Outputs (json_schema mode, strict=true) eliminates parse failures
that hurt the gpt-oss-120b attempt.

Workflow:
  1. Build a Batch input JSONL (one request per case with custom_id=query_id)
  2. Upload via Files API
  3. Create the batch (endpoint /v1/chat/completions, window 24h)
  4. Poll status until completed/failed/expired
  5. Download output JSONL, parse to the existing per-case schema, write
     <output>.json + <output>.meta.json

Usage:
  python -m generator_eval.score_openai_batch_judge categorize \\
      --input  <run-dir>/lynx_fail_cases.json \\
      --output <run-dir>/lynx_fail_categories.json

  python -m generator_eval.score_openai_batch_judge calibrate \\
      --input  <run-dir>/calibration_blind.json \\
      --output <run-dir>/calibration_verdicts.json

Resume a job in progress:
  python -m generator_eval.score_openai_batch_judge wait \\
      --batch-id <id> --mode {categorize|calibrate} \\
      --output <path>
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator_eval.analyze_lynx_fails import (CATEGORIZE_RUBRIC,
                                               VALID_CATEGORIES)
from generator_eval.calibrate import CALIBRATION_RUBRIC

POLL_INTERVAL_SEC = 30


# ── JSON Schemas (Structured Outputs) ────────────────────────────────────────
CATEGORIZE_SCHEMA = {
    "name": "FaithfulnessFailCategory",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query_id": {"type": "string"},
            "category": {"type": "string", "enum": VALID_CATEGORIES},
            "secondary": {"type": "array", "items": {"type": "string"}},
            "justification": {"type": "string"},
            "lynx_verdict": {"type": "string", "enum": ["sound", "questionable"]},
            "lynx_verdict_note": {"type": "string"},
        },
        "required": ["query_id", "category", "secondary", "justification",
                     "lynx_verdict", "lynx_verdict_note"],
    },
}

CALIBRATE_SCHEMA = {
    "name": "FaithfulnessCalibrationVerdict",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
            "reasoning": {"type": "string"},
        },
        "required": ["query_id", "verdict", "reasoning"],
    },
}


# ── Mode dispatch ────────────────────────────────────────────────────────────
def _categorize_user_msg(case: dict) -> str:
    payload = {
        "query_id": case["query_id"],
        "query_text": case.get("query_text", ""),
        "context": case.get("context", ""),
        "answer": case.get("answer", ""),
        "lynx_reasoning": case.get("lynx_reasoning"),
    }
    return ("Categorize this case using the rubric above.\n\nCASE:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2))


def _calibrate_user_msg(case: dict) -> str:
    payload = {
        "query_id": case["query_id"],
        "query_text": case.get("query_text", ""),
        "context": case.get("context", ""),
        "answer": case.get("answer", ""),
    }
    return ("Judge this case using the rubric above.\n\nCASE:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2))


def _validate_categorize(obj: dict, expected_qid: str) -> tuple[dict | None, str | None]:
    # Structured Outputs guarantees the schema, but defend against the API
    # echoing a slightly-mangled qid (e.g. trimming whitespace).
    if obj.get("query_id") != expected_qid:
        obj["query_id"] = expected_qid  # we trust our own id
    if obj.get("category") not in VALID_CATEGORIES:
        return None, f"invalid category {obj.get('category')!r}"
    return obj, None


def _validate_calibrate(obj: dict, expected_qid: str) -> tuple[dict | None, str | None]:
    if obj.get("query_id") != expected_qid:
        obj["query_id"] = expected_qid
    if obj.get("verdict") not in ("PASS", "FAIL"):
        return None, f"invalid verdict {obj.get('verdict')!r}"
    return obj, None


MODES = {
    "categorize": {
        "rubric": CATEGORIZE_RUBRIC,
        "user_msg": _categorize_user_msg,
        "schema": CATEGORIZE_SCHEMA,
        "validate": _validate_categorize,
        "default_effort": "medium",
        "method_name": "openai_batch_categorization",
    },
    "calibrate": {
        "rubric": CALIBRATION_RUBRIC,
        "user_msg": _calibrate_user_msg,
        "schema": CALIBRATE_SCHEMA,
        "validate": _validate_calibrate,
        "default_effort": "high",
        "method_name": "openai_batch_calibration",
    },
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _build_batch_jsonl(cases: list[dict], mode_cfg: dict, model: str,
                       effort: str, max_output_tokens: int) -> bytes:
    """One JSON line per case; matches OpenAI Batch input spec."""
    rubric = mode_cfg["rubric"]
    user_msg_fn = mode_cfg["user_msg"]
    schema = mode_cfg["schema"]

    buf = io.BytesIO()
    for case in cases:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": rubric},
                {"role": "user", "content": user_msg_fn(case)},
            ],
            "reasoning_effort": effort,
            "max_completion_tokens": max_output_tokens,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        line = {
            "custom_id": case["query_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        buf.write(json.dumps(line, ensure_ascii=False).encode("utf-8"))
        buf.write(b"\n")
    return buf.getvalue()


def _submit_batch(client, mode: str, mode_cfg: dict, cases: list[dict],
                  model: str, effort: str, max_output_tokens: int) -> str:
    """Upload input + create batch. Returns batch_id."""
    jsonl_bytes = _build_batch_jsonl(cases, mode_cfg, model, effort,
                                     max_output_tokens)
    n_bytes = len(jsonl_bytes)
    print(f"  built batch JSONL: {len(cases)} requests, "
          f"{n_bytes/1024:.1f} KiB")

    f = client.files.create(
        file=(f"{mode}_input.jsonl", jsonl_bytes),
        purpose="batch",
    )
    print(f"  uploaded input file_id={f.id}")

    b = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"workflow": f"mamai_faithfulness_{mode}"},
    )
    print(f"  batch created: id={b.id}  status={b.status}")
    return b.id


def _poll_batch(client, batch_id: str) -> object:
    """Poll until terminal. Returns the final batch object."""
    last_print = 0
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] terminal: {b.status}")
            return b

        # Throttle log output; print every other poll if still running.
        now = time.time()
        if now - last_print >= POLL_INTERVAL_SEC * 1.5:
            counts = b.request_counts
            done = (counts.completed or 0) if counts else 0
            failed = (counts.failed or 0) if counts else 0
            total = (counts.total or 0) if counts else 0
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] status={b.status}  "
                  f"done={done}/{total} failed={failed}")
            last_print = now
        time.sleep(POLL_INTERVAL_SEC)


def _parse_batch_output(client, output_file_id: str, cases: list[dict],
                        mode_cfg: dict) -> tuple[list[dict], list[dict]]:
    """Download batch output JSONL, parse to per-case label/verdict objects.

    Returns (ok_results, all_results). all_results includes failed rows with
    _ok=False; ok_results is what downstream tools consume.
    """
    raw = client.files.content(output_file_id).read()
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    print(f"  downloaded output: {len(lines)} response lines")

    validate = mode_cfg["validate"]
    case_by_id = {c["query_id"]: c for c in cases}
    all_results: list[dict] = []
    for ln in lines:
        rec = json.loads(ln)
        qid = rec.get("custom_id")
        if not qid:
            continue
        resp = rec.get("response")
        if not resp or resp.get("status_code") != 200:
            all_results.append({
                "query_id": qid, "_ok": False,
                "note": f"http {resp.get('status_code') if resp else '<no resp>'}",
                "raw": (resp or {}).get("body"),
            })
            continue
        body = resp["body"]
        choices = body.get("choices", [])
        if not choices:
            all_results.append({"query_id": qid, "_ok": False,
                                "note": "no choices in response"})
            continue
        msg = choices[0].get("message", {})
        finish = choices[0].get("finish_reason")
        content = msg.get("content")
        # Refusal channel (Structured Outputs spec).
        if msg.get("refusal"):
            all_results.append({"query_id": qid, "_ok": False,
                                "note": f"model refusal: {msg['refusal'][:200]}",
                                "finish_reason": finish})
            continue
        if not content:
            all_results.append({"query_id": qid, "_ok": False,
                                "note": "empty content",
                                "finish_reason": finish})
            continue
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as e:
            all_results.append({"query_id": qid, "_ok": False,
                                "note": f"json decode: {e}",
                                "raw_content": content[:500]})
            continue
        validated, err = validate(obj, qid)
        if validated is None:
            all_results.append({"query_id": qid, "_ok": False, "note": err,
                                "raw_content": content[:500]})
            continue

        usage = body.get("usage", {}) or {}
        validated["_ok"] = True
        validated["finish_reason"] = finish
        validated["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {})
                                .get("reasoning_tokens"),
        }
        all_results.append(validated)

    ok_results = [r for r in all_results if r.get("_ok")]

    # Sort to match input order (Batch output ordering is not guaranteed).
    order = {c["query_id"]: i for i, c in enumerate(cases)}
    ok_results.sort(key=lambda r: order.get(r["query_id"], 1_000_000))
    all_results.sort(key=lambda r: order.get(r["query_id"], 1_000_000))

    # NOTE: don't strip auxiliary keys here — cmd_wait needs `usage` for the
    # cost tally, and `_ok` to filter OK rows. Stripping is done at save time.
    return ok_results, all_results


def _save(output_path: Path, ok_results: list[dict],
          all_results: list[dict], metadata: dict) -> None:
    # Downstream tools (aggregate/score) only care about the per-case label
    # or verdict fields — strip auxiliary keys for the clean output file.
    clean = []
    for r in ok_results:
        c = {k: v for k, v in r.items()
             if k not in ("_ok", "finish_reason", "usage")}
        clean.append(c)
    output_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    # Sidecar keeps everything (usage + raw notes for failed rows).
    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(
        {"metadata": metadata, "results": all_results},
        indent=2, ensure_ascii=False,
    ))


def cmd_submit(args) -> None:
    from openai import OpenAI
    client = OpenAI()
    mode_cfg = MODES[args.mode]
    effort = args.reasoning_effort or mode_cfg["default_effort"]

    input_path = Path(args.input)
    cases_all = json.loads(input_path.read_text())
    if not isinstance(cases_all, list):
        sys.exit(f"ERROR: {input_path} must be a JSON array")
    cases = cases_all[:args.max_questions] if args.max_questions else cases_all
    print(f"Mode:    {args.mode}")
    print(f"Input:   {input_path}  ({len(cases)}/{len(cases_all)} cases)")
    print(f"Output:  {args.output}")
    print(f"Model:   {args.model}  reasoning_effort={effort}  "
          f"max_completion_tokens={args.max_output_tokens}")
    print()
    print("Submitting batch...")
    batch_id = _submit_batch(client, args.mode, mode_cfg, cases,
                             args.model, effort, args.max_output_tokens)

    # Persist the batch state in a sidecar next to the output so a future
    # `wait` invocation can pick up the right id without the user copying it.
    state_path = Path(args.output).with_suffix(".batch_state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "batch_id": batch_id,
        "mode": args.mode,
        "model": args.model,
        "reasoning_effort": effort,
        "max_completion_tokens": args.max_output_tokens,
        "input_file": input_path.name,
        "input_file_sha256": _file_sha256(input_path),
        "n_cases": len(cases),
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"\nState saved: {state_path}")
    print(f"\nTo wait + finalize:")
    print(f"  python -m generator_eval.score_openai_batch_judge wait \\")
    print(f"      --batch-id {batch_id} --mode {args.mode} \\")
    print(f"      --input {input_path} --output {args.output}")


def cmd_wait(args) -> None:
    from openai import OpenAI
    client = OpenAI()
    mode_cfg = MODES[args.mode]

    input_path = Path(args.input)
    cases = json.loads(input_path.read_text())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Waiting on batch {args.batch_id}...")
    b = _poll_batch(client, args.batch_id)
    if b.status != "completed":
        sys.exit(f"ERROR: batch ended in non-completed status {b.status!r}; "
                 f"errors={b.errors}")

    print(f"Parsing batch output (file_id={b.output_file_id})...")
    ok_results, all_results = _parse_batch_output(
        client, b.output_file_id, cases, mode_cfg)

    n_ok = len(ok_results)
    n_bad = len(all_results) - n_ok

    # Cost: tally usage from the sidecar.
    total_prompt = sum((r.get("usage", {}) or {}).get("prompt_tokens") or 0
                       for r in all_results if r.get("_ok"))
    total_completion = sum((r.get("usage", {}) or {}).get("completion_tokens") or 0
                           for r in all_results if r.get("_ok"))
    total_reasoning = sum((r.get("usage", {}) or {}).get("reasoning_tokens") or 0
                          for r in all_results if r.get("_ok"))

    metadata = {
        "batch_id": args.batch_id,
        "mode": args.mode,
        "method": mode_cfg["method_name"],
        "judge_model": "gpt-5",
        "judge_family": "openai (gpt-5)",
        "judge_endpoint": "Batch API",
        "n_cases": len(cases),
        "n_ok": n_ok,
        "n_bad": n_bad,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "usage_totals": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "reasoning_tokens": total_reasoning,
        },
    }
    if args.mode == "categorize":
        from collections import Counter
        dist = Counter(r["category"] for r in ok_results)
        metadata["distribution"] = {c: dist.get(c, 0) for c in VALID_CATEGORIES}
    else:
        from collections import Counter
        v = Counter(r["verdict"] for r in ok_results)
        metadata["verdict_distribution"] = dict(v)

    _save(output_path, ok_results, all_results, metadata)
    print(f"\nSaved: {output_path}  ({n_ok} ok, {n_bad} failed)")
    if args.mode == "categorize":
        for c in VALID_CATEGORIES:
            print(f"  {c:22s}: {metadata['distribution'].get(c, 0)}")
    else:
        print(f"  verdicts: {metadata['verdict_distribution']}")

    # Rough cost at gpt-5 batch tier ($0.625/M input, $5/M output).
    in_cost = total_prompt * 0.625 / 1_000_000
    out_cost = total_completion * 5.0 / 1_000_000
    print(f"\n  usage: prompt={total_prompt:,}  completion={total_completion:,} "
          f"(reasoning={total_reasoning:,})")
    print(f"  cost (batch tier, gpt-5): input ${in_cost:.3f} + "
          f"output ${out_cost:.3f} = ${in_cost+out_cost:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common_io = lambda p: (p.add_argument("--input", required=True),
                           p.add_argument("--output", required=True))
    common_model = lambda p: (
        p.add_argument("--model", default="gpt-5"),
        p.add_argument("--reasoning-effort", default=None,
                       choices=["low", "medium", "high"]),
        p.add_argument("--max-output-tokens", type=int, default=4096))

    for mode in ("categorize", "calibrate"):
        ps = sub.add_parser(mode, help=f"submit + wait, {mode} mode")
        ps.set_defaults(cmd=mode)
        ps.add_argument("mode_alias", nargs="?", default=mode,
                        help=argparse.SUPPRESS)
        common_io(ps)
        common_model(ps)
        ps.add_argument("--max-questions", type=int, default=None)
        ps.add_argument("--submit-only", action="store_true",
                        help="Submit and exit (don't wait). Resume with `wait`.")

    pw = sub.add_parser("wait", help="poll an existing batch + finalize")
    pw.add_argument("--batch-id", required=True)
    pw.add_argument("--mode", required=True, choices=["categorize", "calibrate"])
    pw.add_argument("--input", required=True)
    pw.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.cmd in ("categorize", "calibrate"):
        args.mode = args.cmd
        cmd_submit(args)
        if not args.submit_only:
            # Hand off to wait using the batch_id we just persisted.
            state = json.loads(
                Path(args.output).with_suffix(".batch_state.json").read_text())
            args.batch_id = state["batch_id"]
            cmd_wait(args)
    elif args.cmd == "wait":
        cmd_wait(args)


if __name__ == "__main__":
    main()
