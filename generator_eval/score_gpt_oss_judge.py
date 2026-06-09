"""gpt-oss-120b judge for faithfulness analysis — categorization & calibration.

Single backend, two modes:

  categorize  — applies the pinned CATEGORIZE_RUBRIC (from analyze_lynx_fails)
                to a lynx_fail_cases.json file. Emits one label object per case.
                Downstream consumer: analyze_lynx_fails aggregate.

  calibrate   — applies the pinned CALIBRATION_RUBRIC (from calibrate) to a
                calibration_blind.json file. Emits one verdict object per case.
                Downstream consumer: calibrate score.

Rubrics are imported, not redefined — single source of truth in their owning
scripts so the v0.2.0 numbers stay comparable with v0.1.0.

Inference via vLLM. gpt-oss-120b weights are MXFP4-native (designed to fit on
1×80GB GPU); on 2×A100 tensor-parallel for speed. The model is a reasoning
model: reasoning_effort="high" applied via the chat template's kwarg.

Usage:
  python -m generator_eval.score_gpt_oss_judge categorize \\
      --input  <run-dir>/lynx_fail_cases.json \\
      --output <run-dir>/lynx_fail_categories_gpt_oss.json

  python -m generator_eval.score_gpt_oss_judge calibrate \\
      --input  <run-dir>/calibration_blind.json \\
      --output <run-dir>/calibration_verdicts_gpt_oss.json
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator_eval.analyze_lynx_fails import (CATEGORIZE_RUBRIC,
                                               VALID_CATEGORIES)
from generator_eval.calibrate import CALIBRATION_RUBRIC

CHECKPOINT_CHUNK = 32


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _extract_json_object(text: str) -> dict | None:
    """Pull the first balanced {...} object out of the model's final answer.

    gpt-oss reliably emits JSON when asked, but its harmony output may also
    contain reasoning channel text. Greedy match on the first balanced object.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


# ── Mode dispatch ────────────────────────────────────────────────────────────
def _categorize_user_msg(case: dict) -> str:
    payload = {
        "query_id": case["query_id"],
        "query_text": case.get("query_text", ""),
        "context": case.get("context", ""),
        "answer": case.get("answer", ""),
        "lynx_reasoning": case.get("lynx_reasoning"),
    }
    return ("Categorize this case using the rubric above. Output a single "
            "JSON object — nothing else — with exactly these keys: "
            "query_id, category, secondary, justification, lynx_verdict, "
            "lynx_verdict_note.\n\nCASE:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2))


def _calibrate_user_msg(case: dict) -> str:
    payload = {
        "query_id": case["query_id"],
        "query_text": case.get("query_text", ""),
        "context": case.get("context", ""),
        "answer": case.get("answer", ""),
    }
    return ("Judge this case using the rubric above. Output a single JSON "
            "object — nothing else — with exactly these keys: query_id, "
            "verdict, reasoning.\n\nCASE:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2))


def _validate_categorize(obj: dict, expected_qid: str) -> tuple[dict | None, str | None]:
    if not isinstance(obj, dict):
        return None, "not a JSON object"
    if obj.get("query_id") != expected_qid:
        return None, f"query_id mismatch (got {obj.get('query_id')!r})"
    if obj.get("category") not in VALID_CATEGORIES:
        return None, f"invalid category {obj.get('category')!r}"
    # Coerce missing fields to defaults rather than reject.
    obj.setdefault("secondary", [])
    obj.setdefault("justification", "")
    obj.setdefault("lynx_verdict", "")
    obj.setdefault("lynx_verdict_note", "")
    return obj, None


def _validate_calibrate(obj: dict, expected_qid: str) -> tuple[dict | None, str | None]:
    if not isinstance(obj, dict):
        return None, "not a JSON object"
    if obj.get("query_id") != expected_qid:
        return None, f"query_id mismatch (got {obj.get('query_id')!r})"
    if obj.get("verdict") not in ("PASS", "FAIL"):
        return None, f"invalid verdict {obj.get('verdict')!r}"
    obj.setdefault("reasoning", "")
    return obj, None


MODES = {
    "categorize": {
        "rubric": CATEGORIZE_RUBRIC,
        "user_msg": _categorize_user_msg,
        "validate": _validate_categorize,
        "method_name": "gpt_oss_categorization",
    },
    "calibrate": {
        "rubric": CALIBRATION_RUBRIC,
        "user_msg": _calibrate_user_msg,
        "validate": _validate_calibrate,
        "method_name": "gpt_oss_calibration",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=list(MODES.keys()))
    parser.add_argument("--input", required=True,
                        help="JSON array of cases (lynx_fail_cases.json or "
                             "calibration_blind.json).")
    parser.add_argument("--output", required=True,
                        help="Output path. Explicit to avoid clobbering existing labels.")
    parser.add_argument("--model", default="openai/gpt-oss-120b",
                        help="HF model id. Default: openai/gpt-oss-120b.")
    parser.add_argument("--tensor-parallel", type=int, default=2,
                        help="vLLM tensor_parallel_size.")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="vLLM max_model_len. gpt-oss supports 128k; "
                             "32k is enough for our prompts + reasoning.")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max generation length. Reasoning trace + JSON.")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["low", "medium", "high"],
                        help="gpt-oss reasoning effort (chat template kwarg).")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit to N cases for smoke testing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing output and rescore from scratch.")
    args = parser.parse_args()

    mode_cfg = MODES[args.mode]
    rubric: str = mode_cfg["rubric"]
    build_user_msg = mode_cfg["user_msg"]
    validate = mode_cfg["validate"]

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        sys.exit(f"ERROR: input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases_all = json.loads(input_path.read_text())
    if not isinstance(cases_all, list):
        sys.exit(f"ERROR: {input_path} must be a JSON array")
    cases = cases_all[:args.max_questions] if args.max_questions else cases_all
    print(f"Mode:   {args.mode}")
    print(f"Input:  {input_path}  ({len(cases)}/{len(cases_all)} cases)")
    print(f"Output: {output_path}")

    # Resume from prior output if present (unless --overwrite).
    done_ids: set[str] = set()
    resume_results: list[dict] = []
    if output_path.exists() and args.overwrite:
        print("  --overwrite: starting from scratch")
    elif output_path.exists():
        prev = json.loads(output_path.read_text())
        # Output is either a bare list (final) or {metadata, results} (in-progress).
        prev_results = prev["results"] if isinstance(prev, dict) else prev
        resume_results = list(prev_results)
        done_ids = {r["query_id"] for r in resume_results if "query_id" in r}
        if len(resume_results) >= len(cases):
            print(f"  Already complete ({len(resume_results)}/{len(cases)}), nothing to do")
            return
        print(f"  Resuming: {len(resume_results)}/{len(cases)} already scored")
    pending = [c for c in cases if c["query_id"] not in done_ids]
    print(f"  Pending: {len(pending)}")

    print(f"\nLoading judge via vLLM: {args.model} (tp={args.tensor_parallel})")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    print("Judge loaded.\n")

    input_budget = args.max_model_len - args.max_new_tokens

    metadata = {
        "input_file": input_path.name,
        "input_file_sha256": _file_sha256(input_path),
        "method": mode_cfg["method_name"],
        "judge_model": args.model,
        "judge_family": "openai (gpt-oss open weights)",
        "reasoning_effort": args.reasoning_effort,
        "n_cases": len(cases),
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "sampling": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "max_model_len": args.max_model_len,
    }

    results: list[dict] = list(resume_results)
    t0 = time.time()

    def _save():
        # Final shape consumed by downstream tools is a bare JSON array of
        # label/verdict objects. Keep metadata on a sidecar.
        output_path.write_text(json.dumps(
            [r for r in results if r.get("_ok")],
            indent=2, ensure_ascii=False, default=str,
        ))
        sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
        sidecar.write_text(json.dumps(
            {"metadata": metadata, "results": results},
            indent=2, ensure_ascii=False, default=str,
        ))

    for chunk_start in range(0, len(pending), CHECKPOINT_CHUNK):
        chunk = pending[chunk_start:chunk_start + CHECKPOINT_CHUNK]

        prompts, meta = [], []
        for case in chunk:
            user_msg = build_user_msg(case)
            messages = [
                {"role": "system", "content": rubric},
                {"role": "user", "content": user_msg},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    reasoning_effort=args.reasoning_effort,
                )
            except TypeError:
                # Older transformers/templates may not accept reasoning_effort kwarg.
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            n_tok = len(tokenizer.encode(text))
            prompts.append(text)
            meta.append({"query_id": case["query_id"], "n_input_tokens": n_tok})

        fit_idx = [i for i, m in enumerate(meta) if m["n_input_tokens"] <= input_budget]
        gen_by_idx: dict[int, str] = {}
        if fit_idx:
            outs = llm.generate([prompts[i] for i in fit_idx], sampling)
            for i, o in zip(fit_idx, outs):
                gen_by_idx[i] = o.outputs[0].text

        for i, m in enumerate(meta):
            qid = m["query_id"]
            if i not in gen_by_idx:
                results.append({"query_id": qid, "_ok": False,
                                "note": f"input {m['n_input_tokens']} tok exceeds "
                                        f"budget {input_budget}",
                                "n_input_tokens": m["n_input_tokens"]})
                continue

            raw = gen_by_idx[i]
            obj = _extract_json_object(raw)
            if obj is None:
                results.append({"query_id": qid, "_ok": False,
                                "note": "no JSON object in output",
                                "raw_output": raw[:3000],
                                "n_input_tokens": m["n_input_tokens"]})
                continue

            # Force query_id onto the object before validating — judges sometimes
            # echo a slightly-different qid; we trust our own.
            obj["query_id"] = qid
            validated, err = validate(obj, qid)
            if validated is None:
                results.append({"query_id": qid, "_ok": False,
                                "note": err, "raw_output": raw[:3000],
                                "n_input_tokens": m["n_input_tokens"]})
                continue

            validated["_ok"] = True
            validated["n_input_tokens"] = m["n_input_tokens"]
            results.append(validated)

        _save()
        ok = sum(1 for r in results if r.get("_ok"))
        bad = len(results) - ok
        done = len(results) - len(resume_results)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(pending) - done) / rate / 60 if rate > 0 else float("inf")
        print(f"  [{done}/{len(pending)}] checkpoint — "
              f"ok={ok} bad={bad} rate={rate:.2f}/s ETA={eta:.1f}min")

    elapsed = time.time() - t0
    metadata["total_inference_time_s"] = round(elapsed, 1)
    n_ok = sum(1 for r in results if r.get("_ok"))
    n_bad = len(results) - n_ok
    metadata["n_ok"] = n_ok
    metadata["n_bad"] = n_bad

    if args.mode == "categorize":
        from collections import Counter
        dist = Counter(r["category"] for r in results if r.get("_ok"))
        metadata["distribution"] = {c: dist.get(c, 0) for c in VALID_CATEGORIES}
    else:
        from collections import Counter
        v = Counter(r["verdict"] for r in results if r.get("_ok"))
        metadata["verdict_distribution"] = dict(v)

    _save()
    print(f"\nSaved: {output_path}  ({n_ok} ok, {n_bad} failed)")
    if args.mode == "categorize":
        for c in VALID_CATEGORIES:
            print(f"  {c:22s}: {metadata['distribution'].get(c, 0)}")
    else:
        print(f"  verdicts: {metadata['verdict_distribution']}")


if __name__ == "__main__":
    main()
