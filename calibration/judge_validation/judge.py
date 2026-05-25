"""Judge invocation: run a candidate judge over the calibration triples.

Reuses GRADER_PROMPT + helpers from end_to_end_eval.rescore_rubric so the
calibration uses the EXACT prompt path as production scoring — no drift.

Output format: append-only JSONL, one verdict per line:
    {"row_index": int, "judge_model": str, "criteria_met": bool|None,
     "explanation": str, "error": str|None, "raw": str (optional)}

Resumable: re-running with the same output path skips row_index values
already present in the file.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import the EXACT production prompt + helpers. If these change in
# rescore_rubric.py the calibration follows automatically — that's the
# point of importing rather than duplicating.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from end_to_end_eval.rescore_rubric import (  # noqa: E402
    GRADER_PROMPT,
    _extract_json as extract_json,
    _format_conversation as format_conversation,
)

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_WORKERS = 20


# Strict JSON-schema for the rubric verdict. Pass as `response_format` to the
# OpenAI/vLLM endpoint to force every reply to be parseable structured output —
# eliminates the ~1.5% parse-error rate we saw on the first gpt-oss-120b run.
# Reasoning models with `--enable-reasoning + --reasoning-parser <name>` apply
# this schema to the FINAL channel only; the reasoning channel stays free-form.
CRITERION_VERDICT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "criterion_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["explanation", "criteria_met"],
            "properties": {
                "explanation": {"type": "string"},
                "criteria_met": {"type": "boolean"},
            },
        },
    },
}


# ── Graders (callable: prompt -> raw text) ───────────────────────────────────


def make_openai_grader(
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
    seed: int | None = None,
    response_format: dict | None = None,
):
    """Build a grader that posts to an OpenAI-compatible endpoint (vLLM).

    `extra_body` is forwarded to OpenAI SDK's `extra_body` kwarg — use it for
    vLLM/reasoning-model specifics (e.g. {"reasoning_effort": "medium"} on
    gpt-oss; {"chat_template_kwargs": {"enable_thinking": True}} on Nemotron).

    `max_tokens=None` (default) omits the field from the request so vLLM uses
    the remaining context window (--max-model-len minus input). Safer for
    reasoning models since max_tokens caps reasoning+final combined; setting
    it too tight truncates the JSON.

    `response_format` accepts an OpenAI-style structured-output spec (e.g.
    CRITERION_VERDICT_SCHEMA above). Strongly recommended — guarantees
    parseable JSON every call.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    def grader(prompt: str) -> str:
        kw: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        if seed is not None:
            kw["seed"] = seed
        if response_format is not None:
            kw["response_format"] = response_format
        if extra_body:
            kw["extra_body"] = extra_body
        r = client.chat.completions.create(**kw)
        return (r.choices[0].message.content or "").strip()

    return grader


# ── Single-row grading (pure: takes a grader callable) ───────────────────────


def grade_row(row: dict, grader, max_retries: int = 3) -> dict:
    """Apply the rubric grader prompt to one calibration row.

    `grader` is a callable: `(prompt: str) -> raw: str`. Returns a verdict
    dict; on parse/API failure across retries, criteria_met is None and
    `error` carries the exception summary.
    """
    convo = format_conversation(row["prompt"], row["completion"])
    prompt = (
        GRADER_PROMPT
        .replace("<<conversation>>", convo)
        .replace("<<rubric_item>>", row["rubric"])
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = grader(prompt)
            obj = extract_json(raw)
            return {
                "row_index": row["_orig_idx"],
                "criteria_met": bool(obj.get("criteria_met")),
                "explanation": obj.get("explanation", ""),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {
        "row_index": row["_orig_idx"],
        "criteria_met": None,
        "explanation": "",
        "error": f"{type(last_err).__name__}: {last_err}",
    }


# ── Run over all rows ────────────────────────────────────────────────────────


def _load_existing_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done: set[int] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    done.add(obj["row_index"])
                except (json.JSONDecodeError, KeyError):
                    pass  # ignore corrupt/partial lines
    return done


def run_judge(
    rows: list[dict],
    output_path: Path | str,
    grader,
    judge_model: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress_every: int = 100,
    quiet: bool = False,
) -> int:
    """Grade every row whose row_index is not yet in `output_path`.

    Writes one JSONL verdict per row, with `judge_model` stamped onto each.
    Returns the count of NEW verdicts written (so a re-run shows 0).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_existing_indices(output_path)
    todo = [r for r in rows if r["_orig_idx"] not in done]
    if not quiet:
        print(
            f"Judge {judge_model}: {len(done)} already done · {len(todo)} to do",
            file=sys.stderr,
        )

    if not todo:
        return 0

    lock = threading.Lock()
    written = 0
    with open(output_path, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(grade_row, r, grader) for r in todo]
            for fut in as_completed(futures):
                v = fut.result()
                v["judge_model"] = judge_model
                with lock:
                    out.write(json.dumps(v, ensure_ascii=False) + "\n")
                    out.flush()
                    written += 1
                    if not quiet and written % progress_every == 0:
                        print(f"  ... {written}/{len(todo)}", file=sys.stderr)
    if not quiet:
        print(f"Done. Wrote {written} verdicts to {output_path}", file=sys.stderr)
    return written


# ── Load verdicts back for metrics ───────────────────────────────────────────


def load_verdicts(verdicts_path: Path | str) -> dict[int, bool]:
    """Read JSONL verdicts, return {row_index: criteria_met} dropping errors."""
    verdicts: dict[int, bool] = {}
    with open(verdicts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error") is not None:
                continue
            cm = obj.get("criteria_met")
            if cm is None:
                continue
            verdicts[obj["row_index"]] = bool(cm)
    return verdicts


def count_verdicts(verdicts_path: Path | str) -> dict:
    """Quick summary of a verdicts file — total / good / errors / met-rate."""
    total = good = errors = met = 0
    with open(verdicts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            if obj.get("error") is not None:
                errors += 1
                continue
            good += 1
            if obj.get("criteria_met"):
                met += 1
    return {
        "total": total,
        "good": good,
        "errors": errors,
        "judge_met_rate": round(met / good, 4) if good else None,
    }
