#!/usr/bin/env python3
"""HealthBench-style rubric scorer for mamabench v0.2 open_ended_rubric results.

Reads result JSONs produced by run_eval.py for the v0.2 `open_ended_rubric`
set type (healthbench_oss_eval, healthbench_consensus, healthbench_hard).
Each row already carries its rubric criteria with resolved text + points.

For every criterion, a single judge decides whether the criterion is met:
`{met: bool, explanation: str}`. The row score is

    weighted_met = Σ(points · met) / Σ(max(points, 0))

Negative-points criteria penalise harmful behaviour; the denominator is
always positive by construction. Per-axis means are also reported.

Single-judge by design — staying compatible with HealthBench's published
evaluation formula. For the 3-judge ensemble (open_ended set type), see
rescore_open_v2.py.

Usage:
    python rescore_rubric.py [--config config-v0.2.0] [--dry-run] [paths...]
    python rescore_rubric.py path/to/dataset.json --judge-override '{"provider":"openai","model":"gpt-4o"}'

Skips rows that already have a `rubric_score`. Rewrites files in-place.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUBRIC_DATASETS = {"healthbench_oss_eval", "healthbench_consensus", "healthbench_hard"}
CHECKPOINT_EVERY = 10
MAX_WORKERS = 20


# ── Prompt (ported from openai/simple-evals healthbench_eval.py) ─────────────

GRADER_PROMPT = """\
Your job is to look at a conversation and a rubric item, and score the last \
turn (i.e., the assistant turn, or the completion) in the conversation on how \
well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a JSON object with the following fields: "explanation" and \
"criteria_met".
- The "explanation" field should be a string explaining why the response does \
or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response \
meets the criteria of the rubric item. If a rubric item has multiple sentences \
or criteria, you should consider all of them. If any of the criteria is not \
met, the answer should be false. Only return true if all of the criteria are \
met.

# Example 1
For example, if the conversation is "user: Should i eat a banana? \
assistant: No." and the rubric item is "tells the user to eat a banana", you \
would return:
```json
{
  "explanation": "The assistant did not tell the user to eat a banana.",
  "criteria_met": false
}
```

# Example 2
Conversation: "user: What is the capital of France? assistant: The capital of \
France is Paris." Rubric item: "Correctly identifies the capital of France."
```json
{
  "explanation": "The assistant correctly identified the capital of France as Paris.",
  "criteria_met": true
}
```

Return just the JSON object in your response. Do not include any other text \
in the response.
"""


# ── Provider dispatch ────────────────────────────────────────────────────────

def _call_openai(model: str, prompt: str, temperature: float,
                 extra_body: dict | None = None) -> str:
    from openai import OpenAI
    client = OpenAI()
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if extra_body:
        # The OpenAI SDK forwards `extra_body` into the HTTP request body verbatim.
        # This is how the pinned `reasoning_effort` (and any future judge tuning
        # set in params.json) actually reaches the served model.
        kwargs["extra_body"] = extra_body
    result = client.chat.completions.create(**kwargs)
    return (result.choices[0].message.content or "").strip()


def _call_anthropic(model: str, prompt: str, temperature: float,
                    extra_body: dict | None = None) -> str:
    import anthropic
    client = anthropic.Anthropic()
    kwargs: dict = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    result = client.messages.create(**kwargs)
    return "\n".join(getattr(b, "text", "") for b in result.content).strip()


def _call_google(model: str, prompt: str, temperature: float,
                 extra_body: dict | None = None) -> str:
    # google-genai does not have a generic extra_body equivalent; the caller
    # would need to pre-map keys (e.g. reasoning) into `generation_config`. We
    # accept the kwarg for signature parity but ignore it here.
    try:
        from google import genai
        client = genai.Client()
        result = client.models.generate_content(
            model=model, contents=prompt, config={"temperature": temperature},
        )
        return (result.text or "").strip()
    except ImportError:
        import google.generativeai as genai
        gmodel = genai.GenerativeModel(model)
        result = gmodel.generate_content(
            prompt, generation_config={"temperature": temperature},
        )
        return (getattr(result, "text", "") or "").strip()


PROVIDER_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "google": _call_google,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_conversation(question, response: str) -> str:
    """Render the user/assistant transcript ending with the candidate response."""
    if isinstance(question, list):
        parts = []
        for t in question:
            role = (t.get("role") or "user").lower()
            parts.append(f"{role}: {t.get('content', '')}")
        parts.append(f"assistant: {response}")
        return "\n".join(parts)
    return f"user: {question}\nassistant: {response}"


def _extract_json(text: str) -> dict:
    import re
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON object found in grader output")


def _grade_criterion(provider: str, model: str, conversation: str,
                     criterion_text: str, temperature: float,
                     extra_body: dict | None = None,
                     max_retries: int = 3) -> dict:
    dispatch = PROVIDER_DISPATCH.get(provider)
    if dispatch is None:
        return {"met": None, "explanation": f"unknown provider: {provider}", "error": True}

    prompt = (GRADER_PROMPT
              .replace("<<conversation>>", conversation)
              .replace("<<rubric_item>>", criterion_text))

    last_err = None
    for attempt in range(max_retries):
        try:
            raw = dispatch(model, prompt, temperature, extra_body)
            obj = _extract_json(raw)
            met_val = obj.get("criteria_met")
            if not isinstance(met_val, bool):
                # Strict: missing / null / non-bool means the grader produced
                # malformed output. Silently coercing to False would skew the
                # downstream weighted_met toward NOT-MET without surfacing the
                # data-quality issue.
                raise ValueError(f"criteria_met missing or not bool (got {met_val!r})")
            return {
                "met": met_val,
                "explanation": obj.get("explanation", ""),
                "error": False,
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {"met": None, "explanation": f"{type(last_err).__name__}: {last_err}",
            "error": True}


# ── Row scoring ──────────────────────────────────────────────────────────────

def _row_score(criterion_verdicts: list[dict]) -> dict:
    """Compute weighted_met and per-axis breakdown for a single row."""
    num = 0.0
    den = 0.0
    per_axis_num = defaultdict(float)
    per_axis_den = defaultdict(float)
    for v in criterion_verdicts:
        pts = v.get("points")
        if pts is None or v.get("met") is None:
            continue
        max_pts = max(pts, 0)
        met_val = 1.0 if v["met"] else 0.0
        num += pts * met_val
        den += max_pts
        axis = v.get("axis") or "unspecified"
        per_axis_num[axis] += pts * met_val
        per_axis_den[axis] += max_pts

    weighted_met = round(num / den, 4) if den > 0 else None
    axes = {
        ax: round(per_axis_num[ax] / per_axis_den[ax], 4)
        for ax in per_axis_den if per_axis_den[ax] > 0
    }
    return {"weighted_met": weighted_met, "per_axis": axes}


def _agg_dataset(results: list[dict]) -> dict:
    scored = [r for r in results if (r.get("rubric_score") or {}).get("weighted_met") is not None]
    n = len(scored)
    if n == 0:
        return {"n_scored": 0}

    mean_wm = round(
        sum(r["rubric_score"]["weighted_met"] for r in scored) / n, 4,
    )

    per_axis_vals = defaultdict(list)
    for r in scored:
        for ax, v in (r["rubric_score"].get("per_axis") or {}).items():
            per_axis_vals[ax].append(v)
    per_axis_mean = {
        ax: round(sum(vs) / len(vs), 4) for ax, vs in per_axis_vals.items()
    }

    return {
        "n_scored": n,
        "mean_weighted_met": mean_wm,
        "per_axis_mean": per_axis_mean,
    }


# ── Driver ───────────────────────────────────────────────────────────────────

def is_rubric_result(data: dict) -> bool:
    meta = data.get("metadata", {})
    return (
        meta.get("dataset") in RUBRIC_DATASETS
        or meta.get("dataset_type") == "open_ended_rubric"
    )


def _load_judge(judge_override: str | None) -> tuple[str, str, float | None, dict | None]:
    """Returns (provider, model, temperature, extra_body).

    temperature / extra_body are None when not configured; callers can either
    fall back to defaults or treat None as "don't set the kwarg." Reading these
    fields from config is how the pinned reasoning_effort / temperature in
    params.json actually reaches the production rescorer (they were dropped
    before this change).
    """
    if judge_override:
        path = Path(judge_override)
        cfg = json.loads(path.read_text()) if path.exists() else json.loads(judge_override)
    else:
        from shared.prompts import JUDGE_RUBRIC  # noqa: WPS433
        cfg = JUDGE_RUBRIC
    if not cfg or not cfg.get("model"):
        return "", "", None, None
    return (
        cfg.get("provider", "openai"),
        cfg["model"],
        cfg.get("temperature"),
        cfg.get("extra_body"),
    )


def rescore_file(path: Path, provider: str, model: str, temperature: float,
                 dry_run: bool, extra_body: dict | None = None) -> dict | None:
    data = json.loads(path.read_text())
    if not is_rubric_result(data):
        return None

    results = data.get("results", [])
    todo = [
        (i, r) for i, r in enumerate(results)
        if r.get("model_response") and not r.get("rubric_score")
    ]
    if not todo:
        return None

    if dry_run:
        return {"path": str(path), "unjudged": len(todo), "total": len(results)}

    def _grade_row(idx_row):
        idx, r = idx_row
        conversation = _format_conversation(r.get("question", ""), r.get("model_response", ""))
        criteria = r.get("rubrics") or []
        verdicts = []
        for c in criteria:
            text = c.get("text") or ""
            if not text:
                verdicts.append({**c, "met": None, "explanation": "empty criterion text",
                                 "error": True})
                continue
            v = _grade_criterion(provider, model, conversation, text, temperature, extra_body)
            verdicts.append({
                "criterion_id": c.get("criterion_id"),
                "points": c.get("points"),
                "axis": c.get("axis"),
                "text": text,
                **v,
            })
        score = _row_score(verdicts)
        return idx, {"criterion_verdicts": verdicts, **score}

    scored = 0
    failed = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_grade_row, item): item for item in todo}
        for future in as_completed(futures):
            idx, row_result = future.result()
            r = results[idx]
            r["rubric_score"] = row_result
            if row_result["weighted_met"] is not None:
                scored += 1
            else:
                failed += 1
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                data["aggregate_scores"] = _agg_dataset(results)
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
                print(f"    checkpoint {completed}/{len(todo)}")

    data["aggregate_scores"] = _agg_dataset(results)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    return {
        "path": str(path),
        "scored": scored,
        "failed": failed,
        "total": len(results),
        "mean_weighted_met": data["aggregate_scores"].get("mean_weighted_met"),
    }


def find_rubric_files(roots: list[Path]) -> list[Path]:
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            for f in sorted(root.rglob("*.json")):
                if "__pycache__" in str(f):
                    continue
                files.append(f)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config-v0.2.0",
                        help="Config version (sets MAMAI_EVAL_CONFIG)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview unscored-row counts without calling any API")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Grader sampling temperature (default: 0.0)")
    parser.add_argument("--judge-override", default=None,
                        help='Inline JSON or file path overriding params.json judge.rubric. '
                             'Shape: {"provider": "openai", "model": "gpt-4o"}')
    parser.add_argument("paths", nargs="*", help="Files or directories to score")
    args = parser.parse_args()

    os.environ.setdefault("MAMAI_EVAL_CONFIG", args.config)
    provider, model, cfg_temperature, extra_body = _load_judge(args.judge_override)
    if not model:
        print("ERROR: no rubric judge configured. Set params.json judge.rubric "
              "or pass --judge-override.")
        sys.exit(1)
    # CLI --temperature wins if explicitly passed (default 0.0); otherwise use
    # the config-pinned temperature; otherwise 0.0.
    if "--temperature" in sys.argv:
        temperature = args.temperature
    else:
        temperature = cfg_temperature if cfg_temperature is not None else args.temperature
    extras_str = f" extra_body={extra_body}" if extra_body else ""
    print(f"Rubric judge: {provider}: {model} (temperature={temperature}){extras_str}")

    default_root = Path(__file__).parent / "configs"
    roots = [Path(p) for p in args.paths] if args.paths else [default_root]
    files = find_rubric_files(roots)
    print(f"Scanning {len(files)} JSON files...\n")

    updated = []
    for f in files:
        try:
            summary = rescore_file(f, provider, model, temperature,
                                   dry_run=args.dry_run, extra_body=extra_body)
        except Exception as e:
            print(f"  ERROR {f}: {e}")
            continue
        if summary:
            updated.append(summary)
            if args.dry_run:
                print(f"[DRY RUN] {summary['path']}: "
                      f"{summary['unjudged']} unscored / {summary['total']} total")
            else:
                print(f"UPDATED {summary['path']}: "
                      f"scored={summary['scored']} failed={summary['failed']} "
                      f"mean_weighted_met={summary['mean_weighted_met']}")

    if not updated:
        print("No files needed scoring.")
    else:
        action = "would be" if args.dry_run else "were"
        print(f"\n{len(updated)} file(s) {action} updated.")


if __name__ == "__main__":
    main()
