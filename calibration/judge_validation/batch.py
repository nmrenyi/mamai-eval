"""OpenAI Batch API support for the judge calibration pipeline.

The synchronous judge.py path uses chat.completions.create directly. Batch
is async: we format requests as JSONL, upload as a file, create a batch,
poll until complete, download results, and parse into our standard verdict
format. Batch API gives a 50% discount on both input and output tokens —
the only sensible way to run closed-source calibration at non-toy scale.

Use case: closed-source judges at api.openai.com. Not used for vLLM
(which doesn't have a Batch endpoint anyway).
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

# Reuse production grader prompt + helpers so the batch call shape matches
# rescore_rubric.py exactly. Same as judge.py does for the sync path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from end_to_end_eval.rescore_rubric import (  # noqa: E402
    GRADER_PROMPT,
    _extract_json as extract_json,
    _format_conversation as format_conversation,
)


# ── Build the JSONL batch input ───────────────────────────────────────────────


def build_batch_jsonl(
    rows: list[dict],
    model: str,
    response_format: dict | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    extra_body: dict | None = None,
) -> str:
    """Serialise rows to OpenAI Batch JSONL.

    Each line is one chat-completion request. `custom_id` carries the row's
    `_orig_idx` so we can map responses back to verdicts on download.

    Note on GPT-5 reasoning models: temperature/top_p are locked at 1, so
    we deliberately do NOT include them in the body. max_tokens is also
    deliberately omitted unless explicitly given (max_completion_tokens
    caps reasoning+final combined; the safer default is to let the model
    use whatever's left of max-model-len).
    """
    lines: list[str] = []
    for r in rows:
        prompt = (
            GRADER_PROMPT
            .replace("<<conversation>>", format_conversation(r["prompt"], r["completion"]))
            .replace("<<rubric_item>>", r["rubric"])
        )
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format is not None:
            body["response_format"] = response_format
        if seed is not None:
            body["seed"] = seed
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if max_completion_tokens is not None:
            body["max_completion_tokens"] = max_completion_tokens
        if extra_body:
            # Caller can override / add anything (e.g. service_tier).
            body.update(extra_body)

        lines.append(json.dumps({
            "custom_id": f"row-{r['_orig_idx']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }, ensure_ascii=False))

    return "\n".join(lines) + "\n"


# ── Lifecycle: upload → submit → wait → download ──────────────────────────────


def submit_batch(client, jsonl_content: str, completion_window: str = "24h") -> Any:
    """Upload the JSONL and create a batch. Returns the batch object."""
    fobj = io.BytesIO(jsonl_content.encode("utf-8"))
    fobj.name = "batch_input.jsonl"
    uploaded = client.files.create(file=fobj, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    print(
        f"Batch submitted: id={batch.id} input_file={uploaded.id} status={batch.status}",
        file=sys.stderr,
    )
    return batch


def wait_batch(
    client,
    batch_id: str,
    poll_interval: int = 15,
    timeout: int = 86400,
) -> Any:
    """Poll until the batch reaches a terminal state."""
    start = time.time()
    last_status = None
    while time.time() - start < timeout:
        batch = client.batches.retrieve(batch_id)
        rc = batch.request_counts
        elapsed = int(time.time() - start)
        if batch.status != last_status or elapsed % 60 == 0:
            print(
                f"[{elapsed:>5}s] batch={batch_id} status={batch.status} "
                f"completed={rc.completed}/{rc.total} failed={rc.failed}",
                file=sys.stderr,
            )
            last_status = batch.status
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            return batch
        time.sleep(poll_interval)
    raise TimeoutError(f"Batch {batch_id} did not reach terminal state in {timeout}s")


def download_output(client, file_id: str) -> str:
    """Download a batch output (or error) file as JSONL text."""
    return client.files.content(file_id).text


# ── Parse batch output → standard verdict records ────────────────────────────


def parse_batch_output(jsonl_content: str, judge_model: str) -> list[dict]:
    """Parse batch output JSONL into our standard verdict records.

    Output format matches what judge.run_judge writes — so the metrics
    pipeline doesn't need to know which path produced the verdicts.
    """
    verdicts: list[dict] = []
    for line in jsonl_content.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        custom_id = obj.get("custom_id", "")
        if not custom_id.startswith("row-"):
            continue
        row_index = int(custom_id[len("row-"):])

        api_error = obj.get("error")
        resp = obj.get("response", {}) or {}
        status_code = resp.get("status_code")
        if api_error is not None or status_code != 200:
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": None,
                "explanation": "",
                "error": f"batch_error: status={status_code} err={api_error}",
            })
            continue

        body = resp.get("body", {}) or {}
        choices = body.get("choices") or []
        content = ""
        if choices:
            msg = choices[0].get("message", {}) or {}
            content = (msg.get("content") or "").strip()

        if not content:
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": None,
                "explanation": "",
                "error": "empty_content",
            })
            continue

        try:
            parsed = extract_json(content)
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": bool(parsed.get("criteria_met")),
                "explanation": parsed.get("explanation", ""),
                "error": None,
            })
        except Exception as e:  # noqa: BLE001
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": None,
                "explanation": "",
                "error": f"parse_failed: {type(e).__name__}: {e}",
            })
    return verdicts


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_judge_batch(
    rows: list[dict],
    output_path: Path | str,
    judge_model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    response_format: dict | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    extra_body: dict | None = None,
    completion_window: str = "24h",
    poll_interval: int = 15,
) -> int:
    """End-to-end: build JSONL → upload → submit → poll → download → parse → write.

    Appends verdicts to `output_path` in the same format as judge.run_judge
    (one JSON line per verdict, fields: row_index, judge_model, criteria_met,
    explanation, error).
    """
    from openai import OpenAI

    client_kwargs: dict = {}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl = build_batch_jsonl(
        rows, judge_model,
        response_format=response_format,
        seed=seed,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        extra_body=extra_body,
    )
    print(f"Built batch input: {len(rows)} requests", file=sys.stderr)

    batch = submit_batch(client, jsonl, completion_window=completion_window)
    batch = wait_batch(client, batch.id, poll_interval=poll_interval)

    if batch.status != "completed":
        print(f"Batch did not complete cleanly: status={batch.status}", file=sys.stderr)
        if getattr(batch, "error_file_id", None):
            err = download_output(client, batch.error_file_id)
            print(f"=== error file (first 2000 chars) ===\n{err[:2000]}", file=sys.stderr)
        raise RuntimeError(f"Batch ended in non-completed state: {batch.status}")

    output_content = download_output(client, batch.output_file_id)
    verdicts = parse_batch_output(output_content, judge_model)

    with open(output_path, "a", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    n_good = sum(1 for v in verdicts if v["error"] is None)
    n_err = len(verdicts) - n_good
    print(
        f"Wrote {len(verdicts)} verdicts to {output_path} "
        f"(good={n_good}, errors={n_err})",
        file=sys.stderr,
    )
    return len(verdicts)
