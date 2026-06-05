"""OpenAI Responses API + Batch path for closed-source judge runs.

Replaces the earlier Chat Completions batch path. The architectural
shift: the model reasons once (internal CoT, billed as reasoning_tokens
but normally invisible), OpenAI's reasoning summarizer turns that into a
compact natural-language summary, and we capture
`{criteria_met, reasoning_summary}` instead of asking the model to
re-narrate its reasoning as an `explanation` field. One source of "why,"
no redundant generation.

Endpoint: /v1/responses (not /v1/chat/completions). The Batch API
supports both endpoints; OpenAI's recommended path for new reasoning-
model integrations is Responses.

Use case: closed-source judges at api.openai.com (gpt-5 family). vLLM
clusters keep using the sync chat-completions path in judge.py.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

# Reuse production grader prompt + helpers so the request shape matches
# rescore_rubric.py exactly (same prompt, same conversation formatting).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from end_to_end_eval.rescore_rubric import (  # noqa: E402
    GRADER_PROMPT,
    _extract_json as extract_json,
    _format_conversation as format_conversation,
)


# Minimal JSON schema for the Responses API: just the binary verdict.
# The model's reasoning is NOT asked for here — it surfaces separately
# via reasoning.summary in the response output[] array.
#
# Schema wrapper for Responses API uses `text.format` rather than the
# Chat Completions `response_format`. Inner schema dict is the same.
CRITERION_VERDICT_TEXT_FORMAT = {
    "type": "json_schema",
    "name": "criterion_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["criteria_met"],
        "properties": {
            "criteria_met": {"type": "boolean"},
        },
    },
}


# ── Build the JSONL batch input ───────────────────────────────────────────────


def build_batch_jsonl(
    rows: list[dict],
    model: str,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "auto",
    extra_body: dict | None = None,
) -> str:
    """Serialise rows to OpenAI Batch JSONL for the /v1/responses endpoint.

    Each line is one Responses-API request. `custom_id` carries the row's
    `_orig_idx` so we can map responses back to verdicts on download.

    Notes on GPT-5 reasoning models on the Responses API:
      - `reasoning.effort` (low/medium/high) controls reasoning depth.
      - `reasoning.summary` ("auto"/"concise"/"detailed") asks OpenAI to
        produce a natural-language summary of the internal CoT.
      - `text.format` is the Responses-API equivalent of `response_format`
        for structured JSON output.
      - `max_output_tokens` is intentionally omitted: it caps reasoning+
        final combined; the safer default is to let the server decide.
      - Temperature/top_p are locked at 1 on reasoning models — omitted.
      - `seed` is NOT supported by the Responses API (rejected with
        "Unknown parameter: 'seed'"). With temperature locked at 1 and
        reasoning being non-deterministic anyway, seed wouldn't buy
        meaningful reproducibility even if accepted — so we simply don't
        include it.
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
            "input": [{"role": "user", "content": prompt}],
            "reasoning": {
                "effort": reasoning_effort,
                "summary": reasoning_summary,
            },
            "text": {
                "format": CRITERION_VERDICT_TEXT_FORMAT,
            },
        }
        if extra_body:
            body.update(extra_body)

        lines.append(json.dumps({
            "custom_id": f"row-{r['_orig_idx']}",
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }, ensure_ascii=False))

    return "\n".join(lines) + "\n"


# ── Lifecycle: upload → submit → wait → download ──────────────────────────────


def submit_batch(client, jsonl_content: str, completion_window: str = "24h") -> Any:
    """Upload the JSONL and create a batch on the /v1/responses endpoint."""
    fobj = io.BytesIO(jsonl_content.encode("utf-8"))
    fobj.name = "batch_input.jsonl"
    uploaded = client.files.create(file=fobj, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    print(
        f"Batch submitted: id={batch.id} input_file={uploaded.id} status={batch.status} endpoint=/v1/responses",
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


# ── Parse Responses-API batch output → standard verdict records ───────────────


def _extract_usage(body: dict) -> dict | None:
    """Pull the Responses-API usage block into a compact dict.

    Field names differ from Chat Completions: input_tokens / output_tokens
    instead of prompt_tokens / completion_tokens, and
    output_tokens_details.reasoning_tokens instead of
    completion_tokens_details.reasoning_tokens.
    """
    u = body.get("usage")
    if not isinstance(u, dict):
        return None
    out = {
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "total_tokens": u.get("total_tokens"),
    }
    otd = u.get("output_tokens_details")
    if isinstance(otd, dict):
        rt = otd.get("reasoning_tokens")
        if rt is not None:
            out["reasoning_tokens"] = rt
    itd = u.get("input_tokens_details")
    if isinstance(itd, dict):
        ct = itd.get("cached_tokens")
        if ct:
            out["cached_tokens"] = ct
    return out


def _extract_reasoning_summary(output_blocks: list) -> str | None:
    """Concatenate all reasoning summary text blocks in the output array.

    Responses-API output is a list of typed blocks: `reasoning` blocks
    carry the (optional) summary OpenAI generated from the internal CoT;
    `message` blocks carry the model's actual answer. There can be
    multiple reasoning blocks; we join their summary_text segments.
    """
    parts: list[str] = []
    for block in output_blocks:
        if block.get("type") == "reasoning":
            for s in block.get("summary", []) or []:
                if s.get("type") == "summary_text":
                    txt = s.get("text", "")
                    if txt:
                        parts.append(txt)
    return "\n\n".join(parts) if parts else None


def _extract_message_content(output_blocks: list) -> str:
    """Return the model's textual answer (the JSON our schema enforced)."""
    for block in output_blocks:
        if block.get("type") == "message":
            for c in block.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    return (c.get("text") or "").strip()
    return ""


def parse_batch_output(jsonl_content: str, judge_model: str) -> list[dict]:
    """Parse Responses-API batch output JSONL into verdict records.

    Verdict schema:
      row_index          : int     (recovered from custom_id "row-N")
      judge_model        : str     (caller-supplied)
      criteria_met       : bool|None  (None on error)
      reasoning_summary  : str|None   (OpenAI-generated CoT summary)
      error              : str|None
      status             : str|None   ("completed", "incomplete", ...)
      usage              : dict|None  (input/output/total/reasoning tokens)

    The metrics pipeline only reads `criteria_met` and `error`; the other
    fields are diagnostic (spend, deliberation visibility, debugging).
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
            # Two failure shapes:
            #   (1) per-request 4xx — actual error is at response.body.error.message
            #   (2) batch-level error — top-level `error` is populated
            inner_err = ((resp.get("body") or {}).get("error")) if isinstance(resp.get("body"), dict) else None
            if inner_err and inner_err.get("message"):
                err_str = (
                    f"http_{status_code}: {inner_err.get('code') or inner_err.get('type')}: "
                    f"{inner_err.get('message')}"
                )
            else:
                err_str = f"batch_error: status={status_code} err={api_error}"
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": None,
                "reasoning_summary": None,
                "error": err_str,
                "status": None,
                "usage": None,
            })
            continue

        body = resp.get("body", {}) or {}
        output_blocks = body.get("output", []) or []
        reasoning_summary = _extract_reasoning_summary(output_blocks)
        content = _extract_message_content(output_blocks)
        status = body.get("status")
        usage = _extract_usage(body)

        if not content:
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": None,
                "reasoning_summary": reasoning_summary,
                "error": f"empty_content (status={status})",
                "status": status,
                "usage": usage,
            })
            continue

        try:
            parsed = extract_json(content)
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": bool(parsed.get("criteria_met")),
                "reasoning_summary": reasoning_summary,
                "error": None,
                "status": status,
                "usage": usage,
            })
        except Exception as e:  # noqa: BLE001
            verdicts.append({
                "row_index": row_index,
                "judge_model": judge_model,
                "criteria_met": None,
                "reasoning_summary": reasoning_summary,
                "error": f"parse_failed: {type(e).__name__}: {e}",
                "status": status,
                "usage": usage,
            })
    return verdicts


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_judge_batch(
    rows: list[dict],
    output_path: Path | str,
    judge_model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "auto",
    extra_body: dict | None = None,
    completion_window: str = "24h",
    poll_interval: int = 15,
) -> int:
    """End-to-end: build JSONL → upload → submit → poll → download → parse → write.

    Appends verdicts to `output_path`, one JSON line per row.

    Failure handling: even when the batch as a whole reaches `status=completed`,
    individual requests may have failed (e.g. a bad parameter caused all 20 to
    400). In that case OpenAI sets `output_file_id=None` and provides an
    error_file_id with the per-row error details — we download that, parse the
    same way (`parse_batch_output` recognises the non-200 status shape) so the
    failures land in the verdicts JSONL as error records instead of crashing.
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
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        extra_body=extra_body,
    )
    print(f"Built batch input: {len(rows)} requests", file=sys.stderr)

    batch = submit_batch(client, jsonl, completion_window=completion_window)
    batch = wait_batch(client, batch.id, poll_interval=poll_interval)

    if batch.status not in {"completed"}:
        print(f"Batch did not complete cleanly: status={batch.status}", file=sys.stderr)
        if getattr(batch, "error_file_id", None):
            err = download_output(client, batch.error_file_id)
            print(f"=== error file (first 2000 chars) ===\n{err[:2000]}", file=sys.stderr)
        raise RuntimeError(f"Batch ended in non-completed state: {batch.status}")

    # Collect verdicts from BOTH files (output: successes; error: failures).
    # Either may be None depending on the run's success/failure mix.
    output_content = (
        download_output(client, batch.output_file_id) if batch.output_file_id else ""
    )
    error_content = (
        download_output(client, batch.error_file_id) if batch.error_file_id else ""
    )
    verdicts = parse_batch_output(output_content, judge_model) + \
               parse_batch_output(error_content, judge_model)
    verdicts.sort(key=lambda v: v["row_index"])

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
    if n_err and not n_good:
        # Surface the first error message so the operator sees WHY.
        first_err = next((v for v in verdicts if v["error"]), None)
        if first_err:
            print(f"First error: {first_err['error']}", file=sys.stderr)
    return len(verdicts)
