"""
HuggingFace dataset loader for mamabench v0.2.

Extracted from end_to_end_eval/run_eval.py so the same row-loading +
normalisation code can be used by the generation runner, the on-device
runner, the future faithfulness runner, etc.

Module name is deliberately NOT `datasets` (which would shadow the
HuggingFace `datasets` package on `from datasets import load_dataset`).
"""

import json
import string


# Dataset registry: name → (hf_config, set_type)
HF_CONFIGS = {
    "medmcqa":               ("medmcqa",               "mcq"),
    "medqa_usmle":           ("medqa_usmle",           "mcq"),
    "afrimedqa":             ("afrimedqa",             "mcq"),
    "kenya":                 ("kenya",                 "open_ended"),
    "whb":                   ("whb",                   "open_ended"),
    "afrimedqa_saq":         ("afrimedqa_saq",         "open_ended"),
    "healthbench_oss_eval":  ("healthbench_oss_eval",  "open_ended_rubric"),
    "healthbench_consensus": ("healthbench_consensus", "open_ended_rubric"),
    "healthbench_hard":      ("healthbench_hard",      "open_ended_rubric"),
}


# ── Schema helpers ───────────────────────────────────────────────────────────

def _letter_for_index(idx) -> str:
    if idx is None:
        return ""
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return ""
    if 0 <= i < 26:
        return string.ascii_uppercase[i]
    return ""


def _format_choices(choices: list[str]) -> str:
    return "\n".join(
        f"{string.ascii_uppercase[i]}. {c}" for i, c in enumerate(choices)
    )


def _load_rubric_criteria(repo: str, revision: str) -> dict:
    """Build a criterion_id → {text, level, axis} map from the HF side table.

    Tries the standard `datasets.load_dataset(repo, "healthbench_criteria", ...)`
    config name first, then falls back to fetching the raw JSONL under
    side_tables/ via huggingface_hub — handles publishers who didn't register
    the side table as a queryable config.
    """
    print(f"Loading rubric side table from {repo}@{revision}")
    rows: list[dict] = []
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, "healthbench_criteria", revision=revision, split="test")
        rows = list(ds)
    except Exception as e:
        print(f"  load_dataset(healthbench_criteria) failed ({e}); trying raw side_tables file")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo,
            filename="side_tables/healthbench_criteria.jsonl",
            repo_type="dataset",
            revision=revision,
        )
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

    table = {}
    for row in rows:
        cid = row.get("criterion_id")
        if cid:
            table[cid] = {
                "text": row.get("text", ""),
                "level": row.get("level"),
                "axis": row.get("axis"),
            }
    print(f"  Loaded {len(table)} criteria")
    return table


def _normalize_row(raw: dict, set_type: str, criteria: dict | None) -> dict | None:
    """Project an HF row into the canonical shape the runners consume."""
    row_id = raw.get("id") or (raw.get("source") or {}).get("id") or ""
    question = raw.get("question")
    if not question:
        return None

    if set_type == "mcq":
        choices = raw.get("choices") or []
        answer_index = raw.get("answer_index")
        if not choices or answer_index is None:
            return None
        return {
            "id": row_id,
            "set_type": "mcq",
            "question": str(question),
            "choices": list(choices),
            "choices_formatted": _format_choices(choices),
            "ground_truth_letter": _letter_for_index(answer_index),
            "answer_index": int(answer_index),
        }

    if set_type == "open_ended":
        meta = (raw.get("source") or {}).get("metadata") or {}
        key_facts = (meta.get("key_fact_extraction") or {}).get("key_facts") or []
        return {
            "id": row_id,
            "set_type": "open_ended",
            "question": str(question),
            "reference": str(raw.get("answer") or ""),
            "key_facts": list(key_facts),
        }

    if set_type == "open_ended_rubric":
        resolved = []
        for r in (raw.get("rubrics") or []):
            cid = r.get("criterion_id")
            crit = (criteria or {}).get(cid, {}) if cid else {}
            resolved.append({
                "criterion_id": cid,
                "points": r.get("points"),
                "text": crit.get("text", ""),
                "level": crit.get("level"),
                "axis": crit.get("axis"),
            })
        # question can be str (single-turn) or list[{role, content}] (multi-turn)
        return {
            "id": row_id,
            "set_type": "open_ended_rubric",
            "question": question,
            "rubrics": resolved,
        }

    return None


def _load_dataset(name: str, revision: str, repo: str,
                  max_questions: int | None,
                  row_ids: set[str] | None = None) -> tuple[list[dict], str]:
    """Load + normalize an HF mamabench config. Returns (rows, set_type).

    If `row_ids` is provided, keep only rows whose `id` is in that set.
    Applied before `max_questions` so the subset is deterministic
    irrespective of order. Used by the calibration runner.
    """
    from datasets import load_dataset
    hf_config, set_type = HF_CONFIGS[name]
    print(f"Loading {repo}/{hf_config}@{revision}")
    ds = load_dataset(repo, hf_config, revision=revision, split="test")
    criteria = _load_rubric_criteria(repo, revision) if set_type == "open_ended_rubric" else None

    rows = []
    for raw in ds:
        if row_ids is not None and raw.get("id") not in row_ids:
            continue
        norm = _normalize_row(raw, set_type, criteria)
        if norm is not None:
            rows.append(norm)
        if max_questions and len(rows) >= max_questions:
            break
    print(f"Loaded {len(rows)} rows (set_type={set_type}"
          f"{', row_ids filter applied' if row_ids is not None else ''})")
    return rows, set_type
