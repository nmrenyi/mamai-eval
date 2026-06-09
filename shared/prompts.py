"""
Prompt templates for medical QA evaluation.

Config loading:
  The active config version is set by entry-point scripts via the
  MAMAI_EVAL_CONFIG environment variable before this module is imported.
  Use --config on any entry point (run_eval.py, precompute_retrieval.py)
  and it is handled automatically.

  The open-ended system prompt is read from
  configs/<version>/system_en.txt — the same text as the deployed Android
  app when using a released config version — so open-ended eval scores
  reflect actual app behavior.

  MCQ uses a separate adapter prompt (mcq_system.txt) because the app
  prompt produces clinical prose, not a single letter. See GitHub issue #39.
"""

import hashlib
import json
import os
from pathlib import Path

_CONFIG_VERSION = os.environ.get("MAMAI_EVAL_CONFIG")
if not _CONFIG_VERSION:
    raise EnvironmentError(
        "MAMAI_EVAL_CONFIG must be set to a config version (e.g. config-v0.1.0) "
        "before importing prompts. Entry point scripts set this automatically via --config."
    )

# Repo root is one level above shared/. configs/ lives there.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs" / _CONFIG_VERSION
if not _CONFIG_DIR.exists():
    raise FileNotFoundError(
        f"Config version '{_CONFIG_VERSION}' not found at {_CONFIG_DIR}. "
        f"Available versions: {[d.name for d in (_REPO_ROOT / 'configs').iterdir() if d.is_dir() and d.name != 'exp']}"
    )

_params = json.loads((_CONFIG_DIR / "params.json").read_text())

# --- System prompts ---

APP_SYSTEM_PROMPT: str = (_CONFIG_DIR / "system_en.txt").read_text(encoding="utf-8").rstrip("\n")
APP_SYSTEM_PROMPT_SW: str = (_CONFIG_DIR / "system_sw.txt").read_text(encoding="utf-8").rstrip("\n")

# NOT the app prompt. Required because the app prompt produces clinical prose,
# which breaks single-letter extraction. MCQ scores are a knowledge proxy,
# not a deployment-fidelity measure. See GitHub issue #39.
MCQ_SYSTEM_PROMPT: str = (_CONFIG_DIR / "mcq_system.txt").read_text(encoding="utf-8").rstrip("\n")

# Open-ended eval uses the real app prompt — scores reflect deployed behavior.
OPEN_SYSTEM_PROMPT = APP_SYSTEM_PROMPT

# --- Generation parameters ---
TEMPERATURE: float = _params["generation"]["temperature"]
TOP_P: float = _params["generation"]["top_p"]
TOP_K: int = _params["generation"]["top_k"]
N_CTX: int = _params["generation"]["n_ctx"]

# --- Retrieval parameters ---
RETRIEVAL_TOP_K: int = _params["retrieval"]["top_k"]
RETRIEVAL_THRESHOLD: float = _params["retrieval"]["similarity_threshold"]

# --- Context injection labels ---
CONTEXT_LABEL: str = _params["context_injection"]["context_label_en"]
QUESTION_LABEL: str = _params["context_injection"]["question_label_en"]

# --- Judge parameters ---
JUDGE_MODEL: str = _params["judge"]["model"]
JUDGE_TEMPERATURE: float = _params["judge"]["temperature"]

# v0.2: ensemble (now retired to a single-entry list) for open_ended scoring
# and a separate single judge for open_ended_rubric scoring. Both are optional
# so v0.1 configs still load.
_judge_top: dict = _params.get("judge", {})

# JUDGE_ENSEMBLE = each entry's provider/model/temperature/extra_body, inheriting
# the top-level defaults when the entry doesn't override them. This is how the
# pinned reasoning_effort actually reaches the open-ended SAQ rescorer (the same
# fix pattern as JUDGE_RUBRIC below — previously the ensemble entries lost
# extra_body on the floor).
JUDGE_ENSEMBLE: list[dict] = [
    {
        "provider": e.get("provider", "openai"),
        "model": e.get("model"),
        "temperature": e.get("temperature", _judge_top.get("temperature", 0.0)),
        "extra_body": e.get("extra_body", _judge_top.get("extra_body")),
    }
    for e in (_judge_top.get("ensemble") or [])
    if e.get("model")
]

# JUDGE_RUBRIC = the rubric subsection, but inheriting temperature / extra_body
# from the top-level `judge` block when the subsection doesn't override them.
# This is how the pinned reasoning_effort / temperature from the judge config
# actually reach the production rescorer (previously dropped on the floor).
_rubric_sub: dict = _judge_top.get("rubric", {})
JUDGE_RUBRIC: dict = {
    "provider": _rubric_sub.get("provider", "openai"),
    "model": _rubric_sub.get("model") or _judge_top.get("model"),
    "temperature": _rubric_sub.get("temperature", _judge_top.get("temperature", 0.0)),
    "extra_body": _rubric_sub.get("extra_body", _judge_top.get("extra_body")),
} if (_rubric_sub.get("model") or _judge_top.get("model")) else {}

# --- Dataset source (v0.2: HF dataset; v0.1: local TSV) ---
_dataset_cfg = _params.get("dataset", {})
DATASET_HF_REPO: str | None = _dataset_cfg.get("hf_repo")
DATASET_REVISION: str | None = _dataset_cfg.get("revision")

# --- Protocol versioning ---

PROTOCOL_VERSION = "app_parity_v1"
CONFIG_VERSION = _CONFIG_VERSION


def _spec_sha256() -> str:
    """SHA-256 of the English system prompt for the active config version."""
    return hashlib.sha256((_CONFIG_DIR / "system_en.txt").read_bytes()).hexdigest()


SPEC_SHA256: str = _spec_sha256()


def _prompt_hash(*prompts: str) -> str:
    """Short hash of prompt content — changes automatically when prompts are edited."""
    h = hashlib.sha256("".join(prompts).encode()).hexdigest()[:8]
    return f"v3-{h}"


PROMPT_VERSION = _prompt_hash(MCQ_SYSTEM_PROMPT, OPEN_SYSTEM_PROMPT)


def _format_gemma_it(system: str, user: str) -> str:
    """Wrap system + user content in Gemma IT chat template."""
    return (
        f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def build_mcq_messages(question: str, options: str) -> dict:
    """Return model-agnostic {system, user} messages for MCQ."""
    return {
        "system": MCQ_SYSTEM_PROMPT,
        "user": f"Question: {question}\nOptions:\n{options}",
    }


def build_open_messages(question: str) -> dict:
    """Return model-agnostic {system, user} messages for open-ended."""
    return {
        "system": OPEN_SYSTEM_PROMPT,
        "user": question,
    }


def build_open_messages_multiturn(turns: list[dict]) -> dict:
    """Open-ended messages for HealthBench-style multi-turn rows.

    `turns` is a list[{role, content}] taken straight from the dataset.
    Returned shape matches the single-turn builder so the runner can decide
    once whether to call chat-completion vs prompt-format paths.
    """
    return {
        "system": OPEN_SYSTEM_PROMPT,
        "turns": turns,
    }


def build_mcq_prompt(question: str, options: str) -> str:
    """Build a Gemma IT prompt for a multiple-choice question."""
    msgs = build_mcq_messages(question, options)
    return _format_gemma_it(msgs["system"], msgs["user"])


def build_open_prompt(question: str) -> str:
    """Build a Gemma IT prompt for an open-ended clinical question."""
    msgs = build_open_messages(question)
    return _format_gemma_it(msgs["system"], msgs["user"])


# --- RAG-augmented prompt builders ---

def build_rag_mcq_messages(question: str, options: str, context: str) -> dict:
    """MCQ prompt with RAG context. Same system prompt, context injected in user message."""
    return {
        "system": MCQ_SYSTEM_PROMPT,
        "user": (
            f"{CONTEXT_LABEL}\n{context}\n\n"
            f"{QUESTION_LABEL} {question}\nOptions:\n{options}"
        ),
    }


def build_rag_open_messages(question: str, context: str) -> dict:
    """Open-ended prompt with RAG context. Uses app system prompt."""
    return {
        "system": OPEN_SYSTEM_PROMPT,
        "user": (
            f"{CONTEXT_LABEL}\n{context}\n\n"
            f"{QUESTION_LABEL} {question}"
        ),
    }


def build_rag_open_messages_multiturn(turns: list[dict], context: str) -> dict:
    """Multi-turn open-ended messages with RAG context prepended to the final user turn."""
    if not turns:
        return {"system": OPEN_SYSTEM_PROMPT, "turns": []}
    rewritten = [dict(t) for t in turns]
    # Inject context into the final user turn so retrieval is visible to the model.
    for i in range(len(rewritten) - 1, -1, -1):
        if rewritten[i].get("role") == "user":
            rewritten[i]["content"] = (
                f"{CONTEXT_LABEL}\n{context}\n\n"
                f"{QUESTION_LABEL} {rewritten[i].get('content', '')}"
            )
            break
    return {"system": OPEN_SYSTEM_PROMPT, "turns": rewritten}


def build_rag_mcq_prompt(question: str, options: str, context: str) -> str:
    """Gemma IT prompt for MCQ with RAG context."""
    msgs = build_rag_mcq_messages(question, options, context)
    return _format_gemma_it(msgs["system"], msgs["user"])


def build_rag_open_prompt(question: str, context: str) -> str:
    """Gemma IT prompt for open-ended with RAG context."""
    msgs = build_rag_open_messages(question, context)
    return _format_gemma_it(msgs["system"], msgs["user"])
