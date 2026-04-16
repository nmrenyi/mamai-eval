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

_CONFIG_DIR = Path(__file__).parent / "configs" / _CONFIG_VERSION
if not _CONFIG_DIR.exists():
    raise FileNotFoundError(
        f"Config version '{_CONFIG_VERSION}' not found at {_CONFIG_DIR}. "
        f"Available versions: {[d.name for d in (Path(__file__).parent / 'configs').iterdir() if d.is_dir() and d.name != 'exp']}"
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


def build_rag_mcq_prompt(question: str, options: str, context: str) -> str:
    """Gemma IT prompt for MCQ with RAG context."""
    msgs = build_rag_mcq_messages(question, options, context)
    return _format_gemma_it(msgs["system"], msgs["user"])


def build_rag_open_prompt(question: str, context: str) -> str:
    """Gemma IT prompt for open-ended with RAG context."""
    msgs = build_rag_open_messages(question, context)
    return _format_gemma_it(msgs["system"], msgs["user"])
