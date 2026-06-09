"""Generator faithfulness — stage 3: MiniCheck response-level scoring (solution A).

Loads Gemma's oracle_responses.json (from stage 2), runs MiniCheck-7B on each
(context, response) pair to get a support probability ∈ [0, 1], aggregates,
and writes minicheck_scores_A.json alongside the input.

Per docs/mamai-quality-evaluation.md §3.1, this is Pipeline 1 (no decomposition)
at the response level. The "_A" suffix in the output filename reserves room
for a future paragraph/bullet-level pass ("_B") without overwrite conflicts.

Usage:
  python -m generator_eval.score_minicheck <run-dir>
  python -m generator_eval.score_minicheck <oracle_responses.json>
  python -m generator_eval.score_minicheck <run-dir> --max-questions 5
"""

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

CHECKPOINT_INTERVAL = 100
BATCH_SIZE = 16  # batch MiniCheck calls to reduce Python overhead


def _resolve_input(path: Path) -> Path:
    """Accept either a run dir or the JSON path directly."""
    if path.is_dir():
        candidate = path / "oracle_responses.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No oracle_responses.json in {path}")
        return candidate
    if path.suffix != ".json":
        raise ValueError(f"Expected a directory or .json file, got {path}")
    return path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bootstrap_ci(values: list[float], n_resamples: int = 1000,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean. Deterministic — fixed seed."""
    import random
    rng = random.Random(42)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples)]
    return (lo, hi)


def _save(output_path: Path, metadata: dict, results: list[dict]) -> None:
    output_path.write_text(
        json.dumps({"metadata": metadata, "results": results},
                   indent=2, ensure_ascii=False)
    )


# ── Scorer ──────────────────────────────────────────────────────────────────
#
# We call the Bespoke-MiniCheck-7B model directly via transformers instead of
# the `minicheck` package — that package isn't on PyPI and its GitHub
# pyproject.toml is missing a `name` field, breaking `pip install git+…`.
# Direct invocation also means no vllm dependency, which keeps the cluster
# install lean. The model is a Llama-3-8B fine-tune with a chat template that
# emits a single Yes/No token; we read P(Yes) from its first-token logits.

_PROMPT_TEMPLATE = (
    "Determine whether the provided claim is consistent with the corresponding "
    "document. Consistency in this context implies that all information presented "
    "in the claim is substantiated by the document. If not, it should be considered "
    "inconsistent.\n\n"
    "Document:\n{document}\n\n"
    "Claim:\n{claim}\n\n"
    "Please assess the claim's consistency with the document by responding with "
    "either \"yes\" or \"no\"."
)


class _BespokeMiniCheckScorer:
    """Thin wrapper around bespokelabs/bespoke-minicheck-7b on HF transformers.

    Returns P(supported) ∈ [0, 1] computed as softmax over the {yes, no} token
    logits at the first generated position. Single-call (no batching) — the
    chat-template plus variable-length premises make batched padding awkward,
    and per-call latency on A100 for a 7B model + ~2K input tokens is ≈ 100ms,
    so 2,659 calls ≈ 5 min, well within budget for stage 3.
    """

    HF_MODEL_NAME = "bespokelabs/bespoke-minicheck-7b"

    def __init__(self, model_id: str, cache_dir: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = self._resolve_model_id(model_id)
        print(f"  Loading tokenizer/model from {resolved}")
        # trust_remote_code is required: bespoke-minicheck-7b is an
        # internlm2-architecture model and ships custom modeling code on HF.
        self.tok = AutoTokenizer.from_pretrained(
            resolved, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model.eval()

        # Bespoke's training data uses lowercase "yes"/"no". Get the first
        # token id for each — both should be single tokens for Llama-3.
        self.yes_id = self._first_token_id("yes")
        self.no_id = self._first_token_id("no")
        # Sanity backstop: if for some reason the lowercase mapping looks
        # broken (e.g. tokenizer treats "yes" as multi-token), fall back to
        # "Yes"/"No" — better than silently scoring against garbage tokens.
        if self.yes_id is None or self.no_id is None:
            self.yes_id = self._first_token_id("Yes")
            self.no_id = self._first_token_id("No")
        print(f"  yes_id={self.yes_id}  no_id={self.no_id}")

    @classmethod
    def _resolve_model_id(cls, name: str) -> str:
        # Accept either the friendly name 'Bespoke-MiniCheck-7B' (legacy) or
        # the HF repo id directly.
        if name.lower().startswith("bespoke-minicheck"):
            return cls.HF_MODEL_NAME
        return name

    def _first_token_id(self, text: str) -> int | None:
        ids = self.tok.encode(text, add_special_tokens=False)
        return ids[0] if ids else None

    def score(self, document: str, claim: str) -> float:
        import torch

        prompt = _PROMPT_TEMPLATE.format(document=document, claim=claim)
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(input_text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        last_logits = outputs.logits[0, -1, :]
        yes_no = torch.stack([last_logits[self.yes_id], last_logits[self.no_id]])
        probs = torch.softmax(yes_no.float(), dim=0)
        return float(probs[0].item())

    def score_batch(self, documents: list[str], claims: list[str]) -> list[float]:
        return [self.score(d, c) for d, c in zip(documents, claims)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input",
                        help="Path to oracle_responses.json or its parent run dir")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <input-dir>/minicheck_scores_A.json)")
    parser.add_argument("--minicheck-model", default="Bespoke-MiniCheck-7B",
                        help="MiniCheck variant name (default: Bespoke-MiniCheck-7B).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(supported) threshold for binary supported flag (default 0.5).")
    parser.add_argument("--cache-dir", default=None,
                        help="Local cache dir for MiniCheck weights.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for MiniCheck calls (default {BATCH_SIZE}).")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit to N queries for smoke testing.")
    args = parser.parse_args()

    input_path = _resolve_input(Path(args.input))
    print(f"Input:  {input_path}")
    data = json.loads(input_path.read_text())
    rows_all = data["results"]
    rows = rows_all[:args.max_questions] if args.max_questions else rows_all
    print(f"Loaded {len(rows)}/{len(rows_all)} responses to score")

    output_path = Path(args.output) if args.output else (
        input_path.parent / "minicheck_scores_A.json"
    )
    print(f"Output: {output_path}")

    # Resume from prior output if present.
    done_ids: set[str] = set()
    resume_results: list[dict] = []
    if output_path.exists():
        prev = json.loads(output_path.read_text())
        resume_results = prev.get("results", [])
        done_ids = {r["query_id"] for r in resume_results}
        if len(resume_results) >= len(rows):
            print(f"  Already complete ({len(resume_results)}/{len(rows)}), nothing to do")
            return
        print(f"  Resuming: {len(resume_results)}/{len(rows)} already scored")

    # Filter to the rows that still need scoring, preserving order.
    pending = [r for r in rows if r["query_id"] not in done_ids]
    print(f"  Pending: {len(pending)}")

    print(f"\nLoading MiniCheck: {args.minicheck_model}")
    scorer = _BespokeMiniCheckScorer(model_id=args.minicheck_model,
                                     cache_dir=args.cache_dir)
    print("MiniCheck loaded.\n")

    metadata = {
        "input_file": input_path.name,
        "input_file_sha256": _file_sha256(input_path),
        "method": "response_level",
        "minicheck_model": args.minicheck_model,
        "threshold": args.threshold,
        "n_responses": len(rows),
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "batch_size": args.batch_size,
    }

    results: list[dict] = list(resume_results)
    probs: list[float] = [r["support_prob"] for r in resume_results
                          if isinstance(r.get("support_prob"), (int, float))
                          and not math.isnan(r["support_prob"])]

    t0 = time.time()
    done_since_start = 0

    with tqdm(total=len(pending), desc="MiniCheck") as pbar:
        for i in range(0, len(pending), args.batch_size):
            batch = pending[i:i + args.batch_size]
            docs = [r["context"] for r in batch]
            claims = [r["model_response"] for r in batch]

            # Filter empty claims out of the model call but record them at prob=0.
            valid_idx = [j for j, c in enumerate(claims) if c.strip()]
            batch_probs: list[float] = [float("nan")] * len(batch)
            if valid_idx:
                try:
                    sub_docs = [docs[j] for j in valid_idx]
                    sub_claims = [claims[j] for j in valid_idx]
                    raw_probs = scorer.score_batch(sub_docs, sub_claims)
                    for j, p in zip(valid_idx, raw_probs):
                        batch_probs[j] = float(p)
                except Exception as e:
                    print(f"  ERROR on batch starting at offset {i}: {e}", file=sys.stderr)
                    # Leave NaNs in this batch; they'll be flagged.

            for r, p in zip(batch, batch_probs):
                note = None
                if not r["model_response"].strip():
                    p = 0.0
                    note = "empty_response"
                elif math.isnan(p):
                    note = "scoring_error"
                results.append({
                    "query_id": r["query_id"],
                    "support_prob": None if math.isnan(p) else round(p, 6),
                    "supported": (not math.isnan(p)) and p > args.threshold,
                    "note": note,
                })
                if not math.isnan(p):
                    probs.append(p)

            done_since_start += len(batch)
            pbar.update(len(batch))

            if (done_since_start % CHECKPOINT_INTERVAL) < args.batch_size and done_since_start:
                _save(output_path, metadata, results)
                elapsed = time.time() - t0
                rate = done_since_start / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done_since_start) / rate / 60 if rate > 0 else math.inf
                mean_p = sum(probs) / len(probs) if probs else 0
                pbar.set_postfix(rate=f"{rate:.2f}/s", eta=f"{eta:.1f}min", mean_p=f"{mean_p:.3f}")

    elapsed = time.time() - t0
    metadata["total_inference_time_s"] = round(elapsed, 1)
    metadata["avg_time_per_response_s"] = round(elapsed / len(pending), 3) if pending else 0

    # Aggregate using probs from THIS run plus already-scored ones from resume.
    all_probs = [r["support_prob"] for r in results
                 if isinstance(r.get("support_prob"), (int, float))]
    if all_probs:
        ci = _bootstrap_ci(all_probs)
        metadata["aggregate"] = {
            "mean_support_prob": round(sum(all_probs) / len(all_probs), 4),
            "median_support_prob": round(sorted(all_probs)[len(all_probs) // 2], 4),
            "fraction_supported": round(
                sum(1 for p in all_probs if p > args.threshold) / len(all_probs), 4
            ),
            "n_supported": sum(1 for p in all_probs if p > args.threshold),
            "n_responses_scored": len(all_probs),
            "n_errors": sum(1 for r in results if r.get("note") == "scoring_error"),
            "n_empty_responses": sum(1 for r in results if r.get("note") == "empty_response"),
            "bootstrap_95_ci_mean": [round(x, 4) for x in ci],
        }
    else:
        metadata["aggregate"] = {"error": "no valid probabilities recorded"}

    _save(output_path, metadata, results)
    print(f"\nSaved: {output_path}")
    agg = metadata["aggregate"]
    print(f"  scored:              {agg.get('n_responses_scored')}/{len(rows)}")
    print(f"  mean P(supported):   {agg.get('mean_support_prob')}")
    print(f"  median P:            {agg.get('median_support_prob')}")
    print(f"  fraction P>{args.threshold}:        {agg.get('fraction_supported')}")
    print(f"  95% CI (mean):       {agg.get('bootstrap_95_ci_mean')}")
    print(f"  errors / empty:      {agg.get('n_errors')} / {agg.get('n_empty_responses')}")
    print(f"  total wall:          {elapsed:.1f}s")


if __name__ == "__main__":
    main()
