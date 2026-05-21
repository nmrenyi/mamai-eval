"""Generator faithfulness — stage 3 (Lynx variant): Patronus Lynx 70B scoring.

Reads Gemma's oracle_responses.json (from stage 2) and runs Patronus Lynx —
an open, Llama-3-70B-based RAG hallucination detector — on each
(question, context, answer) triple. Lynx is reference-free: it needs only
the question, the retrieved context, and the answer (no gold reference).

For each response Lynx returns a holistic verdict — PASS (faithful) or FAIL
(not faithful) — plus bullet-point reasoning, as a JSON object. We aggregate
to a dataset-level faithfulness pass rate. See https://arxiv.org/abs/2407.08488.

Inference is via vLLM (the 70B model needs tensor-parallel across 2 GPUs).

Usage:
  python -m generator_eval.score_lynx <run-dir>
  python -m generator_eval.score_lynx <oracle_responses.json> --max-questions 5
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_CHUNK = 256

# Exact prompt template from the Lynx model card
# (PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct).
LYNX_PROMPT = """\
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
"""


def _resolve_input(path: Path) -> Path:
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
    return (means[int(alpha / 2 * n_resamples)],
            means[int((1 - alpha / 2) * n_resamples)])


def _save(output_path: Path, metadata: dict, results: list[dict]) -> None:
    output_path.write_text(
        json.dumps({"metadata": metadata, "results": results},
                   indent=2, ensure_ascii=False)
    )


def _parse_lynx_output(text: str) -> tuple[str | None, object, str | None]:
    """Extract (score, reasoning, parse_note) from Lynx's raw generation.

    score is normalised to "PASS"/"FAIL"/None. reasoning is whatever the
    REASONING field held (list or str), or None. parse_note flags trouble.
    """
    # Find the outermost {...} block — tolerates markdown fences / prose.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = obj.get("SCORE")
            reasoning = obj.get("REASONING")
            if isinstance(score, str):
                norm = score.strip().upper()
                if norm in ("PASS", "FAIL"):
                    return norm, reasoning, None
                return None, reasoning, f"unexpected SCORE value: {score!r}"
            return None, reasoning, "SCORE missing or non-string"
        except json.JSONDecodeError:
            pass  # fall through to regex
    # Fallback: regex straight for the verdict.
    sm = re.search(r'"?SCORE"?\s*:?\s*"?(PASS|FAIL)\b', text, re.IGNORECASE)
    if sm:
        return sm.group(1).upper(), None, "recovered SCORE via regex (no valid JSON)"
    return None, None, "no JSON and no SCORE found"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input",
                        help="Path to oracle_responses.json or its parent run dir")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <input-dir>/lynx_scores.json)")
    parser.add_argument("--lynx-model",
                        default="PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct",
                        help="HF model id for the Lynx judge.")
    parser.add_argument("--tensor-parallel", type=int, default=2,
                        help="vLLM tensor_parallel_size (GPUs). 70B fp16 needs 2.")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max_model_len (Lynx/Llama-3 native is 8192).")
    parser.add_argument("--max-new-tokens", type=int, default=600,
                        help="Max tokens for the reasoning+verdict (model card: 600).")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit to N responses for smoke testing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore any existing output file and score from scratch "
                             "(default: resume from it). Use for smoke iterations.")
    args = parser.parse_args()

    input_path = _resolve_input(Path(args.input))
    print(f"Input:  {input_path}")
    data = json.loads(input_path.read_text())
    rows_all = data["results"]
    rows = rows_all[:args.max_questions] if args.max_questions else rows_all
    print(f"Loaded {len(rows)}/{len(rows_all)} responses to score")

    output_path = Path(args.output) if args.output else (
        input_path.parent / "lynx_scores.json"
    )
    print(f"Output: {output_path}")

    # Resume from prior output if present (unless --overwrite).
    done_ids: set[str] = set()
    resume_results: list[dict] = []
    if output_path.exists() and args.overwrite:
        print("  --overwrite: ignoring existing output, scoring from scratch")
    elif output_path.exists():
        prev = json.loads(output_path.read_text())
        resume_results = prev.get("results", [])
        done_ids = {r["query_id"] for r in resume_results}
        if len(resume_results) >= len(rows):
            print(f"  Already complete ({len(resume_results)}/{len(rows)}), nothing to do")
            return
        print(f"  Resuming: {len(resume_results)}/{len(rows)} already scored")
    pending = [r for r in rows if r["query_id"] not in done_ids]
    print(f"  Pending: {len(pending)}")

    print(f"\nLoading Lynx via vLLM: {args.lynx_model} (tp={args.tensor_parallel})")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.lynx_model,
        tensor_parallel_size=args.tensor_parallel,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    print("Lynx loaded.\n")

    # Token budget: input must leave room for the generation.
    input_budget = args.max_model_len - args.max_new_tokens

    metadata = {
        "input_file": input_path.name,
        "input_file_sha256": _file_sha256(input_path),
        "method": "lynx_holistic_pass_fail",
        "judge_model": args.lynx_model,
        "judge_family": "llama-3 (Patronus Lynx fine-tune)",
        "n_responses": len(rows),
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "sampling": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "max_model_len": args.max_model_len,
    }

    results: list[dict] = list(resume_results)
    t0 = time.time()

    for chunk_start in range(0, len(pending), CHECKPOINT_CHUNK):
        chunk = pending[chunk_start:chunk_start + CHECKPOINT_CHUNK]

        prompts, meta = [], []
        for r in chunk:
            user_msg = LYNX_PROMPT.format(
                question=r["query_text"],
                context=r["context"],
                answer=r["model_response"],
            )
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
            n_tok = len(tokenizer.encode(text))
            prompts.append(text)
            meta.append({"query_id": r["query_id"], "n_input_tokens": n_tok})

        # Flag rows whose prompt won't fit; score the rest.
        fit_idx = [i for i, m in enumerate(meta) if m["n_input_tokens"] <= input_budget]
        gen_by_idx: dict[int, str] = {}
        if fit_idx:
            outs = llm.generate([prompts[i] for i in fit_idx], sampling)
            for i, o in zip(fit_idx, outs):
                gen_by_idx[i] = o.outputs[0].text

        for i, m in enumerate(meta):
            if i not in gen_by_idx:
                results.append({
                    "query_id": m["query_id"],
                    "score": None, "passed": False,
                    "reasoning": None, "n_input_tokens": m["n_input_tokens"],
                    "note": f"input {m['n_input_tokens']} tok exceeds budget {input_budget}",
                })
                continue
            raw = gen_by_idx[i]
            score, reasoning, parse_note = _parse_lynx_output(raw)
            row = {
                "query_id": m["query_id"],
                "score": score,
                "passed": score == "PASS",
                "reasoning": reasoning,
                "n_input_tokens": m["n_input_tokens"],
                "note": parse_note,
            }
            if parse_note is not None:  # keep raw text whenever parsing was imperfect
                row["raw_output"] = raw[:3000]
            results.append(row)

        _save(output_path, metadata, results)
        done = len(results) - len(resume_results)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(pending) - done) / rate / 60 if rate > 0 else float("inf")
        n_pass = sum(1 for r in results if r["score"] == "PASS")
        n_fail = sum(1 for r in results if r["score"] == "FAIL")
        print(f"  [{done}/{len(pending)}] checkpoint — "
              f"PASS={n_pass} FAIL={n_fail} rate={rate:.2f}/s ETA={eta:.1f}min")

    # Aggregate.
    elapsed = time.time() - t0
    metadata["total_inference_time_s"] = round(elapsed, 1)
    verdicts = [1 if r["score"] == "PASS" else 0
                for r in results if r["score"] in ("PASS", "FAIL")]
    n_pass = sum(verdicts)
    n_scored = len(verdicts)
    if n_scored:
        ci = _bootstrap_ci([float(v) for v in verdicts])
        metadata["aggregate"] = {
            "pass_rate": round(n_pass / n_scored, 4),
            "n_pass": n_pass,
            "n_fail": n_scored - n_pass,
            "n_scored": n_scored,
            "n_parse_errors": sum(1 for r in results if r["score"] is None
                                  and r.get("note", "").startswith(("no JSON", "SCORE", "unexpected"))),
            "n_oversize": sum(1 for r in results if r.get("note", "").startswith("input ")),
            "bootstrap_95_ci_pass_rate": [round(x, 4) for x in ci],
        }
    else:
        metadata["aggregate"] = {"error": "no valid verdicts recorded"}

    _save(output_path, metadata, results)
    agg = metadata["aggregate"]
    print(f"\nSaved: {output_path}")
    print(f"  scored:        {agg.get('n_scored')}/{len(rows)}")
    print(f"  PASS rate:     {agg.get('pass_rate')}  ({agg.get('n_pass')} PASS / {agg.get('n_fail')} FAIL)")
    print(f"  95% CI:        {agg.get('bootstrap_95_ci_pass_rate')}")
    print(f"  parse errors:  {agg.get('n_parse_errors')}")
    print(f"  oversize:      {agg.get('n_oversize')}")
    print(f"  total wall:    {elapsed:.1f}s")


if __name__ == "__main__":
    main()
