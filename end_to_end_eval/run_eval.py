"""
Batch evaluation pipeline for mamabench (v0.2 schema).

Reads benchmark rows from the HuggingFace dataset configured in
params.json (`dataset.hf_repo` + `dataset.revision`), runs model
inference, and writes per-row results to
configs/<config>/results/end_to_end_eval/<model>/<run-id>/<dataset>.json.

Open-ended judging is no longer done inline — use rescore_open_v2.py
(3-judge ensemble) or rescore_rubric.py (HealthBench-style) on the
saved result files. The legacy --judge flag still triggers v0.1's
single-judge fallback for backwards compatibility.

Usage:
  python run_eval.py --config config-v0.2.0 --model gemma4-e4b --datasets afrimedqa
  python run_eval.py --config config-v0.2.0 --model gpt-5 --datasets all
  python run_eval.py --config config-v0.2.0 --model gemma4-e4b \
      --datasets kenya --rag rag_contexts/
"""

import argparse
import os

# ── Resolve --config before any prompts imports ──────────────────────────────
# prompts.py reads MAMAI_EVAL_CONFIG at module level; must be set before import.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_pre_args, _ = _pre.parse_known_args()
os.environ["MAMAI_EVAL_CONFIG"] = _pre_args.config
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# Ensure repo root is importable so `from shared.* import …` works when this
# script is invoked directly (`python end_to_end_eval/run_eval.py …`) as well
# as via `python -m end_to_end_eval.run_eval …`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.inference import load_model
from shared.prompts import (TEMPERATURE, TOP_P, TOP_K, N_CTX, PROMPT_VERSION,
                            PROTOCOL_VERSION, SPEC_SHA256, RETRIEVAL_TOP_K,
                            CONFIG_VERSION, JUDGE_MODEL, JUDGE_TEMPERATURE,
                            DATASET_HF_REPO, DATASET_REVISION,
                            build_mcq_prompt, build_mcq_messages,
                            build_open_prompt, build_open_messages,
                            build_open_messages_multiturn,
                            build_rag_mcq_prompt, build_rag_mcq_messages,
                            build_rag_open_prompt, build_rag_open_messages,
                            build_rag_open_messages_multiturn)
from shared.scoring import (JUDGE_DIMENSIONS, _parse_answer_set, create_judge_client,
                            extract_letters, judge_response, score_mcq)
from shared.dataset_loader import HF_CONFIGS, _load_dataset

CHECKPOINT_INTERVAL = 100


# ── Inference dispatch ───────────────────────────────────────────────────────

def _model_call(model, messages_or_prompt, max_tokens: int):
    """Route generation through the right backend method based on model type."""
    if hasattr(model, "is_api") and model.is_api:
        return model.generate(messages_or_prompt, max_tokens=max_tokens)
    if hasattr(model, "supports_chat") and model.supports_chat:
        return model.generate_chat(messages_or_prompt, max_tokens=max_tokens)
    return model.generate(messages_or_prompt, max_tokens=max_tokens)


def _flatten_turns_for_prompt(turns: list[dict]) -> str:
    """Flatten a multi-turn conversation into a single user-side prompt for the
    Gemma prompt-format path. Frontier APIs and chat-template GGUFs use proper
    multi-turn messages instead.
    """
    parts = []
    for t in turns:
        role = (t.get("role") or "user").upper()
        parts.append(f"{role}: {t.get('content', '')}")
    return "\n\n".join(parts)


# ── MCQ runner ───────────────────────────────────────────────────────────────

def run_mcq(model, rows, max_tokens, output_path=None, metadata=None,
            rag_contexts=None, resume_results=None):
    """Run MCQ evaluation: inference + letter extraction + accuracy."""
    n_skip = len(resume_results) if resume_results else 0
    results = list(resume_results) if resume_results else []
    predictions = [r["extracted_answer"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]
    if n_skip:
        print(f"  Resuming from checkpoint: skipping {n_skip} already-completed rows")

    for i, row in enumerate(tqdm(rows, total=len(rows), desc="MCQ inference",
                                 initial=n_skip), 1):
        if i <= n_skip:
            continue

        question = row["question"]
        options = row["choices_formatted"]
        correct = row["ground_truth_letter"]

        context_str = ""
        if rag_contexts and (i - 1) < len(rag_contexts):
            chunks = rag_contexts[i - 1].get("chunks", [])
            context_str = "\n\n".join(chunks)

        t0 = time.time()
        try:
            uses_messages = (
                (hasattr(model, "is_api") and model.is_api)
                or (hasattr(model, "supports_chat") and model.supports_chat)
            )
            if context_str:
                payload = (build_rag_mcq_messages(question, options, context_str)
                           if uses_messages else
                           build_rag_mcq_prompt(question, options, context_str))
            else:
                payload = (build_mcq_messages(question, options)
                           if uses_messages else
                           build_mcq_prompt(question, options))
            response = _model_call(model, payload, max_tokens)
        except Exception as e:
            print(f"  ERROR row {i}: generate() failed: {e}")
            continue
        elapsed = time.time() - t0

        extracted_set = extract_letters(response)
        extracted = ",".join(sorted(extracted_set)) if extracted_set else ""
        predictions.append(extracted)
        ground_truth.append(correct)

        results.append({
            "id": row["id"],
            "question": question,
            "options": options,
            "ground_truth": correct,
            "answer_index": row["answer_index"],
            "rag_context": context_str[:200] + "..." if context_str else "",
            "model_response": response,
            "extracted_answer": extracted,
            "extracted_answers": sorted(extracted_set),
            "correct": extracted_set == _parse_answer_set(correct),
            "inference_time_s": round(elapsed, 2),
        })

        if output_path and i % CHECKPOINT_INTERVAL == 0:
            scores = score_mcq(predictions, ground_truth)
            save_checkpoint(output_path, metadata or {}, scores, results)
            print(f"  Checkpoint saved at {i}/{len(rows)}")

    scores = score_mcq(predictions, ground_truth)
    return results, scores


# ── Open-ended runner ────────────────────────────────────────────────────────

def _open_scores(judgments, n_failed=0):
    """Compute aggregate single-judge scores (v0.1 legacy --judge path only)."""
    if not judgments and not n_failed:
        return {}
    scores = {}
    for dim in JUDGE_DIMENSIONS:
        dim_scores = [j[dim] for j in judgments if j.get(dim) is not None]
        if dim_scores:
            scores[f"mean_{dim}"] = round(sum(dim_scores) / len(dim_scores), 2)
            scores[f"{dim}_distribution"] = {i: dim_scores.count(i) for i in range(1, 6)}
    weighted = [j["weighted_score"] for j in judgments if j.get("weighted_score") is not None]
    if weighted:
        scores["mean_weighted_score"] = round(sum(weighted) / len(weighted), 2)
    scores["n_judged"] = len(judgments)
    scores["n_failed"] = n_failed
    scores["dimension_weights"] = dict(JUDGE_DIMENSIONS)
    return scores


def run_open(model, rows, max_tokens, judge_client, judge_model,
             output_path=None, metadata=None, rag_contexts=None, resume_results=None):
    """Run open-ended inference. Optional v0.1 single-judge scoring inline.

    For v0.2 the canonical scoring path is post-hoc via rescore_open_v2.py.
    """
    n_skip = len(resume_results) if resume_results else 0
    results = list(resume_results) if resume_results else []
    judgments = []
    n_judge_failed = 0
    for r in results:
        if r.get("judge_weighted_score") is not None:
            j = {dim: r["judge_scores"].get(dim) for dim in JUDGE_DIMENSIONS}
            j["weighted_score"] = r["judge_weighted_score"]
            judgments.append(j)
    if n_skip:
        print(f"  Resuming from checkpoint: skipping {n_skip} already-completed rows")

    for i, row in enumerate(tqdm(rows, total=len(rows), desc="Open inference",
                                 initial=n_skip), 1):
        if i <= n_skip:
            continue

        question = row["question"]
        reference = row.get("reference", "")
        key_facts = row.get("key_facts", [])

        context_str = ""
        if rag_contexts and (i - 1) < len(rag_contexts):
            chunks = rag_contexts[i - 1].get("chunks", [])
            context_str = "\n\n".join(chunks)

        t0 = time.time()
        try:
            uses_messages = (
                (hasattr(model, "is_api") and model.is_api)
                or (hasattr(model, "supports_chat") and model.supports_chat)
            )
            if context_str:
                payload = (build_rag_open_messages(question, context_str)
                           if uses_messages else
                           build_rag_open_prompt(question, context_str))
            else:
                payload = (build_open_messages(question)
                           if uses_messages else
                           build_open_prompt(question))
            response = _model_call(model, payload, max_tokens)
        except Exception as e:
            print(f"  ERROR row {i}: generate() failed: {e}")
            continue
        elapsed = time.time() - t0

        result = {
            "id": row["id"],
            "question": question,
            "reference": reference,
            "key_facts": key_facts,
            "model_response": response,
            "inference_time_s": round(elapsed, 2),
        }

        if judge_client is not None and reference:
            judgment = judge_response(question, response, reference, judge_client,
                                      judge_model, temperature=JUDGE_TEMPERATURE)
            if judgment and judgment.get("weighted_score") is not None:
                result["judge_scores"] = {dim: judgment.get(dim) for dim in JUDGE_DIMENSIONS}
                result["judge_weighted_score"] = judgment["weighted_score"]
                result["judge_justification"] = judgment.get("justification")
                judgments.append(judgment)
            else:
                n_judge_failed += 1

        results.append(result)

        if output_path and i % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(output_path, metadata or {},
                            _open_scores(judgments, n_judge_failed), results)
            print(f"  Checkpoint saved at {i}/{len(rows)}")

    return results, _open_scores(judgments, n_judge_failed)


# ── Rubric runner ────────────────────────────────────────────────────────────

def run_rubric(model, rows, max_tokens, output_path=None, metadata=None,
               rag_contexts=None, resume_results=None):
    """Generate responses for HealthBench-style rubric rows.

    No inline scoring — feed the saved result file to rescore_rubric.py.
    """
    n_skip = len(resume_results) if resume_results else 0
    results = list(resume_results) if resume_results else []
    if n_skip:
        print(f"  Resuming from checkpoint: skipping {n_skip} already-completed rows")

    for i, row in enumerate(tqdm(rows, total=len(rows), desc="Rubric inference",
                                 initial=n_skip), 1):
        if i <= n_skip:
            continue

        question = row["question"]
        is_multiturn = isinstance(question, list)
        rubrics = row.get("rubrics", [])

        context_str = ""
        if rag_contexts and (i - 1) < len(rag_contexts):
            chunks = rag_contexts[i - 1].get("chunks", [])
            context_str = "\n\n".join(chunks)

        t0 = time.time()
        try:
            uses_messages = (
                (hasattr(model, "is_api") and model.is_api)
                or (hasattr(model, "supports_chat") and model.supports_chat)
            )
            if is_multiturn and uses_messages:
                payload = (build_rag_open_messages_multiturn(question, context_str)
                           if context_str else
                           build_open_messages_multiturn(question))
            elif is_multiturn:
                # Gemma prompt-format fallback: flatten the conversation into a single user turn.
                flat = _flatten_turns_for_prompt(question)
                payload = (build_rag_open_prompt(flat, context_str)
                           if context_str else
                           build_open_prompt(flat))
            elif uses_messages:
                payload = (build_rag_open_messages(question, context_str)
                           if context_str else
                           build_open_messages(question))
            else:
                payload = (build_rag_open_prompt(question, context_str)
                           if context_str else
                           build_open_prompt(question))
            response = _model_call(model, payload, max_tokens)
        except Exception as e:
            print(f"  ERROR row {i}: generate() failed: {e}")
            continue
        elapsed = time.time() - t0

        results.append({
            "id": row["id"],
            "question": question,
            "rubrics": rubrics,
            "model_response": response,
            "inference_time_s": round(elapsed, 2),
        })

        if output_path and i % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(output_path, metadata or {}, {}, results)
            print(f"  Checkpoint saved at {i}/{len(rows)}")

    return results, {}


# ── Persistence ──────────────────────────────────────────────────────────────

def save_checkpoint(output_path, metadata, scores, results):
    data = {"metadata": metadata, "aggregate_scores": scores, "results": results}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="mamabench v0.2 evaluation pipeline")
    parser.add_argument("--config", required=True,
                        help="Config version to evaluate (e.g. config-v0.2.0)")
    parser.add_argument("--model", required=True, help="Model name (e.g. gemma4-e4b, gpt-5)")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing local model files (ignored for API models)")
    parser.add_argument("--datasets", required=True,
                        help=f"Comma-separated dataset names, or 'all'. "
                             f"Available: {','.join(HF_CONFIGS.keys())}")
    parser.add_argument("--revision", default=None,
                        help="HF dataset revision (default: dataset.revision from params.json, else v0.2)")
    parser.add_argument("--hf-repo", default=None,
                        help="HF dataset repo (default: dataset.hf_repo from params.json)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens to generate (default: from config params.json)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions per dataset (for debugging)")
    parser.add_argument("--row-ids", default=None,
                        help="Path to a calibration manifest JSON. When set, only rows whose "
                             "`id` appears in manifest['ids'] are evaluated. Used to run the "
                             "same row set on both venues for the device-vs-cluster calibration "
                             "comparison.")
    parser.add_argument("--judge", action="store_true",
                        help="(Legacy v0.1 single-judge) Inline open-ended scoring. "
                             "For v0.2 prefer rescore_open_v2.py / rescore_rubric.py.")
    parser.add_argument("--judge-model", default=None,
                        help="OpenAI model for legacy judging (default: from params.json)")
    parser.add_argument("--n-gpu-layers", type=int, default=None,
                        help="GPU layers for GGUF (-1 = all, 0 = CPU, default: auto-detect)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output JSON files "
                             "(default: configs/<config>/results/end_to_end_eval)")
    parser.add_argument("--rag", default=None,
                        help="Path to pre-computed RAG contexts dir (from precompute_retrieval.py)")
    parser.add_argument("--resume", default=None,
                        help="Path to previous run dir to resume incomplete datasets from")
    parser.add_argument("--run-dir", default=None,
                        help="Fixed output dir (reused across restarts for auto-resume).")
    args = parser.parse_args()

    from shared.prompts import _params as _active_params
    max_tokens = args.max_tokens or _active_params["generation"]["max_tokens"]
    judge_model_name = args.judge_model or JUDGE_MODEL
    revision = args.revision or DATASET_REVISION or "v0.2"
    hf_repo = args.hf_repo or DATASET_HF_REPO or "nmrenyi/mamabench"

    output_dir = args.output_dir or str(
        Path(__file__).resolve().parents[1] / "configs" / args.config / "results" / "end_to_end_eval"
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.datasets == "all":
        dataset_names = list(HF_CONFIGS.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        for name in dataset_names:
            if name not in HF_CONFIGS:
                parser.error(f"Unknown dataset: {name}. Available: {list(HF_CONFIGS.keys())}")

    row_ids_filter: set[str] | None = None
    if args.row_ids:
        manifest = json.loads(Path(args.row_ids).read_text())
        row_ids_filter = set(manifest["ids"])
        print(f"Row-ids filter: {len(row_ids_filter)} ids from "
              f"{args.row_ids} ({manifest.get('name', '?')})")

    model = load_model(args.model, args.model_dir, n_gpu_layers=args.n_gpu_layers)

    judge_client, judge_model = None, None
    if args.judge:
        print("WARNING: --judge is the v0.1 single-judge fallback. "
              "Use rescore_open_v2.py for the v0.2 3-judge ensemble.")
        judge_client, judge_model = create_judge_client(judge_model_name)
        if judge_client is None:
            print("WARNING: --judge requested but no OPENAI_API_KEY found. Skipping judge scoring.")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = args.run_dir or os.path.join(output_dir, args.model, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    summary = []
    rag_manifest = None
    rag_manifest_sha256 = None
    if args.rag:
        manifest_path = os.path.join(args.rag, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                rag_manifest = json.load(f)
            rag_manifest_sha256 = _file_sha256(manifest_path)
            print(f"Loaded RAG manifest: {manifest_path}")

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  |  Config: {CONFIG_VERSION}")
        print(f"{'='*60}")

        try:
            rows, set_type = _load_dataset(ds_name, revision, hf_repo, args.max_questions,
                                           row_ids=row_ids_filter)
        except Exception as e:
            print(f"SKIP: failed to load {ds_name}: {e}")
            continue
        if not rows:
            print(f"SKIP: {ds_name} produced 0 normalized rows")
            continue

        output_path = os.path.join(run_dir, f"{ds_name}.json")

        rag_contexts = None
        rag_data = None
        if args.rag:
            rag_path = os.path.join(args.rag, f"{ds_name}.json")
            if os.path.exists(rag_path):
                with open(rag_path) as f:
                    rag_data = json.load(f)
                rag_contexts = rag_data["retrievals"]
                print(f"RAG contexts loaded: {len(rag_contexts)} entries "
                      f"(top-{rag_data['config']['top_k']})")
            else:
                print(f"WARNING: --rag specified but {rag_path} not found. Running without RAG.")

        metadata = {
            "model": args.model,
            "model_dir": args.model_dir,
            "dataset": ds_name,
            "dataset_type": set_type,
            "hf_repo": hf_repo,
            "hf_revision": revision,
            "config_version": CONFIG_VERSION,
            "n_questions": len(rows),
            "timestamp": run_timestamp,
            "protocol_version": PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "spec_sha256": SPEC_SHA256,
            "rag": rag_contexts is not None,
            "generation_params": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "n_ctx": N_CTX,
                "max_tokens": max_tokens,
            },
        }
        if rag_contexts is not None and rag_data is not None:
            rag_meta = {
                "dir": os.path.abspath(args.rag),
                "top_k": rag_data["config"].get("top_k"),
                "n_questions": rag_data["config"].get("n_questions"),
                "context_version": None,
                "manifest_sha256": rag_manifest_sha256,
            }
            if rag_manifest is not None:
                rag_meta.update({
                    "context_version": rag_manifest.get("context_version"),
                    "created_at_utc": rag_manifest.get("created_at_utc"),
                    "repo_ref": rag_manifest.get("repo_ref"),
                    "repo_commit": rag_manifest.get("repo_commit"),
                    "source_lock": rag_manifest.get("source_lock"),
                    "artifacts": rag_manifest.get("artifacts"),
                })
            else:
                rag_meta["context_version"] = rag_data.get("metadata", {}).get("context_version")
            metadata["rag_context"] = rag_meta

        resume_results = None
        resume_path = None
        if os.path.exists(output_path):
            resume_path = output_path
        elif args.resume:
            candidate = os.path.join(args.resume, f"{ds_name}.json")
            if os.path.exists(candidate):
                resume_path = candidate

        if resume_path:
            with open(resume_path) as f:
                prev = json.load(f)
            prev_results = prev.get("results", [])
            if len(prev_results) >= len(rows):
                print(f"  Already complete ({len(prev_results)}/{len(rows)}), skipping")
                if resume_path != output_path:
                    save_checkpoint(output_path, prev.get("metadata", metadata),
                                    prev.get("aggregate_scores", {}), prev_results)
                summary.append(_format_summary(ds_name, set_type,
                                               prev.get("aggregate_scores", {}),
                                               prev_results, resumed=True))
                continue
            resume_results = prev_results
            print(f"  Resuming: {len(resume_results)}/{len(rows)} results from checkpoint")

        t0 = time.time()
        if set_type == "mcq":
            results, scores = run_mcq(model, rows, max_tokens, output_path, metadata,
                                      rag_contexts=rag_contexts,
                                      resume_results=resume_results)
        elif set_type == "open_ended":
            results, scores = run_open(model, rows, max_tokens, judge_client, judge_model,
                                       output_path, metadata, rag_contexts=rag_contexts,
                                       resume_results=resume_results)
        elif set_type == "open_ended_rubric":
            results, scores = run_rubric(model, rows, max_tokens, output_path, metadata,
                                         rag_contexts=rag_contexts,
                                         resume_results=resume_results)
        else:
            print(f"  ERROR: unknown set_type {set_type}")
            continue

        elapsed = time.time() - t0
        metadata["total_inference_time_s"] = round(elapsed, 1)
        metadata["avg_time_per_question_s"] = round(elapsed / len(results), 2) if results else 0

        save_checkpoint(output_path, metadata, scores, results)
        print(f"Saved: {output_path}")
        summary.append(_format_summary(ds_name, set_type, scores, results))

    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}  |  config: {CONFIG_VERSION}")
    print(f"{'='*60}")
    for line in summary:
        print(line)


def _format_summary(ds_name, set_type, scores, results, resumed=False) -> str:
    tag = " [resumed]" if resumed else ""
    if set_type == "mcq":
        acc = scores.get("accuracy", 0)
        partial = scores.get("partial_credit_accuracy", acc)
        return f"  {ds_name}: {acc:.1%} (partial: {partial:.1%}){tag}"
    if set_type == "open_ended":
        mean_score = scores.get("mean_weighted_score")
        if mean_score is not None:
            return f"  {ds_name}: {mean_score}/5{tag}"
        return f"  {ds_name}: {len(results)} responses saved (run rescore_open_v2.py to judge){tag}"
    if set_type == "open_ended_rubric":
        return f"  {ds_name}: {len(results)} responses saved (run rescore_rubric.py to judge){tag}"
    return f"  {ds_name}: {len(results)} responses saved{tag}"


if __name__ == "__main__":
    main()
