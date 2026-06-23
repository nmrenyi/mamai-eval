"""Generator faithfulness — stage 2: generation pass.

Runs a generator (default: Gemma 4 E4B) once per oracle query using oracle
context from mamaretrieval. The output is the raw input to stage 3
(MiniCheck-based sentence-level support scoring); no scoring happens here.

The oracle JSONL is built by `generator_eval/build_oracle.py` and committed
under `configs/<config>/oracle/`. Each row has chunks pre-sorted by judge
score descending. At eval time we take the top-K (default 3, matching the
deployed retrieval depth in params.json) and concatenate as the RAG context.

Usage:
  python -m generator_eval.eval_faithfulness --config config-v0.2.0 \\
      --model gemma4-e4b --top-k 3
  python -m generator_eval.eval_faithfulness --config config-v0.2.0 \\
      --model gemma4-e4b --max-questions 5
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ── Resolve --config before any prompts imports ──────────────────────────────
# shared.prompts reads MAMAI_EVAL_CONFIG at module load time; mirror the
# pattern used by end_to_end_eval/run_eval.py and retrieval_eval/precompute_retrieval.py.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_pre_args, _ = _pre.parse_known_args()
os.environ["MAMAI_EVAL_CONFIG"] = _pre_args.config
# ─────────────────────────────────────────────────────────────────────────────

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.inference import load_model
from shared.prompts import (CONFIG_VERSION, N_CTX, PROMPT_VERSION,
                            PROTOCOL_VERSION, SPEC_SHA256, TEMPERATURE,
                            TOP_K, TOP_P, build_rag_open_messages,
                            build_rag_open_prompt)

CHECKPOINT_INTERVAL = 100


def _model_call(model, payload, max_tokens: int) -> str:
    """Route generation through the right backend method based on model type.

    Same dispatch as end_to_end_eval/run_eval.py — kept inline to avoid
    importing the whole run_eval module just for this helper.
    """
    if hasattr(model, "is_api") and model.is_api:
        return model.generate(payload, max_tokens=max_tokens)
    if hasattr(model, "supports_chat") and model.supports_chat:
        return model.generate_chat(payload, max_tokens=max_tokens)
    return model.generate(payload, max_tokens=max_tokens)


def _load_oracle(path: Path, max_questions: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_questions and len(rows) >= max_questions:
                break
    return rows


def _load_oracle_manifest(oracle_path: Path) -> dict | None:
    """Look for `<stem>.manifest.json` next to the oracle file."""
    manifest_path = oracle_path.with_suffix("").with_suffix(".manifest.json")
    # Above mangles a single-suffix path; recompute more carefully.
    manifest_path = oracle_path.parent / f"{oracle_path.stem}.manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return None


def _gen_responses(model, payloads, max_tokens, workers):
    """One response per payload, order-preserving. Concurrent for API/endpoint
    models when workers>1 (needed for a slow cluster-served model like Qwen-397B).
    Returns list aligned to payloads of (response, elapsed_s); None on error."""
    out: list = [None] * len(payloads)

    def work(idx):
        t0 = time.time()
        try:
            return idx, _model_call(model, payloads[idx], max_tokens), round(time.time() - t0, 2)
        except Exception as e:
            print(f"  ERROR payload {idx}: generate() failed: {e}")
            return idx, None, 0.0

    if workers > 1 and getattr(model, "is_api", False):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, i) for i in range(len(payloads))]
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"Faithfulness gen x{workers}"):
                idx, resp, el = f.result()
                out[idx] = (resp, el)
    else:
        for i in tqdm(range(len(payloads)), desc="Faithfulness gen"):
            _, resp, el = work(i)
            out[i] = (resp, el)
    return out


def _save(output_path: Path, metadata: dict, results: list[dict]) -> None:
    data = {"metadata": metadata, "results": results}
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def run(model, oracle_rows: list[dict], top_k: int, max_tokens: int,
        output_path: Path, metadata: dict,
        resume_results: list[dict] | None, workers: int = 1) -> list[dict]:
    """Generate one response per oracle query and write a single JSON file.

    Resume semantics: identified by query_id, not row index. `workers>1`
    concurrently generates against an API/endpoint model (e.g. Qwen via vLLM).
    """
    done_ids = {r["query_id"] for r in resume_results} if resume_results else set()
    results = list(resume_results) if resume_results else []
    if done_ids:
        print(f"  Resuming from checkpoint: {len(done_ids)} already-completed query_ids; "
              f"unfinished/errored queries will be retried")

    uses_messages = (
        (hasattr(model, "is_api") and model.is_api)
        or (hasattr(model, "supports_chat") and model.supports_chat)
    )

    pending = [r for r in oracle_rows if r["query_id"] not in done_ids]
    payloads, ctxs = [], []
    for row in pending:
        used_chunks = row["chunks"][:top_k]
        context = "\n\n".join(c["text"] for c in used_chunks)
        ctxs.append((used_chunks, context))
        payloads.append(build_rag_open_messages(row["query_text"], context) if uses_messages
                        else build_rag_open_prompt(row["query_text"], context))

    gen = _gen_responses(model, payloads, max_tokens, workers)

    for k, row in enumerate(pending):
        response, elapsed = gen[k]
        if response is None:
            continue
        used_chunks, context = ctxs[k]
        results.append({
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "n_chunks_available": len(row["chunks"]),
            "n_chunks_used": len(used_chunks),
            "chunk_ids": [c["chunk_id"] for c in used_chunks],
            "chunk_scores": [c["score"] for c in used_chunks],
            "context": context,
            "context_chars": len(context),
            "model_response": response,
            "inference_time_s": elapsed,
        })
        if (k + 1) % CHECKPOINT_INTERVAL == 0:
            _save(output_path, metadata, results)

    _save(output_path, metadata, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="Config version (e.g. config-v0.2.0)")
    parser.add_argument("--model", default="gemma4-e4b",
                        help="Model name in shared.inference.MODEL_REGISTRY")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing local model files (ignored for API models)")
    parser.add_argument("--oracle", default=None,
                        help="Oracle JSONL path. Default: configs/<config>/oracle/mamaretrieval-v0.1.0-score5.jsonl")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Max chunks per query fed to the model (default 3, matches deployed retrieval)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens to generate (default: from config params.json)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of queries (for smoke testing)")
    parser.add_argument("--n-gpu-layers", type=int, default=None,
                        help="GPU layers for GGUF (-1 = all, 0 = CPU, default: auto-detect)")
    parser.add_argument("--output-dir", default=None,
                        help="Output root (default: configs/<config>/results/generator)")
    parser.add_argument("--run-dir", default=None,
                        help="Fixed output dir (reused across restarts for auto-resume).")
    parser.add_argument("--resume", default=None,
                        help="Previous run dir to resume from if --run-dir output is empty.")
    parser.add_argument("--system-prompt", default=None,
                        help="Path to an alternate open-ended system prompt (A/B prompt arms). "
                             "Overrides the config's system_en.txt without mutating the config; "
                             "affects the oracle-context generation. Provenance recorded in metadata.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent generation requests for API/endpoint models "
                             "(e.g. cluster-served Qwen via vLLM). Ignored for GGUF.")
    args = parser.parse_args()

    # ── Optional system-prompt override (A/B prompt arms) ────────────────────
    # build_rag_open_{prompt,messages} resolve OPEN_SYSTEM_PROMPT from
    # shared.prompts at call time, so reassigning the module global here takes
    # effect for the faithfulness generation.
    system_prompt_override = None
    override_sha256 = None
    if args.system_prompt:
        import hashlib as _hashlib
        import shared.prompts as _prompts_mod
        _sp = Path(args.system_prompt)
        _prompts_mod.OPEN_SYSTEM_PROMPT = _sp.read_text(encoding="utf-8").rstrip("\n")
        system_prompt_override = str(_sp.resolve())
        override_sha256 = _hashlib.sha256(_sp.read_bytes()).hexdigest()
        print(f"System-prompt override: {system_prompt_override} (sha256={override_sha256[:12]})")

    from shared.prompts import _params as _active_params
    max_tokens = args.max_tokens or _active_params["generation"]["max_tokens"]

    repo_root = Path(__file__).resolve().parents[1]
    oracle_path = Path(args.oracle) if args.oracle else (
        repo_root / "configs" / args.config / "oracle" /
        "mamaretrieval-v0.1.0-score5.jsonl"
    )
    if not oracle_path.is_absolute():
        oracle_path = (Path.cwd() / oracle_path).resolve()
    if not oracle_path.exists():
        parser.error(f"Oracle file not found: {oracle_path}")

    oracle_manifest = _load_oracle_manifest(oracle_path)

    output_dir = Path(args.output_dir) if args.output_dir else (
        repo_root / "configs" / args.config / "results" / "generator"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (
        output_dir / args.model / run_timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "oracle_responses.json"

    print(f"Oracle:     {oracle_path}")
    print(f"Run dir:    {run_dir}")
    print(f"Output:     {output_path}")
    oracle_rows = _load_oracle(oracle_path, args.max_questions)
    print(f"Loaded {len(oracle_rows)} oracle queries"
          f" (max_questions={args.max_questions})")

    resume_results = None
    resume_path = None
    if output_path.exists():
        resume_path = output_path
    elif args.resume:
        candidate = Path(args.resume) / "oracle_responses.json"
        if candidate.exists():
            resume_path = candidate

    if resume_path:
        prev = json.loads(resume_path.read_text())
        prev_results = prev.get("results", [])
        if len(prev_results) >= len(oracle_rows):
            print(f"  Already complete ({len(prev_results)}/{len(oracle_rows)}), nothing to do")
            return
        resume_results = prev_results
        print(f"  Resuming: {len(resume_results)}/{len(oracle_rows)} results from prior run")

    model = load_model(args.model, args.model_dir, n_gpu_layers=args.n_gpu_layers)

    metadata = {
        "model": args.model,
        "model_dir": args.model_dir,
        "config_version": CONFIG_VERSION,
        "oracle_file": oracle_path.name,
        "oracle_path": str(oracle_path),
        "oracle_source": (
            oracle_manifest.get("source") if oracle_manifest else None
        ),
        "oracle_threshold": (
            oracle_manifest.get("threshold") if oracle_manifest else None
        ),
        "oracle_top_k": args.top_k,
        "n_questions": len(oracle_rows),
        "timestamp": run_timestamp,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "spec_sha256": override_sha256 or SPEC_SHA256,
        "system_prompt_override": system_prompt_override,
        "generation_params": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "n_ctx": N_CTX,
            "max_tokens": max_tokens,
        },
    }

    t0 = time.time()
    results = run(model, oracle_rows, args.top_k, max_tokens, output_path,
                  metadata, resume_results=resume_results, workers=args.workers)
    elapsed = time.time() - t0
    metadata["total_inference_time_s"] = round(elapsed, 1)
    metadata["avg_time_per_question_s"] = (
        round(elapsed / len(results), 2) if results else 0
    )

    _save(output_path, metadata, results)
    print(f"\nSaved: {output_path}")
    print(f"  Queries done:        {len(results)}/{len(oracle_rows)}")
    print(f"  Total wall time:     {elapsed:.1f}s")
    if results:
        print(f"  Avg per query:       {metadata['avg_time_per_question_s']}s")


if __name__ == "__main__":
    main()
