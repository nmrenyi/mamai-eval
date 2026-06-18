#!/usr/bin/env python3
"""R2c P1 — precompute the 3-arm RAG contexts for the end-to-end value gate.

For each eval question, produces three retrieval contexts that isolate where
any answer-quality change comes from:
  A. gecko        — Gecko cosine top-3 (the CURRENTLY DEPLOYED retriever)
  B. hybrid       — Gecko+BM25 RRF (alpha=0.5, k=60) top-3 (the R2a config;
                    the baseline that isolates reranking: A->B is fusion,
                    B->C is reranking)
  C. hybrid_rerank— hybrid top-20 -> cross-encoder rerank -> top-3 (arm C)

Writes three run_eval-compatible dirs (<out>/gecko, <out>/hybrid,
<out>/hybrid_rerank), each with manifest.json + <dataset>.json. Retrieval runs
against the same on-device vector store (embeddings.sqlite) + Gecko TFLite the
app uses, so arm A reproduces the deployed retrieval exactly.

Usage (cluster):
  python -m retrieval_eval.precompute_rerank_arms --config config-v0.2.0 \\
      --db-path .../embeddings.sqlite --gecko-model .../Gecko_1024_quant.tflite \\
      --tokenizer .../sentencepiece.model \\
      --reranker mixedbread-ai/mxbai-rerank-base-v1 --reranker-seq-len 256 \\
      --datasets kenya,afrimedqa_saq --out-dir rag_arms
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_pre_args, _ = _pre.parse_known_args()
os.environ["MAMAI_EVAL_CONFIG"] = _pre_args.config

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_eval.retrieval import (GeckoEmbedder, build_index,
                                      format_app_context_chunks, load_vector_store,
                                      parse_chunk_metadata)

ALPHA, RRF_K, POOL = 0.5, 60, 20
HF_CONFIGS = {"kenya": "kenya", "afrimedqa_saq": "afrimedqa_saq", "whb": "whb",
              "afrimedqa": "afrimedqa", "medmcqa": "medmcqa", "medqa_usmle": "medqa_usmle"}
_WORD = re.compile(r"[a-z0-9]+")


def _toks(text):
    return _WORD.findall(parse_chunk_metadata(text)["text"].lower())


def _question_text(q):
    if isinstance(q, str):
        return q
    if isinstance(q, list):
        for turn in reversed(q):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return str(turn.get("content", ""))
    return str(q or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--db-path", required=True)
    ap.add_argument("--gecko-model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--reranker", required=True, help="HF id or local path to CE reranker")
    ap.add_argument("--reranker-seq-len", type=int, default=256)
    ap.add_argument("--hf-repo", default="nmrenyi/mamabench")
    ap.add_argument("--revision", default="v0.2")
    ap.add_argument("--datasets", default="kenya")
    ap.add_argument("--out-dir", default="rag_arms")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = load_vector_store(args.db_path)
    texts, normed = build_index(store)
    n_chunks = len(texts)
    print(f"store: {n_chunks} chunks; building BM25...", flush=True)
    bm25 = BM25Okapi([_toks(t) for t in texts])
    embedder = GeckoEmbedder(args.gecko_model, args.tokenizer)

    rr_tok = AutoTokenizer.from_pretrained(args.reranker)
    rr = AutoModelForSequenceClassification.from_pretrained(args.reranker).eval().to(device)
    rr_labels = rr.config.num_labels

    def rerank_order(query, idxs):
        pairs_t = [parse_chunk_metadata(texts[i])["text"] for i in idxs]
        sc = np.zeros(len(idxs), np.float32)
        with torch.no_grad():
            for s in range(0, len(idxs), 32):
                enc = rr_tok([query] * len(pairs_t[s:s + 32]), pairs_t[s:s + 32],
                             truncation=True, max_length=args.reranker_seq_len,
                             padding=True, return_tensors="pt").to(device)
                lg = rr(**enc).logits
                sc[s:s + 32] = (lg.squeeze(-1) if rr_labels == 1 else lg[:, -1]).float().cpu().numpy()
        order = sorted(range(len(idxs)), key=lambda j: (-sc[j], idxs[j]))
        return [idxs[j] for j in order]

    def arms_for(emb, query_text, query_toks):
        q = emb / (np.linalg.norm(emb) + 1e-10)
        sims = normed @ q
        g_top = np.argsort(sims)[::-1][:50].tolist()         # gecko top-50
        bm = bm25.get_scores(query_toks)
        b_top = np.argsort(bm)[::-1][:50].tolist()           # bm25 top-50
        g_rank = {idx: r + 1 for r, idx in enumerate(g_top)}
        b_rank = {idx: r + 1 for r, idx in enumerate(b_top)}
        rrf = {}
        for idx in set(g_top) | set(b_top):
            s = 0.0
            if idx in g_rank: s += ALPHA / (RRF_K + g_rank[idx])
            if idx in b_rank: s += (1 - ALPHA) / (RRF_K + b_rank[idx])
            rrf[idx] = s
        hyb = sorted(rrf, key=lambda i: (-rrf[i], i))[:POOL]  # hybrid top-20
        return g_top[:args.top_k], hyb[:args.top_k], rerank_order(query_text, hyb)[:args.top_k]

    datasets = [d.strip() for d in args.datasets.split(",")]
    arms = {"gecko": Path(args.out_dir) / "gecko",
            "hybrid": Path(args.out_dir) / "hybrid",
            "hybrid_rerank": Path(args.out_dir) / "hybrid_rerank"}
    for p in arms.values():
        p.mkdir(parents=True, exist_ok=True)
    ctx_version = datetime.now(timezone.utc).strftime("rerankarms-%Y%m%dT%H%M%SZ")

    for ds_name in datasets:
        ds = load_dataset(args.hf_repo, HF_CONFIGS[ds_name], revision=args.revision, split="test")
        rows = list(ds)
        if args.max_questions:
            rows = rows[:args.max_questions]
        print(f"\n=== {ds_name}: {len(rows)} questions ===", flush=True)
        out = {k: [] for k in arms}
        for n, raw in enumerate(rows):
            query_text = _question_text(raw.get("question", ""))
            rid = raw.get("id", "")
            if not query_text:
                for k in arms:
                    out[k].append({"id": rid, "question": "", "chunks": [], "similarities": []})
                continue
            emb = embedder.embed(query_text)
            a, b, c = arms_for(emb, query_text, _toks(query_text))
            for k, idxs in (("gecko", a), ("hybrid", b), ("hybrid_rerank", c)):
                cc, docs = format_app_context_chunks([texts[i] for i in idxs])
                out[k].append({"id": rid, "question": query_text, "chunks": cc,
                               "retrieved_docs": docs, "chunk_indices": idxs})
            if n % 50 == 0:
                print(f"  {n}/{len(rows)}", flush=True)
        for k, p in arms.items():
            payload = {"metadata": {"context_version": ctx_version, "dataset": ds_name,
                                    "arm": k, "hf_repo": args.hf_repo, "hf_revision": args.revision},
                       "config": {"context_version": ctx_version, "top_k": args.top_k,
                                  "embedding_model": "Gecko_1024_quant", "arm": k,
                                  "reranker": args.reranker if k == "hybrid_rerank" else None,
                                  "n_chunks_in_store": n_chunks, "n_questions": len(out[k])},
                       "retrievals": out[k]}
            with open(p / f"{ds_name}.json", "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            man = {"schema_version": 2, "context_version": ctx_version, "arm": k,
                   "retrieval_config": {"top_k": args.top_k, "alpha": ALPHA, "rrf_k": RRF_K,
                                        "pool": POOL, "reranker": args.reranker if k == "hybrid_rerank" else None},
                   "datasets": {ds_name: {"output_file": f"{ds_name}.json", "n_questions": len(out[k])}}}
            (p / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
        print(f"  wrote 3 arms for {ds_name}", flush=True)

    print("\nDone. Arms in", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
