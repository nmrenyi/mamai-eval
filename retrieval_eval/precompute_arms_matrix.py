#!/usr/bin/env python3
"""R2c table — precompute all retriever×reranker arm contexts for a dataset.

For each query, builds the gecko / bm25 / hybrid top-20 pools (same as the prior
arm precompute, rank_bm25 so the hybrid pool matches earlier runs), then emits
run_eval-compatible top-3 contexts for every arm:

  {gecko,bm25,hybrid}__none            — the pool's own top-3
  {gecko,bm25,hybrid}__<reranker>      — pool top-20 reranked -> top-3

One arm dir per cell: <out>/<retr>__<rname>/<dataset>.json (+ manifest). Handles
multi-turn (healthbench) by retrieving on the latest user turn. Rerankers run on
GPU; BM25/gecko on CPU.

Usage (cluster, GPU):
  python -m retrieval_eval.precompute_arms_matrix --config config-v0.2.0 \\
     --db-path ... --gecko-model ... --tokenizer ... \\
     --rerankers minilm_ft=/lightscratch/.../minilm-l6-finetuned-model,mxbai_ft=/lightscratch/.../mxbai-base-finetuned-model \\
     --datasets kenya --out-dir /lightscratch/.../arms_matrix
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
              "healthbench_oss_eval": "healthbench_oss_eval",
              "healthbench_consensus": "healthbench_consensus",
              "healthbench_hard": "healthbench_hard"}
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
    ap.add_argument("--rerankers", required=True, help="csv key=path of fine-tuned CE rerankers")
    ap.add_argument("--reranker-seq-len", type=int, default=256)
    ap.add_argument("--hf-repo", default="nmrenyi/mamabench")
    ap.add_argument("--revision", default="v0.2")
    ap.add_argument("--datasets", default="kenya")
    ap.add_argument("--out-dir", required=True)
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

    rerankers = {}
    for kv in args.rerankers.split(","):
        rname, path = kv.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForSequenceClassification.from_pretrained(path).eval().to(device)
        rerankers[rname] = (tok, m, m.config.num_labels)
    print(f"rerankers: {list(rerankers)}  device={device}", flush=True)

    def rerank_order(query, idxs, rname):
        tok, m, nlab = rerankers[rname]
        docs = [parse_chunk_metadata(texts[i])["text"] for i in idxs]
        sc = np.zeros(len(idxs), np.float32)
        with torch.no_grad():
            for s in range(0, len(idxs), 32):
                enc = tok([query] * len(docs[s:s + 32]), docs[s:s + 32], truncation=True,
                          max_length=args.reranker_seq_len, padding=True, return_tensors="pt").to(device)
                lg = m(**enc).logits
                sc[s:s + 32] = (lg.squeeze(-1) if nlab == 1 else lg[:, -1]).float().cpu().numpy()
        order = sorted(range(len(idxs)), key=lambda j: (-sc[j], idxs[j]))
        return [idxs[j] for j in order]

    def pools_for(emb, query_toks):
        q = emb / (np.linalg.norm(emb) + 1e-10)
        sims = normed @ q
        g_top = np.argsort(sims)[::-1][:50].tolist()
        bm = bm25.get_scores(query_toks)
        b_top = np.argsort(bm)[::-1][:50].tolist()
        g_rank = {idx: r + 1 for r, idx in enumerate(g_top)}
        b_rank = {idx: r + 1 for r, idx in enumerate(b_top)}
        rrf = {}
        for idx in set(g_top) | set(b_top):
            s = 0.0
            if idx in g_rank: s += ALPHA / (RRF_K + g_rank[idx])
            if idx in b_rank: s += (1 - ALPHA) / (RRF_K + b_rank[idx])
            rrf[idx] = s
        return {"gecko": g_top[:POOL], "bm25": b_top[:POOL],
                "hybrid": sorted(rrf, key=lambda i: (-rrf[i], i))[:POOL]}

    arm_keys = [f"{r}__none" for r in ("gecko", "bm25", "hybrid")] + \
               [f"{r}__{rn}" for r in ("gecko", "bm25", "hybrid") for rn in rerankers]
    out_dirs = {k: Path(args.out_dir) / k for k in arm_keys}
    for p in out_dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    ctx_version = datetime.now(timezone.utc).strftime("matrix-%Y%m%dT%H%M%SZ")

    for ds_name in [d.strip() for d in args.datasets.split(",")]:
        ds = load_dataset(args.hf_repo, HF_CONFIGS[ds_name], revision=args.revision, split="test")
        rows = list(ds)
        if args.max_questions:
            rows = rows[:args.max_questions]
        print(f"\n=== {ds_name}: {len(rows)} questions ===", flush=True)
        out = {k: [] for k in arm_keys}
        for n, raw in enumerate(rows):
            qt = _question_text(raw.get("question", ""))
            rid = raw.get("id", "")
            if not qt:
                for k in arm_keys:
                    out[k].append({"id": rid, "question": "", "chunks": [], "retrieved_docs": [], "chunk_indices": []})
                continue
            pools = pools_for(embedder.embed(qt), _toks(qt))
            # build per-arm top-k index lists
            arm_idx = {}
            for retr in ("gecko", "bm25", "hybrid"):
                arm_idx[f"{retr}__none"] = pools[retr][:args.top_k]
                for rn in rerankers:
                    arm_idx[f"{retr}__{rn}"] = rerank_order(qt, pools[retr], rn)[:args.top_k]
            for k, idxs in arm_idx.items():
                cc, docs = format_app_context_chunks([texts[i] for i in idxs])
                out[k].append({"id": rid, "question": qt, "chunks": cc,
                               "retrieved_docs": docs, "chunk_indices": idxs})
            if n % 50 == 0:
                print(f"  {n}/{len(rows)}", flush=True)
        for k in arm_keys:
            payload = {"metadata": {"context_version": ctx_version, "dataset": ds_name, "arm": k},
                       "config": {"context_version": ctx_version, "top_k": args.top_k, "arm": k,
                                  "embedding_model": "Gecko_1024_quant", "n_chunks_in_store": n_chunks,
                                  "n_questions": len(out[k])},
                       "retrievals": out[k]}
            (out_dirs[k] / f"{ds_name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            (out_dirs[k] / "manifest.json").write_text(json.dumps(
                {"schema_version": 2, "context_version": ctx_version, "arm": k,
                 "retrieval_config": {"top_k": args.top_k, "alpha": ALPHA, "rrf_k": RRF_K, "pool": POOL},
                 "datasets": {ds_name: {"output_file": f"{ds_name}.json", "n_questions": len(out[k])}}}, indent=2) + "\n")
        print(f"  wrote {len(arm_keys)} arms for {ds_name}", flush=True)

    print("\nDone. Arms in", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
