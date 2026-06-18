#!/usr/bin/env python3
"""Rerank a screen_embedder-format retrievals JSON with a cross-encoder.

Reads a retrievals file (the {"candidate","dim","datasets":{ds:[{"id","question",
"chunks":[{"idx","text","sim"}]}]}} shape produced by
`screen_embedder embed_retrieve`), re-scores each query's candidate chunks with a
cross-encoder reranker (AutoModelForSequenceClassification logits), and writes a new
retrievals JSON with each query's `chunks` reordered by descending CE score (and a
`ce_score` field added). Feed the output to `screen_embedder arm_format --top-k 3` to
build a value-gate arm whose top-3 = the reranked top-3.

This isolates the reranker on top of an arbitrary base retriever (here EmbeddingGemma),
which the SQLite-based precompute_arms_matrix.py cannot do.

Usage:
  python -m retrieval_eval.rerank_retrievals \
    --retrievals eg_depth20.json --reranker /path/to/minilm-l6-finetuned-model \
    --out eg_minilm_ft.json --seq-len 256
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrievals", required=True,
                    help="comma-separated retrievals JSON files; their datasets are merged")
    ap.add_argument("--reranker", required=True, help="HF id or local path to CE reranker")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.reranker)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker).eval().to(device)
    n_labels = model.config.num_labels

    def score(query, texts):
        sc = np.zeros(len(texts), np.float32)
        with torch.no_grad():
            for s in range(0, len(texts), args.batch):
                chunk = texts[s:s + args.batch]
                enc = tok([query] * len(chunk), chunk, truncation=True,
                          max_length=args.seq_len, padding=True, return_tensors="pt").to(device)
                lg = model(**enc).logits
                sc[s:s + args.batch] = (lg.squeeze(-1) if n_labels == 1 else lg[:, -1]).float().cpu().numpy()
        return sc

    # merge datasets across input files
    merged = {"candidate": None, "dim": None, "datasets": {}}
    for f in [x.strip() for x in args.retrievals.split(",")]:
        d = json.loads(Path(f).read_text())
        merged["candidate"] = merged["candidate"] or d.get("candidate")
        merged["dim"] = merged["dim"] or d.get("dim")
        for ds, recs in d["datasets"].items():
            merged["datasets"].setdefault(ds, []).extend(recs)

    cand = merged.get("candidate") or "candidate"
    rr_tag = Path(args.reranker).name.replace("-model", "")
    out = {"candidate": f"{cand}__{rr_tag}", "dim": merged.get("dim"), "datasets": {}}
    for ds, recs in merged["datasets"].items():
        new_recs = []
        for n, rec in enumerate(recs):
            chunks = rec.get("chunks", [])
            if chunks:
                texts = [c["text"] for c in chunks]
                sc = score(rec["question"], texts)
                order = sorted(range(len(chunks)), key=lambda j: (-sc[j], chunks[j]["idx"]))
                chunks = [dict(chunks[j], ce_score=float(sc[j])) for j in order]
            new_recs.append({"id": rec["id"], "question": rec["question"], "chunks": chunks})
            if n % 100 == 0:
                print(f"[{ds}] reranked {n}/{len(recs)}", flush=True)
        out["datasets"][ds] = new_recs
        print(f"[{ds}] done ({len(new_recs)} queries)", flush=True)

    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
