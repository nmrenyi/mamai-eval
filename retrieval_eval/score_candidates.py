#!/usr/bin/env python3
"""R2c P2 — score every candidate reranker on the held-out test split.

Reranks the hybrid top-20 pool (frozen 70/15/15 split, test fold = 491 queries,
9 820 pairs) with each candidate and reports, per model:
  - rerank quality: P@3 / HR@3 at the lenient (grade>=3) and strict (grade>=5) cuts
  - the R1 Stage-1 score-quality gate (chunk AUC grade>=3 / >=5, within-bundle
    concordance, bundle any-relevant AUC) — i.e. is this model's score
    *thresholdable*, the question R1 answered "no" for cosine.

Two architectures are handled:
  - "ce"    — standard BERT/ELECTRA/RoBERTa cross-encoder via
              AutoModelForSequenceClassification; relevance = the (single) logit,
              or the positive-class logit for 2-class heads. All of these share
              the proven LiteRT-int8 deployability path.
  - "qwen3" — Qwen3-Reranker causal-LM scorer (yes/no token-logit). NOT
              deployable on-device — a strong offline ceiling reference only.

Every model is scored at the SAME seq-len (default 256, the deployed TFLite
config) so the table is apples-to-apples and deployment-relevant. Pass
--max-len 512 for the seq-len sensitivity check (P3.2).

Usage (one model):
  python -m retrieval_eval.score_candidates --only ms-marco-MiniLM-L6-v2 \\
      --features-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \\
      --out-dir      configs/config-v0.2.0/results/retrieval_eval/r2c-rerank/candidates

Usage (all CE candidates; Qwen3 needs a GPU):
  python -m retrieval_eval.score_candidates --group ce ...
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_eval.compare_retriever_gates import gate_stats

# Deployable cross-encoders (same LiteRT-int8 path) + offline Qwen3 references.
# size_m = approx params (M); note = deploy story.
CANDIDATES = [
    # --- tiny (only relevant if we need smaller/faster than L6) ---
    {"key": "jina-tiny",      "id": "jinaai/jina-reranker-v1-tiny-en",      "type": "ce", "size_m": 33,  "trust": True,  "note": "tiny en CE"},
    {"key": "jina-turbo",     "id": "jinaai/jina-reranker-v1-turbo-en",     "type": "ce", "size_m": 38,  "trust": True,  "note": "turbo en CE"},
    # --- small (the MiniLM family; L6 is the deployed baseline) ---
    {"key": "minilm-l6",      "id": "cross-encoder/ms-marco-MiniLM-L6-v2",  "type": "ce", "size_m": 23,  "trust": False, "note": "deployed baseline"},
    {"key": "minilm-l12",     "id": "cross-encoder/ms-marco-MiniLM-L12-v2", "type": "ce", "size_m": 33,  "trust": False, "note": "more depth, same family"},
    # --- mid (~110-280M; stronger, heavier but still deployable) ---
    {"key": "electra-base",   "id": "cross-encoder/ms-marco-electra-base",  "type": "ce", "size_m": 110, "trust": False, "note": "mid general CE"},
    {"key": "bge-base",       "id": "BAAI/bge-reranker-base",               "type": "ce", "size_m": 278, "trust": False, "note": "mid general CE (XLM-R)"},
    {"key": "mxbai-base",     "id": "mixedbread-ai/mxbai-rerank-base-v1",   "type": "ce", "size_m": 184, "trust": False, "note": "mid general CE"},
    {"key": "medcpt",         "id": "ncbi/MedCPT-Cross-Encoder",            "type": "ce", "size_m": 109, "trust": False, "note": "medical-domain CE"},
    # --- offline reference ceilings (NOT deployable; GPU) ---
    {"key": "qwen3-rr-4b",    "id": "Qwen/Qwen3-Reranker-4B",               "type": "qwen3", "size_m": 4000, "trust": False, "note": "offline reference (GPU)"},
    {"key": "qwen3-rr-8b",    "id": "Qwen/Qwen3-Reranker-8B",               "type": "qwen3", "size_m": 8000, "trust": False, "note": "offline reference (GPU)"},
]
BY_KEY = {c["key"]: c for c in CANDIDATES}

QWEN_INSTRUCTION = ("Given a clinical question from a nurse or midwife, retrieve "
                    "the guideline passage that best answers it.")


def rerank_metrics(df, score_col, cut):
    p, hr, n = 0.0, 0, 0
    for _, grp in df.groupby("query_id"):
        top3 = grp.sort_values([score_col, "chunk_id"], ascending=[False, True]).head(3)
        rel = int((top3["grade"] >= cut).sum())
        p += min(rel, 3) / 3.0; hr += int(rel > 0); n += 1
    return round(p / n, 4), round(hr / n, 4)


def gate_on(df, score_col):
    rows = []
    for _, grp in df.groupby("query_id"):
        g = grp.sort_values([score_col, "chunk_id"], ascending=[False, True])
        for rank, r in enumerate(g.itertuples(), 1):
            rows.append({"query_id": r.query_id, "rank": rank,
                         "score": float(getattr(r, score_col)), "grade": int(r.grade)})
    return gate_stats(rows)


def score_ce(cand, pairs, max_len, batch_size, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cand["id"], trust_remote_code=cand["trust"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cand["id"], trust_remote_code=cand["trust"]).eval().to(device)
    n_labels = model.config.num_labels
    scores = np.zeros(len(pairs), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = tok([p[0] for p in batch], [p[1] for p in batch], padding=True,
                      truncation=True, max_length=max_len, return_tensors="pt").to(device)
            logits = model(**enc).logits
            s = logits.squeeze(-1) if n_labels == 1 else logits[:, -1]
            scores[i:i + batch_size] = s.float().cpu().numpy()
            if (i // batch_size) % 50 == 0:
                print(f"    {cand['key']} scored {i}/{len(pairs)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return scores, n_labels


def score_qwen3(cand, pairs, max_len, batch_size, device):
    """Qwen3-Reranker: yes/no token-logit at the final position."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cand["id"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cand["id"], torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None).eval()
    if device == "cpu":
        model = model.to(device)
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements "
              "based on the Query and the Instruct provided. Note that the answer can "
              "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pre_ids = tok.encode(prefix, add_special_tokens=False)
    suf_ids = tok.encode(suffix, add_special_tokens=False)
    scores = np.zeros(len(pairs), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            texts = [f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"
                     for q, d in batch]
            body = tok(texts, add_special_tokens=False, truncation=True,
                       max_length=max_len - len(pre_ids) - len(suf_ids))["input_ids"]
            seqs = [pre_ids + b + suf_ids for b in body]
            enc = tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits[:, -1, :]
            yn = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
            lp = torch.nn.functional.log_softmax(yn, dim=1)[:, 1]
            scores[i:i + batch_size] = lp.float().cpu().numpy()
            if (i // batch_size) % 20 == 0:
                print(f"    {cand['key']} scored {i}/{len(pairs)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return scores, 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=256,
                    help="seq-len; 256 = deployed TFLite config (default)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only", default="", help="comma-separated candidate keys")
    ap.add_argument("--group", default="", choices=["", "ce", "qwen3", "all"])
    args = ap.parse_args()

    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- which candidates ---
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
    elif args.group == "ce":
        keys = [c["key"] for c in CANDIDATES if c["type"] == "ce"]
    elif args.group == "qwen3":
        keys = [c["key"] for c in CANDIDATES if c["type"] == "qwen3"]
    else:
        keys = [c["key"] for c in CANDIDATES]
    print(f"device={device}  max_len={args.max_len}  candidates={keys}", flush=True)

    # --- test-split hybrid pool with text ---
    df = pd.read_parquet(Path(args.features_dir) / "ltr_features.parquet")
    te = df[df["split"] == "test"][["query_id", "chunk_id", "grade", "rrf_score"]].copy()
    q_text = {r.query_id: r.query_text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/queries.parquet", repo_type="dataset", revision=args.revision)).itertuples()}
    c_text = {r.chunk_id: r.text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/chunks.parquet", repo_type="dataset", revision=args.revision)).itertuples()}
    te["q"] = te["query_id"].map(q_text)
    te["c"] = te["chunk_id"].map(c_text)
    pairs = list(zip(te["q"], te["c"]))
    n_q = te["query_id"].nunique()
    print(f"test pool: {len(te)} pairs, {n_q} queries", flush=True)

    cuts = {"lenient_ge3": 3, "strict_ge5": 5}

    # --- floor (RRF) + oracle, computed once ---
    base = {"n_test_queries": int(n_q), "by_cut": {}}
    for ck, cv in cuts.items():
        fp, fhr = rerank_metrics(te, "rrf_score", cv)
        op, ohr = rerank_metrics(te, "grade", cv)
        base["by_cut"][ck] = {"floor_rrf": {"p_at_3": fp, "hr_at_3": fhr},
                              "oracle": {"p_at_3": op, "hr_at_3": ohr}}
    with open(out_dir / "_floor_oracle.json", "w") as f:
        json.dump(base, f, indent=2)
    print("floor/oracle:", base["by_cut"], flush=True)

    for key in keys:
        cand = BY_KEY[key]
        print(f"\n=== {key} ({cand['id']}, {cand['type']}, ~{cand['size_m']}M) ===", flush=True)
        t0 = time.time()
        try:
            if cand["type"] == "ce":
                scores, n_labels = score_ce(cand, pairs, args.max_len, args.batch_size, device)
            else:
                scores, n_labels = score_qwen3(cand, pairs, args.max_len, max(1, args.batch_size // 4), device)
        except Exception as e:
            print(f"  FAILED {key}: {type(e).__name__}: {e}", flush=True)
            with open(out_dir / f"{key}.error.json", "w") as f:
                json.dump({"key": key, "id": cand["id"], "error": f"{type(e).__name__}: {e}"}, f, indent=2)
            continue
        col = f"score_{key}"
        te[col] = scores
        rec = {"created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "key": key, "id": cand["id"], "type": cand["type"], "size_m": cand["size_m"],
               "note": cand["note"], "max_len": args.max_len, "n_labels": int(n_labels),
               "score_seconds": round(time.time() - t0, 1), "device": device, "by_cut": {}}
        for ck, cv in cuts.items():
            cp, chr_ = rerank_metrics(te, col, cv)
            rec["by_cut"][ck] = {"p_at_3": cp, "hr_at_3": chr_}
        rec["stage1_gate"] = {k: gate_on(te, col)[k] for k in
                              ("chunk_auc_grade3", "chunk_auc_grade5",
                               "within_bundle_concordance", "bundle_any_relevant_auc_top1")}
        with open(out_dir / f"{key}.json", "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  {key}: lenient P@3={rec['by_cut']['lenient_ge3']['p_at_3']} "
              f"HR@3={rec['by_cut']['lenient_ge3']['hr_at_3']} | "
              f"strict P@3={rec['by_cut']['strict_ge5']['p_at_3']} | "
              f"gate AUC3={rec['stage1_gate']['chunk_auc_grade3']} "
              f"({rec['score_seconds']}s)", flush=True)

    print("\nAll done. Per-model JSONs in", out_dir, flush=True)


if __name__ == "__main__":
    main()
