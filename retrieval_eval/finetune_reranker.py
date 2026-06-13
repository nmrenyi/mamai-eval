#!/usr/bin/env python3
"""R2c P2.5 — fine-tune a cross-encoder reranker on the 230k graded pairs.

The zero-shot rerankers are trained on web search (MS MARCO) / general domains.
This adapts one to our task: nurse/midwife clinical questions over OBGYN
guideline chunks, graded 0-6 by the rubric judge. We fine-tune the model's
scalar relevance head with a regression target (grade/6 in [0,1]) on the TRAIN
split's graded pairs, select on DEV, and report the held-out TEST reranking
quality + the R1 Stage-1 score-quality gate — the same metrics as the zero-shot
comparison, so the lift is directly attributable to fine-tuning.

Train/dev/test use the frozen by-query split (split.json), so no test query is
ever seen in training. Training pairs are ALL graded (query, chunk) pairs for
train queries (hard negatives from every audit retriever), not just the hybrid
top-20 — more signal, same eval.

Usage (cluster GPU):
  python -m retrieval_eval.finetune_reranker \\
      --model cross-encoder/ms-marco-MiniLM-L6-v2 --key minilm-l6 \\
      --features-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \\
      --out-dir      configs/config-v0.2.0/results/retrieval_eval/r2c-rerank/finetune \\
      --epochs 2 --seq-len 256
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_eval.compare_retriever_gates import gate_stats


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} model={args.model} seq_len={args.seq_len} epochs={args.epochs}", flush=True)

    # --- split + texts ---  (split.json maps query_id -> "train"|"dev"|"test")
    split_of = json.load(open(Path(args.features_dir) / "split.json"))

    judg = pd.read_parquet(hf_hub_download(args.hf_repo, "data/judgments.parquet",
                                           repo_type="dataset", revision=args.revision))
    judg = judg.rename(columns={"score": "grade"})  # judgments.parquet calls the 0-6 grade "score"
    q_text = {r.query_id: r.query_text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/queries.parquet", repo_type="dataset", revision=args.revision)).itertuples()}
    c_text = {r.chunk_id: r.text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/chunks.parquet", repo_type="dataset", revision=args.revision)).itertuples()}
    judg["split"] = judg["query_id"].map(split_of)
    judg = judg.dropna(subset=["split"])
    print("graded pairs by split:", judg["split"].value_counts().to_dict(), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    def make_ds(part):
        d = judg[judg["split"] == part]
        return Dataset.from_dict({
            "q": [q_text.get(x, "") for x in d["query_id"]],
            "c": [c_text.get(x, "") for x in d["chunk_id"]],
            "label": [float(g) / 6.0 for g in d["grade"]],
        })

    def tok_fn(b):
        e = tok(b["q"], b["c"], truncation=True, max_length=args.seq_len,
                padding="max_length")
        e["labels"] = b["label"]
        return e

    train_ds = make_ds("train").map(tok_fn, batched=True, remove_columns=["q", "c", "label"])
    dev_ds = make_ds("dev").map(tok_fn, batched=True, remove_columns=["q", "c", "label"])

    # .float(): some reranker checkpoints (e.g. mxbai-rerank-base-v1) ship fp16
    # params, which break the Trainer's fp16 AMP grad unscaling ("Attempting to
    # unscale FP16 gradients"). Force fp32 master weights; fp16=True still gives
    # mixed-precision compute.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True).float().to(device)

    targs = TrainingArguments(
        output_dir=str(out_dir / f"{args.key}-ckpt"),
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64, eval_strategy="epoch",
        save_strategy="no", logging_steps=100, warmup_ratio=0.1,
        fp16=(device == "cuda"), report_to=[], dataloader_num_workers=4)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds)
    t0 = time.time()
    trainer.train()
    print(f"training done in {time.time()-t0:.0f}s", flush=True)

    # --- final TEST eval: re-score the hybrid top-20 pool ---
    df = pd.read_parquet(Path(args.features_dir) / "ltr_features.parquet")
    te = df[df["split"] == "test"][["query_id", "chunk_id", "grade", "rrf_score"]].copy()
    te["q"] = te["query_id"].map(q_text)
    te["c"] = te["chunk_id"].map(c_text)
    pairs = list(zip(te["q"], te["c"]))
    model.eval()
    scores = np.zeros(len(pairs), np.float32)
    with torch.no_grad():
        for i in range(0, len(pairs), 64):
            batch = pairs[i:i + 64]
            enc = tok([p[0] for p in batch], [p[1] for p in batch], truncation=True,
                      max_length=args.seq_len, padding=True, return_tensors="pt").to(device)
            scores[i:i + 64] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    te["ft"] = scores

    cuts = {"lenient_ge3": 3, "strict_ge5": 5}
    rec = {"created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "key": args.key, "base_model": args.model, "seq_len": args.seq_len,
           "epochs": args.epochs, "lr": args.lr, "finetuned": True, "by_cut": {}}
    for ck, cv in cuts.items():
        cp, chr_ = rerank_metrics(te, "ft", cv)
        rec["by_cut"][ck] = {"p_at_3": cp, "hr_at_3": chr_}
    rec["stage1_gate"] = {k: gate_on(te, "ft")[k] for k in
                          ("chunk_auc_grade3", "chunk_auc_grade5",
                           "within_bundle_concordance", "bundle_any_relevant_auc_top1")}
    with open(out_dir / f"{args.key}-finetuned.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\n=== {args.key} fine-tuned (TEST) ===")
    print("lenient:", rec["by_cut"]["lenient_ge3"], " strict:", rec["by_cut"]["strict_ge5"])
    print("Stage-1 gate:", rec["stage1_gate"])

    if args.save_model:
        save_dir = out_dir / f"{args.key}-finetuned-model"
        model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
        print("saved model to", save_dir, flush=True)


if __name__ == "__main__":
    main()
