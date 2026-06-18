#!/usr/bin/env python3
"""R2c P0 — does the DEPLOYED reranker reproduce the offline-validated quality?

Runs the actual on-device artifact (reranker_minilm_l6_int8.tflite) locally via
the LiteRT interpreter over the full test split, fed two ways:
  (HF)  the HuggingFace BertTokenizerFast inputs, and
  (KT)  a faithful Python port of the app's Kotlin WordPieceTokenizer.

Three faithfulness questions, decomposed:
  1. Tokenizer parity  — do KT and HF produce identical input_ids/type_ids?
     (the app uses KT; we validated quality with HF.)
  2. Conversion+quant  — does tflite(HF) reproduce the offline torch-fp32 @256
     quality (P@3/HR@3/Stage-1 gate)?  [torch-fp32 @256 read from the
     score_candidates minilm-l6 run]
  3. Deployed quality  — tflite(KT) is exactly what the phone computes (modulo
     int64 plumbing the app already exercises). Its P@3/HR@3/gate is the
     deployed model's TRUE seq-256 quality.

This is the local, decisive form of P0.1/0.2/0.3. The only residual vs a logcat
round-trip is on-device int8 kernel arithmetic, which LiteRT computes with the
same integer kernels — confirmatory, not load-bearing.

Usage:
  python -m retrieval_eval.check_parity \\
      --tflite /Users/renyi/Downloads/mamai/device_push/models/reranker_minilm_l6_int8.tflite \\
      --vocab  /Users/renyi/Downloads/mamai/device_push/models/reranker_vocab.txt \\
      --features-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \\
      --out-dir      configs/config-v0.2.0/results/retrieval_eval/r2c-rerank/candidates
"""

import argparse
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_eval.compare_retriever_gates import gate_stats


# ---- faithful Python port of app/.../WordPieceTokenizer.kt ----
class KotlinPortTokenizer:
    def __init__(self, vocab_path, max_len=256):
        self.vocab = {}
        with open(vocab_path) as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.cls = self.vocab.get("[CLS]", 101)
        self.sep = self.vocab.get("[SEP]", 102)
        self.pad = self.vocab.get("[PAD]", 0)
        self.unk = self.vocab.get("[UNK]", 100)
        self.max_len = max_len
        self.max_chars = 100

    @staticmethod
    def _is_punct(ch):
        cp = ord(ch)
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        return unicodedata.category(ch).startswith("P")

    def _basic(self, text):
        lowered = text.lower()
        stripped = "".join(c for c in unicodedata.normalize("NFD", lowered)
                           if unicodedata.category(c) != "Mn")
        out, sb = [], []
        for ch in stripped:
            if ch.isspace():
                if sb:
                    out.append("".join(sb)); sb = []
            elif self._is_punct(ch):
                if sb:
                    out.append("".join(sb)); sb = []
                out.append(ch)
            else:
                sb.append(ch)
        if sb:
            out.append("".join(sb))
        return out

    def _wordpiece(self, token, out):
        if len(token) > self.max_chars:
            out.append(self.unk); return
        start, pieces = 0, []
        while start < len(token):
            end = len(token); cur = -1
            while start < end:
                sub = ("##" if start > 0 else "") + token[start:end]
                if sub in self.vocab:
                    cur = self.vocab[sub]; break
                end -= 1
            if cur == -1:
                out.append(self.unk); return
            pieces.append(cur); start = end
        out.extend(pieces)

    def _ids(self, text):
        ids = []
        for tok in self._basic(text):
            self._wordpiece(tok, ids)
        return ids

    def encode_pair(self, query, doc):
        q = self._ids(query)
        budget = self.max_len - 3
        q = q[:budget]
        d = self._ids(doc)[: budget - len(q)]
        ids = [self.pad] * self.max_len
        mask = [0] * self.max_len
        typ = [0] * self.max_len
        p = 0
        ids[p] = self.cls; mask[p] = 1; typ[p] = 0; p += 1
        for i in q:
            ids[p] = i; mask[p] = 1; typ[p] = 0; p += 1
        ids[p] = self.sep; mask[p] = 1; typ[p] = 0; p += 1
        for i in d:
            ids[p] = i; mask[p] = 1; typ[p] = 1; p += 1
        ids[p] = self.sep; mask[p] = 1; typ[p] = 1; p += 1
        return (np.array(ids, np.int64), np.array(mask, np.int64), np.array(typ, np.int64))


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


def top3_set_agreement(df, col_a, col_b):
    agree, n = 0.0, 0
    for _, grp in df.groupby("query_id"):
        a = set(grp.sort_values([col_a, "chunk_id"], ascending=[False, True]).head(3)["chunk_id"])
        b = set(grp.sort_values([col_b, "chunk_id"], ascending=[False, True]).head(3)["chunk_id"])
        agree += len(a & b) / 3.0; n += 1
    return round(agree / n, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default="nmrenyi/mamaretrieval")
    ap.add_argument("--revision", default="v0.2.0")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--hf-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    import pandas as pd
    from ai_edge_litert.interpreter import Interpreter
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(Path(args.features_dir) / "ltr_features.parquet")
    te = df[df["split"] == "test"][["query_id", "chunk_id", "grade", "rrf_score"]].copy()
    q_text = {r.query_id: r.query_text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/queries.parquet", repo_type="dataset", revision=args.revision)).itertuples()}
    c_text = {r.chunk_id: r.text for r in pd.read_parquet(hf_hub_download(
        args.hf_repo, "data/chunks.parquet", repo_type="dataset", revision=args.revision)).itertuples()}
    te["q"] = te["query_id"].map(q_text)
    te["c"] = te["chunk_id"].map(c_text)
    pairs = list(zip(te["q"], te["c"]))
    print(f"test pool: {len(te)} pairs, {te['query_id'].nunique()} queries", flush=True)

    hf = AutoTokenizer.from_pretrained(args.hf_model)
    kt = KotlinPortTokenizer(args.vocab, args.seq_len)

    it = Interpreter(model_path=args.tflite); it.allocate_tensors()
    ins = it.get_input_details(); outd = it.get_output_details()[0]

    def run_tflite(ids, mask, typ):
        # input order = serving_default_args_0/1/2 = input_ids, attn, type
        it.set_tensor(ins[0]["index"], ids.reshape(1, -1))
        it.set_tensor(ins[1]["index"], mask.reshape(1, -1))
        it.set_tensor(ins[2]["index"], typ.reshape(1, -1))
        it.invoke()
        return float(it.get_tensor(outd["index"])[0, 0])

    n_pairs = len(pairs)
    sc_hf = np.zeros(n_pairs, np.float32)
    sc_kt = np.zeros(n_pairs, np.float32)
    tok_exact = 0          # pairs where KT input_ids == HF input_ids (over real tokens)
    tok_id_disagree = 0    # total token-position disagreements
    tok_total = 0
    t0 = time.time()
    for i, (q, d) in enumerate(pairs):
        h = hf(q, d, max_length=args.seq_len, truncation="only_second",
               padding="max_length", return_token_type_ids=True)
        h_ids = np.array(h["input_ids"], np.int64)
        h_mask = np.array(h["attention_mask"], np.int64)
        h_typ = np.array(h["token_type_ids"], np.int64)
        k_ids, k_mask, k_typ = kt.encode_pair(q, d)
        # token parity over the union of attended positions
        attn = (h_mask | k_mask).astype(bool)
        disagree = int(np.sum(h_ids[attn] != k_ids[attn]))
        tok_id_disagree += disagree
        tok_total += int(attn.sum())
        if disagree == 0 and int(h_mask.sum()) == int(k_mask.sum()):
            tok_exact += 1
        sc_hf[i] = run_tflite(h_ids, h_mask, h_typ)
        sc_kt[i] = run_tflite(k_ids, k_mask, k_typ)
        if i % 1000 == 0:
            print(f"  {i}/{n_pairs} ({time.time()-t0:.0f}s)", flush=True)
    print(f"tflite scoring done in {time.time()-t0:.0f}s", flush=True)

    te["tflite_hf"] = sc_hf
    te["tflite_kt"] = sc_kt

    cuts = {"lenient_ge3": 3, "strict_ge5": 5}
    rec = {"created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "tflite": args.tflite, "seq_len": args.seq_len,
           "tokenizer_parity": {
               "pairs": n_pairs,
               "exact_match_pairs": tok_exact,
               "exact_match_rate": round(tok_exact / n_pairs, 4),
               "token_positions": tok_total,
               "token_disagreements": tok_id_disagree,
               "token_disagree_rate": round(tok_id_disagree / max(tok_total, 1), 6)},
           "score_parity_kt_vs_hf": {
               "max_abs_diff": round(float(np.max(np.abs(sc_kt - sc_hf))), 5),
               "mean_abs_diff": round(float(np.mean(np.abs(sc_kt - sc_hf))), 6),
               "pearson_r": round(float(np.corrcoef(sc_kt, sc_hf)[0, 1]), 6),
               "top3_set_agreement": top3_set_agreement(te, "tflite_kt", "tflite_hf")},
           "quality": {"by_cut": {}}}
    for ck, cv in cuts.items():
        hp, hhr = rerank_metrics(te, "tflite_hf", cv)
        kp, khr = rerank_metrics(te, "tflite_kt", cv)
        rec["quality"]["by_cut"][ck] = {
            "tflite_hf": {"p_at_3": hp, "hr_at_3": hhr},
            "tflite_kt_deployed": {"p_at_3": kp, "hr_at_3": khr}}
    rec["quality"]["stage1_gate_tflite_kt"] = {k: gate_on(te, "tflite_kt")[k] for k in
        ("chunk_auc_grade3", "chunk_auc_grade5", "within_bundle_concordance",
         "bundle_any_relevant_auc_top1")}

    # compare to offline torch-fp32 @ same seq-len, if the score_candidates run exists
    ml = out_dir / "minilm-l6.json"
    if ml.exists():
        off = json.load(open(ml))
        if off.get("max_len") == args.seq_len:
            rec["offline_torch_fp32_same_seqlen"] = off["by_cut"]

    with open(out_dir / "parity.json", "w") as f:
        json.dump(rec, f, indent=2)

    print("\n=== P0 parity ===")
    print("tokenizer exact-match rate:", rec["tokenizer_parity"]["exact_match_rate"],
          " token disagree rate:", rec["tokenizer_parity"]["token_disagree_rate"])
    print("score parity KT vs HF:", rec["score_parity_kt_vs_hf"])
    for ck in cuts:
        c = rec["quality"]["by_cut"][ck]
        print(f"{ck}: tflite-HF P@3={c['tflite_hf']['p_at_3']} | "
              f"tflite-KT(deployed) P@3={c['tflite_kt_deployed']['p_at_3']}")
    print("deployed Stage-1 gate:", rec["quality"]["stage1_gate_tflite_kt"])
    if "offline_torch_fp32_same_seqlen" in rec:
        print("offline torch-fp32 @seqlen:", rec["offline_torch_fp32_same_seqlen"])
    print("Written:", out_dir / "parity.json")


if __name__ == "__main__":
    main()
