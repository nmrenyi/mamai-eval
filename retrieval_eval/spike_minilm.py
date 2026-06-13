#!/usr/bin/env python3
"""R2c Phase-0 + Option-B spike for ms-marco-MiniLM-L6-v2 (BERT cross-encoder).

Joint probe on the smallest standard-BERT cross-encoder:
  - convertibility (layers 1-2): export to ONNX, quantize int8 (ARM-targeted),
  - CPU latency (layer 3 proxy — this Mac is arm64): torch vs onnx-fp32 vs
    onnx-int8, per-(query,chunk) and per 20-chunk query,
  - offline quality: zero-shot rerank of the hybrid top-20 on the held-out test
    split vs the feature-LTR bar and oracle, plus the R1 Stage-1 gate.

The on-phone run (ORT-Mobile/LiteRT + real device latency = layer 4) is handled
separately; this Mac arm64 result is a strong proxy, not the final word.

Usage:
  python -m retrieval_eval.spike_minilm \\
      --features-dir configs/config-v0.2.0/results/retrieval_eval/r2c-rerank \\
      --report-dir   configs/config-v0.2.0/reports/r2c-rerank \\
      --onnx-dir     /tmp/minilm-onnx
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_eval.compare_retriever_gates import gate_stats

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


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
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--onnx-dir", default="/tmp/minilm-onnx")
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    out = {"created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "model": MODEL_ID}

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
    print(f"test pool: {len(te)} pairs, {te['query_id'].nunique()} queries", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    # --- quality: score all pairs (torch CPU, batched) ---
    scores = np.zeros(len(pairs), dtype=np.float32)
    t0 = time.time(); B = 32
    with torch.no_grad():
        for i in range(0, len(pairs), B):
            batch = pairs[i:i + B]
            enc = tok([p[0] for p in batch], [p[1] for p in batch],
                      padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt")
            scores[i:i + B] = model(**enc).logits.squeeze(-1).numpy()
            if (i // B) % 50 == 0:
                print(f"  scored {i}/{len(pairs)}", flush=True)
    te["ce_score"] = scores
    print(f"scoring done in {time.time() - t0:.0f}s", flush=True)

    cuts = {"lenient_ge3": 3, "strict_ge5": 5}
    out["quality"] = {"n_test_queries": int(te["query_id"].nunique()), "by_cut": {}}
    for ck, cv in cuts.items():
        fp, fhr = rerank_metrics(te, "rrf_score", cv)
        cp, chr_ = rerank_metrics(te, "ce_score", cv)
        op, ohr = rerank_metrics(te, "grade", cv)
        out["quality"]["by_cut"][ck] = {
            "floor_rrf": {"p_at_3": fp, "hr_at_3": fhr},
            "minilm_ce": {"p_at_3": cp, "hr_at_3": chr_},
            "oracle": {"p_at_3": op, "hr_at_3": ohr}}
    g = gate_on(te, "ce_score")
    out["quality"]["stage1_gate_on_ce_score"] = {
        k: g[k] for k in ("chunk_auc_grade3", "within_bundle_concordance",
                          "bundle_any_relevant_auc_top1")}

    # --- convertibility: ONNX export + int8 (arm64-targeted) ---
    conv = {"onnx_export": None, "int8_quantize": None}
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        onnx_dir = Path(args.onnx_dir); onnx_dir.mkdir(parents=True, exist_ok=True)
        ort_model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
        ort_model.save_pretrained(onnx_dir)
        fp32 = onnx_dir / "model.onnx"
        conv["onnx_export"] = {"ok": True, "size_mb": round(fp32.stat().st_size / 1e6, 1)}
        quantizer = ORTQuantizer.from_pretrained(onnx_dir)
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=onnx_dir / "int8", quantization_config=qconfig)
        q = next((onnx_dir / "int8").glob("*quantized*.onnx"), None)
        conv["int8_quantize"] = {"ok": True, "arm64_config": True,
                                 "size_mb": round(q.stat().st_size / 1e6, 1) if q else None}
        print(f"ONNX export + arm64 int8 OK ({conv['onnx_export']['size_mb']} MB -> "
              f"{conv['int8_quantize']['size_mb']} MB)", flush=True)
    except Exception as e:
        conv["error"] = f"{type(e).__name__}: {e}"
        print(f"convertibility FAILED: {conv['error']}", flush=True)
    out["convertibility"] = conv

    # --- CPU latency (this Mac is arm64): a 20-chunk query ---
    def bench(fn, n=5):
        fn()  # warmup
        t = time.time()
        for _ in range(n):
            fn()
        return round((time.time() - t) / n * 1000, 1)  # ms
    lat = {}
    sample = pairs[:20]
    enc20 = tok([p[0] for p in sample], [p[1] for p in sample], padding=True,
                truncation=True, max_length=args.max_len, return_tensors="pt")
    with torch.no_grad():
        lat["torch_fp32_20chunks_ms"] = bench(lambda: model(**enc20))
    if conv.get("int8_quantize", {}) and conv["int8_quantize"].get("ok"):
        import onnxruntime as ort
        sess = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
        feed = {k: v.numpy() for k, v in enc20.items()
                if k in {i.name for i in sess.get_inputs()}}
        lat["onnx_int8_20chunks_ms"] = bench(lambda: sess.run(None, feed))
    lat["note"] = "Mac arm64 CPU; proxy for phone ARM, not the on-device figure"
    out["cpu_latency"] = lat

    with open(report_dir / "minilm_spike.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== MiniLM-L6 spike ===")
    for ck in cuts:
        c = out["quality"]["by_cut"][ck]
        print(f"{ck}: floor P@3={c['floor_rrf']['p_at_3']} -> MiniLM {c['minilm_ce']['p_at_3']} "
              f"-> oracle {c['oracle']['p_at_3']} | HR floor {c['floor_rrf']['hr_at_3']} "
              f"-> MiniLM {c['minilm_ce']['hr_at_3']}")
    print("Stage-1 gate (CE score):", out["quality"]["stage1_gate_on_ce_score"])
    print("convertibility:", conv)
    print("latency:", lat)
    print(f"Written: {report_dir}/minilm_spike.json")


if __name__ == "__main__":
    main()
