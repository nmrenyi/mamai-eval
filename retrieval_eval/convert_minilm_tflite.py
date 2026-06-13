#!/usr/bin/env python3
"""Convert ms-marco-MiniLM-L6-v2 to LiteRT/.tflite (the app's runtime).

Proves the convertibility step that matters most for this app: not just ONNX,
but LiteRT/TFLite, which is what the device actually runs. Uses ai-edge-torch
(the PyTorch->LiteRT path). Exports a logits-only wrapper at a fixed shape
(batch=1, seq_len), with and without int8 dynamic quantization, and reports the
.tflite sizes. On-Mac (arm64) LiteRT-interpreter latency is measured as a
proxy; the true on-phone figure needs the device benchmark (see report).

Usage:
  python -m retrieval_eval.convert_minilm_tflite --seq-len 256 --out-dir /tmp/minilm-tflite
"""

import argparse
import json
import time
from pathlib import Path

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out-dir", default="/tmp/minilm-tflite")
    ap.add_argument("--report-dir",
                    default="configs/config-v0.2.0/reports/r2c-rerank")
    args = ap.parse_args()

    import litert_torch as ai_edge_torch
    import torch
    from transformers import AutoModelForSequenceClassification

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    class LogitsOnly(torch.nn.Module):
        """ai-edge-torch needs tensor outputs, not the HF output dataclass."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits

    wrapped = LogitsOnly(base).eval()
    S = args.seq_len
    sample = (torch.ones(1, S, dtype=torch.long),
              torch.ones(1, S, dtype=torch.long),
              torch.zeros(1, S, dtype=torch.long))

    out = {"model": MODEL_ID, "seq_len": S, "fp32": {}, "int8": {}}

    # fp32 LiteRT
    edge = ai_edge_torch.convert(wrapped, sample)
    fp32_path = out_dir / f"minilm_l6_s{S}.tflite"
    edge.export(str(fp32_path))
    out["fp32"] = {"ok": True, "path": str(fp32_path),
                   "size_mb": round(fp32_path.stat().st_size / 1e6, 1)}
    print(f"LiteRT fp32 OK: {out['fp32']['size_mb']} MB -> {fp32_path}", flush=True)

    # int8 dynamic-quantized LiteRT (best-effort; ONNX already proved arm64 int8)
    try:
        from litert_torch.generative.quantize import quant_recipes
        recipe = quant_recipes.full_dynamic_recipe()  # int8 dynamic weights
        edge_q = ai_edge_torch.convert(wrapped, sample, quant_config=recipe)
        int8_path = out_dir / f"minilm_l6_s{S}_int8.tflite"
        edge_q.export(str(int8_path))
        out["int8"] = {"ok": True, "path": str(int8_path),
                       "size_mb": round(int8_path.stat().st_size / 1e6, 1)}
        print(f"LiteRT int8 OK: {out['int8']['size_mb']} MB -> {int8_path}", flush=True)
    except Exception as e:
        out["int8"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"LiteRT int8 failed: {out['int8']['error']}", flush=True)

    # Mac arm64 LiteRT-interpreter latency proxy (one (query,chunk) pass).
    try:
        from ai_edge_litert.interpreter import Interpreter
        import numpy as np
        interp = Interpreter(model_path=str(fp32_path)); interp.allocate_tensors()
        ins = interp.get_input_details()
        feeds = [np.ones(d["shape"], dtype=d["dtype"]) for d in ins]

        def run():
            for d, f in zip(ins, feeds):
                interp.set_tensor(d["index"], f)
            interp.invoke()
        run()
        t = time.time()
        for _ in range(10):
            run()
        per_pair = (time.time() - t) / 10 * 1000
        out["mac_arm64_litert_latency"] = {
            "per_pair_ms": round(per_pair, 1),
            "est_20chunks_ms": round(per_pair * 20, 1),
            "note": "Mac arm64 LiteRT interpreter; proxy for phone, not the device figure"}
        print(f"Mac LiteRT latency: {per_pair:.1f} ms/pair, "
              f"~{per_pair * 20:.0f} ms/20-chunks", flush=True)
    except Exception as e:
        out["mac_arm64_litert_latency"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"latency proxy failed: {out['mac_arm64_litert_latency']['error']}", flush=True)

    with open(report_dir / "minilm_tflite.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {report_dir}/minilm_tflite.json")


if __name__ == "__main__":
    main()
