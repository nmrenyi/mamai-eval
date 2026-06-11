# Device (LiteRT) vs Host (GGUF) fidelity — v0.2 open-ended

**Status: in progress.** Phase B was generated on the *host* proxy (llama.cpp +
Q4_0 GGUF). This compares it against the deployed **LiteRT-LM device** stack on
the same questions, to test how good a proxy the host is.

Device runs via `run_eval_device.py` in fresh-process batches of 8 (the device
OS kills long sustained LLM runs; short batches + retry survive). Same gpt-oss-120b
judge will rescore both sides (pending).

## MCQ (medmcqa, no-RAG, n=100) — done
- Aggregate accuracy: device 0.57 vs host 0.54 (~+3 pp).
- Per-question agreement: **64%** (36/100 differ) — but inflated by temperature 1.0
  sampling noise; a floor, not the true divergence.
- Read: proxy good at the **aggregate** level, loose **per-item**.

## SAQ no-RAG (whb 20 + afrimedqa_saq 37 + kenya 312 = 369) — done, raw only
Same 369 questions both stacks. Judge scoring pending.

| dataset | device defer/refuse* | host defer/refuse* | dev avg chars | host avg chars |
|---|---|---|---|---|
| kenya | 67% | 31% | 991 | 941 |
| whb | 75% | 55% | 467 | 426 |
| afrimedqa_saq | 24% | 27% | 614 | 534 |

\* crude regex on the response prefix (defer/refuse language); same heuristic both sides.

- **Leaked chain-of-thought: 0/369 on device** — confirms the rubric-track leaked-CoT
  is a *host-eval artifact* (manual GGUF template), not device behavior. LiteRT's
  conversation API delivers clean final answers.
- **Response lengths: close** (device slightly longer) — generation is in the same range.
- **Deflection signal higher on device** (kenya 67% vs 31%): *suggestive* that the
  real LiteRT stack deflects MORE than the host proxy — i.e. the proxy may
  understate the safe-but-deflecting problem. Regex-based; to be confirmed by the
  judge-scored recall (lower device recall would confirm).

## Pending
- SAQ +RAG (on-device retrieval) — running.
- Healthbench ±RAG — stratified sample (full 2,339/arm is ~12h/arm on device).
- Judge rescore (gpt-oss-120b, cluster) of all device results → recall/safety
  device-vs-host deltas.
