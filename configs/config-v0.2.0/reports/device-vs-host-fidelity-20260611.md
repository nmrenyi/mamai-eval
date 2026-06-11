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

## SAQ no-RAG (whb 20 + afrimedqa_saq 37 + kenya 312 = 369) — done + judge-scored
Same 369 questions both stacks, both rescored by the pinned gpt-oss-120b judge.

| dataset | recall device / host | harm-rate device / host |
|---|---|---|
| kenya | **0.194 / 0.178** | 19.2% / 20.8% |
| whb | 0.079 / 0.039 | 5.0% / 20.0% |
| afrimedqa_saq | 0.211 / 0.164 | 8.1% / 16.2% |

- **Device is slightly BETTER on both axes** — higher recall *and* lower harm rate
  than the host on all three. So the host GGUF proxy is a **mildly pessimistic**
  approximation (likely Q4_0 GGUF being lossier than the LiteRT bundle); host
  numbers are a safe lower bound. Every Phase B conclusion (low recall,
  safe-but-unhelpful, zero-dangerous floor) holds on-device.
- **Leaked chain-of-thought: 0/369 on device** — confirms the rubric-track leaked-CoT
  is a *host-eval artifact* (manual GGUF template), not device behavior.
- **Response lengths: close** (device slightly longer).
- **Correction:** a crude defer/refuse *regex* had suggested the device deflects MORE
  (kenya 67% vs 31%) — but the judge scores show the opposite (higher device recall).
  The regex was catching extra "consult a doctor" phrasing on answers that still
  convey more content; **disregard the raw deflection signal** in favor of the
  judge-scored recall above.

## Pending
- SAQ +RAG (on-device retrieval) — running.
- Healthbench ±RAG — stratified sample (full 2,339/arm is ~12h/arm on device).
- Judge rescore (gpt-oss-120b, cluster) of all device results → recall/safety
  device-vs-host deltas.
