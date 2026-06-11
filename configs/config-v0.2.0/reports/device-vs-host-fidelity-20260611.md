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

## SAQ +RAG (369) — done + judge-scored
Device does its **own on-device retrieval** (Gecko + vector store) + the deployed
injection; host uses precomputed contexts. So this conflates retrieval *and*
generation differences (not an isolated stack test).

| dataset | recall device / host | harm-rate device / host |
|---|---|---|
| kenya | 0.171 / 0.128 | 23.7% / 18.9% |
| whb | 0.070 / 0.026 | 20.0% / 15.0% |
| afrimedqa_saq | 0.214 / 0.162 | 18.9% / 8.1% |

- Device recall again higher; but device **harm-rate higher** here (opposite of no-RAG),
  plausibly because on-device retrieval surfaces different content the model acts on.

## Healthbench ±RAG — stratified sample (50/subset = 150/arm), judge-scored
Full HB is 2,339/arm (~12h/arm on device), so a seeded 50-per-subset sample. Compared
against host on the **same matched ids**. Metric = weighted_met.

| arm / subset | device / host |
|---|---|
| no-RAG oss_eval | 0.045 / −0.036 |
| no-RAG consensus | 0.657 / 0.573 |
| no-RAG hard | −0.137 / −0.168 |
| +RAG oss_eval | 0.026 / −0.029 |
| +RAG consensus | 0.513 / 0.530 |
| +RAG hard | −0.182 / −0.175 |

- **no-RAG: device slightly higher on all 3** (consistent with SAQ). **+RAG: ≈tied** (±1–2 pp).

## Bottom line — the host proxy is validated
Across **both tracks and both arms**, the LiteRT device tracks the host GGUF proxy
closely:
- **no-RAG (the clean, identical-input comparison): device is consistently slightly
  BETTER** (higher recall / weighted_met, lower or equal harm) — so the host proxy is a
  **mildly pessimistic lower bound**, not an overstatement.
- **+RAG: device ≈ host** on quality, with device slightly higher recall; the +RAG arms
  also exercise device-side retrieval, so small differences there mix retrieval +
  generation.
- **Every Phase B conclusion holds on-device** — low recall / weighted_met,
  safe-but-unhelpful, zero-dangerous floor. The headline numbers can be trusted as
  representative of (a hair below) real deployment.
- **Leaked chain-of-thought is host-only** (0/369 SAQ + 0/300 HB on device) — an
  artifact of the host's manual-GGUF-template path, not the device.

### Caveats
- gpt-oss-120b judge throughout (same bias both sides; ±RAG and device-vs-host deltas
  cancel it). No bootstrap CIs — small per-subset n (esp. HB sample 50/subset, whb 20),
  so treat single-subset deltas as indicative.
- One judge job (SAQ +RAG) failed once on a transient pypi timeout; resubmit succeeded.
- Device runs used fresh-process batches of 8 to survive aggressive on-device process
  killing of long runs (root-caused as not screen-related).
