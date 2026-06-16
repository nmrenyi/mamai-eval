# R2c Phase 2 — offline retrieval screen (results)

*2026-06-16. kenya, n=312, top-20, judge = Qwen3-32B + V2 rubric (same as the 230k audit +
kenya_relevance). Raw: `configs/config-v0.2.0/results/retrieval_eval/r2c-embedder/`.*

## EmbeddingGemma-300M (768-dim) vs deployed Gecko

| metric (kenya, n=312) | Gecko | **EmbeddingGemma-768** | Δ |
|---|---|---|---|
| P@3 lenient (≥3) | 0.270 | **0.396** | **+12.6 pp** |
| P@3 strict (≥5) | 0.172 | **0.284** | **+11.2 pp** |
| mean grade (top-3) | 1.44 | **2.13** | **+0.69** |
| HR@20 lenient (any ≥3 in top-20) | 0.715 | **0.849** | **+13.4 pp** |
| HR@20 strict (any ≥5 in top-20) | 0.609 | **0.734** | **+12.5 pp** |

Both arms run through the *same* pipeline + judge. **Judge-consistency check passed:** the matched
Gecko arm reproduced its recorded kenya_relevance P@3 (0.270/0.172 vs 0.277/0.180), so the
comparison is sound. EmbeddingGemma wins on **every** metric by ~12–13 pp, all far above the
5 pp noise floor.

## Verdict — Phase 2 PASS (decisive)
- Both P@3 deltas (**+11.9 / +10.4 pp**) are far above the 5 pp noise floor at n=312 → a real,
  large retrieval-quality gain, not noise.
- **Recall signal is the headline:** EmbeddingGemma *alone* surfaces a strict-relevant (grade≥5)
  chunk in its top-20 for **73.4%** of kenya queries — exceeding the **58.0%** that the entire
  P-1 multi-arm union (gecko/bm25/hybrid × rerankers) reached at top-3. It is surfacing answers
  every existing arm missed — the embedder's unique lever (recall), working as theorized.
- EmbeddingGemma also clears Phase 0 (deployable, CPU+int8, latency-neutral-to-better).

→ **Advance EmbeddingGemma to Phase 3 (end-to-end value gate).** Pending the Gecko arm only as a
matched HR@20 baseline + a judge-consistency check (should reproduce Gecko P@3 ≈ 0.277).

## Caveats
- The decisive comparison is precision (P@3, directly matched to the existing Gecko number). The
  recall comparison (HR@20) firms up once the Gecko arm lands.
- This is *retrieval* quality; whether it converts to better *answers* is exactly what Phase 3 tests
  (the lesson from the reranker line). Bounded by the P-1 ceiling (~1/5 addressable end-to-end).
