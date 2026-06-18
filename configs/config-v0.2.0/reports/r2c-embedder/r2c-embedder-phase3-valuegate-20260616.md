# R2c Phase 3 — end-to-end value gate (kenya SAQ)

*2026-06-16. Generator gemma4-e4b (deployed), judge gpt-oss-120b + V2 SAQ rubric (key-fact
recall). EmbeddingGemma arm (top-3) vs the matched gemma4-e4b baselines already in-repo.*

## kenya SAQ key-fact recall (gemma4-e4b)
| arm | recall | harm | refusal |
|---|---|---|---|
| **no-RAG** | **0.178** | — | — |
| EmbeddingGemma RAG | 0.1255 | 0.183 | 0.003 |
| Gecko RAG (deployed) | 0.1254 | 0.186 | 0.000 |
| Hybrid RAG | 0.1124 | — | — |
| Hybrid+rerank RAG | 0.1112 | — | — |

## Verdict — the retrieval win does NOT convert end-to-end on kenya
- **EmbeddingGemma 0.1255 ≈ Gecko 0.1254** (Δ ≈ 0). Its decisive +12–13 pp *retrieval* win (Phase 2)
  produces **no** answer-quality gain.
- **RAG is net-negative on kenya:** no-RAG (0.178) beats *every* retrieval arm. The generator does
  worse with the corpus than without it — so the ceiling is the **corpus + generator grounding**,
  not the embedder's ranking.
- Consistent with **P-1** (kenya coverage-poor, ~1/5 addressable) and the **reranker line** (offline
  gains that don't transfer).

## ⚠ Baseline correction (generator trap)
The "gecko 0.256 / hybrid 0.271" figures used earlier are the **retracted gemma3n** run
(`value_gate/`), a different generator. The correct deployed-generator baseline is
`value_gate_g4/` (gemma4-e4b): **gecko 0.1254**. Lesson re-confirmed: verify the generator before
trusting any end-to-end number.

## What this means for R2c
- **EmbeddingGemma is a clear, deployable *retrieval* upgrade** (Phase 0 deployable + latency-neutral;
  Phase 2 +12–13 pp P@3/HR@20) — worth adopting on retrieval grounds, and the right base if/when the
  corpus improves.
- **But it does not move kenya answers**, because kenya's end-to-end ceiling is corpus coverage +
  generator-grounding, not retrieval ranking. For kenya, the lever is **R-corpus (content expansion)
  + generator grounding**, not the embedder.

## Remaining test
**healthbench-oss_eval** (coverage-good, large-n) is the fair end-to-end test of whether the
retrieval win converts *where the corpus actually contains the answers*. kenya (coverage-poor) cannot
show it. Run EmbeddingGemma vs gecko on healthbench (generate + rubric) before the final R2c call.
