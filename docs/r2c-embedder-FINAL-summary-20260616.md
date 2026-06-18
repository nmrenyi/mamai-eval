# R2c embedder bake-off — FINAL summary (2026-06-16)

EmbeddingGemma-300M vs deployed Gecko, on-device medical RAG (LiteRT-only, low-mid Zanzibar,
English-primary). All phases complete. Companion docs: lit-review, test/autonomous plans, and the
per-phase results docs in `docs/r2c-embedder-*`.

## Results by phase

| phase | question | result |
|---|---|---|
| **P0 — deployability** | can EmbeddingGemma ship on-device? | ✅ **Yes.** Official LiteRT int8, 125 ms@4t / 249 ms@1t (seq256, CPU), 187 MB. **Latency-neutral-to-better** vs Gecko (Gecko's real on-device retrieval ≈4 s; embed is search-dominated, EmbeddingGemma's embed is *faster*). |
| **P2 — retrieval quality** | better retriever? | ✅ **Decisively.** kenya, Qwen3-32B judge: P@3 0.396 vs 0.270 (**+12.6 pp**), strict 0.284 vs 0.172 (**+11.2 pp**), HR@20 0.734 vs 0.609 (**+12.5 pp**). All ≫ noise floor. |
| **P1 — corpus coverage** | is the answer even in the corpus? | Corpus reaches **89% strict / 96% lenient** (7-retriever union); only **~10.6%** truly absent. EmbeddingGemma 0.734 strict = best deployable, +10 pp over Gecko, near octen-8B/voyage ceilings. ~26 pp is in-corpus but Gecko-missed (ranking-fixable). |
| **P3 — end-to-end (kenya SAQ)** | does it improve *answers*? | ❌ **No conversion.** EmbeddingGemma 0.1255 ≈ Gecko 0.1254 key-fact recall (gemma4-e4b). **RAG is net-negative** here: no-RAG 0.178 > every retrieval arm. |
| **P3 — end-to-end (healthbench)** | …on a coverage-good set? | ~**Marginal.** weighted_met +0.012 vs −0.004 (Δ+0.016); positive +1.4 pp, penalty −0.4 pp. Both ≈ 0. |

## The synthesis
Three facts that only make sense together:
1. **EmbeddingGemma is a clear, deployable retrieval upgrade** (+12–13 pp retrieval, +10 pp coverage, ships on-device, latency-neutral).
2. **The corpus is mostly fine** — it contains the answer for ~89% of kenya queries; absence is ~10%.
3. **Yet end-to-end answers barely move** (kenya none, healthbench marginal), and RAG is *net-negative* on kenya with the deployed generator.

→ **The binding ceiling is the generator, not the embedder or the corpus.** gemma4-e4b does not
convert retrieved context into better answers — it often does worse *with* context. Better retrieval
(EmbeddingGemma) puts more relevant chunks in front of it, but the generator can't capitalize.

## Recommendation
1. **Adopt EmbeddingGemma-300M as the on-device retriever** — it's a strict, low-risk improvement on
   the retrieval axis (deployable, faster embed, multilingual bonus for future Swahili/German), and
   the best deployable option by a clear margin. Low downside even though it doesn't move today's
   answers.
2. **Prioritize generator RAG-grounding as the end-to-end lever** — prompt/format work and/or
   RAG-aware fine-tuning of gemma4-e4b so it actually uses (and isn't hurt by) injected context. This
   is where the answer-quality ceiling lives. *(New workstream — call it G-RAG.)*
3. **Corpus expansion (R-corpus) is a minor lever** (~10% absent) — lower priority than (2).
4. **R2d (fine-tune the embedder) is not warranted now** — retrieval is already strong; the
   bottleneck moved downstream to the generator.

## Caveats
- End-to-end sets are small/medium (kenya n=312, healthbench n=1209); deltas reported against a
  pre-registered 5 pp noise floor — the marginal healthbench edge is within noise.
- Test device was a flagship (SM8750); low-mid Zanzibar latency will be higher (single-thread CPU is
  the proxy). Deployability holds (CPU+int8), but confirm on a true low-mid device before release.
- bake-off judged with Qwen3-32B (retrieval/coverage) + gpt-oss-120b (SAQ/rubric), per the pinned
  v0.2 judges.
