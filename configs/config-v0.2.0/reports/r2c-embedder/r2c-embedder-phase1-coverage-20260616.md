# R2c Phase 1 — top-20 corpus coverage, 7 retrievers (results)

*2026-06-16. kenya n=312. Per-query UNION across 7 retrievers' top-20, judged Qwen3-32B + V2
rubric (same scale as the 230k audit). 36,471 unique (query,chunk) pairs.
Raw: `configs/config-v0.2.0/results/retrieval_eval/r2c-embedder/kenya_coverage_7retr.json`.*

## Per-retriever coverage (share of kenya queries with a relevant chunk in top-20)
| retriever | lenient (≥3) | strict (≥5) | type |
|---|---|---|---|
| **voyage-4-large** | 0.942 | **0.869** | API ceiling |
| **octen-8B** | 0.914 | 0.817 | 8B ceiling |
| **EmbeddingGemma-300M** | **0.859** | **0.734** | **deployable candidate** |
| gecko (deployed) | 0.753 | 0.635 | deployable baseline |
| bm25 | 0.279 | 0.221 | lexical |
| lateon (ColBERT) | 0.192 | 0.125 | late-interaction |
| medcpt | 0.164 | 0.058 | medical (PubMed) |
| **UNION (all 7)** | **0.965** | **0.894** | corpus reachability |

| split (strict) | value |
|---|---|
| corpus-absent (no relevant chunk in ANY retriever's top-20) | **0.106** |
| ranking-fixable (in corpus, gecko misses, another retriever finds) | **0.260** |

## Verdict — the corpus is largely NOT the ceiling
- **The corpus contains a relevant chunk for ~89% (strict) / ~96% (lenient) of kenya queries** —
  only **~10.6% strict / 3.5% lenient are truly absent**. The earlier P-1 top-3 read (42%
  "not covered") was mostly *buried below rank 3*, now recovered at top-20. Corpus expansion is a
  **minor** lever.
- **EmbeddingGemma is the best deployable retriever** (0.734 strict, +9.9 pp over gecko's 0.635),
  approaching the octen-8B/voyage ceilings — and it recovers a large share of the **26 pp of
  in-corpus answers gecko misses** (ranking-fixable).
- bm25 / lateon / medcpt are weak alone on kenya (lexical-only, ColBERT, and PubMed-domain
  mismatch respectively) — they add little to the union beyond the dense retrievers.

## Implication
The kenya end-to-end failure (no-conversion, RAG net-negative — Phase 3) is **not** a
corpus-coverage problem (corpus reaches ~89%) and **not** a retrieval-ranking problem the embedder
can't help (EmbeddingGemma closes ~10 pp of gecko's 26 pp miss). The binding ceiling is the
**generator**: gemma4-e4b does not convert retrieved context into better answers. The lever is
generator RAG-grounding, with EmbeddingGemma adopted as the better retriever.
