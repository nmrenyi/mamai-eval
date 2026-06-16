# P-1 kenya corpus-coverage diagnostic — verdict

*2026-06-15. Retrieval-ceiling gate before the R2c embedder bake-off. Judge:
Qwen3-32B + the V2 rubric (same as the 230k audit grades). Arms: the full 9-arm
matrix (gecko/bm25/hybrid × none/MiniLM-ft/mxbai-ft), top-3 each, 312 kenya
queries, per-query UNION across all arms. Data:
`kenya_coverage.json`.*

## Question
Any retriever can only reorder chunks the corpus contains. Is the kenya end-to-end
flatness (reranking doesn't help answers) a **ranking** problem (R2c embedder can
help) or a **corpus-coverage** problem (no embedder helps — pivot to content)?

## Result — per-query union coverage (does ANY arm surface a relevant chunk in top-3)

| | lenient (grade≥3) | strict (grade≥5) |
|---|---|---|
| Union covered (some arm) | 217/312 = **69.6%** | 181/312 = **58.0%** |
| Gecko covered (deployed)  | 161/312 = 51.6% | 112/312 = 35.9% |
| (a) ranking-fixable — union found, gecko missed | 56 = **18.0%** | 69 = **22.1%** |
| (b) corpus/depth-limited — no arm found | 95 = **30.5%** | 131 = **42.0%** |

## Kill-rule (pre-registered)
> If ≥60% of kenya queries are NOT strict-covered by ANY arm → corpus is the binding
> ceiling, freeze R2c.

Not-strict-covered = **42.0%** < 60% → **kill-rule NOT triggered. R2c stays justified.**

## Verdict — proceed with R2c, with bounded expectations
- **Ranking headroom is real but modest.** ~18% (lenient) / ~22% (strict) of kenya
  queries have a relevant chunk that *some* retriever surfaces in top-3 but gecko
  misses — the slice a better on-device embedder could plausibly capture. This is
  R2c's addressable upside: roughly **one in five** kenya queries, not the majority.
- **A genuine ceiling remains.** 30–42% of queries have nothing relevant in any arm's
  top-3. Corpus expansion (R-corpus) stays a parallel lever for that fraction.
- **Consistent with the end-to-end nulls.** Even gecko+mxbai-ft (the best retrieval
  arm, relevance P@3 0.279) didn't move SAQ answers — because the realised relevance
  gain over gecko is ~1pp; the bucket-(a) headroom is real but small, and the
  generator can't ground answers on chunks that aren't retrieved.

## Caveat (sizes R2c's true ceiling)
Arms are **top-3 only**, so bucket (b) conflates "not in the corpus" with "in the
corpus but ranked below 3." So 42% is an **upper bound** on the true corpus gap, and
whatever is merely buried is *additional* ranking-fixable headroom that would
**strengthen** the R2c case. **Recommended refinement:** re-run coverage at top-20
(`precompute_arms_matrix.py --top-k 20`) to split corpus-gap from rank-depth before
committing R2c scope. A reference-ceiling retriever (voyage/Qwen3-Embedding-8B) is
not available for kenya (audit rankings exist only for mamaretrieval queries).

## Bottom line
Proceed with the R2c embedder bake-off — the corpus is not the dominant ceiling — but
scope it to the ~1/5 ranking-fixable slice, validate on kenya/healthbench (not
mamaretrieval), and keep corpus expansion on the roadmap for the ~30–42% the corpus
genuinely doesn't cover. Sharpen the split with a top-20 coverage re-run first.
