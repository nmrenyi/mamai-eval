# R2 — On-device retriever upgrade plan

*Filed 2026-06-13. Implements the P1 item "R2 — On-device retriever upgrade" from
`configs/config-v0.2.0/reports/improvement-plan-20260611.html`. Successor to R1
([`r1-threshold-tuning-plan.md`](r1-threshold-tuning-plan.md)), which closed as a
negative result: Gecko's cosine scores cannot be thresholded to clean RAG, so the
fix has to come from retrieving better content, not filtering what Gecko returns.*

---

## Goal

Close as much of the Gecko-vs-top-tier retrieval gap as possible under strict
on-device constraints. The deployed Gecko is weak: P@3 = 0.477 (share of the
injected top-3 that is actually relevant) vs the voyage-4-large ceiling of 0.867,
and it structurally misses ~65% of complete/actionable (strict, grade ≥ 5) chunks
at *any* rank in its top-20. Better retrieval is the precondition for RAG adding
value instead of noise — but it is necessary, not sufficient (corpus answerability
and generator robustness also bound the end-to-end gain; see Guardrails).

## Why these four steps, in this order

The first two steps reshuffle what Gecko already retrieved and are **free** — pure
arithmetic on the mamaretrieval audit artifacts (`rankings.parquet` + 230,964
graded pairs) we already used for the R1 gate. They set the bar the expensive
steps must clear before any embedding, judging, or device work is spent.

| Step | What | Cost | Question it answers |
|---|---|---|---|
| **R2a** | Hybrid: Gecko + BM25, reciprocal-rank fusion | **free** (both already in the audit) | How much of the gap closes with no new model on the phone? |
| **R2b** | Oracle top-20 rerank ceiling (Gecko and hybrid pools) | **free** (rerank by judge grades) | Is building any reranker worth it at all? |
| **R2c** | Swap Gecko for a different on-device embedder | embedding + small top-up judging | Does a better embedder beat the hybrid+rerank bar? |
| **R2d** | Fine-tune / distill an embedder on the 230k pairs | expensive; last resort | Only if the best off-the-shelf R2c candidate still falls short |

### R2a — Hybrid retrieval (Gecko + BM25, RRF)

Fuse Gecko's and BM25's audit top-20 rankings with reciprocal-rank fusion (sweep
the RRF constant and the Gecko/BM25 weighting — a 1–2 parameter sweep), take the
fused top-3, score against the judgments. Report P@3 / HR@3 at lenient (≥3) and
strict (≥5) cuts, against Gecko-alone (0.477) and the voyage ceiling (0.867).
Rationale: BM25 is lexically complementary (drug names, doses, exact terms), beat
Gecko on every score-quality statistic in the R1 gate, and is deployable at
near-zero cost (the app already uses SQLite; FTS5 ships BM25 ranking natively).

### R2b — Oracle rerank ceiling

Rerank the top-20 pool (Gecko's, and the R2a hybrid's) *perfectly* — by the judge
grades themselves — and take the top-3. This is the hard upper bound for any
reranker that can only reorder what the retriever surfaced; it does **not** build a
reranker. Known shape to make precise: lenient HR@20 = 0.977 (a perfect reranker
nearly fixes lenient hit rate) but strict pool recall plateaus at 0.345 (no
reranker can fix the strict-content miss — that needs better retrieval/corpus).
Output: go/no-go on whether a small on-device reranker is worth building.

**Threshold-adaptability note.** Do <em>not</em> run the R1 Stage-1 score-quality
gate on the oracle: its ordering score *is* the judge grade, so every gate
statistic is trivially ~1.0 by construction — circular and uninformative, and the
oracle isn't deployable anyway. The non-circular version belongs to the
**real-reranker follow-up** (below): a trained cross-encoder produces a relevance
score at inference time that is directly optimized to predict relevance — unlike
the geometric (cosine), lexical (BM25), or rank-fusion (RRF) scores already gated
and found sub-viable. That reranker score is the best candidate yet to clear the
0.80 viability bar, so the follow-up should run the Stage-1 gate on it: it both
measures captured ceiling and is the first plausible route to reviving the R1
threshold (the oracle is exempt by construction).

### R2c — Embedder bake-off (the actual swap)

Replace Gecko with a different on-device embedder. Candidates must pass hard
constraints **before** evaluation: LiteRT-compatible, quantized, within the
latency/memory budget, and multilingual (real queries arrive in Swahili; the
corpus is slated to grow German-language textbooks). voyage / octen from the audit
are reference ceilings, **not** candidates (API-only / 8B cluster-only).
Per candidate: re-embed the 63,650 corpus chunks + 3,185 queries, score against the
existing 230k labels, plus a small top-up judging round for chunks it surfaces that
were never judged (outside everyone's top-20). Screen each with the R1 Stage-1
score-quality gate (chunk AUC / within-bundle concordance / bundle-any-relevant
AUC) **and** P@3/HR@3 before any device porting. The R1 table builders +
`compare_retriever_gates.py` are the screening harness.

### R2d — Fine-tune / distill

Fine-tune the chosen on-device embedder contrastively on the 230,964 graded pairs,
and/or distill voyage's rankings. Requires a clean query-level train/test split to
preserve benchmark validity. Most expensive; pursue only after R2c picks a base
model and only if it still falls short.

## Guardrails (carried over from R1)

- **Validate end-to-end, not on retrieval metrics alone.** Whatever wins offline
  is gated on the M1 MCQ ±RAG rerun (the −1.8 pp gap should move) and a SAQ ±RAG
  A/B (key-fact recall must not regress, refusal rate must not rise). The SAQ rows
  (n=369) were never consumed by R1 and remain a clean held-out set.
- **Score-dependent logic must be validated on the deployment platform.** The R1
  audit found the same quantized TFLite model scores differently across x86 vs ARM
  (top-3 membership agreed only 84%); the phone is a third platform.
- **Re-audit contradictions and re-version artifacts after any retriever change.**
  A new embedder or corpus version invalidates cached embeddings, thresholds, and
  the contradiction audit; the self-contradiction count scales with pool size.
- **Score quality is a distinct axis from ranking quality.** The R1 gate showed
  lateon ranking well (P@3 0.738) with zero cross-query score signal (AUC 0.501);
  measure both per candidate, not just P@k.

## Status / results so far

- **R2a (done):** hybrid Gecko+BM25 RRF is a modest, free-to-deploy win (P@3
  0.477→0.516, HR@3 0.814→0.860 at α≈0.5) but doesn't close the gap or fix the
  strict miss; thresholding stays sub-viable on the fused score.
  [`reports/r2-hybrid/r2a-hybrid-result-20260613.html`](../configs/config-v0.2.0/reports/r2-hybrid/r2a-hybrid-result-20260613.html)
- **R2b (done):** oracle rerank ceiling is large — reordering the hybrid top-20
  lifts P@3 0.512→0.926 (lenient) and 0.206→0.545 (strict); **strong go** to scope
  a real reranker, depth ~10. Revised R2a's read: reranking *does* help strict
  (chunks present but mis-ranked), though a ~22% strict structural tail remains for
  R2c/C1. [`reports/r2-rerank/r2b-rerank-ceiling-20260613.html`](../configs/config-v0.2.0/reports/r2-rerank/r2b-rerank-ceiling-20260613.html)

## Suggested sequence

1. **Done (offline, no GPU):** R2a hybrid simulation + R2b oracle-rerank ceiling.
2. **Next — real-reranker follow-up (not free):** evaluate a quantized,
   LiteRT-feasible cross-encoder over the hybrid top-10–20 pool against the same
   labels (clean train/test split). Reports both captured ceiling and the Stage-1
   gate on the reranker's score (the threshold-revival test above).
3. **In parallel / if a structural gap remains:** scope R2c — pick 2–3
   constraint-passing candidate embedders, run the bake-off, screen with the
   Stage-1 gate before device work.
4. **Later, only if needed:** R2d fine-tune/distill on the chosen base.

Each retrieval change loops back through the Guardrails before adoption.
