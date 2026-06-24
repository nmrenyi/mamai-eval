# R2c Phase 3 — end-to-end value gate, healthbench-oss (results)

*2026-06-16. Generator gemma4-e4b (deployed), judge gpt-oss-120b + HealthBench rubric.
n=1209 (full oss_eval test split). EmbeddingGemma vs gecko arms (top-3 contexts), matched
through the same generate+judge pipeline. Raw: `configs/config-v0.2.0/results/retrieval_eval/r2c-embedder/healthbench/`.*

## Rubric scores (n=1209)
| metric | no-RAG | gecko | **EmbeddingGemma** | Δ (EG−gecko) |
|---|---|---|---|---|
| **combined `weighted_met`** | +0.0033 | −0.0039 | **+0.0119** | +0.016 |
| positive — completeness side (got the right guidance in) | 0.1843 | 0.1750 | **0.1891** | +0.014 |
| penalty rate — negative (harmful/incorrect triggered) | 0.3787 | 0.3722 | **0.3683** | −0.004 |

### no-RAG arm (added 2026-06-24)
The no-RAG (k=0) HealthBench arm was **not re-run** — an identical no-RAG generation
already existed: `../../results/end_to_end_eval/gemma4-e4b/20260521T123051-cluster-norag-rubric/healthbench_oss_eval.json`
(gemma4-e4b, config-v0.2.0, temp 1.0 / top_p 0.95 / top_k 64, n=1209, `rag: false`,
scored by the same pinned `gpt-oss-120b` rubric judge — it is a default target in
`cluster/rescore_rubric.sh`). It comes from a parallel app-parity batch, not the r2c
bake-off batch — that batch's own gecko-RAG arm scores −0.0017 vs the r2c −0.0039 shown
here, so the cross-batch comparison is approximate (all arms are ≈0, so the "retrieval
does not convert into answer quality" conclusion holds either way). That file's stored
`mean_weighted_met` is 0.0033 but predates the +/- split, so all three arms were
**recomputed from their per-criterion verdicts with the current `_row_score`**: gecko
and EmbeddingGemma reproduced their published aggregates to the digit
(−0.0039 / 0.1750 / 0.3722 and +0.0119 / 0.1891 / 0.3683), validating the recompute;
the no-RAG split (+0.0033 / 0.1843 / 0.3787) is read off the same pass. no-RAG sits
**between** the two RAG arms on `weighted_met` and slightly **ahead of gecko** —
reinforcing that on this generator, retrieval quality does not convert into answer
quality. (Fills the previously-blank no-RAG → HealthBench cell in the personal-site
Table 2.)

### Per-axis mean (signed)
| axis | gecko | EmbeddingGemma |
|---|---|---|
| completeness | −0.203 | **−0.171** |
| accuracy | 0.144 | **0.158** |
| context_awareness | **0.126** | 0.102 |
| communication_quality | 0.337 | 0.333 |
| instruction_following | **0.193** | 0.166 |

## Verdict — marginal conversion (both ≈ 0)
- EmbeddingGemma is **marginally better** end-to-end: combined `weighted_met` +0.016, positive
  +1.4 pp, penalty −0.4 pp; ahead on completeness + accuracy, behind on context + instruction.
- **But both arms' combined score is essentially zero** (−0.004 / +0.012): the generator's positive
  completeness (~0.18) is almost entirely cancelled by a ~0.37 penalty rate. gemma4-e4b barely
  clears the rubric net *regardless* of which embedder feeds it.
- So even on a **coverage-good, large-n (1209)** set, EmbeddingGemma's decisive **retrieval** win
  (Phase 2: +12–13 pp) produces only a **slim answer-quality gain** — the same generator-grounding
  ceiling seen on kenya SAQ (no conversion). The bottleneck is the generator's ability to use
  retrieved context, not the embedder's ranking.

## Note
The judge job reported `Failed`, but only on its post-scoring copy step — **both arms scored all
1209 rows** with valid aggregates (verified in-place). The scored JSONs are committed here.
