# R2c Phase 3 — end-to-end value gate, healthbench-oss (results)

*2026-06-16. Generator gemma4-e4b (deployed), judge gpt-oss-120b + HealthBench rubric.
n=1209 (full oss_eval test split). EmbeddingGemma vs gecko arms (top-3 contexts), matched
through the same generate+judge pipeline. Raw: `configs/config-v0.2.0/results/retrieval_eval/r2c-embedder/healthbench/`.*

## Rubric scores (n=1209)
| metric | gecko | **EmbeddingGemma** | Δ |
|---|---|---|---|
| **combined `weighted_met`** | −0.0039 | **+0.0119** | +0.016 |
| positive — completeness side (got the right guidance in) | 0.1750 | **0.1891** | +0.014 |
| penalty rate — negative (harmful/incorrect triggered) | 0.3722 | **0.3683** | −0.004 |

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
