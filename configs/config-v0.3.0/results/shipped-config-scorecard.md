# Shipped-config scorecard — G4 + G1 + EmbeddingGemma

> **Derived, not a native run.** These numbers are **extracted from the
> config-v0.2.0 generator × prompt matrix** (cell `g4_g1`, run under
> `config-v0.2.0` with the G1 prompt injected via `--system-prompt`). config-v0.3.0
> has had **no native eval re-run**; this is a labelled snapshot of how the shipped
> stack scored. Source of truth + full matrix:
> `../../config-v0.2.0/results/end_to_end_eval/gen-prompt-matrix-20260622/`.

**Stack:** generator **Gemma 4 E4B (Q4_0)** · prompt **G1** (`system_en.txt`) ·
retriever **EmbeddingGemma** (top-3) · **with RAG** · English only · 2026-06-22.

## Track A — end-to-end (with RAG)

| Metric | Value |
|---|---|
| kenya key-fact recall (n=312) | **0.279** |
| deflection rate | **3.2%** (fixed from G4-baseline's 32.7%) |
| harm rate (potentially-harmful + dangerous) | 16.0% |
| dangerous (raw judge) | 1 |
| **dangerous (adjudicated)** | **0 genuine** — the single flag is a judge over-flag (standard choking first-aid); see `../../config-v0.2.0/results/end_to_end_eval/gen-prompt-matrix-20260622/dangerous-case-adjudication-20260623.md` |
| HealthBench-oss weighted_met (n=1209) | 0.038 (positive 0.221 / penalty 0.373) |

Safety distribution (kenya): safe 234 · minor_concern 28 · potentially_harmful 49 · dangerous 1.
Behaviour: engage_and_refer 274 · engage_only 28 · defer_only 8 · refuse 2.

## Track B — faithfulness (oracle context, n=2989)

| Metric | Value |
|---|---|
| raw Lynx FAIL | 4.72% |
| **categorized true-hallucination** (metric of record) | **3.31%** |
| calibrated (noisy 50-sample extrapolation) | 23.1% (miss-rate 0.22, 95% CI [0.12–0.34]) |

## Verdict

Safest deployable on-device config: **deflection fixed (33% → 3.2%), ~0 genuine
dangerous, harm 16%** (vs 26% on the prior 3n-baseline production) at **comparable
recall**. Open-ended completeness (HealthBench 0.038) is modest — this is the
*safest available* config, not a *solved* product. The capability-with-safety
upside (G-RAG on a stronger on-device generator) is future work, out of scope for
this phase.
