# G1 A/B — system-prompt deflection fix (no-RAG SAQ)

**Status:** in progress — Arm 1 (baseline) judged; Arms 2 & 3 pending.
**Config:** config-v0.2.0 · branch `feat/g1-prompt-fix-20260611`

## Design

Three-arm A/B isolating the effect of the G1 system-prompt revision (improvement
plan §2). Only the system prompt varies; everything else is held constant.

| Arm | Prompt | Isolates |
|---|---|---|
| 1 — baseline | current `system_en.txt` | reproduces the Phase B baseline |
| 2 — +G1 levers | + emergency override, both/and, drugs, geographic, scope, midwife-not-patient, local context | Arm 2 − Arm 1 = pure deflection-fix effect |
| 3 — +G2 skeleton | Arm 2 + consultation-workflow structure | Arm 3 − Arm 2 = pure structure effect |

- **No RAG** — this isolates the generator prompt from retrieval noise.
- **Datasets (full):** kenya 312 (decision set) + afrimedqa_saq 37 + whb 20 = 369/arm.
  whb references are likely under-specced (plan §5) — read as indicative only.
- **Judge:** gpt-oss-120b @ medium, with the new `behavior` (deflection) tag,
  identical across all arms.
- **Two tests:** (1) SAQ — recall + deflection + safety (decision); (2) faithfulness
  spot-check on Arms 2–3 only (guardrail; Arm 1 faithfulness ~9% reused).
- **Generation:** Arm 1 reuses the Phase B no-RAG generations (re-judged only);
  Arms 2 & 3 generated fresh.

## Arm 1 — baseline (re-judged 2026-06-12)

Re-judge of `gemma4-e4b/20260520T104611-cluster-norag-openended` (the responses
the published kenya recall 0.178 was measured on), with the behavior-tagged judge.

| Set | n | recall | deflection_rate | engage / defer_only / refuse | harm_rate | safety (safe/minor/poten.) |
|---|--:|--:|--:|:--|--:|:--|
| **kenya** (decision) | 312 | **0.1772** | **0.372** | 196 / 110 / 6 | 0.112 | 255 / 22 / 35 |
| afrimedqa_saq | 37 | 0.1741 | 0.351 | 24 / 7 / 6 | 0.027 | 34 / 2 / 1 |
| whb | 20 | 0.0312 | 0.700 | 6 / 11 / 3 | 0.100 | 17 / 1 / 2 |

### Two findings to carry forward

1. **Recall reproduces** — kenya 0.1772 vs published 0.1775 → the new judge prompt
   did not disturb the recall metric (sanity check passed).
2. **Deflection is dominated by `defer_only`, not refusal** — kenya 37.2% deflection
   is 110 defer-only + 6 refuse. The old refusal regex read **0.0** because it only
   catches blunt "I can't help" refusals and misses the polite "consult a doctor"
   answer — which is the dominant failure mode and the main target for G1.
3. **Safety calibration shifted under the new judge prompt** — kenya harm 11.2% vs
   Phase B 20.8% (35 vs 65 `potentially_harmful`). Adding the behavior instruction
   nudged the judge's safety calls more lenient. **Consequence:** the A/B safety
   baseline is this re-judged Arm 1 (11.2%), *not* the published 20.8%.
   Within-A/B comparisons remain valid because all arms use this same judge.

## Provenance

- Re-judge job: `mamai-g1-arm1-rejudge` (RunAI light-yiren, H200/gpu307).
- PVC output: `/lightscratch/users/yiren/phase_b_saq/20260612T072157Z/arm1-baseline/`.
- Inputs: this dir's `arm1-baseline/` (Phase B no-RAG gens, judge fields stripped).

## Arm 2 — +G1 levers

_Pending generation + judging._

## Arm 3 — +G2 skeleton

_Pending generation + judging._
