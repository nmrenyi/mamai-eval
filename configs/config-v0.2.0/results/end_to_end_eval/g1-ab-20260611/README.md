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
- **Judge:** gpt-oss-120b @ medium, with the 4-value `behavior` tag
  (`engage_and_refer` / `engage_only` / `defer_only` / `refuse`), identical
  across all arms. `deflection_rate` = `defer_only` + `refuse` (content-free rows).
- **Two tests:** (1) SAQ — recall + deflection + safety (decision); (2) faithfulness
  spot-check on Arms 2–3 only (guardrail; Arm 1 faithfulness ~9% reused).
- **Generation:** Arm 1 reuses the Phase B no-RAG generations (re-judged only);
  Arms 2 & 3 generated fresh.

## Arm 1 — baseline (re-judged 2026-06-12, 4-value scheme)

Re-judge of `gemma4-e4b/20260520T104611-cluster-norag-openended` (the responses
the published kenya recall 0.178 was measured on), with the 4-value behavior judge.

| Set | n | recall | deflection | engage_and_refer / engage_only / defer_only / refuse | harm | safety (safe/minor/poten.) |
|---|--:|--:|--:|:--|--:|:--|
| **kenya** (decision) | 312 | **0.184** | **0.272** | 218 / 9 / 83 / 2 | 0.087 | 273 / 12 / 27 |
| afrimedqa_saq | 37 | 0.178 | 0.297 | 15 / 11 / 6 / 5 | 0.027 | 35 / 1 / 1 |
| whb | 20 | 0.044 | 0.650 | 7 / 0 / 11 / 2 | 0.100 | 18 / 0 / 2 |

### Findings to carry forward

1. **Recall reproduces** — kenya 0.184 ≈ published 0.178 (judge run-to-run noise
   ~±0.006) → the new judge prompt did not disturb the recall metric.
2. **When Gemma engages, it already refers** — of the 227 engaging kenya rows,
   218 are `engage_and_refer` and only 9 `engage_only`. So "manage-then-refer"
   (the G1 ideal) is already the dominant engaging mode; the model is uniformly
   cautious. G1's job is therefore **converting the 85 content-free rows
   (`defer_only` 83 + `refuse` 2) into content-bearing answers**, not adding
   referral to engage-only answers. The both/and lever matters less than the
   "produce first-line content at all" levers (drugs, scope, geographic).
3. **The deflection metric is sensitive to tag wording + run** — the earlier
   3-value run read 37.2%; the 4-value re-run reads 27.2%. Most of that is the
   reframing (the two-question "any first-line content?" test counts borderline
   thin-content as `engage_*` rather than `defer_only`), plus judge run-to-run
   noise (recall 0.177→0.184, harm 0.112→0.087 also drifted). **Consequences:**
   (a) the 4-value scheme is now locked for all arms, so relative A/B deltas stay
   valid; (b) the A/B baseline is deflection 27.2% / harm 8.7% (this run), not the
   published or 3-value numbers; (c) single-run deltas under ~10 pp may be noise —
   consider repeat judge passes or bootstrap CIs before reading them as real
   (ties to the plan's §5 rubric-CI item).

## Provenance

- Re-judge jobs: `mamai-g1-arm1-rejudge` (3-value, superseded),
  `mamai-g1-arm1-rejudge-v2` (4-value, current). RunAI light-yiren, H200.
- PVC output (current): `/lightscratch/users/yiren/phase_b_saq/20260612T074408Z/arm1-baseline/`.
- Inputs: this dir's `arm1-baseline/` (Phase B no-RAG gens, judge fields stripped).

## Arm 2 — +G1 levers

_Pending generation + judging._

## Arm 3 — +G2 skeleton

_Pending generation + judging._
