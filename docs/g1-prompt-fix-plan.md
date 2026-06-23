# G1/G2 Prompt-Improvement Plan + A/B Eval

*Filed 2026-06-19 · branch `feat/g1-prompt-fix-20260611` · config-v0.2.0 prompts · generator **Gemma 3n E4B** · no-RAG*

**Status (2026-06-19):** all three arm prompts drafted and reviewed (see *The three
arms*). Next action is Phase A.1 — the `--system-prompt` override in `run_eval.py`.
Experiment dir: `configs/config-v0.2.0/results/end_to_end_eval/g1-ab-3n-20260619/`.

Implements the prompt-side work items (**G1**, + the **G2 lever-1** pilot) from
`configs/config-v0.2.0/reports/improvement-plan-20260611.html`. Builds on the
instrumentation already committed on this branch (the 4-value `behavior` judge
in `end_to_end_eval/rescore_open_v2.py` and the Arm 1 baseline).

---

## Diagnosis (why this branch exists)

The system is **safe but unhelpful, and the failure is behavioral, not knowledge**:

- kenya SAQ: 26.7% of responses convey **zero key facts**; ~91% of those are
  behavioral (≈50% scope-refusals, ≈41% defer-to-doctor with no first-line
  content), only ~8% genuine knowledge misses.
- HealthBench rubric: 61% of failing criteria are **deflection**; commission ≈ 2%.
- Deployment-blocking: the scope reflex mis-refuses **emergencies** (choking
  child, anaphylaxis, scald) as "not a healthcare question".

The goal is to move **`defer_only` mass → `manage-then-refer`**, *not* remove
caution (the current caution is what keeps commission at ~2%).

Sharpening finding from the committed Arm 1 baseline: of 227 *engaging* kenya
rows, **218 already `engage_and_refer`** — the model already refers when it
engages. So G1's real job is **producing first-line content at all** on the ~85
content-free rows, not adding referral to engaging answers.

---

## Decisions (locked)

| Decision | Choice |
|---|---|
| Generator | **Gemma 3n E4B** (the deployed model; `gemma3n-e4b` in `shared/inference.py`). Re-baseline Arm 1 on 3n; keep the committed Gemma 4 Arm 1 as a cross-check reference only. |
| RAG | **None** — isolates the prompt from retrieval-triggered refusals (retriever upgrade is orthogonal; see R1/R2). |
| Language | **English only** (`system_en.txt`). Port the winning levers to `system_sw.txt` as a separate follow-up. |
| Headline eval sets | **kenya** (SAQ, n=312) + **healthbench_oss_eval** (rubric, n=1209). |
| Secondary sets | afrimedqa_saq (37), whb (20) — SAQ support; **whb out of headline** (under-specced references). |
| Judge | gpt-oss-120b @ medium (pinned v0.2), 4-value `behavior` tag for SAQ; rubric judge for HealthBench. |
| Experiment dir | **fresh** `g1-ab-3n-20260619/` (not the committed Gemma 4 `g1-ab-20260611/`) — keeps the 3n and G4 numbers from being conflated. |
| Prompt delivery | `--system-prompt <path>` override on `run_eval.py` (no mutation of the config `system_en.txt`); arm prompts live in `g1-ab-3n-20260619/prompts/`. |

---

## The three arms (system-prompt only varies)

| Arm | Prompt | Isolates |
|---|---|---|
| **1 — baseline** | current `system_en.txt` | reproduces known numbers |
| **2 — +G1 levers** | + 7 deflection-fix levers (below) | Arm 2 − Arm 1 = pure deflection-fix effect |
| **3 — +G2 skeleton** | Arm 2 + consultation-workflow structure | Arm 3 − Arm 2 = pure structure effect |

**Arm 2 — G1 levers** (priority order, from the plan §2):
1. **Emergency override** (safety-critical, first) — red-flag presentations always
   get minimal safety guidance, never a refusal.
2. **"Either/or" → "both/and"** — first-line management, *then* advise escalation.
3. **Authorize first-line drugs/doses** with a "confirm against local protocol"
   caveat (relax the current "no doses unless retrieved context states them" — moot under no-RAG).
4. **No geographic over-refusal** — the vignette's country must not trigger a scope refusal.
5. **Widen in-scope** (family planning, GBV); when truly out of scope, give a brief
   safe pointer, not a flat refusal.
6. **Address the midwife, not the patient** — clinician-to-clinician; Zanzibar
   nurse-midwives prescribe and decide independently.
7. **Contextualize to local resources** — location informs applicability, never causes refusal.

**Arm 3 — G2 lever 1** (added to Arm 2): consistent answer shape on the standard
consultation workflow (history → exam → tests → diagnosis → plan); definitional
grounding leads the answer; when treatment depends on a finding, give the
**conditional management inline** ("if mild: …; if moderate/severe: …").
History/exam steps **state what the nurse should assess —`run_eval` is single-turn,
so the prompt never asks the user a question back**. (Rationale: a clarifying
question returns zero key facts and the behavior judge would score it `defer_only` —
manufacturing the exact deflection G1 removes. Cost: Arm 3 − Arm 2 is "structure +
room to express it", not a 100%-pure structure delta, because the FORMAT word cap is
also relaxed for structured answers.)

**Prompt files** (drafted, reviewed): `g1-ab-3n-20260619/prompts/arm2_system_en.txt`,
`arm3_system_en.txt`. Arm 1 uses the unchanged `config-v0.2.0/system_en.txt`.

---

## Workflow

Prompts are done; everything below is mechanical. Phases run in dependency order.

**Phase A — Wiring & de-risk** (local, cheap):
1. Add `--system-prompt <path>` override to `run_eval.py` so each arm uses its own
   prompt without mutating the config. *(Blocks all generation — the only real code.)*
2. Confirm the **3n generator is on the cluster PVC** (`gemma3n-e4b` →
   `gemma-3n/gemma-3n-E4B-it-Q4_0.gguf`) — verify the file, don't trust the registry.
3. **Smoke test** ~10 kenya rows × 3 arms: confirm the override loads the right prompt
   and that deflection visibly shifts Arm 1 → 2 → 3, before spending cluster budget.

**Phase B — Generate** (cluster, 3n, no-RAG; one job per arm, ≈1,578 rows/arm):
- SAQ: kenya 312 + afrimedqa_saq 37 + whb 20.
- Rubric: healthbench_oss_eval 1209 (multi-turn) — dominates cost; H200.
- Includes the **Arm 1 3n re-baseline**. Human-in-loop: RunAI submission (NODE_POOL
  H200, check quota first). Output → `g1-ab-3n-20260619/`.

**Phase C — Judge:**
- SAQ → `rescore_open_v2.py` (4-value `behavior` + key-fact recall + safety).
- Rubric → `rescore_rubric.py` (weighted_met / completeness(+) / penalty(−)).
- **Faithfulness spot-check on Arms 2–3** — Lynx pass on a sample (gate's third side).

**Phase D — Read & write up:** aggregate against the acceptance gate, apply the
decision rules, write results into `g1-ab-3n-20260619/README.md`. Pre-commit to a
**repeat judge pass on the safety/harm axis** (the dose lever on 3n is the one likely
to move it; single-run deltas <~10 pp are possible judge noise).

---

## Acceptance gate (three-sided)

A prompt arm ships only if all three hold vs the 3n Arm 1 baseline:

1. **Helpfulness up.**
   - kenya: `defer_only`+`refuse` mass collapses; engaging rows shift to
     `engage_and_refer`; mean key-fact recall up.
   - healthbench_oss_eval: penalty-incurred rate collapses (penalties are
     omission/deflection-phrased); the three emergency cases pass (row IDs in the
     phase-b rubric report); post-G2, positive-credit / completeness climbs.
2. **Safety not worse** — commission stays ~2%, zero "dangerous", no new
   `potentially_harmful` mass. *Emergency-override (lever 1) is the lever most
   likely to push harm up — watch this axis hardest.*
3. **Faithfulness not materially worse** — Lynx spot-check on a sample of Arms 2–3
   (engaging more with concrete drugs/doses mechanically raises hallucination exposure).

**Decision rules:** Arm 2 ≫ 1 confirms G1. Arm 3 ≫ 2 (no safety/faithfulness
regression) → ship the skeleton too. Arm 3 ≈ 2 → ship Arm 2; G2 needs more than
prompting. Treat single-run deltas **< ~10 pp as possible judge noise** — consider
a repeat judge pass / bootstrap CI on the safety + harm axes before reading them.

---

## Notes / open items

- The committed Arm 1 (`results/.../g1-ab-20260611/arm1-baseline/`) is **Gemma 4**;
  it will be superseded by the 3n re-baseline and kept only as a cross-check.
- The improvement plan was written under Gemma 4; 3n already deflects less and
  ~2× recall by default, so the levers' marginal effect will be smaller than the
  plan's G4-based estimates — read deltas, not absolutes.
- Faithfulness must be re-checked after any deflection fix: today's ~9% / zero-
  dangerous numbers were measured on a model that deflects ~a third of the time.
- Swahili (`system_sw.txt`) port and the joint prompt×retrieval (R1) A/B are
  explicit follow-ups, out of scope here.
