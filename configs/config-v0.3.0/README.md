# config-v0.3.0 — final decision config (English-only)

**Status: the decided production configuration for the MAM-AI assistant.**

This config version pins the **G1 system prompt** as the deployed English system
prompt (`system_en.txt`). It is the project's final, evidence-backed decision,
superseding the v0.2.0 default (baseline prompt) for deployment.

## Deployment target

A config version pins the **prompt + generation/retrieval params**; it does **not**
pin the generator model or embedder (those are device/runtime choices). The
decided deployment stack to use *with* this config is:

| Component | Choice |
|---|---|
| System prompt (English) | **G1** (this `system_en.txt`) |
| Generator | **Gemma 4 E4B (Q4_0)** |
| Retriever / embedder | **EmbeddingGemma** (top-3) |
| Generation params | carried forward from v0.2.0 (`params.json`) |

The generator and embedder choices must be reflected in the Android `app_config`
(separate device repo); creating this config does not by itself change the device.

## Why this config (evidence)

From the generator × prompt matrix and the manual dangerous-case adjudication
(both under `../config-v0.2.0/`):

- **G4 + G1 is a net safety improvement over the previously deployed 3n-baseline**
  (dangerous ~0 vs 4, harm 16% vs 26%) at comparable key-fact recall (0.28 vs 0.30).
- **G1 fixes G4's deflection** (33% → 3%) — the problem this branch set out to solve.
- **3n + prompts failed the safety gate** — hand-adjudicated genuine dosing/drug
  errors (doses off 4–1000×, contraindicated drugs); see
  `../config-v0.2.0/results/end_to_end_eval/gen-prompt-matrix-20260622/dangerous-case-adjudication-20260623.md`.
- **G4 + G1's single dangerous flag is a judge over-flag** → effectively zero
  genuine-dangerous.
- Reports: `../config-v0.2.0/reports/generator-prompt-matrix-20260622.html` and
  `faithfulness-g4-vs-3n-prompts-20260622.html`.

**Known limitation:** G4 + G1's open-ended completeness is modest (HealthBench
weighted_met ~0.04); it is the *safest deployable* option, not a *solved* one. The
future lever for capability-with-safety is generator RAG-grounding (G-RAG) on a
stronger on-device model.

## English-only — Swahili deliberately removed

This config ships **no `system_sw.txt`**. The G1 prompt was evaluated in **English
only**; shipping an un-validated Swahili translation of a safety-critical prompt
would be unsafe and is explicitly out of scope. Swahili support requires its own
evaluation before it can be added to a future config version. The harness treats
`system_sw.txt` as optional (`shared/prompts.py`), and `params.json` carries no
Swahili context labels.

## Eval data

The corpus/oracle and dataset (`nmrenyi/mamabench` @ v0.2) are unchanged from
v0.2.0; this config inherits them for any future evaluation. The `results/`
subdirs are scaffolded empty — v0.3.0's deployment decision was justified by the
v0.2.0 matrix, not a re-run.
