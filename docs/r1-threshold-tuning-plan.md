# R1 — Retrieval relevance-threshold tuning plan

*Filed 2026-06-12. Implements the P0 item "R1 — Relevance threshold tuning" from
`configs/config-v0.2.0/reports/improvement-plan-20260611.html`. Work branch:
`feat/r1-retrieval-threshold-20260611`.*

---

## Goal

Pick a per-chunk relevance rule (absolute cosine cutoff, or a relative rule) so the
system stops injecting junk chunks and falls back to no-RAG when the retrieved bundle
is poor. Stated preference: **abstention over noise**.

Evidence of active harm (v0.2 Phase B): MCQ −1.8 pp with RAG; SAQ zero-recall rows
shift toward scope-refusal under RAG (50% → 67%), e.g. geographic over-refusal when
Zanzibar-specific chunks are injected into a Kenya vignette.

Deployment is a config change: `params.json` → `retrieval.similarity_threshold`
(currently `0.0`, loaded as `RETRIEVAL_THRESHOLD` in `shared/prompts.py` but not yet
consumed by `retrieval_eval/retrieval.py:retrieve()` — plumbing the filter is part of
this work). Filter **per-chunk**, not all-or-nothing: keep 1–2 good chunks rather than
forcing 3-or-none.

## Strategy — three-tier funnel

Each tier uses the data it is suited for: **big and free tunes, mid-size and realistic
selects, small and precious judges.**

### Tier 1 — Offline sweep → shortlist (free, no GPU)

Two complementary calibration sources:

- **Source A — mamaretrieval labels.** v0.2 audit: 3,185 queries × 6 retrievers,
  230,964 LLM-graded (query, chunk) pairs on the 0–6 rubric (validated 95%/85% vs
  Opus 4.7 at the ≥3/≥5 cuts). Positive class for this sweep: **lenient grade ≥ 3**
  (R1 targets junk suppression; the harm comes from grade-0–2 chunks).
- **Source B — existing end-to-end ±RAG results.** ~23k judged MCQ rows
  (`configs/config-v0.2.0/results/end_to_end_eval/gemma4-e4b/`), joined per-row with
  the cached per-chunk cosine scores from the retrieval precompute (`similarities`
  field; scores are NOT in the committed result JSONs — join against the rag_contexts
  cache on cluster scratch). Signal: do RAG-harmed rows (correct→incorrect) have
  systematically lower bundle scores than RAG-helped rows?

Steps:

1. **Score parity.** Re-embed the 3,185 audit queries with the deployed pipeline
   (this repo's `GeckoEmbedder` + the app's SQLite store) so the swept scores are
   exactly what `Gecko_1024_quant.tflite` produces at runtime. Do not trust audit
   ranking scores produced by any other Gecko variant — cosine scales are
   model-specific.
2. **Fail-fast check.** AUC/PR of raw cosine vs grade-≥3 on Source A. Known risk:
   Gecko's scores are poorly separated (mean rank-1↔rank-20 gap 0.09; rank-1 p5–p95
   0.685–0.853 overlaps rank-20's 0.573–0.772). If cosine has essentially no
   discriminating power, stop and file a negative result (see Failure mode).
3. **Sweep rule families**, each 1–2 parameters, on both sources:
   - global absolute threshold τ (baseline to beat);
   - relative margin: keep chunks within δ of the query's top-1 score;
   - gated hybrid: abstain entirely if top-1 < floor, else keep within margin
     (prior favourite — separates "whole bundle is junk" from "tail chunks are junk");
   - per-query normalization (z-score over the top-k score curve).
   Outputs per rule: injected-chunk precision vs abstention rate vs lost-hits
   frontier, per dataset.
4. **Shortlist 3–5 discrete candidate operating points** for Tier 2.

Train/test hygiene: split the MCQ rows in half — tune on one half, reserve the other
for acceptance. Open-ended rows are never used in Tier 1.

Distribution-shift caveat: audit queries were generated from corpus chunks, so every
audit query has an in-corpus answer — the audit under-represents the all-junk-bundle
case R1 exists for. MCQ (Source B) covers that shifted distribution. Also compare
Gecko's top-1/top-3 cosine *distributions* across MCQ vs SAQ vs HealthBench queries
(retrieval-side only, no outcomes — no leakage): if SAQ skews low, an absolute cutoff
will over-abstain there, which argues for the scale-invariant relative rules.

### Tier 2 — HealthBench elects (cluster, affordable)

Run each shortlisted candidate end-to-end on the rubric track (healthbench_oss_eval +
consensus + hard, 2,339 rows; gpt-oss-120b judge on H200 per the pinned
`params.json` judge config). Closest affordable proxy to deployment: open-ended,
conversational, OBGYN-filtered, and exercises the app-faithful latest-turn retrieval
path (R3.4).

Cost control — per candidate, only rows whose filtered bundle **shrank** need fresh
generation + judging:

- bundle unchanged (all chunks pass) → reuse the existing +RAG generations/judgments;
- bundle empty (abstain) → reuse the no-RAG arm generations/judgments;
- bundle shrunk (1–2 chunks survive) → regenerate + rejudge (the only new compute).

Prerequisite: a no-RAG rubric-track baseline run must exist (generate once if the
20260521 run was RAG-only; reusable across all candidates).

Pick the winner on weighted_met delta vs both baselines (no-RAG and unfiltered-RAG).
**Report this number as a selection metric** — it is mildly optimistic by
construction and is not acceptance evidence.

### Tier 3 — SAQ ratifies (one run, untouched until now)

One ±RAG A/B at the winning operating point on the open-ended sets (kenya 312,
afrimedqa_saq 37, whb 20). These 369 rows never participate in tuning or selection —
they are the unbiased verdict on the most deployment-realistic data we have.

Acceptance criteria (fixed in advance):

- key-fact recall does not regress;
- refusal rate does not rise;
- on the held-out MCQ half: the −1.8 pp ±RAG gap collapses to ≈ 0;
- safety score distribution unchanged (no new 1s).

Confound to manage: part of the refusal behaviour is prompt-induced (G1). Run the SAQ
A/B jointly with / stratified against the prompt change (2×2 if budget allows) so the
threshold is not blamed or credited for prompt artifacts.

## Operating-point philosophy

Costs are asymmetric today: no-RAG currently *beats* RAG, so a lost hit is cheap and
an injected junk chunk is expensive. Sit aggressively on the high-precision side
(injected-chunk precision ≥ ~0.7–0.8 lenient) and accept the implied abstention rate.
Report per-dataset abstention so a degenerate outcome (e.g. SAQ abstains ~95%, making
"RAG no longer hurts" trivially true) is visible. The knowledge-uplift probe (M1)
needs some injections to measure.

## Failure mode is informative

If Tier 1 shows cosine cannot separate junk from good chunks on either source, the
correct output of this branch is a **documented negative result** recommending effort
move to R2 (better retriever), not a placebo threshold. The sweep + funnel
infrastructure is reusable for R2 candidates either way.

## Relationship to R2 (retriever upgrade)

The chosen threshold *value* is embedder-specific and dies with Gecko; everything
else here (sweep tables, funnel, reuse-aware Tier 2 runner, acceptance gates) is the
standing recalibration pipeline the improvement plan already requires ("re-calibrate
whenever the embedder changes"). Per the plan's sequence, M1 validates R1 first, then
recurs after every R2/C1 change.
