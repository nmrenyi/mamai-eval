# Judge validation (meta-eval) plan

*Drafted 2026-05-21, revised same day. Design notes for choosing and validating
the LLM judge used to score the open-ended evaluation tracks.*

## Why this exists

The `open_ended` and `open_ended_rubric` tracks are scored by an LLM judge:

- **`open_ended`** (kenya / whb / afrimedqa_saq) — `end_to_end_eval/rescore_open_v2.py`,
  headline metric `key_fact_recall` (per-key-fact present/partial/absent).
- **`open_ended_rubric`** (healthbench_oss_eval / consensus / hard) —
  `end_to_end_eval/rescore_rubric.py`, single judge per criterion, HealthBench
  weighted-met formula.

Every headline number on those tracks rests on the judge's verdicts. An
unvalidated judge could silently corrupt the whole v0.2 report. So the judge
must be **measured against human ground truth before it is trusted** — and
before the judge model is pinned in `configs/config-v0.2.0/params.json`
(currently `TBD-pin-before-release`).

## The calibration data: `mamabench@v0.2.1`

mamabench **v0.2.1** (HF tag, released 2026-05-21) ships the judge-calibration
set as a side-file. It is an additive patch over v0.2 — all benchmark rows /
schema / configs are byte-identical, `benchmark_version` stays `"v0.2"`.

- File: `calibration/obgyn_meta_eval.jsonl`
- **Not a loadable config** — fetch it directly, not via `load_dataset`:
  `https://huggingface.co/datasets/nmrenyi/mamabench/resolve/v0.2.1/calibration/obgyn_meta_eval.jsonl`
- An OBGYN-scoped slice of HealthBench's grader meta-evaluation set, filtered
  to mamabench scope by reusing the existing OBGYN classifier verdicts (no new
  classification run). Its prompt set is exactly the `consensus` prompt set.

### What a row is

Each row is a `(conversation, completion, single rubric criterion)` triple
carrying **independent physician labels** — judge-calibration data, not
benchmark questions:

```jsonc
{
  "prompt":        [{role, content}],          // the conversation
  "completion":    "...",                      // a model answer being graded
  "rubric":        "Judge whether the completion ... should: ...",  // ONE criterion
  "binary_labels": [false, true],              // each physician's met/not-met verdict
  "anonymized_physician_ids": [...],
  "category": "cluster:emergency_referrals_...",
  "mamabench_obgyn_category": "CHILD_HEALTH"    // the one field mamabench added
}
```

### Dataset stats (verified against the live v0.2.1 file)

| | |
|---|---:|
| Rows (triples) | 6,853 |
| Unique prompts | 872 (= the `consensus` prompt set) |
| Physician labels total | 14,162 (~2.07 physicians/row) |
| Physicians per row | 2 → 6,409 rows · 3 → 432 · 4 → 12 |
| Physician met-rate | 77.0% |

Per `mamabench_obgyn_category`: CHILD_HEALTH 3,239 · MATERNAL 2,284 ·
SRH 942 · NEONATAL 388.

## Core idea: concordant rows are the clean test bed

Split the set on **physician concordance** — whether all physicians who
labelled a row gave the same verdict:

| Subset | Rows | Share |
|---|---:|---:|
| Concordant (all physicians agree) | 5,241 | 76.5% |
| Discordant (physicians disagree) | 1,612 | 23.5% |

On a **concordant** row the physician verdict is unambiguous ground truth: the
criterion was clear enough that independent physicians agreed. If the judge
gets a concordant row wrong, that is unambiguously the judge's fault.

On a **discordant** row there is no clean ground truth, so it cannot fairly
pass/fail the judge. Discordant rows are excluded from the headline but
reported separately.

> "Consensus" here just means physicians agreed on a calibration row. It is
> unrelated to the `healthbench_consensus` benchmark config.

Concordant subset by category: CHILD_HEALTH 2,463 · MATERNAL 1,777 ·
SRH 710 · NEONATAL 291 — enough rows to slice every category.

## The human baseline reframes what "good" means

Two physicians independently grading the *same* `(completion, criterion)` pair
agree only **77.6%** of the time:

| Human ↔ human baseline | |
|---|---:|
| Pairwise comparisons | 7,777 |
| Raw agreement | **77.6%** |
| Cohen's κ | **0.366** ("fair") |

This is the ceiling. The judge does **not** need to hit 95%. If the
LLM↔human agreement lands near ~78%, the judge is performing at human
inter-rater level — the honest bar. A judge scoring 95% against single humans
would be *suspicious*: over-agreeing with rater noise, or rubber-stamping.

## Metric set

### 1. Agreement — three frames side by side

| Comparison | Rows used | What it tells us |
|---|---|---|
| Human ↔ human | all rows w/ ≥2 physicians | the ceiling — 77.6%, κ 0.37 |
| **LLM ↔ single human** | all 6,853 | judge vs *each individual* physician label, averaged over all (judge, physician) pairs. Apples-to-apples with the human baseline. |
| **LLM ↔ physician consensus** | 5,241 concordant | judge vs the unanimous verdict. The clean pass/fail test. |

### 2. Prediction label distribution — the rubber-stamp detector

The concordant subset is **85% met / 15% not-met**. A judge that rubber-stamps
"met" still scores ~85% raw agreement there. So also report:

- **Judge met-rate vs physician met-rate (77%).** A judge predicting e.g. 92%
  met is rubber-stamping even if headline agreement looks fine.
- **Agreement split by physician label** — agreement on the *met* rows vs the
  *not-met* rows separately. A rubber-stamp judge scores ~100% on met rows and
  ~0% on not-met rows; that asymmetry is invisible in the headline but obvious
  here. (Sensitivity / specificity against the physician label.)

### 3. Per-category breakdown

All of the above, sliced by `mamabench_obgyn_category`.

## Pass criteria

A judge is fit to trust if **all** hold:

1. LLM↔human agreement ≈ human↔human agreement (~78%) — at human inter-rater
   level, not far below and not suspiciously above.
2. Judge met-rate ≈ physician met-rate (~77%) — not lopsided toward "met".
3. Agreement holds on the not-met rows, not only the met rows.

## Candidate judges — open-weight, served on the cluster

Decision: use **open-weight judges served on the EPFL cluster** (vLLM,
OpenAI-compatible endpoint), not a paid API. This sidesteps API-budget
approval; judging cost becomes GPU time already available (8× H100, or
8× H200 when free).

Two families are excluded up front:

- **Gemma** — the model under test is Gemma 4 E4B; a Gemma judge introduces
  self-preference bias (judges favour their own family).
- **Qwen** — already the OBGYN classifier family; kept off the judge side to
  keep the pipeline decorrelated.

Three candidates for this round:

| Model | Params (active) | Reasoning | Run as | Hardware |
|---|---|---|---|---|
| **gpt-oss-120b** | 117B MoE (5.1B) | yes (low/med/high) | native MXFP4 ~63 GB | 1× H100 |
| **Llama-3.1-Nemotron-Ultra-253B-v1** | 253B dense | yes (toggle on/off) | FP8 ~253 GB | 4× H100 |
| **Llama 4 Maverick** | 400B MoE (17B) | no | FP8 ~400 GB | 8× H100 |

Rationale: three distinct lineages (OpenAI / Llama-3.1-via-NVIDIA / Llama-4),
spanning the reasoning axis (2 reasoning, 1 not); none conflicts with Gemma or
Qwen. gpt-oss-120b is GPT-4.1-class and the cheapest to run (1 GPU, 8 replicas
possible); Maverick is externally validated as a strong judge (Judge
Reliability Harness ~0.906); Nemotron-Ultra is the Llama-lineage reasoning
option (the base Llama models are not reasoning models).

## The bake-off and decision rule

Run all three candidates over the calibration set, compare on the metric set
above, and **pick one uniform judge**.

**Decision rule:** pick the model whose concordant agreement is closest to the
77.6% human baseline. **If two are within ~1–2 pp of each other, take the
lighter one** (fewer GPUs) — a statistical tie does not justify Maverick's 8×
footprint over gpt-oss-120b's single GPU, since the chosen judge's footprint
recurs on every future rescore.

Pin the winner in `params.json` → `judge.rubric` (and `judge.ensemble` is
retired — see below).

## One uniform judge for both tracks

The bake-off winner is used **single-judge** for **both** the rubric track and
the open-ended SAQ track. Reasoning:

- The SAQ headline metric `key_fact_recall` is per-key-fact
  present/partial/absent — structurally the same as rubric criterion grading.
  So a judge validated on the rubric meta-eval is, by extension, validated for
  the load-bearing part of SAQ judging.
- HealthBench — the methodological anchor — uses a single judge.

### The 3-judge ensemble is retired as the production path

`rescore_open_v2.py` originally used a 3-judge ensemble across model families.
Its rationale does not survive scrutiny here:

- **Family-bias decorrelation** — only matters when a judge grades its own
  family. The model under test is Gemma 4 E4B; no candidate judge is Gemma, so
  the bias never fires.
- **Variance reduction on Likert axes** — weaker than it looks: three strong
  LLMs make *correlated* errors, which barely shrink under averaging; and the
  variance it reduces is on the Likert axes, which are unvalidated and reported
  as secondary anyway (see below).

So the ensemble is **not** the production scorer. The multi-judge code path is
kept only to run **once, as a diagnostic**: score the SAQ rows with all three
bake-off models and compute inter-judge κ. High agreement confirms the single
judge is stable; low agreement is a flag worth investigating.

### What is and isn't validated on the SAQ track

- **`key_fact_recall`** (the headline) — validated, via the rubric calibration
  transferring.
- **Likert axis scores** (accuracy / completeness / contextual_appropriateness
  on 0–4) + safety enum — *not* covered by any shipped calibration set, and
  single-judge Likert is inherently noisy regardless of model. Report these as
  **secondary / supporting detail**, not as physician-validated numbers.

## Implementation sketch

New directory `calibration/judge_validation/`:

1. **Meta-eval script.** Pulls `obgyn_meta_eval.jsonl` from
   `nmrenyi/mamabench@v0.2.1`; runs a candidate judge over each
   `(prompt, completion, rubric)` triple, reusing the rubric grader prompt from
   `end_to_end_eval/rescore_rubric.py` verbatim so validation matches the real
   scoring path. Talks to a **local vLLM OpenAI-compatible endpoint**
   (`--base-url`), not a paid API. Accepts a judge spec on the command line
   for the bake-off.
2. **vLLM serving recipe** per candidate (gpt-oss-120b 1 GPU; Nemotron-Ultra
   4 GPU; Maverick 8 GPU).
3. **Metrics block** — the three agreement frames + prediction distribution /
   rubber-stamp detector + per-category breakdown.
4. **Verdict cache** — store raw judge outputs so metrics recompute without
   re-judging, and so all candidates stay comparable.

This lives in `calibration/` alongside the device-vs-cluster calibration — it
is meta-evaluation, not generation.

Tracked as Task #25. Follow-ups (separate tasks): run the bake-off; pin the
judge; rescore the rubric + SAQ results with the pinned judge.

## Cost / sizing

- 6,853 judge calls per candidate; ~1,361 input tokens/call (~9.33 M input
  tokens total), ~120 output tokens/call.
- On the cluster this is GPU time, not API spend: each candidate finishes the
  full set in minutes (gpt-oss-120b can run 8 replicas; Maverick is one
  whole-node replica).

## Open / out of scope

- **Likert-axis validation.** No shipped dataset provides physician ground
  truth for 0–4 Likert scoring. Until one exists, the SAQ axis scores stay
  secondary.
