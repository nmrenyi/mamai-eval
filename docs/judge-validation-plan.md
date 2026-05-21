# Judge validation (meta-eval) plan

*Drafted 2026-05-21. Design notes for validating the LLM judge used in the
`open_ended_rubric` track before trusting any of its scores.*

## Why this exists

The `open_ended` and `open_ended_rubric` tracks are scored by an LLM judge:

- **`open_ended`** (kenya / whb / afrimedqa_saq) — 3-judge ensemble
  (`end_to_end_eval/rescore_open_v2.py`), headline `key_fact_recall`.
- **`open_ended_rubric`** (healthbench_oss_eval / consensus / hard) — single
  judge per criterion, HealthBench weighted-met formula
  (`end_to_end_eval/rescore_rubric.py`).

Every headline number on those tracks rests on the judge's verdicts. A judge
that is, say, 70% aligned with clinicians would silently corrupt the whole
report. So the judge must be **measured against human ground truth before it
is trusted** — and before we pin the judge model IDs in
`configs/config-v0.2.0/params.json` (currently `TBD-pin-before-release`).

This document covers the **rubric judge** (single-judge, binary per-criterion).
The same idea extends to the open-ended ensemble but needs a different ground-
truth source — out of scope here.

## The calibration data: `mamabench@v0.2.1`

mamabench **v0.2.1** (HF tag, released 2026-05-21) ships the judge-calibration
set as a side-file. It is an additive patch over v0.2 — all benchmark rows /
schema / configs are byte-identical, `benchmark_version` stays `"v0.2"`.

- File: `calibration/obgyn_meta_eval.jsonl`
- It is **not a loadable config** — fetch it directly, not via `load_dataset`:
  `https://huggingface.co/datasets/nmrenyi/mamabench/resolve/v0.2.1/calibration/obgyn_meta_eval.jsonl`
- It is an OBGYN-scoped slice of HealthBench's grader meta-evaluation set,
  filtered to mamabench scope by reusing the existing OBGYN classifier
  verdicts (no new classification run). Its prompt set is exactly the
  `consensus` prompt set.

### What a row is

Each row is a `(conversation, completion, single rubric criterion)` triple
carrying **independent physician labels** — this is judge-calibration data,
not benchmark questions:

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
gets a concordant row wrong, that is unambiguously the judge's fault — no
"the criterion was ambiguous" excuse.

On a **discordant** row there is no clean ground truth, so it cannot fairly
pass/fail the judge. Discordant rows are excluded from the headline but
reported separately (see below).

> Note: "consensus" here just means physicians agreed on a calibration row.
> It is unrelated to the `healthbench_consensus` benchmark config.

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
inter-rater level — and that is the honest bar. A judge scoring 95% against
single humans would be *suspicious*: it would be over-agreeing with rater
noise, or rubber-stamping.

## Metric set

### 1. Agreement — three frames side by side

| Comparison | Rows used | What it tells us |
|---|---|---|
| Human ↔ human | all rows w/ ≥2 physicians | the ceiling — 77.6%, κ 0.37 |
| **LLM ↔ single human** | all 6,853 | judge vs *each individual* physician label, averaged over all (judge, physician) pairs. Apples-to-apples with the human baseline (both are single-rater-vs-single-rater). |
| **LLM ↔ physician consensus** | 5,241 concordant | judge vs the unanimous verdict. The clean pass/fail test. |

### 2. Prediction label distribution — the rubber-stamp detector

The concordant subset is **85% met / 15% not-met**. A judge that rubber-stamps
"met" still scores ~85% raw agreement there. Raw agreement alone cannot
distinguish a genuine judge from a lazy one. So also report:

- **Judge met-rate vs physician met-rate (77%).** A judge predicting e.g. 92%
  met is rubber-stamping even if headline agreement looks fine.
- **Agreement split by physician label** — agreement on the *met* rows vs the
  *not-met* rows separately. A rubber-stamp judge scores ~100% on met rows and
  ~0% on not-met rows; that asymmetry is invisible in the headline but obvious
  here. (This is sensitivity / specificity against the physician label.)

### 3. Per-category breakdown

All of the above, sliced by `mamabench_obgyn_category`
(CHILD_HEALTH / MATERNAL / SRH / NEONATAL).

## Pass criteria

A judge is fit to trust for the rubric track if **all** hold:

1. LLM↔human agreement ≈ human↔human agreement (~78%) — at human inter-rater
   level, not far below and not suspiciously above.
2. Judge met-rate ≈ physician met-rate (~77%) — not lopsided toward "met".
3. Agreement is not collapsed on one class — it holds on the not-met rows,
   not only the met rows.

## Doubles as a judge bake-off

Run each candidate judge model through the same script and pick the one whose
numbers sit closest to the human baseline. This turns judge-pinning (the four
`TBD-pin-before-release` slots in `params.json`) from a guess into a
measurement.

## Implementation sketch

New directory `calibration/judge_validation/`:

- A script that:
  1. Pulls `obgyn_meta_eval.jsonl` from `nmrenyi/mamabench@v0.2.1`.
  2. Runs a candidate judge over each `(prompt, completion, rubric)` triple,
     reusing the rubric grader prompt from `end_to_end_eval/rescore_rubric.py`
     so the validation matches the real scoring path exactly.
  3. Emits the three agreement frames + prediction distribution +
     per-category breakdown.
  4. Caches raw judge verdicts so metrics can be recomputed without re-paying
     the API cost, and so multiple candidate judges can be compared.
- It should accept a judge spec (`{provider, model}`) on the command line for
  the bake-off, the same shape `rescore_rubric.py --judge-override` takes.

This belongs in `calibration/` alongside the existing device-vs-cluster
calibration — it is meta-evaluation, not generation.

## Open / out of scope

- **Open-ended ensemble validation.** This plan validates the *rubric* judge.
  Validating the 3-judge `key_fact_recall` ensemble needs a different ground-
  truth source (no meta-eval set ships for it) — separate effort.
- **Cost.** Scales with rows judged (6,853 triples per candidate judge); each
  call is small (one criterion + one short conversation).
