# Judge validation bake-off — result and pinning decision

*Drafted 2026-05-22. Final report for Task #25 (judge bake-off).*

## TL;DR

**Pinned: `openai/gpt-oss-120b`** as the uniform single judge for both the
rubric and SAQ tracks. It's the only candidate that passes all three pass
criteria from `docs/judge-validation-plan.md`. It also happens to be the
operationally cheapest to run (1 GPU vs 4 for the other two).

| Candidate | crit 1: ≈ 78% | crit 2: met ≈ 77% | crit 3: not-met agr | Pass | Footprint |
|---|---:|---:|---:|:---:|---|
| **gpt-oss-120b** | 74.3% | **73.2%** ✓ | **67.9%** ✓ | **3/3 ✓** | 1× H100/H200 |
| Nemotron-Ultra-253B (FP8 q.) | **78.8%** | 83.3% ✗ | 55.3% ✗ | 1/3 | 4× H200 |
| Llama 4 Maverick FP8 | 79.8% | 88.8% ✗ | 44.6% ✗ | 1/3 | 4× H200 |

## Side-by-side numbers

All measured on the OBGYN-scoped HealthBench grader meta-eval
(`mamabench@v0.2.1/calibration/obgyn_meta_eval.jsonl`, 6,853 physician-
labeled `(prompt, completion, criterion)` triples; 5,241 concordant rows).

| Metric | Human-human | gpt-oss-120b | Nemotron-Ultra (FP8 q.) | Llama 4 Maverick |
|---|---:|---:|---:|---:|
| Pairs / rows | 7,777 | 13,962 | 14,160 | 14,162 |
| **LLM ↔ single human** | 77.6% | 74.3% [73.4, 75.2] | **78.8%** [78.0, 79.6] | 79.8% [79.0, 80.6] |
| **LLM ↔ consensus** (concordant) | — | 81.6% [80.5, 82.7] | 87.4% [86.4, 88.2] | **88.7%** [87.8, 89.5] |
| Judge met-rate | 77.0% (phys) | **73.2%** | 83.3% | 88.8% |
| Δ met-rate vs phys | — | **−3.9** | +6.3 | +11.8 |
| Agreement on met rows | — | 84.0% | 92.9% | 96.3% |
| **Agreement on not-met rows** | — | **67.9%** | 55.3% | 44.6% |
| Verdicts valid / total | — | 6,753 / 6,853 (98.5%) | 6,852 / 6,853 (99.99%) | 6,853 / 6,853 (100%) |
| Cohen's κ (humans) | 0.366 | — | — | — |
| 95% CI on LLM↔single | — | [73.4, 75.2] | [78.0, 79.6] | [79.0, 80.6] |
| 95% CI on LLM↔consensus | — | [80.5, 82.7] | [86.4, 88.2] | [87.8, 89.5] |

## Per-category (concordant agreement, %)

| Category | n | gpt-oss-120b | Nemotron-Ultra | Llama 4 Maverick | Human baseline |
|---|---:|---:|---:|---:|---:|
| CHILD_HEALTH | 2,463 | 80.0 | 86.8 | 88.5 | 76.8 |
| MATERNAL | 1,777 | 82.7 | 88.4 | 88.1 | 79.1 |
| NEONATAL | 291 | 80.4 | 85.2 | 89.7 | 77.6 |
| SRH | 710 | 84.8 | 87.6 | 90.4 | 76.3 |

All three candidates beat the local human baseline in every category, but
this is partly because a single judge applies the same model consistently
across rows, whereas physicians rotate (the human baseline is variance-
adjusted, the judge isn't).

## Why higher headline ≠ better judge here

A naive read of "LLM↔consensus" alone would pick Maverick (88.7%) over
gpt-oss-120b (81.6%). But the concordant subset is **85% met / 15% not-met**:
a judge that just says "met" all the time gets ~85% agreement by default.
That's exactly what the rubber-stamp detector is for, and it fires hard on
Maverick and Nemotron:

- **Maverick:** met-rate 88.8% (vs phys 77%, +11.8 pp), agreement on met
  rows 96.3%, agreement on not-met rows **44.6%**. Disagrees with
  physicians 55% of the time when they said not-met — that's worse than
  chance on the negative class. Textbook rubber-stamp pattern, just at a
  high enough overall rate that the headline doesn't betray it.
- **Nemotron-Ultra:** less extreme but the same shape. Met-rate 83.3%
  (+6.3 pp), not-met agreement 55.3% vs met-row 92.9%. Asymmetric in the
  same direction.
- **gpt-oss-120b:** met-rate 73.2% — actually 4 pp *below* physicians (a
  slight under-rating, not over-rating). Agreement on met rows 84.0%,
  not-met rows 67.9% — the smallest asymmetry of the three. The 67.9%
  on not-met is the only number among the three judges that's
  meaningfully above chance.

For deployment, an over-rating judge would inflate Gemma's apparent
quality on both the rubric track (`weighted_met`) and the SAQ track
(`key_fact_recall`), exactly the metrics our v0.2 report rests on. An
under-rating judge errs the other way — conservative, defensible.

## Operational considerations (the tiebreak, if we needed it)

| Candidate | GPUs to serve | Approx. recurring rescore cost |
|---|---|---|
| gpt-oss-120b | 1× H100 or H200 | minutes/run, fits in 1-GPU allocation |
| Nemotron-Ultra-253B (FP8 q.) | 4× H200 | full-node allocation, 17 min model load each time |
| Llama 4 Maverick FP8 | 4× H200 | full-node allocation, ~20 min model load |

If two had tied on the pass criteria, the lighter model would have won.
gpt-oss-120b passes the criteria *and* is the cheapest — no conflict.

## What we now pin

In `configs/config-v0.2.0/params.json`:

```json
"judge": {
  "model": "openai/gpt-oss-120b",
  "temperature": 0.0,
  "rubric": {"provider": "openai", "model": "openai/gpt-oss-120b"}
}
```

This judge is the **uniform single judge for both tracks** per the plan
doc:

- **Rubric track** (`open_ended_rubric`: healthbench_oss_eval / consensus
  / hard) — used directly via `rescore_rubric.py`. Atomic call shape is
  identical to the calibration, so the 74.3% LLM↔human and 81.6%
  LLM↔consensus numbers transfer call-for-call.
- **SAQ track** (`open_ended`: kenya / whb / afrimedqa_saq) — same judge,
  same atomic call (per-key-fact present/partial/absent). The 3-judge
  ensemble in `rescore_open_v2.py` is retired as the production scorer
  (family-decorrelation rationale doesn't apply with no Gemma judge);
  the multi-judge code path is kept only as a one-time inter-judge-κ
  diagnostic if useful.

## Serving notes for the production rescore

`rescore_rubric.py` and `rescore_open_v2.py` currently call the OpenAI
SDK against `api.openai.com`. To use gpt-oss-120b served on our cluster,
the scoring step needs to point the SDK at a local vLLM endpoint
(`OPENAI_BASE_URL=http://<host>:8000/v1`, `OPENAI_API_KEY=EMPTY`). vLLM
config that worked for the calibration:

```
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \      # 1 also fine; 4 used during bake-off
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92
```

Extra-body kwarg used in calibration: `{"reasoning_effort": "medium"}`
(gpt-oss-120b is a reasoning model with low/medium/high effort).

## Caveats

- gpt-oss-120b returned 100 unparseable verdicts (1.5% error rate). The
  metric module excludes errored rows. Cluster-side `cat verdicts.jsonl
  | jq '.error'` could surface what the model emitted on those rows; not
  urgent.
- Nemotron's calibration used `--quantization fp8` (on-the-fly), not
  native FP8 weights, because no FP8 release of Nemotron-Ultra-253B
  exists. Quality impact for binary criterion judging is expected to be
  minimal but not directly measured here.
- The bake-off does not directly validate Likert-axis scoring on the SAQ
  track (no physician ground truth for 0-4 Likert). Per the plan doc,
  treat SAQ axis scores as supporting detail; the validated headline
  for SAQ is `key_fact_recall`.

## Artifacts (committed on `feat/judge-validation-20260522`)

- `calibration/judge_validation/verdicts/<judge>/verdicts.jsonl` — raw
  per-row verdicts (one JSONL record per triple).
- `calibration/judge_validation/reports/<judge>/{report.md, report.json}`
  — per-candidate metric report.
- This document.

## Follow-ups (separate tasks)

1. Wire `OPENAI_BASE_URL` into `rescore_rubric.py` and `rescore_open_v2.py`
   so production scoring can target a local vLLM endpoint.
2. Run `rescore_rubric.py` over the HealthBench ±RAG generation results
   (`results/end_to_end_eval/gemma4-e4b/{20260521T123051-cluster-norag-rubric,
   20260521T122626-cluster-rag-rubric}`) with the pinned judge.
3. Run `rescore_open_v2.py` (single-judge mode now) over the SAQ
   ±RAG generation results.
4. Write the open-ended ±RAG comparison report.
