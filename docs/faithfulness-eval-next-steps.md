# Faithfulness eval — next steps & working backlog

*Branch `feat/faithfulness-eval`, PR #2 (draft). Updated 2026-06-09.*

Forward-looking TODO + cold-start handoff. v0.2.0 results report:
[`configs/config-v0.2.0/reports/oracle-v0.2.0-faithfulness.html`](../configs/config-v0.2.0/reports/oracle-v0.2.0-faithfulness.html).
Earlier methodology + v0.1.0 results: [`faithfulness-eval-v0.2.0.md`](faithfulness-eval-v0.2.0.md).
Oracle data-quality findings: [`oracle-self-contradictions-v0.1.0.md`](oracle-self-contradictions-v0.1.0.md),
[`oracle-self-contradictions-v0.2.0.html`](../configs/config-v0.2.0/reports/oracle-self-contradictions-v0.2.0.html).

---

## Where we are (cold-start summary)

Generator faithfulness (`mamai-quality-evaluation.md` §3.1) — **two passes
complete and analysed end-to-end**:

- **`v0.1.0` oracle** (top-3 union, score ≥5, 2,659 queries). Calibration
  via Claude Code subagent — **estimated true hallucination rate ≈ 0.33%**.
  6 self-contradictory oracle contexts found.
  Run dir: `configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/`.
- **`v0.2.0` oracle** (top-20 union, score ≥5, 2,989 queries). Calibration
  via gpt-5 batch (OpenAI API, Structured Outputs) — **calibrated true
  hallucination rate = 9.05%**. 16 self-contradictory oracle contexts found.
  Run dir: `configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321/`.

Pipeline stages:

- **Stage 1** `build_oracle.py` → `configs/config-v0.2.0/oracle/mamaretrieval-{v0.1.0,v0.2.0}-score5.jsonl`
- **Stage 2** `eval_faithfulness.py` → Gemma 4 E4B responses under top-3 oracle context
- **Stage 3** `score_lynx.py` → Patronus Lynx 70B PASS/FAIL + reasoning
- **Stage 4** `analyze_lynx_fails.py extract/aggregate` → 5-bucket FAIL categorization
- **Stage 5** `calibrate.py sample/score` → confusion matrix + calibrated rate
- **Judge driver** `score_openai_batch_judge.py` → drives stages 4-5 via OpenAI Batch (gpt-5)

**Headline (Lynx raw vs gpt-5 calibrated, v0.2.0 oracle):**

| Oracle | n   | Lynx PASS | Lynx FAIL | raw FAIL | calibrated true-H |
|---|---:|---:|---:|---:|---:|
| v0.1.0 | 2,659 | 2,514 | 145 | 5.45% | **0.33%** (Claude subagent) |
| v0.2.0 | 2,989 | 2,815 | 174 | 5.82% | **9.05%** (gpt-5 batch) |

The 30× shift in calibrated rate between revisions is **not** apples-to-apples
— v0.1.0's 0.33% came from an in-conversation subagent pass; v0.2.0's 9.05% is
a clean batch API run with strict-schema outputs. The v0.2.0 number stands on
its own methodology; v0.1.0 is a soft historical reference. See the HTML
report §6 for the explicit methodology callout.

**Judge-model history:**
- MiniCheck 7B — rejected (too small, no reasoning).
- Qwen 3.5-397B — rejected (circular: it built the oracle).
- Patronus Lynx 70B — **chosen** for stage 3 (raw scorer): open, reasoning,
  medical, Llama-3 family.
- For stages 4–5 (categorize + calibrate):
  - v0.1.0 used a Claude Code subagent.
  - For v0.2.0, **gpt-oss-120b was tried first** and **failed** the
    pre-committed v0.1.0 validation gate (categorization +22 off, calibration
    75/100 agreement, calibrated rate 11.20% vs Claude's 0.33%). Root cause:
    rubric-following drift on refusal/omission cases. Evidence committed in
    commit `c6b1a24`.
  - **gpt-5 (OpenAI Batch API, Structured Outputs)** was then chosen and
    used for v0.2.0. Cleanly follows the rubric (omission 0/10, refusal 1/10,
    unclear 0/6 on the FAIL stratum).

---

## Done

- **Stages 1–5 on v0.1.0** — pipeline built, run, analysed (Claude subagent
  categorize + calibrate).
- **Stages 1–5 on v0.2.0** — pipeline run; categorize + calibrate via gpt-5
  batch; 16 self-contradictory contexts audited.
- **Analysis scripts consolidated** (was ad-hoc subagents):
  - `generator_eval/analyze_lynx_fails.py` — `extract` + `aggregate`.
  - `generator_eval/calibrate.py` — `sample` + `score`.
  - `generator_eval/score_openai_batch_judge.py` — drives stages 4–5 via
    OpenAI Batch with Structured Outputs; modes `categorize` and `calibrate`.
  - `generator_eval/score_gpt_oss_judge.py` — vLLM-based runner for the
    (rejected) gpt-oss-120b judge; kept as historical infrastructure.
  - `generator_eval/validate_gpt_oss_v0_1.py` — pre-committed v0.1.0 gate
    check that caught the gpt-oss rubric drift.
  - Categorization & calibration **rubrics pinned** as constants; the
    FAIL-category list is single-sourced in
    `analyze_lynx_fails.VALID_CATEGORIES`.
  - Semantic judging is a **deliberate file boundary** — runners emit
    cases + rubric, judges produce a labels/verdicts file, `aggregate`/`score`
    consume it. Backend-agnostic by design.
- **gpt-oss-120b validation pass** (v0.1.0 gate check) — failed all three
  pre-committed gates by wide margins. Documented in commit `c6b1a24` and
  the HTML report's §5.
- **v0.2.0 categorize + calibrate** — gpt-5 batch, $1.78 total cost, 0
  Structured-Outputs parse failures (3 length-truncated rows re-run sync at
  max 16k). Headline rate 9.05%.
- **v0.2.0 self-contradiction audit** — 16/43 contradiction-bucket FAILs
  confirmed as corpus-quality false-FAILs (3× more than v0.1.0's 6/145).
  1 recurrence vs v0.1.0 (q_02878 suture removal D5–8 vs D7); 5 v0.1.0
  cases dropped; 15 new v0.2.0 cases. Root-cause pattern shifted from
  duplicate-section merging (v0.1.0) to cross-guideline retrieval
  (v0.2.0) — the predicted side effect of widening the chunk pool.
- **v0.2.0 HTML report** — self-contained, in
  `configs/config-v0.2.0/reports/oracle-v0.2.0-faithfulness.html`.
- **`tests/` for `generator_eval/`** — Lynx parser, oracle builder,
  calibration math (5 + 17 = 22 tests).
- **Branch housekeeping** — gitignore for regenerable artifacts
  (`lynx_fail_cases.json`, `lynx_categorize_rubric.txt`, `calibration_rubric.txt`).

---

## Remaining backlog

### 1. Update PR #2 description + flip to ready

The active work item. Refresh the PR body with the v0.2.0 headline, link
to the HTML report, summary of the gpt-oss attempt + GPT-5 decision, and
the self-contradiction count (16 confirmed). Flip from draft → ready for
review.

### 2. Optional — human-adjudicated calibration on v0.2.0

A clinician re-judges the 96 calibration rows. Settles whether gpt-5's
26% Lynx precision / 9.05% calibrated rate is closer to truth than
v0.1.0-Claude's 6% / 0.33%. Cost: clinician time (~3–5 hr). Not blocking
PR #2; can run in parallel.

### 3. Optional — file the v0.2.0 self-contradiction findings upstream

The audit surfaced 16 confirmed corpus-quality issues, several of them
clinically significant (anaphylaxis adrenaline route, antihypertensive
threshold, aspirin prophylaxis duration, neonatal head position, IV
ergometrine dose — see `configs/config-v0.2.0/reports/oracle-self-contradictions-v0.2.0.html`).
Recommendation: file against `mamai-medical-guidelines` and/or
`mamaretrieval` for guideline-owner review.

---

## After this — the next new eval work

- **§3.2 stability** — paraphrase sensitivity, run-to-run variance,
  greedy-vs-sampled. New branch.
- **§3.3 deployment integrity** — citation-existence, guideline-contradiction
  set. New branch.
- **Bigger commitments; start once PR #2 lands.**

---

## Deferred / archived decisions

- **Judge backend** — resolved. gpt-5 batch via OpenAI Batch API was chosen
  after gpt-oss-120b validation failure. See the HTML report §5 for the
  decision chain.
- **Calibration reference (LLM vs clinician)** — gpt-5 is the LLM choice
  for v0.2.0. Clinician calibration is now backlog item #2 (optional
  follow-up). The asymmetry between v0.1.0 (Claude subagent) and v0.2.0
  (gpt-5 batch) is the known limitation, called out in the HTML report.

---

## Artifacts & scripts

```
generator_eval/build_oracle.py             stage 1
generator_eval/eval_faithfulness.py        stage 2
generator_eval/score_lynx.py               stage 3
generator_eval/analyze_lynx_fails.py       stage 4 (extract/aggregate)
generator_eval/calibrate.py                stage 5 (sample/score)
generator_eval/score_openai_batch_judge.py judge driver (gpt-5 batch) — used
generator_eval/score_gpt_oss_judge.py      judge driver (gpt-oss, vLLM)  — rejected
generator_eval/validate_gpt_oss_v0_1.py    v0.1.0 gate check (caught gpt-oss drift)
generator_eval/score_minicheck.py          abandoned MiniCheck approach (reference)

oracles
  configs/config-v0.2.0/oracle/mamaretrieval-v0.1.0-score5.jsonl    7,343 (q,c) pairs
  configs/config-v0.2.0/oracle/mamaretrieval-v0.2.0-score5.jsonl   22,282 (q,c) pairs

v0.1.0 run dir  configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/
  oracle_responses.json                          stage 2: 2,659 Gemma responses
  lynx_scores.json                               stage 3: Lynx PASS/FAIL + reasoning
  lynx_fail_categories.json                      stage 4: 145 FAILs categorised (Claude subagent)
  calibration_{blind,key,independent}.json       stage 5: 100-row calibration (Claude subagent)
  oracle_contradictions.json                     6 verified self-contradictory contexts
  lynx_fail_categories_gpt_oss.json (+ meta)     evidence — gpt-oss attempt
  calibration_verdicts_gpt_oss.json  (+ meta)    evidence — gpt-oss attempt

v0.2.0 run dir  configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321/
  oracle_responses.json                          stage 2: 2,989 Gemma responses
  lynx_scores.json                               stage 3: Lynx PASS/FAIL (94.18% PASS)
  lynx_fail_categories.json (+ meta + summary)   stage 4: 174 FAILs (gpt-5 batch)
  calibration_{blind,key,verdicts,report}.json   stage 5: 96-row calibration (gpt-5 batch)
  calibration_verdicts.json.meta.json            sidecar with per-row usage + reasoning
  *.batch_state.json                             batch IDs + submission audit trail
  oracle_contradictions.json                     16 verified self-contradictory contexts

reports
  configs/config-v0.2.0/reports/oracle-v0.2.0-faithfulness.html   v0.2.0 headline report
```
