# Faithfulness eval — next steps & working backlog

*Branch `feat/faithfulness-eval`, PR #2 (draft). Updated 2026-05-25.*

Forward-looking TODO + cold-start handoff. Full results & methodology:
[`faithfulness-eval-v0.2.0.md`](faithfulness-eval-v0.2.0.md). Oracle
data-quality finding: [`oracle-self-contradictions-v0.1.0.md`](oracle-self-contradictions-v0.1.0.md).

---

## Where we are (cold-start summary)

Generator faithfulness (`mamai-quality-evaluation.md` §3.1) — **two passes
complete**:

- **`v0.1.0` oracle** (top-3 union, score ≥5, 2,659 queries) — first pass,
  fully analysed (categorisation + 100-row calibration).
  Run dir: `configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/`.
- **`v0.2.0` oracle** (top-20 union, score ≥5, 2,989 queries) — stages 1–3
  rerun on the new revision; analysis still pending.
  Run dir: `configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321/`.

Pipeline stages:

- **Stage 1** `build_oracle.py` → `configs/config-v0.2.0/oracle/mamaretrieval-{v0.1.0,v0.2.0}-score5.jsonl`
- **Stage 2** `eval_faithfulness.py` → Gemma 4 E4B responses under top-3 oracle context
- **Stage 3** `score_lynx.py` → Patronus Lynx 70B PASS/FAIL + reasoning
- **Analysis** `analyze_lynx_fails.py` + `calibrate.py` (post-hoc; backend-agnostic)

**Headline (raw, uncalibrated Lynx pass rate):**

| Oracle | n   | PASS | FAIL | rate | 95% CI |
|---|---:|---:|---:|---:|:---:|
| v0.1.0 | 2,659 | 2,514 | 145 | **94.55%** | [93.6%, 95.4%] |
| v0.2.0 | 2,989 | 2,815 | 174 | **94.18%** | [93.3%, 95.0%] |

CIs overlap heavily → **no real movement** between revisions. The richer
oracle was expected to *reduce* false-FAILs (fewer "true but not in our
thin oracle" flags) — that didn't show up at the raw layer. Re-running
calibration on v0.2.0 will say whether the *true*-hallucination estimate
moves.

**v0.1.0 calibration:** Lynx is ~6% precision (heavy over-flagging), 0/50
misses on the PASS side. **Estimated true hallucination rate ≈ 0.3%.**
Also filed upstream: 6 self-contradictory oracle contexts
(mamaretrieval#18).

**Judge-model history:** MiniCheck (rejected — too small, no reasoning) →
Qwen 3.5-397B (rejected — circular: it built the oracle) → **Patronus
Lynx 70B** (chosen — open, reasoning, medical, Llama-3 family).

---

## Done

- **Stages 1–3 on v0.1.0** — pipeline built, run, results in the v0.1.0
  run dir.
- **Stages 1–3 on v0.2.0** — committed; oracle 2,989 q × 22,282 (q,c)
  pairs, stage 2 / 3 outputs in the v0.2.0 run dir.
- **Analysis consolidated into committed scripts** (was ad-hoc subagents):
  - `generator_eval/analyze_lynx_fails.py` — `extract` + `aggregate`.
  - `generator_eval/calibrate.py` — `sample` + `score`.
  - Categorization & calibration **rubrics pinned** as constants in those
    scripts; the FAIL-category list is single-sourced in
    `analyze_lynx_fails.VALID_CATEGORIES`.
  - The semantic judging step is a **deliberate file boundary** — no LLM
    backend wired; `extract`/`sample` emit cases + rubric, `aggregate`/`score`
    consume a labels/verdicts file.
  - **Acceptance-tested** against the v0.1.0 data — scripts reproduce the
    hand-done numbers exactly (categorization 48/22/35/17/23; calibration
    confusion 50/0/47/3, 6% precision, ~0.3%).
- **`tests/` for `generator_eval/`** — 22 tests covering
  `_parse_lynx_output` / `_extract_bullets` (the three runtime bugs from
  the v0.1.0 cluster run), `build_oracle` chunk selection, and the
  `calibrate score` confusion-matrix math.
- **Branch housekeeping** — dropped the regenerable
  `lynx_fails_for_review.json`; refreshed PR #2 description to reflect
  the consolidation scripts, contradiction audit, and docs.
- **Legacy-docs cleanup** — `refactor-plan-tracks.md` removed (executed);
  `v0.2-evaluation-handoff.md` kept (still-useful orientation).

---

## Remaining backlog

### 1. Categorize + calibrate the v0.2.0 FAILs (push-button)

The pipeline exists; only the external semantic-judging step is unwired.

```sh
# 1. Categorize the 174 FAILs
python -m generator_eval.analyze_lynx_fails extract \
    configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321
# → ...lynx_fail_cases.json + ...lynx_categorize_rubric.txt
# [run an LLM judge over the cases using the rubric →
#    ...lynx_fail_categories.json]
python -m generator_eval.analyze_lynx_fails aggregate \
    configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321 \
    --categories .../lynx_fail_categories.json

# 2. Calibrate (100-row stratified blind sample)
python -m generator_eval.calibrate sample \
    configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321
# → ...calibration_blind.json + ...calibration_rubric.txt
# [run an INDEPENDENT judge — second model OR a clinician — over the
#    blinded cases → ...calibration_verdicts.json]
python -m generator_eval.calibrate score \
    configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321 \
    --verdicts .../calibration_verdicts.json
```

Blocked only by the **judge-backend** decision (see Deferred).

### 2. Re-audit oracle self-contradictions on v0.2.0

The 6 v0.1.0 contradictions were a real corpus-quality finding. With a
3× richer chunk pool, the count and shape may change. Mechanical: pull
multi-chunk queries from the v0.2.0 oracle and re-run the per-case audit
that built `oracle-self-contradictions-v0.1.0.md`.

### 3. Update writeup + close PR #2

Add v0.2.0 numbers (and post-calibration true-hallucination estimate) to
`faithfulness-eval-v0.2.0.md`; flip PR #2 to ready once calibration is in.

---

## Deferred — each needs a decision

- **Judge/categorize backend** — scripted LLM API call vs subagent vs human.
  The scripts are backend-agnostic (file interface), so this blocks only
  the *automated* end-to-end run, not the scripts themselves.
- **Calibration reference** (`faithfulness-eval-v0.2.0.md` §8) — human
  clinician (gold standard, the doc's intent) vs a second open model
  (cheap triangulation). v0.1.0's calibration was a Claude subagent; a
  second Claude pass adds nothing. Also gates §3.3's contradiction set.
- **PR #2** — merge as "infrastructure + v0.1.0 + v0.2.0 raw" now, or
  hold until v0.2.0 calibration lands.

---

## After this — the next new eval work

§3.2 stability (paraphrase sensitivity, run-to-run variance, greedy-vs-sampled)
and §3.3 deployment integrity (citation-existence, guideline-contradiction
set). Bigger commitments; start once the backlog above is clear.

---

## Artifacts & scripts

```
generator_eval/build_oracle.py          stage 1
generator_eval/eval_faithfulness.py     stage 2
generator_eval/score_lynx.py            stage 3
generator_eval/analyze_lynx_fails.py    FAIL categorization (extract/aggregate)
generator_eval/calibrate.py             calibration (sample/score)
generator_eval/score_minicheck.py       abandoned MiniCheck approach (reference)

oracles
  configs/config-v0.2.0/oracle/mamaretrieval-v0.1.0-score5.jsonl    7,343 (q,c) pairs
  configs/config-v0.2.0/oracle/mamaretrieval-v0.2.0-score5.jsonl   22,282 (q,c) pairs

v0.1.0 run dir  configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/
  oracle_responses.json        stage-2: 2,659 Gemma responses
  lynx_scores.json             stage-3: Lynx PASS/FAIL + reasoning
  lynx_fail_analysis.json      145 FAILs categorised
  calibration_{blind,key,independent}.json   100-row calibration
  oracle_contradictions.json   6 verified self-contradictory contexts

v0.2.0 run dir  configs/config-v0.2.0/results/generator/gemma4-e4b/20260524T144321/
  oracle_responses.json        stage-2: 2,989 Gemma responses
  lynx_scores.json             stage-3: Lynx PASS/FAIL + reasoning (94.18%)
  (categorisation + calibration pending — backlog item 1)
```
