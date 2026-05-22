# Faithfulness eval — next steps & working backlog

*Branch `feat/faithfulness-eval`, PR #2 (draft). Updated 2026-05-22.*

Forward-looking TODO + cold-start handoff. Full results & methodology:
[`faithfulness-eval-v0.2.0.md`](faithfulness-eval-v0.2.0.md). Oracle
data-quality finding: [`oracle-self-contradictions-v0.1.0.md`](oracle-self-contradictions-v0.1.0.md).

---

## Where we are (cold-start summary)

Generator faithfulness (`mamai-quality-evaluation.md` §3.1) — **first pass
complete** on the mamaretrieval `v0.1.0` oracle (top-3 union, chunks scored ≥5,
2,659 queries).

- **Stage 1** `build_oracle.py` → `configs/config-v0.2.0/oracle/mamaretrieval-v0.1.0-score5.jsonl`
- **Stage 2** `eval_faithfulness.py` → 2,659 Gemma 4 E4B responses under top-3 oracle context
- **Stage 3** `score_lynx.py` → Patronus Lynx 70B PASS/FAIL + reasoning
- Run dir: `configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/`

**Headline:** Lynx raw pass rate **94.55%** (2,514 PASS / 145 FAIL). But the
145 FAILs were audited and a 100-row calibration run: **Lynx is ~6% precision**
(heavy over-flagging), 0/50 misses on the PASS side. **Estimated true
hallucination rate ≈ 0.3%.** Also found & filed upstream: 6 self-contradictory
oracle contexts (mamaretrieval#18).

**Judge-model history:** MiniCheck (rejected — too small, no reasoning) → Qwen
3.5-397B (rejected — circular: it built the oracle) → **Patronus Lynx 70B**
(chosen — open, reasoning, medical, Llama-3 family).

**A mamaretrieval `v0.2.0` (top-20 union) is expected** → a full rerun is
planned once it lands.

---

## Done

- **Stages 1–3** — pipeline built, run, results in the run dir above.
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

## Remaining backlog (do these next — all doable now)

No dependency on the deferred decisions; nothing wasted by the v0.2.0 rerun.

1. **`tests/` for `generator_eval/`.** Cover `_parse_lynx_output` /
   `_extract_bullets` in `score_lynx.py` (three runtime bugs were hit there —
   malformed-JSON, unescaped-apostrophe, NameError), `build_oracle`
   chunk-selection, and the `calibrate score` confusion-matrix math.

2. **Branch housekeeping.**
   - Drop the now-regenerable `lynx_fails_for_review.json` from git
     (`analyze_lynx_fails extract` reproduces it as `lynx_fail_cases.json`).
   - Refresh the PR #2 description (it predates the consolidation scripts, the
     contradiction audit, and the docs).

After these two, the immediate backlog is clear and the branch is a clean,
reviewable, self-contained unit.

---

## Deferred — each needs a decision

- **Judge/categorize backend** — scripted LLM API call vs subagent vs human.
  The scripts are backend-agnostic (file interface), so this blocks only the
  *automated* end-to-end run, not the scripts themselves.
- **Calibration reference** (`faithfulness-eval-v0.2.0.md` §8) — human
  clinician (gold standard, the doc's intent) vs a second open model (cheap
  triangulation). Our one calibration was a Claude subagent; a second Claude
  pass adds nothing. Also gates §3.3's contradiction set.
- **PR #2** — merge as "infrastructure + preliminary results" now, or hold
  until the v0.2.0 result is in.

---

## When mamaretrieval v0.2.0 (top-20 union) lands — the rerun

1. `build_oracle.py --revision v0.2.0` → new oracle.
2. Rerun stage 2 (`eval_faithfulness.py`) and stage 3 (`score_lynx.py`) on the
   cluster — pipeline is parameterized; ~2 h.
3. Rerun `analyze_lynx_fails.py` + `calibrate.py` — push-button now that the
   scripts exist.
4. Re-audit the oracle self-contradictions (may differ with a richer pool).
5. Update `faithfulness-eval-v0.2.0.md` with the v0.2.0 numbers; flip PR #2 to
   ready.

Note: faithfulness is largely invariant to *which* good context is given, so
the headline rate is not expected to move much. The richer oracle should
mainly **reduce Lynx false-FAILs** (fewer "true but not in our thin oracle"
flags), nudging the raw rate up.

---

## After this — the next new eval work

§3.2 stability (paraphrase sensitivity, run-to-run variance, greedy-vs-sampled)
and §3.3 deployment integrity (citation-existence, guideline-contradiction
set). Bigger commitments; start once the backlog above is clear.

## Artifacts & scripts

```
generator_eval/build_oracle.py          stage 1
generator_eval/eval_faithfulness.py     stage 2
generator_eval/score_lynx.py            stage 3
generator_eval/analyze_lynx_fails.py    FAIL categorization (extract/aggregate)
generator_eval/calibrate.py             calibration (sample/score)
generator_eval/score_minicheck.py       abandoned MiniCheck approach (reference)

run dir  configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/
  oracle_responses.json        stage-2: 2,659 Gemma responses
  lynx_scores.json             stage-3: Lynx PASS/FAIL + reasoning
  lynx_fail_analysis.json      145 FAILs categorised
  calibration_{blind,key,independent}.json   100-row calibration
  oracle_contradictions.json   6 verified self-contradictory contexts
```
