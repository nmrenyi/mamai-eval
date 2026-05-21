# Faithfulness eval — next steps & working backlog

*Branch `feat/faithfulness-eval`, PR #2 (draft). Written 2026-05-21.*

This is the forward-looking TODO. For the full results and methodology see
[`faithfulness-eval-v0.2.0.md`](faithfulness-eval-v0.2.0.md); for the oracle
data-quality finding see [`oracle-self-contradictions-v0.1.0.md`](oracle-self-contradictions-v0.1.0.md).

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

## Immediate backlog — all doable now

No dependency on the deferred decisions below, and nothing wasted by the
v0.2.0 rerun (this is oracle-independent tooling).

1. **Pin the rubrics.** The FAIL-categorization prompt and the calibration
   judge prompt currently exist *only* in this session's ephemeral subagent
   calls — they will be lost. Move them into version-controlled constants
   (in the scripts below, or a `generator_eval/prompts_analysis.py`).
   **Highest priority — only item actively at risk of disappearing.**

2. **`generator_eval/calibrate.py` — deterministic parts.**
   - `sample`: seeded (seed=42) stratified draw — 50 PASS + 10×5 FAIL
     categories — shuffled, blinded → sample + key files.
   - `score`: join independent verdicts vs key → confusion matrix,
     precision/recall, population estimate, bootstrap CI → report JSON.
   - `judge` step: define as "reads a verdicts file." How that file is
     produced (API / subagent / human labels) is **deferred** — keep it a
     pluggable file interface.

3. **`generator_eval/analyze_lynx_fails.py` — deterministic parts.**
   - `extract`: join `lynx_scores.json` + `oracle_responses.json` → FAIL cases.
   - `aggregate`: category distribution + true-hallucination count.
   - `categorize` step: reads a labels file — backend deferred (as #2).

4. **Acceptance-test #2 and #3 against the existing v0.1.0 data.** They must
   reproduce the hand-done numbers:
   - categorization of 145 FAILs → contradiction 48, unsupported_addition 22,
     omission 35, refusal 17, unclear 23 (true-hallucination = 70).
   - calibration → confusion 50 / 0 / 47 / 3, precision 6% (3/50), miss 0/50,
     population estimate ≈ 9 → ≈ 0.3%.
   If a script reproduces these, it has faithfully captured the methodology.

5. **`tests/` for `generator_eval/`.** Cover `_parse_lynx_output` /
   `_extract_bullets` in `score_lynx.py` (three runtime bugs were hit there),
   `build_oracle` chunk-selection, and the `calibrate score` math.

6. **Branch housekeeping.** Drop the now-regenerable `lynx_fails_for_review.json`
   from git (script #3 reproduces it). Refresh the PR #2 description to cover
   the contradiction audit + doc updates committed after it was opened.

---

## Deferred — each blocks one task, needs a decision

- **Judge/categorize backend** — scripted LLM API call vs subagent. (Operator
  declined to commit to an API path for now.) Tasks #2/#3 are built backend-
  agnostic (file interface), so they are *not* blocked — only the automated
  end-to-end run is.
- **Calibration reference** (`faithfulness-eval-v0.2.0.md` §8.1) — human
  clinician (gold standard, the doc's intent) vs a second open model
  (cheap triangulation). Our one calibration was a Claude subagent; a second
  Claude pass adds nothing. Also gates §3.3's contradiction set.
- **PR #2** — merge as "infrastructure + preliminary results" now, or hold
  until the v0.2.0 result is in.

---

## When mamaretrieval v0.2.0 (top-20 union) lands — the rerun

1. `build_oracle.py --revision v0.2.0` → new oracle.
2. Rerun stage 2 (`eval_faithfulness.py`) and stage 3 (`score_lynx.py`) on the
   cluster — pipeline is parameterized; ~2 h.
3. Rerun `analyze_lynx_fails.py` + `calibrate.py` (push-button once #1–4 done).
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
set). Bigger commitments; start once the backlog above is clean.

## Artifacts (run dir `…/gemma4-e4b/20260520T094749/`)

```
oracle_responses.json        stage-2: 2,659 Gemma responses
lynx_scores.json             stage-3: Lynx PASS/FAIL + reasoning
lynx_fail_analysis.json      145 FAILs categorised
calibration_{blind,key,independent}.json   100-row calibration
oracle_contradictions.json   6 verified self-contradictory contexts
```
