# Generator Faithfulness Evaluation — Gemma 4 E4B (config-v0.2.0)

*Track 3 (`generator_eval/`). Branch `feat/faithfulness-eval`. Written 2026-05-21.*

Summarises the faithfulness evaluation of Gemma 4 E4B under oracle context — the
pipeline, the methodology decisions taken along the way, the results, and the
calibration finding that reframes the headline number.

---

## 1. Goal

Per `mamai-quality-evaluation.md` §3.1: measure whether Gemma 4 E4B *uses
retrieved clinical-guideline context faithfully* — i.e. without hallucinating
beyond or contradicting it. The generator is evaluated **in isolation under
oracle context**, so any failure is the generator's, not the retriever's.

## 2. Pipeline

Three stages, all under `generator_eval/`:

| Stage | Script | Output | Venue |
|---|---|---|---|
| 1 — Oracle build | `build_oracle.py` | `configs/config-v0.2.0/oracle/mamaretrieval-v0.1.0-score5.jsonl` | local (HF) |
| 2 — Generation | `eval_faithfulness.py` | `…/results/generator/gemma4-e4b/<ts>/oracle_responses.json` | cluster, 1×A100 |
| 3 — Faithfulness scoring | `score_lynx.py` | `…/<ts>/lynx_scores.json` | cluster, 2×A100 (vLLM) |

Run directory: `configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/`.

## 3. Methodology decisions (the discussion)

**Oracle = mamaretrieval `score ≥ 5`.** Oracle context is drawn from
`nmrenyi/mamaretrieval@v0.1.0`, using all chunks a relevance judge scored
`≥ 5` on its 0–6 rubric (the threshold the judge was validated at vs Claude
Opus). Yields **2,659 queries** (of 3,185) with ≥1 oracle chunk, 7,343
(query, chunk) pairs. A stricter `score = 6` oracle (1,749 queries) was
deferred as a sensitivity check.

**Top-3 cap at generation.** Score filtering selects the oracle; at generation
time each query is given its **top-3** highest-scored chunks (matches the
deployed retrieval depth `top_k = 3`). 762 queries (29%) had >3 oracle chunks
and were capped.

**Judge-model journey** — three candidates, two rejected:

1. **MiniCheck (Bespoke-MiniCheck-7B)** — built and smoke-tested. Rejected: at
   7B it is small, and it is a pure classifier — no reasoning, so a verdict
   can't be audited. (`score_minicheck.py` kept in-tree for reference.)
2. **Qwen3.5-397B-A17B** — rejected for **circularity**: that same model
   produced the mamaretrieval relevance labels that *built* the oracle. Using
   it again to judge faithfulness against that oracle means a shared blind
   spot would inflate the score undetectably.
3. **Patronus Lynx 70B** (`PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct`) —
   **chosen.** Open weights; purpose-built for RAG hallucination detection;
   medical-domain trained (PubMedQA); emits bullet-point reasoning; Llama-3
   family — independent of the Qwen oracle judge. Limitation: it gives a
   **holistic PASS/FAIL** per response, not a per-claim breakdown.

## 4. Results

### Stage 2 — generation
2,659 responses from Gemma 4 E4B (Q4_0 GGUF, llama-cpp, deployment params:
T=1.0, top_p=0.95, top_k=64). 0 errors, 0 context-window overflows. ~54 min.

### Stage 3 — Lynx faithfulness scoring
2,659 responses scored, 0 parse errors, 0 oversize, ~27 min on 2×A100.

| Metric | Value |
|---|---:|
| Lynx pass rate | **94.55%** |
| PASS / FAIL | 2,514 / 145 |
| 95% CI (bootstrap) | [93.6%, 95.4%] |

### FAIL categorization (all 145, one agent, one-by-one)

| Category | Count | Genuine faithfulness failure? |
|---|---:|:--:|
| `contradiction` | 48 | yes |
| `unsupported_addition` | 22 | yes |
| `omission` (answer incomplete) | 35 | no |
| `refusal` (model declined) | 17 | no |
| `unclear` (no defect found) | 23 | no |

Only 70/145 FAILs are even candidate hallucinations; 96/145 verdicts were
independently judged "questionable" (over-strict).

### Calibration (verification ladder #5)

100-row **stratified, blinded** sample — 50 Lynx-PASS + 50 Lynx-FAIL (10 per
FAIL category) — independently re-judged by a different model family (Claude
subagent), blind to Lynx's verdict, under a strict hallucination-only
definition (contradiction or unsupported addition only; incompleteness and
refusal are PASS).

Confusion matrix (n=100):

| | ref PASS | ref FAIL |
|---|---:|---:|
| **Lynx PASS** | 50 | **0** — no missed hallucination |
| **Lynx FAIL** | **47** — false alarm | 3 |

- **Lynx precision ≈ 6%** (3 of 50 Lynx-FAILs independently confirmed).
- **Lynx miss rate = 0/50** on the PASS stratum — no hallucination it wrongly
  passed (but a 0/50 bound is loose; rule-of-three upper bound ≈ 6%).
- Per-category confirmation within Lynx-FAIL: contradiction 1/10,
  unsupported_addition 2/10, omission 0/10, refusal 0/10, unclear 0/10.
- **Population estimate of true hallucinations: ~9 / 2,659 ≈ 0.3%.**

## 5. Key findings

1. **Lynx is high-recall, very-low-precision on this dataset.** It catches
   hallucinations but floods false positives — precision ≈ 6%. Its raw 5.45%
   FAIL rate overstates hallucination by roughly an order of magnitude.
2. **Estimated true hallucination rate ≈ 0.3–2.6%** depending on judge; the
   independent calibration puts it near the bottom of that range. Gemma 4 E4B
   is, on this evidence, **highly faithful to oracle context**.
3. **Self-contradictory oracle context.** **6** of the 145 FAILs (verified by
   per-case audit, quotes substring-checked against the raw context) are
   *false* FAILs: the oracle context contains two chunks stating conflicting
   clinical facts (e.g. misoprostol 13–22 wk — table "every 3 h" vs prose
   "every 4–6 h"); the answer matches one chunk, Lynx faults it against the
   other. A **mamaretrieval / guideline-corpus data-quality issue**, not a
   model failure. Full audit with verbatim passages and chunk IDs:
   [`oracle-self-contradictions-v0.1.0.md`](oracle-self-contradictions-v0.1.0.md).
4. **17 refusals** — the model declines clinically answerable questions
   ("consult a doctor", "out of scope") even when the oracle fully answers
   them. Not hallucination, but a real usefulness/product issue worth its own
   tracking.
5. **Omission ≠ unfaithfulness.** Lynx penalises incomplete answers; the §3.1
   definition does not. 35 FAILs are omissions.

## 6. How to report the headline

Not "5.45% hallucination rate." Report tiered:

- Lynx raw FAIL: **5.45%** — conservative upper bound, ~6%-precision over-flagging.
- Categorised true-hallucination candidates: **2.6%**.
- Independent-calibration estimate: **~0.3%** → faithfulness ≈ **99.7%**.

## 7. Limitations

- **Both judges are LLMs, not ground truth.** Lynx over-flags; the calibration
  judge (Claude) may under-flag. The true rate is bracketed, not pinned — only
  human expert adjudication closes that.
- **PASS-side miss rate** rests on 0/50; the upper bound on hallucinations Lynx
  wrongly passed across 2,514 PASS rows is non-trivial.
- **Holistic verdict** — Lynx gives one PASS/FAIL per response, no per-claim
  localisation.
- **Oracle = score ≥ 5 subset**; `score = 6` sensitivity not yet run.
- **Single generation** — no stability probes (§3.2) or deployment-integrity
  checks (§3.3) yet.

## 8. Next steps

1. **Human-adjudicated calibration** — a clinician rates ~50–100 rows; the gold
   standard the doc ultimately wants. Pins the bracket from §7.
2. **`score = 6` oracle sensitivity** — re-aggregate against the stricter oracle.
3. **Stability (§3.2)** — paraphrase sensitivity, run-to-run variance, greedy
   vs sampled.
4. **Deployment integrity (§3.3)** — citation-existence + guideline-contradiction set.
5. **Upstream** — the self-contradictory-context finding is filed against
   mamaretrieval (see `oracle-self-contradictions-v0.1.0.md`); prose-vs-table
   conflicts may also need a guideline-corpus fix.

## Artifacts

```
configs/config-v0.2.0/oracle/mamaretrieval-v0.1.0-score5.jsonl      stage-1 oracle
configs/config-v0.2.0/results/generator/gemma4-e4b/20260520T094749/
  ├── oracle_responses.json        stage-2: 2,659 Gemma responses
  ├── lynx_scores.json             stage-3: Lynx PASS/FAIL + reasoning
  ├── lynx_fail_analysis.json      categorisation of the 145 FAILs
  ├── calibration_blind.json       100-row blinded calibration sample
  ├── calibration_key.json         calibration sample → Lynx verdict/category
  └── calibration_independent.json independent re-judging of the 100
```
