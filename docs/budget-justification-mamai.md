# LLM Judge API Budget — MAM-AI v0.2 HealthBench

Two budget items for the HealthBench rubric track of the v0.2 evaluation, both via the closed-source Batch API. **Starting model: `gpt-5-mini`** — the smallest OpenAI option, used as a low-cost feasibility check before committing to larger spend. *(Separate smaller asks of $30 each cover the SAQ rescore and the faithfulness judge.)*

| Phase | Purpose | Estimated batch cost | **Ask** |
|---|---|---:|---:|
| **A. Calibration** | Validate `gpt-5-mini` as LLM judge against physician ground truth | $6 | **$15** |
| **B. Production rescore** | Score 38,308 model responses on the HealthBench rubric track | $33 | **$50** |

**Current ask total: $65** (gpt-5-mini batch). A further ~$50 covers 2 anticipated retrieval-module iterations (see Forward-looking section) — filed separately when each triggers, not part of this approval. **Full project envelope at gpt-5-mini: ~$115.**

Strategy: gpt-5-mini is the cheapest plausible OpenAI judge. Phase A produces a metric report directly comparable to the open-weight bake-off (Background) on the same 6,853 physician-labeled triples. If gpt-5-mini materially beats the open-weight not-met agreement (67.9%), we run Phase B at gpt-5-mini for the full v0.2 report. If not, we escalate to a higher-tier closed-source judge (gpt-5.4, gpt-5.5, or alternatives — see "Escalation options" below). Either way, Phase A is the empirical gate.

Open-weight judges over-rate "met" on a meaningful fraction of physician-not-met rows (Background), biasing the deployment-headline rubric metric in the wrong direction — so a closed-source judge is necessary for the final v0.2 evaluation. The question Phase A answers is which closed-source tier suffices.

---

## Background

MAM-AI v0.2's open-ended evaluation tracks use an LLM judge to score Gemma 4 E4B responses against physician-written rubric criteria. We **already** ran an open-source bake-off across three candidates (Llama 4 Maverick FP8, Nemotron-Ultra-253B FP8-quantized, gpt-oss-120b) on a 6,853-row physician-labeled meta-evaluation set (`mamabench@v0.2.1/calibration/obgyn_meta_eval.jsonl`). Result: **all three open candidates systematically over-rate "met"** on the rows where physicians unanimously agreed the criterion was **NOT** met.

| Judge (open-weight) | Footprint | Judge met-rate vs phys 77% | Not-met agreement | Over-rated rate on not-met rows |
|---|---|---:|---:|---:|
| Llama 4 Maverick FP8 | 4× H200 | 88.8% (+11.8 pp) | 44.6% | 55.4% |
| Nemotron-Ultra-253B (FP8 q.) | 4× H200 | 83.3% (+6.3 pp) | 55.3% | 44.7% |
| **gpt-oss-120b (medium, pinned)** | **1× H200** | **73.2% (−3.9 pp)** | **67.9%** | **32.1%** |

The pinned candidate (gpt-oss-120b medium) is the **best** of the three. We also confirmed higher reasoning effort doesn't fix it: gpt-oss-120b at `reasoning_effort=high` improved not-met agreement by only 1.7 pp (67.9% → 69.6%) while degrading other metrics — the task is not reasoning-bound. **Bigger ≠ better either**: the two heavier candidates score worse, not better.

A judge that over-rates inflates the rubric headline metric (`weighted_met`) — the exact direction we don't want for a deployment story. We've exhausted the obvious open-source levers (3 models, 2 reasoning settings on the best one) and the bias persists across all of them. **The final evaluation requires a closed-source judge.** We start with the cheapest plausible option — `gpt-5-mini` — and use Phase A to test whether it materially beats the open-weight baseline on the not-met-agreement metric. If yes, Phase B runs at gpt-5-mini. If no, we escalate (see "Escalation options").

---

## Phase A — Calibration (`gpt-5-mini` vs physician ground truth)

### Task
Run `gpt-5-mini` over the OBGYN-scoped HealthBench grader meta-evaluation set (`mamabench@v0.2.1/calibration/obgyn_meta_eval.jsonl`) — **6,853 physician-labeled `(prompt, completion, criterion)` triples**. Compute a 4-metric report: LLM↔single-human agreement, LLM↔consensus, judge met-rate vs physician met-rate, and per-class agreement (rubber-stamp detector).

### Why necessary
The Background section establishes that open-weight judges over-rate "met" on physician-not-met rows. Before committing the Phase B spend, Phase A validates that `gpt-5-mini` materially improves on this metric — by running it over the same 6,853 physician-labeled triples and producing a side-by-side report against the open-weight bake-off. If gpt-5-mini doesn't pass (i.e., not-met agreement isn't materially above 67.9%), the result still has value: it tells us closed-source mini-tier is insufficient and we'd escalate to a higher tier. Phase A is the empirical gate that decides which model goes into Phase B.

### Cost calculation
Pricing: `gpt-5-mini` Batch = $0.125 / 1M input · $1.00 / 1M output (50% off the standard $0.25 / $2.00).

Token volume (measured from actual calibration data):

| | per call | × 6,853 calls |
|---|---:|---:|
| Input (grader template + conversation + completion + criterion) | ~1,361 | **9.33 M** |
| Output at `reasoning_effort=medium` | ~700 | **4.80 M** |

| Component | Batch cost |
|---|---:|
| Input: 9.33 M × $0.125/M | $1.17 |
| Output: 4.80 M × $1.00/M | $4.80 |
| **Estimated total** | **$6** |
| **Ask** (rounded up for safety margin on a small dollar amount) | **$15** |

---

<div style="page-break-after: always;"></div> 

## Phase B — Production Rescore (HealthBench rubric, ±RAG)

### Task
For each (model response, criterion) pair across the 3 HealthBench configs and both ±RAG arms, ask the pinned judge whether the criterion is met. The HealthBench `weighted_met` headline metric is computed per response from these binary verdicts.

### Why necessary
This **is** the published HealthBench evaluation methodology. Without it, the v0.2 report cannot:
- Quantify Gemma's clinical-rubric quality (the deployment-relevant metric).
- Compare ±RAG on the rubric track (key for the RAG deployment claim).
- Compare against the published HealthBench leaderboard.

A closed-source judge is required because the open-weight bake-off (see Background) showed every open candidate biases the headline metric upward by over-rating "met." The judge tier validated in Phase A is the one that goes into production here. Starting tier is `gpt-5-mini`; the cost calc below uses gpt-5-mini batch pricing. If Phase A flags gpt-5-mini as insufficient, Phase B's cost shifts to the chosen escalation tier (see "Escalation options" for tier-specific Phase B totals).

### Cost calculation

Call count (one judge call per criterion, matching HealthBench's published methodology):

| Config | rows × avg criteria/row | criterion calls / arm |
|---|---:|---:|
| healthbench_oss_eval | 1,209 × 11.7 | 14,195 |
| healthbench_consensus | 872 × 2.2 | 1,909 |
| healthbench_hard | 258 × 11.8 | 3,050 |
| Per arm | | 19,154 |
| **Total × 2 arms (±RAG)** | | **38,308** |

Token volume:

| | per call | × 38,308 calls |
|---|---:|---:|
| Input | ~1,361 | **52.14 M** |
| Output at `reasoning_effort=medium` | ~700 | **26.82 M** |

| Component | Batch cost (`gpt-5-mini`) |
|---|---:|
| Input: 52.14 M × $0.125/M | $6.52 |
| Output: 26.82 M × $1.00/M | $26.82 |
| **Estimated total** | **$33** |
| **Ask** (with ~50% buffer for output-token variance) | **$50** |

---

## Forward-looking estimate — future retrieval-iteration runs

The asks above cover **one full v0.2 evaluation of the current MAM-AI app**. Over the coming months, planned retrieval-module improvements will trigger re-runs of the **+RAG arm only** — the no-RAG baseline doesn't change because it doesn't use retrieval.

Each retrieval improvement re-runs roughly *half* of Phase B (one arm of 19,154 criterion calls instead of both arms' 38,308), plus a small SAQ re-run. Cost at the starting `gpt-5-mini` tier:

| Per improvement | Calls | Batch cost |
|---|---:|---:|
| HealthBench rubric, +RAG arm only | 19,154 | ~$17 |
| SAQ, +RAG arm only | 369 | ~$1 |
| **Per-improvement total** | | **~$18** |

Anticipating **2 retrieval-module improvements** over the coming months:

| | Cost |
|---|---:|
| Per improvement | ~$18 |
| × 2 improvements | ~$36 |
| **Forward-looking estimate (with buffer)** | **~$50** |

This is not part of the current ask — each improvement would be a separately approved follow-up. Visibility into the full retrieval-iteration roadmap at gpt-5-mini: ~$65 for the current v0.2 evaluation, then ~$25 per retrieval improvement. (If we escalate to a higher tier after Phase A, these forward-looking numbers scale up proportionally — see "Escalation options" for per-tier project totals.)

---

## Escalation options if `gpt-5-mini` calibration falls short

If Phase A shows gpt-5-mini doesn't materially beat the open-weight baseline (67.9% not-met agreement), we escalate to a higher-tier closed-source judge. The table below is the reference for the new ask we'd file in that case — Phase A is re-run at the escalation tier (~$15–$120 depending on tier), and Phase B runs at the same tier.

Token volumes are fixed regardless of vendor or tier (the workload is the same); only price-per-token differs:

| Phase | Calls | Input tokens | Output tokens (at `reasoning_effort=medium`) |
|---|---:|---:|---:|
| A — Calibration (one-time) | 6,853 | 9.33 M | 4.80 M |
| B — Production rescore (one-time, both ±RAG arms) | 38,308 | 52.14 M | 26.82 M |
| Per future retrieval iteration (+RAG arm only, HealthBench + SAQ) | 19,523 | 27.55 M | 13.78 M |

### Per-candidate cost — pricing + project total (1× Phase A + 1× Phase B + 2× future iterations)

Grouped by vendor (OpenAI → Anthropic → Google). Prices per 1 M tokens. **All three vendors offer a Batch API at 50% off standard — the prices shown are those batch rates.**

| Vendor / Tier | Input $/M | Output $/M | **Project total** | **+~12% buffer** |
|---|---:|---:|---:|---:|
| **OpenAI gpt-5-mini** *(starting point)* | $0.125 | $1.00 | **$74** | **~$85** |
| **OpenAI gpt-5.4-mini** | $0.375 | $2.25 | **$177** | **~$200** |
| **OpenAI gpt-5** | $0.625 | $5.00 | **$369** | **~$420** |
| **OpenAI gpt-5.4** | $1.25 | $7.50 | **$590** | **~$660** |
| **OpenAI gpt-5.5** | $2.50 | $15.00 | **$1,181** | **~$1,320** |
| **Claude Opus 4.7** | $2.50 | $12.50 | **$1,031** | **~$1,160** |
| **Claude Sonnet 4.6** | $1.50 | $7.50 | **$618** | **~$700** |
| **Gemini 3.1 Pro Preview** | $1.00 | $6.00 | **$471** | **~$530** |

### Escalation recommendation

If gpt-5-mini's Phase A result is **insufficient**, the natural escalation order is:

1. **OpenAI gpt-5.4-mini** (~$200 full project) — next step up in the mini tier; double the cost, materially more capable. Best first escalation if mini failed by a small margin.
2. **OpenAI gpt-5** (~$420 full project) — older base flagship, still recognizable as a "frontier OpenAI" name. Best if 5.4-mini is also insufficient and you want to stay in the OpenAI lineage.
3. **OpenAI gpt-5.5** (~$1,320) or **Claude Opus 4.7** (~$1,160) — frontier-tier; pick if all the mid-tier options fail the not-met threshold.
4. **Gemini 3.1 Pro Preview** as a $0-out-of-pocket alternative — frontier-class Google model; new Google Cloud accounts get $300 in free credits each on sign-up, so stacking 2 fresh accounts covers the ~$530 project cost entirely. Tradeoff is logistical (multiple new accounts with separate credit cards), not financial.

In all cases, each escalation would be a separate ask filed after Phase A produces the data justifying it.
