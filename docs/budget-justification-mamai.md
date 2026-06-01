# LLM Judge API Budget — MAM-AI v0.2 HealthBench

Two budget items for the HealthBench rubric track of the v0.2 evaluation, both via the closed-source Batch API. *(Separate smaller asks of $30 each cover the SAQ rescore and the faithfulness judge.)*

| Phase | Purpose | Estimated batch cost | **Ask** |
|---|---|---:|---:|
| **A. Calibration** | Validate `gpt-5.5` as LLM judge against physician ground truth | $96 | **$120** |
| **B. Production rescore** | Score 38,308 model responses on the HealthBench rubric track | $533 | **$600** |

**Current ask total: $720.** A further ~$600 covers 2 anticipated retrieval-module iterations (see Forward-looking section) — filed separately when each triggers, not part of this approval. Full project envelope across v0.2 + 2 iterations: **~$1,320**.

Phase A validates `gpt-5.5` as the judge against physician ground truth before Phase B uses it on production data. Both phases are required: the open-source bake-off (see Background) showed that open-weight judges over-rate "met" on a meaningful fraction of physician-not-met rows — biasing the deployment-headline rubric metric in the wrong direction — so a frontier closed-source judge is necessary for the final v0.2 evaluation.

---

## Background

MAM-AI v0.2's open-ended evaluation tracks use an LLM judge to score Gemma 4 E4B responses against physician-written rubric criteria. We **already** ran an open-source bake-off across three candidates (Llama 4 Maverick FP8, Nemotron-Ultra-253B FP8-quantized, gpt-oss-120b) on a 6,853-row physician-labeled meta-evaluation set (`mamabench@v0.2.1/calibration/obgyn_meta_eval.jsonl`). Result: **all three open candidates systematically over-rate "met"** on the rows where physicians unanimously agreed the criterion was **NOT** met.

| Judge (open-weight) | Footprint | Judge met-rate vs phys 77% | Not-met agreement | Over-rated rate on not-met rows |
|---|---|---:|---:|---:|
| Llama 4 Maverick FP8 | 4× H200 | 88.8% (+11.8 pp) | 44.6% | 55.4% |
| Nemotron-Ultra-253B (FP8 q.) | 4× H200 | 83.3% (+6.3 pp) | 55.3% | 44.7% |
| **gpt-oss-120b (medium, pinned)** | **1× H200** | **73.2% (−3.9 pp)** | **67.9%** | **32.1%** |

The pinned candidate (gpt-oss-120b medium) is the **best** of the three. We also confirmed higher reasoning effort doesn't fix it: gpt-oss-120b at `reasoning_effort=high` improved not-met agreement by only 1.7 pp (67.9% → 69.6%) while degrading other metrics — the task is not reasoning-bound. **Bigger ≠ better either**: the two heavier candidates score worse, not better.

A judge that over-rates inflates the rubric headline metric (`weighted_met`) — the exact direction we don't want for a deployment story. We've exhausted the obvious open-source levers (3 models, 2 reasoning settings on the best one) and the bias persists across all of them. **The final evaluation requires a frontier closed-source judge**: we need to calibrate `gpt-5.5` against the same physician-labeled set (Phase A), confirm it materially improves the not-met agreement, and then run it over the production rubric track (Phase B).

---

## Phase A — Calibration (`gpt-5.5` vs physician ground truth)

### Task
Run `gpt-5.5` over the OBGYN-scoped HealthBench grader meta-evaluation set (`mamabench@v0.2.1/calibration/obgyn_meta_eval.jsonl`) — **6,853 physician-labeled `(prompt, completion, criterion)` triples**. Compute a 4-metric report: LLM↔single-human agreement, LLM↔consensus, judge met-rate vs physician met-rate, and per-class agreement (rubber-stamp detector).

### Why necessary
The Background section establishes that open-weight judges over-rate "met" on physician-not-met rows. Before committing the Phase B spend, Phase A validates that `gpt-5.5` is the upgrade we expect — by running it over the same 6,853 physician-labeled triples and producing a side-by-side metric report against the open-weight bake-off. The output is a precondition for trusting any production Phase B verdicts.

### Cost calculation
Pricing: `gpt-5.5` Batch = $2.50 / 1M input · $15 / 1M output (50% off the standard $5 / $30).

Token volume (measured from actual calibration data):

| | per call | × 6,853 calls |
|---|---:|---:|
| Input (grader template + conversation + completion + criterion) | ~1,361 | **9.33 M** |
| Output at `reasoning_effort=medium` | ~700 | **4.80 M** |

| Component | Batch cost |
|---|---:|
| Input: 9.33 M × $2.50/M | $23.33 |
| Output: 4.80 M × $15/M | $72.00 |
| **Estimated total** | **$96** |
| **Ask** (with ~25% buffer for output-token overruns) | **$120** |

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

A frontier closed-source judge is required because the open-weight bake-off (see Background) showed every open candidate biases the headline metric upward by over-rating "met." The validated `gpt-5.5` judge from Phase A is the one that goes into production here.

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

| Component | Batch cost |
|---|---:|
| Input: 52.14 M × $2.50/M | $130.35 |
| Output: 26.82 M × $15/M | $402.30 |
| **Estimated total** | **$533** |
| **Ask** (with ~13% buffer for output-token variance) | **$600** |

---

## Forward-looking estimate — future retrieval-iteration runs

The asks above cover **one full v0.2 evaluation of the current MAM-AI app**. Over the coming months, planned retrieval-module improvements will trigger re-runs of the **+RAG arm only** — the no-RAG baseline doesn't change because it doesn't use retrieval.

Each retrieval improvement re-runs roughly *half* of Phase B (one arm of 19,154 criterion calls instead of both arms' 38,308), plus a small SAQ re-run:

| Per improvement | Calls | Batch cost |
|---|---:|---:|
| HealthBench rubric, +RAG arm only | 19,154 | ~$267 |
| SAQ, +RAG arm only | 369 | ~$10 |
| **Per-improvement total** | | **~$277** |

Anticipating **2 retrieval-module improvements** over the coming months:

| | Cost |
|---|---:|
| Per improvement | ~$277 |
| × 2 improvements | ~$554 |
| **Forward-looking estimate (with ~10% buffer)** | **~$600** |

This is not part of the current ask — each improvement would be a separately approved follow-up. It is provided so the supervisor has visibility into the full retrieval-iteration roadmap: ~$720 for the current v0.2 evaluation, then ~$300 per retrieval improvement.

---

## Total budget across closed-source judge candidates

Token volumes are fixed regardless of vendor or tier (the workload is the same); only price-per-token differs. **Reference token volumes** for all calculations below:

| Phase | Calls | Input tokens | Output tokens (at `reasoning_effort=medium`) |
|---|---:|---:|---:|
| A — Calibration (one-time) | 6,853 | 9.33 M | 4.80 M |
| B — Production rescore (one-time, both ±RAG arms) | 38,308 | 52.14 M | 26.82 M |
| Per future retrieval iteration (+RAG arm only, HealthBench + SAQ) | 19,523 | 27.55 M | 13.78 M |

### Per-candidate cost — pricing + project total (1× Phase A + 1× Phase B + 2× future iterations)

Grouped by vendor (OpenAI → Anthropic → Google). Prices per 1 M tokens. **All three vendors offer a Batch API at 50% off standard — the prices shown are those batch rates.**

| Vendor / Tier | Input $/M | Output $/M | **Project total** | **+~12% buffer** |
|---|---:|---:|---:|---:|
| **OpenAI gpt-5.5** | $2.50 | $15.00 | **$1,181** | **~$1,320** |
| **OpenAI gpt-5.4** | $1.25 | $7.50 | **$590** | **~$660** |
| **OpenAI gpt-5** | $0.625 | $5.00 | **$369** | **~$420** |
| **Claude Opus 4.7** | $2.50 | $12.50 | **$1,031** | **~$1,160** |
| **Claude Sonnet 4.6** | $1.50 | $7.50 | **$618** | **~$700** |
| **Gemini 3.1 Pro Preview** | $1.00 | $6.00 | **$471** | **~$530** |

### Recommendation

Three paths, depending on how aggressively cost is a binding constraint.

**Path 1 — No compromise on capability.** Pick either:
- **OpenAI gpt-5.5** (~$1,320 with buffer) — OpenAI's top-tier reasoning model.
- **Claude Opus 4.7** (~$1,160 with buffer) — Anthropic's frontier. Comparable capability, slightly cheaper.

The asks in earlier sections of this doc ($120 + $600 + $600) anchor on gpt-5.5.

**Path 2 — Keep OpenAI recognition, keep cost low.** Pick **OpenAI gpt-5** (~$420 with buffer). Still a recognized OpenAI flagship — the methods section reads "OpenAI's reasoning model" — at roughly **a third** the cost of gpt-5.5. Tradeoff: weaker reasoning capacity than 5.5/5.4, so the "frontier" defense in the methods section is softer.

**Path 3 — Save every penny.** Pick **Gemini 3.1 Pro Preview**. Frontier-class capability from Google. API charges would be ~$530 with buffer, but **new Google Cloud accounts receive $300 free credits each on sign-up** (credit-card-linked) — stacking 2 fresh accounts covers the full spend, **net $0 out-of-pocket**. Tradeoff is logistical, not financial: requires onboarding multiple fresh Google Cloud accounts with separate credit cards.

The mid-tier options (gpt-5.4, Sonnet 4.6) sit between Paths 1 and 2 and can be considered if none of the three fit.
