# R2c embedder bake-off — execution plan (deployability-first)

*Filed 2026-06-16. Supersedes the execution ordering in
[`r2c-embedder-bakeoff-next-steps-20260615.md`](r2c-embedder-bakeoff-next-steps-20260615.md)
after the P-1 coverage result, the candidate lit-review, and the architecture decisions
below. Companion artifacts:*
- *Candidate shortlist + feasibility: [`r2c-embedder-litreview-20260616.html`](r2c-embedder-litreview-20260616.html)*
- *P-1 corpus-coverage verdict: `kenya-coverage-summary-20260615.md` (r1-threshold worktree)*

---

## Decisions locked (do not relitigate)

| Decision | Value | Rationale |
|---|---|---|
| **Runtime** | **LiteRT-only**, single runtime, simple architecture | Generator is `.litertlm` (LiteRT-bound); a 2nd runtime is permanent app-size/RAM/maintenance tax for no gain on the primary candidate. |
| **Target hardware** | **Low-/mid-end Android devices (Zanzibar)** | Flagship-NPU benchmarks do **not** apply; latency/memory/storage on weak hardware is the binding constraint. |
| **Language** | **English-primary**; multilingual = *useful bonus*, not central | Restores the full 3-candidate shortlist (no cross-lingual requirement); multilingual is a tiebreaker favoring EmbeddingGemma. |
| **Acceptance margin** | **≥5pp over the deployable hybrid baseline on kenya end-to-end recall, no penalty/refusal regression** | Pre-registered before results. 5pp = the `1/√n` noise floor at n=312. Bootstrap only for a borderline tie-break. |
| **Primary metric** | **Recall** (recall@20 / strict pool-recall), precision (P@3) kept secondary | An embedder's unique lever vs the already-tried reranker is *surfacing missed chunks*, not reordering. |
| **Deployability** | **Tested first, for ALL candidates, before any deep evaluation** | Two of three candidates are "assumed deployable," not confirmed; the binding low-mid budget is unmeasured by any published benchmark. |

## P-1 coverage result that shapes the test (already done)

**Proceed with R2c, but bounded.** Kill-rule not triggered (42% not-strict-covered < 60%).
But the addressable upside is **~1 in 5 kenya queries**: 18% (lenient) / 22% (strict) are
"ranking-fixable" (a relevant chunk is in some arm's top-3 but Gecko misses it); 30–42%
have nothing relevant in any arm's top-3 (corpus/depth-limited — a parallel R-corpus
problem). **Design consequence:** the realistic whole-set end-to-end lift is small
(~2–4pp, at/under the n=312 noise floor), so the test must **lead with targeted retrieval
metrics on the addressable set** and treat a whole-set end-to-end null as *expected-if-small*,
not as disproof.

## Candidates

| # | Model | Role | Deployability prior | Notes |
|---|---|---|---|---|
| 1 | **EmbeddingGemma-300M** | Primary | Conversion ✓ (official LiteRT); **low-mid latency/RAM ✗ unconfirmed** | ~180–220 MB, mixed int4/int8 QAT; multilingual bonus; test 256-dim build. |
| 2 | **MedEmbed-small / bge-small** | Simple + footprint | Conversion ✓ (community-verified); fit ✓ near-certain | ~32 MB, 33M; int8 quality is the risk; cleanest fine-tune base (R2d). |
| 3 | **MedCPT** (query encoder) | Domain challenger | **Conversion ✗ unproven**; most complex (dual-encoder) | Strongest in-domain IR evidence; must clearly win to justify complexity. |

Baselines (always in the comparison): **Gecko** (deployed, English-only ~110M, int8 ~112–146 MB) and **Gecko+BM25 hybrid**.
Reference ceiling (measured once, not deployed): **Qwen3-Embedding-8B / voyage** — gap calibration only.

---

## Execution phases

### Phase 0 — On-device deployability gate (ALL candidates) — **FIRST, hard gate**
No candidate enters the offline screen until it clears this. Lightweight spike, not full
deployment. Two axes, both must pass:

- **(a) Converts + correct vectors:** produce a LiteRT int8 model (use the ready artifact
  for EmbeddingGemma; convert bge-small/MedCPT via `ai-edge-torch`); load it; embed a
  sample; **validate vectors vs PyTorch (`numpy.allclose`, atol≈1e-5)** to catch the
  known silent-garbage conversion bugs.
- **(b) Fits the low-mid budget:** on a **representative low-mid device**, measure
  **per-query embed latency, peak RAM (alongside the resident Gemma generator), and
  on-disk size** (model + projected vector store). Judge against the budget (below).

Per-candidate focus:
- **EmbeddingGemma** — artifact ready; the real test is **latency/RAM on low-mid CPU/GPU**
  (no flagship NPU) at the **256-dim / short-seq** build, and whether the int4/int8 ops
  run (not fall back) on weak hardware.
- **bge-small / MedEmbed-small** — fit is near-certain; confirm **int8 ranking quality**
  doesn't collapse on a 33M model (fall back to int16-activation/fp16 if cosine geometry breaks).
- **MedCPT** — **prove the conversion at all** (no precedent); ship the query encoder only
  (corpus embedded offline with the article encoder). Highest risk; spike before investing.

**Gate output:** a deployability table (pass/fail + latency/RAM/size per candidate).
Only passers proceed. A candidate that converts but blows the budget is **out for Zanzibar**.

> **Blocking inputs needed to run Phase 0** (turn the spike from descriptive → pass/fail):
> 1. A **representative low-mid device** (or target spec: SoC + RAM).
> 2. **Budget thresholds:** acceptable per-query embed latency, RAM headroom beside the
>    Gemma generator, and total storage for embedder + vector store.

### Phase 1 — Sharpen the target (cheap; candidate-independent, may run alongside Phase 0)
Re-run kenya coverage at **top-20** (`precompute_arms_matrix.py --top-k 20`,
r1-threshold worktree). Splits "not in corpus" from "buried below rank 3." Produces:
- the **true addressable size** (sizes the maximum achievable effect → sanity-checks the 5pp margin);
- the **addressable set** = explicit kenya queries where a relevant chunk exists in some
  pool but Gecko misses it. **This set is the high-powered evaluation target downstream.**

### Phase 2 — Offline retrieval screen (cluster, full precision; Phase-0 survivors only)
Harness fairness first: per-candidate **query/doc prefixes**; evaluate EmbeddingGemma at
**both 256 and 768 dim** (quality-vs-footprint for low-mid); **judge-provenance** check
(Qwen3-32B top-up vs the original 230k-label judge). Then per candidate: re-embed corpus +
queries, retrieve top-20, score —

| Metric | Role |
|---|---|
| **Recall@20 / strict pool-recall (≥5)** | the embedder's unique lever |
| **Lift on the addressable set (Phase 1)** | highest-powered, most decision-relevant signal |
| P@3 lenient/strict, HR@3 | secondary precision view |
| Stage-1 gate (chunk AUC) | threshold-revival check |

Evaluate on **kenya-relevance labels (weighted — predictive)** and the mamaretrieval frozen
split (large-n, descriptive only — it's non-predictive for deployment). Run **one reference
ceiling** on mamaretrieval for gap calibration. **Gate to advance:** beat Gecko on recall,
especially on the addressable set.

### Phase 3 — End-to-end value gate (cluster, Gemma 4 E4B; Phase-2 passers only)
Precompute arms (candidate / +hybrid) for **kenya SAQ + healthbench-oss**; generate; judge
SAQ key-fact recall (gpt-oss-120b) + healthbench rubric (+/− split). **Two-tier acceptance:**
- **Primary (powered):** end-to-end recall lift **on the addressable subset** + the retrieval
  recall gain — where a real effect is detectable.
- **Secondary (whole-set):** the pre-registered **≥5pp over hybrid** on full kenya recall, no
  penalty/refusal regression. Given the ~1/5 ceiling, a whole-set null is *expected-if-small*,
  **not** disproof; the addressable-subset + recall results decide. healthbench-oss
  (coverage-good, large-n) is the better-powered end-to-end check.

### Phase 4 — Fine-tune / distill (R2d) — only if needed
Contrastively fine-tune the chosen base (EmbeddingGemma and bge-small both have recipes)
on the 230k graded pairs and/or distill ceiling rankings. Use the frozen `split.json`
(no test-query leakage). Only after Phase 2/3 pick a base and only if it still falls short.

### Phase 5 — Land (LiteRT int8 on device)
Device port + **low-mid on-device parity** (mirror `check_parity.py`), re-version cached
embeddings / thresholds / contradiction audit (a new embedder invalidates them), PR.
Gated on Phase 3 passing.

---

## Decision rule (Phase 4 / final)
Combine **deployability on low-mid (Phase 0)** × **retrieval recall gain (Phase 2)** ×
**addressable-set end-to-end lift (Phase 3)**, with **simplicity + footprint** as the
tiebreak. Outcomes: pick a winner → Phase 4/5, or "even the best deployable embedder doesn't
capture the addressable slice" → the gain is structural → weight **R-corpus** (content expansion).

## Reusable harness (already built)
- `retrieval_eval/precompute_arms_matrix.py` — N-arm contexts (extend for the embedder dim)
- `retrieval_eval/score_pool_matrix.py` — offline P@3 + Stage-1 gate per pool
- `retrieval_eval/judge_kenya_relevance.py` + `_kenya_rubric.py` — Qwen3-32B V2-rubric judging
- `retrieval_eval/compare_retriever_gates.py` — Stage-1 score-quality gate
- `retrieval_eval/check_parity.py` — device-vs-host fidelity (mirror for Phase 0/5)
- `cluster/run_cluster_{arms_matrix,score_matrix,value_gate,value_gate_judge,kenya_relevance}.sh`
- `end_to_end_eval/{run_eval,rescore_open_v2,rescore_rubric}.py` — generate + judge
- Frozen split: `configs/config-v0.2.0/results/retrieval_eval/r2c-rerank/split.json`

## Open risks (empirical only — not resolvable from the literature)
- int8 LiteRT embedding quality is under-documented; small (33M) models degrade most.
- MedCPT domain transfer (PubMed literature ≠ guideline prose) + unproven LiteRT conversion.
- EmbeddingGemma footprint (~200 MB) vs the small models (~32 MB) may decide it on low-mid RAM.
- Cross-benchmark scores are not comparable; the only clean gap comes from the in-house screen.
- The addressable slice is ~1/5 → end-to-end effects are small; rely on targeted metrics.

## Status
- ✅ P-1 coverage gate — proceed (bounded)
- ✅ Candidate lit-review — shortlist of 3 + feasibility priors
- ✅ Architecture decided — LiteRT-only, English-primary, low-mid target
- ✅ Acceptance criterion + metric set — pre-registered
- ▶ **NEXT: Phase 0 on-device deployability gate (all candidates)** — pending target device/spec + budget thresholds
- ⏳ Phase 1 top-20 coverage refinement (can run alongside Phase 0)
