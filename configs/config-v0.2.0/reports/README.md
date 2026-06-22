# config-v0.2.0 — Evaluation report index

The full v0.2 evaluation of the MAM-AI RAG system (on-device Gemma 4 E4B + Gecko retrieval
over the maternal/neonatal guideline corpus) rests on **three pillars**:

1. **Retrieval quality** — does the retriever surface the right guideline chunks?
2. **Generator faithfulness** — does Gemma stay faithful to the retrieved chunks?
3. **End-to-end quality** — does the combined RAG system answer well (MCQ / open-ended SAQ / HealthBench rubric)?

This README says where each report lives, what it contains, and why it matters.

---

## 1. Retrieval quality

> **Lives outside this repo**, in the benchmark project [`nmrenyi/mamaretrieval`](https://github.com/nmrenyi/mamaretrieval)
> (released dataset: HF [`nmrenyi/mamaretrieval`](https://huggingface.co/datasets/nmrenyi/mamaretrieval)).
> The benchmark is 3,185 clinical queries over the 63,650-chunk `rag-bundle-v0.2.0` corpus, with
> LLM-graded relevance labels (Qwen3.5-397B judge, v2 graded rubric: score = d1 × (d2 + d3 + d4) ∈ 0–6,
> validated 95%/85% agreement vs Claude Opus 4.7 reference labels at the lenient/strict thresholds).

| Report | What it is |
|---|---|
| `mamaretrieval/AUDIT_REPORT_v2.md` | **Headline retrieval evaluation.** 6 retrievers (incl. deployed Gecko) at deployment depth k=3, Tier 1 pilot (100 queries) + Tier 2 full (3,185 queries × 36,418 pairs). |
| `mamaretrieval/data/audit/results_v2_full.md` | Auto-generated numerical source of truth for the Tier 2 scoreboard. |
| `mamaretrieval/data/audit/results_v2_tier3.md` | Tier 3: HR/Precision at k ∈ {3,5,10,20} + pool recall, from full top-20 union judging (~231k pairs). |
| `mamaretrieval/AUDIT_REPORT.md` | Phase 3 completeness audit (v1 rubric): the benchmark's candidate pool captures ~half of truly-relevant chunks — the error bar on absolute recall/precision numbers. |

**Headline (Tier 2, k=3, deployed retriever = Gecko):**
Gecko sits in the **middle tier** — HR(≥3) 0.814 / P(≥3) 0.477 lenient; HR(≥5) 0.439 / P(≥5) 0.193 strict.
Clearly above BM25/MedCPT, but roughly **half the precision of the top tier** (voyage 0.996/0.867,
octen 0.991/0.804). In ~19% of queries the deployed top-3 bundle contains *no* clinically useful chunk,
and in ~56% it contains no top-tier (complete/specific) chunk.

**Why it matters for the improvement plan:** this is the structural ceiling on RAG usefulness.
Upgrading or augmenting the on-device retriever (or increasing k) is a candidate lever that is
independent of the generator's behavioral problems.

---

## 2. Generator faithfulness

> Does Gemma 4 E4B, given known-relevant ("oracle") context, stay faithful to it?
> Scored by Patronus Lynx 70B, then calibrated against a stronger judge on a blinded sample.

| Report | What it is |
|---|---|
| `oracle-v0.2.0-faithfulness.html` (this dir) | **Headline faithfulness result** on the v0.2.0 oracle (top-20 union, score ≥ 5; 2,989 queries). Lynx raw PASS 94.2%; gpt-5-calibrated true-hallucination estimate ~9%. |
| `g1-g2-faithfulness-3n-20260619.html` (this dir) | **Gemma 3n re-run** of the oracle faithfulness pipeline across the G1/G2 prompt arms (n=2,989; Lynx → gpt-5 categorize+calibrate). Calibrated true-hallucination: **3n baseline 27% (vs Gemma 4's 9% — 3n ~3× less faithful to context), +G1 46%, +G1+G2 47%.** Categorized rate stayed flat (~6.3–6.7%); only the calibration's Lynx-miss correction (26→46%) revealed G1's effect. Faithfulness gate **fails for G1/G2 on 3n**; deeper signal is that 3n's grounding is the bottleneck. |
| `faithfulness-g4-vs-3n-prompts-20260622.html` (this dir) | **Head-to-head G4 vs 3n × 3 prompts (identical pipeline).** Completes the matrix by running Gemma 4 through the same oracle faithfulness arms. On the defensible **categorized** rate, **G4 is ~2× more faithful than 3n at every arm** (G4 2.6/3.3/3.6% vs 3n 6.3/6.5/6.7%) — the generator gap dwarfs the prompt effect (~4×). Both prompts trade refusals for unsupported additions (the faithfulness cost of the deflection fix), but **G4 absorbs it; 3n was already the weaker grounder.** Shows the calibrated metric is the noisy column (same G4 data scored 9%→19% calibrated across two runs while categorized held at 2.5–2.6%) and that "+G1 doubles 3n hallucination" is largely a Lynx-miss-rate/refusal artifact — the mild categorized reading is the trustworthy one. Reframes the lever as **RAG-grounding, not prompt tuning.** |
| `oracle-self-contradictions-v0.2.0.html` (this dir) | Audit of contradiction-bucket FAILs: 16 are corpus-quality false-FAILs (conflicting facts *inside* the oracle context, mostly cross-guideline retrieval). |
| `../../../docs/faithfulness-eval-v0.2.0.md` | Methodology + v0.1.0-oracle results (2,659 queries; Lynx 94.55% PASS; Claude-calibrated true-hallucination ~0.3%). Explains the verification ladder and why raw Lynx FAIL ≠ hallucination. |
| `../../../docs/oracle-self-contradictions-v0.1.0.md` | The original 6-case corpus-conflict audit (prose-vs-table dosing conflicts, etc.). |
| `../../../docs/faithfulness-eval-next-steps.md` | Pending follow-ups (stability checks, deployment integrity checks). |

**Headline:** when Gemma engages with the retrieved context it is largely faithful; a meaningful share
of flagged "hallucinations" trace to **contradictions within the guideline corpus itself**, which should
be escalated upstream (`mamai-medical-guidelines` / `mamaretrieval`) rather than fixed in the model.
Note the v0.1.0 (~0.3%) and v0.2.0 (~9%) calibrated estimates use different calibration judges and
different oracles — each stands on its own methodology; they are not a trend line.

---

## 3. End-to-end evaluation (full RAG system)

All run on cluster GPU with the deployed prompts/params of this config, ±RAG, judged by the
validated `gpt-oss-120b` judge (see "Judge trust" below). Raw result JSONs are under
`../results/end_to_end_eval/`.

### 3a. Multiple choice (MCQ) — indicator only

Exam-style questions (medmcqa, medqa_usmle, afrimedqa; 23,241 rows). Useful as a broad knowledge
indicator, **not** the deployment-relevant metric — it tests a different distribution than the app's
real use.

| Report | What it is |
|---|---|
| `mcq-rag-effect-20260520.md` | Full MCQ ±RAG. **53.7% no-RAG accuracy; RAG regresses −1.8 pp** (off-domain corpus distracts the model). MCQ headline should be quoted no-RAG. |
| `calibration-mcq-20260519.md` (+ `.csv`) | Device (LiteRT) vs cluster (GGUF Q4_0) on 300 shared rows: κ = 0.558, accuracies within ~3 pp → cluster runs stand in for the device. |

### 3b. Open-ended SAQ — real-world questions, treat carefully

Clinical vignettes answered free-form (kenya 312, afrimedqa_saq 37, whb 20; ±RAG), scored as
key-fact recall (present/partial/absent per pre-extracted fact) + 4-level safety rating.

| Report | What it is |
|---|---|
| `phase-b-saq-result-20260609.html` | **Headline SAQ result.** key_fact_recall **0.178** (kenya, no-RAG); ~34% of responses convey zero key facts, and ~91% of those are **scope-refusal or defer-to-doctor** — behavioral, not knowledge gaps. RAG *lowers* recall (more geographic over-refusal). Safety: **zero "dangerous"** ratings anywhere. Ends with prioritized, mostly prompt-only fixes. |

### 3c. Open-ended HealthBench rubric — treat carefully

HealthBench-style conversations scored against weighted per-response criteria, incl. negative
(penalty) criteria; 3 subsets (oss_eval 1,000 / consensus 1,000 / hard 339; ±RAG).

| Report | What it is |
|---|---|
| `phase-b-rubric-result-20260609.html` | **Headline rubric result.** weighted_met: oss_eval **−0.016**, consensus 0.512, hard −0.178 (no-RAG; report per-subset, never pooled). 61% of failures are **deflection**; active commission of wrong/harmful content ≈ 2%. RAG effect < 1 pp. Same diagnosis as SAQ: safe but unhelpful. |

### 3d. G1/G2 prompt A/B on Gemma 3n — execution of the improvement plan (2026-06-19)

Three-arm prompt A/B (arm1 baseline / arm2 +G1 deflection levers / arm3 +G1+G2 structure) on the
now-deployed **Gemma 3n E4B**, no-RAG. Data + provenance: `../results/end_to_end_eval/g1-ab-3n-20260619/`.

| Report | What it is |
|---|---|
| `g1-g2-prompt-ab-3n-20260619.html` (this dir) | **Headline G1/G2 result (SAQ + rubric).** kenya recall rises 0.288→0.365→0.403 (+40% rel) and deflection is eliminated — but **safety fails monotonically**: harm 22%→34%→35%, dangerous **4→7→9**. Crucially, **3n's baseline deflection is already ~2%** (the Gemma 4→3n upgrade already fixed deflection), so the prompt's net effect is a **recall-for-safety trade**, not a deflection repair. **Verdict: do not ship G1/G2 as-is on 3n.** Faithfulness leg in §2 (`g1-g2-faithfulness-3n-20260619.html`) agrees. |
| `generator-prompt-matrix-20260622.html` (this dir) | **Capstone: generator × prompt matrix (G4 + 3n, both tracks, +RAG).** Extends the 3n-only A/B into a 2×3 matrix and lands the key cross-track result: **the two generators split on helpfulness vs safety** — 3n more helpful (recall 0.30–0.39, ~2× HealthBench) but far less safe/faithful (dangerous **4/11/15** vs G4 **0/1/0**; oracle hallucination 6.3–6.7% vs 2.6–3.6%). **The same prompt has opposite value per generator:** G4 *needs* G1 (fixes its 33% deflection, doubles recall, stays safe) while 3n is *harmed* by it (already non-deflecting; prompts only buy recall by spending safety). **RAG did not rescue 3n** (dangerous 4/11/15 *worse* than no-RAG 4/7/9). Lever = generator RAG-grounding (G-RAG), not prompts/retrieval. *(Qwen-397B ceiling row pending.)* Data: `../results/end_to_end_eval/gen-prompt-matrix-20260622/`. |

---

## Judge trust (methodology backing for §3b/§3c)

The open-ended tracks are LLM-judged, so the judge itself was validated first:

| Report | What it is |
|---|---|
| `../../../docs/judge-validation-result-20260522.md` | Judge bake-off vs 6,853 physician-labeled judgments. **gpt-oss-120b pinned** — only candidate matching human inter-rater agreement (74.3% vs human-human 77.6%) without rubber-stamping "met" (Nemotron/Maverick collapse to 44–55% agreement on not-met rows). |
| `../../../docs/judge-validation-plan.md` | Meta-evaluation design and pass criteria. |
| `../../../docs/judge-validation-phase-a-result-20260608.html` | Phase A calibration of the faithfulness-calibration judge (gpt-5 tiers). |
| `../../../calibration/judge_validation/reports/<judge>/report.md` | Per-candidate raw validation reports (gpt-oss-120b, gpt-oss-120b-high, gpt-5, gpt-5-mini, nemotron-ultra-253b, llama4-maverick). |

---

## Improvement plan

| Report | What it is |
|---|---|
| `improvement-plan-20260611.html` | **Improvement advice for the next cycle**, synthesizing all three pillars. P0: retrieval relevance threshold (stop injecting noise), system-prompt revision (emergency override + deflection fixes, three-arm A/B incl. the G2 structure pilot), corpus expansion with high-quality sources. P1: coverage/completeness of engaged answers (G2), on-device retriever upgrade (hybrid / rerank / multilingual embedder bake-off), evaluation-protocol upgrades (deflection metric, rubric CIs, clinician calibration). P2: query rewriting, MCQ-track instrumentation, corpus-contradiction cleanup. Separates system improvements (§1–4) from evaluation-protocol improvements (§5); every item carries an acceptance gate. Offline simulations from existing audit labels (no new judging) validate the retrieval options before device work. |

### R1 / R2 — retrieval-threshold & retriever-upgrade investigation

Each workstream keeps its report(s), figures, and result JSONs in its own subfolder here.
**Start with the synthesis** — it ties the retriever and generator findings together.

| Report | What it is |
|---|---|
| `r2c-retriever-generator-synthesis-20260618.html` | **Capstone.** Synthesises the reranker + embedder + generator×retriever matrix + thresholdability into ranked recommendations: the **generator is the binding lever** (Gemma 4→3n ~2×), retrieval upgrades win offline but don't convert; adopt EmbeddingGemma (cheap, ~0 online today), don't ship the reranker, no retrieval-side abstention gate. |
| `r1-threshold/` | R1 — relevance-threshold negative result (Gecko cosine carries no usable threshold signal; `similarity_threshold` stays 0.0). |
| `r2-hybrid/` | R2a — Gecko+BM25 hybrid (RRF) offline result. |
| `r2-rerank/` | R2b — rerank-ceiling simulation. |
| `r2c-rerank/` | R2c reranker line — candidate bake-off, in-domain fine-tune (MiniLM-L6-ft / mxbai-ft), device parity, value gate, the **generator×retriever matrix**, and EG+reranker×3n. |
| `r2c-embedder/` | R2c embedder bake-off — EmbeddingGemma vs Gecko (P0 deployability → P2 retrieval → P1 corpus coverage → P3 value gate) + per-phase docs. |
| `r2c-threshold/` | R2c thresholdability — can EmbeddingGemma cosine (or a reranker score) drive an abstention gate? No (chunk AUC 0.69–0.72 < 0.80; answer-quality AUC ≈ 0.50). |

## One-paragraph synthesis

The v0.2 evidence converges: **the system is safe but unhelpful, and the bottlenecks are
(a) generator behavior and (b) retrieval headroom.** End-to-end, Gemma deflects or refuses instead
of giving first-line management (SAQ recall 0.178; rubric weighted_met ≈ 0 driven by deflection,
with only ~2% harmful commission and zero "dangerous" safety ratings). When it does engage, it is
largely faithful to retrieved context — many residual "hallucination" flags are corpus
self-contradictions. The deployed Gecko retriever is serviceable but mid-tier (lenient P@3 ≈ 0.48,
about half of the best available), and the current corpus actively hurts off-domain MCQ. The
highest-leverage improvements are therefore **prompt/behavior fixes** (both/and escalation framing,
authorizing first-line drugs with local-protocol caveats, fixing geographic over-refusal), followed
by **retriever upgrades** and **corpus-contradiction cleanup**.
