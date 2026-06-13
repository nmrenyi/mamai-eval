# R2c — On-device reranker: literature review

*Filed 2026-06-13. Feeds the R2c (embedder/reranker) step of
[`r2-retriever-upgrade-plan.md`](r2-retriever-upgrade-plan.md). Produced by a
deep-research pass (6 search angles, 25 sources fetched, 116 claims extracted,
25 adversarially verified — 0 refuted). Every model below is a
**candidate-to-test**, not a decision: our own offline scoring (Phase 1) and
on-device benchmark (Phase 2) make the call.*

---

## TL;DR

1. **The hard gate is unmet by everyone.** No surveyed reranker — small or
   large — has *any* verified LiteRT/TFLite or ONNX-Runtime-Mobile/ARM
   deployment with int8/int4 on a phone. The only concrete int8 reranker
   quantization recipes (Intel fastRAG; Sentence-Transformers ONNX/OpenVINO) are
   **x86-server-targeted** (require AVX512/VNNI, which ARM does not have) and
   verifiably do not transfer to Android. Convertibility is an unproven
   engineering gate we must settle empirically — and it is the single
   highest-priority thing to test.
2. **All latency numbers in the literature are server-GPU or x86-CPU** (MICE
   113 ms, ColBERT 130 ms, cross-encoder 470 ms; fastRAG 2–3× speedups). There
   is **zero on-device/ARM per-(query,chunk) latency evidence** for any
   candidate, so the "few seconds for 10–20 chunks" budget cannot be confirmed
   from sources — only measured.
3. **Swahili coverage is unverified for every candidate.** mMARCO excludes
   Swahili from training; jina and bge-base cards omit it; bge-v2-m3's
   multilingual claim is self-reported via its bge-m3 base with no isolated
   Swahili reranking number.
4. **Architecture:** at only 10–20 candidate chunks, a *small cross-encoder's*
   N-forward-pass cost may already fit the budget, which would make
   late-interaction's storage-for-speed tradeoff unnecessary at this scale.
5. **Fine-tuning: lean GO, as a signal not proof** — biomedical RAG benefits
   strongly from fine-tuning, but the dramatic published lift is the
   *retriever's*, not the reranker's.

---

## 0. Revised constraints (2026-06-13) — research-only, English-only

After the initial review, two hard constraints were lifted:

- **License: irrelevant.** The app is research-only, not commercial — so
  non-commercial licenses (e.g. jina's CC-BY-NC-4.0) no longer disqualify any
  model. Decisions are now purely technical.
- **Multilingual: not required for now.** English-only is acceptable for the
  reranker at this stage (Swahili/German return later). This is the bigger
  change: it puts the **English MS-MARCO cross-encoder family and MedCPT** —
  small, standard-architecture, best Android-conversion odds — back at the
  center, and demotes the large multilingual model from a deployment candidate
  to an offline reference.

The §1 table below was built under the *old* constraints; the operative
shortlist is the one in this section. Two unchanged constraints still bind and
dominate: **on-device Android convertibility** (LiteRT/TFLite or
ONNX-Runtime-Mobile, int8/int4, fits memory — still zero positive evidence for
any reranker) and the **few-second latency** for 10–20 chunks alongside Gemma.

### Architecture heuristic

Convertibility is unproven for every model, so we can't select on verified
Android support — the best *a priori* proxy is architecture. **Standard
BERT-family encoders (MiniLM, TinyBERT, BERT-base) have the most well-trodden
TFLite / ONNX-Runtime-Mobile path.** Prefer them over T5-based rerankers
(monoT5/RankT5 — encoder-decoder, harder to convert, larger) and over
late-interaction (ColBERT/MICE — storage cost not worth it at 10–20 chunks).

### Operative shortlist — two tiers

**Deployable candidates** (go on the phone; Phase 0 convertibility → Phase 2 device):

| # | Model | Params | Why Android-viable | Role |
|---|---|---|---|---|
| 1 | **cross-encoder/ms-marco-MiniLM-L6-v2** | ~22M | Smallest standard-BERT cross-encoder with strong MS-MARCO reranking; best convertibility + latency odds | **Primary small candidate** |
| 2 | **MedCPT Cross-Encoder** (NCBI) | ~109M (BERT-base) | Biomedical **zero-shot SOTA**, English, standard BERT → convertible; best domain match | **Primary domain candidate** |
| 3 | cross-encoder/ms-marco-MiniLM-L12-v2 | ~33M | Same architecture as #1, higher quality for ~1.5× size | Quality-leaning small option |
| 4 | cross-encoder/ms-marco-TinyBERT-L-2-v2 | ~tiny (single-digit M) | Fastest possible; the latency floor if L6 is too slow on-device | Latency fallback |
| 5 | **Feature-LTR** (GBDT over BM25 score, Gecko cosine, both rank positions, term overlap) | trivial | No transformer to convert — zero deployment risk | Safe fallback |

**Offline quality-reference** (never on the phone; Phase 1 scoring only):

| Model | Params | Role |
|---|---|---|
| **bge-reranker-v2-m3** | 568M | Strong-cross-encoder ceiling on our corpus — the reranker analogue of voyage in R1/R2. Tells us how much the small deployable models leave on the table. Re-enters the deployment conversation later when Swahili/German arrive (latency permitting). |

### The decision that matters: #1 vs #2

Tiny general (MiniLM-L6, web-domain, deployment-safest) versus mid-size
domain-matched (MedCPT, PubMed-trained, 5× larger, the only candidate with real
medical evidence). Phase 1 offline scoring on the hybrid top-20 pool answers
"does domain-matching beat tiny-and-fast for our OBGYN corpus?" directly. bge-base
(278M, English+Chinese) is now also technically eligible but an order of
magnitude larger than #1 — reach for it only if the small models underperform.

---

## 1. Candidate shortlist (original — pre-revision, kept for the full survey)

Verification: all rows below rest on `confidence: high`, 3-0 verifier votes,
against primary sources (HF model cards + arXiv) unless noted.

| Model | Family | Params | Multilingual / Swahili | License | On-device convertibility evidence | Benchmark / domain | On-device latency evidence |
|---|---|---|---|---|---|---|---|
| **mmarco-mMiniLMv2-L12-H384-v1** | cross-encoder (XLM-R distill) | ~117M | mMARCO 14 langs; **Swahili NOT in training**, no per-lang number | Apache-2.0 | **none** | mMARCO; no medical eval | **none** |
| cross-encoder/msmarco-MiniLM-L6-en-de-v1 | cross-encoder | ~0.1B | **EN-DE only** (fails Swahili; relevant to future German corpus) | Apache-2.0 | none | MS MARCO | none |
| **bge-reranker-v2-m3** | cross-encoder (bge-m3 / XLM-R) | 568M | BAAI's recommended multilingual; Swahili **self-reported, unconfirmed** | Apache-2.0 | none | strong multilingual; no isolated Swahili/medical number | none |
| bge-reranker-base | cross-encoder | ~278M | **EN+ZH only**; superseded for multilingual by v2-m3 | MIT | none | — | none |
| jina-reranker-v2-base-multilingual | cross-encoder (XLM-R) | 278M | 26-lang (MKQA); Swahili not listed | **CC-BY-NC-4.0 → DISQUALIFIED** (non-commercial) | none | MKQA/MLDR | none |
| MedCPT Cross-Encoder | cross-encoder | 220M | **English-only** (fails Swahili) | public (NCBI) | none | **biomedical zero-shot SOTA** (TREC-COVID, SciFact, BioASQ) | none |
| MICE | late-interaction / minimal-interaction (MiniLM-L12) | ~33M (26.3M precompute path) | no MIRACL/Swahili/medical eval | (code public) | none | ~97% in-domain nDCG@10; +5.8 BEIR OOD | GPU only (113 ms batch) |
| ColBERTv2 | late-interaction | — | — | none | reference late-interaction | GPU only (130 ms batch) |

**Reading the table:** the two genuinely viable *deployable* candidates by
size + license + multilingual intent are **mmarco-mMiniLMv2-L12-H384** (small,
the realistic on-device target) and **bge-reranker-v2-m3** (the multilingual
quality reference, but ~25× larger and likely too heavy for the phone).
Everything else is eliminated by a hard constraint: jina by license, bge-base
and MedCPT and the en-de model by language, MICE/ColBERT have no small-model
on-device story. **None has any convertibility or on-device-latency evidence at
all** — that column is empty for every row, which is the central finding.

---

## 2. Architecture recommendation

Three families, judged for reranking **10–20 chunks on ARM** (not server IR):

- **Cross-encoder (N forward passes).** Highest quality; the whole shortlist is
  cross-encoders. Cost scales linearly with candidate count — but at only 10–20
  chunks, a *small* (≤120M, int8) cross-encoder's N passes may already fit a
  few-second budget. This is the family to try first.
- **Late-interaction / minimal-interaction (ColBERT, MICE).** Trades inference
  compute for precomputed-chunk-embedding storage; MICE reports ~4× cross-encoder
  speedup matching ColBERT (113 vs 470 ms) and +5.8 BEIR OOD, at 1.8× ColBERT's
  memory. **But the speedup matters at large candidate counts; at 10–20 chunks
  the cross-encoder's N-pass cost is small, so the storage cost (token-level
  vectors per corpus chunk on the phone) is probably not worth it here.** Plus
  all its latencies are server-GPU.
- **Feature-based learning-to-rank** (GBDT/logistic over BM25 score, Gecko
  cosine, both rank positions, term overlap — signals we already compute). No
  external benchmark evidence surfaced, but it is the **cheapest and
  lowest-deployment-risk option by far** (no transformer to convert) and the
  natural safe fallback if the convertibility gate proves hard.

**Recommendation:** lead with a **small multilingual cross-encoder**
(mmarco-mMiniLMv2-L12 class); keep **feature-LTR** as the cheap fallback; treat
late-interaction as out-of-scope at this candidate count unless cross-encoder
conversion fails and we need ColBERT's precompute path for a different reason.

---

## 3. Fine-tune go/no-go

**Lean GO — but as a signal, not proof.** Biomedical RAG performance is
critically dependent on joint fine-tuning: a cited study lifted Recall@10 from
37.9% (zero-shot ModernBERT) to 92.8% (fine-tuned), and noted naive composition
can *degrade* results; a ColBERTv2 reranker added a further 92.8→93.8%. This
supports using our ~230k graded (query, chunk) pairs to fine-tune. Three
caveats temper it: (a) the dramatic lift is the **retriever's** number, not the
reranker's; (b) it rests on a single non-peer-reviewed preprint
(arXiv 2510.04757), reflected in the 2-1 verifier vote; (c) it is
biomedical-English and server-side, not multilingual/on-device. MedCPT shows
biomedical *zero-shot* transfer is real for English, but nothing demonstrates
zero-shot transfer to **Swahili clinical** text.

**Practical reading:** establish a strong **zero-shot** multilingual
cross-encoder baseline first; reserve the by-query train/dev/test split now (per
the R2 plan); fine-tune on the 230k pairs as a second step and measure the
**reranker-specific** delta — the published lift doesn't settle it for us.

---

## 4. Critical caveats (carry into any decision)

1. **Convertibility gate: zero positive evidence** for any model. Prove TFLite /
   ONNX-Runtime-Mobile conversion + ARM int8 execution empirically before
   trusting any shortlist entry.
2. **All latencies are server-GPU/x86** — the few-second on-device budget is
   unconfirmed; measure it.
3. **Swahili unverified for every candidate** — needs an isolated MIRACL-sw /
   Mr.TyDi-sw number, or our own Swahili eval.
4. **No OBGYN-guideline-specific reranker or benchmark exists** — medical
   evidence is general biomedical IR (PubMed) only.
5. **MedCPT (medical SOTA) is English-only** — conflicts head-on with the
   multilingual requirement.
6. **MICE is very recent** (early 2026, public code, limited independent
   corroboration).
7. **The fine-tune "GO" leans on one preprint** whose headline is the retriever's.

---

## 5. How this reshapes the R2c plan

The empty convertibility column flips the phase order I'd assumed. Because the
gate with **zero evidence** is also the one most likely to kill a candidate, it
should be de-risked **first**, not last:

1. **Phase 0 — convertibility spike (new, do first).** Take the smallest
   standard-BERT cross-encoder (ms-marco-MiniLM-L6-v2, ~22M) and *prove* it
   converts to TFLite/ONNX-Runtime-Mobile and runs int8 on the target ARM device,
   with a measured per-(query,chunk) latency for a 10–20 chunk batch. If even the
   smallest standard cross-encoder won't convert, neural reranking on-device is in
   doubt and feature-LTR becomes the lead — a conclusion worth reaching cheaply
   and early.
2. **Phase 1 — offline quality (as planned).** Score the deployable shortlist
   (#1–#5) plus the bge-reranker-v2-m3 reference on the hybrid top-20 pool against
   our labels: P@3/HR@3 (lenient + strict) vs the R2b floor and oracle ceiling,
   plus the R1 Stage-1 gate on the reranker score (the threshold-revival test; a
   trained cross-encoder is the best candidate yet to clear the 0.80 bar). The
   key contrast is #1 (tiny general) vs #2 (MedCPT, domain-matched). Reserve the
   by-query train/dev/test split first.
3. **Phase 2 — fine-tune.** Fine-tune the winning candidate on the 230k pairs and
   measure the reranker-specific delta vs its zero-shot baseline. (Swahili eval
   returns when multilingual does — out of scope for now per §0.)
4. **Gate end-to-end** per the R2 guardrails (M1 MCQ ±RAG rerun, SAQ A/B) before
   any adoption.

The open questions the review could not answer — on-device convertibility,
whether a small cross-encoder's N-pass cost fits the ARM budget at 10–20 chunks,
the #1-vs-#2 domain-match question, and the reranker-specific fine-tune ROI —
*are* the Phase 0–2 test plan.

---

## Sources

Primary (model cards / papers): bge-reranker-v2-m3, bge-reranker-base,
jina-reranker-v2-base-multilingual, mmarco-mMiniLMv2-L12-H384-v1,
msmarco-MiniLM-L6-en-de-v1, Multilingual-MiniLM-L12-H384 (HF); MICE
(arXiv 2602.16299), MedCPT (arXiv 2307.00589), biomedical RAG fine-tuning
(arXiv 2510.04757), BEIR (arXiv 2104.08663), mMARCO (arXiv 2108.13897); Intel
fastRAG quantization guide; Sentence-Transformers CrossEncoder efficiency docs.
Full URL list and per-claim evidence in the workflow result
(`tasks/wbf76bt2o.output`).
