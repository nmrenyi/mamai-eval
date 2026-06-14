# R2c reranker — execution plan / next steps

*Filed 2026-06-13. Where the reranker work stands and everything left to do, in
priority order. Successor planning to
[`r2c-reranker-literature-review-20260613.md`](r2c-reranker-literature-review-20260613.md)
and the Phase-0 spike report.*

---

## Status update (2026-06-13, evening) — P0/P1/P2/P2.5 executed

Full results in
[`configs/config-v0.2.0/reports/r2c-rerank/r2c-reranker-results-20260613.html`](../configs/config-v0.2.0/reports/r2c-rerank/r2c-reranker-results-20260613.html).

- **P0 (correctness) — PASSED.** Deployed int8 `.tflite` + Kotlin-port tokenizer
  reproduce the offline torch-fp32 model at seq-256: P@3 0.626 vs 0.627, tokenizer
  99.97% exact, rank-identical (Pearson 1.0, top-3 agreement 1.0). The
  "approximate tokenizer parity" caveat is closed.
- **P2 (model menu) — DONE.** All deployable cross-encoders + Qwen3-4B/8B
  references scored on the test split @256. Best zero-shot deployable = mxbai-base
  (P@3 0.714); MiniLM-L6 0.627; MedCPT (medical) worst (0.561); Qwen3 refs ~0.79.
- **P2.5 (fine-tune) — DONE, the decisive lever.** Fine-tuning on the 159k
  in-domain pairs: MiniLM-L6 (23M) 0.627→**0.756** (AUC 0.731→0.783), beating
  zero-shot mxbai-base; mxbai-base 0.714→**0.821** with AUC **0.859** — first
  reranker to *cross R1's 0.80 thresholdability bar* (threshold revival is now
  worth re-testing on a fine-tuned score). **Final deployable choice: fine-tuned
  MiniLM-L6**; fine-tuned mxbai-base is the size-upgrade ceiling.
- **P1 (value gate) — MIXED / does not pass on the reliable sample.** 3 arms
  (Gecko / hybrid / hybrid+rerank) generated with the actual on-device generator
  **Gemma 4 E4B** (`app_config` llm_model = gemma-4-E4B-it), SAQ judged by
  gpt-oss-120b. On **kenya SAQ (n=312, the reliable sample) reranking does NOT
  improve key-fact recall** — gecko 0.125 > hybrid 0.112 ≈ rerank 0.111. The
  strong *offline* retrieval gain (P@3 0.51→0.76) does not translate end-to-end
  on this model. afrimedqa_saq (n=37) suggests benefit (rerank 0.171–0.208 vs
  gecko 0.157) but is too small to weight; MCQ: fusion helps (+4.3 pp) but rerank
  adds ~nothing over fusion; minilm-ft has the lowest harm rate (0.151). Caveat:
  offline P@3 was measured on mamaretrieval, the value gate on kenya/afrimedqa —
  a different distribution, so this is "benefit not demonstrated" rather than
  "reranking is useless." **NOTE:** an earlier run reported +12% kenya recall but
  used the wrong generator (gemma3n-e4b) — retracted.
- **P1 diagnostic — resolved: no-transfer, not a model ceiling.** Judged each arm's
  kenya top-3 relevance with the original Qwen3-32B grader + V2 rubric (2,691 pairs).
  Reranking does NOT improve kenya retrieval either: gecko P@3 0.277 (best) vs hybrid
  0.204 / minilm-ft 0.192 / mxbai-ft 0.264. The reranker's mamaretrieval gain
  (0.51→0.76) doesn't generalize to kenya; hybrid/BM25 fusion actively hurts kenya;
  and low absolute relevance across all arms (best mean grade 1.49/6) points to a
  **corpus-coverage** bottleneck. **Do not ship the reranker for kenya.** Any
  retriever/embedder change must be validated on the deployment queries (kenya /
  afrimedqa SAQ), not just mamaretrieval. afrimedqa_saq (n=37) — where the corpus
  covers the query — does show reranking help (mxbai 0.649 > gecko 0.477).

**Remaining (deployment):** convert the fine-tuned MiniLM-L6 to int8 `.tflite`
(same proven path) + re-run P0 parity on it; then P3 latency/batching + P4 PRs.

---

## Where we are

Established so far:
- **Reranking is the high-value lever** — R2b oracle (perfect rerank of the hybrid
  top-20) matches/exceeds the voyage retrieval ceiling.
- **MiniLM-L6 cross-encoder is a viable deployable reranker** — converts to LiteRT
  int8 (24 MB), runs on the real phone at 13.2 ms/pair, beats the feature-LTR bar
  zero-shot (P@3 0.628), int8 ≈ fp32 quality.
- **It runs in the real app** — integrated into `com.example.app` (mamai branch
  `feat/litert-reranker-20260613`); scored a 20-chunk pool in ~842 ms in-app and
  reordered to top-3 before generation.

Two repos hold the work, both on feature branches, neither merged:
- eval repo `feat/r2-retriever-upgrade-20260613` (plan, R2a/R2b, lit review, R2c-A
  feature-LTR, Phase-0 spike).
- app repo `feat/litert-reranker-20260613` (the in-app reranker).

**Guiding principle for ordering:** correctness gates → the decisive value gate →
optimization → ship. Don't optimize or ship a reranker we haven't proven is
(a) correct in-app and (b) actually improves answers.

---

## P0 — Correctness: does the deployed reranker reproduce the validated offline quality?

The in-app reranker *runs*, but its WordPiece tokenizer is a faithful-but-unverified
port. Until we confirm it produces the same scores as the offline model we
validated, the deployed reranker may be reordering on subtly wrong scores — and
every downstream test would inherit that. Cheap; blocks everything below.

> **Seq-len caveat (read first).** The offline quality numbers (P@3 0.628, the
> Stage-1 gate) were scored at **max_len 512**, but the deployed TFLite is fixed
> **seq 256**. So 0.628 is *not* the deployed model's quality, and a naive parity
> check against it would show a spurious mismatch (truncation, not tokenizer).
> Pin the seq-len decision now (it gates the reference and the deployed quality —
> see P3.2), and do parity at the deployed seq-len.

| # | Task | Where | Effort |
|---|---|---|---|
| 0.1 | Tokenizer parity: diff Kotlin `WordPieceTokenizer` token-ids vs HuggingFace on representative (query, chunk) strings (punctuation, accents, long-chunk truncation) | Mac + device log | small |
| 0.2 | Re-score the offline model **at the deployed seq-len (256)** on the test split — the true deployed-config reference (current 0.628 is the 512 number) | Mac | small |
| 0.3 | In-app score parity: log the in-app reranker's chunk-ids + scores for a probe query; reproduce the same (query, chunk) pairs offline **at seq 256** and diff — order + scores within int8 noise | device + Mac | small |
| 0.4 | If parity fails, fix the tokenizer / pair-truncation and re-verify | app repo | small–med |

**Exit:** in-app top-3 and scores match the offline model *at the same seq-len* →
tokenizer caveat closed, deployed reranker trustworthy, and we have the deployed
model's true (seq-256) quality.

---

## P1 — The decisive value gate: does reranking actually improve answers?

Everything so far is *retrieval-quality* evidence (P@3, oracle, latency). The R2
standing rule is **validate end-to-end, not on retrieval metrics alone.** This is
the acceptance gate that decides whether the reranker ships at all. Run it first
with the already-deployed MiniLM-L6 (cheap) before investing in a better model.

**Arms — three, to attribute the gain** (the comparison must isolate reranking
from the R2a hybrid change, else a delta is unattributable):
- **A. Gecko top-3** — the *currently deployed* config (the full-delta reference).
- **B. hybrid top-3 (RRF order)** — the R2a config; **the baseline that isolates
  reranking** (A→B is fusion's effect, B→C is reranking's effect).
- **C. hybrid top-20 → MiniLM rerank → top-3** — the reranker arm.

| # | Task | Where | Effort |
|---|---|---|---|
| 1.1 | Generate the three arms (A/B/C) on a stratified **kenya SAQ** sample — stratify toward the zero-recall / refusal-prone rows reranking should most help | cluster or device harness | med |
| 1.2 | Judge with the pinned gpt-oss-120b ensemble; report **key-fact recall**, refusal/deflection rate, the 4 axis scores, per arm | cluster (H200) | med |
| 1.3 | MCQ three-arm on the held-out MCQ half — does the −1.8 pp RAG gap close (and how much is fusion vs rerank)? | cluster | med |
| 1.4 | Safety-distribution check — no new safety-1s introduced by reranked context | cluster | small |

**Acceptance criteria (pre-registered):** key-fact recall does not regress (ideally
rises B→C); refusal rate does not rise; MCQ −1.8 pp gap moves toward 0; safety
distribution unchanged. **If reranking (B→C) fails this gate even with a strong
reranker, the whole direction is reconsidered** — so 1.x is the make-or-break.

Notes:
- Run 1.x with MiniLM-L6 first (deployed, cheap). Bound the ceiling by also
  scoring an **oracle-reranked arm** (we have the grades) — if even the oracle
  doesn't move answers, no reranker will.
- **G1 confound:** R1 established SAQ over-refusal is partly *prompt*-induced, not
  retrieval-induced. The refusal metric here inherits that — interpret the refusal
  delta as rerank-only by holding the prompt fixed across arms, and ideally run
  jointly with / stratified against the G1 prompt change so reranking isn't
  blamed/credited for a prompt artifact.

---

## P2 — Model optimization (only if P1 shows reranking helps)

Pick the best deployable reranker. Don't spend this effort unless P1 says reranking
moves answers.

**The deployable candidate menu** (all standard-BERT cross-encoders → same proven
LiteRT-int8 convertibility path; choice is purely quality vs size/latency):

- *Tiny (~4–38M):* TinyBERT-L2, MiniLM-L2/L4, jina-reranker-v1-tiny/turbo-en —
  latency/size floor; only relevant if we need smaller than MiniLM-L6.
- *Small (~22–33M):* **MiniLM-L6 (done, baseline)**, **MiniLM-L12**.
- *Mid (~110–280M):* **MedCPT** (medical domain), bge-reranker-base, mxbai-rerank-base,
  ms-marco-electra-base — stronger but heavier.
- *Non-CE:* feature-LTR (done, R2c-A); late-interaction (ColBERT/MICE) — different
  family, deprioritized at 10–20 chunks.

We test the few that answer a distinct question, not the whole menu:

| # | Task | Where | Effort |
|---|---|---|---|
| 2.1 | **MedCPT** (domain, ~109M) on the test split — does medical-domain pretraining beat tiny-general MiniLM-L6? (the #1-vs-#2 question) | Mac (in reach) | med |
| 2.2 | **MiniLM-L12** (~33M) — does a bit more depth help, same family, near-free to test? | Mac | small |
| 2.3 | One **mid English generalist** (bge-reranker-base or mxbai-rerank-base) — is a stronger non-domain model worth the size? | Mac/cluster | med |
| 2.4 | Strong **offline references** Qwen3-8B / bge-v2-m3 (NOT deployable — too big) — bound how much going small costs vs a top reranker | cluster (GPU) | med |
| 2.5 | **Fine-tune** the best candidate on the 230k graded pairs (split already frozen); re-measure quality **and the Stage-1 gate** — best shot at pushing the score past 0.80 and reviving the R1 threshold | cluster | med–large |
| 2.6 | (deprioritized) TinyBERT/L2/L4 — only if we need smaller/faster than MiniLM-L6; 13 ms already fits budget | Mac | small |
| 2.7 | Decide the final deployable model; re-run P0 parity + P1 gate with it | — | — |

---

## P3 — Engineering refinement (production-readiness)

| # | Task | Where | Effort | Note |
|---|---|---|---|---|
| 3.1 | **Batch the 20 forward passes** (re-export with batch dim) — the big latency lever, bigger than thread count (842 ms → expected ~few×100 ms) | app + eval | med | |
| 3.2 | seq-len decision: 256 (current, truncates long chunks) vs 512 (more faithful, slower) — measure quality/latency tradeoff. **Pin this early** — it gates the P0 reference and the deployed quality (see P0 caveat), not just production polish | eval + device | small | |
| 3.3 | Robustness: reranker-fail fallback (init already try/caught), thread-count tuning, warmup at init to remove first-call cost | app | small | |
| 3.4 | Config: tune `rerank_depth` (20 default); confirm the rerank/no-rerank toggle and defaults | app | small | |

---

## P4 — Landing

| # | Task | Where | Effort |
|---|---|---|---|
| 4.1 | PR: eval-repo `feat/r2-retriever-upgrade-20260613` (R2a/b + lit review + R2c-A + Phase-0) — the full Option-B story | eval repo | small |
| 4.2 | PR: app-repo `feat/litert-reranker-20260613` (the in-app reranker) — gate on P0+P1 passing | app repo | small |
| 4.3 | Update the v0.2 improvement-plan index with the reranker outcome + decision | eval repo | small |
| 4.4 | Cleanup: remove the pushed `benchmark_model` + scratch `.tflite` from `/data/local/tmp`; note the venvs | — | small |

---

## Sequencing / dependencies

```
P0 (correctness)  ──►  P1 first-pass with MiniLM-L6 (does reranking help?)
                            │
                   ┌────────┴─────────┐
              fails gate          passes gate
                   │                  │
        reconsider direction    P2 (best model) ──► P1 final (acceptance with chosen model)
        (corpus C1 / prompt G1)        │
                                   P3 (latency/robustness) ──► P4 (PRs, ship)
```

- **P0 before everything** — never build on an unverified reranker.
- **P1-first-pass before P2/P3** — the cheap "is this worth pursuing" check; don't
  optimize a model or latency that doesn't move answers.
- **P2 before P1-final** — the acceptance gate should ultimately use the chosen
  model; the first pass just decides whether to bother.
- **Latency (P3) is refinement, not blocking** — 842 ms already fits the 1-min
  budget; only matters for production polish.

## Immediate next action

**P0.1 + P0.2** — verify the deployed reranker reproduces the offline scores
(tokenizer parity + in-app score parity). Cheap, and it's the prerequisite for
trusting the P1 value gate.
