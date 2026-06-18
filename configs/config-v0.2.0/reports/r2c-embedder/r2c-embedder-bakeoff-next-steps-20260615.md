# R2c (embedder) — bake-off execution plan / scaffold

*Filed 2026-06-15. The on-device **embedder** swap (replace Gecko) — the retrieval
lever still on the table after the reranker direction under-delivered. Successor to
the R2c **embedder** sections of
[`r2-retriever-upgrade-plan.md`](r2-retriever-upgrade-plan.md) (§R2c/§R2d). Note the
naming overlap: the `r2c-reranker-*` docs are the *reranker* line; this is the
*embedder* line.*

---

## Why this, and what we learned from the reranker first

The reranker work (see [`r2c-reranker-results-20260613.html`](../configs/config-v0.2.0/reports/r2c-rerank/r2c-reranker-results-20260613.html))
landed on a clear, sobering result: reranking lifts **offline** retrieval (P@3
0.51→0.82 on mamaretrieval) but **does not convert end-to-end** — on kenya it
doesn't even improve *retrieval relevance* (gecko stays best), and the low absolute
relevance across every arm (best mean grade 1.49/6) points to a **corpus-coverage**
bottleneck, not a ranking one.

**Three lessons that must shape the embedder bake-off up front:**
1. **Validate on the deployment queries, not just mamaretrieval.** Offline gains on
   the audit set did not transfer to kenya. Every embedder candidate must be scored
   on kenya/healthbench, not only the 230k audit labels.
2. **Corpus coverage may be the real ceiling.** A better embedder cannot retrieve an
   answer the corpus doesn't contain. Measure coverage explicitly (how often *any*
   retriever surfaces a grade≥3 chunk per kenya query) before concluding an embedder
   is the bottleneck — and consider whether R-corpus (content expansion) outranks R2c.
3. **Score quality ≠ ranking quality.** Carry the R1 Stage-1 gate (chunk AUC /
   concordance) alongside P@3 — a new embedder could revive the thresholding idea.

**Guiding principle (carried from the reranker line):** hard constraints → offline
screen → the decisive end-to-end gate → fine-tune only if needed. Don't port a
candidate to device until it's beaten the hybrid+rerank bar AND moved answers.

---

## P0 — Candidate shortlist + hard-constraint screen (do first, cheap)

Candidates must pass **before** any embedding run:
- **LiteRT-convertible + quantizable** (int8) within the app's runtime (same gate the
  reranker passed; sentence-transformer/BERT-family encoders are the safe path).
- **Latency/memory budget** on-device (Gecko is 768-dim; stay near that footprint).
- **Multilingual** — real queries arrive in Swahili; corpus is slated to add German
  textbooks. This is a *hard* filter (Gecko is multilingual; English-only encoders
  are out).

**Candidate menu (to refine in a short lit-review, P0.1):**
- *On-device-first:* **EmbeddingGemma-300M** (Google, multilingual, MRL dims,
  on-device-targeted) — the headline candidate; **multilingual-e5-small (118M)** /
  **-base (278M)**; **gte-multilingual-base (305M)**; **snowflake-arctic-embed-m-v2.0**.
- *Heavier, maybe out of budget:* bge-m3 (568M), jina-embeddings-v3 (572M), LaBSE (471M).
- *Reference ceilings (NOT deployable):* voyage-3 / Qwen3-Embedding-8B / octen — bound
  how much we lose by staying on-device (already partly in the audit rankings).
- Baseline to beat: **Gecko** (deployed) and **Gecko+BM25 hybrid** (R2a).

| # | task | where | effort |
|---|---|---|---|
| 0.1 | Short lit-review → pick 2–3 constraint-passing candidates (+ confirm LiteRT path) | desk | small |
| 0.2 | Confirm each converts to int8 LiteRT + fits the latency/memory budget (MiniLM-style spike) | mac/device | small–med |

## P1 — Offline screen on the held-out test split (reuse the reranker harness)

Re-embed the 63,650 corpus chunks + the queries with each candidate, retrieve top-k,
score against the 230k labels. **Top-up judging** for chunks a candidate surfaces that
were never judged (outside everyone's audit top-20) — Qwen3-32B + the V2 rubric
(`retrieval_eval/_kenya_rubric.py`, `judge_kenya_relevance.py`).

| # | task | reuse |
|---|---|---|
| 1.1 | Re-embed corpus+queries per candidate; build the candidate's top-20 pool | `precompute_*`, `retrieval.py` |
| 1.2 | Score P@3/HR@3 + Stage-1 gate vs 230k (+ top-up judge new chunks) | `score_pool_matrix.py`, `compare_retriever_gates.py` |
| 1.3 | Add candidate rows to the **2D scoreboard** (embedder × reranker × {offline, kenya-relevance, end-to-end}) | the table in the reranker report |

**Gate:** a candidate must beat Gecko (and ideally hybrid) on the test split before
the (expensive) end-to-end run.

## P2 — The decisive end-to-end gate (the lesson from the reranker)

For candidates that pass P1, run the **same value gate the reranker used**, on the
**deployment** sets, with the actual on-device generator **Gemma 4 E4B**:
- **kenya SAQ** (key-fact recall) — coverage-poor case
- **healthbench-oss_eval** (rubric: overall / completeness(+) / penalty(−)) — coverage-good, large-n

| # | task | reuse |
|---|---|---|
| 2.1 | Precompute candidate arms (candidate / +hybrid / +rerank) for kenya + healthbench | `precompute_arms_matrix.py` |
| 2.2 | Generate Gemma 4 E4B; judge SAQ recall (gpt-oss-120b) + healthbench rubric | `run_cluster_value_gate*.sh`, `rescore_open_v2.py`, `rescore_rubric.py` |
| 2.3 | Coverage check: per kenya query, does *any* candidate surface a grade≥3 chunk? (is the answer even in the corpus?) | `judge_kenya_relevance.py` |

**Acceptance:** end-to-end recall/rubric rises over the deployed Gecko on the reliable
sets, no refusal/penalty regression. If even the best embedder doesn't move answers
AND the coverage check is low → the bottleneck is the corpus (pivot to content
expansion), not the embedder.

## P3 — Fine-tune / distill (R2d, only if needed)

Contrastively fine-tune the chosen embedder on the 230k graded pairs and/or distill
voyage/Qwen3 rankings. Use the frozen `split.json` (no test-query leakage). Most
expensive; only after P1/P2 pick a base and only if it still falls short.

## P4 — Land

Device port (LiteRT int8) + on-device parity (mirror `check_parity.py`), re-version
cached embeddings/thresholds/contradiction audit (a new embedder invalidates them),
PR. Gate on P2 passing.

---

## Reusable harness (already built for the reranker)

- `retrieval_eval/precompute_arms_matrix.py` — N retriever × reranker arm contexts (extend for embedder dim)
- `retrieval_eval/score_pool_matrix.py` — offline P@3 + Stage-1 gate per pool
- `retrieval_eval/judge_kenya_relevance.py` + `_kenya_rubric.py` — Qwen3-32B V2-rubric relevance judging
- `retrieval_eval/compare_retriever_gates.py` — Stage-1 score-quality gate
- `cluster/run_cluster_{arms_matrix,score_matrix,value_gate,value_gate_judge,kenya_relevance}.sh`
- `end_to_end_eval/{run_eval,rescore_open_v2,rescore_rubric}.py` — generate + judge (rubric has +/− split)
- Frozen split: `configs/config-v0.2.0/results/retrieval_eval/r2c-rerank/split.json`

## Open questions to resolve early
- Does EmbeddingGemma convert to LiteRT int8 within budget? (P0.2 — the make-or-break feasibility gate.)
- ~~Is the kenya ceiling corpus coverage or retrieval?~~ **ANSWERED (2026-06-15, P-1
  coverage diagnostic — `kenya-coverage-summary-20260615.md`):** mixed. Union top-3
  coverage 70% lenient / 58% strict; ranking-fixable (some arm found, gecko missed)
  ~18–22%; corpus/depth-limited (no arm) 30–42%. Kill-rule (≥60% not-strict-covered)
  NOT triggered → **R2c proceeds, but scope it to the ~1/5 ranking-fixable slice**;
  corpus expansion stays a parallel lever. Caveat: top-3-only → re-run coverage at
  **top-20** first to split true corpus-gap from rank-depth and size R2c's real ceiling.
- Multilingual eval: do we have non-English (Swahili) eval queries, or only English so far?

## Immediate next action
**P0.1 + P0.2** — lit-review the multilingual on-device shortlist and confirm
EmbeddingGemma (+ one e5/gte fallback) converts to int8 LiteRT in budget. Cheap, and
it gates everything below.
