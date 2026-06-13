# R2c reranker — execution plan / next steps

*Filed 2026-06-13. Where the reranker work stands and everything left to do, in
priority order. Successor planning to
[`r2c-reranker-literature-review-20260613.md`](r2c-reranker-literature-review-20260613.md)
and the Phase-0 spike report.*

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

| # | Task | Where | Effort |
|---|---|---|---|
| 0.1 | Tokenizer parity: diff Kotlin `WordPieceTokenizer` token-ids vs HuggingFace on representative (query, chunk) strings (punctuation, accents, long-chunk truncation) | Mac + device log | small |
| 0.2 | In-app score parity: log the in-app per-chunk scores for a probe query, compare to the offline fp32/int8 model on the same 20 pairs — order + scores within int8 noise | device + Mac | small |
| 0.3 | If parity fails, fix the tokenizer (or pair-truncation) and re-verify | app repo | small–med |

**Exit:** in-app top-3 and scores match the offline validated model → tokenizer
caveat closed, deployed reranker trustworthy.

---

## P1 — The decisive value gate: does reranking actually improve answers?

Everything so far is *retrieval-quality* evidence (P@3, oracle, latency). The R2
standing rule is **validate end-to-end, not on retrieval metrics alone.** This is
the acceptance gate that decides whether the reranker ships at all. Run it first
with the already-deployed MiniLM-L6 (cheap) before investing in a better model.

| # | Task | Where | Effort |
|---|---|---|---|
| 1.1 | Build the ±rerank eval: hybrid top-20 retrieval → MiniLM rerank → top-3, vs the no-rerank top-3 baseline, on a stratified **kenya SAQ** sample (held-out, deployment-realistic) | cluster or device harness | med |
| 1.2 | Judge with the pinned gpt-oss-120b ensemble; report **key-fact recall**, refusal/deflection rate, the 4 axis scores | cluster (H200) | med |
| 1.3 | MCQ ±rerank on the held-out MCQ half — does the −1.8 pp RAG gap close? | cluster | med |
| 1.4 | Safety-distribution check — no new safety-1s introduced by reranked context | cluster | small |

**Acceptance criteria (pre-registered):** key-fact recall does not regress (ideally
rises); refusal rate does not rise; MCQ −1.8 pp gap moves toward 0; safety
distribution unchanged. **If reranking fails this gate even with a strong reranker,
the whole direction is reconsidered** — so 1.x is the make-or-break.

Note: run 1.x with MiniLM-L6 first (deployed, free-ish). Optionally bound the
ceiling by also scoring an oracle-reranked arm (we have the grades) — if even the
oracle doesn't help answers, no reranker will.

---

## P2 — Model optimization (only if P1 shows reranking helps)

Pick the best deployable reranker. Don't spend this effort unless P1 says reranking
moves answers.

| # | Task | Where | Effort |
|---|---|---|---|
| 2.1 | Option B quality scaling: score **MedCPT** (domain, ~109M) on the test split — does domain-match beat MiniLM-L6? | Mac (in reach) | med |
| 2.2 | Strong references **Qwen3-8B / bge-v2-m3** on the test split — how much does going small cost vs a strong reranker? | cluster (GPU) | med |
| 2.3 | **Fine-tune** MiniLM-L6 (or the winner) on the 230k graded pairs (clean by-query train/dev/test split already frozen); re-measure quality **and the Stage-1 gate** — best shot at pushing the score past 0.80 and reviving the R1 threshold | cluster | med–large |
| 2.4 | Decide the final deployable model; re-run P0 parity + P1 gate with it | — | — |

---

## P3 — Engineering refinement (production-readiness)

| # | Task | Where | Effort | Note |
|---|---|---|---|---|
| 3.1 | **Batch the 20 forward passes** (re-export with batch dim) — the big latency lever, bigger than thread count (842 ms → expected ~few×100 ms) | app + eval | med | |
| 3.2 | seq-len decision: 256 (current, truncates long chunks) vs 512 (more faithful, slower) — measure quality/latency tradeoff | eval + device | small | |
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
