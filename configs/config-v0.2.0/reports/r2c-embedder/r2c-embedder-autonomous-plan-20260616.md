# R2c embedder — autonomous execution plan

*2026-06-16. Purpose: run the embedder bake-off unattended. Every gate has a
**pre-registered decision rule + fallback** so no mid-run human input is needed, and the
points where I *will* surface are listed explicitly. Companion: the test plan
([`r2c-embedder-test-plan-20260616.md`](r2c-embedder-test-plan-20260616.md)), Phase 0 results
([`r2c-embedder-phase0-deviceability-20260616.md`](r2c-embedder-phase0-deviceability-20260616.md)),
lit-review ([`r2c-embedder-litreview-20260616.html`](r2c-embedder-litreview-20260616.html)).*

## Autonomy status
| Resource | State | Autonomous? |
|---|---|---|
| Android device (adb, flagship SM8750) | connected | ✅ local benchmarks |
| **Cluster — LOCAL `runai` on the Mac** | v2.22.73, token **VALID** (`runai list` → project `light-yiren`), `~/.kube/config` fresh | ✅ **submit via `RUNAI_LOCAL=1`** |
| Cluster login node (`ssh light` → `runai`) | reachable, but its runai token expired (`invalid_grant`) | ⚠ use only for key-file reads + `ssh`, not job submission |
| `ssh moi` (Ubuntu, <1 GB RAM) | up | ❌ too small for conversion |
| HF EmbeddingGemma license | accepted | ✅ |

### ✅ NO HUMAN UNBLOCK NEEDED
Earlier read was wrong: only the *cluster-side* runai token is expired. The **local Mac token is
valid**, so jobs submit via `submit_job.sh` with `RUNAI_LOCAL=1` (local `runai submit`; `ssh light`
still serves the API-key files). **Long unattended execution is feasible now** — pending a one-shot
smoke-submit to confirm submission (not just `list`) succeeds.

## Gecko baseline confound — resolution
`app_config.json`: `Gecko_1024_quant.tflite`, **CPU** (`use_gpu_for_embeddings:false`), dim 768.
Stock `benchmark_model` (full seq1024) = 562 ms; app reports ~20 ms ⇒ the AI Edge RAG library
feeds the **actual short query length** (dynamic seq). Candidate builds are **fixed-seq** (always
compute 256/512). **Implication: candidates are likely a latency regression vs Gecko for short
queries.** Confirm autonomously: re-run the app RAG benchmark, read `[TIMING] GeckoEmbeddingModel`.
**Decision: latency is a FACTOR, not a kill-gate** — query embed is once-per-turn vs ~14 s
generation; record the delta and weigh it only at the final winner decision.

## Pre-registered acceptance (no mid-run human input)
- **Primary metric:** recall (recall@20 / strict pool-recall) + lift on the *addressable set*.
- **End-to-end acceptance:** ≥5 pp over the deployable hybrid baseline on kenya key-fact recall,
  no penalty/refusal regression (5 pp = `1/√n` noise floor at n=312). Whole-set null is
  *expected-if-small* (≈1/5 addressable), not disproof.
- **Tiebreak:** simplicity + on-device footprint/latency.
- **Candidate priority:** EmbeddingGemma (primary) > MedEmbed-small/bge (simple) > MedCPT (drop if it fights us).

## Phases — decision rules + fallbacks (execute top-to-bottom)

### Phase 0 closeout — device only (autonomous NOW)
1. Re-run the app RAG benchmark → real Gecko per-query embed latency → finalize baseline +
   the regression delta vs EmbeddingGemma (125 ms@4t / 249 ms@1t) and bge.
2. **Done already:** EmbeddingGemma deployable (CPU+int8, 187 MB); GPU dead for all embedders;
   int8 mandatory.
- **Rule:** record latency; do NOT kill any candidate on latency alone.

### Phase 1 — top-20 coverage refinement (cluster, after re-auth)
`precompute_arms_matrix.py --top-k 20` on kenya (r1-threshold worktree) → true addressable size +
the **addressable set**. Update `kenya-coverage-summary` in place.
- **Rule:** if addressable slice <10 % even at top-20 → tag corpus-dominant, still run Phase 2 for
  EmbeddingGemma but with lowered expectations and a prominent R-corpus recommendation.

### Conversions (cluster, after re-auth)
int8 bge-small (for a fair number vs the fp32 measured) + MedCPT query-encoder.
- **Fallback:** if MedCPT conversion fails after 2 attempts → **drop it** and proceed with
  EmbeddingGemma + bge/MedEmbed (MedCPT is complex, English-only, low-priority).

### Phase 2 — offline retrieval screen (cluster)
Re-embed corpus + queries per surviving candidate (apply per-candidate prefixes; EmbeddingGemma at
256 **and** 768). Score recall@20 + addressable-set lift + P@3 + Stage-1 gate on kenya-relevance
(weighted) + mamaretrieval (descriptive) + 1 reference ceiling.
- **Advance rule:** a candidate goes to Phase 3 **iff** it beats Gecko on strict recall@20 on
  kenya-relevance **or** on the addressable set.
- **STOP rule:** if NO candidate beats Gecko on recall → halt, write the verdict ("no embedder
  improves retrieval recall → bottleneck is corpus, not embedder"), recommend R-corpus, **surface**.

### Phase 3 — end-to-end value gate (cluster)
Phase-2 passers → arms → Gemma 4 E4B generate → judge SAQ recall (gpt-oss-120b) + healthbench rubric.
- **PASS:** addressable-subset recall lift positive + retrieval recall gain + no penalty/refusal
  regression (secondary: ≥5 pp whole-set over hybrid).
- **FAIL/null everywhere:** record "no end-to-end benefit," recommend corpus expansion, **surface**.

### Phase 4 — fine-tune (conditional)
Only if a base passes Phase 2 but misses the Phase 3 margin AND addressable headroom remains. Frozen
`split.json`, no leakage. EmbeddingGemma/bge both have recipes.

### Phase 5 — land (HUMAN)
Device port + on-device parity (`check_parity.py`) + re-version cached artifacts + PR. App
integration & release are human — **surface**.

## When I surface (otherwise I run unattended)
1. **Now:** RunAI re-auth needed.
2. A STOP/pivot verdict (no candidate beats Gecko → corpus).
3. Final winner + the latency-vs-quality tradeoff decision (after Phase 3).
4. Phase 5 landing (human integration/release).
5. Any hard failure I can't route around (quota denied, repeated job failures, conversion OOM with no fallback).

## Resource defaults (pre-decided)
- Node pool order: **H200 > H100 > A100** (per quota; pass `NODE_POOL`).
- HF token: `~/.cache/huggingface/token` (has EmbeddingGemma access).
- Judges: Qwen3-32B (relevance) / gpt-oss-120b (SAQ + rubric), per pinned config.
- Spend: loop-until-done (no token budget cap set); prefer cluster for >30-min jobs.

## Execution mechanics
- Submit cluster jobs via `cluster/run_cluster_*.sh` + `submit_job.sh`; monitor via `runai`;
  background + self-paced wakeups between polls.
- Flushed progress + probe-ETA before any >10-min job.
- Commit + push per completed phase (separate scoped commits).

## Live status
- ✅ Phase 0: EmbeddingGemma deployable; GPU-dead/int8-mandatory; Gecko confound resolved (pending app-benchmark confirm)
- ⏳ Phase 0 closeout: app RAG benchmark re-run (device, autonomous)
- ⛔ Phases 1–3 + conversions: **blocked on RunAI re-auth**
