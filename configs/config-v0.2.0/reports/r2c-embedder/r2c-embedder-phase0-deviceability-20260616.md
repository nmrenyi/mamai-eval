# R2c Phase 0 — on-device deployability gate (results, in progress)

*2026-06-16. Executes Phase 0 of [`r2c-embedder-test-plan-20260616.md`](r2c-embedder-test-plan-20260616.md):
convert/obtain each candidate as LiteRT int8, run it on the connected Android device,
measure latency / memory / size. Gate: deployable within the low-mid budget before any
deep evaluation.*

## Test rig
- **Device:** OnePlus `OPD2413`, **SoC SM8750 = Snapdragon 8 Elite**, 16 GB RAM (~5 GB free),
  8 cores, Android 15 (SDK 35), arm64-v8a.
- **⚠ This is a flagship, not the low-mid Zanzibar target** (accepted by decision).
  All latency/memory here is an **optimistic upper bound**; low-mid CPU is ~1.5–3× slower.
  Single-thread CPU numbers are the closest low-mid proxy.
- **Tool:** LiteRT `benchmark_model` (android_aarch64 nightly) at `/data/local/tmp/`.
- **Conversion env:** `ssh moi` (Ubuntu 24.04, Python 3.12) — but only **2 cores / <1 GB RAM**,
  too small to trace a 300M model through `ai-edge-torch`. ⇒ prefer prebuilt LiteRT artifacts.

## Baseline to beat (deployed Gecko, this device)
| | value |
|---|---|
| Embed + SQLite retrieval latency | **~20 ms** median (prior RAG benchmark) |
| Model on disk (int8) | **139 MB** |
| Vector store | ~195 MB (63,650 × 768 × fp32) |

## Results (all via `benchmark_model`, flagship CPU; latency = ms/embed)
| Candidate | precision / seq | CPU 4t | CPU 2t | CPU 1t | GPU | peak RAM | disk |
|---|---|---|---|---|---|---|---|
| **EmbeddingGemma-300M** (generic) | mixed int4/int8 / 256 | **125** | 135 | **249** | ❌ fails | **187 MB** | 171 MB |
| **bge-small** (proxy for MedEmbed-small) | **fp32** / 512 | 190 | 271 | 537 | ❌ fails | 213 MB | 127 MB |
| Gecko (deployed, same tool) | int8 / **1024** | 562 | 935 | >1 s | ❌ fails | **664 MB** | 139 MB |
| EmbeddingGemma sm8750 NPU build | int4/int8 / 256 | — needs LiteRT QNN runtime (`DISPATCH_OP`); not runnable via stock `benchmark_model` | | | | | 173 MB |
| **MedCPT** (query enc.) | — pending cluster conversion | | | | | | |

## Findings
1. **EmbeddingGemma is deployable and the strongest tested.** At seq256 (mixed int4/int8) it is
   **faster and lighter than fp32 bge-small**, and dramatically lighter in RAM than the deployed
   Gecko_1024 (187 MB vs 664 MB). On low-mid CPU expect ~250 ms × 1.5–3× ≈ **0.4–0.75 s** single-thread.
2. **GPU offload is not viable for any embedder here.** The GPU delegate fails on both bge
   (35/678 ops + batch mismatch) and EmbeddingGemma (CAST/EMBEDDING_LOOKUP/GREATER_EQUAL
   unsupported). **On-device path is CPU (XNNPACK).**
3. **NPU precompiled builds need the LiteRT QNN runtime** (`DISPATCH_OP` custom op) — not measurable
   with stock `benchmark_model`, and irrelevant for low-mid (all SoC builds are flagship anyway).
4. **int8 is mandatory.** fp32 bge-small is 190 ms (4t) → 537 ms (1t) on a *flagship* CPU; int8
   should cut ~2–4× and shrink disk ~4×.
5. **Gecko baseline confound — RESOLVED, and it favors EmbeddingGemma.** A fresh on-device RAG
   benchmark (`benchmark_20260616T122950`) measures real Gecko retrieval at **~4 s** (median
   4645 ms; embed seq1024 + SQLite cosine scan over 63k chunks) — the earlier "~20 ms" was bogus
   (No-RAG/noise). Two consequences: (a) **the embedder is not the latency bottleneck — the SQLite
   brute-force search dominates and is embedder-independent**; (b) EmbeddingGemma's embed (125 ms@4t)
   is *faster* than Gecko's (562 ms), so swapping is **latency-neutral-to-better, NOT a regression**.
   The latency objection to EmbeddingGemma is removed. (Separately: ~4 s retrieval is ~23% of the
   ~17 s total query — search optimization is its own lever, independent of R2c.)
6. **Static seq length** — the graph computes the full sequence regardless of query length, so pick
   the shortest seq build that fits real medical queries (seq256 is ample).

## Blockers / actions
- **A — EmbeddingGemma is HF-gated (HTTP 403).** The local token authenticates but the account
  lacks access to `litert-community/embeddinggemma-300m` (Gemma license not accepted).
  **USER ACTION:** accept the license at `huggingface.co/google/embeddinggemma-300m` and
  `huggingface.co/litert-community/embeddinggemma-300m`, then confirm the token has access.
  This is the *primary* candidate (only one with an official low-risk LiteRT path) — top priority.
- **B — bge-small export is fp32, not int8.** The only number above is fp32 (upper bound).
  Need an int8 build for a fair result (quantize via `ai-edge-torch`/converter).
- **C — MedCPT needs conversion** (no prebuilt LiteRT). `moi` may OOM at <1 GB RAM; needs a
  larger box or a different conversion route. Lowest priority (complex, English-only, dual-encoder).

## Available EmbeddingGemma artifacts (once unblocked)
Official `litert-community/embeddinggemma-300m` ships per-seq × per-SoC builds. Relevant:
`seq256_mixed-precision.tflite` (171 MB, generic CPU/GPU = **low-mid proxy**),
`seq256_mixed-precision.qualcomm.sm8750.tflite` (173 MB, this device's NPU = upper bound).
Note: **all** SoC-precompiled builds are flagship (Qualcomm 8-series / flagship MediaTek /
Tensor) — a real low-mid phone would run the **generic build on CPU**.

## Status — Phase 0 PASS for EmbeddingGemma
- ✅ Device characterized; benchmark pipeline proven; baseline anchored
- ✅ **EmbeddingGemma (generic seq256) — DEPLOYABLE** (125 ms@4t / 249 ms@1t, 187 MB, 171 MB disk; CPU-only)
- ✅ GPU-dead + int8-mandatory findings (all embedders)
- ✅ **Gecko confound RESOLVED** — real retrieval ~4 s (search-dominated, embedder-independent); EmbeddingGemma is latency-neutral-to-better → **latency objection removed**
- ✅ **Phase 0 verdict: EmbeddingGemma clears the deployability gate** (deployable, fits budget, not a latency regression)
- ⏳ bge-small int8 + MedCPT conversion (secondary; cluster) — only needed if they reach the final on-device cut
