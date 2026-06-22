# Generator × Prompt Matrix — comprehensive comparison plan

*Filed 2026-06-22 · branch `feat/g1-prompt-fix-20260611` · extends the G1/G2 prompt A/B and the oracle-faithfulness A/B into a full generator×prompt grid.*

## Why

The G1/G2 prompt A/B was run **no-RAG on Gemma 3n only**, and the faithfulness A/B revealed **3n is ~3× less
faithful to context than Gemma 4** — a generator property that was *unmeasured* when the app shifted G4→3n.
Two gaps to close: (1) end-to-end must be measured **with RAG** (the deployed config), and (2) the generator
choice must be compared **across recall, safety, *and* faithfulness**, not just recall/completeness. This plan
crosses **generator × prompt** and evaluates every cell on **both** an end-to-end track and a faithfulness track.

## The grid

**Generators** (all cluster-served — ~$0 API):
- **G4** — Gemma 4 E4B (GGUF, llama-cpp)
- **3n** — Gemma 3n E4B (GGUF) — the deployed generator
- **Qwen** — `Qwen3.5-397B-A17B-FP8` (vLLM, ~5–6× 80 GB tensor-parallel) — **capability ceiling**, not deployable

**Prompts** (the committed arm files):
- **baseline** = `config-v0.2.0/system_en.txt`
- **+G1** = `g1-ab-3n-20260619/prompts/arm2_system_en.txt`
- **+G1+G2** = `g1-ab-3n-20260619/prompts/arm3_system_en.txt`

**Two evaluation tracks** per cell:
- **Track A — end-to-end, +RAG only** (deployed EmbeddingGemma retrieval, top-3). Datasets: **kenya (312)** +
  **healthbench_oss_eval (1209)**. Scored by pinned `gpt-oss-120b`: kenya → 4-value behavior + key-fact recall +
  safety; healthbench → rubric (weighted_met / completeness / penalty).
- **Track B — faithfulness, oracle context** (mamaretrieval v0.2.0, score ≥ 5, top-3, **2,989 q**). Scored:
  Patronus **Lynx-70B** → **gpt-5** categorize + calibrate (the validated pipeline; ~$10 total API — kept for
  comparability, the *only* dollar cost).
  *Track B stays oracle (not real retrieval) on purpose — it isolates the generator and stays comparable to the
  prior G4 number and the done 3n cells.*

3 generators × 3 prompts = **9 cells × 2 tracks**.

## Status — what's DONE, what's TO-RUN

### Track A — end-to-end (+RAG, kenya + healthbench)
| prompt ↓ / generator → | G4 | 3n | Qwen |
|---|---|---|---|
| baseline | ⬜ TO-RUN | ⬜ TO-RUN | ⬜ TO-RUN |
| +G1 | ⬜ TO-RUN | ⬜ TO-RUN | ⬜ TO-RUN |
| +G1+G2 | ⬜ TO-RUN | ⬜ TO-RUN | ⬜ TO-RUN |

*All 9 new. (We have 3n × 3 prompts **no-RAG** done in `g1-ab-3n-20260619/` — kept only as the no-RAG reference
for the 3n ±RAG delta; it is **not** a +RAG cell.)*

### Track B — faithfulness (oracle)
| prompt ↓ / generator → | G4 | 3n | Qwen |
|---|---|---|---|
| baseline | ⬜ TO-RUN *(prior ~9% as cross-check)* | ✅ **DONE — 27.1%** | ⬜ TO-RUN |
| +G1 | ⬜ TO-RUN | ✅ **DONE — 45.7%** | ⬜ TO-RUN |
| +G1+G2 | ⬜ TO-RUN | ✅ **DONE — 47.2%** | ⬜ TO-RUN |

*3n column ✅ done (`report-faithfulness.html`). New: **G4 × 3 + Qwen × 3 = 6 cells**. (The prior
`oracle-v0.2.0-faithfulness.html` G4 ~9% used a slightly different pool; G4-baseline is re-run through the
identical pipeline for clean apples-to-apples, with the old number as a sanity check.)*

**Total new generation: Track A 9 + Track B 6 = 15 generation runs.**

## What's reusable (no rebuild)
- **Infra:** `--system-prompt` override (`run_eval.py` + `eval_faithfulness.py`); combined judge runner
  (`run_cluster_g1_judge.sh`); combined Lynx runner (`run_cluster_g1_faith_lynx.sh`); the per-arm F3+F4 gpt-5
  driver (`/tmp/g1_f34.sh` pattern); the **GPU-race + heartbeat** machinery; the dangerous-case adjudication script.
- **Data:** 3n × 3 Track-B faithfulness cells (done). 3n × 3 no-RAG Track-A (reference only).

## Build items (NEW infra — Phase 0)
1. **Qwen-as-generator path.** Serve `Qwen3.5-397B-A17B-FP8` on vLLM and have `run_eval` / `eval_faithfulness`
   generate against the local OpenAI-compatible endpoint (`OPENAI_BASE_URL=local`). Needs a small generator
   entry / base-url plumbing + a `run_cluster_qwen_serve+gen.sh` recipe (model is FP8 ~400 GB, TP ~5–6).
2. **EmbeddingGemma retrievals for Track A.** Verify R2c precomputed EmbeddingGemma top-3 retrievals for
   kenya + healthbench exist on PVC (`eval_output/rag_arms_eg*` / `value_gate_eg*`); precompute if missing
   (`screen_embedder embed_retrieve` + `arm_format`) against the deployed `rag-bundle-v0.3.0` embeddings.

## Execution phases (ordered)
- **Phase 0 — prereqs** *(needs VPN)*: confirm G4 / 3n / Qwen-397B on PVC; verify or precompute EmbeddingGemma
  retrievals (kenya + hb); stand up the Qwen vLLM serving recipe + the run_eval base-url generator path.
- **Phase 1 — Track A generation (+RAG), 9 cells**: G4×3 + 3n×3 on GGUF (cheap, raced in parallel); Qwen×3 on the
  big vLLM serving job (one boot, 3 prompts sequential).
- **Phase 2 — Track A judging**: `gpt-oss-120b` (one boot) over all 9 cells — SAQ behavior + recall + safety, rubric.
- **Phase 3 — Track B generation (oracle), 6 cells**: G4×3 + Qwen×3 (3n reused).
- **Phase 4 — Track B scoring**: Lynx-70B (one boot) over the 6 new cells → gpt-5 categorize + calibrate (drivers).
- **Phase 5 — analysis**: assemble generator×prompt tables per axis (kenya recall / deflection / harm / **dangerous
  count**; hb weighted_met / completeness / penalty; Track-B raw / categorized / **calibrated** + CI / Lynx-miss).
  **Per-cell dangerous-case re-adjudication** — does +RAG and/or a stronger generator *ground* the lever-3 dose
  errors (oxytocin units, ceftriaxone-in-neonate, benzathine-penicillin dose, PEP, blisters)?
- **Phase 6 — report**: a combined **generator×prompt matrix** report under `configs/config-v0.2.0/reports/`
  (e.g. `generator-prompt-matrix-3n-g4-qwen-202606xx.html`), folding in the existing SAQ/rubric + faithfulness
  results; index it in `reports/README.md`.

## Cost
- **API: ~$10 total** — only the gpt-5 faithfulness categorize/calibrate on the 6 new Track-B cells. (GPT-5 as a
  *generator* was dropped for budget; replaced by cluster-served Qwen.)
- **Cluster GPU: the real cost** — Qwen-397B-FP8 serving needs a multi-GPU slice (~5–6× 80 GB) held for ~2–3 h
  across its 9 (Track A) + ... wait, Qwen rows = 3 prompts × (1,521 + 2,989) ≈ 13.5k generations; plus G4/3n GGUF
  (1 GPU each), gpt-oss-120b judge (1 GPU), Lynx-70B (2 GPU). Free in dollars, heavy on RunAI quota — the race
  matters most for the big Qwen allocation.

## Decisions locked
- Generators: G4, 3n, Qwen3.5-397B-A17B-FP8 (ceiling). Prompts: all 3 on every generator.
- Track A: **+RAG only** (deployed EmbeddingGemma), kenya + healthbench_oss_eval.
- Track B: oracle faithfulness, 2,989 q. gpt-5 kept only for the validated calibration judging.
- Per-cell dangerous-case re-adjudication is part of the deliverable.
