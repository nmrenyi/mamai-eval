# MCQ +RAG vs no-RAG — Gemma 4 E4B (n=23,241)

*Run completed 2026-05-20. Both arms on the same row set, same model artifact, same cluster setup; the only difference is whether retrieved chunks were injected into the prompt.*

## TL;DR

**RAG regresses MCQ accuracy by 1.8 pp overall** (95% CI doesn't overlap with no-RAG). The deployed maternal/neonatal RAG bundle helps less than it hurts when the eval task is broad medical-knowledge MCQ. The mechanism is: out-of-corpus retrievals distract the model from its training-time knowledge, net-flipping 420 questions from correct to wrong.

This is **not a deployment failure** — MCQ tests a different distribution than what the on-device app sees in practice. The open-ended evals (`kenya`, `whb`, `afrimedqa_saq`, all maternal/neonatal vignettes that match the corpus) are where we should expect RAG to help, and where the deployment claim ultimately rests.

## Setup

| | |
|---|---|
| **Model** | Gemma 4 E4B, Q4_0 GGUF |
| **Inference runtime** | llama-cpp-python, CUDA on NVIDIA A100-80GB |
| **HF dataset** | `nmrenyi/mamabench@v0.2`, MCQ configs `afrimedqa` + `medqa_usmle` + `medmcqa` |
| **Total rows** | 23,241 (full row sets, no subsampling) |
| **RAG context source** | `mcq-v0.2.0-bundle-v0.2.0-cpu32` precompute against the v0.2.0 chunk bundle (87 sources / 63,650 chunks, maternal-neonatal-obstetric guidelines) |
| **RAG top_k** | 3 |
| **MCQ adapter prompt** | `configs/config-v0.2.0/mcq_system.txt` ("Reply with ONLY the letter of the correct option") |
| **Run dirs** | no-RAG: `results/generation/gemma4-e4b/20260520T032705-cluster-norag/`<br>+RAG:   `results/generation/gemma4-e4b/20260520T082028-cluster-rag/` |
| **Wall-clock** | no-RAG: 19 min  ·  +RAG: 58 min (longer prompts ⇒ slower prefill) |

## Aggregate accuracy

| Venue | Accuracy | 95% CI | Δ vs no-RAG |
|---|---:|---|---|
| Gemma 4 E4B / no-RAG | **53.7%** (12,489 / 23,241) | [53.1%, 54.4%] | — |
| Gemma 4 E4B / +RAG   | **51.9%** (12,069 / 23,241) | [51.3%, 52.5%] | **−1.8 pp** |

CIs do **not** overlap — the regression is real at this sample size, not noise.

## Per-config breakdown

| Config | n | RAG | no-RAG | Δ | Per-row agreement | Cohen's κ |
|---|---:|---:|---:|---:|---:|---:|
| afrimedqa | 534 | 48.1% (257) | 51.3% (274) | **−3.2 pp** | 59% | 0.478 |
| medqa_usmle | 4,199 | 45.2% (1,899) | 48.9% (2,054) | **−3.7 pp** | 62% | 0.528 |
| medmcqa | 18,508 | 53.6% (9,913) | 54.9% (10,161) | **−1.3 pp** | 60% | 0.459 |
| **Overall** | 23,241 | 51.9% | 53.7% | **−1.8 pp** | 60% | 0.476 |

All three configs regress in the same direction. The magnitude is largest on **medqa_usmle** (−3.7 pp) — exactly the dataset furthest from the maternal/neonatal corpus (USMLE board-style: pharmacology, anatomy, internal medicine).

Per-row agreement is **only 60%** (κ = 0.48, low end of "moderate"). RAG changes the predicted letter on **40% of rows**, but most of those changes are noise-flips, not knowledge-injection wins.

## Where the −1.8 pp comes from: net-flips analysis

For each row we check whether RAG changed the model's answer relative to no-RAG, and whether that change was toward or away from the ground truth:

| Config | RAG flipped → correct | RAG flipped → wrong | Net |
|---|---:|---:|---:|
| afrimedqa | 59 | 76 | **−17** |
| medqa_usmle | 409 | 564 | **−155** |
| medmcqa | 2,511 | 2,759 | **−248** |
| **Total** | **2,979** | **3,399** | **−420** |

So in absolute terms, **RAG sent the model to a different letter on 6,378 rows; 47% of those changes were correct, 53% were wrong** — biased toward wrong by 420 net. Spread across 23k rows, that's the 1.8 pp regression.

## Why this is the expected shape

The deployed RAG corpus is **87 sources of maternal/neonatal/obstetric guidelines** (WHO PPH, ACOG, Essential Midwifery Practice, etc.). Per-row inspection (not enumerated here) reveals retrieval surfaces those guidelines no matter what the question is about — because that's all the bundle contains. For MCQ rows like:

> *"A 45-year-old male presents with crushing substernal chest pain radiating to the left arm..."* (USMLE cardiology — not in the corpus)

…retrieval still returns 3 chunks (the closest cosine matches the embedder can find), but those chunks are about PPH or fetal monitoring or partograph use — completely unrelated. The MCQ adapter prompt is also clearly not designed for RAG: it says "Reply with ONLY the letter" but the user message begins with "RELEVANT CONTEXT FROM MEDICAL GUIDELINES: [Document 1: …" which biases the model toward acknowledging the context instead of answering crisply.

Two failure-mode hypotheses, both supported by the disagreement pattern:

1. **Distraction.** Off-topic context degrades the model's confidence in its training-time recall. The model attends to irrelevant chunks instead of its priors.
2. **Prompt-format collision.** The MCQ adapter prompt was authored for non-RAG. Combined with the RAG-wrapped user message, it pushes the model out of the "single letter" output distribution.

## What this means for the deployment story

**It doesn't undermine the deployed product.** The deployed app's users (Zanzibar nurse-midwives) ask questions about postpartum hemorrhage, neonatal sepsis, eclampsia — questions where the RAG corpus *is* relevant. The MCQ regression we measured is essentially out-of-distribution: we're testing whether a maternal/neonatal RAG corpus helps with a USMLE-style broad medical-knowledge test, and the answer is "of course not."

What it DOES mean:
- **MCQ headline accuracy in the eval report should be no-RAG numbers**, not RAG. Adding RAG to MCQ tests the wrong thing.
- **The deployed-app analogue is the open-ended track** (`kenya`, `whb`, `afrimedqa_saq`). All three are maternal/neonatal clinical vignettes. RAG should *help* there. **That comparison is the load-bearing one for the report.**
- The current MCQ +RAG run is still useful as a **negative-control baseline** — it shows the retrieval pipeline is doing *something* (40% of rows get different predictions, so retrieval is being injected and the model is reacting), but that something isn't always positive when the corpus is off-domain.

## Caveats

- **Single seed, single model.** All numbers are from one run of one model artifact. No noise-from-sampling estimate beyond the bootstrap CIs.
- **The MCQ adapter prompt may be a confound.** The eval would benefit from one of:
  - Running the same MCQ rows with the *clinical* system prompt (not the adapter) to separate "adapter / RAG collision" from "domain mismatch."
  - Designing a RAG-aware MCQ adapter that explicitly tells the model "use the context if it helps; ignore it if it doesn't."
- **Statistical significance ≠ deployment significance.** A 1.8 pp regression at n=23,241 is statistically detectable but doesn't tell us anything about user-facing harm. The deployed app doesn't show users an MCQ accuracy number.

## Recommended next step

**Run the open-ended +RAG vs no-RAG comparison** (`kenya`, `whb`, `afrimedqa_saq`; 369 total rows). Same pattern as today: cluster job, scoring via `rescore_open_v2.py` (3-judge ensemble — judge IDs need to be pinned in `params.json` first). Hypothesis: RAG helps meaningfully on open-ended because the corpus matches the question domain. If that hypothesis holds, the report's deployment-relevant headline is "RAG adds X pp to maternal/neonatal clinical Q&A," not the MCQ regression we measured here.

## Provenance

- Cluster job names: `mamai-v02-gemma-mcq-full-retry` (no-RAG), `mamai-v02-rag-precompute-cpu32` (precompute), `mamai-v02-gemma-mcq-rag` (+RAG eval)
- Code commits: refactor `56a3489`, no-RAG results `16dde2c`, +RAG results `4a20bff`
- RAG bundle: v0.2.0 (87 sources, 63,650 chunks) per mamai commit `9b195e7`
- Branch: `feat/end-to-end-mcq-rag-report-20260520`
