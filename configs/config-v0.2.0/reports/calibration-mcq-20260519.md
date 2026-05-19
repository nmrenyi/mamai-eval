# MCQ device-vs-cluster calibration

- **Manifest**: `configs/config-v0.2.0/calibration/mcq_300.json` (name=`mcq_calibration_300`, seed=42, created=2026-05-19T13:43:52Z)
- **Cluster run**: `configs/config-v0.2.0/results/generation/gemma4-e4b/20260519T135210-cluster-cal/`
- **Device run**: `configs/config-v0.2.0/results/generation/gemma4-e4b-device/20260519T134622/`
- **HF dataset**: `nmrenyi/mamabench@v0.2`
- **Sampled**: 100 per config × 3 configs = 300 rows
- **Joined**: 300 overlapping rows scored on both venues

## Runtime

| Venue | Model artifact | Inference runtime | Hardware backend | Numeric precision |
|---|---|---|---|---|
| Device | gemma-4-E4B-it.litertlm | LiteRT-LM | GPU (Android, OpenCL) | **FP16** |
| Cluster | google_gemma-4-E4B-it-Q4_0.gguf | llama-cpp-python | CUDA (NVIDIA A100 80GB) | **Q4_0 (4-bit integer quant)** |

> The two venues run the same model family but very different precision regimes:
> device uses LiteRT-LM's GPU FP16 attention (the default — known FP32 escape via
> the artifact's `prefer_activation_type=float32` metadata key is **not** set on
> this artifact), while cluster uses llama-cpp-python's GGUF Q4_0 (integer 4-bit
> quantisation) on CUDA. Any accuracy gap below the ±5 pp noise floor at n=300
> is dominated by precision differences, not model differences.

## Aggregate accuracy

| Venue | Accuracy | 95% CI | n |
|---|---:|---|---:|
| Device (LiteRT, .litertlm) | **52.3%** (157/300) | [47.0%, 58.0%] | 300 |
| Cluster (GGUF Q4_0, llama-cpp) | **49.7%** (149/300) | [44.0%, 55.0%] | 300 |
| **Δ (device − cluster)** | **+2.7%** | | |

## Per-row agreement

- **Same predicted letter on both venues**: 195/300 (65.0%)
- **Cohen's κ** (chance-corrected agreement, 4-letter classes): **0.558**
  - κ ≥ 0.41 → "moderate" agreement.

## Per-config breakdown

| Config | n | Device acc | Cluster acc | Δ | Agree | κ |
|---|---:|---:|---:|---:|---:|---:|
| afrimedqa | 100 | 48.0% (48) | 42.0% (42) | +6.0% | 66.0% (66) | 0.569 |
| medmcqa | 100 | 57.0% (57) | 54.0% (54) | +3.0% | 64.0% (64) | 0.519 |
| medqa_usmle | 100 | 52.0% (52) | 53.0% (53) | -1.0% | 65.0% (65) | 0.570 |

## Disagreement patterns

- 105 rows (35.0%) where device and cluster predict different letters.
- Of those, **device-correct & cluster-wrong**: 41
- **cluster-correct & device-wrong**: 33
- **both wrong, different letters**: 31

### Confusion matrix (rows=cluster prediction, cols=device prediction)

| cluster ↓ \ device → |  | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|---|
| **** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **A** | 0 | 53 | 3 | 10 | 3 | 0 | 1 | 0 | 0 |
| **B** | 0 | 17 | 49 | 3 | 6 | 5 | 0 | 0 | 0 |
| **C** | 0 | 11 | 3 | 42 | 0 | 2 | 0 | 0 | 0 |
| **D** | 0 | 9 | 3 | 9 | 27 | 2 | 0 | 0 | 0 |
| **E** | 0 | 7 | 4 | 3 | 1 | 20 | 0 | 0 | 0 |
| **F** | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **G** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| **H** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |

## Interpretation

- Accuracy delta is within ±3 pp — the two runtimes are **effectively interchangeable** at this sample size. Cluster runs alone are sufficient for the rest of the pilot.
