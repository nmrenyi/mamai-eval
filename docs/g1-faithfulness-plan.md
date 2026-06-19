# G1/G2 Faithfulness Eval (Option B) — oracle-context faithfulness A/B on Gemma 3n

*Filed 2026-06-19 · branch `feat/g1-prompt-fix-20260611` · parallels [`oracle-v0.2.0-faithfulness.html`](../configs/config-v0.2.0/reports/oracle-v0.2.0-faithfulness.html)*

## Goal & framing

The G1/G2 SAQ A/B ([`g1-ab-3n-20260619`](../configs/config-v0.2.0/results/end_to_end_eval/g1-ab-3n-20260619/report.html))
showed recall ↑ / harm ↑ on **no-RAG** kenya, but could not measure **faithfulness** (no context to be
faithful to). This eval closes that gate: re-run the Track-3 oracle-faithfulness pipeline — **with known-good
context** — swapping in **Gemma 3n** + the three arm prompts, so we can answer *"does the G1 prompt make 3n
hallucinate more against retrieved guideline context?"* Deployment-relevant: the shipped system is RAG.

| | prior report | this eval |
|---|---|---|
| generator | Gemma 4 E4B | **Gemma 3n E4B** |
| prompt | deployed `system_en.txt` | **arm1 / arm2 / arm3** (same files as the SAQ A/B) |
| oracle | mamaretrieval v0.2.0, top-3 ≥5 (2,989 q) | same |
| scorer | Patronus Lynx 70B | same |
| categorize + calibrate | gpt-5 batch (medium / high) | same |

**Key confound (why Option B, not raw Lynx):** Lynx's FAIL bucket is ~57% non-hallucination (omission/refusal).
G1 *eliminates refusal*, mechanically lowering those FAILs while potentially raising the real-hallucination
buckets (contradiction + unsupported_addition). So raw Lynx FAIL deltas are confounded — we need the gpt-5
**categorize** step to isolate real hallucination, and the **calibrate** step to also correct Lynx's ~8%
PASS-side miss rate and produce an absolute calibrated rate comparable to the prior 9.05% headline.

## Pipeline (5 stages, scripts mapped)

| # | Stage | Script | Where | Cost |
|---|---|---|---|---|
| F1 | Generate (3n, oracle ctx, per arm) | `generator_eval/eval_faithfulness.py` (+ `--system-prompt` override, to add) | cluster GPU ×3 | ~2,989 rows/arm |
| F2 | Lynx 70B PASS/FAIL | `generator_eval/score_lynx.py` (one boot scores all 3 arms) | cluster, 2 GPUs | ~2,989 ×3 scorings |
| F3 | Categorize FAILs | `analyze_lynx_fails.py extract` → `score_openai_batch_judge.py categorize` (gpt-5 medium) → `analyze_lynx_fails.py aggregate` | gpt-5 Batch API | only FAIL rows/arm |
| F4 | Calibrate Lynx | `calibrate.py sample` → `score_openai_batch_judge.py calibrate` (gpt-5 high) → `calibrate.py score` | gpt-5 Batch API | ~96 blind rows/arm |
| F5 | HTML report | new `report-faithfulness.html` in the experiment dir | local | — |

## Execution plan

**F0 — wire (local, ~10 lines):** add `--system-prompt <path>` to `eval_faithfulness.py` (identical pattern to
`run_eval.py`: override `shared.prompts.OPEN_SYSTEM_PROMPT` before generation; it already generates via
`build_rag_open_prompt`, which reads that global). Commit + push (cluster clones the branch).

**F1 — generate, 3 arms in parallel** (`run_cluster_faithfulness.sh`, `MODEL=gemma3n-e4b`,
`ORACLE=.../mamaretrieval-v0.2.0-score5.jsonl`, `SYSTEM_PROMPT=<arm>`): race across H200/H100/A100 like the SAQ
run; per-arm output `oracle_responses.json` on PVC. ~1.5–2 h/arm.

**F2 — Lynx, one job, all 3 arms** (combined cluster script, 1× Lynx-70B vLLM boot, TP=2): score arm1→arm2→arm3
sequentially to amortize the ~140 GB model load. Writes `lynx_scored.json` per arm. Heavy (2 GPUs).

**F3 — categorize** (per arm, gpt-5 Batch, medium reasoning): extract FAIL cases → submit batch → on completion
aggregate to bucket counts → real-hallucination rate (contradiction + unsupported_addition).

**F4 — calibrate** (per arm, gpt-5 Batch, high reasoning): draw stratified blind sample (~96: PASS + FAIL by
bucket) → submit batch → score → confusion matrix → Lynx precision + miss rate → **calibrated true-hallucination
rate + CI** per arm.

**F5 — report:** 3-arm faithfulness comparison (raw Lynx PASS/FAIL · categorized real-hallucination ·
calibrated rate ± CI), against the prior Gemma-4 baseline (raw 5.82% / calibrated 9.05%). Verdict: does engaging
more (G1/G2) cost faithfulness to context?

## Decisions / caveats

- **Different distribution from the SAQ arms** — this is oracle (with-context); faithfulness only has meaning
  with context. Complements (does not replace) the no-RAG SAQ harm signal.
- **Calibration scope:** full per-arm calibration = 3× the gpt-5-high cost and 3 wide-CI numbers (sample
  n~96/arm → band like the prior 3–17%). Cheaper alt: calibrate only the **extremes** (arm1 baseline + arm3
  most-engaged) to bound the effect, use categorized-FAIL deltas for arm2. **Default: per-arm (user chose B).**
- **gpt-5 Batch latency:** up to a 24 h window (usually faster); F3/F4 are async — submit, then poll/`wait`.
- **Lynx GPU:** 2 GPUs/job; one-boot-three-arms keeps it to a single 2-GPU job.
- Gemma 3n is the Q4_0 cluster proxy for on-device int4 (direction robust, magnitudes shift).
