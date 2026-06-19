# G1/G2 prompt A/B on Gemma 3n — no-RAG SAQ + rubric

**Status:** complete (2026-06-19). Full writeup: [`report.html`](report.html). All numbers: [`summary.json`](summary.json).

Three-arm prompt A/B isolating the G1 (deflection-fix) and G2-lever-1 (structure) prompt
revisions on the now-deployed **Gemma 3n E4B** generator, **no-RAG**. Plan:
[`docs/g1-prompt-fix-plan.md`](../../../../../docs/g1-prompt-fix-plan.md).

| Arm | Prompt | Isolates |
|---|---|---|
| arm1 | baseline `system_en.txt` | 3n no-RAG baseline |
| arm2 | + G1 deflection levers (`prompts/arm2_system_en.txt`) | arm2 − arm1 = deflection-fix effect |
| arm3 | arm2 + G2 consultation skeleton (`prompts/arm3_system_en.txt`) | arm3 − arm2 = structure effect |

## Verdict — do not ship G1/G2 as-is on 3n

Helpfulness rises but **safety fails the gate on the kenya decision set:**

| kenya (n=312) | arm1 | arm2 (+G1) | arm3 (+G1+G2) |
|---|--:|--:|--:|
| key-fact recall | 0.288 | 0.365 | **0.403** |
| deflection | 1.9% | 0% | 0% |
| harm rate | 22.4% | 34.0% | **35.3%** |
| dangerous | 4 | 7 | **9** |

Recall and harm rise together, monotonically (a dose–response curve). The most helpful arm is the
least safe. And the **premise shifted**: 3n's baseline deflection is already ~2% (the Gemma 4 → 3n
upgrade already fixed deflection), so the prompt's net effect is a **recall-for-safety trade**, not
deflection repair. HealthBench rubric is milder (completeness up, penalty flat) — but it's a different
question set; the free-form kenya `safety` enum governs the gate. See `report.html` §5 for the full
verdict + recommendations (re-scope to non-commission levers; pursue RAG grounding + RAG-context
faithfulness as the real safety path).

## Files
- `report.html` — full writeup. `summary.json` — every aggregate (incl. healthbench).
- `arm{1,2,3}/{kenya,afrimedqa_saq,whb}.json` — scored SAQ result rows (behavior + recall + safety).
- `prompts/arm{2,3}_system_en.txt` — the A/B prompt variants. arm1 = unmodified config prompt.
- HealthBench raw scored rows (~38 MB/arm) kept on PVC:
  `/lightscratch/users/yiren/eval_output/g1_ab_3n_20260619/arm{1,2,3}/run/healthbench_oss_eval.json`.

## Provenance
- Generation: `cluster/run_cluster_g1_arm.sh` (no-RAG, `gemma3n-e4b`, `--system-prompt` override).
  arm1 ran on H200 (won a 3-pool GPU race), arm2 H100, arm3 A100.
- Judging: `cluster/run_cluster_g1_judge.sh` — one gpt-oss-120b vLLM boot per arm, both tracks.
- Code: branch `feat/g1-prompt-fix-20260611`.
