# PLAN — generator × retriever matrix (kenya + healthbench-oss)

*Filed 2026-06-16. Self-contained execution plan; written to survive /compact. Re-read this
file, then execute. Goal: a unified **generator × retriever** table on kenya SAQ (key-fact recall)
and healthbench-oss (rubric overall / completeness+ / penalty−), merging the reranker line
(value_gate_matrix*) and the embedder line (value_gate_eg*). Reranker dimension is collapsed to
"none" (no rerank) — it was shown not to matter.*

## Repo / cluster facts (do not re-derive)
- Workdir: `/Users/renyi/Downloads/mamai-eval-r1-threshold`, branch `feat/r2-retriever-upgrade-20260613` (HEAD has all scripts; local==origin).
- Cluster: `RUNAI_LOCAL=1` + `cluster/submit_job.sh <jobname> <script.sh> KEY=VAL...`. Pools: `h200`,`h100`,`default`(A100-80G). Quota ~15 GPU. Spread across pools to dodge contention.
- **Path gotcha:** containers write `/lightscratch/users/yiren/...`; from `ssh light` the SAME path is `/mnt/light/scratch/users/yiren/...`. Use the latter for ssh/scp reads.
- **REPO_REF gotcha:** cluster scripts clone from GitHub. `run_cluster_value_gate_eg.sh` DEFAULTS to `REPO_REF=main` — MUST pass `REPO_REF=feat/r2-retriever-upgrade-20260613`. Any new script must be pushed before submit.
- Judge serving: `run_cluster_value_gate_judge.sh` serves gpt-oss-120b (TP=2) and runs `JUDGE_MODULE` over `RUN_DIRS` for each dataset in `SAQ_DS`. `JUDGE_MODULE=end_to_end_eval.rescore_open_v2` (kenya recall) or `end_to_end_eval.rescore_rubric` (healthbench, pass `RUBRIC_WORKERS=64`).
- **Waiter discipline:** run_eval/rescore checkpoint incrementally → gate completion waiters on FULL counts (kenya 312, healthbench_oss 1209; rubric: `aggregate_scores.n_scored>=1209`; SAQ recall: `aggregate_scores.mean_key_fact_recall is not None`). Use background `until ssh light '<py check>'; do sleep 90; done` waiters; never bare foreground sleep. Add a 3-min failure watchdog (grep runai for Failed) + ~10-min heartbeat (re-arm each fire) — the user wants continuous monitoring.

## Cells ALREADY HAVE (hard-code into the table; do not re-run)
kenya key-fact recall:  G4: no-RAG 0.178, Gecko 0.1254, EmbeddingGemma 0.1255, Hybrid 0.112, BM25 0.109 ; G3n: Gecko 0.263, Hybrid 0.265, BM25 0.271
healthbench overall / positive / penalty:
  G4:  Gecko −0.004/0.173/0.371 · EmbeddingGemma +0.012/0.189/0.368 · Hybrid −0.007/0.174/0.376 · BM25 +0.000/0.177/0.368
  G3n: Gecko 0.089/0.260/0.356 · Hybrid 0.091/0.264/0.363 · BM25 0.074/0.248/0.361
(Source: value_gate_matrix/* = G4 arms_matrix; value_gate_matrix_3n/* = G3n; value_gate_eg* = EmbeddingGemma×G4. "Gecko/Hybrid/BM25" = the `*__none` no-rerank arms.)

## GAPS TO RUN (3 generation jobs, then judge)
1. **EmbeddingGemma × Gemma 3n** (kenya + healthbench) — re-run the EG value-gate with gemma3n.
2. **no-RAG × Gemma 3n** (kenya + healthbench).
3. **no-RAG × Gemma 4** (healthbench only; kenya already 0.178).

## WAVE 1 — generation (launch all 3 in parallel; ~80-90 min, healthbench is the long pole)
From `cd cluster`, `export SUPPRESS_DEPRECATION_MESSAGE=true`:

```
# J-C  EmbeddingGemma × gemma3n  (reuses precomputed EG retrievals; OUT_DIR distinct from the G4 run)
NODE_POOL=h200 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
  ./submit_job.sh mamai-eg-3n run_cluster_value_gate_eg.sh \
  MODEL=gemma3n-e4b DATASETS=kenya,healthbench_oss_eval \
  OUT_DIR=/lightscratch/users/yiren/eval_output/value_gate_eg_3n \
  REPO_REF=feat/r2-retriever-upgrade-20260613

# J-A  no-RAG × gemma3n  (kenya + healthbench)
NODE_POOL=h100 GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
  ./submit_job.sh mamai-norag-3n run_cluster_norag_gen.sh \
  MODEL=gemma3n-e4b DATASETS=kenya,healthbench_oss_eval \
  OUT_DIR=/lightscratch/users/yiren/eval_output/norag_gemma3n

# J-B  no-RAG × gemma4  (healthbench only)
NODE_POOL=default GPU_REQUEST=1 MEMORY_REQUEST=64G CPU_REQUEST=8 RUNAI_LOCAL=1 \
  ./submit_job.sh mamai-norag-g4 run_cluster_norag_gen.sh \
  MODEL=gemma4-e4b DATASETS=healthbench_oss_eval \
  OUT_DIR=/lightscratch/users/yiren/eval_output/norag_gemma4
```
Health-check each (llama-cpp build ~8 min, then "Loading GGUF" / "Dataset:" / "Open inference"). For J-C confirm `screen_embedder arm_format` built the arm before generation.

Output run dirs (host paths for ssh/scp):
- EG×3n:    `/mnt/light/scratch/users/yiren/eval_output/value_gate_eg_3n/embeddinggemma/run/{kenya,healthbench_oss_eval}.json`
- norag×3n: `/mnt/light/scratch/users/yiren/eval_output/norag_gemma3n/run/{kenya,healthbench_oss_eval}.json`
- norag×g4: `/mnt/light/scratch/users/yiren/eval_output/norag_gemma4/run/healthbench_oss_eval.json`

## WAVE 2 — judging (after the relevant gens hit full counts; 2 jobs, parallel; spread pools)
```
# Kenya recall (rescore_open_v2) — the 2 new kenya arms (EG×3n, norag×3n)
NODE_POOL=h100 GPU_REQUEST=2 MEMORY_REQUEST=128G CPU_REQUEST=12 RUNAI_LOCAL=1 \
  ./submit_job.sh mamai-mtx2-kenya-judge run_cluster_value_gate_judge.sh \
  "RUN_DIRS=/lightscratch/users/yiren/eval_output/value_gate_eg_3n/embeddinggemma/run /lightscratch/users/yiren/eval_output/norag_gemma3n/run" \
  SAQ_DS=kenya JUDGE_MODULE=end_to_end_eval.rescore_open_v2

# Healthbench rubric (rescore_rubric, 64 workers) — the 3 new hb arms (EG×3n, norag×3n, norag×g4)
NODE_POOL=h200 GPU_REQUEST=2 MEMORY_REQUEST=128G CPU_REQUEST=12 RUNAI_LOCAL=1 \
  ./submit_job.sh mamai-mtx2-hb-judge run_cluster_value_gate_judge.sh \
  "RUN_DIRS=/lightscratch/users/yiren/eval_output/value_gate_eg_3n/embeddinggemma/run /lightscratch/users/yiren/eval_output/norag_gemma3n/run /lightscratch/users/yiren/eval_output/norag_gemma4/run" \
  SAQ_DS=healthbench_oss_eval JUDGE_MODULE=end_to_end_eval.rescore_rubric RUBRIC_WORKERS=64
```
Read results via ssh-python from `aggregate_scores`: SAQ → `mean_key_fact_recall`; rubric → `mean_weighted_met` / `mean_positive_score` / `mean_penalty_rate`. If scp blocked by `nobody:0600`, `ssh light 'chmod -R a+rX <hostpath>'` first, or just read values via ssh-python.

## WAVE 3 — assemble + commit
Build the **generator × retriever** table (rows: no-RAG, Gecko, EmbeddingGemma, Hybrid, BM25; cols: Gemma 4 | Gemma 3n) for BOTH kenya recall AND healthbench (overall/+/−). Add it as a new section to
`configs/config-v0.2.0/reports/r2c-rerank/r2c-reranker-results-20260613.html` (or the embedder report) — pick the reranker report (it already has the generator-comparison section). Read down a column = retriever effect (small); across a row = generator effect (large). Commit + push (Co-Authored-By: Claude Opus 4.8). Then clean up finished jobs.

## Expected read (hypothesis to confirm/refute)
G3n ≈ 2× G4 on kenya recall for every retriever; no-RAG likely ≥ RAG for G4 (RAG net-negative) — check if 3n also shows RAG net-negative or whether 3n actually benefits from RAG (key open question this matrix answers). Healthbench: 3n higher completeness across the board.
