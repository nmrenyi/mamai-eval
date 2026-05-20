# Track-based directory refactor

*Drafted 2026-05-20 while the v0.2 Gemma 4 +RAG cluster chain was running. Execute as a single atomic commit AFTER that chain completes — never while a cluster job is in flight, because cluster jobs `git clone --branch main --depth 1` at submit time and will 404 on moved files.*

## Motivation

The repo today is flat: 14+ Python files at the root, all originally built for one eval track ("generation quality" — MCQ + open-ended + rubric). The `configs/<v>/results/` tree is already organised by track (`generation/`, `retrieval/`, `latency/`, `safety/`), but the code is not. As we add **retrieval evaluation** and **generator faithfulness evaluation** (two new tracks), the flat layout will get worse.

This refactor makes the code mirror the results layout: one folder per evaluation track, with a small `shared/` for cross-cutting infrastructure.

## Target layout

```
mamai-eval/
├── end_to_end_eval/                  # whole-system eval: model → score
│   ├── run_eval.py
│   ├── run_eval_device.py
│   ├── rescore_mcq.py                # MCQ scoring
│   ├── rescore_open_v2.py            # open-ended (3-judge ensemble)
│   ├── rescore_rubric.py             # HealthBench-style rubric
│   ├── safety_eval.py                # NEW placeholder for v0.3 safety
│   └── legacy/
│       └── rescore_open.py           # v0.1 single-judge fallback
│
├── retrieval_eval/                   # retrieval-only quality metrics
│   ├── precompute_retrieval.py       # produces rag_contexts/<v>/<dataset>.json
│   ├── retrieval.py                  # Gecko embedder + vector store reader
│   └── eval_retrieval.py             # NEW: precision@k / recall@k / MRR
│
├── generator_eval/                   # Gemma 4 faithfulness vs retrieved context
│   └── eval_faithfulness.py          # NEW
│
├── latency_eval/                     # on-device latency benchmark
│   └── benchmark_latency.py
│
├── shared/                           # imported by ≥2 tracks
│   ├── inference.py                  # model backends (Llama-cpp, OpenAI, …)
│   ├── prompts.py                    # prompt templates + params.json loader
│   ├── scoring.py                    # MCQ extract_letters + judge utilities
│   └── datasets.py                   # NEW: HF loader + _normalize_row
│                                     #      (extracted from run_eval.py so faithfulness
│                                     #       eval can load the same row set)
│
├── calibration/                      # device-vs-cluster comparison tool
│   ├── prepare_calibration_set.py
│   └── compare_calibration.py
│
├── configs/                          # unchanged
├── cluster/                          # unchanged structurally, paths updated
├── tests/                            # adjusted import paths
├── docs/                             # adjusted
├── Dockerfile, requirements.txt, README.md
```

## File-by-file move table

| Current path | New path | Notes |
|---|---|---|
| `run_eval.py` | `end_to_end_eval/run_eval.py` | extract `_normalize_row` + `_load_dataset` to `shared/datasets.py` |
| `run_eval_device.py` | `end_to_end_eval/run_eval_device.py` | |
| `rescore_mcq.py` | `end_to_end_eval/rescore_mcq.py` | |
| `rescore_open_v2.py` | `end_to_end_eval/rescore_open_v2.py` | |
| `rescore_rubric.py` | `end_to_end_eval/rescore_rubric.py` | |
| `rescore_open.py` | `end_to_end_eval/legacy/rescore_open.py` | v0.1 fallback, preserved for reproducibility |
| `inference.py` | `shared/inference.py` | |
| `prompts.py` | `shared/prompts.py` | |
| `scoring.py` | `shared/scoring.py` | |
| `precompute_retrieval.py` | `retrieval_eval/precompute_retrieval.py` | |
| `retrieval.py` | `retrieval_eval/retrieval.py` | grep confirmed: only `precompute_retrieval.py` imports it |
| `benchmark_latency.py` | `latency_eval/benchmark_latency.py` | |
| `prepare_calibration_set.py` | `calibration/prepare_calibration_set.py` | |
| `compare_calibration.py` | `calibration/compare_calibration.py` | |

Plus new stub files (track placeholders, empty `main()`):

| New file | Purpose |
|---|---|
| `end_to_end_eval/safety_eval.py` | v0.3 safety eval per `docs/v0.2-evaluation-handoff.md` |
| `retrieval_eval/eval_retrieval.py` | precision@k / recall@k / MRR on retrieved chunks |
| `generator_eval/eval_faithfulness.py` | per-claim context-support rate for Gemma 4 output |

## Import-path changes (~30 lines across the codebase)

Old form → new form (mechanical search/replace):

```
from prompts import …              →  from shared.prompts import …
from inference import …            →  from shared.inference import …
from scoring import …              →  from shared.scoring import …
from retrieval import …            →  from retrieval_eval.retrieval import …
from run_eval import …             →  from end_to_end_eval.run_eval import …
```

Each track-folder + `shared/` + `calibration/` gets an `__init__.py` so they're proper Python packages.

## Cluster-script changes

`cluster/run_cluster.sh` (and `run_cluster_precompute.sh`, `run_cluster_gemma4_e4b_open.sh`):

```diff
- python3 run_eval.py "${DATASET_ARGS[@]}"
+ python3 -m end_to_end_eval.run_eval "${DATASET_ARGS[@]}"

- python3 precompute_retrieval.py "${DATASET_ARGS[@]}"
+ python3 -m retrieval_eval.precompute_retrieval "${DATASET_ARGS[@]}"
```

Using `python3 -m <module>` from `$WORKTREE` keeps imports clean — no `sys.path` manipulation needed.

## Test changes

`tests/test_imports.py` and `tests/test_config_schema.py` reference modules by name. Update imports:

```diff
- import scoring
+ from shared import scoring

- import prompts
+ from shared import prompts
```

`tests/test_score_mcq.py` and `tests/test_scoring.py` likely also need the same shift.

## Code extractions (small, alongside the moves)

**`shared/datasets.py`** — extract from `run_eval.py`:
- `HF_CONFIGS` dict
- `_letter_for_index`, `_format_choices`
- `_load_rubric_criteria`
- `_normalize_row`
- `_load_dataset`

These are needed by both `end_to_end_eval/run_eval.py` and the new `generator_eval/eval_faithfulness.py` (which needs to load the same row set to feed Gemma).

Everything else stays in `run_eval.py`.

## When to execute

**Prerequisite**: no in-flight cluster jobs. Specifically wait for `mamai-v02-rag-precompute-cpu32` and the chained `mamai-v02-gemma-mcq-rag` to both complete.

**Sequence**:

1. Confirm prerequisite — both jobs Succeeded, results pulled.
2. Single atomic commit `refactor: track-based directory layout`:
   - `git mv` each file (preserves history)
   - Create new package dirs with `__init__.py`
   - Rewrite imports
   - Update cluster scripts
   - Update tests
   - Create stub `safety_eval.py`, `eval_retrieval.py`, `eval_faithfulness.py`
3. Run `pytest tests/` — should pass green.
4. Push.
5. Verify a smoke-style cluster submission still works (e.g. afrimedqa max-questions 5) to catch any missed import-path issue before the next big run.

## Locked-in design decisions

- `end_to_end_eval/` is **flat** — MCQ + open-ended + rubric + safety files live alongside each other, no sub-folders by set_type.
- `retrieval.py` lives in `retrieval_eval/` (only consumer is `precompute_retrieval.py`).
- `generator_eval/` may import from `shared/` (inference + prompts) — no duplication.
- `latency_eval/` is its own top-level bucket — measures performance, not quality, so distinct from the other three.
- `calibration/` is a cross-cutting tool, not a track — it compares results between venues for any track.
- `shared/datasets.py` is a new file extracted from `run_eval.py` for the HF loader + normalisation.
- Backward-compat shims at root (e.g. `run_eval.py` re-exporting from `end_to_end_eval/`) — **not done**; clean break is healthier than parallel paths.

## Future considerations (not part of this refactor)

- **Per-track requirements files** (e.g. `requirements-judges.txt` for `anthropic`+`google-genai`+`openai`) if dependency footprints diverge meaningfully.
- **Cluster image** — a pre-built mamai-eval container image with Python deps baked in, avoiding the per-job `apt-get update && pip install` cycle. Mentioned in handoff doc; out of scope here.
- **Backward-compat with v0.1 runs** — the refactor doesn't touch v0.1 result JSONs already on disk. v0.1 reproducibility lives in the git history (release tag `config-v0.1.0`), not the working tree.
