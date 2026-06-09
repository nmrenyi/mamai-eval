# mamai-eval

Evaluation harness for the [MAM-AI](https://github.com/nmrenyi/mamai) medical assistant app.

Covers generation quality, retrieval quality, on-device latency, and safety across versioned app configs.

> **Picking up the v0.2 evaluation cycle?** Start with [`docs/v0.2-evaluation-handoff.md`](docs/v0.2-evaluation-handoff.md) — it covers what changed since v0.1 (HF dataset loading, three set types including the new `open_ended_rubric`, the 3-judge ensemble for open-ended), what's already here vs what needs to be built, and the recommended build order.

## ⚠ Safety evaluation is the highest priority

MAM-AI is used by nurses and midwives making real clinical decisions. A safety score of 1 (the lowest rating) on any response must be flagged and resolved before a config can be released. Safety results are a mandatory gate — not an optional metric.

## Structure

```
configs/                  versioned app configs — each with its own eval results
  config-v0.x.y/
    system_en.txt         English system prompt (same text as deployed app)
    system_sw.txt         Swahili system prompt
    mcq_system.txt        MCQ adapter prompt
    params.json           generation + retrieval + judge params
    calibration/          row-id manifests for device-vs-cluster runs
    results/
      safety/             *** safety-specific evaluation — must pass before release ***
      retrieval/          retrieval quality metrics
      generation/         per-model generation quality (MCQ accuracy, open-ended judge scores)
      latency/            on-device latency benchmarks
    reports/              human-readable summary writeups
  exp/                    experimental configs — never released

end_to_end_eval/          track 1: whole-system runners + scorers (MCQ/open/rubric/safety)
retrieval_eval/           track 2: precompute retrieval + retrieval-quality metrics
generator_eval/           track 3: generator (Gemma) faithfulness vs retrieved context
latency_eval/             track 4: on-device latency benchmark via ADB

shared/                   cross-cutting: model backends, prompts, scoring, HF loader
calibration/              device-vs-cluster comparison tools
cluster/                  RunAI cluster submission scripts
tests/
docs/                     handoff + refactor + design docs
```

## Config versioning

Released configs are tagged on this repo (e.g. `config-v0.1.0`) and published as GitHub releases.
The MAM-AI app repo pins the active config version in `app_config.lock.json`.

A config under `configs/config-v*` is immutable after its release tag is created.
Experimental work goes under `configs/exp/`.

## Running eval

End-to-end generation runs are invoked as modules from the repo root:

```bash
# MCQ on cluster GPU (one of the v0.2 configs):
python -m end_to_end_eval.run_eval --config config-v0.2.0 --model gemma4-e4b \
    --datasets afrimedqa,medqa_usmle,medmcqa

# On the connected Android device (requires ADB + the eval-mode APK):
python -m end_to_end_eval.run_eval_device --config config-v0.2.0 \
    --datasets afrimedqa --max-questions 20

# RAG context precompute (needed before any +RAG eval):
python -m retrieval_eval.precompute_retrieval --config config-v0.2.0 \
    --db-path ... --gecko-model ... --tokenizer ... \
    --datasets afrimedqa,medqa_usmle,medmcqa

# Post-hoc rescoring on saved result JSONs:
python -m end_to_end_eval.rescore_mcq      configs/config-v0.2.0/results/end_to_end_eval/<model>/<ts>/
python -m end_to_end_eval.rescore_open_v2  configs/config-v0.2.0/results/end_to_end_eval/<model>/<ts>/
python -m end_to_end_eval.rescore_rubric   configs/config-v0.2.0/results/end_to_end_eval/<model>/<ts>/
```

Results land under `configs/<config>/results/end_to_end_eval/<model>/<ts>/`.
