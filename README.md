# mamai-eval

Evaluation harness for the [MAM-AI](https://github.com/nmrenyi/mamai) medical assistant app.

Covers generation quality, retrieval quality, on-device latency, and safety across versioned app configs.

## ⚠ Safety evaluation is the highest priority

MAM-AI is used by nurses and midwives making real clinical decisions. A safety score of 1 (the lowest rating) on any response must be flagged and resolved before a config can be released. Safety results are a mandatory gate — not an optional metric.

## Structure

```
configs/              versioned app configs — each with its own eval results
  config-v0.x.y/
    system_en.txt     English system prompt (same text as deployed app)
    system_sw.txt     Swahili system prompt
    mcq_system.txt    MCQ adapter prompt
    params.json       generation + retrieval + judge params
    results/
      safety/         *** safety-specific evaluation — must pass before release ***
        <model>/
      retrieval/      retrieval quality metrics
      generation/     per-model generation quality (MCQ accuracy, open-ended judge scores)
        <model>/
      latency/        on-device latency benchmarks
        <model>/
  exp/                experimental configs — never released

datasets/             benchmark datasets (TSV)
cluster/              RunAI cluster submission scripts
tests/
```

## Config versioning

Released configs are tagged on this repo (e.g. `config-v0.1.0`) and published as GitHub releases.
The MAM-AI app repo pins the active config version in `app_config.lock.json`.

A config under `configs/config-v*` is immutable after its release tag is created.
Experimental work goes under `configs/exp/`.

## Running eval

```bash
python run_eval.py --config config-v0.1.0 --model gemma4-e4b --datasets afrimedqa_mcq
python run_eval.py --config config-v0.1.0 --model gpt-5 --datasets all --judge
```

Results are written to `configs/<config>/results/generation/<model>/`.
