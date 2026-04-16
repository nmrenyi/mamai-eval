"""Validate config folder schemas on every push.

Catches the class of bug where params.json is missing a required key,
has a malformed value, or a prompt file is empty — all of which would
silently break the eval harness or produce misleading scores.

Each configs/config-v*/ folder is validated independently so a broken
experimental config doesn't hide a valid one.
"""

import json
import re
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parents[1] / "configs"


def _versioned_configs() -> list[Path]:
    """Return all configs/config-v*/ directories (excludes configs/exp/)."""
    return sorted(
        d for d in _CONFIGS_DIR.iterdir()
        if d.is_dir() and re.match(r"config-v\d+\.\d+\.\d+", d.name)
    )


# ── At least one config must exist ───────────────────────────────────────────

def test_at_least_one_versioned_config():
    configs = _versioned_configs()
    assert configs, f"No config-v* directories found under {_CONFIGS_DIR}"


# ── params.json ───────────────────────────────────────────────────────────────

def _params(cfg_dir: Path) -> dict:
    return json.loads((cfg_dir / "params.json").read_text())


def test_params_parses_as_json():
    for cfg in _versioned_configs():
        _params(cfg)  # raises if malformed


def test_params_top_level_sections():
    for cfg in _versioned_configs():
        p = _params(cfg)
        for section in ("generation", "retrieval", "context_injection", "judge"):
            assert section in p, f"{cfg.name}/params.json missing section '{section}'"


def test_params_generation_values():
    for cfg in _versioned_configs():
        gen = _params(cfg)["generation"]
        assert 0.0 < gen["temperature"] <= 2.0, f"{cfg.name}: temperature out of range"
        assert 0.0 < gen["top_p"] <= 1.0, f"{cfg.name}: top_p out of range"
        assert isinstance(gen["top_k"], int) and gen["top_k"] > 0, f"{cfg.name}: top_k invalid"
        assert isinstance(gen["n_ctx"], int) and gen["n_ctx"] > 0, f"{cfg.name}: n_ctx invalid"
        assert isinstance(gen["max_tokens"], int) and gen["max_tokens"] > 0, f"{cfg.name}: max_tokens invalid"


def test_params_retrieval_top_k_positive():
    for cfg in _versioned_configs():
        top_k = _params(cfg)["retrieval"]["top_k"]
        assert isinstance(top_k, int) and top_k > 0, f"{cfg.name}: retrieval.top_k must be a positive int"


def test_params_context_injection_labels_nonempty():
    for cfg in _versioned_configs():
        ci = _params(cfg)["context_injection"]
        for key in ("context_label_en", "question_label_en"):
            assert ci.get(key, "").strip(), f"{cfg.name}: context_injection.{key} is empty"


def test_params_judge_temperature_in_range():
    for cfg in _versioned_configs():
        t = _params(cfg)["judge"]["temperature"]
        assert 0.0 <= t <= 2.0, f"{cfg.name}: judge.temperature {t} out of range [0, 2]"


def test_params_judge_model_nonempty():
    for cfg in _versioned_configs():
        model = _params(cfg)["judge"].get("model", "")
        assert model.strip(), f"{cfg.name}: judge.model is empty"


# ── Prompt files ──────────────────────────────────────────────────────────────

def test_prompt_files_exist():
    for cfg in _versioned_configs():
        for fname in ("system_en.txt", "system_sw.txt", "mcq_system.txt"):
            path = cfg / fname
            assert path.exists(), f"{cfg.name}/{fname} is missing"


def test_prompt_files_nonempty():
    for cfg in _versioned_configs():
        for fname in ("system_en.txt", "system_sw.txt", "mcq_system.txt"):
            content = (cfg / fname).read_text(encoding="utf-8").strip()
            assert len(content) > 50, f"{cfg.name}/{fname} looks suspiciously short ({len(content)} chars)"


# ── Results directory structure ───────────────────────────────────────────────

def test_results_subdirs_exist():
    for cfg in _versioned_configs():
        results = cfg / "results"
        assert results.exists(), f"{cfg.name}/results/ directory is missing"
        for subdir in ("safety", "generation", "retrieval", "latency"):
            assert (results / subdir).exists(), f"{cfg.name}/results/{subdir}/ directory is missing"
