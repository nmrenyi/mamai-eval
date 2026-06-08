"""Tests for calibration/judge_validation/metrics.py.

The decisive Phase-1 check is `test_human_human_baseline_reproduces_77_6` —
it confirms the metric module, run on the real obgyn_meta_eval.jsonl,
reproduces the 77.6% / kappa 0.366 human baseline we computed earlier.
That number is what every judge candidate will be measured against, so
the math has to be right before any judge runs.

Skips gracefully if the calibration file hasn't been fetched yet.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from calibration.judge_validation import fetch, metrics

_CACHE = fetch.DEFAULT_CACHE / fetch.DEFAULT_FILENAME


def _require_calibration() -> Path:
    if not _CACHE.exists():
        pytest.skip(
            f"Calibration file not fetched. Run: "
            f"python -m calibration.judge_validation fetch"
        )
    return _CACHE


# ── Synthetic mini-fixture (deterministic, doesn't need the HF download) ─────


def _mini_rows() -> list[dict]:
    """3 concordant rows + 1 discordant row. Mixed categories.

    Indices 0-2 concordant (2 met, 1 not-met). Index 3 discordant.
    """
    rows = [
        # 0: 2 physicians both met, MATERNAL
        {"binary_labels": [True, True],   "mamabench_obgyn_category": "MATERNAL"},
        # 1: 2 physicians both met, MATERNAL
        {"binary_labels": [True, True],   "mamabench_obgyn_category": "MATERNAL"},
        # 2: 2 physicians both not-met, NEONATAL
        {"binary_labels": [False, False], "mamabench_obgyn_category": "NEONATAL"},
        # 3: discordant, MATERNAL
        {"binary_labels": [True, False],  "mamabench_obgyn_category": "MATERNAL"},
    ]
    for i, r in enumerate(rows):
        r["_orig_idx"] = i
    return rows


# ── Sanity on the synthetic fixture ──────────────────────────────────────────


def test_split_concordant_synthetic():
    c, d = metrics.split_concordant(_mini_rows())
    assert len(c) == 3
    assert len(d) == 1
    # _orig_idx is preserved through the split
    assert sorted(r["_orig_idx"] for r in c) == [0, 1, 2]
    assert [r["_orig_idx"] for r in d] == [3]


def test_human_human_synthetic():
    # rows 0,1,2 each contribute 1 pair (all agree)
    # row 3 contributes 1 pair (disagree)
    # → 3 of 4 pairs agree → 75%
    result = metrics.human_human_agreement(_mini_rows())
    assert result["pairs"] == 4
    assert result["agreement"] == pytest.approx(0.75, abs=1e-4)


# ── The decisive Phase-1 check: real-data human baseline ─────────────────────


def test_human_human_baseline_reproduces_77_6():
    rows = metrics.load_rows(_require_calibration())
    h = metrics.human_human_agreement(rows)
    assert h["pairs"] == 7777
    assert abs(h["agreement"] - 0.776) < 0.002, (
        f"Expected ~77.6% but got {h['agreement']}"
    )
    assert abs(h["kappa"] - 0.366) < 0.005, (
        f"Expected kappa ~0.366 but got {h['kappa']}"
    )


def test_concordant_split_reproduces_5241_1612():
    rows = metrics.load_rows(_require_calibration())
    c, d = metrics.split_concordant(rows)
    assert len(c) == 5241
    assert len(d) == 1612


def test_concordant_subset_is_85pc_met():
    """The rubber-stamp baseline: concordant subset is 85% met / 15% not-met."""
    rows = metrics.load_rows(_require_calibration())
    c, _ = metrics.split_concordant(rows)
    n_met = sum(1 for r in c if r["binary_labels"][0])
    frac = n_met / len(c)
    assert abs(frac - 0.85) < 0.01


# ── Synthetic verdict regimes — pins down what each metric means ─────────────


def test_perfect_oracle_concordant_agreement_is_100():
    """Oracle = first physician's label.

    On concordant rows the first physician's label IS the consensus, so
    `llm_vs_consensus` agreement must be exactly 100%.
    """
    rows = metrics.load_rows(_require_calibration())
    verdicts = {r["_orig_idx"]: r["binary_labels"][0] for r in rows}
    c, _ = metrics.split_concordant(rows)
    cons = metrics.llm_vs_consensus(c, verdicts)
    assert cons["agreement"] == 1.0
    # judge met-rate equals consensus met-rate on concordant rows ≈ 85%
    assert abs(cons["judge_met_rate"] - 0.85) < 0.01


def test_rubber_stamp_detector_flags_always_met():
    """Always-met judge: high overall agreement on concordant (85% met
    population) BUT 100% met-rate (vs physician 77%) and 0% agreement on
    physician-not-met rows. Detector must surface this."""
    rows = metrics.load_rows(_require_calibration())
    verdicts = {r["_orig_idx"]: True for r in rows}
    c, _ = metrics.split_concordant(rows)
    cons = metrics.llm_vs_consensus(c, verdicts)
    pred = metrics.prediction_distribution(rows, verdicts)

    # Looks fine if you only read the headline
    assert abs(cons["agreement"] - 0.85) < 0.01
    # But the rubber-stamp signals are screaming:
    assert pred["judge_met_rate"] == 1.0
    assert abs(pred["physician_met_rate"] - 0.77) < 0.01
    assert cons["agreement_on_met_rows"] == 1.0
    assert cons["agreement_on_not_met_rows"] == 0.0


def test_random_baseline_is_about_50pc():
    """Random Bernoulli(0.5) judge → ~50% LLM-vs-human regardless of base rate."""
    rng = random.Random(42)
    rows = metrics.load_rows(_require_calibration())
    verdicts = {r["_orig_idx"]: rng.random() > 0.5 for r in rows}
    sh = metrics.llm_vs_single_human(rows, verdicts)
    assert 0.46 < sh["agreement"] < 0.54


# ── Per-category sanity ──────────────────────────────────────────────────────


def test_per_category_covers_four_categories():
    """The 5,241 concordant rows should split across all 4 mamabench categories."""
    rows = metrics.load_rows(_require_calibration())
    verdicts = {r["_orig_idx"]: True for r in rows}  # rubber stamp, just to populate
    by_cat = metrics.per_category(
        rows,
        verdicts,
        lambda rs, vs: metrics.llm_vs_consensus(
            [r for r in rs if len(set(r["binary_labels"])) == 1], vs
        ),
    )
    assert set(by_cat) == {
        "MATERNAL", "CHILD_HEALTH", "SEXUAL_AND_REPRODUCTIVE_HEALTH", "NEONATAL",
    }
    total_n = sum(c["n"] for c in by_cat.values())
    assert total_n == 5241


# ── Bootstrap sanity ─────────────────────────────────────────────────────────


def test_bootstrap_ci_brackets_point_estimate():
    """Bootstrap CI should contain the point estimate."""
    rows = metrics.load_rows(_require_calibration())
    verdicts = {r["_orig_idx"]: r["binary_labels"][0] for r in rows}  # oracle
    sh = metrics.llm_vs_single_human(rows, verdicts)
    lo, hi = metrics.bootstrap_ci(
        rows, verdicts, metrics.llm_vs_single_human,
        key="agreement", n_resamples=200, seed=0,
    )
    assert lo <= sh["agreement"] <= hi


# ── Markdown renderer doesn't crash ──────────────────────────────────────────


def test_render_markdown_smoke():
    rows = metrics.load_rows(_require_calibration())
    verdicts = {r["_orig_idx"]: r["binary_labels"][0] for r in rows}
    report = metrics.full_report(rows, verdicts, judge_model="test-oracle")
    md = metrics.render_markdown(report)
    assert "Judge validation" in md
    assert "Human ↔ human" in md
    assert "Rubber-stamp detector" in md
