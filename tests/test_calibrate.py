"""Tests for calibration math in generator_eval/calibrate.py.

`cmd_score` is the headline calculation: confusion matrix, precision, miss
rate, and population estimate of true hallucinations. The acceptance run
against v0.1.0 produced 50/0/47/3 and ~0.3% — these tests pin the math on
synthetic inputs so a refactor can't silently break it.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from generator_eval.calibrate import cmd_score


def _write_run_dir(tmp_path, key, verdicts, lynx_full):
    """Lay out the three files cmd_score reads in a fresh run dir."""
    (tmp_path / "calibration_key.json").write_text(json.dumps(key))
    (tmp_path / "lynx_scores.json").write_text(json.dumps({"results": lynx_full}))
    verdicts_path = tmp_path / "verdicts.json"
    verdicts_path.write_text(json.dumps(verdicts))
    return verdicts_path


def test_confusion_matrix_and_precision(tmp_path, capsys):
    # 4-row sample, one cell per quadrant — easiest to read off the math.
    key = [
        {"query_id": "a", "lynx_score": "PASS", "lynx_category": None},
        {"query_id": "b", "lynx_score": "PASS", "lynx_category": None},
        {"query_id": "c", "lynx_score": "FAIL", "lynx_category": "contradiction"},
        {"query_id": "d", "lynx_score": "FAIL", "lynx_category": "contradiction"},
    ]
    verdicts = [
        {"query_id": "a", "verdict": "PASS"},   # PASS / PASS
        {"query_id": "b", "verdict": "FAIL"},   # PASS / FAIL — Lynx miss
        {"query_id": "c", "verdict": "PASS"},   # FAIL / PASS — false alarm
        {"query_id": "d", "verdict": "FAIL"},   # FAIL / FAIL
    ]
    # Population: arbitrary — pick numbers that make the est easy to verify.
    lynx_full = (
        [{"query_id": f"p{i}", "score": "PASS"} for i in range(100)]
        + [{"query_id": f"f{i}", "score": "FAIL"} for i in range(20)]
    )
    verdicts_path = _write_run_dir(tmp_path, key, verdicts, lynx_full)

    cmd_score(tmp_path, verdicts_path)

    report = json.loads((tmp_path / "calibration_report.json").read_text())

    cm = report["confusion_matrix"]
    assert cm["lynx_PASS_ref_PASS"] == 1
    assert cm["lynx_PASS_ref_FAIL"] == 1   # missed
    assert cm["lynx_FAIL_ref_PASS"] == 1   # false alarm
    assert cm["lynx_FAIL_ref_FAIL"] == 1

    # precision = 1/2, miss = 1/2 on this toy sample.
    assert report["lynx_precision"] == 0.5
    assert report["lynx_miss_rate"] == 0.5

    # Population estimate = n_fail * prec + n_pass * miss = 20*0.5 + 100*0.5 = 60.
    assert report["estimated_true_hallucinations"] == pytest.approx(60.0)
    assert report["estimated_true_hallucination_rate"] == pytest.approx(60 / 120, abs=1e-4)


def test_perfect_precision_zero_misses(tmp_path):
    # The v0.1.0-shape calibration: PASS stratum has zero ref-FAILs, FAIL
    # stratum has very few ref-FAILs. Pin both edges (0/n and small/n).
    key = (
        [{"query_id": f"p{i}", "lynx_score": "PASS", "lynx_category": None}
         for i in range(10)]
        + [{"query_id": f"f{i}", "lynx_score": "FAIL", "lynx_category": "contradiction"}
           for i in range(10)]
    )
    verdicts = (
        [{"query_id": f"p{i}", "verdict": "PASS"} for i in range(10)]
        + [{"query_id": f"f{i}", "verdict": "FAIL" if i == 0 else "PASS"}
           for i in range(10)]  # 1/10 FAILs confirmed
    )
    lynx_full = (
        [{"query_id": f"P{i}", "score": "PASS"} for i in range(900)]
        + [{"query_id": f"F{i}", "score": "FAIL"} for i in range(100)]
    )
    verdicts_path = _write_run_dir(tmp_path, key, verdicts, lynx_full)

    cmd_score(tmp_path, verdicts_path)
    report = json.loads((tmp_path / "calibration_report.json").read_text())

    assert report["lynx_precision"] == 0.1
    assert report["lynx_miss_rate"] == 0.0
    # est = 100 * 0.1 + 900 * 0 = 10.
    assert report["estimated_true_hallucinations"] == pytest.approx(10.0)


def test_per_category_confirmation_counts(tmp_path):
    # Two FAIL categories, mixed confirmations.
    key = [
        {"query_id": "c1", "lynx_score": "FAIL", "lynx_category": "contradiction"},
        {"query_id": "c2", "lynx_score": "FAIL", "lynx_category": "contradiction"},
        {"query_id": "u1", "lynx_score": "FAIL", "lynx_category": "unsupported_addition"},
        {"query_id": "u2", "lynx_score": "FAIL", "lynx_category": "unsupported_addition"},
    ]
    verdicts = [
        {"query_id": "c1", "verdict": "FAIL"},   # contradiction confirmed
        {"query_id": "c2", "verdict": "PASS"},   # contradiction rejected
        {"query_id": "u1", "verdict": "FAIL"},   # unsupported confirmed
        {"query_id": "u2", "verdict": "FAIL"},   # unsupported confirmed
    ]
    lynx_full = [{"query_id": q, "score": "FAIL"} for q in ("c1", "c2", "u1", "u2")]
    verdicts_path = _write_run_dir(tmp_path, key, verdicts, lynx_full)

    cmd_score(tmp_path, verdicts_path)
    report = json.loads((tmp_path / "calibration_report.json").read_text())

    per_cat = report["per_fail_category_confirmation"]
    assert per_cat["contradiction"] == {"n": 2, "confirmed_fail": 1}
    assert per_cat["unsupported_addition"] == {"n": 2, "confirmed_fail": 2}
    # Categories with no rows still appear (single source of truth in
    # VALID_CATEGORIES — they must be present even at zero).
    assert per_cat["omission"] == {"n": 0, "confirmed_fail": 0}


def test_missing_verdict_aborts(tmp_path):
    # `cmd_score` must hard-fail if the verdicts file doesn't cover every
    # sampled query — otherwise precision is silently computed against a
    # partial denominator.
    key = [
        {"query_id": "a", "lynx_score": "FAIL", "lynx_category": "contradiction"},
        {"query_id": "b", "lynx_score": "FAIL", "lynx_category": "contradiction"},
    ]
    verdicts = [{"query_id": "a", "verdict": "FAIL"}]   # b is missing
    lynx_full = [{"query_id": "a", "score": "FAIL"}, {"query_id": "b", "score": "FAIL"}]
    verdicts_path = _write_run_dir(tmp_path, key, verdicts, lynx_full)

    with pytest.raises(SystemExit):
        cmd_score(tmp_path, verdicts_path)


def test_invalid_verdict_value_aborts(tmp_path):
    key = [{"query_id": "a", "lynx_score": "FAIL", "lynx_category": "contradiction"}]
    verdicts = [{"query_id": "a", "verdict": "maybe"}]
    lynx_full = [{"query_id": "a", "score": "FAIL"}]
    verdicts_path = _write_run_dir(tmp_path, key, verdicts, lynx_full)

    with pytest.raises(SystemExit):
        cmd_score(tmp_path, verdicts_path)
