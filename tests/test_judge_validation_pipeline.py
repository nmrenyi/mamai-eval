"""End-to-end pipeline test for calibration/judge_validation/judge.py.

Phase-2 milestone: confirm the full call shape (load rows -> grade -> JSONL
verdicts -> load verdicts -> metrics) works without ever hitting a network.
The grader is a callable, so we inject a synthetic one.

Skips gracefully if the calibration file hasn't been fetched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.judge_validation import fetch, judge, metrics


_CACHE = fetch.DEFAULT_CACHE / fetch.DEFAULT_FILENAME


def _require_calibration() -> Path:
    if not _CACHE.exists():
        pytest.skip(
            "Calibration file not fetched. "
            "Run: python -m calibration.judge_validation fetch"
        )
    return _CACHE


# ── Synthetic graders ────────────────────────────────────────────────────────


def _always_met_grader(prompt: str) -> str:
    """Mock judge: always returns met=True with valid JSON."""
    return '{"explanation": "synthetic always-met", "criteria_met": true}'


def _flaky_grader(state: dict):
    """Mock judge that fails ONCE GLOBALLY across the whole test run, then
    succeeds for every subsequent call (per-row or otherwise). Used to verify
    the retry path catches a single transient error without retrying forever.
    Not suitable for multi-row retry tests — the same `state` dict is shared,
    so only the very first call sees the failure."""
    def grader(prompt: str) -> str:
        if not state.get("failed_once"):
            state["failed_once"] = True
            raise RuntimeError("simulated transient API error")
        return '{"explanation": "ok", "criteria_met": false}'
    return grader


# ── Pipeline smoke tests ─────────────────────────────────────────────────────


def test_run_judge_writes_verdicts(tmp_path):
    rows = metrics.load_rows(_require_calibration())[:30]
    out = tmp_path / "verdicts.jsonl"

    n = judge.run_judge(
        rows, out, _always_met_grader, "mock-always-met",
        max_workers=4, quiet=True,
    )
    assert n == 30

    # All lines parse, no errors, all met=True, judge_model stamped.
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 30
    for v in lines:
        assert v["error"] is None
        assert v["criteria_met"] is True
        assert v["judge_model"] == "mock-always-met"


def test_resume_skips_done_rows(tmp_path):
    """Re-running with the same output dir should write 0 new verdicts."""
    rows = metrics.load_rows(_require_calibration())[:10]
    out = tmp_path / "verdicts.jsonl"

    n1 = judge.run_judge(rows, out, _always_met_grader, "mock",
                        max_workers=4, quiet=True)
    n2 = judge.run_judge(rows, out, _always_met_grader, "mock",
                        max_workers=4, quiet=True)
    assert n1 == 10
    assert n2 == 0  # resume worked — nothing re-judged


def test_load_verdicts_round_trip(tmp_path):
    rows = metrics.load_rows(_require_calibration())[:5]
    out = tmp_path / "verdicts.jsonl"
    judge.run_judge(rows, out, _always_met_grader, "mock",
                    max_workers=2, quiet=True)

    verdicts = judge.load_verdicts(out)
    assert set(verdicts) == {r["_orig_idx"] for r in rows}
    assert all(v is True for v in verdicts.values())


def test_grade_row_retries_on_transient_error():
    rows = metrics.load_rows(_require_calibration())[:1]
    grader = _flaky_grader({})
    v = judge.grade_row(rows[0], grader, max_retries=3)
    # After one transient failure, the retry succeeds.
    assert v["error"] is None
    assert v["criteria_met"] is False


def test_full_pipeline_metrics_on_mock_judge(tmp_path):
    """End-to-end: 100 rows, mock 'always-met' judge, compute report. Rubber-
    stamp detector should flag this judge."""
    rows = metrics.load_rows(_require_calibration())[:100]
    out = tmp_path / "verdicts.jsonl"
    judge.run_judge(rows, out, _always_met_grader, "mock-always-met",
                    max_workers=4, quiet=True)

    verdicts = judge.load_verdicts(out)
    report = metrics.full_report(rows, verdicts, judge_model="mock-always-met")

    # The rubber-stamp signals are unambiguous:
    assert report["prediction_distribution"]["judge_met_rate"] == 1.0
    consensus = report["llm_vs_consensus"]
    if consensus["n"] > 0:
        # Agreement on physician-not-met rows must be 0 (judge always says met)
        assert consensus["agreement_on_not_met_rows"] == 0.0
        # And 100% on met rows
        assert consensus["agreement_on_met_rows"] == 1.0

    # And the markdown renders without crashing
    md = metrics.render_markdown(report)
    assert "mock-always-met" in md
