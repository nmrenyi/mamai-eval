"""Tests for Lynx output parsing in generator_eval/score_lynx.py.

Three runtime bugs were hit in this code path during the v0.1.0 oracle run:
  1. Lynx emits a Python-dict literal, not strict JSON (single quotes + bare
     verdict). `json.loads` alone fails — must fall back to `ast.literal_eval`
     after quoting the bare verdict.
  2. Some bullets contain unescaped apostrophes ("newborn's"), which break
     even `ast.literal_eval`. A delimiter-aware regex fallback must recover
     the bullets.
  3. The SCORE regex must accept the bare (unquoted) verdict — losing the
     SCORE means losing the metric.

These tests pin all three.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from generator_eval.score_lynx import _extract_bullets, _parse_lynx_output


# ── _parse_lynx_output ───────────────────────────────────────────────────────

def test_parse_strict_json_pass():
    text = '{"REASONING": ["one", "two"], "SCORE": "PASS"}'
    score, reasoning, note = _parse_lynx_output(text)
    assert score == "PASS"
    assert reasoning == ["one", "two"]
    assert note is None


def test_parse_python_literal_bare_verdict_pass():
    # Lynx's actual output shape: single-quoted strings + unquoted verdict.
    text = "{\"REASONING\": ['bullet one', 'bullet two'], \"SCORE\": PASS}"
    score, reasoning, note = _parse_lynx_output(text)
    assert score == "PASS"
    assert reasoning == ["bullet one", "bullet two"]
    assert note is None


def test_parse_python_literal_bare_verdict_fail():
    text = "{\"REASONING\": ['unsupported claim'], \"SCORE\": FAIL}"
    score, reasoning, note = _parse_lynx_output(text)
    assert score == "FAIL"
    assert reasoning == ["unsupported claim"]
    assert note is None


def test_parse_unescaped_apostrophe_falls_back_to_bullet_extraction():
    # `ast.literal_eval` chokes on 'newborn's' — must recover via regex.
    text = "{\"REASONING\": ['a newborn's ears', 'next bullet'], \"SCORE\": FAIL}"
    score, reasoning, note = _parse_lynx_output(text)
    assert score == "FAIL"
    assert reasoning == ["a newborn's ears", "next bullet"]
    assert note is not None and "delimiter extraction" in note


def test_parse_score_recovered_even_when_block_malformed():
    # Even if REASONING is unrecoverable, SCORE must be returned.
    text = "garbage no opening brace SCORE: PASS later text"
    score, reasoning, note = _parse_lynx_output(text)
    assert score == "PASS"
    # No REASONING in this input — reasoning should be None and note set.
    assert reasoning is None
    assert note is not None


def test_parse_no_score_returns_none_and_notes():
    text = '{"REASONING": ["one"], "SCORE": "MAYBE"}'
    score, reasoning, note = _parse_lynx_output(text)
    assert score is None
    assert reasoning == ["one"]
    assert note is not None and "no parseable SCORE" in note


def test_parse_lowercase_verdict_normalised():
    text = "{\"REASONING\": ['x'], \"SCORE\": pass}"
    score, reasoning, _ = _parse_lynx_output(text)
    assert score == "PASS"
    assert reasoning == ["x"]


# ── _extract_bullets — the delimiter-anchored regex itself ───────────────────

def test_extract_bullets_simple_single_quoted():
    assert _extract_bullets("['a', 'b', 'c']") == ["a", "b", "c"]


def test_extract_bullets_simple_double_quoted():
    assert _extract_bullets('["a", "b"]') == ["a", "b"]


def test_extract_bullets_inner_apostrophe_preserved():
    # The whole point of the delimiter-anchored regex: an apostrophe inside
    # a single-quoted bullet must NOT split the bullet in two.
    out = _extract_bullets("['a newborn's ears', 'next']")
    assert out == ["a newborn's ears", "next"]


def test_extract_bullets_empty_list():
    assert _extract_bullets("[]") == []


def test_extract_bullets_whitespace_tolerated():
    # The regex uses `\s*` around the comma — but there must be a delimiter.
    assert _extract_bullets("[ 'a' ,   'b'  ]") == ["a", "b"]
