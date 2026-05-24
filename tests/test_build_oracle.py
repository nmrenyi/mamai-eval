"""Tests for `build_oracle` chunk selection in generator_eval/build_oracle.py.

The oracle is the foundation of every downstream metric, so the three rules
that govern chunk inclusion and ordering MUST stay exactly as documented:
  1. include a chunk iff its judgment score >= threshold,
  2. order chunks per query by (score desc, chunk_id asc),
  3. drop a query entirely if no chunk qualifies.

These are deterministic on the inputs, so we mock `datasets.load_dataset`
rather than touch HF — keeps the test offline and fast.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def _install_fake_datasets(monkeypatch, queries, judgments, chunks):
    """Install a fake `datasets` module exposing `load_dataset(...)`."""

    def fake_load_dataset(repo, name, revision=None, split=None):
        if name == "queries":
            return queries
        if name == "judgments":
            return judgments
        if name == "chunks":
            return chunks
        raise AssertionError(f"unexpected config name {name}")

    fake = types.ModuleType("datasets")
    fake.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_threshold_filters_below_cutoff(monkeypatch):
    from generator_eval.build_oracle import build_oracle

    queries = [{"query_id": "q1", "query_text": "Q?"}]
    judgments = [
        {"query_id": "q1", "chunk_id": "c_low", "score": 4},
        {"query_id": "q1", "chunk_id": "c_hi",  "score": 6},
    ]
    chunks = [
        {"chunk_id": "c_low", "text": "low"},
        {"chunk_id": "c_hi",  "text": "hi"},
    ]
    _install_fake_datasets(monkeypatch, queries, judgments, chunks)

    rows, counts = build_oracle("repo", "rev", threshold=5)

    assert len(rows) == 1
    chunk_ids = [c["chunk_id"] for c in rows[0]["chunks"]]
    assert chunk_ids == ["c_hi"]  # score=4 dropped
    assert counts["n_queries_with_oracle"] == 1
    assert counts["n_qc_pairs"] == 1


def test_chunks_sorted_score_desc_then_chunk_id_asc(monkeypatch):
    from generator_eval.build_oracle import build_oracle

    queries = [{"query_id": "q1", "query_text": "Q?"}]
    # Two score=6 chunks with non-alphabetical insertion order, plus a score=5.
    judgments = [
        {"query_id": "q1", "chunk_id": "c_b", "score": 6},
        {"query_id": "q1", "chunk_id": "c_a", "score": 6},
        {"query_id": "q1", "chunk_id": "c_z", "score": 5},
    ]
    chunks = [
        {"chunk_id": "c_a", "text": "a"},
        {"chunk_id": "c_b", "text": "b"},
        {"chunk_id": "c_z", "text": "z"},
    ]
    _install_fake_datasets(monkeypatch, queries, judgments, chunks)

    rows, _ = build_oracle("repo", "rev", threshold=5)

    assert [c["chunk_id"] for c in rows[0]["chunks"]] == ["c_a", "c_b", "c_z"]
    assert [c["score"] for c in rows[0]["chunks"]] == [6, 6, 5]


def test_queries_with_no_qualifying_chunks_are_dropped(monkeypatch):
    from generator_eval.build_oracle import build_oracle

    queries = [
        {"query_id": "q_keep", "query_text": "keep"},
        {"query_id": "q_drop", "query_text": "drop"},
    ]
    judgments = [
        {"query_id": "q_keep", "chunk_id": "c1", "score": 5},
        {"query_id": "q_drop", "chunk_id": "c2", "score": 4},  # below cutoff
    ]
    chunks = [
        {"chunk_id": "c1", "text": "x"},
        {"chunk_id": "c2", "text": "y"},
    ]
    _install_fake_datasets(monkeypatch, queries, judgments, chunks)

    rows, counts = build_oracle("repo", "rev", threshold=5)

    assert [r["query_id"] for r in rows] == ["q_keep"]
    assert counts["n_queries_total"] == 2
    assert counts["n_queries_with_oracle"] == 1


def test_threshold_6_is_strict_subset_of_threshold_5(monkeypatch):
    from generator_eval.build_oracle import build_oracle

    queries = [
        {"query_id": "q1", "query_text": "Q1"},
        {"query_id": "q2", "query_text": "Q2"},
    ]
    judgments = [
        {"query_id": "q1", "chunk_id": "c1", "score": 6},
        {"query_id": "q2", "chunk_id": "c2", "score": 5},  # disappears at thr=6
    ]
    chunks = [
        {"chunk_id": "c1", "text": "x"},
        {"chunk_id": "c2", "text": "y"},
    ]
    _install_fake_datasets(monkeypatch, queries, judgments, chunks)

    rows5, _ = build_oracle("repo", "rev", threshold=5)
    rows6, _ = build_oracle("repo", "rev", threshold=6)

    assert {r["query_id"] for r in rows5} == {"q1", "q2"}
    assert {r["query_id"] for r in rows6} == {"q1"}


def test_chunk_text_inlined(monkeypatch):
    from generator_eval.build_oracle import build_oracle

    queries = [{"query_id": "q1", "query_text": "Q?"}]
    judgments = [{"query_id": "q1", "chunk_id": "c1", "score": 5}]
    chunks = [{"chunk_id": "c1", "text": "the actual chunk text"}]
    _install_fake_datasets(monkeypatch, queries, judgments, chunks)

    rows, _ = build_oracle("repo", "rev", threshold=5)

    # Oracle JSONL must be self-contained — no chunk-table lookup downstream.
    assert rows[0]["chunks"][0]["text"] == "the actual chunk text"
