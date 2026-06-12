"""Unit tests for the R1 threshold table builders (pure logic only —
no TFLite, sqlite, or HF access)."""

import unittest

from retrieval_eval.build_mcq_outcome_table import outcome_label, split_of


class TestOutcomeLabel(unittest.TestCase):
    def test_hurt(self):
        self.assertEqual(outcome_label(True, False), "hurt")

    def test_helped(self):
        self.assertEqual(outcome_label(False, True), "helped")

    def test_unchanged_both_right(self):
        self.assertEqual(outcome_label(True, True), "unchanged")

    def test_unchanged_both_wrong(self):
        self.assertEqual(outcome_label(False, False), "unchanged")


class TestSplit(unittest.TestCase):
    def test_deterministic(self):
        rid = "mamabench_v0.2_medmcqa_abc123"
        self.assertEqual(split_of(rid), split_of(rid))

    def test_values(self):
        ids = [f"mamabench_v0.2_medmcqa_{i}" for i in range(200)]
        splits = {split_of(i) for i in ids}
        self.assertEqual(splits, {"tune", "holdout"})
        # Roughly balanced — md5 parity on 200 ids should not be degenerate.
        n_tune = sum(1 for i in ids if split_of(i) == "tune")
        self.assertGreater(n_tune, 60)
        self.assertLess(n_tune, 140)


if __name__ == "__main__":
    unittest.main()
