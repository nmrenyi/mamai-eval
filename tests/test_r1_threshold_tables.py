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


class TestSplitAssign(unittest.TestCase):
    def test_deterministic_and_three_way(self):
        from retrieval_eval.build_split import assign
        ids = [f"q_{i:05d}" for i in range(2000)]
        a = [assign(i) for i in ids]
        self.assertEqual(a, [assign(i) for i in ids])  # deterministic
        self.assertEqual(set(a), {"train", "dev", "test"})
        # Roughly 70/15/15 over 2000 ids.
        frac = {s: a.count(s) / len(a) for s in ("train", "dev", "test")}
        self.assertAlmostEqual(frac["train"], 0.70, delta=0.04)
        self.assertAlmostEqual(frac["dev"], 0.15, delta=0.04)
        self.assertAlmostEqual(frac["test"], 0.15, delta=0.04)


class TestLexicalFeatures(unittest.TestCase):
    def test_overlap_and_coverage(self):
        from retrieval_eval.build_ltr_features import lexical_features
        f = lexical_features("magnesium sulfate dose 4 g",
                             "give magnesium sulfate 4 g loading dose")
        self.assertGreater(f["overlap_count"], 0)
        self.assertGreater(f["q_coverage"], 0.5)   # most query content words present
        self.assertEqual(f["num_overlap"], 1)       # shared "4"
        self.assertGreaterEqual(f["jaccard"], 0.0)
        self.assertLessEqual(f["jaccard"], 1.0)

    def test_no_overlap(self):
        from retrieval_eval.build_ltr_features import lexical_features
        f = lexical_features("postpartum hemorrhage", "neonatal jaundice phototherapy")
        self.assertEqual(f["overlap_count"], 0)
        self.assertEqual(f["q_coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
