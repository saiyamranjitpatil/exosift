"""Offline unit tests for the pure-logic helpers in src/exosift_pipeline.py.

No network access and no TensorFlow model loading — these exercise the gap/run
detection, the composite-score weight renormalization, and the harmonic-relation
labeling used by the periodogram and the caution flags.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exosift_pipeline import (  # noqa: E402
    COMPOSITE_BASE_WEIGHTS,
    compute_composite,
    find_all_runs,
    find_gaps,
    harmonic_relation,
)
from explainability import rank_candidate  # noqa: E402


class TestRunsAndGaps(unittest.TestCase):
    def test_single_run_no_gaps(self):
        t = np.linspace(0, 10, 500)
        self.assertEqual(find_all_runs(t), [(0, 500)])
        self.assertEqual(find_gaps(t), [])

    def test_one_large_gap_splits_into_two_runs(self):
        t = np.concatenate([np.linspace(0, 10, 300), np.linspace(20, 30, 300)])
        runs = find_all_runs(t)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0], (0, 300))
        self.assertEqual(runs[1], (300, 600))
        gaps = find_gaps(t)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0]["days"], t[300] - t[299], places=9)

    def test_runs_partition_every_point_exactly_once(self):
        rng = np.random.default_rng(1)
        t = np.sort(rng.uniform(0, 100, 1000))
        # Insert three real gaps.
        t[300:] += 5
        t[600:] += 8
        t[850:] += 2
        runs = find_all_runs(t)
        covered = sum(b - a for a, b in runs)
        self.assertEqual(covered, len(t))
        self.assertEqual(runs[0][0], 0)
        self.assertEqual(runs[-1][1], len(t))
        for (_, stop), (start2, _) in zip(runs, runs[1:]):
            self.assertEqual(stop, start2)  # runs are contiguous and non-overlapping


class TestHarmonicRelation(unittest.TestCase):
    def test_exact_multiples_and_fractions(self):
        self.assertEqual(harmonic_relation(1.0, 1.0), "1x")
        self.assertEqual(harmonic_relation(2.0, 1.0), "2x")
        self.assertEqual(harmonic_relation(0.5, 1.0), "0.5x")
        self.assertEqual(harmonic_relation(1.5, 1.0), "1.5x")
        self.assertEqual(harmonic_relation(3.0, 1.0), "3x")

    def test_within_tolerance_still_matches(self):
        self.assertEqual(harmonic_relation(2.01, 1.0), "2x")

    def test_unrelated_period_returns_none(self):
        self.assertIsNone(harmonic_relation(1.37, 1.0))

    def test_zero_reference_returns_none(self):
        self.assertIsNone(harmonic_relation(1.0, 0.0))


class TestComposite(unittest.TestCase):
    def test_weights_sum_to_one(self):
        comp = compute_composite(cnn_score=0.5, bls_power=20.0, depth_rel=0.01, repeatability=0.7)
        self.assertAlmostEqual(sum(comp["weights"].values()), 1.0, places=9)

    def test_matches_direct_rank_candidate_call(self):
        total = sum(COMPOSITE_BASE_WEIGHTS.values())
        weights = {k: w / total for k, w in COMPOSITE_BASE_WEIGHTS.items()}
        expected_score, _ = rank_candidate(
            cnn_probability=0.5674, rf_probability=None, bls_sde=215.3,
            transit_depth=0.000282, repeatability_score=0.781, weights=weights,
        )
        comp = compute_composite(cnn_score=0.5674, bls_power=215.3, depth_rel=0.000282, repeatability=0.781)
        self.assertAlmostEqual(comp["score"], expected_score, places=9)

    def test_matches_known_worked_example(self):
        # KIC 757076 Q02: see docs/scientific_audit.md for the full derivation.
        comp = compute_composite(cnn_score=0.567409, bls_power=215.3, depth_rel=0.00028155, repeatability=0.7815)
        self.assertAlmostEqual(comp["score"], 0.613, places=3)

    def test_bls_term_saturates_above_power_10(self):
        comp = compute_composite(cnn_score=0.0, bls_power=1000.0, depth_rel=0.0, repeatability=0.0)
        bls_row = next(r for r in comp["rows"] if r["component"] == "bls")
        self.assertEqual(bls_row["normalised"], 1.0)

    def test_rf_term_is_absent_not_zero_weighted(self):
        comp = compute_composite(cnn_score=0.5, bls_power=5.0, depth_rel=0.01, repeatability=0.5)
        self.assertNotIn("rf", comp["weights"])
        self.assertTrue(all(r["component"] != "rf" for r in comp["rows"]))


if __name__ == "__main__":
    unittest.main()
