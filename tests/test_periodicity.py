"""Regression tests for the detect_periods(return_spectrum=...) change.

These run entirely offline on synthetic data (no MAST download, no TensorFlow), so they
are fast and don't depend on network access. The live-data regression against the
original (pre-edit) periodicity.py — synthetic transit, real KIC 757076 Q02, and
shuffled flux, each with default/no-detrend/custom-grid kwargs — was run manually during
development (see docs/scientific_audit.md, section "detect_periods change") and found
byte-identical default-path output; it is not repeated here to keep the suite offline.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from periodicity import detect_periods  # noqa: E402


def _synthetic_transit(period=3.5, depth=0.01, duration=0.1, n=4000, span=90.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    flux = 1.0 + rng.normal(0, 1e-3, n)
    phase = (t % period) / period
    flux[phase < duration / period] -= depth
    return t, flux


class TestDetectPeriodsSpectrum(unittest.TestCase):
    def setUp(self):
        self.t, self.f = _synthetic_transit()

    def test_default_return_unchanged_in_shape(self):
        candidates = detect_periods(self.t, self.f)
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        for key in ("period_days", "power", "sde", "depth", "duration_days", "transit_time"):
            self.assertIn(key, candidates[0])

    def test_return_spectrum_false_is_default(self):
        a = detect_periods(self.t, self.f)
        b = detect_periods(self.t, self.f, return_spectrum=False)
        self.assertEqual(a, b)

    def test_spectrum_consistent_with_candidates(self):
        candidates, spectrum = detect_periods(self.t, self.f, return_spectrum=True)
        for key in ("period", "power", "depth", "duration", "transit_time"):
            self.assertIn(key, spectrum)
            self.assertEqual(len(spectrum[key]), len(spectrum["period"]))
        # The best candidate must be the spectrum's maximum-power point.
        i_max = int(np.argmax(spectrum["power"]))
        self.assertAlmostEqual(spectrum["power"][i_max], candidates[0]["power"], places=6)
        self.assertEqual(spectrum["period"][i_max], candidates[0]["period_days"])

    def test_candidates_identical_with_and_without_spectrum(self):
        a = detect_periods(self.t, self.f, top_k=4)
        b, _ = detect_periods(self.t, self.f, top_k=4, return_spectrum=True)
        self.assertEqual(a, b)

    def test_recovers_injected_period(self):
        candidates = detect_periods(self.t, self.f, min_period_days=1.0, max_period_days=10.0)
        best = candidates[0]["period_days"]
        self.assertTrue(
            any(abs(best - 3.5 * m) / (3.5 * m) < 0.02 for m in (0.5, 1.0, 2.0)),
            f"expected ~3.5 d (or a harmonic), got {best}",
        )


if __name__ == "__main__":
    unittest.main()
