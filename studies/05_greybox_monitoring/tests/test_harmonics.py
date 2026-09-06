"""
Tests for the harmonic diagnostics of Study 05 (spec D6).

Run from studies/:  python 05_greybox_monitoring/tests/test_harmonics.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import coupling, monitoring, prediction  # noqa: E402


def _days(n_days, amplitude=10.0, peak_hour=14.0, freq='20min', seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n_days * 72, freq=freq, tz='UTC')
    hours = index.hour + index.minute / 60.0
    values = amplitude * np.cos(2 * np.pi * (hours - peak_hour) / 24.0)
    return pd.Series(values + rng.normal(0, 0.1, len(index)), index=index)


class TestDailyHarmonic(unittest.TestCase):

    def test_amplitude_and_peak_hour_are_recovered(self):
        out = monitoring.daily_harmonic(_days(5), min_slots=60)
        self.assertEqual(len(out), 5)
        self.assertTrue(np.allclose(out['amplitude'], 10.0, atol=0.2))
        self.assertTrue(np.allclose(out['phase_h'], 14.0, atol=0.2))

    def test_a_short_day_is_dropped(self):
        series = _days(3)
        series.iloc[72:72 + 20] = np.nan   # second day keeps 52 of 72 slots
        out = monitoring.daily_harmonic(series, min_slots=60)
        self.assertEqual(len(out), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
