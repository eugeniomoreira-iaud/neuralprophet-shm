"""
Tests for shmlib.coupling.reset_thermal_lag_filter.

Run from studies/:  python shmlib/tests/test_coupling.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import coupling  # noqa: E402


def _one_pole(values, tau_hours, dt_hours=1.0):
    """Reference one-pole recursion, computed independently of shmlib."""
    alpha = dt_hours / (tau_hours + dt_hours)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * values[i]
    return out


class TestResetThermalLagFilter(unittest.TestCase):

    def test_matches_a_known_one_pole_filter_when_no_gap_is_long(self):
        index = pd.date_range('2024-01-01', periods=200, freq='1h', tz='UTC')
        rng = np.random.default_rng(0)
        values = 10.0 + 5.0 * np.sin(np.arange(200) / 12.0) + rng.normal(0, 0.1, 200)
        series = pd.Series(values, index=index)

        filtered, warmup = coupling.reset_thermal_lag_filter(
            series, tau_hours=6.0, reset_gap='10D', dt_hours=1.0)

        expected = _one_pole(values, tau_hours=6.0, dt_hours=1.0)
        np.testing.assert_allclose(filtered.to_numpy(), expected, rtol=0, atol=1e-9)
        # No run of missing values exists at all, so no segment can carry a
        # reset and the mask covers only the first segment's own warm-up.
        self.assertEqual(int(warmup.sum()), int(np.ceil(3.0 * 6.0 / 1.0)))

    def test_a_long_gap_resets_the_state_the_second_segment_sees(self):
        index = pd.date_range('2024-01-01', periods=140, freq='1h', tz='UTC')

        def build(level_before_gap):
            values = np.full(140, np.nan)
            values[:40] = level_before_gap
            # positions 40..89 (50 hours) stay NaN: a gap longer than the
            # 24-hour reset_gap below.
            values[90:] = 5.0 + np.sin(np.arange(50) / 8.0)
            return pd.Series(values, index=index)

        filtered_high, _ = coupling.reset_thermal_lag_filter(
            build(1000.0), tau_hours=4.0, reset_gap='24h', dt_hours=1.0)
        filtered_low, _ = coupling.reset_thermal_lag_filter(
            build(-1000.0), tau_hours=4.0, reset_gap='24h', dt_hours=1.0)

        # The pre-gap segment differs wildly between the two runs...
        self.assertFalse(np.allclose(filtered_high.iloc[:40].to_numpy(),
                                     filtered_low.iloc[:40].to_numpy()))
        # ...but the post-gap segment does not depend on it at all.
        np.testing.assert_allclose(filtered_high.iloc[90:].to_numpy(),
                                   filtered_low.iloc[90:].to_numpy(),
                                   rtol=0, atol=1e-12)
        # The excluded gap itself stays NaN in both.
        self.assertTrue(filtered_high.iloc[40:90].isna().all())

    def test_a_short_gap_does_not_reset_and_matches_thermal_lag_filter(self):
        index = pd.date_range('2024-01-01', periods=80, freq='1h', tz='UTC')
        rng = np.random.default_rng(1)
        values = 20.0 + rng.normal(0, 1.0, 80)
        values[30:35] = np.nan   # a 5-hour gap

        series = pd.Series(values, index=index)
        filtered, _ = coupling.reset_thermal_lag_filter(
            series, tau_hours=5.0, reset_gap='10h', dt_hours=1.0)
        direct = coupling.thermal_lag_filter(series, tau_hours=5.0, dt_hours=1.0)

        pd.testing.assert_series_equal(filtered, direct, check_names=False)

    def test_warmup_mask_length_per_segment_and_independence_from_values(self):
        index = pd.date_range('2024-01-01', periods=140, freq='1h', tz='UTC')

        def build(rng_seed):
            rng = np.random.default_rng(rng_seed)
            values = np.full(140, np.nan)
            values[:40] = rng.normal(0, 1.0, 40)
            values[90:] = rng.normal(0, 1.0, 50)
            return pd.Series(values, index=index)

        tau_hours, warmup_factor, dt_hours = 4.0, 3.0, 1.0
        expected_run = int(np.ceil(warmup_factor * tau_hours / dt_hours))

        _, warmup_a = coupling.reset_thermal_lag_filter(
            build(2), tau_hours=tau_hours, reset_gap='24h', dt_hours=dt_hours,
            warmup_factor=warmup_factor)
        _, warmup_b = coupling.reset_thermal_lag_filter(
            build(3), tau_hours=tau_hours, reset_gap='24h', dt_hours=dt_hours,
            warmup_factor=warmup_factor)

        pd.testing.assert_series_equal(warmup_a, warmup_b, check_names=False)
        self.assertEqual(int(warmup_a.iloc[:40].sum()), min(expected_run, 40))
        self.assertEqual(int(warmup_a.iloc[90:].sum()), min(expected_run, 50))
        self.assertTrue(warmup_a.iloc[40:90].eq(False).all())

    def test_tau_zero_returns_the_driver_untouched_and_an_all_false_mask(self):
        index = pd.date_range('2024-01-01', periods=10, freq='1h', tz='UTC')
        series = pd.Series(np.arange(10, dtype=float), index=index)
        series.iloc[4] = np.nan

        filtered, warmup = coupling.reset_thermal_lag_filter(
            series, tau_hours=0.0, reset_gap='1h', dt_hours=1.0)

        pd.testing.assert_series_equal(filtered, series, check_names=False)
        self.assertFalse(warmup.any())

    def test_nan_is_preserved_wherever_the_input_was_missing(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
        rng = np.random.default_rng(4)
        values = rng.normal(0, 1.0, 100)
        values[10:12] = np.nan     # short gap, bridged internally
        values[50:75] = np.nan     # long gap, a reset boundary
        series = pd.Series(values, index=index)

        filtered, _ = coupling.reset_thermal_lag_filter(
            series, tau_hours=3.0, reset_gap='12h', dt_hours=1.0)

        pd.testing.assert_series_equal(filtered.isna(), series.isna(),
                                       check_names=False)

    def test_changing_a_later_value_does_not_change_earlier_output(self):
        index = pd.date_range('2024-01-01', periods=60, freq='1h', tz='UTC')
        rng = np.random.default_rng(5)
        values = 5.0 + np.sin(np.arange(60) / 10.0) + rng.normal(0, 0.05, 60)
        series = pd.Series(values, index=index)

        baseline, _ = coupling.reset_thermal_lag_filter(
            series, tau_hours=6.0, reset_gap='10D', dt_hours=1.0)

        altered = series.copy()
        altered.iloc[30] += 500.0
        changed, _ = coupling.reset_thermal_lag_filter(
            altered, tau_hours=6.0, reset_gap='10D', dt_hours=1.0)

        pd.testing.assert_series_equal(baseline.iloc[:30], changed.iloc[:30],
                                       check_names=False)
        # Sanity: the perturbation does propagate forward, so the equality
        # above is a real causality check and not a vacuous one.
        self.assertFalse(np.allclose(baseline.iloc[30:].to_numpy(),
                                     changed.iloc[30:].to_numpy()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
