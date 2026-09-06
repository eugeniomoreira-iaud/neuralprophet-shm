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


class TestAnnualModulation(unittest.TestCase):

    def _daily(self, years=4, seed=1):
        rng = np.random.default_rng(seed)
        days = pd.date_range('2019-01-01', periods=365 * years, freq='D', tz='UTC')
        doy = days.dayofyear.to_numpy()
        truth = 20.0 + 8.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)
        return pd.Series(truth + rng.normal(0, 1.0, len(days)), index=days)

    def test_one_harmonic_is_chosen_and_the_peak_day_is_recovered(self):
        table, fit = coupling.annual_modulation(self._daily(), harmonics=(1, 2))
        self.assertEqual(fit['order'], 1)
        self.assertTrue(table.loc[table['chosen'], 'order'].item() == 1)
        year = pd.date_range('2023-01-01', periods=365, freq='D', tz='UTC')
        curve = coupling.evaluate_modulation(fit, year)
        self.assertAlmostEqual(curve.idxmax().dayofyear, 200, delta=6)
        self.assertAlmostEqual(curve.max() - curve.min(), 16.0, delta=1.0)


class TestCycleSurfaceRank(unittest.TestCase):

    def test_a_single_modulated_shape_has_rank_one(self):
        index = pd.date_range('2019-01-01', periods=72 * 365 * 3, freq='20min', tz='UTC')
        hours = index.hour + index.minute / 60.0
        doy = index.dayofyear.to_numpy()
        envelope = 1.0 + 0.5 * np.cos(2 * np.pi * (doy - 200) / 365.25)
        series = pd.Series(envelope * np.cos(2 * np.pi * (hours - 14) / 24.0), index=index)
        out = coupling.cycle_surface_rank(series, doy_bins=52)
        self.assertGreater(out['table']['variance_share'].iloc[0], 0.98)
        self.assertEqual(out['daily_shapes'].shape[0], 72)
        self.assertEqual(out['annual_weights'].shape[0], 52)


class TestSeasonalWeights(unittest.TestCase):

    def test_weights_sum_to_one_and_peak_in_july_by_default(self):
        index = pd.date_range('2024-01-01', periods=366, freq='D', tz='UTC')
        out = prediction.seasonal_weights(index)
        self.assertTrue(np.allclose(out['summer_w'] + out['winter_w'], 1.0))
        self.assertEqual(out['summer_w'].idxmax().month, 7)
        self.assertGreaterEqual(out['summer_w'].min(), 0.0)
        self.assertLessEqual(out['summer_w'].max(), 1.0)

    def test_a_measured_modulation_is_normalised_to_the_unit_interval(self):
        fit = {'order': 1, 'coef': np.array([20.0, -8.0, 0.0]),
               'period_days': 365.25, 'n': 1000}
        index = pd.date_range('2024-01-01', periods=366, freq='D', tz='UTC')
        out = prediction.seasonal_weights(index, modulation=fit)
        self.assertAlmostEqual(out['summer_w'].max(), 1.0, places=6)
        self.assertAlmostEqual(out['summer_w'].min(), 0.0, places=6)


class TestPeriodScanSpacing(unittest.TestCase):

    def _series(self, n_days=400, seed=2):
        rng = np.random.default_rng(seed)
        index = pd.date_range('2020-01-01', periods=n_days * 72, freq='20min', tz='UTC')
        t_days = (index - index[0]) / pd.Timedelta(days=1)
        values = np.cos(2 * np.pi * t_days) + rng.normal(0, 0.05, len(index))
        return pd.Series(values, index=index)

    def test_the_daily_peak_is_certified_on_a_log_grid(self):
        scan = prediction.period_scan(
            self._series(), min_days=0.5, max_days=30.0, n_periods=4000,
            spacing='log', top=3)
        top = scan.iloc[0]
        self.assertLess(abs(top['period_days'] - 1.0), top['resolution_days'])

    def test_an_unknown_spacing_raises(self):
        with self.assertRaises(ValueError):
            prediction.period_scan(
                self._series(), min_days=0.5, max_days=30.0, n_periods=4000,
                spacing='bogus', top=3)

    def test_chunked_evaluation_matches_a_single_call(self):
        series = self._series()
        kwargs = dict(min_days=0.5, max_days=30.0, n_periods=4000,
                     spacing='log', top=3)
        default_chunk = prediction._LOMBSCARGLE_CHUNK
        whole = prediction.period_scan(series, **kwargs)
        prediction._LOMBSCARGLE_CHUNK = 7
        try:
            chunked = prediction.period_scan(series, **kwargs)
        finally:
            prediction._LOMBSCARGLE_CHUNK = default_chunk
        pd.testing.assert_frame_equal(whole, chunked)


class TestOlsResidual(unittest.TestCase):

    def test_gains_and_residual_on_paired_rows_with_missing_data(self):
        rng = np.random.default_rng(3)
        n = 500
        index = pd.date_range('2021-01-01', periods=n, freq='D', tz='UTC')
        a = pd.Series(rng.normal(0, 1, n), index=index, name='a')
        b = pd.Series(rng.normal(0, 1, n), index=index, name='b')
        a.iloc[10] = np.nan
        a.iloc[200] = np.nan
        target = pd.Series(2 + 3 * a + (-1) * b + rng.normal(0, 0.01, n),
                           index=index, name='y')
        residual, gains = prediction.ols_residual(target, pd.concat([a, b], axis=1))
        self.assertTrue(np.allclose(
            gains[['intercept', 'a', 'b']].to_numpy(), [2.0, 3.0, -1.0], atol=0.05))
        self.assertEqual(len(residual), 498)
        self.assertAlmostEqual(residual.mean(), 0.0, delta=0.01)


class TestHasCertifiedPeriod(unittest.TestCase):

    def _scan(self):
        return pd.DataFrame({
            'period_days': [365.2, 1.0],
            'resolution_days': [20.0, 0.0003],
        })

    def test_own_resolution_certifies_the_annual_peak(self):
        self.assertTrue(prediction.has_certified_period(self._scan(), 365.25))

    def test_own_resolution_rejects_a_far_off_period(self):
        self.assertFalse(prediction.has_certified_period(self._scan(), 182.6))

    def test_fixed_tolerance_rejects_a_sub_daily_peak_absent_from_the_scan(self):
        self.assertFalse(prediction.has_certified_period(
            self._scan(), 0.5, tolerance_days=0.05))

    def test_fixed_tolerance_certifies_the_daily_peak(self):
        self.assertTrue(prediction.has_certified_period(
            self._scan(), 1.0, tolerance_days=0.05))


if __name__ == '__main__':
    unittest.main(verbosity=2)
