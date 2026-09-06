"""
Tests for the monitor additions of Study 05 (spec D10, D11).

Run from studies/:  python 05_greybox_monitoring/tests/test_monitor.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import monitoring  # noqa: E402


def _ar1(n=20000, phi=0.99, seed=0, freq='20min'):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1.0, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    index = pd.date_range('2019-01-01', periods=n, freq=freq, tz='UTC')
    return pd.Series(r, index=index)


class TestPrewhiten(unittest.TestCase):

    def test_innovations_of_an_ar1_are_white_and_phi_is_recovered(self):
        residual = _ar1()
        innovations, phi = monitoring.prewhiten(residual, start='2019-01-01', end='2019-06-30')
        self.assertAlmostEqual(phi, 0.99, delta=0.01)
        self.assertLess(abs(innovations.autocorr(1)), 0.05)
        self.assertAlmostEqual(innovations.std(), 1.0, delta=0.05)

    def test_a_gap_is_not_bridged(self):
        residual = _ar1(n=500)
        residual.iloc[200:210] = np.nan
        innovations, _ = monitoring.prewhiten(residual, phi=0.99)
        self.assertTrue(innovations.iloc[200:211].isna().all())
        self.assertFalse(np.isnan(innovations.iloc[211]))


class TestChannelCoincidence(unittest.TestCase):

    def test_alarms_are_attributed_by_the_coincident_channel(self):
        index = pd.date_range('2024-01-01', periods=1000, freq='20min', tz='UTC')
        rng = np.random.default_rng(1)
        channels = pd.DataFrame({'tair': rng.normal(0, 0.1, 1000),
                                 'rh': rng.normal(0, 0.1, 1000),
                                 'batt': rng.normal(0, 0.001, 1000)}, index=index)
        channels.loc[index[500], 'batt'] += 1.0     # instrument excursion
        channels.loc[index[700], 'tair'] += 10.0    # environment excursion
        alarm = pd.Series(False, index=index)
        alarm.iloc[[500, 700, 900]] = True
        out = monitoring.channel_coincidence(alarm, channels)
        self.assertEqual(out.loc[index[500]], 'instrument')
        self.assertEqual(out.loc[index[700]], 'environment')
        self.assertEqual(out.loc[index[900]], 'unattributed')


class TestStatisticAwareDetectability(unittest.TestCase):

    def test_innovation_statistic_finds_a_step_the_raw_residual_needs_a_wide_limit_for(self):
        residual = _ar1(n=30000, phi=0.99)
        innovations, phi = monitoring.prewhiten(residual, start='2019-01-01', end='2019-04-30')
        reference = monitoring.reference_stats(innovations, start='2019-01-01', end='2019-04-30')
        curve = monitoring.detectability_curve(
            residual, reference['mu'], reference['sigma'], magnitudes=(8.0,),
            durations=('24h',), kind='step', statistic='innovation', phi=phi,
            lam=0.2, L=4.0)
        self.assertTrue(bool(curve.loc[0, 'detected']))

    def test_daily_amplitude_statistic_runs_on_a_daily_grid(self):
        index = pd.date_range('2019-01-01', periods=72 * 400, freq='20min', tz='UTC')
        hours = index.hour + index.minute / 60.0
        rng = np.random.default_rng(2)
        residual = pd.Series(2.0 * np.cos(2 * np.pi * (hours - 14) / 24.0)
                             + rng.normal(0, 0.5, len(index)), index=index)
        daily = monitoring.daily_harmonic(residual)['amplitude']
        reference = monitoring.reference_stats(daily, start='2019-01-01', end='2019-06-30')
        curve = monitoring.detectability_curve(
            residual, reference['mu'], reference['sigma'], magnitudes=(6.0,),
            durations=('168h',), kind='amplitude', statistic='daily_amplitude',
            injection_starts=('2019-09-01', '2019-11-01'))
        self.assertIn(curve.loc[0, 'detected'], (0.5, 1.0, True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
