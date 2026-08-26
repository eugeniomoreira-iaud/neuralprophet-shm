"""
Unit tests for shmlib.monitoring.

Run from studies/:  python shmlib/tests/test_monitoring.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import monitoring


def _quiet(n=2000, seed=0, freq='20min'):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq=freq)
    return pd.Series(rng.normal(0.0, 1.0, n), index=index)


class TestReferenceStats(unittest.TestCase):

    def test_robust_centre_and_scale_ignore_a_contaminated_tail(self):
        residuals = _quiet(1000)
        residuals.iloc[-50:] += 40.0
        stats = monitoring.reference_stats(residuals, robust=True)
        self.assertLess(abs(stats['mu']), 0.3)
        self.assertLess(stats['sigma'], 2.0)

    def test_a_window_restricts_which_rows_are_used(self):
        residuals = _quiet(1000)
        stats = monitoring.reference_stats(
            residuals, start=residuals.index[0], end=residuals.index[99])
        self.assertEqual(stats['n'], 100)


class TestEwmaChart(unittest.TestCase):

    def test_quiet_residuals_rarely_alarm(self):
        residuals = _quiet(2000)
        stats = monitoring.reference_stats(residuals)
        chart = monitoring.ewma_chart(residuals, stats['mu'], stats['sigma'])
        self.assertLess(chart['alarm'].mean(), 0.02)

    def test_a_sustained_step_alarms(self):
        residuals = _quiet(2000)
        residuals.iloc[1000:] += 3.0
        stats = monitoring.reference_stats(
            residuals, end=residuals.index[500])
        chart = monitoring.ewma_chart(residuals, stats['mu'], stats['sigma'])
        self.assertTrue(chart['alarm'].iloc[1000:1100].any())

    def test_limits_widen_with_L_and_narrow_with_lambda(self):
        residuals = _quiet(500)
        wide = monitoring.ewma_chart(residuals, 0.0, 1.0, lam=0.2, L=4.0)
        narrow = monitoring.ewma_chart(residuals, 0.0, 1.0, lam=0.2, L=2.0)
        self.assertTrue((wide['ucl'] > narrow['ucl']).all())


class TestCusumChart(unittest.TestCase):

    def test_a_small_persistent_shift_accumulates(self):
        residuals = _quiet(2000)
        residuals.iloc[1000:] += 0.8
        chart = monitoring.cusum_chart(residuals, 0.0, 1.0, k=0.5, h=5.0)
        # In control this chart signals rarely rather than never: with k = 0.5
        # and h = 5 the in-control average run length is a few hundred samples,
        # so a handful of isolated exceedances in 900 draws of noise is the
        # design point and not a defect. What distinguishes the shift is the
        # rate, which is what these two assertions compare.
        quiet_rate = chart['alarm'].iloc[:900].mean()
        shifted_rate = chart['alarm'].iloc[1000:1400].mean()
        self.assertLess(quiet_rate, 0.05)
        self.assertTrue(chart['alarm'].iloc[1000:1400].any())
        self.assertGreater(shifted_rate, 10 * quiet_rate)

    def test_both_directions_are_watched(self):
        residuals = _quiet(1000)
        residuals.iloc[500:] -= 2.0
        chart = monitoring.cusum_chart(residuals, 0.0, 1.0)
        self.assertTrue(chart['alarm'].iloc[500:].any())
        self.assertTrue((chart['cusum_low'].iloc[500:] > 0).any())


class TestJointAlarm(unittest.TestCase):

    def test_alarms_coincide_within_the_window(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        ewma = pd.Series(False, index=index)
        cusum = pd.Series(False, index=index)
        ewma.iloc[10] = True
        cusum.iloc[12] = True
        joint = monitoring.joint_alarm(ewma, cusum, window='2h')
        self.assertTrue(joint.iloc[10:13].any())

    def test_isolated_alarms_on_one_chart_do_not_survive(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        ewma = pd.Series(False, index=index)
        cusum = pd.Series(False, index=index)
        ewma.iloc[10] = True
        joint = monitoring.joint_alarm(ewma, cusum, window='2h')
        self.assertFalse(joint.any())


class TestAlarmEpisodes(unittest.TestCase):

    def test_consecutive_alarms_collapse_into_one_episode(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        alarm = pd.Series(False, index=index)
        alarm.iloc[10:16] = True
        alarm.iloc[50] = True
        episodes = monitoring.alarm_episodes(alarm)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes['n_slots'].iloc[0], 6)
        self.assertAlmostEqual(episodes['duration_h'].iloc[0], 2.0)

    def test_residual_statistics_are_attached_when_supplied(self):
        index = pd.date_range('2024-01-01', periods=20, freq='20min')
        alarm = pd.Series(False, index=index)
        alarm.iloc[5:8] = True
        residuals = pd.Series(np.arange(20.0), index=index)
        episodes = monitoring.alarm_episodes(alarm, residuals)
        self.assertAlmostEqual(episodes['peak_abs_z'].iloc[0], 7.0)
        self.assertAlmostEqual(episodes['mean_z'].iloc[0], 6.0)


class TestAverageRunLength(unittest.TestCase):

    def test_run_length_is_watched_time_divided_by_episode_count(self):
        index = pd.date_range('2024-01-01', periods=144, freq='20min')
        alarm = pd.Series(False, index=index)
        alarm.iloc[10] = True
        alarm.iloc[100] = True
        result = monitoring.average_run_length(alarm, freq='20min')
        self.assertEqual(result['n_episodes'], 2)
        self.assertAlmostEqual(result['hours'], 48.0)
        self.assertAlmostEqual(result['arl_hours'], 24.0)
        self.assertAlmostEqual(result['arl_days'], 1.0)

    def test_a_chart_that_never_alarms_reports_an_infinite_run_length(self):
        index = pd.date_range('2024-01-01', periods=144, freq='20min')
        result = monitoring.average_run_length(
            pd.Series(False, index=index), freq='20min')
        self.assertEqual(result['n_episodes'], 0)
        self.assertTrue(np.isinf(result['arl_hours']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
