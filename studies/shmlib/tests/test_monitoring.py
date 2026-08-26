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

    def test_a_zero_sigma_raises(self):
        residuals = _quiet(100)
        with self.assertRaises(ValueError):
            monitoring.ewma_chart(residuals, 0.0, 0.0)

    def test_a_non_finite_sigma_raises(self):
        residuals = _quiet(100)
        with self.assertRaises(ValueError):
            monitoring.ewma_chart(residuals, 0.0, float('nan'))


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

    def test_a_zero_sigma_raises(self):
        residuals = _quiet(100)
        with self.assertRaises(ValueError):
            monitoring.cusum_chart(residuals, 0.0, 0.0)


class TestGapHandling(unittest.TestCase):

    def test_a_gap_does_not_disturb_either_accumulator(self):
        residuals = _quiet(200)
        residuals.iloc[100:110] = np.nan

        ewma = monitoring.ewma_chart(residuals, 0.0, 1.0)
        before = ewma['ewma'].iloc[99]
        self.assertTrue((ewma['ewma'].iloc[100:110] == before).all())

        cusum = monitoring.cusum_chart(residuals, 0.0, 1.0)
        high_before = cusum['cusum_high'].iloc[99]
        low_before = cusum['cusum_low'].iloc[99]
        self.assertTrue((cusum['cusum_high'].iloc[100:110] == high_before).all())
        self.assertTrue((cusum['cusum_low'].iloc[100:110] == low_before).all())


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

    def test_a_dropped_row_breaks_the_regular_grid_and_raises(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        index = index.delete(50)
        alarm = pd.Series(False, index=index)
        alarm.iloc[10:16] = True
        with self.assertRaises(ValueError):
            monitoring.alarm_episodes(alarm)


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

    def test_a_freq_that_disagrees_with_the_index_raises(self):
        index = pd.date_range('2024-01-01', periods=144, freq='20min')
        alarm = pd.Series(False, index=index)
        with self.assertRaises(ValueError):
            monitoring.average_run_length(alarm, freq='30min')


class TestInjectAnomaly(unittest.TestCase):

    def test_a_step_shifts_everything_from_its_start(self):
        series = _quiet(100)
        moved = monitoring.inject_anomaly(
            series, 'step', 5.0, start=series.index[50])
        np.testing.assert_allclose(moved.iloc[:50], series.iloc[:50])
        np.testing.assert_allclose(moved.iloc[50:], series.iloc[50:] + 5.0)

    def test_a_pulse_ends_after_its_duration(self):
        series = _quiet(100)
        moved = monitoring.inject_anomaly(
            series, 'pulse', 5.0, start=series.index[50], duration='2h')
        self.assertAlmostEqual(moved.iloc[50] - series.iloc[50], 5.0)
        self.assertAlmostEqual(moved.iloc[90] - series.iloc[90], 0.0)

    def test_a_ramp_reaches_its_magnitude_at_the_end_of_its_duration(self):
        series = pd.Series(0.0, index=pd.date_range(
            '2024-01-01', periods=100, freq='20min'))
        moved = monitoring.inject_anomaly(
            series, 'ramp', 6.0, start=series.index[10], duration='3h')
        self.assertAlmostEqual(moved.iloc[10], 0.0, places=6)
        self.assertAlmostEqual(moved.iloc[19], 6.0, places=6)
        self.assertAlmostEqual(moved.iloc[50], 6.0, places=6)

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            monitoring.inject_anomaly(
                _quiet(10), 'wobble', 1.0, start=_quiet(10).index[0])


class TestDetectabilityCurve(unittest.TestCase):

    def test_a_large_step_is_detected_and_a_small_one_is_not(self):
        residuals = _quiet(3000)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[0.1, 5.0], durations=['24h'])
        small = curve[curve['magnitude'] == 0.1].iloc[0]
        large = curve[curve['magnitude'] == 5.0].iloc[0]
        self.assertFalse(bool(small['detected']))
        self.assertTrue(bool(large['detected']))
        # A large step alarms on its first contaminated sample, so a delay of
        # zero is the best case rather than a defect. What the delay must be is
        # finite and inside the window the sweep actually searched.
        self.assertGreaterEqual(large['delay_h'], 0.0)
        self.assertLessEqual(large['delay_h'], 48.0)

    def test_a_larger_departure_is_found_sooner(self):
        residuals = _quiet(3000)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[1.0, 2.0], durations=['24h'])
        small = curve[curve['magnitude'] == 1.0].iloc[0]
        large = curve[curve['magnitude'] == 2.0].iloc[0]
        self.assertTrue(bool(small['detected']))
        self.assertTrue(bool(large['detected']))
        self.assertLess(large['delay_h'], small['delay_h'])

    def test_one_row_per_magnitude_and_duration_pair(self):
        residuals = _quiet(1500)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[1.0, 2.0],
            durations=['6h', '24h'])
        self.assertEqual(len(curve), 4)
        self.assertEqual(list(curve.columns),
                         ['magnitude', 'duration_h', 'detected', 'delay_h'])

    def test_an_undetected_case_reports_a_missing_delay(self):
        residuals = _quiet(1500)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[0.01], durations=['6h'])
        self.assertFalse(bool(curve['detected'].iloc[0]))
        self.assertTrue(np.isnan(curve['delay_h'].iloc[0]))

    def test_an_alarm_the_clean_record_raises_anyway_is_not_a_detection(self):
        residuals = _quiet(1500)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[0.0], durations=['6h'])
        self.assertFalse(bool(curve['detected'].iloc[0]))
        self.assertTrue(np.isnan(curve['delay_h'].iloc[0]))


class TestPhaseShiftAmplitude(unittest.TestCase):

    def test_no_shift_implies_no_residual(self):
        self.assertAlmostEqual(
            monitoring.phase_shift_amplitude(10.0, 0.0), 0.0)

    def test_half_a_period_inverts_the_cycle(self):
        # A cycle shifted by half its period is its own negation, so the
        # difference between shifted and unshifted has twice the amplitude.
        self.assertAlmostEqual(
            monitoring.phase_shift_amplitude(10.0, 12.0), 20.0)

    def test_a_small_shift_follows_the_chord_formula(self):
        expected = 2.0 * 10.0 * np.sin(np.pi * 1.0 / 24.0)
        self.assertAlmostEqual(
            monitoring.phase_shift_amplitude(10.0, 1.0), expected)


class TestPhysicalInjections(unittest.TestCase):

    def _flat(self, n=6 * 24 * 40):
        index = pd.date_range('2024-01-01', periods=n, freq='20min')
        return pd.Series(0.0, index=index)

    def test_an_amplitude_growth_reaches_its_size_and_holds(self):
        series = self._flat()
        start = series.index[100]
        moved = monitoring.inject_anomaly(
            series, 'amplitude', 4.0, start=start, duration='10d')
        np.testing.assert_allclose(moved.iloc[:100], 0.0, atol=1e-12)
        first_day = moved.loc[start:start + pd.Timedelta('1d')]
        last_day = moved.loc[moved.index[-1] - pd.Timedelta('1d'):]
        self.assertLess(first_day.abs().max(), 1.0)
        self.assertAlmostEqual(last_day.abs().max(), 4.0, places=1)

    def test_an_amplitude_growth_adds_no_level_once_it_has_settled(self):
        series = self._flat()
        start = series.index[0]
        moved = monitoring.inject_anomaly(
            series, 'amplitude', 4.0, start=start, duration='1d')
        # A wider swing is not a shifted one: over whole cycles at the settled
        # amplitude the injection averages to zero. The ramp itself is excluded
        # deliberately, because a rising envelope weights the two halves of each
        # cycle differently and must leave a small mean behind.
        settled = moved.loc[start + pd.Timedelta('1d'):
                            start + pd.Timedelta('31d')]
        self.assertAlmostEqual(float(settled.mean()), 0.0, places=2)

    def test_a_phase_change_is_in_quadrature_with_an_amplitude_growth(self):
        series = self._flat()
        start = series.index[0]
        amplitude = monitoring.inject_anomaly(
            series, 'amplitude', 1.0, start=start, duration='1h')
        phase = monitoring.inject_anomaly(
            series, 'phase', 1.0, start=start, duration='1h')
        window = slice(start + pd.Timedelta('2d'), start + pd.Timedelta('32d'))
        overlap = float((amplitude.loc[window] * phase.loc[window]).mean())
        self.assertAlmostEqual(overlap, 0.0, places=2)

    def test_a_drift_accumulates_at_its_stated_yearly_rate(self):
        series = self._flat()
        start = series.index[0]
        moved = monitoring.inject_anomaly(series, 'drift', 12.0, start=start)
        after_30_days = float(moved.loc[start + pd.Timedelta('30d')])
        self.assertAlmostEqual(after_30_days, 12.0 * 30.0 / 365.25, places=3)

    def test_a_drift_needs_no_duration_and_leaves_the_past_alone(self):
        series = self._flat()
        start = series.index[500]
        moved = monitoring.inject_anomaly(series, 'drift', 5.0, start=start)
        np.testing.assert_allclose(moved.iloc[:500], 0.0, atol=1e-12)
        self.assertGreater(float(moved.iloc[-1]), 0.0)

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            monitoring.inject_anomaly(
                self._flat(), 'settlement', 1.0, start='2024-01-02')

    def test_the_existing_kinds_are_untouched(self):
        series = self._flat(200)
        start = series.index[50]
        step = monitoring.inject_anomaly(series, 'step', 3.0, start=start)
        self.assertAlmostEqual(float(step.iloc[-1]), 3.0)
        self.assertAlmostEqual(float(step.iloc[49]), 0.0)


class TestDetectabilityKinds(unittest.TestCase):

    def test_the_swept_kind_reaches_the_injector(self):
        rng = np.random.default_rng(0)
        index = pd.date_range('2024-01-01', periods=6 * 24 * 40, freq='20min')
        residuals = pd.Series(rng.normal(0.0, 1.0, len(index)), index=index)
        curve = monitoring.detectability_curve(
            residuals, 0.0, 1.0, magnitudes=[0.01, 12.0], durations=['72h'],
            kind='amplitude')
        self.assertEqual(list(curve.columns),
                         ['magnitude', 'duration_h', 'detected', 'delay_h'])
        self.assertFalse(bool(curve['detected'].iloc[0]))
        self.assertTrue(bool(curve['detected'].iloc[1]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
