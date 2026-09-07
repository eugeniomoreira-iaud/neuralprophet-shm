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


def _quiet(n=2000, seed=0, freq='20min'):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq=freq)
    return pd.Series(rng.normal(0.0, 1.0, n), index=index)


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

    def test_a_short_window_catches_a_jump_the_24h_default_buries_in_the_diurnal_cycle(self):
        # Fix-round ruling 7: a 24-hour centred median leaves the diurnal
        # cycle itself in the "departure", inflating the MAD-based scale
        # until no realistic swing crosses the threshold. A window short
        # enough to track the cycle out (two hours) isolates a genuine
        # twenty-minute jump instead.
        index = pd.date_range('2024-01-01', periods=72 * 10, freq='20min', tz='UTC')
        hours = index.hour + index.minute / 60.0
        tair = pd.Series(5.0 * np.cos(2 * np.pi * (hours - 14) / 24.0), index=index)
        alarm_slot = index[72 * 5]
        tair.loc[alarm_slot] += 3.0
        channels = pd.DataFrame({'tair': tair})
        alarm = pd.Series(False, index=index)
        alarm.loc[alarm_slot] = True
        short_window = monitoring.channel_coincidence(
            alarm, channels, window='2h', threshold=5.0)
        default_window = monitoring.channel_coincidence(alarm, channels)
        self.assertEqual(short_window.loc[alarm_slot], 'environment')
        self.assertEqual(default_window.loc[alarm_slot], 'unattributed')


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


class TestChartSeries(unittest.TestCase):

    def test_returns_the_four_names_and_a_finite_phi(self):
        residual = _ar1(n=72 * 60, phi=0.9)
        series, phi = monitoring.chart_series(
            residual, '20min', 60, residual.index[0], residual.index[-1])
        self.assertEqual(set(series), {'fast', 'daily_amplitude', 'daily_phase', 'slow'})
        self.assertTrue(np.isfinite(phi))

    def test_a_day_too_short_to_fit_is_a_gap_not_a_dropped_row(self):
        # daily_harmonic silently drops a day below min_slots rather than
        # carrying it as missing, which leaves its own output on an
        # irregular grid the moment the window holds one short day --
        # exactly what alarm_episodes and average_run_length refuse to
        # chart downstream, in tune_limit_to_budget and run_chart.
        residual = _ar1(n=72 * 30, phi=0.9)
        residual.iloc[72 * 10:72 * 10 + 40] = np.nan  # day 10 drops below min_slots=60
        series, _ = monitoring.chart_series(
            residual, '20min', 60, residual.index[0], residual.index[-1])
        for name in ('daily_amplitude', 'daily_phase'):
            diffs = np.unique(np.diff(series[name].index.to_numpy()))
            self.assertEqual(len(diffs), 1, f'{name} is not on a regular grid')
        self.assertTrue(np.isnan(series['daily_amplitude'].iloc[10]))


class TestTuneLimitToBudget(unittest.TestCase):

    def test_a_low_budget_returns_the_smallest_candidate_and_a_table(self):
        series = _quiet(3000)
        L, sweep = monitoring.tune_limit_to_budget(
            series, series.index[0], series.index[-1], budget_days=0.1,
            candidates=np.arange(2.0, 6.01, 0.5), lam=0.2, k=0.5, h=5.0,
            joint_window='6h', freq='20min')
        self.assertEqual(L, 2.0)
        self.assertIn('arl_days', sweep.columns)

    def test_an_impossible_budget_raises(self):
        # A single, tight candidate alarms often enough on plain noise (a
        # finite average run length of about ten days), so a budget far
        # beyond what any candidate in the grid can reach is genuinely
        # unreachable rather than trivially satisfied by an all-quiet run.
        series = _quiet(3000)
        with self.assertRaises(RuntimeError):
            monitoring.tune_limit_to_budget(
                series, series.index[0], series.index[-1], budget_days=1e9,
                candidates=(2.0,), lam=0.2, k=0.5, h=5.0,
                joint_window='6h', freq='20min')


class TestRunChart(unittest.TestCase):

    def test_returns_the_four_keys_with_the_episodes_frame_columns(self):
        series = _quiet(3000)
        reference = monitoring.reference_stats(
            series, start=series.index[0], end=series.index[1500])
        out = monitoring.run_chart(
            series, reference, L=3.0, lam=0.2, k=0.5, h=5.0, joint_window='6h',
            monitored_start=series.index[1500], freq='20min')
        self.assertEqual(set(out), {'ewma', 'cusum', 'joint', 'episodes'})
        self.assertEqual(list(out['episodes'].columns),
                         ['start', 'end', 'duration_h', 'n_slots', 'mean_z', 'peak_abs_z'])

    def test_mean_z_is_the_standardised_mean_not_the_raw_one(self):
        # Fix-round ruling 6: GM_11's mean_z/peak_abs_z read as z-scores, so
        # they must be computed on the standardised series, not on the
        # chart's own native units (hours for the daily phase, mdeg for the
        # slow chart's daily mean). The episode's own start/end are read
        # back from run_chart's own output, so this holds regardless of
        # exactly which slots the joint alarm picks out.
        index = pd.date_range('2024-01-01', periods=300, freq='20min')
        rng = np.random.default_rng(3)
        values = rng.normal(50.0, 0.5, 300)  # baseline near 50, native units
        values[150:160] += 40.0              # a clear, sustained spike
        series = pd.Series(values, index=index)
        reference = monitoring.reference_stats(series, start=index[0], end=index[99])
        out = monitoring.run_chart(
            series, reference, L=3.0, lam=0.3, k=0.5, h=3.0, joint_window='1h',
            monitored_start=index[0], freq='20min')
        self.assertFalse(out['episodes'].empty)
        episode = out['episodes'].iloc[0]
        watched_z = (series - reference['mu']) / reference['sigma']
        window = watched_z.loc[episode['start']:episode['end']]
        self.assertAlmostEqual(episode['mean_z'], window.mean(), places=6)
        self.assertAlmostEqual(episode['peak_abs_z'], window.abs().max(), places=6)
        # Guard against silently reverting to the raw-units behaviour: the
        # episode's own native-unit mean sits near 50-90, nowhere close to a
        # z-score.
        raw_window = series.loc[episode['start']:episode['end']]
        self.assertGreater(abs(episode['mean_z'] - raw_window.mean()), 1.0)


class TestAttributeEpisodes(unittest.TestCase):

    def test_one_episode_by_the_mode_and_another_by_the_default(self):
        index = pd.date_range('2024-01-01', periods=100, freq='20min')
        episodes = pd.DataFrame({'start': [index[10], index[60]],
                                 'end': [index[12], index[62]]})
        labels = pd.Series(['instrument', 'instrument', 'environment'],
                           index=[index[10], index[11], index[12]])
        out = monitoring.attribute_episodes(episodes, labels)
        self.assertEqual(out['attribution'].iloc[0], 'instrument')
        self.assertEqual(out['attribution'].iloc[1], 'unattributed')


class TestDailyResponseAmplitude(unittest.TestCase):

    def test_recovers_the_amplitude_of_a_synthetic_daily_sinusoid(self):
        index = pd.date_range('2020-01-01', periods=72 * 30, freq='20min')
        hours = index.hour + index.minute / 60.0
        components = pd.DataFrame({
            'future_regressor_tair': 5.0 * np.cos(2 * np.pi * (hours - 14) / 24.0),
            'season_daily': 0.0}, index=index)
        amplitude = monitoring.daily_response_amplitude(
            components, ('2020-01-05', '2020-01-10'), window=72, min_slots=60)
        self.assertAlmostEqual(amplitude, 5.0, delta=0.2)


class TestDetectabilityByMechanism(unittest.TestCase):

    def test_two_mechanisms_return_both_labels(self):
        residual = _quiet(6000)
        reference = monitoring.reference_stats(
            residual, start=residual.index[0], end=residual.index[-1])
        tuned = {'fast': {'reference': reference, 'L': 3.0},
                'slow': {'reference': reference, 'L': 3.0}}
        specs = {'fast': {'lam': 0.2}, 'slow': {'lam': 0.1}}
        mechanisms = {'step': ('fast', 'residual'), 'drift': ('slow', 'residual')}
        magnitudes = {'step': (5.0,), 'drift': (5.0,)}
        out = monitoring.detectability_by_mechanism(
            residual, tuned, specs, mechanisms, magnitudes, durations=('24h',),
            freq='20min', k=0.5, h=5.0, phi=None, response_window='24h',
            injection_starts=None, min_slots=60)
        self.assertEqual(set(out['mechanism']), {'step', 'drift'})

    def test_a_dict_of_durations_gives_each_mechanism_its_own_duration_h(self):
        # Fix-round ruling 1: a mechanism whose damage accumulates on a
        # different timescale (drift, watched for weeks) is not forced
        # through the same duration grid as the others.
        residual = _quiet(6000)
        reference = monitoring.reference_stats(
            residual, start=residual.index[0], end=residual.index[-1])
        tuned = {'fast': {'reference': reference, 'L': 3.0},
                'slow': {'reference': reference, 'L': 3.0}}
        specs = {'fast': {'lam': 0.2}, 'slow': {'lam': 0.1}}
        mechanisms = {'step': ('fast', 'residual'), 'drift': ('slow', 'residual')}
        magnitudes = {'step': (5.0,), 'drift': (5.0,)}
        durations = {'step': ('24h',), 'drift': ('720h', '1440h')}
        out = monitoring.detectability_by_mechanism(
            residual, tuned, specs, mechanisms, magnitudes, durations=durations,
            freq='20min', k=0.5, h=5.0, phi=None, response_window='24h',
            injection_starts=None, min_slots=60)
        step_hours = set(out.loc[out['mechanism'] == 'step', 'duration_h'])
        drift_hours = set(out.loc[out['mechanism'] == 'drift', 'duration_h'])
        self.assertEqual(step_hours, {24.0})
        self.assertEqual(drift_hours, {720.0, 1440.0})

    def test_all_charts_true_sweeps_every_mechanism_on_every_chart(self):
        # Fix-round ruling 2: with two mechanisms on two charts and
        # all_charts=True, four (mechanism, chart) pairs come back and
        # exactly one per mechanism is primary.
        residual = _quiet(6000)
        reference = monitoring.reference_stats(
            residual, start=residual.index[0], end=residual.index[-1])
        tuned = {'fast': {'reference': reference, 'L': 3.0},
                'slow': {'reference': reference, 'L': 3.0}}
        specs = {'fast': {'lam': 0.2}, 'slow': {'lam': 0.1}}
        mechanisms = {'step': ('fast', 'residual'), 'drift': ('slow', 'residual')}
        magnitudes = {'step': (5.0,), 'drift': (5.0,)}
        out = monitoring.detectability_by_mechanism(
            residual, tuned, specs, mechanisms, magnitudes, durations=('24h',),
            freq='20min', k=0.5, h=5.0, phi=None, response_window='24h',
            injection_starts=None, min_slots=60, all_charts=True)
        pairs = set(zip(out['mechanism'], out['chart']))
        self.assertEqual(pairs, {('step', 'fast'), ('step', 'slow'),
                                 ('drift', 'fast'), ('drift', 'slow')})
        for mechanism in ('step', 'drift'):
            primaries = out.loc[out['mechanism'] == mechanism, 'primary']
            self.assertEqual(primaries.sum(), 1)


class TestDetectionThresholds(unittest.TestCase):

    def test_smallest_magnitudes_and_the_all_zero_case(self):
        detectability = pd.DataFrame([
            {'mechanism': 'step', 'chart': 'fast', 'primary': True,
             'magnitude': 1.0, 'duration_h': 24.0, 'detected': 0.0, 'delay_h': np.nan},
            {'mechanism': 'step', 'chart': 'fast', 'primary': True,
             'magnitude': 2.0, 'duration_h': 24.0, 'detected': 0.5, 'delay_h': 3.0},
            {'mechanism': 'step', 'chart': 'fast', 'primary': True,
             'magnitude': 4.0, 'duration_h': 24.0, 'detected': 1.0, 'delay_h': 1.0},
            {'mechanism': 'drift', 'chart': 'slow', 'primary': True,
             'magnitude': 1.0, 'duration_h': 720.0, 'detected': 0.0, 'delay_h': np.nan},
            {'mechanism': 'drift', 'chart': 'slow', 'primary': True,
             'magnitude': 2.0, 'duration_h': 720.0, 'detected': 0.0, 'delay_h': np.nan},
        ])
        out = monitoring.detection_thresholds(detectability)
        step = out.loc[out['mechanism'] == 'step'].iloc[0]
        self.assertEqual(step['smallest_any'], 2.0)
        self.assertEqual(step['smallest_all'], 4.0)
        self.assertEqual(step['delay_h_at_smallest_all'], 1.0)
        drift = out.loc[out['mechanism'] == 'drift'].iloc[0]
        self.assertTrue(np.isnan(drift['smallest_any']))
        self.assertTrue(np.isnan(drift['smallest_all']))
        self.assertTrue(np.isnan(drift['delay_h_at_smallest_all']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
