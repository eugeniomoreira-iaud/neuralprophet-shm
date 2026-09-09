"""
Tests for the temporal alignment side quest.

Run from studies/:  python shmlib/tests/test_temporal_alignment.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import temporal_alignment  # noqa: E402


def _daily_signal(index, phase=0.0, trend=0.0):
    """A repeatable diurnal signal with enough structure for lag tests."""
    slots = np.arange(len(index), dtype=float)
    radians = 2.0 * np.pi * slots / 72.0 + phase
    return np.sin(radians) + 0.35 * np.cos(2.0 * radians) + trend * slots


class TestPairedShiftScan(unittest.TestCase):

    def test_positive_shift_assigns_sensor_observation_to_a_later_timestamp(self):
        index = pd.date_range('2025-01-01', periods=72 * 30, freq='20min')
        reference = pd.Series(_daily_signal(index), index=index, name='reference')
        sensor = reference.copy()
        sensor.index = sensor.index - pd.Timedelta(minutes=60)
        sensor.name = 'sensor'

        out = temporal_alignment.paired_shift_scan(
            sensor, reference, [-60, 0, 60], freq='20min',
            min_pairs=100, min_days=10)

        best = out.sort_values('r_daily', ascending=False).iloc[0]
        self.assertEqual(int(best['shift_minutes']), 60)
        self.assertGreater(best['r_daily'], 0.999)

    def test_injected_delay_is_recovered_with_irregular_holes(self):
        index = pd.date_range('2025-06-01', periods=72 * 35, freq='20min')
        reference = pd.Series(_daily_signal(index, phase=0.3), index=index)
        sensor = reference.copy()
        sensor.index = sensor.index - pd.Timedelta(minutes=40)

        sensor.iloc[100:125] = np.nan
        sensor.iloc[900:940] = np.nan
        reference.iloc[300:330] = np.nan
        reference.iloc[1500:1520] = np.nan

        out = temporal_alignment.paired_shift_scan(
            sensor, reference, [-80, -40, 0, 40, 80], freq='20min',
            min_pairs=200, min_days=14)

        best = out.sort_values('r_daily', ascending=False).iloc[0]
        self.assertEqual(int(best['shift_minutes']), 40)
        self.assertGreater(best['r_daily'], 0.98)

    def test_all_shifts_use_identical_common_support(self):
        index = pd.date_range('2025-01-01', periods=72 * 20, freq='20min')
        reference = pd.Series(_daily_signal(index), index=index)
        sensor = pd.Series(_daily_signal(index, phase=0.2), index=index)
        sensor.iloc[:6] = np.nan
        sensor.iloc[100:112] = np.nan
        reference.iloc[-6:] = np.nan
        reference.iloc[500:517] = np.nan

        out = temporal_alignment.paired_shift_scan(
            sensor, reference, [-40, 0, 40], freq='20min',
            min_pairs=50, min_days=5)

        self.assertEqual(out['n_pairs'].nunique(), 1)
        self.assertEqual(out['n_days'].nunique(), 1)

    def test_shift_does_not_compress_index_gaps(self):
        index = pd.date_range('2025-01-01', periods=72 * 16, freq='20min')
        reference = pd.Series(_daily_signal(index), index=index)
        sensor = reference.copy()
        sensor.index = sensor.index - pd.Timedelta(minutes=20)
        sensor.iloc[200:215] = np.nan
        sensor.iloc[600:621] = np.nan

        out = temporal_alignment.paired_shift_scan(
            sensor, reference, [0, 20], freq='20min',
            min_pairs=200, min_days=14)

        r_zero = out.loc[out['shift_minutes'] == 0, 'r_daily'].iloc[0]
        r_twenty = out.loc[out['shift_minutes'] == 20, 'r_daily'].iloc[0]
        self.assertGreater(r_twenty, 0.999)
        # A one-grid-step (20 min) residual misalignment on a smooth
        # 24-hour signal is a ~5-degree phase error, which correlates at
        # roughly 0.995 by construction (verified analytically against the
        # two-harmonic daily_signal) -- not the <0.99 an actual
        # index-compression bug would produce. 0.999 keeps the two shifts
        # clearly distinguished without asserting an unreachable bound.
        self.assertLess(r_zero, 0.999)

    def test_period_boundaries_are_applied_before_timestamp_shift(self):
        index = pd.date_range('2025-01-01', periods=72 * 18, freq='20min')
        reference = pd.Series(_daily_signal(index), index=index)
        sensor = reference.copy()
        sensor.index = sensor.index - pd.Timedelta(minutes=60)

        start = pd.Timestamp('2025-01-03 00:00')
        end = pd.Timestamp('2025-01-16 23:40')
        pre_period = sensor.index < start
        sensor.loc[pre_period] = 1000.0 + np.arange(pre_period.sum())

        out = temporal_alignment.paired_shift_scan(
            sensor, reference, [60], freq='20min', start=start, end=end,
            min_pairs=100, min_days=10)

        self.assertGreater(out['r_daily'].iloc[0], 0.999)

    def test_daily_demeaning_removes_slow_level_agreement(self):
        index = pd.date_range('2025-01-01', periods=72 * 20, freq='20min')
        day = np.repeat(np.arange(20, dtype=float), 72)
        within_day = np.tile(np.linspace(-1.0, 1.0, 72), 20)
        sensor = pd.Series(day + within_day, index=index)
        reference = pd.Series(day - within_day, index=index)

        out = temporal_alignment.paired_shift_scan(
            sensor, reference, [0], freq='20min',
            min_pairs=100, min_days=10)

        self.assertGreater(out['r_levels'].iloc[0], 0.9)
        self.assertLess(out['r_daily'].iloc[0], -0.99)

    def test_off_grid_shift_is_rejected(self):
        index = pd.date_range('2025-01-01', periods=72 * 15, freq='20min')
        series = pd.Series(_daily_signal(index), index=index)

        with self.assertRaises(ValueError):
            temporal_alignment.paired_shift_scan(
                series, series, [10], freq='20min',
                min_pairs=100, min_days=10)

    def test_insufficient_support_returns_nan_correlations(self):
        index = pd.date_range('2025-01-01', periods=50, freq='20min')
        series = pd.Series(np.arange(50.0), index=index)

        out = temporal_alignment.paired_shift_scan(
            series, series, [0], freq='20min',
            min_pairs=200, min_days=14)

        row = out.iloc[0]
        self.assertEqual(row['n_pairs'], 50)
        self.assertEqual(row['n_days'], 1)
        self.assertTrue(np.isnan(row['r_levels']))
        self.assertTrue(np.isnan(row['r_daily']))

    def test_shared_shift_preserves_within_package_correlations_and_values(self):
        index = pd.date_range('2025-07-01', periods=72 * 16, freq='20min')
        sensor = pd.DataFrame({
            'inc_comp_cleaned': _daily_signal(index, phase=0.1),
            'tair_str': _daily_signal(index, phase=0.4),
            'rh_str': _daily_signal(index, phase=1.2),
        }, index=index)
        shifted = sensor.copy()
        shifted.index = shifted.index + pd.Timedelta(minutes=60)

        before = sensor.corr()
        after = shifted.corr()
        pd.testing.assert_frame_equal(before, after)
        np.testing.assert_array_equal(
            sensor['inc_comp_cleaned'].to_numpy(),
            shifted['inc_comp_cleaned'].to_numpy())


class TestBootstrapCorrelationDelta(unittest.TestCase):

    def test_bootstrap_delta_is_deterministic(self):
        index = pd.date_range('2025-01-01', periods=72 * 70, freq='20min')
        reference = pd.Series(_daily_signal(index), index=index)
        baseline = pd.Series(_daily_signal(index, phase=0.35), index=index)
        candidate = pd.Series(_daily_signal(index, phase=0.05), index=index)

        first = temporal_alignment.bootstrap_correlation_delta(
            reference, baseline, candidate, block_days=7, n_boot=200,
            seed=20260906, min_blocks=8)
        second = temporal_alignment.bootstrap_correlation_delta(
            reference, baseline, candidate, block_days=7, n_boot=200,
            seed=20260906, min_blocks=8)

        self.assertEqual(first, second)
        self.assertGreater(first['delta'], 0.0)
        self.assertEqual(first['n_boot_valid'], 200)

    def test_bootstrap_refuses_intervals_with_too_few_blocks(self):
        index = pd.date_range('2025-01-01', periods=72 * 21, freq='20min')
        reference = pd.Series(_daily_signal(index), index=index)
        baseline = pd.Series(_daily_signal(index, phase=0.3), index=index)
        candidate = pd.Series(_daily_signal(index, phase=0.1), index=index)

        out = temporal_alignment.bootstrap_correlation_delta(
            reference, baseline, candidate, block_days=7, n_boot=200,
            seed=20260906, min_blocks=8)

        self.assertTrue(np.isnan(out['ci_low']))
        self.assertTrue(np.isnan(out['ci_high']))
        self.assertEqual(out['n_blocks'], 3)
        self.assertEqual(out['n_boot_valid'], 0)

    def test_absolute_delta_uses_absolute_correlations(self):
        index = pd.date_range('2025-01-01', periods=72 * 70, freq='20min')
        reference = pd.Series(_daily_signal(index), index=index)
        baseline = -reference + 0.4 * pd.Series(_daily_signal(index, phase=1.0), index=index)
        candidate = -reference + 0.05 * pd.Series(_daily_signal(index, phase=1.0), index=index)

        signed = temporal_alignment.bootstrap_correlation_delta(
            reference, baseline, candidate, block_days=7, n_boot=100,
            seed=20260906, min_blocks=8, absolute=False)
        absolute = temporal_alignment.bootstrap_correlation_delta(
            reference, baseline, candidate, block_days=7, n_boot=100,
            seed=20260906, min_blocks=8, absolute=True)

        self.assertLess(signed['delta'], 0.0)
        self.assertGreater(absolute['delta'], 0.0)


class TestRunExperimentContract(unittest.TestCase):

    def test_runner_contract_preserves_compensation_and_writes_expected_outputs(self):
        index = pd.date_range('2025-01-01', periods=72 * 16, freq='20min')
        rh = 50.0 + _daily_signal(index, phase=0.5)
        archive = pd.DataFrame({
            'inc_comp_cleaned': _daily_signal(index),
            'inc_spike': False,
            'tair': _daily_signal(index, phase=0.1) + 20.0,
            'rh': rh,
            # load_sensor_package reads humidity from the archive's own
            # current- and legacy-era verdict columns, not from a generic
            # 'rh' column: n_rh_ok (current block) and st02_rh_ok (legacy
            # block, TARGET_STATION='st02').
            'n_rh_ok': rh,
            'st02_rh_ok': rh,
        }, index=index)
        archive.index.name = 'datetime'

        ground_index = pd.date_range(index.min(), index.max(), freq='30min')
        ground = pd.DataFrame({
            'Temp': np.interp(ground_index.view('int64'), index.view('int64'), archive['tair']),
            'Umid': np.interp(ground_index.view('int64'), index.view('int64'), archive['rh']),
            'Rad.Sol.': 100.0,
        }, index=ground_index)
        ground.index.name = 'datetime'

        era_index = pd.date_range(index.min(), index.max(), freq='1h')
        era5 = pd.DataFrame({
            'temperature (degC)': np.interp(era_index.view('int64'), index.view('int64'), archive['tair']),
            'relative_humidity (0-1)': np.interp(era_index.view('int64'), index.view('int64'), archive['rh']) / 100.0,
            'surface_solar_radiation (W/m^2)': 100.0,
        }, index=era_index)
        era5.index.name = 'datetime'

        periods = [{
            'name': 'synthetic_train',
            'era': 'current',
            'role': 'train',
            'start': '2025-01-01',
            'end': '2025-01-16',
        }]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive_path = tmp_path / 'archive.csv'
            ground_path = tmp_path / 'ground.csv'
            era5_path = tmp_path / 'era5.csv'
            out_dir = tmp_path / 'outputs'
            archive.to_csv(archive_path)
            ground.to_csv(ground_path)
            era5.to_csv(era5_path)

            before = pd.read_csv(archive_path, index_col=0, parse_dates=True)
            paths = temporal_alignment.run_experiment(
                archive_path, ground_path, era5_path, out_dir,
                shifts_minutes=[0], periods=periods, freq='20min',
                n_boot=20, seed=20260906, min_pairs=100, min_days=10,
                block_days=7, min_blocks=2)
            after = pd.read_csv(archive_path, index_col=0, parse_dates=True)

            self.assertEqual(
                set(paths),
                {'scan', 'choices', 'validation', 'manifest'})
            for output in paths.values():
                self.assertTrue(Path(output).exists())
            pd.testing.assert_series_equal(
                before['inc_comp_cleaned'], after['inc_comp_cleaned'])
            self.assertNotIn('inc_comp', before.columns)
            self.assertNotIn('inc_comp', after.columns)

            manifest = json.loads(Path(paths['manifest']).read_text())
            self.assertEqual(manifest['parameters']['freq'], '20min')
            self.assertEqual(manifest['parameters']['shifts_minutes'], [0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
