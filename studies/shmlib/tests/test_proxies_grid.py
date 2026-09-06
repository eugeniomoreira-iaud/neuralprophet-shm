"""
Tests for the proxy grid helpers Study 05 adds to shmlib.proxies.

Run from studies/:  python shmlib/tests/test_proxies_grid.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import proxies  # noqa: E402


class TestToNativeGrid(unittest.TestCase):

    def test_hourly_ramp_is_interpolated_to_twenty_minutes(self):
        index = pd.date_range('2024-01-01', periods=4, freq='1h', tz='UTC')
        frame = pd.DataFrame({'tair_era5': [0.0, 3.0, 6.0, 9.0]}, index=index)
        frame.attrs['source'] = 'gs'
        out = proxies.to_native_grid(frame, freq='20min')
        self.assertEqual(out.index.freqstr, '20min')
        self.assertAlmostEqual(out.loc['2024-01-01 00:20', 'tair_era5'], 1.0)
        self.assertAlmostEqual(out.loc['2024-01-01 02:40', 'tair_era5'], 8.0)
        self.assertEqual(out.attrs.get('source'), 'gs')

    def test_a_gap_longer_than_one_native_step_is_not_bridged(self):
        index = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00',
                                  '2024-01-01 02:00', '2024-01-01 05:00'], tz='UTC')
        frame = pd.DataFrame({'tair_gs': [0.0, 1.0, 2.0, 5.0]}, index=index)
        out = proxies.to_native_grid(frame, freq='20min')
        self.assertTrue(np.isnan(out.loc['2024-01-01 03:20', 'tair_gs']))
        self.assertAlmostEqual(out.loc['2024-01-01 00:40', 'tair_gs'], 2.0 / 3.0)

    def test_an_accumulation_is_centred_half_a_step_earlier(self):
        index = pd.date_range('2024-06-01', periods=3, freq='1h', tz='UTC')
        frame = pd.DataFrame({'sr_era5': [0.0, 100.0, 200.0]}, index=index)
        out = proxies.to_native_grid(frame, freq='20min', accumulations=('sr',))
        # value 100 belongs to 00:30, so 00:40 sits a sixth of the way to 200
        self.assertAlmostEqual(out.loc['2024-06-01 00:40', 'sr_era5'], 100.0 + 100.0 / 6.0, places=6)


class TestFillShortGaps(unittest.TestCase):

    def _frame(self):
        index = pd.date_range('2024-01-01', periods=12, freq='20min', tz='UTC')
        values = np.arange(12, dtype=float)
        values[2] = np.nan            # one slot: 20 min gap
        values[5:10] = np.nan         # five slots: 2h between the bracketing observations
        return pd.DataFrame({'tair_gs': values}, index=index)

    def test_gaps_within_the_limit_are_filled_and_flagged(self):
        out = proxies.fill_short_gaps(self._frame(), ['tair_gs'], max_gap='2h')
        self.assertAlmostEqual(out['tair_gs'].iloc[2], 2.0)
        self.assertAlmostEqual(out['tair_gs'].iloc[7], 7.0)
        self.assertTrue(out['tair_gs_filled'].iloc[2])
        self.assertFalse(out['tair_gs_filled'].iloc[3])

    def test_a_gap_beyond_the_limit_stays_missing(self):
        out = proxies.fill_short_gaps(self._frame(), ['tair_gs'], max_gap='1h')
        self.assertAlmostEqual(out['tair_gs'].iloc[2], 2.0)
        self.assertTrue(out['tair_gs'].iloc[5:10].isna().all())
        self.assertFalse(out['tair_gs_filled'].iloc[5:10].any())

    def test_no_flag_column_when_flag_is_false(self):
        out = proxies.fill_short_gaps(self._frame(), ['tair_gs'], flag=False)
        self.assertNotIn('tair_gs_filled', out.columns)


class TestRegressorSets(unittest.TestCase):

    def _record(self):
        index = pd.date_range('2024-01-01', periods=24, freq='20min', tz='UTC')
        sr = np.arange(24, dtype=float)
        tair = np.full(24, 10.0)
        tair[5] = np.nan
        record = pd.DataFrame({'y': np.linspace(0.0, 1.0, 24), 'tair_a': tair,
                               'rh_a': 50.0, 'sr_a': sr, 'tair_b': 11.0,
                               'rh_b': 51.0, 'sr_b': sr + 100.0}, index=index)
        record.loc[record.index[3], 'y'] = np.nan
        mapping = {'a': {'tair': 'tair_a', 'rh': 'rh_a', 'sr': 'sr_a'},
                   'b': {'tair': 'tair_b', 'rh': 'rh_b', 'sr': 'sr_b'}}
        return record, mapping

    def test_radiation_is_delayed_by_whole_slots_and_the_frame_is_assembled(self):
        record, mapping = self._record()
        sets, frame = proxies.build_regressor_sets(
            record, mapping, target='y', fill_max_gap='2h',
            radiation_delay_h=1.0, freq='20min')
        pd.testing.assert_series_equal(sets['a']['sr'], record['sr_a'].shift(3),
                                       check_names=False)
        self.assertEqual(list(frame.columns)[0], 'y')
        self.assertIn('sr_b', frame.columns)
        self.assertTrue(frame['tair_a_filled'].iloc[5])
        self.assertAlmostEqual(frame['tair_a'].iloc[5], 10.0)
        self.assertFalse(frame['sr_a_filled'].any())
        self.assertTrue(np.isnan(frame['y'].iloc[3]))   # the target is never filled

    def test_coverage_table_has_one_row_per_set_and_role_plus_the_target(self):
        record, mapping = self._record()
        sets, frame = proxies.build_regressor_sets(record, mapping, target='y',
                                                   freq='20min')
        coverage = proxies.regressor_set_coverage(sets, frame['y'], 'inc')
        self.assertEqual(len(coverage), 2 * 3 + 1)
        self.assertEqual(list(coverage.columns),
                         ['set', 'role', 'accepted', 'filled', 'coverage'])
        target_row = coverage.iloc[-1]
        self.assertEqual(target_row['set'], 'target')
        self.assertEqual(target_row['accepted'], 23)
        self.assertAlmostEqual(target_row['coverage'], 23 / 24)
        a_tair = coverage[(coverage['set'] == 'a') & (coverage['role'] == 'tair')].iloc[0]
        self.assertEqual(a_tair['filled'], 1)
        self.assertEqual(a_tair['accepted'], 24)


if __name__ == '__main__':
    unittest.main(verbosity=2)
