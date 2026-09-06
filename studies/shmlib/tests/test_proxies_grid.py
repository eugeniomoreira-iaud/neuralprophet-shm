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
        out = proxies.to_native_grid(frame, freq='20min')
        self.assertEqual(out.index.freqstr, '20min')
        self.assertAlmostEqual(out.loc['2024-01-01 00:20', 'tair_era5'], 1.0)
        self.assertAlmostEqual(out.loc['2024-01-01 02:40', 'tair_era5'], 8.0)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
