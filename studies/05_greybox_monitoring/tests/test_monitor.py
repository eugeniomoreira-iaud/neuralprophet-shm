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


if __name__ == '__main__':
    unittest.main(verbosity=2)
