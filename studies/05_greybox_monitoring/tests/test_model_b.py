"""
Tests for Model B's learned impulse response (spec D9).

Run from studies/:  python 05_greybox_monitoring/tests/test_model_b.py
"""
import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction  # noqa: E402

logging.getLogger('NP').setLevel(logging.ERROR)


class TestLaggedWeights(unittest.TestCase):

    def test_the_peak_weight_sits_at_the_injected_lag(self):
        rng = np.random.default_rng(0)
        n = 24 * 60
        index = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
        x = rng.normal(0, 1, n)
        y = 5.0 + 3.0 * np.roll(x, 3) + rng.normal(0, 0.1, n)
        frame = pd.DataFrame({'y': y, 'x': x}, index=index)
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=(), task='nowcast', epochs=20, freq='1h',
            yearly=False, daily_order=1, quantiles=(), lagged_regressors=('x',),
            lagged_n_lags=8, learning_rate=0.05)
        weights = prediction.lagged_regressor_weights(model)
        peak = int(weights.loc[weights['weight'].abs().idxmax(), 'lag'])
        self.assertEqual(peak, 3)

    def test_summary_reads_delay_and_time_constant_from_a_known_response(self):
        lags = np.arange(1, 25)
        w = 2.0 * np.exp(-(lags - 1) / 4.0) * (lags >= 1)
        weights = pd.DataFrame({'regressor': 'x', 'lag': lags, 'weight': w})
        summary = prediction.impulse_response_summary(weights, dt_hours=1.0)
        self.assertAlmostEqual(summary.loc[0, 'tau_h'], 4.0, delta=1.0)
        self.assertGreater(summary.loc[0, 'gain'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
