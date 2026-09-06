"""
Tests for the Model A additions Study 05 makes to shmlib.prediction.

Run from studies/:  python 05_greybox_monitoring/tests/test_model_a.py
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


def _frame(n=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    hours = np.arange(n)
    tair = 10.0 + 6.0 * np.sin(2 * np.pi * hours / 24.0) + rng.normal(0, 0.2, n)
    y = 50.0 - 2.5 * tair + 0.01 * hours + rng.normal(0, 0.3, n)
    weights = prediction.seasonal_weights(index)
    return pd.DataFrame({'y': y, 'tair': tair, 'summer_w': weights['summer_w'],
                         'winter_w': weights['winter_w']}, index=index)


class TestBacktestAdditions(unittest.TestCase):

    def test_defaults_are_unchanged(self):
        frame = _frame()
        model, out = prediction.neuralprophet_backtest(
            frame.iloc[:1000], frame.iloc[1000:], regressors=('tair',),
            task='nowcast', epochs=3, freq='1h')
        self.assertEqual(model.config_trend.trend_reg, 0)
        self.assertEqual(len(out), len(frame) - 1000)

    def test_orders_regularisation_and_validation_reach_the_model(self):
        frame = _frame()
        model, _ = prediction.neuralprophet_backtest(
            frame.iloc[:1000], frame.iloc[1000:], regressors=('tair',),
            task='nowcast', epochs=3, freq='1h', trend_reg=1.5,
            growth='linear', yearly_order=2, daily_order=3,
            learning_rate=0.02, validation=frame.iloc[1000:])
        # NeuralProphet 0.8.0 rescales a positive trend_reg by 0.001 once
        # changepoints exist (growth='linear' with the wrapper's default
        # n_changepoints=10 here), and would zero it instead with none, so
        # the value stored on the fitted model is never the raw input.
        self.assertAlmostEqual(model.config_trend.trend_reg, 0.001 * 1.5)
        self.assertTrue(hasattr(model, 'fit_metrics_'))
        self.assertIn('MAE_val', model.fit_metrics_.columns)

    def test_conditional_daily_seasonality_is_decomposed(self):
        frame = _frame()
        model, out = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=3,
            freq='1h', daily_order=3,
            conditional_seasonality={'daily_summer': 'summer_w',
                                     'daily_winter': 'winter_w'},
            decompose=True)
        components = prediction.decompose_components(
            model, frame, regressors=('tair',))
        self.assertIn('season_daily_summer', components.columns)
        self.assertIn('season_daily_winter', components.columns)

    def test_lagged_regressor_in_a_nowcast_model(self):
        frame = _frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=(), task='nowcast', epochs=3, freq='1h',
            lagged_regressors=('tair',), lagged_n_lags=6)
        weights = model.model.get_covar_weights()
        self.assertIn('tair', weights)

    def test_tz_aware_index_matches_predictions_to_test_rows(self):
        # Regression test for _long_predictions: NeuralProphet's predict()
        # returns 'ds' tz-naive, converted through UTC internally (e.g. a
        # Europe/Rome midnight comes back as 23:00 the previous day), so
        # matching predictions back to a tz-aware test index must compare
        # UTC-naive representations rather than raw Timestamps, or every
        # row is silently dropped instead of being reported.
        frame = _frame()
        frame = frame.tz_convert('Europe/Rome')
        model, out = prediction.neuralprophet_backtest(
            frame.iloc[:200], frame.iloc[200:], regressors=('tair',),
            task='nowcast', epochs=2, freq='1h')
        self.assertEqual(len(out), len(frame) - 200)
        self.assertTrue(pd.DatetimeIndex(out['ds']).equals(frame.index[200:]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
