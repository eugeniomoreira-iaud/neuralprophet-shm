"""
Smoke tests for the NeuralProphet 0.8.0 capabilities Study 05 relies on.

Run from studies/:  python 05_greybox_monitoring/tests/test_neuralprophet_capabilities.py

Each test states one capability the design assumes (spec D6-D9, D14). A
failure here means the design's fallback for that capability applies.
"""
import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

logging.getLogger('NP').setLevel(logging.ERROR)


def _frame(n=24 * 40, freq='1h', seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range('2024-01-01', periods=n, freq=freq)
    hours = np.arange(n)
    x = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24.0)
    y = 100.0 - 2.5 * np.roll(x, 3) + rng.normal(0, 0.3, n)
    doy = ds.dayofyear.to_numpy()
    summer_w = 0.5 * (1 - np.cos(2 * np.pi * (doy - 15) / 365.0))
    return pd.DataFrame({'ds': ds, 'y': y, 'x': x,
                         'summer_w': summer_w, 'winter_w': 1 - summer_w})


def _model(**kwargs):
    from neuralprophet import NeuralProphet
    base = dict(n_lags=0, n_forecasts=1, yearly_seasonality=False,
                weekly_seasonality=False, daily_seasonality=True,
                epochs=5, learning_rate=0.05, impute_missing=False,
                drop_missing=False, quantiles=[0.05, 0.95])
    base.update(kwargs)
    return NeuralProphet(**base)


class TestLaggedRegressorWithoutAutoregression(unittest.TestCase):
    """Spec D9: Model B carries lagged regressors with n_lags=0 on the target."""

    def test_fit_and_predict_with_lagged_regressor_only(self):
        df = _frame()
        m = _model()
        m.add_lagged_regressor('x', n_lags=6)
        m.fit(df, freq='1h', progress='none', minimal=True)
        out = m.predict(df, decompose=True)
        self.assertIn('yhat1', out.columns)
        lagged = [c for c in out.columns if c.startswith('lagged_regressor')]
        self.assertTrue(lagged, 'no lagged_regressor component column')


class TestConformalPredict(unittest.TestCase):
    """Spec D8: split conformal prediction with the cqr method."""

    def test_conformal_columns(self):
        df = _frame()
        train, cal, test = df.iloc[:600], df.iloc[600:800], df.iloc[800:]
        m = _model()
        m.add_future_regressor('x')
        m.fit(train, freq='1h', progress='none', minimal=True)
        out = m.conformal_predict(test, calibration_df=cal, alpha=0.1,
                                  method='cqr')
        self.assertIn('yhat1 - qhat1', out.columns)
        self.assertIn('yhat1 + qhat1', out.columns)


class TestParameterExtractors(unittest.TestCase):
    """Spec D14: redraws read public methods, not figures."""

    def test_predict_trend_and_seasonal_components(self):
        df = _frame()
        m = _model()
        m.fit(df, freq='1h', progress='none', minimal=True)
        trend = m.predict_trend(df)
        seasonal = m.predict_seasonal_components(df)
        self.assertEqual(len(trend), len(df))
        self.assertIn('daily', seasonal.columns)

    def test_matplotlib_backend_returns_figure(self):
        import matplotlib
        matplotlib.use('Agg')
        df = _frame()
        m = _model()
        m.fit(df, freq='1h', progress='none', minimal=True)
        fig = m.plot_parameters(plotting_backend='matplotlib')
        self.assertTrue(hasattr(fig, 'savefig'))


class TestConditionalSeasonalityWithFloatWeights(unittest.TestCase):
    """Spec D7: two daily series blended by float weights in 0..1."""

    def test_float_conditions_are_accepted_and_decomposed(self):
        df = _frame()
        m = _model(daily_seasonality=False)
        m.add_seasonality(name='daily_summer', period=1, fourier_order=3,
                          condition_name='summer_w')
        m.add_seasonality(name='daily_winter', period=1, fourier_order=3,
                          condition_name='winter_w')
        m.fit(df, freq='1h', progress='none', minimal=True)
        out = m.predict(df, decompose=True)
        self.assertIn('daily_summer', out.columns)
        self.assertIn('daily_winter', out.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
