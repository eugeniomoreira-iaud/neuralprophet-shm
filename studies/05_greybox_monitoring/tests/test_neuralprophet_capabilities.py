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


def _frame(n=24 * 40, freq='1h', seed=0, extra=()):
    rng = np.random.default_rng(seed)
    ds = pd.date_range('2024-01-01', periods=n, freq=freq)
    hours = np.arange(n)
    x = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24.0)
    y = 100.0 - 2.5 * np.roll(x, 3) + rng.normal(0, 0.3, n)
    doy = ds.dayofyear.to_numpy()
    summer_w = 0.5 * (1 - np.cos(2 * np.pi * (doy - 15) / 365.0))
    columns = {'ds': ds, 'y': y, 'x': x,
               'summer_w': summer_w, 'winter_w': 1 - summer_w}
    return pd.DataFrame({'ds': ds, 'y': y,
                         **{name: columns[name] for name in extra}})


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
        df = _frame(extra=('x',))
        m = _model()
        m.add_lagged_regressor('x', n_lags=6)
        m.fit(df, freq='1h', progress='none', minimal=True)
        out = m.predict(df, decompose=True)
        self.assertIn('yhat1', out.columns)
        lagged = [c for c in out.columns if c.startswith('lagged_regressor')]
        self.assertTrue(lagged, 'no lagged_regressor component column')


class TestConformalPredict(unittest.TestCase):
    """Spec D8: split conformal prediction with the cqr method.

    For method='cqr', NeuralProphet 0.8.0 adjusts the existing quantile
    columns in place rather than adding new ones; the '± qhat1' column
    names it produces exist only for method='naive'.
    """

    def test_conformal_columns(self):
        df = _frame(extra=('x',))
        train, cal, test = df.iloc[:600], df.iloc[600:800], df.iloc[800:]
        m = _model()
        m.add_future_regressor('x')
        m.fit(train, freq='1h', progress='none', minimal=True)
        out = m.conformal_predict(test, calibration_df=cal, alpha=0.1,
                                  method='cqr')
        self.assertIn('yhat1 5.0%', out.columns)
        self.assertIn('yhat1 95.0%', out.columns)
        self.assertEqual(len(out), len(test))

    def test_conformal_plot_requires_show_all_pi(self):
        """conformal_plot raises ValueError unless conformal_predict was
        called with show_all_PI=True: NeuralProphet 0.8.0's conformal_plot
        looks for a '+'-joined interval-width column that 'cqr' only writes
        when every retained interval is kept, not only the narrowest one
        the method returns by default. Study 05's native diagnostic cell
        (Movement 3) passes show_all_PI=True for exactly this reason, so
        this test pins that the fitted-and-conformalised pair the study
        depends on actually plots once that flag is set.
        """
        df = _frame(extra=('x',))
        train, cal, test = df.iloc[:600], df.iloc[600:800], df.iloc[800:]
        m = _model(epochs=2)
        m.add_future_regressor('x')
        m.fit(train, freq='1h', progress='none', minimal=True)
        out = m.conformal_predict(test, calibration_df=cal, alpha=0.1,
                                  method='cqr', show_all_PI=True)
        fig = m.conformal_plot(out, plotting_backend='plotly')
        self.assertIsNotNone(fig)


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

    @unittest.expectedFailure
    def test_matplotlib_backend_returns_figure(self):
        """NeuralProphet 0.8.0's matplotlib plot_parameters always returns
        None, because plot_model_parameters_matplotlib.py assigns
        fig = fig.tight_layout(), and Figure.tight_layout() returns None.
        No study code calls the matplotlib backend: D14 redraws every
        figure through shmlib.figures from the extractor methods instead.
        """
        import matplotlib
        matplotlib.use('Agg')
        df = _frame()
        m = _model()
        m.fit(df, freq='1h', progress='none', minimal=True)
        fig = m.plot_parameters(plotting_backend='matplotlib')
        self.assertTrue(hasattr(fig, 'savefig'))


class TestConditionalSeasonalityWithFloatWeights(unittest.TestCase):
    """Spec D7: two daily series blended by float weights in 0..1.
    predict(decompose=True) prefixes every seasonal component with
    'season_'.
    """

    def test_float_conditions_are_accepted_and_decomposed(self):
        df = _frame(extra=('summer_w', 'winter_w'))
        m = _model(daily_seasonality=False)
        m.add_seasonality(name='daily_summer', period=1, fourier_order=3,
                          condition_name='summer_w')
        m.add_seasonality(name='daily_winter', period=1, fourier_order=3,
                          condition_name='winter_w')
        m.fit(df, freq='1h', progress='none', minimal=True)
        out = m.predict(df, decompose=True)
        self.assertIn('season_daily_summer', out.columns)
        self.assertIn('season_daily_winter', out.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
