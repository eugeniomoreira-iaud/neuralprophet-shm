"""
Unit tests for the decomposition, changepoint and metric additions to shmlib.

The NeuralProphet-backed tests are deliberately tiny: they check that the
wrapper hands the model what it promised and reshapes what comes back, not that
the model is accurate.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction


class TestCoveredChangepoints(unittest.TestCase):

    def test_changepoints_avoid_an_uncovered_stretch(self):
        index = pd.date_range('2024-01-01', periods=1000, freq='1h')
        observed = pd.Series(True, index=index)
        observed.iloc[300:700] = False
        points = prediction.covered_changepoints(index, 5, observed)
        self.assertEqual(len(points), 5)
        hole = pd.Interval(index[300].value, index[699].value)
        for point in points:
            self.assertNotIn(point.value, hole)

    def test_without_a_mask_every_timestamp_counts_as_covered(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h')
        points = prediction.covered_changepoints(index, 4)
        self.assertEqual(len(points), 4)
        self.assertTrue(points.is_monotonic_increasing)

    def test_asking_for_more_changepoints_than_covered_samples_is_clipped(self):
        index = pd.date_range('2024-01-01', periods=10, freq='1h')
        observed = pd.Series(False, index=index)
        observed.iloc[:3] = True
        points = prediction.covered_changepoints(index, 8, observed)
        self.assertLessEqual(len(points), 3)


class TestDecomposeComponents(unittest.TestCase):

    def _frame(self, n=24 * 40):
        index = pd.date_range('2024-01-01', periods=n, freq='1h')
        driver = 10.0 * np.sin(np.arange(n) / 24.0 * 2 * np.pi)
        return pd.DataFrame({'y': -2.0 * driver + 0.001 * np.arange(n),
                             'tair': driver}, index=index)

    def test_returns_named_components_that_sum_towards_the_prediction(self):
        frame = self._frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=1,
            growth='linear', n_changepoints=2, quantiles=())
        components = prediction.decompose_components(
            model, frame, regressors=('tair',))
        self.assertIn('trend', components.columns)
        self.assertIn('season_daily', components.columns)
        self.assertIn('future_regressor_tair', components.columns)
        self.assertIn('residual', components.columns)
        parts = components[['trend', 'season_daily',
                            'future_regressor_tair']].sum(axis=1)
        np.testing.assert_allclose(parts.to_numpy(),
                                   components['yhat1'].to_numpy(),
                                   rtol=1e-3, atol=1e-3)

    def test_residual_is_observed_minus_prediction(self):
        frame = self._frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=1,
            quantiles=())
        components = prediction.decompose_components(
            model, frame, regressors=('tair',))
        expected = components['y'] - components['yhat1']
        np.testing.assert_allclose(components['residual'].dropna().to_numpy(),
                                   expected.dropna().to_numpy(), atol=1e-9)


class TestBacktestDefaultsAreUnchanged(unittest.TestCase):

    def test_growth_still_defaults_to_off(self):
        import inspect
        signature = inspect.signature(prediction.neuralprophet_backtest)
        self.assertEqual(signature.parameters['growth'].default, 'off')
        self.assertIs(signature.parameters['changepoints'].default, None)
        self.assertIs(signature.parameters['decompose'].default, False)
        self.assertIs(signature.parameters['freq'].default, None)


class TestComponentVarianceShares(unittest.TestCase):

    def _components(self, n=500):
        index = pd.date_range('2024-01-01', periods=n, freq='1h')
        return pd.DataFrame({
            'trend': np.linspace(0.0, 1.0, n),
            'season_daily': 10.0 * np.sin(np.arange(n) / 24.0 * 2 * np.pi),
            'future_regressor_tair': np.zeros(n),
            'residual': np.full(n, 0.5),
            'y': np.zeros(n),
            'yhat1': np.zeros(n),
        }, index=index)

    def test_shares_sum_to_one_and_rank_by_variance(self):
        table = prediction.component_variance_shares(self._components())
        self.assertAlmostEqual(table['share'].sum(), 1.0, places=6)
        self.assertEqual(table['component'].iloc[0], 'season_daily')

    def test_a_constant_component_has_zero_variance_and_a_reported_mean(self):
        table = prediction.component_variance_shares(
            self._components()).set_index('component')
        self.assertAlmostEqual(table.loc['residual', 'variance'], 0.0)
        self.assertAlmostEqual(table.loc['residual', 'mean'], 0.5)

    def test_y_and_yhat_are_never_treated_as_components(self):
        table = prediction.component_variance_shares(self._components())
        self.assertNotIn('y', list(table['component']))
        self.assertNotIn('yhat1', list(table['component']))


class TestResidualDiagnostics(unittest.TestCase):

    def test_white_noise_is_not_rejected(self):
        rng = np.random.default_rng(0)
        index = pd.date_range('2024-01-01', periods=2000, freq='1h')
        residuals = pd.Series(rng.normal(size=2000), index=index)
        table = prediction.residual_diagnostics(residuals, lags=(1, 24))
        self.assertTrue((table['lb_pvalue'] > 0.01).all())
        self.assertEqual(list(table['lag']), [1, 24])

    def test_a_periodic_residual_is_rejected(self):
        index = pd.date_range('2024-01-01', periods=2000, freq='1h')
        residuals = pd.Series(np.sin(np.arange(2000) / 24.0 * 2 * np.pi),
                              index=index)
        table = prediction.residual_diagnostics(residuals, lags=(24,))
        self.assertLess(table['lb_pvalue'].iloc[0], 0.01)

    def test_scale_columns_describe_the_residual(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h')
        residuals = pd.Series(np.arange(100.0), index=index)
        table = prediction.residual_diagnostics(residuals, lags=(1,))
        self.assertEqual(table['n'].iloc[0], 100)
        self.assertGreater(table['std'].iloc[0], 0.0)
        self.assertGreater(table['mad'].iloc[0], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
