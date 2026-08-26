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

    def test_a_lag_not_shorter_than_the_series_returns_nan_rather_than_raising(self):
        index = pd.date_range('2024-01-01', periods=50, freq='1h')
        residuals = pd.Series(np.arange(50.0), index=index)
        table = prediction.residual_diagnostics(residuals, lags=(72,))
        self.assertEqual(len(table), 1)
        self.assertEqual(table['lag'].iloc[0], 72)
        self.assertTrue(np.isnan(table['lb_stat'].iloc[0]))
        self.assertTrue(np.isnan(table['lb_pvalue'].iloc[0]))

    def test_a_mixed_request_keeps_the_testable_lag(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h')
        residuals = pd.Series(np.arange(100.0), index=index)
        table = prediction.residual_diagnostics(residuals, lags=(1, 500))
        self.assertEqual(list(table['lag']), [1, 500])
        self.assertFalse(np.isnan(table['lb_stat'].iloc[0]))
        self.assertTrue(np.isnan(table['lb_stat'].iloc[1]))


class TestScorePredictionsExtension(unittest.TestCase):

    def _frame(self):
        return pd.DataFrame({
            'y': [0.0, 1.0, 2.0, 3.0],
            'yhat': [0.5, 1.5, 1.5, 3.5],
            'q05': [-1.0, 0.0, 1.0, 2.0],
            'q95': [1.0, 2.0, 3.0, 4.0],
            'model': ['a', 'a', 'b', 'b'],
        })

    def test_existing_columns_keep_their_names_order_and_values(self):
        scores = prediction.score_predictions(self._frame(), ['model'])
        expected = ['model', 'n', 'mae', 'rmse', 'bias', 'r2',
                    'coverage_q05_q95', 'width_q05_q95']
        self.assertEqual(list(scores.columns)[:len(expected)], expected)
        self.assertAlmostEqual(scores['mae'].iloc[0], 0.5)

    def test_new_columns_are_appended_after_the_existing_ones(self):
        scores = prediction.score_predictions(self._frame(), ['model'])
        self.assertEqual(list(scores.columns)[-4:],
                         ['mase', 'pinball_q05', 'pinball_q95',
                          'interval_score'])

    def test_mase_is_missing_until_a_naive_scale_is_supplied(self):
        without = prediction.score_predictions(self._frame(), ['model'])
        self.assertTrue(without['mase'].isna().all())
        withscale = prediction.score_predictions(
            self._frame(), ['model'], naive_scale=0.5)
        self.assertAlmostEqual(withscale['mase'].iloc[0], 1.0)

    def test_pinball_loss_penalises_the_wrong_side_of_each_quantile(self):
        frame = pd.DataFrame({'y': [10.0], 'yhat': [0.0],
                              'q05': [0.0], 'q95': [1.0], 'model': ['a']})
        scores = prediction.score_predictions(frame, ['model'])
        # y above q95: the 0.95 quantile is penalised at weight 0.95.
        self.assertAlmostEqual(scores['pinball_q95'].iloc[0], 0.95 * 9.0)
        self.assertAlmostEqual(scores['pinball_q05'].iloc[0], 0.05 * 10.0)

    def test_interval_score_adds_a_violation_penalty_to_the_width(self):
        frame = pd.DataFrame({'y': [3.0], 'yhat': [0.0],
                              'q05': [0.0], 'q95': [1.0], 'model': ['a']})
        scores = prediction.score_predictions(frame, ['model'], alpha=0.10)
        self.assertAlmostEqual(scores['interval_score'].iloc[0],
                               1.0 + (2.0 / 0.10) * 2.0)


class TestModelFigures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')

    def test_every_figure_builds_and_writes_png_and_svg(self):
        import tempfile
        from pathlib import Path

        import matplotlib.pyplot as plt

        from shmlib import figures, monitoring

        index = pd.date_range('2024-01-01', periods=400, freq='20min')
        components = pd.DataFrame({
            'trend': np.linspace(0.0, 2.0, 400),
            'season_daily': np.sin(np.arange(400) / 72.0 * 2 * np.pi),
            'future_regressor_tair': np.cos(np.arange(400) / 72.0 * 2 * np.pi),
            'residual': np.zeros(400),
            'y': np.zeros(400),
            'yhat1': np.zeros(400),
        }, index=index)
        observed = pd.Series(np.sin(np.arange(400) / 20.0), index=index)
        expected = observed * 0.9
        chart = monitoring.ewma_chart(observed - expected, 0.0, 1.0)
        metrics = pd.DataFrame({
            'horizon_h': [1, 6, 24, 1, 6, 24],
            'model': ['ar'] * 3 + ['ar+tair'] * 3,
            'mae': [1.0, 2.0, 3.0, 0.9, 1.8, 2.9],
        })
        curve = pd.DataFrame({
            'magnitude': [1.0, 2.0, 1.0, 2.0],
            'duration_h': [6.0, 6.0, 24.0, 24.0],
            'detected': [False, True, True, True],
            'delay_h': [np.nan, 2.0, 1.0, 0.5],
        })

        with tempfile.TemporaryDirectory() as tmp:
            cases = (
                ('NP_F05_decomposition_stack',
                 lambda p, f: figures.plot_decomposition_stack(
                     components, title='t', save_path=p, filename=f)),
                ('NP_F07_observed_vs_expected',
                 lambda p, f: figures.plot_prediction_band(
                     observed, expected, expected - 1.0, expected + 1.0,
                     title='t', save_path=p, filename=f)),
                ('NP_F08_control_chart',
                 lambda p, f: figures.plot_control_chart(
                     chart, title='t', save_path=p, filename=f)),
                ('NP_F10_skill_vs_horizon',
                 lambda p, f: figures.plot_metric_vs_horizon(
                     metrics, metric='mae', by='model', title='t',
                     save_path=p, filename=f)),
                ('NP_F09_detectability',
                 lambda p, f: figures.plot_detectability(
                     curve, title='t', save_path=p, filename=f)),
            )
            for name, call in cases:
                call(tmp, name)
                for ext in ('png', 'svg'):
                    self.assertTrue((Path(tmp) / f'{name}.{ext}').exists(),
                                    f'{name}.{ext} was not written')
            plt.close('all')

    def test_legends_sit_below_their_axes(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        metrics = pd.DataFrame({
            'horizon_h': [1, 6, 1, 6],
            'model': ['ar', 'ar', 'ar+tair', 'ar+tair'],
            'mae': [1.0, 2.0, 0.9, 1.8],
        })
        fig = figures.plot_metric_vs_horizon(metrics, metric='mae', by='model')
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                # Compare both boxes in display coordinates: the legend's top
                # must sit at or below the axes' bottom. Reading the anchor's
                # y1 instead measures a pixel position, which is positive for
                # any legend inside the canvas and so cannot discriminate.
                self.assertLessEqual(legend.get_window_extent(renderer).y1,
                                     ax.get_window_extent().y0)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
