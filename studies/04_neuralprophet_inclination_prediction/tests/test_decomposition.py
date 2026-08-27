"""
Unit tests for the decomposition, changepoint and metric additions to shmlib.

The NeuralProphet-backed tests are deliberately tiny: they check that the
wrapper hands the model what it promised and reshapes what comes back, not that
the model is accurate.
"""
import os
import sys
import unittest
import warnings

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

    def _components_with_regressor_split(self, n=500, include_parts=True):
        index = pd.date_range('2024-01-01', periods=n, freq='1h')
        tair = np.linspace(-1.0, 1.0, n)
        rh = np.full(n, 0.01)
        data = {
            'trend': np.linspace(0.0, 1.0, n),
            'season_daily': 10.0 * np.sin(np.arange(n) / 24.0 * 2 * np.pi),
            'future_regressors_additive': tair + rh,
        }
        if include_parts:
            data['future_regressor_tair'] = tair
            data['future_regressor_rh'] = rh
        data['residual'] = np.full(n, 0.5)
        data['y'] = np.zeros(n)
        data['yhat1'] = np.zeros(n)
        return pd.DataFrame(data, index=index)

    def test_an_aggregate_is_dropped_when_its_constituents_are_present(self):
        table = prediction.component_variance_shares(
            self._components_with_regressor_split(include_parts=True))
        components = list(table['component'])
        self.assertNotIn('future_regressors_additive', components)
        self.assertIn('future_regressor_tair', components)
        self.assertIn('future_regressor_rh', components)
        self.assertAlmostEqual(table['share'].sum(), 1.0, places=6)

    def test_an_aggregate_is_kept_when_no_constituent_is_present(self):
        table = prediction.component_variance_shares(
            self._components_with_regressor_split(include_parts=False))
        self.assertIn('future_regressors_additive', list(table['component']))
        self.assertAlmostEqual(table['share'].sum(), 1.0, places=6)


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

    def test_decomposition_stack_panels_match_the_components_the_table_keeps(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        n = 200
        index = pd.date_range('2024-01-01', periods=n, freq='20min')
        tair = np.linspace(-1.0, 1.0, n)
        rh = np.full(n, 0.01)
        components = pd.DataFrame({
            'trend': np.linspace(0.0, 1.0, n),
            'season_daily': np.sin(np.arange(n) / 72.0 * 2 * np.pi),
            'future_regressors_additive': tair + rh,
            'future_regressor_tair': tair,
            'future_regressor_rh': rh,
            'residual': np.full(n, 0.5),
            'y': np.zeros(n),
            'yhat1': np.zeros(n),
        }, index=index)

        table = prediction.component_variance_shares(components)
        kept = list(table['component'])
        # The aggregate must not survive into the table's kept components:
        # counting it beside its own parts is exactly the bug this guards.
        self.assertNotIn('future_regressors_additive', kept)

        fig = figures.plot_decomposition_stack(components, title='t')
        self.assertEqual(len(fig.axes), len(kept))
        expected_labels = {c.replace('future_regressor_', '')
                           .replace('lagged_regressor_', '')
                           .replace('_', ' ')
                           for c in kept}
        actual_labels = {ax.get_ylabel() for ax in fig.axes}
        self.assertEqual(actual_labels, expected_labels)
        plt.close(fig)

    def _sparse_components(self):
        # Segmentation drops a missing timestamp outright rather than
        # carrying it as a NaN row, so the interior stretch below is deleted
        # from the index, not filled with NaN.
        full_index = pd.date_range('2024-01-01', periods=100, freq='20min')
        sparse_index = full_index.delete(np.arange(30, 60))
        n = len(sparse_index)
        components = pd.DataFrame({
            'trend': np.linspace(0.0, 1.0, n),
            'residual': np.zeros(n),
            'y': np.zeros(n),
            'yhat1': np.zeros(n),
        }, index=sparse_index)
        return full_index, components

    def test_decomposition_stack_breaks_the_line_across_a_dropped_stretch(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        full_index, components = self._sparse_components()
        fig = figures.plot_decomposition_stack(
            components, freq='20min', title='t')
        ydata = fig.axes[0].get_lines()[0].get_ydata()
        self.assertEqual(len(ydata), len(full_index))
        self.assertTrue(np.any(~np.isfinite(ydata)))
        plt.close(fig)

    def test_decomposition_stack_freq_none_plots_the_sparse_data_as_given(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        _, components = self._sparse_components()
        fig = figures.plot_decomposition_stack(components, title='t')
        ydata = fig.axes[0].get_lines()[0].get_ydata()
        self.assertEqual(len(ydata), len(components))
        self.assertTrue(np.all(np.isfinite(ydata)))
        plt.close(fig)

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

    def test_saving_and_colour_and_legend_conventions_hold(self):
        import tempfile
        from pathlib import Path

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        from shmlib import figures, monitoring, viz

        index = pd.date_range('2024-01-01', periods=200, freq='20min')
        components = pd.DataFrame({
            'trend': np.linspace(0.0, 1.0, 200),
            'season_daily': np.sin(np.arange(200) / 72.0 * 2 * np.pi),
            'residual': np.zeros(200),
            'y': np.zeros(200),
            'yhat1': np.zeros(200),
        }, index=index)
        observed = pd.Series(np.sin(np.arange(200) / 20.0), index=index)
        expected = observed * 0.9
        chart = monitoring.ewma_chart(observed - expected, 0.0, 1.0)
        metrics = pd.DataFrame({
            'horizon_h': [1, 6, 1, 6],
            'model': ['ar', 'ar', 'ar+tair', 'ar+tair'],
            'mae': [1.0, 2.0, 0.9, 1.8],
        })
        curve = pd.DataFrame({
            'magnitude': [1.0, 2.0, 1.0, 2.0],
            'duration_h': [6.0, 6.0, 24.0, 24.0],
            'detected': [False, True, True, True],
            'delay_h': [np.nan, 2.0, 1.0, 0.5],
        })

        cases = (
            lambda p, f: figures.plot_decomposition_stack(
                components, title='t', save_path=p, filename=f),
            lambda p, f: figures.plot_prediction_band(
                observed, expected, expected - 1.0, expected + 1.0,
                title='t', save_path=p, filename=f),
            lambda p, f: figures.plot_control_chart(
                chart, title='t', save_path=p, filename=f),
            lambda p, f: figures.plot_metric_vs_horizon(
                metrics, metric='mae', by='model', title='t',
                save_path=p, filename=f),
            lambda p, f: figures.plot_detectability(
                curve, title='t', save_path=p, filename=f),
        )

        # 1. Nothing is written to disk when only one of save_path/filename
        # is given: a figure that dropped the guard on either argument would
        # leave a stray file behind in an otherwise empty directory.
        with tempfile.TemporaryDirectory() as tmp:
            for call in cases:
                call(tmp, None)
                call(None, 'partial')
                self.assertEqual(list(Path(tmp).iterdir()), [],
                                 'a figure wrote a file with only one of '
                                 'save_path/filename given')
            plt.close('all')

        # 2. Every data line carries its intended colour, compared through
        # matplotlib's own colour parsing rather than a string, so the
        # assertion does not turn on hex case or an alpha suffix.
        inc_rgba = mcolors.to_rgba(viz.INC_COLOUR)

        stack_fig = figures.plot_decomposition_stack(components, title='t')
        for ax in stack_fig.axes:
            lines = ax.get_lines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(mcolors.to_rgba(lines[0].get_color()), inc_rgba)

        band_fig = figures.plot_prediction_band(
            observed, expected, expected - 1.0, expected + 1.0, title='t')
        band_ax = band_fig.axes[0]
        observed_line, expected_line = band_ax.get_lines()
        self.assertEqual(mcolors.to_rgba(observed_line.get_color()), inc_rgba)
        self.assertEqual(mcolors.to_rgba(expected_line.get_color()), inc_rgba)
        self.assertEqual(expected_line.get_linestyle(), '--')
        self.assertNotEqual(observed_line.get_linestyle(), '--')

        # 3. The other two legend-bearing figures also clear their axes,
        # using the same display-coordinate comparison as the previous test.
        control_fig = figures.plot_control_chart(chart, title='t')
        for fig in (band_fig, control_fig):
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend is not None:
                    self.assertLessEqual(
                        legend.get_window_extent(renderer).y1,
                        ax.get_window_extent().y0)

        plt.close('all')


class TestConformalInterval(unittest.TestCase):

    def test_coverage_on_held_out_rows_is_close_to_nominal(self):
        n = 4000
        rng = np.random.default_rng(1)
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        yhat = np.zeros(n)
        residual = rng.normal(scale=1.0, size=n)
        frame = pd.DataFrame({'ds': idx, 'y': yhat + residual, 'yhat': yhat})
        cutoff = idx[n // 2 - 1]

        out = prediction.conformal_interval(
            frame, alpha=0.10, calibration_end=cutoff)

        held_out = out[out['ds'] > cutoff]
        covered = ((held_out['y'] >= held_out['q05'])
                   & (held_out['y'] <= held_out['q95']))
        self.assertAlmostEqual(covered.mean(), 0.90, delta=0.02)

    def test_interval_is_asymmetric_for_skewed_residuals(self):
        n = 4000
        rng = np.random.default_rng(2)
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        yhat = np.zeros(n)
        # Gamma residuals are always non-negative and heavily right-skewed:
        # the upper tail must be pushed out much further than the lower one.
        residual = rng.gamma(shape=2.0, scale=1.0, size=n)
        frame = pd.DataFrame({'ds': idx, 'y': yhat + residual, 'yhat': yhat})

        out = prediction.conformal_interval(frame, alpha=0.10)

        lower_offset = out['q05'].iloc[0] - out['yhat'].iloc[0]
        upper_offset = out['q95'].iloc[0] - out['yhat'].iloc[0]
        self.assertGreater(abs(upper_offset) - abs(lower_offset), 1.0)

    def test_calibration_end_uses_only_the_calibration_periods_scale(self):
        n = 2000
        rng = np.random.default_rng(3)
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        calib_residual = rng.normal(scale=1.0, size=n // 2)
        post_residual = rng.normal(scale=100.0, size=n - n // 2)
        residual = np.concatenate([calib_residual, post_residual])
        yhat = np.zeros(n)
        frame = pd.DataFrame({'ds': idx, 'y': yhat + residual, 'yhat': yhat})
        cutoff = idx[n // 2 - 1]

        out = prediction.conformal_interval(
            frame, alpha=0.10, calibration_end=cutoff)
        width = out['q95'].iloc[0] - out['q05'].iloc[0]

        calib_only = frame[frame['ds'] <= cutoff]
        calib_res = calib_only['y'] - calib_only['yhat']
        expected_width = (calib_res.quantile(0.95)
                          - calib_res.quantile(0.05))

        self.assertAlmostEqual(width, expected_width, places=6)
        # A width set by the wild post-cutoff scale would be roughly two
        # orders of magnitude wider than this.
        self.assertLess(width, 20.0)

    def test_input_is_not_mutated_and_interval_column_is_set(self):
        idx = pd.date_range('2025-01-01', periods=10, freq='1h')
        frame = pd.DataFrame({'ds': idx, 'y': np.arange(10.0),
                              'yhat': np.zeros(10)})
        original = frame.copy()

        out = prediction.conformal_interval(frame)

        pd.testing.assert_frame_equal(frame, original)
        self.assertTrue((out['interval'] == 'conformal').all())
        self.assertIn('q05', out.columns)
        self.assertIn('q95', out.columns)

    def test_calibration_window_with_no_rows_raises(self):
        idx = pd.date_range('2025-01-01', periods=10, freq='1h')
        frame = pd.DataFrame({'ds': idx, 'y': np.arange(10.0),
                              'yhat': np.zeros(10)})
        with self.assertRaises(ValueError):
            prediction.conformal_interval(
                frame, calibration_end=idx[0] - pd.Timedelta(hours=1))

    def test_all_missing_calibration_residuals_raises(self):
        idx = pd.date_range('2025-01-01', periods=10, freq='1h')
        frame = pd.DataFrame({'ds': idx, 'y': [np.nan] * 10,
                              'yhat': np.zeros(10)})
        with self.assertRaises(ValueError):
            prediction.conformal_interval(frame)


class TestRollingNowcast(unittest.TestCase):

    def test_task_in_model_kwargs_raises_rather_than_colliding(self):
        idx = pd.date_range('2025-01-01', periods=10, freq='1h')
        frame = pd.DataFrame({'y': np.arange(10.0)}, index=idx)
        with self.assertRaises(ValueError):
            prediction.rolling_nowcast(
                frame, min_train='1h', refit_every='1h', task='forecast')

    @classmethod
    def setUpClass(cls):
        n = 200
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        steps = np.arange(n, dtype=float)
        rng = np.random.default_rng(0)
        cls.frame = pd.DataFrame({
            'y': np.sin(steps / 12.0) + 0.01 * rng.normal(size=n),
            'x': np.cos(steps / 10.0),
        }, index=idx)
        cls.min_train = '72h'
        cls.refit_every = '24h'
        cls.out = prediction.rolling_nowcast(
            cls.frame, regressors=('x',), refit_every=cls.refit_every,
            min_train=cls.min_train, epochs=1, n_lags=0, quantiles=())

    def test_every_row_has_an_origin_at_or_before_its_own_timestamp_and_no_duplicates(self):
        self.assertFalse(self.out.empty)
        self.assertTrue((self.out['origin'] <= self.out['ds']).all())
        self.assertFalse(self.out['ds'].duplicated().any())

    def test_window_count_matches_the_schedule_and_respects_min_train(self):
        span = self.frame.index.max() - self.frame.index.min()
        min_train_delta = pd.Timedelta(self.min_train)
        step = pd.Timedelta(self.refit_every)
        expected_windows = int((span - min_train_delta) // step) + 1

        self.assertEqual(self.out['origin'].nunique(), expected_windows)
        self.assertTrue(
            (self.out['ds'] >= self.frame.index.min() + min_train_delta).all())

    def test_rolling_bias_is_materially_smaller_than_a_frozen_fit_on_a_drift_the_model_cannot_represent(self):
        n = 300
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        steps = np.arange(n, dtype=float)
        rng = np.random.default_rng(1)
        # A linear drift that a growth='off' model has no term to express:
        # a frozen fit's bias must grow with distance from its own origin,
        # which is exactly the failure a walk-forward refit schedule caps.
        drift = 0.05 * steps
        frame = pd.DataFrame({
            'y': drift + np.sin(steps / 12.0) + 0.01 * rng.normal(size=n),
            'x': np.cos(steps / 10.0),
        }, index=idx)

        # A small first slice, deliberately: the frozen fit's mean stays
        # pinned near that slice's own low values while the drift keeps
        # climbing underneath it, and rolling_nowcast's periodic refits keep
        # each window's fit anchored closer to wherever the drift currently
        # is.
        split = int(n * 0.1)
        train_first = frame.iloc[:split]
        eval_region = frame.iloc[split:]

        _, frozen = prediction.neuralprophet_backtest(
            train_first, eval_region, regressors=('x',), task='nowcast',
            epochs=1, n_lags=0, quantiles=(), growth='off')

        min_train_hours = int((idx[split] - idx[0]) / pd.Timedelta(hours=1))
        rolling = prediction.rolling_nowcast(
            frame, regressors=('x',), refit_every='24h',
            min_train=f'{min_train_hours}h', epochs=1, n_lags=0,
            quantiles=(), growth='off')

        common_ds = set(rolling['ds']) & set(frozen['ds'])
        self.assertGreater(
            len(common_ds), 50,
            'not enough overlap between the frozen and rolling evaluation '
            'windows to compare bias fairly')

        frozen_common = frozen[frozen['ds'].isin(common_ds)]
        rolling_common = rolling[rolling['ds'].isin(common_ds)]

        frozen_bias = abs((frozen_common['yhat'] - frozen_common['y']).mean())
        rolling_bias = abs(
            (rolling_common['yhat'] - rolling_common['y']).mean())

        # Measured on this fixture: frozen bias ~7.17, rolling bias ~5.59
        # (about 22% smaller) — deterministic given the fixed seeds, so the
        # 0.9 threshold below leaves comfortable margin without being tight
        # enough to flake on minor numerical differences across machines.
        self.assertLess(rolling_bias, 0.9 * frozen_bias)


class TestPeriodScan(unittest.TestCase):

    def _dates(self, n, freq='1D'):
        return pd.date_range('2020-01-01', periods=n, freq=freq)

    def test_a_clean_annual_sine_is_recovered_as_the_top_peak(self):
        idx = self._dates(3 * 365)
        t = np.arange(len(idx), dtype=float)
        series = pd.Series(np.sin(2 * np.pi * t / 365.0), index=idx)

        table = prediction.period_scan(series)

        self.assertAlmostEqual(table['period_days'].iloc[0], 365.0,
                               delta=365.0 * 0.02)

    def test_single_sine_over_three_years_returns_one_near_annual_entry(self):
        # Periodogram resolution scales with period^2/span. On a 3-year
        # record an annual peak's sidelobes spread roughly +-120 days, so a
        # fixed 5% neighbour-suppression radius (+-18 days here) leaves
        # several sidelobes standing as separate rows a reader could
        # mistake for distinct cycles — exactly the misreading
        # resolution-based suppression exists to prevent.
        idx = self._dates(3 * 365)
        t = np.arange(len(idx), dtype=float)
        series = pd.Series(np.sin(2 * np.pi * t / 365.0), index=idx)

        table = prediction.period_scan(series)

        top_period = table['period_days'].iloc[0]
        top_resolution = table['resolution_days'].iloc[0]
        within_resolution = ((table['period_days'] - top_period).abs()
                             < top_resolution)
        self.assertEqual(int(within_resolution.sum()), 1)

    def test_resolution_days_matches_period_squared_over_span(self):
        idx = self._dates(3 * 365)
        t = np.arange(len(idx), dtype=float)
        series = pd.Series(np.sin(2 * np.pi * t / 365.0), index=idx)

        table = prediction.period_scan(series)

        span_days = float((idx[-1] - idx[0]) / pd.Timedelta(days=1))
        expected = table['period_days'] ** 2 / span_days
        np.testing.assert_allclose(table['resolution_days'].to_numpy(),
                                   expected.to_numpy(), rtol=1e-6)

    def test_the_same_sine_survives_30_percent_of_samples_missing(self):
        # This is the property that justifies Lomb-Scargle over an FFT: an
        # irregular, gappy sample must not need imputation to be scanned.
        idx = self._dates(3 * 365)
        t = np.arange(len(idx), dtype=float)
        series = pd.Series(np.sin(2 * np.pi * t / 365.0), index=idx)
        rng = np.random.default_rng(0)
        drop = rng.choice(len(series), size=int(0.3 * len(series)),
                          replace=False)
        series.iloc[drop] = np.nan

        table = prediction.period_scan(series)

        self.assertAlmostEqual(table['period_days'].iloc[0], 365.0,
                               delta=365.0 * 0.02)

    def test_two_superposed_periods_both_appear_and_the_stronger_ranks_first(self):
        # A Lomb-Scargle peak's width in period-space scales with p^2/span:
        # over only three years the two components' sidelobes overlap and
        # the second period never surfaces. Fifteen years narrows both
        # peaks enough to separate cleanly, which is what this test needs
        # to check, not an artefact of a too-short fixture.
        idx = self._dates(15 * 365)
        t = np.arange(len(idx), dtype=float)
        series = pd.Series(
            2.0 * np.sin(2 * np.pi * t / 365.0)
            + 1.0 * np.sin(2 * np.pi * t / 120.0),
            index=idx)

        table = prediction.period_scan(series, top=5)

        top_three = table['period_days'].iloc[:3].to_numpy()
        self.assertTrue(np.any(np.abs(top_three - 365.0) < 365.0 * 0.02))
        self.assertTrue(np.any(np.abs(top_three - 120.0) < 120.0 * 0.02))
        self.assertAlmostEqual(table['period_days'].iloc[0], 365.0,
                               delta=365.0 * 0.02)

    def test_fewer_than_50_samples_raises(self):
        idx = self._dates(10)
        series = pd.Series(np.arange(10.0), index=idx)
        with self.assertRaises(ValueError):
            prediction.period_scan(series)


class TestSingletonSegmentDroppedBeforePrediction(unittest.TestCase):
    """A one-row segment carries no frequency; slicing by time makes one eventually."""

    @classmethod
    def setUpClass(cls):
        # A model's first fit/predict call in a fresh process disturbs the
        # warnings machinery once (observed: pytorch-lightning's first
        # Trainer setup replaces `warnings.filters` wholesale, which can
        # swallow a `catch_warnings(record=True)` capture started before
        # it). One throwaway fit here absorbs that one-time cost so the
        # warning assertions below test this module's behaviour, not an
        # unrelated cold-start artefact of the training library.
        idx = pd.date_range('2025-01-01 00:00', periods=10, freq='20min')
        warm = pd.DataFrame({'y': np.arange(10.0)}, index=idx)
        prediction.neuralprophet_backtest(
            warm, warm, task='nowcast', epochs=1, n_lags=0, quantiles=(),
            freq='20min')

    def _train(self):
        idx = pd.date_range('2025-01-01 00:00', periods=30, freq='20min')
        steps = np.arange(len(idx), dtype=float)
        return pd.DataFrame({
            'y': np.sin(steps / 5.0),
            'x': np.cos(steps / 7.0),
            'segment_id': ['TR'] * len(idx),
        }, index=idx)

    def _test_with_fragment(self):
        idx_s1 = pd.date_range('2025-01-01 10:00', periods=5, freq='20min')
        idx_s2 = pd.date_range('2025-01-01 12:00', periods=5, freq='20min')
        idx_s3 = pd.date_range('2025-01-01 14:00', periods=1, freq='20min')
        idx = idx_s1.append(idx_s2).append(idx_s3)
        steps = np.arange(len(idx), dtype=float)
        segment_id = ['S1'] * 5 + ['S2'] * 5 + ['S3'] * 1
        return pd.DataFrame({
            'y': np.sin(steps / 5.0),
            'x': np.cos(steps / 7.0),
            'segment_id': segment_id,
        }, index=idx)

    def test_a_one_row_segment_is_dropped_not_crashed(self):
        # Record every warning rather than assertWarns/assertWarnsRegex:
        # NeuralProphet's own import and fit/predict path already raises
        # unrelated UserWarnings (pkg_resources deprecation, MPS
        # availability) ahead of ours, and both of those unittest helpers
        # only examine the first warning of the matching class, which would
        # make this test fail on noise rather than on the property it
        # checks.
        train = self._train()
        test = self._test_with_fragment()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, out = prediction.neuralprophet_backtest(
                train, test, regressors=('x',), task='nowcast', epochs=1,
                n_lags=0, quantiles=(), freq='20min')
        drop_warnings = [w for w in caught
                         if 'single-row segment' in str(w.message)]
        self.assertEqual(len(drop_warnings), 1)
        self.assertTrue(issubclass(drop_warnings[0].category, UserWarning))

        self.assertEqual(set(out['ID']), {'S1', 'S2'})
        self.assertEqual(len(out), 10)
        self.assertTrue(np.isfinite(out['yhat']).all())

    def test_dropping_the_fragment_is_a_no_op_on_the_healthy_segments(self):
        train = self._train()
        test_with_fragment = self._test_with_fragment()
        test_without_fragment = test_with_fragment[
            test_with_fragment['segment_id'] != 'S3']

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, with_fragment = prediction.neuralprophet_backtest(
                train, test_with_fragment, regressors=('x',), task='nowcast',
                epochs=1, n_lags=0, quantiles=(), freq='20min')
        self.assertEqual(
            len([w for w in caught
                if 'single-row segment' in str(w.message)]), 1)

        # Record every warning instead of escalating any to an exception:
        # NeuralProphet's own fit/predict already raises unrelated,
        # known-benign warnings this test has no business failing on. Only
        # the drop warning's absence is asserted.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, without_fragment = prediction.neuralprophet_backtest(
                train, test_without_fragment, regressors=('x',),
                task='nowcast', epochs=1, n_lags=0, quantiles=(),
                freq='20min')
        drop_warnings = [w for w in caught
                         if 'single-row segment' in str(w.message)]
        self.assertEqual(drop_warnings, [])

        self.assertEqual(len(with_fragment), len(without_fragment))
        self.assertEqual(set(with_fragment['ds']), set(without_fragment['ds']))
        with_sorted = with_fragment.sort_values('ds').reset_index(drop=True)
        without_sorted = without_fragment.sort_values(
            'ds').reset_index(drop=True)
        np.testing.assert_allclose(with_sorted['yhat'].to_numpy(),
                                   without_sorted['yhat'].to_numpy())


class TestSingletonSegmentDroppedBeforeFitting(unittest.TestCase):
    """The training frame is sliced by the same walk-forward, so it grows
    one-row segments for the same reason the evaluation frame does."""

    @classmethod
    def setUpClass(cls):
        # A model's first fit/predict call in a fresh process disturbs the
        # warnings machinery once (observed: pytorch-lightning's first
        # Trainer setup replaces `warnings.filters` wholesale, which can
        # swallow a `catch_warnings(record=True)` capture started before
        # it). One throwaway fit here absorbs that one-time cost so the
        # warning assertions below test this module's behaviour, not an
        # unrelated cold-start artefact of the training library. It is
        # repeated in this class rather than shared, because unittest gives
        # no guarantee that the sibling class holding the prediction-side
        # tests runs first.
        idx = pd.date_range('2025-01-01 00:00', periods=10, freq='20min')
        warm = pd.DataFrame({'y': np.arange(10.0)}, index=idx)
        prediction.neuralprophet_backtest(
            warm, warm, task='nowcast', epochs=1, n_lags=0, quantiles=(),
            freq='20min')

    def _train_with_fragment(self):
        # Two healthy segments and a one-row stub, exactly the shape a
        # time-based slice leaves when it cuts a long segment at a window
        # boundary.
        idx_s1 = pd.date_range('2025-01-01 00:00', periods=15, freq='20min')
        idx_s2 = pd.date_range('2025-01-01 06:00', periods=15, freq='20min')
        idx_s3 = pd.date_range('2025-01-01 09:00', periods=1, freq='20min')
        idx = idx_s1.append(idx_s2).append(idx_s3)
        steps = np.arange(len(idx), dtype=float)
        return pd.DataFrame({
            'y': np.sin(steps / 5.0),
            'x': np.cos(steps / 7.0),
            'segment_id': ['TR1'] * 15 + ['TR2'] * 15 + ['TR3'] * 1,
        }, index=idx)

    def _evaluation(self):
        idx = pd.date_range('2025-01-01 12:00', periods=10, freq='20min')
        steps = np.arange(len(idx), dtype=float)
        return pd.DataFrame({
            'y': np.sin(steps / 5.0),
            'x': np.cos(steps / 7.0),
            'segment_id': ['EV'] * len(idx),
        }, index=idx)

    def test_a_one_row_training_segment_is_dropped_not_crashed(self):
        # Before the drop, `model.fit` raised `ValueError: Invalid frequency:
        # NaT` on this frame: NeuralProphet re-infers a sampling frequency per
        # segment on the training path too, and a single timestamp offers none.
        _, out = prediction.neuralprophet_backtest(
            self._train_with_fragment(), self._evaluation(),
            regressors=('x',), task='nowcast', epochs=1, n_lags=0,
            quantiles=(), freq='20min')

        self.assertEqual(len(out), 10)
        self.assertTrue(np.isfinite(out['yhat']).all())

    def test_the_warning_names_the_training_side(self):
        # Record every warning rather than assertWarns/assertWarnsRegex:
        # NeuralProphet's own import and fit/predict path already raises
        # unrelated UserWarnings ahead of ours, and both of those unittest
        # helpers only examine the first warning of the matching class.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prediction.neuralprophet_backtest(
                self._train_with_fragment(), self._evaluation(),
                regressors=('x',), task='nowcast', epochs=1, n_lags=0,
                quantiles=(), freq='20min')

        drop_warnings = [w for w in caught
                         if 'single-row segment' in str(w.message)]
        self.assertEqual(len(drop_warnings), 1)
        message = str(drop_warnings[0].message)
        self.assertTrue(issubclass(drop_warnings[0].category, UserWarning))
        self.assertIn('before fitting', message)
        self.assertIn('removed from training', message)
        self.assertIn('dropped 1 row(s) across 1 single-row segment(s)',
                      message)

    def test_a_training_frame_with_no_short_segments_is_untouched(self):
        # The drop must be a no-op on every frame that already fits: no
        # warning, and predictions identical to the run whose fragment the
        # library removed for itself.
        train_with_fragment = self._train_with_fragment()
        train_without_fragment = train_with_fragment[
            train_with_fragment['segment_id'] != 'TR3']
        evaluation = self._evaluation()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, without_fragment = prediction.neuralprophet_backtest(
                train_without_fragment, evaluation, regressors=('x',),
                task='nowcast', epochs=1, n_lags=0, quantiles=(),
                freq='20min')
        self.assertEqual([w for w in caught
                          if 'single-row segment' in str(w.message)], [])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, with_fragment = prediction.neuralprophet_backtest(
                train_with_fragment, evaluation, regressors=('x',),
                task='nowcast', epochs=1, n_lags=0, quantiles=(),
                freq='20min')
        self.assertEqual(
            len([w for w in caught
                 if 'single-row segment' in str(w.message)]), 1)

        self.assertEqual(list(with_fragment['ds']),
                         list(without_fragment['ds']))
        np.testing.assert_allclose(with_fragment['yhat'].to_numpy(),
                                   without_fragment['yhat'].to_numpy())


if __name__ == '__main__':
    unittest.main(verbosity=2)
