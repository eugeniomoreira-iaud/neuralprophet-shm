"""
Unit tests for the gap-anatomy and cadence functions this study relies on.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction


def _series(values, start='2024-01-01', freq='20min'):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


class TestGapInventory(unittest.TestCase):

    def test_each_run_of_missing_slots_becomes_one_row(self):
        values = [1.0, np.nan, np.nan, 2.0, 3.0, np.nan, 4.0]
        table = prediction.gap_inventory(_series(values), freq='20min')
        self.assertEqual(len(table), 2)
        self.assertEqual(list(table['n_slots']), [2, 1])
        self.assertAlmostEqual(table['duration_h'].iloc[0], 2 / 3)
        self.assertEqual(table['start'].iloc[0],
                         pd.Timestamp('2024-01-01 00:20:00'))
        self.assertEqual(table['end'].iloc[0],
                         pd.Timestamp('2024-01-01 00:40:00'))

    def test_a_series_with_no_gaps_returns_an_empty_typed_table(self):
        table = prediction.gap_inventory(_series([1.0, 2.0, 3.0]), freq='20min')
        self.assertTrue(table.empty)
        self.assertEqual(list(table.columns),
                         ['start', 'end', 'duration_h', 'n_slots', 'gap_class'])

    def test_gaps_are_classified_by_duration(self):
        values = [1.0] + [np.nan] * 3 + [1.0] + [np.nan] * 60 + [1.0]
        table = prediction.gap_inventory(_series(values), freq='20min')
        self.assertEqual(list(table['gap_class']), ['<=1h', '6-24h'])

    def test_an_irregular_index_is_reindexed_onto_the_grid_first(self):
        index = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00'])
        table = prediction.gap_inventory(
            pd.Series([1.0, 2.0], index=index), freq='20min')
        self.assertEqual(len(table), 1)
        self.assertEqual(table['n_slots'].iloc[0], 2)


class TestSegmentSurvival(unittest.TestCase):

    def _frame(self):
        # Two complete runs of 12 slots (4 h at 20 min) separated by one gap.
        index = pd.date_range('2024-01-01', periods=25, freq='20min')
        y = np.arange(25, dtype=float)
        y[12] = np.nan
        return pd.DataFrame({'y': y, 'x': np.ones(25)}, index=index)

    def test_counts_windows_that_fit_inside_a_segment(self):
        table = prediction.segment_survival(
            self._frame(), required=['y', 'x'],
            lag_hours=1, forecast_hours=1, freq='20min')
        row = table.iloc[0]
        self.assertEqual(row['n_segments'], 2)
        self.assertEqual(row['n_rows'], 24)
        # Each segment holds 12 slots; a window needs 3 + 3 = 6, so 7 fit.
        self.assertEqual(row['n_surviving'], 2)
        self.assertEqual(row['n_windows'], 14)

    def test_a_window_longer_than_every_segment_survives_nowhere(self):
        table = prediction.segment_survival(
            self._frame(), required=['y', 'x'],
            lag_hours=24, forecast_hours=24, freq='20min')
        self.assertEqual(table['n_surviving'].iloc[0], 0)
        self.assertEqual(table['n_windows'].iloc[0], 0)

    def test_several_configurations_return_several_rows(self):
        table = prediction.segment_survival(
            self._frame(), required=['y'],
            lag_hours=[1, 2], forecast_hours=[1], freq='20min')
        self.assertEqual(len(table), 2)
        self.assertEqual(list(table['lag_hours']), [1, 2])

    def test_a_required_column_that_is_never_present_yields_zero_rows(self):
        frame = self._frame()
        frame['z'] = np.nan
        table = prediction.segment_survival(
            frame, required=['y', 'z'], lag_hours=1, forecast_hours=1,
            freq='20min')
        self.assertEqual(table['n_rows'].iloc[0], 0)
        self.assertEqual(table['n_segments'].iloc[0], 0)


class TestCadenceEvidence(unittest.TestCase):

    def _pair(self, n=6 * 24 * 30):
        index = pd.date_range('2024-01-01', periods=n, freq='20min')
        phase = np.arange(n) / (3 * 24) * 2 * np.pi
        driver = pd.Series(10.0 * np.sin(phase), index=index)
        # Drift kept well below the coupled amplitude so it does not dilute
        # the level correlation this test asserts.
        response = pd.Series(-2.0 * driver.to_numpy() + 0.002 * np.arange(n),
                             index=index)
        return response, driver

    def test_reports_one_row_per_cadence_with_the_documented_columns(self):
        response, driver = self._pair()
        table = prediction.cadence_evidence(response, driver)
        self.assertEqual(list(table['cadence']), ['20min', '1h'])
        for column in ('level_autocorr1', 'change_autocorr1', 'change_std',
                       'change_mad', 'corr_level', 'corr_change',
                       'drift_per_year'):
            self.assertIn(column, table.columns)

    def test_recovers_the_sign_of_a_known_coupling(self):
        response, driver = self._pair()
        table = prediction.cadence_evidence(response, driver).set_index('cadence')
        self.assertLess(table.loc['1h', 'corr_level'], -0.9)
        self.assertLess(table.loc['1h', 'corr_change'], -0.9)

    def test_white_noise_on_the_level_shows_as_negative_change_autocorrelation(self):
        response, driver = self._pair()
        rng = np.random.default_rng(0)
        noisy = response + rng.normal(0.0, 20.0, len(response))
        table = prediction.cadence_evidence(noisy, driver).set_index('cadence')
        self.assertLess(table.loc['20min', 'change_autocorr1'], 0.0)

    def test_an_era_boundary_is_never_differenced_across(self):
        response, driver = self._pair(n=200)
        era = pd.Series('legacy', index=response.index)
        era.iloc[100:] = 'current'
        table = prediction.cadence_evidence(response, driver, era=era)
        without = prediction.cadence_evidence(response, driver)
        self.assertLess(table['n_change'].iloc[0], without['n_change'].iloc[0])

    def test_a_sparse_index_is_reindexed_before_the_level_autocorrelation(self):
        response, driver = self._pair(n=200)
        dropped = response.index[50:53]
        sparse_response = response.drop(dropped)
        nan_response = response.copy()
        nan_response.loc[dropped] = np.nan

        sparse_table = prediction.cadence_evidence(
            sparse_response, driver, cadences=('20min',))
        nan_table = prediction.cadence_evidence(
            nan_response, driver, cadences=('20min',))

        self.assertAlmostEqual(sparse_table['level_autocorr1'].iloc[0],
                               nan_table['level_autocorr1'].iloc[0], places=9)


class TestPhaseOneFigures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')

    def test_each_figure_builds_and_saves_both_formats(self):
        import tempfile
        from pathlib import Path

        import matplotlib.pyplot as plt

        from shmlib import figures

        index = pd.date_range('2024-01-01', periods=300, freq='20min')
        values = np.sin(np.arange(300) / 10.0)
        values[50:70] = np.nan
        series = pd.Series(values, index=index)
        frame = pd.DataFrame({'y': series, 'x': 1.0}, index=index)

        inventory = prediction.gap_inventory(series, freq='20min')
        survival = prediction.segment_survival(
            frame, ['y', 'x'], lag_hours=[1, 2], forecast_hours=[1],
            freq='20min')
        evidence = prediction.cadence_evidence(
            series.ffill(), pd.Series(np.arange(300.0), index=index))

        with tempfile.TemporaryDirectory() as tmp:
            for name, call in (
                    ('NP_F02_gap_anatomy',
                     lambda p, f: figures.plot_gap_anatomy(
                         inventory, title='t', save_path=p, filename=f)),
                    ('NP_F03_segment_survival',
                     lambda p, f: figures.plot_segment_survival(
                         survival, title='t', save_path=p, filename=f)),
                    ('NP_F04_cadence_evidence',
                     lambda p, f: figures.plot_cadence_evidence(
                         evidence, title='t', save_path=p, filename=f))):
                call(tmp, name)
                for ext in ('png', 'svg'):
                    self.assertTrue((Path(tmp) / f'{name}.{ext}').exists(),
                                    f'{name}.{ext} was not written')
            plt.close('all')

    def test_no_legend_is_drawn_inside_the_axes(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        index = pd.date_range('2024-01-01', periods=300, freq='20min')
        values = np.sin(np.arange(300) / 10.0)
        values[50:70] = np.nan
        frame = pd.DataFrame({'y': values, 'x': 1.0}, index=index)
        inventory = prediction.gap_inventory(
            pd.Series(values, index=index), freq='20min')
        survival = prediction.segment_survival(
            frame, ['y', 'x'], lag_hours=[1, 2], forecast_hours=[1],
            freq='20min')

        # plot_gap_anatomy draws no legend, so it alone cannot tell a correct
        # placement from a broken guard; plot_segment_survival does draw one,
        # which keeps this check live rather than vacuously passing.
        for fig in (figures.plot_gap_anatomy(inventory, title='t'),
                   figures.plot_segment_survival(survival, title='t')):
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend is not None:
                    # Compare both boxes in display coordinates: the legend's
                    # top must sit at or below the axes' bottom. Reading the
                    # anchor's y1 instead measures a pixel position, which is
                    # positive for any legend inside the canvas and so cannot
                    # discriminate.
                    self.assertLessEqual(
                        legend.get_window_extent(renderer).y1,
                        ax.get_window_extent().y0,
                        'A legend must sit below its axes.')
            plt.close(fig)

    def test_each_forecast_horizon_group_gets_a_distinct_line_style(self):
        import matplotlib.pyplot as plt

        from shmlib import figures

        survival = pd.DataFrame({
            'lag_hours': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'forecast_hours': [1, 1, 1, 8, 8, 8, 24, 24, 24],
            'n_windows': [10, 9, 8, 6, 5, 4, 3, 2, 1],
        })
        fig = figures.plot_segment_survival(survival, title='t')
        ax = fig.axes[0]
        drawn_styles = [line.get_linestyle() for line in ax.get_lines()]
        self.assertEqual(len(drawn_styles), 3)
        self.assertEqual(len(set(drawn_styles)), 3,
                         'Each forecast-horizon group must draw a distinct '
                         'line style.')
        plt.close(fig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
