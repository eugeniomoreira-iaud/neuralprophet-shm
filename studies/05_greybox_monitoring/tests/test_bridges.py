"""
Tests for the outage bridge (spec D12).

Run from studies/:  python 05_greybox_monitoring/tests/test_bridges.py
"""
import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import figures, prediction  # noqa: E402

logging.getLogger('NP').setLevel(logging.ERROR)


def _frame(n=24 * 90, seed=0):
    """A tiny hourly frame with a target and one regressor, for fitting a
    real (if minimal) NeuralProphet model through outage_bridge's default
    path."""
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    hours = np.arange(n)
    tair = 10.0 + 6.0 * np.sin(2 * np.pi * hours / 24.0) + rng.normal(0, 0.2, n)
    y = 50.0 - 2.0 * tair + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': y, 'tair': tair}, index=index)


class TestOutageBridge(unittest.TestCase):

    def test_a_level_shift_across_an_outage_is_reported_outside_the_band(self):
        index = pd.date_range('2024-01-01', periods=24 * 90, freq='1h', tz='UTC')
        y = pd.Series(np.zeros(len(index)), index=index)
        y.loc['2024-02-20':] = 10.0                      # the wall moved during the gap
        y.loc['2024-02-01':'2024-02-19'] = np.nan        # the outage
        frame = pd.DataFrame({'y': y})

        def runner(train, test):
            return pd.DataFrame({'ds': test.index, 'y': test['y'].to_numpy(),
                                 'yhat': 0.0, 'q05': -1.0, 'q95': 1.0})

        table, paths = prediction.outage_bridge(
            runner, frame, [('2024-02-01', '2024-02-19')], settle_days=1, window_days=7)
        self.assertEqual(table.loc[0, 'verdict'], 'outside')
        self.assertAlmostEqual(table.loc[0, 'shift'], 10.0, places=6)
        self.assertIn('outage', paths.columns)

    def test_table_columns_are_exactly_the_brief_s(self):
        index = pd.date_range('2024-01-01', periods=24 * 40, freq='1h', tz='UTC')
        frame = pd.DataFrame({'y': np.zeros(len(index))}, index=index)

        def runner(train, test):
            return pd.DataFrame({'ds': test.index, 'y': test['y'].to_numpy(),
                                 'yhat': 0.0, 'q05': -1.0, 'q95': 1.0})

        table, _ = prediction.outage_bridge(
            runner, frame, [('2024-01-20', '2024-01-22')])
        self.assertEqual(
            list(table.columns),
            ['outage', 'start', 'end', 'expected', 'lower', 'upper',
             'observed', 'shift', 'n_observed', 'verdict'])

    def test_works_on_a_tz_naive_frame_index(self):
        # The study's own def_frames carry a tz-naive DatetimeIndex (no UTC
        # localisation), unlike this test file's other frames built with
        # tz='UTC'; outage_bridge must not force UTC onto dates it parses.
        index = pd.date_range('2024-01-01', periods=24 * 60, freq='1h')
        self.assertIsNone(index.tz)
        frame = pd.DataFrame({'y': np.zeros(len(index))}, index=index)

        def runner(train, test):
            return pd.DataFrame({'ds': test.index, 'y': test['y'].to_numpy(),
                                 'yhat': 0.0, 'q05': -1.0, 'q95': 1.0})

        table, paths = prediction.outage_bridge(
            runner, frame, [('2024-01-15', '2024-01-17')],
            settle_days=1, window_days=3)
        self.assertNotEqual(table.loc[0, 'verdict'], 'no data')
        self.assertIn('outage', paths.columns)

    def test_default_runner_fits_neuralprophet_backtest_itself(self):
        # runner=None: outage_bridge fits neuralprophet_backtest on its own,
        # nowcast task, changepoints from covered_changepoints on the
        # train side, n_changepoints read out of model_kwargs.
        frame = _frame()
        outages = [('2024-02-10', '2024-02-12')]
        table, paths = prediction.outage_bridge(
            None, frame, outages, settle_days=1, window_days=3,
            regressors=('tair',), epochs=2, freq='1h', quantiles=(0.05, 0.95),
            n_changepoints=2, seed=0)
        self.assertEqual(
            list(table.columns),
            ['outage', 'start', 'end', 'expected', 'lower', 'upper',
             'observed', 'shift', 'n_observed', 'verdict'])
        self.assertNotEqual(table.loc[0, 'verdict'], 'no data')
        self.assertIn('yhat', paths.columns)
        self.assertIn('q05', paths.columns)
        self.assertIn('q95', paths.columns)

    def test_min_train_guard_reports_no_data_without_fitting(self):
        # An outage too close to the start of the record, against
        # min_train, is 'no data' and the runner is never called.
        index = pd.date_range('2024-01-01', periods=24 * 30, freq='1h', tz='UTC')
        frame = pd.DataFrame({'y': np.zeros(len(index))}, index=index)

        def runner(train, test):
            raise AssertionError('runner must not be called under the min_train guard')

        table, paths = prediction.outage_bridge(
            runner, frame, [('2024-01-10', '2024-01-12')],
            min_train='60d')
        self.assertEqual(table.loc[0, 'verdict'], 'no data')
        self.assertTrue(paths.empty)

    def test_n_jobs_two_matches_n_jobs_one(self):
        frame = _frame(n=24 * 150)
        outages = [('2024-02-20', '2024-02-22'), ('2024-04-10', '2024-04-12')]
        kwargs = dict(regressors=('tair',), settle_days=1, window_days=2,
                      epochs=2, freq='1h', n_changepoints=2, seed=0)
        serial, _ = prediction.outage_bridge(None, frame, outages, n_jobs=1, **kwargs)
        parallel, _ = prediction.outage_bridge(None, frame, outages, n_jobs=2, **kwargs)
        pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-5)


class TestPlotOutageBridge(unittest.TestCase):

    def test_one_axes_per_outage_in_paths(self):
        index = pd.date_range('2024-01-01', periods=24 * 60, freq='1h', tz='UTC')
        frame = pd.DataFrame({'y': np.zeros(len(index))}, index=index)

        def runner(train, test):
            return pd.DataFrame({'ds': test.index, 'y': test['y'].to_numpy(),
                                 'yhat': 0.0, 'q05': -1.0, 'q95': 1.0})

        outages = [('2024-01-15', '2024-01-17'), ('2024-02-05', '2024-02-07')]
        table, paths = prediction.outage_bridge(
            runner, frame, outages, settle_days=1, window_days=3)
        fig = figures.plot_outage_bridge(paths, table)
        self.assertEqual(len(fig.axes), paths['outage'].nunique())


if __name__ == '__main__':
    unittest.main(verbosity=2)
