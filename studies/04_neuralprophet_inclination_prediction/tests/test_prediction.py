"""
Tests for Study 04 prediction helpers.

Run directly, with no test runner installed::

    python studies/04_neuralprophet_inclination_prediction/tests/test_prediction.py
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import prediction  # noqa: E402


class TestHourlyChange(unittest.TestCase):
    """First differences must not bridge records the model may not learn across."""

    def test_change_stops_at_gaps_changeover_and_invalid_endpoints(self):
        idx = pd.to_datetime([
            '2025-01-01 00:00',
            '2025-01-01 01:00',
            '2025-01-01 03:00',
            '2025-01-01 04:00',
            '2025-01-01 05:00',
            '2025-01-01 06:00',
        ])
        series = pd.Series([10.0, 12.0, 20.0, 23.0, 30.0, 31.0], index=idx)
        era = pd.Series(['legacy', 'legacy', 'legacy', 'current', 'current',
                         'current'], index=idx)
        invalid = pd.Series([False, False, False, False, True, False],
                            index=idx)

        out = prediction.hourly_change(series, era=era, invalid=invalid)

        expected = pd.Series([np.nan, 2.0, np.nan, np.nan, np.nan, np.nan],
                             index=idx)
        pd.testing.assert_series_equal(out, expected)


class TestContiguousSegments(unittest.TestCase):
    """Only complete rows on an unbroken grid can enter segmented folds."""

    def test_complete_rows_receive_deterministic_segment_ids(self):
        idx = pd.to_datetime([
            '2025-01-01 00:00',
            '2025-01-01 01:00',
            '2025-01-01 02:00',
            '2025-01-01 04:00',
            '2025-01-01 05:00',
            '2025-01-01 06:00',
        ])
        frame = pd.DataFrame({
            'y': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            'x': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            'ignored': [np.nan] * 6,
        }, index=idx)

        out = prediction.contiguous_segments(
            frame, required=['y', 'x'], min_length=2)

        self.assertEqual(list(out.index), [idx[0], idx[1], idx[3], idx[4],
                                           idx[5]])
        self.assertEqual(list(out['segment_id']),
                         ['S001', 'S001', 'S002', 'S002', 'S002'])
        self.assertIn('ignored', out.columns)

    def test_short_segments_are_removed(self):
        idx = pd.date_range('2025-01-01', periods=3, freq='1h')
        frame = pd.DataFrame({'y': [1.0, np.nan, 3.0], 'x': [1.0, 2.0, 3.0]},
                             index=idx)

        out = prediction.contiguous_segments(
            frame, required=['y', 'x'], min_length=2)

        self.assertTrue(out.empty)
        self.assertIn('segment_id', out.columns)


class TestExpandingSegmentFolds(unittest.TestCase):
    """Folds train on past segments and test on all later segments."""

    def test_remaining_segments_are_partitioned_into_nonempty_test_groups(self):
        segment_ids = pd.Series(['S001', 'S001', 'S002', 'S003', 'S003',
                                 'S004', 'S005', 'S006', 'S007'])

        folds = prediction.expanding_segment_folds(
            segment_ids, initial_segments=2, n_folds=3)

        self.assertEqual(folds, [
            {'fold': 1, 'train_segments': ['S001', 'S002'],
             'test_segments': ['S003', 'S004']},
            {'fold': 2, 'train_segments': ['S001', 'S002', 'S003', 'S004'],
             'test_segments': ['S005', 'S006']},
            {'fold': 3, 'train_segments': ['S001', 'S002', 'S003', 'S004',
                                           'S005', 'S006'],
             'test_segments': ['S007']},
        ])


class TestAvailabilityRoute(unittest.TestCase):
    """Routes must reflect what information exists for each prediction task."""

    def test_nowcast_prefers_rich_then_core(self):
        self.assertEqual(
            prediction.availability_route(True, True, task='nowcast'), 'rich')
        self.assertEqual(
            prediction.availability_route(True, False, task='nowcast'), 'core')
        self.assertEqual(
            prediction.availability_route(False, False, ar_ok=True,
                                          task='nowcast'), 'unavailable')

    def test_forecast_uses_rich_only_inside_one_day_then_falls_back(self):
        self.assertEqual(
            prediction.availability_route(True, True, task='forecast',
                                          horizon_h=24), 'rich')
        self.assertEqual(
            prediction.availability_route(True, True, task='forecast',
                                          horizon_h=25), 'core')
        self.assertEqual(
            prediction.availability_route(False, False, ar_ok=True,
                                          task='forecast', horizon_h=25),
            'ar_only')


class TestScorePredictions(unittest.TestCase):
    """Prediction scores use paired observed and predicted values only."""

    def test_grouped_scores_and_quantile_interval_metrics(self):
        frame = pd.DataFrame({
            'model': ['core', 'core', 'core', 'rich', 'rich', 'rich'],
            'y': [1.0, 2.0, 3.0, 1.0, 3.0, np.nan],
            'yhat': [1.5, 1.0, 3.0, 1.0, 5.0, 4.0],
            'q05': [0.0, 1.5, -47.0, 0.5, 2.0, 3.0],
            'q95': [2.0, 2.5, 53.0, 1.5, 2.5, 5.0],
        })

        out = prediction.score_predictions(frame, group_cols=['model'])
        core = out.set_index('model').loc['core']
        rich = out.set_index('model').loc['rich']

        self.assertEqual(core['n'], 3)
        self.assertAlmostEqual(core['mae'], 0.5)
        self.assertAlmostEqual(core['rmse'], np.sqrt(1.25 / 3.0))
        self.assertAlmostEqual(core['bias'], -1.0 / 6.0)
        self.assertAlmostEqual(core['r2'], 0.375)
        self.assertAlmostEqual(core['coverage_q05_q95'], 1.0)
        self.assertAlmostEqual(core['width_q05_q95'], 2.0)
        self.assertEqual(rich['n'], 2)
        self.assertAlmostEqual(rich['coverage_q05_q95'], 0.5)


class TestPairedMaeSkill(unittest.TestCase):
    """Bootstrap skill must keep parent and child errors timestamp-aligned."""

    def test_aligned_block_bootstrap_is_reproducible(self):
        idx = pd.date_range('2025-01-01', periods=4, freq='1h')
        parent = pd.Series([2.0, 4.0, 6.0, 8.0], index=idx)
        child = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        child = child.drop(idx[2])

        out = prediction.paired_mae_skill(
            parent, child, block_hours=2, repetitions=200, seed=42)

        self.assertEqual(out['n'], 3)
        self.assertAlmostEqual(out['parent_mae'], 14.0 / 3.0)
        self.assertAlmostEqual(out['child_mae'], 7.0 / 3.0)
        self.assertAlmostEqual(out['skill'], 0.5)
        self.assertLessEqual(out['skill_q05'], out['skill'])
        self.assertGreaterEqual(out['skill_q95'], out['skill'])

    def test_horizon_adaptive_block_hours(self):
        idx = pd.date_range('2025-01-01', periods=100, freq='1h')
        parent = pd.Series(np.ones(100) * 2.0, index=idx)
        child = pd.Series(np.ones(100) * 1.0, index=idx)

        # When horizon_hours=24, effective block size becomes max(24, 2*24)=48h
        out = prediction.paired_mae_skill(
            parent, child, block_hours=24, horizon_hours=24, repetitions=50, seed=0)
        self.assertEqual(out['n'], 100)
        self.assertAlmostEqual(out['skill'], 0.5)
        self.assertAlmostEqual(out['skill_q05'], 0.5)
        self.assertAlmostEqual(out['skill_q95'], 0.5)


class TestQuantileColumn(unittest.TestCase):
    """Robust quantile column matching across varied library string formats."""

    def test_quantile_column_variations(self):
        cols = ['yhat1', 'yhat1 5.0%', 'yhat1 95.0%', 'yhat2 5%', 'yhat2 95%',
                'yhat3 0.05', 'yhat3 0.95', 'yhat4 50%']
        self.assertEqual(prediction._quantile_column(cols, 1, 0.05), 'yhat1 5.0%')
        self.assertEqual(prediction._quantile_column(cols, 1, 0.95), 'yhat1 95.0%')
        self.assertEqual(prediction._quantile_column(cols, 2, 0.05), 'yhat2 5%')
        self.assertEqual(prediction._quantile_column(cols, 2, 0.95), 'yhat2 95%')
        self.assertEqual(prediction._quantile_column(cols, 3, 0.05), 'yhat3 0.05')
        self.assertEqual(prediction._quantile_column(cols, 3, 0.95), 'yhat3 0.95')
        self.assertEqual(prediction._quantile_column(cols, 4, 0.50), 'yhat4 50%')
        self.assertIsNone(prediction._quantile_column(cols, 1, 0.50))
        self.assertIsNone(prediction._quantile_column(cols, 5, 0.05))


class TestNeuralProphetBacktest(unittest.TestCase):
    """A tiny real fit protects the wrapper contract at the library boundary."""

    def test_nowcast_smoke_returns_aligned_finite_predictions(self):
        idx = pd.date_range('2025-01-01', periods=48, freq='1h')
        steps = np.arange(len(idx), dtype=float)
        data = pd.DataFrame({
            'y': 0.2 * steps + np.sin(steps / 6.0),
            'x': np.cos(steps / 8.0),
            'segment_id': ['A'] * len(idx),
        }, index=idx)

        _, out = prediction.neuralprophet_backtest(
            data.iloc[:36], data.iloc[36:], regressors=('x',),
            task='nowcast', epochs=1, quantiles=(0.05, 0.95), seed=7)

        self.assertEqual(set(['ds', 'ID', 'horizon_h', 'y', 'yhat',
                              'q05', 'q95']), set(out.columns))
        self.assertEqual(set(out['ds']), set(data.index[36:]))
        self.assertTrue((out['ID'] == 'A').all())
        self.assertTrue((out['horizon_h'] == 1).all())
        self.assertTrue(np.isfinite(out['yhat']).all())
        self.assertTrue(np.isfinite(out[['q05', 'q95']]).all().all())

    def test_segmented_forecast_uses_each_test_segments_own_history(self):
        idx = pd.date_range('2025-01-01', periods=64, freq='1h')
        steps = np.arange(len(idx), dtype=float)
        data = pd.DataFrame({
            'y': np.sin(steps / 6.0),
            'x': np.cos(steps / 8.0),
            'segment_id': ['train'] * 32 + ['test'] * 32,
        }, index=idx)

        _, out = prediction.neuralprophet_backtest(
            data.iloc[:32], data.iloc[32:], regressors=('x',),
            task='forecast', n_lags=4, n_forecasts=3,
            regressor_lags=4, horizons=(1, 3), epochs=1,
            quantiles=(), seed=7)

        self.assertFalse(out.empty)
        self.assertEqual(set(out['ID']), {'test'})
        self.assertEqual(set(out['horizon_h']), {1, 3})
        self.assertTrue(np.isfinite(out['yhat']).all())


class TestBacktestSpecifications(unittest.TestCase):
    """Notebook orchestration must keep fold boundaries explicit."""

    def test_fake_runner_gets_whole_past_segments_and_scores_each_test_once(self):
        idx = pd.date_range('2025-01-01', periods=10, freq='1h')
        segmented = pd.DataFrame({
            'y': np.arange(10, dtype=float),
            'x': np.arange(10, dtype=float) * 10.0,
            'segment_id': ['S001'] * 2 + ['S002'] * 2 + ['S003'] * 2
            + ['S004'] * 2 + ['S005'] * 2,
        }, index=idx)
        folds = prediction.expanding_segment_folds(
            segmented['segment_id'], initial_segments=2, n_folds=2)
        calls = []

        def fake_runner(train, test, **kwargs):
            calls.append({
                'train': list(pd.unique(train['segment_id'])),
                'test': list(pd.unique(test['segment_id'])),
                'kwargs': kwargs,
            })
            out = pd.DataFrame({
                'ds': test.index,
                'ID': test['segment_id'].to_numpy(),
                'horizon_h': 1,
                'y': test['y'].to_numpy(),
                'yhat': test['y'].to_numpy() + 0.5,
            })
            return object(), out

        out = prediction.backtest_specifications(
            segmented, folds,
            [{'name': 'rich', 'task': 'nowcast', 'regressors': ('x',),
              'epochs': 3}],
            epochs=9, seed=4, runner=fake_runner)

        self.assertEqual(calls[0]['train'], ['S001', 'S002'])
        self.assertEqual(calls[0]['test'], ['S003', 'S004'])
        self.assertEqual(calls[1]['train'], ['S001', 'S002', 'S003', 'S004'])
        self.assertEqual(calls[1]['test'], ['S005'])
        self.assertEqual(calls[0]['kwargs']['task'], 'nowcast')
        self.assertEqual(calls[0]['kwargs']['regressors'], ('x',))
        self.assertEqual(calls[0]['kwargs']['epochs'], 3)
        self.assertEqual(calls[0]['kwargs']['seed'], 4)
        self.assertEqual(set(out['ID']), {'S003', 'S004', 'S005'})
        self.assertEqual(out.groupby('ID')['ds'].nunique().to_dict(),
                         {'S003': 2, 'S004': 2, 'S005': 2})
        self.assertEqual(set(out['model']), {'rich'})
        self.assertEqual(set(out['fold']), {1, 2})


class TestBaselinePredictions(unittest.TestCase):
    """Baselines must never borrow values across segment boundaries."""

    def test_zero_persistence_and_seasonal_naive_on_test_rows_only(self):
        idx = pd.date_range('2025-01-01', periods=56, freq='1h')
        segmented = pd.DataFrame({
            'y': np.arange(56, dtype=float),
            'segment_id': ['S001'] * 28 + ['S002'] * 28,
        }, index=idx)
        folds = [{'fold': 1, 'train_segments': ['S001'],
                  'test_segments': ['S002']}]

        out = prediction.baseline_predictions(
            segmented, folds, horizons=(1, 25), task='forecast')

        self.assertEqual(set(out['model']),
                         {'zero', 'persistence', 'seasonal_naive'})
        self.assertEqual(set(out['ID']), {'S002'})
        self.assertEqual(set(out['ds']), set(segmented.index[28:]))

        first = segmented.index[28]
        row = out[(out['model'] == 'persistence') & (out['horizon_h'] == 1)
                  & (out['ds'] == first)].iloc[0]
        self.assertTrue(pd.isna(row['yhat']))

        row = out[(out['model'] == 'persistence') & (out['horizon_h'] == 1)
                  & (out['ds'] == segmented.index[29])].iloc[0]
        self.assertEqual(row['yhat'], 28.0)

        row = out[(out['model'] == 'seasonal_naive') & (out['horizon_h'] == 1)
                  & (out['ds'] == segmented.index[52])].iloc[0]
        self.assertEqual(row['yhat'], 28.0)

        horizon_25 = out[(out['model'] == 'seasonal_naive')
                         & (out['horizon_h'] == 25)]
        self.assertTrue(horizon_25['yhat'].isna().all())

        zero = out[out['model'] == 'zero']
        self.assertTrue((zero['yhat'] == 0.0).all())

    def test_nowcast_baseline_only_zero(self):
        idx = pd.date_range('2025-01-01', periods=4, freq='1h')
        segmented = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'segment_id': ['S001', 'S001', 'S002', 'S002'],
        }, index=idx)
        folds = [{'fold': 1, 'train_segments': ['S001'],
                  'test_segments': ['S002']}]

        out = prediction.baseline_predictions(
            segmented, folds, horizons=(1,), task='nowcast')

        self.assertEqual(set(out['model']), {'zero'})
        self.assertEqual(list(out['yhat']), [0.0, 0.0])


class TestNeuralProphetPredict(unittest.TestCase):
    """Predicting new rows must preserve ID and tolerate unknown y."""

    def test_fake_model_output_is_reshaped_without_requiring_y(self):
        idx = pd.date_range('2025-01-03', periods=2, freq='1h')
        frame = pd.DataFrame({
            'x': [5.0, 6.0],
            'segment_id': ['live', 'live'],
        }, index=idx)

        class FakeModel:
            def predict(self, model_frame, decompose=False):
                self.model_frame = model_frame
                return pd.DataFrame({
                    'ds': model_frame['ds'],
                    'y': [np.nan, np.nan],
                    'ID': model_frame['ID'],
                    'yhat1': [10.0, 11.0],
                    'yhat1 5.0%': [9.0, 10.0],
                    'yhat1 95.0%': [12.0, 13.0],
                })

        fake = FakeModel()
        out = prediction.neuralprophet_predict(
            fake, frame, regressors=('x',), horizons=(1,),
            quantiles=(0.05, 0.95))

        self.assertTrue(fake.model_frame['y'].isna().all())
        self.assertEqual(list(out['ID']), ['live', 'live'])
        self.assertEqual(list(out['yhat']), [10.0, 11.0])
        self.assertTrue(out['y'].isna().all())
        self.assertEqual(list(out['q05']), [9.0, 10.0])
        self.assertEqual(list(out['q95']), [12.0, 13.0])

    def test_zero_lag_model_gets_placeholder_y_but_output_keeps_missing_y(self):
        idx = pd.date_range('2025-01-03', periods=3, freq='1h')
        frame = pd.DataFrame({
            'y': [1.0, np.nan, np.nan],
            'x': [5.0, 6.0, 7.0],
            'segment_id': ['live', 'live', 'live'],
        }, index=idx)

        class FakeZeroLagModel:
            n_lags = 0

            def predict(self, model_frame, decompose=False):
                self.model_frame = model_frame.copy()
                return pd.DataFrame({
                    'ds': model_frame['ds'],
                    'y': model_frame['y'],
                    'ID': model_frame['ID'],
                    'yhat1': [10.0, 11.0, 12.0],
                })

        fake = FakeZeroLagModel()
        out = prediction.neuralprophet_predict(
            fake, frame, regressors=('x',), horizons=(1,), quantiles=())

        self.assertEqual(list(fake.model_frame['y']), [1.0, 0.0, 0.0])
        self.assertEqual(out['y'].isna().tolist(), [False, True, True])
        self.assertEqual(list(out['yhat']), [10.0, 11.0, 12.0])


class TestExecutionFolds(unittest.TestCase):
    """Notebook can choose frozen-origin or per-fold refits explicitly."""

    def test_frozen_origin_uses_first_train_and_all_test_segments(self):
        folds = [
            {'fold': 1, 'train_segments': ['S001', 'S002'],
             'test_segments': ['S003', 'S004']},
            {'fold': 2, 'train_segments': ['S001', 'S002', 'S003', 'S004'],
             'test_segments': ['S005']},
        ]

        frozen = prediction.execution_folds(folds, refit_each_fold=False)

        self.assertEqual(frozen, [{
            'fold': 1,
            'train_segments': ['S001', 'S002'],
            'test_segments': ['S003', 'S004', 'S005'],
        }])
        self.assertEqual(prediction.execution_folds(folds, refit_each_fold=True),
                         folds)


class TestGapClosureSummary(unittest.TestCase):
    """Gap closures compare observed recovery with summed predicted changes."""

    def test_bracketed_gap_closure_and_terminal_gap_status(self):
        idx = pd.date_range('2025-01-01', periods=9, freq='1h')
        inclination = pd.Series(
            [10.0, np.nan, np.nan, 16.0, 17.0, 18.0, np.nan, np.nan, np.nan],
            index=idx)
        estimates = pd.Series(
            [np.nan, 2.0, 2.0, 2.0, 1.0, 1.0, 4.0, 4.0, 4.0], index=idx)
        era = pd.Series(['A'] * len(idx), index=idx)

        out = prediction.gap_closure_summary(
            inclination, estimates, era=era)

        self.assertEqual(len(out), 2)
        first = out.iloc[0]
        self.assertEqual(first['gap_id'], 'G001')
        self.assertEqual(first['start'], idx[1])
        self.assertEqual(first['end'], idx[2])
        self.assertEqual(first['recovery'], idx[3])
        self.assertEqual(first['n_missing'], 2)
        self.assertEqual(first['status'], 'available')
        self.assertEqual(first['prior_anchor'], 10.0)
        self.assertEqual(first['observed_recovery'], 6.0)
        self.assertEqual(first['predicted_recovery'], 6.0)
        self.assertEqual(first['closure_error'], 0.0)

        second = out.iloc[1]
        self.assertEqual(second['gap_id'], 'G002')
        self.assertEqual(second['status'], 'unbracketed')
        self.assertTrue(pd.isna(second['recovery']))
        self.assertTrue(pd.isna(second['closure_error']))

    def test_cross_era_and_incomplete_estimate_statuses_are_explicit(self):
        idx = pd.date_range('2025-01-01', periods=8, freq='1h')
        inclination = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, np.nan, 10.0,
                                 11.0], index=idx)
        estimates = pd.Series([np.nan, 2.0, 2.0, np.nan, 1.0, np.nan, 1.0,
                               1.0], index=idx)
        era = pd.Series(['A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'], index=idx)

        out = prediction.gap_closure_summary(
            inclination, estimates, era=era)

        self.assertEqual(list(out['status']),
                         ['cross_era', 'incomplete_estimates'])


if __name__ == '__main__':
    unittest.main()
