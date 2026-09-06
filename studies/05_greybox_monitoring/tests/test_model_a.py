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


def _regressor_sets(n=24 * 60, seed=0):
    """Two hand-built regressor sets sharing one target, for the Movement 2
    (attribution-fit) helpers: `sets` mimics `proxies.build_regressor_sets`'s
    first return, a dict of set name to a block already carrying the plain
    role columns `tair`, `rh`, `sr` on the target's own index."""
    base = _frame(n=n, seed=seed)
    hours = np.arange(n)
    rng = np.random.default_rng(seed + 1)
    rh = 50.0 + 5.0 * np.sin(2 * np.pi * hours / 24.0 + 1.0) + rng.normal(0, 0.5, n)
    sr = np.clip(200.0 * np.sin(2 * np.pi * (hours % 24) / 24.0), 0.0, None)
    sets = {
        'a': pd.DataFrame({'tair': base['tair'], 'rh': rh, 'sr': sr}, index=base.index),
        'b': pd.DataFrame({'tair': base['tair'] + 1.0, 'rh': rh - 2.0, 'sr': sr * 0.9},
                          index=base.index),
    }
    return sets, base['y']


class TestMovementTwoHelpers(unittest.TestCase):

    def test_regressor_set_frames_joins_and_drops_missing(self):
        index = pd.date_range('2024-01-01', periods=10, freq='1h', tz='UTC')
        target = pd.Series(np.arange(10.0), index=index, name='y')
        target.iloc[0] = np.nan
        block_a = pd.DataFrame({'tair': np.arange(10.0), 'rh': np.arange(10.0),
                                'sr': np.arange(10.0)}, index=index)
        block_a.loc[index[3], 'tair'] = np.nan
        block_b = pd.DataFrame({'tair': np.arange(10.0), 'rh': np.arange(10.0),
                                'sr': np.arange(10.0)}, index=index)
        block_b.loc[index[7], 'sr'] = np.nan
        sets = {'a': block_a, 'b': block_b}

        out = prediction.regressor_set_frames(sets, target)

        self.assertEqual(set(out), {'a', 'b'})
        self.assertEqual(list(out['a'].columns),
                         ['y', 'tair', 'rh', 'sr', 'summer_w', 'winter_w'])
        self.assertEqual(list(out['b'].columns),
                         ['y', 'tair', 'rh', 'sr', 'summer_w', 'winter_w'])
        # The row missing in the target (index[0]) is dropped from both sets;
        # each set additionally loses its own NaN row.
        self.assertEqual(len(out['a']), 8)
        self.assertEqual(len(out['b']), 8)
        self.assertNotIn(index[0], out['a'].index)
        self.assertNotIn(index[3], out['a'].index)
        self.assertNotIn(index[0], out['b'].index)
        self.assertNotIn(index[7], out['b'].index)

    def test_sweep_trend_reg_flags_one_minimum(self):
        frame = _frame()
        train, valid = frame.iloc[:1000], frame.iloc[1000:]

        sweep = prediction.sweep_trend_reg(
            train, valid, (0.0, 1.0), ('tair',), epochs=2, freq='1h')

        self.assertEqual(len(sweep), 2)
        self.assertEqual(sweep['chosen'].sum(), 1)
        self.assertEqual(sweep.loc[sweep['chosen'], 'trend_reg'].iloc[0],
                         sweep.loc[sweep['mae_val'].idxmin(), 'trend_reg'])

    def test_compare_daily_terms_reports_two_rows(self):
        frame = _frame()
        train, valid = frame.iloc[:1000], frame.iloc[1000:]
        conditions = {'daily_summer': 'summer_w', 'daily_winter': 'winter_w'}

        table, keep = prediction.compare_daily_terms(
            train, valid, ('tair',), conditions, 0.01, epochs=2, freq='1h')

        self.assertEqual(list(table['daily_term']), ['plain', 'conditional'])
        self.assertIsInstance(keep, bool)
        self.assertTrue((table['daily_share'] >= 0).all())
        self.assertTrue((table['daily_share'] <= 1).all())

    def test_attribution_fits_builds_fits_and_tables(self):
        sets, target = _regressor_sets()
        frames = prediction.regressor_set_frames(sets, target)
        study03_gains = {('a', 'tair'): -2.5}

        fits, shares, gains, diagnostics = prediction.attribution_fits(
            frames, ('tair', 'rh', 'sr'), valid_p=0.2, n_changepoints=2,
            study03_gains=study03_gains, diagnostic_lags=(1, 5),
            weight_curve='sentinel', epochs=2, freq='1h')

        self.assertEqual(set(fits), {'a', 'b'})
        for fit in fits.values():
            self.assertTrue(hasattr(fit['model'], 'fit_metrics_'))
            self.assertEqual(fit['model'].weight_curve_, 'sentinel')
        self.assertEqual(shares.columns[0], 'set')
        self.assertEqual(set(shares['set']), {'a', 'b'})
        self.assertEqual(gains.columns[0], 'set')
        self.assertEqual(set(gains['set']), {'a', 'b'})
        self.assertEqual(diagnostics.columns[0], 'set')
        self.assertEqual(set(diagnostics['set']), {'a', 'b'})

        known = gains[(gains['set'] == 'a') & (gains['regressor'] == 'tair')].iloc[0]
        self.assertAlmostEqual(known['study03_gain'], -2.5)
        unknown = gains[(gains['set'] == 'a') & (gains['regressor'] == 'rh')].iloc[0]
        self.assertTrue(np.isnan(unknown['study03_gain']))

    def test_fold_stability_returns_expected_rows(self):
        sets, target = _regressor_sets()
        frames = prediction.regressor_set_frames(sets, target)
        fits, _, _, _ = prediction.attribution_fits(
            frames, ('tair', 'rh', 'sr'), valid_p=0.2, n_changepoints=2,
            epochs=2, freq='1h')

        stability = prediction.fold_stability(
            fits, frames, ('tair', 'rh', 'sr'), n_changepoints=2, k=2,
            fold_pct=0.1, fold_overlap_pct=0.0, freq='1h', epochs=2)

        self.assertEqual(len(stability), 4)
        self.assertEqual(list(stability.columns),
                         ['set', 'fold', 'tair_gain', 'yearly_peak_to_peak',
                          'trend_rate', 'mae_val'])
        self.assertFalse(stability['mae_val'].isna().any())

    def test_select_by_ds_localises_naive_ds_through_utc(self):
        # Regression test for the Task 2.4 fix round 1: crossvalidation_
        # split_df's naive 'ds' is a UTC wall-clock reading (NeuralProphet
        # 0.8.0 strips any zone through UTC), not a reading already in the
        # block's own zone. Localising it straight to 'Europe/Rome' (the
        # bug) would shift every row by the zone's UTC offset and select
        # the wrong instants; going through UTC first must select exactly
        # the block rows at the instants 'ds' actually denotes. The
        # expected rows are built independently of the function under
        # test, through the same UTC-naive membership check
        # decompose_components uses.
        index = pd.date_range('2024-01-01', periods=48, freq='1h',
                              tz='Europe/Rome')
        block = pd.DataFrame({'y': np.arange(48.0)}, index=index)
        chosen = index[[3, 10, 25, 40]]
        ds = pd.Series(chosen.tz_convert('UTC').tz_localize(None))

        selected = prediction._select_by_ds(block, ds)

        naive_utc = index.tz_convert('UTC').tz_localize(None)
        expected = block.loc[naive_utc.isin(ds)]
        pd.testing.assert_frame_equal(selected, expected)


class TestExtractors(unittest.TestCase):

    def test_regressor_gain_is_the_slope_of_the_component(self):
        index = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
        tair = pd.Series(np.linspace(0, 10, 100), index=index)
        components = pd.DataFrame({'future_regressor_tair': -2.5 * tair + 3.0}, index=index)
        frame = pd.DataFrame({'tair': tair}, index=index)
        gains = prediction.regressor_gains(components, frame, ('tair',))
        self.assertAlmostEqual(gains.loc[0, 'gain'], -2.5, places=6)
        self.assertAlmostEqual(gains.loc[0, 'r2'], 1.0, places=6)

    def test_decompose_components_keeps_the_frame_s_own_timezone(self):
        # Regression test: NeuralProphet's predict() always returns 'ds'
        # tz-naive (see the tz_aware_index test above), and decompose_
        # components used to leave that naive index uncorrected. The
        # observed-value reindex a few lines later then aligned a tz-aware
        # Series against a tz-naive one, which pandas does not raise on --
        # it just matches nothing, so 'y' and 'residual' came back entirely
        # NaN with no error to notice. This pins the fix: on a tz-aware
        # frame, decompose_components' own index carries that same
        # timezone, and 'y'/'residual' are populated exactly as they are on
        # a tz-naive frame.
        frame = _frame().tz_convert('Europe/Rome')  # exercise a real UTC offset
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=2,
            freq='1h', quantiles=())
        components = prediction.decompose_components(
            model, frame, regressors=('tair',))
        self.assertEqual(components.index.tz, frame.index.tz)
        self.assertTrue(components.index.equals(frame.index))
        self.assertFalse(components['y'].isna().any())
        self.assertFalse(components['residual'].isna().any())

    def test_trend_rates_recover_a_linear_drift(self):
        frame = _frame(n=24 * 120)
        changepoints = prediction.covered_changepoints(frame.index, 2)
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=8,
            freq='1h', growth='linear', changepoints=changepoints,
            n_changepoints=2, quantiles=())
        trend, rates = prediction.trend_parameters(model, frame, changepoints,
                                                   regressors=('tair',))
        self.assertEqual(len(trend), len(frame))
        self.assertGreater(len(rates), 0)
        # 0.01 per hour is 87.6 per year; NeuralProphet's fit is stochastic
        self.assertAlmostEqual(rates['rate_mdeg_per_year'].mean(), 87.6, delta=30.0)

    def test_seasonal_parameters_return_one_curve_per_date(self):
        frame = _frame()
        model, _ = prediction.neuralprophet_backtest(
            frame, frame, regressors=('tair',), task='nowcast', epochs=3,
            freq='1h', daily_order=3, quantiles=())
        curves = prediction.seasonal_parameters(
            model, ['2024-06-21', '2024-12-21'], freq='1h', regressors=('tair',))
        daily = curves[curves['component'] == 'daily']
        self.assertEqual(daily['date'].nunique(), 2)
        self.assertEqual(daily.groupby('date').size().iloc[0], 24)
        # 'date' must be one comparable dtype across daily and yearly rows:
        # tz-naive on every row, so a caller can sort or compare the whole
        # column without NeuralProphet's tz-aware yearly grid leaking through.
        self.assertIsNone(pd.DatetimeIndex(curves['date']).tz)
        sorted_curves = curves.sort_values('date')
        self.assertEqual(len(sorted_curves), len(curves))


if __name__ == '__main__':
    unittest.main(verbosity=2)
