"""
Tests for the Model A additions Study 05 makes to shmlib.prediction.

Run from studies/:  python 05_greybox_monitoring/tests/test_model_a.py
"""
import logging
import os
import sys
import time
import unittest
import warnings

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
            freq='1h', daily_order=3, yearly_order=1, quantiles=())
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
        # Regression test: the per-day evaluation used to also carry
        # 'yearly' rows (NeuralProphet reports that component on every
        # prediction regardless of horizon), one non-representative value
        # per hour of each evaluated date, alongside the 365 real values the
        # synthetic-year sweep produces. Only the year sweep may contribute
        # 'yearly' rows, so every one of them sits at hour 0.0 and there are
        # exactly 365 distinct dates among them.
        yearly = curves[curves['component'] == 'yearly']
        self.assertTrue((yearly['hour'] == 0.0).all())
        self.assertEqual(yearly['date'].nunique(), 365)


class TestLadder(unittest.TestCase):
    """`prediction.ladder_frame` and `prediction.channel_ladder`, the current-era
    channel ladder Movement 2b of Study 05 orchestrates (D4)."""

    def test_ladder_frame_adds_six_columns_treating_radiation_and_lead_correctly(self):
        index = pd.date_range('2024-01-01', periods=48, freq='1h')
        frame = pd.DataFrame({'y': np.arange(48.0)}, index=index)
        sensor = pd.DataFrame({
            'twall_str': np.linspace(10.0, 20.0, 48),
            'sr_str': np.linspace(0.0, 100.0, 48),
        }, index=index)

        out = prediction.ladder_frame(
            frame, sensor, index[0], twall_tau_h=4.0, twall_lead_h=-2.0,
            radiation_delay_h=1.0, freq='1h')

        for column in ('twall', 'sr_wall', 'twall_tau', 'twall_lead',
                      'summer_w', 'winter_w'):
            self.assertIn(column, out.columns)

        # A one-hour delay on a one-hour grid is a shift of one slot, with
        # no inertia (tau=0), exactly as build_regressor_sets treats every
        # radiation source.
        expected_sr_wall = sensor['sr_str'].reindex(out.index).shift(1)
        pd.testing.assert_series_equal(out['sr_wall'], expected_sr_wall,
                                       check_names=False)

        # A lead of -2 hours means the probe at time t already shows the
        # value the deformation will see two hours later, so the lead
        # column at t must carry the probe's own reading at t + 2h.
        self.assertAlmostEqual(out['twall_lead'].iloc[10], out['twall'].iloc[12])

    def test_channel_ladder_two_rungs_report_skill_only_from_the_second(self):
        frame = _frame()
        rng = np.random.default_rng(2)
        hours = np.arange(len(frame))
        frame = frame.copy()
        frame['sr'] = (np.clip(200.0 * np.sin(2 * np.pi * (hours % 24) / 24.0), 0.0, None)
                      + rng.normal(0, 5.0, len(frame)))
        rungs = [
            ('1 base', ['tair'], None),
            ('2 + sr', ['tair', 'sr'], 'sr'),
        ]

        ladder, errors = prediction.channel_ladder(
            frame, rungs, valid_p=0.2, n_changepoints=2, block_hours=24,
            repetitions=50, seed=0, epochs=2, freq='1h')

        self.assertEqual(list(ladder['rung']), ['1 base', '2 + sr'])
        self.assertTrue(np.isnan(ladder['skill'].iloc[0]))
        self.assertTrue(np.isnan(ladder['skill_q05'].iloc[0]))
        self.assertTrue(np.isnan(ladder['skill_q95'].iloc[0]))
        self.assertFalse(np.isnan(ladder['skill'].iloc[1]))
        self.assertFalse(np.isnan(ladder['skill_q05'].iloc[1]))
        self.assertFalse(np.isnan(ladder['skill_q95'].iloc[1]))
        self.assertEqual(set(errors), {'1 base', '2 + sr'})

        # skill_vs_first is chain skill's own value with only two rungs
        # (the rung below the second is also the first), but the column is
        # its own computation, not an alias, and must exist and follow the
        # same NaN-on-the-first-rung rule.
        for column in ('skill_vs_first', 'skill_vs_first_q05', 'skill_vs_first_q95'):
            self.assertIn(column, ladder.columns)
            self.assertTrue(np.isnan(ladder[column].iloc[0]))
            self.assertFalse(np.isnan(ladder[column].iloc[1]))

    def test_channel_ladder_multi_column_gate_matches_all_columns_present(self):
        # A rung's gate may name channels the rung does not itself regress
        # on, exactly how D4's ladder holds every rung to one matched
        # window defined by the deployable channels rather than by each
        # rung's own regressor coverage. Two gate columns with only
        # partially overlapping gaps exercise that the restriction is the
        # intersection of both, not either alone.
        frame = _frame()
        frame = frame.copy()
        frame['gate_a'] = 1.0
        frame['gate_b'] = 1.0
        frame.loc[frame.index[:50], 'gate_a'] = np.nan
        frame.loc[frame.index[30:80], 'gate_b'] = np.nan
        rungs = [('1 gated', ['tair'], ['gate_a', 'gate_b'])]

        ladder, _ = prediction.channel_ladder(
            frame, rungs, valid_p=0.2, n_changepoints=2, block_hours=24,
            repetitions=10, seed=0, epochs=1, freq='1h')

        expected_rows = len(frame.dropna(subset=['y', 'tair', 'gate_a', 'gate_b']))
        self.assertEqual(int(ladder.loc[0, 'rows']), expected_rows)


class TestRollingAdditions(unittest.TestCase):

    def test_rolling_conformal_covers_at_the_nominal_rate(self):
        rng = np.random.default_rng(3)
        ds = pd.date_range('2024-01-01', periods=24 * 400, freq='1h', tz='UTC')
        y = rng.normal(0, 2.0, len(ds))
        origins = ds.floor('30D')
        predictions = pd.DataFrame({'ds': ds, 'horizon_h': 0.0, 'y': y, 'yhat': 0.0,
                                    'q05': -1.0, 'q95': 1.0, 'origin': origins})
        out = prediction.rolling_conformal(predictions, alpha=0.10, window='120d')
        scored = out.dropna(subset=['q05', 'q95'])
        covered = ((scored['y'] >= scored['q05']) & (scored['y'] <= scored['q95'])).mean()
        self.assertAlmostEqual(covered, 0.90, delta=0.03)
        self.assertIn('q05_model', out.columns)
        self.assertTrue(out['qhat'].dropna().gt(0).all())

    def test_train_window_and_per_window_changepoints_are_accepted(self):
        frame = _frame(n=24 * 120)
        out = prediction.rolling_nowcast(
            frame, regressors=('tair',), refit_every='20d', min_train='40d',
            train_window='60d', changepoints_per_window=True, freq='1h',
            epochs=2, growth='linear', n_changepoints=2, quantiles=(0.05, 0.95))
        self.assertIn('staleness_d', out.columns)
        self.assertGreater(out['origin'].nunique(), 1)

    def test_score_predictions_skips_an_unused_category_without_warning(self):
        # Carried-over Phase 3 review fix: pandas 2.3 warns (soon errors)
        # when groupby() on a categorical column is not told whether to
        # restrict itself to the categories that actually occur. Before the
        # fix, an unused category silently produced an all-NaN row here;
        # after it, that row must not appear at all, and the call itself
        # must raise no FutureWarning.
        frame = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'yhat': [1.1, 1.9, 3.2, 3.8],
            'group': pd.Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'c']),
        })
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            out = prediction.score_predictions(frame, ['group'])
        self.assertEqual(sorted(out['group'].astype(str)), ['a', 'b'])


class TestParallelFits(unittest.TestCase):
    """Task 5.4c: process-level parallelism over independent fits reproduces
    the serial loop's own output, in order, up to floating-point reduction
    order."""

    def test_parallel_map_matches_serial_values(self):
        self.assertEqual(
            prediction._parallel_map(lambda x: x * 2, [1, 2, 3], n_jobs=2),
            [2, 4, 6])

    def test_parallel_map_preserves_item_order_under_uneven_timing(self):
        # Item 0 sleeps longest and finishes last; the result must still
        # come back first, because _parallel_map orders by item, not by
        # completion.
        def _sleep_and_return(item):
            index, delay = item
            time.sleep(delay)
            return index

        items = [(0, 0.3), (1, 0.2), (2, 0.1)]
        out = prediction._parallel_map(_sleep_and_return, items, n_jobs=3)
        self.assertEqual(out, [0, 1, 2])

    def test_rolling_nowcast_parallel_matches_serial(self):
        frame = _frame(n=24 * 120)
        kwargs = dict(regressors=('tair',), refit_every='20d', min_train='40d',
                     freq='1h', epochs=2)
        serial = prediction.rolling_nowcast(frame, n_jobs=1, **kwargs)
        parallel = prediction.rolling_nowcast(frame, n_jobs=2, **kwargs)
        self.assertGreaterEqual(serial['origin'].nunique(), 3)
        self.assertTrue(serial['origin'].equals(parallel['origin']))
        pd.testing.assert_frame_equal(serial, parallel, check_exact=False,
                                      rtol=1e-5)

    def test_sweep_trend_reg_parallel_matches_serial(self):
        frame = _frame(n=24 * 60)
        train, valid = frame.iloc[:1000], frame.iloc[1000:]
        serial = prediction.sweep_trend_reg(
            train, valid, [0.0, 1.0], ('tair',), n_jobs=1, epochs=2, freq='1h')
        parallel = prediction.sweep_trend_reg(
            train, valid, [0.0, 1.0], ('tair',), n_jobs=2, epochs=2, freq='1h')
        self.assertEqual(list(serial['trend_reg']), list(parallel['trend_reg']))
        pd.testing.assert_frame_equal(serial, parallel, check_exact=False,
                                      rtol=1e-5)

    def test_detached_model_pickles_small_and_predicts_the_same_once_restored(self):
        # A fitted NeuralProphet drags its whole Lightning Trainer graph
        # (loops, connectors, dataloaders with the training set) through
        # any pickle, and that graph grows far faster than the record: at
        # the notebook's scale one worker's return payload exhausted 143 GB.
        # The worker must ship the model without its trainer and the parent
        # must rebuild one before predicting.
        import cloudpickle
        frame = _frame(n=24 * 40)
        train, valid = frame.iloc[:800], frame.iloc[800:]
        model, _ = prediction.neuralprophet_backtest(
            train, valid, regressors=('tair',), task='nowcast', epochs=2, freq='1h')
        df_valid = valid.reset_index().rename(columns={'index': 'ds'})
        df_valid['ds'] = pd.DatetimeIndex(df_valid['ds']).tz_convert(None)
        before = model.predict(df_valid[['ds', 'y', 'tair']])['yhat1'].to_numpy()
        attached = len(cloudpickle.dumps(model, protocol=4))

        prediction._detach_trainer(model)
        detached = len(cloudpickle.dumps(model, protocol=4))
        self.assertIsNone(model.trainer)
        self.assertLess(detached * 10, attached)

        restored = cloudpickle.loads(cloudpickle.dumps(model, protocol=4))
        prediction._restore_trainer(restored)
        self.assertIsNotNone(restored.trainer)
        after = restored.predict(df_valid[['ds', 'y', 'tair']])['yhat1'].to_numpy()
        np.testing.assert_allclose(after, before, rtol=0, atol=0)

    def test_attribution_fits_parallel_returns_models_that_still_predict(self):
        sets, target = _regressor_sets(n=24 * 40)
        frames = prediction.regressor_set_frames(sets, target)
        kwargs = dict(valid_p=0.2, n_changepoints=2, diagnostic_lags=(1, 5),
                      epochs=2, freq='1h')
        serial_fits, serial_shares, *_ = prediction.attribution_fits(
            frames, ('tair', 'rh', 'sr'), n_jobs=1, **kwargs)
        parallel_fits, parallel_shares, *_ = prediction.attribution_fits(
            frames, ('tair', 'rh', 'sr'), n_jobs=2, **kwargs)
        self.assertEqual(list(parallel_fits), list(serial_fits))
        pd.testing.assert_frame_equal(serial_shares, parallel_shares,
                                      check_exact=False, rtol=1e-5)
        for name in serial_fits:
            self.assertIsNotNone(parallel_fits[name]['model'].trainer)
            valid = serial_fits[name]['valid']
            df_valid = valid.reset_index().rename(columns={'index': 'ds'})
            df_valid['ds'] = pd.DatetimeIndex(df_valid['ds']).tz_convert(None)
            columns = ['ds', 'y', 'tair', 'rh', 'sr']
            np.testing.assert_allclose(
                parallel_fits[name]['model'].predict(df_valid[columns])['yhat1'].to_numpy(),
                serial_fits[name]['model'].predict(df_valid[columns])['yhat1'].to_numpy(),
                rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
