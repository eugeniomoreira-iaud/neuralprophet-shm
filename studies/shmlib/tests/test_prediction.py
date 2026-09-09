"""
Tests for shmlib.prediction.radiation_filter_sweep (spec D15).

Fitting even a small NeuralProphet model is far too slow for a unit-test
budget, so every test here monkeypatches ``prediction.attribution_fits`` —
the fitting engine ``radiation_filter_sweep`` reuses rather than duplicates —
and ``prediction.decompose_components``, which ``radiation_filter_sweep``
calls a second time on each fit's own held-out tail, with small deterministic
stand-ins. That isolates exactly what ``radiation_filter_sweep`` contributes
on top of ``attribution_fits``: the per-``tau`` candidate frame construction,
the warm-up exclusion before scoring, the ``improves`` rule, and the
daily-sideband measure. No test here exercises a real NeuralProphet fit, so
none of them can catch a regression in ``attribution_fits`` or
``neuralprophet_backtest`` themselves — that is intentionally left to their
own (integration-level) coverage.

Run from studies/:  python shmlib/tests/test_prediction.py
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import coupling, prediction  # noqa: E402


class _FakeModel:
    """Stand-in for a fitted NeuralProphet model; carries only its name."""

    def __init__(self, name):
        self.name = name


def _base_frame(index):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {'y': rng.normal(size=len(index)),
         'sr': rng.normal(200.0, 50.0, len(index)).clip(min=0.0),
         'tair': rng.normal(15.0, 5.0, len(index))},
        index=index)


def _fake_decompose_components(components_by_name):
    """
    A stand-in for ``decompose_components`` keyed by the fake model's name.

    Real ``decompose_components`` re-predicts from a fitted model;
    ``components_by_name[model.name]`` hands back a canned ``y``/``yhat1``/
    ``residual`` frame instead, reindexed onto whatever frame the caller
    passed (normally one fit's own held-out ``valid``).
    """
    def _fake(model, frame, regressors=(), freq=None):
        return components_by_name[model.name].reindex(frame.index)
    return _fake


class TestCandidateFrameConstruction(unittest.TestCase):

    def test_each_candidate_replaces_only_the_radiation_column(self):
        index = pd.date_range('2024-01-01', periods=200, freq='1h')
        frame = _base_frame(index)
        radiation = pd.Series(
            np.abs(np.sin(np.arange(200) / 12.0)) * 300.0, index=index)
        taus = [2.0, 8.0]
        captured = {}

        def _fake_attribution_fits(frames, regressors, valid_p, n_changepoints,
                                   study03_gains=None, n_jobs=1, **model_kwargs):
            captured['frames'] = frames
            fits = {name: {'model': _FakeModel(name), 'valid': block.iloc[-10:]}
                   for name, block in frames.items()}
            gains = pd.DataFrame(
                [{'set': name, 'regressor': 'sr', 'gain': 0.0, 'r2': np.nan,
                 'study03_gain': np.nan} for name in frames])
            return fits, None, gains, None

        flat = pd.DataFrame({'y': 0.0, 'yhat1': 0.0, 'residual': 0.0}, index=index)
        components_by_name = {name: flat for name in
                              ['baseline', 'tau_2h', 'tau_8h']}

        with mock.patch.object(prediction, 'attribution_fits',
                               _fake_attribution_fits), \
             mock.patch.object(prediction, 'decompose_components',
                               _fake_decompose_components(components_by_name)):
            prediction.radiation_filter_sweep(
                frame, radiation, taus, ['sr', 'tair'], valid_p=0.2,
                n_changepoints=3, reset_gap='3D', freq='1h')

        frames = captured['frames']
        self.assertEqual(set(frames), {'baseline', 'tau_2h', 'tau_8h'})
        pd.testing.assert_frame_equal(frames['baseline'], frame)
        for tau, name in ((2.0, 'tau_2h'), (8.0, 'tau_8h')):
            expected_sr, _ = coupling.reset_thermal_lag_filter(
                radiation, tau, '3D', dt_hours=1.0, warmup_factor=3.0)
            pd.testing.assert_series_equal(
                frames[name]['sr'], expected_sr, check_names=False)
            pd.testing.assert_series_equal(frames[name]['tair'], frame['tair'])
            pd.testing.assert_series_equal(frames[name]['y'], frame['y'])


class TestWarmupExclusion(unittest.TestCase):

    def test_warmup_rows_are_dropped_from_both_the_mae_and_the_skill(self):
        index = pd.date_range('2024-01-01', periods=300, freq='1h')
        frame = _base_frame(index)
        radiation = pd.Series(100.0, index=index)
        # A 60-hour outage, longer than the 48-hour reset_gap, forces a
        # filter reset; the segment that resumes at position 250 starts
        # cold, so its first 3 * tau = 9 hours are flagged warm-up.
        radiation.iloc[190:250] = np.nan
        valid = frame.iloc[-50:]

        def _fake_attribution_fits(frames, regressors, valid_p, n_changepoints,
                                   study03_gains=None, n_jobs=1, **model_kwargs):
            fits = {name: {'model': _FakeModel(name), 'valid': valid}
                   for name in frames}
            gains = pd.DataFrame(
                [{'set': name, 'regressor': 'sr', 'gain': 0.0, 'r2': np.nan,
                 'study03_gain': np.nan} for name in frames])
            return fits, None, gains, None

        # Baseline: a constant absolute error of 2.0 on every held-out row.
        baseline_components = pd.DataFrame(
            {'y': 2.0, 'yhat1': 0.0, 'residual': 2.0}, index=valid.index)
        # Candidate: absolute error 5.0 for its first 9 (warm-up) rows and
        # 1.0 afterwards, on the same rows.
        candidate_y = np.where(np.arange(50) < 9, 5.0, 1.0)
        candidate_components = pd.DataFrame(
            {'y': candidate_y, 'yhat1': 0.0, 'residual': candidate_y},
            index=valid.index)
        components_by_name = {'baseline': baseline_components,
                              'tau_3h': candidate_components}

        with mock.patch.object(prediction, 'attribution_fits',
                               _fake_attribution_fits), \
             mock.patch.object(prediction, 'decompose_components',
                               _fake_decompose_components(components_by_name)):
            result = prediction.radiation_filter_sweep(
                frame, radiation, [3.0], ['sr', 'tair'], valid_p=0.2,
                n_changepoints=3, reset_gap='2D', freq='1h')

        row = result.loc[~result['is_baseline']].iloc[0]
        self.assertEqual(row['n_warmup'], 9)
        self.assertEqual(row['n_scored'], 41)
        self.assertEqual(row['n'], 41)
        self.assertAlmostEqual(row['mae_val'], 1.0)
        self.assertAlmostEqual(row['skill'], 0.5)

        baseline_row = result.loc[result['is_baseline']].iloc[0]
        self.assertEqual(baseline_row['n_warmup'], 0)
        self.assertAlmostEqual(baseline_row['mae_val'], 2.0)


class TestImprovesRule(unittest.TestCase):

    def test_improves_only_when_the_skill_lower_bound_excludes_zero(self):
        index = pd.date_range('2024-01-01', periods=50, freq='1h')
        frame = _base_frame(index)
        radiation = pd.Series(100.0, index=index)
        taus = [1.0, 2.0, 3.0, 4.0]
        valid = frame.iloc[-5:]

        def _fake_attribution_fits(frames, regressors, valid_p, n_changepoints,
                                   study03_gains=None, n_jobs=1, **model_kwargs):
            fits = {name: {'model': _FakeModel(name), 'valid': valid}
                   for name in frames}
            gains = pd.DataFrame(
                [{'set': name, 'regressor': 'sr', 'gain': 0.0, 'r2': np.nan,
                 'study03_gain': np.nan} for name in frames])
            return fits, None, gains, None

        zero_components = pd.DataFrame(
            {'y': 0.0, 'yhat1': 0.0, 'residual': 0.0}, index=valid.index)
        components_by_name = {'baseline': zero_components,
                              **{f'tau_{t:g}h': zero_components for t in taus}}

        skill_q05_by_tau = {1.0: 0.01, 2.0: 0.0, 3.0: -0.01, 4.0: np.nan}
        canned = [{'n': 5, 'parent_mae': 1.0, 'child_mae': 0.5, 'skill': 0.5,
                  'skill_q05': skill_q05_by_tau[t], 'skill_q95': 0.9}
                 for t in taus]

        with mock.patch.object(prediction, 'attribution_fits',
                               _fake_attribution_fits), \
             mock.patch.object(prediction, 'decompose_components',
                               _fake_decompose_components(components_by_name)), \
             mock.patch.object(prediction, 'paired_mae_skill',
                               side_effect=canned):
            result = prediction.radiation_filter_sweep(
                frame, radiation, taus, ['sr', 'tair'], valid_p=0.2,
                n_changepoints=3, reset_gap='2D', freq='1h')

        candidates = result.loc[~result['is_baseline']].set_index('tau_h')
        self.assertEqual(candidates.loc[1.0, 'improves'], True)
        self.assertEqual(candidates.loc[2.0, 'improves'], False)
        self.assertEqual(candidates.loc[3.0, 'improves'], False)
        self.assertEqual(candidates.loc[4.0, 'improves'], False)
        baseline_row = result.loc[result['is_baseline']].iloc[0]
        self.assertEqual(baseline_row['improves'], False)


class TestReturnShape(unittest.TestCase):

    def test_no_candidate_frame_reaches_the_fit_carrying_a_missing_driver(self):
        # The filter returns nothing inside a reset gap or where the driver
        # was absent, and the model refuses a frame with missing inputs. Those
        # rows must leave the candidate's fit rather than reach it.
        index = pd.date_range('2024-01-01', periods=60, freq='1h')
        frame = _base_frame(index)
        radiation = pd.Series(100.0, index=index)
        radiation.iloc[20:35] = np.nan          # an outage past any reset_gap
        valid = frame.iloc[-5:]
        seen = {}

        def _fake_attribution_fits(frames, regressors, valid_p, n_changepoints,
                                   study03_gains=None, n_jobs=1, **model_kwargs):
            for name, candidate in frames.items():
                seen[name] = int(candidate.isna().sum().sum())
            fits = {name: {'model': _FakeModel(name), 'valid': valid}
                    for name in frames}
            gains = pd.DataFrame(
                [{'set': name, 'regressor': 'sr', 'gain': -0.03, 'r2': 0.9,
                  'study03_gain': np.nan} for name in frames])
            return fits, None, gains, None

        flat = pd.DataFrame({'y': 0.0, 'yhat1': 0.0, 'residual': 0.0},
                            index=valid.index)
        components_by_name = {'baseline': flat, 'tau_4h': flat}

        with mock.patch.object(prediction, 'attribution_fits',
                               _fake_attribution_fits), \
             mock.patch.object(prediction, 'decompose_components',
                               _fake_decompose_components(components_by_name)):
            prediction.radiation_filter_sweep(
                frame, radiation, [4.0], ['sr', 'tair'], valid_p=0.2,
                n_changepoints=3, reset_gap='2h', freq='1h')

        self.assertEqual(seen['tau_4h'], 0)
        self.assertEqual(seen['baseline'], 0)

    def test_the_study_grid_reaches_every_fit(self):
        # ``freq`` is the sweep's own parameter and so never lands in
        # ``model_kwargs``. If it is not handed on explicitly, every fit runs
        # on the backtest's default grid instead of the study's, silently.
        index = pd.date_range('2024-01-01', periods=50, freq='1h')
        frame = _base_frame(index)
        radiation = pd.Series(100.0, index=index)
        valid = frame.iloc[-5:]
        seen = {}

        def _fake_attribution_fits(frames, regressors, valid_p, n_changepoints,
                                   study03_gains=None, n_jobs=1, **model_kwargs):
            seen.update(model_kwargs)
            fits = {name: {'model': _FakeModel(name), 'valid': valid}
                    for name in frames}
            gains = pd.DataFrame(
                [{'set': name, 'regressor': 'sr', 'gain': -0.03, 'r2': 0.9,
                  'study03_gain': np.nan} for name in frames])
            return fits, None, gains, None

        flat = pd.DataFrame({'y': 0.0, 'yhat1': 0.0, 'residual': 0.0},
                            index=valid.index)
        components_by_name = {'baseline': flat, 'tau_4h': flat}

        with mock.patch.object(prediction, 'attribution_fits',
                               _fake_attribution_fits), \
             mock.patch.object(prediction, 'decompose_components',
                               _fake_decompose_components(components_by_name)):
            prediction.radiation_filter_sweep(
                frame, radiation, [4.0], ['sr', 'tair'], valid_p=0.2,
                n_changepoints=3, reset_gap='2D', freq='1h')

        self.assertEqual(seen.get('freq'), '1h')

    def test_one_row_per_candidate_plus_the_baseline_in_order(self):
        index = pd.date_range('2024-01-01', periods=50, freq='1h')
        frame = _base_frame(index)
        radiation = pd.Series(100.0, index=index)
        taus = [4.0, 12.0]
        valid = frame.iloc[-5:]

        def _fake_attribution_fits(frames, regressors, valid_p, n_changepoints,
                                   study03_gains=None, n_jobs=1, **model_kwargs):
            fits = {name: {'model': _FakeModel(name), 'valid': valid}
                   for name in frames}
            gains = pd.DataFrame(
                [{'set': name, 'regressor': 'sr', 'gain': -0.03, 'r2': 0.9,
                 'study03_gain': study03_gains.get((name, 'sr'), np.nan)}
                for name in frames])
            return fits, None, gains, None

        flat = pd.DataFrame({'y': 0.0, 'yhat1': 0.0, 'residual': 0.0},
                            index=valid.index)
        components_by_name = {'baseline': flat, 'tau_4h': flat, 'tau_12h': flat}

        with mock.patch.object(prediction, 'attribution_fits',
                               _fake_attribution_fits), \
             mock.patch.object(prediction, 'decompose_components',
                               _fake_decompose_components(components_by_name)):
            result = prediction.radiation_filter_sweep(
                frame, radiation, taus, ['sr', 'tair'], valid_p=0.2,
                n_changepoints=3, reset_gap='2D', freq='1h',
                study03_gain=-0.035)

        self.assertEqual(list(result.columns), [
            'tau_h', 'is_baseline', 'mae_val', 'skill', 'skill_q05',
            'skill_q95', 'n', 'gain', 'study03_gain', 'sideband_amplitude',
            'n_warmup', 'n_scored', 'improves'])
        self.assertEqual(len(result), 3)
        self.assertTrue(result.iloc[0]['is_baseline'])
        self.assertTrue(np.isnan(result.iloc[0]['tau_h']))
        self.assertEqual(list(result.iloc[1:]['tau_h']), [4.0, 12.0])
        self.assertFalse(result.iloc[1]['is_baseline'])
        self.assertFalse(result.iloc[2]['is_baseline'])
        self.assertTrue((result['study03_gain'] == -0.035).all())


class TestSidebandAmplitude(unittest.TestCase):

    def _run_with_residual(self, residual, freq='1h'):
        index = residual.index
        frame = _base_frame(index)
        radiation = pd.Series(100.0, index=index)
        valid = frame

        def _fake_attribution_fits(frames, regressors, valid_p, n_changepoints,
                                   study03_gains=None, n_jobs=1, **model_kwargs):
            fits = {name: {'model': _FakeModel(name), 'valid': valid}
                   for name in frames}
            gains = pd.DataFrame(
                [{'set': name, 'regressor': 'sr', 'gain': 0.0, 'r2': np.nan,
                 'study03_gain': np.nan} for name in frames])
            return fits, None, gains, None

        components = pd.DataFrame(
            {'y': residual.to_numpy(), 'yhat1': 0.0,
            'residual': residual.to_numpy()}, index=index)
        components_by_name = {'baseline': components, 'tau_1h': components}

        with mock.patch.object(prediction, 'attribution_fits',
                               _fake_attribution_fits), \
             mock.patch.object(prediction, 'decompose_components',
                               _fake_decompose_components(components_by_name)):
            return prediction.radiation_filter_sweep(
                frame, radiation, [1.0], ['sr', 'tair'], valid_p=0.2,
                n_changepoints=3, reset_gap='2D', freq=freq)

    def test_too_short_a_held_out_tail_reports_nan_not_an_exception(self):
        index = pd.date_range('2024-01-01', periods=50, freq='1h')
        residual = pd.Series(
            np.sin(np.arange(50) * 2.0 * np.pi / 24.0), index=index)

        result = self._run_with_residual(residual)

        self.assertTrue(result['sideband_amplitude'].isna().all())

    def test_a_clean_two_year_annual_swing_is_recovered(self):
        index = pd.date_range('2024-01-01', periods=2 * 365 * 72, freq='20min')
        hour = (index.hour + index.minute / 60.0).to_numpy()
        doy = index.dayofyear.to_numpy(dtype=float)
        # A pure daily cosine whose own amplitude is modulated across the
        # year as 2.0 + 1.0 * cos(2 pi doy / 365.25): true annual amplitude
        # of that modulation is 1.0.
        daily_amplitude = 2.0 + 1.0 * np.cos(2.0 * np.pi * doy / 365.25)
        residual = pd.Series(
            daily_amplitude * np.cos(2.0 * np.pi * hour / 24.0), index=index)

        result = self._run_with_residual(residual, freq='20min')

        candidate = result.loc[~result['is_baseline']].iloc[0]
        self.assertFalse(np.isnan(candidate['sideband_amplitude']))
        self.assertAlmostEqual(candidate['sideband_amplitude'], 1.0, delta=0.25)


if __name__ == '__main__':
    unittest.main()
