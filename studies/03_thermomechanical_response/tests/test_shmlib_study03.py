"""
Unit tests for the decisions study 3 relies on, which live in ``shmlib``.

Run without a test runner, as the other studies' tests are run::

    python studies/03_thermomechanical_response/tests/test_shmlib_study03.py

What is tested here is the set of choices this study's measurement makes, not
the arithmetic of pandas or of statsmodels. Five of them carry the study's
conclusions and would be silent if they broke:

* the lag search itself, which must recover a delay that was put into a
  synthetic pair and must refuse to score a lag the record cannot support
  (:func:`shmlib.coupling.lag_scan`);
* the objective the search maximises, since a coupling of expected negative
  sign is invisible to a search that maximises the signed correlation
  (:func:`shmlib.coupling.best_lag`);
* the band separation, which must remove a slow drift without altering the
  daily cycle whose amplitude the gain is then fitted to
  (:func:`shmlib.coupling.diurnal_band`);
* the Newey-West standard error, which is the difference between a gain that
  is significant and one that only looks it on an autocorrelated series
  (:func:`shmlib.coupling.gain_at_lag`);
* the site's sign convention, asserted here so that the expectation stated in
  ``docs/raw-data-format.md`` section 7.5 is executable rather than only
  written down (:func:`shmlib.coupling.shortlist`).

One further test is a regression rather than a specification.
:func:`shmlib.quality.clock_check` carried its own copy of the lag search until
study 3 extracted it, and study 2's clock results depend on that search
answering exactly as it did before. ``TestClockCheckRegression`` reimplements
the loop as it stood and requires the shared search to agree with it.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.abspath(os.path.join(_HERE, '..'))
_STUDIES = os.path.abspath(os.path.join(_STUDY, '..'))
for path in (_STUDY, _STUDIES):
    if path not in sys.path:
        sys.path.insert(0, path)

from shmlib import coupling, meteo, proxies, quality, site        # noqa: E402


def _hourly(values, start='2025-03-01', tz='UTC'):
    """A Series of `values` on an hourly index, for a synthetic pair."""
    index = pd.date_range(start, periods=len(values), freq='1h', tz=tz)
    return pd.Series(np.asarray(values, dtype=float), index=index)


def _diurnal(hours, amplitude=1.0, phase=0.0):
    """A clean daily cycle, the signal every lag scan at this site works on."""
    t = np.arange(hours, dtype=float)
    return amplitude * np.sin(2.0 * np.pi * (t - phase) / 24.0)


class TestLagScan(unittest.TestCase):
    """The search, against a delay that was put into the pair on purpose."""

    def test_scan_recovers_an_injected_delay(self):
        driver = _hourly(_diurnal(24 * 30))
        # The response reproduces the driver four hours later and inverted,
        # which is the sign the site's geometry predicts for a heating driver.
        response = -driver.shift(4)
        curve = coupling.lag_scan(response, driver, range(0, 13))
        winner = coupling.best_lag(curve, objective='absolute')
        self.assertEqual(winner['lag'], 4.0)
        self.assertLess(winner['r'], -0.99)

    def test_scan_reports_every_lag_it_was_given(self):
        driver = _hourly(_diurnal(24 * 10))
        curve = coupling.lag_scan(driver, driver, range(-6, 7))
        self.assertEqual(list(curve['lag']), list(range(-6, 7)))
        self.assertEqual(len(curve), 13)

    def test_a_lag_without_enough_overlap_is_not_scored(self):
        driver = _hourly(_diurnal(30))
        response = -driver.shift(2)
        curve = coupling.lag_scan(response, driver, [2], min_paired=100)
        self.assertTrue(pd.isna(curve['r'].iloc[0]))
        # The count is still reported, so the curve says where the record ran
        # out rather than silently dropping the lag.
        self.assertGreater(curve['n'].iloc[0], 0)

    def test_a_curve_with_nothing_scored_has_no_winner(self):
        driver = _hourly(_diurnal(30))
        curve = coupling.lag_scan(driver, driver, [0], min_paired=100)
        winner = coupling.best_lag(curve)
        self.assertTrue(np.isnan(winner['lag']))
        self.assertEqual(winner['n'], 0)


class TestThermalOperator(unittest.TestCase):
    """The second parameter: inertia, which a pure delay cannot express."""

    def test_a_zero_time_constant_changes_nothing(self):
        driver = _hourly(_diurnal(48))
        filtered = coupling.thermal_lag_filter(driver, 0.0)
        self.assertTrue(np.allclose(filtered.to_numpy(), driver.to_numpy()))

    def test_the_filter_attenuates_and_delays_a_daily_cycle(self):
        driver = _hourly(_diurnal(24 * 30, amplitude=1.0))
        filtered = coupling.thermal_lag_filter(driver, 6.0)
        settled = filtered.iloc[48:]
        # A one-pole filter of time constant tau cannot pass a daily cycle
        # undiminished, and it moves the peak later. Both are the physics the
        # instantaneous assumption throws away.
        self.assertLess(float(settled.abs().max()), 1.0)
        self.assertGreater(float(settled.abs().max()), 0.1)
        peak_driver = int(np.argmax(driver.iloc[48:96].to_numpy()))
        peak_filtered = int(np.argmax(filtered.iloc[48:96].to_numpy()))
        self.assertGreater(peak_filtered, peak_driver)

    def test_a_gap_is_bridged_for_the_filter_and_restored_after_it(self):
        driver = _hourly(_diurnal(24 * 5))
        driver.iloc[30:36] = np.nan
        filtered = coupling.thermal_lag_filter(driver, 4.0)
        # The gap comes back as a gap: the bridge exists to keep the filter
        # state running across it, not to invent six hours of measurement.
        self.assertTrue(filtered.iloc[30:36].isna().all())
        self.assertFalse(filtered.iloc[36:48].isna().any())

    def test_the_delay_and_the_inertia_commute(self):
        driver = _hourly(_diurnal(24 * 10))
        operated = coupling.thermal_operator(driver, delay=3, tau=5.0)
        self.assertTrue(np.allclose(
            operated.dropna().to_numpy(),
            coupling.thermal_lag_filter(driver, 5.0).shift(3).dropna().to_numpy()))
        # Both operations are linear and time-invariant, so the order is a
        # matter of exposition and not of arithmetic; away from the ends of the
        # series the two orders give the same numbers. The docstring states the
        # physical order because a reader has to be told which way the model
        # runs, not because the result would change.
        other_order = coupling.thermal_lag_filter(driver.shift(3), 5.0)
        self.assertTrue(np.allclose(operated.iloc[10:40].to_numpy(),
                                    other_order.iloc[10:40].to_numpy()))


class TestOperatorScan(unittest.TestCase):
    """The grid, and what it says that a single lag curve cannot."""

    def test_an_injected_time_constant_is_recovered(self):
        driver = _hourly(_diurnal(24 * 40))
        response = -2.0 * coupling.thermal_lag_filter(driver, 6.0)
        scan = coupling.operator_scan(response, driver, range(0, 13),
                                      taus=(0, 1, 2, 4, 6, 8, 12, 24))
        best = coupling.best_operator(scan)
        self.assertEqual(best['delay'], 0.0)
        self.assertEqual(best['tau'], 6.0)
        # And the instantaneous case, which a delay-only scan would have been
        # stuck with, explains far less.
        self.assertLess(best['r2_instantaneous'], 0.6)
        self.assertGreater(best['gain_from_operator'], 0.3)

    def test_an_injected_delay_is_recovered_without_inventing_inertia(self):
        rng = np.random.default_rng(5)
        driver = _hourly(_diurnal(24 * 40) + rng.normal(scale=0.05, size=24 * 40))
        response = -driver.shift(4)
        scan = coupling.operator_scan(response, driver, range(0, 13),
                                      taus=(0, 1, 2, 4, 8))
        best = coupling.best_operator(scan)
        self.assertEqual(best['delay'], 4.0)
        self.assertEqual(best['tau'], 0.0)

    def test_a_grid_of_one_time_constant_reproduces_the_lag_scan(self):
        # The bridge between the two: with no inertia in the grid, the joint
        # scan is the delay scan, cell for cell. This is what lets
        # ``quality.clock_check`` keep using the one-dimensional form.
        rng = np.random.default_rng(9)
        driver = _hourly(_diurnal(24 * 20) + rng.normal(scale=0.1, size=24 * 20))
        response = -driver.shift(3)
        grid = coupling.operator_scan(response, driver, range(-6, 7), taus=(0.0,))
        curve = coupling.lag_scan(response, driver, range(-6, 7))
        self.assertTrue(np.allclose(grid['r'].to_numpy(), curve['r'].to_numpy(),
                                    equal_nan=True))
        self.assertTrue(np.array_equal(grid['n'].to_numpy(),
                                       curve['n'].to_numpy()))

    def test_the_causal_restriction_costs_what_the_lead_was_worth(self):
        # A driver with a second harmonic and some noise, because a pure daily
        # sine is its own alias: a lead of four hours and a delay of eight are
        # the same alignment on it, and no restriction can cost anything.
        rng = np.random.default_rng(31)
        hours = 24 * 40
        t = np.arange(hours, dtype=float)
        driver = _hourly(np.sin(2.0 * np.pi * t / 24.0)
                         + 0.6 * np.sin(2.0 * np.pi * t / 12.0)
                         + rng.normal(scale=0.05, size=hours))
        # A response that leads its driver: the wall-temperature case, where the
        # probe follows the movement instead of preceding it.
        response = -driver.shift(-4)
        scan = coupling.operator_scan(response, driver, range(-12, 13),
                                      taus=(0, 2, 4))
        free = coupling.best_operator(scan)
        causal = coupling.best_operator(scan, causal_only=True)
        self.assertEqual(free['delay'], -4.0)
        self.assertGreaterEqual(causal['delay'], 0.0)
        self.assertGreater(free['r2'] - causal['r2'], 0.0)


class TestObjective(unittest.TestCase):
    """A negative coupling is invisible to a search that maximises ``r``."""

    def test_absolute_and_signed_disagree_on_a_negative_coupling(self):
        driver = _hourly(_diurnal(24 * 30))
        response = -driver.shift(4)
        # The range an external forcing is admitted over: a forcing precedes
        # its response, and twelve hours is where a delay stops being
        # distinguishable from a lead of what is left of the day.
        curve = coupling.lag_scan(response, driver, range(0, 13))

        absolute = coupling.best_lag(curve, objective='absolute')
        signed = coupling.best_lag(curve, objective='signed')

        self.assertEqual(absolute['lag'], 4.0)
        self.assertLess(absolute['r'], -0.99)
        # The signed search walks away from a perfect anti-correlation towards
        # whatever positive value the range still contains, which is exactly
        # why this study does not use it.
        self.assertEqual(signed['lag'], 12.0)
        self.assertGreater(signed['r'], 0.0)

    def test_an_unknown_objective_is_refused(self):
        curve = coupling.lag_scan(_hourly(_diurnal(48)), _hourly(_diurnal(48)), [0])
        with self.assertRaises(ValueError):
            coupling.best_lag(curve, objective='largest')


class TestDiurnalBand(unittest.TestCase):
    """The filter must remove the drift and leave the daily cycle alone."""

    def test_a_linear_drift_is_removed(self):
        hours = 24 * 20
        drift = _hourly(np.linspace(0.0, 100.0, hours))
        band = coupling.diurnal_band(drift, window=24)
        # The interior, where the window is full. At the two ends the window
        # is truncated and its mean is drawn from one side of the centre only,
        # so the filter cannot be unbiased there and the study does not read
        # the first and last day of any stratum as though it were.
        interior = band.iloc[24:-24]
        # The drift is gone: what remains is flat. It is not identically zero,
        # because a 24-sample centred window has no centre sample and its mean
        # sits half a step off the point it is subtracted from. That leaves a
        # constant, which changes neither a correlation nor a slope.
        self.assertLess(float(interior.std()), 1e-9)
        self.assertLess(float(interior.max() - interior.min()), 1e-9)

    def test_a_daily_cycle_keeps_its_amplitude(self):
        hours = 24 * 20
        cycle = _hourly(_diurnal(hours, amplitude=3.0))
        band = coupling.diurnal_band(cycle, window=24)
        interior = band.iloc[24:-24]
        self.assertAlmostEqual(float(interior.max()), 3.0, places=6)
        self.assertAlmostEqual(float(interior.min()), -3.0, places=6)

    def test_the_two_bands_can_disagree(self):
        # A pair that shares a rising drift but whose daily cycles are
        # anti-phase correlates positively on the levels and negatively on the
        # band. Both numbers are true of the same pair, which is why the study
        # reports both.
        hours = 24 * 30
        drift = np.linspace(0.0, 40.0, hours)
        driver = _hourly(drift + _diurnal(hours, amplitude=2.0))
        response = _hourly(drift - _diurnal(hours, amplitude=2.0))

        level = coupling.lag_scan(response, driver, [0])['r'].iloc[0]
        banded = coupling.lag_scan(coupling.diurnal_band(response),
                                   coupling.diurnal_band(driver), [0])['r'].iloc[0]
        self.assertGreater(level, 0.0)
        self.assertLess(banded, 0.0)


class TestGain(unittest.TestCase):
    """The slope, and the standard error an hourly series actually deserves."""

    def test_a_known_slope_is_recovered(self):
        rng = np.random.default_rng(20260822)
        driver = _hourly(rng.normal(size=2000))
        response = -3.37 * driver + _hourly(rng.normal(scale=0.01, size=2000))
        gain = coupling.gain_at_lag(response, driver, 0)
        self.assertAlmostEqual(gain['slope'], -3.37, places=2)
        self.assertEqual(gain['n'], 2000)

    def test_a_lagged_slope_is_recovered_at_its_lag(self):
        rng = np.random.default_rng(7)
        driver = _hourly(rng.normal(size=1000))
        response = 2.0 * driver.shift(5)
        gain = coupling.gain_at_lag(response, driver, 5)
        self.assertAlmostEqual(gain['slope'], 2.0, places=6)

    def test_the_hac_error_exceeds_the_ordinary_one(self):
        import statsmodels.api as sm

        rng = np.random.default_rng(11)
        hours = 4000
        noise = np.zeros(hours)
        for i in range(1, hours):
            noise[i] = 0.95 * noise[i - 1] + rng.normal()
        driver = _hourly(_diurnal(hours, amplitude=5.0) + rng.normal(size=hours))
        response = -1.5 * driver + _hourly(noise)

        hac = coupling.gain_at_lag(response, driver, 0, hac_maxlags=24)
        paired = pd.concat([driver, response], axis=1).dropna()
        ordinary = sm.OLS(paired.iloc[:, 1].to_numpy(),
                          sm.add_constant(paired.iloc[:, 0].to_numpy())).fit()
        self.assertGreater(hac['slope_se'], float(ordinary.bse[1]))

    def test_a_missing_lag_returns_a_missing_gain(self):
        driver = _hourly(_diurnal(48))
        gain = coupling.gain_at_lag(driver, driver, np.nan)
        self.assertTrue(np.isnan(gain['slope']))
        self.assertEqual(gain['n'], 0)


class TestCouple(unittest.TestCase):
    """The orchestration: every driver, in every band, over every stratum."""

    def _frame(self):
        hours = 24 * 60
        driver = _diurnal(hours, amplitude=6.0)
        rng = np.random.default_rng(3)
        frame = pd.DataFrame({
            'inc_comp_cleaned': -0.5 * np.roll(driver, 3),
            'tair_str': driver,
            'batt_str': rng.normal(size=hours),
        }, index=pd.date_range('2025-03-01', periods=hours, freq='1h', tz='UTC'))
        return frame

    def test_every_combination_produces_a_row(self):
        frame = self._frame()
        table, curves = coupling.couple(
            frame, 'inc_comp_cleaned', ['tair_str', 'batt_str'],
            lags={None: range(0, 13)},
            strata={'all': pd.Series(True, index=frame.index)})
        self.assertEqual(len(table), 2 * 2)          # two drivers, two bands
        self.assertEqual(len(curves), 4)
        self.assertEqual(set(table['band']), {coupling.BAND_LEVEL,
                                              coupling.BAND_DIURNAL})

    def test_a_driver_keeps_its_own_lag_range(self):
        frame = self._frame()
        table, _ = coupling.couple(
            frame, 'inc_comp_cleaned', ['tair_str'],
            lags={'tair_str': range(0, 13)}, bands=[coupling.BAND_LEVEL])
        self.assertTrue((table['lag'] >= 0).all())
        self.assertTrue((table['lag'] <= 12).all())

    def test_a_driver_without_a_lag_range_is_skipped(self):
        frame = self._frame()
        table, _ = coupling.couple(
            frame, 'inc_comp_cleaned', ['tair_str', 'batt_str'],
            lags={'tair_str': range(0, 13)}, bands=[coupling.BAND_LEVEL])
        self.assertEqual(list(table['driver']), ['tair_str'])


class TestCoupleOperators(unittest.TestCase):
    """The screen, run over the grid rather than over delays alone."""

    def _inertial_frame(self):
        hours = 24 * 60
        index = pd.date_range('2025-03-01', periods=hours, freq='1h', tz='UTC')
        driver = pd.Series(_diurnal(hours, amplitude=5.0), index=index)
        return pd.DataFrame({
            'inc_comp_cleaned': -2.0 * coupling.thermal_lag_filter(driver, 8.0),
            'tair_str': driver,
        })

    def test_the_operator_columns_are_reported(self):
        frame = self._inertial_frame()
        table, _ = coupling.couple(frame, 'inc_comp_cleaned', ['tair_str'],
                                   lags={None: range(0, 13)},
                                   taus=(0, 2, 4, 8, 16),
                                   bands=[coupling.BAND_LEVEL])
        row = table.iloc[0]
        self.assertEqual(row['tau'], 8.0)
        self.assertGreater(row['gain_from_operator'], 0.0)
        self.assertEqual(row['r2_lost_to_causality'], 0.0)

    def test_the_gain_is_fitted_against_the_operator_that_won(self):
        # The slope is -2 against the filtered driver. Fitted against the raw
        # one it is not, and a study reporting that number would be quoting the
        # slope of a different variable.
        frame = self._inertial_frame()
        table, _ = coupling.couple(frame, 'inc_comp_cleaned', ['tair_str'],
                                   lags={None: range(0, 13)},
                                   taus=(0, 2, 4, 8, 16),
                                   bands=[coupling.BAND_LEVEL])
        self.assertAlmostEqual(float(table.iloc[0]['slope']), -2.0, places=6)

    def test_a_delay_only_screen_mistakes_the_inertia_for_a_delay(self):
        # The defect this whole family exists to fix, made executable. The
        # frame's response is the driver put through an eight-hour time
        # constant and no delay at all. Screened over delays alone, the scan
        # has no way to say so: it reports a transport delay that never
        # happened, and fits the gain against the unfiltered driver, which is
        # not the variable the response actually follows.
        frame = self._inertial_frame()
        delays_only, _ = coupling.couple(frame, 'inc_comp_cleaned', ['tair_str'],
                                         lags={None: range(0, 13)},
                                         bands=[coupling.BAND_LEVEL])
        self.assertEqual(float(delays_only.iloc[0]['tau']), 0.0)
        self.assertGreater(float(delays_only.iloc[0]['lag']), 0.0)
        self.assertLess(abs(float(delays_only.iloc[0]['slope'])), 2.0)

        joint, _ = coupling.couple(frame, 'inc_comp_cleaned', ['tair_str'],
                                   lags={None: range(0, 13)},
                                   taus=(0, 2, 4, 8, 16),
                                   bands=[coupling.BAND_LEVEL])
        self.assertEqual(float(joint.iloc[0]['tau']), 8.0)
        self.assertEqual(float(joint.iloc[0]['lag']), 0.0)
        self.assertGreater(float(joint.iloc[0]['r2']),
                           float(delays_only.iloc[0]['r2']))


class TestShortlist(unittest.TestCase):
    """The verdict, and the sign convention of section 7.5."""

    def _table(self):
        return pd.DataFrame([
            {'stratum': 'all', 'band': 'level', 'driver': 'tair_str',
             'lag': 3.0, 'r': -0.90, 'n': 1000, 'slope': -3.37,
             'slope_se': 0.1, 'ci_low': -3.6, 'ci_high': -3.1, 'sign': '-'},
            {'stratum': 'all', 'band': 'level', 'driver': 'pres_gs',
             'lag': 1.0, 'r': 0.20, 'n': 1000, 'slope': 0.4,
             'slope_se': 0.5, 'ci_low': -0.6, 'ci_high': 1.4, 'sign': '+'},
            {'stratum': 'all', 'band': 'level', 'driver': 'batt_str',
             'lag': 0.0, 'r': 0.30, 'n': 1000, 'slope': 0.2,
             'slope_se': 0.1, 'ci_low': 0.0, 'ci_high': 0.4, 'sign': '+'},
        ])

    def test_the_control_sets_the_floor_and_leaves_the_table(self):
        result = coupling.shortlist(self._table(), control='batt_str')
        self.assertNotIn('batt_str', list(result['driver']))
        self.assertTrue((result['control_r'] == 0.30).all())

    def test_a_candidate_below_the_control_does_not_clear_it(self):
        result = coupling.shortlist(self._table(), control='batt_str')
        by_driver = result.set_index('driver')
        self.assertTrue(bool(by_driver.loc['tair_str', 'clears_control']))
        self.assertFalse(bool(by_driver.loc['pres_gs', 'clears_control']))

    def test_a_gain_whose_interval_spans_zero_is_not_significant(self):
        result = coupling.shortlist(self._table(), control='batt_str')
        by_driver = result.set_index('driver')
        self.assertTrue(bool(by_driver.loc['tair_str', 'gain_significant']))
        self.assertFalse(bool(by_driver.loc['pres_gs', 'gain_significant']))

    def test_the_expected_sign_at_this_site_is_negative(self):
        # The wall stands in an embankment whose valley face is the more
        # exposed, so daytime heating tips it towards the mountain, which is a
        # negative change in inclination. Stated in docs/raw-data-format.md
        # section 7.5; asserted here so that a change of default has to break a
        # test rather than pass unnoticed.
        result = coupling.shortlist(self._table(), control='batt_str')
        by_driver = result.set_index('driver')
        self.assertTrue(bool(by_driver.loc['tair_str', 'expected_sign']))
        self.assertFalse(bool(by_driver.loc['pres_gs', 'expected_sign']))

        flipped = coupling.shortlist(self._table(), control='batt_str',
                                     expected_sign='+')
        self.assertFalse(bool(flipped.set_index('driver')
                              .loc['tair_str', 'expected_sign']))


class TestGainStability(unittest.TestCase):
    """A gain re-fitted over windows, at a lag held fixed."""

    def test_a_constant_coupling_holds_its_slope(self):
        hours = 24 * 120
        index = pd.date_range('2025-01-01', periods=hours, freq='1h', tz='UTC')
        driver = pd.Series(_diurnal(hours, amplitude=5.0), index=index)
        frame = pd.DataFrame({'inc_comp_cleaned': -2.0 * driver.shift(2),
                              'tair_str': driver})
        stability = coupling.gain_stability(frame, 'inc_comp_cleaned',
                                            ['tair_str'], {'tair_str': 2})
        self.assertGreaterEqual(len(stability), 3)
        self.assertTrue(np.allclose(stability['slope'], -2.0, atol=1e-6))

    def test_a_driver_without_an_optimum_is_skipped(self):
        index = pd.date_range('2025-01-01', periods=100, freq='1h', tz='UTC')
        frame = pd.DataFrame({'inc_comp_cleaned': np.arange(100.0),
                              'tair_str': np.arange(100.0)}, index=index)
        stability = coupling.gain_stability(frame, 'inc_comp_cleaned',
                                            ['tair_str'], {'tair_str': np.nan})
        self.assertTrue(stability.empty)


class TestWindComponents(unittest.TestCase):
    """A bearing is not a number a lag scan may be run on."""

    def test_the_components_recover_the_direction(self):
        direction = pd.Series([0.0, 90.0, 180.0, 270.0, 359.0],
                              index=pd.date_range('2025-01-01', periods=5,
                                                  freq='1h', tz='UTC'),
                              name='wdir')
        components = meteo.wind_components(direction)
        recovered = np.degrees(np.arctan2(components['wdir_sin'],
                                          components['wdir_cos'])) % 360.0
        self.assertTrue(np.allclose(recovered.to_numpy(),
                                    direction.to_numpy(), atol=1e-9))

    def test_the_components_do_not_break_at_the_wrap(self):
        # 350 degrees and 10 degrees are twenty degrees apart, and their
        # components average to due north. The bearings themselves average to
        # 180, pointing exactly backwards, which is why the scan never sees a
        # bearing.
        direction = pd.Series([350.0, 10.0],
                              index=pd.date_range('2025-01-01', periods=2,
                                                  freq='1h', tz='UTC'),
                              name='wdir')
        components = meteo.wind_components(direction)
        mean_angle = np.degrees(np.arctan2(components['wdir_sin'].mean(),
                                           components['wdir_cos'].mean())) % 360.0
        # Compared the way a bearing has to be compared: due north can arrive
        # as 0 or as 360 depending on the last bit of a float, and only the
        # difference along the shorter arc is the same number either way.
        self.assertAlmostEqual(
            float(meteo.circular_difference(mean_angle, 0.0)), 0.0, places=9)
        self.assertAlmostEqual(float(np.mean([350.0, 10.0])), 180.0)

    def test_a_speed_scales_the_components(self):
        index = pd.date_range('2025-01-01', periods=2, freq='1h', tz='UTC')
        direction = pd.Series([90.0, 90.0], index=index, name='wdir')
        speed = pd.Series([2.0, 4.0], index=index)
        components = meteo.wind_components(direction, speed=speed)
        self.assertTrue(np.allclose(components['wdir_sin'].to_numpy(),
                                    [2.0, 4.0]))


class TestResponseLoading(unittest.TestCase):
    """Study 1's verdicts are consumed here, never revisited."""

    def _archive(self, directory):
        index = pd.date_range('2025-03-01 00:00', periods=9, freq='20min')
        frame = pd.DataFrame({
            'inc_comp_cleaned': np.arange(9.0),
            'inc_spike': [False] * 3 + [True] * 3 + [False] * 3,
        }, index=index)
        frame.index.name = 'datetime'
        path = os.path.join(directory, 'archive.csv')
        frame.to_csv(path)
        return path

    def test_interpolated_samples_are_removed_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._archive(directory)
            response, provenance = proxies.load_response(path)
            self.assertEqual(provenance['interpolated_samples_removed'], 3)
            # The middle hour was made entirely of interpolated samples, so it
            # carries no hourly mean at all rather than a mean of nothing.
            self.assertTrue(pd.isna(response.iloc[1]))
            self.assertFalse(pd.isna(response.iloc[0]))

    def test_the_verdict_can_be_set_aside_as_a_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._archive(directory)
            response, provenance = proxies.load_response(path,
                                                         honour_spike=False)
            self.assertEqual(provenance['interpolated_samples_removed'], 0)
            self.assertFalse(pd.isna(response.iloc[1]))

    def test_the_index_is_converted_off_the_civil_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._archive(directory)
            response, provenance = proxies.load_response(path)
            # Naive UTC, the grid the external sources are already on, so that
            # a response and a proxy on the same index mean the same instant.
            self.assertIsNone(response.index.tz)
            self.assertEqual(provenance['clock'], site.SITE_TZ)
            # The archive was written at 00:00 Italian civil time on 1 March,
            # which is winter time here and so 23:00 UTC the day before. An
            # hour that did not move would mean the civil clock had been read
            # as though it were UTC.
            self.assertEqual(response.index[0],
                             pd.Timestamp('2025-02-28 23:00'))

    def test_an_absent_response_column_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._archive(directory)
            with self.assertRaises(KeyError):
                proxies.load_response(path, column='inc_comp_joined')


class TestJoinEras(unittest.TestCase):
    """One channel per quantity, from the block that was recording."""

    def test_the_first_frame_wins_where_both_recorded(self):
        index = pd.date_range('2025-01-01', periods=3, freq='1h', tz='UTC')
        current = pd.DataFrame({'rh_str': [1.0, np.nan, 3.0]}, index=index)
        legacy = pd.DataFrame({'rh_str': [9.0, 9.0, 9.0]}, index=index)
        joined = proxies.join_eras([current, legacy])
        self.assertTrue(np.allclose(joined['rh_str'].to_numpy(),
                                    [1.0, 9.0, 3.0]))

    def test_columns_absent_from_one_frame_survive(self):
        index = pd.date_range('2025-01-01', periods=2, freq='1h', tz='UTC')
        current = pd.DataFrame({'sr_str': [1.0, 2.0]}, index=index)
        legacy = pd.DataFrame({'rh_str': [3.0, 4.0]}, index=index)
        joined = proxies.join_eras([current, legacy])
        self.assertEqual(sorted(joined.columns), ['rh_str', 'sr_str'])


class TestClockCheckRegression(unittest.TestCase):
    """
    Study 2's clock result must not move when the search moves to ``shmlib``.

    The loop below is the one :func:`shmlib.quality.clock_check` carried before
    study 3 extracted it. It is kept here, in the study that moved it, so that
    the extraction is checked against the code it replaced rather than against
    a memory of what that code did.
    """

    @staticmethod
    def _former_search(target, reference, max_lag_hours):
        best = None
        target = pd.to_numeric(target, errors='coerce')
        for candidate in range(-max_lag_hours, max_lag_hours + 1):
            paired = pd.concat([target.shift(candidate), reference],
                               axis=1).dropna()
            if len(paired) < 24:
                continue
            score = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
            if best is None or score > best[1]:
                best = (candidate, score)
        return (np.nan, np.nan) if best is None else (float(best[0]), best[1])

    def test_the_shared_search_reproduces_the_former_loop(self):
        rng = np.random.default_rng(1234)
        hours = 24 * 40
        reference = _hourly(_diurnal(hours, amplitude=300.0)
                            + rng.normal(scale=20.0, size=hours))
        for true_lag in (-3, 0, 2, 7):
            target = reference.shift(-true_lag)
            former = self._former_search(target, reference, 12)
            curve = coupling.lag_scan(reference, target, range(-12, 13),
                                      min_paired=24)
            shared = coupling.best_lag(curve, objective='signed')
            self.assertEqual(shared['lag'], former[0])
            self.assertAlmostEqual(shared['r'], former[1], places=12)

    def test_clock_check_still_answers_on_a_synthetic_pair(self):
        hours = 24 * 40
        index = pd.date_range('2025-05-01', periods=hours, freq='1h', tz='UTC')
        radiation = np.clip(_diurnal(hours, amplitude=400.0, phase=6.0), 0.0, None)
        frame = pd.DataFrame({'sr_era5': radiation,
                              'sr_gs': np.roll(radiation, 2)}, index=index)
        result = quality.clock_check(frame, 'sr', sources=('era5', 'gs'),
                                     reference='era5')
        by_source = result.set_index('source')
        self.assertEqual(by_source.loc['era5', 'xcorr_lag_h'], 0.0)
        self.assertEqual(by_source.loc['gs', 'xcorr_lag_h'], -2.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
