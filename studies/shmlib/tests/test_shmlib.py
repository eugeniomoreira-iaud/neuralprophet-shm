"""
Tests for the shared library.

These cover the properties every study depends on and that no study checks for
itself: that the parsers accept exactly the records of their own era, that the
mixed decimal separator is normalised per field, that the compensation matches
the documented formula, and that the solar geometry places the sun where the
almanac says it is.

Run directly, with no test runner installed::

    python studies/shmlib/tests/test_shmlib.py
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '01_data_exploration')))

from shmlib import adc, figures, site, solar, meteo, tables, viz  # noqa: E402
import de_lib                                                   # noqa: E402


def write_adc(lines):
    """Write lines to a temporary .adc file and return its path."""
    fh = tempfile.NamedTemporaryFile('w', suffix='.adc', delete=False)
    fh.write('\n'.join(lines) + '\n')
    fh.close()
    return fh.name


CURRENT_RECORD = '\t'.join(
    ['21/02/25', '10:20:00']
    + ['0.000'] * 12
    + ['12,600', '7.500', '61.200', '2503.125', '180.500', '9.400'])

LEGACY_RECORD = '\t'.join(
    ['20/02/25', '10:20:00']
    + ['12.700', '6,300', '58.100', '2498.750']
    + ['0.000'] * 4
    + ['12.100', '5.900', '60.000', '2501.250'])


class TestParsers(unittest.TestCase):
    """The two era parsers, and the boundary between them."""

    def test_current_parser_reads_its_own_era(self):
        path = write_adc([CURRENT_RECORD])
        df = adc.parse_file(path)
        os.unlink(path)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), adc.CURRENT_COLUMNS)
        self.assertEqual(df.index[0], pd.Timestamp('2025-02-21 10:20:00'))
        self.assertAlmostEqual(df['sr'].iloc[0], 180.5)
        self.assertAlmostEqual(df['twall'].iloc[0], 9.4)

    def test_legacy_parser_reads_its_own_era(self):
        path = write_adc([LEGACY_RECORD])
        df = adc.parse_legacy_file(path)
        os.unlink(path)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), adc.LEGACY_COLUMNS)
        self.assertAlmostEqual(df['st01_batt'].iloc[0], 12.7)
        self.assertAlmostEqual(df['st03_i'].iloc[0], 2501.25)

    def test_each_parser_ignores_the_other_era(self):
        """
        A file spanning the changeover carries both layouts. Each parser must
        take its own records and leave the rest, because the era of a record is
        decided by its own field count and never by the date of its file.
        """
        path = write_adc([LEGACY_RECORD, CURRENT_RECORD, LEGACY_RECORD])
        current = adc.parse_file(path)
        legacy = adc.parse_legacy_file(path)
        os.unlink(path)
        self.assertEqual(len(current), 1)
        self.assertEqual(len(legacy), 2)

    def test_empty_result_still_has_its_columns(self):
        path = write_adc([LEGACY_RECORD])
        df = adc.parse_file(path)
        os.unlink(path)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), adc.CURRENT_COLUMNS)

    def test_malformed_records_are_skipped_not_fatal(self):
        path = write_adc(['nonsense', CURRENT_RECORD, 'also\tnonsense'])
        df = adc.parse_file(path)
        os.unlink(path)
        self.assertEqual(len(df), 1)

    def test_unparseable_timestamp_is_skipped(self):
        broken = CURRENT_RECORD.replace('21/02/25', '99/99/99', 1)
        path = write_adc([broken, CURRENT_RECORD])
        df = adc.parse_file(path)
        os.unlink(path)
        self.assertEqual(len(df), 1)


class TestDecimalSeparator(unittest.TestCase):
    """The separator is mixed, sometimes within a single record."""

    def test_both_separators_parse_to_the_same_number(self):
        self.assertEqual(adc.to_float('3.500'), adc.to_float('3,500'))

    def test_non_numeric_becomes_nan(self):
        self.assertTrue(np.isnan(adc.to_float('')))
        self.assertTrue(np.isnan(adc.to_float('---')))
        self.assertTrue(np.isnan(adc.to_float(None)))

    def test_mixed_separators_within_one_record(self):
        """`CURRENT_RECORD` writes batt with a comma and tair with a point."""
        path = write_adc([CURRENT_RECORD])
        df = adc.parse_file(path)
        os.unlink(path)
        self.assertAlmostEqual(df['batt'].iloc[0], 12.6)
        self.assertAlmostEqual(df['tair'].iloc[0], 7.5)


class TestCompensation(unittest.TestCase):
    """The documented formula, and the anchoring it depends on."""

    def setUp(self):
        idx = pd.date_range('2025-03-01', periods=4, freq='20min')
        self.df = pd.DataFrame(
            {'inc': [100.0, 100.0, 100.0, 100.0],
             'tair': [10.0, 11.0, 12.0, 10.0]}, index=idx)

    def test_five_mdeg_per_degree(self):
        """The coefficient is 0.005, applied times 1000: 5 mdeg per °C."""
        out = adc.compensate(self.df, normalise=False)
        self.assertAlmostEqual(out.iloc[0], 100.0)
        self.assertAlmostEqual(out.iloc[1], 95.0)
        self.assertAlmostEqual(out.iloc[2], 90.0)

    def test_normalise_starts_the_series_at_zero(self):
        out = adc.compensate(self.df)
        self.assertAlmostEqual(out.iloc[0], 0.0)
        self.assertAlmostEqual(out.iloc[2], -10.0)

    def test_reference_temperature_is_the_first_complete_record(self):
        df = self.df.copy()
        df.loc[df.index[0], 'inc'] = np.nan
        out = adc.compensate(df, normalise=False)
        # T_ref is now 11.0, the temperature of the first record carrying both.
        self.assertAlmostEqual(out.iloc[1], 100.0)

    def test_no_complete_record_yields_all_nan(self):
        df = self.df.copy()
        df['inc'] = np.nan
        out = adc.compensate(df)
        self.assertTrue(out.isna().all())
        self.assertEqual(len(out), len(df))


class TestSolar(unittest.TestCase):
    """Solar geometry, against quantities an almanac can confirm."""

    def test_solar_noon_is_near_eleven_utc_at_this_longitude(self):
        noon = solar.solar_noon_utc(pd.date_range('2025-06-21', periods=1))
        self.assertTrue(10.5 < noon.iloc[0] < 11.5)

    def test_elevation_peaks_at_solar_noon(self):
        day = pd.date_range('2025-06-21', periods=24 * 6, freq='10min')
        elevation = solar.solar_elevation(day)
        noon = solar.solar_noon_utc(pd.DatetimeIndex(['2025-06-21'])).iloc[0]
        peak = elevation.idxmax()
        self.assertLess(abs(peak.hour + peak.minute / 60.0 - noon), 0.25)

    def test_summer_noon_is_higher_than_winter_noon(self):
        summer = solar.solar_elevation(
            pd.DatetimeIndex(['2025-06-21 11:00'])).iloc[0]
        winter = solar.solar_elevation(
            pd.DatetimeIndex(['2025-12-21 11:00'])).iloc[0]
        self.assertGreater(summer, winter)
        # At 43.35 degrees north the solstice elevations are about 70 and 23.
        self.assertTrue(65 < summer < 75)
        self.assertTrue(18 < winter < 28)

    def test_sun_is_below_the_horizon_at_local_midnight(self):
        self.assertLess(
            solar.solar_elevation(pd.DatetimeIndex(['2025-06-21 23:00'])).iloc[0],
            0.0)

    def test_daylight_mask_follows_the_elevation(self):
        day = pd.date_range('2025-06-21', periods=24, freq='1h')
        mask = solar.daylight_mask(day)
        elevation = solar.solar_elevation(day)
        self.assertTrue(((elevation > 0) == mask).all())

    def test_elevation_never_leaves_its_range(self):
        year = pd.date_range('2025-01-01', '2026-01-01', freq='1h')
        elevation = solar.solar_elevation(year)
        self.assertTrue((elevation >= -90).all() and (elevation <= 90).all())


class TestMeteo(unittest.TestCase):
    """Circular statistics, against the arithmetic-mean answer they must avoid."""

    def test_mean_of_350_and_10_is_zero_not_180(self):
        self.assertAlmostEqual(meteo.circular_mean([350, 10]), 0.0, places=6)

    def test_mean_ignores_nan(self):
        self.assertAlmostEqual(
            meteo.circular_mean([350, np.nan, 10]), 0.0, places=6)

    def test_mean_of_empty_or_all_nan_is_nan(self):
        self.assertTrue(np.isnan(meteo.circular_mean([])))
        self.assertTrue(np.isnan(meteo.circular_mean([np.nan, np.nan])))

    def test_difference_is_signed_and_shortest_way_around(self):
        self.assertAlmostEqual(meteo.circular_difference(10, 350), 20.0)
        self.assertAlmostEqual(meteo.circular_difference(350, 10), -20.0)

    def test_difference_wraps_to_the_half_open_interval(self):
        # (-180, 180]: the boundary itself is +180, never -180.
        self.assertAlmostEqual(meteo.circular_difference(180, 0), 180.0)

    def test_resample_honours_min_count(self):
        idx = pd.date_range('2025-01-01 00:00', periods=4, freq='1h')
        series = pd.Series([350.0, 10.0, np.nan, np.nan], index=idx)
        # The first hourly bin (00:00-01:00 on a 1-hour grid resampled to
        # 2 hours) holds both readings; the second holds none.
        out = meteo.circular_resample(series, '2h', min_count=2)
        self.assertAlmostEqual(out.iloc[0], 0.0, places=6)
        self.assertTrue(np.isnan(out.iloc[1]))

    def test_resample_produces_the_expected_direction(self):
        idx = pd.date_range('2025-01-01 00:00', periods=3, freq='20min')
        series = pd.Series([350.0, 10.0, 30.0], index=idx)
        out = meteo.circular_resample(series, '1h', min_count=1)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.iloc[0], meteo.circular_mean([350, 10, 30]),
                               places=6)

    def test_std_is_near_zero_for_a_constant_direction(self):
        std = meteo.circular_std([90.0, 90.0, 90.0, 90.0])
        self.assertLess(std, 1e-6)

    def test_std_is_large_for_a_uniform_spread(self):
        std = meteo.circular_std([0.0, 90.0, 180.0, 270.0])
        self.assertGreater(std, 100.0)

    def test_std_of_empty_or_all_nan_is_nan(self):
        self.assertTrue(np.isnan(meteo.circular_std([])))
        self.assertTrue(np.isnan(meteo.circular_std([np.nan, np.nan])))


class TestSite(unittest.TestCase):
    """Deployment facts, and the archive-extent helper."""

    def test_eras_do_not_overlap_and_do_not_gap(self):
        self.assertEqual(
            pd.Timestamp(site.LEGACY_END) + pd.Timedelta(days=1),
            pd.Timestamp(site.CURRENT_START))

    def test_changeover_agrees_with_the_era_boundary(self):
        self.assertEqual(adc.CHANGEOVER, pd.Timestamp(site.CURRENT_START))

    def test_target_station_is_one_of_the_stations(self):
        self.assertIn(site.TARGET_STATION, site.STATIONS)

    def test_last_archive_day_reads_the_newest_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('GUBBIO_20250220.adc', 'GUBBIO_20250221.adc',
                         'notes.txt'):
                open(os.path.join(tmp, name), 'w').close()
            self.assertEqual(site.last_archive_day(tmp), '2025-02-21')

    def test_last_archive_day_raises_on_an_empty_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                site.last_archive_day(tmp)


class TestViz(unittest.TestCase):
    """The figure conventions that the Graphical Guidelines make binding."""

    def test_every_channel_has_a_distinct_identity_colour(self):
        base = ['inc_comp', 'tair', 'rh', 'batt', 'twall', 'sr']
        colours = [viz.CHANNEL_COLOUR[c] for c in base]
        self.assertEqual(len(set(colours)), len(base))

    def test_aliases_of_a_quantity_share_its_colour(self):
        self.assertEqual(viz.CHANNEL_COLOUR['inc_comp_cleaned'],
                         viz.CHANNEL_COLOUR['inc_comp'])
        self.assertEqual(viz.CHANNEL_COLOUR['twall_filtered'],
                         viz.CHANNEL_COLOUR['twall'])

    def test_accent_is_never_a_channel_colour(self):
        self.assertNotIn(viz.MARK_COLOUR, viz.CHANNEL_COLOUR.values())

    def test_span_highlight_is_black_at_five_per_cent_with_no_edge(self):
        self.assertEqual(viz.SPAN_STYLE['color'], '#000000')
        self.assertEqual(viz.SPAN_STYLE['alpha'], 0.05)
        self.assertEqual(viz.SPAN_STYLE['lw'], 0)

    def test_unknown_channel_raises_rather_than_defaulting(self):
        with self.assertRaises(KeyError):
            viz.channel_style('not_a_channel', 1.0)

    def test_figsize_scales_with_the_active_context(self):
        viz.set_context('notebook')
        self.assertEqual(viz.figsize(6.0, 4.0), (6.0, 4.0))
        viz.set_context('paper')
        scaled = viz.figsize(6.0, 4.0)
        self.assertLess(scaled[0], 6.0)
        viz.set_context('notebook')

    def test_finish_without_a_destination_writes_nothing(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            viz.finish(fig)
            plt.close(fig)
            self.assertEqual(os.listdir(tmp), [])

    def test_de_lib_diurnal_cycle_helpers_are_aliases_of_viz(self):
        """
        Study 1's `_complete_days` and `_draw_cycle` moved here as
        `complete_days` and `draw_cycle`; this pins the alias down so a future
        edit to either name cannot silently break the other without a test
        noticing.
        """
        self.assertIs(de_lib._complete_days, viz.complete_days)
        self.assertIs(de_lib._draw_cycle, viz.draw_cycle)


class TestDrawCycle(unittest.TestCase):
    """`viz.draw_cycle`, the single diurnal panel every study draws with."""

    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt

    def test_a_short_day_is_dropped_from_the_mean_and_from_n_days(self):
        """
        With `complete_day` set, a day short of that many slots contributes
        neither to the mean cycle nor to the day count: only the one full day
        below should reach the panel.
        """
        idx = pd.DatetimeIndex(
            ['2025-03-01 00:00', '2025-03-01 08:00', '2025-03-01 16:00',
             '2025-03-02 00:00', '2025-03-02 08:00', '2025-03-02 16:00'])
        data = pd.DataFrame(
            {'inc': [10.0, 20.0, 30.0, 11.0, 21.0, np.nan]}, index=idx)
        fig, ax = self.plt.subplots()
        cycle, amplitude, n_days = viz.draw_cycle(
            ax, data, 'inc', '#0072B2', complete_day=3, centre=False,
            background=False)
        self.plt.close(fig)
        self.assertEqual(n_days, 1)
        self.assertAlmostEqual(cycle[0.0], 10.0)
        self.assertAlmostEqual(cycle[8.0], 20.0)
        self.assertAlmostEqual(cycle[16.0], 30.0)

    def test_centring_zeroes_the_cycles_own_mean_but_not_its_amplitude(self):
        """
        `centre=True` returns a cycle whose own mean is zero and
        `centre=False` returns it in the channel's own units; the amplitude,
        a peak-to-trough distance, must not depend on which is asked for.
        """
        idx = pd.DatetimeIndex(
            ['2025-03-01 00:00', '2025-03-01 08:00', '2025-03-01 16:00',
             '2025-03-02 00:00', '2025-03-02 08:00', '2025-03-02 16:00'])
        data = pd.DataFrame(
            {'inc': [10.0, 20.0, 30.0, 110.0, 120.0, 130.0]}, index=idx)

        fig, ax = self.plt.subplots()
        raw, raw_amp, _ = viz.draw_cycle(
            ax, data, 'inc', '#0072B2', centre=False, background=False)
        self.plt.close(fig)

        fig, ax = self.plt.subplots()
        centred, centred_amp, _ = viz.draw_cycle(
            ax, data, 'inc', '#0072B2', centre=True, background=False)
        self.plt.close(fig)

        self.assertAlmostEqual(raw.mean(), 70.0)
        self.assertAlmostEqual(centred.mean(), 0.0, places=9)
        self.assertAlmostEqual(raw_amp, centred_amp)
        self.assertAlmostEqual(raw_amp, 20.0)

    def test_min_days_empties_a_position_too_few_days_carry(self):
        """
        A position within the day is left `NaN` when fewer than `min_days`
        distinct days carry it, while a position at or above the threshold is
        kept.
        """
        idx = pd.DatetimeIndex(
            ['2025-03-01 00:00', '2025-03-01 08:00',
             '2025-03-02 00:00', '2025-03-02 08:00',
             '2025-03-03 00:00'])
        data = pd.DataFrame(
            {'inc': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        fig, ax = self.plt.subplots()
        cycle, amplitude, n_days = viz.draw_cycle(
            ax, data, 'inc', '#0072B2', min_days=3, centre=False,
            background=False)
        self.plt.close(fig)
        self.assertAlmostEqual(cycle[0.0], 3.0)
        self.assertTrue(np.isnan(cycle[8.0]))

    def test_circular_matches_meteo_and_returns_no_amplitude_or_background(self):
        """
        `circular=True` averages by the unit-vector method, matching
        `meteo.circular_mean` on the same slot's values; the peak-to-trough
        amplitude is undefined for a circular quantity and comes back `NaN`;
        and no gray background day is drawn, even when `background=True` is
        passed, because a wrapping direction drawn as a connected line lies
        about the data.
        """
        idx = pd.DatetimeIndex(
            ['2025-03-01 00:00', '2025-03-01 08:00',
             '2025-03-02 00:00', '2025-03-02 08:00'])
        data = pd.DataFrame(
            {'wd': [350.0, 170.0, 10.0, 190.0]}, index=idx)
        fig, ax = self.plt.subplots()
        cycle, amplitude, n_days = viz.draw_cycle(
            ax, data, 'wd', '#CC79A7', circular=True, background=True)
        line_count = len(ax.lines)
        self.plt.close(fig)
        self.assertAlmostEqual(
            cycle[0.0], meteo.circular_mean([350.0, 10.0]), places=6)
        self.assertAlmostEqual(
            cycle[8.0], meteo.circular_mean([170.0, 190.0]), places=6)
        self.assertTrue(np.isnan(amplitude))
        self.assertEqual(n_days, 2)
        self.assertEqual(line_count, 1)

    def test_background_toggles_the_individual_day_lines(self):
        """
        `background=True` draws every individual day behind the mean and
        `background=False` draws none of them; the mean cycle itself is
        drawn either way. Individual-day and mean lines are told apart by
        counting the `Line2D` artists a real Agg-backed axes ends up holding.
        """
        idx = pd.DatetimeIndex(
            ['2025-03-01 00:00', '2025-03-01 08:00', '2025-03-01 16:00',
             '2025-03-02 00:00', '2025-03-02 08:00', '2025-03-02 16:00'])
        data = pd.DataFrame(
            {'inc': [10.0, 20.0, 30.0, 11.0, 21.0, 31.0]}, index=idx)

        fig, ax = self.plt.subplots()
        viz.draw_cycle(ax, data, 'inc', '#0072B2', background=True)
        with_background = len(ax.lines)
        self.plt.close(fig)

        fig, ax = self.plt.subplots()
        viz.draw_cycle(ax, data, 'inc', '#0072B2', background=False)
        without_background = len(ax.lines)
        self.plt.close(fig)

        self.assertEqual(with_background, 3)
        self.assertEqual(without_background, 1)


class TestPlotDiurnalSeasonGrid(unittest.TestCase):
    """`figures.plot_diurnal_season_grid`, the channel-by-season diurnal grid."""

    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
        self.season_months = {'summer': [6, 7, 8], 'winter': [12, 1, 2]}

    def _frame(self):
        """Five hourly days each of summer and winter, on two channels of very different magnitude."""
        summer = pd.date_range('2025-06-01', '2025-06-05 23:00', freq='1h')
        winter = pd.date_range('2025-12-01', '2025-12-05 23:00', freq='1h')
        idx = summer.append(winter)
        hour = idx.hour.values.astype(float)
        frame = pd.DataFrame(index=idx)
        frame['ch1_src'] = 5.0 + 1.0 * np.sin(2 * np.pi * hour / 24.0)
        frame['ch2_src'] = 500.0 + 200.0 * np.sin(2 * np.pi * hour / 24.0)
        return frame

    def _circular_frame(self):
        """A temperature channel and a wind-direction channel, five hourly days each season."""
        summer = pd.date_range('2025-06-01', '2025-06-05 23:00', freq='1h')
        winter = pd.date_range('2025-12-01', '2025-12-05 23:00', freq='1h')
        idx = summer.append(winter)
        hour = idx.hour.values
        frame = pd.DataFrame(index=idx)
        frame['tair_src'] = 20.0 + 5.0 * np.sin(2 * np.pi * hour.astype(float) / 24.0)
        frame['wdir_src'] = np.where(hour % 2 == 0, 350.0, 10.0)
        return frame

    def test_the_grid_has_one_axes_per_channel_and_season_in_order(self):
        """
        The returned figure carries exactly `len(columns) * len(seasons)`
        axes, laid out one row per channel and one column per season, both in
        the order the caller gave them.
        """
        colours = {'ch1': '#0072B2', 'ch2': '#E69F00'}
        labels = {'ch1': 'Ch1', 'ch2': 'Ch2'}
        fig = figures.plot_diurnal_season_grid(
            self._frame(), ['ch1_src', 'ch2_src'], ['summer', 'winter'],
            self.season_months, 'Test grid', colours=colours, labels=labels,
            circular=())
        axes = fig.get_axes()
        positions = [(ax.get_subplotspec().rowspan.start,
                      ax.get_subplotspec().colspan.start) for ax in axes]
        self.plt.close(fig)
        self.assertEqual(len(axes), 4)
        self.assertEqual(positions, [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_a_row_shares_its_vertical_scale_but_rows_do_not_match(self):
        """
        The two seasonal panels of one row must return the same `get_ylim()`,
        since a shared vertical scale within a row is the entire reason the
        two seasons are drawn in one figure rather than two; the two rows
        here carry channels of very different magnitude, so their y limits
        must not agree with each other.
        """
        colours = {'ch1': '#0072B2', 'ch2': '#E69F00'}
        labels = {'ch1': 'Ch1', 'ch2': 'Ch2'}
        fig = figures.plot_diurnal_season_grid(
            self._frame(), ['ch1_src', 'ch2_src'], ['summer', 'winter'],
            self.season_months, 'Test grid', colours=colours, labels=labels,
            circular=())
        axes = np.array(fig.get_axes()).reshape(2, 2)
        top_left, top_right = axes[0][0].get_ylim(), axes[0][1].get_ylim()
        bottom_left, bottom_right = axes[1][0].get_ylim(), axes[1][1].get_ylim()
        self.plt.close(fig)
        self.assertEqual(top_left, top_right)
        self.assertEqual(bottom_left, bottom_right)
        self.assertNotEqual(top_left, bottom_left)

    def test_the_season_names_only_the_top_row_and_every_title_names_its_days(self):
        """
        The season name appears once, capitalised, at the start of the
        top-row panel titles, and is not repeated on the panels beneath them;
        every panel's title, top row included, still states the number of
        days its own mean rests on.
        """
        colours = {'ch1': '#0072B2', 'ch2': '#E69F00'}
        labels = {'ch1': 'Ch1', 'ch2': 'Ch2'}
        fig = figures.plot_diurnal_season_grid(
            self._frame(), ['ch1_src', 'ch2_src'], ['summer', 'winter'],
            self.season_months, 'Test grid', colours=colours, labels=labels,
            circular=())
        axes = np.array(fig.get_axes()).reshape(2, 2)
        titles = [[ax.get_title() for ax in row] for row in axes]
        self.plt.close(fig)
        self.assertTrue(titles[0][0].startswith('Summer'))
        self.assertTrue(titles[0][1].startswith('Winter'))
        self.assertFalse(titles[1][0].startswith('Summer'))
        self.assertFalse(titles[1][1].startswith('Winter'))
        for row in titles:
            for title in row:
                self.assertIn('5 days', title)

    def test_a_circular_row_draws_a_compass_axis_and_no_amplitude(self):
        """
        The row of a channel declared circular is drawn on a 0-360 degree
        compass axis at the compass bearings, and its panel titles report no
        amplitude, since the peak-to-trough distance of a direction depends on
        which way round the circle it is measured; the non-circular row above
        it does report an amplitude.

        The tick *labels* are asserted on the leftmost panel alone. Every panel
        of a row shares one y axis, so matplotlib draws that axis once, at the
        outer edge of the row, and the inner panels carry no tick labels of
        their own to read back — which is the intended layout, not a defect in
        it. The bearings the ticks sit at are shared and are checked across the
        whole row.
        """
        fig = figures.plot_diurnal_season_grid(
            self._circular_frame(), ['tair_src', 'wdir_src'],
            ['summer', 'winter'], self.season_months, 'Test grid')
        axes = np.array(fig.get_axes()).reshape(2, 2)
        wdir_ylims = [ax.get_ylim() for ax in axes[1]]
        wdir_ticks = [list(ax.get_yticks()) for ax in axes[1]]
        wdir_ticklabels = [text.get_text()
                           for text in axes[1][0].get_yticklabels()]
        titles = [[ax.get_title() for ax in row] for row in axes]
        self.plt.close(fig)
        for ylim in wdir_ylims:
            self.assertEqual(ylim, (0.0, 360.0))
        for ticks in wdir_ticks:
            self.assertEqual(ticks, list(viz.COMPASS_TICKS.keys()))
        self.assertEqual(wdir_ticklabels, list(viz.COMPASS_TICKS.values()))
        for title in titles[1]:
            self.assertNotIn('amplitude', title)
        for title in titles[0]:
            self.assertIn('amplitude', title)


class TestPlotChannelPanels(unittest.TestCase):
    """`figures.plot_channel_panels`, the channel-named resolver over `plot_source_panels`."""

    def setUp(self):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt

    def _frame(self):
        """Three hourly days on the inclination and two source-suffixed channels."""
        idx = pd.date_range('2025-03-01', periods=72, freq='1h')
        frame = pd.DataFrame(index=idx)
        frame['inc_comp_cleaned'] = 100.0
        frame['tair_str'] = 10.0
        frame['twall_str'] = 9.0
        return frame

    def test_each_column_draws_its_own_channels_identity_colour(self):
        """
        `'inc_comp_cleaned'` already names its channel and needs no suffix
        stripped; `'tair_str'` and `'twall_str'` each carry a source suffix
        that the default resolution strips before it is looked up. All three
        columns reach the figure as three panels, each drawn in its own
        channel's `viz.CHANNEL_COLOUR`.
        """
        fig = figures.plot_channel_panels(
            self._frame(), ['inc_comp_cleaned', 'tair_str', 'twall_str'],
            'Test panels')
        axes = fig.get_axes()
        drawn = [ax.lines[0].get_color() for ax in axes]
        self.plt.close(fig)
        self.assertEqual(len(axes), 3)
        self.assertEqual(drawn, [viz.CHANNEL_COLOUR['inc_comp_cleaned'],
                                 viz.CHANNEL_COLOUR['tair'],
                                 viz.CHANNEL_COLOUR['twall']])

    def test_a_column_resolving_to_an_unknown_channel_raises(self):
        """A column naming no channel `viz.CHANNEL_LABEL` knows about must fail loudly, not draw silently in a default colour."""
        frame = self._frame()
        frame['not_a_channel'] = 1.0
        with self.assertRaises(KeyError):
            figures.plot_channel_panels(
                frame, ['inc_comp_cleaned', 'not_a_channel'], 'Test panels')


class TestTables(unittest.TestCase):
    """The LaTeX row convention, and what a missing value prints as."""

    def setUp(self):
        self.frame = pd.DataFrame(
            {'name': ['inc_comp', 'n_sr'], 'value': [1.234, np.nan]},
            index=['a', 'b'])

    def test_rows_carry_no_separator_of_their_own(self):
        rows = tables.to_rows(self.frame, [('name', None), ('value', '.2f')])
        self.assertEqual(rows, ['inc_comp & 1.23', 'n_sr & ---'])
        self.assertFalse(any('\\\\' in row for row in rows))

    def test_body_separates_rows_and_never_terminates_the_last(self):
        """A trailing separator opens an empty row that \\bottomrule lands inside."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'body.tex')
            tables.write_table_body(['a & b', 'c & d'], path)
            body = open(path).read()
        self.assertEqual(body, 'a & b \\\\\nc & d\n')
        self.assertFalse(body.rstrip('\n').endswith('\\\\'))

    def test_single_row_body_has_no_separator_at_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'body.tex')
            tables.write_table_body(['only & row'], path)
            self.assertEqual(open(path).read(), 'only & row\n')

    def test_underscores_are_escaped(self):
        self.assertEqual(tables.latex_escape('inc_comp'), 'inc\\_comp')
        self.assertEqual(tables.texttt('n_sr'), '\\texttt{n\\_sr}')

    def test_a_callable_source_reads_the_index(self):
        rows = tables.to_rows(self.frame, [(lambda r: r.name, None)])
        self.assertEqual(rows, ['a', 'b'])

    def test_percent_escapes_the_sign(self):
        # A bare % opens a LaTeX comment and swallows the row terminator, so
        # the row below merges into this one and the table fails to compile.
        self.assertEqual(tables.percent(0.755429), '75.5\\%')
        self.assertEqual(tables.percent(0.755429, decimals=2), '75.54\\%')
        self.assertNotIn('%', tables.percent(1.0).replace('\\%', ''))

    def test_percent_renders_a_missing_value_as_the_marker(self):
        self.assertEqual(tables.percent(np.nan), tables.MISSING)
        self.assertEqual(tables.percent(None, missing='n/a'), 'n/a')

    def test_percent_survives_a_round_trip_through_a_table_body(self):
        frame = pd.DataFrame({'share': [0.755429, np.nan]})
        rows = tables.to_rows(frame, [('share', tables.percent)])
        self.assertEqual(rows, ['75.5\\%', tables.MISSING])

    def test_a_composed_cell_is_not_escaped(self):
        """Unit strings are markup, not data, and must reach LaTeX intact."""
        rows = tables.to_rows(
            self.frame,
            [(lambda r: f'{r["value"]:.1f} \\textdegree C'
              if pd.notna(r['value']) else None, None)])
        self.assertEqual(rows[0], '1.2 \\textdegree C')
        self.assertEqual(rows[1], tables.MISSING)

    def test_write_table_returns_the_path_it_wrote(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'body.tex')
            written = tables.write_table(self.frame, path, [('name', None)])
            self.assertEqual(written, path)
            self.assertTrue(os.path.exists(path))


class TestFiguresStudy05(unittest.TestCase):

    def test_plot_regressor_sets_draws_one_panel_per_record(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        index = pd.date_range('2024-01-01', periods=48, freq='1h', tz='UTC')
        frame = pd.DataFrame({
            'y': np.sin(np.arange(48) / 4.0), 'tair_str': 10.0, 'rh_str': 50.0,
            'sr_gs': 100.0, 'tair_gs': 11.0, 'rh_gs': 51.0, 'tair_era5': 9.0,
            'rh_era5': 49.0, 'sr_era5': 90.0}, index=index)
        sets = {'str': {'tair': 'tair_str', 'rh': 'rh_str', 'sr': 'sr_gs'},
                'gs': {'tair': 'tair_gs', 'rh': 'rh_gs', 'sr': 'sr_gs'},
                'era5': {'tair': 'tair_era5', 'rh': 'rh_era5', 'sr': 'sr_era5'}}
        fig = figures.plot_regressor_sets(frame, 'y', sets)
        # Three sets, with `sr_gs` shared by `str` and `gs`: 1 target panel
        # plus 8 distinct records (tair x3, rh x3, sr x2).
        self.assertEqual(len(fig.axes), 9)
        # The two `tair` panels are drawn from different columns but must
        # share their vertical scale, which pins the per-role shared axis.
        self.assertEqual(fig.axes[1].get_ylim(), fig.axes[2].get_ylim())
        plt.close(fig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
