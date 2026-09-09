"""
Unit tests for the decisions study 2 relies on, now that they live in
``shmlib``.

Run without a test runner, as the other studies' tests are run::

    python studies/02_proxy_forcing_characterization/tests/test_shmlib_study02.py

What is tested here is the set of choices this study's characterisation makes,
not the arithmetic of pandas. Four of them carry the study's conclusions and
would be silent if they broke:

* the conversion of the archive's civil-time index to UTC, including the two
  hours a year the civil clock steps (:func:`shmlib.site.to_utc`);
* the honouring of study 1's ``sr_suspect`` verdict, which must remove whole
  days and must be visible in the provenance the loader returns
  (:func:`shmlib.proxies.load_sensor_forcings`);
* the measurement of the certified radiation window as runs and gaps rather
  than as a single span (:func:`shmlib.quality.certified_sr_window`);
* the refusal of a verdict row that does not say what window its evidence
  covers (:func:`shmlib.quality.verdict_table`).

The remaining tests guard the circular handling of wind direction, which is the
one place where an ordinary arithmetic mean produces an answer that is not
merely imprecise but pointing the opposite way, and they exercise
:mod:`shmlib.compare` and :mod:`shmlib.proxies` alongside :mod:`shmlib.site`
and :mod:`shmlib.quality`, the four modules study 2's own dissolved private
library moved into.

Three further classes guard step 5, the native-grid harmonisation scan added
for this study: that :func:`shmlib.proxies.load_ground_station`'s
``stamp_offset`` moves the value read at one timestamp to exactly the earlier
timestamp requested and does nothing when left at its default; that
:func:`shmlib.temporal_alignment.reference_shift_scan` recovers, with the
documented sign, a residual reference-side displacement injected into a
synthetic pair; and that :func:`shmlib.temporal_alignment.shift_summary`'s
gain and sign arithmetic matches a summary worked out by hand.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.abspath(os.path.join(_HERE, '..'))
_STUDIES = os.path.abspath(os.path.join(_STUDY, '..'))
for path in (_STUDY, _STUDIES, os.path.join(_STUDIES, '01_data_exploration')):
    if path not in sys.path:
        sys.path.insert(0, path)

from shmlib import compare, proxies, quality, site, temporal_alignment  # noqa: E402


def _write_csv(frame, directory, name):
    """Write a frame to a temporary CSV and return its path."""
    path = os.path.join(directory, name)
    frame.to_csv(path)
    return path


class TestClockConversion(unittest.TestCase):
    """The archive is written on the site's civil clock; everything else is UTC."""

    def test_winter_timestamp_loses_one_hour(self):
        index = pd.DatetimeIndex(['2025-01-15 12:00:00'])
        self.assertEqual(site.to_utc(index)[0], pd.Timestamp('2025-01-15 11:00'))

    def test_summer_timestamp_loses_two_hours(self):
        # Central European Summer Time. A conversion that subtracted a fixed
        # hour would put every summer reading sixty minutes late, which is the
        # error this study exists to avoid making.
        index = pd.DatetimeIndex(['2025-07-15 12:00:00'])
        self.assertEqual(site.to_utc(index)[0], pd.Timestamp('2025-07-15 10:00'))

    def test_ambiguous_autumn_hour_becomes_missing(self):
        # On 26 October 2025 the civil clock falls back and 02:30 happens twice.
        # There is no way to tell which pass a record belongs to, so it is
        # dropped rather than assigned to one of them.
        index = pd.DatetimeIndex(['2025-10-26 02:30:00'])
        self.assertTrue(pd.isna(site.to_utc(index)[0]))

    def test_nonexistent_spring_hour_is_shifted_forward(self):
        # On 30 March 2025 the civil clock springs forward and 02:30 never
        # happens. A record stamped with it is shifted out of the gap.
        index = pd.DatetimeIndex(['2025-03-30 02:30:00'])
        self.assertFalse(pd.isna(site.to_utc(index)[0]))

    def test_conversion_is_monotonic_across_a_dst_boundary(self):
        index = pd.date_range('2025-03-29 22:00', '2025-03-30 06:00', freq='h')
        converted = site.to_utc(index)
        converted = converted[converted.notna()]
        self.assertTrue(converted.is_monotonic_increasing)


class TestSensorLoading(unittest.TestCase):
    """Study 1's verdicts are consumed, never revisited."""

    def setUp(self):
        import tempfile
        self.directory = tempfile.mkdtemp()

        index = pd.date_range('2025-06-01 00:00', '2025-06-03 23:40',
                              freq='20min')
        frame = pd.DataFrame(index=index)
        frame.index.name = 'datetime'
        frame['tair'] = 20.0
        frame['n_sr_ok'] = 100.0
        frame['n_sr_flag'] = ''
        # The middle day is the one study 1 condemned.
        frame['sr_suspect'] = (index.normalize()
                               == pd.Timestamp('2025-06-02'))
        # Six hours of the first day carry the substituted night zero.
        night = (index.normalize() == pd.Timestamp('2025-06-01')) & (index.hour < 2)
        frame.loc[night, 'n_sr_flag'] = quality.FLAG_NIGHT
        frame.loc[night, 'n_sr_ok'] = 0.0

        self.path = _write_csv(frame, self.directory, 'archive.csv')

    def test_suspect_days_are_removed_from_radiation(self):
        sensor, provenance = proxies.load_sensor_forcings(self.path,
                                                           honour_suspect=True)
        condemned = sensor.loc['2025-06-02 00:00':'2025-06-02 21:00', 'sr_str']
        self.assertTrue(condemned.isna().all())
        self.assertEqual(provenance['suspect_days'], 1)
        self.assertGreater(provenance['suspect_samples_removed'], 0)

    def test_suspect_days_survive_when_the_verdict_is_waived(self):
        # Waiving the verdict is a diagnostic and never a result, but it must
        # actually show the condemned days when it is asked to.
        sensor, provenance = proxies.load_sensor_forcings(self.path,
                                                           honour_suspect=False)
        condemned = sensor.loc['2025-06-02 00:00':'2025-06-02 21:00', 'sr_str']
        self.assertTrue(condemned.notna().any())
        self.assertEqual(provenance['suspect_samples_removed'], 0)
        self.assertFalse(provenance['honour_suspect'])

    def test_air_temperature_is_untouched_by_the_radiation_verdict(self):
        sensor, _ = proxies.load_sensor_forcings(self.path, honour_suspect=True)
        self.assertTrue(sensor['tair_str'].notna().all())

    def test_night_corrected_samples_are_counted_not_removed(self):
        # The substituted zero is the only unmeasured number in any _ok column.
        # It must reach the study, and the study must know how many there are.
        _, provenance = proxies.load_sensor_forcings(self.path)
        self.assertEqual(provenance['night_corrected_samples'], 6)

    def test_index_is_converted_to_utc(self):
        sensor, _ = proxies.load_sensor_forcings(self.path)
        # June, so the civil clock runs two hours ahead of UTC and the record
        # that was stamped midnight now sits at 22:00 of the previous day.
        self.assertEqual(sensor.index.min(), pd.Timestamp('2025-05-31 22:00'))


class TestCertifiedWindow(unittest.TestCase):
    """The window is measured as runs and gaps, because that is what it is."""

    def _sensor(self, days):
        rows = []
        for day in days:
            rows.append(pd.date_range(f'{day} 06:00', f'{day} 18:00', freq='h'))
        index = pd.DatetimeIndex(np.concatenate([r.to_numpy() for r in rows]))
        return pd.DataFrame({'sr_str': 100.0}, index=index)

    def test_contiguous_days_form_one_run(self):
        sensor = self._sensor(['2025-06-01', '2025-06-02', '2025-06-03'])
        days, summary = quality.certified_sr_window(sensor)
        self.assertEqual(summary['n_days'], 3)
        self.assertEqual(summary['n_runs'], 1)
        self.assertEqual(summary['longest_gap_days'], 0)
        self.assertEqual(summary['coverage_pct'], 100.0)

    def test_a_gap_is_reported_rather_than_spanned(self):
        # Three days either side of a week's absence is not "a fortnight of
        # data", and the summary has to be able to say so.
        sensor = self._sensor(['2025-06-01', '2025-06-02',
                               '2025-06-10', '2025-06-11'])
        days, summary = quality.certified_sr_window(sensor)
        self.assertEqual(summary['n_days'], 4)
        self.assertEqual(summary['n_runs'], 2)
        self.assertEqual(summary['longest_gap_days'], 7)
        self.assertEqual(summary['span_days'], 11)
        self.assertLess(summary['coverage_pct'], 40.0)

    def test_an_empty_channel_yields_no_days(self):
        sensor = pd.DataFrame(
            {'sr_str': [np.nan] * 24},
            index=pd.date_range('2025-06-01', periods=24, freq='h'))
        days, summary = quality.certified_sr_window(sensor)
        self.assertEqual(len(days), 0)
        self.assertEqual(summary['n_days'], 0)

    def test_min_hours_excludes_a_barely_present_day(self):
        sensor = self._sensor(['2025-06-01', '2025-06-02'])
        sensor.loc['2025-06-02 07:00':, 'sr_str'] = np.nan
        days, summary = quality.certified_sr_window(sensor, min_hours=5)
        self.assertEqual(summary['n_days'], 1)


class TestRestrictToDays(unittest.TestCase):
    """Restriction is by calendar day, not by timestamp range."""

    def test_only_the_named_days_survive(self):
        index = pd.date_range('2025-06-01', '2025-06-05 23:00', freq='h')
        frame = pd.DataFrame({'x': 1.0}, index=index)
        days = pd.DatetimeIndex(['2025-06-02', '2025-06-04'])
        kept = quality.restrict_to_days(frame, days)
        self.assertEqual(set(pd.DatetimeIndex(kept.index).normalize()),
                         set(days))

    def test_an_empty_day_set_yields_an_empty_frame(self):
        index = pd.date_range('2025-06-01', periods=24, freq='h')
        frame = pd.DataFrame({'x': 1.0}, index=index)
        self.assertEqual(len(quality.restrict_to_days(frame,
                                                       pd.DatetimeIndex([]))), 0)


class TestCircularHandling(unittest.TestCase):
    """Wind direction is a direction, and the study must never average it flat."""

    def test_wdir_is_declared_circular(self):
        self.assertIn('wdir', proxies.CIRCULAR)

    def test_diurnal_profile_averages_direction_on_the_circle(self):
        index = pd.date_range('2025-06-01', '2025-06-10 23:00', freq='h')
        frame = pd.DataFrame(index=index)
        # Alternate day by day, so that every hour-of-day slot averages one
        # reading of 350 degrees against one of 10. The arithmetic mean of
        # those two is 180 — due south, the opposite way.
        odd_day = pd.DatetimeIndex(index).day % 2 == 1
        frame['wdir_gs'] = np.where(odd_day, 350.0, 10.0)
        profile = compare.diurnal_profile(frame, ['wdir_gs'], 'summer',
                                          {'summer': [6, 7, 8]}, min_days=1)
        means = profile['wdir_gs'].dropna()
        self.assertTrue(len(means) > 0)
        near_north = np.minimum(means % 360.0, 360.0 - (means % 360.0))
        self.assertTrue((near_north < 1.0).all(),
                        f'circular mean drifted off north: {means.tolist()}')

    def test_agreement_differences_wrap(self):
        index = pd.date_range('2025-06-01', periods=48, freq='h')
        frame = pd.DataFrame(index=index)
        frame['wdir_str'] = 350.0
        frame['wdir_gs'] = 10.0
        scored = compare.pairwise_agreement(frame, 'wdir', reference='str',
                                            compared=['gs'])
        # Twenty degrees apart across north, not three hundred and forty.
        self.assertAlmostEqual(float(scored['bias'].iloc[0]), 20.0, places=6)


class TestAgreement(unittest.TestCase):
    """Bias, error and phase, on constructed series whose answer is known."""

    def _pair(self, offset=0.0, shift_hours=0):
        index = pd.date_range('2025-06-01', '2025-06-30 23:00', freq='h')
        base = 20.0 + 5.0 * np.sin(2 * np.pi * np.arange(len(index)) / 24.0)
        frame = pd.DataFrame(index=index)
        frame['tair_str'] = base
        frame['tair_gs'] = np.roll(base, shift_hours) + offset
        return frame

    def test_a_constant_offset_is_reported_as_bias(self):
        scored = compare.pairwise_agreement(self._pair(offset=2.5), 'tair',
                                            reference='str', compared=['gs'])
        self.assertAlmostEqual(float(scored['bias'].iloc[0]), 2.5, places=6)
        self.assertAlmostEqual(float(scored['r'].iloc[0]), 1.0, places=6)

    def test_identical_series_have_unit_amplitude_ratio(self):
        scored = compare.pairwise_agreement(self._pair(), 'tair', reference='str',
                                            compared=['gs'])
        self.assertAlmostEqual(float(scored['amplitude_ratio'].iloc[0]), 1.0,
                               places=6)
        self.assertAlmostEqual(float(scored['phase_lag_h'].iloc[0]), 0.0,
                               places=6)

    def test_a_shifted_series_is_reported_as_a_phase_lag(self):
        scored = compare.pairwise_agreement(self._pair(shift_hours=3), 'tair',
                                            reference='str', compared=['gs'])
        self.assertAlmostEqual(abs(float(scored['phase_lag_h'].iloc[0])), 3.0,
                               places=6)

    def test_an_absent_channel_is_skipped_rather_than_raising(self):
        index = pd.date_range('2025-06-01', periods=48, freq='h')
        frame = pd.DataFrame({'tair_str': 20.0}, index=index)
        scored = compare.pairwise_agreement(frame, 'tair', reference='str',
                                            compared=['gs'])
        self.assertEqual(int(scored['n'].iloc[0]), 0)

    def test_a_drifting_bias_shows_as_a_changing_monthly_bias(self):
        index = pd.date_range('2025-01-01', '2025-12-31 23:00', freq='h')
        frame = pd.DataFrame(index=index)
        frame['tair_str'] = 10.0
        # A bias that grows through the year is the case a pooled statistic
        # would report as a small average offset and call stable.
        frame['tair_gs'] = 10.0 + np.linspace(-5.0, 5.0, len(index))
        stability = compare.agreement_stability(frame, 'tair', reference='str',
                                                compared=['gs'])
        self.assertGreater(len(stability), 6)
        self.assertLess(float(stability['bias'].iloc[0]), -3.0)
        self.assertGreater(float(stability['bias'].iloc[-1]), 3.0)


class TestSeasons(unittest.TestCase):
    """Season definitions match study 1's, and unnamed months stay unnamed."""

    def test_months_map_to_their_season(self):
        months = {'winter': [12, 1, 2], 'summer': [6, 7, 8]}
        index = pd.DatetimeIndex(['2025-01-15', '2025-07-15', '2025-04-15'])
        labels = site.season_of(index, months)
        self.assertEqual(labels.iloc[0], 'winter')
        self.assertEqual(labels.iloc[1], 'summer')
        self.assertTrue(pd.isna(labels.iloc[2]))


class TestVerdictTable(unittest.TestCase):
    """A verdict that does not say what it rests on is not a weaker verdict."""

    def _row(self, **overrides):
        row = {'quantity': 'tair', 'source': 'era5', 'verdict': 'usable',
               'transformation': 'none', 'evidence_window': '2025-2026',
               'limitation': 'grid-cell average'}
        row.update(overrides)
        return row

    def test_a_complete_row_is_accepted(self):
        table = quality.verdict_table([self._row()])
        self.assertEqual(len(table), 1)
        self.assertEqual(list(table.columns),
                         ['quantity', 'source', 'verdict', 'transformation',
                          'evidence_window', 'limitation'])

    def test_a_row_without_its_evidence_window_is_refused(self):
        row = self._row()
        del row['evidence_window']
        with self.assertRaises(ValueError):
            quality.verdict_table([row])

    def test_a_row_without_its_limitation_is_refused(self):
        row = self._row()
        del row['limitation']
        with self.assertRaises(ValueError):
            quality.verdict_table([row])


class TestInventories(unittest.TestCase):
    """The inventory counts defects; it never removes them."""

    def test_implausible_values_are_counted_and_kept(self):
        index = pd.date_range('2025-06-01', periods=48, freq='h')
        frame = pd.DataFrame(index=index)
        frame['rain_gs'] = 0.0
        frame.iloc[0, 0] = 1e9        # the documented corrupt rainfall value
        inventory = proxies.channel_inventory(frame, 'gs',
                                              column_map={'rain': 'Pioggia'})
        self.assertEqual(int(inventory['n_implausible'].iloc[0]), 1)
        self.assertEqual(float(frame['rain_gs'].iloc[0]), 1e9)

    def test_a_circular_channel_reports_no_minimum_or_maximum(self):
        # Every direction is both the smallest and the largest, depending on
        # where the axis is cut, so reporting either would be meaningless.
        index = pd.date_range('2025-06-01', periods=48, freq='h')
        frame = pd.DataFrame({'wdir_gs': np.linspace(0, 359, 48)}, index=index)
        inventory = proxies.channel_inventory(frame, 'gs',
                                              column_map={'wdir': 'Dir'})
        self.assertTrue(pd.isna(inventory['min'].iloc[0]))
        self.assertTrue(pd.isna(inventory['max'].iloc[0]))

    def test_unconfirmed_units_are_marked(self):
        index = pd.date_range('2025-06-01', periods=48, freq='h')
        frame = pd.DataFrame({'rain_rate_gs': 0.0}, index=index)
        inventory = proxies.channel_inventory(frame, 'gs',
                                              column_map={'rain_rate': 'Int.Pio.'})
        self.assertFalse(bool(inventory['unit_confirmed'].iloc[0]))


class TestMaskImplausible(unittest.TestCase):
    """The counterpart to the inventory's count: what it counts, this removes."""

    def test_the_impossible_value_is_masked_and_the_count_reports_it(self):
        index = pd.date_range('2025-06-01', periods=48, freq='h')
        frame = pd.DataFrame(index=index)
        frame['rain_gs'] = 0.0
        frame.iloc[0, 0] = 1.7e9        # the documented corrupt rainfall value
        masked, n_masked = proxies.mask_implausible(frame)
        self.assertTrue(pd.isna(masked['rain_gs'].iloc[0]))
        self.assertTrue((masked['rain_gs'].iloc[1:] == 0.0).all())
        self.assertEqual(int(n_masked['rain_gs']), 1)

    def test_the_input_frame_is_never_modified(self):
        # This is the property the whole design rests on — the reading stays
        # in the data everywhere else in the study — so it is asserted
        # directly on the caller's own frame, not on a copy of it.
        index = pd.date_range('2025-06-01', periods=48, freq='h')
        frame = pd.DataFrame(index=index)
        frame['rain_gs'] = 0.0
        frame.iloc[0, 0] = 1.7e9
        proxies.mask_implausible(frame)
        self.assertEqual(float(frame['rain_gs'].iloc[0]), 1.7e9)

    def test_bounds_are_inclusive_and_missing_values_are_not_counted(self):
        low, high = proxies.PLAUSIBLE_RANGE['rh']
        index = pd.date_range('2025-06-01', periods=4, freq='h')
        frame = pd.DataFrame({'rh_gs': [low, high, np.nan, 50.0]}, index=index)
        masked, n_masked = proxies.mask_implausible(frame)
        self.assertEqual(float(masked['rh_gs'].iloc[0]), low)
        self.assertEqual(float(masked['rh_gs'].iloc[1]), high)
        self.assertTrue(pd.isna(masked['rh_gs'].iloc[2]))
        self.assertEqual(int(n_masked['rh_gs']), 0)

    def test_an_undocumented_quantity_is_left_untouched_by_default(self):
        index = pd.date_range('2025-06-01', periods=4, freq='h')
        frame = pd.DataFrame({'foo_gs': [1.0, -999.0, 5.0, 1e9]}, index=index)
        masked, n_masked = proxies.mask_implausible(frame)
        self.assertTrue((masked['foo_gs'] == frame['foo_gs']).all())
        self.assertNotIn('foo_gs', n_masked.index)
        # An explicit mapping overrides the default rather than extending it.
        masked, n_masked = proxies.mask_implausible(
            frame, plausible_range={'foo': (0.0, 10.0)})
        self.assertTrue(pd.isna(masked['foo_gs'].iloc[1]))
        self.assertTrue(pd.isna(masked['foo_gs'].iloc[3]))
        self.assertEqual(int(n_masked['foo_gs']), 2)


class TestGroundStationStampOffset(unittest.TestCase):
    """``stamp_offset`` moves the value read at a timestamp, nothing else."""

    def setUp(self):
        import tempfile
        self.directory = tempfile.mkdtemp()

        # Raw timestamps every 30 minutes, each carrying a distinct value, so a
        # shift in which raw sample lands under which grid label is visible.
        index = pd.date_range('2025-06-01 00:00', periods=6, freq='30min')
        frame = pd.DataFrame({'Temp': np.arange(len(index), dtype=float)},
                             index=index)
        frame.index.name = 'datetime'
        self.path = _write_csv(frame, self.directory, 'ground.csv')

    def test_offset_moves_the_value_to_the_earlier_label(self):
        # A grid fine enough (5 minutes) that a 15-minute offset, itself a
        # multiple of the grid step, cannot be absorbed by the resampling
        # bin edges the way it would be on a coarser grid.
        vendor = proxies.load_ground_station(
            self.path, column_map={'tair': 'Temp'}, freq='5min', min_count=1)
        offset = proxies.load_ground_station(
            self.path, column_map={'tair': 'Temp'}, freq='5min', min_count=1,
            stamp_offset='15min')

        vendor_label = pd.Timestamp('2025-06-01 01:00')  # the raw sample valued 2.0
        offset_label = vendor_label - pd.Timedelta(minutes=15)
        self.assertEqual(float(vendor['tair_gs'].loc[vendor_label]), 2.0)
        self.assertEqual(float(offset['tair_gs'].loc[offset_label]), 2.0)
        self.assertTrue(pd.isna(offset['tair_gs'].loc[vendor_label]))

    def test_offset_is_recorded_in_attrs(self):
        offset = proxies.load_ground_station(
            self.path, column_map={'tair': 'Temp'}, freq='5min', min_count=1,
            stamp_offset='15min')
        self.assertEqual(offset.attrs['stamp_offset'], '0 days 00:15:00')

    def test_default_leaves_the_vendor_stamp_untouched(self):
        vendor = proxies.load_ground_station(
            self.path, column_map={'tair': 'Temp'}, freq='30min', min_count=1)
        self.assertIsNone(vendor.attrs['stamp_offset'])
        self.assertEqual(float(vendor['tair_gs'].loc['2025-06-01 00:00']), 0.0)


class TestReferenceShiftScan(unittest.TestCase):
    """Displacing the reference recovers an injected displacement, with sign."""

    def test_argmax_shift_matches_the_injected_reference_delay(self):
        freq = '20min'
        rng = np.random.default_rng(20260906)
        index = pd.date_range('2025-06-01', periods=300, freq=freq)
        # Unstructured values, so only the correct alignment correlates: any
        # other candidate compares unrelated samples and scores near zero.
        base = pd.Series(rng.normal(size=len(index)), index=index)

        d_true = 40  # minutes; the reference's own stamp sits 40 minutes late
        reference = base.copy()
        reference.index = reference.index + pd.Timedelta(minutes=d_true)

        scan = temporal_alignment.reference_shift_scan(
            base, reference, shifts_minutes=[-40, -20, 0, 20, 40, 60],
            freq=freq, min_pairs=1, min_days=1)
        best = scan.loc[scan['r_levels'].idxmax()]
        self.assertEqual(int(best['shift_minutes']), d_true)
        self.assertGreater(float(best['r_levels']), 0.99)

    def test_negative_injected_delay_recovers_a_negative_argmax(self):
        freq = '20min'
        rng = np.random.default_rng(20260907)
        index = pd.date_range('2025-06-01', periods=300, freq=freq)
        base = pd.Series(rng.normal(size=len(index)), index=index)

        d_true = -40
        reference = base.copy()
        reference.index = reference.index + pd.Timedelta(minutes=d_true)

        scan = temporal_alignment.reference_shift_scan(
            base, reference, shifts_minutes=[-60, -40, -20, 0, 20],
            freq=freq, min_pairs=1, min_days=1)
        best = scan.loc[scan['r_levels'].idxmax()]
        self.assertEqual(int(best['shift_minutes']), d_true)

    def test_matches_paired_shift_scan_for_the_same_arguments(self):
        # The two functions read the same number two different ways; the
        # numbers themselves must not differ.
        freq = '20min'
        rng = np.random.default_rng(20260908)
        index = pd.date_range('2025-06-01', periods=200, freq=freq)
        sensor = pd.Series(rng.normal(size=len(index)), index=index)
        reference = pd.Series(rng.normal(size=len(index)), index=index)

        shifts = [-20, 0, 20, 40]
        one = temporal_alignment.paired_shift_scan(sensor, reference, shifts,
                                                    freq=freq, min_pairs=1,
                                                    min_days=1)
        other = temporal_alignment.reference_shift_scan(sensor, reference,
                                                         shifts, freq=freq,
                                                         min_pairs=1,
                                                         min_days=1)
        pd.testing.assert_frame_equal(one.reset_index(drop=True),
                                      other.reset_index(drop=True))


class TestShiftSummary(unittest.TestCase):
    """The gain and sign arithmetic, on a scan worked out by hand."""

    def _scan(self):
        return pd.DataFrame({
            'pair': ['str-gs'] * 4 + ['str-era5'] * 4,
            'period': ['current'] * 8,
            'shift_minutes': [-20, 0, 20, 40] * 2,
            'r_daily': [0.10, 0.20, 0.55, 0.30,   # str-gs: best at +20
                       -0.05, -0.40, -0.20, -0.10],  # str-era5: best at 0
        })

    def test_gain_and_argmax_for_the_improving_pair(self):
        summary = temporal_alignment.shift_summary(self._scan())
        row = summary[summary['pair'] == 'str-gs'].iloc[0]
        self.assertAlmostEqual(float(row['r_shift0']), 0.20, places=6)
        self.assertEqual(int(row['shift_argmax']), 20)
        self.assertAlmostEqual(float(row['r_argmax']), 0.55, places=6)
        self.assertAlmostEqual(float(row['gain']), 0.35, places=6)
        self.assertEqual(row['sign'], 'positive')

    def test_zero_is_already_the_best_shift_for_the_other_pair(self):
        summary = temporal_alignment.shift_summary(self._scan())
        row = summary[summary['pair'] == 'str-era5'].iloc[0]
        self.assertAlmostEqual(float(row['r_shift0']), -0.40, places=6)
        self.assertEqual(int(row['shift_argmax']), 0)
        self.assertAlmostEqual(float(row['gain']), 0.0, places=6)
        self.assertEqual(row['sign'], 'zero')

    def test_ties_break_toward_the_smaller_absolute_shift(self):
        scan = pd.DataFrame({
            'pair': ['x'] * 3,
            'period': ['p'] * 3,
            'shift_minutes': [-20, 0, 20],
            'r_daily': [0.50, 0.20, 0.50],
        })
        summary = temporal_alignment.shift_summary(scan)
        self.assertEqual(int(summary.iloc[0]['shift_argmax']), -20)

    def test_a_group_with_no_finite_correlation_is_undefined(self):
        scan = pd.DataFrame({
            'pair': ['x'] * 2,
            'period': ['p'] * 2,
            'shift_minutes': [-20, 0],
            'r_daily': [np.nan, np.nan],
        })
        summary = temporal_alignment.shift_summary(scan)
        self.assertTrue(pd.isna(summary.iloc[0]['shift_argmax']))
        self.assertEqual(summary.iloc[0]['sign'], 'undefined')


if __name__ == '__main__':
    unittest.main(verbosity=2)
