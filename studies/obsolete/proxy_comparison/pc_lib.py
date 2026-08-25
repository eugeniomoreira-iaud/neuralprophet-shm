"""
Module: pc_lib.py

Support library for the proxy-comparison study
(``studies/proxy_comparison/``).

Three independent sources now measure the environment at Gubbio: the logger on
the structure, a ground station in the town, and ERA5 reanalysis for the grid
cell containing the site. They share channel names and they do not measure the
same thing — a housing on a sun-exposed wall, a standard screen in town and a
nine-kilometre grid average are three different quantities. This module puts
them on one grid under one naming scheme and measures how far apart they are.

Two rules govern the comparison and neither is negotiable.

**The target is the calibrated reading and is never rebuilt.** ``inc_comp`` is
the manufacturer's formula applied with the temperature measured at the
transducer. That is an instrument calibration, not a regression: the transducer
experiences the temperature at the transducer, so substituting a grid-cell or
in-town temperature into the formula would be meaningless. Nothing here
re-estimates the coefficient, and the only adjustment permitted is the levelling
correction across the 2025-02-21 regime change.

**The clock is measured, not assumed.** The Oikolab export is documented as UTC
while the logger runs Italian local time with daylight saving. An uncorrected
offset would misplace every diurnal phase in the study, so the offset is
recovered from the solar-radiation channels, whose daily maximum is set by the
sun rather than by a convention.

Loading of the sensor side comes from ``ud_lib``; operator scans and ranking
from ``ip_lib``; slope and diurnal helpers from ``tc_lib``. Only what is new to
this study is implemented below.

Nothing in this module writes to the raw archive.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sib in ('thermal_compensation', 'thermal_compensation_legacy',
             'inclination_prediction', 'inclination_prediction_legacy',
             'unified_dataset'):
    _p = os.path.abspath(os.path.join(_HERE, '..', _sib))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tc_lib as tc                                             # noqa: E402
import ip_lib as ip                                             # noqa: E402
import ud_lib as ud                                             # noqa: E402

from tc_lib import set_context, figsize, _finish                # noqa: E402,F401


# ──────────────────────────────────────────────────────────────────────
# Canonical naming
# ──────────────────────────────────────────────────────────────────────
#
# One scheme, ``{quantity}_{source}``. The source suffix is always present, so
# a column name states where the number came from without reference to a
# docstring. ``docs/proxy-data-dictionary.md`` is the reference copy of this
# mapping and is generated from the tables below, so the two cannot drift.

#: Source identifiers, in order of physical proximity to the wall.
SOURCES = ('str', 'gs', 'era5')

#: Human-readable source names, for figures and tables.
SOURCE_LABEL = {
    'str': 'on-structure',
    'gs': 'ground station',
    'era5': 'ERA5 reanalysis',
}

#: What each source physically measures. These are not replicates and the study
#: exists because they are not.
SOURCE_DESCRIPTION = {
    'str': 'Air inside the instrument housing on the monitored wall.',
    'gs': 'Standard-exposure weather station in Gubbio town.',
    'era5': 'Reanalysis average over a grid cell of roughly nine kilometres.',
}

#: Canonical quantity names and their units after harmonisation.
QUANTITY_UNIT = {
    'tair': 'degC',
    'rh': '%',
    'sr': 'W/m2',
    'tdew': 'degC',
    'pres': 'hPa',
    'wspd': 'm/s',
    'rain': 'mm',
    'twall': 'degC',
    'tskin': 'degC',
    'lwrad': 'W/m2',
}

#: On-structure channels, as published by ``ud_lib``. The unified sensor table
#: keeps these names; they are aliased here rather than renamed at source, so
#: no existing artefact changes.
STR_MAP = {
    'tair': 'tair',
    'rh': 'rh',
    'sr': 'sr',
    'twall': 'twall',
}

#: Ground-station channels, as exported by ``auxiliary/meteosystem_italy.py``.
GS_MAP = {
    'tair': 'Temp',
    'rh': 'Umid',
    'sr': 'Rad.Sol.',
    'tdew': 'Dew pt',
    'pres': 'Press',
    'wspd': 'Vento',
    'rain': 'Pioggia',
}

#: ERA5 channels, as exported by ``auxiliary/oiko.py``.
ERA5_MAP = {
    'tair': 'temperature (degC)',
    'rh': 'relative_humidity (0-1)',
    'sr': 'surface_solar_radiation (W/m^2)',
    'tdew': 'dewpoint_temperature (degC)',
    'wspd': 'wind_speed (m/s)',
    'rain': 'total_precipitation (mm of water equivalent)',
    'tskin': 'skin_temperature (degC)',
    'lwrad': 'surface_thermal_radiation (W/m^2)',
}

#: Channels needing a unit conversion on load, as (source, quantity) → factor.
#: ERA5 reports relative humidity as a fraction where every other source reports
#: per cent; left uncorrected it would appear as a 99 % dry bias.
UNIT_FACTOR = {
    ('era5', 'rh'): 100.0,
}

#: Plausible physical range per quantity, used by :func:`harmonisation_report`
#: as a guard rather than a filter. A value outside these bounds is a defect to
#: be reported, not silently removed.
PLAUSIBLE_RANGE = {
    'tair': (-25.0, 45.0),
    'rh': (0.0, 100.0),
    'sr': (0.0, 1400.0),
    'tdew': (-30.0, 30.0),
    'pres': (900.0, 1060.0),
    'wspd': (0.0, 60.0),
    'rain': (0.0, 200.0),
    'twall': (-25.0, 60.0),
    'tskin': (-30.0, 70.0),
    'lwrad': (0.0, 600.0),
}

#: The target. Fixed, and never rebuilt by this study.
TARGET = 'inc_comp'

#: The same target with the levelling correction applied across the instrument
#: changeover, for work that spans it.
TARGET_JOINED = 'inc_comp_joined'


# ──────────────────────────────────────────────────────────────────────
# Loading and harmonisation
# ──────────────────────────────────────────────────────────────────────

def load_ground_station(path, freq=ud.ANALYSIS_FREQ):
    """
    Load the ground-station export onto the analysis grid.

    The export is natively half-hourly and carries a small number of duplicated
    timestamps, which are dropped rather than merged: unlike the sensor archive,
    where conflicting copies of a record routinely differ in which block holds
    real values, these duplicates are re-scrapes of the same observation.

    Parameters
    ----------
    path : str
        Path to the assembled CSV.
    freq : str, optional
        Analysis grid. Default :data:`ud_lib.ANALYSIS_FREQ`.

    Returns
    -------
    pd.DataFrame
        Canonically named columns with the ``_gs`` suffix.
    """
    raw = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    n_dup = int(raw.index.duplicated().sum())
    raw = raw[~raw.index.duplicated(keep='first')]

    out = pd.DataFrame(index=raw.index)
    for quantity, native in GS_MAP.items():
        if native in raw.columns:
            out[f'{quantity}_gs'] = pd.to_numeric(raw[native], errors='coerce')

    out = out.resample(freq).mean()
    out.attrs['duplicates_dropped'] = n_dup
    out.attrs['native_step'] = str(raw.index.to_series().diff().median())
    return out


def load_era5(path, freq=ud.ANALYSIS_FREQ):
    """
    Load the ERA5 export onto the analysis grid.

    Relative humidity is converted from the fraction ERA5 reports to the per
    cent every other source reports. No other unit is changed.

    Parameters
    ----------
    path : str
        Path to the Oikolab export.
    freq : str, optional
        Analysis grid. Default :data:`ud_lib.ANALYSIS_FREQ`.

    Returns
    -------
    pd.DataFrame
        Canonically named columns with the ``_era5`` suffix.
    """
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    raw = raw[~raw.index.duplicated(keep='first')]

    out = pd.DataFrame(index=raw.index)
    for quantity, native in ERA5_MAP.items():
        if native not in raw.columns:
            continue
        values = pd.to_numeric(raw[native], errors='coerce')
        factor = UNIT_FACTOR.get(('era5', quantity))
        out[f'{quantity}_era5'] = values * factor if factor else values

    return out.resample(freq).mean()


def alias_structure(sensor):
    """
    Rename the unified sensor table's channels into the canonical scheme.

    The unified table keeps its published names on disk; this aliases them in
    memory so that every column in the comparison carries an explicit source.

    Parameters
    ----------
    sensor : pd.DataFrame
        Output of :func:`ud_lib.load_unified`.

    Returns
    -------
    pd.DataFrame
        The environmental channels with the ``_str`` suffix, plus the target
        and provenance columns unchanged.
    """
    out = pd.DataFrame(index=sensor.index)
    for quantity, native in STR_MAP.items():
        if native in sensor.columns:
            out[f'{quantity}_str'] = sensor[native]
    for keep in (TARGET, TARGET_JOINED, 'inc', 'era', 'inc_source'):
        if keep in sensor.columns:
            out[keep] = sensor[keep]
    return out


def harmonisation_report(df):
    """
    One row per canonical channel: extent, coverage and range.

    Also flags values outside :data:`PLAUSIBLE_RANGE`. These are reported rather
    than removed — a radiation channel reading sixty watts at midnight is a
    finding about the sensor, and deleting it would hide the finding.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying suffixed channels.

    Returns
    -------
    pd.DataFrame
        One row per channel present.
    """
    rows = []
    for column in df.columns:
        if '_' not in column:
            continue
        quantity, _, source = column.rpartition('_')
        if source not in SOURCES or quantity not in QUANTITY_UNIT:
            continue
        values = pd.to_numeric(df[column], errors='coerce')
        observed = values.dropna()
        low, high = PLAUSIBLE_RANGE.get(quantity, (-np.inf, np.inf))
        rows.append({
            'channel': column,
            'quantity': quantity,
            'source': source,
            'unit': QUANTITY_UNIT.get(quantity, ''),
            'first': str(values.first_valid_index()),
            'last': str(values.last_valid_index()),
            'coverage_%': round(100.0 * values.notna().mean(), 1),
            'min': round(float(observed.min()), 2) if len(observed) else np.nan,
            'max': round(float(observed.max()), 2) if len(observed) else np.nan,
            'mean': round(float(observed.mean()), 2) if len(observed) else np.nan,
            'outside_plausible': int(((observed < low) | (observed > high)).sum()),
        })
    return pd.DataFrame(rows).set_index('channel')


# ──────────────────────────────────────────────────────────────────────
# The clock
# ──────────────────────────────────────────────────────────────────────

def clock_offset(df, quantity='sr', reference='str', sources=None,
                 max_shift=6, by_season=True):
    """
    Recover each source's time offset from its solar-radiation channel.

    The daily maximum of incident radiation is set by the sun, not by a
    recording convention, so radiation is the one channel whose phase is known a
    priori to be identical at all three sources. Any shift between them is a
    clock difference.

    The test is run separately for winter and summer because the logger runs
    Italian local time with daylight saving while the reanalysis is documented
    as UTC; a single annual figure would average a one-hour and a two-hour
    offset into a meaningless ninety minutes.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying ``{quantity}_{source}`` columns.
    quantity : str, optional
        Channel to align on. Default ``'sr'``.
    reference : str, optional
        Source every other is measured against. Default ``'str'``.
    sources : iterable of str, optional
        Sources to test. Defaults to every source present besides the reference.
    max_shift : int, optional
        Largest shift considered, in steps. Default ``6``.
    by_season : bool, optional
        Report winter and summer separately. Default ``True``.

    Returns
    -------
    pd.DataFrame
        One row per source and season: the shift maximising the correlation, and
        the correlation attained at that shift and at zero.
    """
    ref_col = f'{quantity}_{reference}'
    if ref_col not in df.columns:
        raise ValueError(f'reference channel {ref_col} not present')

    if sources is None:
        sources = [s for s in SOURCES if s != reference
                   and f'{quantity}_{s}' in df.columns]

    seasons = ([('winter', [12, 1, 2]), ('summer', [6, 7, 8])] if by_season
               else [('all', list(range(1, 13)))])

    rows = []
    for source in sources:
        col = f'{quantity}_{source}'
        for label, months in seasons:
            mask = df.index.month.isin(months)
            a = df.loc[mask, ref_col]
            b = df.loc[mask, col]
            if a.notna().sum() < 100 or b.notna().sum() < 100:
                continue
            scores = {}
            for shift in range(-max_shift, max_shift + 1):
                joined = pd.concat([a, b.shift(shift)], axis=1).dropna()
                if len(joined) < 100:
                    continue
                scores[shift] = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if not scores:
                continue
            best = max(scores, key=scores.get)
            rows.append({
                'source': source,
                'season': label,
                'best_shift_h': best,
                'r_at_best': round(scores[best], 4),
                'r_at_zero': round(scores.get(0, np.nan), 4),
                'n': int(pd.concat([a, b], axis=1).dropna().shape[0]),
            })
    return pd.DataFrame(rows)


def apply_clock(df, offsets):
    """
    Shift each source's channels by its measured offset.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    offsets : dict
        Source to shift in steps. A positive value moves that source's values
        later in time.

    Returns
    -------
    pd.DataFrame
        A copy with the shifts applied.
    """
    out = df.copy()
    for source, shift in offsets.items():
        if not shift:
            continue
        columns = [c for c in out.columns if c.endswith(f'_{source}')]
        out[columns] = out[columns].shift(shift)
    return out


def peak_hour_profile(df, quantity='sr', by_season=True):
    """
    Mean value by hour of day, per source, for a stated channel.

    The direct visual counterpart to :func:`clock_offset`.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str, optional
        Channel to profile. Default ``'sr'``.
    by_season : bool, optional
        Split winter and summer. Default ``True``.

    Returns
    -------
    pd.DataFrame
        Indexed by hour, one column per source and season.
    """
    seasons = ([('winter', [12, 1, 2]), ('summer', [6, 7, 8])] if by_season
               else [('all', list(range(1, 13)))])
    out = {}
    for source in SOURCES:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue
        for label, months in seasons:
            sub = df.loc[df.index.month.isin(months), col].dropna()
            if len(sub) < 100:
                continue
            out[f'{source}·{label}'] = sub.groupby(sub.index.hour).mean()
    return pd.DataFrame(out)


# ──────────────────────────────────────────────────────────────────────
# Absolute agreement
# ──────────────────────────────────────────────────────────────────────

def pairwise_agreement(df, quantity, sources=SOURCES, min_days=30,
                       dt_hours=1.0):
    """
    How far apart two sources are, for one quantity, on their common overlap.

    Reports the difference in three complementary ways, because they answer
    different questions. Bias and RMSE say how large the disagreement is. The
    regression slope and intercept say what *kind* it is: a slope near one with
    a non-zero intercept is an offset, a slope away from one is a scale error,
    and the two call for different remedies. The limits of agreement are the
    Bland-Altman interval, which is what a practitioner needs in order to know
    how far a single substituted value might be from the one it replaces.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame, clock already corrected.
    quantity : str
        Canonical quantity name.
    sources : iterable of str, optional
        Sources to compare pairwise.
    min_days : float, optional
        Shortest overlap for which a comparison is reported. Default ``30``.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per ordered pair, the second source measured against the first.
    """
    rows = []
    present = [s for s in sources if f'{quantity}_{s}' in df.columns]
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            ca, cb = f'{quantity}_{a}', f'{quantity}_{b}'
            joined = df[[ca, cb]].dropna()
            days = len(joined) * dt_hours / 24.0
            if days < min_days:
                continue
            diff = joined[cb] - joined[ca]
            fit = np.polyfit(joined[ca].to_numpy(), joined[cb].to_numpy(), 1)
            rows.append({
                'quantity': quantity,
                'reference': a,
                'compared': b,
                'n': int(len(joined)),
                'overlap_days': round(days, 1),
                'bias': round(float(diff.mean()), 3),
                'MAE': round(float(diff.abs().mean()), 3),
                'RMSE': round(float(np.sqrt((diff ** 2).mean())), 3),
                'sd_of_diff': round(float(diff.std()), 3),
                'slope': round(float(fit[0]), 4),
                'intercept': round(float(fit[1]), 3),
                'r': round(float(joined[ca].corr(joined[cb])), 4),
                # Bland-Altman limits of agreement: where 95 % of the
                # differences between the two sources actually fall.
                'loa_lower': round(float(diff.mean() - 1.96 * diff.std()), 3),
                'loa_upper': round(float(diff.mean() + 1.96 * diff.std()), 3),
            })
    return pd.DataFrame(rows)


def agreement_breakdown(df, quantity, reference='str', compared='era5',
                        by='season'):
    """
    The same difference, split by season or by hour of day.

    A source can agree on annual average and disagree at every noon, which is
    the disagreement that matters for a wall driven by the daily cycle.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    reference, compared : str, optional
        Sources to difference.
    by : {'season', 'hour', 'month'}, optional
        Grouping. Default ``'season'``.

    Returns
    -------
    pd.DataFrame
        One row per group.
    """
    ca, cb = f'{quantity}_{reference}', f'{quantity}_{compared}'
    joined = df[[ca, cb]].dropna()
    if joined.empty:
        return pd.DataFrame()

    diff = (joined[cb] - joined[ca]).rename('diff')
    if by == 'season':
        key = pd.Series([ud._season(m) for m in joined.index.month],
                        index=joined.index, name='group')
    elif by == 'hour':
        key = pd.Series(joined.index.hour, index=joined.index, name='group')
    else:
        key = pd.Series(joined.index.month, index=joined.index, name='group')

    grouped = pd.concat([diff, key], axis=1).groupby('group')['diff']
    out = pd.DataFrame({
        'n': grouped.size(),
        'bias': grouped.mean().round(3),
        'MAE': grouped.apply(lambda s: s.abs().mean()).round(3),
        'sd': grouped.std().round(3),
    })
    out.attrs['pair'] = f'{compared} - {reference}'
    out.attrs['quantity'] = quantity
    return out


# ──────────────────────────────────────────────────────────────────────
# Dynamic agreement
# ──────────────────────────────────────────────────────────────────────

def diurnal_comparison(df, quantity, sources=SOURCES, reference='str'):
    """
    Amplitude and phase of the daily cycle, per source.

    A proxy can sit two degrees low and still drive the wall correctly, or match
    on average and miss every daily swing. Amplitude and phase separate those
    cases, and the amplitude ratio against the reference is the single number
    that says whether a gridded product can stand in for a sensor on the wall.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame, clock corrected.
    quantity : str
        Canonical quantity name.
    sources : iterable of str, optional
        Sources to profile.
    reference : str, optional
        Source the amplitude ratio is taken against. Default ``'str'``.

    Returns
    -------
    pd.DataFrame
        One row per source.
    """
    rows = []
    ref_amp = np.nan
    for source in sources:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 24 * 30:
            continue
        profile = series.groupby(series.index.hour).mean()
        amplitude = float((profile.max() - profile.min()) / 2.0)
        if source == reference:
            ref_amp = amplitude
        rows.append({
            'source': source,
            'n': int(len(series)),
            'mean': round(float(series.mean()), 3),
            'diurnal_amplitude': round(amplitude, 3),
            'peak_hour': int(profile.idxmax()),
            'trough_hour': int(profile.idxmin()),
            # tc_lib measures the amplitude day by day and averages, which is
            # not the same as the amplitude of the averaged day: the first
            # survives phase jitter, the second is flattened by it.
            'daily_amplitude': round(float(tc.diurnal_amplitude(series)), 3),
        })
    out = pd.DataFrame(rows).set_index('source')
    if np.isfinite(ref_amp) and ref_amp:
        out['amplitude_ratio'] = (out['diurnal_amplitude'] / ref_amp).round(3)
    return out


def difference_agreement(df, quantity, sources=SOURCES, min_days=30,
                         dt_hours=1.0):
    """
    Agreement between sources in the first-difference domain.

    The wall responds to change, not to level, and the first difference is
    invariant to every arbitrary constant in the chain. Two sources can share a
    trend and disagree completely about the hour-to-hour movement, and this is
    the statistic that catches it.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    sources : iterable of str, optional
        Sources to compare.
    min_days : float, optional
        Shortest overlap reported. Default ``30``.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per ordered pair.
    """
    diffs = pd.DataFrame(index=df.index)
    for source in sources:
        col = f'{quantity}_{source}'
        if col in df.columns:
            # A difference is only meaningful when its two endpoints are one
            # step apart; a compacted index would silently span a gap.
            diffs[f'{quantity}_{source}'] = df[col].diff().where(
                df[col].notna() & df[col].shift().notna())
    return pairwise_agreement(diffs, quantity, sources=sources,
                              min_days=min_days, dt_hours=dt_hours)


# ──────────────────────────────────────────────────────────────────────
# Relationship with the inclination
# ──────────────────────────────────────────────────────────────────────

def target_relationship(df, quantity, target=TARGET, sources=SOURCES,
                        hac_lags=24, min_days=30, dt_hours=1.0,
                        blocks=None, tier=None):
    """
    Each source against the calibrated inclination, in levels and differences.

    The target is fixed. It is the manufacturer's calibration applied with the
    temperature measured at the transducer, and this function never rebuilds it
    — the slope it reports describes the residual response that survives the
    calibration, and is not a candidate coefficient.

    That the on-structure temperature appears inside the target is a property of
    the calibrated measurement and is stated in the report rather than corrected
    for.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying the target.
    quantity : str
        Canonical quantity name.
    target : str, optional
        Target column. Default :data:`TARGET`.
    sources : iterable of str, optional
        Sources to test.
    hac_lags : int, optional
        Newey-West lag truncation. Default ``24``. The residuals here are
        strongly autocorrelated and ordinary standard errors would be far too
        small.
    min_days : float, optional
        Shortest overlap reported. Default ``30``.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    blocks : pd.DataFrame or None, optional
        Block inventory from :func:`tier_blocks`. When given together with
        ``tier``, every statistic is computed inside that tier's blocks: levels
        on the union, differences taken within each block and concatenated. The
        default ``None`` computes on the whole frame, which pools both
        instrument eras and every ragged stretch between them, and is retained
        only as the contrast case.
    tier : str or None, optional
        Tier name to select from ``blocks``. Required when ``blocks`` is given.

    Returns
    -------
    pd.DataFrame
        One row per source and domain, carrying a ``tier`` column when
        ``blocks`` is given.
    """
    if blocks is not None and tier is None:
        raise ValueError('tier must be given when blocks is given')

    rows = []
    for source in sources:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue

        if blocks is None:
            levels = df[[target, col]]
            differences = df[[target, col]].diff().where(
                df[[target, col]].notna()
                & df[[target, col]].shift().notna())
        else:
            levels = tier_frame(df, blocks, tier)[[target, col]]
            differences = tier_differences(df, blocks, tier,
                                           cols=[target, col])

        for domain, frame in (('levels', levels),
                              ('differences', differences)):
            joined = frame.dropna()
            days = len(joined) * dt_hours / 24.0
            if days < min_days:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = tc.fit_slope(
                    joined.rename(columns={target: 'inc', col: 'driver'}),
                    'inc', 'driver', differenced=False, hac_lags=hac_lags)
            rows.append({
                'quantity': quantity,
                'tier': tier,
                'source': source,
                'domain': domain,
                'n': int(len(joined)),
                'overlap_days': round(days, 1),
                'r': round(float(joined[target].corr(joined[col])), 4),
                'slope_mdeg_per_unit': round(float(fit['slope']), 4),
                'slope_se': round(float(fit['se']), 4),
                't': round(float(fit['slope'] / fit['se']), 2)
                if fit['se'] else np.nan,
                'r2': round(float(fit.get('r2', np.nan)), 4),
            })

    columns = (['quantity'] + (['tier'] if blocks is not None else [])
               + ['source', 'domain', 'n', 'overlap_days', 'r',
                  'slope_mdeg_per_unit', 'slope_se', 't', 'r2'])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def operator_by_source(df, quantity, target=TARGET, sources=SOURCES,
                       delays=None, taus=None, detrend_hours=168,
                       dt_hours=1.0):
    """
    Transport delay and thermal inertia between each source and the target.

    A grid cell showing a different apparent delay from a sensor on the wall is
    a measurement rather than an error: the reanalysis reports an hourly average
    over nine kilometres, the ground station an instantaneous reading in town,
    and the on-structure channel the air inside a housing that the sun warms
    directly. They should not agree, and how far they disagree is what decides
    whether one can substitute for another.

    Delays are bounded at twelve hours for external forcings, per
    ``docs/raw-data-format.md`` Section 7.5: beyond half a diurnal cycle a delay
    is indistinguishable from a lead and the scan returns aliases.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying the target.
    quantity : str
        Canonical quantity name.
    target : str, optional
        Target column. Default :data:`TARGET`.
    sources : iterable of str, optional
        Sources to scan.
    delays : iterable of int, optional
        Delay grid in steps. Default ``range(0, 13)``.
    taus : iterable of float, optional
        Inertia time constants in hours.
    detrend_hours : int, optional
        Rolling mean removed before scanning, isolating the diurnal band.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    summary : pd.DataFrame
        One row per source: the optimum and the score there.
    scans : dict
        Source to the full scan grid, for the heat maps.
    """
    if delays is None:
        delays = list(range(0, 13))
    if taus is None:
        taus = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168]

    rows, scans = [], {}
    for source in sources:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue
        sub = df[[target, col]].dropna()
        if len(sub) < 24 * 30:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scan = ip.lag_inertia_scan(
                df[[target, col]], target, col, delays, taus,
                dt_hours=dt_hours, detrend_hours=detrend_hours)
        scans[source] = scan
        best = ip.best_operator(scan)
        rows.append({
            'quantity': quantity,
            'source': source,
            'delay_h': best['delay_h'],
            'tau_h': best['tau_h'],
            'r2': round(float(best['r2']), 4),
            'r': round(float(best.get('r', np.nan)), 4),
            'n': int(len(sub)),
        })
    return pd.DataFrame(rows), scans


# ──────────────────────────────────────────────────────────────────────
# The clock, settled against the sun rather than against another sensor
# ──────────────────────────────────────────────────────────────────────
#
# Cross-correlating one radiation channel against another only ever gives a
# *relative* offset, and it inherits every defect of whichever channel is chosen
# as the reference. The on-structure radiation sensor is a poor reference on
# both counts: it exists only from 2025-02-21, and it reads about sixty watts at
# midnight, which destroys the correlation in winter when the true signal is
# small.
#
# Solar noon is computable exactly from the site's longitude and the day of the
# year. Comparing each source's observed radiation phase against it gives an
# absolute offset per source, with no reference channel and no shared defect.

# The site coordinates originate in ``auxiliary/oiko.py``, lines 84 and 85,
# where they are the ``LAT`` and ``LON`` arguments of the Oikolab API request
# that produced ``data/raw/proxies/oikolab_weather.csv``. They therefore fix two
# things at once: the point at which the ERA5 grid cell was extracted, and the
# solar geometry used by the clock tests and by the daylight mask. Changing one
# without the other would silently compare a reanalysis series against the sun
# seen from somewhere else, so the two must always be edited together.

#: Site longitude, degrees east. See the provenance note above.
SITE_LONGITUDE = 12.582047

#: Site latitude, degrees north. See the provenance note above.
SITE_LATITUDE = 43.35343


def solar_noon_utc(dates, longitude=SITE_LONGITUDE):
    """
    Solar noon in UTC hours, for each date, by the NOAA formulation.

    Accounts for the equation of time, which moves solar noon by up to about
    sixteen minutes over the year and would otherwise be mistaken for a clock
    drift.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates to evaluate.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.

    Returns
    -------
    pd.Series
        Solar noon, in hours after UTC midnight.
    """
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    minutes = 720.0 - 4.0 * longitude - eqtime
    return pd.Series(minutes / 60.0, index=pd.DatetimeIndex(dates))


def solar_elevation(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE):
    """
    Solar elevation angle at each timestamp, by the NOAA formulation.

    The elevation is the angle of the sun above the horizon. It is computed from
    the date and the site coordinates alone, so it is independent of every
    measured channel and is defined at timestamps where a radiation sensor is
    missing. That independence is the reason it is preferred to a data-derived
    daylight indicator: the same mask can then be applied to all three sources
    without any of them helping to define it.

    The calculation takes no account of cloud, of the local horizon formed by
    the surrounding terrain, or of atmospheric refraction near the horizon. It
    answers where the sun is, not how much radiation reached the wall.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to evaluate. Interpreted as UTC, consistent with
        :func:`solar_noon_utc`.
    latitude : float, optional
        Degrees north. Default :data:`SITE_LATITUDE`.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.

    Returns
    -------
    pd.Series
        Elevation in degrees, indexed by ``index``.
    """
    index = pd.DatetimeIndex(index)
    doy = index.dayofyear.to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0)

    # The equation of time, in minutes. Identical to the term used by
    # solar_noon_utc, so the two cannot drift apart.
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    minutes = (index.hour.to_numpy(dtype=float) * 60.0
               + index.minute.to_numpy(dtype=float)
               + index.second.to_numpy(dtype=float) / 60.0)
    true_solar_time = minutes + eqtime + 4.0 * longitude
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)

    lat = np.radians(latitude)
    cos_zenith = (np.sin(lat) * np.sin(declination)
                  + np.cos(lat) * np.cos(declination) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return pd.Series(90.0 - np.degrees(np.arccos(cos_zenith)), index=index)


def daylight_mask(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE,
                  min_elevation=0.0):
    """
    True where the sun stands above the horizon.

    Roughly half of every solar-radiation record is a structural night-time
    zero. Those zeros are real measurements and are never removed from the
    record, but they distort every agreement statistic computed over the full
    day: correlations are inflated by a large block of samples on which all
    sources trivially agree, limits of agreement are computed over a bimodal
    distribution, and the trough hour of a mean diurnal cycle is an arbitrary
    point inside the night. This mask allows those statistics to be reported a
    second time over the daylight hours alone, beside the all-hours figures
    rather than in place of them.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to evaluate.
    latitude : float, optional
        Degrees north. Default :data:`SITE_LATITUDE`.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.
    min_elevation : float, optional
        Elevation threshold in degrees. Default ``0.0``, the geometric horizon.

    Returns
    -------
    pd.Series
        Boolean, indexed by ``index``.
    """
    return solar_elevation(index, latitude=latitude,
                           longitude=longitude) > min_elevation


def robust_z(series, window=25, min_periods=5):
    """
    Distance from a local median, in robust standard deviations.

    A rolling median is used as the reference rather than a mean, and the
    scatter is measured by the median absolute deviation rather than the
    standard deviation, because both statistics are being computed on a record
    that contains the very outliers they are meant to expose. A mean and a
    standard deviation would be dragged towards a spike and would then fail to
    call it one.

    Parameters
    ----------
    series : pd.Series
        Series to score.
    window : int, optional
        Width of the centred rolling window, in samples. Default ``25``, which
        is one day either side of centre on the hourly grid.
    min_periods : int, optional
        Minimum observations in the window. Default ``5``.

    Returns
    -------
    pd.Series
        Signed robust z-score, aligned to ``series``.
    """
    median = series.rolling(window, center=True, min_periods=min_periods).median()
    residual = series - median
    mad = residual.abs().rolling(window, center=True,
                                 min_periods=min_periods).median()
    sigma = 1.4826 * mad
    # Where the record is locally constant the MAD is zero and the ratio would
    # be infinite. The series-wide scatter is the fallback, which is
    # conservative: it under-reports rather than inventing outliers.
    sigma = sigma.replace(0.0, np.nan).fillna(residual.abs().std())
    return residual / sigma


def acquisition_faults(sensor, target='inc_comp',
                       corroborating=('inc', 'tair', 'batt'),
                       window=25, k=8.0):
    """
    Register the single-hour acquisition faults in the calibrated inclination.

    The current era of this archive contains hours in which the inclination
    moves by hundreds of millidegrees and returns on the following sample. The
    cause is not structural and it is not the compensation. At 2025-05-10 14:00
    the raw inclination moves 220 mdeg, the air temperature rises 5.9 degC and
    the battery voltage dips, all within the same hour and all recovering at the
    next one. Three physically independent channels do not fail together for one
    sample because a wall moved; they fail together because the acquisition
    failed. The compensated channel inherits the excursion from the raw one and
    slightly enlarges it wherever the temperature spiked too, so differencing the
    calibrated reading neither creates these events nor escapes them.

    They matter because they live almost entirely in the first difference. A
    single unreversed hour contributes two large differences of opposite sign,
    which is why an unscreened current-era first-difference correlation measures
    the faults rather than the wall.

    This function **marks** them rather than removing them. It returns one row
    per flagged hour carrying the evidence that justifies the flag, so that the
    register can be read, audited and reported. :func:`mask_faults` applies it.

    Parameters
    ----------
    sensor : pd.DataFrame
        Unified sensor frame on the hourly grid, carrying ``target`` and
        whichever of ``corroborating`` are available.
    target : str, optional
        Channel whose outliers define the register. Default ``'inc_comp'``, the
        calibrated reading, because that is the series every statistic in this
        study is computed on.
    corroborating : iterable of str, optional
        Channels checked for a simultaneous excursion. These do not gate the
        flag; they are the evidence recorded beside it.
    window : int, optional
        Rolling window for :func:`robust_z`. Default ``25``.
    k : float, optional
        Robust z above which an hour is flagged. Default ``8.0``. The result is
        insensitive to this choice over a wide range, which is what a genuine
        outlier population looks like.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, one row per flagged hour, with the target's
        z-score and signed step, the number of corroborating channels that were
        simultaneously anomalous, whether the excursion reversed on the
        following hour, and the era. Empty with those columns if nothing is
        flagged.
    """
    columns = ['era', 'z_' + target, 'delta_mdeg', 'n_corroborating',
               'reverses_next_hour']

    if target not in sensor.columns:
        return pd.DataFrame(columns=columns)

    z_target = robust_z(sensor[target], window=window)
    flagged = z_target.abs() > k

    others = [c for c in corroborating
              if c in sensor.columns and c != target]
    z_others = {c: robust_z(sensor[c], window=window) for c in others}

    step = sensor[target].diff()
    next_step = step.shift(-1)

    rows = []
    for stamp in sensor.index[flagged.fillna(False)]:
        corroborated = sum(abs(z_others[c].get(stamp, np.nan)) > k
                           for c in others
                           if pd.notna(z_others[c].get(stamp, np.nan)))
        this, following = step.get(stamp, np.nan), next_step.get(stamp, np.nan)
        reverses = bool(
            pd.notna(this) and pd.notna(following)
            and np.sign(this) != np.sign(following)
            and abs(following) > 0.5 * abs(this))
        rows.append({
            'datetime': stamp,
            'era': 'legacy' if stamp < ERA_BOUNDARY else 'current',
            'z_' + target: round(float(z_target[stamp]), 1),
            'delta_mdeg': round(float(this), 2) if pd.notna(this) else np.nan,
            'n_corroborating': int(corroborated),
            'reverses_next_hour': reverses,
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).set_index('datetime')[columns]


def mask_faults(df, faults, cols=('inc_comp',)):
    """
    Blank the registered fault hours, in levels and therefore in differences.

    The hours are set to ``NaN`` rather than interpolated. An interpolated value
    would be an invention, and the study's statistics already handle missing
    hours: a level statistic skips them and a first difference is not taken
    across them. Masking the level is what removes the fault from both domains
    at once, which is why the register is applied here and not to the
    differenced series.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to mask. Not modified in place.
    faults : pd.DataFrame
        Register from :func:`acquisition_faults`.
    cols : iterable of str, optional
        Columns to blank at the flagged hours. Default ``('inc_comp',)``.

    Returns
    -------
    pd.DataFrame
        A copy with the flagged hours blanked in the named columns.
    """
    out = df.copy()
    stamps = out.index.intersection(faults.index)
    for col in cols:
        if col in out.columns:
            out.loc[stamps, col] = np.nan
    return out


#: The instrument at station st02 was physically reinstalled on this date. The
#: legacy package and the current one share no baseline, so no window and no
#: statistic in this study may span the boundary.
ERA_BOUNDARY = pd.Timestamp('2025-02-21')

#: Column order of the block inventory, fixed so that an empty inventory has the
#: same shape as a populated one and downstream code needs no special case.
BLOCK_COLUMNS = ['tier', 'era', 'block', 'start', 'end', 'days', 'coverage_%']


def tier_blocks(df, tiers, era_boundary=ERA_BOUNDARY, min_days=20,
                max_gap_hours=6):
    """
    Contiguous blocks per tier, within each instrument era.

    A tier is a set of columns that must be simultaneously present. The binding
    channel of this study is the on-structure pyranometer, which exists only
    from the era boundary onwards, so a tier requiring it can never reach the
    legacy era. That asymmetry is not worked around: it is what the tiers exist
    to express.

    Blocks are found within an era rather than across the whole record, because
    the instrument was physically reinstalled at the boundary and the two
    packages share no baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame on a regular hourly index.
    tiers : dict
        Tier name to the list of columns that tier requires.
    era_boundary : pd.Timestamp, optional
        First timestamp of the current era. Default :data:`ERA_BOUNDARY`.
    min_days : int, optional
        Shortest block reported, in days. Default ``20``, the value used by the
        sibling studies, so inventories are directly comparable.
    max_gap_hours : int, optional
        Longest interruption absorbed into a block. Default ``6``.

    Returns
    -------
    pd.DataFrame
        One row per block, with the columns of :data:`BLOCK_COLUMNS`. Empty with
        those same columns if no tier yields a block.
    """
    eras = {
        'legacy': df.loc[df.index < era_boundary],
        'current': df.loc[df.index >= era_boundary],
    }

    rows = []
    for tier, cols in tiers.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < len(cols):
            continue
        for era, frame in eras.items():
            if frame.empty:
                continue
            found = ip.contiguous_blocks(frame, present, min_days=min_days,
                                         max_gap_hours=max_gap_hours)
            for n, (_, block) in enumerate(found.iterrows(), start=1):
                rows.append({
                    'tier': tier,
                    'era': era,
                    'block': f'{tier}-{era[:3]}{n}',
                    'start': block['start'],
                    'end': block['end'],
                    'days': block['days'],
                    'coverage_%': block['coverage_%'],
                })

    if not rows:
        return pd.DataFrame(columns=BLOCK_COLUMNS)
    return (pd.DataFrame(rows)[BLOCK_COLUMNS]
            .sort_values(['tier', 'days'], ascending=[True, False])
            .reset_index(drop=True))


def tier_index(inventory, tier):
    """
    The union of a tier's blocks, as a single sorted index.

    This is the window for every *level* statistic. Differenced statistics must
    not use it directly — see :func:`tier_differences`.

    Parameters
    ----------
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.

    Returns
    -------
    pd.DatetimeIndex
        Sorted and unique. Empty if the tier has no block.
    """
    parts = [pd.date_range(row['start'], row['end'], freq='h')
             for _, row in inventory[inventory['tier'] == tier].iterrows()]
    if not parts:
        return pd.DatetimeIndex([])
    index = parts[0]
    for part in parts[1:]:
        index = index.union(part)
    return index


def tier_frame(df, inventory, tier):
    """
    ``df`` restricted to the union of a tier's blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.

    Returns
    -------
    pd.DataFrame
        Rows of ``df`` inside the tier's blocks, in index order.
    """
    index = tier_index(inventory, tier)
    return df.loc[df.index.intersection(index)]


def tier_differences(df, inventory, tier, cols=None):
    """
    First differences taken inside each block and concatenated.

    A first difference must never be taken across a block boundary. The two
    rows on either side of a boundary are both valid and both present, but they
    are not adjacent in time, and differencing them manufactures a step out of a
    gap. Differencing per block and concatenating afterwards is the only correct
    way to obtain a differenced frame over a tier.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.
    cols : list of str or None, optional
        Columns to difference. Default ``None``, meaning every **numeric**
        column of ``df``. The harmonised frame carries non-numeric channels
        alongside the measurements, and differencing one raises rather than
        returning something useless, so the default excludes them instead of
        requiring every caller to remember.

    Returns
    -------
    pd.DataFrame
        Differences, with the first row of every block dropped.
    """
    cols = (list(df.select_dtypes(include='number').columns) if cols is None
            else list(cols))
    parts = []
    for _, row in inventory[inventory['tier'] == tier].iterrows():
        block = df.loc[row['start']:row['end'], cols]
        if len(block) < 2:
            continue
        parts.append(block.diff().iloc[1:])
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts).sort_index()


def longest_block(inventory, tier):
    """
    The single longest block of a tier.

    Operator scans run on one contiguous block rather than on the union,
    because a transport delay is applied with a shift, a time constant with a
    recursive filter, and the band limit with a rolling mean. Every one of those
    reads across adjacent rows, so a union would let a filter draw values from
    the far side of a multi-month interruption.

    Parameters
    ----------
    inventory : pd.DataFrame
        Output of :func:`tier_blocks`.
    tier : str
        Tier name.

    Returns
    -------
    pd.Series or None
        The inventory row of the longest block, or ``None`` if the tier is
        empty.
    """
    rows = inventory[inventory['tier'] == tier]
    if rows.empty:
        return None
    return rows.loc[rows['days'].idxmax()]


def radiation_phase(series, min_daily_total=200.0, night_baseline=True):
    """
    Observed solar-noon hour per day, as the centroid of the daily radiation.

    The centroid is used rather than the hour of the maximum because the maximum
    is quantised to the hour and jumps by a whole step under a little cloud,
    while the centroid moves smoothly and estimates the phase to a fraction of
    an hour.

    Parameters
    ----------
    series : pd.Series
        A radiation channel on an hourly index.
    min_daily_total : float, optional
        Days whose total falls below this are dropped: an overcast day has no
        well-defined phase. Default ``200``.
    night_baseline : bool, optional
        Subtract each day's minimum before taking the centroid, which removes a
        constant sensor offset. The on-structure channel needs this. Default
        ``True``.

    Returns
    -------
    pd.Series
        One value per day, the observed centroid hour in the series' own clock.
    """
    values = series.dropna()
    if values.empty:
        return pd.Series(dtype=float)

    frame = pd.DataFrame({
        'value': values.to_numpy(dtype=float),
        'day': values.index.normalize(),
        'hour': values.index.hour + values.index.minute / 60.0,
    })
    if night_baseline:
        frame['value'] = frame['value'] - frame.groupby('day')['value'].transform('min')
    frame['value'] = frame['value'].clip(lower=0.0)

    totals = frame.groupby('day')['value'].sum()
    weighted = frame.assign(w=frame['value'] * frame['hour']) \
                    .groupby('day')['w'].sum()
    keep = totals >= min_daily_total
    return (weighted[keep] / totals[keep]).rename('centroid_hour')


def clock_against_sun(df, quantity='sr', sources=SOURCES,
                      longitude=SITE_LONGITUDE, by_season=True,
                      min_daily_total=200.0):
    """
    Each source's clock offset from UTC, measured against computed solar noon.

    A source recording in UTC should show a centroid close to computed solar
    noon. A source recording Italian local time should sit one hour later in
    winter and two in summer, and the split by season is what makes daylight
    saving visible rather than averaged away.

    The centroid of a daily radiation curve sits slightly after solar noon for a
    real sensor, because afternoon convective cloud is more common than morning
    cloud. The comparison is therefore made *between* sources on the same
    quantity, with the computed noon as the common origin, so that shared bias
    cancels.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str, optional
        Radiation channel. Default ``'sr'``.
    sources : iterable of str, optional
        Sources to test.
    longitude : float, optional
        Site longitude, degrees east.
    by_season : bool, optional
        Split winter and summer. Default ``True``.
    min_daily_total : float, optional
        Passed to :func:`radiation_phase`.

    Returns
    -------
    pd.DataFrame
        One row per source and season: the median offset from computed solar
        noon, and the offset rounded to the nearest hour.
    """
    seasons = ([('winter', [12, 1, 2]), ('summer', [6, 7, 8])] if by_season
               else [('all', list(range(1, 13)))])

    rows = []
    for source in sources:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue
        centroid = radiation_phase(df[col], min_daily_total=min_daily_total)
        if centroid.empty:
            continue
        noon = solar_noon_utc(centroid.index, longitude=longitude)
        offset = centroid - noon
        for label, months in seasons:
            sub = offset[offset.index.month.isin(months)]
            if len(sub) < 10:
                continue
            rows.append({
                'source': source,
                'season': label,
                'n_days': int(len(sub)),
                'median_offset_h': round(float(sub.median()), 2),
                'iqr_h': round(float(sub.quantile(0.75) - sub.quantile(0.25)), 2),
                'implied_shift_h': int(np.round(sub.median())),
            })
    return pd.DataFrame(rows)


def apply_seasonal_clock(df, offsets, summer_months=(4, 5, 6, 7, 8, 9)):
    """
    Shift each source by a different amount inside and outside summer.

    The measured offsets between the sources are zero in winter and one hour in
    summer, which is the signature of a daylight-saving mismatch: one clock
    advances in spring and the other does not. A single annual shift would be
    wrong in both halves of the year, so the correction is applied by season.

    The boundary is taken as the European summer-time period rather than the
    meteorological seasons used elsewhere in this study, because that is what
    the mismatch actually follows.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    offsets : dict
        Source to ``(winter_shift, summer_shift)`` in steps.
    summer_months : tuple of int, optional
        Months treated as summer time. Default April through September, which
        brackets the European changeover dates in late March and late October.

    Returns
    -------
    pd.DataFrame
        A copy with the shifts applied.
    """
    out = df.copy()
    summer = pd.Series(out.index.month.isin(summer_months), index=out.index)
    for source, (winter_shift, summer_shift) in offsets.items():
        columns = [c for c in out.columns if c.endswith(f'_{source}')]
        if not columns:
            continue
        shifted_winter = out[columns].shift(winter_shift)
        shifted_summer = out[columns].shift(summer_shift)
        mask = pd.DataFrame(
            np.repeat(summer.to_numpy()[:, None], len(columns), axis=1),
            index=out.index, columns=columns)
        out[columns] = shifted_winter.where(~mask, shifted_summer)
    return out


def clock_evidence(df, reference='str'):
    """
    Both clock tests side by side, with the disagreement made explicit.

    Two independent tests are available and they do not have to agree. The
    astronomical test is absolute and depends on a radiation channel; the
    cross-correlation test is relative and can use any channel. Where a source's
    radiation and temperature imply different clocks, the channel is defective
    and the study needs to know which one to trust.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    reference : str, optional
        Source for the relative test. Default ``'str'``.

    Returns
    -------
    pd.DataFrame
        One row per source and season, carrying both estimates.
    """
    sun = clock_against_sun(df, quantity='sr')
    rel_t = clock_offset(df, quantity='tair', reference=reference)
    rel_s = clock_offset(df, quantity='sr', reference=reference)

    out = sun.rename(columns={'implied_shift_h': 'sun_shift_h',
                              'median_offset_h': 'sun_offset_h'})
    out = out[['source', 'season', 'n_days', 'sun_offset_h', 'sun_shift_h']]

    for frame, label in ((rel_t, 'tair'), (rel_s, 'sr')):
        if frame.empty:
            continue
        keyed = frame.set_index(['source', 'season'])
        out[f'rel_{label}_shift_h'] = [
            keyed['best_shift_h'].get((s, e), np.nan)
            for s, e in zip(out['source'], out['season'])]
        out[f'rel_{label}_r'] = [
            keyed['r_at_best'].get((s, e), np.nan)
            for s, e in zip(out['source'], out['season'])]

    return out


# ──────────────────────────────────────────────────────────────────────
# The data dictionary
# ──────────────────────────────────────────────────────────────────────

def write_data_dictionary(path, report=None, extents=None, notes=None):
    """
    Write the canonical channel reference to a Markdown file.

    Generated from the mapping tables in this module rather than maintained by
    hand, so the documentation and the code cannot drift apart. Any study that
    consumes these channels reads this file.

    Parameters
    ----------
    path : str
        Destination, normally ``docs/proxy-data-dictionary.md``.
    report : pd.DataFrame, optional
        Output of :func:`harmonisation_report`, for measured extent and
        coverage.
    extents : dict, optional
        Source to a human-readable extent string.
    notes : dict, optional
        Channel name to a defect note.

    Returns
    -------
    str
        The path written.
    """
    notes = notes or {}
    maps = {'str': STR_MAP, 'gs': GS_MAP, 'era5': ERA5_MAP}

    lines = [
        '# Proxy data dictionary',
        '',
        'Canonical channel names for every environmental source at Gubbio.',
        '**Generated by `studies/proxy_comparison/pc_lib.write_data_dictionary`',
        '— do not edit by hand.**',
        '',
        'Naming is `{quantity}_{source}`. The source suffix is always present, so',
        'a column name states where the number came from without reference to a',
        'docstring.',
        '',
        '## Sources',
        '',
        '| Suffix | Source | What it measures |',
        '|---|---|---|',
    ]
    for source in SOURCES:
        extent = f' Extent: {extents[source]}.' if extents and source in extents else ''
        lines.append(f'| `{source}` | {SOURCE_LABEL[source]} | '
                     f'{SOURCE_DESCRIPTION[source]}{extent} |')

    lines += [
        '',
        '**They are not replicates.** A housing on a sun-exposed wall, a standard',
        'screen in town and a nine-kilometre grid average are three different',
        'quantities that happen to share a name. Disagreement between them is',
        'expected and is measured in `studies/proxy_comparison/`.',
        '',
        '## Site',
        '',
        'The coordinates below fix both the point at which the ERA5 grid cell',
        'was extracted and the solar geometry used by the clock tests and the',
        'daylight mask. They must always be changed together.',
        '',
        '| Field | Value |',
        '|---|---|',
        f'| Latitude | {SITE_LATITUDE:.6f} degrees north |',
        f'| Longitude | {SITE_LONGITUDE:.6f} degrees east |',
        '| Origin | `auxiliary/oiko.py`, lines 84 and 85 |',
        '',
        '## Channels',
        '',
        '| Canonical | Quantity | Unit | Source | Native column | Coverage |',
        '|---|---|---|---|---|---|',
    ]
    for source in SOURCES:
        for quantity, native in maps[source].items():
            column = f'{quantity}_{source}'
            coverage = ''
            if report is not None and column in report.index:
                coverage = f'{report.loc[column, "coverage_%"]:.1f} %'
            lines.append(
                f'| `{column}` | {quantity} | {QUANTITY_UNIT.get(quantity, "")} '
                f'| {SOURCE_LABEL[source]} | `{native}` | {coverage} |')

    lines += [
        '',
        '## Conversions applied on load',
        '',
        '| Channel | Conversion | Why |',
        '|---|---|---|',
        '| `rh_era5` | × 100 | ERA5 reports relative humidity as a fraction '
        'where every other source reports per cent. Left uncorrected it would '
        'appear as a 99 % dry bias. |',
        '| all `_gs` | 30 min → hourly mean | The ground station is natively '
        'half-hourly. |',
        '',
        '## The clock',
        '',
        'The sources do not share a clock, and the offset is **measured, not',
        'assumed**. Two independent tests are run in',
        '`studies/proxy_comparison/`: an absolute one against computed solar',
        'noon, and a relative one by cross-correlation. Where they disagree for a',
        'channel, that channel is defective.',
        '',
        '## Known defects',
        '',
        '| Channel | Defect |',
        '|---|---|',
    ]
    for column, note in sorted(notes.items()):
        lines.append(f'| `{column}` | {note} |')

    lines += ['', '## The target', '',
              'The compensated inclination `inc_comp` is the **calibrated',
              'reading**: the manufacturer\'s formula applied with the',
              'temperature measured at the transducer. It is an instrument',
              'calibration, not a regression, and no study re-estimates its',
              'coefficient. The only permitted adjustment is the levelling',
              'correction across the 2025-02-21 regime change, published as',
              '`inc_comp_joined`.', '']

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def plot_clock(profiles, evidence=None, title='', save_path=None,
               filename=None):
    """
    Mean diurnal radiation profile per source and season, with the clock result.

    Parameters
    ----------
    profiles : pd.DataFrame
        Output of :func:`peak_hour_profile`.
    evidence : pd.DataFrame, optional
        Output of :func:`clock_evidence`, annotated in the caption area.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.4), sharey=True)
    palette = dict(zip(SOURCES, sns.color_palette('colorblind', len(SOURCES))))
    for ax, season in zip(axes, ('winter', 'summer')):
        for column in profiles.columns:
            source, _, label = column.partition('·')
            if label != season:
                continue
            ax.plot(profiles.index, profiles[column], marker='o', ms=3, lw=1.3,
                    color=palette.get(source), label=SOURCE_LABEL.get(source))
        ax.set_title(season)
        ax.set_xlabel('hour of day')
    axes[0].set_ylabel('mean value')
    axes[0].legend(fontsize='small')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_agreement(df, quantity, reference='str', compared=('gs', 'era5'),
                   title='', save_path=None, filename=None):
    """
    Scatter and Bland-Altman for each comparison against the reference.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame, clock corrected.
    quantity : str
        Canonical quantity name.
    reference : str, optional
        Source on the x axis.
    compared : tuple of str, optional
        Sources compared against it.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    others = [s for s in compared if f'{quantity}_{s}' in df.columns]
    fig, axes = plt.subplots(2, len(others), figsize=figsize(11, 6.0),
                             squeeze=False)
    palette = dict(zip(SOURCES, sns.color_palette('colorblind', len(SOURCES))))
    ref_col = f'{quantity}_{reference}'

    for k, source in enumerate(others):
        col = f'{quantity}_{source}'
        joined = df[[ref_col, col]].dropna()

        ax = axes[0, k]
        ax.scatter(joined[ref_col], joined[col], s=1, alpha=0.05,
                   color=palette.get(source), rasterized=True)
        lo = float(min(joined[ref_col].min(), joined[col].min()))
        hi = float(max(joined[ref_col].max(), joined[col].max()))
        ax.plot([lo, hi], [lo, hi], color='0.3', lw=1.0, ls='--')
        ax.set_xlabel(f'{SOURCE_LABEL[reference]} [{QUANTITY_UNIT.get(quantity, "")}]')
        ax.set_ylabel(f'{SOURCE_LABEL[source]}')
        ax.set_title(f'{SOURCE_LABEL[source]} against {SOURCE_LABEL[reference]}')

        ax = axes[1, k]
        mean = (joined[ref_col] + joined[col]) / 2.0
        diff = joined[col] - joined[ref_col]
        ax.scatter(mean, diff, s=1, alpha=0.05, color=palette.get(source),
                   rasterized=True)
        for level, style in ((diff.mean(), '-'),
                             (diff.mean() + 1.96 * diff.std(), '--'),
                             (diff.mean() - 1.96 * diff.std(), '--')):
            ax.axhline(level, color='#A5202B', lw=1.0, ls=style)
        ax.set_xlabel('mean of the two')
        ax.set_ylabel('difference')

    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def densest_window(df, cols, days=14):
    """
    The window in which every listed column is most completely present.

    Used to choose an honest zoom: a window picked by eye would invite the
    suspicion that it was picked to flatter, and a window picked at random can
    land in an outage and show nothing.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    cols : list of str
        Columns required simultaneously.
    days : int, optional
        Window length in days. Default ``14``.

    Returns
    -------
    tuple of pd.Timestamp or None
        Start and end of the window, or ``None`` if no column set is present.
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    joint = df[present].notna().all(axis=1)
    if not joint.any():
        return None
    width = int(days * 24)
    counted = joint.rolling(width, min_periods=1).sum()
    end = counted.idxmax()
    return end - pd.Timedelta(hours=width - 1), end


def plot_series_by_source(df, quantity, sources=SOURCES, span='overlap',
                          zoom=None, zoom_days=14, envelope='30D', title='',
                          save_path=None, filename=None):
    """
    The measured series itself, per source, over the whole record and up close.

    Every other figure in this study reports a statistic: a mean diurnal
    profile, an agreement scatter, a correlation. Those hide what the raw
    channel looks like, and for solar radiation the raw channel is where the
    on-structure defect is directly visible. This figure plots the values.

    The top panels give one source each over the full span, at hourly
    resolution, with a thirty-day rolling maximum drawn over them. Eight years
    of hourly radiation is denser than the page can resolve, so the hourly cloud
    carries the range and the rolling maximum carries the seasonal ceiling;
    neither is an average over sources, and neither smooths a value away. Hours
    with no measurement are left as gaps rather than joined, so an outage reads
    as an outage. The bottom panel overlays all three at full resolution across
    a short window, which is the only scale at which individual hourly values
    can actually be read.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame, clock corrected.
    quantity : str
        Canonical quantity name.
    sources : iterable of str, optional
        Sources to draw, in order.
    span : str or tuple, optional
        Horizontal extent. ``'overlap'``, the default, restricts every panel to
        the window in which all drawn sources have some data, which for solar
        radiation is the on-structure record and is the only period over which
        the three can be compared at all. ``'full'`` uses the whole index, which
        shows how much longer the proxy records are but leaves most of the
        figure empty for the on-structure panel. A tuple is used as given.
    zoom : tuple or None, optional
        ``(start, end)`` for the detail panel, anything the index accepts.
        Default ``None``, which selects the window where all available sources
        are most completely present, via :func:`densest_window`.
    zoom_days : int, optional
        Width of the automatically chosen window, in days. Default ``14``.
    envelope : str, optional
        Rolling window for the maximum drawn over the values. Default
        ``'30D'``.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    available = [s for s in sources if f'{quantity}_{s}' in df.columns
                 and df[f'{quantity}_{s}'].notna().any()]
    if not available:
        raise ValueError(f'no source carries {quantity}')

    cols = [f'{quantity}_{s}' for s in available]
    unit = QUANTITY_UNIT.get(quantity, '')
    palette = dict(zip(SOURCES, sns.color_palette('colorblind', len(SOURCES))))

    if span == 'overlap':
        limits = (max(df[c].first_valid_index() for c in cols),
                  min(df[c].last_valid_index() for c in cols))
    elif span == 'full':
        limits = (df.index.min(), df.index.max())
    else:
        limits = (pd.Timestamp(span[0]), pd.Timestamp(span[1]))

    df = df.loc[limits[0]:limits[1]]

    if zoom is None:
        zoom = densest_window(df, cols, days=zoom_days)

    n = len(available)
    fig, axes = plt.subplots(n + 1, 1, figsize=figsize(11, 1.9 * n + 2.6))
    for ax, source in zip(axes[:n], available):
        col = f'{quantity}_{source}'
        # Plotted with the missing hours left as NaN rather than dropped, so
        # that an outage appears as a break. Dropping them would join the two
        # sides of a ten-month interruption with a straight line and show data
        # where there is none.
        series = df[col]
        colour = palette.get(source)
        ax.plot(series.index, series.values, lw=0.3, alpha=0.55, color=colour)
        # A thirty-day rolling maximum, not a daily one: the daily maximum on
        # this span is as dense as the data it summarises and hides it, while
        # the monthly envelope reads as the seasonal ceiling it is meant to be.
        ceiling = series.rolling(envelope, min_periods=24).max()
        ax.plot(ceiling.index, ceiling.values, lw=0.9, color='0.15')
        ax.set_xlim(*limits)
        ax.set_ylabel(f'[{unit}]')
        coverage = 100.0 * series.notna().mean()
        ax.set_title(
            f'{SOURCE_LABEL[source]} — {coverage:.1f} % of this window, '
            f'peak {series.max():.0f} {unit}, '
            f'median {series[series > 0].median():.0f} {unit} when positive',
            fontsize=8, loc='left')
        if limits[0] < ERA_BOUNDARY < limits[1]:
            ax.axvline(ERA_BOUNDARY, color='0.35', lw=0.8, ls='--')
        if zoom is not None:
            ax.axvspan(zoom[0], zoom[1], color='0.2', alpha=0.18, lw=0)

    detail = axes[n]
    if zoom is not None:
        window = df.loc[zoom[0]:zoom[1]]
        for source in available:
            series = window[f'{quantity}_{source}']
            if series.notna().sum() == 0:
                continue
            detail.plot(series.index, series.values, lw=1.0,
                        color=palette.get(source), label=SOURCE_LABEL[source])
        detail.set_xlim(zoom[0], zoom[1])
        detail.legend(fontsize='small', ncol=len(available))
        detail.set_title(
            f'hourly values, {zoom[0]:%Y-%m-%d} to {zoom[1]:%Y-%m-%d}',
            fontsize=8, loc='left')
        detail.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        detail.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    detail.set_ylabel(f'[{unit}]')

    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_diurnal_profiles(df, quantity, title='', save_path=None,
                          filename=None):
    """
    Mean diurnal cycle per source, overlaid.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame, clock corrected.
    quantity : str
        Canonical quantity name.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.4))
    palette = dict(zip(SOURCES, sns.color_palette('colorblind', len(SOURCES))))

    for source in SOURCES:
        col = f'{quantity}_{source}'
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        profile = series.groupby(series.index.hour).mean()
        axes[0].plot(profile.index, profile.values, marker='o', ms=3, lw=1.4,
                     color=palette.get(source), label=SOURCE_LABEL[source])
        anomaly = profile - profile.mean()
        axes[1].plot(anomaly.index, anomaly.values, marker='o', ms=3, lw=1.4,
                     color=palette.get(source), label=SOURCE_LABEL[source])

    axes[0].set_ylabel(f'mean [{QUANTITY_UNIT.get(quantity, "")}]')
    axes[1].set_ylabel('departure from the daily mean')
    for ax in axes:
        ax.set_xlabel('hour of day')
    axes[0].legend(fontsize='small')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_target_relationship(relationship, title='', save_path=None,
                             filename=None):
    """
    Correlation and slope against the target, per source and domain.

    Parameters
    ----------
    relationship : pd.DataFrame
        Output of :func:`target_relationship`, possibly for several quantities.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.4))
    palette = dict(zip(SOURCES, sns.color_palette('colorblind', len(SOURCES))))
    domains = ['levels', 'differences']
    width = 0.25

    for k, source in enumerate(SOURCES):
        rows = relationship[relationship['source'] == source]
        if rows.empty:
            continue
        positions = np.arange(len(domains)) + (k - 1) * width
        values = [float(rows[rows['domain'] == d]['r'].iloc[0])
                  if (rows['domain'] == d).any() else np.nan for d in domains]
        axes[0].bar(positions, values, width, color=palette.get(source),
                    label=SOURCE_LABEL[source])
        slopes = [float(rows[rows['domain'] == d]['slope_mdeg_per_unit'].iloc[0])
                  if (rows['domain'] == d).any() else np.nan for d in domains]
        errors = [float(rows[rows['domain'] == d]['slope_se'].iloc[0])
                  if (rows['domain'] == d).any() else np.nan for d in domains]
        axes[1].bar(positions, slopes, width, yerr=errors, capsize=3,
                    color=palette.get(source), label=SOURCE_LABEL[source])

    for ax, label in ((axes[0], 'correlation with the target'),
                      (axes[1], 'slope [mdeg per unit]')):
        ax.set_xticks(np.arange(len(domains)))
        ax.set_xticklabels(domains)
        ax.set_ylabel(label)
        ax.axhline(0, color='0.3', lw=0.8)
    axes[0].legend(fontsize='small')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)
