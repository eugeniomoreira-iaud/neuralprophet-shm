"""
Module: lc_lib.py

Support library for the legacy-era thermal-compensation study
(``studies/thermal_compensation_legacy/``).

This study repeats the question asked in ``studies/thermal_compensation`` — is the
fixed thermal correction good for the prediction system? — on the legacy
instrument era, where only air temperature and relative humidity are available as
candidate regressors. It adds a NeuralProphet decomposition of every series and a
NeuralProphet-based imputation of the gaps.

**Shared code.** The generic analysis machinery — correlation and lag structure,
slope fitting, the compensation acceptance tests, the coefficient sweep, the
forecasting experiment and the figures — is imported from the sibling study rather
than duplicated. Using literally the same functions is what makes the two studies
comparable: a difference in results is then a difference in the data, not a
difference in the tests. Only what is genuinely specific to the legacy era is
implemented here: the 14-column parser, the completeness scan used to choose a year,
and the NeuralProphet decomposition and imputation.

Nothing in this module writes to the raw archive.
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The sibling study is the single source of truth for the shared analysis.
_SIBLING = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', 'thermal_compensation'))
if _SIBLING not in sys.path:
    sys.path.insert(0, _SIBLING)

import tc_lib as tc                                            # noqa: E402
from tc_lib import (                                           # noqa: E402,F401
    _to_float, sync_cache, screen_spikes, coverage_table, valid_windows,
    to_hourly, sampling_hours,
    compensate, compensation_term, correlation_matrix, cross_correlation,
    peak_lags, fit_slope, slope_table, multivariate_fit, diurnal_amplitude,
    thermal_lag_filter, inertia_scan, best_inertia, add_lagged, add_inertia,
    compensation_tests, signed_correlation_check, coefficient_sweep,
    build_supervised, run_ridge_experiment, run_neuralprophet_experiment,
    set_context, figsize,
    plot_overview, plot_diurnal, plot_correlation_heatmaps,
    plot_cross_correlation, plot_lag_scan, plot_inertia_scan,
    plot_thermal_scatter, plot_compensation_effect,
    plot_coefficient_sweep, plot_experiment,
    INCLINOMETER_ZERO_MV, INCLINOMETER_RANGE_MDEG, DOCUMENTED_COEFF,
    SAMPLING, ANALYSIS_FREQ, MIN_SAMPLES_PER_HOUR, N_COLS_LEGACY,
)

#: Legacy column names in file order. Blocks appear in station order:
#: b1 = st01, b2 = st02, b3 = st03 (docs/raw-data-format.md, section 3.3).
LEGACY_COLUMNS = [
    'st01_batt', 'st01_tair', 'st01_rh', 'st01_i',
    'st02_batt', 'st02_tair', 'st02_rh', 'st02_i',
    'st03_batt', 'st03_tair', 'st03_rh', 'st03_i',
]

#: Channels in which 0.000 is a missing-data sentinel. In the legacy era this is
#: every channel — there is no solar radiation column to exempt.
LEGACY_ZERO_IS_SENTINEL = LEGACY_COLUMNS


# ──────────────────────────────────────────────────────────────────────
# Loading — 14-column legacy era
# ──────────────────────────────────────────────────────────────────────

def parse_legacy_file(path):
    """
    Parse one legacy-era ``.adc`` file.

    Differs from the current-era parser in the expected field count and in
    tolerating the defects that are concentrated in this era: duplicate
    timestamps whose copies disagree, decimal separators that change within a
    single file, and records that belong to the previous calendar day.

    Parameters
    ----------
    path : str
        Path to a ``.adc`` file.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, columns as :data:`LEGACY_COLUMNS`. Empty if the
        file holds no legacy-era records.
    """
    rows, stamps = [], []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != N_COLS_LEGACY:
                continue
            try:
                ts = pd.to_datetime(parts[0] + ' ' + parts[1],
                                    format='%d/%m/%y %H:%M:%S')
            except ValueError:
                continue
            stamps.append(ts)
            rows.append([_to_float(p) for p in parts[2:]])

    if not rows:
        return pd.DataFrame(columns=LEGACY_COLUMNS)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(stamps),
                        columns=LEGACY_COLUMNS)


def load_legacy(archive_dir, cache_dir, start, end, station='st02',
                verbose=True, freq=ANALYSIS_FREQ):
    """
    Load one legacy station over a date range, on the analysis grid.

    Duplicate timestamps are merged field-wise, taking the first non-null value
    in each column. This matters far more here than in the current era: a third
    of legacy files contain duplicated timestamps, and where the copies differ
    it is usually because one carries real values in a block where the other
    carries zeros. Dropping whole records would discard real measurements.

    Parameters
    ----------
    archive_dir : str
        Read-only ``.adc`` archive.
    cache_dir : str
        Local working copy directory.
    start, end : str or pd.Timestamp
        Inclusive date bounds.
    station : str, optional
        Station prefix to extract. Default ``'st02'``.
    verbose : bool, optional
        Print a loading report. Default ``True``.
    freq : str or None, optional
        Aggregation interval, applied by :func:`tc_lib.to_hourly`. Default
        :data:`tc_lib.ANALYSIS_FREQ`, matching the hourly rate of every
        external proxy. Pass ``None`` for the native 20-minute record.

    Returns
    -------
    pd.DataFrame
        Columns ``batt``, ``tair``, ``rh``, ``inc`` on a regular grid at
        ``freq``, with ``inc`` in millidegrees about the instrument's
        calibrated zero.
    """
    paths = sync_cache(archive_dir, cache_dir, start, end, verbose=verbose)

    frames = [parse_legacy_file(p) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError('No legacy-era records found in the requested range.')
    raw = pd.concat(frames).sort_index()
    n_records = len(raw)

    dup_mask = raw.index.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    conflicts = 0
    if n_dup:
        conflicts = int(raw[dup_mask].groupby(level=0).nunique().gt(1).sum().sum())
    merged = raw.groupby(level=0).first()

    cols = {f'{station}_batt': 'batt', f'{station}_tair': 'tair',
            f'{station}_rh': 'rh', f'{station}_i': 'i_mv'}
    df = merged[list(cols)].rename(columns=cols).copy()

    n_zero = 0
    for col in df.columns:
        hit = df[col] == 0.0
        n_zero += int(hit.sum())
        df.loc[hit, col] = np.nan

    df['inc'] = df['i_mv'] - INCLINOMETER_ZERO_MV
    df = df.drop(columns=['i_mv'])

    n_range = int((df['inc'].abs() > INCLINOMETER_RANGE_MDEG).sum())
    df.loc[df['inc'].abs() > INCLINOMETER_RANGE_MDEG, 'inc'] = np.nan

    grid = pd.date_range(pd.Timestamp(start).floor('D'),
                         pd.Timestamp(end).ceil('D'), freq=SAMPLING)
    df = df.reindex(grid)
    df.index.name = 'datetime'

    if verbose:
        print(f'  parsed    : {n_records} records from {len(paths)} files')
        print(f'  duplicates: {n_dup} records share a timestamp, '
              f'{conflicts} field-level conflicts merged')
        print(f'  sentinels : {n_zero} zeros to NaN, '
              f'{n_range} inclinometer out of range')
        print(f'  grid      : {len(df)} slots, {df.index.min()} '
              f'to {df.index.max()}')
    if freq:
        df = to_hourly(df, freq=freq, verbose=verbose)
    return df


def year_completeness(archive_dir, cache_dir, years, station='st02',
                      required=('inc', 'tair', 'rh')):
    """
    Rank calendar years by how complete one station's record is.

    Whole-day file counts are a poor guide, because a day can be present as a
    file while the station's own block is written as zeros throughout. This
    measures what the study actually needs: grid slots in which every required
    channel carries a real value.

    Parameters
    ----------
    archive_dir : str
        Read-only ``.adc`` archive.
    cache_dir : str
        Local working copy directory.
    years : iterable of int
        Calendar years to evaluate.
    station : str, optional
        Station to measure. Default ``'st02'``.
    required : tuple of str, optional
        Channels that must all be present. Default ``('inc', 'tair', 'rh')``.

    Returns
    -------
    pd.DataFrame
        One row per year, sorted by completeness, with per-channel counts and
        the longest contiguous complete run.
    """
    rows = []
    for y in years:
        df = load_legacy(archive_dir, cache_dir, f'{y}-01-01', f'{y}-12-31',
                         station=station, verbose=False)
        df = df.loc[f'{y}-01-01':f'{y}-12-31 23:59']
        ok = df[list(required)].notna().all(axis=1)
        per_day = int(round(24 / sampling_hours(df)))
        # Longest contiguous run of complete slots.
        grp = (~ok).cumsum()[ok]
        longest = int(ok[ok].groupby(grp).size().max()) if ok.any() else 0
        row = {'year': y, 'slots': len(df), 'complete': int(ok.sum()),
               'complete_%': round(100 * ok.mean(), 2),
               'longest_run_days': round(longest / per_day, 1)}
        for c in required:
            row[f'{c}_%'] = round(100 * df[c].notna().mean(), 2)
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values('complete_%', ascending=False)
            .set_index('year'))


# ──────────────────────────────────────────────────────────────────────
# NeuralProphet decomposition and imputation
# ──────────────────────────────────────────────────────────────────────

def _quiet_neuralprophet():
    """Silence NeuralProphet and Lightning chatter."""
    from neuralprophet import set_log_level
    set_log_level('ERROR')
    for name in ('pytorch_lightning', 'lightning.pytorch', 'NP.forecaster'):
        logging.getLogger(name).setLevel(logging.ERROR)


def decompose_series(series, freq=ANALYSIS_FREQ, epochs=25, yearly=True,
                     weekly=True, daily=True, verbose=False):
    """
    Decompose one series into trend and seasonal components with NeuralProphet.

    Fitted without autoregression so that every component is a function of time
    alone. That is what makes the fit usable for imputation as well: a model
    with autoregressive terms cannot produce a value inside a gap, because the
    lags it needs are themselves missing.

    Parameters
    ----------
    series : pd.Series
        Series on a datetime index; missing values may be present.
    freq : str, optional
        Sampling frequency. Default 20 minutes.
    epochs : int, optional
        Training epochs. Default ``25``.
    yearly, weekly, daily : bool, optional
        Seasonal terms to include.
    verbose : bool, optional
        Let NeuralProphet print progress. Default ``False``.

    Returns
    -------
    components : pd.DataFrame
        Indexed like ``series``, with a column per fitted component plus
        ``yhat`` and ``observed``.
    model : NeuralProphet
        The fitted model.
    """
    from neuralprophet import NeuralProphet
    if not verbose:
        _quiet_neuralprophet()

    obs = series.dropna()
    train = pd.DataFrame({'ds': obs.index, 'y': obs.values})

    model = NeuralProphet(
        n_lags=0, n_forecasts=1,
        yearly_seasonality=yearly, weekly_seasonality=weekly,
        daily_seasonality=daily,
        epochs=epochs, learning_rate=0.01, impute_missing=False,
    )
    # NeuralProphet 0.8.0 raises `UnboundLocalError: conditional_cols` when a
    # prediction frame ends in NaN and `impute_missing` is off: the variable is
    # bound only inside the imputation branch, but it is read again when the
    # trailing NaN rows are restored. The trailing gap is therefore trimmed
    # before predicting and the components are reindexed back onto the full
    # span afterwards, which leaves those rows empty — the honest result, since
    # nothing can be fitted where the record has already ended.
    last_valid = series.last_valid_index()
    full = pd.DataFrame({'ds': series.index, 'y': series.values})
    if last_valid is not None:
        full = full[full['ds'] <= last_valid]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(train, freq=freq, progress=None)
        fcst = model.predict(full)

    fcst = fcst.set_index('ds')
    keep = [c for c in fcst.columns
            if c == 'trend' or c.startswith('season_')]
    out = fcst[keep].reindex(series.index)
    out['yhat'] = fcst['yhat1'].reindex(series.index)
    out['observed'] = series.values
    return out, model


def impute_series(series, components=None, freq=ANALYSIS_FREQ, epochs=25,
                  verbose=False):
    """
    Fill gaps in a series with its own NeuralProphet trend-plus-seasonal fit.

    The filled values carry no information the model did not already have: they
    are the trend and seasonal expectation at that timestamp. That is
    appropriate for a regressor, whose role is to describe the environment
    rather than to be predicted, and it is what allows a model requiring
    contiguous input to run across a short gap. It is **not** appropriate for
    the target, and this study never imputes the inclinometer.

    Parameters
    ----------
    series : pd.Series
        Series with gaps.
    components : pd.DataFrame or None, optional
        Output of :func:`decompose_series`, reused if already computed.
    freq : str, optional
        Sampling frequency.
    epochs : int, optional
        Training epochs if a fit is needed.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    filled : pd.Series
        Series with gaps filled.
    imputed_mask : pd.Series
        Boolean mask marking the filled positions.
    """
    if components is None:
        components, _ = decompose_series(series, freq=freq, epochs=epochs,
                                         verbose=verbose)
    mask = series.isna()
    filled = series.copy()
    filled[mask] = components['yhat'].reindex(series.index)[mask]
    return filled, mask


def gap_profile(series, freq_minutes=60):
    """
    Summarise the gap structure of a series.

    The bands are expressed in slots and converted to hours from
    ``freq_minutes``, so the same function describes the native record and the
    hourly analysis grid without retuning.

    Parameters
    ----------
    series : pd.Series
        Series on a regular grid.
    freq_minutes : int, optional
        Sampling interval in minutes. Default ``60``, the analysis grid.

    Returns
    -------
    pd.DataFrame
        One row per gap-length band with counts and total slots lost.
    """
    miss = series.isna()
    if not miss.any():
        return pd.DataFrame(columns=['band', 'gaps', 'slots', 'hours'])
    grp = (~miss).cumsum()[miss]
    lengths = miss[miss].groupby(grp).size()
    per_hour = 60 / freq_minutes
    bands = [(1, 1, 'single slot'),
             (2, int(round(6 * per_hour)), 'up to 6 h'),
             (int(round(6 * per_hour)) + 1, int(round(24 * per_hour)), '6-24 h'),
             (int(round(24 * per_hour)) + 1, int(round(168 * per_hour)),
              '1-7 days'),
             (int(round(168 * per_hour)) + 1, 10 ** 9, 'over 7 days')]
    rows = []
    for lo, hi, label in bands:
        sel = lengths[(lengths >= lo) & (lengths <= hi)]
        if len(sel):
            rows.append({'band': label, 'gaps': int(len(sel)),
                         'slots': int(sel.sum()),
                         'hours': round(sel.sum() * freq_minutes / 60, 1)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Figures specific to this study
# ──────────────────────────────────────────────────────────────────────

def plot_decomposition(components, name, save_path=None, filename=None):
    """
    Draw the NeuralProphet decomposition of one series.

    Parameters
    ----------
    components : pd.DataFrame
        Output of :func:`decompose_series`.
    name : str
        Series name, used in titles.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    comp_cols = [c for c in components.columns
                 if c == 'trend' or c.startswith('season_')]
    n = len(comp_cols) + 1
    fig, axes = plt.subplots(n, 1, figsize=tc.figsize(13, 2.1 * n),
                             sharex=False)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    sns.lineplot(x=components.index, y=components['observed'].values, ax=ax,
                 linewidth=0.5, label='observed')
    sns.lineplot(x=components.index, y=components['yhat'].values, ax=ax,
                 linewidth=0.7, label='fitted')
    ax.set_ylabel(name)
    ax.legend()
    ax.set_title(f'{name} — NeuralProphet decomposition')

    for ax, c in zip(axes[1:], comp_cols):
        if c == 'season_daily':
            hour = components.index.hour + components.index.minute / 60
            prof = components[c].groupby(hour).mean()
            sns.lineplot(x=prof.index, y=prof.values, ax=ax)
            ax.set_xlabel('Hour of day')
            ax.set_xticks(range(0, 25, 3))
        elif c == 'season_weekly':
            dow = components.index.dayofweek + components.index.hour / 24
            prof = components[c].groupby(dow).mean()
            sns.lineplot(x=prof.index, y=prof.values, ax=ax)
            ax.set_xlabel('Day of week')
        else:
            sns.lineplot(x=components.index, y=components[c].values, ax=ax,
                         linewidth=0.9)
        ax.set_ylabel(c)

    sns.despine(fig=fig)
    return tc._finish(fig, save_path, filename)


def plot_imputation(original, filled, mask, title='', save_path=None,
                    filename=None):
    """
    Show which points were imputed and what was put there.

    Parameters
    ----------
    original : pd.Series
        Series before imputation.
    filled : pd.Series
        Series after imputation.
    mask : pd.Series
        Boolean mask of imputed positions.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    palette = sns.color_palette()
    per_day = int(round(24 / tc.sampling_hours(filled)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=tc.figsize(13, 6))
    sns.lineplot(x=filled.index, y=filled.values, ax=ax1, linewidth=0.5,
                 color=palette[1], label='imputed values')
    sns.lineplot(x=original.index, y=original.values, ax=ax1, linewidth=0.5,
                 color=palette[0], label='observed')
    ax1.legend()
    ax1.set_title(title or 'Imputation over the full record')

    # Zoom on the largest gap so the fill can actually be inspected.
    if mask.any():
        grp = (~mask).cumsum()[mask]
        sizes = mask[mask].groupby(grp).size()
        big = sizes.idxmax()
        idx = mask[mask].groupby(grp).apply(lambda s: s.index)[big]
        pad = 3 * per_day
        lo = max(0, filled.index.get_loc(idx[0]) - pad)
        hi = min(len(filled), filled.index.get_loc(idx[-1]) + pad)
        seg = slice(lo, hi)
        sns.lineplot(x=filled.index[seg], y=filled.iloc[seg].values, ax=ax2,
                     linewidth=1.1, color=palette[1], label='imputed')
        sns.lineplot(x=original.index[seg], y=original.iloc[seg].values,
                     ax=ax2, linewidth=1.1, color=palette[0], label='observed')
        hours = len(idx) * 24 / per_day
        ax2.set_title(f'Largest gap ({len(idx)} slots, {hours:.1f} h) '
                      f'with three days either side')
        ax2.legend()
    sns.despine(fig=fig)
    return tc._finish(fig, save_path, filename)


def plot_year_completeness(table, save_path=None, filename=None):
    """
    Bar chart of per-year completeness for the candidate years.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`year_completeness`.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tc.figsize(12, 4.2))
    t = table.sort_index()
    chans = [c for c in t.columns if c.endswith('_%') and c != 'complete_%']
    long = (t[chans].reset_index()
            .melt(id_vars='year', var_name='channel', value_name='available'))
    sns.barplot(data=long, x='year', y='available', hue='channel', ax=ax1)
    ax1.set_ylabel('Available [%]')
    ax1.set_xlabel('year')
    ax1.set_title('Per-channel availability by year')

    sns.barplot(x=t.index.astype(str), y=t['complete_%'].values, ax=ax2,
                color=sns.color_palette()[2])
    for i, v in enumerate(t['complete_%']):
        ax2.text(i, v, f' {v:.1f}', ha='center', va='bottom')
    ax2.set_ylabel('All channels present [%]')
    ax2.set_title('Usable completeness by year')
    sns.despine(fig=fig)
    return tc._finish(fig, save_path, filename)
