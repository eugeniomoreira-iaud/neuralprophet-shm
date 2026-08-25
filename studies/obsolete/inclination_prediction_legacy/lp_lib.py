"""
Module: lp_lib.py

Support library for the legacy-era inclination-prediction study
(``studies/inclination_prediction_legacy/``).

This study asks the questions of ``studies/inclination_prediction/`` again, on
the era that preceded the extended sensor package: station 02 of the 14-column
legacy network, 2018-07-26 to 2025-02-20. That era carries only air temperature,
relative humidity, battery voltage and inclination — no solar radiation, no wall
temperature — so the predictor-selection questions shrink to almost nothing.

What it buys instead is length. Six and a half years is the first record in this
project long enough to identify an annual cycle, so the work here is spectral
rather than combinatorial:

* which periodicities are actually present, established on gapped data without
  imputing anything first;
* over which stretch of record they should be decomposed, and whether they
  persist across the whole span;
* how missing values should be filled, and up to what gap length that is
  defensible;
* how far ahead the inclination can be predicted, and how the error at each
  horizon divides between the components that produce it.

The premises are those of the sibling study and are not re-tested here: the
logged ``I`` channel is read as an inclination in millidegrees whatever the
datasheet range says, the documented compensation is taken as correct, and the
analysis grid is one hour.

Loading, sentinel handling and the legacy-era defect tolerances come from
``lc_lib``; the operator scan, cross-validation, ranking, subset search,
importance diagnostics and horizon machinery come from ``ip_lib``. Only what is
genuinely new to this study is implemented below.

Nothing in this module writes to the raw archive.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.signal import lombscargle, welch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sib in ('thermal_compensation', 'inclination_prediction',
             'thermal_compensation_legacy'):
    _p = os.path.abspath(os.path.join(_HERE, '..', _sib))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tc_lib as tc                                             # noqa: E402
import ip_lib as ip                                             # noqa: E402
import lc_lib as lc                                             # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

#: Legacy era bounds. The 14-column network ran until the changeover.
LEGACY_ERA = ('2018-07-26', '2025-02-20')

#: Station this study is about. The legacy blocks appear in station order and
#: b2 is st02 (docs/raw-data-format.md, section 3.3).
STATION = 'st02'

#: Candidate predictors available in this era. Solar radiation and wall
#: temperature do not exist before 2025-02-21.
LEGACY_DRIVERS = ['tair', 'rh']

#: Housekeeping channel carried as a negative control, exactly as in the sibling
#: study: it correlates with the real drivers through insolation but no mechanism
#: connects it to the structure.
CONTROL_DRIVER = 'batt'

#: Periods, in days, that the harmonic decomposition is built from. The annual
#: term and its second harmonic are the components of interest; the diurnal term
#: and its second harmonic carry the daily cycle.
#:
#: Note on 180 against 182.62: these cannot be separated by this record. The
#: frequency difference is 8.0e-5 cycles per day, which needs roughly 34 years to
#: resolve. A semi-annual peak is therefore reported as the annual second
#: harmonic unless the phase test in :func:`sliding_harmonics` says otherwise.
YEAR_DAYS = 365.25
HARMONIC_PERIODS = {
    'annual': YEAR_DAYS,
    'semiannual': YEAR_DAYS / 2.0,
    'diurnal': 1.0,
    'semidiurnal': 0.5,
}


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_station(archive_dir, cache_dir, start=LEGACY_ERA[0],
                 end=LEGACY_ERA[1], station=STATION, freq=tc.ANALYSIS_FREQ,
                 verbose=True):
    """
    Load one legacy station onto the analysis grid, ready for this study.

    Delegates parsing and sentinel handling to :func:`lc_lib.load_legacy`, which
    tolerates the defects concentrated in this era — duplicate timestamps whose
    copies disagree, decimal separators that change within a file, and records
    dated to the previous day. The inclinometer offset is then restored so that
    ``inc`` carries the number the logger wrote, read as millidegrees, matching
    the premise of the sibling study.

    Parameters
    ----------
    archive_dir : str
        Read-only ``.adc`` archive.
    cache_dir : str
        Local working copy directory.
    start, end : str, optional
        Inclusive date bounds. Default :data:`LEGACY_ERA`.
    station : str, optional
        Station identifier. Default :data:`STATION`.
    freq : str, optional
        Analysis interval. Default :data:`tc_lib.ANALYSIS_FREQ`.
    verbose : bool, optional
        Print the loader's report. Default ``True``.

    Returns
    -------
    pd.DataFrame
        Columns ``batt``, ``tair``, ``rh``, ``inc`` on a regular index.
    """
    df = lc.load_legacy(archive_dir, cache_dir, start, end, station=station,
                        verbose=verbose, freq=freq)
    df['inc'] = df['inc'] + ip.INCLINOMETER_OFFSET_MDEG
    if verbose:
        print(f'  premise   : inclinometer offset of '
              f'{ip.INCLINOMETER_OFFSET_MDEG:.0f} restored; the logged channel '
              f'is read as millidegrees regardless of the datasheet range')
    return df


# ──────────────────────────────────────────────────────────────────────
# Spectral estimation on gapped data
# ──────────────────────────────────────────────────────────────────────
#
# The record is missing a fifth of its days, and the missingness is structured:
# seven contiguous outages, the longest of 271 days. That rules out a plain
# periodogram, which needs a complete regular grid, because filling those gaps
# before estimating the spectrum lets the filling method decide the answer. A
# 271-day hole interpolated linearly injects power at exactly the low frequencies
# this study is trying to measure.
#
# The Lomb-Scargle periodogram is the standard estimator for this situation: it
# fits a sinusoid at each trial frequency by least squares over whatever samples
# exist, so it consumes the gaps rather than requiring them to be filled. The
# spectrum is therefore computed *before* any imputation, and the imputation
# stage that follows cannot influence it.

def _as_xy(series):
    """Return finite (time in days, value) arrays from a series."""
    s = series.dropna()
    t0 = s.index[0]
    x = (s.index - t0).total_seconds().to_numpy() / 86400.0
    return x.astype(float), s.to_numpy(dtype=float)


def period_grid(min_days, max_days, n=4000, log=True):
    """
    Trial periods for a periodogram, in days.

    Parameters
    ----------
    min_days, max_days : float
        Shortest and longest period to scan.
    n : int, optional
        Number of trial periods. Default ``4000``.
    log : bool, optional
        Space them logarithmically. Default ``True``, which gives even
        resolution per octave rather than crowding the short periods.

    Returns
    -------
    np.ndarray
    """
    if log:
        return np.logspace(np.log10(min_days), np.log10(max_days), n)
    return np.linspace(min_days, max_days, n)


def lomb_scargle(series, periods, n_bootstrap=0, seed=0, detrend_hours=None):
    """
    Normalised Lomb-Scargle power against trial period, with an optional null.

    Parameters
    ----------
    series : pd.Series
        Source series on a regular index; missing values are simply skipped,
        which is the point of the estimator.
    periods : array-like
        Trial periods in days.
    n_bootstrap : int, optional
        Permutation replicates used to set a false-alarm level. Default ``0``,
        meaning none. Each replicate shuffles the values against the observation
        times, destroying any real periodicity while preserving the sampling
        pattern exactly, and records the largest power obtained. The 95th and
        99th percentiles of those maxima are the levels a real peak must clear.
    seed : int, optional
        Random seed for the permutation null. Default ``0``.
    detrend_hours : int or None, optional
        Subtract a centred rolling mean of this width before estimating.
        Default ``None``. This matters whenever a high-frequency band is being
        examined: the normalised power is a share of total variance, and on this
        record the annual term holds so much of that variance that the diurnal
        peak is pushed to the fourth decimal place and becomes unreadable even
        though it is the clearest feature in its own band.

    Returns
    -------
    result : pd.DataFrame
        Indexed by ``period_days``, with column ``power`` in ``[0, 1]``.
    levels : dict
        Empty when ``n_bootstrap`` is zero, otherwise ``{'p95': float,
        'p99': float, 'n': int}``.
    """
    if detrend_hours:
        dt = tc.sampling_hours(series) if len(series) > 1 else 1.0
        series = ip.band_limit(series, detrend_hours, dt)
    x, y = _as_xy(series)
    periods = np.asarray(periods, dtype=float)
    ang = 2.0 * np.pi / periods

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        power = lombscargle(x, y, ang, precenter=True, normalize=True)
    result = pd.DataFrame({'power': power},
                          index=pd.Index(periods, name='period_days'))

    levels = {}
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        maxima = np.empty(n_bootstrap)
        for k in range(n_bootstrap):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                maxima[k] = lombscargle(x, rng.permutation(y), ang,
                                        precenter=True, normalize=True).max()
        levels = {'p95': float(np.percentile(maxima, 95)),
                  'p99': float(np.percentile(maxima, 99)),
                  'n': int(n_bootstrap)}
    return result, levels


def window_spectrum(series, periods):
    """
    Lomb-Scargle power of the *sampling pattern* alone.

    This is the control that a periodogram of gapped data requires. The
    observation mask has its own spectrum — seven contiguous outages at
    irregular spacing are not white — and any peak that appears here is a
    property of when the instrument was running, not of the structure. A peak in
    the data spectrum that coincides with a peak here has to be discounted.

    Parameters
    ----------
    series : pd.Series
        Source series on a regular index, used only for its missingness pattern.
    periods : array-like
        Trial periods in days, normally the same grid as the data spectrum.

    Returns
    -------
    pd.DataFrame
        Indexed by ``period_days``, with column ``power``.
    """
    mask = series.notna().astype(float)
    t0 = mask.index[0]
    x = (mask.index - t0).total_seconds().to_numpy().astype(float) / 86400.0
    periods = np.asarray(periods, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        power = lombscargle(x, mask.to_numpy(), 2.0 * np.pi / periods,
                            precenter=True, normalize=True)
    return pd.DataFrame({'power': power},
                        index=pd.Index(periods, name='period_days'))


def spectral_peaks(spectrum, levels=None, window=None, top=12,
                   min_separation=0.15):
    """
    Rank the local maxima of a periodogram.

    Parameters
    ----------
    spectrum : pd.DataFrame
        Output of :func:`lomb_scargle`.
    levels : dict or None, optional
        Output of :func:`lomb_scargle`, used to mark significance.
    window : pd.DataFrame or None, optional
        Output of :func:`window_spectrum`, used to flag peaks that the sampling
        pattern also produces.
    top : int, optional
        Number of peaks to return. Default ``12``.
    min_separation : float, optional
        Minimum separation between reported peaks, in log10 period units.
        Default ``0.15``, about a factor of 1.4, which stops one broad peak
        being reported as several.

    Returns
    -------
    pd.DataFrame
        One row per peak: period in days, a readable period label, power,
        whether it clears the permutation levels, and the window-function power
        at the same period.
    """
    p = spectrum['power'].to_numpy()
    periods = spectrum.index.to_numpy()
    is_peak = np.r_[False, (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]), False]
    idx = np.flatnonzero(is_peak)
    idx = idx[np.argsort(p[idx])[::-1]]

    chosen = []
    for i in idx:
        lg = np.log10(periods[i])
        if all(abs(lg - np.log10(periods[j])) >= min_separation
               for j in chosen):
            chosen.append(i)
        if len(chosen) >= top:
            break

    rows = []
    for i in chosen:
        entry = {'period_days': round(float(periods[i]), 4),
                 'label': _period_label(periods[i]),
                 'power': round(float(p[i]), 4)}
        if levels:
            entry['above_95'] = bool(p[i] > levels['p95'])
            entry['above_99'] = bool(p[i] > levels['p99'])
        if window is not None:
            entry['window_power'] = round(
                float(np.interp(periods[i], window.index.to_numpy(),
                                window['power'].to_numpy())), 4)
        rows.append(entry)
    return pd.DataFrame(rows).sort_values('power', ascending=False) \
        .reset_index(drop=True)


def _period_label(days):
    """Human-readable label for a period expressed in days."""
    hours = days * 24.0
    if days < 1.5:
        return f'{hours:.2f} h'
    if days < 45:
        return f'{days:.2f} d'
    return f'{days:.1f} d ({days / YEAR_DAYS:.2f} yr)'


def welch_psd(series, freq_hours=1.0, segment_days=60, overlap=0.5):
    """
    Welch power spectral density on a gap-free stretch, as a cross-check.

    Lomb-Scargle is the primary estimator because it tolerates the gaps. Welch
    is run on a contiguous block as an independent confirmation in the diurnal
    band, where segment averaging gives a much lower-variance estimate. It
    requires a complete grid and therefore says nothing about the low
    frequencies this record cannot supply gap-free.

    Parameters
    ----------
    series : pd.Series
        Source series, expected to be gap-free over the stretch supplied.
    freq_hours : float, optional
        Sampling interval in hours. Default ``1.0``.
    segment_days : int, optional
        Segment length in days. Default ``60``.
    overlap : float, optional
        Fractional overlap between segments. Default ``0.5``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``period_days``, with column ``psd``.
    """
    y = series.interpolate(method='time', limit_direction='both').to_numpy()
    nper = int(segment_days * 24 / freq_hours)
    nper = min(nper, len(y))
    f, pxx = welch(y - np.nanmean(y), fs=1.0 / freq_hours, nperseg=nper,
                   noverlap=int(nper * overlap))
    keep = f > 0
    periods_days = (1.0 / f[keep]) / 24.0
    return pd.DataFrame({'psd': pxx[keep]},
                        index=pd.Index(periods_days, name='period_days')
                        ).sort_index()


# ──────────────────────────────────────────────────────────────────────
# Harmonic decomposition
# ──────────────────────────────────────────────────────────────────────

def harmonic_design(index, periods, t0=None, trend=True):
    """
    Sine and cosine design matrix for a set of periods.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to build the matrix for.
    periods : dict
        Mapping of component name to period in days.
    t0 : pd.Timestamp or None, optional
        Time origin. Defaults to the first timestamp. Fixing it explicitly
        matters when a model fitted on one window is evaluated on another: the
        phases are only comparable against a common origin.
    trend : bool, optional
        Include a linear term in days. Default ``True``.

    Returns
    -------
    pd.DataFrame
        Design matrix with a constant, an optional trend, and a sine/cosine
        pair per period.
    """
    t0 = index[0] if t0 is None else t0
    t = (index - t0).total_seconds().to_numpy() / 86400.0
    out = pd.DataFrame({'const': 1.0}, index=index)
    if trend:
        out['trend_days'] = t
    for name, period in periods.items():
        w = 2.0 * np.pi * t / period
        out[f'{name}_sin'] = np.sin(w)
        out[f'{name}_cos'] = np.cos(w)
    return out


def fit_harmonics(series, periods, t0=None, trend=True, hac_lags=24):
    """
    Least-squares amplitude and phase for each periodic component.

    Parameters
    ----------
    series : pd.Series
        Target on a regular index.
    periods : dict
        Mapping of component name to period in days.
    t0 : pd.Timestamp or None, optional
        Common time origin for the phases.
    trend : bool, optional
        Include a linear term. Default ``True``.
    hac_lags : int, optional
        Newey-West lag truncation for the standard errors. Default ``24``.
        The residuals of a harmonic fit to this signal are strongly
        autocorrelated, so ordinary standard errors would be far too small.

    Returns
    -------
    table : pd.DataFrame
        One row per component: amplitude, its standard error, phase in days
        after ``t0``, and the share of variance the component alone explains.
    fitted : pd.Series
        The fitted series, on the input index.
    model : statsmodels results
        The underlying fit, for diagnostics.
    """
    X = harmonic_design(series.index, periods, t0=t0, trend=trend)
    both = X.join(series.rename('__y__')).dropna()
    model = sm.OLS(both['__y__'], both.drop(columns='__y__')).fit(
        cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    fitted = pd.Series(model.predict(X), index=series.index)

    total_var = float(series.dropna().var())
    rows = []
    for name, period in periods.items():
        a = float(model.params[f'{name}_sin'])
        b = float(model.params[f'{name}_cos'])
        amp = float(np.hypot(a, b))
        # Delta method for the amplitude standard error.
        cov = model.cov_params().loc[[f'{name}_sin', f'{name}_cos'],
                                     [f'{name}_sin', f'{name}_cos']].to_numpy()
        grad = np.array([a, b]) / amp if amp > 0 else np.zeros(2)
        se = float(np.sqrt(grad @ cov @ grad))
        # Phase expressed as the time of the component's maximum after t0.
        phase_days = float((np.arctan2(a, b) / (2 * np.pi)) * period) % period
        comp = (a * X[f'{name}_sin'] + b * X[f'{name}_cos'])
        rows.append({
            'component': name,
            'period_days': round(period, 3),
            'amplitude_mdeg': round(amp, 3),
            'amp_se': round(se, 3),
            'peak_day_after_t0': round(phase_days, 2),
            'var_share': round(float(comp.var() / total_var), 4)
                if total_var > 0 else np.nan,
        })
    table = pd.DataFrame(rows).set_index('component')
    table.attrs['r2'] = float(model.rsquared)
    return table, fitted, model


def sliding_harmonics(series, periods, window_days=730, step_days=90,
                      t0=None, min_coverage=0.5):
    """
    Track each component's amplitude and phase along the record.

    A component that is a genuine, persistent feature of the structure keeps a
    stable amplitude and a stable phase. One that appears only in some years, or
    whose phase wanders, is not the same object.

    This also supplies the only available test of whether a semi-annual
    component is independent or is simply the second harmonic of the annual
    cycle. The two cannot be separated by frequency — see
    :data:`HARMONIC_PERIODS` — but they can be separated by phase behaviour. If
    the semi-annual term is harmonic distortion of the annual one, the quantity
    ``phase_semiannual - 2 * phase_annual`` is constant along the record; if it
    is an independent process, that quantity drifts.

    Parameters
    ----------
    series : pd.Series
        Target on a regular index.
    periods : dict
        Mapping of component name to period in days.
    window_days : int, optional
        Width of each fitting window. Default ``730``, two years, which is the
        minimum that identifies an annual term at all.
    step_days : int, optional
        Step between windows. Default ``90``.
    t0 : pd.Timestamp or None, optional
        Common phase origin for every window. Defaults to the first timestamp,
        and must be common or the phases are not comparable.
    min_coverage : float, optional
        Minimum fraction of the window that must carry data. Default ``0.5``.

    Returns
    -------
    pd.DataFrame
        One row per window and component, with the window centre, amplitude and
        phase.
    """
    t0 = series.index[0] if t0 is None else t0
    start, end = series.index[0], series.index[-1]
    width = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)

    rows, left = [], start
    while left + width <= end:
        chunk = series.loc[left:left + width]
        if chunk.notna().mean() >= min_coverage:
            try:
                table, _, _ = fit_harmonics(chunk, periods, t0=t0)
            except Exception:
                left = left + step
                continue
            centre = left + width / 2
            for name, row in table.iterrows():
                rows.append({
                    'centre': centre,
                    'component': name,
                    'amplitude_mdeg': row['amplitude_mdeg'],
                    'amp_se': row['amp_se'],
                    'peak_day_after_t0': row['peak_day_after_t0'],
                    'coverage': round(float(chunk.notna().mean()), 3),
                })
        left = left + step
    return pd.DataFrame(rows)


def phase_lock_test(sliding, fundamental='annual', harmonic='semiannual',
                    periods=HARMONIC_PERIODS):
    """
    Test whether a component is harmonic distortion of a lower-frequency one.

    For a second harmonic, the phase of the harmonic advances at exactly twice
    the rate of the fundamental, so the combination
    ``2 * phi_fundamental - phi_harmonic`` is constant along the record. This
    computes that combination window by window and reports its circular spread.
    A small spread means locked, and the harmonic reading stands; a large spread
    means the two components are drifting independently, and the higher
    frequency is a separate process.

    Parameters
    ----------
    sliding : pd.DataFrame
        Output of :func:`sliding_harmonics`.
    fundamental, harmonic : str, optional
        Component names.
    periods : dict, optional
        Mapping of component name to period in days.

    Returns
    -------
    dict
        ``n_windows``, the circular standard deviation of the locked phase in
        degrees, the equivalent spread of an unlocked null, and a boolean
        ``locked``.
    """
    f = sliding[sliding['component'] == fundamental].set_index('centre')
    h = sliding[sliding['component'] == harmonic].set_index('centre')
    common = f.index.intersection(h.index)
    if len(common) < 3:
        return {'n_windows': len(common), 'circ_sd_deg': np.nan,
                'null_sd_deg': 103.9, 'locked': None}

    # Phases as angles, then the locked combination.
    phi_f = 2 * np.pi * f.loc[common, 'peak_day_after_t0'] / periods[fundamental]
    phi_h = 2 * np.pi * h.loc[common, 'peak_day_after_t0'] / periods[harmonic]
    delta = (2 * phi_f - phi_h).to_numpy()

    resultant = np.abs(np.mean(np.exp(1j * delta)))
    circ_sd = float(np.degrees(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))))
    # A uniform phase gives a circular standard deviation of about 103.9 deg.
    return {'n_windows': int(len(common)),
            'circ_sd_deg': round(circ_sd, 1),
            'null_sd_deg': 103.9,
            'resultant': round(float(resultant), 4),
            'locked': bool(circ_sd < 45.0)}


# ──────────────────────────────────────────────────────────────────────
# Window selection
# ──────────────────────────────────────────────────────────────────────

def window_report(df, cols, longest_period_days, min_cycles=2.0,
                  min_days=30, max_gap_hours=6):
    """
    Rank the contiguous blocks by their fitness for a decomposition.

    A block can only identify a component whose period fits into it several
    times over. This applies that rule explicitly rather than leaving the choice
    of analysis window to inspection.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    cols : list of str
        Columns required to be simultaneously present.
    longest_period_days : float
        Period of the slowest component to be identified.
    min_cycles : float, optional
        Cycles of that component a block must contain to qualify. Default
        ``2.0``.
    min_days, max_gap_hours
        Passed to :func:`ip_lib.contiguous_blocks`.

    Returns
    -------
    pd.DataFrame
        The blocks, with the number of cycles each contains and whether it
        qualifies.
    """
    blocks = ip.contiguous_blocks(df, cols, min_days=min_days,
                                 max_gap_hours=max_gap_hours)
    if blocks.empty:
        return blocks
    blocks['cycles'] = (blocks['days'] / longest_period_days).round(2)
    blocks['qualifies'] = blocks['cycles'] >= min_cycles
    return blocks


# ──────────────────────────────────────────────────────────────────────
# Gap inventory and imputation
# ──────────────────────────────────────────────────────────────────────

def gap_inventory(series, dt_hours=1.0):
    """
    Length distribution of the missing stretches.

    Parameters
    ----------
    series : pd.Series
        Source series on a regular index.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per length band: the count of gaps and the total hours lost.
    """
    missing = series.isna().to_numpy()
    if not missing.any():
        return pd.DataFrame(columns=['band', 'n_gaps', 'hours_lost',
                                     'share_of_missing_%'])
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
    lengths = (edges[1::2] - edges[::2]) * dt_hours

    bands = [(0, 3), (3, 6), (6, 24), (24, 72), (72, 168), (168, 720),
             (720, np.inf)]
    names = ['<= 3 h', '3-6 h', '6-24 h', '1-3 d', '3-7 d', '7-30 d', '> 30 d']
    total = lengths.sum()
    rows = []
    for (lo, hi), name in zip(bands, names):
        sel = (lengths > lo) & (lengths <= hi)
        rows.append({'band': name, 'n_gaps': int(sel.sum()),
                     'hours_lost': float(lengths[sel].sum()),
                     'share_of_missing_%': round(
                         100.0 * lengths[sel].sum() / total, 1)
                     if total else 0.0})
    out = pd.DataFrame(rows).set_index('band')
    out.attrs['n_gaps'] = int(len(lengths))
    out.attrs['longest_hours'] = float(lengths.max())
    return out


def impute_benchmark(df, target, drivers, periods, gap_lengths=(1, 3, 6, 12,
                                                                24, 72, 168,
                                                                720),
                     n_trials=40, seed=0, min_context_hours=48):
    """
    Measure imputation error against gap length by injecting known gaps.

    Filling a hole is only defensible up to the length at which the filler stops
    knowing anything the data did not already contain. That length is not a
    matter of opinion: it can be measured by removing stretches whose values are
    known, filling them, and comparing. This does that for four fillers of
    increasing ambition, so that the imputation policy is chosen from a curve
    rather than asserted.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index, containing ``target`` and ``drivers``.
    target : str
        Column to blank and refill.
    drivers : list of str
        Exogenous columns available to the model-based fillers.
    periods : dict
        Harmonic periods for the seasonal filler.
    gap_lengths : tuple of int, optional
        Gap lengths to test, in hours.
    n_trials : int, optional
        Randomly placed gaps per length. Default ``40``.
    seed : int, optional
        Random seed. Default ``0``.
    min_context_hours : int, optional
        Observed data required on each side of a candidate gap. Default ``48``.
        This has to be modest: the record carries hundreds of scattered short
        gaps, so demanding weeks of clean context either side finds no
        candidates at all and returns an empty benchmark. Two days is what the
        fillers actually consume — interpolation needs one endpoint on each
        side and the seasonal filler needs the previous day.

    Returns
    -------
    pd.DataFrame
        One row per gap length and method, with the mean absolute error over
        trials and the number of trials that could be placed.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(seed)
    y = df[target]
    observed = y.notna().to_numpy()
    n = len(y)
    context = int(min_context_hours)

    # Harmonic model fitted once on everything observed, used as one filler.
    harm_table, harm_fit, _ = fit_harmonics(y, periods)

    # Ridge on the drivers, fitted once on everything observed.
    feats = df[drivers]
    both = feats.join(y.rename('__y__')).dropna()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(both[drivers], both['__y__'])
    ridge_pred = pd.Series(ridge.predict(feats.ffill().bfill()), index=df.index)

    rows = []
    for L in gap_lengths:
        errs = {'interpolate': [], 'seasonal_naive': [], 'harmonic': [],
                'ridge_drivers': []}
        placed = 0
        for _ in range(n_trials * 6):
            if placed >= n_trials:
                break
            i = int(rng.integers(context, max(n - context - L, context + 1)))
            sl = slice(i, i + L)
            if not observed[max(i - context, 0):i + L + context].all():
                continue
            truth = y.iloc[sl]
            holed = y.copy()
            holed.iloc[sl] = np.nan

            errs['interpolate'].append(float(
                (holed.interpolate(method='time').iloc[sl] - truth)
                .abs().mean()))
            cycles = max(int(np.ceil(L / 24)), 1)
            errs['seasonal_naive'].append(float(
                (y.shift(24 * cycles).iloc[sl] - truth).abs().mean()))
            errs['harmonic'].append(float(
                (harm_fit.iloc[sl] - truth).abs().mean()))
            errs['ridge_drivers'].append(float(
                (ridge_pred.iloc[sl] - truth).abs().mean()))
            placed += 1

        for method, values in errs.items():
            if not values:
                continue
            rows.append({'gap_hours': L, 'method': method,
                         'MAE_mdeg': round(float(np.mean(values)), 4),
                         'sd': round(float(np.std(values)), 4),
                         'n_trials': placed})
    return pd.DataFrame(rows)


def impute_policy(bench, tolerance_mdeg):
    """
    Longest gap each method can fill within a stated error tolerance.

    Parameters
    ----------
    bench : pd.DataFrame
        Output of :func:`impute_benchmark`.
    tolerance_mdeg : float
        Error the study is prepared to accept from a filled value.

    Returns
    -------
    pd.DataFrame
        One row per method with the longest qualifying gap length.
    """
    if bench.empty or 'method' not in bench.columns:
        raise ValueError(
            'The imputation benchmark placed no gaps, so no policy can be '
            'derived. The usual cause is min_context_hours being larger than '
            'the clean stretches this record actually contains.')
    rows = []
    for method, grp in bench.groupby('method'):
        ok = grp[grp['MAE_mdeg'] <= tolerance_mdeg]
        rows.append({
            'method': method,
            'max_gap_h': int(ok['gap_hours'].max()) if len(ok) else 0,
            'MAE_at_max': (round(float(ok.loc[ok['gap_hours'].idxmax(),
                                              'MAE_mdeg']), 4)
                           if len(ok) else np.nan),
        })
    return (pd.DataFrame(rows).set_index('method')
            .sort_values('max_gap_h', ascending=False))


# ──────────────────────────────────────────────────────────────────────
# How much of the record a model can actually use
# ──────────────────────────────────────────────────────────────────────
#
# "Use the longest unbroken block" is the obvious answer to a gapped record and,
# for a model with autoregressive terms, the wrong one. Such a model does not
# need one continuous series; it needs windows of ``n_lags`` consecutive
# observations followed by ``n_forecasts`` more, and those windows are available
# in every block, not only the longest. NeuralProphet supplies exactly this
# behaviour through ``drop_missing``, which discards the windows straddling a
# gap and trains on the remainder.
#
# That changes what imputation is for. Filling a gap is not a way of making the
# record continuous; it is a way of buying back the windows the gap destroyed.
# One missing hour costs on the order of ``n_lags + n_forecasts`` windows however
# short it is, so filling the many brief interruptions is cheap in fabricated
# data and expensive to skip, while filling one long outage is the reverse. The
# functions below measure that trade rather than assuming which way it falls.

def usable_windows(series, n_lags=24, n_forecasts=1, fill_limit_h=0,
                   dt_hours=1.0):
    """
    Count the autoregressive windows a fill policy leaves usable.

    A window is usable when every one of its ``n_lags`` input steps and
    ``n_forecasts`` target steps is present, either because it was observed or
    because the policy filled it. Both quantities are returned, so a policy can
    be judged on windows gained per hour fabricated rather than on either number
    alone.

    Parameters
    ----------
    series : pd.Series
        Response series on a regular index, missing values present.
    n_lags : int, optional
        Autoregressive depth. Default ``24``.
    n_forecasts : int, optional
        Length of the forecast head. Default ``1``.
    fill_limit_h : int or None, optional
        Longest interruption, in hours, the policy fills. ``0`` fills nothing,
        ``None`` fills everything. Default ``0``.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    dict
        ``n_windows``, ``hours_fabricated``, ``hours_observed``,
        ``hours_available`` and ``span_days``.
    """
    observed = series.notna().to_numpy()
    limit_steps = (None if fill_limit_h is None
                   else int(round(fill_limit_h / dt_hours)))

    # The fill is delegated rather than reimplemented. An accounting function
    # that models the policy instead of applying it will eventually disagree
    # with the code that runs, and the disagreement will be silent.
    present = ip._fill_short_runs(series, limit_steps).notna().to_numpy()

    width = int(n_lags + n_forecasts)
    if width > len(present):
        n_windows = 0
    else:
        # A window is usable when it contains no missing step. Counting the
        # missing steps in every window with a cumulative sum keeps this linear
        # in the length of the record rather than quadratic.
        gaps = np.cumsum(np.r_[0, (~present).astype(np.int64)])
        per_window = gaps[width:] - gaps[:-width]
        n_windows = int((per_window == 0).sum())

    return {
        'n_windows': n_windows,
        'hours_fabricated': float((present & ~observed).sum() * dt_hours),
        'hours_observed': float(observed.sum() * dt_hours),
        'hours_available': float(present.sum() * dt_hours),
        'span_days': round(len(series) * dt_hours / 24.0, 1),
    }


def window_policies(series, policies, n_lags=24, n_forecasts=1, dt_hours=1.0,
                    baseline=None):
    """
    Apply :func:`usable_windows` to several policies and tabulate the result.

    Parameters
    ----------
    series : pd.Series
        Response series on a regular index.
    policies : dict
        Maps a label to either a fill limit in hours, or a ``(series, limit)``
        pair when the policy also restricts the span.
    n_lags, n_forecasts, dt_hours
        Passed to :func:`usable_windows`.
    baseline : str, optional
        Label whose window count the ``windows_vs_baseline`` column is measured
        against. Defaults to the first policy.

    Returns
    -------
    pd.DataFrame
        One row per policy, indexed by label.
    """
    rows = []
    for label, spec in policies.items():
        target, limit = spec if isinstance(spec, tuple) else (series, spec)
        entry = {'policy': label, 'fill_limit_h': limit}
        entry.update(usable_windows(target, n_lags=n_lags,
                                    n_forecasts=n_forecasts,
                                    fill_limit_h=limit, dt_hours=dt_hours))
        entry['fabricated_%'] = round(
            100.0 * entry['hours_fabricated'] / max(len(target), 1), 3)
        entry['annual_cycles'] = round(entry['span_days'] / YEAR_DAYS, 2)
        rows.append(entry)

    out = pd.DataFrame(rows).set_index('policy')
    ref_label = baseline if baseline in out.index else out.index[0]
    ref = out.loc[ref_label, 'n_windows']
    out['windows_vs_baseline'] = (out['n_windows'] / ref).round(2) if ref else np.nan
    # Windows bought per hour invented: the number that decides whether a fill
    # policy is worth its fabricated data.
    gained = out['n_windows'] - ref
    out['windows_per_hour_filled'] = np.where(
        out['hours_fabricated'] > 0,
        (gained / out['hours_fabricated'].replace(0, np.nan)).round(1),
        np.nan)
    return out


def bridge_blocks(df, cols, max_gap_hours, policy=None, dt_hours=1.0):
    """
    Build one continuous block by filling the interruptions inside it.

    The longest span that :func:`ip_lib.contiguous_blocks` reports at the given
    tolerance is extracted and its remaining holes are filled, each by the method
    the imputation policy licenses for a hole of that length. Holes longer than
    any licensed method are filled by interpolation regardless, because the block
    is continuous by definition — which is exactly why the mask is returned
    alongside it.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    cols : list of str
        Columns required to be simultaneously present when locating the block.
    max_gap_hours : float
        Interruption the block may absorb, in hours.
    policy : pd.DataFrame, optional
        Output of :func:`impute_policy`. Used only to report which holes exceed
        what the benchmark licenses; it does not change the filling.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    block : pd.DataFrame
        The continuous block, every column filled.
    fabricated : pd.Series
        Boolean, indexed like ``block``, true where the response was invented.
    summary : dict
        Extent of the block and what filling it cost.
    """
    blocks = ip.contiguous_blocks(df, cols, min_days=1,
                                  max_gap_hours=max_gap_hours)
    if blocks.empty:
        raise ValueError(
            f'no contiguous block at a tolerance of {max_gap_hours} h')

    best = blocks.iloc[0]
    block = df.loc[best['start']:best['end'], cols].copy()
    fabricated = block[cols[0]].isna()

    licensed = float(policy['max_gap_h'].max()) if policy is not None else 0.0
    missing = fabricated.to_numpy()
    over_licence = 0
    if missing.any():
        edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
        runs = (edges[1::2] - edges[::2]) * dt_hours
        over_licence = int((runs > licensed).sum())
        longest_run = float(runs.max())
    else:
        longest_run = 0.0

    block = block.interpolate(method='time', limit_direction='both')

    summary = {
        'start': best['start'],
        'end': best['end'],
        'span_days': round(float(best['days']), 1),
        'annual_cycles': round(float(best['days']) / YEAR_DAYS, 2),
        'hours_fabricated': float(fabricated.sum() * dt_hours),
        'fabricated_%': round(100.0 * float(fabricated.mean()), 2),
        'longest_hole_h': longest_run,
        'holes_beyond_policy': over_licence,
        'policy_limit_h': licensed,
    }
    return block, fabricated, summary


def baseline_step_test(series, left_end, right_start, window_days=30,
                       control=None, deseason=None):
    """
    Test whether the level differs across a long interruption.

    A long outage is an opportunity for the instrument to be serviced, remounted
    or replaced, and an inclinometer's absolute level carries no meaning across
    such an event. A model fitted through the outage would read any offset as
    structural drift, so the offset has to be looked for before the two sides are
    treated as one series.

    The naive comparison — the last observations before against the first after
    — confounds the offset with the season, because the two windows sit at
    different points of the annual cycle. The test is therefore run twice. A
    control channel governed by the calendar and not by the instrument shows how
    much of the apparent step is seasonal; subtracting a fitted seasonal model
    from the response, via ``deseason``, removes that contribution directly. Only
    a step that survives the second treatment is evidence about the instrument.

    Parameters
    ----------
    series : pd.Series
        Response series on a datetime index.
    left_end : str or pd.Timestamp
        Last instant of the block preceding the interruption.
    right_start : str or pd.Timestamp
        First instant of the block following it.
    window_days : float, optional
        Length of the comparison window on each side. Default ``30``.
    control : pd.Series, optional
        Channel governed by the calendar alone, tested identically.
    deseason : pd.Series, optional
        Seasonal expectation to subtract from the response before the second
        test. Typically the fitted series from :func:`fit_harmonics`.

    Returns
    -------
    pd.DataFrame
        One row per test: the two means, the step, Welch's statistic and its
        p-value.
    """
    from scipy import stats

    left_end = pd.Timestamp(left_end)
    right_start = pd.Timestamp(right_start)
    span = pd.Timedelta(days=window_days)

    tests = [('response', series)]
    if deseason is not None:
        tests.append(('response, deseasoned',
                      (series - deseason.reindex(series.index)).rename(
                          series.name)))
    if control is not None:
        tests.append(('control', control))

    rows = []
    for name, s in tests:
        before = s.loc[left_end - span:left_end].dropna()
        after = s.loc[right_start:right_start + span].dropna()
        entry = {'test': name, 'channel': s.name,
                 'n_before': int(len(before)), 'n_after': int(len(after))}
        if len(before) < 2 or len(after) < 2:
            entry.update({'mean_before': np.nan, 'mean_after': np.nan,
                          'step': np.nan, 't': np.nan, 'p': np.nan,
                          'significant': None})
        else:
            t, p = stats.ttest_ind(after, before, equal_var=False)
            entry.update({
                'mean_before': round(float(before.mean()), 3),
                'mean_after': round(float(after.mean()), 3),
                'step': round(float(after.mean() - before.mean()), 3),
                't': round(float(t), 2),
                'p': float(p),
                'significant': bool(p < 0.05),
            })
        rows.append(entry)

    out = pd.DataFrame(rows).set_index('test')
    out.attrs['interruption_days'] = round(
        (right_start - left_end).total_seconds() / 86400.0, 1)
    return out


def compare_decompositions(components, harmonic=None, periods=None,
                           label='neuralprophet', t0=None):
    """
    Summarise a NeuralProphet decomposition beside a harmonic reference.

    NeuralProphet's trend is piecewise linear with changepoints, and a trend of
    that shape is free to absorb an annual cycle instead of leaving it to the
    yearly Fourier term. The harmonic fit cannot do this — its basis has one
    place to put an annual signal — so it serves as the reference against which
    the split is judged. The diagnostic is the standard deviation of the trend
    against that of the yearly component: a trend that moves as much as the
    season it is supposed to sit beneath has taken part of it.

    Parameters
    ----------
    components : pd.DataFrame
        Output of :func:`lc_lib.decompose_series`, carrying ``trend`` and
        ``season_*`` columns plus ``observed``.
    harmonic : pd.DataFrame, optional
        The table returned by :func:`fit_harmonics`, for the reference amplitude
        and phase.
    periods : dict, optional
        Component periods in days. Only ``'annual'`` is read, to convert the
        reference phase into a day of year.
    label : str, optional
        Row label. Default ``'neuralprophet'``.
    t0 : pd.Timestamp, optional
        Time origin of the harmonic phases. Supplying it converts the reference
        phase, which :func:`fit_harmonics` reports as days after the origin, into
        the day of year that NeuralProphet's component is read on, so that the
        two phases can be compared directly.

    Returns
    -------
    pd.DataFrame
        One row, carrying the component standard deviations, the implied annual
        amplitude and phase, and the harmonic reference where one was supplied.
    """
    obs = components['observed']
    entry = {'configuration': label, 'n_observed': int(obs.notna().sum())}

    for col in components.columns:
        if col == 'trend' or col.startswith('season_'):
            entry[f'std_{col}'] = round(float(components[col].std()), 3)

    if 'season_yearly' in components.columns:
        yearly = components['season_yearly'].dropna()
        # Amplitude of a component recovered as a series is half its range, and
        # the phase is the day of year at which it peaks.
        entry['np_annual_amplitude'] = round(
            float((yearly.max() - yearly.min()) / 2.0), 2)
        entry['np_annual_peak_doy'] = int(
            yearly.idxmax().dayofyear) if len(yearly) else -1
        if 'trend' in components.columns:
            trend_std = float(components['trend'].std())
            season_std = float(yearly.std())
            total = trend_std + season_std
            entry['trend_share_%'] = (round(100.0 * trend_std / total, 1)
                                      if total else np.nan)

    if harmonic is not None and 'annual' in harmonic.index:
        entry['ref_annual_amplitude'] = round(
            float(harmonic.loc['annual', 'amplitude_mdeg']), 2)
        entry['ref_annual_amp_se'] = round(
            float(harmonic.loc['annual', 'amp_se']), 2)
        peak_after_t0 = float(harmonic.loc['annual', 'peak_day_after_t0'])
        entry['ref_annual_peak_day'] = round(peak_after_t0, 1)
        if t0 is not None:
            entry['ref_annual_peak_doy'] = int(
                (pd.Timestamp(t0)
                 + pd.Timedelta(days=peak_after_t0)).dayofyear)
        if 'np_annual_amplitude' in entry:
            entry['amplitude_ratio'] = round(
                entry['np_annual_amplitude']
                / entry['ref_annual_amplitude'], 3)
        if 'np_annual_peak_doy' in entry and 'ref_annual_peak_doy' in entry:
            # Phase distance on a circle of 365 days: a difference of 350 days
            # is a difference of 15, and reporting the former would be absurd.
            raw = abs(entry['np_annual_peak_doy'] - entry['ref_annual_peak_doy'])
            entry['phase_gap_days'] = int(min(raw, 365 - raw))

    if 'yhat' in components.columns:
        err = (obs - components['yhat']).dropna()
        entry['fit_MAE_mdeg'] = round(float(err.abs().mean()), 3)

    return pd.DataFrame([entry]).set_index('configuration')


# ──────────────────────────────────────────────────────────────────────
# Error budget: the nested ladder
# ──────────────────────────────────────────────────────────────────────
#
# A single forecast error tells you how well the system does. It does not tell
# you what the error is made of, and therefore does not tell you what would
# improve it. The ladder below adds one component at a time and scores every
# rung at every horizon, so the reduction attributable to each component is read
# directly off consecutive rows.
#
# The rungs divide into two kinds, and the division is the point. Rungs built
# only from the calendar — trend and harmonics — are deterministic functions of
# time, so their error does not grow with the horizon. Rungs that use the
# signal's own recent history or its drivers do grow, because the information
# they rest on goes stale. Where the two curves cross is the horizon beyond
# which nothing but the seasonal model is contributing.

#: The ladder. Each entry is ``(name, harmonic components, trend, autoregression,
#: drivers)``, cumulative from the rung above.
#:
#: The linear trend is deliberately placed *after* the harmonics rather than
#: immediately after the mean. Fitted on part of a record and extrapolated across
#: the rest, a straight line through this signal is the single most damaging term
#: available — it turns a 49 mdeg error into a 138 mdeg one. Putting it first
#: would hide every seasonal result behind that one decision; putting it here
#: shows what the seasonal model achieves on its own and then prices the drift
#: term honestly against it.
_SEASONAL = ['annual', 'semiannual', 'diurnal', 'semidiurnal']
LADDER = [
    ('climatology',  [],                        False, False, False),
    ('+ annual',     ['annual'],                False, False, False),
    ('+ semiannual', ['annual', 'semiannual'],  False, False, False),
    ('+ diurnal',    _SEASONAL,                 False, False, False),
    ('+ AR',         _SEASONAL,                 False, True,  False),
    ('+ drivers',    _SEASONAL,                 False, True,  True),
]

#: Two rungs differing only in the linear trend, for pricing it on its own.
#:
#: No rung of :data:`LADDER` carries a trend. Fitted on part of this record and
#: extrapolated across the rest, a straight line is the single most damaging term
#: available — it turns a 49 mdeg error into a 134 mdeg one. Leaving it inside
#: the ladder would make every gain below it a measure of recovery from that one
#: decision rather than of the component's own worth, so it is priced separately
#: here, which is the honest place for a term that only subtracts.
TREND_PROBE = [
    ('seasonal, no trend', _SEASONAL, False, False, False),
    ('seasonal + trend',   _SEASONAL, True,  False, False),
]


def error_budget(df, target, drivers, horizons, origins=(0.5, 0.6, 0.7),
                 n_lags=24, periods=HARMONIC_PERIODS, alpha=1.0,
                 ladder=LADDER, future_regressors=False, min_test=200):
    """
    Score every rung of the nested ladder at every horizon.

    Each rung is fitted on the data before a chronological cut and scored on
    everything after it, at an exact lead time. A rung built only from the
    calendar is a function of time, so one fit serves every horizon; a rung
    using history or drivers is fitted once per horizon by direct multi-step
    regression, which lets a single fitted model be evaluated at every issue
    time in the test region while keeping the lead exact.

    Scoring across the whole test region rather than at one timestamp per origin
    matters: the latter yields a handful of samples per horizon and an error
    estimate dominated by which day the cut happened to land on.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    target : str
        Column to predict.
    drivers : list of str
        Exogenous columns for the final rung.
    horizons : iterable of int
        Horizons in steps.
    origins : tuple of float, optional
        Chronological training fractions to repeat the whole exercise at.
        Default ``(0.5, 0.6, 0.7)``. Errors are pooled across them.
    n_lags : int, optional
        Autoregressive depth for the rungs that use it. Default ``24``.
    periods : dict, optional
        Available harmonic components. Default :data:`HARMONIC_PERIODS`.
    alpha : float, optional
        Ridge penalty. Default ``1.0``.
    ladder : list, optional
        The rungs. Default :data:`LADDER`.
    future_regressors : bool, optional
        Give the driver rung the driver values at the predicted timestamp
        instead of at the issue time. Default ``False``, the honest case.
    min_test : int, optional
        Minimum scorable test samples for a cell to be reported. Default
        ``200``.

    Returns
    -------
    pd.DataFrame
        Rungs as rows, horizons as columns, mean absolute error in the cells.
        The sample count behind each column is attached as ``.attrs['n']``.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    n = len(df)
    t0 = df.index[0]
    y = df[target]
    results = {name: {h: [] for h in horizons} for name, *_ in ladder}
    counts = {h: 0 for h in horizons}

    for frac in origins:
        cut = int(n * frac)
        cut_stamp = df.index[cut]
        for name, comps, use_trend, use_ar, use_drv in ladder:
            # A component the caller has removed simply does not enter. This
            # keeps the sensitivity sweep able to ask what dropping the
            # semi-annual term, or every seasonal term, would cost.
            sub = {k: periods[k] for k in comps if k in periods}

            if not use_ar and not use_drv:
                # Calendar-only rung: a function of time. One fit serves every
                # horizon, because the prediction for a timestamp does not
                # depend on how far ahead it was requested.
                X = harmonic_design(df.index, sub, t0=t0, trend=use_trend)
                tr = X.loc[X.index < cut_stamp].join(
                    y.rename('__y__')).dropna()
                if len(tr) < min_test:
                    continue
                model = sm.OLS(tr['__y__'], tr.drop(columns='__y__')).fit()
                pred = pd.Series(model.predict(X), index=df.index)
                for h in horizons:
                    # Test rows are those whose issue time is also after the
                    # cut, so the lead is exactly h everywhere.
                    sel = df.index[cut + h:]
                    err = (pred.reindex(sel) - y.reindex(sel)).abs().dropna()
                    if len(err) >= min_test:
                        results[name][h].append(err)
            else:
                seasonal = harmonic_design(df.index, sub, t0=t0,
                                           trend=use_trend)
                seas_cols = [c for c in seasonal.columns if c != 'const']
                for h in horizons:
                    frame = seasonal[seas_cols].copy()
                    if use_drv:
                        for c in drivers:
                            frame[c] = (df[c] if future_regressors
                                        else df[c].shift(h))
                    if use_ar:
                        for k in range(1, n_lags + 1):
                            frame[f'ar{k}'] = y.shift(h + k - 1)
                    both = frame.join(y.rename('__y__')).dropna()
                    tr = both.loc[both.index < cut_stamp]
                    te = both.loc[both.index >= cut_stamp]
                    if len(tr) < min_test or len(te) < min_test:
                        continue
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(tr.drop(columns='__y__'), tr['__y__'])
                    pred = model.predict(te.drop(columns='__y__'))
                    results[name][h].append(
                        pd.Series(np.abs(pred - te['__y__'].to_numpy()),
                                  index=te.index))

    table = {}
    for h in horizons:
        col = {}
        for name, *_ in ladder:
            parts = results[name][h]
            if parts:
                pooled = pd.concat(parts)
                col[name] = round(float(pooled.mean()), 4)
                counts[h] = max(counts[h], len(pooled))
            else:
                col[name] = np.nan
        table[h] = col
    out = pd.DataFrame(table)
    out.index.name = 'rung'
    out.columns.name = 'horizon_h'
    out.attrs['n'] = counts
    return out


def budget_gains(budget):
    """
    Marginal error reduction attributable to each rung.

    Parameters
    ----------
    budget : pd.DataFrame
        Output of :func:`error_budget`.

    Returns
    -------
    pd.DataFrame
        Same shape, holding the drop in mean absolute error from the rung above.
        The first row holds the climatology error itself, which is the total the
        remaining rows divide up.
    """
    gains = budget.diff().mul(-1.0)
    gains.iloc[0] = budget.iloc[0]
    return gains.round(4)


# ──────────────────────────────────────────────────────────────────────
# Sensitivity
# ──────────────────────────────────────────────────────────────────────

def sensitivity_sweep(run_fn, variations, baseline_label='baseline'):
    """
    One-at-a-time sensitivity of a headline number to the study's choices.

    Every result in a study of this kind rests on decisions that could
    defensibly have gone another way: the compensation coefficient, the analysis
    window, whether a seasonal term is included, where the training split falls.
    Reporting a single number without showing how far it moves under those
    choices overstates its precision.

    Parameters
    ----------
    run_fn : callable
        Takes a keyword dictionary and returns a scalar. Called once per
        variation.
    variations : dict
        Mapping of a readable label to the keyword dictionary that produces it.
        Include the baseline under ``baseline_label``.
    baseline_label : str, optional
        Key of the baseline configuration. Default ``'baseline'``.

    Returns
    -------
    pd.DataFrame
        One row per variation with the value, the absolute change from the
        baseline, and the change as a percentage.
    """
    values = {}
    for label, kwargs in variations.items():
        try:
            values[label] = float(run_fn(**kwargs))
        except Exception as exc:                              # noqa: BLE001
            values[label] = np.nan
            print(f'  sensitivity: "{label}" failed - {exc}')

    base = values.get(baseline_label, np.nan)
    rows = [{'variation': label, 'value': round(v, 4),
             'delta': round(v - base, 4) if v == v and base == base else np.nan,
             'delta_%': (round(100.0 * (v - base) / base, 2)
                         if v == v and base == base and base else np.nan)}
            for label, v in values.items()]
    out = pd.DataFrame(rows).set_index('variation')
    return out.reindex(
        [baseline_label] + [k for k in values if k != baseline_label])


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

set_context = tc.set_context
figsize = tc.figsize
_finish = tc._finish


def plot_spectrum(spectrum, window=None, levels=None, marks=None, title='',
                  xlabel='period [days]', save_path=None, filename=None):
    """
    Periodogram against period, with the sampling-pattern control.

    Parameters
    ----------
    spectrum : pd.DataFrame
        Output of :func:`lomb_scargle`.
    window : pd.DataFrame or None, optional
        Output of :func:`window_spectrum`, drawn beneath the data spectrum.
    levels : dict or None, optional
        Permutation levels from :func:`lomb_scargle`.
    marks : dict or None, optional
        Periods to annotate, as ``{label: period_days}``.
    title, xlabel : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 2 if window is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize(11, 3.2 * n_panels),
                             sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(spectrum.index, spectrum['power'], lw=0.8,
            color=sns.color_palette('colorblind')[0])
    if levels:
        for key, style in (('p95', '--'), ('p99', ':')):
            if key in levels:
                ax.axhline(levels[key], color='0.35', ls=style, lw=1,
                           label=f'{key} permutation level')
    if marks:
        for label, period in marks.items():
            ax.axvline(period, color='0.6', lw=0.8, ls='-.')
            ax.text(period, ax.get_ylim()[1] * 0.95, f' {label}',
                    rotation=90, va='top', fontsize='small', color='0.35')
    ax.set_xscale('log')
    ax.set_ylabel('normalised power')
    if levels:
        ax.legend(fontsize='small')

    if window is not None:
        axw = axes[1, 0]
        axw.plot(window.index, window['power'], lw=0.8, color='0.45')
        axw.set_xscale('log')
        axw.set_ylabel('window function')
        axw.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_sliding_harmonics(sliding, components=('annual', 'semiannual'),
                           title='', save_path=None, filename=None):
    """
    Amplitude and phase of each component along the record.

    Parameters
    ----------
    sliding : pd.DataFrame
        Output of :func:`sliding_harmonics`.
    components : tuple of str, optional
        Components to draw.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.4))
    palette = sns.color_palette('colorblind')
    for k, comp in enumerate(components):
        grp = sliding[sliding['component'] == comp]
        if grp.empty:
            continue
        axes[0].plot(grp['centre'], grp['amplitude_mdeg'], marker='o', ms=3,
                     lw=1.2, color=palette[k], label=comp)
        axes[0].fill_between(grp['centre'],
                             grp['amplitude_mdeg'] - grp['amp_se'],
                             grp['amplitude_mdeg'] + grp['amp_se'],
                             color=palette[k], alpha=0.2)
        axes[1].plot(grp['centre'], grp['peak_day_after_t0'], marker='o', ms=3,
                     lw=1.2, color=palette[k], label=comp)
    axes[0].set_ylabel('amplitude [mdeg]')
    axes[1].set_ylabel('phase — peak, days after origin')
    for ax in axes:
        ax.set_xlabel('')
        ax.legend(fontsize='small')
        ax.tick_params(axis='x', rotation=30)
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_gap_inventory(inventory, bench=None, title='', save_path=None,
                       filename=None):
    """
    Where the missing hours are, and what it costs to fill them.

    Parameters
    ----------
    inventory : pd.DataFrame
        Output of :func:`gap_inventory`.
    bench : pd.DataFrame or None, optional
        Output of :func:`impute_benchmark`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 2 if bench is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize(11, 3.4),
                             squeeze=False)
    ax = axes[0, 0]
    labels = inventory.index.tolist()
    sns.barplot(x=inventory['share_of_missing_%'].values, y=labels, ax=ax,
                hue=labels, palette='colorblind', legend=False, orient='h')
    for i, n in enumerate(inventory['n_gaps']):
        ax.text(inventory['share_of_missing_%'].iloc[i] + 0.5, i,
                f'{int(n)} gaps', va='center', fontsize='small')
    ax.set_xlabel('share of all missing hours [%]')
    ax.set_ylabel('')

    if bench is not None:
        axb = axes[0, 1]
        for method, grp in bench.groupby('method'):
            grp = grp.sort_values('gap_hours')
            axb.plot(grp['gap_hours'], grp['MAE_mdeg'], marker='o', ms=4,
                     lw=1.2, label=method)
        axb.set_xscale('log')
        axb.set_xlabel('injected gap length [h]')
        axb.set_ylabel('imputation MAE [mdeg]')
        axb.legend(fontsize='small')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_error_budget(budget, gains=None, title='', save_path=None,
                      filename=None):
    """
    The ladder: error against horizon per rung, and the marginal gains.

    Parameters
    ----------
    budget : pd.DataFrame
        Output of :func:`error_budget`.
    gains : pd.DataFrame or None, optional
        Output of :func:`budget_gains`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 2 if gains is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize(11, 3.8),
                             squeeze=False)
    ax = axes[0, 0]
    palette = sns.color_palette('colorblind', n_colors=len(budget))
    for k, (rung, row) in enumerate(budget.iterrows()):
        ax.plot(row.index, row.values, marker='o', ms=4, lw=1.2,
                color=palette[k], label=rung)
    ax.set_xscale('log')
    ax.set_xticks(list(budget.columns))
    ax.set_xticklabels([f'{int(h)}' for h in budget.columns])
    ax.set_xlabel('forecast horizon [h]')
    ax.set_ylabel('MAE [mdeg]')

    if gains is not None:
        axg = axes[0, 1]
        share = gains.drop(index=gains.index[0]).clip(lower=0)
        bottom = np.zeros(len(share.columns))
        xs = np.arange(len(share.columns))
        for k, (rung, row) in enumerate(share.iterrows()):
            axg.bar(xs, row.values, bottom=bottom, label=rung,
                    color=palette[k + 1], width=0.7)
            bottom = bottom + np.nan_to_num(row.values)
        axg.set_xticks(xs)
        axg.set_xticklabels([f'{int(h)}' for h in share.columns])
        axg.set_xlabel('forecast horizon [h]')
        axg.set_ylabel('MAE reduction attributed [mdeg]')

    # One shared legend beneath both panels. Placed inside an axis it covers the
    # bars, which in this figure are the result.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 4), fontsize='small',
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path, filename)


def plot_sensitivity(sens, title='', save_path=None, filename=None):
    """
    Tornado chart of the headline number against the study's choices.

    Parameters
    ----------
    sens : pd.DataFrame
        Output of :func:`sensitivity_sweep`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    body = sens.drop(index=sens.index[0]).dropna(subset=['delta'])
    body = body.reindex(body['delta'].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=figsize(11, 0.45 * len(body) + 1.4))
    colours = ['#A5202B' if d > 0 else '#1B6B3A' for d in body['delta']]
    ax.barh(body.index.tolist(), body['delta'].values, color=colours)
    ax.axvline(0, color='0.3', lw=1)
    ax.set_xlabel(f'change in headline MAE [mdeg] '
                  f'(baseline {sens["value"].iloc[0]:.3f})')
    for i, (d, p) in enumerate(zip(body['delta'], body['delta_%'])):
        ax.text(d, i, f'  {p:+.1f} %', va='center', fontsize='small')
    if title:
        ax.set_title(title)
    return _finish(fig, save_path, filename)


def plot_window_accounting(policies, title='', save_path=None, filename=None):
    """
    What each fill policy buys, against what it invents.

    Parameters
    ----------
    policies : pd.DataFrame
        Output of :func:`window_policies`.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.6))
    labels = policies.index.tolist()
    palette = sns.color_palette('colorblind', len(labels))

    axes[0].barh(labels, policies['n_windows'].values, color=palette)
    axes[0].set_xlabel('usable autoregressive windows')
    for i, (n, c) in enumerate(zip(policies['n_windows'],
                                   policies['annual_cycles'])):
        axes[0].text(n, i, f'  {c:.2f} cycles', va='center', fontsize='small')

    axes[1].barh(labels, policies['fabricated_%'].values, color=palette)
    axes[1].set_xlabel('fabricated share of the span [%]')
    axes[1].set_yticklabels([])
    for i, h in enumerate(policies['hours_fabricated']):
        axes[1].text(policies['fabricated_%'].iloc[i], i, f'  {h:.0f} h',
                     va='center', fontsize='small')

    for ax in axes:
        ax.set_ylabel('')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_decomposition(components, harmonic_fit=None, title='',
                       save_path=None, filename=None):
    """
    A NeuralProphet decomposition, with the harmonic fit for comparison.

    Parameters
    ----------
    components : pd.DataFrame
        Output of :func:`lc_lib.decompose_series`.
    harmonic_fit : pd.Series or None, optional
        Fitted series from :func:`fit_harmonics`, drawn over the observations.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    drawable = ['trend'] + [c for c in components.columns
                            if c.startswith('season_')]
    drawable = [c for c in drawable if c in components.columns]
    fig, axes = plt.subplots(len(drawable) + 1, 1,
                             figsize=figsize(11, 1.7 * (len(drawable) + 1)),
                             sharex=True)
    palette = sns.color_palette('colorblind')

    ax = axes[0]
    ax.plot(components.index, components['observed'], lw=0.4, color='0.55',
            label='observed')
    if harmonic_fit is not None:
        ax.plot(harmonic_fit.index, harmonic_fit.values, lw=1.0,
                color=palette[3], label='harmonic fit')
    ax.set_ylabel('mdeg')
    ax.legend(fontsize='small', ncol=2)

    for k, col in enumerate(drawable):
        axes[k + 1].plot(components.index, components[col], lw=0.9,
                         color=palette[k])
        axes[k + 1].set_ylabel(col.replace('season_', ''))
    axes[-1].set_xlabel('')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)
