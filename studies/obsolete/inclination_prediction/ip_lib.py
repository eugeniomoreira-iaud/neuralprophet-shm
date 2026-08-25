"""
Module: ip_lib.py

Support library for the inclination-prediction study
(``studies/inclination_prediction/``).

The study asks three questions of the extended sensor package installed at
station 02 on 21 February 2025:

1. Can any single sensor channel predict the inclination, and which one is best?
2. Can a combination of channels beat the best single one, and by what procedure
   should that combination be diagnosed?
3. Under the best combination, how far ahead can NeuralProphet forecast, and with
   what quantified uncertainty?

Three premises are imposed by the study design and are **not** re-examined here.
They differ deliberately from those of ``studies/thermal_compensation/``:

* The logged ``I`` channel is taken to be an inclination in millidegrees, not a
  conditioner voltage. The 2500 offset that the sibling study subtracts is added
  back by :func:`load_current`, so the series carries the number the acquisition
  system recorded.
* The temperature compensation implemented in Notebook 00 is taken to be
  correct. The compensated series is the target of every model below, and no
  test in this module questions the coefficient.
* The analysis grid is one hour, because every external proxy the production
  pipeline aligns against is published hourly.

Loading, hourly aggregation, sentinel handling, spike screening, the thermal-lag
filter and the seaborn figure scaffolding are imported from the sibling study's
``tc_lib`` rather than reimplemented, so that any difference in results between
the two studies is a difference in the question asked and not in the machinery.

Nothing in this module writes to the raw archive.
"""

import os
import sys
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'thermal_compensation')))

import tc_lib as tc                                             # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Premise constants
# ──────────────────────────────────────────────────────────────────────

#: Offset that ``tc.load_archive`` removes from the logged ``I`` channel when it
#: treats that channel as a conditioner voltage. This study treats the logged
#: number as an inclination in millidegrees, so the offset is added back
#: immediately after loading. Nothing downstream depends on the choice: the
#: compensation normalises the series to start at zero, which removes every
#: additive constant. The offset is restored so that the printed levels match the
#: raw files rather than so that any model behaves differently.
INCLINOMETER_OFFSET_MDEG = tc.INCLINOMETER_ZERO_MV

#: Compensation coefficient, taken as given. See the module docstring.
COMP_COEFF = tc.DOCUMENTED_COEFF

#: Channel driving the compensation, as implemented in Notebook 00.
COMP_TEMP_COL = 'tair'

#: Candidate predictors carrying a plausible physical mechanism.
PHYSICAL_DRIVERS = ['tair', 'twall', 'sr', 'rh']

#: Housekeeping channel included in the single-predictor ranking as a negative
#: control. Battery voltage rises with insolation and temperature, so it
#: correlates with the real drivers, but no mechanism connects it to the
#: structure. If it ranks near the physical channels, the ranking procedure is
#: measuring shared diurnal shape rather than predictive content, and the study
#: says so instead of reporting the ranking.
CONTROL_DRIVER = 'batt'


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_current(archive_dir, cache_dir, start, end, freq=tc.ANALYSIS_FREQ,
                 verbose=True):
    """
    Load the current-era station 02 record onto the analysis grid.

    Delegates parsing, duplicate merging, sentinel handling and hourly
    aggregation to :func:`tc_lib.load_archive`, then restores the inclinometer
    offset so that ``inc`` carries the number the logger wrote, interpreted as
    millidegrees per this study's first premise.

    Parameters
    ----------
    archive_dir : str
        Read-only ``.adc`` archive.
    cache_dir : str
        Local working copy directory.
    start, end : str or pd.Timestamp
        Inclusive date bounds.
    freq : str, optional
        Analysis interval. Default :data:`tc_lib.ANALYSIS_FREQ`.
    verbose : bool, optional
        Print the loader's report. Default ``True``.

    Returns
    -------
    pd.DataFrame
        Columns ``batt``, ``tair``, ``rh``, ``inc``, ``sr``, ``twall`` on a
        regular index at ``freq``.
    """
    df = tc.load_archive(archive_dir, cache_dir, start, end, verbose=verbose,
                         freq=freq)
    df['inc'] = df['inc'] + INCLINOMETER_OFFSET_MDEG
    if verbose:
        print(f'  premise   : inclinometer offset of '
              f'{INCLINOMETER_OFFSET_MDEG:.0f} restored; the logged channel is '
              f'read as millidegrees')
    return df


def add_target(df, temp_col=COMP_TEMP_COL, coeff=COMP_COEFF):
    """
    Attach the compensated inclination, which is this study's target.

    The compensation is applied exactly as Notebook 00 implements it and is
    taken as correct. The normalisation it performs subtracts the first valid
    value, so the target starts at zero; only differences of the target carry
    information, and the level is an artefact of where the window begins.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc`` and ``temp_col``.
    temp_col : str, optional
        Compensating temperature channel. Default :data:`COMP_TEMP_COL`.
    coeff : float, optional
        Compensation coefficient. Default :data:`COMP_COEFF`.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``inc_comp`` column.
    """
    out = df.copy()
    out['inc_comp'] = tc.compensate(out, temp_col=temp_col, coeff=coeff,
                                    normalise=True)
    return out


def sign_table(df, targets, drivers, detrend_hours=None, dt_hours=1.0):
    """
    Compare the thermal response of the raw and the compensated inclination.

    The site's structural expectation fixes a sign: the valley face of the
    embankment receives more insolation than the mountain face, expands further
    when heated, and tips the wall towards the mountain, which is a negative
    change under the instrument's convention. A negative association between a
    heating driver and the inclination is therefore the physically expected
    result rather than an anomaly.

    That expectation is about the structure, so it must be tested on the signal
    before the compensation touches it. The compensation subtracts a multiple of
    air temperature, and if that multiple exceeds the instrument's real thermal
    slope it will manufacture a negative association out of a positive one. This
    function reports both series side by side so that the two cases can be told
    apart: a sign that is already negative in the raw channel is structural, and
    a sign that only appears after compensation is arithmetic.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    targets : list of str
        Inclination columns to test, normally the raw and the compensated one.
    drivers : list of str
        Heating drivers to test against.
    detrend_hours : int or None, optional
        If given, the same statistics are also computed on the band-limited
        series. Default ``None``.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per target and driver, with the correlation and the ordinary
        least-squares slope of the target on the driver, on levels and, when
        requested, on the band-limited series.
    """
    rows = []
    for tgt in targets:
        for drv in drivers:
            entry = {'target': tgt, 'driver': drv}
            pairs = [('levels', df[tgt], df[drv])]
            if detrend_hours:
                pairs.append(('band',
                              band_limit(df[tgt], detrend_hours, dt_hours),
                              band_limit(df[drv], detrend_hours, dt_hours)))
            for label, y, x in pairs:
                both = pd.concat([y.rename('y'), x.rename('x')],
                                 axis=1).dropna()
                if len(both) < 100:
                    entry[f'r_{label}'] = np.nan
                    entry[f'slope_{label}'] = np.nan
                    continue
                entry[f'r_{label}'] = round(
                    float(both['y'].corr(both['x'])), 4)
                entry[f'slope_{label}'] = round(
                    float(np.polyfit(both['x'], both['y'], 1)[0]), 4)
            rows.append(entry)
    return pd.DataFrame(rows).set_index(['target', 'driver'])


def contiguous_blocks(df, cols, min_days=20, max_gap_hours=6):
    """
    Find the runs in which every listed column is continuously available.

    A run is broken by any stretch longer than ``max_gap_hours`` in which at
    least one of the columns is missing. Short interruptions are tolerated
    because the forecasting models bridge them internally and score only at
    observed timestamps; a multi-day interruption is a different object and must
    split the record instead of being interpolated across.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    cols : list of str
        Columns required to be simultaneously present.
    min_days : int, optional
        Minimum run length to report, in days. Default ``20``.
    max_gap_hours : int, optional
        Longest interruption absorbed into a run. Default ``6``.

    Returns
    -------
    pd.DataFrame
        One row per run with ``start``, ``end``, ``days`` and ``coverage_%``,
        sorted longest first.
    """
    step = tc.sampling_hours(df)
    ok = df[cols].notna().all(axis=1)
    tolerance = int(round(max_gap_hours / step))

    runs, start, gap = [], None, 0
    for stamp, good in ok.items():
        if good:
            if start is None:
                start = stamp
            gap = 0
            last = stamp
        else:
            if start is not None:
                gap += 1
                if gap > tolerance:
                    runs.append((start, last))
                    start = None
    if start is not None:
        runs.append((start, last))

    rows = []
    for lo, hi in runs:
        days = (hi - lo).total_seconds() / 86400.0
        if days < min_days:
            continue
        block = ok.loc[lo:hi]
        rows.append({'start': lo, 'end': hi, 'days': round(days, 1),
                     'coverage_%': round(100.0 * block.mean(), 1)})
    if not rows:
        return pd.DataFrame(columns=['start', 'end', 'days', 'coverage_%'])
    return (pd.DataFrame(rows).sort_values('days', ascending=False)
            .reset_index(drop=True))


def availability(df, cols):
    """
    Fraction of the grid on which every listed column is present.

    This is the quantity that prices a predictor against its downtime. A channel
    that explains the signal well but is absent for half the record buys less
    than its accuracy suggests, and the decision between them cannot be made from
    error metrics alone.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    cols : list of str
        Columns required to be simultaneously present.

    Returns
    -------
    float
        Fraction in ``[0, 1]``.
    """
    return float(df[cols].notna().all(axis=1).mean())


# ──────────────────────────────────────────────────────────────────────
# Operators: transport delay and thermal inertia
# ──────────────────────────────────────────────────────────────────────
#
# Two distinct things can delay a structural response to an environmental
# driver, and they are not interchangeable.
#
# A *transport delay* shifts the driver in time without changing its shape. It
# represents the time a thermal front takes to travel from the exposed face to
# the depth that governs the inclination.
#
# A *thermal inertia* low-passes the driver: the structure integrates the
# forcing with a time constant tau, so its temperature follows a smoothed and
# attenuated version of the driver rather than a shifted copy of it. This is the
# lumped-capacitance response, implemented in ``tc_lib.thermal_lag_filter`` as a
# single-pole filter with alpha = dt / (tau + dt).
#
# The sibling study scanned each of these separately. Here they are scanned
# jointly, because a filter with the right time constant and no delay and a
# delay with no filter can produce similar peak cross-correlations while
# implying entirely different physics.

def apply_operator(series, delay=0, tau=0.0, dt_hours=1.0):
    """
    Apply a thermal inertia followed by a transport delay.

    Order matters and follows the physics: the structure first integrates the
    forcing, then the resulting thermal state takes time to reach the depth that
    moves the instrument.

    Parameters
    ----------
    series : pd.Series
        Driver on a regular index.
    delay : int, optional
        Transport delay in steps. Default ``0``.
    tau : float, optional
        Thermal time constant in hours. ``0`` disables the filter. Default
        ``0.0``.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.Series
        Transformed driver, same index.
    """
    out = tc.thermal_lag_filter(series, tau, dt_hours=dt_hours)
    if delay:
        out = out.shift(delay)
    return out


def band_limit(series, hours=168, dt_hours=1.0):
    """
    Remove the slow component of a series by subtracting a centred rolling mean.

    The diurnal and the seasonal responses of this structure do not share a sign,
    let alone a time constant: the sibling study found the inclination leading
    air temperature by roughly two months at the annual scale while following it
    within the hour at the daily scale. A scan run on the raw series mixes the
    two and reports an operator that suits neither. Band-limiting to periods
    shorter than ``hours`` isolates the diurnal response, which is the one a
    causal filter can represent.

    Parameters
    ----------
    series : pd.Series
        Source series on a regular index.
    hours : int, optional
        Width of the rolling mean, in hours. Default ``168`` (one week).
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.Series
        Band-limited series, same index.
    """
    window = max(int(round(hours / dt_hours)), 3)
    slow = series.rolling(window, center=True,
                          min_periods=max(3, window // 4)).mean()
    return series - slow


def lag_inertia_scan(df, target, driver, delays, taus, dt_hours=1.0,
                     detrend_hours=None):
    """
    Score every combination of transport delay and thermal time constant.

    Each cell holds the squared Pearson correlation between the target and the
    driver transformed by that operator pair, which for a single predictor is the
    explained variance of the corresponding linear fit. The sign of the
    correlation is kept separately, because a strong negative correlation is a
    real relationship and a scan that reports only magnitude cannot distinguish
    it from a positive one.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    target : str
        Column to explain.
    driver : str
        Column to transform.
    delays : iterable of int
        Transport delays, in steps.
    taus : iterable of float
        Thermal time constants, in hours.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    detrend_hours : int or None, optional
        If given, both series are band-limited to periods shorter than this
        before scoring. Default ``None``.

    Returns
    -------
    pd.DataFrame
        Long-form table with ``delay_h``, ``tau_h``, ``r``, ``r2`` and ``n``.
    """
    y = df[target]
    x0 = df[driver]
    if detrend_hours:
        y = band_limit(y, detrend_hours, dt_hours)
        x0 = band_limit(x0, detrend_hours, dt_hours)

    rows = []
    for tau in taus:
        filtered = apply_operator(x0, delay=0, tau=tau, dt_hours=dt_hours)
        for d in delays:
            x = filtered.shift(d) if d else filtered
            both = pd.concat([y, x], axis=1).dropna()
            if len(both) < 100:
                r = np.nan
            else:
                r = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
            rows.append({'delay_h': d * dt_hours, 'tau_h': tau, 'r': r,
                         'r2': r ** 2 if r == r else np.nan, 'n': len(both)})
    return pd.DataFrame(rows)


def best_operator(scan):
    """
    Pick the operator pair with the highest explained variance.

    Parameters
    ----------
    scan : pd.DataFrame
        Output of :func:`lag_inertia_scan`.

    Returns
    -------
    dict
        ``delay_h``, ``tau_h``, ``r``, ``r2`` at the maximum.
    """
    row = scan.loc[scan['r2'].idxmax()]
    return {'delay_h': float(row['delay_h']), 'tau_h': float(row['tau_h']),
            'r': float(row['r']), 'r2': float(row['r2'])}


def operator_table(df, target, drivers, delays, taus, dt_hours=1.0,
                   detrend_hours=None):
    """
    Run :func:`lag_inertia_scan` for several drivers and summarise the optima.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Column to explain.
    drivers : list of str
        Columns to scan.
    delays : iterable of int
        Transport delays, in steps.
    taus : iterable of float
        Thermal time constants, in hours.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    detrend_hours : int or None, optional
        Band-limit width. Default ``None``.

    Returns
    -------
    summary : pd.DataFrame
        One row per driver, indexed by driver name, with the best operator, its
        explained variance, and the explained variance of the instantaneous
        operator for comparison.
    scans : dict
        Driver name to the full scan table, for plotting.
    """
    scans, rows = {}, []
    for drv in drivers:
        scan = lag_inertia_scan(df, target, drv, delays, taus,
                                dt_hours=dt_hours,
                                detrend_hours=detrend_hours)
        scans[drv] = scan
        best = best_operator(scan)
        flat = scan[(scan['delay_h'] == 0) & (scan['tau_h'] == 0)]
        instant = float(flat['r2'].iloc[0]) if len(flat) else np.nan
        rows.append({
            'driver': drv,
            'delay_h': best['delay_h'],
            'tau_h': best['tau_h'],
            'r': round(best['r'], 4),
            'r2': round(best['r2'], 4),
            'r2_instantaneous': round(instant, 4),
            'gain_from_operator': round(best['r2'] - instant, 4),
        })
    summary = pd.DataFrame(rows).set_index('driver')
    return summary, scans


def operators_from_summary(summary, dt_hours=1.0, causal_only=True,
                           verbose=False):
    """
    Convert an operator summary into the mapping the model builders expect.

    A scan run over signed delays can return a *negative* optimum, meaning the
    driver lags the response rather than leading it. That is a meaningful
    physical measurement — see :func:`cross_phase` — but it is not something a
    forecasting system can use, because applying it would require the driver's
    future values at the moment the forecast is issued. ``causal_only`` clamps
    such an optimum to zero so that a diagnostic scan can never leak into a
    predictive model.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of :func:`operator_table`.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    causal_only : bool, optional
        Clamp negative delays to zero. Default ``True``. Set ``False`` only for
        diagnostic use, never to build features for a deployable model.
    verbose : bool, optional
        Report any clamping that took place. Default ``False``.

    Returns
    -------
    dict
        Driver name to ``{'delay': steps, 'tau': hours}``.
    """
    out, clamped = {}, []
    for drv, row in summary.iterrows():
        delay = int(round(row['delay_h'] / dt_hours))
        if causal_only and delay < 0:
            clamped.append((drv, delay))
            delay = 0
        out[drv] = {'delay': delay, 'tau': float(row['tau_h'])}
    if verbose and clamped:
        for drv, delay in clamped:
            print(f'  causal guard: {drv} optimum at {delay * dt_hours:+.0f} h '
                  f'is a lead; clamped to 0 for predictive use')
    return out


def cross_phase(df, pairs, delays, detrend_hours=None, dt_hours=1.0):
    """
    Signed phase between pairs of channels, on the band of interest.

    Unlike :func:`lag_inertia_scan`, this admits **negative** delays, so it can
    report that the first series of a pair *lags* the second. That direction is
    excluded from every predictive model in this study, but it carries physical
    information that a causal scan destroys.

    The case this exists for is the wall-temperature probe. The probe sits at an
    unknown depth inside the masonry, while the deformation is governed by the
    temperature field near the sun-exposed face. A probe close to the surface
    reads a fast signal that the bulk deformation then follows, so the
    inclination lags it. A probe set deep reads a damped and delayed signal, so
    the deformation happens first and the inclination *leads* it. The sign of the
    phase measured here therefore discriminates between a shallow and a deep
    installation, and its magnitude is the quantity to check against the depth
    once that depth is established.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    pairs : list of tuple of str
        ``(a, b)`` pairs. A positive optimum means ``b`` must be shifted forward
        to align with ``a``, that is ``b`` leads ``a``; a negative optimum means
        ``b`` lags ``a``.
    delays : iterable of int
        Signed delays to scan, in steps.
    detrend_hours : int or None, optional
        Band-limit width. Default ``None``.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per pair, indexed by ``a`` and ``b``, with the signed optimum in
        hours, the correlation there, the correlation at zero delay, and the
        gain the phase buys.
    """
    rows = []
    for a, b in pairs:
        y, x0 = df[a], df[b]
        if detrend_hours:
            y = band_limit(y, detrend_hours, dt_hours)
            x0 = band_limit(x0, detrend_hours, dt_hours)
        best_r, best_d, r_zero = np.nan, np.nan, np.nan
        for d in delays:
            x = x0.shift(d) if d else x0
            both = pd.concat([y, x], axis=1).dropna()
            if len(both) < 100:
                continue
            r = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
            if d == 0:
                r_zero = r
            if not (best_r == best_r) or abs(r) > abs(best_r):
                best_r, best_d = r, d
        rows.append({
            'a': a, 'b': b,
            'phase_h': best_d * dt_hours if best_d == best_d else np.nan,
            'r_at_phase': round(best_r, 4) if best_r == best_r else np.nan,
            'r_at_zero': round(r_zero, 4) if r_zero == r_zero else np.nan,
            'r2_gain': (round(best_r ** 2 - r_zero ** 2, 4)
                        if best_r == best_r and r_zero == r_zero else np.nan),
        })
    return pd.DataFrame(rows).set_index(['a', 'b'])


def build_features(df, drivers, operators, dt_hours=1.0):
    """
    Apply each driver's fitted operator and return the resulting feature frame.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    drivers : list of str
        Columns to transform.
    operators : dict
        Output of :func:`operators_from_summary`.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One column per driver, keeping the driver's own name.
    """
    out = pd.DataFrame(index=df.index)
    for drv in drivers:
        op = operators.get(drv, {'delay': 0, 'tau': 0.0})
        out[drv] = apply_operator(df[drv], delay=op['delay'], tau=op['tau'],
                                  dt_hours=dt_hours)
    return out


# ──────────────────────────────────────────────────────────────────────
# Cross-validation machinery
# ──────────────────────────────────────────────────────────────────────

def rolling_origin_splits(n, n_splits=5, min_train_frac=0.5):
    """
    Generate expanding-window chronological train/test splits.

    Every test block lies strictly after its training block, so no model is ever
    scored on data that precedes what it was fitted on. Reporting a single split
    would make the whole ranking depend on where that one cut happened to fall;
    several origins turn that dependence into a spread that can be inspected.

    Parameters
    ----------
    n : int
        Number of samples.
    n_splits : int, optional
        Number of folds. Default ``5``.
    min_train_frac : float, optional
        Fraction of the record reserved for the first training block. Default
        ``0.5``.

    Yields
    ------
    tuple of slice
        ``(train_slice, test_slice)``.
    """
    start = int(n * min_train_frac)
    block = max((n - start) // n_splits, 1)
    for k in range(n_splits):
        train_end = start + k * block
        test_end = n if k == n_splits - 1 else train_end + block
        if train_end < 50 or test_end <= train_end:
            continue
        yield slice(0, train_end), slice(train_end, test_end)


def ridge_cv(X, y, n_splits=5, min_train_frac=0.5, alpha=1.0,
             return_predictions=False):
    """
    Score a feature set by pooled error over rolling-origin folds.

    Ridge with standardised inputs is used throughout the screening stages
    because it is fast enough for an exhaustive subset search and stable under
    the strong collinearity these channels exhibit. It is not the model the
    project deploys; its role is to order candidate predictor sets cheaply so
    that NeuralProphet is run only on the survivors.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    n_splits : int, optional
        Number of folds. Default ``5``.
    min_train_frac : float, optional
        First training fraction. Default ``0.5``.
    alpha : float, optional
        Ridge penalty. Default ``1.0``.
    return_predictions : bool, optional
        Also return the pooled out-of-sample predictions. Default ``False``.

    Returns
    -------
    metrics : dict
        ``MAE``, ``RMSE``, ``R2``, ``MAE_sd`` (spread across folds), ``n_test``
        and ``n_features``.
    predictions : pd.Series
        Only when ``return_predictions`` is true.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    both = X.join(y.rename('__y__')).dropna()
    if len(both) < 200:
        empty = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAE_sd': np.nan,
                 'n_test': 0, 'n_features': X.shape[1]}
        return (empty, pd.Series(dtype=float)) if return_predictions else empty

    Xc = both.drop(columns='__y__')
    yc = both['__y__']

    preds, fold_mae = [], []
    for tr, te in rolling_origin_splits(len(both), n_splits, min_train_frac):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(Xc.iloc[tr], yc.iloc[tr])
        p = pd.Series(model.predict(Xc.iloc[te]), index=Xc.index[te])
        preds.append(p)
        fold_mae.append(float((p - yc.iloc[te]).abs().mean()))

    if not preds:
        empty = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAE_sd': np.nan,
                 'n_test': 0, 'n_features': X.shape[1]}
        return (empty, pd.Series(dtype=float)) if return_predictions else empty

    pooled = pd.concat(preds)
    truth = yc.reindex(pooled.index)
    err = pooled - truth
    ss_res = float((err ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    metrics = {
        'MAE': float(err.abs().mean()),
        'RMSE': float(np.sqrt((err ** 2).mean())),
        'R2': 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        'MAE_sd': float(np.std(fold_mae)),
        'n_test': int(len(pooled)),
        'n_features': int(Xc.shape[1]),
    }
    return (metrics, pooled) if return_predictions else metrics


# ──────────────────────────────────────────────────────────────────────
# Question 1 — the best single predictor
# ──────────────────────────────────────────────────────────────────────

def rank_single_predictors(df, target, drivers, operators, dt_hours=1.0,
                           n_splits=5, min_train_frac=0.5, alpha=1.0,
                           full_record=None):
    """
    Rank each driver on its own out-of-sample skill, without autoregression.

    Autoregression is deliberately excluded. One hour ahead, the inclination's
    own recent history explains almost all of its variance, and every driver
    added on top of it looks equally negligible: a ranking computed that way
    measures how little room the autoregressive term leaves, not how much the
    driver knows. Removing the autoregressive term asks the question the study
    actually poses — what does this environmental channel, on its own, say about
    the structure.

    Parameters
    ----------
    df : pd.DataFrame
        Source data for the window being ranked.
    target : str
        Target column.
    drivers : list of str
        Candidate predictors.
    operators : dict
        Fitted operator per driver, from :func:`operators_from_summary`.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    n_splits, min_train_frac, alpha
        Passed to :func:`ridge_cv`.
    full_record : pd.DataFrame or None, optional
        Record against which availability is measured. Defaults to ``df``, but
        the honest denominator is usually the whole era rather than the window
        in which the driver happens to be present.

    Returns
    -------
    pd.DataFrame
        One row per driver, sorted by MAE, with the fitted operator, the
        out-of-sample metrics and the availability of the channel.
    """
    base = df if full_record is None else full_record
    rows = []
    for drv in drivers:
        feats = build_features(df, [drv], operators, dt_hours=dt_hours)
        met = ridge_cv(feats, df[target], n_splits=n_splits,
                       min_train_frac=min_train_frac, alpha=alpha)
        op = operators.get(drv, {'delay': 0, 'tau': 0.0})
        rows.append({
            'driver': drv,
            'delay_h': op['delay'] * dt_hours,
            'tau_h': op['tau'],
            'MAE_mdeg': round(met['MAE'], 4),
            'RMSE_mdeg': round(met['RMSE'], 4),
            'R2': round(met['R2'], 4),
            'MAE_sd': round(met['MAE_sd'], 4),
            'availability_%': round(100 * availability(base, [drv]), 1),
            'n_test': met['n_test'],
        })
    out = pd.DataFrame(rows).set_index('driver').sort_values('MAE_mdeg')
    best = out['MAE_mdeg'].min()
    out['MAE_vs_best_%'] = (100 * (out['MAE_mdeg'] - best) / best).round(2)
    return out


# ──────────────────────────────────────────────────────────────────────
# Question 2 — combinations, and how to diagnose them
# ──────────────────────────────────────────────────────────────────────

def vif_table(df, cols):
    """
    Variance inflation factors among the candidate predictors.

    These channels are physically coupled — radiation heats the air, the air and
    the radiation together heat the wall — so a regression that includes several
    of them distributes one physical effect across correlated coefficients in a
    way that is not stable between folds. The variance inflation factor
    quantifies that instability before any model is read for interpretation.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    cols : list of str
        Columns to test.

    Returns
    -------
    pd.DataFrame
        One row per column with its VIF and its correlation with the strongest
        of the others.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    data = df[cols].dropna()
    mat = sm.add_constant(data.values)
    rows = []
    corr = data.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    for i, col in enumerate(cols):
        rows.append({
            'column': col,
            'VIF': round(float(variance_inflation_factor(mat, i + 1)), 2),
            'max_|r|_with_others': round(float(corr[col].max()), 3),
            'closest': corr[col].idxmax(),
        })
    return pd.DataFrame(rows).set_index('column')


def subset_search(df, target, drivers, operators, dt_hours=1.0, n_splits=5,
                  min_train_frac=0.5, alpha=1.0, full_record=None):
    """
    Score every non-empty subset of the candidate predictors.

    With four candidates the search is exhaustive at fifteen fits, which removes
    any need for a stepwise procedure and the selection bias that comes with one.
    Each subset is scored by pooled rolling-origin error and, separately, by the
    Bayesian information criterion of the corresponding ordinary least-squares
    fit, so that a subset which wins on error only by spending parameters is
    visible as such.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Target column.
    drivers : list of str
        Candidate predictors.
    operators : dict
        Fitted operator per driver.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    n_splits, min_train_frac, alpha
        Passed to :func:`ridge_cv`.
    full_record : pd.DataFrame or None, optional
        Denominator for availability. Defaults to ``df``.

    Returns
    -------
    pd.DataFrame
        One row per subset, sorted by MAE, with size, metrics, BIC, availability
        and an availability-weighted utility.
    """
    base = df if full_record is None else full_record
    rows = []
    for k in range(1, len(drivers) + 1):
        for combo in itertools.combinations(drivers, k):
            combo = list(combo)
            feats = build_features(df, combo, operators, dt_hours=dt_hours)
            met = ridge_cv(feats, df[target], n_splits=n_splits,
                           min_train_frac=min_train_frac, alpha=alpha)
            if met['n_test'] == 0:
                continue
            both = feats.join(df[target].rename('__y__')).dropna()
            ols = sm.OLS(both['__y__'],
                         sm.add_constant(both.drop(columns='__y__'))).fit()
            avail = availability(base, combo)
            rows.append({
                'subset': ' + '.join(combo),
                'k': k,
                'MAE_mdeg': round(met['MAE'], 4),
                'RMSE_mdeg': round(met['RMSE'], 4),
                'R2': round(met['R2'], 4),
                'MAE_sd': round(met['MAE_sd'], 4),
                'BIC': round(float(ols.bic), 1),
                'availability_%': round(100 * avail, 1),
                'n_test': met['n_test'],
            })
    out = pd.DataFrame(rows).sort_values('MAE_mdeg').set_index('subset')
    best = out['MAE_mdeg'].min()
    out['MAE_vs_best_%'] = (100 * (out['MAE_mdeg'] - best) / best).round(2)
    # Utility prices a subset by what it delivers over the record as a whole
    # rather than over the window in which it happens to be complete: skill on
    # the fraction of time the subset exists, and nothing on the rest.
    out['utility'] = ((best / out['MAE_mdeg']) *
                      (out['availability_%'] / 100.0)).round(4)
    return out


def incremental_gain(df, target, subset, operators, dt_hours=1.0, n_splits=5,
                     min_train_frac=0.5, alpha=1.0):
    """
    Measure what each member of a subset contributes given all the others.

    Leaving one predictor out and re-scoring is the diagnostic that survives
    collinearity. A regression coefficient on a channel that is 0.9-correlated
    with another says nothing reliable about importance; the change in
    out-of-sample error when the channel is removed says exactly what dropping
    it would cost.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Target column.
    subset : list of str
        Members of the combination under test.
    operators : dict
        Fitted operator per driver.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    n_splits, min_train_frac, alpha
        Passed to :func:`ridge_cv`.

    Returns
    -------
    pd.DataFrame
        One row per member with the full-subset MAE, the MAE without it, and the
        increase caused by its removal, in millidegrees and per cent.
    """
    full_feats = build_features(df, subset, operators, dt_hours=dt_hours)
    full = ridge_cv(full_feats, df[target], n_splits=n_splits,
                    min_train_frac=min_train_frac, alpha=alpha)

    rows = []
    for drv in subset:
        rest = [d for d in subset if d != drv]
        if rest:
            feats = build_features(df, rest, operators, dt_hours=dt_hours)
            met = ridge_cv(feats, df[target], n_splits=n_splits,
                           min_train_frac=min_train_frac, alpha=alpha)
            mae_without = met['MAE']
        else:
            # Removing the only member leaves the mean of the training block.
            mae_without = float(
                (df[target] - df[target].mean()).abs().mean())
        rows.append({
            'driver': drv,
            'MAE_full': round(full['MAE'], 4),
            'MAE_without': round(mae_without, 4),
            'gain_mdeg': round(mae_without - full['MAE'], 4),
            'gain_%': round(100 * (mae_without - full['MAE']) / mae_without, 2),
        })
    return (pd.DataFrame(rows).set_index('driver')
            .sort_values('gain_mdeg', ascending=False))


def permutation_importance(df, target, subset, operators, dt_hours=1.0,
                           n_splits=5, min_train_frac=0.5, alpha=1.0,
                           n_repeats=10, seed=0):
    """
    Importance by destroying one predictor's time alignment in the test folds.

    The model is fitted once per fold on the intact data; each predictor is then
    shuffled within the test block and the increase in error is recorded. Unlike
    the leave-one-out gain, this keeps the fitted model fixed, so it measures how
    much the deployed model *relies* on the channel rather than how much the
    channel would be missed if the model were refitted without it. The two
    disagree exactly when collinearity lets a refit recover the lost information
    from a correlated channel, and that disagreement is itself the diagnosis.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target : str
        Target column.
    subset : list of str
        Members of the combination under test.
    operators : dict
        Fitted operator per driver.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.
    n_splits, min_train_frac, alpha
        Passed to the same cross-validation scheme as :func:`ridge_cv`.
    n_repeats : int, optional
        Shuffles per predictor per fold. Default ``10``.
    seed : int, optional
        Random seed. Default ``0``.

    Returns
    -------
    pd.DataFrame
        One row per member with the mean and standard deviation of the MAE
        increase caused by shuffling it.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    feats = build_features(df, subset, operators, dt_hours=dt_hours)
    both = feats.join(df[target].rename('__y__')).dropna()
    X, y = both.drop(columns='__y__'), both['__y__']
    rng = np.random.default_rng(seed)

    increases = {d: [] for d in subset}
    for tr, te in rolling_origin_splits(len(both), n_splits, min_train_frac):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X.iloc[tr], y.iloc[tr])
        Xte, yte = X.iloc[te], y.iloc[te]
        base = float(np.abs(model.predict(Xte) - yte).mean())
        for drv in subset:
            for _ in range(n_repeats):
                shuffled = Xte.copy()
                shuffled[drv] = rng.permutation(shuffled[drv].values)
                mae = float(np.abs(model.predict(shuffled) - yte).mean())
                increases[drv].append(mae - base)

    rows = [{'driver': d,
             'MAE_increase': round(float(np.mean(v)), 4),
             'sd': round(float(np.std(v)), 4)}
            for d, v in increases.items()]
    return (pd.DataFrame(rows).set_index('driver')
            .sort_values('MAE_increase', ascending=False))


# ──────────────────────────────────────────────────────────────────────
# Question 3 — horizon and uncertainty
# ──────────────────────────────────────────────────────────────────────

def build_direct(df, target, features, n_lags, horizon, future_known):
    """
    Assemble a supervised matrix for direct multi-step forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    target : str
        Column to predict.
    features : list of str
        Exogenous columns, already carrying their fitted operators.
    n_lags : int
        Autoregressive depth, in steps.
    horizon : int
        Forecast horizon, in steps.
    future_known : bool
        If true the exogenous columns are read at the predicted timestamp, which
        assumes a perfect forecast of them; if false they are read at the time
        the forecast is issued.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    frame = pd.DataFrame(index=df.index)
    for k in range(1, n_lags + 1):
        frame[f'{target}_lag{k}'] = df[target].shift(horizon + k - 1)
    for f in features:
        frame[f] = df[f] if future_known else df[f].shift(horizon)
    both = frame.join(df[target].rename('__y__')).dropna()
    return both.drop(columns='__y__'), both['__y__']


def baseline_errors(series, horizons, test_index, dt_hours=1.0):
    """
    Error of the two reference forecasts a model must beat.

    Persistence carries the last observation forward and is the standard against
    which any short-horizon claim is measured. The seasonal-naive forecast
    repeats the value from an integer number of days earlier, which is the
    cheapest way to exploit the diurnal cycle and is therefore the standard for
    horizons beyond a few hours. A model that beats neither has no forecasting
    content, whatever its absolute error looks like.

    Parameters
    ----------
    series : pd.Series
        Observed target on a regular index.
    horizons : iterable of int
        Horizons in steps.
    test_index : pd.DatetimeIndex
        Timestamps over which the comparison is made.
    dt_hours : float, optional
        Step length in hours. Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        One row per horizon with the MAE of both baselines.
    """
    per_day = max(int(round(24.0 / dt_hours)), 1)
    rows = []
    for h in horizons:
        pers = series.shift(h)
        cycles = max(int(np.ceil(h / per_day)), 1)
        seas = series.shift(cycles * per_day)
        both = pd.DataFrame({'y': series, 'p': pers, 's': seas}) \
            .reindex(test_index).dropna()
        if both.empty:
            continue
        rows.append({
            'horizon_h': h * dt_hours,
            'MAE_persistence': round(float((both['p'] - both['y'])
                                           .abs().mean()), 4),
            'MAE_seasonal_naive': round(float((both['s'] - both['y'])
                                              .abs().mean()), 4),
            'n': len(both),
        })
    return pd.DataFrame(rows).set_index('horizon_h')


def run_horizon_experiment(df, target, regressors, max_horizon, horizons,
                           freq=tc.ANALYSIS_FREQ, regime='lagged', n_lags=24,
                           epochs=20, quantiles=(0.05, 0.95), train_frac=0.7,
                           seed=0, verbose=False, keep_forecast=False,
                           yearly=False, fill_limit_h=None,
                           n_changepoints=None):
    """
    Fit one NeuralProphet model that forecasts every horizon at once.

    ``n_forecasts`` is set to the longest horizon of interest, so a single fit
    produces ``yhat1 … yhat{max_horizon}`` and the whole horizon sweep comes from
    one training run rather than from one run per horizon. The multi-horizon head
    is trained jointly, which can cost a little accuracy at any individual
    horizon relative to a model dedicated to it; the study checks that cost
    separately rather than assuming it away.

    Two regimes are supported and the difference between them is the point of the
    exercise. Under ``'lagged'`` the regressors enter only through their past
    values, which is what a deployed system has. Under ``'future'`` the
    regressors are known over the forecast window, which is what a system coupled
    to a perfect weather forecast would have. The first is the honest number, the
    second is the ceiling, and reporting only one of them misstates the result in
    a predictable direction.

    Gaps are bridged by interpolation so that the autoregressive window has
    continuous history, and the positions that were originally missing are
    recorded so that scoring happens only at observed timestamps. The model may
    read a filled value; it is never rewarded for reproducing one.

    That unlimited bridging is safe on a window chosen for being unbroken, and
    indefensible on a record whose longest interruption runs to months, because
    it would fabricate every one of those hours. ``fill_limit_h`` caps the length
    of an interruption the interpolation is allowed to cross. A record needing
    that cap should go to :func:`run_horizon_segments` instead, which fits the
    same model across every complete stretch without inventing anything.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index.
    target : str
        Target column.
    regressors : list of str
        Exogenous columns, already carrying their fitted operators.
    max_horizon : int
        Longest horizon, in steps. Becomes ``n_forecasts``.
    horizons : iterable of int
        Horizons to report, in steps. Must not exceed ``max_horizon``.
    freq : str, optional
        Sampling frequency string. Default :data:`tc_lib.ANALYSIS_FREQ`.
    regime : {'lagged', 'future'}, optional
        Regressor regime. Default ``'lagged'``.
    n_lags : int, optional
        Autoregressive depth. Default ``24``.
    epochs : int, optional
        Training epochs. Default ``20``.
    quantiles : tuple of float, optional
        Lower and upper prediction quantiles. Default ``(0.05, 0.95)``.
    train_frac : float, optional
        Chronological training fraction. Default ``0.7``.
    seed : int, optional
        Random seed, applied before fitting. Default ``0``.
    verbose : bool, optional
        Let NeuralProphet print progress. Default ``False``.
    keep_forecast : bool, optional
        Also return the raw forecast frame, for the fan chart. Default
        ``False``.
    yearly : bool, optional
        Enable the yearly Fourier seasonality. Default ``False``, which is
        appropriate for a window too short to identify an annual term and wrong
        for one that is not; set it deliberately rather than by inheritance.
    fill_limit_h : int or None, optional
        Longest interruption, in hours, that the interpolation may cross. ``None``
        places no limit and reproduces the historical behaviour of this function.
        When a limit is given, NeuralProphet's own imputation is switched off so
        that the fill applied here is the only one performed.
    n_changepoints : int or None, optional
        Override the number of trend changepoints. ``None`` keeps the
        NeuralProphet default. Worth setting to zero on a record with long
        interruptions, because changepoints are spread uniformly over the span
        and those landing inside an interruption are constrained by no
        observation.

    Returns
    -------
    results : pd.DataFrame
        One row per horizon with MAE, RMSE, the empirical coverage of the
        prediction interval, its median width, and the number of scored points.
    forecast : pd.DataFrame
        Only when ``keep_forecast`` is true.
    """
    import logging
    from neuralprophet import NeuralProphet, set_log_level, set_random_seed

    if not verbose:
        set_log_level('ERROR')
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    set_random_seed(seed)

    cols = [target] + list(regressors)
    grid = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    data = df[cols].reindex(grid)
    observed = data[target].notna()
    limit_steps = _fill_limit_steps(fill_limit_h, freq)
    filled = data.apply(lambda s: _fill_short_runs(s, limit_steps))

    frame = pd.DataFrame({'ds': grid, 'y': filled[target].values})
    for r in regressors:
        frame[r] = filled[r].values

    cut = int(len(frame) * train_frac)
    train, test = frame.iloc[:cut], frame.iloc[cut:]

    np_kwargs = dict(
        n_lags=n_lags,
        n_forecasts=max_horizon,
        quantiles=list(quantiles),
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=True,
        epochs=epochs,
        learning_rate=0.01,
        trend_reg=0.0,
        # With a fill limit in force the interpolation above is deliberately
        # partial, and NeuralProphet's own imputer would quietly undo that by
        # filling a further ten steps in each direction plus a rolling mean.
        impute_missing=(limit_steps is None),
        drop_missing=False,
    )
    if n_changepoints is not None:
        np_kwargs['n_changepoints'] = n_changepoints
    model = NeuralProphet(**np_kwargs)
    for r in regressors:
        if regime == 'future':
            model.add_future_regressor(r)
        else:
            model.add_lagged_regressor(r)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(train, freq=freq, progress=None)
        forecast = model.predict(pd.concat([train.tail(n_lags + max_horizon),
                                            test]))

    fc = forecast.set_index('ds')
    truth = df[target].reindex(fc.index)
    was_observed = observed.reindex(fc.index).fillna(False)

    results = _score_horizons(fc, truth, was_observed, horizons, quantiles,
                              regime)
    if keep_forecast:
        return results, fc
    return results


def _score_horizons(fc, truth, was_observed, horizons, quantiles, regime):
    """
    Score a forecast frame at each horizon, at observed timestamps only.

    Shared by every configuration in this module so that a result computed one
    way is comparable with a result computed another. The model may have read an
    interpolated value; it is never scored on reproducing one.

    Parameters
    ----------
    fc : pd.DataFrame
        Forecast frame indexed by timestamp, carrying ``yhat{h}`` columns.
    truth : pd.Series
        Observed target on the same index.
    was_observed : pd.Series
        Boolean, true where the target was measured rather than filled.
    horizons : iterable of int
        Horizons to report, in steps.
    quantiles : tuple of float
        Lower and upper prediction quantiles.
    regime : str
        Recorded on the result for downstream labelling.

    Returns
    -------
    pd.DataFrame
        One row per horizon, indexed by ``horizon_h``.
    """
    lo_name, hi_name = _quantile_columns(fc, quantiles)

    rows = []
    for h in horizons:
        col = f'yhat{h}'
        if col not in fc.columns:
            continue
        yhat = fc[col]
        ok = yhat.notna() & truth.notna() & was_observed
        if int(ok.sum()) == 0:
            continue
        err = yhat[ok] - truth[ok]
        entry = {
            'horizon_h': h,
            'MAE_mdeg': round(float(err.abs().mean()), 4),
            'RMSE_mdeg': round(float(np.sqrt((err ** 2).mean())), 4),
            'n': int(ok.sum()),
        }
        lo_col, hi_col = lo_name.format(h=h), hi_name.format(h=h)
        if lo_col in fc.columns and hi_col in fc.columns:
            lo, hi = fc[lo_col][ok], fc[hi_col][ok]
            inside = (truth[ok] >= lo) & (truth[ok] <= hi)
            entry['coverage_%'] = round(100 * float(inside.mean()), 1)
            entry['PI_width_mdeg'] = round(float((hi - lo).median()), 4)
        rows.append(entry)

    results = pd.DataFrame(rows).set_index('horizon_h')
    results.attrs['regime'] = regime
    results.attrs['nominal_coverage_%'] = round(
        100 * (max(quantiles) - min(quantiles)), 1)
    return results


def run_horizon_segments(df, target, regressors, max_horizon, horizons,
                         freq=tc.ANALYSIS_FREQ, regime='lagged', n_lags=24,
                         epochs=20, quantiles=(0.05, 0.95), train_frac=0.7,
                         seed=0, verbose=False, keep_forecast=False,
                         yearly=True, fill_limit_h=6, n_changepoints=None,
                         min_segment=None):
    """
    Fit one model across every complete stretch of a gapped record.

    :func:`run_horizon_experiment` needs a window that is continuous, or that can
    honestly be made continuous, and on a multi-year record interrupted for
    months at a time neither is available. Restricting the fit to the longest
    unbroken block answers that by discarding most of the record, which costs
    both training data and — more seriously — the seasonal span that a yearly
    component needs in order to be identified at all.

    This function keeps the whole record instead. The short interruptions are
    interpolated, since filling them is cheap and each one would otherwise
    destroy a full window's worth of samples; every stretch that survives is then
    presented to NeuralProphet as a separate series through its ``ID`` column.
    The model fits one set of shared components across all of them, so the
    seasonal terms see every year in the record, while no autoregressive window
    is ever formed across an interruption and not one hour of the long outages is
    invented.

    Treating the stretches as separate series has a second benefit that matters
    for an inclinometer. Normalisation is per series by default, so an offset
    introduced by the instrument being serviced during a long outage does not
    propagate into the fit as though it were structural movement.

    The obvious alternative, NeuralProphet's own ``drop_missing``, does not work
    in version 0.8.0: training on a frame that actually requires a drop returns a
    NaN loss from the first epoch and leaves every weight NaN, and the prediction
    path raises a length mismatch. Both were verified on this record.

    Parameters
    ----------
    df : pd.DataFrame
        Source data on a regular index, gaps present.
    target : str
        Target column.
    regressors : list of str
        Exogenous columns, already carrying their fitted operators.
    max_horizon : int
        Longest horizon, in steps. Becomes ``n_forecasts``.
    horizons : iterable of int
        Horizons to report, in steps.
    freq : str, optional
        Sampling frequency string. Default :data:`tc_lib.ANALYSIS_FREQ`.
    regime : {'lagged', 'future'}, optional
        Regressor regime. Default ``'lagged'``.
    n_lags : int, optional
        Autoregressive depth. Default ``24``.
    epochs : int, optional
        Training epochs. Default ``20``.
    quantiles : tuple of float, optional
        Lower and upper prediction quantiles. Default ``(0.05, 0.95)``.
    train_frac : float, optional
        Chronological training fraction, applied inside each stretch so that
        every part of the record contributes to both sides. Default ``0.7``.
    seed : int, optional
        Random seed. Default ``0``.
    verbose : bool, optional
        Let NeuralProphet print progress. Default ``False``.
    keep_forecast : bool, optional
        Also return the forecast frame. Default ``False``.
    yearly : bool, optional
        Enable the yearly Fourier seasonality. Default ``True``, which is the
        point of using the whole record.
    fill_limit_h : int or None, optional
        Longest interruption the interpolation may cross, in hours. Default
        ``6``, the limit the imputation benchmark licenses for interpolation.
    n_changepoints : int or None, optional
        Override the number of trend changepoints. ``None`` keeps the default.
    min_segment : int or None, optional
        Shortest stretch worth keeping, in steps. Defaults to twice one full
        window, below which a stretch cannot be split into train and test.

    Returns
    -------
    results : pd.DataFrame
        One row per horizon, as :func:`run_horizon_experiment` returns.
    forecast : pd.DataFrame
        Only when ``keep_forecast`` is true.
    """
    import logging
    from neuralprophet import NeuralProphet, set_log_level, set_random_seed

    if not verbose:
        set_log_level('ERROR')
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    set_random_seed(seed)

    window = int(n_lags + max_horizon)
    if min_segment is None:
        min_segment = 2 * window

    cols = [target] + list(regressors)
    grid = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    data = df[cols].reindex(grid)
    observed = data[target].notna()
    limit_steps = _fill_limit_steps(fill_limit_h, freq)
    filled = data.apply(lambda s: _fill_short_runs(s, limit_steps))

    complete = filled.notna().all(axis=1).to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, complete.astype(int), 0]))

    train_parts, predict_parts = [], []
    for k, (start, stop) in enumerate(zip(edges[::2], edges[1::2])):
        if (stop - start) < min_segment:
            continue
        chunk = filled.iloc[start:stop]
        piece = pd.DataFrame({'ds': chunk.index, 'y': chunk[target].values,
                              'ID': f'seg{k:03d}'})
        for r in regressors:
            piece[r] = chunk[r].values

        cut = int(len(piece) * train_frac)
        if cut < window or (len(piece) - cut) < window:
            continue
        train_parts.append(piece.iloc[:cut])
        # The forecast origin needs its lags, so each test stretch carries the
        # tail of its own training portion rather than starting cold.
        predict_parts.append(piece.iloc[cut - window:])

    if not train_parts:
        raise ValueError(
            f'no stretch of {min_segment} steps survives a fill limit of '
            f'{fill_limit_h} h')

    train = pd.concat(train_parts, ignore_index=True)
    predict_frame = pd.concat(predict_parts, ignore_index=True)

    np_kwargs = dict(
        n_lags=n_lags,
        n_forecasts=max_horizon,
        quantiles=list(quantiles),
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=True,
        epochs=epochs,
        learning_rate=0.01,
        trend_reg=0.0,
        impute_missing=False,
        drop_missing=False,
    )
    if n_changepoints is not None:
        np_kwargs['n_changepoints'] = n_changepoints
    model = NeuralProphet(**np_kwargs)
    for r in regressors:
        if regime == 'future':
            model.add_future_regressor(r)
        else:
            model.add_lagged_regressor(r)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(train, freq=freq, progress=None)
        forecast = model.predict(predict_frame)

    fc = forecast.set_index('ds')
    fc = fc[~fc.index.duplicated(keep='first')]
    truth = df[target].reindex(fc.index)
    was_observed = observed.reindex(fc.index).fillna(False)

    results = _score_horizons(fc, truth, was_observed, horizons, quantiles,
                              regime)
    results.attrs['n_segments'] = len(train_parts)
    results.attrs['train_rows'] = int(len(train))
    results.attrs['fill_limit_h'] = fill_limit_h
    results.attrs['hours_fabricated'] = float(
        (filled[target].notna() & ~observed).sum())
    if keep_forecast:
        return results, fc
    return results


def _fill_limit_steps(fill_limit_h, freq):
    """
    Convert a fill limit expressed in hours into steps of the analysis grid.

    Parameters
    ----------
    fill_limit_h : int or None
        Limit in hours, or ``None`` for no limit.
    freq : str
        Sampling frequency string of the grid.

    Returns
    -------
    int or None
        The limit in steps, or ``None``.
    """
    if fill_limit_h is None:
        return None
    step_hours = pd.Timedelta(
        pd.tseries.frequencies.to_offset(freq)).total_seconds() / 3600.0
    return int(round(fill_limit_h / step_hours))


def _fill_short_runs(series, limit_steps):
    """
    Interpolate only the missing runs no longer than a stated length.

    ``Series.interpolate(limit=n)`` does not express this: its limit counts
    consecutive fills from each direction, so with ``limit_direction='both'`` a
    run of ``2n`` is filled completely. The rule wanted here is a property of the
    run, not of the direction of travel, so the series is interpolated in full
    and the runs that are too long are then restored to missing.

    Parameters
    ----------
    series : pd.Series
        Series on a datetime index, missing values present.
    limit_steps : int or None
        Longest run, in steps, that may be filled. ``None`` fills every run.

    Returns
    -------
    pd.Series
        The series with the short runs filled and the long ones left missing.
    """
    filled = series.interpolate(method='time', limit_direction='both')
    if limit_steps is None:
        return filled

    missing = series.isna().to_numpy()
    if not missing.any():
        return filled

    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
    out = filled.copy()
    for start, stop in zip(edges[::2], edges[1::2]):
        if (stop - start) > limit_steps:
            out.iloc[start:stop] = np.nan
    return out


def _quantile_columns(fc, quantiles):
    """
    Discover how this NeuralProphet build names its quantile columns.

    The naming has changed between releases — ``yhat1 5.0%`` in some, ``yhat1
    5%`` in others — and a hard-coded pattern silently produces a table with no
    coverage column rather than an error. This probes the frame once and returns
    format strings keyed on the horizon.

    Parameters
    ----------
    fc : pd.DataFrame
        Forecast frame.
    quantiles : tuple of float
        Lower and upper quantiles as fractions.

    Returns
    -------
    tuple of str
        Format strings for the lower and upper column names, each containing
        ``{h}``.
    """
    lo, hi = min(quantiles), max(quantiles)
    for lo_fmt, hi_fmt in [('yhat{h} %.1f%%' % (100 * lo),
                            'yhat{h} %.1f%%' % (100 * hi)),
                           ('yhat{h} %g%%' % (100 * lo),
                            'yhat{h} %g%%' % (100 * hi))]:
        if lo_fmt.format(h=1) in fc.columns:
            return lo_fmt, hi_fmt
    return 'yhat{h}__missing_lo', 'yhat{h}__missing_hi'


def add_skill(results, baselines):
    """
    Attach skill scores against the two reference forecasts.

    Skill is one minus the ratio of errors, so zero means the model matches the
    baseline and negative means it is worse. The horizon at which skill against
    persistence reaches zero is the answer to how far ahead the system can
    usefully forecast.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :func:`run_horizon_experiment`.
    baselines : pd.DataFrame
        Output of :func:`baseline_errors`.

    Returns
    -------
    pd.DataFrame
        ``results`` with ``skill_vs_persistence`` and
        ``skill_vs_seasonal_naive`` added.
    """
    out = results.join(baselines[['MAE_persistence', 'MAE_seasonal_naive']],
                       how='left')
    out['skill_vs_persistence'] = (
        1 - out['MAE_mdeg'] / out['MAE_persistence']).round(4)
    out['skill_vs_seasonal_naive'] = (
        1 - out['MAE_mdeg'] / out['MAE_seasonal_naive']).round(4)
    return out


def horizon_limit(results, column='skill_vs_persistence'):
    """
    Longest horizon at which the model still beats its baseline.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :func:`add_skill`.
    column : str, optional
        Skill column to test. Default ``'skill_vs_persistence'``.

    Returns
    -------
    float or None
        The horizon in hours, or ``None`` if the model never beats the baseline.
    """
    positive = results[results[column] > 0]
    if positive.empty:
        return None
    return float(positive.index.max())


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
#
# Every figure is drawn through seaborn at the context set by
# ``tc.set_context``, so switching between a notebook layout and a manuscript
# layout is one call at the top of the study.

set_context = tc.set_context
figsize = tc.figsize
_finish = tc._finish


def plot_operator_heatmaps(scans, title='', save_path=None, filename=None,
                           ncols=2):
    """
    Explained variance over the delay-by-inertia grid, one panel per driver.

    Parameters
    ----------
    scans : dict
        Driver name to the scan table from :func:`lag_inertia_scan`.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.
    ncols : int, optional
        Panels per row. Default ``2``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(scans)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize(11, 3.6 * nrows),
                             squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        grid = scans[name].pivot(index='tau_h', columns='delay_h', values='r2')
        sns.heatmap(grid, ax=ax, cmap='viridis', cbar_kws={'label': 'R²'},
                    xticklabels=max(len(grid.columns) // 8, 1),
                    yticklabels=1)
        best = best_operator(scans[name])
        ax.set_title(f'{name} — best R² {best["r2"]:.3f} at '
                     f'delay {best["delay_h"]:.0f} h, τ {best["tau_h"]:.0f} h')
        ax.set_xlabel('transport delay [h]')
        ax.set_ylabel('thermal time constant τ [h]')
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_single_ranking(rank, title='', save_path=None, filename=None):
    """
    Out-of-sample error and availability of each single predictor.

    Parameters
    ----------
    rank : pd.DataFrame
        Output of :func:`rank_single_predictors`.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.6))
    order = rank.index.tolist()

    sns.barplot(x=rank['MAE_mdeg'].values, y=order, ax=axes[0],
                hue=order, palette='colorblind', legend=False, orient='h')
    axes[0].errorbar(rank['MAE_mdeg'].values, range(len(order)),
                     xerr=rank['MAE_sd'].values, fmt='none', ecolor='0.25',
                     capsize=3, lw=1)
    axes[0].set_xlabel('out-of-sample MAE [mdeg], no autoregression')
    axes[0].set_ylabel('')

    sns.barplot(x=rank['R2'].values, y=order, ax=axes[1],
                hue=order, palette='colorblind', legend=False, orient='h')
    for i, (r2, av) in enumerate(zip(rank['R2'], rank['availability_%'])):
        axes[1].text(max(r2, 0) + 0.01, i, f'{av:.0f} % available',
                     va='center', fontsize='small')
    axes[1].set_xlabel('out-of-sample R²')
    axes[1].set_ylabel('')
    axes[1].set_xlim(min(0, rank['R2'].min() * 1.1),
                     max(rank['R2'].max() * 1.45, 0.1))
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_subset_search(subsets, title='', save_path=None, filename=None):
    """
    Error against subset size, and utility once availability is priced in.

    Parameters
    ----------
    subsets : pd.DataFrame
        Output of :func:`subset_search`.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 4.0))
    data = subsets.reset_index()

    sns.scatterplot(data=data, x='k', y='MAE_mdeg', hue='availability_%',
                    palette='viridis', s=70, ax=axes[0], legend='brief')
    frontier = data.groupby('k')['MAE_mdeg'].min()
    axes[0].plot(frontier.index, frontier.values, color='0.3', lw=1,
                 ls='--', zorder=0)
    for _, row in data.nsmallest(3, 'MAE_mdeg').iterrows():
        axes[0].annotate(row['subset'], (row['k'], row['MAE_mdeg']),
                         textcoords='offset points', xytext=(6, 4),
                         fontsize='small')
    axes[0].set_xlabel('number of predictors')
    axes[0].set_ylabel('pooled out-of-sample MAE [mdeg]')
    axes[0].set_xticks(sorted(data['k'].unique()))

    top = subsets.nlargest(8, 'utility')
    sns.barplot(x=top['utility'].values, y=top.index.tolist(), ax=axes[1],
                hue=top.index.tolist(), palette='colorblind', legend=False,
                orient='h')
    axes[1].set_xlabel('utility = accuracy ratio × availability')
    axes[1].set_ylabel('')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_diagnostics(gain, importance, title='', save_path=None,
                     filename=None):
    """
    Leave-one-out gain beside permutation importance for the chosen subset.

    Parameters
    ----------
    gain : pd.DataFrame
        Output of :func:`incremental_gain`.
    importance : pd.DataFrame
        Output of :func:`permutation_importance`.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.4))

    sns.barplot(x=gain['gain_mdeg'].values, y=gain.index.tolist(), ax=axes[0],
                hue=gain.index.tolist(), palette='colorblind', legend=False,
                orient='h')
    axes[0].set_xlabel('MAE cost of removing the channel and refitting [mdeg]')
    axes[0].set_ylabel('')

    sns.barplot(x=importance['MAE_increase'].values,
                y=importance.index.tolist(), ax=axes[1],
                hue=importance.index.tolist(), palette='colorblind',
                legend=False, orient='h')
    axes[1].errorbar(importance['MAE_increase'].values,
                     range(len(importance)), xerr=importance['sd'].values,
                     fmt='none', ecolor='0.25', capsize=3, lw=1)
    axes[1].set_xlabel('MAE cost of shuffling the channel, model fixed [mdeg]')
    axes[1].set_ylabel('')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_horizon_skill(results, title='', save_path=None, filename=None):
    """
    Error and skill against horizon, for every regime and window supplied.

    Parameters
    ----------
    results : dict
        Label to the table returned by :func:`add_skill`.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.8))
    for label, res in results.items():
        axes[0].plot(res.index, res['MAE_mdeg'], marker='o', ms=4, lw=1.2,
                     label=label)
        axes[1].plot(res.index, res['skill_vs_persistence'], marker='o', ms=4,
                     lw=1.2, label=label)
    ref = next(iter(results.values()))
    if 'MAE_persistence' in ref:
        axes[0].plot(ref.index, ref['MAE_persistence'], color='0.3', ls='--',
                     lw=1, label='persistence')
        axes[0].plot(ref.index, ref['MAE_seasonal_naive'], color='0.6',
                     ls=':', lw=1, label='seasonal naive')
    axes[1].axhline(0, color='0.3', lw=1, ls='--')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(list(ref.index))
        ax.set_xticklabels([f'{int(h)}' for h in ref.index])
        ax.set_xlabel('forecast horizon [h]')
    axes[0].set_ylabel('MAE [mdeg]')
    axes[1].set_ylabel('skill vs persistence')
    axes[0].legend(fontsize='small')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_uncertainty(results, title='', save_path=None, filename=None):
    """
    Calibration and width of the prediction interval against horizon.

    A nominal ninety per cent interval is only worth quoting if roughly ninety
    per cent of observations fall inside it. The left panel tests that; the right
    panel shows what the interval costs in width, since an interval can always be
    made to cover by being made useless.

    Parameters
    ----------
    results : dict
        Label to the table returned by :func:`run_horizon_experiment`.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize(11, 3.8))
    nominal = None
    for label, res in results.items():
        if 'coverage_%' not in res:
            continue
        nominal = res.attrs.get('nominal_coverage_%', 90.0)
        axes[0].plot(res.index, res['coverage_%'], marker='o', ms=4, lw=1.2,
                     label=label)
        axes[1].plot(res.index, res['PI_width_mdeg'], marker='o', ms=4,
                     lw=1.2, label=label)
    if nominal is not None:
        axes[0].axhline(nominal, color='0.3', ls='--', lw=1,
                        label=f'nominal {nominal:.0f} %')
    for ax in axes:
        ax.set_xscale('log')
        ref = next(iter(results.values()))
        ax.set_xticks(list(ref.index))
        ax.set_xticklabels([f'{int(h)}' for h in ref.index])
        ax.set_xlabel('forecast horizon [h]')
    axes[0].set_ylabel('empirical coverage [%]')
    axes[1].set_ylabel('median interval width [mdeg]')
    axes[0].legend(fontsize='small')
    if title:
        fig.suptitle(title)
    return _finish(fig, save_path, filename)


def plot_fan(forecast, truth, horizon, quantiles=(0.05, 0.95), days=10,
             title='', save_path=None, filename=None):
    """
    One stretch of the test period with the forecast and its interval.

    Parameters
    ----------
    forecast : pd.DataFrame
        Forecast frame indexed by timestamp, as returned by
        :func:`run_horizon_experiment` with ``keep_forecast``.
    truth : pd.Series
        Observed target.
    horizon : int
        Horizon to draw, in steps.
    quantiles : tuple of float, optional
        Quantiles used when fitting. Default ``(0.05, 0.95)``.
    days : int, optional
        Length of the drawn window. Default ``10``.
    title : str, optional
        Figure title.
    save_path, filename : str or None, optional
        Destination for the saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    col = f'yhat{horizon}'
    lo_fmt, hi_fmt = _quantile_columns(forecast, quantiles)
    valid = forecast[col].dropna()
    end = valid.index.max()
    start = end - pd.Timedelta(days=days)
    window = forecast.loc[start:end]

    fig, ax = plt.subplots(figsize=figsize(11, 3.4))
    lo_col, hi_col = lo_fmt.format(h=horizon), hi_fmt.format(h=horizon)
    if lo_col in window and hi_col in window:
        ax.fill_between(window.index, window[lo_col], window[hi_col],
                        alpha=0.25, color=sns.color_palette('colorblind')[0],
                        label=f'{100 * (max(quantiles) - min(quantiles)):.0f} % '
                              f'prediction interval')
    tc._ts(ax, truth.loc[start:end], label='observed', lw=1.0, color='0.2')
    tc._ts(ax, window[col], label=f'forecast, {horizon} h ahead', lw=1.0,
           color=sns.color_palette('colorblind')[0])
    ax.set_ylabel('compensated inclination [mdeg]')
    ax.set_xlabel('')
    ax.legend(fontsize='small')
    if title:
        ax.set_title(title)
    return _finish(fig, save_path, filename)
