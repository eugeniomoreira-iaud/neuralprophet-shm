"""
Prediction helpers for Study 04.

The functions in this module prepare and score prediction experiments without
owning the model itself. They keep the study's choices visible in the notebook:
which channels are required, how many complete segments are needed, which route
is eligible for a given task, and how large a paired bootstrap block should be.
"""

import re
import warnings

import numpy as np
import pandas as pd

from . import coupling

# scipy's lombscargle evaluates every (frequency, sample) pair in one shot
# internally; on a multi-year record at native resolution (order 2e5
# samples) a scan of tens of thousands of candidate periods materialises a
# broadcast array large enough to exhaust ordinary machine memory and kill
# the interpreter with no Python traceback to show for it. period_scan
# below evaluates the periodogram in slices of this many frequencies at a
# time instead — each frequency's normalised power is independent of every
# other, so the sliced result is identical to the one-shot computation,
# just bounded in peak memory.
_LOMBSCARGLE_CHUNK = 2000


def _as_series(values, index, name=None):
    """Return ``values`` as a Series aligned to ``index``."""
    if values is None:
        return None
    if isinstance(values, pd.Series):
        return values.reindex(index)
    return pd.Series(values, index=index, name=name)


def hourly_change(series, era=None, invalid=None, freq='1h'):
    """
    First difference on a regular grid, with unsafe endpoints removed.

    A structural prediction target may use change but must not bridge a data
    gap, an instrument-era boundary, or a sample already rejected by the study.
    The output has the same index as the input; a value is present only where
    the current timestamp is exactly ``freq`` after the previous retained row,
    both values are finite, the two era labels match when supplied, and neither
    endpoint is flagged invalid when supplied.

    Parameters
    ----------
    series : pd.Series
        Numeric signal indexed by timestamp.
    era : pd.Series or array-like or None, optional
        Era label for each timestamp. A change crossing different labels is
        returned as ``NaN``. Default ``None``.
    invalid : pd.Series or array-like or None, optional
        Boolean rejection flag for each timestamp. A change touching a true flag
        is returned as ``NaN``. Default ``None``.
    freq : str, optional
        Expected spacing between adjacent rows. Default ``'1h'``.

    Returns
    -------
    pd.Series
        First difference, aligned to ``series.index``.
    """
    values = pd.to_numeric(series, errors='coerce')
    step = pd.Timedelta(freq)
    out = values.diff()

    index = pd.DatetimeIndex(values.index)
    adjacent = pd.Series(index.to_series().diff().to_numpy() == step,
                         index=values.index)
    valid = adjacent & values.notna() & values.shift(1).notna()

    era_values = _as_series(era, values.index, name='era')
    if era_values is not None:
        valid &= era_values.notna() & era_values.shift(1).notna()
        valid &= era_values == era_values.shift(1)

    invalid_values = _as_series(invalid, values.index, name='invalid')
    if invalid_values is not None:
        bad = invalid_values.fillna(True).astype(bool)
        valid &= ~bad & ~bad.shift(1, fill_value=True)

    return out.where(valid)


def contiguous_segments(frame, required, min_length=1, freq='1h'):
    """
    Complete rows grouped into deterministic contiguous segment IDs.

    Rows with missing values in any required column are removed before
    contiguity is assessed. The remaining rows are split whenever their
    timestamp spacing is not exactly ``freq``. Segments shorter than
    ``min_length`` are dropped. IDs are assigned after dropping short segments,
    in chronological order, as ``S001``, ``S002`` and so on.

    Parameters
    ----------
    frame : pd.DataFrame
        Candidate modelling frame indexed by timestamp.
    required : sequence of str
        Columns that must be non-missing.
    min_length : int, optional
        Minimum number of rows a segment must contain. Default ``1``.
    freq : str, optional
        Expected timestamp spacing. Default ``'1h'``.

    Returns
    -------
    pd.DataFrame
        Complete rows with a ``segment_id`` column.
    """
    required = list(required)
    out = frame.dropna(subset=required).copy()
    out['segment_id'] = pd.Series(dtype=object)
    if out.empty:
        return out

    step = pd.Timedelta(freq)
    index = pd.DatetimeIndex(out.index)
    breaks = index.to_series().diff().ne(step).to_numpy()
    raw_ids = pd.Series(np.cumsum(breaks), index=out.index)
    sizes = raw_ids.groupby(raw_ids).transform('size')
    out = out.loc[sizes >= int(min_length)].copy()
    if out.empty:
        return out

    kept = pd.Series(raw_ids.loc[out.index].to_numpy(), index=out.index)
    labels = {raw: f'S{i:03d}' for i, raw in enumerate(pd.unique(kept), 1)}
    out['segment_id'] = kept.map(labels).to_numpy()
    return out


DEFAULT_GAP_CLASSES = (
    (0.0, 1.0, '<=1h'),
    (1.0, 6.0, '1-6h'),
    (6.0, 24.0, '6-24h'),
    (24.0, 168.0, '1-7d'),
    (168.0, np.inf, '>7d'),
)


def gap_inventory(series, freq='20min', classes=DEFAULT_GAP_CLASSES):
    """
    One row per maximal run of missing slots, classified by duration.

    The series is first reindexed onto a regular grid of spacing ``freq``, so
    that time absent from the index counts as missing rather than disappearing.
    A coverage percentage says how much time is missing; this says how that time
    is shaped, which is what decides whether filling it is interpolation or
    reconstruction.

    Parameters
    ----------
    series : pd.Series
        Numeric signal indexed by timestamp. Missing is ``NaN`` or an absent
        timestamp.
    freq : str, optional
        Spacing of the analysis grid. Default ``'20min'``.
    classes : sequence of (float, float, str), optional
        Half-open duration bins in hours, as ``(low, high, label)``; a gap falls
        in the first bin with ``low < duration_h <= high``. Bins are expected to
        tile the whole range of possible durations; a duration matching no bin
        is not an error, it silently takes the label of the last class in
        ``classes``. Default ``DEFAULT_GAP_CLASSES``.

    Returns
    -------
    pd.DataFrame
        Columns ``start``, ``end``, ``duration_h``, ``n_slots``, ``gap_class``,
        in chronological order. Empty with those columns when nothing is
        missing.
    """
    columns = ['start', 'end', 'duration_h', 'n_slots', 'gap_class']
    values = pd.to_numeric(series, errors='coerce')
    index = pd.DatetimeIndex(values.index)
    if len(index) == 0:
        return pd.DataFrame(columns=columns)

    grid = pd.date_range(index.min(), index.max(), freq=freq)
    values = values.reindex(grid)
    step_hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)

    missing = values.isna().to_numpy()
    if not missing.any():
        return pd.DataFrame(columns=columns)

    positions = np.flatnonzero(missing)
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = positions[np.r_[0, breaks + 1]]
    ends = positions[np.r_[breaks, positions.size - 1]]
    n_slots = ends - starts + 1
    duration_h = n_slots * step_hours

    labels = []
    for hours in duration_h:
        label = classes[-1][2]
        for low, high, name in classes:
            if low < hours <= high:
                label = name
                break
        labels.append(label)

    return pd.DataFrame({
        'start': grid[starts],
        'end': grid[ends],
        'duration_h': duration_h,
        'n_slots': n_slots,
        'gap_class': labels,
    }, columns=columns)


def segment_survival(frame, required, lag_hours, forecast_hours, freq='20min'):
    """
    How many training windows survive contiguous segmentation, per configuration.

    Segmenting a gapped record is not free: a model that consumes ``lag_hours``
    of history and predicts ``forecast_hours`` ahead can only be trained inside a
    run of complete rows long enough to hold both. This counts what is left, so
    that a lag length is chosen against the record rather than against habit.

    Parameters
    ----------
    frame : pd.DataFrame
        Candidate modelling frame indexed by timestamp on a regular grid.
    required : sequence of str
        Columns that must be present for a row to count as complete.
    lag_hours, forecast_hours : int or sequence of int
        Configurations to evaluate. Scalars are broadcast, and every combination
        of the two is reported.
    freq : str, optional
        Spacing of the analysis grid. Default ``'20min'``.

    Returns
    -------
    pd.DataFrame
        One row per configuration, with ``lag_hours``, ``forecast_hours``,
        ``n_rows``, ``coverage``, ``n_segments``, ``median_segment_h``,
        ``max_segment_h``, ``n_surviving`` and ``n_windows``.
    """
    required = list(required)
    step_hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)
    segmented = contiguous_segments(frame, required, min_length=1, freq=freq)

    if segmented.empty:
        lengths = np.array([], dtype=int)
    else:
        lengths = segmented.groupby('segment_id').size().to_numpy()

    lags = np.atleast_1d(lag_hours)
    horizons = np.atleast_1d(forecast_hours)
    rows = []
    for lag in lags:
        for horizon in horizons:
            need = int(round((float(lag) + float(horizon)) / step_hours))
            usable = lengths[lengths >= need] if lengths.size else lengths
            windows = int((usable - need + 1).sum()) if usable.size else 0
            rows.append({
                'lag_hours': int(lag),
                'forecast_hours': int(horizon),
                'n_rows': int(len(segmented)),
                'coverage': (len(segmented) / len(frame)) if len(frame) else np.nan,
                'n_segments': int(lengths.size),
                'median_segment_h': (float(np.median(lengths) * step_hours)
                                     if lengths.size else np.nan),
                'max_segment_h': (float(lengths.max() * step_hours)
                                  if lengths.size else np.nan),
                'n_surviving': int(usable.size),
                'n_windows': windows,
            })
    return pd.DataFrame(rows)


def cadence_evidence(response, driver, era=None, cadences=('20min', '1h'),
                     freq='20min'):
    """
    Statistics that decide the modelling cadence and the prediction target.

    Reported per candidate cadence: the persistence of the level, the memory
    left in its gap-safe first difference, the scale of that difference, its
    coupling to a driver, and the drift implied by its mean. A level whose
    lag-one autocorrelation is near unity cannot be scored honestly, and a
    difference whose lag-one autocorrelation is negative is dominated by
    measurement noise rather than by the increment it is meant to carry.

    Parameters
    ----------
    response : pd.Series
        Structural response, on the finest available grid. Reindexed onto a
        complete grid of spacing ``freq`` before any statistic is computed,
        so that time absent from the index counts as missing rather than
        silently making two non-adjacent samples look adjacent to the
        lag-one autocorrelation.
    driver : pd.Series
        Environmental driver to correlate against, same index.
    era : pd.Series or None, optional
        Instrument-era label per timestamp; a difference crossing a change of
        label is discarded. Default ``None``.
    cadences : sequence of str, optional
        Grids to evaluate. The first must be the native one. Default
        ``('20min', '1h')``.
    freq : str, optional
        Native spacing of the inputs. Default ``'20min'``.

    Returns
    -------
    pd.DataFrame
        One row per cadence, with ``cadence``, ``n_level``, ``n_change``,
        ``level_autocorr1``, ``change_autocorr1``, ``change_std``,
        ``change_mad``, ``corr_level``, ``corr_change`` and ``drift_per_year``.
    """
    response = pd.to_numeric(response, errors='coerce')
    index = pd.DatetimeIndex(response.index)
    if len(index) > 0:
        grid = pd.date_range(index.min(), index.max(), freq=freq)
        response = response.reindex(grid)
    driver = pd.to_numeric(driver, errors='coerce').reindex(response.index)
    era_values = _as_series(era, response.index, name='era')

    rows = []
    for cadence in cadences:
        if cadence == freq:
            level, force = response, driver
            labels = era_values
        else:
            level = response.resample(cadence).mean()
            force = driver.resample(cadence).mean()
            labels = (era_values.resample(cadence).first()
                      if era_values is not None else None)

        change = hourly_change(level, era=labels, freq=cadence)
        driver_change = hourly_change(force, era=labels, freq=cadence)
        steps_per_year = pd.Timedelta(days=365) / pd.Timedelta(cadence)

        level_pair = pd.concat([level, force], axis=1).dropna()
        change_pair = pd.concat([change, driver_change], axis=1).dropna()

        rows.append({
            'cadence': cadence,
            'n_level': int(level.notna().sum()),
            'n_change': int(change.notna().sum()),
            'level_autocorr1': float(level.autocorr(1)),
            'change_autocorr1': float(change.autocorr(1)),
            'change_std': float(change.std()),
            'change_mad': float((change - change.median()).abs().median()),
            'corr_level': (float(level_pair.iloc[:, 0].corr(level_pair.iloc[:, 1]))
                           if len(level_pair) > 1 else np.nan),
            'corr_change': (float(change_pair.iloc[:, 0].corr(change_pair.iloc[:, 1]))
                            if len(change_pair) > 1 else np.nan),
            'drift_per_year': float(change.mean() * steps_per_year),
        })
    return pd.DataFrame(rows)


def expanding_segment_folds(segment_ids, initial_segments, n_folds):
    """
    Chronological expanding folds over whole contiguous segments.

    Each fold trains on the first ``initial_segments`` plus every test group
    used by earlier folds. All remaining segments are partitioned
    chronologically into up to ``n_folds`` non-empty test groups.

    Parameters
    ----------
    segment_ids : sequence
        Segment labels in chronological row order.
    initial_segments : int
        Number of earliest segments in the first training set.
    n_folds : int
        Maximum number of folds to return.

    Returns
    -------
    list of dict
        Fold number, training segment labels, and test segment labels.
    """
    unique = list(pd.unique(pd.Series(segment_ids).dropna()))
    initial_segments = int(initial_segments)
    n_folds = int(n_folds)
    remaining = unique[initial_segments:]
    if initial_segments < 1 or n_folds < 1 or not remaining:
        return []

    n_groups = min(len(remaining), n_folds)
    groups = [list(group) for group in np.array_split(remaining, n_groups)]
    folds = []
    tested = []
    for i, group in enumerate(groups):
        folds.append({
            'fold': i + 1,
            'train_segments': unique[:initial_segments] + tested,
            'test_segments': group,
        })
        tested.extend(group)
    return folds


def availability_route(core_ok, rich_ok, ar_ok=False, task='nowcast',
                       horizon_h=1):
    """
    Select the most informative route available for a prediction timestamp.

    ``nowcast`` can use rich predictors when present, then the core route, and
    otherwise stays unavailable. ``forecast`` can use rich predictors only up to
    a one-day horizon because longer horizons would require future exogenous
    information; after that it falls back to core predictors, then to an
    autoregressive-only route if explicitly allowed.

    Parameters
    ----------
    core_ok : bool
        Core predictors are available.
    rich_ok : bool
        Rich predictors are available.
    ar_ok : bool, optional
        Autoregressive-only route is available. Default ``False``.
    task : {'nowcast', 'forecast'}, optional
        Prediction task. Default ``'nowcast'``.
    horizon_h : int or float, optional
        Forecast horizon in hours. Default ``1``.

    Returns
    -------
    str
        One of ``'rich'``, ``'core'``, ``'ar_only'`` or ``'unavailable'``.
    """
    if task == 'nowcast':
        if rich_ok:
            return 'rich'
        if core_ok:
            return 'core'
        return 'unavailable'
    if task != 'forecast':
        raise ValueError("task must be 'nowcast' or 'forecast'")

    if rich_ok and horizon_h <= 24:
        return 'rich'
    if core_ok:
        return 'core'
    if ar_ok:
        return 'ar_only'
    return 'unavailable'


def _score_group(group, naive_scale=None, alpha=0.10):
    """Compute scalar scores for one already selected group."""
    data = group[['y', 'yhat']].apply(pd.to_numeric, errors='coerce').dropna()
    scores = {
        'n': int(len(data)),
        'mae': np.nan,
        'rmse': np.nan,
        'bias': np.nan,
        'r2': np.nan,
    }
    if len(data):
        error = data['yhat'] - data['y']
        scores['mae'] = float(error.abs().mean())
        scores['rmse'] = float(np.sqrt(np.mean(error ** 2)))
        scores['bias'] = float(error.mean())
        if len(data) > 1:
            total = float(((data['y'] - data['y'].mean()) ** 2).sum())
            if total > 0:
                residual = float((error ** 2).sum())
                scores['r2'] = 1.0 - residual / total

    if {'q05', 'q95'}.issubset(group.columns):
        interval = group[['y', 'q05', 'q95']].apply(
            pd.to_numeric, errors='coerce').dropna()
        if len(interval):
            covered = ((interval['y'] >= interval['q05'])
                       & (interval['y'] <= interval['q95']))
            scores['coverage_q05_q95'] = float(covered.mean())
            scores['width_q05_q95'] = float((interval['q95']
                                             - interval['q05']).median())
        else:
            scores['coverage_q05_q95'] = np.nan
            scores['width_q05_q95'] = np.nan

    # Scale-free error, so that horizons and cadences are comparable on one
    # axis. The scale is the caller's in-sample naive mean absolute error; it is
    # not derived here, because a scale computed on the evaluation rows would
    # make the metric self-referential.
    scores['mase'] = (scores['mae'] / naive_scale
                      if naive_scale not in (None, 0) and np.isfinite(naive_scale)
                      else np.nan)

    lower_col, upper_col = 'q05', 'q95'
    if lower_col in group.columns and upper_col in group.columns:
        y = pd.to_numeric(group['y'], errors='coerce')
        lower = pd.to_numeric(group[lower_col], errors='coerce')
        upper = pd.to_numeric(group[upper_col], errors='coerce')
        complete = y.notna() & lower.notna() & upper.notna()
        y, lower, upper = y[complete], lower[complete], upper[complete]

        # Pinball loss scores each quantile on its own terms rather than only
        # asking whether the pair happened to bracket the observation.
        for column, quantile, forecast in ((f'pinball_{lower_col}', 0.05, lower),
                                           (f'pinball_{upper_col}', 0.95, upper)):
            error = y - forecast
            loss = np.where(error >= 0, quantile * error,
                            (quantile - 1.0) * error)
            scores[column] = float(np.mean(loss)) if len(loss) else np.nan

        # Winkler interval score: width, plus a penalty proportional to how far
        # outside the interval the observation fell.
        width = upper - lower
        penalty = np.where(y < lower, (2.0 / alpha) * (lower - y), 0.0) \
            + np.where(y > upper, (2.0 / alpha) * (y - upper), 0.0)
        scores['interval_score'] = (float(np.mean(width + penalty))
                                    if len(width) else np.nan)
    else:
        scores['pinball_q05'] = np.nan
        scores['pinball_q95'] = np.nan
        scores['interval_score'] = np.nan
    return scores


def score_predictions(frame, group_cols, naive_scale=None, alpha=0.10):
    """
    Score observed and predicted values, optionally by group.

    The input must contain ``y`` and ``yhat``. Optional ``q05`` and ``q95``
    columns are scored as a nominal central interval: coverage is the fraction
    of observed values inside the interval, and width is the median interval
    width. All metrics use only rows complete for the columns they need.

    Parameters
    ----------
    frame : pd.DataFrame
        Prediction table.
    group_cols : sequence of str or None
        Columns defining independent score groups. Use ``None`` or an empty
        sequence for a single pooled score row.
    naive_scale : float, optional
        In-sample mean absolute error of a naive (e.g. persistence) forecast,
        supplied by the caller. Used as the denominator of MASE. When
        ``None`` (the default) or ``0`` or non-finite, ``mase`` is ``NaN``
        rather than the column being absent, so the returned schema never
        depends on this argument.
    alpha : float, optional
        Nominal miscoverage rate of the ``q05``/``q95`` interval, used to
        weight the Winkler interval-score penalty for observations that fall
        outside the interval. Default ``0.10``.

    Returns
    -------
    pd.DataFrame
        One row per group with ``n``, ``mae``, ``rmse``, ``bias``, ``r2``
        and, when interval columns are present, coverage and width, followed
        by ``mase`` (scale-free error against ``naive_scale``),
        ``pinball_q05`` and ``pinball_q95`` (per-quantile pinball loss) and
        ``interval_score`` (Winkler interval score).
    """
    group_cols = list(group_cols or [])
    if not group_cols:
        return pd.DataFrame([_score_group(frame, naive_scale, alpha)])

    rows = []
    grouped = frame.groupby(group_cols, dropna=False, sort=True, observed=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(_score_group(group, naive_scale, alpha))
        rows.append(row)
    return pd.DataFrame(rows)


def conformal_interval(predictions, alpha=0.10, calibration_end=None,
                       y_col='y', yhat_col='yhat'):
    """
    Replace a model's own quantile columns with an empirical conformal interval.

    NeuralProphet's quantile regression fits the *shape* of the training
    residual, but in Study 04 that shape does not survive out of sample: the
    nominal 90 % interval covers 68.7 % of the calibration rows and only
    5.8 % of the held-out ones, because the residual has a tight core and fat
    tails (MAD 3.3 against a standard deviation of 11.8) that quantile
    regression smooths over rather than reproduces. This function sidesteps
    the model's own quantiles entirely: it reads the ``alpha / 2`` and
    ``1 - alpha / 2`` empirical quantiles of the residual ``y - yhat`` on a
    chosen calibration slice, and adds those two fixed offsets to every row's
    ``yhat``. The resulting interval's calibration-period coverage is, by
    construction, exactly what those residuals show it to be — nothing is
    asked to extrapolate a shape it was never fit to reproduce.

    The two offsets are computed independently and are never centred or
    symmetrised around zero: a residual distribution with a heavy lower tail
    must produce a wider lower side than upper side, and forcing symmetry
    would misstate the coverage on whichever side is actually heavier.

    Parameters
    ----------
    predictions : pd.DataFrame
        Long prediction table with a ``ds`` timestamp column and the columns
        named by ``y_col`` and ``yhat_col``.
    alpha : float, optional
        Nominal miscoverage rate; the returned interval targets
        ``1 - alpha`` central coverage on the calibration rows. Default
        ``0.10``.
    calibration_end : timestamp-like or None, optional
        Rows with ``ds`` at or before this timestamp form the calibration
        slice whose residuals set the offsets. ``None`` calibrates on every
        row in ``predictions``, including the ones the interval is then
        applied to. That is a diagnostic only — useful for asking how wide
        an interval would have to be to describe its own data — and must
        never be reported as a result, because calibrating on the rows being
        scored is exactly the self-reference this study exists to avoid.
        Default ``None``.
    y_col : str, optional
        Observed-value column. Default ``'y'``.
    yhat_col : str, optional
        Point-prediction column. Default ``'yhat'``.

    Returns
    -------
    pd.DataFrame
        A copy of ``predictions`` with ``q05`` and ``q95`` overwritten by the
        conformal lower and upper bounds — the names are kept so that
        :func:`score_predictions` scores the interval unchanged — and a new
        column ``interval`` holding the constant string ``'conformal'``. The
        input ``predictions`` is never mutated.

    Raises
    ------
    ValueError
        If the calibration slice holds no row with both ``y_col`` and
        ``yhat_col`` finite. Returning a zero-width interval in that case
        would silently produce a monitoring band with no meaning, a failure
        mode this project has already been bitten by twice.
    """
    y = pd.to_numeric(predictions[y_col], errors='coerce')
    yhat = pd.to_numeric(predictions[yhat_col], errors='coerce')
    residual = y - yhat

    if calibration_end is None:
        calibration_mask = pd.Series(True, index=predictions.index)
    else:
        ds = pd.to_datetime(predictions['ds'])
        calibration_mask = ds <= pd.Timestamp(calibration_end)

    calibration_residual = residual[calibration_mask].dropna()
    if calibration_residual.empty:
        raise ValueError(
            'conformal_interval: the calibration slice holds no row with a '
            'finite y/yhat residual pair; check calibration_end and the '
            'y_col/yhat_col arguments.')

    lower_offset = float(calibration_residual.quantile(alpha / 2.0))
    upper_offset = float(calibration_residual.quantile(1.0 - alpha / 2.0))

    out = predictions.copy()
    out['q05'] = yhat + lower_offset
    out['q95'] = yhat + upper_offset
    out['interval'] = 'conformal'
    return out


def paired_mae_skill(parent, child, block_hours=24, repetitions=2000, seed=0,
                     horizon_hours=None):
    """
    Paired block-bootstrap skill of a child error series against its parent.

    ``parent`` and ``child`` are absolute-error series indexed by prediction
    timestamp. They are aligned on their shared complete timestamps before the
    observed MAE and every bootstrap draw are computed. Blocks are calendar
    chunks of ``block_hours`` from the first aligned timestamp. If ``horizon_hours``
    is provided, the effective block length is adjusted to at least
    ``max(block_hours, 2.0 * horizon_hours)`` to preserve residual autocorrelation.

    Parameters
    ----------
    parent, child : pd.Series
        Absolute errors for the baseline and the candidate route.
    block_hours : int or float, optional
        Width of bootstrap blocks in hours. Default ``24``.
    repetitions : int, optional
        Bootstrap repetitions. Default ``2000``.
    seed : int, optional
        Random seed. Default ``0``.
    horizon_hours : int or float, optional
        Forecast horizon in hours. If provided, effective block width is
        adjusted to ``max(block_hours, 2.0 * horizon_hours)``. Default ``None``.

    Returns
    -------
    dict
        Aligned sample size, observed MAEs, observed fractional skill
        ``1 - child_mae / parent_mae``, and the 5th/95th percentiles of the
        bootstrap skill distribution.
    """
    paired = pd.concat({'parent': parent, 'child': child}, axis=1)
    paired = paired.apply(pd.to_numeric, errors='coerce').dropna()
    if paired.empty:
        return {
            'n': 0,
            'parent_mae': np.nan,
            'child_mae': np.nan,
            'skill': np.nan,
            'skill_q05': np.nan,
            'skill_q95': np.nan,
        }

    parent_mae = float(paired['parent'].mean())
    child_mae = float(paired['child'].mean())
    skill = np.nan if parent_mae == 0 else 1.0 - child_mae / parent_mae

    effective_block_hours = float(block_hours)
    if horizon_hours is not None:
        effective_block_hours = max(effective_block_hours, 2.0 * float(horizon_hours))

    index = pd.DatetimeIndex(paired.index)
    elapsed = (index - index.min()) / pd.Timedelta(hours=effective_block_hours)
    blocks = np.floor(elapsed).astype(int)
    block_values = [paired.iloc[np.flatnonzero(blocks == block)]
                    for block in np.unique(blocks)]

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(repetitions)):
        chosen = rng.integers(0, len(block_values), size=len(block_values))
        sample = pd.concat([block_values[i] for i in chosen], axis=0)
        sample_parent = float(sample['parent'].mean())
        if sample_parent == 0:
            draws.append(np.nan)
        else:
            draws.append(1.0 - float(sample['child'].mean()) / sample_parent)

    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    q05, q95 = (np.nan, np.nan) if not len(finite) else np.quantile(
        finite, [0.05, 0.95])
    return {
        'n': int(len(paired)),
        'parent_mae': parent_mae,
        'child_mae': child_mae,
        'skill': float(skill),
        'skill_q05': float(q05),
        'skill_q95': float(q95),
    }


def _model_frame(frame, regressors):
    """Convert a study frame to NeuralProphet's ``ds``/``y`` table."""
    columns = list(regressors)
    if 'segment_id' in frame.columns:
        columns.append('segment_id')
    out = frame.loc[:, columns].copy()
    out.insert(0, 'y', pd.to_numeric(frame['y'], errors='coerce')
               if 'y' in frame.columns else np.nan)
    out.insert(0, 'ds', pd.DatetimeIndex(frame.index))
    if 'segment_id' in out.columns:
        out['ID'] = out.pop('segment_id').astype(str)
    return out


def _drop_singleton_segments(model_frame, context, phase):
    """
    Drop segments of fewer than two rows before a frame reaches NeuralProphet.

    NeuralProphet re-infers a sampling frequency per segment independently,
    both when fitting and when predicting, regardless of whether the caller
    already supplied an explicit ``freq`` — ``fit`` calls
    ``df_utils.infer_frequency`` on the training frame unconditionally, and
    ``predict`` does the same on the frame it is given. A segment holding
    exactly one timestamp has no interval for either call to measure a
    frequency from: ``df_utils.infer_frequency`` returns ``NaT``, and the
    library's own ``pd.to_timedelta(NaT)`` then raises ``ValueError: Invalid
    frequency: NaT``. This is not a corner case a caller can design around:
    any time-based slice of a segmented record — which is exactly what a
    walk-forward window is, on both its training and its evaluation side —
    can cut a long segment at the window boundary and leave a one-row stub
    behind, so a one-row segment is a matter of when, not if.

    A frame with no ``ID`` column, or whose segments all hold two or more
    rows, is returned unchanged — this function is a no-op on every frame
    that does not contain a singleton segment. When rows are dropped, a
    ``UserWarning`` names how many rows and how many segments were removed
    and which side of the model they were dropped from, so a caller is
    never silently short of predictions or training data, and a log
    carrying both warnings cannot confuse one drop for the other. On the
    prediction side those timestamps simply carry no row in the caller's
    returned long frame, which is correct: a one-row fragment is not a
    window any model can judge. On the training side those rows are not
    training data in any meaningful sense either — a single observation
    carries no lag, no seasonality and no changepoint for the model to
    estimate from — and are removed rather than merged into a neighbouring
    segment, because merging would silently bridge the gap segmentation
    exists to keep the model from spanning.

    Parameters
    ----------
    model_frame : pd.DataFrame
        NeuralProphet-format frame (``ds``, ``y``, regressor columns, and
        optionally ``ID``) about to be passed to ``model.fit`` or
        ``model.predict``.
    context : str
        Name of the calling function, included in the warning message so a
        caller can tell where the drop happened.
    phase : {'fitting', 'prediction'}
        Which call the frame is about to be passed to, named in the warning
        message so the training-side and prediction-side drops read as two
        distinct events rather than one.

    Returns
    -------
    pd.DataFrame
        ``model_frame``, or a copy with singleton-segment rows removed.
    """
    if 'ID' not in model_frame.columns:
        return model_frame

    sizes = model_frame.groupby('ID')['ID'].transform('size')
    singleton = sizes < 2
    if not singleton.any():
        return model_frame

    n_rows = int(singleton.sum())
    n_segments = int(model_frame.loc[singleton, 'ID'].nunique())
    consequence = ('those rows are removed from training, not merely left '
                   'unscored' if phase == 'fitting' else
                   'those timestamps receive no prediction')
    # The phrase 'single-row segment' is load-bearing: the tests match on it
    # to pick this warning out of the unrelated UserWarnings NeuralProphet
    # and its dependencies raise around every fit and predict call. Keep it
    # verbatim in any rewording.
    warnings.warn(
        f'{context}: dropped {n_rows} row(s) across {n_segments} '
        f'single-row segment(s) before {phase}, because a segment of one '
        f'row carries no frequency for NeuralProphet to infer; '
        f'{consequence}.',
        stacklevel=2)
    return model_frame.loc[~singleton]


def _analysis_freq(index):
    """Infer an hourly-style frequency string for NeuralProphet."""
    freq = pd.infer_freq(pd.DatetimeIndex(index))
    if freq is not None:
        return freq
    diffs = pd.DatetimeIndex(index).to_series().diff().dropna()
    if diffs.empty:
        return '1h'
    hours = diffs.median() / pd.Timedelta(hours=1)
    return f'{hours:g}h'


def _quantile_column(columns, horizon, quantile):
    """Return NeuralProphet's quantile column name when present."""
    target_pct = 100.0 * quantile
    for suffix in (f'{target_pct:.1f}%', f'{target_pct:g}%', f'{int(round(target_pct))}%'):
        target = f'yhat{horizon} {suffix}'
        if target in columns:
            return target
    pattern = re.compile(rf'^yhat{horizon}\s+([0-9.]+)%?$')
    for col in columns:
        m = pattern.match(col)
        if m:
            try:
                val = float(m.group(1))
                if np.isclose(val, target_pct, atol=0.05) or np.isclose(val, quantile, atol=0.0005):
                    return col
            except ValueError:
                continue
    return None


def _long_predictions(predictions, test_index, has_id, horizons, quantiles):
    """
    Convert NeuralProphet's wide prediction table to study-long format.

    NeuralProphet's ``predict`` always returns its ``ds`` column tz-naive,
    converted through UTC internally, even when the frame it was given
    carried a tz-aware ``DatetimeIndex``. Comparing that naive value against
    a tz-aware ``test_index`` by direct set membership never matches, since a
    naive and a tz-aware ``Timestamp`` hash differently even at the same
    instant, so every row would be silently dropped rather than raising an
    error. The lookup below keys on each side's UTC-naive representation —
    a no-op when ``test_index`` is itself tz-naive, which is how every
    caller before this one used it — and reports each matched row under the
    original, tz-aware timestamp from ``test_index``, so a tz-aware caller
    gets tz-aware ``ds`` values back exactly as it would expect.
    """
    test_index = pd.DatetimeIndex(test_index)
    if test_index.tz is not None:
        lookup = dict(zip(test_index.tz_convert('UTC').tz_localize(None),
                          test_index))
    else:
        lookup = {stamp: stamp for stamp in test_index}
    test_ds = set(lookup)
    rows = []
    quantiles = list(quantiles or [])
    lower = min(quantiles) if quantiles else None
    upper = max(quantiles) if quantiles else None
    for _, record in predictions.iterrows():
        raw_ds = pd.Timestamp(record['ds'])
        if raw_ds not in test_ds:
            continue
        ds = lookup[raw_ds]
        for horizon in horizons:
            yhat_col = f'yhat{horizon}'
            if yhat_col not in predictions.columns or pd.isna(record[yhat_col]):
                continue
            row = {
                'ds': ds,
                'horizon_h': int(horizon),
                'y': record['y'],
                'yhat': record[yhat_col],
            }
            if has_id:
                row['ID'] = record['ID']
            if lower is not None and upper is not None:
                q05_col = _quantile_column(predictions.columns, horizon, lower)
                q95_col = _quantile_column(predictions.columns, horizon, upper)
                if q05_col is not None:
                    row['q05'] = record[q05_col]
                if q95_col is not None:
                    row['q95'] = record[q95_col]
            rows.append(row)

    base = ['ds'] + (['ID'] if has_id else []) + ['horizon_h', 'y', 'yhat']
    interval = ['q05', 'q95'] if quantiles else []
    return pd.DataFrame(rows, columns=base + interval)


def neuralprophet_backtest(train, test, regressors=(), task='forecast',
                           n_lags=24, n_forecasts=24, regressor_lags=12,
                           horizons=None, epochs=30, yearly=False,
                           quantiles=(0.05, 0.95), seed=0, growth='off',
                           changepoints=None, n_changepoints=10, freq=None,
                           decompose=False, trend_reg=0.0,
                           changepoints_range=None, learning_rate=0.01,
                           yearly_order=None, daily_order=None,
                           conditional_seasonality=None, lagged_regressors=(),
                           lagged_n_lags=None, lagged_regularization=None,
                           validation=None):
    """
    Fit one NeuralProphet model and return long out-of-sample predictions.

    The wrapper is deliberately thin: it translates the study's DatetimeIndex
    frames to NeuralProphet's ``ds``/``y`` format, maps ``segment_id`` to
    ``ID`` when present, registers regressors according to task, and reshapes
    the wide ``yhatN`` output to one row per target timestamp and horizon. It
    does not interpolate missing rows.

    Parameters
    ----------
    train, test : pd.DataFrame
        Datetime-indexed frames with ``y``, optional regressors, and optional
        ``segment_id``.
    regressors : sequence of str, optional
        Regressor columns to expose to NeuralProphet. Default empty.
    task : {'forecast', 'nowcast'}, optional
        Forecast uses autoregression and past lagged regressors. Nowcast uses
        same-time future regressors and forces one-step, zero-lag prediction.
        Default ``'forecast'``.
    n_lags, n_forecasts : int, optional
        NeuralProphet autoregressive lag and forecast lengths for forecast
        tasks. Defaults ``24`` and ``24``.
    regressor_lags : int, optional
        Exact lag length for past-only forecast regressors. Default ``12``.
    horizons : sequence of int or None, optional
        Horizons to keep in the long output. Default all model horizons.
    epochs : int, optional
        Training epochs. Default ``30``.
    yearly : bool, optional
        Whether to enable yearly seasonality. Default ``False``.
    quantiles : sequence of float, optional
        Prediction quantiles. Default ``(0.05, 0.95)``.
    seed : int, optional
        Random seed. Default ``0``.
    growth : {'off', 'linear'}, optional
        Trend specification. ``'off'`` fits a constant offset and is the
        default, which is what a change-valued target needs; ``'linear'``
        fits a piecewise-linear trend and is what a level-valued target needs.
    changepoints : pd.DatetimeIndex or None, optional
        Explicit changepoint locations, normally from
        ``covered_changepoints``. ``None`` lets NeuralProphet space
        ``n_changepoints`` of them along the training range, which on a gapped
        record can place one inside an outage. Default ``None``.
    n_changepoints : int, optional
        Number of changepoints when ``changepoints`` is ``None``. Ignored
        otherwise. Default ``10``.
    freq : str or None, optional
        Frequency handed to NeuralProphet. ``None`` infers it from the training
        index. Default ``None``.
    decompose : bool, optional
        Whether ``model.predict`` returns component columns beside the
        prediction. Default ``False``, which is what a scoring run needs.
    trend_reg : float, optional
        L1 regularisation strength on the trend's changepoint deltas, passed
        to the constructor unchanged. ``0.0`` reproduces NeuralProphet's own
        default of an unregularised, freely bending trend. NeuralProphet
        0.8.0 rescales a positive value by ``0.001`` internally when
        changepoints exist, and zeroes it when there are none (``growth``
        ``'off'`` or ``n_changepoints=0``), so the number this argument
        takes is in this wrapper's user-facing scale, not the scale stored
        on the fitted model's ``config_trend.trend_reg``. Default ``0.0``.
    changepoints_range : float or None, optional
        Fraction of the training range, from its start, in which
        changepoints may be placed. ``None`` omits the argument from the
        constructor entirely, so NeuralProphet's own default range applies
        exactly as it did before this parameter existed. Default ``None``.
    learning_rate : float, optional
        Optimiser learning rate, passed to the constructor unchanged.
        Default ``0.01``, NeuralProphet's own default and this wrapper's
        previous hard-coded value.
    yearly_order : int or None, optional
        Fourier order of yearly seasonality. ``None`` falls back to the
        boolean ``yearly`` flag, exactly as before this parameter existed;
        an integer overrides ``yearly`` and requests that many harmonics
        instead of NeuralProphet's own automatic order. Default ``None``.
    daily_order : int or None, optional
        Fourier order of daily seasonality. ``None`` requests NeuralProphet's
        automatic daily seasonality (``daily_seasonality=True``), the
        wrapper's previous behaviour; an integer requests that many
        harmonics instead. When ``conditional_seasonality`` is also given,
        this order is reused for each conditional daily term and the
        unconditional daily seasonality is turned off, since the two would
        otherwise explain the same variation twice. Default ``None``.
    conditional_seasonality : dict or None, optional
        Maps a seasonality name to the frame column that gates it, for
        example ``{'daily_summer': 'summer_w', 'daily_winter': 'winter_w'}``.
        Each entry becomes one call to ``model.add_seasonality(name,
        period=1, fourier_order=daily_order or 6, condition_name=column)``,
        and every named condition column is carried through into the model
        frames alongside the regressors. Giving this argument turns the
        model's unconditional daily seasonality off, on the understanding
        that the conditional terms replace it. ``None`` (the default)
        registers no conditional seasonality and leaves daily seasonality
        exactly as ``daily_order`` and ``yearly`` already describe it.
    lagged_regressors : sequence of str, optional
        Regressor columns registered as NeuralProphet lagged (autoregressive)
        regressors regardless of ``task`` — unlike ``regressors``, which
        follows the forecast/nowcast switch, every name here always reaches
        ``model.add_lagged_regressor``, so a nowcast model can still see a
        driver's recent history. Each such column is also carried through
        into the model frames. Default empty, which registers none and
        leaves nowcast models exactly as free of lagged regressors as
        before this parameter existed.
    lagged_n_lags : int or None, optional
        Number of past steps used for every column in ``lagged_regressors``.
        ``None`` falls back to ``regressor_lags``. Ignored when
        ``lagged_regressors`` is empty. Default ``None``.
    lagged_regularization : float or None, optional
        Regularisation strength passed to ``add_lagged_regressor`` for every
        column in ``lagged_regressors``. ``None`` requests NeuralProphet's
        own default (no regularisation). Ignored when ``lagged_regressors``
        is empty. Default ``None``.
    validation : pd.DataFrame or None, optional
        A held-out frame, shaped like ``train``, scored after every training
        epoch. Giving it switches the constructor's ``collect_metrics`` on
        and calls ``model.fit`` with ``validation_df=`` and
        ``minimal=False``, and the metrics frame ``fit`` returns is stored
        on the fitted model as ``model.fit_metrics_`` for a caller to read
        back (see :func:`~shmlib.figures.plot_fit_metrics`). ``None`` (the
        default) fits exactly as before this parameter existed: no
        validation frame, ``collect_metrics=False``, ``minimal=True``, and
        no ``fit_metrics_`` attribute is set.

    Returns
    -------
    model, pd.DataFrame
        Fitted NeuralProphet model and long prediction table. A segment
        (identified by ``segment_id``) shorter than two rows is dropped
        before it can reach NeuralProphet, on both sides of the fit: from
        ``train`` before ``model.fit``, and from ``test`` before
        ``model.predict``. NeuralProphet infers a sampling frequency per
        segment on both calls, regardless of the explicit ``freq`` this
        wrapper already supplies, and a single timestamp carries none to
        infer; see :func:`_drop_singleton_segments`. This is a no-op
        whenever ``train`` and ``test`` carry no ``segment_id`` or every
        segment already holds two or more rows, and a separate
        ``UserWarning`` — naming which side, and how many rows and segments
        — is raised for each side that is not. The fitted model also carries
        ``extra_columns_``, the tuple of lagged-regressor and condition
        columns (beyond ``regressors``) that were passed through to fit it;
        :func:`decompose_components` and :func:`trend_parameters` read it
        back so a caller does not have to repeat those column names at
        prediction time.
    """
    if task not in {'forecast', 'nowcast'}:
        raise ValueError("task must be 'forecast' or 'nowcast'")

    from neuralprophet import NeuralProphet, set_random_seed

    set_random_seed(seed)
    np.random.seed(seed)

    regressors = tuple(regressors or ())
    lagged_regressors = tuple(lagged_regressors or ())
    conditions = dict(conditional_seasonality or {})
    if task == 'nowcast':
        n_lags = 0
        n_forecasts = 1

    daily_setting = True if daily_order is None else int(daily_order)
    if conditions:
        daily_setting = False
    constructor = dict(
        growth=growth,
        changepoints=(list(pd.DatetimeIndex(changepoints))
                      if changepoints is not None else None),
        n_changepoints=int(n_changepoints),
        trend_reg=float(trend_reg),
        n_lags=int(n_lags),
        n_forecasts=int(n_forecasts),
        daily_seasonality=daily_setting,
        weekly_seasonality=False,
        yearly_seasonality=(yearly if yearly_order is None else int(yearly_order)),
        normalize='standardize',
        global_normalization=True,
        global_time_normalization=True,
        unknown_data_normalization=True,
        impute_missing=False,
        drop_missing=False,
        loss_func='SmoothL1Loss',
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        quantiles=list(quantiles or ()),
        collect_metrics=validation is not None,
    )
    if changepoints_range is not None:
        constructor['changepoints_range'] = float(changepoints_range)
    model = NeuralProphet(**constructor)
    for name, column in conditions.items():
        model.add_seasonality(name=name, period=1,
                              fourier_order=int(daily_order or 6),
                              condition_name=column)
    for regressor in regressors:
        if task == 'nowcast':
            model.add_future_regressor(regressor)
        else:
            model.add_lagged_regressor(regressor, n_lags=int(regressor_lags))
    for regressor in lagged_regressors:
        model.add_lagged_regressor(
            regressor, n_lags=int(lagged_n_lags or regressor_lags),
            regularization=lagged_regularization)

    passthrough = tuple(regressors) + lagged_regressors + tuple(conditions.values())
    model.extra_columns_ = tuple(c for c in passthrough if c not in regressors)
    train_df = _model_frame(train, passthrough)
    test_df = _model_frame(test, passthrough)
    fit_freq = freq if freq is not None else _analysis_freq(train.index)
    train_df = _drop_singleton_segments(train_df, 'neuralprophet_backtest',
                                        'fitting')
    if validation is not None:
        validation_df = _drop_singleton_segments(
            _model_frame(validation, passthrough), 'neuralprophet_backtest',
            'validation')
        model.fit_metrics_ = model.fit(train_df, freq=fit_freq,
                                       validation_df=validation_df,
                                       progress='none', minimal=False)
    else:
        model.fit(train_df, freq=fit_freq, progress='none', minimal=True)

    segmented_test = 'segment_id' in test.columns
    predict_df = test_df if task == 'nowcast' or segmented_test else pd.concat(
        [train_df.tail(max(int(n_lags), int(regressor_lags),
                           int(lagged_n_lags or 0))), test_df],
        ignore_index=True)
    predict_df = _drop_singleton_segments(predict_df, 'neuralprophet_backtest',
                                          'prediction')
    wide = model.predict(predict_df, decompose=bool(decompose))
    keep_horizons = list(horizons or range(1, int(n_forecasts) + 1))
    long = _long_predictions(
        wide, test.index, 'segment_id' in test.columns, keep_horizons,
        quantiles)
    return model, long


def rolling_nowcast(frame, regressors=(), refit_every='30d', min_train='180d',
                    freq=None, train_window=None, changepoints_per_window=False,
                    **model_kwargs):
    """
    Walk-forward nowcast evaluation: keep every prediction as fresh as a
    deployed model would be, by refitting on a schedule rather than once.

    A single frozen fit goes stale as the record it was fitted to recedes
    into the past. In Study 04 a model fitted to 2025-09 and scored on
    everything after sits +28.75 mdeg above its own expectation out of
    sample, and 82 % of its mean square error is that constant offset rather
    than scatter around a moving target — the residual's spread barely
    changes (MAD 2.93 out of sample against 3.28 in). This function is the
    walk-forward discipline a deployed system would use instead: starting at
    ``frame.index.min() + min_train`` and stepping by ``refit_every``, each
    window fits on every row strictly before the window's origin and scores
    only the rows in ``[origin, origin + refit_every)``. No prediction this
    function returns is ever more than one ``refit_every`` step past the fit
    that produced it.

    Every window is fit through :func:`neuralprophet_backtest` at
    ``task='nowcast'`` — this function answers the nowcast question the
    study asks, not the multi-step forecast question — passing
    ``regressors``, ``freq`` and every entry of ``model_kwargs`` straight
    through, so a caller controls ``n_lags``, ``epochs``, ``yearly``,
    ``quantiles``, ``seed``, ``growth`` and ``n_changepoints`` exactly as for
    a single fit. A window is skipped outright when its training rows or its
    evaluation rows are empty, and also when the training frame carries a
    ``segment_id`` column with fewer than two distinct segments, since
    :func:`neuralprophet_backtest` needs at least that much structure to
    fit.

    This fits one NeuralProphet model per window, so cost scales with record
    length divided by ``refit_every``: a three-year record at the default
    ``refit_every='30d'`` is roughly three dozen fits, not one. Keep
    ``epochs`` and the record short when calling this outside of a full
    study run.

    Parameters
    ----------
    frame : pd.DataFrame
        Datetime-indexed modelling frame with ``y``, the regressor columns,
        and optionally ``segment_id``.
    regressors : sequence of str, optional
        Regressor columns exposed to the model. Default empty.
    refit_every : str, optional
        Pandas offset alias giving both the refit cadence and the width of
        each window's scored slice (e.g. ``'30d'``). Default ``'30d'``.
    min_train : str, optional
        Pandas offset alias giving the minimum history required before the
        first fit (e.g. ``'180d'``). Default ``'180d'``.
    freq : str or None, optional
        Frequency handed to :func:`neuralprophet_backtest`. ``None`` infers
        it per window from that window's own training index. Default
        ``None``.
    train_window : str or None, optional
        Pandas offset alias bounding each window's training data to
        ``[origin - train_window, origin)`` rather than all history back to
        ``frame.index.min()``. ``None`` (the default) reproduces the
        original expanding-window behaviour, training on everything strictly
        before ``origin``. A bounded window keeps every refit at a
        comparable size on a multi-year record, where an expanding window
        would otherwise make the last fit an order of magnitude larger than
        the first.
    changepoints_per_window : bool, optional
        When ``True``, each window computes its own trend changepoints
        through :func:`covered_changepoints` on that window's training
        index and ``model_kwargs['n_changepoints']`` (falling back to
        :func:`neuralprophet_backtest`'s own default of ``10`` when
        ``n_changepoints`` is not given), and passes them as
        ``changepoints`` instead of leaving NeuralProphet to space
        ``n_changepoints`` of them uniformly along the window — which, on a
        gapped record, can place one inside an outage. Default ``False``,
        which reproduces the original behaviour of letting
        :func:`neuralprophet_backtest` place changepoints itself.
    **model_kwargs
        Forwarded unchanged to :func:`neuralprophet_backtest` for every
        window (e.g. ``n_lags``, ``epochs``, ``yearly``, ``quantiles``,
        ``seed``, ``growth``, ``n_changepoints``).

    Returns
    -------
    pd.DataFrame
        Long predictions in the same shape :func:`neuralprophet_backtest`
        returns, plus an ``origin`` column giving the fit origin each row was
        predicted from and a ``staleness_d`` column giving, for each row,
        how many days its ``ds`` sits past that origin — ``0`` for a
        same-day nowcast and up to ``refit_every`` at the far edge of a
        window. Chronological by ``ds``, with no duplicated timestamps.

    Raises
    ------
    ValueError
        If ``'task'`` appears in ``model_kwargs``. ``rolling_nowcast``
        evaluates nowcasts only; a caller passing ``task='forecast'`` (or
        even a redundant ``task='nowcast'``) would otherwise collide with
        the ``task='nowcast'`` this function already passes to
        :func:`neuralprophet_backtest`, raising an opaque
        ``TypeError: got multiple values for keyword argument 'task'``. A
        walk-forward evaluation of the forecast task is a different
        function, not yet written.
    """
    if 'task' in model_kwargs:
        raise ValueError(
            "rolling_nowcast() evaluates nowcasts only and always calls "
            "neuralprophet_backtest with task='nowcast'; it does not accept "
            "'task' in model_kwargs. A walk-forward evaluation of the "
            "forecast task is a different function.")

    ordered = frame.sort_index()
    index = pd.DatetimeIndex(ordered.index)
    empty_columns = ['ds', 'horizon_h', 'y', 'yhat', 'origin']
    if len(index) == 0:
        return pd.DataFrame(columns=empty_columns)

    step = pd.Timedelta(refit_every)
    origin = index.min() + pd.Timedelta(min_train)
    last = index.max()

    windows = []
    while origin <= last:
        lower = (origin - pd.Timedelta(train_window)) if train_window else index.min()
        train = ordered.loc[(index >= lower) & (index < origin)]
        test = ordered.loc[(index >= origin) & (index < origin + step)]
        eligible = not train.empty and not test.empty
        if eligible and 'segment_id' in train.columns:
            eligible = train['segment_id'].nunique(dropna=True) >= 2
        if eligible:
            window_kwargs = dict(model_kwargs)
            if changepoints_per_window:
                window_kwargs['changepoints'] = covered_changepoints(
                    train.index, int(window_kwargs.get('n_changepoints', 10)))
            _, predictions = neuralprophet_backtest(
                train, test, regressors=regressors, task='nowcast',
                freq=freq, **window_kwargs)
            if not predictions.empty:
                predictions = predictions.copy()
                predictions['origin'] = origin
                predictions['staleness_d'] = (
                    pd.DatetimeIndex(predictions['ds']) - origin) / pd.Timedelta(days=1)
                windows.append(predictions)
        origin += step

    if not windows:
        return pd.DataFrame(columns=empty_columns)

    out = pd.concat(windows, ignore_index=True)
    return out.sort_values('ds', kind='stable').reset_index(drop=True)


def rolling_conformal(predictions, alpha=0.10, window='180d', method='cqr'):
    """
    Conformal bounds calibrated on the walk-forward output's own recent past.

    For each refit origin the calibration set is every row of ``predictions``
    with ``ds`` in ``[origin - window, origin)``. Those rows were predicted by
    earlier fits and are out of sample by construction, so the split
    conformal guarantee holds while every fit stays as fresh as the refit
    schedule allows (spec D8). The ``'cqr'`` method is conformalised quantile
    regression on the model's own quantile columns; ``'naive'`` is the
    symmetric absolute-residual interval about ``yhat``.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of :func:`rolling_nowcast` with ``ds``, ``y``, ``yhat``,
        ``origin`` and, for ``'cqr'``, ``q05`` and ``q95``.
    alpha : float, optional
        Miscoverage. Default ``0.10``.
    window : str, optional
        Length of the calibration set behind each origin. Default ``'180d'``.
    method : {'cqr', 'naive'}, optional

    Returns
    -------
    pd.DataFrame
        Copy with ``q05_model``/``q95_model`` (when present), conformal
        ``q05``/``q95``, ``qhat`` and ``n_cal``.
    """
    out = predictions.copy()
    ds = pd.DatetimeIndex(out['ds'])
    if method == 'cqr':
        for column in ('q05', 'q95'):
            out[f'{column}_model'] = out[column]
        scores = np.maximum(out['q05_model'] - out['y'], out['y'] - out['q95_model'])
    elif method == 'naive':
        scores = (out['y'] - out['yhat']).abs()
    else:
        raise ValueError("method must be 'cqr' or 'naive'")
    scores = pd.Series(scores.to_numpy(dtype=float), index=out.index)
    span = pd.Timedelta(window)
    qhat = pd.Series(np.nan, index=out.index)
    n_cal = pd.Series(0, index=out.index, dtype=int)
    for origin, rows in out.groupby('origin').groups.items():
        origin = pd.Timestamp(origin)
        calibration = scores[(ds >= origin - span) & (ds < origin)].dropna()
        n = int(calibration.size)
        n_cal.loc[rows] = n
        if n == 0:
            continue
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhat.loc[rows] = float(np.quantile(calibration, level))
    out['qhat'] = qhat
    out['n_cal'] = n_cal
    if method == 'cqr':
        out['q05'] = out['q05_model'] - out['qhat']
        out['q95'] = out['q95_model'] + out['qhat']
    else:
        out['q05'] = out['yhat'] - out['qhat']
        out['q95'] = out['yhat'] + out['qhat']
    return out


def neuralprophet_predict(model, frame, regressors=(), horizons=(1,),
                          quantiles=(0.05, 0.95), decompose=False):
    """
    Predict new same-time rows with an already fitted NeuralProphet model.

    Parameters
    ----------
    model : object
        Fitted model exposing ``predict(df, decompose=False)``.
    frame : pd.DataFrame
        Datetime-indexed rows with optional ``y`` and optional ``segment_id``.
    regressors : sequence of str, optional
        Regressor columns to pass through. Default empty.
    horizons : sequence of int, optional
        Horizons to keep from the model output. Default ``(1,)``.
    quantiles : sequence of float, optional
        Prediction quantiles to reshape when present. Default ``(0.05, 0.95)``.
    decompose : bool, optional
        Whether component columns are requested from the model. Default
        ``False``.

    Returns
    -------
    pd.DataFrame
        Long prediction table with ``ds``, optional ``ID``, horizon, observed
        value, prediction, and optional interval columns. A segment
        (identified by ``segment_id``) shorter than two rows in ``frame`` is
        dropped before it can reach NeuralProphet's ``model.predict``.
        NeuralProphet infers a sampling frequency per segment in ``fit`` as
        well as in ``predict``, regardless of any explicit ``freq`` a caller
        supplied elsewhere, and a single timestamp carries none to infer;
        see :func:`_drop_singleton_segments`. This is a no-op whenever
        ``frame`` carries no ``segment_id`` or every segment already holds
        two or more rows, and a ``UserWarning`` naming the row and segment
        counts is raised whenever it is not.
    """
    model_frame = _model_frame(frame, regressors)
    original_y = pd.to_numeric(
        frame['y'], errors='coerce') if 'y' in frame.columns else pd.Series(
            np.nan, index=frame.index, dtype=float)
    if getattr(model, 'n_lags', None) == 0:
        # NeuralProphet 0.9.0 crashes while restoring trailing missing y even
        # though a zero-lag model does not consume the target at prediction.
        model_frame['y'] = model_frame['y'].fillna(0.0)
    model_frame = _drop_singleton_segments(model_frame, 'neuralprophet_predict',
                                           'prediction')
    wide = model.predict(model_frame, decompose=bool(decompose))
    long = _long_predictions(
        wide, frame.index, 'segment_id' in frame.columns, list(horizons),
        quantiles)
    if 'segment_id' in frame.columns:
        observed = {
            (pd.Timestamp(ds), str(segment)): value
            for ds, segment, value in zip(
                frame.index, frame['segment_id'], original_y)
        }
        long['y'] = [observed.get((pd.Timestamp(ds), str(segment)), np.nan)
                     for ds, segment in zip(long['ds'], long['ID'])]
    else:
        observed = {pd.Timestamp(ds): value
                    for ds, value in original_y.items()}
        long['y'] = [observed.get(pd.Timestamp(ds), np.nan)
                     for ds in long['ds']]
    return long


def backtest_specifications(segmented, folds, specifications, epochs=30,
                            quantiles=(0.05, 0.95), seed=0, runner=None):
    """
    Run model specifications over expanding whole-segment folds.

    Parameters
    ----------
    segmented : pd.DataFrame
        Modelling frame with ``segment_id``.
    folds : sequence of dict
        Fold dictionaries from :func:`expanding_segment_folds`.
    specifications : sequence of dict
        Each specification has ``name``, ``task`` and ``regressors`` plus
        optional keyword arguments for the model runner.
    epochs : int, optional
        Default epochs passed to the runner. Default ``30``.
    quantiles : sequence of float, optional
        Default quantiles passed to the runner. Default ``(0.05, 0.95)``.
    seed : int, optional
        Seed passed to the runner. Default ``0``.
    runner : callable or None, optional
        Backtest runner. Default :func:`neuralprophet_backtest`.

    Returns
    -------
    pd.DataFrame
        Concatenated long predictions with ``fold`` and ``model`` metadata.
    """
    if runner is None:
        runner = neuralprophet_backtest
    rows = []
    for fold in folds:
        train = segmented[segmented['segment_id'].isin(
            fold['train_segments'])]
        test = segmented[segmented['segment_id'].isin(fold['test_segments'])]
        for specification in specifications:
            options = dict(specification)
            name = options.pop('name')
            options.setdefault('task', 'forecast')
            options.setdefault('regressors', ())
            options.setdefault('epochs', epochs)
            options.setdefault('quantiles', quantiles)
            options.setdefault('seed', seed)
            _, predictions = runner(train, test, **options)
            if predictions.empty:
                continue
            predictions = predictions.copy()
            predictions['fold'] = fold['fold']
            predictions['model'] = name
            rows.append(predictions)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _segment_lag(values, horizon):
    """Lag within each segment, never across segment IDs."""
    return values.groupby(values['segment_id'])['y'].shift(int(horizon))


def baseline_predictions(segmented, folds, horizons, task='forecast'):
    """
    Long zero, persistence and seasonal-naive predictions for fold test rows.

    Parameters
    ----------
    segmented : pd.DataFrame
        Modelling frame with ``y`` and ``segment_id``.
    folds : sequence of dict
        Fold dictionaries with ``fold`` and ``test_segments``.
    horizons : sequence of int
        Forecast horizons to score.
    task : {'forecast', 'nowcast'}, optional
        Nowcast emits only the zero baseline. Forecast also emits persistence
        and seasonal naive baselines. Default ``'forecast'``.

    Returns
    -------
    pd.DataFrame
        Columns ``fold``, ``model``, ``horizon_h``, ``ds``, ``ID``, ``y`` and
        ``yhat``.
    """
    if task not in {'forecast', 'nowcast'}:
        raise ValueError("task must be 'forecast' or 'nowcast'")
    work = segmented.copy()
    work = work.sort_index()
    rows = []
    for fold in folds:
        test = work[work['segment_id'].isin(fold['test_segments'])]
        for horizon in horizons:
            horizon = int(horizon)
            candidates = [('zero', pd.Series(0.0, index=work.index))]
            if task == 'forecast':
                lag = max(horizon, int(np.ceil(horizon / 24.0) * 24))
                candidates.extend([
                    ('persistence', _segment_lag(work, horizon)),
                    ('seasonal_naive', _segment_lag(work, lag)),
                ])
            for model_name, yhat in candidates:
                for ds, record in test.iterrows():
                    rows.append({
                        'fold': fold['fold'],
                        'model': model_name,
                        'horizon_h': horizon,
                        'ds': ds,
                        'ID': record['segment_id'],
                        'y': record['y'],
                        'yhat': yhat.loc[ds],
                    })
    return pd.DataFrame(
        rows, columns=['fold', 'model', 'horizon_h', 'ds', 'ID', 'y', 'yhat'])


def execution_folds(folds, refit_each_fold=False):
    """
    Choose expanding refits or one frozen-origin evaluation fold.

    Parameters
    ----------
    folds : sequence of dict
        Expanding fold dictionaries.
    refit_each_fold : bool, optional
        Return folds unchanged when true. Otherwise freeze the origin at the
        first fold's training segments and concatenate every test segment in
        chronological fold order. Default ``False``.

    Returns
    -------
    list of dict
        Folds to execute.
    """
    folds = list(folds)
    if refit_each_fold or not folds:
        return folds
    test_segments = []
    for fold in folds:
        test_segments.extend(fold['test_segments'])
    return [{
        'fold': folds[0]['fold'],
        'train_segments': list(folds[0]['train_segments']),
        'test_segments': test_segments,
    }]


def gap_closure_summary(inclination, estimates, era=None, freq='1h'):
    """
    Summarise whether predicted hourly changes close observed inclination gaps.

    Contiguous missing runs in ``inclination`` are reported without
    interpolation. A closure is available only when the gap has observed anchors
    immediately before and after the run, the two anchors are in the same era
    when era labels are supplied, and estimates exist from the first missing
    timestamp through the recovery timestamp.

    Parameters
    ----------
    inclination : pd.Series
        Observed inclination level.
    estimates : pd.Series
        Predicted hourly changes on the same timestamp grid.
    era : pd.Series or None, optional
        Era labels. Default ``None``.
    freq : str, optional
        Expected spacing. Default ``'1h'``.

    Returns
    -------
    pd.DataFrame
        One row per gap with anchors, observed and predicted recovery, closure
        error, and explicit status.
    """
    values = pd.to_numeric(inclination, errors='coerce')
    estimates = pd.to_numeric(estimates, errors='coerce').reindex(values.index)
    era_values = _as_series(era, values.index, name='era')
    step = pd.Timedelta(freq)
    index = pd.DatetimeIndex(values.index)
    missing = values.isna()
    rows = []
    gap_no = 1
    pos = 0
    while pos < len(values):
        if not missing.iloc[pos]:
            pos += 1
            continue
        start_pos = pos
        while (pos + 1 < len(values) and missing.iloc[pos + 1]
               and index[pos + 1] - index[pos] == step):
            pos += 1
        end_pos = pos
        prior_pos = start_pos - 1
        recovery_pos = end_pos + 1
        has_prior = prior_pos >= 0 and pd.notna(values.iloc[prior_pos])
        has_recovery = (recovery_pos < len(values)
                        and pd.notna(values.iloc[recovery_pos])
                        and index[recovery_pos] - index[end_pos] == step)

        status = 'available'
        prior_anchor = values.iloc[prior_pos] if has_prior else np.nan
        recovery = index[recovery_pos] if has_recovery else pd.NaT
        observed = np.nan
        predicted = np.nan
        error = np.nan

        if not has_prior or not has_recovery:
            status = 'unbracketed'
        elif (era_values is not None
              and era_values.iloc[prior_pos] != era_values.iloc[recovery_pos]):
            status = 'cross_era'
        else:
            closure = estimates.iloc[start_pos:recovery_pos + 1]
            if closure.isna().any():
                status = 'incomplete_estimates'
            else:
                observed = float(values.iloc[recovery_pos] - prior_anchor)
                predicted = float(closure.sum())
                error = predicted - observed

        rows.append({
            'gap_id': f'G{gap_no:03d}',
            'start': index[start_pos],
            'end': index[end_pos],
            'recovery': recovery,
            'n_missing': int(end_pos - start_pos + 1),
            'status': status,
            'prior_anchor': float(prior_anchor) if pd.notna(prior_anchor)
            else np.nan,
            'observed_recovery': observed,
            'predicted_recovery': predicted,
            'closure_error': error,
        })
        gap_no += 1
        pos += 1

    return pd.DataFrame(rows, columns=[
        'gap_id', 'start', 'end', 'recovery', 'n_missing', 'status',
        'prior_anchor', 'observed_recovery', 'predicted_recovery',
        'closure_error'])


def covered_changepoints(index, n_changepoints, observed_mask=None):
    """
    Trend changepoints placed on time the record actually covers.

    A changepoint placed inside an outage is constrained by no observation, and
    the trend is free to move arbitrarily across it. Placing changepoints at
    quantiles of the observed timestamps rather than uniformly along the axis
    keeps every one of them anchored to data.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full analysis grid, covered and uncovered alike.
    n_changepoints : int
        Number of changepoints requested. Silently clipped when fewer covered
        samples exist.
    observed_mask : pd.Series or array-like or None, optional
        Boolean per timestamp, true where a value is present. ``None`` treats
        every timestamp as covered. Default ``None``.

    Returns
    -------
    pd.DatetimeIndex
        Increasing changepoint locations, of length at most ``n_changepoints``.
    """
    index = pd.DatetimeIndex(index)
    if observed_mask is None:
        covered = index
    else:
        mask = _as_series(observed_mask, index).fillna(False).astype(bool)
        covered = index[mask.to_numpy()]
    if len(covered) == 0:
        return pd.DatetimeIndex([])

    count = int(min(int(n_changepoints), len(covered)))
    if count <= 0:
        return pd.DatetimeIndex([])
    quantiles = np.linspace(0.0, 1.0, count + 2)[1:-1]
    positions = np.unique((quantiles * (len(covered) - 1)).round().astype(int))
    return pd.DatetimeIndex(covered[positions])


def _frame_columns(model, frame, regressors):
    """
    Every column a model frame needs: the requested regressors, plus
    whatever the model itself was fit with beyond them.

    ``neuralprophet_backtest`` records the lagged-regressor and condition
    columns it passed through at fit time on the fitted model, as
    ``extra_columns_``, because NeuralProphet requires those same columns in
    every frame handed to ``predict`` afterwards — not only ``fit`` — or it
    raises a ``KeyError`` on the missing condition column. This adds back
    whichever of them are present in ``frame`` and not already named in
    ``regressors``, so a caller of :func:`decompose_components` or
    :func:`trend_parameters` does not have to repeat them by hand. A model
    without ``extra_columns_`` (for instance one built outside
    ``neuralprophet_backtest``) is treated as having none.

    Parameters
    ----------
    model : object
        A NeuralProphet model, optionally carrying ``extra_columns_``.
    frame : pd.DataFrame
        The frame the columns will be read from.
    regressors : sequence of str
        Regressor columns already requested by the caller.

    Returns
    -------
    tuple of str
        ``regressors`` followed by any extra column present in ``frame``.
    """
    regressors = tuple(regressors)
    extra = tuple(c for c in getattr(model, 'extra_columns_', ())
                  if c in frame.columns and c not in regressors)
    return regressors + extra


def decompose_components(model, frame, regressors=(), freq=None):
    """
    The additive parts NeuralProphet fitted, aligned to the study's index.

    NeuralProphet returns its decomposition as extra columns beside the
    prediction. This reshapes them into one timestamp-indexed table, adds the
    observed value and the residual, and leaves the component names as the model
    produced them, so that a reader can trace any column back to the term that
    made it. Columns the model needed at fit time beyond ``regressors`` —
    lagged-regressor and condition columns registered through
    ``neuralprophet_backtest``'s ``lagged_regressors`` and
    ``conditional_seasonality`` — are supplied again automatically, through
    :func:`_frame_columns`, because NeuralProphet requires them again at
    predict time.

    NeuralProphet's ``predict`` always returns its ``ds`` column tz-naive,
    converted through UTC internally, exactly as documented on
    :func:`_long_predictions`. Left uncorrected, the returned table's index
    would be tz-naive even when ``frame`` is tz-aware, and the ``observed =
    ... .reindex(out.index)`` step below would then align a tz-aware
    ``Series`` against a tz-naive one: pandas raises no error there, it
    silently matches nothing, so ``y`` and therefore ``residual`` would come
    back entirely ``NaN`` with no warning. This restores ``ds`` to
    ``frame``'s own zone by the same UTC-naive lookup :func:`_long_predictions`
    uses, before that reindex ever runs, so the returned table lands on
    exactly ``frame``'s index whether or not it carries a timezone.

    Parameters
    ----------
    model : object
        Fitted NeuralProphet model, exposing ``predict(df, decompose=True)``.
    frame : pd.DataFrame
        Datetime-indexed rows with ``y``, the regressors, and optionally
        ``segment_id``.
    regressors : sequence of str, optional
        Regressor columns to pass through. Default empty.
    freq : str or None, optional
        Unused by the model at prediction time; accepted so callers may pass
        the study's grid for symmetry with ``neuralprophet_backtest``. Default
        ``None``.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, carrying every component column the model
        produced, plus ``y``, ``yhat1``, ``residual`` and, when the input was
        segmented, ``ID``.
    """
    model_frame = _model_frame(frame, _frame_columns(model, frame, regressors))
    if getattr(model, 'n_lags', None) == 0:
        model_frame['y'] = model_frame['y'].fillna(0.0)
    wide = model.predict(model_frame, decompose=True)

    reserved = {'ds', 'y'}
    components = [c for c in wide.columns
                  if c not in reserved and not c.startswith('yhat')
                  and '%' not in c and c != 'ID']

    out = wide.loc[:, ['ds'] + components].copy()
    out['yhat1'] = wide['yhat1'] if 'yhat1' in wide.columns else np.nan
    if 'ID' in wide.columns:
        out['ID'] = wide['ID']

    frame_index = pd.DatetimeIndex(frame.index)
    if frame_index.tz is not None:
        lookup = dict(zip(frame_index.tz_convert('UTC').tz_localize(None),
                          frame_index))
        out['ds'] = pd.DatetimeIndex(out['ds']).map(lookup)
    out = out.set_index('ds')
    out.index.name = frame.index.name

    observed = pd.to_numeric(frame['y'], errors='coerce') if 'y' in frame else None
    out['y'] = observed.reindex(out.index) if observed is not None else np.nan
    out['residual'] = out['y'] - out['yhat1']
    return out


_NON_COMPONENT_COLUMNS = ('y', 'yhat1', 'ID')


# NeuralProphet emits a family aggregate column (the plural-plus-suffix form,
# left) alongside the per-term columns that sum to it (the shared
# singular-prefixed family, right) whenever both the aggregate and its parts
# are requested from the model. Counting the aggregate beside its own parts
# double-counts the same variation, so this mapping lets both
# ``component_variance_shares`` and ``figures.plot_decomposition_stack`` drop
# an aggregate whenever at least one column of its family is present, and keep
# it otherwise, from one place.
AGGREGATE_CONSTITUENT_PREFIXES = {
    'future_regressors_additive': 'future_regressor_',
    'future_regressors_multiplicative': 'future_regressor_',
    'seasonalities_additive': 'season_',
    'seasonalities_multiplicative': 'season_',
    'events_additive': 'event_',
    'events_multiplicative': 'event_',
}


def decomposition_columns(components, columns=None):
    """
    Component columns to treat as one term each, redundant aggregates dropped.

    NeuralProphet's decomposition can include both a family aggregate (e.g.
    ``future_regressors_additive``) and the per-term columns that sum to it
    (e.g. ``future_regressor_tair``, ``future_regressor_rh``). Keeping both
    counts the same contribution twice, so this drops an aggregate whenever at
    least one column of its family (see ``AGGREGATE_CONSTITUENT_PREFIXES``) is
    also present in ``columns``. When a model's parts were never emitted or
    never requested, the aggregate is the only representation of that family's
    contribution and is kept.

    Parameters
    ----------
    components : pd.DataFrame
        Output of ``decompose_components``.
    columns : sequence of str or None, optional
        Candidate component columns. ``None`` takes every column except
        ``y``, ``yhat1`` and ``ID``. Default ``None``.

    Returns
    -------
    list of str
        ``columns`` (or the default column set), in their original order,
        with a redundant aggregate removed for each family whose constituents
        are present.
    """
    if columns is None:
        columns = [c for c in components.columns
                   if c not in _NON_COMPONENT_COLUMNS]
    columns = list(columns)
    present = set(columns)
    return [c for c in columns
            if not (c in AGGREGATE_CONSTITUENT_PREFIXES
                    and any(other != c and other.startswith(
                        AGGREGATE_CONSTITUENT_PREFIXES[c])
                        for other in present))]


def component_variance_shares(components, columns=None):
    """
    What share of the fitted variation each additive component carries.

    This is the decomposition's headline claim expressed as a number: a record
    whose daily component carries most of the variance is a thermometer, and one
    whose trend does is a structure that is moving. The residual is included as
    a component so that the shares are comparable and sum to one. A family
    aggregate (e.g. ``future_regressors_additive``) is dropped whenever at
    least one of its constituent columns (e.g. ``future_regressor_tair``) is
    also present, via :func:`decomposition_columns`, so that a regressor's
    contribution is never counted once whole and once in parts.

    Parameters
    ----------
    components : pd.DataFrame
        Output of ``decompose_components``.
    columns : sequence of str or None, optional
        Components to include. ``None`` takes every column except ``y``,
        ``yhat1`` and ``ID``. Default ``None``.

    Returns
    -------
    pd.DataFrame
        Columns ``component``, ``variance``, ``share``, ``mean`` and
        ``peak_to_peak``, ordered by descending share.
    """
    columns = decomposition_columns(components, columns)
    frame = components.loc[:, columns].apply(
        pd.to_numeric, errors='coerce')

    variance = frame.var(ddof=0)
    total = float(variance.sum())
    table = pd.DataFrame({
        'component': variance.index,
        'variance': variance.to_numpy(),
        'share': (variance / total).to_numpy() if total > 0 else np.nan,
        'mean': frame.mean().to_numpy(),
        'peak_to_peak': (frame.max() - frame.min()).to_numpy(),
    })
    return (table.sort_values('share', ascending=False, kind='stable')
            .reset_index(drop=True))


def residual_diagnostics(residuals, lags=(1, 24, 72)):
    """
    Whether anything is left in the residual, and on what scale it sits.

    Structure surviving in the residual means a component of the model is
    missing. That matters twice over here: it makes the decomposition's
    attribution wrong, and it makes a residual-based alarm fire on model error
    rather than on the structure.

    Parameters
    ----------
    residuals : pd.Series
        Observed minus predicted, indexed by timestamp. Missing values are
        dropped before testing. Dropping compacts the series, so on a gapped
        record a lag counts positions in the surviving subsequence rather
        than a fixed interval of calendar time.
    lags : sequence of int, optional
        Ljung-Box lags to test. Default ``(1, 24, 72)``. A lag that is not
        shorter than the number of surviving observations cannot be tested
        and is reported as ``NaN`` rather than raising.

    Returns
    -------
    pd.DataFrame
        One row per lag, with ``lag``, ``lb_stat``, ``lb_pvalue``, and the
        scale columns ``n``, ``std`` and ``mad`` repeated on every row.

    Notes
    -----
    Requires ``statsmodels``.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    values = pd.to_numeric(residuals, errors='coerce').dropna()
    lags = [int(lag) for lag in lags]

    testable = [lag for lag in lags if 0 < lag < values.size]
    if testable:
        result = acorr_ljungbox(values, lags=testable, return_df=True)
        statistics = dict(zip(testable, result['lb_stat'].to_numpy()))
        pvalues = dict(zip(testable, result['lb_pvalue'].to_numpy()))
    else:
        statistics, pvalues = {}, {}

    scale = {
        'n': int(values.size),
        'std': float(values.std(ddof=1)) if values.size > 1 else np.nan,
        'mad': float((values - values.median()).abs().median()),
    }
    return pd.DataFrame({
        'lag': lags,
        'lb_stat': [statistics.get(lag, np.nan) for lag in lags],
        'lb_pvalue': [pvalues.get(lag, np.nan) for lag in lags],
        'n': scale['n'],
        'std': scale['std'],
        'mad': scale['mad'],
    })


def period_scan(series, min_days=30.0, max_days=900.0, n_periods=4000, top=5,
                spacing='linear'):
    """
    Confirm, by measurement, that a claimed periodic component has the
    period it is claimed to have — on a record too gappy for an FFT.

    This exists to answer a specific kind of question honestly: when a
    decomposition names a component ``season_yearly``, is there actually a
    roughly annual period in the underlying record, or is that just the
    label the model happened to give a term the fit found useful? An FFT
    cannot answer this on a gapped structural record without first
    imputing the gaps, which would let the imputation choice, not the
    record, decide the answer. The Lomb-Scargle periodogram takes irregular,
    gappy sampling as it comes: it fits a sinusoid of each candidate
    frequency to the surviving samples directly, with no interpolation
    step, so a claim about a component's period can rest on a measurement
    of the actual timestamps present rather than on a figure the reader has
    to trust.

    Missing values are dropped, time is measured in days from the first
    surviving sample, and the series mean is subtracted before the
    periodogram is evaluated (:func:`scipy.signal.lombscargle` assumes a
    zero baseline). Power is evaluated at ``n_periods`` candidate periods
    spaced linearly between ``min_days`` and ``max_days``, normalized so
    that a perfectly matched sinusoid is close to the periodogram's own
    maximum.

    A periodogram built from a finite record cannot resolve a period more
    finely than about ``period ** 2 / span_days``: a genuinely single cycle
    near that period shows up as one broad peak with sidelobes on either
    side, not as several distinct nearby periods. A candidate is therefore
    treated as the same feature as an already-reported peak, and skipped,
    whenever it falls within that peak's own resolution of it — this is
    what keeps a single annual cycle on a three-year record from being
    reported as three or four separate ones. The strongest ``top`` peaks
    that survive this suppression are returned.

    Parameters
    ----------
    series : pd.Series
        Numeric signal indexed by timestamp. Missing values (``NaN`` or an
        absent index entry) are dropped before scanning.
    min_days, max_days : float, optional
        Bounds, in days, of the candidate period grid. Defaults ``30.0`` and
        ``900.0``, which spans from a month to roughly two and a half years
        and so brackets an annual cycle with headroom on both sides.
    n_periods : int, optional
        Number of candidate periods evaluated, linearly spaced between
        ``min_days`` and ``max_days``. Default ``4000``.
    top : int, optional
        Maximum number of peaks to return. Default ``5``.
    spacing : {'linear', 'log'}, optional
        How the candidate grid is laid out between ``min_days`` and
        ``max_days``. Default ``'linear'``, ``np.linspace`` — unchanged
        from every call site that predates this parameter. A peak's own
        resolution near period ``p`` is about ``p ** 2 / span_days``: on a
        record spanning years, a linear grid built to reach from half a
        day to hundreds of days takes steps at the long-period end far
        wider than the width of the daily or twelve-hour peak, and can
        step past it entirely. ``'log'`` (``np.geomspace``) holds the
        *relative* spacing constant instead, so the grid remains dense
        enough near a day to certify a sub-daily peak on the same scan
        that also reaches out to the annual one. Any other value raises
        ``ValueError``.

    Returns
    -------
    pd.DataFrame
        Columns ``rank`` (1-indexed, strongest first), ``period_days``,
        ``period_years``, ``power``, ``resolution_days``, plus ``n`` and
        ``span_days`` describing the input series and repeated on every row.
        ``resolution_days`` is ``period_days ** 2 / span_days``, the width
        this record can resolve around that period: two rows closer
        together than the smaller one's ``resolution_days`` would in truth
        be one feature, not two, so a reader comparing ``period_days``
        against its own ``resolution_days`` can see at a glance how many
        digits of that number are real. Fewer than ``top`` rows are
        returned if fewer independent peaks survive the resolution-based
        suppression.

    Raises
    ------
    ValueError
        If fewer than 50 samples survive dropping missing values. A
        periodogram built from fewer points than that is not a measurement
        of a period, it is noise with a shape. Also raised if ``spacing``
        is neither ``'linear'`` nor ``'log'``.

    Notes
    -----
    ``scipy.signal.lombscargle`` evaluates the full (frequencies, samples)
    pair set in one call. On a multi-year record at native temporal
    resolution and a scan of tens of thousands of candidate periods, that
    array is large enough to exceed ordinary machine memory and be killed
    by the operating system with no Python exception to explain it. To
    keep peak memory bounded, the periodogram is evaluated in slices of
    :data:`_LOMBSCARGLE_CHUNK` frequencies at a time; each frequency's
    normalized power is independent of every other, so the sliced result
    is numerically identical to evaluating all frequencies at once.
    """
    from scipy.signal import lombscargle

    values = pd.to_numeric(series, errors='coerce').dropna()
    if values.size < 50:
        raise ValueError(
            'period_scan: only {0} usable samples after dropping missing '
            'values; at least 50 are required, because a periodogram built '
            'from fewer points is noise, not a measurement.'.format(
                values.size))

    index = pd.DatetimeIndex(values.index)
    t_days = ((index - index[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    y = values.to_numpy(dtype=float) - float(values.mean())
    span_days = float(t_days[-1] - t_days[0])

    if spacing == 'linear':
        periods = np.linspace(float(min_days), float(max_days), int(n_periods))
    elif spacing == 'log':
        periods = np.geomspace(float(min_days), float(max_days), int(n_periods))
    else:
        raise ValueError(
            "period_scan: spacing must be 'linear' or 'log', got {0!r}."
            .format(spacing))
    angular_freqs = 2.0 * np.pi / periods
    power = np.empty(len(angular_freqs))
    for start in range(0, len(angular_freqs), _LOMBSCARGLE_CHUNK):
        stop = start + _LOMBSCARGLE_CHUNK
        power[start:stop] = lombscargle(
            t_days, y, angular_freqs[start:stop], normalize=True)

    # A peak's own resolution — period**2/span_days — is the width around
    # it that this record cannot tell apart from the peak itself, so a
    # later candidate falling inside an already-kept peak's resolution is
    # one of its sidelobes, not a second cycle.
    order = np.argsort(power)[::-1]
    kept_periods, kept_power = [], []
    for i in order:
        candidate = periods[i]
        if any(abs(candidate - kept) < (kept ** 2 / span_days)
               for kept in kept_periods):
            continue
        kept_periods.append(float(candidate))
        kept_power.append(float(power[i]))
        if len(kept_periods) >= int(top):
            break

    return pd.DataFrame({
        'rank': np.arange(1, len(kept_periods) + 1),
        'period_days': kept_periods,
        'period_years': [p / 365.25 for p in kept_periods],
        'power': kept_power,
        'resolution_days': [p ** 2 / span_days for p in kept_periods],
        'n': int(values.size),
        'span_days': span_days,
    })


def seasonal_weights(index, modulation=None, peak_doy=196):
    """
    Condition columns for the smoothly weighted daily seasonality (spec D7).

    Two weights that sum to one at every timestamp and vary only with the
    calendar. With a measured annual modulation the summer weight is that
    curve min-max normalised over one year of days; without one it is a
    cosine peaking at ``peak_doy``.

    Parameters
    ----------
    index : pd.DatetimeIndex
    modulation : dict or None, optional
        ``fit`` from :func:`shmlib.coupling.annual_modulation`. Default
        ``None`` (cosine fallback).
    peak_doy : int, optional
        Day of year of the fallback cosine's maximum. Default ``196``.

    Returns
    -------
    pd.DataFrame
        ``summer_w`` and ``winter_w`` indexed by ``index``.
    """
    index = pd.DatetimeIndex(index)
    if modulation is None:
        doy = index.dayofyear.to_numpy(dtype=float)
        summer = 0.5 * (1.0 - np.cos(2.0 * np.pi * (doy - (peak_doy - 182.625))
                                     / 365.25))
    else:
        year = pd.date_range('2001-01-01', periods=365, freq='D')
        curve = coupling.evaluate_modulation(modulation, year)
        low, high = float(curve.min()), float(curve.max())
        raw = coupling.evaluate_modulation(modulation, index).to_numpy()
        summer = (raw - low) / (high - low) if high > low else np.full_like(raw, 0.5)
    summer = np.clip(summer, 0.0, 1.0)
    return pd.DataFrame({'summer_w': summer, 'winter_w': 1.0 - summer},
                        index=index)


def ols_residual(target, drivers):
    """
    Residual of an ordinary-least-squares fit of the target on the drivers.

    The plain linear fit is not a model of the wall; it is the quickest way
    to remove what the regressors explain, so that the harmonic diagnostic
    of spec D6 can look at what the seasonal terms will actually have to
    carry. Only rows where the target and every driver are present enter
    the fit, and the residual is returned on those rows.

    Parameters
    ----------
    target : pd.Series
        The response.
    drivers : pd.DataFrame
        The regressors, on an index compatible with ``target``.

    Returns
    -------
    residual : pd.Series
        ``target`` minus the fit, on the paired index, named ``'residual'``.
    gains : pd.Series
        Fitted coefficients indexed ``['intercept', *drivers.columns]``.
    """
    paired = pd.concat([target.rename('__target__'), drivers], axis=1).dropna()
    columns = list(drivers.columns)
    design = np.column_stack([np.ones(len(paired)),
                              paired[columns].to_numpy(dtype=float)])
    coef, _, _, _ = np.linalg.lstsq(design, paired['__target__'].to_numpy(dtype=float),
                                    rcond=None)
    residual = pd.Series(paired['__target__'].to_numpy(dtype=float) - design @ coef,
                         index=paired.index, name='residual')
    gains = pd.Series(coef, index=['intercept', *columns])
    return residual, gains


def has_certified_period(scan, period_days, tolerance_days=None):
    """
    Whether a candidate period is certified by a :func:`period_scan` table.

    A period is certified when some row of ``scan`` reports a
    ``period_days`` within tolerance of the candidate. The honest default
    tolerance is that row's own ``resolution_days`` — the width the record
    itself cannot resolve around that period — rather than a value chosen
    to make the answer come out a particular way. A fixed
    ``tolerance_days`` is for a sub-daily candidate (the twelve-hour tide
    of a diurnal cycle, say) whose ``resolution_days`` on a multi-year scan
    is thousandths of a day: too fine a tolerance to allow for the grid's
    own finite spacing, so a fixed, coarser tolerance is supplied instead.

    Parameters
    ----------
    scan : pd.DataFrame
        Output of :func:`period_scan`, with columns ``period_days`` and
        ``resolution_days``.
    period_days : float
        The candidate period to check for, in days.
    tolerance_days : float or None, optional
        Fixed tolerance, in days. Default ``None``, meaning each row's own
        ``resolution_days`` is used as that row's tolerance.

    Returns
    -------
    bool
        ``True`` if any row of ``scan`` certifies ``period_days``.
    """
    period_days = float(period_days)
    diff = (scan['period_days'].astype(float) - period_days).abs()
    tolerance = (scan['resolution_days'].astype(float) if tolerance_days is None
                else float(tolerance_days))
    return bool((diff < tolerance).any())


def regressor_gains(components, frame, regressors):
    """
    The gain each future regressor learned, read from its component.

    NeuralProphet's additive future regressor is linear in the regressor, so
    the slope of the component against the regressor value is the learned
    coefficient in the series' own units per unit of the driver.

    Parameters
    ----------
    components : pd.DataFrame
        Output of :func:`decompose_components`.
    frame : pd.DataFrame
        The frame the components were computed on, holding the regressors.
    regressors : sequence of str

    Returns
    -------
    pd.DataFrame
        ``regressor``, ``gain``, ``r2``.
    """
    rows = []
    for name in regressors:
        paired = pd.concat([components[f'future_regressor_{name}'], frame[name]],
                           axis=1).dropna()
        paired.columns = ['contribution', 'driver']
        slope, intercept = np.polyfit(paired['driver'], paired['contribution'], 1)
        fitted = slope * paired['driver'] + intercept
        total = float(((paired['contribution'] - paired['contribution'].mean()) ** 2).sum())
        r2 = 1.0 - float(((paired['contribution'] - fitted) ** 2).sum()) / total if total else np.nan
        rows.append({'regressor': name, 'gain': float(slope), 'r2': r2})
    return pd.DataFrame(rows, columns=['regressor', 'gain', 'r2'])


def trend_parameters(model, frame, changepoints, regressors=()):
    """
    The fitted trend on the record, and its rate on each segment.

    Read through the public ``predict_trend``; the rate of each segment
    between consecutive changepoints is the trend's change across that
    segment per year, which is what a rate-change plot shows without
    touching the model's internal deltas. The model frame is built through
    :func:`_frame_columns`, so a model fitted with lagged regressors or
    conditional seasonality (via ``neuralprophet_backtest``'s
    ``lagged_regressors`` or ``conditional_seasonality``) gets those columns
    back automatically, exactly as :func:`decompose_components` does.

    Parameters
    ----------
    model : object
        Fitted NeuralProphet model, exposing ``predict_trend``.
    frame : pd.DataFrame
        Datetime-indexed rows the trend is evaluated on.
    changepoints : pd.DatetimeIndex
        Changepoint locations the segments are cut at, normally the same
        ones the model was fitted with.
    regressors : sequence of str, optional
        Regressor columns to pass through. Default empty.

    Returns
    -------
    (pd.Series, pd.DataFrame)
        ``trend`` indexed like ``frame``; ``rates`` with ``start``, ``end``,
        ``rate_mdeg_per_year``.
    """
    df = _model_frame(frame, _frame_columns(model, frame, regressors))
    predicted = model.predict_trend(df)
    trend = pd.Series(predicted['trend'].to_numpy(dtype=float),
                      index=pd.DatetimeIndex(frame.index[:len(predicted)]))
    edges = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(changepoints))
                                    | {trend.index.min(), trend.index.max()}))
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        segment = trend.loc[start:end].dropna()
        if len(segment) < 2:
            continue
        days = (segment.index[-1] - segment.index[0]) / pd.Timedelta(days=1)
        rows.append({'start': segment.index[0], 'end': segment.index[-1],
                     'rate_mdeg_per_year': float(
                         (segment.iloc[-1] - segment.iloc[0]) / days * 365.25)})
    return trend, pd.DataFrame(rows, columns=['start', 'end', 'rate_mdeg_per_year'])


def seasonal_parameters(model, dates, freq='20min', conditions=None,
                        regressors=()):
    """
    The fitted seasonal curves, evaluated on synthetic days and one year.

    For each date one day is built on the grid, with condition columns from
    :func:`seasonal_weights` where the model was fitted with them, and every
    regressor set to zero; ``predict_seasonal_components`` then gives each
    seasonal term over that day. The yearly term is evaluated separately,
    over one synthetic year at daily resolution: NeuralProphet reports the
    ``'yearly'`` component on every prediction regardless of the horizon it
    was asked for, so a per-day evaluation would otherwise also carry
    ``'yearly'`` rows — one non-representative value per hour of that single
    day, mixed in with the 365 actual daily values from the year sweep, on a
    ``date`` that duplicates none of them. The per-day evaluation therefore
    keeps only the components that are not ``'yearly'``, and the year sweep
    below is the sole source of ``'yearly'`` rows. The notebook sets
    ``model.weight_curve_ = WEIGHT_CURVE`` after each fit so this function
    can rebuild the condition columns with the same seasonal-weight
    modulation the model was fitted under.

    Parameters
    ----------
    model : object
        Fitted NeuralProphet model, exposing ``predict_seasonal_components``.
    dates : sequence
        Calendar dates (parseable by ``pd.Timestamp``) to evaluate the daily
        term on.
    freq : str, optional
        Sampling frequency of each synthetic day. Default ``'20min'``.
    conditions : dict or None, optional
        Maps a seasonality name to its condition column, as passed to
        ``neuralprophet_backtest``'s ``conditional_seasonality``. ``None``
        (default) evaluates no conditional term.
    regressors : sequence of str, optional
        Regressor columns the model expects; each is set to ``0.0`` on every
        synthetic row. Default empty.

    Returns
    -------
    pd.DataFrame
        Long: ``date``, ``hour``, ``component``, ``value``. ``date`` is a
        tz-naive calendar date on every row — the daily rows already carry
        it that way, and the yearly rows' timestamps (built tz-aware, on the
        internal ``'2001-01-01'`` synthetic year) are stripped of their
        ``tz`` before being written, so the whole column stays one
        comparable, sortable dtype rather than raising ``TypeError`` on a
        tz-naive/tz-aware mix.
    """
    conditions = dict(conditions or {})
    rows = []

    def _evaluate(index, date_label):
        frame = pd.DataFrame({'y': 0.0}, index=index)
        for name in regressors:
            frame[name] = 0.0
        if conditions:
            weights = seasonal_weights(index, modulation=getattr(model, 'weight_curve_', None))
            for column in conditions.values():
                frame[column] = weights[column].to_numpy()
        df = _model_frame(frame, tuple(regressors) + tuple(conditions.values()))
        out = model.predict_seasonal_components(df)
        for component in [c for c in out.columns if c not in ('ds', 'ID', 'yearly')]:
            for stamp, value in zip(index, out[component].to_numpy()):
                rows.append({'date': date_label, 'hour': stamp.hour + stamp.minute / 60.0,
                             'component': component, 'value': float(value)})

    for date in dates:
        day = pd.Timestamp(date, tz='UTC')
        _evaluate(pd.date_range(day, day + pd.Timedelta('1D'), freq=freq,
                                inclusive='left'), pd.Timestamp(date))
    year = pd.date_range('2001-01-01', periods=365, freq='D', tz='UTC')
    frame = pd.DataFrame({'y': 0.0}, index=year)
    for name in regressors:
        frame[name] = 0.0
    if conditions:
        weights = seasonal_weights(year, modulation=getattr(model, 'weight_curve_', None))
        for column in conditions.values():
            frame[column] = weights[column].to_numpy()
    out = model.predict_seasonal_components(
        _model_frame(frame, tuple(regressors) + tuple(conditions.values())))
    if 'yearly' in out.columns:
        for stamp, value in zip(year, out['yearly'].to_numpy()):
            rows.append({'date': stamp.tz_localize(None), 'hour': 0.0,
                         'component': 'yearly', 'value': float(value)})
    return pd.DataFrame(rows, columns=['date', 'hour', 'component', 'value'])


def regressor_set_frames(sets, target, roles=('tair', 'rh', 'sr'), weight_curve=None):
    """
    Model-ready frames for Model A, one per regressor set (spec D7).

    Each regressor set produced by ``proxies.build_regressor_sets`` already
    carries the study's three roles under plain names on the record's index;
    this joins the shared target onto each set in turn, attaches the
    conditional-seasonality weight columns every Model A fit needs whether or
    not it ends up using them, and drops whatever a set cannot cover. Rows
    are dropped rather than filled, because a regressor set's whole purpose
    is to say what that combination of sources can support, and imputing a
    gap here would blur that comparison between sets.

    Parameters
    ----------
    sets : dict
        ``{set_name: DataFrame}``, the first return of
        :func:`shmlib.proxies.build_regressor_sets`: each block carries the
        columns named in ``roles`` (and, unused here, a ``<role>_filled``
        flag per role) on the record's index.
    target : pd.Series
        The response, on an index compatible with every set in ``sets``.
        Renamed ``'y'`` in the output regardless of its own name.
    roles : sequence of str, optional
        Role columns to carry from each set. Default ``('tair', 'rh', 'sr')``.
    weight_curve : dict or None, optional
        Annual modulation passed to :func:`seasonal_weights` as
        ``modulation``, evaluated on each set's own index. ``None`` (the
        default) uses that function's cosine fallback.

    Returns
    -------
    dict
        ``{set_name: DataFrame}``, each with columns ``'y'``, the roles (in
        the order of ``roles``), then ``'summer_w'`` and ``'winter_w'``. Rows
        where the target or any role is missing are dropped; the sets'
        ``_filled`` flags are not carried through.
    """
    roles = list(roles)
    target = pd.Series(target).rename('y')
    frames = {}
    for name, block in sets.items():
        weights = seasonal_weights(block.index, modulation=weight_curve)
        joined = pd.concat([target, block.loc[:, roles], weights], axis=1)
        joined = joined.dropna(subset=['y'] + roles)
        frames[name] = joined.loc[:, ['y'] + roles + ['summer_w', 'winter_w']]
    return frames


def sweep_trend_reg(train, valid, candidates, regressors, **model_kwargs):
    """
    Score a nowcast fit at each candidate trend regularisation (spec D7).

    One :func:`neuralprophet_backtest` fit per candidate, all else held
    fixed, scored on the same held-out tail by the mean absolute error of
    its long predictions. Reporting every candidate's score, rather than
    only the winner, is what lets the study show that the choice is not
    close, or admit it when it is.

    Parameters
    ----------
    train, valid : pd.DataFrame
        Training and held-out frames, as accepted by
        :func:`neuralprophet_backtest`.
    candidates : sequence of float
        Trend regularisation values to try, in the wrapper's user-facing
        scale (see :func:`neuralprophet_backtest`'s ``trend_reg``).
    regressors : sequence of str
        Regressor columns registered on every fit.
    **model_kwargs
        Passed unchanged to every :func:`neuralprophet_backtest` call beside
        ``train``, ``valid``, ``regressors``, ``task`` and ``trend_reg``,
        which this function already supplies.

    Returns
    -------
    pd.DataFrame
        One row per candidate, columns ``trend_reg``, ``mae_val`` and
        ``chosen`` — ``True`` on the row of lowest ``mae_val``, the first
        such row on a tie, ``False`` everywhere else.
    """
    rows = []
    for candidate in candidates:
        _, out = neuralprophet_backtest(
            train, valid, regressors=regressors, task='nowcast',
            trend_reg=candidate, **model_kwargs)
        rows.append({'trend_reg': float(candidate),
                     'mae_val': float((out['y'] - out['yhat']).abs().mean())})
    table = pd.DataFrame(rows, columns=['trend_reg', 'mae_val'])
    table['chosen'] = False
    table.loc[table['mae_val'].idxmin(), 'chosen'] = True
    return table


def compare_daily_terms(train, valid, regressors, conditions, keep_min_share,
                        **model_kwargs):
    """
    Whether the conditional daily term earns its place over the plain one
    (spec D7).

    Two otherwise identical nowcast fits, one with no conditional
    seasonality and one with ``conditions`` registered as
    ``conditional_seasonality``. For each, the held-out mean absolute error
    and the share of the fitted variance the daily term(s) carry — the sum
    of :func:`component_variance_shares`' shares over every component whose
    name starts with ``'season_daily'``, which covers both the plain fit's
    single ``season_daily`` column and the conditional fit's per-condition
    columns (``season_daily_summer``, ``season_daily_winter``, ...), so the
    two fits' daily shares are directly comparable even though they are
    spread over a different number of columns.

    Parameters
    ----------
    train, valid : pd.DataFrame
        Training and held-out frames, as accepted by
        :func:`neuralprophet_backtest`.
    regressors : sequence of str
        Regressor columns registered on both fits.
    conditions : dict
        ``conditional_seasonality`` mapping used by the ``'conditional'``
        fit; the ``'plain'`` fit is run with ``None`` instead.
    keep_min_share : float
        Minimum daily variance share the conditional fit must clear to be
        preferred when its held-out error is not already lower.
    **model_kwargs
        Passed unchanged to both :func:`neuralprophet_backtest` calls beside
        ``train``, ``valid``, ``regressors``, ``task`` and
        ``conditional_seasonality``, which this function already supplies.

    Returns
    -------
    table : pd.DataFrame
        Two rows, ``daily_term`` (``'plain'`` then ``'conditional'``),
        ``mae_val`` and ``daily_share``.
    keep : bool
        ``True`` when the conditional fit's ``mae_val`` is lower than the
        plain fit's, or its ``daily_share`` exceeds ``keep_min_share``.
    """
    rows = []
    for label, spec in (('plain', None), ('conditional', dict(conditions))):
        model, out = neuralprophet_backtest(
            train, valid, regressors=regressors, task='nowcast',
            conditional_seasonality=spec, **model_kwargs)
        components = decompose_components(model, train, regressors=regressors)
        shares = component_variance_shares(components)
        daily_share = float(
            shares.loc[shares['component'].str.startswith('season_daily'),
                       'share'].sum())
        rows.append({'daily_term': label,
                     'mae_val': float((out['y'] - out['yhat']).abs().mean()),
                     'daily_share': daily_share})
    table = pd.DataFrame(rows, columns=['daily_term', 'mae_val', 'daily_share'])
    keep = bool(table.loc[1, 'mae_val'] < table.loc[0, 'mae_val']
               or table.loc[1, 'daily_share'] > keep_min_share)
    return table, keep


def attribution_fits(frames, regressors, valid_p, n_changepoints,
                     study03_gains=None, diagnostic_lags=(1, 72, 216),
                     weight_curve=None, **model_kwargs):
    """
    Fit Model A once per regressor set, for attribution (spec D7).

    Each set's frame is split into a training head and a held-out tail,
    changepoints are placed on the training head's own covered time, and a
    single nowcast fit is scored against the held-out tail — the same fit a
    caller then reads shares, gains and residual diagnostics from, rather
    than a second, silently different model. The fitted model carries
    ``weight_curve_``, so :func:`seasonal_parameters` can later rebuild the
    same conditional-seasonality weights this fit was trained under.

    Parameters
    ----------
    frames : dict
        ``{set_name: DataFrame}``, normally :func:`regressor_set_frames`'s
        return: each frame carries ``'y'`` and every column ``regressors``
        names.
    regressors : sequence of str
        Regressor columns registered on every fit.
    valid_p : float
        Fraction of each set's frame held out as the tail scored by
        ``mae_val`` and passed as ``validation`` to
        :func:`neuralprophet_backtest`.
    n_changepoints : int
        Passed to :func:`covered_changepoints` for each set's training head,
        and to :func:`neuralprophet_backtest` as ``n_changepoints``.
    study03_gains : dict or None, optional
        ``{(set_name, regressor): gain}`` lookup placed beside every learned
        gain as ``study03_gain``. ``None`` (the default), or a lookup
        lacking a given ``(set_name, regressor)`` key, reports ``NaN`` for
        that row rather than raising.
    diagnostic_lags : sequence of int, optional
        Lags passed to :func:`residual_diagnostics`. Default ``(1, 72,
        216)``.
    weight_curve : dict or None, optional
        Stored on every fitted model as ``weight_curve_``. Default ``None``.
    **model_kwargs
        Passed unchanged to every :func:`neuralprophet_backtest` call beside
        ``train``, ``valid``, ``regressors``, ``task``, ``changepoints``,
        ``n_changepoints`` and ``validation``, which this function already
        supplies.

    Returns
    -------
    fits : dict
        ``{set_name: {'model', 'train', 'valid', 'changepoints',
        'components'}}``, the fitted model, its training and held-out
        frames, its changepoints, and its :func:`decompose_components`
        output.
    shares : pd.DataFrame
        :func:`component_variance_shares` for every set, concatenated, with
        ``'set'`` as the first column.
    gains : pd.DataFrame
        :func:`regressor_gains` for every set, concatenated, with ``'set'``
        as the first column and ``study03_gain`` appended.
    diagnostics : pd.DataFrame
        :func:`residual_diagnostics` of each set's residual, concatenated,
        with ``'set'`` as the first column.
    """
    study03_gains = study03_gains or {}
    fits = {}
    shares_rows, gain_rows, diag_rows = [], [], []
    for name, block in frames.items():
        split_at = int(len(block) * (1 - valid_p))
        train, valid = block.iloc[:split_at], block.iloc[split_at:]
        changepoints = covered_changepoints(train.index, n_changepoints)
        model, _ = neuralprophet_backtest(
            train, valid, regressors=regressors, task='nowcast',
            changepoints=changepoints, n_changepoints=n_changepoints,
            validation=valid, **model_kwargs)
        model.weight_curve_ = weight_curve
        components = decompose_components(model, train, regressors=regressors)
        fits[name] = {'model': model, 'train': train, 'valid': valid,
                      'changepoints': changepoints, 'components': components}

        share_table = component_variance_shares(components)
        share_table.insert(0, 'set', name)
        shares_rows.append(share_table)

        gain_table = regressor_gains(components, train, regressors)
        gain_table.insert(0, 'set', name)
        gain_table['study03_gain'] = [study03_gains.get((name, regressor), np.nan)
                                      for regressor in gain_table['regressor']]
        gain_rows.append(gain_table)

        diag_table = residual_diagnostics(components['residual'], lags=diagnostic_lags)
        diag_table.insert(0, 'set', name)
        diag_rows.append(diag_table)

    shares = pd.concat(shares_rows, ignore_index=True)
    gains = pd.concat(gain_rows, ignore_index=True)
    diagnostics = pd.concat(diag_rows, ignore_index=True)
    return fits, shares, gains, diagnostics


def _select_by_ds(block, ds):
    """
    Rows of ``block`` whose index appears in ``ds``, a fold boundary column
    from ``crossvalidation_split_df``.

    NeuralProphet's ``crossvalidation_split_df`` returns its ``ds`` column
    tz-naive regardless of whether the frame it split carried a tz-aware
    index — NeuralProphet 0.8.0's own preprocessing strips any zone through
    UTC (``df_utils.py``: ``pd.to_datetime(ds, utc=True).dt.tz_convert(None)``),
    so a naive ``ds`` it returns is a UTC wall-clock reading, not a reading
    already expressed in the block's own zone. A tz-aware ``block`` is
    therefore matched by first attaching UTC to ``ds`` and only then
    converting to the block's zone; localising ``ds`` directly to the
    block's zone, as an earlier version of this function did, would treat a
    UTC instant as if it were already local wall time and silently
    mis-select rows for any zone other than UTC itself. A tz-naive ``block``
    is matched as given, unchanged.

    Parameters
    ----------
    block : pd.DataFrame
        Datetime-indexed frame, one of Model A's regressor-set frames.
    ds : pd.Series
        The ``'ds'`` column of one side of one fold.

    Returns
    -------
    pd.DataFrame
        The subset of ``block`` whose index timestamps appear in ``ds``.
    """
    if ds.dt.tz is None and block.index.tz is not None:
        ds = ds.dt.tz_localize('UTC').dt.tz_convert(block.index.tz)
    return block.loc[block.index.isin(ds)]


def fold_stability(fits, frames, regressors, n_changepoints, k, fold_pct,
                   fold_overlap_pct, freq, gain_regressor='tair',
                   **model_kwargs):
    """
    Component stability across NeuralProphet's own chronological folds
    (spec §4.1).

    A learned gain, trend rate or yearly amplitude that changes sign or
    order of magnitude between folds is not a finding about the wall, it is
    noise the single full-window fit happened to land on. Each set's
    already-fitted model (from :func:`attribution_fits`) supplies the fold
    boundaries through its own ``crossvalidation_split_df``, so the folds
    match the same chronological scheme NeuralProphet itself uses for
    walk-forward validation; each fold is then refit from scratch on its own
    training slice, because a fold's stability is about repeated fitting,
    not about reusing one fit's parameters.

    Parameters
    ----------
    fits : dict
        ``{set_name: {'model', ...}}``, :func:`attribution_fits`'s first
        return; only ``fits[name]['model']`` is used, to call
        ``crossvalidation_split_df``.
    frames : dict
        ``{set_name: DataFrame}``, normally the same
        :func:`regressor_set_frames` mapping ``attribution_fits`` was given.
    regressors : sequence of str
        Regressor columns registered on every fold's fit.
    n_changepoints : int
        Passed to :func:`covered_changepoints` for each fold's training
        slice, and to :func:`neuralprophet_backtest` as ``n_changepoints``.
    k : int
        Number of folds, passed to ``crossvalidation_split_df``.
    fold_pct, fold_overlap_pct : float
        Fold size and overlap, as fractions of the frame, passed to
        ``crossvalidation_split_df``.
    freq : str
        Sampling frequency, passed to ``crossvalidation_split_df`` and, as
        ``freq``, to every fold's :func:`neuralprophet_backtest` call.
    gain_regressor : str, optional
        The regressor whose learned gain is reported per fold. Default
        ``'tair'``.
    **model_kwargs
        Passed unchanged to every fold's :func:`neuralprophet_backtest` call
        beside ``fold_train``, ``fold_test``, ``regressors``, ``task``,
        ``changepoints``, ``n_changepoints``, ``freq`` and ``quantiles``,
        which this function already supplies — folds are always fitted
        without quantiles, so ``model_kwargs`` must not also carry
        ``quantiles``.

    Returns
    -------
    pd.DataFrame
        One row per set and fold: ``set``, ``fold``,
        ``'<gain_regressor>_gain'``, ``yearly_peak_to_peak`` (``NaN`` when
        the model has no ``season_yearly`` component), ``trend_rate`` (the
        mean of that fold's :func:`trend_parameters` rates), and the fold's
        held-out ``mae_val``.
    """
    conditions = dict(model_kwargs.get('conditional_seasonality') or {})
    passthrough = tuple(regressors) + tuple(conditions.values())
    gain_column = f'{gain_regressor}_gain'
    rows = []
    for name, block in frames.items():
        model0 = fits[name]['model']
        folds = model0.crossvalidation_split_df(
            _model_frame(block, passthrough), freq=freq, k=k,
            fold_pct=fold_pct, fold_overlap_pct=fold_overlap_pct)
        for fold_index, (fold_train_df, fold_test_df) in enumerate(folds, start=1):
            fold_train = _select_by_ds(block, fold_train_df['ds'])
            fold_test = _select_by_ds(block, fold_test_df['ds'])
            changepoints = covered_changepoints(fold_train.index, n_changepoints)
            model, out = neuralprophet_backtest(
                fold_train, fold_test, regressors=regressors, task='nowcast',
                changepoints=changepoints, n_changepoints=n_changepoints,
                freq=freq, quantiles=(), **model_kwargs)
            components = decompose_components(model, fold_train, regressors=regressors)
            gain = float(regressor_gains(components, fold_train, (gain_regressor,))
                        .loc[0, 'gain'])
            _, rates = trend_parameters(model, fold_train, changepoints,
                                        regressors=regressors)
            yearly = (components['season_yearly']
                     if 'season_yearly' in components.columns else None)
            rows.append({
                'set': name, 'fold': fold_index, gain_column: gain,
                'yearly_peak_to_peak': (float(yearly.max() - yearly.min())
                                       if yearly is not None else np.nan),
                'trend_rate': float(rates['rate_mdeg_per_year'].mean()),
                'mae_val': float((out['y'] - out['yhat']).abs().mean()),
            })
    return pd.DataFrame(rows, columns=['set', 'fold', gain_column,
                                       'yearly_peak_to_peak', 'trend_rate',
                                       'mae_val'])


def ladder_frame(frame, sensor, start, twall_tau_h, twall_lead_h,
                 radiation_delay_h, freq, weight_curve=None,
                 twall_column='twall_str', sr_column='sr_str'):
    """
    The current-era window plus the wall probe and the on-structure
    pyranometer, ready for the channel ladder (D4).

    Wall temperature and the on-structure pyranometer exist only from the
    instrument change at ``start`` onward, so neither ever enters the main
    line's three regressor sets; this assembles the matched window and the
    handful of derived columns the ladder's rungs need in one call, rather
    than repeating the same reindex-and-filter steps in the notebook. The
    on-structure radiation is put through the same pure transport delay
    :func:`shmlib.proxies.build_regressor_sets` already applies to every
    radiation source in the main line (D3), so a rung that swaps ``frame``'s
    borrowed station radiation for the wall's own pyranometer differs from
    the rung below it in source alone, never in treatment.

    Parameters
    ----------
    frame : pd.DataFrame
        The study's main-line record: ``y`` plus every regressor set's
        ``<role>_<set>`` columns, as returned by
        :func:`shmlib.proxies.build_regressor_sets`. Its index is zone-naive
        UTC clock time, the same convention ``sensor`` is expected to share.
    sensor : pd.DataFrame
        The on-structure sensor package joined across instrument eras, as
        returned by :func:`shmlib.proxies.join_eras`, carrying at least
        ``twall_column`` and ``sr_column``.
    start : str or pd.Timestamp
        First instant of the ladder's window. ``frame`` is cut to
        ``frame.loc[start:]`` before any derived column is computed, so
        every rung fits on exactly this window.
    twall_tau_h : float
        Thermal time constant applied to the wall probe at the ladder's
        deployable rung, in hours, passed to
        :func:`shmlib.coupling.thermal_operator` as ``tau``.
    twall_lead_h : float
        The wall probe's measured lead over the deformation, in hours,
        negative when the probe leads. Used only to build the diagnostic
        rung's ``twall_lead`` column; this function does not clamp a
        positive value to zero; a caller that reports a lag rather than a
        lead is expected to say so, since a leading probe is the entire
        reason that rung exists as a diagnostic rather than a candidate.
    radiation_delay_h : float
        Transport delay applied to the on-structure radiation, in hours;
        normally the same value the main line applies to every radiation
        source (D3), so that the ladder's radiation rung differs from the
        main line's `'str'` set in source alone.
    freq : str
        Sampling frequency of ``frame`` and ``sensor``, used to convert
        ``radiation_delay_h`` and ``twall_lead_h`` from hours into grid
        slots.
    weight_curve : dict or None, optional
        Passed to :func:`seasonal_weights` as ``modulation``. Default
        ``None``, the cosine fallback.
    twall_column, sr_column : str, optional
        Columns of ``sensor`` holding the wall probe and the pyranometer.
        Defaults ``'twall_str'`` and ``'sr_str'``, :func:`shmlib.proxies.\
load_sensor_forcings`'s own names for those quantities.

    Returns
    -------
    pd.DataFrame
        ``frame.loc[start:]`` with six columns added: ``twall`` (the probe,
        reindexed as given), ``sr_wall`` (the pyranometer, delayed),
        ``twall_tau`` (the probe through the thermal-inertia filter),
        ``twall_lead`` (the probe at its measured lead), ``summer_w`` and
        ``winter_w`` (the conditional-seasonality weights).
    """
    current = frame.loc[start:].copy()
    step_hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)
    slots_per_hour = 1.0 / step_hours
    delay_slots = int(round(radiation_delay_h / step_hours))

    current['twall'] = sensor[twall_column].reindex(current.index)
    current['sr_wall'] = coupling.thermal_operator(
        sensor[sr_column].reindex(current.index), delay=delay_slots, tau=0.0,
        dt_hours=step_hours)
    current['twall_tau'] = coupling.thermal_operator(
        current['twall'], delay=0, tau=twall_tau_h, dt_hours=step_hours)
    current['twall_lead'] = current['twall'].shift(
        int(round(twall_lead_h * slots_per_hour)))

    weights = seasonal_weights(current.index, modulation=weight_curve)
    return pd.concat([current, weights], axis=1)


def channel_ladder(current, rungs, valid_p, n_changepoints, block_hours,
                   repetitions, seed, **model_kwargs):
    """
    Refit one specification rung by rung on a matched window (D4).

    Each rung adds one on-structure channel to the specification before it,
    fitted on the same held-out split and the same window as every other
    rung, so an improvement in held-out error is attributable to what the
    rung added rather than to a different fit window or regressor count.
    The ladder's last rung conventionally uses the wall probe's own future
    values (a negative ``twall_lead_h`` in :func:`ladder_frame`) to bound
    what the probe could buy under a lead nobody can exploit at prediction
    time; this function scores it exactly like every other rung, on the
    understanding that the caller reports it as a diagnostic ceiling, never
    as a candidate specification.

    Parameters
    ----------
    current : pd.DataFrame
        The ladder's window, as returned by :func:`ladder_frame`, carrying
        ``y`` and every column any rung names.
    rungs : sequence of (str, sequence of str, gate)
        ``(label, columns, gate)`` triples, ``gate`` a column name, a
        sequence of column names, or ``None``, in the order they are fitted
        and reported. ``columns`` are the regressor columns of that rung's
        specification. ``gate`` is the column, or sequence of columns, whose
        presence further restricts the rung's matched window beyond
        ``columns``' own coverage; every column named in ``gate`` must be
        non-missing for a row to enter the rung's block, whether or not that
        column is also one of ``columns`` — a rung with no channel of its
        own to gate on can still be held to a window another rung's gate
        defines, which is exactly how every rung of D4's ladder is matched
        to one common window rather than each to its own regressors' own
        coverage. ``None`` applies no restriction beyond ``columns``.
    valid_p : float
        Fraction of each rung's matched block held out at its tail for
        scoring, mirroring the main line's own held-out split.
    n_changepoints : int
        Passed to :func:`covered_changepoints` for each rung's training
        head, and to :func:`neuralprophet_backtest` as ``n_changepoints``.
    block_hours, repetitions : int or float
        Passed to :func:`paired_mae_skill` as ``block_hours`` and
        ``repetitions``.
    seed : int
        Passed to every rung's :func:`neuralprophet_backtest` call and to
        :func:`paired_mae_skill`.
    **model_kwargs
        Passed unchanged to every rung's :func:`neuralprophet_backtest` call
        beside ``regressors``, ``task``, ``changepoints``, ``n_changepoints``
        and ``seed``, which this function already supplies.

    Returns
    -------
    ladder : pd.DataFrame
        One row per rung, in the order ``rungs`` was given: ``rung``,
        ``rows`` (the matched block's length), ``mae_val`` (held-out mean
        absolute error), ``coverage`` (the fraction of held-out rows inside
        ``[q05, q95]``, ``NaN`` when the fit carries no interval),
        ``gain_last`` (the last-named regressor's learned gain),
        ``residual_r1`` (lag-1 autocorrelation of the fit's own residual,
        computed on its training window), ``skill``, ``skill_q05``,
        ``skill_q95`` (:func:`paired_mae_skill`'s paired block-bootstrap
        skill of this rung's held-out error over the rung immediately
        below's, evaluated on their shared timestamps; the first rung has no
        rung below it, so its three skill columns are ``NaN``), and
        ``skill_vs_first``, ``skill_vs_first_q05``, ``skill_vs_first_q95``
        (the same paired skill, but always against the first rung rather
        than the one immediately below, so a rung's improvement over the
        ladder's starting specification can be read off directly rather
        than compounded from the chain; ``NaN`` on the first rung itself).
    errors : dict
        Rung label to its held-out absolute-error series, indexed by
        prediction timestamp, so a caller can re-inspect a pairing the
        ``skill`` columns already summarised.
    """
    ladder_rows = []
    errors = {}
    for label, columns, gate in rungs:
        columns = list(columns)
        block = current.dropna(subset=['y'] + columns)
        if gate is not None:
            gate_columns = [gate] if isinstance(gate, str) else list(gate)
            block = block.dropna(subset=gate_columns)
        split = int(len(block) * (1 - valid_p))
        train, valid = block.iloc[:split], block.iloc[split:]
        changepoints = covered_changepoints(train.index, n_changepoints)
        model, out = neuralprophet_backtest(
            train, valid, regressors=tuple(columns), task='nowcast',
            changepoints=changepoints, n_changepoints=n_changepoints,
            seed=seed, **model_kwargs)
        components = decompose_components(model, train, regressors=tuple(columns))
        gains = regressor_gains(components, train, tuple(columns))
        abs_error = (out['y'] - out['yhat']).abs()
        abs_error.index = pd.DatetimeIndex(out['ds'])
        errors[label] = abs_error
        coverage = (float(((out['y'] >= out['q05']) & (out['y'] <= out['q95'])).mean())
                   if 'q05' in out else np.nan)
        ladder_rows.append({
            'rung': label, 'rows': len(block), 'mae_val': float(abs_error.mean()),
            'coverage': coverage, 'gain_last': float(gains['gain'].iloc[-1]),
            'residual_r1': float(components['residual'].autocorr(1)),
        })

    ladder = pd.DataFrame(ladder_rows)
    labels = [label for label, _, _ in rungs]
    skills = [{'skill': np.nan, 'skill_q05': np.nan, 'skill_q95': np.nan}]
    for below, above in zip(labels[:-1], labels[1:]):
        shared = errors[below].index.intersection(errors[above].index)
        result = paired_mae_skill(
            errors[below].loc[shared], errors[above].loc[shared],
            block_hours=block_hours, repetitions=repetitions, seed=seed)
        skills.append({'skill': result['skill'], 'skill_q05': result['skill_q05'],
                       'skill_q95': result['skill_q95']})

    first_label = labels[0]
    skills_vs_first = [{'skill_vs_first': np.nan, 'skill_vs_first_q05': np.nan,
                        'skill_vs_first_q95': np.nan}]
    for label in labels[1:]:
        shared = errors[first_label].index.intersection(errors[label].index)
        result = paired_mae_skill(
            errors[first_label].loc[shared], errors[label].loc[shared],
            block_hours=block_hours, repetitions=repetitions, seed=seed)
        skills_vs_first.append({
            'skill_vs_first': result['skill'],
            'skill_vs_first_q05': result['skill_q05'],
            'skill_vs_first_q95': result['skill_q95'],
        })

    ladder = pd.concat(
        [ladder, pd.DataFrame(skills), pd.DataFrame(skills_vs_first)], axis=1)
    return ladder, errors


def lagged_regressor_weights(model, physical=False):
    """
    The weight a fitted model attaches to every past lag of every lagged
    regressor, one row per lag.

    Read straight from ``model.model.get_covar_weights()``, the dict a
    fitted NeuralProphet model exposes from lagged-regressor name to a
    tensor of shape ``(n_forecasts, n_lags)``. Only the first forecast row
    is read — Model B is always a one-step nowcast, so ``n_forecasts`` is
    always 1 — and NeuralProphet stores that row's lags oldest first, so
    its last entry is the most recent past sample; this function reverses
    that order into ``lag = 1`` for the most recent sample and
    ``lag = n_lags`` for the oldest, the convention every other quantity in
    this module (``lagged_n_lags``, the study's `LAGGED_N_LAGS` parameter)
    already uses. This is the same data
    ``model.plot_parameters(components=['lagged_regressors'])`` draws, read
    back as a table rather than a native plot.

    Parameters
    ----------
    model : neuralprophet.NeuralProphet
        A fitted model carrying at least one lagged regressor.
    physical : bool, optional
        The raw weight NeuralProphet stores acts on whatever normalised
        representation of ``y`` and of the regressor the model actually
        trained on — every column normalised by its own affine map
        ``value = shift + scale * value_norm``, with a per-column
        ``norm_type`` (``'standardize'``, ``'soft'``, ``'minmax'``, ...)
        that this wrapper's constructor does not fix, so the raw weight is
        in units of a standard deviation of ``y`` per unit of whatever the
        regressor's own normalisation happens to be, not in the regressor's
        physical units. When ``physical=True``, every regressor's weight is
        rescaled by ``scale_y / scale_regressor``, the two scales the
        fitted model recorded for itself in
        ``model.config_normalization.global_data_params`` (a dict of column
        name to a ``neuralprophet.df_utils.ShiftScale`` carrying ``.shift``
        and ``.scale``). Because every normalisation type NeuralProphet
        supports is one affine map per column, this rescaling is exact
        regardless of which type either column was actually given — it is
        not a re-derivation from the raw series (a ``'soft'`` column's
        ``scale`` is a quantile width, not a standard deviation, and
        recomputing ``std()`` from the data would silently use the wrong
        number). Default ``False``, which returns exactly the weight
        NeuralProphet's own tensor carries, unchanged from this function's
        original behaviour.

    Returns
    -------
    pd.DataFrame
        ``regressor``, ``lag`` (``1`` is the most recent past sample),
        ``weight`` — in the target's units per unit of the regressor when
        ``physical=True``, otherwise in the model's own normalised units —
        sorted by regressor and then by lag.
    """
    scales = model.config_normalization.global_data_params if physical else None
    scale_y = scales['y'].scale if physical else None
    rows = []
    for name, tensor in model.model.get_covar_weights().items():
        values = np.asarray(tensor.detach().cpu().numpy(), dtype=float)
        values = values.reshape(values.shape[0], -1)[0]
        n_lags = values.size
        factor = (scale_y / scales[name].scale) if physical else 1.0
        for i, weight in enumerate(values):
            rows.append({'regressor': name, 'lag': n_lags - i,
                        'weight': float(weight) * factor})
    out = pd.DataFrame(rows, columns=['regressor', 'lag', 'weight'])
    return out.sort_values(['regressor', 'lag']).reset_index(drop=True)


def impulse_response_summary(weights, dt_hours):
    """
    The effective delay and time constant a learned impulse response
    implies, read the same way Study 03 measured them by scanning.

    The gain is the response's total weight, the sum over every lag. The
    delay is the response's first moment in hours — the weighted mean lag,
    weighted by the weight itself — which is the discrete analogue of the
    transport delay :func:`shmlib.coupling.thermal_operator` applies as a
    whole-sample shift. The time constant is found by fitting the one-pole
    step response ``G * (1 - exp(-t / tau))`` to the impulse response's own
    cumulative sum by grid search over ``tau`` from 0.1 to 48 hours in the
    same two-stage resolution (0.1 h steps below 2 h, 0.5 h steps above) a
    coarse-to-fine scan would use, since the cumulative sum of a one-pole
    impulse response is exactly that step response. Reporting these two
    numbers beside Study 03's own delay-and-time-constant scan is the
    whole point of Model B (D9): the same physical quantities, this time
    read from what the model learned rather than imposed on it.

    Parameters
    ----------
    weights : pd.DataFrame
        ``regressor``, ``lag``, ``weight``, as returned by
        :func:`lagged_regressor_weights`.
    dt_hours : float
        Duration of one lag slot, in hours.

    Returns
    -------
    pd.DataFrame
        One row per regressor, with ``regressor``, ``gain`` (the summed
        weight), ``delay_h`` (the weight-weighted mean lag in hours),
        ``tau_h`` (the grid-searched one-pole time constant) and
        ``r2_onepole`` (how well that one-pole step response explains the
        response's own cumulative sum).
    """
    rows = []
    taus = np.concatenate([np.arange(0.1, 2.0, 0.1), np.arange(2.0, 48.5, 0.5)])
    for name, group in weights.groupby('regressor'):
        group = group.sort_values('lag')
        w = group['weight'].to_numpy(dtype=float)
        t = group['lag'].to_numpy(dtype=float) * float(dt_hours)
        gain = float(w.sum())
        delay = float((w * t).sum() / w.sum()) if w.sum() != 0 else np.nan
        cumulative = np.cumsum(w)
        best_tau, best_sse = np.nan, np.inf
        for tau in taus:
            fitted = gain * (1.0 - np.exp(-t / tau))
            sse = float(((cumulative - fitted) ** 2).sum())
            if sse < best_sse:
                best_tau, best_sse = float(tau), sse
        total = float(((cumulative - cumulative.mean()) ** 2).sum())
        rows.append({'regressor': name, 'gain': gain, 'delay_h': delay,
                    'tau_h': best_tau,
                    'r2_onepole': 1.0 - best_sse / total if total > 0 else np.nan})
    return pd.DataFrame(
        rows, columns=['regressor', 'gain', 'delay_h', 'tau_h', 'r2_onepole'])
