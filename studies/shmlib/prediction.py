"""
Prediction helpers for Study 04.

The functions in this module prepare and score prediction experiments without
owning the model itself. They keep the study's choices visible in the notebook:
which channels are required, how many complete segments are needed, which route
is eligible for a given task, and how large a paired bootstrap block should be.
"""

import re

import numpy as np
import pandas as pd


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


def _score_group(group):
    """Compute scalar scores for one already selected group."""
    data = group[['y', 'yhat']].apply(pd.to_numeric, errors='coerce').dropna()
    row = {
        'n': int(len(data)),
        'mae': np.nan,
        'rmse': np.nan,
        'bias': np.nan,
        'r2': np.nan,
    }
    if len(data):
        error = data['yhat'] - data['y']
        row['mae'] = float(error.abs().mean())
        row['rmse'] = float(np.sqrt(np.mean(error ** 2)))
        row['bias'] = float(error.mean())
        if len(data) > 1:
            total = float(((data['y'] - data['y'].mean()) ** 2).sum())
            if total > 0:
                residual = float((error ** 2).sum())
                row['r2'] = 1.0 - residual / total

    if {'q05', 'q95'}.issubset(group.columns):
        interval = group[['y', 'q05', 'q95']].apply(
            pd.to_numeric, errors='coerce').dropna()
        if len(interval):
            covered = ((interval['y'] >= interval['q05'])
                       & (interval['y'] <= interval['q95']))
            row['coverage_q05_q95'] = float(covered.mean())
            row['width_q05_q95'] = float((interval['q95']
                                          - interval['q05']).median())
        else:
            row['coverage_q05_q95'] = np.nan
            row['width_q05_q95'] = np.nan
    return row


def score_predictions(frame, group_cols):
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

    Returns
    -------
    pd.DataFrame
        One row per group with ``n``, ``mae``, ``rmse``, ``bias``, ``r2`` and,
        when interval columns are present, coverage and width.
    """
    group_cols = list(group_cols or [])
    if not group_cols:
        return pd.DataFrame([_score_group(frame)])

    rows = []
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(_score_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


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
    """Convert NeuralProphet's wide prediction table to study-long format."""
    test_ds = set(pd.DatetimeIndex(test_index))
    rows = []
    quantiles = list(quantiles or [])
    lower = min(quantiles) if quantiles else None
    upper = max(quantiles) if quantiles else None
    for _, record in predictions.iterrows():
        ds = pd.Timestamp(record['ds'])
        if ds not in test_ds:
            continue
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
                           decompose=False):
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

    Returns
    -------
    model, pd.DataFrame
        Fitted NeuralProphet model and long prediction table.
    """
    if task not in {'forecast', 'nowcast'}:
        raise ValueError("task must be 'forecast' or 'nowcast'")

    from neuralprophet import NeuralProphet, set_random_seed

    set_random_seed(seed)
    np.random.seed(seed)

    regressors = tuple(regressors or ())
    if task == 'nowcast':
        n_lags = 0
        n_forecasts = 1

    model = NeuralProphet(
        growth=growth,
        changepoints=(list(pd.DatetimeIndex(changepoints))
                      if changepoints is not None else None),
        n_changepoints=int(n_changepoints),
        n_lags=int(n_lags),
        n_forecasts=int(n_forecasts),
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=yearly,
        normalize='standardize',
        global_normalization=True,
        global_time_normalization=True,
        unknown_data_normalization=True,
        impute_missing=False,
        drop_missing=False,
        loss_func='SmoothL1Loss',
        learning_rate=0.01,
        epochs=int(epochs),
        quantiles=list(quantiles or ()),
        collect_metrics=False,
    )
    for regressor in regressors:
        if task == 'nowcast':
            model.add_future_regressor(regressor)
        else:
            model.add_lagged_regressor(regressor, n_lags=int(regressor_lags))

    train_df = _model_frame(train, regressors)
    test_df = _model_frame(test, regressors)
    fit_freq = freq if freq is not None else _analysis_freq(train.index)
    model.fit(train_df, freq=fit_freq, progress='none', minimal=True)

    segmented_test = 'segment_id' in test.columns
    predict_df = test_df if task == 'nowcast' or segmented_test else pd.concat(
        [train_df.tail(max(int(n_lags), int(regressor_lags))), test_df],
        ignore_index=True)
    wide = model.predict(predict_df, decompose=bool(decompose))
    keep_horizons = list(horizons or range(1, int(n_forecasts) + 1))
    long = _long_predictions(
        wide, test.index, 'segment_id' in test.columns, keep_horizons,
        quantiles)
    return model, long


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
        value, prediction, and optional interval columns.
    """
    model_frame = _model_frame(frame, regressors)
    original_y = pd.to_numeric(
        frame['y'], errors='coerce') if 'y' in frame.columns else pd.Series(
            np.nan, index=frame.index, dtype=float)
    if getattr(model, 'n_lags', None) == 0:
        # NeuralProphet 0.9.0 crashes while restoring trailing missing y even
        # though a zero-lag model does not consume the target at prediction.
        model_frame['y'] = model_frame['y'].fillna(0.0)
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


def decompose_components(model, frame, regressors=(), freq=None):
    """
    The additive parts NeuralProphet fitted, aligned to the study's index.

    NeuralProphet returns its decomposition as extra columns beside the
    prediction. This reshapes them into one timestamp-indexed table, adds the
    observed value and the residual, and leaves the component names as the model
    produced them, so that a reader can trace any column back to the term that
    made it.

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
    model_frame = _model_frame(frame, regressors)
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
    out = out.set_index('ds')
    out.index.name = frame.index.name

    observed = pd.to_numeric(frame['y'], errors='coerce') if 'y' in frame else None
    out['y'] = observed.reindex(out.index) if observed is not None else np.nan
    out['residual'] = out['y'] - out['yhat1']
    return out


_NON_COMPONENT_COLUMNS = ('y', 'yhat1', 'ID')


def component_variance_shares(components, columns=None):
    """
    What share of the fitted variation each additive component carries.

    This is the decomposition's headline claim expressed as a number: a record
    whose daily component carries most of the variance is a thermometer, and one
    whose trend does is a structure that is moving. The residual is included as
    a component so that the shares are comparable and sum to one.

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
    if columns is None:
        columns = [c for c in components.columns
                   if c not in _NON_COMPONENT_COLUMNS]
    frame = components.loc[:, list(columns)].apply(
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
        dropped before testing.
    lags : sequence of int, optional
        Ljung-Box lags to test. Default ``(1, 24, 72)``.

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
    result = acorr_ljungbox(values, lags=lags, return_df=True)

    scale = {
        'n': int(values.size),
        'std': float(values.std(ddof=1)) if values.size > 1 else np.nan,
        'mad': float((values - values.median()).abs().median()),
    }
    return pd.DataFrame({
        'lag': lags,
        'lb_stat': result['lb_stat'].to_numpy(),
        'lb_pvalue': result['lb_pvalue'].to_numpy(),
        'n': scale['n'],
        'std': scale['std'],
        'mad': scale['mad'],
    })
