"""
Module: shmlib.meteo

Circular statistics for directional meteorological data — wind direction and
any other quantity measured on a compass rather than on a line.

An angle wraps at 360 degrees, so the ordinary arithmetic mean is the wrong
tool for it: the mean of a reading of 350 degrees and one of 10 degrees is 180
by ordinary arithmetic, which points exactly backwards from a direction the two
readings in fact agree on to within 20 degrees. Every function here works
instead on the unit vector each angle represents — averaging its sine and
cosine separately and recovering the direction with ``arctan2`` — which has no
branch cut and therefore no direction at which it breaks.

Inputs and outputs are angles in degrees by default, the meteorological
convention of a compass bearing measured clockwise from north, with a
``degrees`` flag on every function for a caller who instead wants radians.
``NaN`` is treated as a missing reading throughout: it is dropped before an
average is taken, never propagated into one.
"""

import numpy as np
import pandas as pd


#: Floor imposed on the mean resultant length before its logarithm is taken in
#: :func:`circular_std`, so that a perfectly uniform sample returns a large
#: finite circular standard deviation rather than raising a divide-by-zero
#: warning at ``log(0)``.
_RESULTANT_FLOOR = np.finfo(float).eps


def circular_mean(values, degrees=True):
    """
    Mean direction of a sequence of angles, by the unit-vector method.

    The ordinary arithmetic mean of two angles that straddle the wrap point is
    meaningless — the mean of 350 degrees and 10 degrees is 180 by ordinary
    arithmetic, which points exactly backwards from a direction the two
    readings actually agree on to within 20 degrees. This averages the sine
    and the cosine of each angle separately and recovers the direction of the
    resulting vector with ``arctan2``, which is defined everywhere on the
    circle and so cannot break at any particular direction.

    Parameters
    ----------
    values : array-like
        Angles to average. May contain ``NaN``, which is dropped before the
        average is taken.
    degrees : bool, optional
        If ``True`` (the default), `values` are in degrees and the result is
        returned in degrees; if ``False``, both are in radians.

    Returns
    -------
    float
        Mean direction, in ``[0, 360)`` when `degrees` is ``True`` or in
        ``[0, 2*pi)`` when `degrees` is ``False``. ``NaN`` when `values` is
        empty or holds only ``NaN``.
    """
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan

    angles = np.radians(valid) if degrees else valid
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    full_circle = 360.0 if degrees else 2.0 * np.pi
    result = float(np.degrees(mean_angle) if degrees else mean_angle) % full_circle
    # A mean direction that is mathematically exactly 0 can arrive as a tiny
    # negative float (rounding in the sine/cosine average), which the modulo
    # above then rounds up to the full circle itself rather than to 0 — the
    # one value the ``[0, 360)`` contract must not return.
    return 0.0 if result == full_circle else result


def circular_resample(series, freq, min_count=1):
    """
    Resample a Series of angles onto a new frequency by the circular mean.

    Parameters
    ----------
    series : pd.Series
        Angles indexed by a ``DatetimeIndex``, in degrees. May contain ``NaN``.
    freq : str
        Pandas offset alias for the target frequency, e.g. ``'1h'``.
    min_count : int, optional
        Minimum number of non-``NaN`` values a bin must hold to receive a
        result. A bin with fewer becomes ``NaN`` even though
        :func:`circular_mean` could still average what little it has, because
        a bin's mean should say something about the bin rather than about one
        stray reading inside it. Default 1.

    Returns
    -------
    pd.Series
        Circular mean per bin, indexed on the resampled grid. ``NaN`` where a
        bin holds fewer than `min_count` non-``NaN`` values.
    """
    resampler = series.resample(freq)
    means = resampler.apply(
        lambda bin_values: circular_mean(bin_values.to_numpy(dtype=float)))
    counts = resampler.count()
    return means.where(counts >= min_count)


def circular_difference(a, b, degrees=True):
    """
    Signed smallest-angle difference ``a - b``, wrapped to ``(-180, 180]``.

    Ordinary subtraction of two directions can overshoot by a full turn when
    they straddle the wrap point: 10 degrees minus 350 degrees is -340 by
    ordinary arithmetic, when the two directions are in fact only 20 degrees
    apart. This returns the signed difference along the shorter arc between
    them instead.

    Parameters
    ----------
    a, b : scalar, array-like or pd.Series
        Angles to compare. Any combination that ordinary subtraction accepts;
        the wrap is applied elementwise.
    degrees : bool, optional
        If ``True`` (the default), `a` and `b` are in degrees and the result
        is wrapped to ``(-180, 180]``; if ``False``, both are in radians and
        the result is wrapped to ``(-pi, pi]``.

    Returns
    -------
    scalar, array-like or pd.Series
        The signed smallest-angle difference, of the same shape ``a - b``
        would produce.
    """
    diff = a - b
    full_circle = 360.0 if degrees else 2.0 * np.pi
    half_circle = full_circle / 2.0
    return half_circle - ((half_circle - diff) % full_circle)


def circular_std(values, degrees=True):
    """
    Circular standard deviation, from the resultant length of the sample.

    Computed as ``sqrt(-2 * ln(R))``, where ``R`` is the length of the mean
    unit vector: 1 for a sample with no spread at all, falling towards 0 as
    the sample spreads towards uniform over the circle. This is the standard
    directional-statistics analogue of the linear standard deviation, and
    unlike the linear one it has no fixed relationship to the spread in
    degrees near ``R = 0`` — it grows without bound as the sample approaches a
    uniform spread rather than saturating at some largest possible value.

    Parameters
    ----------
    values : array-like
        Angles to summarise. May contain ``NaN``, which is dropped before the
        statistic is computed.
    degrees : bool, optional
        If ``True`` (the default), `values` are in degrees and the result is
        returned in degrees; if ``False``, both are in radians.

    Returns
    -------
    float
        Circular standard deviation. ``NaN`` when `values` is empty or holds
        only ``NaN``.
    """
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan

    angles = np.radians(valid) if degrees else valid
    resultant_length = np.hypot(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    resultant_length = max(resultant_length, _RESULTANT_FLOOR)

    std_rad = np.sqrt(-2.0 * np.log(resultant_length))
    return float(np.degrees(std_rad) if degrees else std_rad)


def wind_components(direction, speed=None, degrees=True, prefix=None):
    """
    A direction as its two Cartesian components, for a linear statistic.

    A correlation, a regression or a lag scan on a compass bearing is
    meaningless: the number 359 sits at the far end of the scale from the
    number 1, while the directions they name are one degree apart, so every
    linear statistic computed on degrees is dominated by an artefact of where
    the scale was cut. Splitting the direction into the sine and cosine of its
    angle removes the cut — the pair varies smoothly all the way round the
    circle — at the cost of turning one channel into two, each of which has to
    be screened and reported separately.

    Where a speed is given the components are scaled by it, which is the
    ordinary meteorological wind vector: the pair then carries how hard the
    wind blew as well as where it came from, and a calm hour contributes
    nothing rather than contributing a direction the anemometer could not
    resolve.

    Parameters
    ----------
    direction : pd.Series
        Bearings, by convention measured clockwise from north.
    speed : pd.Series or None, optional
        Wind speed on the same index. Default ``None``, which returns the unit
        vector.
    degrees : bool, optional
        If ``True`` (the default), `direction` is in degrees; if ``False``, in
        radians.
    prefix : str or None, optional
        Stem of the two output column names. Default ``None``, which uses the
        name of `direction`, or ``'wdir'`` when it has none.

    Returns
    -------
    pd.DataFrame
        Two columns, ``{prefix}_sin`` and ``{prefix}_cos``, on the index of
        `direction`. ``NaN`` propagates from either input.
    """
    if prefix is None:
        prefix = direction.name if direction.name is not None else 'wdir'

    angles = np.radians(direction.astype(float)) if degrees else direction.astype(float)
    sin_component = np.sin(angles)
    cos_component = np.cos(angles)
    if speed is not None:
        sin_component = sin_component * speed.astype(float)
        cos_component = cos_component * speed.astype(float)

    return pd.DataFrame({f'{prefix}_sin': sin_component,
                         f'{prefix}_cos': cos_component},
                        index=direction.index)
