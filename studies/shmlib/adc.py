"""
Module: shmlib.adc

The raw ``.adc`` archive: its file format, the constants of the instrument that
writes it, and the manufacturer's temperature compensation.

This module is the project's single implementation of the file-format contract
specified in ``docs/raw-data-format.md``. That document is binding on everything
here, and where the two disagree the document wins.

Both parsers are **pure**: they read a file and return what it contained. They
apply no cleaning, reject no sentinel and fill no gap. That is deliberate, and it
is what allows the study that documents the archive to open the record before any
verdict has been passed on it. Deciding what in the parsed record is a
measurement belongs to the study, not to the parser.

Two eras share the archive, and they are told apart by the field count of each
individual record rather than by the date of the file that carries it. Files
bleed records into the previous calendar day, and a file written across the
changeover carries both layouts, so a parser that trusted the filename would
mangle exactly the records that matter most.
"""

import os
import re
import shutil

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Format constants — see docs/raw-data-format.md
# ──────────────────────────────────────────────────────────────────────

#: Number of fields in a current-era record (2 timestamp + 18 measurement).
N_COLS_CURRENT = 20

#: Number of fields in a legacy-era record (2 timestamp + 12 measurement).
N_COLS_LEGACY = 14

#: Date on which the 20-column instrument package replaced the legacy network.
CHANGEOVER = pd.Timestamp('2025-02-21')

#: Current-era column names, in file order. Fields 3-14 are the three legacy
#: blocks, which are identically zero in this era.
CURRENT_COLUMNS = [
    'b1_batt', 'b1_tair', 'b1_rh', 'b1_i',
    'b2_batt', 'b2_tair', 'b2_rh', 'b2_i',
    'b3_batt', 'b3_tair', 'b3_rh', 'b3_i',
    'batt', 'tair', 'rh', 'i_mv', 'sr', 'twall',
]

#: Legacy column names in file order. Blocks appear in station order:
#: b1 = st01, b2 = st02, b3 = st03 (``docs/raw-data-format.md`` Section 3.3).
LEGACY_COLUMNS = [
    'st01_batt', 'st01_tair', 'st01_rh', 'st01_i',
    'st02_batt', 'st02_tair', 'st02_rh', 'st02_i',
    'st03_batt', 'st03_tair', 'st03_rh', 'st03_i',
]

#: Channels in which ``0.000`` marks an unavailable channel rather than a
#: reading. Solar radiation is deliberately absent: its night-time zeros are
#: real measurements.
ZERO_IS_SENTINEL = ['batt', 'tair', 'rh', 'i_mv']

#: Nominal sampling interval of the raw archive.
SAMPLING = '20min'


# ──────────────────────────────────────────────────────────────────────
# Instrument constants
# ──────────────────────────────────────────────────────────────────────

#: Conditioner output zero point, in the millivolt units the logger records.
#: Sensitivity is 1000 mV per degree, so millivolts and millidegrees share a
#: scale and only this offset separates the logged number from a true angle.
INCLINOMETER_ZERO_MV = 2500.0

#: Full-scale range of the calibrated inclinometer, in millidegrees.
INCLINOMETER_RANGE_MDEG = 2000.0

#: Wall-temperature probe failure sentinel, in °C. The bottom of the sensor
#: range, emitted as an open-circuit indicator.
TWALL_SENTINEL = -55.0

#: Physical ceiling for incident solar radiation at this latitude, in W/m².
#: Records above this are the unsigned wrap-around artefact, not measurements.
SR_MAX_PHYSICAL = 1400.0

#: The compensation coefficient documented by the manufacturer and hard-coded in
#: Notebook 00, in mdeg · °C⁻¹ · 10⁻³. Multiplied by 1000 it becomes 5 mdeg per
#: °C. It is an instrument calibration rather than a fitted parameter, and no
#: study re-estimates it.
DOCUMENTED_COEFF = 0.005


# ──────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────

def to_float(token):
    """
    Parse one numeric field, accepting either decimal separator.

    The archive mixes ``3.500`` and ``3,500`` forms, sometimes within a single
    file, so the separator is normalised per field rather than per file.

    Parameters
    ----------
    token : str
        Raw field text.

    Returns
    -------
    float
        Parsed value, or ``nan`` if the field is not numeric.
    """
    try:
        return float(token.replace(',', '.'))
    except (ValueError, AttributeError):
        return np.nan


def _parse(path, n_cols, columns):
    """
    Parse the records of one era out of one ``.adc`` file.

    Shared body of :func:`parse_file` and :func:`parse_legacy_file`. A record is
    accepted only if its own field count matches ``n_cols``, so a file holding
    both layouts yields each era to the parser that asked for it and neither
    parser sees the other's records.

    Parameters
    ----------
    path : str
        Path to a ``.adc`` file.
    n_cols : int
        Field count identifying the era.
    columns : list of str
        Column names of that era, in file order.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, with ``columns``. Empty if the file holds no
        record of this era.
    """
    rows, stamps = [], []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != n_cols:
                continue
            try:
                ts = pd.to_datetime(parts[0] + ' ' + parts[1],
                                    format='%d/%m/%y %H:%M:%S')
            except ValueError:
                continue
            stamps.append(ts)
            rows.append([to_float(p) for p in parts[2:]])

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(stamps), columns=columns)


def parse_file(path):
    """
    Parse the current-era records of one ``.adc`` file.

    Handles the mixed decimal separator, takes the date from the record rather
    than from the filename (files bleed one record into the previous day), and
    branches on the record's own field count so that a file whose content
    disagrees with its expected era is skipped rather than mangled. No cleaning
    is applied: sentinels arrive as the numbers they were written as.

    Parameters
    ----------
    path : str
        Path to a ``.adc`` file.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, columns as :data:`CURRENT_COLUMNS`. Empty if the
        file holds no current-era records.
    """
    return _parse(path, N_COLS_CURRENT, CURRENT_COLUMNS)


def parse_legacy_file(path):
    """
    Parse the legacy-era records of one ``.adc`` file.

    Differs from :func:`parse_file` only in the expected field count and the
    column names. The defects concentrated in this era — duplicate timestamps
    whose copies disagree, decimal separators that change within a single file,
    and records belonging to the previous calendar day — are all tolerated here
    and resolved by the caller, which is the only place that knows what to do
    about them.

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
    return _parse(path, N_COLS_LEGACY, LEGACY_COLUMNS)


def sync_cache(archive_dir, cache_dir, start, end, verbose=True,
               progress=False):
    """
    Copy the ``.adc`` files covering a date range to a local cache.

    The archive is served by Google Drive streaming at roughly ten files per
    minute, which makes repeated in-place reads impractical. This copies each
    required file once; subsequent runs reuse the cache. The archive itself is
    only ever read.

    Parameters
    ----------
    archive_dir : str
        Path to the read-only ``.adc`` archive.
    cache_dir : str
        Local directory for the working copy. Created if absent.
    start, end : str or pd.Timestamp
        Inclusive date bounds.
    verbose : bool, optional
        Print a one-line summary. Default ``True``.
    progress : bool, optional
        Draw a progress bar over the files that still have to be copied.
        Default ``False``. Worth enabling on a cold cache: copying is the one
        slow step here, while a warm cache returns immediately and draws no bar
        at all.

    Returns
    -------
    list of str
        Absolute paths of the cached files, sorted by date.

    Notes
    -----
    Writes into ``cache_dir``. Never writes to ``archive_dir``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    wanted = []
    for name in sorted(os.listdir(archive_dir)):
        m = re.fullmatch(r'GUBBIO_(\d{8})\.adc', name)
        if not m:
            continue
        day = pd.Timestamp(m.group(1))
        if start <= day <= end:
            wanted.append(name)

    missing = [n for n in wanted
               if not os.path.exists(os.path.join(cache_dir, n))]
    copied = len(missing)

    pending = missing
    if progress and missing:
        # Imported here rather than at module scope so that a study without
        # tqdm installed still runs.
        from tqdm.auto import tqdm
        pending = tqdm(missing, desc='  caching', unit='file')

    for name in pending:
        shutil.copy2(os.path.join(archive_dir, name),
                     os.path.join(cache_dir, name))

    if verbose:
        print(f'  cache: {len(wanted)} files in range, {copied} newly copied, '
              f'{len(wanted) - copied} already present')
    return [os.path.join(cache_dir, n) for n in wanted]


# ──────────────────────────────────────────────────────────────────────
# Compensation
# ──────────────────────────────────────────────────────────────────────

def compensate(df, temp_col='tair', coeff=DOCUMENTED_COEFF, normalise=True):
    """
    Apply the documented temperature compensation to the inclination.

    The correction is ``inc - (T - T_ref) * coeff * 1000`` with ``T_ref`` taken
    from the first record carrying both a signal and a temperature, followed
    optionally by a shift so that the series starts at zero. This is the
    manufacturer's formula as Notebook 00 implements it, and it is an instrument
    calibration rather than a regression: the coefficient is never swept here.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inc`` and ``temp_col``.
    temp_col : str, optional
        Temperature channel driving the correction. Default ``'tair'``.
    coeff : float, optional
        Coefficient in mdeg · °C⁻¹ · 10⁻³. Default :data:`DOCUMENTED_COEFF`.
    normalise : bool, optional
        Subtract the first valid value, so the series starts at zero. Default
        ``True``.

    Returns
    -------
    pd.Series
        Compensated signal named ``inc_comp``, on the index of ``df``.

    Notes
    -----
    Anything spanning both instrument eras must be anchored **once**, as a
    single series. Anchoring each era separately and concatenating plants a step
    that is arithmetic rather than instrumentation
    (``docs/raw-data-format.md`` Section 7.3).
    """
    valid = df[['inc', temp_col]].dropna()
    if valid.empty:
        return pd.Series(np.nan, index=df.index, name='inc_comp')
    t_ref = valid[temp_col].iloc[0]
    out = df['inc'] - (df[temp_col] - t_ref) * coeff * 1000.0
    if normalise:
        first = out.dropna()
        if len(first):
            out = out - first.iloc[0]
    return out.rename('inc_comp')
