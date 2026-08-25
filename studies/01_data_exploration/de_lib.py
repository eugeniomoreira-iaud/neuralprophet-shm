"""
Module: de_lib.py

Support library for the data-exploration study
(``studies/01_data_exploration/``).

This study is the project's documentation of the archive, and a document that
describes a record cannot be handed that record already cleaned. Everything
here therefore reads the raw ``.adc`` files directly, on their **native
20-minute grid**, and keeps every value as it was written alongside the verdict
passed on it and the corrected value that verdict produces.

Three properties govern the design.

**Nothing is aggregated.** The hourly grid every other study works on exists so
that the pipeline can align against hourly external proxies. That is a
downstream requirement, not a property of the archive, and it belongs to the
study that prepares the proxy dataset. Here the record is reported at the rate
it was recorded.

**Nothing is silently discarded.** Each channel appears three times: as recorded,
as a flag naming what was decided about the value, and as the corrected value.
The recorded column is never modified, so a reader who disagrees with a decision
can recover the number behind it. A flag is either a *rejection*, which empties
the corrected column because nothing is known in the value's place, or a
*correction*, which fills it because the true value is known.

**The two instruments never share a column.** The package installed on
2025-02-21 replaced the legacy network and its inclinometer shares no baseline
with legacy block b2 (``docs/raw-data-format.md`` Section 3.3). Legacy block b2
is ``st02_*`` and the current package is ``n_*``, kept apart in the table itself
rather than by a convention someone downstream has to remember.

The file-format contract is not reimplemented here. Parsing comes from
:func:`shmlib.adc.parse_file` and :func:`shmlib.adc.parse_legacy_file`, which
handle the mixed decimal separator, take the date from the record rather than
from the filename, and branch on the record's own field count; both are pure
parsers that apply no cleaning, which is exactly what this study needs. The
documented thresholds and the compensation come from :mod:`shmlib.adc`, the
extent of the archive and the eras from :mod:`shmlib.site`, and the solar
geometry behind the night correction from :mod:`shmlib.solar`, which carries the
site coordinates and their provenance. The flag vocabulary this module writes
into every ``_flag`` column is likewise defined once, in :mod:`shmlib.quality`,
because study 2 reads the identical codes back out of this study's exported
archive; this module keeps a thin alias at each original name so nothing below
had to change when the vocabulary moved.

Nothing in this module writes to the raw archive.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

#: ``studies/``, the parent of every study folder, carries the shared library.
_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDIES = os.path.abspath(os.path.join(_HERE, '..'))
if _STUDIES not in sys.path:
    sys.path.insert(0, _STUDIES)

from shmlib import adc, quality, site, solar, viz                # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

#: First day of the archive.
ARCHIVE_START = site.ARCHIVE_START

#: Last day of the legacy 14-column era, inclusive.
LEGACY_END = site.LEGACY_END

#: First day of the current 20-column era.
CURRENT_START = site.CURRENT_START

#: The native sampling interval of the archive, and the grid this study works
#: on throughout. Nothing here resamples.
SAMPLING = adc.SAMPLING

#: The station the current-era package instruments, and the one every later
#: study analyses.
TARGET_STATION = site.TARGET_STATION

#: Legacy blocks, in file order: b1 = st01, b2 = st02, b3 = st03.
LEGACY_BLOCKS = ('st01', 'st02', 'st03')

#: The current-era package. Named for the block it occupies in the file rather
#: than for the station it stands at, because its inclinometer shares no
#: baseline with the legacy block at that station.
#:
#: Moved to ``shmlib.site``: study 2 needs the identical fact and ``shmlib``
#: must never be reached into from one study by way of another. Kept here as
#: an alias at the original name, so every call site in this module is
#: unaffected.
CURRENT_BLOCK = site.CURRENT_BLOCK

#: Every acquisition block written to the unified table.
BLOCKS = LEGACY_BLOCKS + (CURRENT_BLOCK,)

#: Channels each block carries, in file order.
BLOCK_CHANNELS = {
    'st01': ('batt', 'tair', 'rh', 'i_mv'),
    'st02': ('batt', 'tair', 'rh', 'i_mv'),
    'st03': ('batt', 'tair', 'rh', 'i_mv'),
    'n': ('batt', 'tair', 'rh', 'i_mv', 'sr', 'twall'),
}

#: Every column of the record as written, in file order.
RAW_COLUMNS = [f'{block}_{channel}'
               for block in BLOCKS
               for channel in BLOCK_CHANNELS[block]]

#: Channels in which ``0.000`` marks an unavailable channel rather than a
#: reading. Solar radiation is deliberately absent: its night-time zeros are
#: real measurements.
ZERO_IS_SENTINEL = ('batt', 'tair', 'rh', 'i_mv')

#: Rejection and correction codes written into the ``_flag`` columns, and the
#: two tuples that group them.
#:
#: Moved to ``shmlib.quality``: study 2 needs the identical vocabulary to read
#: this study's ``_flag`` columns, and ``shmlib`` must never be reached into
#: from one study by way of another. Kept here as aliases at the original
#: names, so every call site in this module is unaffected. The documentation
#: prose — what a rejection is, what a correction is, and why the
#: inclinometer's calibrated range is deliberately not a rejection criterion —
#: now lives with the definitions in ``shmlib.quality``.
FLAG_SENTINEL = quality.FLAG_SENTINEL
FLAG_WRAP = quality.FLAG_WRAP
FLAG_UNPHYSICAL = quality.FLAG_UNPHYSICAL
FLAG_NIGHT = quality.FLAG_NIGHT
REJECTION_CODES = quality.REJECTION_CODES
CORRECTION_CODES = quality.CORRECTION_CODES

#: Wall-temperature probe failure sentinel, in °C. The bottom of the sensor range,
#: emitted as an open-circuit indicator.
TWALL_SENTINEL = adc.TWALL_SENTINEL

#: Floor of the solar-radiation wrap-around, in W/m². The artefact of the raw-data
#: specification is the full scale of the field, 2**13 - 0.125 = 8191.875, so any
#: wrapped reading lands in the top half of the unsigned 13-bit range. The cut is
#: placed at half scale rather than on the exact value so that a differently
#: quantised wrap is still recognised as one, and it clears the largest reading
#: of the other, unexplained failure mode by a factor of nearly two.
SR_WRAP_MIN = 4096.0

#: Compensation coefficient, in mdeg · °C⁻¹ · 10⁻³.
COMP_COEFF = adc.DOCUMENTED_COEFF

#: The logger's civil clock, from :data:`shmlib.site.SITE_TZ`. Measured on this
#: archive rather than assumed: the midpoint of the radiation curve on clean days
#: sits +0.95 h from UTC in February and March and +2.11 h from April onwards,
#: stepping between the two on 2025-03-30, the last Sunday of March and the
#: European daylight-saving changeover. The logger therefore records Italian
#: civil time and does observe daylight saving. Every position within the day
#: reported by this study is in this clock.
SITE_TZ = site.SITE_TZ

#: Solar elevation below which no natural light reaches the sensor, and the
#: elevation above which a working sensor must report substantial radiation,
#: both in degrees.
#:
#: Moved to ``shmlib.solar``: study 2 needs the identical thresholds, and the
#: two studies must not disagree about where the sun was. Kept here as aliases
#: at the original names, so every call site in this module is unaffected. The
#: measured basis for each threshold now lives with the definitions in
#: ``shmlib.solar``.
NIGHT_ELEVATION = solar.NIGHT_ELEVATION
SUN_HIGH_ELEVATION = solar.SUN_HIGH_ELEVATION

#: A day is condemned when its high-sun radiation is not this many times its
#: dark radiation. The two populations are far apart: days that fail sit at a
#: median of 1.06 and never exceed 1.91, while days that pass sit at a median
#: near 2500.
SR_DAY_RATIO = 3.0

#: Samples required in each of the two windows before a day may be judged.
SR_DAY_MIN_SAMPLES = 6


# ──────────────────────────────────────────────────────────────────────
# Reading the archive
# ──────────────────────────────────────────────────────────────────────

def last_archive_day(archive_dir):
    """
    Date of the most recent ``.adc`` file present in the archive.

    Parameters
    ----------
    archive_dir : str
        Read-only archive directory.

    Returns
    -------
    str
        Date in ``YYYY-MM-DD`` form.
    """
    return site.last_archive_day(archive_dir)


def sync_archive(archive_dir, cache_dir, start, end, verbose=True):
    """
    Copy the archive files covering a date range to the local cache.

    A thin wrapper on :func:`shmlib.adc.sync_cache` with the progress bar enabled.
    Copying is the one slow step in this study: Google Drive streams the archive
    at roughly ten files per minute, so a cold cache takes hours while a warm
    one returns at once.

    Parameters
    ----------
    archive_dir : str
        Read-only ``.adc`` archive. Only ever read.
    cache_dir : str
        Local working copy directory. Created if absent.
    start, end : str or pd.Timestamp
        Inclusive date bounds.
    verbose : bool, optional
        Print a one-line summary. Default ``True``.

    Returns
    -------
    list of str
        Paths of the cached files, sorted by date.
    """
    return adc.sync_cache(archive_dir, cache_dir, start, end,
                         verbose=verbose, progress=True)


def scan_raw_text(paths, progress=True):
    """
    Count the file-level defects that survive no parser.

    A parser has to skip a malformed record to do its job, and once skipped the
    record is invisible to everything downstream. This pass reads the same files
    as text and counts what was skipped and why, so that the study can report
    the defects of the archive rather than only the defects of the values that
    made it through.

    Parameters
    ----------
    paths : list of str
        Cached ``.adc`` files.
    progress : bool, optional
        Draw a progress bar. Default ``True``.

    Returns
    -------
    dict
        ``files``, ``lines``, ``n_legacy`` and ``n_current`` records by field
        count, ``rejected`` lines matching neither layout, ``comma_fields``
        numeric fields written with a comma decimal separator, and
        ``mixed_separator_files``, the number of files in which both separators
        occur.
    """
    stats = {'files': len(paths), 'lines': 0, 'n_legacy': 0, 'n_current': 0,
             'rejected': 0, 'comma_fields': 0, 'mixed_separator_files': 0}

    iterator = paths
    if progress and paths:
        from tqdm.auto import tqdm
        iterator = tqdm(paths, desc='  scanning', unit='file')

    for path in iterator:
        saw_comma = saw_dot = False
        with open(path, 'r', errors='replace') as handle:
            for line in handle:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                stats['lines'] += 1
                fields = line.split('\t')
                if len(fields) == adc.N_COLS_CURRENT:
                    stats['n_current'] += 1
                elif len(fields) == adc.N_COLS_LEGACY:
                    stats['n_legacy'] += 1
                else:
                    stats['rejected'] += 1
                    continue
                for token in fields[2:]:
                    if ',' in token:
                        stats['comma_fields'] += 1
                        saw_comma = True
                    elif '.' in token:
                        saw_dot = True
        if saw_comma and saw_dot:
            stats['mixed_separator_files'] += 1

    return stats


def parse_era(paths, era, progress=True, verbose=True):
    """
    Parse one era's files into a single frame, without cleaning anything.

    Duplicate timestamps are merged field by field, taking the first non-null
    value in each column, which is the behaviour both existing loaders use: a
    third of the legacy files carry duplicated timestamps, and where the copies
    disagree it is usually because one holds real values in a block where the
    other holds zeros. Dropping whole records would discard measurements.

    One pass extracts every block. The unified ingest calls the legacy loader
    once per station and so parses each legacy file three times.

    Pass the **whole** file list for both eras rather than the files whose names
    fall inside one of them. Each parser keeps only the records matching its own
    field count and silently ignores the rest, so calling it twice over
    everything branches on the record as the specification requires, rather than
    on the date, which it forbids. It is not hypothetical: seven current-era
    records live in files dated before the changeover, and splitting by filename
    loses them.

    Parameters
    ----------
    paths : list of str
        Cached ``.adc`` files. The whole archive, for both calls.
    era : {'legacy', 'current'}
        Which layout to keep. Selects the parser and the expected field count.
    progress : bool, optional
        Draw a progress bar over the files. Default ``True``.
    verbose : bool, optional
        Print a summary of what was read. Default ``True``.

    Returns
    -------
    frame : pd.DataFrame
        Every record of the era, indexed by timestamp, duplicates merged,
        **values exactly as written** — sentinels included, no unit conversion.
    stats : dict
        ``files``, ``records`` before merging, ``duplicates`` sharing a
        timestamp, ``conflicts`` resolved field by field, and ``rows`` after
        merging.
    """
    if era == 'legacy':
        parser, columns = adc.parse_legacy_file, adc.LEGACY_COLUMNS
    elif era == 'current':
        parser, columns = adc.parse_file, adc.CURRENT_COLUMNS
    else:
        raise ValueError(f"era must be 'legacy' or 'current', not {era!r}")

    iterator = paths
    if progress and paths:
        from tqdm.auto import tqdm
        iterator = tqdm(paths, desc=f'  parsing {era}', unit='file')

    frames = [parser(path) for path in iterator]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return (pd.DataFrame(columns=columns),
                {'files': len(paths), 'records': 0, 'duplicates': 0,
                 'conflicts': 0, 'rows': 0})

    raw = pd.concat(frames).sort_index()
    n_records = len(raw)

    duplicated = raw.index.duplicated(keep=False)
    n_duplicates = int(duplicated.sum())
    n_conflicts = 0
    if n_duplicates:
        grouped = raw[duplicated].groupby(level=0)
        n_conflicts = int(grouped.nunique().gt(1).sum().sum())
    merged = raw.groupby(level=0).first()

    stats = {'files': len(paths), 'records': n_records,
             'duplicates': n_duplicates, 'conflicts': n_conflicts,
             'rows': int(len(merged))}

    if verbose:
        print(f'  {era:8s}: {n_records} records from {len(paths)} files, '
              f'{n_duplicates} sharing a timestamp '
              f'({n_conflicts} field conflicts merged), '
              f'{len(merged)} distinct timestamps')
    return merged, stats


# ──────────────────────────────────────────────────────────────────────
# Assembling the record as written
# ──────────────────────────────────────────────────────────────────────

def build_grid(start, end, freq=SAMPLING):
    """
    The regular grid the whole study works on.

    Parameters
    ----------
    start, end : str or pd.Timestamp
        Inclusive day bounds of the archive.
    freq : str, optional
        Sampling interval. Default :data:`SAMPLING`, the native rate.

    Returns
    -------
    pd.DatetimeIndex
        One entry per slot, from the start of ``start`` to the end of ``end``.
    """
    return pd.date_range(pd.Timestamp(start).floor('D'),
                         pd.Timestamp(end).ceil('D'), freq=freq,
                         name='datetime')


def assemble_wide(legacy_raw, current_raw, grid):
    """
    Place both eras on the common grid, in the layout of the files themselves.

    The legacy blocks keep their station names and the current-era package
    becomes the ``n`` block. Its twelve legacy columns are dropped: they are
    written as constant zero on every current-era record and carry no station's
    data. Slots with no record are ``NaN``, so that completeness is always
    measured against the intended sampling grid rather than against the records
    that happen to exist.

    The ``era`` column is taken from the parser that produced the row, which is
    to say from the record's own field count, so that a file whose content
    disagrees with its date is labelled by what it contains. It is **empty where
    nothing was recorded**: the column names the layout that produced a slot, not
    the era the calendar puts the slot in. That is deliberately narrower than the
    unified dataset's column of the same name, which labels every slot in an
    era's date range whether or not anything was written there.

    Parameters
    ----------
    legacy_raw : pd.DataFrame
        Legacy-era records from :func:`parse_era`.
    current_raw : pd.DataFrame
        Current-era records from :func:`parse_era`. May be empty.
    grid : pd.DatetimeIndex
        The common grid, from :func:`build_grid`.

    Returns
    -------
    pd.DataFrame
        One row per slot; :data:`RAW_COLUMNS` as written, plus ``era``.
    """
    wide = pd.DataFrame(index=grid, columns=RAW_COLUMNS, dtype=float)
    wide['era'] = pd.Series(pd.NA, index=grid, dtype='object')

    if len(legacy_raw):
        legacy = legacy_raw.rename(
            columns={f'{block}_i': f'{block}_i_mv' for block in LEGACY_BLOCKS})
        index = legacy.index.intersection(grid)
        for column in RAW_COLUMNS:
            if column in legacy.columns:
                wide.loc[index, column] = legacy.loc[index, column].astype(float)
        wide.loc[index, 'era'] = 'legacy'

    if current_raw is not None and len(current_raw):
        current = current_raw.rename(columns={
            channel: f'{CURRENT_BLOCK}_{channel}'
            for channel in BLOCK_CHANNELS[CURRENT_BLOCK]})
        index = current.index.intersection(grid)
        for column in RAW_COLUMNS:
            if column in current.columns:
                wide.loc[index, column] = current.loc[index, column].astype(float)
        wide.loc[index, 'era'] = 'current'

    return wide


# ──────────────────────────────────────────────────────────────────────
# Flagging and correcting
# ──────────────────────────────────────────────────────────────────────

def solar_night_mask(index, elevation=NIGHT_ELEVATION):
    """
    True where the sun stands below a given elevation at the site.

    The archive timestamps are the logger's civil clock, :data:`SITE_TZ`, while
    the solar geometry is defined against UTC, so the index is converted before
    the elevation is evaluated. The two daylight-saving transition hours are
    resolved rather than raised on: the logger writes wall-clock time and neither
    repeats nor skips a reading, so either resolution names the same physical
    instant to within an hour, far inside the margin this mask carries.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps in the logger's civil clock.
    elevation : float, optional
        Threshold in degrees. Default :data:`NIGHT_ELEVATION`.

    Returns
    -------
    np.ndarray of bool
        True where the sun is below ``elevation``.
    """
    utc = (pd.DatetimeIndex(index)
           .tz_localize(SITE_TZ, ambiguous=True, nonexistent='shift_forward')
           .tz_convert('UTC').tz_localize(None))
    return (solar.solar_elevation(utc, latitude=solar.SITE_LATITUDE,
                               longitude=solar.SITE_LONGITUDE).to_numpy()
            < elevation)


def flag_and_correct(wide):
    """
    Add the flag and corrected columns beside every recorded channel.

    Each raw column gains ``{col}_flag``, saying what was decided about the
    value, and ``{col}_ok``, the value the study goes on to use. **The recorded
    column is never modified**, so a reader who disagrees with any decision here
    can recover the number behind it.

    Two kinds of code are written, and they differ in what they do to
    ``{col}_ok``:

    - **Rejections** (:data:`REJECTION_CODES`) say the value is not a
      measurement and nothing is known in its place. ``{col}_ok`` is missing.
    - **Corrections** (:data:`CORRECTION_CODES`) say the value is wrong but its
      true value is known. ``{col}_ok`` carries the substituted value. The only
      one is :data:`FLAG_NIGHT`: radiation reported while the sun is below civil
      twilight, whose true value is zero.

    So the invariant is: **``{col}_ok`` is missing exactly where a rejection code
    is set, and carries a substituted value exactly where a correction code is.**
    A consumer that treats every flag as a gap will be right about the
    rejections and needlessly lose the corrected nights.

    Every rejection here is a marker the acquisition system wrote, never a
    reading the instrument produced. The calibrated range of the inclinometer
    conditioner is therefore not among the tests: see :data:`REJECTION_CODES`.
    The night correction runs last and only on values no rejection has already
    claimed, since a wrapped reading is not a measurement of anything and cannot
    be corrected into one.

    The radiation ceiling is crossed by two different defects, so it carries two
    codes. :data:`FLAG_WRAP` is the documented artefact of the raw-data
    specification, a single sample returned at the full scale of the field;
    :data:`FLAG_UNPHYSICAL` is everything else above
    :data:`shmlib.adc.SR_MAX_PHYSICAL`, which in this archive is a six-week run of
    daylight-scale values reported around the clock and is not documented
    anywhere. Both empty ``{col}_ok``, exactly as the single test they replace
    did, so no value the project consumes changes; only the account of why it was
    rejected does. The wrap test runs first and the second test sees only what it
    left, so no sample is counted twice.

    Two further verdicts in this study condemn a reading without either rejecting
    or correcting it — the Hampel spike test and the day-quality test of
    :func:`flag_sr_day_quality` — and each carries its own boolean column.

    Parameters
    ----------
    wide : pd.DataFrame
        The record as written, from :func:`assemble_wide`.

    Returns
    -------
    out : pd.DataFrame
        A copy carrying ``{col}_flag`` and ``{col}_ok`` for every raw column.
    counts : dict
        Number of samples carrying each code, keyed ``'{col}:{code}'``.
    """
    out = wide.copy()
    counts = {}

    for column in RAW_COLUMNS:
        block, channel = column.split('_', 1)
        values = out[column]
        flag = pd.Series('', index=out.index, dtype='object')

        if channel in ZERO_IS_SENTINEL:
            hit, code = values == 0.0, FLAG_SENTINEL
        elif channel == 'twall':
            hit, code = values == adc.TWALL_SENTINEL, FLAG_SENTINEL
        elif channel == 'sr':
            hit, code = values >= SR_WRAP_MIN, FLAG_WRAP
        else:
            hit, code = pd.Series(False, index=out.index), None

        if code is not None and int(hit.sum()):
            flag[hit] = code
            counts[f'{column}:{code}'] = int(hit.sum())

        if channel == 'sr':
            # Everything above the physical ceiling that is not the documented
            # wrap-around. Rejected for the same reason — no pyranometer at this
            # latitude can report it — but counted under its own code, because it
            # is a different defect with a different signature: not a single
            # wrapped sample but a six-week run of daylight-scale values reported
            # around the clock, which the specification does not describe.
            beyond = (flag == '') & (values > adc.SR_MAX_PHYSICAL)
            flag[beyond] = FLAG_UNPHYSICAL
            if int(beyond.sum()):
                counts[f'{column}:{FLAG_UNPHYSICAL}'] = int(beyond.sum())

        out[f'{column}_flag'] = flag
        out[f'{column}_ok'] = values.where(flag == '')

    return out, counts


def correct_sr_night(wide, elevation=NIGHT_ELEVATION):
    """
    Set the radiation recorded while the sun is down to the zero it must be.

    A correction rather than a rejection: the reading is wrong, but unlike a
    sentinel or a wrap-around its true value is known exactly. ``n_sr_ok``
    therefore carries zero and ``n_sr_flag`` says :data:`FLAG_NIGHT`, while the
    recorded column keeps whatever the instrument claimed.

    The cut is civil twilight, not the geometric horizon: real diffuse radiation
    survives to about -4 degrees and zeroing it would destroy measurements rather
    than correct them. Below :data:`NIGHT_ELEVATION` what the channel reports is
    its own floor.

    Applied only where no rejection has already claimed the sample, because a
    wrapped reading measures nothing and cannot be corrected into a zero, and
    only where a value was actually recorded, because writing a zero into an
    empty slot would invent a measurement.

    **Run this after** :func:`flag_sr_day_quality`. That test divides by the dark
    radiation, which this function sets to zero.

    Parameters
    ----------
    wide : pd.DataFrame
        Frame carrying ``n_sr``, ``n_sr_flag`` and ``n_sr_ok``.
    elevation : float, optional
        Threshold in degrees. Default :data:`NIGHT_ELEVATION`.

    Returns
    -------
    out : pd.DataFrame
        A copy with the correction applied.
    n_corrected : int
        Samples set to zero.
    """
    out = wide.copy()
    flag = out['n_sr_flag']
    night = ((flag == '') & solar_night_mask(out.index)
             & out['n_sr'].notna())
    n_corrected = int(night.sum())
    if n_corrected:
        out.loc[night, 'n_sr_flag'] = FLAG_NIGHT
        out.loc[night, 'n_sr_ok'] = 0.0
    return out, n_corrected


def sr_measurements(wide):
    """
    The radiation that survived rejection, before the night correction.

    Both diagnostics that justify or apply the night correction need this
    intermediate state: the samples no rejection has claimed, still carrying what
    the instrument reported when the sun was down. Reconstructing it from the
    flag rather than depending on being called at the right moment is what keeps
    :func:`flag_sr_day_quality` and :func:`sr_by_elevation` correct whether they
    run before or after :func:`correct_sr_night`.

    Parameters
    ----------
    wide : pd.DataFrame
        Frame carrying ``n_sr``, ``n_sr_flag`` and ``n_sr_ok``.

    Returns
    -------
    pd.Series
        Corrected radiation with the night correction undone.
    """
    values = wide['n_sr_ok'].copy()
    restored = wide['n_sr_flag'] == FLAG_NIGHT
    values[restored] = wide.loc[restored, 'n_sr']
    return values


def flag_sr_day_quality(wide, ratio=SR_DAY_RATIO, high=SUN_HIGH_ELEVATION,
                        night=NIGHT_ELEVATION, min_samples=SR_DAY_MIN_SAMPLES):
    """
    Condemn the days on which the radiation channel has no diurnal cycle.

    A working pyranometer reports hundreds of watts when the sun is high and
    nothing when it is down, so the ratio between the two is enormous — a median
    near 2500 on this archive. The channel fails in two ways and both collapse
    that ratio to about one. It sticks near a constant, reporting as much at
    midnight as at noon; or it dies, reporting exactly zero through a summer
    midday. Testing the ratio catches both, and it is the failure itself rather
    than a symptom of it.

    The windows are defined by where the sun is, not by hour labels, so that the
    test means the same thing in December as in June.

    The test reads :func:`sr_measurements`: the samples that survived rejection,
    with the night correction undone. Rejected values must not decide a day, and
    the corrected zeros cannot, because they are the denominator.

    Days too short to judge — fewer than ``min_samples`` in either window — are
    left unjudged rather than condemned; a December day barely clears 15° of
    elevation and would otherwise be condemned for being December.

    Parameters
    ----------
    wide : pd.DataFrame
        Frame carrying the recorded ``n_sr`` column.
    ratio : float, optional
        Condemn below this high-sun-to-dark ratio. Default :data:`SR_DAY_RATIO`.
    high : float, optional
        Elevation defining the high-sun window. Default
        :data:`SUN_HIGH_ELEVATION`.
    night : float, optional
        Elevation defining the dark window. Default :data:`NIGHT_ELEVATION`.
    min_samples : int, optional
        Samples required in each window. Default :data:`SR_DAY_MIN_SAMPLES`.

    Returns
    -------
    out : pd.DataFrame
        A copy carrying the boolean ``sr_suspect`` column.
    days : pd.DatetimeIndex
        The condemned days.
    per_day : pd.DataFrame
        One row per judged day: the two medians, the ratio, and the verdict.
        The body of the day-quality artefact.
    """
    out = wide.copy()
    recorded = sr_measurements(out).dropna()
    if recorded.empty:
        out['sr_suspect'] = False
        return out, pd.DatetimeIndex([]), pd.DataFrame()

    elevation = pd.Series(
        solar.solar_elevation(
            pd.DatetimeIndex(recorded.index)
            .tz_localize(SITE_TZ, ambiguous=True, nonexistent='shift_forward')
            .tz_convert('UTC').tz_localize(None)).to_numpy(),
        index=recorded.index)

    rows = []
    for day, group in recorded.groupby(recorded.index.normalize()):
        angle = elevation.loc[group.index]
        sunlit = group[angle > high]
        dark = group[angle < night]
        if len(sunlit) < min_samples or len(dark) < min_samples:
            continue
        dark_median = float(dark.median())
        sunlit_median = float(sunlit.median())
        # The floor keeps a legitimately zero night from making every ratio
        # infinite; it is far below any value the defect produces.
        day_ratio = sunlit_median / max(dark_median, 0.1)
        rows.append({'day': day, 'n_samples': int(len(group)),
                     'dark_median': round(dark_median, 3),
                     'sunlit_median': round(sunlit_median, 2),
                     'ratio': round(day_ratio, 2),
                     'condemned': day_ratio < ratio})

    per_day = pd.DataFrame(rows)
    days = (pd.DatetimeIndex(sorted(per_day.loc[per_day['condemned'], 'day']))
            if len(per_day) else pd.DatetimeIndex([]))
    out['sr_suspect'] = out.index.normalize().isin(days)
    return out, days, per_day


def sr_by_elevation(wide, suspect=None, edges=(-90, -18, -12, -9, -6, -4, -2,
                                               0, 2, 5, 10, 90)):
    """
    Solar radiation grouped by solar elevation, condemned days against the rest.

    The evidence behind :data:`NIGHT_ELEVATION`. On days the channel is working,
    the radiation decays smoothly through twilight and reaches the instrument's
    own floor by about -6°; below that there is no sky brightness left to lose,
    so a correction applied there destroys nothing. On condemned days the same
    table is flat at a couple of hundred watts from one end to the other, which
    is what having no diurnal cycle looks like when it is tabulated.

    Uses :func:`sr_measurements`, so that the correction under test does not
    define its own justification and values already rejected do not appear as if
    they were measurements.

    Parameters
    ----------
    wide : pd.DataFrame
        Frame carrying the recorded ``n_sr`` column.
    suspect : pd.Series of bool, optional
        Day-quality verdict. Defaults to ``wide['sr_suspect']`` when present.
    edges : tuple of float, optional
        Elevation band edges in degrees.

    Returns
    -------
    pd.DataFrame
        One row per band, with sample count, median and 95th percentile for the
        working and the condemned population side by side.
    """
    if suspect is None:
        suspect = wide.get('sr_suspect', pd.Series(False, index=wide.index))
    suspect = suspect.astype(bool)

    recorded = sr_measurements(wide)
    elevation = pd.Series(
        solar.solar_elevation(
            pd.DatetimeIndex(wide.index)
            .tz_localize(SITE_TZ, ambiguous=True, nonexistent='shift_forward')
            .tz_convert('UTC').tz_localize(None)).to_numpy(),
        index=wide.index)
    band = pd.cut(elevation, list(edges))

    rows = []
    for interval, group in recorded.groupby(band, observed=True):
        keep = group.dropna()
        if keep.empty:
            continue
        good = keep[~suspect.reindex(keep.index).fillna(False)]
        bad = keep[suspect.reindex(keep.index).fillna(False)]
        rows.append({
            'elevation_from': interval.left, 'elevation_to': interval.right,
            'n_working': int(len(good)),
            'median_working': round(float(good.median()), 2) if len(good) else np.nan,
            'p95_working': round(float(good.quantile(0.95)), 1) if len(good) else np.nan,
            'n_condemned': int(len(bad)),
            'median_condemned': round(float(bad.median()), 1) if len(bad) else np.nan,
        })
    return pd.DataFrame(rows)


def clock_offset(wide, min_peak=200.0, threshold=50.0, min_samples=60):
    """
    The logger's clock offset from UTC, measured against computed solar noon.

    The archive carries no timezone, and the solar geometry every night test in
    this study depends on is defined against UTC, so the offset has to be
    measured rather than assumed. On a day the radiation channel is working, the
    midpoint between the first and last crossing of a small threshold is a robust
    estimate of solar noon in the logger's own clock; computed solar noon comes
    from the site longitude and the equation of time. Their difference is the
    offset.

    Only days the day-quality test passed are used. Including the condemned days
    is what made the same measurement unreliable in the proxy-comparison study,
    where a channel with no diurnal cycle has no meaningful midpoint.

    Parameters
    ----------
    wide : pd.DataFrame
        Frame carrying ``n_sr_ok``, ``sr_suspect`` and a datetime index.
    min_peak : float, optional
        A day must reach this radiation to be used. Default ``200.0`` W/m².
    threshold : float, optional
        Crossing level defining the start and end of the day. Default ``50.0``.
    min_samples : int, optional
        Slots a day must carry to be used. Default ``60`` of the 72 a full day
        holds, which excludes the truncated days at either end of the archive:
        a day missing its afternoon has a midpoint but not a meaningful one.

    Returns
    -------
    pd.DataFrame
        One row per usable day: the observed midpoint, computed solar noon in
        UTC hours, and the offset between them in hours.
    """
    series = wide.loc[~wide.get('sr_suspect', False).astype(bool), 'n_sr_ok']
    series = series.dropna()

    rows = []
    for day, group in series.groupby(series.index.normalize()):
        if len(group) < min_samples or group.max() < min_peak:
            continue
        lit = group[group > threshold]
        if len(lit) < 10:
            continue
        first, last = lit.index[0], lit.index[-1]
        midpoint = ((first.hour + first.minute / 60.0)
                    + (last.hour + last.minute / 60.0)) / 2.0
        noon = float(solar.solar_noon_utc([day]).iloc[0])
        rows.append({'day': day, 'midpoint_local': round(midpoint, 3),
                     'solar_noon_utc': round(noon, 3),
                     'offset_hours': round(midpoint - noon, 3)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# The inclination correction chain
# ──────────────────────────────────────────────────────────────────────

def hampel_filter(series, window_size, n_sigmas=3.0, floor=10.0):
    """
    Replace isolated spikes with a linear interpolation across them.

    Each sample is compared with the median of the window centred on it and
    flagged when it lies more than ``n_sigmas`` robust standard deviations away.
    The robust standard deviation is the median absolute deviation scaled by
    1.4826, the factor that makes it agree with the ordinary standard deviation
    for Gaussian noise. Because a rolling median follows a step change within
    half a window, a genuine shift in level survives while a single-sample
    excursion does not — which is the property that matters here, since a step
    is what structural damage would look like.

    Parameters
    ----------
    series : pd.Series
        The series to filter, on a regular time grid.
    window_size : int
        Length of the centred rolling window, in samples. On the native grid a
        24-hour window is 72 samples.
    n_sigmas : float, optional
        Threshold in robust standard deviations. Default ``3.0``.
    floor : float, optional
        Lower bound on the threshold, in the units of ``series``. Over quiet
        stretches the rolling deviation falls close to zero, and without this
        floor the test would reject ordinary measurement noise. Default ``10.0``.

    Returns
    -------
    filtered : pd.Series
        The series with flagged samples replaced by linear interpolation.
    outliers : pd.Series of bool
        True where a sample was flagged and replaced.
    """
    rolling_median = series.rolling(
        window=window_size, center=True, min_periods=1).median()
    rolling_mad = (series - rolling_median).abs().rolling(
        window=window_size, center=True, min_periods=1).median()

    threshold = np.maximum(n_sigmas * 1.4826 * rolling_mad, floor)
    outliers = (series - rolling_median).abs() > threshold

    filtered = series.copy()
    filtered[outliers] = np.nan
    return filtered.interpolate(method='linear'), outliers


def filter_impulsive_segment(series, start, window_size, n_sigmas=3.0,
                             floor=10.0):
    """
    Filter isolated impulses inside one dated segment of a longer series.

    Values before ``start`` are copied without testing. Within the selected
    segment, each contiguous run of observed values is tested independently, so
    neither the rolling window nor the interpolation can cross an original gap.
    A run boundary is never replaced because it lacks observations on both
    sides. Flagged interior samples are replaced by linear interpolation.

    Parameters
    ----------
    series : pd.Series
        Full series, including any missing slots.
    start : str or pandas-compatible datetime
        First timestamp included in the anomaly test.
    window_size : int
        Length of the centred Hampel window, in observed samples.
    n_sigmas : float, optional
        Threshold in robust standard deviations. Default ``3.0``.
    floor : float, optional
        Lower bound on the threshold, in the units of ``series``. Default
        ``10.0``.

    Returns
    -------
    filtered : pd.Series
        Full series with only flagged values inside the segment replaced.
    outliers : pd.Series of bool
        True exactly where a value was flagged and replaced.
    """
    filtered = series.copy()
    outliers = pd.Series(False, index=series.index, dtype=bool)

    segment = series.loc[pd.Timestamp(start):]
    observed = segment.notna()
    run_ids = (~observed).cumsum()
    edge = window_size // 2

    for _, run in segment[observed].groupby(run_ids[observed]):
        if len(run) < window_size:
            continue
        _, run_outliers = hampel_filter(
            run,
            window_size=window_size,
            n_sigmas=n_sigmas,
            floor=floor,
        )
        if edge:
            run_outliers.iloc[:edge] = False
            run_outliers.iloc[-edge:] = False

        run_filtered = run.copy()
        run_filtered.loc[run_outliers] = np.nan
        run_filtered = run_filtered.interpolate(method='linear', limit_area='inside')
        filtered.loc[run.index] = run_filtered
        outliers.loc[run.index] = run_outliers.astype(bool)
    return filtered, outliers


def add_inclination(wide, coeff=COMP_COEFF):
    """
    Derive the inclination in millidegrees, and compensate it, block by block.

    Two steps, both per block. The recorded field is the conditioner output in
    millivolts about a 2500 mV zero, and the sensitivity is 1000 mV per degree,
    so millivolts and millidegrees share a scale and only that offset separates
    the logged number from an angle. The thermal correction then subtracts the
    instrument's own response to temperature, using that block's own air
    temperature and anchored on that block's own first record.

    These per-block series are the right thing for the three legacy units, which
    stand at three different places and have nothing to anchor in common. They
    are **not** how the target station's series is built: the two units at
    station 02 measure the same wall, so concatenating two separately anchored
    series there would plant an arbitrary step at the changeover.
    :func:`coalesce_target` builds that series instead, anchoring once.

    Parameters
    ----------
    wide : pd.DataFrame
        Frame carrying the corrected columns, from :func:`flag_and_correct`.
    coeff : float, optional
        Compensation coefficient in mdeg · °C⁻¹ · 10⁻³. Default
        :data:`COMP_COEFF`.

    Returns
    -------
    pd.DataFrame
        A copy carrying ``{block}_inc`` and ``{block}_inc_comp`` in millidegrees.
    """
    out = wide.copy()
    for block in BLOCKS:
        out[f'{block}_inc'] = (out[f'{block}_i_mv_ok']
                               - adc.INCLINOMETER_ZERO_MV)
        frame = pd.DataFrame({'inc': out[f'{block}_inc'],
                              'tair': out[f'{block}_tair_ok']})
        out[f'{block}_inc_comp'] = adc.compensate(
            frame, temp_col='tair', coeff=coeff, normalise=True)
    return out


def coalesce_target(wide, legacy_block='st02', current_block=CURRENT_BLOCK,
                    coeff=COMP_COEFF):
    """
    Build one continuous series for the target station across both eras.

    The legacy unit and the current package stand at the same place and measure
    the same wall, but they are different instruments and are kept in different
    columns for that reason. The analysis still needs one series, and this builds
    it: first the recorded channel and the recording unit's own air temperature
    are taken era by era from whichever instrument produced them, and only then
    are the compensation and the anchor applied, **once, to the whole series**.

    The order matters and is the correction of an earlier defect. Compensating
    and anchoring each unit separately and concatenating the results afterwards
    gives each era its own additive constant, and the difference between those
    two constants appears at the changeover as a step of order a hundred
    millidegrees. That step is arithmetic, not instrumentation: measured on the
    recorded channel the two units agree across the boundary to within a couple
    of millidegrees. Anchoring once removes it at the source, and there is then
    no offset left to estimate and nothing to join.

    The slope term still uses the recording unit's own air temperature, because
    the bias being cancelled is each instrument's response to the temperature it
    actually sat at. Only the reference is global.

    Parameters
    ----------
    wide : pd.DataFrame
        Frame carrying the per-block inclination columns, from
        :func:`add_inclination`.
    legacy_block : str, optional
        Unit recording the legacy era at the target station. Default ``'st02'``.
    current_block : str, optional
        Unit recording the current era. Default :data:`CURRENT_BLOCK`.
    coeff : float, optional
        Compensation coefficient in mdeg · °C⁻¹ · 10⁻³. Default
        :data:`COMP_COEFF`.

    Returns
    -------
    pd.DataFrame
        A copy carrying ``inc``, ``tair`` and ``inc_comp`` for the target
        station, the last compensated and anchored on a single reference.
    """
    out = wide.copy()
    legacy = out['era'] == 'legacy'
    current = out['era'] == 'current'

    for stage, suffix in (('inc', 'inc'), ('tair', 'tair_ok')):
        joined = pd.Series(np.nan, index=out.index)
        joined[legacy] = out.loc[legacy, f'{legacy_block}_{suffix}']
        joined[current] = out.loc[current, f'{current_block}_{suffix}']
        out[stage] = joined

    compensated = out['inc'] - (out['tair'] * coeff * 1000)
    anchored = compensated.dropna()
    out['inc_comp'] = (compensated - anchored.iloc[0]
                       if len(anchored) else compensated)
    return out


# ──────────────────────────────────────────────────────────────────────
# Views onto the wide table
# ──────────────────────────────────────────────────────────────────────

#: Position within the day, in fractional hours.
#:
#: Moved to ``shmlib.site``: study 2's diurnal profiles need the identical
#: computation on their own (hourly) grid, and ``shmlib`` must never be reached
#: into from one study by way of another — including a function that, before
#: this move, only this study defined but a later study's dissolved library was
#: already reaching across into. Kept here as an alias at the original name, so
#: every call site in this module is unaffected. The documentation prose now
#: lives with the definition in ``shmlib.site``.
time_of_day = site.time_of_day


def cycle_extreme(cycle, kind='min', flat_positions=6, tol=1e-6):
    """
    Position of a cycle's extreme, or nothing when the extreme is a plateau.

    A mean diurnal cycle normally has one maximum and one minimum, and reporting
    the hour of each says something. The corrected radiation does not: its night
    is exactly zero for half the day, so the position of its minimum is whichever
    slot the search happens to visit first and carries no information at all.
    Returning nothing there is more honest than returning midnight.

    Parameters
    ----------
    cycle : pd.Series
        Mean cycle indexed by position within the day.
    kind : {'min', 'max'}, optional
        Which extreme. Default ``'min'``.
    flat_positions : int, optional
        Report nothing when more than this many positions sit at the extreme.
        Default ``6``, two hours on the native grid.
    tol : float, optional
        Absolute tolerance for counting a position as at the extreme.

    Returns
    -------
    float
        Position within the day, or ``nan`` when the extreme is a plateau.
    """
    value = cycle.min() if kind == 'min' else cycle.max()
    if int((cycle - value).abs().le(tol).sum()) > flat_positions:
        return float('nan')
    return float(cycle.idxmin() if kind == 'min' else cycle.idxmax())


def format_clock(hours):
    """
    Render a fractional hour as a clock time.

    Parameters
    ----------
    hours : float
        Fractional hour, as returned by :func:`time_of_day`.

    Returns
    -------
    str
        ``HH:MM``, or ``'---'`` when the value is missing.
    """
    if hours is None or (isinstance(hours, float) and np.isnan(hours)):
        return '---'
    total = int(round(float(hours) * 60.0))
    return f'{(total // 60) % 24:02d}:{total % 60:02d}'


def block_view(wide, block):
    """
    One acquisition block as a tidy frame, on corrected values.

    The wide table is faithful to the files and awkward to plot from. This
    returns one block under plain channel names, carrying the corrected values
    rather than the recorded ones, which is what every figure in the study wants.

    Parameters
    ----------
    wide : pd.DataFrame
        The assembled, flagged and corrected frame.
    block : str
        One of :data:`BLOCKS`.

    Returns
    -------
    pd.DataFrame
        Columns ``batt``, ``tair``, ``rh``, ``inc``, ``inc_comp`` and ``era``,
        plus ``sr`` and ``twall`` for the current-era block.
    """
    out = pd.DataFrame(index=wide.index)
    for channel in BLOCK_CHANNELS[block]:
        if channel == 'i_mv':
            continue
        out[channel] = wide[f'{block}_{channel}_ok']
    out['inc'] = wide[f'{block}_inc']
    out['inc_comp'] = wide[f'{block}_inc_comp']
    out['era'] = wide['era']
    return out


def target_view(wide, station=TARGET_STATION, legacy_block='st02',
                current_block=CURRENT_BLOCK):
    """
    The target station across both eras, as the analysis reads it.

    Every environmental channel is taken from the instrument that was recording
    at the time, and the inclination arrives with its whole correction chain
    already attached. This is the frame the report's per-variable sections work
    on, and it is deliberately the only place in this study where the two
    instruments appear under one set of column names.

    Parameters
    ----------
    wide : pd.DataFrame
        The assembled frame, after :func:`inclination_chain`.
    station : str, optional
        Station identifier written into the frame. Default
        :data:`TARGET_STATION`.
    legacy_block, current_block : str, optional
        Blocks recording the two eras.

    Returns
    -------
    pd.DataFrame
        The four core channels, the two current-era channels, the inclination at
        every stage, the verdict columns, and provenance.
    """
    legacy = wide['era'] == 'legacy'
    current = wide['era'] == 'current'

    out = pd.DataFrame(index=wide.index)
    for channel in ('batt', 'tair', 'rh'):
        joined = pd.Series(np.nan, index=wide.index)
        joined[legacy] = wide.loc[legacy, f'{legacy_block}_{channel}_ok']
        joined[current] = wide.loc[current, f'{current_block}_{channel}_ok']
        out[channel] = joined

    for channel in ('sr', 'twall'):
        out[channel] = wide[f'{current_block}_{channel}_ok']
    if f'{current_block}_twall_filtered' in wide.columns:
        out['twall_filtered'] = wide[f'{current_block}_twall_filtered']

    for stage in ('inc', 'tair', 'inc_comp', 'inc_comp_cleaned'):
        if stage in wide.columns:
            out[stage] = wide[stage]

    for verdict in ('inc_spike', 'twall_spike', 'sr_suspect'):
        if verdict in wide.columns:
            out[verdict] = wide[verdict]

    out['era'] = wide['era']
    out['station'] = station
    out['instrument'] = np.where(
        current, f'current_package_{station}',
        np.where(legacy, f'legacy_block_{station}', None))
    out['inc_source'] = np.where(out['inc'].notna(), 'observed', 'missing')
    return out


# ──────────────────────────────────────────────────────────────────────
# Coverage, and the shape of each station's record
# ──────────────────────────────────────────────────────────────────────

def coverage_by_period(frames, stations, column, freq):
    """
    Fraction of each period observed, one column per station.

    Parameters
    ----------
    frames : dict of str to pd.DataFrame
        One tidy frame per station, as returned by :func:`block_view` and
        :func:`target_view`.
    stations : sequence of str
        Stations to include, in the order the columns should appear.
    column : str
        Channel whose presence defines coverage.
    freq : str
        Resampling frequency, e.g. ``'D'`` for daily or ``'YE'`` for annual.

    Returns
    -------
    pd.DataFrame
        Indexed by period, one column per station, values between 0 and 1.
    """
    return pd.DataFrame({
        station: frames[station][column].notna().resample(freq).mean()
        for station in stations
    })


def yearly_coverage(frames, stations, column='inc_comp'):
    """
    Fraction of each calendar year observed, one column per station.

    Parameters
    ----------
    frames : dict of str to pd.DataFrame
        One tidy frame per station.
    stations : sequence of str
        Stations to include, in column order.
    column : str, optional
        Channel whose presence defines coverage. Default ``'inc_comp'``.

    Returns
    -------
    pd.DataFrame
        Indexed by year as an integer, one column per station.
    """
    yearly = coverage_by_period(frames, stations, column, 'YE')
    yearly.index = yearly.index.year
    yearly.index.name = 'year'
    return yearly


def station_classification(frames, stations, slots_per_day,
                           dual_era_station=TARGET_STATION, column='inc_comp'):
    """
    What each station actually contributed: when it ran, and how much of it.

    Two coverage figures are reported and they answer different questions.
    Coverage of the span asks how reliably a station recorded while it was
    alive; coverage of the archive asks how much of the whole record it
    accounts for. A unit that ran perfectly for two years and then died scores
    high on the first and low on the second, and reporting only one of them
    would describe it wrongly either way.

    Parameters
    ----------
    frames : dict of str to pd.DataFrame
        One tidy frame per station.
    stations : sequence of str
        Stations to classify, in row order.
    slots_per_day : int
        Slots of the analysis grid in one day, used to convert an observation
        count into observed days.
    dual_era_station : str, optional
        The station instrumented in both eras. Default
        :data:`TARGET_STATION`. Every other station is legacy only.
    column : str, optional
        Channel whose presence defines an observation. Default ``'inc_comp'``.

    Returns
    -------
    pd.DataFrame
        Indexed by station, carrying the eras it spans, its first and last
        reading, the span between them in days, the days observed, and the two
        coverage percentages.
    """
    rows = []
    for station in stations:
        frame = frames[station]
        first = frame[column].first_valid_index()
        last = frame[column].last_valid_index()
        span = (last - first).total_seconds() / 86400
        observed = frame[column].notna().sum() / slots_per_day
        archive_days = ((frame.index[-1] - frame.index[0]).total_seconds()
                        / 86400)
        rows.append({
            'station': station,
            'eras': ('legacy+current' if station == dual_era_station
                     else 'legacy'),
            'first_reading': str(first),
            'last_reading': str(last),
            'span_days': span,
            'observed_days': observed,
            'coverage_of_span_%': 100 * observed / span,
            'coverage_of_archive_%': 100 * observed / archive_days,
        })
    return pd.DataFrame(rows).set_index('station')


def summary_by_era(df, channels, decimals=3):
    """
    Count, mean, spread and extremes of each channel, era by era.

    The eras are summarised separately throughout this study. Pooling them
    would average two unrelated inclinometer baselines into one number, and the
    standard deviation of that pooled series would be dominated by the step
    between the instruments rather than by anything the wall did.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame, carrying an ``era`` column.
    channels : sequence of str
        Channels to summarise, in row order within each era.
    decimals : int, optional
        Decimal places for the statistics. Default 3.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(era, channel)``, carrying ``count``, ``mean``, ``sd``,
        ``min``, ``median`` and ``max``. A statistic that the data cannot
        support — a mean of nothing, a standard deviation of one sample — is
        missing rather than zero.
    """
    rows = []
    for era, sub in df.groupby('era', sort=False):
        for column in channels:
            series = sub[column].dropna()
            rows.append({
                'era': era,
                'channel': column,
                'count': int(series.size),
                'mean': round(float(series.mean()), decimals) if series.size else np.nan,
                'sd': round(float(series.std()), decimals) if series.size > 1 else np.nan,
                'min': round(float(series.min()), decimals) if series.size else np.nan,
                'median': round(float(series.median()), decimals) if series.size else np.nan,
                'max': round(float(series.max()), decimals) if series.size else np.nan,
            })
    return pd.DataFrame(rows).set_index(['era', 'channel'])


# ──────────────────────────────────────────────────────────────────────
# The current era, and the channels that exist only in it
# ──────────────────────────────────────────────────────────────────────

def current_era_availability(current, channels, season_months, complete_day):
    """
    What each channel of the current era actually holds, season by season.

    Two measures, because they answer different questions. The share of the era
    observed says how much of the channel exists; the count of *complete* days
    says how much of it can carry a diurnal average, which is what the rest of
    the step needs. A channel present in scattered hours scores well on the
    first and not at all on the second.

    Parameters
    ----------
    current : pd.DataFrame
        The current era of a station's tidy frame.
    channels : sequence of str
        Channels to report, in row order.
    season_months : dict of str to sequence of int
        Months belonging to each season, in column order.
    complete_day : int
        Slots a day must carry to count as complete.

    Returns
    -------
    pd.DataFrame
        One row per channel: ``observed``, ``pct_of_era``, and one
        ``complete_days_{season}`` column per season.
    """
    rows = []
    for column in channels:
        row = {'channel': column,
               'observed': int(current[column].notna().sum()),
               'pct_of_era': round(100 * float(current[column].notna().mean()), 1)}
        for season, months in season_months.items():
            group = current[current.index.month.isin(months)]
            counts = group.groupby(group.index.date)[column].count()
            row[f'complete_days_{season}'] = int((counts == complete_day).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def diurnal_response(diurnal_stats, response_channel, driver_channel,
                     seasons=('summer', 'winter'), eras=('legacy', 'current')):
    """
    The structural response normalised by the forcing that produced it.

    The amplitude of the response means little until the amplitude of the
    forcing is known: a summer warmer than the one before it enlarges the
    inclination cycle without anything about the wall or the instrument having
    changed. Dividing by the driver's amplitude over the same complete days
    removes that, and what remains is a sensitivity in millidegrees per degree
    Celsius. That is the quantity to compare across the changeover — an
    instrument whose gain changed would move it, and a merely warmer season
    would not.

    Parameters
    ----------
    diurnal_stats : pd.DataFrame
        Diurnal statistics carrying ``channel``, ``era``, ``season``,
        ``amplitude`` and ``complete_days``.
    response_channel : str
        The structural response, e.g. the cleaned inclination.
    driver_channel : str
        The forcing to normalise by, e.g. the air temperature.
    seasons : sequence of str, optional
        Seasons to report, in row order. Default ``('summer', 'winter')``.
    eras : sequence of str, optional
        Eras to report, within each season. Default
        ``('legacy', 'current')``.

    Returns
    -------
    pd.DataFrame
        One row per season and era, carrying both amplitudes, the number of
        complete days, and their ratio.
    """
    amplitude = diurnal_stats.set_index(['channel', 'era', 'season'])['amplitude']
    days = diurnal_stats.set_index(['channel', 'era', 'season'])['complete_days']

    rows = []
    for season in seasons:
        for era in eras:
            response = float(amplitude[(response_channel, era, season)])
            driver = float(amplitude[(driver_channel, era, season)])
            rows.append({
                'season': season, 'era': era,
                'complete_days': int(days[(response_channel, era, season)]),
                'inc_amplitude_mdeg': round(response, 2),
                'tair_amplitude_degC': round(driver, 2),
                'response_mdeg_per_degC': round(response / driver, 3),
            })
    return pd.DataFrame(rows)


#: The states a day of the radiation failure can be in, in the order they are
#: tested. The order is what makes them a partition rather than a description: a
#: day on which nothing was written cannot also be a day on which nothing
#: survived rejection, and a day on which nothing survived cannot be judged at
#: all. They run from the most complete failure to the least.
SR_STATES = ('no record at all',
             'pegged above the ceiling, every sample rejected',
             'dead, zero at high sun',
             'stuck near a constant',
             'passes the ratio test')


def sr_failure_window(quality):
    """
    The extent of the radiation failure, from the day-level verdict.

    The window opens on the first condemned day and closes on the day before the
    channel next produces a day that passes, so an outage interrupting it falls
    inside: an instrument that stops recording has not recovered, and the record
    cannot show that it has until it reports again.

    Parameters
    ----------
    quality : pd.DataFrame
        Day-quality table indexed by day, carrying ``condemned``.

    Returns
    -------
    start : pd.Timestamp
        First condemned day.
    end : pd.Timestamp
        Last day of the failure.
    recovery : pd.Timestamp
        First day that passes again.
    """
    condemned = quality.index[quality['condemned']]
    start = condemned.min()
    after = quality.loc[quality.index > condemned.max()]
    recovery = after.index[~after['condemned']].min()
    return start, recovery - pd.Timedelta(days=1), recovery


def sr_day_states(wide, quality, window, dead_threshold=10.0):
    """
    Name the single state each day of the failure window belongs to.

    The window is long enough that discarding it wholesale deserves an argument
    rather than an assertion, and the argument is that every day in it falls
    into one of :data:`SR_STATES` and none of those states contains a day whose
    radiation could be used.

    Parameters
    ----------
    wide : pd.DataFrame
        The flagged archive table.
    quality : pd.DataFrame
        Day-quality table indexed by day, carrying ``condemned`` and
        ``sunlit_median``.
    window : pd.DatetimeIndex
        The calendar days of the failure window.
    dead_threshold : float, optional
        High-sun median below which a condemned day counts as dead rather than
        stuck, in W/m². Default 10.0.

    Returns
    -------
    pd.Series
        One state per day of ``window``, indexed by day.

    Raises
    ------
    AssertionError
        If the states do not partition the window, which would mean the
        taxonomy had developed a gap or an overlap.
    """
    span = wide.loc[window.min():window.max() + pd.Timedelta(days=1)]
    recorded = set(span['n_sr'].dropna().index.normalize())
    survived = set(sr_measurements(span).dropna().index.normalize())

    def classify(day):
        if day not in recorded:
            return SR_STATES[0]
        if day not in survived:
            return SR_STATES[1]
        if day not in quality.index:
            return SR_STATES[1]          # recorded and survived, too short to judge
        row = quality.loc[day]
        if not row['condemned']:
            return SR_STATES[4]
        return (SR_STATES[2] if row['sunlit_median'] < dead_threshold
                else SR_STATES[3])

    state = pd.Series([classify(day) for day in window], index=window)
    assert state.isin(SR_STATES).all(), 'states must partition the window'
    return state


def condemned_modes(quality, dead_threshold=10.0):
    """
    Split the condemned radiation days into the two ways the channel fails.

    The two modes partition the condemned days rather than merely describing
    them, so that the counts drawn from them and the accounting of the whole
    failure window cannot disagree. Dead is the sharp test — nothing at all at
    high sun — and stuck is every other condemned day.

    Parameters
    ----------
    quality : pd.DataFrame
        Day-quality table indexed by day, carrying ``condemned`` and
        ``sunlit_median``.
    dead_threshold : float, optional
        High-sun median below which a day counts as dead, in W/m². Default
        10.0.

    Returns
    -------
    dead : pd.DataFrame
        Condemned days reporting essentially nothing at high sun.
    stuck : pd.DataFrame
        Condemned days reporting as much in the dark as at high sun.
    """
    condemned = quality[quality['condemned']]
    return (condemned[condemned['sunlit_median'] < dead_threshold],
            condemned[condemned['sunlit_median'] >= dead_threshold])


def condemned_by_month(quality):
    """
    The condemned days month by month, with the level they reported.

    Parameters
    ----------
    quality : pd.DataFrame
        Day-quality table indexed by day, carrying ``condemned``,
        ``sunlit_median`` and ``dark_median``.

    Returns
    -------
    pd.DataFrame
        Indexed by month, carrying the number of condemned ``days`` and the
        median of the day medians at high sun and in the dark.
    """
    condemned = quality[quality['condemned']]
    rows = []
    for month, group in condemned.groupby(condemned.index.to_period('M')):
        rows.append({'month': str(month), 'days': int(len(group)),
                     'sunlit_median': float(group['sunlit_median'].median()),
                     'dark_median': float(group['dark_median'].median())})
    return pd.DataFrame(rows).set_index('month')


def response_ratios(response, seasons=('summer', 'winter')):
    """
    How the response changed across the changeover, before and after normalising.

    The bare amplitude ratio still carries the season; the normalised one does
    not. Reporting both is what separates an instrument whose gain changed from
    a season that was merely warmer.

    Parameters
    ----------
    response : pd.DataFrame
        Output of :func:`diurnal_response`.
    seasons : sequence of str, optional
        Seasons to report. Default ``('summer', 'winter')``.

    Returns
    -------
    pd.DataFrame
        Indexed by season, carrying ``amplitude_ratio`` and ``response_ratio``,
        each current over legacy.
    """
    rows = []
    for season in seasons:
        sub = response[response['season'] == season].set_index('era')
        rows.append({
            'season': season,
            'amplitude_ratio': (sub.loc['current', 'inc_amplitude_mdeg']
                                / sub.loc['legacy', 'inc_amplitude_mdeg']),
            'response_ratio': (sub.loc['current', 'response_mdeg_per_degC']
                               / sub.loc['legacy', 'response_mdeg_per_degC']),
        })
    return pd.DataFrame(rows).set_index('season')


def sr_valid_days(quality, fail_start, fail_end, index):
    """
    The days whose radiation may be used, and the mask that selects them.

    A day qualifies when the day-quality test judged it and let it stand, **and**
    when it falls outside the failure window. The second condition is not implied
    by the first: the window contains days the test could not judge at all, and
    an unjudged day has not been approved by anything. Relying on the condemned
    flag alone would let them through.

    Parameters
    ----------
    quality : pd.DataFrame
        Day-quality table indexed by day, carrying ``condemned``.
    fail_start, fail_end : pd.Timestamp
        Bounds of the failure window, inclusive.
    index : pd.DatetimeIndex
        Index the mask should cover, normally the current era's.

    Returns
    -------
    days : pd.DatetimeIndex
        The qualifying days.
    mask : pd.Series
        Boolean over ``index``, true on slots belonging to a qualifying day.
    """
    judged = quality.index
    inside = (judged >= fail_start) & (judged <= fail_end)
    days = pd.DatetimeIndex(judged[~quality['condemned'].to_numpy() & ~inside])
    mask = pd.Series(index.normalize().isin(days), index=index)
    return days, mask


def clock_step(clock, smooth_days=7):
    """
    Locate the daylight-saving step in the measured clock offset.

    A single day's estimate scatters by half an hour, so the step is sought in a
    centred rolling median rather than in the raw daily values, where ordinary
    scatter would out-rank it.

    Parameters
    ----------
    clock : pd.DataFrame
        Per-day offsets from :func:`clock_offset`, carrying ``day`` and
        ``offset_hours``.
    smooth_days : int, optional
        Length of the centred rolling median, in days. Default 7.

    Returns
    -------
    dict
        ``before`` and ``after``, each a ``(day, offset)`` pair bracketing the
        largest step in the smoothed offset.
    """
    ordered = clock.sort_values('day').reset_index(drop=True)
    smooth = ordered['offset_hours'].rolling(smooth_days, center=True,
                                             min_periods=3).median()
    jump = smooth.diff().idxmax()
    return {'before': (ordered.loc[jump - 1, 'day'], float(smooth[jump - 1])),
            'after': (ordered.loc[jump, 'day'], float(smooth[jump]))}


# ──────────────────────────────────────────────────────────────────────
# The summer 2026 anomaly
# ──────────────────────────────────────────────────────────────────────

def anomaly_and_reference(df, column, anomaly_start, anomaly_end,
                          historical_end):
    """
    The anomaly window, and the same calendar window averaged over earlier years.

    Comparing the excursion against the same days of every preceding year is
    what separates the ordinary seasonal shape from whatever is peculiar to the
    anomaly year. The reference stops at ``historical_end`` so that the anomaly
    cannot contribute to its own baseline.

    The average is taken over month, day and position within the day rather than
    over day of year, because a leap year shifts the day of year by one; and the
    position within the day is the fractional hour, since on the native
    twenty-minute grid the hour alone names three slots.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame.
    column : str
        Channel to compare.
    anomaly_start, anomaly_end : pd.Timestamp
        Bounds of the anomaly window, inclusive.
    historical_end : pd.Timestamp
        Last timestamp counted as history. Everything after it belongs to the
        anomaly year and is excluded from the reference.

    Returns
    -------
    anomaly : pd.Series
        Observed values inside the anomaly window.
    reference : pd.Series
        The historical average, on the timestamps of ``anomaly``. Missing where
        no earlier year observed that position in the day.
    """
    anomaly = df.loc[anomaly_start:anomaly_end, column].dropna()

    historical = df.loc[:historical_end, column].dropna()
    first_day = (anomaly_start.month, anomaly_start.day)
    last_day = (anomaly_end.month, anomaly_end.day)
    in_window = [first_day <= day <= last_day
                 for day in zip(historical.index.month, historical.index.day)]
    historical = historical[in_window]

    average = historical.groupby(
        [historical.index.month, historical.index.day,
         time_of_day(historical.index)]).mean()

    reference = pd.Series(index=anomaly.index, dtype=float)
    for stamp, clock in zip(anomaly.index, time_of_day(anomaly.index)):
        try:
            reference[stamp] = average.loc[(stamp.month, stamp.day, clock)]
        except KeyError:
            pass
    return anomaly, reference


def extreme_excursions(series, limit):
    """
    Split a series at a threshold, into the extreme population and the rest.

    The threshold is not a filter — nothing is removed by it — only a line drawn
    beyond the ordinary spread, so that the extreme population can be counted
    and dated separately from the record it is buried in.

    Parameters
    ----------
    series : pd.Series
        Observed values of one era.
    limit : float
        Values strictly below this are extreme.

    Returns
    -------
    extreme : pd.Series
        Values below ``limit``.
    ordinary : pd.Series
        Everything else.
    """
    return series[series < limit], series[series >= limit]


def excursions_by_month(extreme):
    """
    Count and depth of an extreme population, month by month.

    Parameters
    ----------
    extreme : pd.Series
        The extreme values, as returned by :func:`extreme_excursions`.

    Returns
    -------
    pd.DataFrame
        Indexed by month period, carrying ``n``, ``minimum`` and the number of
        distinct ``days`` the month's excursions fall on.
    """
    rows = []
    for month, group in extreme.groupby(extreme.index.to_period('M')):
        rows.append({'month': str(month), 'n': int(len(group)),
                     'minimum': float(group.min()),
                     'days': int(len(np.unique(group.index.date)))})
    return pd.DataFrame(rows).set_index('month')


def spike_replacement_summary(df, changeover, column='inc_comp',
                              spike_column='inc_spike'):
    """
    How much of each era the impulsive filter replaced.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame, after filtering.
    changeover : pd.Timestamp
        First timestamp of the current era.
    column : str, optional
        The observed channel. Default ``'inc_comp'``.
    spike_column : str, optional
        Boolean column marking replaced samples. Default ``'inc_spike'``.

    Returns
    -------
    pd.DataFrame
        Indexed by era, carrying ``observed``, ``replaced`` and ``pct``.
    """
    rows = []
    for era, mask in (('legacy', df.index < changeover),
                      ('current', df.index >= changeover)):
        observed = int(df.loc[mask, column].notna().sum())
        replaced = int(df.loc[mask, spike_column].sum())
        rows.append({'era': era, 'observed': observed, 'replaced': replaced,
                     'pct': 100 * replaced / observed if observed else np.nan})
    return pd.DataFrame(rows).set_index('era')


def recorded_thermal_slope(df, eras=('legacy', 'current'),
                           target='inc', driver='tair',
                           spike_column='inc_spike'):
    """
    Slope of the recorded inclination against its own air temperature, by era.

    The embankment geometry predicts that heating tips the wall towards the
    mountain, a negative change by the instrument's convention, while the
    compensation subtracts 5 mdeg per degree. The slope of the *recorded*
    channel decides whether that subtraction is correcting the sign or
    producing it.

    This is a coarse estimate and must be reported as one: a single regression
    over a whole era mixes the diurnal response with the annual cycle and with
    whatever drift the record carries, and the correlations are weak. It
    supports only the claim that the recorded slope is positive on both
    instruments and smaller in magnitude than the coefficient the compensation
    removes. It is not a measurement of the instrument's thermal response.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame, carrying ``era`` and the spike marker.
    eras : sequence of str, optional
        Eras to fit, in row order. Default ``('legacy', 'current')``.
    target : str, optional
        Recorded response channel. Default ``'inc'``.
    driver : str, optional
        Temperature channel, already the recording unit's own, era by era.
        Default ``'tair'``.
    spike_column : str, optional
        Boolean column whose samples are excluded from the fit. Default
        ``'inc_spike'``.

    Returns
    -------
    pd.DataFrame
        Indexed by era, carrying ``slope`` in mdeg per degree Celsius, the
        Pearson ``r``, and the number of paired samples ``n``. An era with fewer
        than two paired samples is absent from the table.
    """
    rows = []
    for era in eras:
        era_rows = (df['era'] == era) & ~df[spike_column].astype(bool)
        temperature = df.loc[era_rows, driver]
        inclination = df.loc[era_rows, target]
        both = temperature.notna() & inclination.notna()
        if both.sum() < 2:
            continue
        slope = np.polyfit(temperature[both], inclination[both], 1)[0]
        correlation = float(np.corrcoef(temperature[both],
                                        inclination[both])[0, 1])
        rows.append({'era': era, 'slope': float(slope), 'r': correlation,
                     'n': int(both.sum())})
    return pd.DataFrame(rows).set_index('era')


def residual_and_scale(df, column, start, reference_end, window):
    """
    Departure of a channel from its own rolling median, and its ordinary scale.

    Reducing each channel to its departure from a centred rolling median removes
    the diurnal and seasonal shape and leaves the short-lived part of the signal.
    The scale of that departure is measured before the event under test, so the
    threshold is set by the channel's own ordinary behaviour and the two periods
    are judged on identical terms.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame.
    column : str
        Channel to reduce.
    start : pd.Timestamp
        First timestamp considered. Only one era is used, so that another
        instrument cannot set the scale for a judgement about this one.
    reference_end : pd.Timestamp
        End of the quiet period the scale is measured over.
    window : int
        Length of the centred rolling median, in samples.

    Returns
    -------
    residual : pd.Series
        The series minus its centred rolling median.
    scale : float
        Robust standard deviation of the residual over the quiet period.
    """
    series = df.loc[start:, column].dropna()
    residual = series - series.rolling(window, center=True,
                                       min_periods=1).median()
    scale = 1.4826 * residual.loc[:reference_end].abs().median()
    return residual, scale


def anomaly_by_channel(df, channels, start, anomaly_start, anomaly_end,
                       window, n_sigmas, daylight_hours=(9, 16)):
    """
    Whether each channel carries the anomaly, judged on its own ordinary noise.

    The impulsive noise was found on the inclination. Whether it is a property
    of the wall, of the site, or of the acquisition system is decided by whether
    the other channels carry it over the same days: a real thermal or hygric
    event has to appear in the environmental channels, while a fault in the
    acquisition chain need not appear in them at all.

    Where an excursion falls in the day, and which way it points, separate a
    systematic artefact from noise: random disturbance is spread over the clock
    and symmetric about zero, while something driven by sun or by the power
    system is neither.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame.
    channels : sequence of str
        Channels to judge, in row order.
    start : pd.Timestamp
        First timestamp considered, normally the era boundary.
    anomaly_start, anomaly_end : pd.Timestamp
        Bounds of the anomaly window.
    window : int
        Length of the centred rolling median, in samples.
    n_sigmas : float
        How many robust standard deviations bound ordinary behaviour.
    daylight_hours : tuple of int, optional
        Inclusive hour range counted as daylight. Default ``(9, 16)``.

    Returns
    -------
    stats : pd.DataFrame
        One row per channel: the ``scale``, the share of samples beyond the
        threshold before and during the window, the largest departure in each
        period, and the count, sign and timing of the excursions.
    residuals : dict of str to pd.Series
        The residual of each channel, for plotting.
    limits : dict of str to float
        The threshold of each channel, for plotting.
    excursions : dict of str to pd.DatetimeIndex
        Where each channel's excursions fall, for the coincidence test.
    """
    rows, residuals, limits, excursions = [], {}, {}, {}
    for column in channels:
        residual, scale = residual_and_scale(df, column, start, anomaly_start,
                                             window)
        before = residual.loc[:anomaly_start]
        during = residual.loc[anomaly_start:anomaly_end]
        limit = n_sigmas * scale

        residuals[column] = residual
        limits[column] = limit

        beyond = during[during.abs() > limit]
        excursions[column] = beyond.index
        daylight = (beyond.index.hour.to_series().between(*daylight_hours)
                    if len(beyond) else None)
        rows.append({
            'channel': column,
            'scale': round(float(scale), 3),
            'pct_beyond_before': round(100 * float((before.abs() > limit).mean()), 2),
            'pct_beyond_during': round(100 * float((during.abs() > limit).mean()), 2),
            'max_abs_residual_before': round(float(before.abs().max()), 2),
            'max_abs_residual_during': round(float(during.abs().max()), 2),
            'n_excursions': int(len(beyond)),
            'pct_negative': round(100 * float((beyond < 0).mean()), 1) if len(beyond) else np.nan,
            'pct_in_daylight': round(100 * float(daylight.mean()), 1) if len(beyond) else np.nan,
        })
    return pd.DataFrame(rows), residuals, limits, excursions


def coincidence(first, second, n_slots):
    """
    Whether two channels' excursions keep company, against independence.

    A table that counts each channel separately cannot answer this. The
    comparison is made against what independence would produce: if the two
    channels excurred without reference to each other, the share of the first's
    excursions landing in a slot that also carries one of the second's would be
    the second's own excursion rate.

    Parameters
    ----------
    first, second : pd.DatetimeIndex
        Where each channel's excursions fall.
    n_slots : int
        Slots in the window over which both were counted.

    Returns
    -------
    dict
        ``shared``, ``expected`` under independence, and the ``ratio`` of the
        two.
    """
    shared = first.intersection(second)
    expected = len(first) * len(second) / n_slots if n_slots else np.nan
    return {'shared': int(len(shared)),
            'expected': float(expected),
            'ratio': float(len(shared) / expected) if expected else np.nan}


# ──────────────────────────────────────────────────────────────────────
# The raw-defect census
# ──────────────────────────────────────────────────────────────────────

def service_windows(wide):
    """
    The interval over which each acquisition block was actually in service.

    A block's service window runs from its first surviving measurement to its
    last, within an era. It exists because the acquisition units did not stop
    together: station 01 fell silent in September 2022 and station 03 in June
    2023, while the files went on writing their fields as constant zeros until
    the February 2025 changeover. Measured against the whole legacy era those
    trailing zeros are indistinguishable from downtime, and a unit that died
    early is reported as though it had spent years failing intermittently. Cut
    at the last real reading instead, the count says what it should: how much of
    the time the unit was alive did it fail to record.

    Parameters
    ----------
    wide : pd.DataFrame
        The flagged frame, from :func:`flag_and_correct`.

    Returns
    -------
    dict
        Keyed ``(block, era)``. Each value is a dict carrying ``mask``, a
        boolean Series true on the slots of that era inside the window;
        ``slots``, its length; and ``first`` and ``last``, the bounding
        timestamps. Blocks that never reported in an era are absent.
    """
    windows = {}
    for block in BLOCKS:
        observed = pd.concat(
            [wide[f'{block}_{channel}_ok'] for channel in BLOCK_CHANNELS[block]],
            axis=1).notna().any(axis=1)
        for era in ('legacy', 'current'):
            in_era = wide['era'] == era
            live = observed & in_era
            if not live.any():
                continue
            first, last = wide.index[live].min(), wide.index[live].max()
            mask = in_era & (wide.index >= first) & (wide.index <= last)
            windows[(block, era)] = {
                'mask': mask,
                'slots': int(mask.sum()),
                'first': first,
                'last': last,
            }
    return windows


def sentinel_census(wide, text_stats=None, parse_stats=None):
    """
    Measure every documented defect of the archive, block by block and channel
    by channel.

    The raw-data specification states which values are sentinels and which are
    artefacts; this counts them. Two kinds of defect appear. Those visible in
    the values — the ``0.000`` zero-blocks, the ``-55.0`` probe failure, the
    radiation wrap-around and the unphysical radiation below it — are counted
    from the flag columns. Those that no parser can pass on, because a malformed
    record has to be skipped to be handled at all, are counted by
    :func:`scan_raw_text` and :func:`parse_era` and are reported here under the
    pseudo-channel ``file``.

    Channel rows are confined to the reporting block's service window, as
    returned by :func:`service_windows`, and their percentages are taken against
    it. File-level rows are properties of the text rather than of any one unit,
    so their denominator remains the records the era contributed.

    Parameters
    ----------
    wide : pd.DataFrame
        The flagged frame, from :func:`flag_and_correct`.
    text_stats : dict, optional
        Output of :func:`scan_raw_text`. Adds the file-level rows.
    parse_stats : dict of dict, optional
        Era name to the stats returned by :func:`parse_era`. Adds the duplicate
        and conflict rows.

    Returns
    -------
    pd.DataFrame
        One row per (era, block, channel, code): ``n`` samples affected,
        ``pct_of_service`` against the denominator described above, ``days``
        distinct calendar days touched, ``first`` and ``last`` timestamp of the
        defect, and ``service_first`` and ``service_last`` bounding the window
        the row was measured over.
    """
    rows = []
    windows = service_windows(wide)

    for column in RAW_COLUMNS:
        block, channel = column.split('_', 1)
        flags = wide[f'{column}_flag']
        for code in REJECTION_CODES + CORRECTION_CODES:
            hit = flags == code
            if not hit.any():
                continue
            kind = 'reject' if code in REJECTION_CODES else 'correct'
            for era in ('legacy', 'current'):
                window = windows.get((block, era))
                if window is None:
                    continue
                mask = hit & window['mask']
                if not mask.any():
                    continue
                stamps = wide.index[mask]
                denominator = window['slots']
                rows.append({
                    'era': era,
                    'block': block,
                    'channel': channel,
                    'code': code,
                    'kind': kind,
                    'n': int(mask.sum()),
                    'pct_of_service': round(100 * mask.sum() / denominator, 2)
                    if denominator else np.nan,
                    'days': int(len(np.unique(stamps.date))),
                    'first': str(stamps.min()),
                    'last': str(stamps.max()),
                    'service_first': str(window['first']),
                    'service_last': str(window['last']),
                })

    if parse_stats:
        for era, stats in parse_stats.items():
            denominator = stats.get('records', 0)
            for code, key in (('duplicate_timestamp', 'duplicates'),
                              ('field_conflict', 'conflicts')):
                if not stats.get(key):
                    continue
                rows.append({
                    'era': era, 'block': 'file', 'channel': 'record',
                    'code': code, 'kind': 'parse', 'n': int(stats[key]),
                    'pct_of_service': round(100 * stats[key] / denominator, 2)
                    if denominator else np.nan,
                    'days': np.nan, 'first': '', 'last': '',
                    'service_first': '', 'service_last': '',
                })

    if text_stats:
        # Only the rejected-record count is a share of the records read. The
        # separator counts are per numeric field and per file respectively, and
        # a percentage against a line count would be meaningless for both.
        lines = text_stats.get('lines', 0)
        for code, key, denominator in (
                ('rejected_field_count', 'rejected', lines),
                ('comma_separator', 'comma_fields', None),
                ('mixed_separator_file', 'mixed_separator_files', None)):
            if not text_stats.get(key):
                continue
            rows.append({
                'era': 'archive', 'block': 'file', 'channel': 'text',
                'code': code, 'kind': 'parse', 'n': int(text_stats[key]),
                'pct_of_service': round(100 * text_stats[key] / denominator, 2)
                if denominator else np.nan,
                'days': np.nan, 'first': '', 'last': '',
                'service_first': '', 'service_last': '',
            })

    return pd.DataFrame(rows)


def clock_cell(row, key):
    """
    A phase hour formatted for a table, or nothing where there is no cycle.

    A season with no complete day has no mean cycle, so it has no peak hour and
    no trough hour. Returning nothing rather than a formatted zero lets the
    table print its missing marker there.

    Parameters
    ----------
    row : pd.Series
        A row of a diurnal-cycle statistics table, carrying ``amplitude`` and
        the requested phase column.
    key : str
        Which phase to format, ``'peak_hour'`` or ``'trough_hour'``.

    Returns
    -------
    str or None
        The hour as ``HH:MM``, or ``None`` where the cycle does not exist.
    """
    if pd.isna(row['amplitude']) or pd.isna(row[key]):
        return None
    return format_clock(row[key])


def service_span(row):
    """
    The dated span of a census row, as one cell of text.

    Parameters
    ----------
    row : pd.Series
        A row of :func:`sentinel_census`, carrying ``first`` and ``last``.

    Returns
    -------
    str or float
        ``'YYYY-MM-DD to YYYY-MM-DD'``, or ``nan`` for a file-level row, which
        counts records rather than samples and therefore has no span. The
        missing value is what makes the table print its missing marker there.
    """
    if not row['first']:
        return np.nan
    return f'{row["first"][:10]} to {row["last"][:10]}'


# ──────────────────────────────────────────────────────────────────────
# Persisting the record
# ──────────────────────────────────────────────────────────────────────

def export_column_doc(station):
    """
    What each column of the exported table means, for the manifest.

    The schema of the export is a property of this library rather than of any
    run of the notebook, so its documentation lives beside the exporter. What
    the notebook contributes is the run: the parameters, the counts and the
    extent, which it records in the manifest around this block.

    Parameters
    ----------
    station : str
        The target station, named in the description of the coalesced columns.

    Returns
    -------
    dict of str to str
        Column pattern to description.
    """
    return {
        '{block}_{channel}': 'the field as recorded, sentinels included',
        '{block}_{channel}_flag': 'what was decided about the value, empty if it stands. '
                                  'Rejections: sentinel, wrap, unphysical. '
                                  'Corrections: night',
        '{block}_{channel}_ok': 'missing exactly where a rejection is set; carries the '
                                'substituted value where a correction is',
        'n_sr:night': 'radiation recorded below %.0f degrees of solar elevation, set to '
                      'zero. Its recorded value is preserved in n_sr'
                      % NIGHT_ELEVATION,
        '{block}_inc': 'inclination in mdeg about the calibrated zero',
        '{block}_inc_comp': 'thermally compensated, anchored within the block',
        'inc': f'{station} across both eras, as recorded by whichever instrument',
        'inc_comp': 'compensated, current-era offset removed. Carries an estimate',
        'inc_comp_cleaned': 'compensated, offset removed, impulsive noise filtered',
        'inc_spike': 'True where inc_comp_cleaned is an interpolation, not a measurement',
        'n_twall_filtered': 'n_twall_ok with impulsive anomalies in the final recording '
                            'segment replaced by linear interpolation',
        'twall_spike': 'True where n_twall_filtered is an interpolation, not a measurement',
        'sr_suspect': 'True on days whose high-sun radiation does not exceed their dark '
                      'radiation by a factor of %g; the whole day is condemned, see Step 7'
                      % SR_DAY_RATIO,
        'era': "legacy or current, from the record's own field count. Empty where "
               'nothing was recorded, so the column says which layout produced a '
               'slot rather than which era the calendar puts it in',
    }


def export_columns(wide):
    """
    The column order of the exported table.

    Grouped so that the file reads as what it is: the record as written, then
    the verdict on each value, then the corrected value, then what was derived
    from them.

    Parameters
    ----------
    wide : pd.DataFrame
        The finished frame.

    Returns
    -------
    list of str
        Column names present in ``wide``, in export order.
    """
    order = ['era']
    order += RAW_COLUMNS
    order += [f'{column}_flag' for column in RAW_COLUMNS]
    order += [f'{column}_ok' for column in RAW_COLUMNS]
    for block in BLOCKS:
        order += [f'{block}_inc', f'{block}_inc_comp']
    order += ['inc', 'tair', 'inc_comp', 'inc_comp_cleaned',
              'inc_spike', 'n_twall_filtered', 'twall_spike', 'sr_suspect']
    return [column for column in order if column in wide.columns]


def save_archive(wide, out_dir, manifest, filename='gubbio_archive_20min.csv',
                 compress=False, float_format='%.4g'):
    """
    Write the archive table and the manifest that describes how it was built.

    Parameters
    ----------
    wide : pd.DataFrame
        The finished frame.
    out_dir : str
        Destination directory. Created if absent.
    manifest : dict
        Build record, written beside the table as JSON.
    filename : str, optional
        Table filename. Default ``'gubbio_archive_20min.csv'``.
    compress : bool, optional
        Append ``.gz`` and write compressed. Default ``False``.
    float_format : str, optional
        Passed to ``to_csv``. The default keeps four significant digits, which
        is more precision than any channel here carries and avoids writing
        seventeen digits of floating-point noise for every cell.

    Returns
    -------
    dict
        ``table`` and ``manifest`` paths, and the table's size in megabytes.

    Notes
    -----
    Writes two files. Nothing in the raw archive is touched.
    """
    import json

    os.makedirs(out_dir, exist_ok=True)
    if compress and not filename.endswith('.gz'):
        filename += '.gz'

    table_path = os.path.join(out_dir, filename)
    export = wide[export_columns(wide)].copy()
    export.index.name = 'datetime'
    export.to_csv(table_path, float_format=float_format)

    stem = filename.replace('.csv.gz', '').replace('.csv', '')
    manifest_path = os.path.join(out_dir, f'{stem}_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, default=str)

    return {'table': table_path, 'manifest': manifest_path,
            'size_mb': round(os.path.getsize(table_path) / 1e6, 1)}


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
#
# Every figure this study publishes is built here and displayed by the
# notebook. Each function takes the data and the choices, draws, optionally
# saves, and returns the figure; none of them reads a path of its own or reaches
# for a module-level default. Colour comes from :mod:`shmlib.viz`, where a
# channel's identity is fixed for the whole project.
#
# These are this study's figures rather than the project's. A figure a second
# study also needs is promoted to ``shmlib`` when that study needs it, not in
# anticipation of it.

#: Figure width in inches: the manuscript column width every report figure is
#: exported at.
#:
#: Moved to ``shmlib.viz``: study 2 needs the identical width. Kept here as an
#: alias at the original name, so every call site in this module is
#: unaffected.
FIGURE_WIDTH = viz.FIGURE_WIDTH


# `_complete_days` and `_draw_cycle` moved to `shmlib.viz` as `complete_days`
# and `draw_cycle`: both state a fact about how to average a diurnal cycle
# correctly, not a decision this study is making, and study 2 needs both.
# Kept here as aliases, at the name every call site in this module already
# uses, so behaviour and defaults are unchanged.
_complete_days = viz.complete_days
_draw_cycle = viz.draw_cycle

# `robust_limits` moved to `shmlib.viz` for the same reason: clipping an axis to
# the bulk of a record, and reporting how many samples that leaves outside, is a
# display convention of the project rather than a decision this study makes.
robust_limits = viz.robust_limits


def diurnal_cycle_grid(df, column, changeover, season_months, complete_day,
                       ylabel, suptitle, colour=None, seasons=('summer', 'winter'),
                       eras=('legacy', 'current'), save_path=None, filename=None):
    """
    Average diurnal cycle of one channel, by instrument era and season.

    A grid with the eras across the columns and the seasons down the rows.
    Pooling the eras would average two unrelated instrument baselines, so they
    are always drawn apart.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame, spanning both eras.
    column : str
        Channel to analyse.
    changeover : pd.Timestamp
        First timestamp of the current era.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    complete_day : int
        Slots a day must carry to enter the average.
    ylabel : str
        Vertical axis label, including the unit.
    suptitle : str
        Figure title.
    colour : str, optional
        Colour of the mean cycle. Default ``None``, meaning the channel's own
        identity colour.
    seasons : sequence of str, optional
        Seasons, down the rows. Default ``('summer', 'winter')``.
    eras : sequence of str, optional
        Eras, across the columns. Default ``('legacy', 'current')``.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, for the notebook to display.
    stats : pd.DataFrame
        One row per era and season: the number of complete days, the
        peak-to-trough amplitude of the mean cycle, and the hours at which it
        peaks and troughs.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    colour = colour or viz.CHANNEL_COLOUR[column]
    viz.apply_report_style()
    fig, axes = plt.subplots(len(seasons), len(eras),
                             figsize=(FIGURE_WIDTH, 4.5),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    era_frames = {'legacy': df[df.index < changeover].dropna(subset=[column]),
                  'current': df[df.index >= changeover].dropna(subset=[column])}

    rows = []
    for row_index, season in enumerate(seasons):
        for col_index, era in enumerate(eras):
            ax = axes[row_index, col_index]
            source = era_frames[era]
            data = source[source.index.month.isin(season_months[season])]
            complete, days = _complete_days(data, column, complete_day)

            if days:
                centred, amplitude, _ = _draw_cycle(ax, complete, column,
                                                   colour)
                ax.set_title(f'{era.capitalize()} {season}\n'
                             f'({days} complete days, amplitude {amplitude:.1f})',
                             fontsize='small')
                rows.append({'channel': column, 'era': era, 'season': season,
                             'complete_days': days,
                             'amplitude': round(float(amplitude), 2),
                             'peak_hour': float(centred.idxmax()),
                             'trough_hour': float(centred.idxmin())})
            else:
                ax.set_title(f'{era.capitalize()} {season}\n(0 complete days)',
                             fontsize='small')
                rows.append({'channel': column, 'era': era, 'season': season,
                             'complete_days': 0, 'amplitude': np.nan,
                             'peak_hour': np.nan, 'trough_hour': np.nan})

            ax.set_xticks(range(0, 25, 4))
            ax.grid(True, alpha=0.3)
            viz.format_spines(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel('Hour of day')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    fig.suptitle(suptitle, y=1.02, fontweight='bold')
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig, pd.DataFrame(rows)


def diurnal_cycle_pair(current, column, season_months, complete_day, ylabel,
                       suptitle, seasons=('summer', 'winter'), valid=None,
                       save_path=None, filename=None):
    """
    Average diurnal cycle of a current-era channel, by season.

    The current-era counterpart of :func:`diurnal_cycle_grid`: one row of
    panels, one per season, with no era split because the channel exists in only
    one era.

    Parameters
    ----------
    current : pd.DataFrame
        The current era of a station's tidy frame.
    column : str
        Channel to analyse.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    complete_day : int
        Slots a day must carry to enter the average.
    ylabel : str
        Vertical axis label, including the unit.
    suptitle : str
        Figure title.
    seasons : sequence of str, optional
        Seasons to draw, one panel each. Default ``('summer', 'winter')``.
    valid : pd.Series of bool, optional
        Boolean over the index of ``current``, true on the slots eligible to
        enter the average. Days it excludes are dropped before the completeness
        test, so a day the mask rejects cannot become a complete day. Default
        ``None``, meaning every slot is eligible.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    stats : pd.DataFrame
        One row per season with a cycle: complete days, amplitude, peak hour and
        trough hour. A season with no complete day contributes no row.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, axes = plt.subplots(1, len(seasons), figsize=(FIGURE_WIDTH, 2.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    source = current if valid is None else current[valid]
    rows = []
    for ax, season in zip(axes, seasons):
        data = source[source.index.month.isin(season_months[season])].dropna(
            subset=[column])
        complete, days = _complete_days(data, column, complete_day)

        if days:
            centred, amplitude, _ = _draw_cycle(
                ax, complete, column, viz.CHANNEL_COLOUR[column],
                background_alpha=0.25)
            ax.set_title(f'{season.capitalize()}\n({days} complete days, '
                         f'amplitude {amplitude:.1f})', fontsize='small')
            rows.append({'channel': column, 'season': season,
                         'complete_days': days,
                         'amplitude': round(float(amplitude), 2),
                         'peak_hour': cycle_extreme(centred, 'max'),
                         'trough_hour': cycle_extreme(centred, 'min')})
        else:
            ax.set_title(f'{season.capitalize()}\n(0 complete days)',
                         fontsize='small')
        ax.set_xticks(range(0, 25, 6))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Hour of day')
        viz.format_spines(ax)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(suptitle, y=1.04, fontweight='bold')
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig, pd.DataFrame(rows)


def plot_channel_series(df, column, changeover, title, save_path=None,
                        filename=None):
    """
    One full-archive plot of a single channel, in that channel's colour.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame.
    column : str
        Channel to plot.
    changeover : pd.Timestamp
        Instrument changeover, marked in the accent colour.
    title : str
        Axes title.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.0))
    ax.plot(df.index, df[column], **viz.channel_style(column, 0.4))
    ax.axvline(changeover, color=viz.MARK_COLOUR, lw=1.0, ls='--')
    ax.set_ylabel(viz.channel_unit(column))
    ax.set_title(title)
    viz.format_spines(ax)
    viz.finish(fig, save_path, filename)
    return fig


def plot_current_era_series(current, column, title, highlight_start=None,
                            save_path=None, filename=None):
    """
    One plot of a current-era channel over the era it exists in.

    Unlike :func:`plot_channel_series` this draws no changeover marker: the era
    begins at the changeover, so the line would sit on the axis.

    Parameters
    ----------
    current : pd.DataFrame
        The current era of a station's tidy frame.
    column : str
        Channel to plot.
    title : str
        Axes title.
    highlight_start : str or pd.Timestamp, optional
        Shade from this timestamp to the end of the era. Default ``None``.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.0))
    ax.plot(current.index, current[column], **viz.channel_style(column, 0.4))
    if highlight_start is not None:
        ax.axvspan(pd.Timestamp(highlight_start), current.index.max(),
                   **viz.SPAN_STYLE)
    ax.set_ylabel(viz.channel_unit(column))
    ax.set_title(title)
    viz.format_spines(ax)
    viz.finish(fig, save_path, filename)
    return fig


def plot_coverage_heatmap(daily, title, tick_when, tick_format, height=2.23,
                          marker=None, save_path=None, filename=None):
    """
    Daily coverage of several channels or stations, as a heatmap.

    One strip per column of ``daily``, showing the fraction of each day
    observed. The colormap is Cividis reversed, so that present data reads as
    ink and absent data as empty page, and a day outside a strip's range is left
    white rather than coloured as zero coverage.

    Parameters
    ----------
    daily : pd.DataFrame
        Indexed by day, one column per strip, values between 0 and 1.
    title : str
        Axes title.
    tick_when : callable
        Takes a day and returns whether it should carry a tick. This is how the
        axis is labelled by year, by quarter, or by anything else, without the
        function having to guess the span.
    tick_format : str
        ``strftime`` format for the tick labels.
    height : float, optional
        Figure height in inches. Default 2.23.
    marker : pd.Timestamp, optional
        Draw a vertical accent line at the nearest day to this timestamp.
        Default ``None``.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    cmap = sns.color_palette('cividis_r', as_cmap=True)
    cmap.set_bad('white')
    sns.heatmap(daily.T, cmap=cmap, cbar_kws={'label': 'Coverage Fraction'},
                ax=ax, rasterized=True, vmin=0, vmax=1)

    ax.set_title(title, pad=15 if marker is not None else 12,
                 fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')

    ticks = [i for i, day in enumerate(daily.index) if tick_when(day)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([daily.index[i].strftime(tick_format) for i in ticks],
                       rotation=0)

    if marker is not None:
        position = daily.index.get_indexer([pd.to_datetime(marker)],
                                           method='nearest')[0]
        ax.axvline(position, color=viz.MARK_COLOUR, lw=1.5, ls='--', zorder=10)

    plt.tight_layout()
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    viz.finish(fig, save_path, filename)
    return fig


def plot_inclination_series(df, column, changeover, span_start, limits,
                            title=None, note=None, height=2.4,
                            save_path=None, filename=None):
    """
    The inclination over the whole archive, on a clipped vertical axis.

    A handful of very large excursions would otherwise compress eight years of
    record into a thin band. The clipping is a display choice only: nothing is
    removed from the data or from any computed statistic, and the note names how
    many samples fall outside the visible range and how deep the deepest goes.

    Parameters
    ----------
    df : pd.DataFrame
        A station's tidy frame.
    column : str
        Inclination column to draw.
    changeover : pd.Timestamp
        Instrument changeover, marked in the accent colour.
    span_start : pd.Timestamp
        Start of the highlighted interval, which runs to the end of the record.
    limits : tuple of float
        Vertical axis limits, normally from :func:`robust_limits`.
    title : str, optional
        Axes title. Default ``None``, meaning no title.
    note : str, optional
        Text placed inside the axes in the accent colour, explaining what the
        clipping hides. Default ``None``.
    height : float, optional
        Figure height in inches. Default 2.4.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    ax.plot(df.index, df[column], lw=0.4, color=viz.INC_COLOUR)
    ax.axvline(changeover, color=viz.MARK_COLOUR, lw=1.0, ls='--')
    ax.axvspan(span_start, df.index[-1], **viz.SPAN_STYLE)
    ax.set_ylabel('[mdeg]')
    if title:
        ax.set_title(title)
    ax.set_ylim(limits)

    if note:
        ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize='small',
                color=viz.MARK_COLOUR, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none',
                          pad=2))

    viz.format_spines(ax)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig


def plot_anomaly_comparison(reference, series, title, series_label,
                            reference_label, linewidth=0.8, height=3.0,
                            day_interval=5, save_path=None, filename=None):
    """
    The anomaly window against the same calendar window of earlier years.

    Parameters
    ----------
    reference : pd.Series
        The historical average, on the timestamps of ``series``.
    series : pd.Series
        The anomaly year.
    title : str
        Axes title.
    series_label, reference_label : str
        Legend entries.
    linewidth : float, optional
        Width of the anomaly line. Default 0.8.
    height : float, optional
        Figure height in inches. Default 3.0.
    day_interval : int, optional
        Spacing of the date ticks, in days. Default 5.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    The historical average is drawn in gray: it is a reference rather than a
    measured channel, so it carries no channel identity.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    ax.plot(reference.index, reference.values, lw=1.5, color='0.6',
            label=reference_label)
    ax.plot(series.index, series.values, lw=linewidth, color=viz.INC_COLOUR,
            alpha=0.8, label=series_label)

    ax.set_ylabel('[mdeg]')
    ax.set_title(title)
    # Below the axes, per the Graphical Guidelines.
    ax.legend(fontsize='small', ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    viz.format_spines(ax)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig


def plot_anomaly_channels(residuals, limits, channels, labels, anomaly_start,
                          anomaly_end, n_sigmas, suptitle, xlabel,
                          day_interval=7, height=6.4, save_path=None,
                          filename=None):
    """
    Each channel's departure from its own rolling median, over the anomaly.

    One panel per channel, each with its own threshold band, so that channels in
    different units are judged on identical terms: whether the window departs
    from what that channel ordinarily does.

    Parameters
    ----------
    residuals : dict of str to pd.Series
        Residual of each channel, from :func:`anomaly_by_channel`.
    limits : dict of str to float
        Threshold of each channel, from the same call.
    channels : sequence of str
        Channels to draw, in panel order.
    labels : dict of str to str
        Axis label of each channel, as ``Name [unit]``.
    anomaly_start, anomaly_end : pd.Timestamp
        Bounds of the window drawn.
    n_sigmas : float
        Threshold in robust standard deviations, quoted in each panel title.
    suptitle : str
        Figure title.
    xlabel : str
        Label of the shared horizontal axis.
    day_interval : int, optional
        Spacing of the date ticks, in days. Default 7.
    height : float, optional
        Figure height in inches. Default 6.4.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, axes = plt.subplots(len(channels), 1, figsize=(FIGURE_WIDTH, height),
                             sharex=True)

    for ax, column in zip(np.atleast_1d(axes), channels):
        residual, limit = residuals[column], limits[column]
        before = residual.loc[:anomaly_start]
        during = residual.loc[anomaly_start:anomaly_end]

        ax.axhspan(-limit, limit, zorder=0, **viz.SPAN_STYLE)
        ax.plot(during.index, during.values, lw=0.5,
                color=viz.CHANNEL_COLOUR[column], zorder=2)
        ax.axhline(0, color='0.4', lw=0.6, zorder=1)
        ax.set_ylabel(labels[column])
        ax.set_title(f'{labels[column].split(" [")[0]}: '
                     f'{100 * (during.abs() > limit).mean():.1f}% beyond '
                     f'{n_sigmas}$\\sigma$, against '
                     f'{100 * (before.abs() > limit).mean():.1f}% before the anomaly',
                     fontsize='small')
        viz.format_spines(ax)

    last = np.atleast_1d(axes)[-1]
    last.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
    last.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    last.set_xlabel(xlabel)
    fig.suptitle(suptitle, y=1.01, fontweight='bold')
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig


def plot_twall_raw(wide, title, height=2.2, save_path=None, filename=None):
    """
    The wall-temperature channel as written, sentinel included.

    The corrected record shows a gap where the probe failed. The raw record
    shows what is actually in the archive over those months: the open-circuit
    sentinel, on every sample. The probe did not stop reporting — it reported
    its own failure, continuously — and only the raw channel distinguishes that
    from an acquisition unit that was switched off. The sentinel stretch is
    drawn in the channel's own colour at a heavier weight, and the interval it
    covers is highlighted.

    Parameters
    ----------
    wide : pd.DataFrame
        The flagged archive table.
    title : str
        Axes title.
    height : float, optional
        Figure height in inches. Default 2.2.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    # The era is judged by what was recorded rather than by how many slots the
    # calendar holds, so the stretch drawn here is the one the census counts.
    recorded = wide['era'] == 'current'
    raw = wide.loc[recorded, 'n_twall']
    sentinel = wide.loc[recorded, 'n_twall_flag'] == FLAG_SENTINEL

    ax.plot(raw.index, raw.where(~sentinel), **viz.channel_style('twall', 0.6))
    ax.plot(raw.index, raw.where(sentinel), color=viz.CHANNEL_COLOUR['twall'],
            lw=1.8, solid_capstyle='butt')

    first, last = raw.index[sentinel][[0, -1]]
    ax.axvspan(first, last, **viz.SPAN_STYLE)

    ax.set_ylabel(viz.channel_unit('twall'))
    ax.set_title(title)
    viz.format_spines(ax)
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig


def plot_segment(segment, column, title, flagged_column=None, flag_label=None,
                 linewidth=0.8, height=3.0, day_interval=7, save_path=None,
                 filename=None):
    """
    One recording segment, optionally marking the samples an impulse test flagged.

    Parameters
    ----------
    segment : pd.DataFrame
        The rows of the segment.
    column : str
        Channel to draw.
    title : str
        Axes title.
    flagged_column : str, optional
        Boolean column marking rejected samples, drawn as accent crosses.
        Default ``None``, which marks nothing.
    flag_label : str, optional
        Legend entry for the markers, which may name their count. Required when
        ``flagged_column`` is given.
    linewidth : float, optional
        Width of the data line. Default 0.8.
    height : float, optional
        Figure height in inches. Default 3.0.
    day_interval : int, optional
        Spacing of the date ticks, in days. Default 7.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    ax.plot(segment.index, segment[column],
            **viz.channel_style(column, linewidth))

    if flagged_column is not None:
        flagged = segment[segment[flagged_column]]
        ax.scatter(flagged.index, flagged[column], color=viz.MARK_COLOUR,
                   marker='x', s=34, linewidths=1.2, zorder=4,
                   label=flag_label)
        # Below the axes, per the Graphical Guidelines.
        ax.legend(fontsize='small', ncol=2, loc='upper center',
                  bbox_to_anchor=(0.5, -0.30), frameon=False)

    ax.set_ylabel(viz.channel_unit(column))
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    viz.format_spines(ax)
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig


def plot_sr_raw(wide, start, fail_start, recovery, title, height=2.2,
                axis_step=500.0, save_path=None, filename=None):
    """
    The radiation channel as written, over the era it exists in.

    Sliced by date rather than by the era mask: the mask drops the slots the
    acquisition unit never wrote, which closes the index over every outage and
    lets the line join two points months apart as though the record ran between
    them. Keeping the empty slots keeps the gaps.

    The wrap-around samples are isolated, and a line joining a point whose two
    neighbours are both missing draws nothing at all — which is why they were
    invisible while the axis they force was not. They carry a dot instead, in
    the channel's own colour: they are readings of the same channel, and the
    accent is reserved for annotations.

    Parameters
    ----------
    wide : pd.DataFrame
        The flagged archive table.
    start : pd.Timestamp
        First timestamp drawn, normally the changeover.
    fail_start, recovery : pd.Timestamp
        Bounds of the highlighted failure window.
    title : str
        Axes title.
    height : float, optional
        Figure height in inches. Default 2.2.
    axis_step : float, optional
        The vertical axis is rounded up to a multiple of this, so the
        wrap-around is shown at the magnitude it actually claims. Default 500.0.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    raw = wide.loc[start:, 'n_sr']
    wrap = wide.loc[start:, 'n_sr_flag'] == FLAG_WRAP

    # The whole failure, marked once. Its internal structure is the subject of
    # the next figure.
    ax.axvspan(fail_start, recovery, **viz.SPAN_STYLE)

    ax.plot(raw.index, raw.where(~wrap), **viz.channel_style('sr', 0.6))
    ax.plot(raw.index[wrap], raw[wrap], linestyle='none', marker='.',
            markersize=3, **viz.channel_style('sr', 0.6))

    ax.set_ylim(0, float(np.ceil(raw.max() / axis_step) * axis_step))
    ax.set_ylabel(viz.channel_unit('sr'))
    ax.set_title(title)
    viz.format_spines(ax)
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig


def plot_sr_failure(wide, state, fail_start, recovery, title, short_labels,
                    margin_days=10, height=2.4, axis_step=500.0,
                    save_path=None, filename=None):
    """
    The radiation failure at close range, with its phases named.

    One label per phase, placed over the longest unbroken run of that phase.
    Labelling every run would crowd the axes, and the states interleave at their
    boundaries: the pegged window sits inside the span of the stuck days, so a
    label at the midpoint of the whole state would point at the wrong stretch of
    record.

    Parameters
    ----------
    wide : pd.DataFrame
        The flagged archive table.
    state : pd.Series
        One state per day of the window, from :func:`sr_day_states`.
    fail_start, recovery : pd.Timestamp
        The two events that bound the failure: the first condemned day, and the
        first day the channel passes again. Both drawn in the accent colour.
    title : str
        Axes title.
    short_labels : dict of str to str
        Short name for each state that should be labelled. A state absent from
        this mapping is drawn but not named.
    margin_days : int, optional
        Days of context drawn either side of the failure. Default 10.
    height : float, optional
        Figure height in inches. Default 2.4.
    axis_step : float, optional
        The vertical axis is rounded up to a multiple of this. Default 500.0.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    margin = pd.Timedelta(days=margin_days)
    view = wide.loc[fail_start - margin:recovery + margin]
    ax.plot(view.index, view['n_sr'], **viz.channel_style('sr', 0.7))

    for edge in (fail_start, recovery):
        ax.axvline(edge, color=viz.MARK_COLOUR, lw=0.9, ls='--', zorder=3)

    top = float(np.ceil(view['n_sr'].max() / axis_step) * axis_step)
    ax.set_ylim(0, top)

    runs = (state != state.shift()).cumsum()
    for name, label in short_labels.items():
        blocks = [days for _, days in state[state == name].groupby(runs)]
        if not blocks:
            continue
        longest = max(blocks, key=len)
        middle = (longest.index.min()
                  + (longest.index.max() - longest.index.min()) / 2)
        ax.text(middle, top * 0.94, label, ha='center', va='top', fontsize=7,
                color=viz.MARK_COLOUR)

    ax.set_ylabel(viz.channel_unit('sr'))
    ax.set_title(title)
    viz.format_spines(ax)
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig


def phase_chain(current, channels, season_months, complete_day, season,
                valid, title, height=3.3, save_path=None, filename=None):
    """
    The four quantities of the thermal chain, on one clock and one scale.

    The sun that arrives, the air it warms, the masonry the air and the sun
    together heat, and the deformation that follows. Each cycle is centred and
    then divided by its own largest excursion, so that four different units can
    share an axis: the figure is about *when*, not about how much.

    One day set is used for all four channels rather than one per channel. The
    figure compares the timing of four cycles against each other, and that
    comparison is only meaningful if they are the same days — a channel averaged
    over days its neighbours were missing is answering a slightly different
    question. The set is the days complete in every channel at once, drawn from
    the days whose radiation is usable, so the sun the other three are compared
    against is a measurement throughout.

    Parameters
    ----------
    current : pd.DataFrame
        The current era of a station's tidy frame.
    channels : sequence of str
        Channels of the chain, in drawing order.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    complete_day : int
        Slots a day must carry to be complete.
    season : str
        Season to draw.
    valid : pd.Series of bool
        Boolean over the index of ``current``, true on slots whose radiation may
        be used.
    title : str
        Axes title.
    height : float, optional
        Figure height in inches. Default 3.3.
    save_path, filename : str, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    stats : pd.DataFrame
        One row per channel: the number of common complete days, the amplitude
        in the channel's own unit, and the peak and trough hours.
    days : pd.DatetimeIndex
        The days every channel is complete on.

    Notes
    -----
    Writes two image files when ``save_path`` and ``filename`` are given.
    """
    viz.apply_report_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, height))

    season_data = current[valid & current.index.month.isin(season_months[season])]

    days = None
    for column in channels:
        data = season_data.dropna(subset=[column])
        counts = data.groupby(data.index.normalize())[column].count()
        complete = pd.DatetimeIndex(counts[counts == complete_day].index)
        days = complete if days is None else days.intersection(complete)

    common = season_data[season_data.index.normalize().isin(days)]

    rows = []
    for column in channels:
        cycle = common.groupby(time_of_day(common.index))[column].mean()
        centred = cycle - cycle.mean()
        normalised = centred / centred.abs().max()
        ax.plot(normalised.index, normalised.values,
                label=viz.channel_name(column),
                **viz.channel_style(column, 2.0))
        rows.append({'channel': column, 'complete_days': len(days),
                     'amplitude': round(float(cycle.max() - cycle.min()), 2),
                     'peak_hour': cycle_extreme(centred, 'max'),
                     'trough_hour': cycle_extreme(centred, 'min')})

    ax.axhline(0, color='0.4', lw=0.6, zorder=0)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Cycle, scaled to its own peak')
    ax.set_title(title)
    # Below the axes: every corner inside them is crossed by one of the cycles.
    ax.legend(fontsize='small', ncol=len(channels), loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    ax.grid(True, alpha=0.3)
    viz.format_spines(ax)
    plt.tight_layout()
    viz.finish(fig, save_path, filename)
    return fig, pd.DataFrame(rows), days
