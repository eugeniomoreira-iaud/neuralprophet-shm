"""
Module: shmlib.proxies

The external forcings that drive the wall, in every source that carries them:
what a source is, the canonical channel vocabulary that names its quantities
regardless of which file they came from, the file formats of the three sources
this project has retrieved, and the loaders that put each of them on a shared
analysis grid under that vocabulary.

Three properties govern the design, carried over unchanged from study 2, the
first study to need this module.

**The sensor side is study 1's, unaltered.** The on-structure loader reads
``data/interim/archive/gubbio_archive_20min.csv``, the product of
``studies/01_data_exploration/``. It does not open an ``.adc`` file, does not
re-decide a rejection, and does not recompute a correction. It consumes the
``_ok`` columns and honours the ``sr_suspect`` verdict by default. Where a study
needs to depart from one of those decisions it does so visibly, from the
recorded column that sits beside it.

**The sources are not replicates.** Air inside a sun-exposed instrument
housing, a standard-exposure screen in a town, and an average over a
nine-kilometre reanalysis grid cell are three different physical quantities
that happen to share a name. Their disagreement is a finding, not a defect to
be corrected away, and no source is rescaled onto another here.

**A source is characterised on its own terms before it is compared with
anything.** :func:`source_inventory` and :func:`channel_inventory` measure a
source's own extent, coverage and range; nothing here computes how two sources
agree — that is :mod:`shmlib.compare` — nor which day a channel is fit to be
trusted — that is :mod:`shmlib.quality`, which this module's on-structure
loader consumes the verdicts of but never decides.

Facts about the site — the eras, the logger's clock, the solar geometry and the
site coordinates — come from :mod:`shmlib.site` and :mod:`shmlib.solar`. Nothing
here imports from ``studies/obsolete/``.

Inputs are file paths a study's parameter cell supplies; outputs are
DataFrames. No function here reads a path of its own or writes one that was not
passed to it.
"""

import numpy as np
import pandas as pd

from . import adc, meteo, site


# ──────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────

#: The sources compared, by the suffix that names them in every column of this
#: study. Canonical names and provenance are in ``docs/proxy-data-dictionary.md``.
SOURCES = ('str', 'gs', 'era5')

#: Human-readable name of each source, for figure legends and table headers.
SOURCE_LABEL = {
    'str': 'On-structure',
    'gs': 'Ground station',
    'era5': 'ERA5',
}

#: What each source physically measures. Quoted wherever a figure invites the
#: reader to treat the three as interchangeable, which they are not.
SOURCE_DESCRIPTION = {
    'str': 'Air inside the instrument housing on the monitored wall',
    'gs': 'Standard-exposure weather station in Gubbio town',
    'era5': 'Reanalysis average over a grid cell of roughly nine kilometres',
}

#: The nature of each source, for the inventory table that opens the report.
SOURCE_NATURE = {
    'str': 'Instrument housing on the monitored wall',
    'gs': 'Ground-based meteorological station in Gubbio town',
    'era5': 'Gridded atmospheric reanalysis',
}

#: Native sampling interval of each source, as documented by its retrieval
#: script. Measured extents and intervals are recomputed on load; these are the
#: documented values the measurement is checked against.
SOURCE_NATIVE_STEP = {
    'str': '20min',
    'gs': '30min',
    'era5': '1h',
}


# ──────────────────────────────────────────────────────────────────────
# Quantities
# ──────────────────────────────────────────────────────────────────────

#: The quantities that enter the three-source comparison. Every other quantity
#: below is characterised within its own source and goes no further: the wall is
#: driven by heat and light, and this study does not claim a structural role for
#: pressure or rainfall.
QUANTITIES = ('tair', 'sr')

#: Unit of each quantity, as reported after the conversions in
#: :data:`UNIT_FACTOR` have been applied.
QUANTITY_UNIT = {
    'tair': 'degC',
    'rh': '%',
    'tdew': 'degC',
    'sr': 'W/m2',
    'rain': 'mm',
    'rain_rate': 'mm/h',
    'wspd': 'm/s',
    'wdir': 'deg',
    'pres': 'hPa',
}

#: Human-readable name of each quantity, for axis labels and table rows.
QUANTITY_LABEL = {
    'tair': 'Air temperature',
    'rh': 'Relative humidity',
    'tdew': 'Dew-point temperature',
    'sr': 'Solar radiation',
    'rain': 'Precipitation',
    'rain_rate': 'Precipitation intensity',
    'wspd': 'Wind speed',
    'wdir': 'Wind direction',
    'pres': 'Atmospheric pressure',
}

#: Quantities whose unit could not be confirmed from the source's own
#: documentation. Reported as such rather than silently asserted: the ground
#: station publishes a precipitation intensity without stating the interval it
#: is accumulated over, so its unit is this study's inference and is marked.
UNIT_UNCONFIRMED = ('rain_rate',)

#: Quantities that are directions rather than magnitudes. They are averaged by
#: the unit-vector method in :mod:`shmlib.meteo` and drawn on a circular axis,
#: because the arithmetic mean of 350 and 10 degrees is 180, which points
#: exactly backwards.
CIRCULAR = ('wdir',)

#: Plausible physical range per quantity. A guard, never a filter: a value
#: outside these bounds is counted and reported, and the number behind it is
#: left in the frame. A radiation channel reading sixty watts at midnight is a
#: finding about the instrument, and deleting it would hide the finding.
PLAUSIBLE_RANGE = {
    'tair': (-25.0, 45.0),
    'rh': (0.0, 100.0),
    'tdew': (-30.0, 30.0),
    'sr': (0.0, adc.SR_MAX_PHYSICAL),
    'rain': (0.0, 200.0),
    'rain_rate': (0.0, 300.0),
    'wspd': (0.0, 60.0),
    'wdir': (0.0, 360.0),
    'pres': (900.0, 1060.0),
}


# ──────────────────────────────────────────────────────────────────────
# Native column names, one map per source
# ──────────────────────────────────────────────────────────────────────

#: ERA5 channels, as exported by ``auxiliary/oiko.py``. The seven retained here
#: are those the report characterises; the Oikolab response also carries skin
#: temperature, downward thermal radiation and snowfall, which this study does
#: not use and therefore does not load.
ERA5_MAP = {
    'tair': 'temperature (degC)',
    'rh': 'relative_humidity (0-1)',
    'tdew': 'dewpoint_temperature (degC)',
    'sr': 'surface_solar_radiation (W/m^2)',
    'rain': 'total_precipitation (mm of water equivalent)',
    'wspd': 'wind_speed (m/s)',
    'wdir': 'wind_direction (deg)',
}

#: Ground-station channels, as exported by ``auxiliary/meteosystem_italy.py``.
#: The station also reports the interval minimum and maximum of the temperature
#: and the gust speed and direction; those describe the sampling within an
#: interval rather than a distinct quantity, and are not characterised here.
GS_MAP = {
    'tair': 'Temp',
    'rh': 'Umid',
    'tdew': 'Dew pt',
    'wspd': 'Vento',
    'wdir': 'Dir',
    'pres': 'Press',
    'rain': 'Pioggia',
    'rain_rate': 'Int.Pio.',
    'sr': 'Rad.Sol.',
}

#: On-structure channels, as exported by study 1. Air temperature comes from the
#: archive's joined ``tair`` column, which carries the corrected value of
#: whichever acquisition block was recording; radiation exists only in the
#: current era and is read from its corrected column.
STR_MAP = {
    'tair': 'tair',
    'sr': f'{site.CURRENT_BLOCK}_sr_ok',
}

#: Channels needing a unit conversion on load, as ``(source, quantity)`` to
#: factor. ERA5 reports relative humidity as a fraction where every other source
#: reports per cent; left uncorrected it would appear as a 99 % dry bias.
UNIT_FACTOR = {
    ('era5', 'rh'): 100.0,
}


# ──────────────────────────────────────────────────────────────────────
# The sensor side: columns of study 1's archive
# ──────────────────────────────────────────────────────────────────────

#: For each quantity, the current-era columns of the archive: the field as
#: recorded, the verdict passed on it, and the value study 1 uses.
SENSOR_COLUMNS = {
    quantity: (f'{site.CURRENT_BLOCK}_{quantity}',
               f'{site.CURRENT_BLOCK}_{quantity}_flag',
               f'{site.CURRENT_BLOCK}_{quantity}_ok')
    for quantity in QUANTITIES
}

#: Study 1's verdict on the radiation channel: ``True`` on days the channel
#: shows no diurnal cycle, whatever it reported. A rejection of the day, not of
#: a sample, and it condemns readings that pass every per-value test.
SUSPECT_COLUMN = 'sr_suspect'


# ──────────────────────────────────────────────────────────────────────
# Loading and harmonisation
# ──────────────────────────────────────────────────────────────────────

def _resample_quantity(series, quantity, freq, circular, min_count):
    """
    Put one channel on the analysis grid with the average its kind admits.

    Directions are averaged by the unit-vector method and everything else by the
    arithmetic mean. The distinction is not cosmetic: a wind that blows from just
    either side of north averages to south under an arithmetic mean.

    Parameters
    ----------
    series : pd.Series
        Channel to resample, indexed by timestamp.
    quantity : str
        Canonical quantity name, used only to decide whether the channel is
        circular.
    freq : str
        Pandas offset alias of the target grid.
    circular : sequence of str
        Quantities to average by the unit-vector method.
    min_count : int
        Non-missing samples a bin must carry to produce a value. Bins holding
        fewer become missing rather than being averaged from too little.

    Returns
    -------
    pd.Series
        The channel on the target grid.
    """
    if quantity in circular:
        return meteo.circular_resample(series, freq, min_count=min_count)
    # ``min_count`` cannot be passed through ``Resampler.mean`` — pandas rejects
    # it as a numpy keyword — so the count is taken separately and used to mask.
    resampler = series.resample(freq)
    return resampler.mean().where(resampler.count() >= min_count)


def _read_proxy(path, column_map, source, freq, circular, unit_factor,
                min_count):
    """
    Read one external proxy export and put it on the analysis grid.

    Shared by :func:`load_era5` and :func:`load_ground_station`, which differ
    only in the map they pass. Duplicated timestamps are dropped rather than
    merged: unlike the sensor archive, where conflicting copies of a record
    routinely differ in which block holds real values, these are re-scrapes of
    one observation and the copies agree.

    Parameters
    ----------
    path : str
        Path to the export. The first column is the timestamp.
    column_map : dict of str to str
        Canonical quantity name to the column that carries it in this file.
    source : str
        Source suffix appended to every canonical name.
    freq : str
        Pandas offset alias of the analysis grid.
    circular : sequence of str
        Quantities averaged by the unit-vector method.
    unit_factor : dict
        ``(source, quantity)`` to multiplicative factor applied on load.
    min_count : int
        Non-missing samples a bin must carry to produce a value.

    Returns
    -------
    pd.DataFrame
        Columns named ``{quantity}_{source}``, on the analysis grid. Carries in
        ``attrs`` the measured native step, the duplicate count, the raw extent,
        and the names in ``column_map`` that the file did not contain.
    """
    raw = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    raw = raw.sort_index()

    n_duplicates = int(raw.index.duplicated().sum())
    raw = raw[~raw.index.duplicated(keep='first')]

    steps = raw.index.to_series().diff()
    native_step = steps.median()

    out = pd.DataFrame(index=raw.index)
    missing = []
    for quantity, native in column_map.items():
        if native not in raw.columns:
            missing.append(native)
            continue
        values = pd.to_numeric(raw[native], errors='coerce')
        factor = unit_factor.get((source, quantity))
        if factor is not None:
            values = values * factor
        out[f'{quantity}_{source}'] = values

    # Both edges are floored rather than rounded outwards. Rounding the last
    # edge up appends a slot that no sample can fall into, which reads as an
    # hour of missing data at the end of every source.
    gridded = pd.DataFrame(index=pd.date_range(raw.index.min().floor(freq),
                                               raw.index.max().floor(freq),
                                               freq=freq))
    for column in out.columns:
        quantity = column.rpartition('_')[0]
        gridded[column] = _resample_quantity(out[column], quantity, freq,
                                             circular, min_count)

    gridded.attrs['source'] = source
    gridded.attrs['native_step'] = str(native_step)
    gridded.attrs['duplicates_dropped'] = n_duplicates
    gridded.attrs['raw_extent'] = (raw.index.min(), raw.index.max())
    gridded.attrs['raw_rows'] = int(len(raw))
    gridded.attrs['columns_absent'] = missing
    return gridded


def load_era5(path, column_map=None, freq=site.ANALYSIS_FREQ, circular=CIRCULAR,
              unit_factor=None, min_count=1):
    """
    Load the ERA5 export onto the analysis grid.

    The export is natively hourly and its timestamps are already UTC, so on the
    hourly grid this is a rename and a unit conversion rather than an
    aggregation. Relative humidity is converted from the fraction ERA5 reports
    to the per cent every other source reports; no other unit is changed.

    Parameters
    ----------
    path : str
        Path to the Oikolab export.
    column_map : dict of str to str or None, optional
        Canonical quantity name to native column. Default ``None``, which uses
        :data:`ERA5_MAP`.
    freq : str, optional
        Analysis grid. Default :data:`shmlib.site.ANALYSIS_FREQ`.
    circular : sequence of str, optional
        Quantities averaged by the unit-vector method. Default
        :data:`CIRCULAR`.
    unit_factor : dict or None, optional
        Conversions applied on load. Default ``None``, which uses
        :data:`UNIT_FACTOR`.
    min_count : int, optional
        Non-missing samples a bin must carry. Default ``1``: the source is
        already on the analysis grid, so a bin holds one sample or none.

    Returns
    -------
    pd.DataFrame
        Columns named ``{quantity}_era5``, on the analysis grid, with load
        provenance in ``attrs``.
    """
    return _read_proxy(path, ERA5_MAP if column_map is None else column_map,
                       'era5', freq, circular,
                       UNIT_FACTOR if unit_factor is None else unit_factor,
                       min_count)


def load_ground_station(path, column_map=None, freq=site.ANALYSIS_FREQ,
                        circular=CIRCULAR, unit_factor=None, min_count=1):
    """
    Load the ground-station export onto the analysis grid.

    The export is natively half-hourly, so each hour of the grid is the mean of
    the two observations inside it, and each direction the unit-vector mean of
    the same two. Its timestamps were converted from Italian civil time to UTC
    by the retrieval script and arrive here already in UTC.

    Parameters
    ----------
    path : str
        Path to the assembled CSV.
    column_map : dict of str to str or None, optional
        Canonical quantity name to native column. Default ``None``, which uses
        :data:`GS_MAP`.
    freq : str, optional
        Analysis grid. Default :data:`shmlib.site.ANALYSIS_FREQ`.
    circular : sequence of str, optional
        Quantities averaged by the unit-vector method. Default
        :data:`CIRCULAR`.
    unit_factor : dict or None, optional
        Conversions applied on load. Default ``None``, which uses
        :data:`UNIT_FACTOR`.
    min_count : int, optional
        Non-missing samples an hour must carry. Default ``1``, which lets a
        half-hour that lost its partner still report the hour it was in.

    Returns
    -------
    pd.DataFrame
        Columns named ``{quantity}_gs``, on the analysis grid, with load
        provenance in ``attrs``.
    """
    return _read_proxy(path, GS_MAP if column_map is None else column_map,
                       'gs', freq, circular,
                       UNIT_FACTOR if unit_factor is None else unit_factor,
                       min_count)


def load_sensor_forcings(path, column_map=None, freq=site.ANALYSIS_FREQ,
                         tz=site.SITE_TZ, suspect_column=SUSPECT_COLUMN,
                         honour_suspect=True, night_flag=None,
                         min_count=site.MIN_SAMPLES_PER_HOUR, usecols=None):
    """
    Load the on-structure forcings from study 1's archive onto the analysis grid.

    Study 1's decisions are consumed, never revisited. Air temperature is taken
    from the archive's joined ``tair`` column and radiation from the corrected
    ``_ok`` column of the current-era block, and the ``sr_suspect`` verdict is
    honoured by default: on the days study 1 condemned, the radiation becomes
    missing before anything is averaged, so no statistic built on this frame is
    built on a day whose diurnal cycle study 1 could not find.

    The archive is kept on the logger's civil clock and the proxies are in UTC,
    so the index is converted here — before any resampling, because an hourly
    mean taken on one clock and labelled with another is wrong by up to two
    hours for half the year.

    Parameters
    ----------
    path : str
        Path to ``gubbio_archive_20min.csv``.
    column_map : dict of str to str or None, optional
        Canonical quantity name to archive column. Default ``None``, which uses
        :data:`STR_MAP`.
    freq : str, optional
        Analysis grid. Default :data:`shmlib.site.ANALYSIS_FREQ`.
    tz : str, optional
        The logger's civil clock. Default :data:`shmlib.site.SITE_TZ`.
    suspect_column : str, optional
        Study 1's day-level radiation verdict. Default :data:`SUSPECT_COLUMN`.
    honour_suspect : bool, optional
        Exclude the condemned days from the radiation channel. Default ``True``.
        ``False`` shows what the channel reported on those days, which is a
        diagnostic and never a result.
    night_flag : str or None, optional
        Flag code marking a radiation value study 1 replaced with a substituted
        zero. Default ``None``, which uses :data:`shmlib.quality.FLAG_NIGHT`.
        Resolved lazily, at call time rather than at definition time, because
        :mod:`shmlib.quality` is the flag vocabulary's home and itself depends
        on this module for :data:`SUSPECT_COLUMN` and :data:`SOURCES`; a
        module-level import in both directions would be a real import cycle,
        and this is the one-line fix that keeps it out. Counted, not removed.
    min_count : int, optional
        Twenty-minute samples an hour must carry. Default
        :data:`shmlib.site.MIN_SAMPLES_PER_HOUR`.
    usecols : sequence of str or None, optional
        Columns to read from the archive. Default ``None``, which reads the
        mapped channels, the radiation flag and the suspect verdict. The archive
        is a wide table of roughly a hundred columns and forty-five megabytes,
        and reading only what is needed is the difference between a load of
        seconds and one of minutes.

    Returns
    -------
    sensor : pd.DataFrame
        Columns named ``{quantity}_str`` on the analysis grid, in UTC.
    provenance : dict
        What was read and what was set aside: the native extent, the count of
        ambiguous civil timestamps dropped by the clock conversion, the number
        of condemned days and of samples they cost, and the number of radiation
        values carrying the night correction.
    """
    from . import quality
    night_flag = quality.FLAG_NIGHT if night_flag is None else night_flag

    column_map = STR_MAP if column_map is None else column_map
    sr_flag_column = f'{site.CURRENT_BLOCK}_sr_flag'

    if usecols is None:
        wanted = ['datetime'] + list(column_map.values())
        wanted += [sr_flag_column, suspect_column]
    else:
        wanted = list(usecols)

    header = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in dict.fromkeys(wanted) if column in header]

    raw = pd.read_csv(path, index_col=0, parse_dates=True, usecols=usecols,
                      low_memory=False)
    raw = raw.sort_index()
    native_extent = (raw.index.min(), raw.index.max())

    suspect = pd.Series(False, index=raw.index)
    if suspect_column in raw.columns:
        suspect = raw[suspect_column].fillna(False).astype(bool)
    n_suspect_days = int(pd.Series(raw.index[suspect]).dt.normalize().nunique())

    n_night = 0
    if sr_flag_column in raw.columns:
        n_night = int((raw[sr_flag_column] == night_flag).sum())

    values = pd.DataFrame(index=raw.index)
    n_suspect_samples = 0
    for quantity, native in column_map.items():
        if native not in raw.columns:
            continue
        channel = pd.to_numeric(raw[native], errors='coerce')
        if quantity == 'sr' and honour_suspect:
            n_suspect_samples = int(channel.notna().where(suspect, False).sum())
            channel = channel.where(~suspect)
        values[f'{quantity}_str'] = channel

    utc_index = site.to_utc(values.index, tz=tz)
    n_ambiguous = int(pd.isna(utc_index).sum())
    values.index = utc_index
    values = values[values.index.notna()].sort_index()

    grid = pd.date_range(values.index.min().floor(freq),
                         values.index.max().floor(freq), freq=freq)
    sensor = pd.DataFrame(index=grid)
    for column in values.columns:
        resampler = values[column].resample(freq)
        hourly = resampler.mean().where(resampler.count() >= min_count)
        sensor[column] = hourly.reindex(grid)

    provenance = {
        'native_extent': native_extent,
        'native_rows': int(len(raw)),
        'native_step': adc.SAMPLING,
        'clock': tz,
        'ambiguous_timestamps_dropped': n_ambiguous,
        'suspect_days': n_suspect_days,
        'suspect_samples_removed': n_suspect_samples,
        'night_corrected_samples': n_night,
        'honour_suspect': bool(honour_suspect),
    }
    sensor.attrs.update(provenance)
    sensor.attrs['source'] = 'str'
    return sensor, provenance


#: Study 1's derived inclination chain, in the order it was built. The last is
#: the analysis column: compensated, joined across the instrument change, and
#: with the spikes study 1 found replaced inside observed stretches.
INC_CHAIN = ('inc', 'inc_comp', 'inc_comp_cleaned')

#: Study 1's verdict on the inclination: ``True`` where the sample it stands
#: beside was interpolated rather than measured.
INC_SPIKE_COLUMN = 'inc_spike'


def load_response(path, column='inc_comp_cleaned',
                  spike_column=INC_SPIKE_COLUMN, honour_spike=True,
                  freq=site.ANALYSIS_FREQ, tz=site.SITE_TZ,
                  min_count=site.MIN_SAMPLES_PER_HOUR):
    """
    Load the structural response from study 1's archive onto the analysis grid.

    The response is one column of the same file :func:`load_sensor_forcings`
    reads, and it is put on the grid by the same rules: the archive's civil
    timestamps are converted to UTC before anything is averaged, and an hour
    produces a value only if it carries `min_count` of the twenty-minute
    samples underneath it.

    Study 1's spike verdict is honoured by default. The samples it marks were
    replaced by interpolation inside observed stretches, so they are the
    interpolator's output rather than the instrument's; leaving them in a
    correlation would let study 1's interpolation choices into a coupling
    measured against them. They are counted, so the report can say what
    honouring the verdict cost.

    Parameters
    ----------
    path : str
        Path to ``gubbio_archive_20min.csv``.
    column : str, optional
        Column carrying the response. Default ``'inc_comp_cleaned'``, study 1's
        analysis column. The alternatives are named in :data:`INC_CHAIN`; note
        that the absolute level of every one of them is set by an arbitrary
        anchor, so only their changes carry structural information.
    spike_column : str, optional
        Study 1's interpolation verdict. Default :data:`INC_SPIKE_COLUMN`.
    honour_spike : bool, optional
        Drop the samples the verdict marks before resampling. Default ``True``.
        ``False`` shows what the cleaned series contains at those samples,
        which is a diagnostic and never a result.
    freq : str, optional
        Analysis grid. Default :data:`shmlib.site.ANALYSIS_FREQ`.
    tz : str, optional
        The logger's civil clock. Default :data:`shmlib.site.SITE_TZ`.
    min_count : int, optional
        Twenty-minute samples an hour must carry. Default
        :data:`shmlib.site.MIN_SAMPLES_PER_HOUR`.

    Returns
    -------
    response : pd.Series
        The response on the analysis grid, in UTC, named after `column`.
    provenance : dict
        The native extent and row count, the clock converted from, the
        ambiguous civil timestamps the conversion dropped, the interpolated
        samples the spike verdict removed, and the hours the grid carries.
    """
    header = pd.read_csv(path, nrows=0).columns
    wanted = [name for name in ('datetime', column, spike_column)
              if name in header]
    if column not in wanted:
        raise KeyError(f'{column!r} is not a column of {path}')

    raw = pd.read_csv(path, index_col=0, parse_dates=True, usecols=wanted,
                      low_memory=False).sort_index()
    native_extent = (raw.index.min(), raw.index.max())

    values = pd.to_numeric(raw[column], errors='coerce')
    n_spike = 0
    if honour_spike and spike_column in raw.columns:
        spike = raw[spike_column].fillna(False).astype(bool)
        n_spike = int(values.notna().where(spike, False).sum())
        values = values.where(~spike)

    utc_index = site.to_utc(values.index, tz=tz)
    n_ambiguous = int(pd.isna(utc_index).sum())
    values.index = utc_index
    values = values[values.index.notna()].sort_index()

    resampler = values.resample(freq)
    response = resampler.mean().where(resampler.count() >= min_count)
    response.name = column

    provenance = {
        'column': column,
        'native_extent': native_extent,
        'native_rows': int(len(raw)),
        'native_step': adc.SAMPLING,
        'clock': tz,
        'ambiguous_timestamps_dropped': n_ambiguous,
        'interpolated_samples_removed': n_spike,
        'honour_spike': bool(honour_spike),
        'hours_observed': int(response.notna().sum()),
    }
    response.attrs.update(provenance)
    return response, provenance


def join_eras(frames):
    """
    One channel per quantity, from the block that was recording.

    The archive keeps the legacy blocks and the current package in separate
    columns, which is right — they are different units of hardware and study 1
    keeps them structurally apart rather than by convention. A study that wants
    a channel spanning both eras therefore has to say how they are joined, and
    this is the rule study 1 itself applies to air temperature: the eras do not
    overlap, so the value is whichever block wrote one, and where both did the
    earlier frame in the sequence wins.

    Parameters
    ----------
    frames : sequence of pd.DataFrame
        Frames sharing a naming scheme, in order of precedence. Each may hold
        any subset of the columns; a column absent from one is taken from the
        next.

    Returns
    -------
    pd.DataFrame
        The frames combined on the union of their indices and columns.
    """
    joined = None
    for frame in frames:
        joined = frame.copy() if joined is None else joined.combine_first(frame)
    return joined


def harmonise(frames, freq=site.ANALYSIS_FREQ):
    """
    Join the loaded sources onto one index under one naming scheme.

    An outer join, deliberately. Each source is characterised over its own
    extent before any comparison narrows it, and a join that kept only the
    shared timestamps would silently answer a different question — it would
    describe the overlap and call it the record.

    Parameters
    ----------
    frames : sequence of pd.DataFrame
        The loaded sources, each already on ``freq`` and already carrying
        suffixed column names.
    freq : str, optional
        Analysis grid. Default :data:`shmlib.site.ANALYSIS_FREQ`.

    Returns
    -------
    pd.DataFrame
        One frame spanning the union of the sources' extents, with the ``attrs``
        of each input preserved under its source suffix.
    """
    frames = [frame for frame in frames if frame is not None and len(frame)]
    if not frames:
        raise ValueError('harmonise received no non-empty frame')

    start = min(frame.index.min() for frame in frames)
    end = max(frame.index.max() for frame in frames)
    grid = pd.date_range(start, end, freq=freq)

    out = pd.DataFrame(index=grid)
    out.index.name = 'datetime'
    provenance = {}
    for frame in frames:
        provenance[frame.attrs.get('source', 'unknown')] = dict(frame.attrs)
        for column in frame.columns:
            out[column] = frame[column].reindex(grid)

    out.attrs['provenance'] = provenance
    out.attrs['freq'] = freq
    return out


# ──────────────────────────────────────────────────────────────────────
# Characterisation, one source at a time
# ──────────────────────────────────────────────────────────────────────

def source_inventory(frames, labels=None, natures=None, descriptions=None,
                     native_steps=None):
    """
    One row per source: what it is, how often it reports, and what it spans.

    Every extent and every interval in the row is measured from the file that
    was loaded. The documented values sit in :data:`SOURCE_NATIVE_STEP` and in
    the report's introduction, and the point of measuring them again is that the
    two can then be compared rather than conflated.

    Parameters
    ----------
    frames : dict of str to pd.DataFrame
        Source suffix to the frame returned by that source's loader.
    labels, natures, descriptions : dict or None, optional
        Source suffix to display name, nature, and what it physically measures.
        Default ``None`` for each, which uses the module constants.
    native_steps : dict or None, optional
        Source suffix to documented native interval. Default ``None``, which
        uses :data:`SOURCE_NATIVE_STEP`.

    Returns
    -------
    pd.DataFrame
        One row per source, in the order of ``frames``.
    """
    labels = SOURCE_LABEL if labels is None else labels
    natures = SOURCE_NATURE if natures is None else natures
    descriptions = SOURCE_DESCRIPTION if descriptions is None else descriptions
    native_steps = (SOURCE_NATIVE_STEP if native_steps is None
                    else native_steps)

    rows = []
    for source, frame in frames.items():
        attrs = frame.attrs
        extent = attrs.get('raw_extent') or attrs.get('native_extent')
        first, last = (extent if extent else (frame.index.min(),
                                              frame.index.max()))
        rows.append({
            'source': source,
            'label': labels.get(source, source),
            'nature': natures.get(source, ''),
            'measures': descriptions.get(source, ''),
            'native_step_documented': native_steps.get(source, ''),
            'native_step_measured': str(attrs.get('native_step', '')),
            'first': pd.Timestamp(first),
            'last': pd.Timestamp(last),
            'days': int((pd.Timestamp(last).normalize()
                         - pd.Timestamp(first).normalize()).days) + 1,
            'native_rows': int(attrs.get('raw_rows',
                                         attrs.get('native_rows', 0))),
            'duplicates_dropped': int(attrs.get('duplicates_dropped', 0)),
            'n_channels': int(len(frame.columns)),
        })
    return pd.DataFrame(rows)


def mask_implausible(df, plausible_range=None, columns=None):
    """
    A copy of a frame with impossible values replaced by ``NaN``.

    The counterpart to the count :func:`channel_inventory` reports, for the
    statistics that cannot survive one. A count says the ground station's
    rainfall carries a reading of order a billion millimetres; a mean square
    computed over that reading says nothing at all, and a table of agreement
    scores built from it is worse than no table, because it looks like a
    result. Where a statistic is to be computed rather than a defect reported,
    the impossible values come out first — and the study that does so says it
    has, which is why this returns the count alongside the frame.

    Nothing is dropped and no row disappears: an excluded value becomes missing,
    so every statistic downstream reports the sample count it actually used. The
    input frame is never modified.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying ``{quantity}_{source}`` columns.
    plausible_range : dict of str to tuple or None, optional
        Canonical quantity name to its ``(low, high)`` bounds, inclusive.
        Default ``None``, which uses :data:`PLAUSIBLE_RANGE`. A quantity absent
        from the mapping is left untouched.
    columns : sequence of str or None, optional
        Columns to mask. Default ``None``, meaning every column of ``df`` whose
        quantity has documented bounds.

    Returns
    -------
    masked : pd.DataFrame
        A copy of ``df`` with out-of-range values replaced by ``NaN``.
    n_masked : pd.Series
        How many values were masked in each column touched, indexed by column
        name. Columns that lost nothing are still listed, at zero, so that a
        caller can report the whole set it considered.
    """
    plausible_range = (PLAUSIBLE_RANGE if plausible_range is None
                       else plausible_range)
    masked = df.copy()
    counts = {}
    for column in (df.columns if columns is None else columns):
        quantity = column.rpartition('_')[0]
        bounds = plausible_range.get(quantity)
        if bounds is None:
            continue
        values = df[column]
        outside = values.notna() & ~values.between(bounds[0], bounds[1])
        counts[column] = int(outside.sum())
        masked.loc[outside, column] = np.nan
    return masked, pd.Series(counts, dtype=int)


def channel_inventory(df, source, column_map=None, quantity_unit=None,
                      plausible_range=None, unit_factor=None,
                      unit_unconfirmed=UNIT_UNCONFIRMED, circular=CIRCULAR):
    """
    One row per channel of one source: extent, coverage, range and defects.

    Coverage is measured against the source's own extent rather than against the
    study window, because this is the characterisation of a source on its own
    terms: a source that starts late is not thereby incomplete.

    Values outside :data:`PLAUSIBLE_RANGE` are counted and reported. They are
    not removed, and this function never modifies ``df``. The ground station's
    rainfall carries at least one value of order a billion millimetres, and the
    count is how the report says so.

    Parameters
    ----------
    df : pd.DataFrame
        Frame carrying this source's suffixed channels.
    source : str
        Source suffix whose channels are to be described.
    column_map : dict of str to str or None, optional
        Canonical quantity to native column, reported so a row states where its
        number came from. Default ``None``, which selects the map belonging to
        ``source``.
    quantity_unit, plausible_range, unit_factor : dict or None, optional
        Default ``None`` for each, which uses the module constants.
    unit_unconfirmed : sequence of str, optional
        Quantities whose unit is this study's inference rather than the source's
        statement. Default :data:`UNIT_UNCONFIRMED`.
    circular : sequence of str, optional
        Quantities for which a minimum and a maximum are meaningless, since
        every direction is both. Default :data:`CIRCULAR`.

    Returns
    -------
    pd.DataFrame
        One row per channel present, ordered as ``column_map``.
    """
    default_maps = {'era5': ERA5_MAP, 'gs': GS_MAP, 'str': STR_MAP}
    column_map = default_maps.get(source, {}) if column_map is None \
        else column_map
    quantity_unit = QUANTITY_UNIT if quantity_unit is None else quantity_unit
    plausible_range = (PLAUSIBLE_RANGE if plausible_range is None
                       else plausible_range)
    unit_factor = UNIT_FACTOR if unit_factor is None else unit_factor

    present = df[[column for column in df.columns
                  if column.endswith(f'_{source}')]]
    if not len(present.columns):
        return pd.DataFrame()

    span = present.dropna(how='all')
    first_seen = span.index.min() if len(span) else pd.NaT
    last_seen = span.index.max() if len(span) else pd.NaT
    extent = present.loc[first_seen:last_seen] if len(span) else present

    rows = []
    for quantity, native in column_map.items():
        column = f'{quantity}_{source}'
        if column not in present.columns:
            continue
        values = pd.to_numeric(present[column], errors='coerce')
        observed = values.dropna()
        in_extent = extent[column] if column in extent.columns else values
        low, high = plausible_range.get(quantity, (-np.inf, np.inf))
        outside = observed[(observed < low) | (observed > high)]
        factor = unit_factor.get((source, quantity))
        rows.append({
            'channel': column,
            'quantity': quantity,
            'name': QUANTITY_LABEL.get(quantity, quantity),
            'native_column': native,
            'unit': quantity_unit.get(quantity, ''),
            'unit_confirmed': quantity not in unit_unconfirmed,
            'conversion': f'x {factor:g}' if factor else '',
            'first': observed.index.min() if len(observed) else pd.NaT,
            'last': observed.index.max() if len(observed) else pd.NaT,
            'n_valid': int(len(observed)),
            'n_slots': int(len(in_extent)),
            'coverage_pct': (100.0 * len(observed) / len(in_extent)
                             if len(in_extent) else np.nan),
            'min': np.nan if quantity in circular else
                   (float(observed.min()) if len(observed) else np.nan),
            'median': float(observed.median()) if len(observed) else np.nan,
            'max': np.nan if quantity in circular else
                   (float(observed.max()) if len(observed) else np.nan),
            'n_implausible': int(len(outside)),
        })
    return pd.DataFrame(rows)
