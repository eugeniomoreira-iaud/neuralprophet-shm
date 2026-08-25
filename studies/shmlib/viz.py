"""
Module: shmlib.viz

Figure scaffolding shared by every study: the colour scheme, the report style,
and the save helper.

The rules encoded here come from the Graphical Guidelines in
``instructions-pipeline.md`` and are binding on every figure the project
produces. The most important of them is that **colour carries variable identity
and nothing else**. A channel is drawn in its own colour in every figure of every
study, so that a reader recognises the quantity before reading the axis label.
When two instrument eras, two sources or two subsets must be told apart inside a
single axes, the distinction is line style: colour is already spent on identity
and cannot also carry the grouping.

Two consequences follow, and both are easy to violate by accident. The accent
colour marks events and annotations — changeover lines, isolated markers, the
note that explains a clipped axis — and never encodes a data category. Span
highlights are black at five per cent opacity with no edge, which is not a
category colour either. Quantitative areas such as confidence intervals drawn
with ``fill_between`` are exempt from the span rule.

Nothing here saves a file unless it is given a destination.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns

from . import meteo, proxies


# ──────────────────────────────────────────────────────────────────────
# Colour: identity, accent, spans
# ──────────────────────────────────────────────────────────────────────

#: Okabe-Ito colour per measured channel, held constant across every figure in
#: the project. Data lines are single-colour strokes: no outline, no halo, no
#: ``path_effects.Stroke``. A channel that needs more contrast gets a darker
#: identity colour or a wider line, never a boundary in a second colour.
CHANNEL_COLOUR = {
    'inc_comp': '#0072B2',       # Blue
    'tair': '#E69F00',           # Orange
    'rh': '#009E73',             # Bluish Green
    'batt': '#000000',           # Black
    'twall': '#DAA520',          # Goldenrod
    'sr': '#CC79A7',             # Reddish Purple
}

# The same quantity appears under several names as it is corrected, and each
# alias must be drawn in the colour of the quantity rather than in a new one.
CHANNEL_COLOUR['twall_filtered'] = CHANNEL_COLOUR['twall']
CHANNEL_COLOUR['inc_comp_cleaned'] = CHANNEL_COLOUR['inc_comp']
CHANNEL_COLOUR['inc'] = CHANNEL_COLOUR['inc_comp']

#: Shorthand for the inclination, the quantity most figures are about.
INC_COLOUR = CHANNEL_COLOUR['inc_comp']

#: Axis label of each channel, as ``Name [unit]``. Paired with
#: :data:`CHANNEL_COLOUR` so that a channel is named and coloured identically
#: wherever it appears. Split on ``' ['`` to recover the name alone, which is
#: what a title or a legend entry wants.
CHANNEL_LABEL = {
    'inc_comp': 'Compensated inclination [mdeg]',
    'tair': 'Air temperature [°C]',
    'rh': 'Relative humidity [%]',
    'batt': 'Supply voltage [V]',
    'twall': 'Wall temperature [°C]',
    'sr': 'Solar radiation [W/m²]',
}

# A quantity keeps one label through every stage of its correction, as it keeps
# one colour: the quantity is the same, only its treatment differs.
CHANNEL_LABEL['inc_comp_cleaned'] = CHANNEL_LABEL['inc_comp']
CHANNEL_LABEL['inc'] = CHANNEL_LABEL['inc_comp']
CHANNEL_LABEL['twall_filtered'] = CHANNEL_LABEL['twall']


def channel_name(column):
    """
    The channel's name without its unit, for a title or a legend entry.

    Parameters
    ----------
    column : str
        Channel. Must be a key of :data:`CHANNEL_LABEL`.

    Returns
    -------
    str
        The label up to the unit, e.g. ``'Air temperature'``.
    """
    return CHANNEL_LABEL[column].split(' [')[0]


def channel_unit(column):
    """
    The channel's unit, bracketed, for an axis label.

    Parameters
    ----------
    column : str
        Channel. Must be a key of :data:`CHANNEL_LABEL`.

    Returns
    -------
    str
        The unit including its brackets, e.g. ``'[°C]'``.
    """
    return '[' + CHANNEL_LABEL[column].split(' [')[1]


#: Okabe-Ito Vermilion, the accent. Instrument-changeover lines, isolated event
#: markers, and the short text notes that explain a clipped axis. Reserved for
#: those roles: it never encodes a data category and never fills a span.
MARK_COLOUR = '#D55E00'

#: Every highlighted interval or reference band drawn with ``axvspan`` or
#: ``axhspan``. Black at five per cent, no edge.
SPAN_STYLE = {'color': '#000000', 'alpha': 0.05, 'lw': 0}

#: Era is line style, so that colour stays free for identity.
LEGACY_STYLE = '--'
CURRENT_STYLE = '-'


def channel_style(column, linewidth):
    """
    Colour and width for a channel's single-colour data line.

    Parameters
    ----------
    column : str
        Channel being drawn. Must be a key of :data:`CHANNEL_COLOUR`; an unknown
        name raises rather than silently falling back to a default colour, since
        a channel drawn in the wrong colour is worse than a figure that fails to
        build.
    linewidth : float
        Requested line width.

    Returns
    -------
    dict
        Keyword arguments for ``Axes.plot``.
    """
    return {'color': CHANNEL_COLOUR[column], 'lw': linewidth}


# ──────────────────────────────────────────────────────────────────────
# Quantity colour and label identity — multi-source comparisons
# ──────────────────────────────────────────────────────────────────────
#
# :data:`CHANNEL_COLOUR` above names the six channels this project measures at
# the wall. Study 2 compares those channels against external proxy sources that
# report several further quantities the project has never plotted — dew point,
# precipitation, wind, pressure — so this section extends the same identity
# rule to the full vocabulary of :mod:`shmlib.proxies`. Moved here from that
# study's now-deleted private library, because the identity a quantity is drawn in must stay
# fixed for any later study that plots the same comparison, not just for the
# one that first needed it.

#: Okabe-Ito Yellow darkened until a one-point line is legible against white.
#: The Graphical Guidelines permit a darker variant of an identity colour where
#: the approved one lacks contrast against the plotting ground, which pure
#: ``#F0E442`` does at every line width these figures draw.
DARK_YELLOW = '#B8A400'

#: Colour identifying each quantity, held constant across every figure that
#: draws it.
#:
#: :data:`CHANNEL_COLOUR` names six measured channels, of which three appear
#: here — air temperature, relative humidity and solar radiation — and those
#: keep their project colours unchanged. The six remaining quantities are ones
#: the project has never plotted, so they are assigned, in three steps and in
#: this order.
#:
#: First, the Okabe-Ito colours the project has not spent: Sky Blue. Second,
#: three colours the project *has* assigned, but to channels a proxy comparison
#: never plots — the inclination, the supply voltage and the wall temperature
#: appear in no figure built from this vocabulary, so re-using their colours
#: cannot collide with them inside any axes a reader will see. Each such re-use
#: is named in the comment beside it, so that a later study which does plot
#: both can find them. Third, and only because the palette is then exhausted, a
#: darkened Okabe-Ito Yellow.
#:
#: Precipitation depth and precipitation intensity share one colour
#: deliberately: they are one quantity reported at two integrations, they never
#: appear in the same axes, and giving them separate colours would assert a
#: distinction that is not made.
#:
#: Vermilion is absent by rule. It is the accent, and it marks events and
#: annotations rather than encoding a quantity.
QUANTITY_COLOUR = {
    'tair': CHANNEL_COLOUR['tair'],            # Orange, project channel
    'rh': CHANNEL_COLOUR['rh'],                # Bluish Green, project channel
    'sr': CHANNEL_COLOUR['sr'],                # Reddish Purple, project channel
    'tdew': '#56B4E9',                         # Sky Blue, unspent Okabe-Ito
    'rain': '#0072B2',                         # Blue; the project's inclination
    'rain_rate': '#0072B2',                    # colour, and one quantity with it
    'wspd': '#000000',                         # Black; the project's supply voltage
    'wdir': '#DAA520',                         # Goldenrod; the project's wall temp.
    'pres': DARK_YELLOW,                       # Okabe-Ito Yellow, darkened
}

#: Line style telling several sources of one quantity apart inside one axes.
#: Colour is already spent on the quantity and cannot also carry the source, so
#: the source is style — the same rule study 1 applies to the two instrument
#: eras.
SOURCE_STYLE = {
    'str': '-',
    'gs': '--',
    'era5': ':',
}

#: Axis label of each quantity, as ``Name [unit]``, built from
#: :data:`shmlib.proxies.QUANTITY_LABEL` and :data:`shmlib.proxies.QUANTITY_UNIT`
#: — the identity of a channel belongs with the data dictionary in
#: :mod:`shmlib.proxies`; this is its presentation.
QUANTITY_AXIS_LABEL = {
    quantity: f'{proxies.QUANTITY_LABEL[quantity]} [{proxies.QUANTITY_UNIT[quantity]}]'
    for quantity in proxies.QUANTITY_LABEL
}

#: Abbreviated name of each quantity. A figure of seven or nine stacked panels
#: gives each of them about an inch of height, and a label reading
#: "Dew-point temperature [degC]" is physically longer than the panel it
#: belongs to, so it collides with its neighbours. These names are for that
#: case and no other; a single-panel figure uses the full name.
QUANTITY_SHORT = {
    'tair': 'Air temp.',
    'rh': 'Rel. humidity',
    'tdew': 'Dew point',
    'sr': 'Solar rad.',
    'rain': 'Precip.',
    'rain_rate': 'Precip. rate',
    'wspd': 'Wind speed',
    'wdir': 'Wind dir.',
    'pres': 'Pressure',
}

#: Two-line axis label of each quantity: the abbreviated name above its unit.
#: Two short lines fit a panel that one long line does not.
QUANTITY_PANEL_LABEL = {
    quantity: f'{QUANTITY_SHORT[quantity]}\n[{proxies.QUANTITY_UNIT[quantity]}]'
    for quantity in QUANTITY_SHORT
}

#: Compass labels and their bearings, for the ticks of a wind-direction axis.
COMPASS_TICKS = {0.0: 'N', 90.0: 'E', 180.0: 'S', 270.0: 'W', 360.0: 'N'}


# ──────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────

#: Figure width in inches: the manuscript column width every report figure is
#: exported at, matching study 1.
#:
#: Moved from study 1's ``de_lib.py``, where it was defined for that study's
#: own use, because study 2 needs the identical width and ``shmlib`` must never
#: be reached into from one study by way of another. ``de_lib.py`` keeps a thin
#: alias at the original name, so every call site there is unaffected.
FIGURE_WIDTH = 6.38

#: Suffixes a decomposed circular channel carries, stripped before a colour or
#: a label is looked up so that both components of one direction are drawn as
#: that direction rather than as two unrelated quantities.
COMPONENT_SUFFIXES = ('_sin', '_cos')


def split_column(column, sources=proxies.SOURCES):
    """
    A harmonised column name split into its quantity and its source.

    The comparison studies name a column ``{quantity}_{source}`` — ``tair_era5``,
    ``sr_str`` — and the identity a figure draws it in belongs to the quantity
    while the line style belongs to the source. This is the one place that
    split is performed, so a name that does not carry a source suffix, such as
    a study's own derived column, comes back with ``None`` for it rather than
    losing its last underscored word.

    Parameters
    ----------
    column : str
        Column name.
    sources : sequence of str, optional
        Recognised source suffixes. Default :data:`shmlib.proxies.SOURCES`.

    Returns
    -------
    quantity : str
        The name with any recognised source suffix removed.
    source : str or None
        The suffix, or ``None`` when the name carries none.
    """
    for source in sources:
        suffix = f'_{source}'
        if column.endswith(suffix):
            return column[:-len(suffix)], source
    return column, None


def driver_colour(column, sources=proxies.SOURCES):
    """
    The identity colour of a driver column, however it is suffixed.

    Resolved in the order the project's vocabulary was built: the quantity
    scheme of the multi-source comparison first, then the six channels measured
    at the wall. A component of a decomposed direction takes the colour of the
    direction it came from. An unknown name raises, because a channel drawn in
    the wrong colour is worse than a figure that fails to build.

    Parameters
    ----------
    column : str
        Column name, with or without a source suffix.
    sources : sequence of str, optional
        Recognised source suffixes. Default :data:`shmlib.proxies.SOURCES`.

    Returns
    -------
    str
        Hex colour.
    """
    quantity, _ = split_column(column, sources=sources)
    for suffix in COMPONENT_SUFFIXES:
        if quantity.endswith(suffix):
            quantity = quantity[:-len(suffix)]
    if quantity in QUANTITY_COLOUR:
        return QUANTITY_COLOUR[quantity]
    if quantity in CHANNEL_COLOUR:
        return CHANNEL_COLOUR[quantity]
    raise KeyError(f'no identity colour for {column!r}')


def driver_label(column, sources=proxies.SOURCES):
    """
    A short legend entry for a driver column: its name, and where it came from.

    Parameters
    ----------
    column : str
        Column name, with or without a source suffix.
    sources : sequence of str, optional
        Recognised source suffixes. Default :data:`shmlib.proxies.SOURCES`.

    Returns
    -------
    str
        For example ``'Air temp. (ERA5)'``, or ``'Wind dir. sin'`` for a
        component of a decomposed direction with no source suffix.
    """
    quantity, source = split_column(column, sources=sources)

    component = ''
    for suffix in COMPONENT_SUFFIXES:
        if quantity.endswith(suffix):
            quantity, component = quantity[:-len(suffix)], ' ' + suffix[1:]
            break

    if quantity in QUANTITY_SHORT:
        name = QUANTITY_SHORT[quantity]
    elif quantity in CHANNEL_LABEL:
        name = CHANNEL_LABEL[quantity].split(' [')[0]
    else:
        name = quantity

    label = f'{name}{component}'
    if source is not None:
        label += f' ({proxies.SOURCE_LABEL.get(source, source)})'
    return label


def apply_report_style():
    """
    Set the seaborn theme used by every figure destined for a study report.

    Fixed at seaborn's ``paper`` context. Figure width is set per call through
    ``figsize=(6.38, ...)``, the manuscript column width in inches.

    Returns
    -------
    None

    Notes
    -----
    Mutates global matplotlib state, which is what a theme is for. Call it once
    before a block of figures rather than inside a plotting function.
    """
    sns.set_theme(context='paper', style='ticks', rc={
        'axes.labelsize': 9,
        'axes.labelweight': 'bold',
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 1.5,
    })


def format_spines(ax):
    """
    Remove the top and right spines of an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to despine.

    Returns
    -------
    None
    """
    sns.despine(ax=ax, top=True, right=True)


#: Canvas scale factor per seaborn context, for studies that switch context
#: rather than fixing it at ``paper``. ``'paper'`` lands a full-width figure at
#: about seven inches, the usual two-column text width, while seaborn's own
#: context handling takes care of fonts, line widths and markers.
CONTEXT_SCALE = {'paper': 0.55, 'notebook': 1.0, 'talk': 1.15, 'poster': 1.35}

#: Active context, updated by :func:`set_context`.
_ACTIVE = {'context': 'notebook', 'scale': 1.0}


def set_context(context='notebook', style='ticks', palette='colorblind',
                rc=None):
    """
    Set the seaborn theme and the canvas scale for the figures that follow.

    Switching between a notebook layout and a paper layout is a single call:
    seaborn rescales the typography and :func:`figsize` rescales the canvas, so
    a figure exported for the manuscript is the same figure at column width
    rather than a different one.

    Parameters
    ----------
    context : str, optional
        Seaborn context: ``'paper'``, ``'notebook'``, ``'talk'`` or
        ``'poster'``. Default ``'notebook'``.
    style : str, optional
        Seaborn style. Default ``'ticks'``.
    palette : str, optional
        Seaborn palette. Default ``'colorblind'``, the approximation of
        Okabe-Ito used where colours are assigned automatically rather than by
        channel identity.
    rc : dict or None, optional
        Extra matplotlib rcParams, merged over the defaults.

    Returns
    -------
    str
        The context that was applied.

    Notes
    -----
    Mutates global matplotlib state and the module-level active context.
    """
    base_rc = {'axes.axisbelow': True, 'figure.autolayout': False}
    if rc:
        base_rc.update(rc)
    sns.set_theme(context=context, style=style, palette=palette, rc=base_rc)
    _ACTIVE['context'] = context
    _ACTIVE['scale'] = CONTEXT_SCALE.get(context, 1.0)
    return context


def figsize(width, height):
    """
    Scale a nominal figure size by the active context.

    Parameters
    ----------
    width, height : float
        Size in inches at the ``'notebook'`` context.

    Returns
    -------
    tuple of float
        Size in inches at the active context.
    """
    s = _ACTIVE['scale']
    return (width * s, height * s)


# ──────────────────────────────────────────────────────────────────────
# Axis limits
# ──────────────────────────────────────────────────────────────────────

def robust_limits(series, lower=0.005, upper=0.995, pad=0.05, within=None):
    """
    Axis limits that exclude the extreme tails of a series.

    A handful of very large excursions can compress an eight-year record into a
    fraction of its axis. Clipping to an inner quantile range restores the shape
    of the bulk of the record; the count left outside is returned so that the
    figure can say what it is not showing. Nothing is removed from the data or
    from any computed statistic — this is a display choice only.

    Moved here from study 1's ``de_lib.py``, where it was written for that
    study's own use, because study 2 needs the same treatment for a channel of
    its own and ``shmlib`` must never be reached into from one study by way of
    another. ``de_lib.py`` keeps a thin alias at the original name, so every
    call site there is unaffected.

    Parameters
    ----------
    series : pd.Series
        The series about to be plotted. Missing values are ignored.
    lower, upper : float, optional
        Quantiles bounding the retained range. Defaults 0.005 and 0.995.
    pad : float, optional
        Fraction of the retained span added to each side, so that points at the
        quantile itself are not drawn on the axis line. Default 0.05.
    within : tuple of float or None, optional
        Range the quantiles are taken over, values outside it being ignored
        when the limits are computed. Default ``None``, which takes them over
        everything. Pass a channel's documented plausible range where the
        values to be excluded are impossible rather than merely extreme: one
        reading of a thousand millimetres of rain in an hour is not a tail of
        the distribution to be trimmed by a quantile, it is a defect, and a
        quantile chosen large enough to exclude it also excludes real weather.

    Returns
    -------
    limits : tuple of float
        The lower and upper axis limits.
    n_outside : int
        How many observed samples fall outside those limits. Counted over the
        whole series, ``within`` included, since the figure must account for
        every sample it does not show.
    """
    values = series.dropna()
    considered = (values if within is None
                  else values[values.between(within[0], within[1])])
    if not len(considered):
        considered = values
    q_lower, q_upper = considered.quantile([lower, upper])
    span = q_upper - q_lower
    limits = (q_lower - pad * span, q_upper + pad * span)
    n_outside = int(((values < limits[0]) | (values > limits[1])).sum())
    return limits, n_outside


# ──────────────────────────────────────────────────────────────────────
# Diurnal cycles
# ──────────────────────────────────────────────────────────────────────
#
# Averaging a diurnal cycle correctly — restricting to the days that carry
# every slot of the grid, and drawing the individual days behind their mean —
# is a fact about how to average a cycle, not a decision any one study makes,
# so it lives here rather than in a study's own library. Moved from study 1's
# ``de_lib.py``, where the study keeps a thin alias at the original name so
# every existing call site there is unaffected.
#
# :func:`draw_cycle` is the one panel every diurnal figure in the project is
# drawn with. A study decides what a panel is about — an era and a season for
# one channel, or one season for every channel a source reports — and how the
# cycle should be treated, through the arguments it passes; it does not decide
# how a cycle is averaged or drawn, because two implementations of that
# disagree eventually and the disagreement surfaces as two figures a reader
# cannot reconcile.

def complete_days(data, column, complete_day):
    """
    The days on which a channel carries every slot of the grid.

    A partly observed day cannot bias a mean towards the part of the day it
    happens to cover if it never enters the mean.

    Parameters
    ----------
    data : pd.DataFrame
        Frame to search, already restricted to the group of interest.
    column : str
        Channel that must be complete.
    complete_day : int
        Slots a day must carry.

    Returns
    -------
    complete : pd.DataFrame
        The rows belonging to complete days.
    n_days : int
        How many complete days there are.
    """
    counts = data.groupby(data.index.date)[column].count()
    dates = counts[counts == complete_day].index
    return data[np.isin(data.index.date, dates)], len(dates)


def draw_cycle(ax, data, column, colour, complete_day=None, min_days=1,
               centre=True, background=True, background_alpha=0.2,
               circular=False, linewidth=2.0, linestyle='-', label=None):
    """
    Draw one group's mean diurnal cycle over the days behind it.

    This is the panel every diurnal figure in the project is built from: study
    1's grid of eras and seasons for one channel, and study 2's grid of the
    channels one source reports, are the same drawing repeated over different
    facets, and both obtain it here rather than each drawing a cycle its own
    way.

    Every day behind the mean is drawn in gray beneath it, so that the spread
    around the average is visible and a mean built from few days cannot be
    mistaken for a well determined one. Whether the cycle is centred is the
    caller's decision and follows from what the panel is about. Centring
    subtracts each day's own mean before averaging, which removes the seasonal
    drift and leaves the shape of the day alone: the right treatment for a
    channel whose absolute level is arbitrary, such as the inclination, or
    where the object is the shape of one instrument's day. A climatology
    describing what a source actually reports is drawn uncentred, because there
    the level is part of the answer.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    data : pd.DataFrame
        Rows of this group, already restricted to the era, season or other
        facet the panel is about. Rows where ``column`` is missing are ignored.
    column : str
        Channel to average.
    colour : str
        Colour of the mean cycle. The era or season is carried by the panel,
        not by the colour.
    complete_day : int or None, optional
        Slots a day must carry to contribute at all, applied through
        :func:`complete_days`. Default ``None``, which admits every day and
        expects ``data`` to have been restricted by the caller if it wanted
        that rule.
    min_days : int, optional
        Days a position within the day must draw on before its mean is drawn.
        Positions below it are left empty rather than drawn from a handful of
        days. Default ``1``, which draws every position that has any data.
    centre : bool, optional
        Subtract each cycle's own mean, the individual days included, so that
        what is drawn is the excursion rather than the level. Default ``True``.
    background : bool, optional
        Draw the individual days in gray behind the mean. Default ``True``.
        Ignored when ``circular`` is true, since a wrapping direction drawn as
        a line is a lie about the data.
    background_alpha : float, optional
        Opacity of the individual days. Default 0.2. A panel averaging years
        rather than months wants a smaller value, or the days fill it solid.
    circular : bool, optional
        Average by the unit-vector method and draw as unconnected markers,
        for directional data such as a wind direction. Default ``False``.
    linewidth : float, optional
        Width of the mean cycle. Default 2.0.
    linestyle : str, optional
        Style of the mean cycle. Default ``'-'``. Colour carries the identity
        of the quantity and cannot also carry a grouping, so where two sources
        of one quantity share an axes they are told apart by this — the rule
        :data:`SOURCE_STYLE` states for the project.
    label : str or None, optional
        Legend label for the mean cycle. Default ``None``, no label.

    Returns
    -------
    cycle : pd.Series
        The mean cycle as drawn, indexed by position within the day in
        fractional hours: centred on its own mean when ``centre`` is true, in
        the channel's own units otherwise.
    amplitude : float
        Peak-to-trough amplitude of the mean cycle, which centring does not
        change. ``NaN`` for a circular quantity, whose peak-to-trough distance
        on a circle is not defined.
    n_days : int
        Days behind the panel: the complete days when ``complete_day`` is
        given, and every day carrying at least one observation otherwise.
    """
    if complete_day is not None:
        data, n_days = complete_days(data, column, complete_day)

    kept = data.dropna(subset=[column])
    if complete_day is None:
        n_days = len(set(kept.index.date))
    if not len(kept):
        return pd.Series(dtype=float), np.nan, 0

    # Position within the day, in fractional hours (0.0, 0.333, 0.667, ...).
    # Computed inline rather than through a study's own helper: this figure
    # primitive must not depend on any one study's library, and the formula
    # is a one-line fact about a `DatetimeIndex` rather than something worth
    # a name of its own here.
    clock = (kept.index.hour + kept.index.minute / 60.0
             + kept.index.second / 3600.0)

    if background and not circular:
        cycles = kept.pivot_table(values=column, index=clock,
                                  columns=kept.index.date)
        if centre:
            cycles = cycles.sub(cycles.mean(axis=0), axis=1)
        ax.plot(cycles.index, cycles, lw=0.5, color='0.7',
                alpha=background_alpha, zorder=1)

    values = kept[column]
    if circular:
        cycle = values.groupby(clock).apply(
            lambda group: meteo.circular_mean(group.to_numpy()))
    else:
        cycle = values.groupby(clock).mean()

    if min_days > 1:
        behind = pd.Series(kept.index.date, index=clock).groupby(
            level=0).nunique()
        cycle = cycle.where(behind.reindex(cycle.index) >= min_days)

    amplitude = np.nan if circular else float(cycle.max() - cycle.min())
    if centre and not circular:
        cycle = cycle - cycle.mean()

    if circular:
        ax.plot(cycle.index, cycle.values, linestyle='none', marker='.',
                markersize=3.5, color=colour, zorder=2, label=label)
    else:
        ax.plot(cycle.index, cycle.values, lw=linewidth, color=colour,
                linestyle=linestyle, zorder=2, label=label)
    return cycle, amplitude, n_days


# ──────────────────────────────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────────────────────────────

def finish(fig, save_path=None, filename=None):
    """
    Lay out a figure and, when a destination is given, save it as PNG and SVG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finish.
    save_path : str or None, optional
        Output directory. Created if absent. Default ``None``, which saves
        nothing.
    filename : str or None, optional
        Base name, without extension. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure, for chaining.

    Notes
    -----
    Writes two files when both ``save_path`` and ``filename`` are given: a PNG
    at 200 dpi and an SVG. With either omitted it only lays the figure out.
    """
    fig.tight_layout()
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, filename + '.png'),
                    dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(save_path, filename + '.svg'),
                    bbox_inches='tight')
    return fig
