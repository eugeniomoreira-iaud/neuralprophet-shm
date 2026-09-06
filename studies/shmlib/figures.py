"""
Module: shmlib.figures

The multi-source figures a proxy-forcing comparison draws.

Each function here takes the data and the choices, draws, optionally saves, and
returns the figure; none of them reads a path of its own or reaches for a
module-level default that was not passed as an argument. Colour comes from
:mod:`shmlib.viz`, where a quantity's identity is fixed for the whole project.
:func:`plot_diurnal_grid` lays out panels and draws each one with
:func:`shmlib.viz.draw_cycle`, the diurnal panel every study shares, while
:func:`plot_diurnal_comparison` — which overlays several sources in one axes
and so cannot draw the days behind any of them — takes its means from
:mod:`shmlib.compare`.

Moved here from study 2's now-deleted private library, where these seven functions were
written to inspect and compare the on-structure record against the two
external proxy sources. None of them is specific to that comparison — the
columns, the sources and the season definitions all arrive as arguments — so a
later study drawing the same kind of figure over a different comparison finds
it here rather than writing it a second time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from . import compare, coupling, meteo, proxies, site, viz


def _clip_note(series, limits, unit):
    """
    The note a clipped axis carries, naming what it leaves outside.

    Parameters
    ----------
    series : pd.Series
        The channel as drawn. Missing values are ignored.
    limits : tuple of float
        The axis limits the panel was clipped to.
    unit : str
        Unit of the channel, written after the extreme value.

    Returns
    -------
    str or None
        The note, or ``None`` when every sample is inside the limits and there
        is nothing to explain.
    """
    values = series.dropna()
    below = values[values < limits[0]]
    above = values[values > limits[1]]
    if not len(below) and not len(above):
        return None

    def phrase(n):
        return f'{n} sample' if n == 1 else f'{n} samples'

    if len(below) and len(above):
        return (f'{phrase(len(below) + len(above))} outside this range, '
                f'from {values.min():.3g} to {values.max():.3g} {unit}')
    if len(above):
        return (f'{phrase(len(above))} above this range, '
                f'to {values.max():.3g} {unit}')
    return (f'{phrase(len(below))} below this range, '
            f'to {values.min():.3g} {unit}')


def _reindex_regular(data, freq):
    """
    Reindex a series or frame onto a regular grid, or leave it untouched.

    Segmented modelling data skips a missing timestamp outright rather than
    carrying it as a ``NaN`` row, so a line plotted straight from it joins
    across an outage instead of breaking there — a figure that quietly draws
    the interpolation the study explicitly refuses to compute. Reindexing
    onto ``pd.date_range(data.index.min(), data.index.max(), freq=freq)``
    first restores every absent slot as ``NaN``, which matplotlib then
    renders as a gap in the line.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time-indexed data about to be plotted.
    freq : str or None
        Grid spacing to reindex onto. ``None`` returns ``data`` unchanged, so
        that a caller which does not pass a frequency keeps today's
        behaviour exactly.

    Returns
    -------
    pd.Series or pd.DataFrame
        ``data`` as given when ``freq`` is ``None``, otherwise reindexed onto
        the regular grid, same type as the input.
    """
    if freq is None:
        return data
    grid = pd.date_range(data.index.min(), data.index.max(), freq=freq)
    return data.reindex(grid)


def plot_source_panels(df, columns, title, colours=None, labels=None,
                       quantities=None, units=None, ranges=None,
                       circular=proxies.CIRCULAR, panel_height=1.15,
                       tick_years=1, clip=(), clip_quantiles=(0.0, 1.0),
                       clip_pad=0.05, save_path=None, filename=None):
    """
    The complete record of one source, one panel per channel, on one shared clock.

    Every channel a source reports is drawn as its own panel on a common time
    axis, each named by its own title and carrying its unit on the axis, so
    that a reader sees the source whole: when it starts, where it drops out,
    and whether different channels of the same instrument tend to go missing
    together. The panels are separated rather than butted together, on the
    layout study 1 uses for its own multi-channel figures, because a title per
    panel is what lets a reader name a channel without counting rows.
    Nothing is interpolated and nothing is dropped before drawing — a gap in
    the record stays a gap in the figure, exactly as it is in ``df``. This
    function is used twice in study 2, once for each external source, on the
    channels and in the order that study's notebook names, so that the two
    sources can be inspected in the same way before either is compared with
    the on-structure measurement.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying the source's suffixed columns on the
        analysis grid.
    columns : sequence of str
        Channels to draw, one panel each, in the order given, top to bottom.
    title : str
        Figure title.
    colours : dict of str to str or None, optional
        Canonical quantity name to line colour. Default ``None``, which uses
        :data:`shmlib.viz.QUANTITY_COLOUR`.
    labels : dict of str to str or None, optional
        Canonical quantity name to the title its panel is given. Default
        ``None``, which uses :data:`shmlib.proxies.QUANTITY_LABEL`. The unit
        is not taken from here: it is written on the y axis, from
        :data:`shmlib.proxies.QUANTITY_UNIT`.
    quantities : dict of str to str or None, optional
        Column name to the quantity key that resolves its colour, its label,
        its unit and, if clipped, its plausible range. Default ``None``,
        which keeps the rule this function was written for exactly:
        ``quantity = column.rpartition('_')[0]``, the source suffix stripped
        from a column such as ``'tair_str'``. Any column absent from
        ``quantities`` still falls back to that same rule, so a caller needs
        to name only the columns the rule gets wrong. The hook exists
        because a study whose columns are named for measured channels rather
        than for source-suffixed quantities — ``'inc_comp_cleaned'`` rather
        than, say, ``'inc_comp_str'`` — has no suffix for the rpartition
        rule to strip.
    units : dict of str to str or None, optional
        Quantity key to unit string, written on the y axis and in the clip
        note. Default ``None``, which uses
        :data:`shmlib.proxies.QUANTITY_UNIT` — the right default for a
        source-suffixed study, and the reason this parameter exists is the
        same as ``quantities``: a study keyed on measured channels needs its
        own unit for each one instead.
    ranges : dict of str to tuple of float or None, optional
        Quantity key to its plausible ``(low, high)`` range, read only inside
        the clipping branch to bound the quantiles a clipped axis is fitted
        to. Default ``None``, which uses
        :data:`shmlib.proxies.PLAUSIBLE_RANGE`. A study whose quantity keys
        are measured channels rather than external-proxy quantities passes
        its own ranges here, for the channels it wants clipped and knows a
        plausible bound for.
    circular : sequence of str, optional
        Quantities drawn as unconnected markers on a 0-360 degree axis rather
        than as a connected line, since a line joining 350 degrees to 10
        degrees would sweep the wrong way around the compass. Default
        :data:`shmlib.proxies.CIRCULAR`.
    panel_height : float, optional
        Height of one panel, in inches. The figure height is this value
        times the number of channels. Default ``1.15``, which leaves room for
        the title each panel carries.
    tick_years : int, optional
        Spacing of the year ticks on the shared bottom axis, in years.
        Default ``1``.
    clip : sequence of str, optional
        Canonical quantity names whose panel is drawn on a clipped vertical
        axis, with a note in the accent colour saying what is left outside.
        Default ``()``, which clips nothing. A channel carrying a value its
        own documented plausible range excludes compresses its whole record
        into one line and a spike, and the figure then shows the defect
        instead of the channel; clipping shows the channel and names the
        defect in words.
    clip_quantiles : tuple of float, optional
        Quantiles bounding a clipped axis, taken over the values inside the
        quantity's plausible range in :data:`shmlib.proxies.PLAUSIBLE_RANGE`.
        Default ``(0.0, 1.0)``, which keeps every plausible value and excludes
        only the impossible ones. Tighten it for a channel whose plausible
        values are themselves too skewed to draw.
    clip_pad : float, optional
        Fraction of the retained span added to each side of a clipped axis.
        Default ``0.05``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. Clipping changes the axis only: no value is dropped from
    the frame, from the channel inventory, or from any statistic computed
    elsewhere in the study.
    """
    colours = viz.QUANTITY_COLOUR if colours is None else colours
    labels = proxies.QUANTITY_LABEL if labels is None else labels
    quantities = {} if quantities is None else quantities
    units = proxies.QUANTITY_UNIT if units is None else units
    ranges = proxies.PLAUSIBLE_RANGE if ranges is None else ranges

    n = len(columns)
    fig, axes = plt.subplots(
        n, 1, figsize=viz.figsize(viz.FIGURE_WIDTH, panel_height * n),
        sharex=True)
    axes = np.atleast_1d(axes)

    for ax, column in zip(axes, columns):
        quantity = quantities.get(column, column.rpartition('_')[0])
        colour = colours[quantity]
        series = df[column]
        name = labels.get(quantity, quantity)
        if quantity in circular:
            # Eight years of hourly directions drawn per sample fill the panel
            # solid and say nothing. The daily circular mean is the coarsest
            # summary that still carries the seasonal preference this panel
            # exists to show, and the panel title says that is what it is.
            daily = meteo.circular_resample(series.dropna(), '1D', min_count=1)
            ax.plot(daily.index, daily.values, linestyle='none', marker='.',
                    markersize=1.2, alpha=0.5, color=colour)
            ax.set_ylim(0.0, 360.0)
            ax.set_yticks(list(viz.COMPASS_TICKS.keys()))
            ax.set_yticklabels(list(viz.COMPASS_TICKS.values()))
            name = f'{name}, daily circular mean'
        else:
            ax.plot(series.index, series.values, color=colour, lw=0.5)

        if quantity in clip:
            limits, n_outside = viz.robust_limits(
                series, *clip_quantiles, pad=clip_pad,
                within=ranges.get(quantity))
            ax.set_ylim(limits)
            note = _clip_note(series, limits,
                              units.get(quantity, ''))
            if note:
                # Placed opposite the excursions it explains, so that the note
                # does not cover the very samples the axis is clipping.
                above = (series.dropna() > limits[1]).any()
                ax.text(0.02, 0.94 if above else 0.06, note,
                        transform=ax.transAxes, fontsize='x-small',
                        color=viz.MARK_COLOUR, ha='left',
                        va='top' if above else 'bottom',
                        bbox=dict(facecolor='white', alpha=0.75,
                                  edgecolor='none', pad=2))

        # The panel names its channel and the axis carries the unit alone. Both
        # naming it would spend two lines of a short panel saying one thing.
        ax.set_ylabel(f'[{units.get(quantity, "")}]',
                      fontsize='x-small')
        ax.set_title(name, fontsize='small')
        viz.format_spines(ax)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=tick_years))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle(title, y=1.0, fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_channel_panels(df, columns, title, channels=None, panel_height=1.15,
                        tick_years=1, clip=(), clip_quantiles=(0.0, 1.0),
                        clip_pad=0.05, save_path=None, filename=None):
    """
    The complete record of the on-structure package, one panel per measured channel.

    A thin resolver over :func:`plot_source_panels`, written for the study
    that draws the project's own measured channels — inclination, air
    temperature, relative humidity, supply voltage, wall temperature, solar
    radiation — rather than a source's suffixed quantities. A channel's
    colour and label are fixed project-wide in :mod:`shmlib.viz`, not decided
    per study, so this function resolves each column to its channel and
    reads both from there, sparing a study that draws the on-structure
    package from restating them the way a study comparing external sources
    states its quantities' colours and labels through :func:`plot_source_panels`
    directly. It draws nothing of its own: every panel is the one
    :func:`plot_source_panels` draws, and this function's whole job is
    deciding what each column is called before calling it.

    Parameters
    ----------
    df : pd.DataFrame
        Frame carrying the channels on the analysis grid.
    columns : sequence of str
        Channels to draw, one panel each, in the order given, top to bottom.
    title : str
        Figure title.
    channels : dict of str to str or None, optional
        Column name to a key of :data:`shmlib.viz.CHANNEL_LABEL`. Default
        ``None``, which takes each column to name its own channel after
        stripping any source suffix in :data:`shmlib.proxies.SOURCES` it
        carries — ``'tair_str'`` resolves to ``'tair'`` and ``'twall_str'``
        to ``'twall'`` — so a column already named for its channel, such as
        ``'inc_comp_cleaned'``, is left as it is. A column resolving to a key
        absent from :data:`shmlib.viz.CHANNEL_LABEL` raises ``KeyError``
        naming the column, never a silent default colour, for the reason
        :func:`shmlib.viz.channel_style` already gives in its own docstring:
        a channel drawn in the wrong colour is worse than a figure that
        fails to build.
    panel_height : float, optional
        Height of one panel, in inches. Default ``1.15``. Forwarded to
        :func:`plot_source_panels`.
    tick_years : int, optional
        Spacing of the year ticks on the shared bottom axis, in years.
        Default ``1``. Forwarded to :func:`plot_source_panels`.
    clip : sequence of str, optional
        Channel keys whose panel is drawn on a clipped vertical axis, with a
        note in the accent colour saying what is left outside. Default
        ``()``, which clips nothing. Forwarded to :func:`plot_source_panels`.
    clip_quantiles : tuple of float, optional
        Quantiles bounding a clipped axis. Default ``(0.0, 1.0)``. Forwarded
        to :func:`plot_source_panels`.
    clip_pad : float, optional
        Fraction of the retained span added to each side of a clipped axis.
        Default ``0.05``. Forwarded to :func:`plot_source_panels`.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure :func:`plot_source_panels` returns.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. No measured channel is a compass direction, so
    :func:`plot_source_panels` is called with ``circular=()`` regardless of
    :data:`shmlib.proxies.CIRCULAR`.
    """
    if channels is None:
        channels = {}
        for column in columns:
            channel = column
            for source in proxies.SOURCES:
                suffix = f'_{source}'
                if column.endswith(suffix):
                    channel = column[:-len(suffix)]
                    break
            channels[column] = channel

    for column in columns:
        channel = channels.get(column, column)
        if channel not in viz.CHANNEL_LABEL:
            raise KeyError(f"column '{column}' resolves to channel "
                           f"'{channel}', not a key of viz.CHANNEL_LABEL")

    colours = {channels[column]: viz.CHANNEL_COLOUR[channels[column]]
              for column in columns}
    labels = {channels[column]: viz.channel_name(channels[column])
             for column in columns}
    units = {channels[column]: viz.channel_unit(channels[column]).strip('[]')
            for column in columns}

    return plot_source_panels(
        df, columns, title, colours=colours, labels=labels,
        quantities=channels, units=units, ranges=None, circular=(),
        panel_height=panel_height, tick_years=tick_years, clip=clip,
        clip_quantiles=clip_quantiles, clip_pad=clip_pad,
        save_path=save_path, filename=filename)


def plot_diurnal_grid(df, columns, season, season_months, title, colours=None,
                      labels=None, ncols=2, circular=proxies.CIRCULAR,
                      complete_day=None, min_days=1, centre=False,
                      background=True, background_alpha=0.08,
                      panel_height=1.9, save_path=None, filename=None):
    """
    The mean daily cycle of every channel of one source, as a grid of panels.

    One small panel per channel, laid out ``ncols`` across and filled in
    reading order, so that a reader can compare the shape and timing of every
    channel a source reports without paging through separate figures. One
    season to a figure: this is the layout for showing many channels of one
    source at once, where the seasonal comparison is not the question being
    asked. Where it is, :func:`plot_diurnal_season_grid` puts the seasons in
    the columns of a single figure and shares a vertical scale along each row,
    which is what study 2 draws — so this function currently has no caller.

    Each panel is drawn by :func:`shmlib.viz.draw_cycle`, the panel every
    diurnal figure in the project shares: the individual days behind the mean
    in gray, the mean over them, and the days and the amplitude named in the
    panel title. The report requires every diurnal profile to carry an
    explicit representation of the observations it rests on, and the days
    drawn behind the mean, counted in the title, are where that requirement is
    met.

    Parameters
    ----------
    df : pd.DataFrame
        Frame on the analysis grid, carrying the source's suffixed columns.
        Restricted to ``season`` here rather than by the caller, so that the
        days behind each mean are available to be drawn.
    columns : sequence of str
        Channels to draw, one panel each.
    season : str
        Season to restrict to, as named in ``season_months``.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    title : str
        Figure title.
    colours : dict of str to str or None, optional
        Canonical quantity name to line colour. Default ``None``, which uses
        :data:`shmlib.viz.QUANTITY_COLOUR`.
    labels : dict of str to str or None, optional
        Canonical quantity name to y-axis label. Default ``None``, which uses
        :data:`shmlib.viz.QUANTITY_PANEL_LABEL`.
    ncols : int, optional
        Panels per row. Default ``2``.
    circular : sequence of str, optional
        Quantities averaged by the unit-vector method and drawn as markers on
        a 0-360 degree axis rather than as a connected line. Default
        :data:`shmlib.proxies.CIRCULAR`. Pass an empty sequence for a source
        that reports no directional channel.
    complete_day : int or None, optional
        Slots a day must carry to contribute at all. Default ``None``, which
        admits every day — the right rule for a climatology spanning years,
        where a completeness rule would discard most of the record and gain
        nothing.
    min_days : int, optional
        Days a position within the day must draw on before its mean is drawn.
        Default ``1``.
    centre : bool, optional
        Draw the excursion rather than the level, each day centred on its own
        mean. Default ``False``: a panel describing what a source reports is
        about the level as well as the shape.
    background : bool, optional
        Draw the individual days in gray behind each mean. Default ``True``.
    background_alpha : float, optional
        Opacity of the individual days. Default ``0.08``, low because a
        climatology over several years puts hundreds of days in one panel and
        a heavier stroke fills it solid.
    panel_height : float, optional
        Height of one row of panels, in inches. Default ``1.9``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. Unused cells of the grid, when the number of channels
    does not fill it exactly, are switched off rather than left blank axes.
    """
    colours = viz.QUANTITY_COLOUR if colours is None else colours
    labels = viz.QUANTITY_PANEL_LABEL if labels is None else labels

    season_labels = site.season_of(df.index, season_months)
    data = df.loc[season_labels.values == season]
    day_noun = 'complete days' if complete_day is not None else 'days'

    n = len(columns)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=viz.figsize(viz.FIGURE_WIDTH, panel_height * nrows),
        squeeze=False)
    flat = axes.flatten()

    for index, column in enumerate(columns):
        ax = flat[index]
        quantity = column.rpartition('_')[0]
        is_circular = quantity in circular

        _, amplitude, n_days = viz.draw_cycle(
            ax, data, column, colours[quantity], complete_day=complete_day,
            min_days=min_days, centre=centre, background=background,
            background_alpha=background_alpha, circular=is_circular,
            linewidth=1.5)

        if is_circular:
            ax.set_ylim(0.0, 360.0)
            ax.set_yticks(list(viz.COMPASS_TICKS.keys()))
            ax.set_yticklabels(list(viz.COMPASS_TICKS.values()))

        name = proxies.QUANTITY_LABEL.get(quantity, quantity)
        count = f'{n_days:,} {day_noun}'
        # A circular quantity has no peak-to-trough amplitude to report: the
        # distance between two bearings on a circle depends on which way round
        # it is measured.
        summary = (count if np.isnan(amplitude)
                   else f'{count}, amplitude {amplitude:.1f}')
        ax.set_title(f'{name}\n({summary})', fontsize='small')
        ax.set_ylabel(labels.get(quantity, quantity), fontsize='x-small')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlabel('Hour of day', fontsize='x-small')
        ax.grid(True, alpha=0.3)
        viz.format_spines(ax)

    for index in range(n, nrows * ncols):
        flat[index].set_visible(False)

    fig.suptitle(title, y=1.02, fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_diurnal_season_grid(df, columns, seasons, season_months, title,
                             colours=None, labels=None,
                             circular=proxies.CIRCULAR, complete_day=None,
                             min_days=1, centre=False, background=True,
                             background_alpha=0.08, panel_height=1.5,
                             save_path=None, filename=None):
    """
    One source's mean daily cycles, a channel per row and a season per column.

    The seasonal comparison :func:`plot_diurnal_grid` splits across separate
    figures, brought into one. A reader asking how a channel's day differs
    between summer and winter has the two answers side by side and on one
    vertical scale, because the panels of a row share their y axis: the
    difference between the seasons is then a difference in the drawn shape
    rather than something to be reconstructed from two axes with different
    limits, which is what comparing two figures across a page requires.

    Rows follow the order of ``columns`` and columns the order of ``seasons``,
    both as given, since the order channels are read in is an editorial
    decision belonging to the study rather than to this function.

    Each panel is drawn by :func:`shmlib.viz.draw_cycle`, the diurnal panel
    every study shares: the contributing days in gray, the mean over them, and
    the count and amplitude named above the panel.

    Parameters
    ----------
    df : pd.DataFrame
        Frame on the analysis grid, carrying the source's suffixed columns.
        Restricted to each season here rather than by the caller, so that the
        days behind every mean are available to be drawn.
    columns : sequence of str
        Channels to draw, one row each, top to bottom in the order given.
    seasons : sequence of str
        Seasons to draw, one column each, left to right in the order given, as
        named in ``season_months``.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    title : str
        Figure title.
    colours : dict of str to str or None, optional
        Canonical quantity name to line colour. Default ``None``, which uses
        :data:`shmlib.viz.QUANTITY_COLOUR`.
    labels : dict of str to str or None, optional
        Canonical quantity name to y-axis label. Default ``None``, which uses
        :data:`shmlib.viz.QUANTITY_PANEL_LABEL`.
    circular : sequence of str, optional
        Quantities averaged by the unit-vector method and drawn as markers on
        a 0-360 degree axis rather than as a connected line. Default
        :data:`shmlib.proxies.CIRCULAR`.
    complete_day : int or None, optional
        Slots a day must carry to contribute at all. Default ``None``, which
        admits every day.
    min_days : int, optional
        Days a position within the day must draw on before its mean is drawn.
        Default ``1``.
    centre : bool, optional
        Draw the excursion rather than the level, each day centred on its own
        mean. Default ``False``.
    background : bool, optional
        Draw the individual days in gray behind each mean. Default ``True``.
    background_alpha : float, optional
        Opacity of the individual days. Default ``0.08``.
    panel_height : float, optional
        Height of one row of panels, in inches. Default ``1.5``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given.
    """
    colours = viz.QUANTITY_COLOUR if colours is None else colours
    labels = viz.QUANTITY_PANEL_LABEL if labels is None else labels

    season_labels = site.season_of(df.index, season_months)
    day_noun = 'complete days' if complete_day is not None else 'days'

    nrows, ncols = len(columns), len(seasons)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=viz.figsize(viz.FIGURE_WIDTH, panel_height * nrows),
        squeeze=False, sharex=True, sharey='row')

    for row, column in enumerate(columns):
        quantity = column.rpartition('_')[0]
        is_circular = quantity in circular

        for col, season in enumerate(seasons):
            ax = axes[row][col]
            data = df.loc[season_labels.values == season]

            _, amplitude, n_days = viz.draw_cycle(
                ax, data, column, colours[quantity], complete_day=complete_day,
                min_days=min_days, centre=centre, background=background,
                background_alpha=background_alpha, circular=is_circular,
                linewidth=1.5)

            if is_circular:
                ax.set_ylim(0.0, 360.0)
                ax.set_yticks(list(viz.COMPASS_TICKS.keys()))
                ax.set_yticklabels(list(viz.COMPASS_TICKS.values()))

            count = f'{n_days:,} {day_noun}'
            # A circular quantity has no peak-to-trough amplitude to report:
            # the distance between two bearings on a circle depends on which
            # way round it is measured.
            summary = (count if np.isnan(amplitude)
                       else f'{count}, amplitude {amplitude:.1f}')
            # The season names the column, so it is written once, above the
            # top row. Every panel carries what its own mean rests on.
            heading = f'{season.capitalize()}\n{summary}' if row == 0 else summary
            ax.set_title(heading, fontsize='x-small')

            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 6))
            ax.grid(True, alpha=0.3)
            viz.format_spines(ax)

        axes[row][0].set_ylabel(labels.get(quantity, quantity),
                                fontsize='x-small')

    for ax in axes[-1]:
        ax.set_xlabel('Hour of day', fontsize='x-small')

    fig.suptitle(title, y=1.01, fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_three_source_series(df, quantity, sources=proxies.SOURCES, title='',
                             labels=None, span=None, height=2.2,
                             save_path=None, filename=None):
    """
    Several sources of one quantity on one time axis, over a stated span.

    Colour carries the quantity, held constant across every line, and line
    style tells the sources apart — the same convention the rest of the
    project uses to separate a grouping from an identity once colour is
    already spent. Drawn before any agreement statistic is computed, so a
    reader sees what several sources of one forcing look like together before
    being told a number for how far apart they stand.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    sources : sequence of str, optional
        Sources to draw, in legend order. Default
        :data:`shmlib.proxies.SOURCES`. A source whose column is absent from
        ``df`` is skipped rather than raising — the ground station carries no
        wall temperature, and asking for one must not be an error.
    title : str, optional
        Axes title. Default ``''``.
    labels : dict of str to str or None, optional
        Source suffix to legend label. Default ``None``, which uses
        :data:`shmlib.proxies.SOURCE_LABEL`.
    span : tuple or None, optional
        ``(start, end)`` bounds, each a string or timestamp, restricting the
        window drawn. Default ``None``, the full index. When given, the
        bounds are appended to the axes title so the restriction is visible
        on the figure itself.
    height : float, optional
        Figure height in inches. Default ``2.2``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given.
    """
    labels = proxies.SOURCE_LABEL if labels is None else labels
    colour = viz.QUANTITY_COLOUR[quantity]

    view = df if span is None else df.loc[span[0]:span[1]]
    full_title = title
    if span is not None:
        start, end = pd.Timestamp(span[0]), pd.Timestamp(span[1])
        window = f'{start:%Y-%m-%d} to {end:%Y-%m-%d}'
        full_title = f'{title} ({window})' if title else window

    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, height))
    for source in sources:
        column = f'{quantity}_{source}'
        if column not in view.columns:
            continue
        ax.plot(view.index, view[column], color=colour,
               linestyle=viz.SOURCE_STYLE[source], lw=0.7,
               label=labels.get(source, source))

    ax.set_ylabel(viz.QUANTITY_AXIS_LABEL[quantity])
    ax.set_title(full_title)
    # Below the axes, per the Graphical Guidelines: a legend inside covers the
    # record it is labelling, and `loc='best'` covers a different part of it
    # every time the figure is rebuilt.
    ax.legend(fontsize='small', ncol=max(len(sources), 1), loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.format_spines(ax)
    viz.finish(fig, save_path, filename)
    return fig


def plot_source_scatter(df, quantity, reference='str', compared=None,
                        fits=None, title='', height=2.6, save_path=None,
                        filename=None):
    """
    Each external source against the reference, one panel per source.

    Drawn as a hexbin rather than a scatter of points, because the record
    behind these panels runs to years of hourly samples and a point cloud at
    that density would show nothing but ink; the density-aware colouring
    lets the shape of the agreement come through where a plain scatter would
    saturate. The panels are square with equal axis limits on both sides, so
    a constant bias is visible as a parallel displacement of the cloud from
    the 1:1 line rather than as a change of slope that a reader has to
    compute.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    reference : str, optional
        Source suffix drawn on the horizontal axis of every panel. Default
        ``'str'``, the on-structure measurement.
    compared : sequence of str or None, optional
        Sources to score, one panel each, in order. Default ``None``, every
        other source in :data:`shmlib.proxies.SOURCES`. A source whose column
        is absent from ``df`` is skipped rather than raising.
    fits : pd.DataFrame or None, optional
        The frame :func:`shmlib.compare.calibrate_against_sensor` returns,
        carrying at least ``compared``, ``slope``, ``intercept``, ``r2`` and
        ``n``. When given, the row for each source's overall window
        (``split == 'all'`` when a ``split`` column is present, since the
        frame may also carry a row per season) is drawn as a fitted line over
        the panel, with its slope, intercept, R-squared and sample count
        printed in the panel's corner. Default ``None``, which draws no fit.
    title : str, optional
        Figure title. Default ``''``.
    height : float, optional
        Figure height in inches. Default ``2.6``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. Pairs with a missing value on either side are dropped
    before drawing, since a hexbin and a linear fit both require paired,
    non-missing observations; this is a scatter of paired values rather than
    a time series, so it is not covered by the project's rule against
    dropping missing samples from a series before plotting it.
    """
    compared = ([source for source in proxies.SOURCES if source != reference]
               if compared is None else compared)
    reference_column = f'{quantity}_{reference}'
    present = [source for source in compared
              if reference_column in df.columns
              and f'{quantity}_{source}' in df.columns]

    fig, axes = plt.subplots(
        1, len(present), figsize=viz.figsize(viz.FIGURE_WIDTH, height),
        squeeze=False)
    axes = axes[0]

    for ax, source in zip(axes, present):
        compared_column = f'{quantity}_{source}'
        paired = df[[reference_column, compared_column]].dropna()
        x = paired[reference_column].to_numpy()
        y = paired[compared_column].to_numpy()

        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))

        # Log-scaled bins. The count distribution is extremely skewed — one
        # bin near the origin holds more than all the daylight bins together —
        # so on a linear scale every bin that carries the relationship reads as
        # the palest colour and the panel says nothing.
        hexbin = ax.hexbin(x, y, gridsize=35, cmap='cividis_r', mincnt=1,
                           bins='log',
                           extent=(low, high, low, high))
        colourbar = fig.colorbar(hexbin, ax=ax, shrink=0.85)
        colourbar.set_label('count (log)', fontsize='x-small')
        colourbar.ax.tick_params(labelsize='x-small')

        ax.plot([low, high], [low, high], color=viz.MARK_COLOUR, lw=1.0,
               ls='--', zorder=3)

        if fits is not None:
            candidates = fits[fits['compared'] == source]
            if 'split' in candidates.columns:
                candidates = candidates[candidates['split'] == 'all']
            if len(candidates):
                row = candidates.iloc[0]
                fitted = (np.array([low, high]) * row['slope']
                         + row['intercept'])
                ax.plot([low, high], fitted, color=viz.QUANTITY_COLOUR[quantity],
                       lw=1.2, zorder=4)
                ax.text(0.03, 0.97,
                       f"slope {row['slope']:.2f}\n"
                       f"intercept {row['intercept']:.2f}\n"
                       f"$R^2$ {row['r2']:.2f}\nn={int(row['n']):,}",
                       transform=ax.transAxes, fontsize='x-small',
                       va='top', ha='left')

        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect('equal')
        ax.set_xlabel(f'{proxies.SOURCE_LABEL.get(reference, reference)} '
                     f'[{proxies.QUANTITY_UNIT[quantity]}]', fontsize='small')
        ax.set_title(proxies.SOURCE_LABEL.get(source, source), fontsize='small')
        viz.format_spines(ax)

    axes[0].set_ylabel(f'External source [{proxies.QUANTITY_UNIT[quantity]}]',
                       fontsize='small')
    fig.suptitle(title, y=1.03, fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_diurnal_comparison(df, quantity, seasons, season_months,
                            sources=proxies.SOURCES, title='', profile_fn=None,
                            height=2.2, save_path=None, filename=None):
    """
    Several sources' mean daily cycle side by side, one panel per season.

    Puts the diurnal comparison a proxy-forcing study is built around in one
    figure: the same quantity, the same seasons as every other figure in the
    study, and every source drawn on top of each other so that an amplitude
    or phase difference between the sources is a shape a reader can see
    rather than a number they have to look up in a table.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame.
    quantity : str
        Canonical quantity name.
    seasons : sequence of str
        Seasons to draw, one panel each, in order.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    sources : sequence of str, optional
        Sources to draw, told apart by line style. Default
        :data:`shmlib.proxies.SOURCES`. A source whose column is absent from
        ``df`` is skipped.
    title : str, optional
        Figure title. Default ``''``.
    profile_fn : callable or None, optional
        Function computing the diurnal profile of a set of columns, called
        as ``profile_fn(df, columns, season, season_months)``. Default
        ``None``, which uses :func:`shmlib.compare.diurnal_profile`. Kept as
        an argument rather than a hard-coded call so that the notebook can
        pass a version built with different day-completeness rules without
        this function growing a second set of parameters that duplicate
        :func:`shmlib.compare.diurnal_profile`'s own.
    height : float, optional
        Figure height in inches. Default ``2.2``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given.
    """
    profile_fn = compare.diurnal_profile if profile_fn is None else profile_fn
    present = [source for source in sources
              if f'{quantity}_{source}' in df.columns]
    columns = [f'{quantity}_{source}' for source in present]

    fig, axes = plt.subplots(
        1, len(seasons), figsize=viz.figsize(viz.FIGURE_WIDTH, height),
        sharey=True)
    axes = np.atleast_1d(axes)

    for ax, season in zip(axes, seasons):
        profile = profile_fn(df, columns, season, season_months)
        for source in present:
            column = f'{quantity}_{source}'
            if not len(profile) or column not in profile.columns:
                continue
            ax.plot(profile.index, profile[column],
                   color=viz.QUANTITY_COLOUR[quantity],
                   linestyle=viz.SOURCE_STYLE[source], lw=1.5,
                   label=proxies.SOURCE_LABEL.get(source, source))
        ax.set_title(season.capitalize(), fontsize='small')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlabel('Hour of day')
        viz.format_spines(ax)

    axes[0].set_ylabel(viz.QUANTITY_AXIS_LABEL[quantity])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize='small', loc='upper center',
              bbox_to_anchor=(0.5, -0.02), ncol=max(len(present), 1),
              frameon=False)
    fig.suptitle(title, y=1.08, fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_pair_scatter_grid(df, quantities, reference, compared, scores=None,
                           ncols=2, panel_height=1.9, gridsize=30,
                           save_path=None, filename=None):
    """
    Two sources against each other, one square panel per quantity.

    The marginal view of a source-against-source comparison: every hour on
    which both report, drawn as a hexbin against the 1:1 line, with one panel
    per quantity so that the whole vocabulary the two sources share can be
    read at once. A constant offset appears as a cloud displaced parallel to
    the diagonal; a source that compresses the range appears as a cloud
    flatter than the diagonal, which is what an amplitude ratio below one
    looks like before it is reduced to a number.

    Drawn as a hexbin rather than a scatter of points, on log-scaled bins, for
    the reason :func:`plot_source_scatter` gives: years of hourly samples
    saturate a point cloud, and the count distribution is skewed enough that
    linear bins leave every informative cell at the palest colour. Darker is
    more samples. No colour bar is drawn, because the panels are small and
    each carries its own count scale; the shape of the cloud against the
    diagonal is what these panels are for, not the reading of a density.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying both sources' suffixed columns.
    quantities : sequence of str
        Canonical quantity names, one panel each, in the order given.
    reference : str
        Source suffix on the horizontal axis of every panel.
    compared : str
        Source suffix on the vertical axis of every panel.
    scores : pd.DataFrame or None, optional
        The frame :func:`shmlib.compare.pairwise_agreement` returns. When
        given, the bias and correlation of each quantity's whole-record row
        are printed in its panel, taken from that frame rather than computed
        again here, so the figure and the table cannot disagree. Default
        ``None``, which prints nothing.
    ncols : int, optional
        Panels per row. Default ``2``.
    panel_height : float, optional
        Height of one row of panels, in inches. Default ``1.9``.
    gridsize : int, optional
        Hexbin resolution. Default ``30``.
    save_path, filename : str or None, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. Pairs missing on either side are dropped before drawing:
    a hexbin requires paired observations, and this is a scatter of pairs
    rather than a series, so the project's rule against dropping missing
    samples from a plotted series does not apply.
    """
    present = [quantity for quantity in quantities
               if f'{quantity}_{reference}' in df.columns
               and f'{quantity}_{compared}' in df.columns]

    n = len(present)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=viz.figsize(viz.FIGURE_WIDTH, panel_height * nrows),
        squeeze=False)
    flat = axes.flatten()

    for index, quantity in enumerate(present):
        ax = flat[index]
        paired = df[[f'{quantity}_{reference}',
                     f'{quantity}_{compared}']].dropna()
        x = paired[f'{quantity}_{reference}'].to_numpy()
        y = paired[f'{quantity}_{compared}'].to_numpy()
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))

        ax.hexbin(x, y, gridsize=gridsize, cmap='cividis_r', mincnt=1,
                  bins='log', extent=(low, high, low, high))
        ax.plot([low, high], [low, high], color=viz.MARK_COLOUR, lw=1.0,
                ls='--', zorder=3)

        if scores is not None:
            row = scores[(scores['quantity'] == quantity)
                         & (scores['compared'] == compared)]
            if 'split' in row.columns:
                row = row[row['split'] == 'all']
            if len(row):
                row = row.iloc[0]
                ax.text(0.04, 0.96,
                        f"bias {row['bias']:.2f}\nr {row['r']:.2f}",
                        transform=ax.transAxes, fontsize='xx-small',
                        va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.75,
                                  edgecolor='none', pad=1.5))

        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect('equal')
        ax.set_title(f'{proxies.QUANTITY_LABEL.get(quantity, quantity)} '
                     f'[{proxies.QUANTITY_UNIT.get(quantity, "")}]',
                     fontsize='small')
        ax.tick_params(labelsize='xx-small')
        viz.format_spines(ax)

    for index in range(n, nrows * ncols):
        flat[index].set_visible(False)

    fig.supxlabel(proxies.SOURCE_LABEL.get(reference, reference),
                  fontsize='small')
    fig.supylabel(proxies.SOURCE_LABEL.get(compared, compared),
                  fontsize='small')
    viz.finish(fig, save_path, filename)
    return fig


def plot_pair_bias_grid(df, quantities, reference, compared, freq='MS',
                        min_hours=100, ncols=2, panel_height=1.5,
                        profile_fn=None, save_path=None, filename=None):
    """
    Whether two sources' disagreement holds still, one panel per quantity.

    The temporal counterpart of :func:`plot_pair_scatter_grid`. Whether one
    source may stand in for another turns on whether their difference is
    *stable*, not on whether it is small: a constant offset can be subtracted
    and the substitution documented, while an offset that changes sign with
    the season cannot, and the two are indistinguishable in any statistic
    pooled over the whole record. Each panel draws the bias window by window
    with the interquartile range of the difference behind it, so that a drift
    and a widening are told apart.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying both sources' suffixed columns.
    quantities : sequence of str
        Canonical quantity names, one panel each, in the order given.
    reference : str
        Source the difference is measured against.
    compared : str
        Source measured against it. A positive bias is this source reading
        high.
    freq : str, optional
        Window the bias is tracked over, as a pandas offset alias. Default
        ``'MS'``, calendar months.
    min_hours : int, optional
        Paired hours a window must hold to be reported. Default ``100``.
    ncols : int, optional
        Panels per row. Default ``2``.
    panel_height : float, optional
        Height of one row of panels, in inches. Default ``1.5``.
    profile_fn : callable or None, optional
        Function computing the windowed bias, called as
        ``profile_fn(df, quantity, reference=..., compared=[...], freq=...,
        min_hours=...)``. Default ``None``, which uses
        :func:`shmlib.compare.agreement_stability`. Kept as an argument so a
        notebook can pass a version built with different rules rather than
        this function growing a second set of parameters duplicating that
        function's own.
    save_path, filename : str or None, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. A quantity whose windows all fall below ``min_hours``
    contributes an empty panel rather than being dropped, so that the grid
    stays aligned with the quantity list the caller gave.
    """
    profile_fn = compare.agreement_stability if profile_fn is None else profile_fn
    present = [quantity for quantity in quantities
               if f'{quantity}_{reference}' in df.columns
               and f'{quantity}_{compared}' in df.columns]

    n = len(present)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=viz.figsize(viz.FIGURE_WIDTH, panel_height * nrows),
        squeeze=False, sharex=True)
    flat = axes.flatten()

    for index, quantity in enumerate(present):
        ax = flat[index]
        colour = viz.QUANTITY_COLOUR[quantity]
        stability = profile_fn(df, quantity, reference=reference,
                               compared=[compared], freq=freq,
                               min_hours=min_hours)
        subset = (stability[stability['compared'] == compared]
                  .sort_values('window') if len(stability) else stability)

        if len(subset):
            ax.fill_between(subset['window'],
                            subset['bias'] - subset['iqr'] / 2.0,
                            subset['bias'] + subset['iqr'] / 2.0,
                            color=colour, alpha=0.15, lw=0, zorder=2)
            ax.plot(subset['window'], subset['bias'], color=colour, lw=1.2,
                    zorder=3)
        ax.axhline(0.0, color='#000000', lw=0.6, zorder=1)
        ax.set_title(f'{proxies.QUANTITY_LABEL.get(quantity, quantity)} '
                     f'[{proxies.QUANTITY_UNIT.get(quantity, "")}]',
                     fontsize='small')
        ax.tick_params(labelsize='xx-small')
        viz.format_spines(ax)

    for index in range(n, nrows * ncols):
        flat[index].set_visible(False)

    # The date axis is shared, so matplotlib labels only the bottom cell of
    # each column — and where the grid does not divide evenly that cell is one
    # of the hidden ones, leaving its column with no dates at all. The last
    # *visible* panel of every column is labelled instead.
    for column in range(ncols):
        visible = [index for index in range(column, n, ncols)]
        if not visible:
            continue
        ax = flat[visible[-1]]
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(labelbottom=True, labelsize='xx-small')

    fig.supylabel(f'{proxies.SOURCE_LABEL.get(compared, compared)} minus '
                  f'{proxies.SOURCE_LABEL.get(reference, reference)}',
                  fontsize='small')
    viz.finish(fig, save_path, filename)
    return fig


def plot_diurnal_source_grid(df, quantities, seasons, season_months, sources,
                             colours=None, circular=proxies.CIRCULAR,
                             complete_day=None, min_days=1, centre=False,
                             panel_height=1.5, save_path=None, filename=None):
    """
    Several sources' mean daily cycles overlaid, a quantity per row and a
    season per column.

    The within-the-day view of a source comparison. Two sources may agree on
    the level of a quantity and still disagree about its day — the hour it
    peaks, and how far it travels between its extremes — and neither a scatter
    of paired hours nor a windowed bias can show that, because both average
    the time of day away. Here the two cycles are drawn on one axes per season,
    the sources told apart by line style because colour is already spent on the
    identity of the quantity, and the panels of a row share a vertical scale so
    that a difference in amplitude is a difference in the drawn shape.

    Each cycle is drawn by :func:`shmlib.viz.draw_cycle`, the diurnal panel
    every study shares. The individual days are not drawn behind these means:
    two sources' day clouds over the same panel would obscure both cycles,
    which are what the panel exists to compare.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised frame carrying every source's suffixed columns.
    quantities : sequence of str
        Canonical quantity names, one row each, in the order given.
    seasons : sequence of str
        Seasons, one column each, in the order given.
    season_months : dict of str to sequence of int
        Months belonging to each season.
    sources : sequence of str
        Source suffixes to overlay, told apart by
        :data:`shmlib.viz.SOURCE_STYLE`. A source whose column is absent for a
        quantity is skipped for that row.
    colours : dict of str to str or None, optional
        Canonical quantity name to line colour. Default ``None``, which uses
        :data:`shmlib.viz.QUANTITY_COLOUR`.
    circular : sequence of str, optional
        Quantities averaged by the unit-vector method and drawn as markers on
        a compass axis. Default :data:`shmlib.proxies.CIRCULAR`.
    complete_day : int or None, optional
        Slots a day must carry to contribute at all. Default ``None``.
    min_days : int, optional
        Days a position within the day must draw on. Default ``1``.
    centre : bool, optional
        Draw the excursion rather than the level. Default ``False``: two
        sources' levels are part of what is being compared.
    panel_height : float, optional
        Height of one row of panels, in inches. Default ``1.5``.
    save_path, filename : str or None, optional
        Where to save. Default ``None``, which saves nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given.
    """
    colours = viz.QUANTITY_COLOUR if colours is None else colours
    season_labels = site.season_of(df.index, season_months)

    nrows, ncols = len(quantities), len(seasons)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=viz.figsize(viz.FIGURE_WIDTH, panel_height * nrows),
        squeeze=False, sharex=True, sharey='row')

    for row, quantity in enumerate(quantities):
        is_circular = quantity in circular
        for col, season in enumerate(seasons):
            ax = axes[row][col]
            data = df.loc[season_labels.values == season]

            for source in sources:
                column = f'{quantity}_{source}'
                if column not in df.columns:
                    continue
                viz.draw_cycle(
                    ax, data, column, colours[quantity],
                    complete_day=complete_day, min_days=min_days,
                    centre=centre, background=False, circular=is_circular,
                    linewidth=1.4,
                    linestyle=viz.SOURCE_STYLE.get(source, '-'))

            if is_circular:
                ax.set_ylim(0.0, 360.0)
                ax.set_yticks(list(viz.COMPASS_TICKS.keys()))
                ax.set_yticklabels(list(viz.COMPASS_TICKS.values()))
            if row == 0:
                ax.set_title(season.capitalize(), fontsize='small')
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 6))
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize='xx-small')
            viz.format_spines(ax)

        axes[row][0].set_ylabel(
            viz.QUANTITY_PANEL_LABEL.get(quantity, quantity),
            fontsize='xx-small')

    for ax in axes[-1]:
        ax.set_xlabel('Hour of day', fontsize='x-small')

    # Neutral handles, drawn for the legend alone. Taking them from a panel
    # would give every entry that panel's quantity colour, and colour in this
    # project names the quantity — a legend saying the sources are orange
    # would be asserting something false about the five rows drawn in other
    # colours. Style is what distinguishes the sources, so style is all the
    # legend shows.
    handles = [Line2D([], [], color='0.3', lw=1.4,
                      linestyle=viz.SOURCE_STYLE.get(source, '-'),
                      label=proxies.SOURCE_LABEL.get(source, source))
               for source in sources]
    fig.legend(handles=handles, fontsize='small', loc='upper center',
               bbox_to_anchor=(0.5, -0.01), ncol=max(len(sources), 1),
               frameon=False)
    viz.finish(fig, save_path, filename)
    return fig


def plot_certified_window(census, title='', state_colours=None, height=1.9,
                          tick_months=1, save_path=None, filename=None):
    """
    What became of every day of the on-structure radiation channel's life.

    A day-by-day strip spanning the whole current instrument era, one thin
    vertical band per calendar day, coloured by the state
    :func:`shmlib.quality.sr_day_census` assigned it. The certified window a
    proxy-forcing study calibrates against is a subset of the days coloured
    ``'certified'`` here, and the point of the figure is to show, at a
    glance, how much of the channel's life that subset actually is.

    Parameters
    ----------
    census : pd.DataFrame
        Output of :func:`shmlib.quality.sr_day_census`: one row per calendar
        day, carrying ``day``, ``n_hours``, ``n_night``, ``condemned`` and
        ``state``, where ``state`` is one of ``'no record'``,
        ``'condemned'`` or ``'certified'``.
    title : str, optional
        Axes title. Default ``''``.
    state_colours : dict of str to str or None, optional
        State name to colour. Default ``None``, which colours ``'certified'``
        in :data:`shmlib.viz.QUANTITY_COLOUR`'s radiation colour — the state
        that *is* radiation keeps radiation's identity colour —
        ``'condemned'`` in light gray, and ``'no record'`` in white with a
        hairline gray edge, drawn separately from the fill.
    height : float, optional
        Figure height in inches. Default ``1.9``.
    tick_months : int, optional
        Spacing of the month ticks, in months. Default ``1``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given.
    """
    default_colours = {'certified': viz.QUANTITY_COLOUR['sr'],
                       'condemned': '0.75', 'no record': '#FFFFFF'}
    state_colours = (default_colours if state_colours is None
                     else state_colours)

    days = pd.DatetimeIndex(census['day'])
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, height))

    for state in ('no record', 'condemned', 'certified'):
        mask = (census['state'] == state).to_numpy()
        if not mask.any():
            continue
        colour = state_colours.get(state, '0.5')
        edge = '0.6' if state == 'no record' else 'none'
        width = 0.4 if state == 'no record' else 0.0
        ax.bar(days[mask], 1.0, width=1.0, color=colour, edgecolor=edge,
              linewidth=width, align='edge',
              label=f'{state} ({int(mask.sum())})')

    ax.set_xlim(days.min(), days.max() + pd.Timedelta(days=1))
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title(title)
    ax.legend(fontsize='small', loc='upper center',
             bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)

    viz.format_spines(ax)
    ax.spines['left'].set_visible(False)
    viz.finish(fig, save_path, filename)
    return fig


def plot_agreement_stability(stability, quantity, title='', sources=None,
                             height=2.0, save_path=None, filename=None):
    """
    Whether the bias between each external source and the reference holds still.

    A source may stand in for the local measurement only if its bias is
    stable rather than merely small: a constant offset can be subtracted and
    the substitution documented, while a bias that drifts with the season
    cannot be, and the two look identical in any statistic pooled over the
    whole record. This figure is where the distinction becomes visible — a
    flat line is a stable bias, and a sloped or wandering one is the finding
    that rules the source out.

    Parameters
    ----------
    stability : pd.DataFrame
        Output of :func:`shmlib.compare.agreement_stability`: one row per
        window per pair, carrying ``window``, ``compared``, ``bias``,
        ``iqr`` and ``n``.
    quantity : str
        Canonical quantity name, fixing the line colour and the axis label.
    title : str, optional
        Axes title. Default ``''``.
    sources : sequence of str or None, optional
        Sources to draw, one line each, told apart by line style. Default
        ``None``, every source present in ``stability['compared']``, in the
        order it first lists them.
    height : float, optional
        Figure height in inches. Default ``2.0``.
    save_path : str or None, optional
        Output directory. Default ``None``, which saves nothing.
    filename : str or None, optional
        Base file name, without extension. Default ``None``, which saves
        nothing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. The shaded band around each line is centred on the
    plotted bias at plus and minus half the reported interquartile range:
    ``stability`` carries only the spread of the difference within each
    window, not its own upper and lower bound, so this is the symmetric
    reading of that spread rather than a reconstruction of the true
    quartiles.
    """
    sources = (list(pd.unique(stability['compared'])) if sources is None
              else sources)
    colour = viz.QUANTITY_COLOUR[quantity]

    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, height))
    for source in sources:
        subset = stability[stability['compared'] == source] \
            .sort_values('window')
        if not len(subset):
            continue
        ax.plot(subset['window'], subset['bias'], color=colour,
               linestyle=viz.SOURCE_STYLE.get(source, '-'), lw=1.3,
               label=proxies.SOURCE_LABEL.get(source, source), zorder=3)
        ax.fill_between(subset['window'],
                        subset['bias'] - subset['iqr'] / 2.0,
                        subset['bias'] + subset['iqr'] / 2.0,
                        color=colour, alpha=0.15, lw=0, zorder=2)

    ax.axhline(0.0, color='#000000', lw=0.6, zorder=1)
    ax.set_ylabel(f'Bias [{proxies.QUANTITY_UNIT[quantity]}]')
    ax.set_title(title)
    ax.legend(fontsize='small', ncol=max(len(sources), 1), loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.format_spines(ax)
    viz.finish(fig, save_path, filename)
    return fig


def plot_lag_curves(curves, stratum, band, drivers=None, title='',
                    height=2.6, width=None, save_path=None, filename=None):
    """
    The correlation of each driver against the response, at every lag scanned.

    The curve says what its maximum cannot. A broad flat curve is a coupling
    whose delay the record does not resolve, however confident the single lag
    reported beside it looks; two comparable peaks a day apart are the aliasing
    a bounded scan exists to avoid; and a curve still climbing at the edge of
    the range is a warning that the range was drawn too narrow. Drawing the
    whole scan puts all three in front of the reader rather than in a footnote.

    Parameters
    ----------
    curves : dict
        The scans :func:`shmlib.coupling.couple` returns, keyed by
        ``(stratum, band, driver)``.
    stratum, band : str
        Which slice of `curves` to draw.
    drivers : sequence of str or None, optional
        Drivers to draw, in order. Default ``None``, every driver present for
        this stratum and band, in the order the scan produced them. A figure
        drawing more than about six lines stops being readable, so a study with
        a wide screen passes the subset worth reading hour by hour and leaves
        the rest to its table.
    title : str, optional
        Figure title.
    height : float, optional
        Figure height in inches. Default 2.6.
    width : float or None, optional
        Figure width in inches. Default ``None``, the manuscript column width.
    save_path, filename : str or None, optional
        Output directory and base name. Both are needed before anything is
        written.

    Returns
    -------
    matplotlib.figure.Figure
    """
    present = [key[2] for key in curves if key[0] == stratum and key[1] == band]
    if drivers is None:
        drivers = list(dict.fromkeys(present))

    fig, ax = plt.subplots(figsize=(width or viz.FIGURE_WIDTH, height))

    for driver in drivers:
        curve = curves.get((stratum, band, driver))
        if curve is None or curve['r'].isna().all():
            continue
        delay_column = 'delay' if 'delay' in curve.columns else 'lag'
        if 'tau' in curve.columns:
            # A two-parameter scan has no single curve, so the one drawn is the
            # row of the grid at this driver's own winning time constant. The
            # legend says which, because the same delay axis at a different
            # time constant is a different curve.
            best_tau = float(curve.loc[curve['r'].abs().idxmax(), 'tau'])
            curve = curve[curve['tau'] == best_tau]
            label = f'{viz.driver_label(driver)}, $\\tau$ {best_tau:g} h'
        else:
            label = viz.driver_label(driver)

        colour = viz.driver_colour(driver)
        ax.plot(curve[delay_column], curve['r'], color=colour, lw=1.2,
                label=label)

        peak = curve.loc[curve['r'].abs().idxmax()]
        ax.plot(peak[delay_column], peak['r'], marker='o', ms=3.5,
                color=viz.MARK_COLOUR, zorder=5, ls='none')

    ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
    ax.axvline(0.0, color='0.6', lw=0.8, zorder=1)
    ax.set_xlabel('Transport delay applied to the driver [h]', fontsize='small')
    ax.set_ylabel('Correlation with the response', fontsize='small')
    ax.legend(fontsize='x-small', ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.set_title(title, fontsize='small', fontweight='bold')
    viz.format_spines(ax)
    viz.finish(fig, save_path, filename)
    return fig


def plot_coupling_scatter(frame, response, driver, lag, tau=0.0, gain=None,
                          title='', height=2.8, width=None, save_path=None,
                          filename=None):
    """
    The response against one driver, at the lag the scan chose.

    Drawn as a hexbin for the same reason the source comparison is: years of
    hourly samples saturate a point cloud, and the density-aware colouring is
    what lets the shape of the relation show through. Where a fitted gain is
    passed it is drawn over the cloud with its confidence interval in the
    annotation, so the number in the table and the cloud it was fitted to are
    read together.

    Parameters
    ----------
    frame : pd.DataFrame
        Response and driver on one grid, in the band being reported.
    response, driver : str
        Columns to plot. The driver is shifted by `lag` before pairing, the
        same convention :func:`shmlib.coupling.lag_scan` uses.
    lag : int or float
        Transport delay in samples of the grid.
    tau : float, optional
        Thermal time constant applied to the driver before the delay, in hours.
        Default 0.0. Where a scan chose one, the cloud has to be drawn against
        the same operator the gain was fitted to, or the fitted line will not
        lie on the points.
    gain : dict or None, optional
        A row of a coupling table, carrying at least ``slope`` and
        ``intercept`` and optionally ``ci_low``, ``ci_high``, ``r`` and ``n``.
        Default ``None``, which draws the cloud alone.
    title : str, optional
        Figure title.
    height : float, optional
        Figure height in inches. Default 2.8.
    width : float or None, optional
        Figure width in inches. Default ``None``, the manuscript column width.
    save_path, filename : str or None, optional
        Output directory and base name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    operated = coupling.thermal_operator(
        pd.to_numeric(frame[driver], errors='coerce'), delay=int(lag), tau=tau)
    paired = pd.concat([operated,
                        pd.to_numeric(frame[response], errors='coerce')],
                       axis=1).dropna()

    fig, ax = plt.subplots(figsize=(width or viz.FIGURE_WIDTH, height))
    if len(paired):
        x = paired.iloc[:, 0].to_numpy(dtype=float)
        y = paired.iloc[:, 1].to_numpy(dtype=float)
        hexbin = ax.hexbin(x, y, gridsize=40, cmap='cividis_r', mincnt=1,
                           bins='log')
        colourbar = fig.colorbar(hexbin, ax=ax, shrink=0.85)
        colourbar.set_label('count (log)', fontsize='x-small')
        colourbar.ax.tick_params(labelsize='x-small')

        if gain is not None and not np.isnan(gain.get('slope', np.nan)):
            span = np.array([x.min(), x.max()])
            ax.plot(span, span * gain['slope'] + gain['intercept'],
                    color=viz.driver_colour(driver), lw=1.4, zorder=4)
            note = f"slope {gain['slope']:.3g}"
            if not np.isnan(gain.get('ci_low', np.nan)):
                note += f"\n95 % CI [{gain['ci_low']:.3g}, {gain['ci_high']:.3g}]"
            if not np.isnan(gain.get('r', np.nan)):
                note += f"\nr {gain['r']:.3f}"
            if gain.get('n'):
                note += f"\nn={int(gain['n']):,}"
            ax.text(0.03, 0.97, note, transform=ax.transAxes,
                    fontsize='x-small', va='top', ha='left')

    operator_note = (f'delayed {int(lag)} h' if not tau
                     else f'$\\tau$ {tau:g} h, delayed {int(lag)} h')
    ax.set_xlabel(f'{viz.driver_label(driver)}, {operator_note}',
                  fontsize='small')
    ax.set_ylabel(viz.CHANNEL_LABEL.get(response, response), fontsize='small')
    ax.set_title(title, fontsize='small', fontweight='bold')
    viz.format_spines(ax)
    viz.finish(fig, save_path, filename)
    return fig


def plot_gain_stability(stability, drivers=None, title='', panel_height=1.5,
                        width=None, save_path=None, filename=None):
    """
    Each driver's gain re-estimated window by window, one panel per driver.

    One panel per driver rather than one axes for all of them, because the
    gains are in different units — millidegrees per degree Celsius beside
    millidegrees per watt per square metre — and an axis that holds both says
    nothing about either. What the reader is asked to judge is whether a
    driver's band stays where it was, not how two drivers compare in size.

    Parameters
    ----------
    stability : pd.DataFrame
        The frame :func:`shmlib.coupling.gain_stability` returns.
    drivers : sequence of str or None, optional
        Drivers to draw, one panel each, in order. Default ``None``, every
        driver in `stability`.
    title : str, optional
        Figure title.
    panel_height : float, optional
        Height of one panel in inches. Default 1.5.
    width : float or None, optional
        Figure width in inches. Default ``None``, the manuscript column width.
    save_path, filename : str or None, optional
        Output directory and base name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if drivers is None:
        drivers = list(dict.fromkeys(stability['driver']))

    fig, axes = plt.subplots(len(drivers), 1, sharex=True, squeeze=False,
                             figsize=(width or viz.FIGURE_WIDTH,
                                      panel_height * max(len(drivers), 1)))
    axes = axes.ravel()

    for ax, driver in zip(axes, drivers):
        block = stability[stability['driver'] == driver].sort_values('window')
        colour = viz.driver_colour(driver)
        ax.fill_between(block['window'], block['ci_low'], block['ci_high'],
                        color=colour, alpha=0.20, lw=0)
        ax.plot(block['window'], block['slope'], color=colour, lw=1.2)
        ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
        ax.set_ylabel(viz.driver_label(driver), fontsize='x-small')
        viz.format_spines(ax)

    axes[-1].set_xlabel('Window', fontsize='small')
    fig.suptitle(title, y=1.02, fontsize='small', fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_band_separation(series, window=24, min_periods=None, span=None,
                         title='', height=3.2, width=None, save_path=None,
                         filename=None):
    """
    One series as its level and as its diurnal band, on a shared time axis.

    Drawn before any number rests on the separation, so that a reader can see
    what the filter did rather than take it on the word of the study. The level
    panel carries a position on the axis that an inclinometer's arbitrary
    anchor sets and that means nothing on its own; the band panel carries the
    daily excursion, which is the part every lag in this project is measured
    on.

    Parameters
    ----------
    series : pd.Series
        Series to separate, on a regular grid.
    window, min_periods : optional
        Passed to :func:`shmlib.coupling.diurnal_band`.
    span : tuple of pd.Timestamp or None, optional
        The window drawn, as ``(start, end)``. Default ``None``, the whole
        series, which at years of hourly data is a solid band; a study drawing
        this figure to show the filter passes a few days.
    title : str, optional
        Figure title.
    height : float, optional
        Figure height in inches. Default 3.2.
    width : float or None, optional
        Figure width in inches. Default ``None``, the manuscript column width.
    save_path, filename : str or None, optional
        Output directory and base name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    band = coupling.diurnal_band(series, window=window, min_periods=min_periods)
    if span is not None:
        series = series.loc[span[0]:span[1]]
        band = band.loc[span[0]:span[1]]

    name = series.name
    style = viz.channel_style(name, 1.2) if name in viz.CHANNEL_COLOUR else {
        'color': viz.INC_COLOUR, 'lw': 1.2}
    unit = (viz.channel_unit(name) if name in viz.CHANNEL_LABEL else '')

    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(width or viz.FIGURE_WIDTH, height))
    axes[0].plot(series.index, series.to_numpy(dtype=float), **style)
    axes[1].plot(band.index, band.to_numpy(dtype=float), **style)
    axes[1].axhline(0.0, color='0.6', lw=0.8, zorder=1)
    axes[0].set_ylabel(f'Level {unit}'.strip(), fontsize='small')
    axes[1].set_ylabel(f'Diurnal band {unit}'.strip(), fontsize='small')
    for ax in axes:
        viz.format_spines(ax)
    fig.suptitle(title, fontsize='small', fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_operator_grid(scans, stratum, band, drivers, title='', ncols=2,
                       vmin=None, vmax=None, panel_height=2.1, width=None,
                       save_path=None, filename=None):
    """
    The whole delay-by-time-constant grid, one panel per driver.

    The figure that a single winning cell cannot replace. What it shows is the
    *shape* of the optimum, and that shape is usually a ridge rather than a
    peak: a transport delay and a thermal time constant substitute for one
    another over a wide range, so a scan can report a confident pair of numbers
    from a region in which many pairs do almost equally well. A reader looking
    at the ridge can see how much of the delay is really resolved, which a
    table of maxima hides.

    Each panel is titled with its own best cell. The colour scale is per-panel
    by default, because the drivers differ by an order of magnitude in how much
    they explain and one scale over all of them renders the weaker ones flat.
    Passing `vmax` overrides that with a scale shared by every panel, which is
    the right choice when the question is how the drivers compare rather than
    what shape each one has: a shade then means the same number in every panel
    and across every figure drawn on the same range. The cost is that a weak
    panel occupies only the bottom of the range and shows less of its own
    structure, which is the comparison being made rather than a defect.
    A saturated panel is marked with its true maximum in the accent colour, and
    its title still reports the unclipped value, so nothing is hidden by the
    choice of range.

    Parameters
    ----------
    scans : dict
        The grids :func:`shmlib.coupling.couple` returns, keyed by
        ``(stratum, band, driver)``.
    stratum, band : str
        Which slice of `scans` to draw.
    drivers : sequence of str
        Drivers to draw, one panel each, in order.
    title : str, optional
        Figure title.
    ncols : int, optional
        Panels across. Default 2.
    vmin, vmax : float or None, optional
        Shared colour range. Default ``None`` for each, which scales every
        panel to its own values and draws a colour bar beside each. Giving
        `vmax` switches to one range and one colour bar for the whole figure;
        `vmin` then defaults to 0, the smallest an ``R^2`` can be.
    panel_height : float, optional
        Height of one panel row in inches. Default 2.1.
    width : float or None, optional
        Figure width in inches. Default ``None``, the manuscript column width.
    save_path, filename : str or None, optional
        Output directory and base name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    drivers = [driver for driver in drivers
               if (stratum, band, driver) in scans]
    nrows = int(np.ceil(len(drivers) / ncols)) if drivers else 1

    shared = vmax is not None
    if shared and vmin is None:
        vmin = 0.0

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(width or viz.FIGURE_WIDTH,
                                      panel_height * nrows))
    flat_axes = axes.ravel()

    mesh = None
    for ax, driver in zip(flat_axes, drivers):
        scan = scans[(stratum, band, driver)]
        # R-squared rather than r, so that the colour is the share of variance
        # explained and the sign of the association does not split the scale.
        grid = scan.pivot(index='tau', columns='delay', values='r2')
        values = grid.to_numpy(dtype=float)
        # Cividis reversed: the meaningful end is the dark one, as the project's
        # graphical rules require of every scalar map.
        mesh = ax.pcolormesh(range(len(grid.columns)), range(len(grid.index)),
                             values, cmap='cividis_r', shading='nearest',
                             vmin=vmin, vmax=vmax)
        panel_max = (np.nanmax(values) if np.isfinite(values).any() else np.nan)
        clipped = bool(shared and panel_max > vmax)

        # A colour bar per panel, even where the range is shared. One bar for
        # the figure would say the same thing in less ink, but a figure-level
        # bar and the tight layout every figure in this project is finished
        # with fight over the same space and the bar lands across the panels.
        # The over-range arrow is drawn only where a panel actually runs past
        # the top, so that an arrow always means something was clipped.
        colourbar = fig.colorbar(mesh, ax=ax, shrink=0.9,
                                 extend='max' if clipped else 'neither')
        colourbar.set_label('$R^2$', fontsize='x-small')
        colourbar.ax.tick_params(labelsize='xx-small')

        if clipped:
            # The note the project's rules require of any clipped scale: this
            # panel runs past the top of the shared range, and the reader is
            # told by how much rather than left to infer it.
            ax.text(0.98, 0.04, f'max {panel_max:.2f}', transform=ax.transAxes,
                    fontsize='xx-small', color=viz.MARK_COLOUR, ha='right',
                    va='bottom')

        scored = scan.dropna(subset=['r2'])
        if len(scored):
            best = scored.loc[scored['r2'].idxmax()]
            ax.plot(list(grid.columns).index(best['delay']),
                    list(grid.index).index(best['tau']), marker='o', ms=4.0,
                    color=viz.MARK_COLOUR, ls='none', zorder=5)
            ax.set_title(f"{viz.driver_label(driver)} — $R^2$ {best['r2']:.3f} "
                         f"at {best['delay']:g} h, $\\tau$ {best['tau']:g} h",
                         fontsize='x-small')

        # A signed scan carries four times the columns of a causal one, and
        # forty-nine labels in a panel this size are a black smear rather than
        # an axis, so the labels are thinned to at most thirteen while every
        # cell keeps its place.
        step = max(1, int(np.ceil(len(grid.columns) / 13)))
        positions = list(range(0, len(grid.columns), step))
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{grid.columns[i]:g}' for i in positions],
                           fontsize='xx-small', rotation=90)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f'{value:g}' for value in grid.index],
                           fontsize='xx-small')
        ax.set_xlabel('Transport delay [h]', fontsize='x-small')
        ax.set_ylabel('Time constant $\\tau$ [h]', fontsize='x-small')

    for ax in flat_axes[len(drivers):]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.01, fontsize='small', fontweight='bold')
    viz.finish(fig, save_path, filename)
    return fig


def plot_gap_anatomy(inventory, classes=None, title='', save_path=None,
                     filename=None):
    """
    How the missing time is shaped: gap count and missing hours, by duration class.

    Two panels answer two different questions about the same table. The left
    counts gaps, which is what governs how badly contiguity is broken; the right
    sums their hours, which is what governs how much record is absent. A record
    can be dominated by one class on the left and another on the right, and the
    difference decides what kind of problem filling it is.

    Parameters
    ----------
    inventory : pd.DataFrame
        Output of ``prediction.gap_inventory``.
    classes : sequence of str or None, optional
        Class order along the category axis. Default: the order in which the
        classes appear in ``prediction.DEFAULT_GAP_CLASSES``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from shmlib import prediction as _prediction

    if classes is None:
        classes = [label for _, _, label in _prediction.DEFAULT_GAP_CLASSES]

    counts = (inventory.groupby('gap_class')['n_slots'].size()
              .reindex(classes).fillna(0.0))
    hours = (inventory.groupby('gap_class')['duration_h'].sum()
             .reindex(classes).fillna(0.0))

    fig, axes = plt.subplots(1, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))
    for ax, values, label in ((axes[0], counts, 'Number of gaps'),
                              (axes[1], hours, 'Missing time [h]')):
        ax.bar(range(len(classes)), values.to_numpy(), color=viz.INC_COLOUR,
               width=0.72)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_ylabel(label)
        ax.set_xlabel('Gap duration')
        viz.format_spines(ax)

    total = hours.sum()
    if total > 0:
        share = hours / total
        for position, value in enumerate(share.to_numpy()):
            axes[1].annotate(f'{value:.0%}',
                             (position, hours.to_numpy()[position]),
                             ha='center', va='bottom', fontsize='small',
                             color=viz.MARK_COLOUR)

    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_segment_survival(survival, title='', save_path=None, filename=None):
    """
    Training windows surviving segmentation, against the length of window asked for.

    Parameters
    ----------
    survival : pd.DataFrame
        Output of ``prediction.segment_survival``, one row per configuration.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))
    handles = []
    styles = ('-', '--', ':', '-.')
    for position, (horizon, group) in enumerate(
            survival.groupby('forecast_hours')):
        style = styles[position % len(styles)]
        line, = ax.plot(group['lag_hours'], group['n_windows'],
                        color=viz.INC_COLOUR, linestyle=style, marker='o',
                        linewidth=1.6, label=f'{int(horizon)} h horizon')
        handles.append(line)
    ax.set_xlabel('Autoregressive window [h]')
    ax.set_ylabel('Training windows')
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    ax.legend(fontsize='small', ncol=len(handles), loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_cadence_evidence(evidence, title='', save_path=None, filename=None):
    """
    The three measurements that fix the cadence and the target, side by side.

    Parameters
    ----------
    evidence : pd.DataFrame
        Output of ``prediction.cadence_evidence``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = (('level_autocorr1', 'Level lag-1\nautocorrelation'),
              ('change_autocorr1', 'Change lag-1\nautocorrelation'),
              ('corr_change', 'corr(change, driver change)'))
    fig, axes = plt.subplots(1, len(panels),
                             figsize=viz.figsize(viz.FIGURE_WIDTH, 2.2))
    positions = range(len(evidence))
    for ax, (column, label) in zip(axes, panels):
        ax.bar(positions, evidence[column].to_numpy(), color=viz.INC_COLOUR,
               width=0.6)
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(evidence['cadence'])
        ax.set_ylabel(label)
        viz.format_spines(ax)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_decomposition_stack(components, columns=None, freq=None, title='',
                             save_path=None, filename=None):
    """
    One panel per additive component, on a shared clock.

    The panels are stacked rather than overlaid because the components differ in
    scale by orders of magnitude: a trend of a few millidegrees a year and a
    daily cycle of tens of millidegrees cannot share an axis without one of them
    becoming a flat line.

    Parameters
    ----------
    components : pd.DataFrame
        Output of ``prediction.decompose_components``.
    columns : sequence of str or None, optional
        Components to draw, in panel order. ``None`` draws every component
        column, ending with the residual. Either way, a family aggregate
        (e.g. ``future_regressors_additive``) is dropped whenever one of its
        constituent columns (e.g. ``future_regressor_tair``) is also present,
        via ``prediction.decomposition_columns``, so the panel drawn here can
        never disagree with what ``prediction.component_variance_shares``
        counts. Default ``None``.
    freq : str or None, optional
        Grid spacing ``components`` is reindexed onto before plotting, so
        that a timestamp segmentation dropped outright — as opposed to
        carrying as ``NaN`` — reappears as a gap in the line rather than a
        straight interpolation across it. ``None`` plots the data as given.
        Default ``None``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from shmlib import prediction as _prediction

    components = _reindex_regular(components, freq)
    columns = _prediction.decomposition_columns(components, columns)

    fig, axes = plt.subplots(
        len(columns), 1, sharex=True,
        figsize=viz.figsize(viz.FIGURE_WIDTH, 1.15 * len(columns)))
    axes = np.atleast_1d(axes)

    for ax, column in zip(axes, columns):
        ax.plot(components.index, components[column], color=viz.INC_COLOUR,
                linewidth=1.0)
        ax.set_ylabel(column.replace('future_regressor_', '')
                      .replace('lagged_regressor_', '')
                      .replace('_', ' '))
        viz.format_spines(ax)
    axes[-1].set_xlabel('')

    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_prediction_band(observed, expected, lower, upper, freq=None,
                         title='', highlight=None, save_path=None,
                         filename=None):
    """
    Observed against expected, with the prediction interval drawn behind them.

    Parameters
    ----------
    observed, expected : pd.Series
        Measured and predicted values, on a shared index.
    lower, upper : pd.Series
        Interval bounds, same index.
    freq : str or None, optional
        Grid spacing each of ``observed``, ``expected``, ``lower`` and
        ``upper`` is reindexed onto before plotting, so that a timestamp
        segmentation dropped outright — as opposed to carrying as ``NaN`` —
        reappears as a gap in the line rather than a straight interpolation
        across it. ``None`` plots the data as given. Default ``None``.
    title : str, optional
        Figure title. Default ``''``.
    highlight : sequence of (start, end) or None, optional
        Intervals to shade, drawn in the project's span style. Default ``None``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    observed = _reindex_regular(observed, freq)
    expected = _reindex_regular(expected, freq)
    lower = _reindex_regular(lower, freq)
    upper = _reindex_regular(upper, freq)

    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))

    ax.fill_between(observed.index, lower, upper, color=viz.INC_COLOUR,
                    alpha=0.18, linewidth=0.0, label='90 % interval')
    ax.plot(observed.index, observed, color=viz.INC_COLOUR, linewidth=1.0,
            label='observed')
    ax.plot(expected.index, expected, color=viz.INC_COLOUR, linewidth=1.0,
            linestyle='--', label='expected')

    for span in (highlight or ()):
        ax.axvspan(span[0], span[1], **viz.SPAN_STYLE)

    ax.set_ylabel('Inclination [mdeg]')
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    ax.legend(fontsize='small', ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_control_chart(chart, statistic='ewma', episodes=None, freq=None,
                       title='', save_path=None, filename=None):
    """
    A control statistic against its limits, with alarming episodes shaded.

    Parameters
    ----------
    chart : pd.DataFrame
        Output of ``monitoring.ewma_chart`` or ``monitoring.cusum_chart``.
    statistic : str, optional
        Column to draw. Default ``'ewma'``; pass ``'cusum_high'`` for a CUSUM
        chart.
    episodes : pd.DataFrame or None, optional
        Output of ``monitoring.alarm_episodes``, shaded behind the statistic.
        Default ``None``.
    freq : str or None, optional
        Grid spacing ``chart`` is reindexed onto before plotting, so that a
        timestamp segmentation dropped outright — as opposed to carrying as
        ``NaN`` — reappears as a gap in the line rather than a straight
        interpolation across it. ``None`` plots the data as given. Default
        ``None``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    chart = _reindex_regular(chart, freq)

    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))

    for _, episode in (episodes if episodes is not None
                       else pd.DataFrame()).iterrows():
        ax.axvspan(episode['start'], episode['end'], **viz.SPAN_STYLE)

    ax.plot(chart.index, chart[statistic], color=viz.INC_COLOUR,
            linewidth=1.0, label=statistic.replace('_', ' '))
    limit_labelled = False
    for limit in ('ucl', 'lcl', 'limit'):
        if limit in chart.columns:
            # First-one-wins: whichever limit column is drawn first gets the
            # single 'limit' legend entry, so a chart carrying only 'lcl'
            # still gets a legend entry instead of silently losing it to a
            # name-specific special case.
            ax.plot(chart.index, chart[limit], color=viz.MARK_COLOUR,
                    linewidth=0.9, linestyle='--',
                    label=None if limit_labelled else 'limit')
            limit_labelled = True

    ax.set_ylabel('Standardised residual')
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize='small', ncol=len(labels),
              loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_metric_vs_horizon(metrics, metric='mae', by='model', title='',
                           save_path=None, filename=None):
    """
    One curve per model, showing how a metric degrades with forecast horizon.

    Parameters
    ----------
    metrics : pd.DataFrame
        Long table carrying ``horizon_h``, the grouping column and the metric.
    metric : str, optional
        Column to draw. Default ``'mae'``.
    by : str, optional
        Grouping column, one curve per level. Default ``'model'``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))

    styles = ['-', '--', '-.', ':']
    for position, (name, group) in enumerate(metrics.groupby(by, sort=False)):
        ordered = group.sort_values('horizon_h')
        ax.plot(ordered['horizon_h'], ordered[metric], color=viz.INC_COLOUR,
                linestyle=styles[position % len(styles)], marker='o',
                markersize=3, linewidth=1.4, label=str(name))

    ax.set_xlabel('Forecast horizon [h]')
    ax.set_ylabel(metric.upper() if len(metric) <= 4 else metric)
    viz.format_spines(ax)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize='small', ncol=min(len(labels), 4),
              loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=False)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_detectability(curve, title='', save_path=None, filename=None):
    """
    The smallest departure the charts find, against how long it persists.

    Two panels: whether each injected departure was detected at all, as a
    Cividis field over magnitude and duration, and how late the detection came.

    Parameters
    ----------
    curve : pd.DataFrame
        Output of ``monitoring.detectability_curve``.
    title : str, optional
        Figure title. Default ``''``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``; nothing is written when either is ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    detected = curve.pivot(index='magnitude', columns='duration_h',
                           values='detected').astype(float)
    delay = curve.pivot(index='magnitude', columns='duration_h',
                        values='delay_h')

    fig, axes = plt.subplots(1, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))
    for ax, frame, label, cmap in ((axes[0], detected, 'Detected', 'cividis_r'),
                                   (axes[1], delay, 'Detection delay [h]',
                                    'cividis')):
        mesh = ax.pcolormesh(frame.columns.to_numpy(),
                             frame.index.to_numpy(),
                             frame.to_numpy(), cmap=cmap, shading='nearest')
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xlabel('Duration [h]')
        ax.set_ylabel('Magnitude [mdeg]')
        viz.format_spines(ax)

    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_harmonic_diagnostics(scans, dailies, fits, surface, title='',
                              save_path=None, filename=None):
    """
    The harmonic diagnostic of spec D6, on one page.

    Four panels: the periods the spectral scan certifies for each series;
    the daily amplitude by day of year with its annual fit; the daily phase
    likewise; and the singular-value shares of the daily-by-annual surface.

    Parameters
    ----------
    scans, dailies : dict of pd.DataFrame
        Keyed by series name; outputs of ``prediction.period_scan`` and
        ``monitoring.daily_harmonic``.
    fits : dict of dict
        ``fits[name]['amplitude']`` and ``fits[name]['phase']`` from
        ``coupling.annual_modulation``.
    surface : dict
        Output of ``coupling.cycle_surface_rank``.
    title : str, optional
    save_path, filename : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 5.2))
    styles = {'target': '-', 'residual': '--'}
    colour = viz.INC_COLOUR
    year = pd.date_range('2001-01-01', periods=365, freq='D')

    ax = axes[0, 0]
    for name, scan in scans.items():
        ax.stem(scan['period_days'], scan['power'], linefmt=colour,
                markerfmt=' ', basefmt=' ', label=name)
    ax.set_xscale('log')
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('Normalised power')
    for period in (0.5, 1.0, 182.6, 365.25):
        ax.axvline(period, color=viz.MARK_COLOUR, linewidth=0.6, alpha=0.6)

    for ax, key, label in ((axes[0, 1], 'amplitude', 'Daily amplitude [mdeg]'),
                           (axes[1, 0], 'phase_h', 'Hour of daily maximum')):
        for name, daily in dailies.items():
            doy = daily.index.dayofyear
            ax.scatter(doy, daily[key], s=3, color=colour, alpha=0.15)
            fit = fits[name]['amplitude' if key == 'amplitude' else 'phase']
            curve = coupling.evaluate_modulation(fit, year)
            ax.plot(year.dayofyear, curve, color=colour, linewidth=1.6,
                    linestyle=styles.get(name, '-'), label=name)
        ax.set_xlabel('Day of year')
        ax.set_ylabel(label)

    ax = axes[1, 1]
    table = surface['table'].head(6)
    ax.bar(table['component'], table['variance_share'], color=colour, width=0.7)
    ax.set_xlabel('Component')
    ax.set_ylabel('Share of variance')

    for ax in axes.ravel():
        viz.format_spines(ax)
    # Per the project's graphical guidelines a multi-panel figure's legend
    # belongs to the figure rather than to one panel, so it is built here
    # from one neutral handle per series in `dailies` rather than taken from
    # a single axes.
    handles = [Line2D([], [], color='#000000', linestyle=styles.get(name, '-'),
                      label=name)
               for name in dailies]
    fig.legend(handles=handles, labels=list(dailies), loc='upper center',
              bbox_to_anchor=(0.5, -0.01), ncol=len(dailies), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_regressor_sets(frame, target, sets, target_channel='inc_comp', title='',
                        tick_years=1, save_path=None, filename=None):
    """
    The target and every regressor set's own record, one panel each.

    Overlaying the three regressor sets in one panel per role — the
    figure's first form — draws three lines of the same identity colour on
    top of each other and is unreadable: a reader cannot tell the sets
    apart, since colour here already carries the role and cannot also carry
    the set. This layout instead gives every distinct record its own panel:
    the target first, then one panel per distinct record found across
    ``sets``, in role order ``'tair'``, ``'rh'``, ``'sr'`` and then set
    order as ``sets`` is given. Two sets naming the same role are merged
    into one panel when their columns hold identical values, checked with
    ``frame[a].equals(frame[b])`` rather than by column name alone: the
    on-structure set stores the ground station's borrowed radiation under
    its own suffix, so its ``'sr'`` column and the ground station's own
    differ in name but not in content, and the two must still collapse to
    one panel rather than drawing the same record twice under two names.
    Two columns sharing a literal name are the same check, since a column
    trivially equals itself. A merged panel's title names every set that
    reads it. Every panel of one role shares its vertical
    scale, fitted to the finite range of every column that role draws
    (shared even where the columns differ), so that an amplitude difference
    between two sets is a difference in the drawn shape rather than an
    artefact of two independently chosen axes.

    Parameters
    ----------
    frame : pd.DataFrame
        Datetime-indexed frame holding ``target`` and every set's columns.
    target : str
        Column of the response in ``frame``.
    sets : dict
        ``{set_name: {role: column}}``; roles are ``'tair'``, ``'rh'``, ``'sr'``.
    target_channel : str, optional
        Channel identity used for the target's colour and unit. Default
        ``'inc_comp'``.
    title : str, optional
        Figure title. Default ``''``, which draws no title.
    tick_years : int, optional
        Years between x ticks. Default ``1``.
    save_path, filename : str or None, optional
        Passed to ``viz.finish``.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Writes two image files (PNG and SVG) when ``save_path`` and ``filename``
    are both given. Every panel is titled with its role and the set names
    that share it, so no legend is drawn.
    """
    roles = ['tair', 'rh', 'sr']

    # The distinct records, in role order then set order. Two sets of the
    # same role merge into one panel when their columns hold identical
    # values (`frame[a].equals(frame[b])`), not only when they share a
    # column name — the on-structure set's borrowed radiation is the case
    # this exists for: same values, different suffix.
    panels = []
    for role in roles:
        for name, mapping in sets.items():
            if role not in mapping:
                continue
            column = mapping[role]
            series = frame[column]
            matched = next(
                (panel for panel in panels if panel[0] == role
                 and (column == panel[1] or series.equals(frame[panel[1]]))),
                None)
            if matched is not None:
                matched[2].append(name)
            else:
                panels.append([role, column, [name]])

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1 + n_panels, 1, sharex=True,
        figsize=viz.figsize(viz.FIGURE_WIDTH, 0.85 * n_panels + 0.4))
    axes = np.atleast_1d(axes)

    axes[0].plot(frame.index, frame[target],
                 **viz.channel_style(target_channel, 0.6))
    axes[0].set_ylabel(viz.channel_unit(target_channel))
    axes[0].set_title(target, loc='left', fontsize='small')

    for ax, (role, column, set_names) in zip(axes[1:], panels):
        ax.plot(frame.index, frame[column], **viz.channel_style(role, 0.6))
        ax.set_ylabel(viz.channel_unit(role))
        ax.set_title(f"{role} · {' and '.join(set_names)}",
                    loc='left', fontsize='small')

    # One vertical scale per role, shared by every panel of that role.
    role_panel_indices = {}
    for index, (role, column, set_names) in enumerate(panels):
        role_panel_indices.setdefault(role, []).append(index)

    for role, indices in role_panel_indices.items():
        columns = [panels[index][1] for index in indices]
        values = frame[columns].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        low, high = float(finite.min()), float(finite.max())
        pad = 0.05 * (high - low)
        for index in indices:
            axes[1 + index].set_ylim(low - pad, high + pad)

    for ax in axes:
        viz.format_spines(ax)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(tick_years))

    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_fit_metrics(metrics, title='', save_path=None, filename=None):
    """
    Training and validation MAE by epoch, the tutorial's own curve, redrawn.

    Parameters
    ----------
    metrics : pd.DataFrame
        NeuralProphet's own fit-metrics frame (``model.fit_metrics_``),
        indexed by epoch, with a ``MAE`` column and, when a validation frame
        was fitted with, ``MAE_val``.
    title : str, optional
        Figure title. Default ``''``, which draws none.
    save_path, filename : optional
        Passed to :func:`shmlib.viz.finish`. Default ``None``, no save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.4))
    ax.plot(metrics.index, metrics['MAE'], color=viz.INC_COLOUR, label='training')
    if 'MAE_val' in metrics.columns:
        ax.plot(metrics.index, metrics['MAE_val'], color=viz.INC_COLOUR,
                linestyle='--', label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE [mdeg]')
    viz.format_spines(ax)
    ax.legend(fontsize='small', ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_trend_parameters(trend, rates, changepoints, freq=None, title='',
                          save_path=None, filename=None):
    """
    The trend on covered time above, the rate of each segment below;
    changepoints in the accent colour.

    ``trend`` is indexed on covered time only — a segmented fit skips a
    missing timestamp outright rather than carrying it as a ``NaN`` row — so
    drawing it straight joins the line across any gap between segments as if
    the trend were known there, which it is not. Passing ``freq`` reindexes
    the trend onto the regular grid first, through
    :func:`_reindex_regular`, the same mechanism :func:`plot_decomposition_stack`
    uses for its own panels, so a gap longer than one step becomes a break in
    the drawn line rather than a silent interpolation.

    Parameters
    ----------
    trend : pd.Series
        Datetime-indexed fitted trend, as returned by
        :func:`shmlib.prediction.trend_parameters`.
    rates : pd.DataFrame
        ``start``, ``end``, ``rate_mdeg_per_year`` per segment, as returned
        by the same function.
    changepoints : pd.DatetimeIndex
        Changepoint locations, marked with a vertical line on both panels.
    freq : str or None, optional
        Grid spacing ``trend`` is reindexed onto before drawing, so a gap
        longer than one step breaks the line instead of being bridged.
        Default ``None``, which keeps today's behaviour of drawing ``trend``
        exactly as given.
    title : str, optional
        Figure title. Default ``''``, which draws none.
    save_path, filename : optional
        Passed to :func:`shmlib.viz.finish`. Default ``None``, no save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    trend = _reindex_regular(trend, freq)
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=viz.figsize(viz.FIGURE_WIDTH, 3.6))
    axes[0].plot(trend.index, trend, color=viz.INC_COLOUR, linewidth=1.0)
    axes[0].set_ylabel('Trend [mdeg]')
    for stamp in pd.DatetimeIndex(changepoints):
        for ax in axes:
            ax.axvline(stamp, color=viz.MARK_COLOUR, linewidth=0.6)
    for _, row in rates.iterrows():
        axes[1].hlines(row['rate_mdeg_per_year'], row['start'], row['end'],
                       color=viz.INC_COLOUR, linewidth=2.0)
    axes[1].axhline(0.0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Rate [mdeg/yr]')
    for ax in axes:
        viz.format_spines(ax)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_seasonal_parameters(curves, title='', save_path=None, filename=None):
    """
    The yearly curve on the left; the daily curve per evaluated date on the
    right.

    A two-panel figure, so per the project's graphical guidelines its legend
    belongs to the figure rather than to the right-hand axes: one
    :class:`matplotlib.lines.Line2D` handle is built per plotted date, in the
    accent-free identity colour every panel already shares, and the figure's
    own legend is drawn centred below both panels. The yearly block is
    sorted by ``date`` before it is drawn, because
    :func:`shmlib.prediction.seasonal_parameters` builds it from a
    synthetic-year sweep whose row order is not guaranteed to already be
    calendar order; drawing it unsorted would connect points out of
    sequence and draw a curve with spurious jumps.

    Parameters
    ----------
    curves : pd.DataFrame
        Long ``date``, ``hour``, ``component``, ``value``, as returned by
        :func:`shmlib.prediction.seasonal_parameters`.
    title : str, optional
        Figure title. Default ``''``, which draws none.
    save_path, filename : optional
        Passed to :func:`shmlib.viz.finish`. Default ``None``, no save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))
    yearly = curves[curves['component'] == 'yearly'].sort_values('date')
    axes[0].plot(pd.DatetimeIndex(yearly['date']).dayofyear, yearly['value'],
                 color=viz.INC_COLOUR)
    axes[0].set_xlabel('Day of year')
    axes[0].set_ylabel('Yearly term [mdeg]')
    daily = curves[curves['component'] != 'yearly']
    styles = ['-', '--', ':', '-.']
    handles = []
    for style, (date, group) in zip(styles * 4, daily.groupby('date')):
        total = group.groupby('hour')['value'].sum()
        axes[1].plot(total.index, total, color=viz.INC_COLOUR, linestyle=style)
        handles.append(Line2D([], [], color=viz.INC_COLOUR, linestyle=style,
                              label=pd.Timestamp(date).strftime('%d %b')))
    axes[1].set_xlabel('Hour of day (UTC)')
    axes[1].set_ylabel('Daily term [mdeg]')
    for ax in axes:
        viz.format_spines(ax)
    fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
              bbox_to_anchor=(0.5, -0.01), ncol=len(handles), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_regressor_gains(gains, title='', save_path=None, filename=None):
    """
    Learned gain per driver and set, with Study 03's measurement beside
    each.

    Parameters
    ----------
    gains : pd.DataFrame
        ``set``, ``regressor``, ``gain``, ``study03_gain``, one row per
        driver evaluated in one regressor set.
    title : str, optional
        Figure title. Default ``''``, which draws none.
    save_path, filename : optional
        Passed to :func:`shmlib.viz.finish`. Default ``None``, no save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=viz.figsize(viz.FIGURE_WIDTH, 2.6))
    labels = [f"{r['set']} · {r['regressor']}" for _, r in gains.iterrows()]
    positions = np.arange(len(gains))
    ax.bar(positions - 0.2, gains['gain'], width=0.4, color=viz.INC_COLOUR,
           label='learned')
    ax.bar(positions + 0.2, gains['study03_gain'], width=0.4,
           color=viz.INC_COLOUR, alpha=0.4, label='Study 03')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize='small')
    ax.axhline(0.0, color='black', linewidth=0.5)
    ax.set_ylabel('Gain [mdeg per unit]')
    viz.format_spines(ax)
    ax.legend(fontsize='small', ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.45), frameon=False)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig


def plot_ladder(ladder, title='', save_path=None, filename=None):
    """
    Held-out MAE per rung above, its paired bootstrap skill below.

    The first rung has no rung below it to be scored against, so its
    ``skill`` column and interval are ``NaN`` there; the lower panel's
    error bar is simply undrawn at that position rather than shown at zero,
    since zero would misreport "no improvement" where the true statement is
    "not applicable".

    Parameters
    ----------
    ladder : pd.DataFrame
        ``rung``, ``mae_val``, ``skill``, ``skill_q05``, ``skill_q95``, one
        row per rung in fitting order, as returned by
        :func:`shmlib.prediction.channel_ladder`.
    title : str, optional
        Figure title. Default ``''``, which draws none.
    save_path, filename : optional
        Passed to :func:`shmlib.viz.finish`. Default ``None``, no save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=viz.figsize(viz.FIGURE_WIDTH, 3.4))
    positions = np.arange(len(ladder))
    axes[0].bar(positions, ladder['mae_val'], color=viz.INC_COLOUR, width=0.6)
    axes[0].set_ylabel('Held-out MAE [mdeg]')
    axes[1].errorbar(
        positions, ladder['skill'],
        yerr=[ladder['skill'] - ladder['skill_q05'],
              ladder['skill_q95'] - ladder['skill']],
        fmt='o', color=viz.INC_COLOUR, capsize=3)
    axes[1].axhline(0.0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Skill over rung below')
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(ladder['rung'], rotation=20, ha='right',
                            fontsize='small')
    for ax in axes:
        viz.format_spines(ax)
    if title:
        fig.suptitle(title)
    viz.finish(fig, save_path=save_path, filename=filename)
    return fig
