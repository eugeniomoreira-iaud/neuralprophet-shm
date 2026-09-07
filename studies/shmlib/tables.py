"""
Module: shmlib.tables

Writing the LaTeX table bodies that the study reports ``\\input``.

A report holds the header, the column specification and the rules of each table;
the study writes only the rows, into ``outputs/{PREFIX}_T##_{name}.tex``. Nothing
in a report is then transcribed by hand, and a rebuilt archive changes every
number in the document by changing the files it reads.

**A LaTeX row is separated from the next by ``\\\\``, not terminated by it.** A
trailing separator after the last row opens an empty row, which the following
``\\bottomrule`` — a ``\\noalign`` — then lands inside; that either breaks the
build or, papered over with a bare ``\\cr``, typesets as a visible blank row. So
every row is accumulated first and the rows are joined, and the last one never
carries a separator. That single rule is the reason this module exists rather
than a line of formatting in each notebook cell.

The formatting of a given table stays with the study that publishes it: which
columns appear, in what order and to how many decimals is a presentation choice
about that table, declared in the notebook cell that writes it. What is shared
is the row convention, the escaping, and the treatment of a missing value.
"""

import os

import numpy as np
import pandas as pd


#: What a missing value prints as. An em-dash triple is the convention of every
#: table in this project, and it is deliberately not an empty cell: a blank
#: reads as an oversight, a dash as a measurement that does not exist.
MISSING = '---'


def latex_escape(text):
    """
    Escape the characters LaTeX would otherwise interpret.

    Channel and column names carry underscores throughout this project, and an
    unescaped underscore in text mode is a compile error rather than a character.

    Parameters
    ----------
    text : str
        Text to escape.

    Returns
    -------
    str
        The same text, safe to typeset in LaTeX text mode.
    """
    out = str(text)
    for char in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        out = out.replace(char, '\\' + char) if char != '\\' else out
    return out


def texttt(value):
    """
    Typeset a value as monospaced text, escaped.

    Used for anything that names a thing in the data rather than describing it:
    a column, a channel, a flag code, a station.

    Parameters
    ----------
    value : object
        Value to typeset.

    Returns
    -------
    str
        A ``\\texttt{...}`` group.
    """
    return r'\texttt{' + latex_escape(value) + '}'


def _cell(value, fmt, missing=MISSING):
    """
    Format one cell, or the missing marker where there is no value.

    Parameters
    ----------
    value : object
        The value to format.
    fmt : str or callable or None
        A format specification such as ``'.2f'`` or ``',.0f'``, a callable
        taking the value and returning a string, or ``None`` for ``str``.
    missing : str, optional
        What to print where the value is missing. Default :data:`MISSING`.

    Returns
    -------
    str
        The formatted cell.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return missing
    if callable(fmt):
        return fmt(value)
    if fmt in (None, ''):
        return str(value)
    return format(value, fmt)


def to_rows(frame, columns, missing=MISSING):
    """
    Format a table into LaTeX rows, one string per row of ``frame``.

    Parameters
    ----------
    frame : pd.DataFrame
        Source table. Iterated in order; the row's index value is available to a
        callable source as ``row.name``.
    columns : sequence of tuple
        One ``(source, fmt)`` pair per column of the output table, in the order
        the report's column specification expects them.

        ``source`` is either a column name of ``frame``, or a callable taking
        the row and returning the value — which is how a column that combines
        two fields, or the index itself, is written.

        ``fmt`` is a format specification such as ``'.2f'`` or ``',.0f'``, a
        callable such as :func:`texttt`, or ``None`` for plain ``str``.
    missing : str, optional
        What to print where a value is missing. Default :data:`MISSING`.

    Returns
    -------
    list of str
        The rows, each a string of cells joined by ``' & '`` and carrying no
        separator of its own.

    Examples
    --------
    >>> rows = to_rows(coverage, [
    ...     (lambda r: r.name, texttt),       # the index
    ...     ('eras', None),
    ...     ('span_days', '.1f'),
    ... ])
    """
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for source, fmt in columns:
            value = source(row) if callable(source) else row[source]
            cells.append(_cell(value, fmt, missing=missing))
        rows.append(' & '.join(cells))
    return rows


def write_table_body(rows, path):
    """
    Write LaTeX rows to a table body file.

    The rows are joined by the row separator and the file ends with a newline,
    so the last row carries no separator — see the module docstring for why that
    matters.

    Parameters
    ----------
    rows : sequence of str
        Formatted rows, without separators. Typically from :func:`to_rows`.
    path : str
        Destination file. Its directory must exist.

    Returns
    -------
    str
        The path written, so a notebook cell can report it.

    Notes
    -----
    Writes a file. Overwrites any existing body at ``path``.
    """
    body = ' \\\\\n'.join(rows) + '\n'
    with open(path, 'w') as fh:
        fh.write(body)
    return path


def write_table(frame, path, columns, missing=MISSING):
    """
    Format a table and write it in one call.

    Convenience wrapper over :func:`to_rows` and :func:`write_table_body`, which
    is how a notebook cell writes a table body in a single statement.

    Parameters
    ----------
    frame : pd.DataFrame
        Source table.
    path : str
        Destination file.
    columns : sequence of tuple
        Column specification, as in :func:`to_rows`.
    missing : str, optional
        What to print where a value is missing. Default :data:`MISSING`.

    Returns
    -------
    str
        The path written.

    Notes
    -----
    Writes a file.
    """
    return write_table_body(to_rows(frame, columns, missing=missing), path)


def date_cell(field, fmt='%Y-%m-%d', missing=None):
    """
    A table formatter rendering one date field of a row.

    Timestamps reach a table as ``2018-01-01 00:00:00``, which spends eleven
    characters saying midnight in a column whose rows are all midnight. This
    renders the date alone, and renders a missing date as the table's own
    missing marker rather than as the string ``NaT``.

    Parameters
    ----------
    field : str
        Column of the row to render.
    fmt : str, optional
        ``strftime`` pattern. Default ``'%Y-%m-%d'``.
    missing : str or None, optional
        What to print where the date is missing. Default ``None``, which uses
        :data:`MISSING`.

    Returns
    -------
    callable
        A function of one row, suitable as the ``source`` of a column
        specification passed to :func:`write_table`.
    """
    marker = MISSING if missing is None else missing

    def render(row):
        value = row.get(field)
        if value is None or pd.isna(value):
            return marker
        return pd.Timestamp(value).strftime(fmt)

    return render


def yes_no(value, missing='---'):
    """
    Render a boolean as a word.

    A table column reading ``True`` and ``False`` is a column of Python
    repr, and the reports these tables feed are read by people who do not
    write Python.

    Parameters
    ----------
    value : bool or float
        The value to render. A missing value becomes ``missing``.
    missing : str, optional
        What to print where the value is missing. Default ``'---'``.

    Returns
    -------
    str
        ``'yes'``, ``'no'``, or the missing marker.
    """
    if value is None or (not isinstance(value, (bool, np.bool_))
                         and pd.isna(value)):
        return missing
    return 'yes' if bool(value) else 'no'


def percent(value, decimals=1, missing=MISSING):
    """
    Render a fraction as a percentage with the sign LaTeX needs.

    Python's ``'.1%'`` format produces a bare ``%``, which in LaTeX opens a
    comment and swallows the rest of the line -- including the ``\\\\`` that
    ends the table row, so the row after it merges into this one and the table
    fails to compile with an alignment error that names the wrong line. This
    escapes the sign, which is the only difference from the plain format.

    Parameters
    ----------
    value : float
        A fraction, where ``0.755`` renders as ``75.5\\%``.
    decimals : int, optional
        Digits after the decimal point. Default ``1``.
    missing : str, optional
        What to print where the value is missing. Default :data:`MISSING`.

    Returns
    -------
    str
        The formatted percentage, or the missing marker.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return missing
    return format(float(value), f'.{int(decimals)}%').replace('%', r'\%')


def basename(output_dir, artefact):
    """
    Path of an artefact inside a study's output directory.

    Keeps the notebook free of string concatenation in the cells that save, and
    keeps every artefact name in one recognisable shape.

    Parameters
    ----------
    output_dir : str
        The study's output directory, from its parameter cell.
    artefact : str
        File name, including extension, e.g. ``'DE_T01_raw_census.tex'``.

    Returns
    -------
    str
        The joined path.
    """
    return os.path.join(output_dir, artefact)


def run_metadata(parameters, versions=None):
    """
    Summarise a run's parameters and library versions in one two-column table.

    A study's closing movement records what it was run on: every parameter the
    notebook's own parameter cells declared, in the order the notebook states
    them, followed by the versions of the four libraries whose behaviour the
    result depends on. Keeping this as one function rather than a table
    literal typed into the notebook means the row order and the version
    lookup are tested once here rather than re-typed, and possibly
    mistyped, in every study that closes this way.

    Parameters
    ----------
    parameters : mapping
        Parameter name to value, in the order the output table should list
        them. Typically one dict literal built in the notebook's closing
        cell, naming its own parameter-cell variables.
    versions : mapping, optional
        Version-row name to value, e.g. ``{'neuralprophet_version': '0.8.0',
        ...}``. Default ``None``, which reads the installed versions of
        ``neuralprophet``, ``pandas``, ``numpy`` and ``scipy`` inside this
        function and names the four rows ``neuralprophet_version``,
        ``pandas_version``, ``numpy_version`` and ``scipy_version``, in that
        order. A mapping given instead is used as is, unchanged.

    Returns
    -------
    pd.DataFrame
        Columns ``parameter`` and ``value``: one row per entry of
        ``parameters``, in its order, followed by one row per entry of
        ``versions``. Every value is rendered with ``str``, so a value that is
        itself a dict or a tuple — a set of regressor sets, a bundle of tuned
        control limits — survives :func:`write_table` as readable text rather
        than raising on an unformattable type.
    """
    if versions is None:
        import neuralprophet
        import scipy
        versions = {
            'neuralprophet_version': neuralprophet.__version__,
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'scipy_version': scipy.__version__,
        }
    rows = list(parameters.items()) + list(versions.items())
    return pd.DataFrame(
        [(name, str(value)) for name, value in rows], columns=['parameter', 'value'])
