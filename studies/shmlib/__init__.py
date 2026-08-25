"""
Package: shmlib

The library shared by the studies under ``studies/``, and the only one they
have.

What belongs here is everything: the raw ``.adc`` file format and the
instrument that writes it, the deployment at Gubbio, the solar geometry of the
site, the external forcing sources and their canonical channel vocabulary, the
verdicts a study passes on a record and the windows they leave, how two series
compare, the figure conventions every report follows — and equally a function
that encodes a decision, such as which day to condemn or which lag to accept.
There is no second, per-study library tier and no judgement to make about
which library a function is destined for.

Where a decision lives, then, is in the notebook that makes it, as the
arguments it passes. A function that encodes a choice takes that choice as a
documented argument; the study states its value in its parameter cell. That is
what keeps a decision arguable — visible on the page a reader is already
reading, rather than in a private module one directory away. It is also what
keeps the studies independent of each other: before this package existed,
study 2 duplicated its own copy of facts study 1 already stated, and before its
per-study library tier was dissolved into this one, a decision buried in a
study's private module could drift from what its notebook actually did without
a reader ever seeing the drift.

**Before adding anything here, in this order:** look for a function that
already does what is needed — ``/graphify`` answers "what already does X" and
"what calls Y" across the tree faster than reading files one at a time; if none
does, look for one that can be adapted, which is allowed only if every existing
caller keeps its present behaviour and defaults; only then write a new
function, here, in the module whose subject it shares, or in a new module when
it shares no subject with any existing one. The rule is stated in full in
``instructions-pipeline`` § Studies at the repository root.

Nothing here hard-codes a path, and nothing here decides a threshold on a
study's behalf. Paths and governing parameters arrive as arguments from the
notebook that owns them. A default is acceptable only where it is a documented
constant of the domain — :data:`shmlib.adc.DOCUMENTED_COEFF` is the
manufacturer's calibration — and never where it is a choice.

| Module | Holds |
|---|---|
| :mod:`shmlib.adc` | The ``.adc`` format, its parsers, the instrument constants, the documented compensation |
| :mod:`shmlib.site` | Eras, stations, channels, the logger's clock, the archive's extent, clock conversion, season labels |
| :mod:`shmlib.solar` | Site coordinates and NOAA solar geometry |
| :mod:`shmlib.meteo` | Circular statistics for directional data, such as wind direction |
| :mod:`shmlib.proxies` | External forcing sources, the canonical channel vocabulary, their file formats and loaders |
| :mod:`shmlib.quality` | The flag vocabulary, the certified radiation window, day censuses, clock checks, verdicts |
| :mod:`shmlib.compare` | Diurnal cycles, pairwise agreement, stability over time, calibration against a reference |
| :mod:`shmlib.coupling` | Thermal operators, delay-and-time-constant scans, band separation, gains and their stability |
| :mod:`shmlib.viz` | Colour and label identity, report style, figure saving |
| :mod:`shmlib.figures` | Multi-source comparison figures |
| :mod:`shmlib.tables` | LaTeX table bodies for the reports |

``shmlib`` is not installed. Studies reach it by putting ``studies/`` on the
path, which every study notebook does in its imports cell::

    import os
    import sys
    sys.path.insert(0, os.path.abspath('..'))       # studies/, for shmlib
    sys.path.insert(0, os.path.abspath('../..'))    # repo root, for heritageshm

    from shmlib import adc, site, solar, meteo, proxies, quality, compare, coupling, tables, viz, figures

It is a different library from ``heritageshm/`` at the repository root: that one
is the production pipeline, this one is the studies' common ground.
"""

from . import adc, site, solar, meteo, proxies, quality, compare, coupling, tables, viz, figures  # noqa: E501,F401

__all__ = ['adc', 'site', 'solar', 'meteo', 'proxies', 'quality', 'compare',
           'coupling', 'tables', 'viz', 'figures']
