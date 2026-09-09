"""
Regenerate ``test_24_changepoints.py`` from the parent study script.

Run this with the working directory set to ``sidequest/`` (the directory
this file lives in) -- every relative path below is written against that
cwd. It reads ``../greybox_monitoring_study.py``, truncates it at the end of
the ``GM_F04_trend`` figure call, applies a small set of guarded
substitutions, and splices in the season-changepoint block that this
sidequest introduced in place of ``prediction.covered_changepoints``. It
OVERWRITES ``test_24_changepoints.py`` with the result: re-running this
script discards whatever is currently in that file and replaces it with a
fresh regeneration from the parent, so it must only be run again once the
season-changepoint block and guidance markdown embedded below have been
updated to match any further change the parent study or this sidequest have
made in the meantime.
"""

import os

source_file = '../greybox_monitoring_study.py'
dest_file = 'test_24_changepoints.py'

os.makedirs('outputs', exist_ok=True)

with open(source_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Truncate just after the GM_F04_trend figure call. Found by content, not by
# a hardcoded line number: the parent study keeps growing past this cutoff
# (it was line 1272 when this generator was first written, and had drifted
# to line 1293 by the time of this rewrite), and a hardcoded cutoff goes
# stale again every time the parent grows further.
CUTOFF_MARKER = "filename='GM_F04_trend')"
cutoff_index = None
for i, line in enumerate(lines):
    if CUTOFF_MARKER in line:
        cutoff_index = i
        break
if cutoff_index is None:
    raise RuntimeError(
        f"could not find cutoff marker {CUTOFF_MARKER!r} in {source_file}; "
        "the parent study must have renamed or removed the GM_F04_trend figure call, "
        "and this generator's truncation point needs to be updated to match."
    )
lines = lines[:cutoff_index + 1]

substitutions_applied = []


def substitute_line(old_prefix, new_line, label):
    """Replace, in place, the single line in ``lines`` that starts with
    ``old_prefix``, with ``new_line``. Raises if no such line is found, so
    that a miss is a loud failure rather than the silent no-op that let the
    previous version of this generator drift out of date with the parent
    study without anyone noticing."""
    for i, line in enumerate(lines):
        if line.startswith(old_prefix):
            lines[i] = new_line
            substitutions_applied.append(label)
            return
    raise RuntimeError(
        f"substitution {label!r} failed: no line starting with {old_prefix!r} "
        f"was found in the truncated copy of {source_file}"
    )


substitute_line('N_CHANGEPOINTS = 12', 'N_CHANGEPOINTS = 24\n', 'N_CHANGEPOINTS: 12 -> 24')

# The generated file is executed with the working directory set to the study
# root (05_greybox_monitoring/), not to sidequest/, so its OUTPUT_DIR must
# point at sidequest/outputs from that vantage point -- unlike this
# generator's own os.makedirs('outputs', ...) call above, which runs with
# cwd already inside sidequest/.
substitute_line("OUTPUT_DIR = Path('outputs')", "OUTPUT_DIR = Path('sidequest/outputs')\n",
                'OUTPUT_DIR: outputs -> sidequest/outputs')

# N_JOBS needs no substitution: the parent study already uses 32 workers,
# the same value this sidequest wants. Asserted rather than assumed, so
# that if the parent's value ever changes this generator fails loudly
# instead of silently reusing a worker count nobody chose.
if not any(line.startswith('N_JOBS = 32') for line in lines):
    raise RuntimeError(
        "expected the parent study's N_JOBS to already read 32 (no substitution needed); "
        "it has apparently changed, and this generator needs a real substitution added for it"
    )
substitutions_applied.append('N_JOBS already 32 in the parent, no substitution needed')

# Replace the parent's N_CHANGEPOINTS / CHANGEPOINTS_RANGE guidance markdown
# with the guidance this sidequest actually needs, describing the
# season-changepoint replacement rather than the parent's quantile-based one.
GUIDANCE_BLOCK = """# **`N_CHANGEPOINTS`** — ignored by this sidequest, and kept only so that
# `LADDER_N_CHANGEPOINTS` below still has a number to divide. The parent
# study places this many changepoints at quantiles of covered time through
# `prediction.covered_changepoints`; here the placement is the calendar's
# instead. `season_changepoints`, defined beside the fits in Movement 2,
# puts a changepoint on every astronomical season boundary the training
# window spans — the March equinox, the June solstice, the September
# equinox and the December solstice — snapped to the nearest covered
# timestamp and dropped when the nearest covered timestamp is more than half
# a season away. How many changepoints a fit gets is therefore decided by
# the length of its own training window, and is reported by the notebook
# when it runs.
#
# **`CHANGEPOINTS_RANGE`** — fraction of the training range eligible to
# carry a changepoint; default `0.95`. Inert here: NeuralProphet 0.8.0
# ignores both `changepoints_range` and `n_changepoints` whenever an
# explicit `changepoints` list is supplied, which is the case for every fit
# in this sidequest.
"""

guidance_start = "# **`N_CHANGEPOINTS`** — trend changepoints, placed on covered time through\n"
guidance_end = "# carry a changepoint; default `0.95`.\n"
try:
    i0 = lines.index(guidance_start)
    i1 = lines.index(guidance_end, i0)
except ValueError as exc:
    raise RuntimeError(
        "could not find the parent's N_CHANGEPOINTS/CHANGEPOINTS_RANGE guidance block "
        "(start or end anchor line missing); the parent study's wording must have changed"
    ) from exc
lines[i0:i1 + 1] = GUIDANCE_BLOCK.splitlines(keepends=True)
substitutions_applied.append('N_CHANGEPOINTS/CHANGEPOINTS_RANGE guidance markdown -> sidequest guidance')

# Replace the parent's quantile-based changepoint placement with the
# season-changepoint block this sidequest actually runs: a signature-
# compatible replacement for prediction.covered_changepoints that places
# changepoints on astronomical season boundaries instead of quantiles of
# covered time.
SEASON_BLOCK = '''SEASON_BOUNDARIES = (('03', '21'), ('06', '21'), ('09', '23'), ('12', '21'))


def season_changepoints(index, n_changepoints=None, observed_mask=None):
    """
    Trend changepoints at the astronomical season boundaries.

    A signature-compatible replacement for
    :func:`shmlib.prediction.covered_changepoints`, which spaces changepoints
    at quantiles of covered time. Here the candidate locations are fixed by
    the calendar instead — the March equinox, the June solstice, the
    September equinox and the December solstice of every year the index
    spans, taken at local midnight — so that each trend segment covers one
    season and its rate is the rate of that season.

    Every candidate is snapped to the nearest covered timestamp, because a
    changepoint placed inside an outage is constrained by no observation and
    leaves the trend free to move arbitrarily across it — the property
    ``covered_changepoints`` exists to guarantee and which this function must
    not give up. A candidate whose nearest covered timestamp lies further
    away than half the spacing to its neighbouring boundary is dropped rather
    than snapped: a boundary moved more than halfway towards the next one no
    longer marks the season it was named for.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full analysis grid, covered and uncovered alike.
    n_changepoints : int or None, optional
        Ignored. Accepted, and accepted positionally, only so that this
        function can stand in for ``covered_changepoints`` at call sites that
        pass a count. How many changepoints come back is decided by the
        calendar and by how much of it the record covers, never by a count.
    observed_mask : pd.Series or array-like or None, optional
        Boolean per timestamp, true where a value is present. ``None`` (the
        default) treats every timestamp as covered.

    Returns
    -------
    pd.DatetimeIndex
        Increasing, unique changepoint locations, every one of them a
        timestamp the record actually covers.
    """
    index = pd.DatetimeIndex(index)
    if observed_mask is None:
        covered = index
    else:
        mask = pd.Series(np.asarray(observed_mask), index=index).fillna(False).astype(bool)
        covered = index[mask.to_numpy()]
    if len(covered) == 0:
        return pd.DatetimeIndex([])

    covered = covered.sort_values()
    first, last = covered[0], covered[-1]

    # One extra year on each side so that the first and last in-range
    # candidate still have a neighbour to measure their tolerance against.
    candidates = []
    for year in range(first.year - 1, last.year + 2):
        for month, day in SEASON_BOUNDARIES:
            stamp = pd.Timestamp(f'{year}-{month}-{day}')
            if first.tz is not None:
                stamp = stamp.tz_localize(first.tz)
            candidates.append(stamp)
    candidates = pd.DatetimeIndex(sorted(candidates))

    kept = []
    for position, stamp in enumerate(candidates):
        if not first < stamp < last:
            continue
        tolerance = min(abs(candidates[position + step] - stamp)
                        for step in (-1, 1)
                        if 0 <= position + step < len(candidates)) / 2
        slot = covered.searchsorted(stamp)
        nearest = min((covered[p] for p in (slot - 1, slot)
                       if 0 <= p < len(covered)),
                      key=lambda c: abs(c - stamp))
        if abs(nearest - stamp) <= tolerance:
            kept.append(nearest)

    return pd.DatetimeIndex(sorted(set(kept)))


# `prediction.attribution_fits` and `prediction.fold_stability` take a
# changepoint *count* and place the changepoints themselves, by calling
# `covered_changepoints` as a bare global from inside the closure they
# dispatch to their workers (prediction.py:2900 and :3056). Rebinding the
# module attribute is therefore the only way to reach the fit that draws
# GM_F04 without editing `shmlib`, which this sidequest may not touch. It
# does reach the workers: cloudpickle captures that global by value when it
# pickles the closure, so a replacement defined here in `__main__` is
# serialised into each loky process.
#
# The proper fix, for whenever the no-touch constraint lifts, is a
# `changepoints=` passthrough on both functions, matching the one
# `sweep_trend_reg` and `compare_daily_terms` already have.
prediction.covered_changepoints = season_changepoints

changepoints_str = season_changepoints(train_str.index)
N_CHANGEPOINTS = len(changepoints_str)
print(f'season changepoints on the on-structure training head: {N_CHANGEPOINTS}')
for stamp in changepoints_str:
    print('   ', stamp)
'''

season_line = 'changepoints_str = prediction.covered_changepoints(train_str.index, N_CHANGEPOINTS)\n'
try:
    i = lines.index(season_line)
except ValueError as exc:
    raise RuntimeError(
        "could not find the parent's changepoints_str = prediction.covered_changepoints(...) line; "
        "the parent study's Movement 2 fit setup must have changed"
    ) from exc
lines[i:i + 1] = SEASON_BLOCK.splitlines(keepends=True)
substitutions_applied.append('changepoints_str line -> season-changepoint block')

# newline='' disables text-mode newline translation on write, so the '\n'
# already present in every line (normalized on read above, regardless of the
# parent file's own convention) is written out as a bare LF instead of being
# translated to the host platform's line separator -- CRLF on Windows, which
# is where this generator runs. The rest of the repository's tracked files
# are LF-terminated, and the generated file must match.
with open(dest_file, 'w', encoding='utf-8', newline='') as f:
    f.writelines(lines)

print(f"wrote {len(lines)} lines to {dest_file}")
print("substitutions applied:")
for item in substitutions_applied:
    print(f"  - {item}")
